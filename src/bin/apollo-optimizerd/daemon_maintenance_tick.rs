//! Maintenance Purge tick — opportunistic non-crisis page reclaim.
//!
//! See docs/superpowers/specs/2026-05-10-maintenance-purge-design.md
//!
//! Spec invariants:
//! - Pressure window: 0.65 ≤ raw < 0.85 (no overlap with survival ≥0.85)
//! - Swap floor: max(1.5 GB, 50% × swap_total)
//! - Swap delta sustained < 256 KB/s for 90s (via SwapDeltaWindow)
//! - User idle ≥120s + 10s post-wake quiet
//! - Media-active bypass: audio playing / video call / generic sleep-assertion
//!   `purge` invalidates the entire file-backed page cache; processes with
//!   active media re-fault frames from SSD causing audio glitches and video
//!   stutter. UserContext.audio_active/call_in_progress/has_sleep_assertion
//!   are sticky 60s-window signals (pmset assertions).
//! - Build-active bypass (caller passes bool from BuildTracker)
//! - Reads + writes shared last_any_purge_at (30 min)

use std::sync::atomic::Ordering;

use apollo_engine::collector::SystemSnapshot;
use apollo_engine::engine::coreaudio_active::AudioActivitySnapshot;
use apollo_engine::engine::daemon_helpers::spawn_reaped_purge;
use apollo_engine::engine::lse_counters::LockFreeMetrics;
use apollo_engine::engine::maintenance_state::MaintenanceState;
use apollo_engine::engine::shadow_signals;
use apollo_engine::engine::user_context::UserContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    PressureLow,
    PressureSurvival,
    /// B.4 purge band (2026-06-10): pressure entered [0.70, 0.75) without
    /// prior eligibility in the safe band — likely a fast ramp toward
    /// crisis; purging now would add jank to the ramp.
    PressureRisingEdge,
    SwapFloor,
    Growing,
    Idle,
    PostWake,
    /// Audio playing / video call active / generic sleep-assertion held.
    /// Skipping prevents page-cache invalidation glitches in active media.
    MediaActive,
    BuildMode,
    RateLimit,
    /// Sprint 12 Convergence #5 (2026-05-17). Unified-memory bus is
    /// saturated (entropy_anomaly > 2.0 fallback on M1; or amc>80% with
    /// IOReport entitlement). A vm_purge while the bus is busy contends
    /// with whatever drives the bandwidth (usually LLM inference) and
    /// induces user-visible jank — the gate must yield until the bus
    /// quiets. [Hennessy & Patterson 2017 §2.2]
    BusSaturated,
}

// B.4 purge band (2026-06-10): widened + hysteresis. Old gate [0.65, 0.85)
// left purge nearly unreachable in practice (3882 pressure skips vs 4-7
// fires); the band starts earlier (0.55) so reclaim happens BEFORE the
// compressor is saturated, and hard-skips earlier (0.75) because purging
// inside a crisis ramp adds I/O jank — survival tick (>=0.85) and the
// Gate-F thrashing bypass remain the crisis paths.
const PURGE_BAND_ENTRY_LOW: f64 = 0.55;
const PURGE_HARD_SKIP: f64 = 0.75;

const EMERGENCY_THRASHING_PURGE_SCORE: f64 = 25_000.0;
const CRITICAL_THRASHING_PURGE_SCORE: f64 = 50_000.0;
/// Memory-pressure floor for the emergency thrashing-bypass purge
/// (2026-06-13). A high thrashing_score with LOW physical pressure is
/// almost always benign page-cache churn from a cooperative app — prod
/// caught 12 emergency purges firing at pressure 0.51-0.66 with Brave
/// renderers spiking thrash to 50k-260k while swap stayed flat. Purging
/// then is a pure ~tens-of-ms stall: the "stutter at random moments" the
/// user reported. Below this floor, only a genuinely predicted OOM may
/// purge; the streak heuristic alone does not.
const EMERGENCY_PURGE_PRESSURE_FLOOR: f64 = 0.70;
const EMERGENCY_THRASHING_STREAK_SCORE: f64 = 15_000.0;
const EMERGENCY_THRASHING_MIN_CYCLES: u32 = 3;
const EMERGENCY_PURGE_COOLDOWN_SECS: u64 = 300;
const CRITICAL_THRASHING_P_OOM: f64 = 0.80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AutomaticPurgeMediaGuard {
    active: bool,
    source: &'static str,
}

fn automatic_purge_media_guard(
    ctx: &UserContext,
    audio: AudioActivitySnapshot,
) -> AutomaticPurgeMediaGuard {
    if ctx.call_in_progress {
        return AutomaticPurgeMediaGuard {
            active: true,
            source: "call",
        };
    }
    if ctx.audio_active || audio.output_active || audio.input_active {
        return AutomaticPurgeMediaGuard {
            active: true,
            source: "audio",
        };
    }
    if ctx.has_sleep_assertion {
        return AutomaticPurgeMediaGuard {
            active: true,
            source: "assertion",
        };
    }
    if !audio.session_supported || !audio.output_probe_available || !audio.input_probe_available {
        return AutomaticPurgeMediaGuard {
            active: true,
            source: "hal-unobservable",
        };
    }
    AutomaticPurgeMediaGuard {
        active: false,
        source: "none",
    }
}

/// Returns true if the maintenance tick fired a purge in this cycle.
/// Caller should record `system_maintenance_purge` in the CausalGraph
/// for observational outcome tracking (≥30 samples before trusting).
pub fn run_maintenance_tick(
    snap: &SystemSnapshot,
    ctx: &UserContext,
    state: &mut MaintenanceState,
    lf_metrics: &LockFreeMetrics,
    build_active: bool,
    bus_saturated: bool,
) -> bool {
    run_maintenance_tick_with_audio(
        snap,
        ctx,
        state,
        lf_metrics,
        build_active,
        bus_saturated,
        apollo_engine::engine::coreaudio_active::audio_activity_snapshot(),
    )
}

fn run_maintenance_tick_with_audio(
    snap: &SystemSnapshot,
    ctx: &UserContext,
    state: &mut MaintenanceState,
    lf_metrics: &LockFreeMetrics,
    build_active: bool,
    bus_saturated: bool,
    audio_snapshot: AudioActivitySnapshot,
) -> bool {
    state.push_swap_delta(snap.pressure.swap_delta_bytes_per_sec);
    // B.4: advance the Schmitt trigger before should_fire reads it.
    state.tick_pressure_band(snap.pressure.memory_pressure);

    // Gate F (2026-05-12): emergency thrashing-triggered purge bypass.
    // The normal maintenance gate requires idle_long + 1800s rate-limit,
    // both legitimate for "background maintenance". But the 180s stress
    // test revealed Apollo's generic-pressure response gap: thrashing
    // sustained at 22k while pressure peaked 0.75 (below survival 0.85),
    // user-visible "system unresponsive" with no Apollo action available.
    //
    // Emergency path: thrashing > 25k for ≥3 cycles AND no media/call AND
    // build_active false → purge bypass with 300s cooldown (not 1800s).
    // Critical non-media path may use a sustained 50k streak. Any positive
    // media guard requires a genuinely predicted OOM before purge is allowed.
    // [Camacho 2007] predictive control under sustained flow-crisis must
    // override level gates that are tuned for level thresholds.
    let thrash = snap.pressure.thrashing_score;
    state.push_thrashing(thrash);
    let p_oom_30s = shadow_signals::get_p_oom_30s().unwrap_or(0.0);
    // Compose direct HAL with pmset/user context. A root LaunchDaemon cannot
    // observe login-session default devices, so unavailable HAL is treated as
    // uncertainty and blocks automatic purge instead of pretending silence.
    let physical_pressure = if snap.pressure.memory_pressure_raw > 0.0 {
        snap.pressure.memory_pressure_raw
    } else {
        snap.pressure.memory_pressure
    };
    // The EMERGENCY purge gets the ACTUAL call signal, not high_bw. Its
    // internal logic already handles storms correctly: a genuine call gets
    // glitch protection (purge only on predicted OOM), while a non-call thrash
    // STREAK (consecutive_thrash_50k_cycles >= 10) still purges for relief.
    // Passing high_bw here (the 2026-06-15 regression) gave storms the call
    // treatment — blocking the streak relief → thrash strangled to 69k.
    let media_guard = automatic_purge_media_guard(ctx, audio_snapshot);
    let emergency = emergency_thrashing_purge_allowed(
        thrash,
        p_oom_30s,
        physical_pressure,
        ctx,
        state,
        build_active,
        bus_saturated,
        media_guard.active,
    );
    // Phase 1 (fixed 2026-06-15): the gentler NORMAL purge holds off during a
    // TRANSIENT high-volume workload (call or fault-in storm). The survival
    // escape inside is_high_bw_workload_active yields once physical pressure
    // crosses the floor, and the emergency streak path above is the relief
    // backstop, so this can no longer strangle.
    let high_bw_workload = apollo_engine::engine::coreaudio_active::is_high_bw_workload_active(
        snap.pressure.refault_delta_per_sec,
        physical_pressure,
    );
    if emergency && spawn_reaped_purge() {
        state.mark_purged();
        state.mark_compressor_flushing(snap.pressure.swap_delta_bytes_per_sec < 0.0);
        lf_metrics
            .maintenance_purge_total
            .fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            thrashing = thrash as u64,
            pressure = snap.pressure.memory_pressure,
            "maintenance: emergency thrashing-bypass purge"
        );
        return true;
    }

    // Normal-path hard guard: a live full-duplex call blocks the background
    // purge regardless of the (possibly stale) pmset audio_active flag. The
    // normal path is "background maintenance" — it has no business stalling a
    // call. Counted under the idle aggregate, same as the MediaActive skip.
    if high_bw_workload {
        lf_metrics
            .maintenance_purge_skipped_idle_total
            .fetch_add(1, Ordering::Relaxed);
        return false;
    }

    match should_fire_with_media_guard(
        snap,
        ctx,
        state,
        build_active,
        bus_saturated,
        media_guard.active,
    ) {
        None => {
            if spawn_reaped_purge() {
                state.mark_purged();
                state.mark_compressor_flushing(snap.pressure.swap_delta_bytes_per_sec < 0.0);
                lf_metrics
                    .maintenance_purge_total
                    .fetch_add(1, Ordering::Relaxed);
                return true;
            }
            false
        }
        Some(reason) => {
            if reason == SkipReason::MediaActive && media_guard.active {
                tracing::debug!(
                    source = media_guard.source,
                    "maintenance: media guard skipped purge"
                );
            }
            // B.4: split counters disambiguate the legacy aggregate (which
            // keeps incrementing as the sum for dashboard continuity).
            match reason {
                SkipReason::PressureLow => {
                    lf_metrics
                        .maintenance_purge_skipped_pressure_low_total
                        .fetch_add(1, Ordering::Relaxed);
                }
                SkipReason::PressureSurvival => {
                    lf_metrics
                        .maintenance_purge_skipped_pressure_survival_total
                        .fetch_add(1, Ordering::Relaxed);
                }
                SkipReason::PressureRisingEdge => {
                    lf_metrics
                        .maintenance_purge_skipped_rising_edge_total
                        .fetch_add(1, Ordering::Relaxed);
                }
                _ => {}
            }
            let counter = match reason {
                SkipReason::PressureLow
                | SkipReason::PressureSurvival
                | SkipReason::PressureRisingEdge => {
                    &lf_metrics.maintenance_purge_skipped_pressure_total
                }
                SkipReason::SwapFloor => &lf_metrics.maintenance_purge_skipped_swap_floor_total,
                SkipReason::Growing => &lf_metrics.maintenance_purge_skipped_growing_total,
                SkipReason::Idle | SkipReason::PostWake | SkipReason::MediaActive => {
                    &lf_metrics.maintenance_purge_skipped_idle_total
                }
                SkipReason::BuildMode => &lf_metrics.maintenance_purge_skipped_build_mode_total,
                SkipReason::RateLimit => &lf_metrics.maintenance_purge_skipped_rate_limit_total,
                SkipReason::BusSaturated => {
                    &lf_metrics.maintenance_purge_skipped_bus_saturated_total
                }
            };
            counter.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
}

fn emergency_thrashing_purge_allowed(
    thrash: f64,
    p_oom_30s: f64,
    pressure: f64,
    ctx: &UserContext,
    state: &MaintenanceState,
    build_active: bool,
    bus_saturated: bool,
    media_guarded: bool,
) -> bool {
    if thrash <= EMERGENCY_THRASHING_PURGE_SCORE
        || !state.thrashing_streak_above(
            EMERGENCY_THRASHING_STREAK_SCORE,
            EMERGENCY_THRASHING_MIN_CYCLES,
        )
        || build_active
        || state.secs_since_any_purge() < EMERGENCY_PURGE_COOLDOWN_SECS
    {
        return false;
    }

    let critical_lockup = thrash > CRITICAL_THRASHING_PURGE_SCORE
        && (p_oom_30s >= CRITICAL_THRASHING_P_OOM || state.consecutive_thrash_50k_cycles >= 10);
    let genuine_oom =
        thrash > CRITICAL_THRASHING_PURGE_SCORE && p_oom_30s >= CRITICAL_THRASHING_P_OOM;

    if bus_saturated && !critical_lockup {
        return false;
    }

    // 2026-06-11 fight-hunt fix — "se purgea mucho en llamadas / ruido raro":
    // a `purge` stalls the whole system for tens of ms → CoreAudio buffer
    // underrun (audible crackle) on a live call. The B.5 bypass below was too
    // aggressive: a thrash STREAK alone (consecutive_thrash_50k_cycles >= 10,
    // which also flips critical_lockup) bypassed the media gate, firing purges
    // mid-Meet. During a GENUINE full-duplex call (mic+output both running —
    // a fresh CoreAudio probe, robust where the pmset audio_active flag is
    // stale or a browser owns the assertion), ONLY a genuinely PREDICTED OOM
    // (p_oom_30s past the critical bar) may eat the stall. The streak
    // heuristic does not qualify: a flow-crisis is survivable, a mid-call
    // glitch is certain. An imminent OOM kill would drop the call anyway, so
    // there the stall is the lesser evil.
    if media_guarded || ctx.audio_active || ctx.call_in_progress || ctx.has_sleep_assertion {
        return genuine_oom;
    }

    // 2026-06-13 pressure-floor fix — "stutter at random moments": a thrash
    // STREAK with low physical pressure is benign page-cache churn from a
    // cooperative app (Brave renderers spike thrashing_score to 50k-260k
    // while pressure stays ~0.52 and swap is flat). Purging then is a pure
    // stall, not relief — purge frees page-cache, but at 0.52 pressure there
    // is no scarcity to relieve. Below the floor, ONLY a genuinely predicted
    // OOM (the same critical bar used for live calls) justifies the emergency
    // purge; real escalating scarcity (pressure >= 0.70) still purges.
    if pressure < EMERGENCY_PURGE_PRESSURE_FLOOR && !genuine_oom {
        return false;
    }

    // B.5 (2026-06-09): sustained 50k+ thrashing (≥10 cycles) remains a
    // non-media relief path. The media branch returned above and only a
    // predicted OOM can override it.
    critical_lockup || state.consecutive_thrash_50k_cycles >= 10
}

pub(crate) fn should_fire(
    snap: &SystemSnapshot,
    ctx: &UserContext,
    state: &MaintenanceState,
    build_active: bool,
    bus_saturated: bool,
) -> Option<SkipReason> {
    should_fire_with_media_guard(snap, ctx, state, build_active, bus_saturated, false)
}

fn should_fire_with_media_guard(
    snap: &SystemSnapshot,
    ctx: &UserContext,
    state: &MaintenanceState,
    build_active: bool,
    bus_saturated: bool,
    media_guarded: bool,
) -> Option<SkipReason> {
    // Fight-hunt fix (2026-06-10): the purge gate judges PHYSICAL pressure.
    // The 2026-05-10 design spec mandated raw ("purge addresses memory
    // pressure only; effective includes thermal/hw/llm/battery boosts that
    // purge cannot fix" — Skeptic verdict), but the per-cycle aggregator
    // overwrites memory_pressure with effective BEFORE this tick runs.
    // Fallback to effective when raw is unset (tests build snapshots
    // without the aggregation pass).
    let p = if snap.pressure.memory_pressure_raw > 0.0 {
        snap.pressure.memory_pressure_raw
    } else {
        snap.pressure.memory_pressure
    };
    if p < PURGE_BAND_ENTRY_LOW {
        return Some(SkipReason::PressureLow);
    }
    if p >= PURGE_HARD_SKIP {
        return Some(SkipReason::PressureSurvival);
    }
    // [0.70, 0.75): only proceed when the Schmitt trigger was armed in the
    // safe band — fresh entry here is a fast ramp; skip and let the crisis
    // paths (survival >=0.85, Gate-F thrashing) own it. [Hellerstein 2004 §9]
    if p >= 0.70 && !state.purge_band_eligible {
        return Some(SkipReason::PressureRisingEdge);
    }

    let swap_used = snap.pressure.swap_used_bytes;
    let swap_total = snap.pressure.swap_total_bytes;
    let swap_floor = std::cmp::max(1_536u64 * 1024 * 1024, swap_total / 2);
    if swap_used < swap_floor {
        return Some(SkipReason::SwapFloor);
    }

    if !state.swap_delta_window.sustained_below(256_000.0, 90) {
        return Some(SkipReason::Growing);
    }
    if !ctx.is_idle_long() {
        return Some(SkipReason::Idle);
    }
    if state.secs_since_wake() < 10 {
        return Some(SkipReason::PostWake);
    }
    // Media-active gate: audio playback / video calls / sleep-assertion
    // holders cannot tolerate page-cache invalidation. UserContext flags are
    // refreshed every cycle (pmset -g assertions polled with TTL) and combine
    // coreaudiod NoIdleSleep + NSPreventIdleSystemSleep + conferencing apps.
    if media_guarded || ctx.audio_active || ctx.call_in_progress || ctx.has_sleep_assertion {
        return Some(SkipReason::MediaActive);
    }
    // Sprint 12 Convergence #5 (2026-05-17): bus-saturation gate.
    // Same "now is dangerous" cohort as MediaActive — the system is
    // actively transferring data and a vm_purge would contend.
    // [Hennessy & Patterson 2017 §2.2] unified memory contention.
    if bus_saturated {
        return Some(SkipReason::BusSaturated);
    }
    if build_active {
        return Some(SkipReason::BuildMode);
    }
    if state.secs_since_any_purge() < 1800 {
        return Some(SkipReason::RateLimit);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::collector::{CpuStats, MemoryStats, PressureStats, SystemSnapshot};
    use apollo_engine::engine::user_context::UserContext;
    use chrono::Utc;

    fn synth_snap(pressure: f64, swap_used: u64, swap_total: u64) -> SystemSnapshot {
        SystemSnapshot {
            timestamp: Utc::now(),
            cpu: CpuStats {
                global_usage: 0.0,
                core_count: 8,
            },
            memory: MemoryStats {
                total_ram: 8 * 1024 * 1024 * 1024,
                used_ram: 4 * 1024 * 1024 * 1024,
                free_ram: 4 * 1024 * 1024 * 1024,
                total_swap: swap_total,
                used_swap: swap_used,
            },
            pressure: PressureStats {
                memory_pressure: pressure,
                swap_used_bytes: swap_used,
                swap_total_bytes: swap_total,
                swap_delta_bytes_per_sec: 0.0,
                thermal_level: "nominal".into(),
                compressor_pressure: 0.0,
                thrashing_score: 0.0,
                memory_pressure_raw: 0.0,
                refault_delta_per_sec: 0.0,
            },
            disks: vec![],
            networks: vec![],
            top_processes: vec![],
        }
    }

    fn idle_ctx() -> UserContext {
        UserContext {
            idle_secs: 200.0,
            ..Default::default()
        }
    }

    fn make_ready_state() -> MaintenanceState {
        let mut state = MaintenanceState::default();
        let now = std::time::SystemTime::now();
        for i in 0..45 {
            let t =
                now - std::time::Duration::from_secs(89) + std::time::Duration::from_secs(i * 2);
            state.swap_delta_window.push(t, 50_000.0);
        }
        // B.4: arm the Schmitt trigger — fixtures at 0.70 model a system
        // that was already in the safe band (eligibility carried forward).
        state.purge_band_eligible = true;
        state
    }

    #[test]
    fn purge_gate_follows_raw_pressure_not_effective() {
        // Fight-hunt fix (2026-06-10): effective pressure (battery/thermal
        // boosted) reads 0.80 — old code would hard-skip (PressureSurvival).
        // Physical pressure is 0.58 (in band) — purge is exactly what helps.
        let state = make_ready_state();
        let mut snap = synth_snap(0.80, 3_000_000_000, 4_000_000_000);
        snap.pressure.memory_pressure_raw = 0.58;
        let ctx = idle_ctx();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            None,
            "gate must judge physical pressure, not boosted effective"
        );
        // And the inverse: raw says crisis (0.80) even if effective were low.
        let mut snap2 = synth_snap(0.60, 3_000_000_000, 4_000_000_000);
        snap2.pressure.memory_pressure_raw = 0.80;
        assert_eq!(
            should_fire(&snap2, &ctx, &state, false, false),
            Some(SkipReason::PressureSurvival)
        );
    }

    #[test]
    fn band_pressure_058_fires_in_new_band() {
        // The paradox fix: 0.58 was PressureLow under the old 0.65 gate.
        let state = make_ready_state();
        let snap = synth_snap(0.58, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        assert_eq!(should_fire(&snap, &ctx, &state, false, false), None);
    }

    #[test]
    fn band_pressure_072_skips_on_fresh_entry() {
        let mut state = make_ready_state();
        state.purge_band_eligible = false; // fresh ramp, never in safe band
        let snap = synth_snap(0.72, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::PressureRisingEdge)
        );
    }

    #[test]
    fn band_pressure_072_fires_when_band_eligible() {
        let state = make_ready_state(); // eligible = true
        let snap = synth_snap(0.72, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        assert_eq!(should_fire(&snap, &ctx, &state, false, false), None);
    }

    #[test]
    fn band_pressure_076_hard_skips_even_when_eligible() {
        let state = make_ready_state();
        let snap = synth_snap(0.76, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::PressureSurvival)
        );
    }

    #[test]
    fn band_schmitt_trigger_state_machine() {
        let mut state = MaintenanceState::default();
        assert!(!state.purge_band_eligible);
        state.tick_pressure_band(0.60); // safe band → arm
        assert!(state.purge_band_eligible);
        state.tick_pressure_band(0.72); // hysteresis hold
        assert!(state.purge_band_eligible);
        state.tick_pressure_band(0.76); // crisis ramp → clear
        assert!(!state.purge_band_eligible);
        state.tick_pressure_band(0.72); // hold (still not eligible)
        assert!(!state.purge_band_eligible);
        state.tick_pressure_band(0.60); // re-arm
        assert!(state.purge_band_eligible);
        state.tick_pressure_band(0.45); // calm → clear
        assert!(!state.purge_band_eligible);
    }

    #[test]
    fn should_fire_pressure_below_returns_pressure_low() {
        let snap = synth_snap(0.50, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let state = MaintenanceState::default();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::PressureLow)
        );
    }

    #[test]
    fn should_fire_pressure_at_survival_returns_pressure_survival() {
        let snap = synth_snap(0.90, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let state = MaintenanceState::default();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::PressureSurvival)
        );
    }

    #[test]
    fn should_fire_swap_floor_traps_m1_cold_boot() {
        // M1 cold boot: swap_total=800MB, swap_used=500MB (62.5% by ratio).
        // 1.5 GB absolute floor MUST kick in to skip.
        let snap = synth_snap(0.70, 500 * 1024 * 1024, 800 * 1024 * 1024);
        let ctx = idle_ctx();
        let mut state = MaintenanceState {
            purge_band_eligible: true, // B.4: bypass rising-edge, test swap floor
            ..Default::default()
        };
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::SwapFloor)
        );
    }

    #[test]
    fn should_fire_growing_swap_returns_growing() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let mut state = MaintenanceState {
            purge_band_eligible: true, // B.4: bypass rising-edge, test growing
            ..Default::default()
        };
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::Growing)
        );

        let now = std::time::SystemTime::now();
        for i in 0..45 {
            let t =
                now - std::time::Duration::from_secs(89) + std::time::Duration::from_secs(i * 2);
            state.swap_delta_window.push(t, 50_000.0);
        }
        assert_ne!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::Growing)
        );
    }

    #[test]
    fn should_fire_user_active_returns_idle() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = UserContext {
            idle_secs: 10.0,
            ..Default::default()
        };
        let state = make_ready_state();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::Idle)
        );
    }

    #[test]
    fn should_fire_post_wake_quiet_returns_postwake() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let mut state = make_ready_state();
        state.observe_wake();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::PostWake)
        );
    }

    #[test]
    fn should_fire_audio_active_returns_media_active() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = UserContext {
            idle_secs: 200.0,
            audio_active: true,
            ..Default::default()
        };
        let state = make_ready_state();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::MediaActive)
        );
    }

    #[test]
    fn should_fire_call_in_progress_returns_media_active() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = UserContext {
            idle_secs: 200.0,
            call_in_progress: true,
            ..Default::default()
        };
        let state = make_ready_state();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::MediaActive)
        );
    }

    #[test]
    fn should_fire_sleep_assertion_returns_media_active() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = UserContext {
            idle_secs: 200.0,
            has_sleep_assertion: true,
            ..Default::default()
        };
        let state = make_ready_state();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::MediaActive)
        );
    }

    #[test]
    fn unobservable_hal_arms_app_agnostic_purge_guard() {
        let ctx = idle_ctx();
        let fallback = AudioActivitySnapshot::default();
        let guard = automatic_purge_media_guard(&ctx, fallback);
        assert!(guard.active);
        assert_eq!(guard.source, "hal-unobservable");

        let direct_silence = AudioActivitySnapshot {
            session_supported: true,
            output_probe_available: true,
            input_probe_available: true,
            ..AudioActivitySnapshot::default()
        };
        assert!(
            !automatic_purge_media_guard(&ctx, direct_silence).active,
            "only a complete direct HAL silence sample may clear uncertainty"
        );
    }

    #[test]
    fn direct_hal_playback_arms_purge_guard() {
        let ctx = idle_ctx();
        let audio = AudioActivitySnapshot {
            output_active: true,
            output_probe_available: true,
            session_supported: true,
            ..AudioActivitySnapshot::default()
        };
        let guard = automatic_purge_media_guard(&ctx, audio);
        assert!(guard.active);
        assert_eq!(guard.source, "audio");
    }

    #[test]
    fn fallback_media_guard_preserves_normal_gate_order() {
        let state = make_ready_state();
        let ctx = idle_ctx();
        let eligible = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        assert_eq!(
            should_fire_with_media_guard(&eligible, &ctx, &state, false, false, true),
            Some(SkipReason::MediaActive)
        );

        let low_pressure = synth_snap(0.30, 3_000_000_000, 4_000_000_000);
        assert_eq!(
            should_fire_with_media_guard(&low_pressure, &ctx, &state, false, false, true),
            Some(SkipReason::PressureLow),
            "media uncertainty must not hide the primary pressure diagnosis"
        );
    }

    #[test]
    fn maintenance_tick_never_spawns_purge_when_hal_is_unobservable() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let mut state = make_ready_state();
        let metrics = LockFreeMetrics::new();

        assert!(!run_maintenance_tick_with_audio(
            &snap,
            &ctx,
            &mut state,
            &metrics,
            false,
            false,
            AudioActivitySnapshot::default(),
        ));
        assert_eq!(metrics.maintenance_purge_total.load(Ordering::Relaxed), 0);
        assert_eq!(
            metrics
                .maintenance_purge_skipped_idle_total
                .load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn emergency_thrashing_respects_media_until_critical() {
        let ctx = UserContext {
            idle_secs: 200.0,
            audio_active: true,
            ..Default::default()
        };
        let mut state = MaintenanceState {
            consecutive_thrash_cycles: EMERGENCY_THRASHING_MIN_CYCLES,
            ..Default::default()
        };

        assert!(
            !emergency_thrashing_purge_allowed(
                30_000.0, 0.90, 0.80, &ctx, &state, false, false, false
            ),
            "moderate emergency thrashing should still respect active media"
        );
        assert!(
            !emergency_thrashing_purge_allowed(
                60_000.0, 0.40, 0.80, &ctx, &state, false, false, false
            ),
            "critical thrashing without high p_oom should still respect active media"
        );
        assert!(
            emergency_thrashing_purge_allowed(
                60_000.0, 0.90, 0.80, &ctx, &state, false, false, false
            ),
            "critical sustained thrashing plus high p_oom should bypass media politeness"
        );
    }

    #[test]
    fn emergency_thrashing_realtime_call_blocks_streak_bypass() {
        // 2026-06-11: a live full-duplex call must NOT be interrupted by the
        // B.5 thrash-streak bypass — only true OOM imminence (critical_lockup)
        // may eat the purge stall. Reproduces "se purgea mucho en llamadas".
        let ctx = idle_ctx(); // no pmset media flag — the call is browser-based
        let mut state = MaintenanceState {
            consecutive_thrash_cycles: EMERGENCY_THRASHING_MIN_CYCLES,
            ..Default::default()
        };
        state.consecutive_thrash_50k_cycles = 12; // would trip the B.5 streak bypass

        // Without a realtime call, the 50k streak bypasses → purge allowed.
        assert!(
            emergency_thrashing_purge_allowed(
                60_000.0, 0.40, 0.80, &ctx, &state, false, false, false
            ),
            "50k streak bypasses media politeness when NO realtime call"
        );
        // With a realtime call but p_oom below critical → blocked (the fix).
        assert!(
            !emergency_thrashing_purge_allowed(
                60_000.0, 0.40, 0.80, &ctx, &state, false, false, true
            ),
            "live call + sub-critical p_oom must NOT purge (audio-glitch guard)"
        );
        // But genuine OOM imminence (high p_oom) still bypasses even a call —
        // an OOM kill would drop the call anyway; the stall is the lesser evil.
        assert!(
            emergency_thrashing_purge_allowed(
                60_000.0, 0.90, 0.80, &ctx, &state, false, false, true
            ),
            "critical_lockup (high p_oom) overrides the realtime-call guard"
        );
    }

    #[test]
    fn emergency_thrashing_fallback_media_guard_requires_predicted_oom() {
        let ctx = idle_ctx();
        let mut state = MaintenanceState {
            consecutive_thrash_cycles: EMERGENCY_THRASHING_MIN_CYCLES,
            ..Default::default()
        };
        state.consecutive_thrash_50k_cycles = 12;

        assert!(
            !emergency_thrashing_purge_allowed(
                70_000.0, 0.30, 0.75, &ctx, &state, false, false, true
            ),
            "session-fallback media evidence must block the old streak bypass"
        );
        assert!(
            emergency_thrashing_purge_allowed(
                70_000.0, 0.90, 0.75, &ctx, &state, false, false, true
            ),
            "a genuinely predicted OOM remains the last-resort escape"
        );
    }

    #[test]
    fn emergency_thrashing_pressure_floor_blocks_benign_browser_churn() {
        // 2026-06-13: prod caught 12 emergency purges firing at pressure
        // 0.51-0.66 with Brave renderers spiking thrash to 50k-260k while
        // swap stayed flat — benign page-cache churn, not scarcity. The
        // purge stall was the user's "stutter at random moments".
        let ctx = idle_ctx();
        let mut state = MaintenanceState {
            consecutive_thrash_cycles: EMERGENCY_THRASHING_MIN_CYCLES,
            ..Default::default()
        };
        state.consecutive_thrash_50k_cycles = 12; // streak that used to bypass

        // Low pressure (0.55) + sub-critical p_oom → the streak no longer
        // purges: it's browser churn, not memory scarcity.
        assert!(
            !emergency_thrashing_purge_allowed(
                70_000.0, 0.30, 0.55, &ctx, &state, false, false, false
            ),
            "thrash streak with LOW pressure must NOT purge (benign churn)"
        );
        // Same streak but pressure now genuinely elevated (0.75) → real
        // escalating scarcity, purge proceeds.
        assert!(
            emergency_thrashing_purge_allowed(
                70_000.0, 0.30, 0.75, &ctx, &state, false, false, false
            ),
            "elevated pressure + thrash streak still purges (real scarcity)"
        );
        // Low pressure but a genuinely predicted OOM (high p_oom + critical
        // thrash) → purge overrides the floor (imminent kill is worse).
        assert!(
            emergency_thrashing_purge_allowed(
                60_000.0, 0.90, 0.55, &ctx, &state, false, false, false
            ),
            "predicted OOM overrides the pressure floor"
        );
    }

    #[test]
    fn emergency_thrashing_keeps_build_and_bus_blocks() {
        let ctx = idle_ctx();
        let mut state = MaintenanceState {
            consecutive_thrash_cycles: EMERGENCY_THRASHING_MIN_CYCLES,
            ..Default::default()
        };

        assert!(
            !emergency_thrashing_purge_allowed(
                60_000.0, 0.90, 0.80, &ctx, &state, true, false, false
            ),
            "build mode remains protected under critical thrashing"
        );
        assert!(
            !emergency_thrashing_purge_allowed(
                60_000.0, 0.40, 0.80, &ctx, &state, false, true, false
            ),
            "bus saturation remains protected without high p_oom"
        );
        assert!(
            emergency_thrashing_purge_allowed(
                60_000.0, 0.90, 0.80, &ctx, &state, false, true, false
            ),
            "high p_oom critical thrashing may bypass bus saturation to avoid lockup"
        );
    }

    #[test]
    fn should_fire_build_mode_returns_build_mode() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let state = make_ready_state();
        assert_eq!(
            should_fire(&snap, &ctx, &state, true, false),
            Some(SkipReason::BuildMode)
        );
    }

    #[test]
    fn should_fire_bus_saturated_returns_bus_saturated() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let state = make_ready_state();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, true),
            Some(SkipReason::BusSaturated),
            "bus_saturated=true while all other gates pass → BusSaturated"
        );
    }

    #[test]
    fn should_fire_bus_saturated_yields_to_media_active() {
        // MediaActive must be checked BEFORE BusSaturated so a call-in-progress
        // is reported as MediaActive (correct user-facing reason) even when the
        // bus is also saturated. Verifies the gate order documented at
        // run_maintenance_tick line ~165.
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = UserContext {
            idle_secs: 200.0,
            call_in_progress: true,
            ..Default::default()
        };
        let state = make_ready_state();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, true),
            Some(SkipReason::MediaActive),
            "MediaActive precedence over BusSaturated"
        );
    }

    #[test]
    fn should_fire_rate_limit_returns_rate_limit() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let mut state = make_ready_state();
        state.last_any_purge_at =
            Some(std::time::SystemTime::now() - std::time::Duration::from_secs(100));
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::RateLimit)
        );
    }

    #[test]
    fn should_fire_all_gates_pass_returns_none() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let state = make_ready_state();
        assert_eq!(should_fire(&snap, &ctx, &state, false, false), None);
    }
}
