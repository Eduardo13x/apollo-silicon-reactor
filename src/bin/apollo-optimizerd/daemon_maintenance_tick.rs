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
use apollo_engine::engine::lse_counters::LockFreeMetrics;
use apollo_engine::engine::maintenance_state::MaintenanceState;
use apollo_engine::engine::shadow_signals;
use apollo_engine::engine::user_context::UserContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    PressureLow,
    PressureSurvival,
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

const EMERGENCY_THRASHING_PURGE_SCORE: f64 = 25_000.0;
const CRITICAL_THRASHING_PURGE_SCORE: f64 = 50_000.0;
const EMERGENCY_THRASHING_STREAK_SCORE: f64 = 15_000.0;
const EMERGENCY_THRASHING_MIN_CYCLES: u32 = 3;
const EMERGENCY_PURGE_COOLDOWN_SECS: u64 = 300;
const CRITICAL_THRASHING_P_OOM: f64 = 0.80;

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
    state.push_swap_delta(snap.pressure.swap_delta_bytes_per_sec);

    // Gate F (2026-05-12): emergency thrashing-triggered purge bypass.
    // The normal maintenance gate requires idle_long + 1800s rate-limit,
    // both legitimate for "background maintenance". But the 180s stress
    // test revealed Apollo's generic-pressure response gap: thrashing
    // sustained at 22k while pressure peaked 0.75 (below survival 0.85),
    // user-visible "system unresponsive" with no Apollo action available.
    //
    // Emergency path: thrashing > 25k for ≥3 cycles AND no media/call AND
    // build_active false → purge bypass with 300s cooldown (not 1800s).
    // Critical path: thrashing > 50k for ≥3 cycles can bypass media/assertion
    // politeness too; at that flow rate the user is already paying the stall.
    // [Camacho 2007] predictive control under sustained flow-crisis must
    // override level gates that are tuned for level thresholds.
    let thrash = snap.pressure.thrashing_score;
    state.push_thrashing(thrash);
    let p_oom_30s = shadow_signals::get_p_oom_30s().unwrap_or(0.0);
    let emergency = emergency_thrashing_purge_allowed(
        thrash,
        p_oom_30s,
        ctx,
        state,
        build_active,
        bus_saturated,
    );
    if emergency && std::process::Command::new("purge").spawn().is_ok() {
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

    match should_fire(snap, ctx, state, build_active, bus_saturated) {
        None => {
            if std::process::Command::new("purge").spawn().is_ok() {
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
            let counter = match reason {
                SkipReason::PressureLow | SkipReason::PressureSurvival => {
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
    ctx: &UserContext,
    state: &MaintenanceState,
    build_active: bool,
    bus_saturated: bool,
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

    let media_or_assertion = ctx.audio_active || ctx.call_in_progress || ctx.has_sleep_assertion;
    let critical_lockup = thrash > CRITICAL_THRASHING_PURGE_SCORE
        && (p_oom_30s >= CRITICAL_THRASHING_P_OOM
            || state.consecutive_thrash_50k_cycles >= 10);

    if bus_saturated && !critical_lockup {
        return false;
    }

    // B.5 (2026-06-09): sustained 50k+ thrashing (≥10 cycles) bypasses the
    // MediaActive gate — a streak at this level is a flow crisis, not a
    // transient audio glitch. Without this, Meet/streaming/audio sessions
    // permanently block purge while thrashing climbs toward 62k+.
    !media_or_assertion || critical_lockup || state.consecutive_thrash_50k_cycles >= 10
}

pub(crate) fn should_fire(
    snap: &SystemSnapshot,
    ctx: &UserContext,
    state: &MaintenanceState,
    build_active: bool,
    bus_saturated: bool,
) -> Option<SkipReason> {
    let p = snap.pressure.memory_pressure;
    if p < 0.65 {
        return Some(SkipReason::PressureLow);
    }
    if p >= 0.85 {
        return Some(SkipReason::PressureSurvival);
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
    if ctx.audio_active || ctx.call_in_progress || ctx.has_sleep_assertion {
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
        state
    }

    #[test]
    fn should_fire_pressure_below_returns_pressure_low() {
        let snap = synth_snap(0.55, 3_000_000_000, 4_000_000_000);
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
        let state = MaintenanceState::default();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false, false),
            Some(SkipReason::SwapFloor)
        );
    }

    #[test]
    fn should_fire_growing_swap_returns_growing() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let mut state = MaintenanceState::default();
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
    fn emergency_thrashing_respects_media_until_critical() {
        let ctx = UserContext {
            idle_secs: 200.0,
            audio_active: true,
            ..Default::default()
        };
        let mut state = MaintenanceState::default();
        state.consecutive_thrash_cycles = EMERGENCY_THRASHING_MIN_CYCLES;

        assert!(
            !emergency_thrashing_purge_allowed(30_000.0, 0.90, &ctx, &state, false, false),
            "moderate emergency thrashing should still respect active media"
        );
        assert!(
            !emergency_thrashing_purge_allowed(60_000.0, 0.40, &ctx, &state, false, false),
            "critical thrashing without high p_oom should still respect active media"
        );
        assert!(
            emergency_thrashing_purge_allowed(60_000.0, 0.90, &ctx, &state, false, false),
            "critical sustained thrashing plus high p_oom should bypass media politeness"
        );
    }

    #[test]
    fn emergency_thrashing_keeps_build_and_bus_blocks() {
        let ctx = idle_ctx();
        let mut state = MaintenanceState::default();
        state.consecutive_thrash_cycles = EMERGENCY_THRASHING_MIN_CYCLES;

        assert!(
            !emergency_thrashing_purge_allowed(60_000.0, 0.90, &ctx, &state, true, false),
            "build mode remains protected under critical thrashing"
        );
        assert!(
            !emergency_thrashing_purge_allowed(60_000.0, 0.40, &ctx, &state, false, true),
            "bus saturation remains protected without high p_oom"
        );
        assert!(
            emergency_thrashing_purge_allowed(60_000.0, 0.90, &ctx, &state, false, true),
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
