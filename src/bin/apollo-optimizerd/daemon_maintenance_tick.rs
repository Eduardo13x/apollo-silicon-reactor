//! Maintenance Purge tick — opportunistic non-crisis page reclaim.
//!
//! See docs/superpowers/specs/2026-05-10-maintenance-purge-design.md
//!
//! Spec invariants:
//! - Pressure window: 0.65 ≤ raw < 0.85 (no overlap with survival ≥0.85)
//! - Swap floor: max(1.5 GB, 50% × swap_total)
//! - Swap delta sustained < 256 KB/s for 90s (via SwapDeltaWindow)
//! - User idle ≥120s + 10s post-wake quiet
//! - Build-active bypass (caller passes bool from BuildTracker)
//! - Reads + writes shared last_any_purge_at (30 min)

use std::sync::atomic::Ordering;

use apollo_engine::collector::SystemSnapshot;
use apollo_engine::engine::lse_counters::LockFreeMetrics;
use apollo_engine::engine::maintenance_state::MaintenanceState;
use apollo_engine::engine::user_context::UserContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    PressureLow,
    PressureSurvival,
    SwapFloor,
    Growing,
    Idle,
    PostWake,
    BuildMode,
    RateLimit,
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
) -> bool {
    state.push_swap_delta(snap.pressure.swap_delta_bytes_per_sec);

    match should_fire(snap, ctx, state, build_active) {
        None => {
            if std::process::Command::new("purge").spawn().is_ok() {
                state.mark_purged();
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
                SkipReason::Idle | SkipReason::PostWake => {
                    &lf_metrics.maintenance_purge_skipped_idle_total
                }
                SkipReason::BuildMode => &lf_metrics.maintenance_purge_skipped_build_mode_total,
                SkipReason::RateLimit => &lf_metrics.maintenance_purge_skipped_rate_limit_total,
            };
            counter.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
}

pub(crate) fn should_fire(
    snap: &SystemSnapshot,
    ctx: &UserContext,
    state: &MaintenanceState,
    build_active: bool,
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
            let t = now - std::time::Duration::from_secs(89)
                + std::time::Duration::from_secs(i * 2);
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
            should_fire(&snap, &ctx, &state, false),
            Some(SkipReason::PressureLow)
        );
    }

    #[test]
    fn should_fire_pressure_at_survival_returns_pressure_survival() {
        let snap = synth_snap(0.90, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let state = MaintenanceState::default();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false),
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
            should_fire(&snap, &ctx, &state, false),
            Some(SkipReason::SwapFloor)
        );
    }

    #[test]
    fn should_fire_growing_swap_returns_growing() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let mut state = MaintenanceState::default();
        assert_eq!(
            should_fire(&snap, &ctx, &state, false),
            Some(SkipReason::Growing)
        );

        let now = std::time::SystemTime::now();
        for i in 0..45 {
            let t = now - std::time::Duration::from_secs(89)
                + std::time::Duration::from_secs(i * 2);
            state.swap_delta_window.push(t, 50_000.0);
        }
        assert_ne!(
            should_fire(&snap, &ctx, &state, false),
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
            should_fire(&snap, &ctx, &state, false),
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
            should_fire(&snap, &ctx, &state, false),
            Some(SkipReason::PostWake)
        );
    }

    #[test]
    fn should_fire_build_mode_returns_build_mode() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let state = make_ready_state();
        assert_eq!(
            should_fire(&snap, &ctx, &state, true),
            Some(SkipReason::BuildMode)
        );
    }

    #[test]
    fn should_fire_rate_limit_returns_rate_limit() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let mut state = make_ready_state();
        state.last_any_purge_at = Some(
            std::time::SystemTime::now() - std::time::Duration::from_secs(100),
        );
        assert_eq!(
            should_fire(&snap, &ctx, &state, false),
            Some(SkipReason::RateLimit)
        );
    }

    #[test]
    fn should_fire_all_gates_pass_returns_none() {
        let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
        let ctx = idle_ctx();
        let state = make_ready_state();
        assert_eq!(should_fire(&snap, &ctx, &state, false), None);
    }
}
