//! # Daemon Wake Unfreeze
//!
//! Staggered wake unfreeze queue drain extracted from main.rs (Wave 25).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Drain wake_unfreeze_queue in small batches each cycle
//! - Apply thermal + swap-velocity bulkhead to shrink batch when system is stressed
//! - SIGCONT pids via unfreeze_pids_verified, remove from frozen_state, restore QoS
//!
//! ## Ordering invariant
//! Must run AFTER sleep/wake detection (daemon_wake_handler) and BEFORE the main
//! snapshot/decision pass so thawed processes are visible as live this cycle.

use std::collections::VecDeque;
use std::path::Path;

use apollo_optimizer::engine::daemon_helpers::{unfreeze_pids_verified, write_frozen_state};
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::mach_qos::SchedulingTier;
use apollo_optimizer::engine::background_collectors::PressureCollector;

/// Maximum PIDs to SIGCONT per cycle under normal conditions.
const WAKE_UNFREEZE_BATCH: usize = 5;

/// Drain one batch from the wake-unfreeze queue.
///
/// # Parameters
/// - `wake_unfreeze_queue` — queue of PIDs waiting for SIGCONT after wake
/// - `wake_thaw_pids` — accumulates PIDs SIGCONT'd this cycle (for ODE τ learning)
/// - `state` — SharedState (thermal_level_real, frozen_state, mach_qos locks)
/// - `pressure_collector` — for swap_delta_bps velocity (bulkhead gate)
/// - `frozen_state_path` — path for write_frozen_state WAL update
pub fn run_wake_unfreeze(
    wake_unfreeze_queue: &mut VecDeque<u32>,
    wake_thaw_pids: &mut Vec<u32>,
    state: &SharedState,
    pressure_collector: &PressureCollector,
    frozen_state_path: &Path,
) {
    if wake_unfreeze_queue.is_empty() {
        return;
    }

    let wake_batch = {
        // G21 — Thermal Bulkhead: serious/critical thermal → single-process
        // thaw prevents CPU surge from simultaneous reactivation.
        // [Nygard 2018 §4.3 — bulkhead limits blast radius under resource stress]
        let thermal_str = state.metrics.lock_recover().thermal_level_real.clone();
        if thermal_str == "serious" || thermal_str == "critical" {
            1_usize
        } else {
            // dM/dt proxy: swap_delta_bps > 0 = swap growing.
            // 50 MB/s growth → rate_factor = 1.0 → batch = 1.
            let rate_factor = (pressure_collector.latest().swap_delta_bps
                / (50.0 * 1024.0 * 1024.0))
                .clamp(0.0, 1.0);
            (WAKE_UNFREEZE_BATCH as f64 * (1.0 - rate_factor * 0.8))
                .max(1.0)
                .round() as usize
        }
    };

    let batch: Vec<u32> = wake_unfreeze_queue
        .drain(..wake_unfreeze_queue.len().min(wake_batch))
        .collect();

    // A-B-A defense: lock frozen_guard first to read identity
    // (start_sec) before signalling. Crash before SIGCONT leaves
    // PIDs in frozen_state for recovery on restart (WAL semantics).
    // [Saltzer & Kaashoek 2009] §3.3 Complete Mediation.
    {
        let mut frozen_guard = state.frozen_state.lock_recover();
        let entries: std::collections::HashMap<u32, apollo_optimizer::engine::types::FrozenEntry> =
            batch
                .iter()
                .filter_map(|&pid| frozen_guard.get(&pid).map(|e| (pid, e.clone())))
                .collect();
        unfreeze_pids_verified(&entries);
        for pid in &batch {
            frozen_guard.remove(pid);
        }
        write_frozen_state(frozen_state_path, &frozen_guard);
    }

    // Restore Mach QoS from Background (E-cores) → Normal so
    // processes resume on P-cores. Wake unfreeze is the highest-
    // urgency thaw path (user just returned to desktop), so P-core
    // routing is critical for perceived responsiveness.
    {
        let mut qos = state.mach_qos.lock_recover();
        for pid in &batch {
            let _ = qos.set_tier(*pid, SchedulingTier::Normal);
        }
    }

    // Record actual-SIGCONT T0 for unfreeze_decay ODE τ learning.
    wake_thaw_pids.extend_from_slice(&batch);
}
