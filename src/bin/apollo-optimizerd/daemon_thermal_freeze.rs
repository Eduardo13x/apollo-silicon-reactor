//! # Daemon Thermal Freeze
//!
//! Thermal pre-throttle freeze/unfreeze extracted from main.rs (Wave 20).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Phase3/Phase4: SIGSTOP background processes before hardware throttles (5-10°C ahead)
//! - Cooled: SIGCONT processes frozen by thermal (hysteresis via ThermalBailout)
//!
//! ## Safety
//! - `is_protected_name` is the single truth point [Saltzer & Kaashoek 2009]
//! - Max 80 freezes per cycle to bound per-cycle cost
//!
//! ## Ordering invariant
//! Must run AFTER thermal_action is computed (thermal_bailout.evaluate()) and BEFORE
//! the main decide_actions pass (thermal_frozen PIDs visible to downstream dedup).

use std::path::Path;

use apollo_engine::collector::SystemCollector;
use apollo_engine::engine::daemon_helpers::{unfreeze_pids, write_frozen_state};
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::outcome_tracker::OutcomeTracker;
use apollo_engine::engine::process_identity::ProcessIdentity;
use apollo_engine::engine::safety::is_protected_name;
use apollo_engine::engine::thermal_bailout::ThermalAction;
use apollo_engine::engine::types::{FreezeSource, FrozenEntry};
use chrono::Utc;

/// Apply thermal pre-throttle freeze or unfreeze for this cycle.
///
/// # Parameters
/// - `thermal_action` — action computed by ThermalBailout::evaluate()
/// - `state` — SharedState (frozen_state + metrics locks)
/// - `collector` — SystemCollector (process iterator)
/// - `foreground_pid` — current foreground PID (never frozen thermally)
/// - `memory_pressure` — effective memory pressure for this cycle
/// - `frozen_state_path` — path to persist frozen state JSON
/// - `outcome_tracker` — recipient of survival-bias closure events: each
///   freeze candidate skipped here is `record_blocked` so the learning
///   loop can later infer (counterfactually) whether the block was a
///   missed opportunity. SHADOW-MODE-ONLY signal.
pub fn run_thermal_freeze(
    thermal_action: &ThermalAction,
    state: &SharedState,
    collector: &SystemCollector,
    foreground_pid: Option<u32>,
    memory_pressure: f64,
    frozen_state_path: &Path,
    outcome_tracker: &mut OutcomeTracker,
) {
    if thermal_action.freeze_background || thermal_action.freeze_all_non_critical {
        let policy_protected = state
            .policy
            .lock_recover()
            .learned_policy
            .protected_patterns
            .clone();
        let mut frozen_guard = state.frozen_state.lock_recover();
        let mut thermal_frozen = 0u32;
        let cpu_threshold: f32 = if thermal_action.freeze_all_non_critical {
            100.0 // Phase4: no CPU filter
        } else {
            2.0 // Phase3: only idle processes
        };

        for (pid, process) in collector.system().processes() {
            let pid_u32 = pid.as_u32();
            let name = process.name().to_string();
            let cpu = process.cpu_usage();
            // Survival-bias closure: when is_protected_name blocks a freeze
            // candidate that otherwise qualified (cpu/foreground/duplicate
            // checks already passed), record the block so OutcomeTracker can
            // post-hoc infer whether the protection was a missed opportunity.
            // Only the is_protected_name branch is recorded; the other
            // skip reasons (foreground, already-frozen, apollo itself,
            // policy-protected user pattern) are not learning signals.
            //
            // [Bengio 2013] Counterfactual reasoning needs the unobserved
            // branch. SHADOW-MODE-ONLY — never used to auto-unblock.
            if cpu <= cpu_threshold
                && Some(pid_u32) != foreground_pid
                && !frozen_guard.contains_key(&pid_u32)
                && name != "apollo-optimizerd"
                && !policy_protected.iter().any(|p| name.contains(p.as_str()))
                && is_protected_name(&name)
            {
                outcome_tracker.record_blocked("freeze", "is-protected-name", memory_pressure);
            }
            if cpu > cpu_threshold
                || Some(pid_u32) == foreground_pid
                || is_protected_name(&name)
                || policy_protected.iter().any(|p| name.contains(p.as_str()))
                || name == "apollo-optimizerd"
                || frozen_guard.contains_key(&pid_u32)
            {
                continue;
            }
            if thermal_frozen >= 80 {
                break;
            }
            if unsafe { libc::kill(pid_u32 as i32, libc::SIGSTOP) } == 0 {
                frozen_guard.insert(
                    pid_u32,
                    FrozenEntry {
                        frozen_at: Utc::now(),
                        source: FreezeSource::ThermalPreThrottle,
                        pressure_at_freeze: memory_pressure,
                        process_name: Some(name.clone()),
                        start_sec: ProcessIdentity::from_pid(pid_u32)
                            .map(|pi| pi.start_sec)
                            .unwrap_or(0),
                        original_jetsam_priority: None,
                    },
                );
                thermal_frozen += 1;
            }
        }
        if thermal_frozen > 0 {
            write_frozen_state(frozen_state_path, &frozen_guard);
            state.metrics.lock_recover().metrics.freezes_applied += thermal_frozen as u64;
            println!(
                "[thermal] Phase {:?}: froze {} background processes (pre-throttle)",
                thermal_action.phase, thermal_frozen
            );
        }
        drop(frozen_guard);
    } else {
        // Temperature dropped back to Phase2 or below — unfreeze any thermally-frozen PIDs.
        let thermal_frozen_pids: Vec<u32> = {
            let frozen_guard = state.frozen_state.lock_recover();
            frozen_guard
                .iter()
                .filter(|(_, e)| e.source == FreezeSource::ThermalPreThrottle)
                .map(|(&pid, _)| pid)
                .collect()
        };
        if !thermal_frozen_pids.is_empty() {
            let n = unfreeze_pids(thermal_frozen_pids.iter().copied());
            let mut frozen_guard = state.frozen_state.lock_recover();
            for pid in &thermal_frozen_pids {
                frozen_guard.remove(pid);
            }
            write_frozen_state(frozen_state_path, &frozen_guard);
            drop(frozen_guard);
            state.metrics.lock_recover().metrics.unfreezes_applied += n;
            println!("[thermal] Cooled: unfroze {} pre-throttled processes", n);
        }
    }
}
