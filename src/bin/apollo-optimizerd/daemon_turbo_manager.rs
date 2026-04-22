//! # Daemon Turbo Manager
//!
//! Display-Off Turbo: Android Doze-like power management for the daemon.
//!
//! When the display turns off for more than a configurable dwell period (2s on battery,
//! 5s on AC), Apollo freezes all non-essential background processes to cut power and
//! memory pressure. When the display turns back on, all frozen processes are unfrozen
//! immediately.
//!
//! This module contains the per-cycle turbo tick extracted from the main daemon loop.
//! [Nygard 2018] bulkhead pattern: bound the blast radius of state transitions.

use std::path::Path;

use apollo_optimizer::collector::SystemCollector;
use apollo_optimizer::engine::background_collectors::PressureCollector;
use apollo_optimizer::engine::daemon_helpers::{unfreeze_pids_verified, write_frozen_state};
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::display_turbo::{DisplayTurbo, TurboAction};
use apollo_optimizer::engine::foreground::ForegroundDetector;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::process_identity::ProcessIdentity;
use apollo_optimizer::engine::safety::is_protected_name;
use apollo_optimizer::engine::stability_oracle::StabilityOracle;
use apollo_optimizer::engine::types::{FreezeSource, FrozenEntry};
use chrono::Utc;

/// Run one display-off turbo tick.
///
/// Sets the dwell sensitivity based on power source, then handles the result of
/// `DisplayTurbo::tick()`:
/// - `ActivateTurbo`: freeze all non-essential background processes (SIGSTOP).
/// - `DeactivateTurbo`: SIGCONT all previously-turbo-frozen PIDs (A-B-A verified).
/// - `None`: record absence of jank for StabilityOracle EMA.
///
/// # Parameters
/// - `display_turbo` — Display-off turbo state machine.
/// - `state` — Shared daemon state (policy, frozen_state, metrics).
/// - `fg_detector` — Current foreground process detector.
/// - `collector` — System process collector (provides live process list).
/// - `pressure_collector` — Background pressure collector (for `pressure_at_freeze`).
/// - `frozen_state_path` — Path to `frozen_state.json` for atomic persistence.
/// - `stability_oracle` — Records display jank events for RL reward signal.
/// - `is_on_battery` — If true, use shorter dwell (2s) for faster turbo activation.
#[allow(clippy::too_many_arguments)]
pub fn run_turbo_tick(
    display_turbo: &mut DisplayTurbo,
    state: &SharedState,
    fg_detector: &ForegroundDetector,
    collector: &SystemCollector,
    pressure_collector: &PressureCollector,
    frozen_state_path: &Path,
    stability_oracle: &mut StabilityOracle,
    is_on_battery: bool,
) {
    // Battery-aware dwell: on battery shorten to 2s so turbo activates faster
    // → more aggressive power savings when user steps away.
    display_turbo.set_dwell_secs(if is_on_battery { 2 } else { 5 });

    match display_turbo.tick() {
        TurboAction::ActivateTurbo => {
            // Freeze non-essential background processes.
            let policy_protected = state
                .policy
                .lock_recover()
                .learned_policy
                .protected_patterns
                .clone();
            let fg_pid = fg_detector.detect().pid();
            let mut frozen_guard = state.frozen_state.lock_recover();
            let mut turbo_frozen = 0u32;
            let max_freeze = display_turbo.max_freeze_count();

            for (pid, process) in collector.system().processes() {
                let pid_u32 = pid.as_u32();
                let name = process.name().to_string();
                // Never freeze: foreground, OS/infra/dev-runtime/policy-protected,
                // or Apollo itself. [Saltzer & Kaashoek 2009] Complete Mediation.
                if Some(pid_u32) == fg_pid
                    || is_protected_name(&name)
                    || policy_protected.iter().any(|p| name.contains(p.as_str()))
                    || name == "apollo-optimizerd"
                    || frozen_guard.contains_key(&pid_u32)
                {
                    continue;
                }
                if turbo_frozen as usize >= max_freeze {
                    break;
                }
                if unsafe { libc::kill(pid_u32 as i32, libc::SIGSTOP) } == 0 {
                    display_turbo.record_turbo_freeze(pid_u32);
                    frozen_guard.insert(
                        pid_u32,
                        FrozenEntry {
                            frozen_at: Utc::now(),
                            source: FreezeSource::MainLoop,
                            pressure_at_freeze: pressure_collector.latest().memory_pressure,
                            process_name: Some(name.clone()),
                            // A3: capture start_sec for identity check on unfreeze.
                            start_sec: ProcessIdentity::from_pid(pid_u32)
                                .map(|pi| pi.start_sec)
                                .unwrap_or(0),
                            original_jetsam_priority: None,
                        },
                    );
                    turbo_frozen += 1;
                }
            }
            write_frozen_state(frozen_state_path, &frozen_guard);
            drop(frozen_guard);
            state.metrics.lock_recover().metrics.freezes_applied += turbo_frozen as u64;
        }

        TurboAction::DeactivateTurbo {
            unfreeze_pids: pids,
        } => {
            // A-B-A defense: verify PID identity before SIGCONT.
            // Lock frozen_guard first so we can read start_sec for each PID
            // before signalling. PIDs recycled between the display-off freeze
            // and the display-on thaw are skipped.
            // [Saltzer & Kaashoek 2009] §3.3 Complete Mediation.
            let mut frozen_guard = state.frozen_state.lock_recover();
            let entries_to_unfreeze: std::collections::HashMap<u32, FrozenEntry> = pids
                .iter()
                .filter_map(|&pid| frozen_guard.get(&pid).map(|e| (pid, e.clone())))
                .collect();
            let unfreeze_count = unfreeze_pids_verified(&entries_to_unfreeze);
            for pid in &pids {
                frozen_guard.remove(pid);
            }
            write_frozen_state(frozen_state_path, &frozen_guard);
            drop(frozen_guard);
            // Clear turbo internal state so stale PIDs don't block re-freeze on
            // the next display-off cycle.
            display_turbo.clear_frozen();
            state.metrics.lock_recover().metrics.unfreezes_applied += unfreeze_count;
            // Jank is recorded only when we ACTUALLY froze processes during turbo.
            // Pure display on/off cycles with zero freezes are normal user behavior.
            // [Nielsen 1993] usability heuristics — count only user-perceptible impact.
            stability_oracle.record_display_jank(unfreeze_count > 0);
        }

        TurboAction::None => {
            stability_oracle.record_display_jank(false);
        }
    }
}
