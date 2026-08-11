//! # Daemon Context-Switch Tick
//!
//! Context-switch burst detector + reactive foreground unfreeze extracted
//! from main.rs (Wave 31). [Fowler 2004] Strangler Fig — pure move.
//!
//! ## Responsibilities
//! - Detect foreground change, push timestamp to ctx_switch_times ring buffer
//! - Reactively unfreeze foreground PID immediately on switch (before process_tree)
//! - Update last_fg_name + GC ctx_switch_times window (300 s)
//!
//! ## Ordering invariant
//! Must run AFTER ForegroundDetector (foreground_app/pid) and Markov tick,
//! and BEFORE process_tree build — reactive unfreeze needs only fg_pid, not
//! the full process family.

use std::collections::VecDeque;
use std::path::Path;
use std::time::{Duration, Instant};

use apollo_engine::engine::daemon_helpers::{
    unfreeze_outcome_events, unfreeze_pids_verified_outcome, write_frozen_state, UnfreezeOutcome,
};
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::decision_ledger::CycleDecisionEvents;
use apollo_engine::engine::lock_ext::LockRecover;

/// Run context-switch burst detector and reactive foreground unfreeze.
///
/// # Parameters
/// - `foreground_app` — current foreground app name (owned, stored as last_fg_name)
/// - `foreground_pid` — current foreground PID (reactive unfreeze target)
/// - `last_fg_name` — previous cycle fg name (detects transition)
/// - `ctx_switch_times` — ring buffer of recent switch timestamps
/// - `state` — SharedState (frozen_state + metrics)
/// - `frozen_state_path` — WAL path for write_frozen_state
pub fn run_ctx_switch_tick(
    foreground_app: Option<String>,
    foreground_pid: Option<u32>,
    last_fg_name: &mut Option<String>,
    ctx_switch_times: &mut VecDeque<Instant>,
    state: &SharedState,
    frozen_state_path: &Path,
    cycle: u64,
) -> CycleDecisionEvents {
    let mut decision_events = CycleDecisionEvents::default();
    let fg_changed =
        foreground_app.is_some() && last_fg_name.is_some() && foreground_app != *last_fg_name;

    if fg_changed {
        ctx_switch_times.push_back(Instant::now());
    }

    if let Some(fg_pid) = foreground_pid {
        let mut frozen_guard = state.frozen_state.lock_recover();
        if let Some(entry) = frozen_guard.get(&fg_pid).cloned() {
            let entries = std::collections::HashMap::from([(fg_pid, entry)]);
            let outcome = unfreeze_pids_verified_outcome(&entries);
            for pid in outcome.forgettable_pids() {
                frozen_guard.remove(&pid);
            }
            write_frozen_state(frozen_state_path, &frozen_guard);
            drop(frozen_guard);
            state.metrics.lock_recover().metrics.unfreezes_applied += outcome.applied_count();
            decision_events.extend_buffer(&foreground_unfreeze_events(cycle, &outcome));
        }
    }

    *last_fg_name = foreground_app;
    let cutoff = Instant::now() - Duration::from_secs(300);
    ctx_switch_times.retain(|t| *t > cutoff);
    decision_events
}

fn foreground_unfreeze_events(cycle: u64, outcome: &UnfreezeOutcome) -> CycleDecisionEvents {
    unfreeze_outcome_events(
        "unfreeze:foreground_switch",
        "context-switch-recovery",
        cycle,
        outcome,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome;

    #[test]
    fn foreground_sigcont_receipts_preserve_exact_dispositions() {
        let outcome = UnfreezeOutcome {
            applied_pids: vec![41],
            stale_pids: vec![42],
            failed_pids: vec![43],
        };

        let events = foreground_unfreeze_events(12, &outcome);

        assert_eq!(events.len(), 3);
        assert_eq!(events.as_slice()[0].proposal.target, "pid:41");
        assert_eq!(
            events.as_slice()[0].outcome,
            ActuatorDecisionOutcome::Reverted
        );
        assert_eq!(events.as_slice()[1].proposal.target, "pid:42");
        assert_eq!(
            events.as_slice()[1].outcome,
            ActuatorDecisionOutcome::Blocked
        );
        assert_eq!(events.as_slice()[2].proposal.target, "pid:43");
        assert_eq!(
            events.as_slice()[2].outcome,
            ActuatorDecisionOutcome::Failed
        );
    }
}
