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

use apollo_engine::engine::background_collectors::PressureCollector;
use apollo_engine::engine::daemon_helpers::{
    unfreeze_pids_verified_outcome, write_frozen_state, UnfreezeOutcome,
};
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::decision_ledger::{
    ActuatorDecisionEvent, ActuatorDecisionOutcome, CycleDecisionEvents,
};
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::mach_qos::SchedulingTier;

/// Maximum PIDs to SIGCONT per cycle under normal conditions.
const WAKE_UNFREEZE_BATCH: usize = 5;

#[derive(Debug, Default)]
pub struct WakeUnfreezeOutput {
    pub decision_events: CycleDecisionEvents,
}

fn wake_recovery_event(
    pid: u32,
    cycle: u64,
    outcome: ActuatorDecisionOutcome,
    detail: impl Into<String>,
) -> ActuatorDecisionEvent {
    ActuatorDecisionEvent::local(
        "unfreeze:wake_recovery",
        format!("pid:{pid}"),
        cycle,
        outcome,
        "wake-recovery",
        detail,
    )
}

fn wake_qos_restore_event(
    pid: u32,
    cycle: u64,
    success: bool,
    mutated: bool,
    error: Option<&str>,
) -> ActuatorDecisionEvent {
    let (outcome, detail) = if mutated {
        (
            ActuatorDecisionOutcome::Reverted,
            "wake recovery restored normal Mach tier".to_string(),
        )
    } else if success {
        (
            ActuatorDecisionOutcome::NoOp,
            "wake recovery Mach tier already normal".to_string(),
        )
    } else {
        (
            ActuatorDecisionOutcome::Failed,
            error.unwrap_or("Mach tier restore failed").to_string(),
        )
    };
    ActuatorDecisionEvent::local(
        "thread_qos:wake_recovery",
        format!("pid:{pid}"),
        cycle,
        outcome,
        "wake-recovery",
        detail,
    )
}

pub fn recovery_unfreeze_events(
    outcome: &UnfreezeOutcome,
    cycle: u64,
    action_key: &str,
    source: &str,
) -> CycleDecisionEvents {
    let mut events = CycleDecisionEvents::default();
    for &pid in &outcome.applied_pids {
        events.push(ActuatorDecisionEvent::local(
            action_key,
            format!("pid:{pid}"),
            cycle,
            ActuatorDecisionOutcome::Reverted,
            source,
            "verified SIGCONT applied",
        ));
    }
    for &pid in &outcome.stale_pids {
        events.push(ActuatorDecisionEvent::local(
            action_key,
            format!("pid:{pid}"),
            cycle,
            ActuatorDecisionOutcome::Blocked,
            source,
            "PID identity stale; signal suppressed",
        ));
    }
    for &pid in &outcome.failed_pids {
        events.push(ActuatorDecisionEvent::local(
            action_key,
            format!("pid:{pid}"),
            cycle,
            ActuatorDecisionOutcome::Failed,
            source,
            "SIGCONT failed",
        ));
    }
    events
}

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
    cycle: u64,
) -> WakeUnfreezeOutput {
    let mut output = WakeUnfreezeOutput::default();
    if wake_unfreeze_queue.is_empty() {
        return output;
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
    let outcome = {
        let mut frozen_guard = state.frozen_state.lock_recover();
        let entries: std::collections::HashMap<u32, apollo_engine::engine::types::FrozenEntry> =
            batch
                .iter()
                .filter_map(|&pid| frozen_guard.get(&pid).map(|e| (pid, e.clone())))
                .collect();
        let outcome = unfreeze_pids_verified_outcome(&entries);
        for pid in outcome.forgettable_pids() {
            frozen_guard.remove(&pid);
        }
        write_frozen_state(frozen_state_path, &frozen_guard);
        outcome
    };
    for pid in outcome.failed_pids.iter().rev() {
        wake_unfreeze_queue.push_front(*pid);
    }
    let applied = outcome.applied_count();
    for pid in &outcome.applied_pids {
        output.decision_events.push(wake_recovery_event(
            *pid,
            cycle,
            ActuatorDecisionOutcome::Reverted,
            "verified SIGCONT applied",
        ));
    }
    for pid in &outcome.stale_pids {
        output.decision_events.push(wake_recovery_event(
            *pid,
            cycle,
            ActuatorDecisionOutcome::Blocked,
            "stale process identity",
        ));
        output.decision_events.push(ActuatorDecisionEvent::local(
            "thread_qos:wake_recovery",
            format!("pid:{pid}"),
            cycle,
            ActuatorDecisionOutcome::Blocked,
            "wake-recovery",
            "Mach tier restore skipped for stale process identity",
        ));
    }
    for pid in &outcome.failed_pids {
        output.decision_events.push(wake_recovery_event(
            *pid,
            cycle,
            ActuatorDecisionOutcome::Failed,
            "SIGCONT failed and was requeued",
        ));
        output.decision_events.push(ActuatorDecisionEvent::local(
            "thread_qos:wake_recovery",
            format!("pid:{pid}"),
            cycle,
            ActuatorDecisionOutcome::Blocked,
            "wake-recovery",
            "Mach tier restore skipped because SIGCONT failed",
        ));
    }
    if applied > 0 {
        let mut metrics = state.metrics.lock_recover();
        metrics.metrics.post_wake_defensive_unfreezes += applied;
        metrics.metrics.unfreezes_applied += applied;
        metrics.metrics.throttle_reverted += applied;
    }

    // Mark thawed PIDs in cooldown to prevent gate_e re-freeze oscillation.
    // [Nygard 2018] §8.5 — circuit breaker hold-down after recovery.
    {
        let mut cooldown = state.freeze_cooldown.lock_recover();
        for pid in &outcome.applied_pids {
            cooldown.mark_thawed(*pid);
        }
    }

    // Restore Mach QoS from Background (E-cores) → Normal so
    // processes resume on P-cores. Wake unfreeze is the highest-
    // urgency thaw path (user just returned to desktop), so P-core
    // routing is critical for perceived responsiveness.
    {
        let mut qos = state.mach_qos.lock_recover();
        for pid in &outcome.applied_pids {
            let qos_outcome = qos.set_tier(*pid, SchedulingTier::Normal);
            output.decision_events.push(wake_qos_restore_event(
                *pid,
                cycle,
                qos_outcome.success,
                qos_outcome.mutated,
                qos_outcome.error.as_deref(),
            ));
        }
    }

    // Record actual-SIGCONT T0 for unfreeze_decay ODE τ learning.
    wake_thaw_pids.extend_from_slice(&outcome.applied_pids);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome;

    #[test]
    fn wake_recovery_event_is_a_reverted_unfreeze() {
        let event = wake_recovery_event(42, 8, ActuatorDecisionOutcome::Reverted, "SIGCONT");

        assert_eq!(event.proposal.action_key, "unfreeze:wake_recovery");
        assert_eq!(event.proposal.target, "pid:42");
        assert_eq!(event.outcome, ActuatorDecisionOutcome::Reverted);
    }

    #[test]
    fn wake_qos_restore_has_an_independent_exact_disposition() {
        let reverted = wake_qos_restore_event(42, 8, true, true, None);
        let noop = wake_qos_restore_event(43, 8, true, false, None);
        let failed = wake_qos_restore_event(44, 8, false, false, Some("mach denied"));

        assert_eq!(reverted.proposal.action_key, "thread_qos:wake_recovery");
        assert_eq!(reverted.outcome, ActuatorDecisionOutcome::Reverted);
        assert_eq!(noop.outcome, ActuatorDecisionOutcome::NoOp);
        assert_eq!(failed.outcome, ActuatorDecisionOutcome::Failed);
        assert_eq!(failed.proposal.target, "pid:44");
    }

    #[test]
    fn recovery_outcome_closes_applied_stale_and_failed_pids() {
        let outcome = apollo_engine::engine::daemon_helpers::UnfreezeOutcome {
            applied_pids: vec![1],
            stale_pids: vec![2],
            failed_pids: vec![3],
        };

        let events = recovery_unfreeze_events(
            &outcome,
            11,
            "unfreeze:deadlock_recovery",
            "deadlock-recovery",
        );

        assert_eq!(events.len(), 3);
        assert_eq!(
            events.as_slice()[0].outcome,
            ActuatorDecisionOutcome::Reverted
        );
        assert_eq!(
            events.as_slice()[1].outcome,
            ActuatorDecisionOutcome::Blocked
        );
        assert_eq!(
            events.as_slice()[2].outcome,
            ActuatorDecisionOutcome::Failed
        );
    }
}
