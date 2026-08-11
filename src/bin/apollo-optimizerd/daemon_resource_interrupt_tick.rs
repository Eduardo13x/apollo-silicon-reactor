//! Main-loop executor for bounded resource-interrupt proposals.
//!
//! The resource sentinel is an observer only. This module is the sole bridge
//! from its typed proposals to Apollo's existing broker and mediator effectors.

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::Ordering;

use apollo_engine::engine::actuation_broker::{ActuationBroker, ActuationRequest, BrokerMode};
use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::daemon_helpers::write_frozen_state;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::decision_ledger::{
    ActuatorDecisionEvent, ActuatorDecisionOutcome, CycleDecisionEvents,
};
use apollo_engine::engine::effect_ledger::{
    forget_global_if_justification, is_global_owner, is_global_tracked, record_global,
    AppliedEffect, DEFAULT_TTL,
};
use apollo_engine::engine::io_tiering::{apply_io_tier, IOTier};
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::mediator::{
    mediate, BlockReason, Effect, MachPolicyEffector, MachPolicyKind, PreCondition, SignalEffector,
};
use apollo_engine::engine::process_identity::ProcessIdentity;
use apollo_engine::engine::safety::is_protected_name;
use apollo_engine::engine::thermal_interrupt::{
    ResourceInterruptActuationWindow, ResourceInterruptProposal, ResourceInterruptProposalBatch,
    ResourceInterruptUnfreezeReason,
};
use apollo_engine::engine::types::{CapabilityReport, FreezeSource, FrozenEntry, RootAction};
use chrono::Utc;

const MIGRATION_OWNER: &str = "thermal: E-core migration";

pub struct ResourceInterruptExecutor<'a> {
    pub state: &'a SharedState,
    pub caps: &'a CapabilityReport,
    pub journal_path: &'a Path,
    pub frozen_state_path: &'a Path,
    pub dry_run: bool,
    pub cycle: u64,
    pub memory_pressure: f64,
}

impl ResourceInterruptExecutor<'_> {
    pub fn execute_batch(
        &mut self,
        window: &ResourceInterruptActuationWindow,
        batch: ResourceInterruptProposalBatch,
    ) -> CycleDecisionEvents {
        window.resolve(batch, self.cycle, |proposal| self.execute_one(proposal))
    }

    fn execute_one(&mut self, proposal: ResourceInterruptProposal) -> CycleDecisionEvents {
        if let Some(events) = self.authority_denial(&proposal) {
            return events;
        }
        match proposal {
            ResourceInterruptProposal::MigrateToBackground {
                pid,
                name,
                start_sec,
                start_usec,
            } => self.migrate(pid, name, start_sec, start_usec),
            ResourceInterruptProposal::Freeze {
                pid,
                name,
                start_sec,
                start_usec,
            } => self.freeze(pid, name, start_sec, start_usec),
            ResourceInterruptProposal::RestoreScheduling {
                pid,
                name,
                start_sec,
                start_usec,
            } => self.restore_scheduling(pid, name, start_sec, start_usec),
            ResourceInterruptProposal::Unfreeze {
                pid,
                name,
                start_sec,
                start_usec,
                reason,
            } => self.unfreeze(pid, name, start_sec, start_usec, reason),
        }
    }

    fn authority_denial(
        &self,
        proposal: &ResourceInterruptProposal,
    ) -> Option<CycleDecisionEvents> {
        match ActuationBroker::from_runtime(self.caps, self.dry_run).mode() {
            BrokerMode::InProcessRoot => None,
            BrokerMode::DryRun => Some(self.single(
                proposal.action_key(),
                proposal.pid(),
                ActuatorDecisionOutcome::NoOp,
                "dry-run: resource-interrupt proposal not executed",
            )),
            BrokerMode::Denied => Some(self.single(
                proposal.action_key(),
                proposal.pid(),
                ActuatorDecisionOutcome::Rejected,
                "actuation broker denied: root authority mismatch",
            )),
        }
    }

    fn single(
        &self,
        action_key: &str,
        pid: u32,
        outcome: ActuatorDecisionOutcome,
        detail: impl Into<String>,
    ) -> CycleDecisionEvents {
        let mut events = CycleDecisionEvents::default();
        events.push(ActuatorDecisionEvent::local(
            action_key,
            format!("pid:{pid}"),
            self.cycle,
            outcome,
            "resource-interrupt-main-loop",
            detail,
        ));
        events
    }

    fn checked_identity(
        &self,
        action_key: &str,
        pid: u32,
        name: &str,
        start_sec: u64,
        start_usec: u64,
    ) -> Result<ProcessIdentity, CycleDecisionEvents> {
        let Some(identity) = ProcessIdentity::from_pid(pid) else {
            return Err(self.single(
                action_key,
                pid,
                ActuatorDecisionOutcome::NoOp,
                "process exited before proposal execution",
            ));
        };
        let expected_name = (!name.is_empty()).then_some(name);
        if !identity.matches(expected_name, start_sec, start_usec) {
            return Err(self.single(
                action_key,
                pid,
                ActuatorDecisionOutcome::Blocked,
                "stale process identity before proposal execution",
            ));
        }
        Ok(identity)
    }

    fn migrate(
        &mut self,
        pid: u32,
        name: String,
        start_sec: u64,
        start_usec: u64,
    ) -> CycleDecisionEvents {
        let action_key = if self.caps.can_taskpolicy {
            "thermal_interrupt:mach_tier_background"
        } else {
            "thermal_interrupt:darwin_bg_enable"
        };
        if self
            .state
            .resource_interrupt
            .interrupt_migrated_pids
            .lock_recover()
            .contains(&pid)
        {
            return self.single(
                action_key,
                pid,
                ActuatorDecisionOutcome::NoOp,
                "resource-interrupt scheduling effect already tracked",
            );
        }
        if self.state.frozen_state.lock_recover().contains_key(&pid) {
            return self.single(
                action_key,
                pid,
                ActuatorDecisionOutcome::NoOp,
                "freeze takes precedence over scheduling migration",
            );
        }
        let identity = match self.checked_identity(action_key, pid, &name, start_sec, start_usec) {
            Ok(identity) => identity,
            Err(events) => return events,
        };
        if is_protected_name(&identity.name)
            || apollo_engine::engine::process_identity::is_apple_platform_process(pid)
        {
            return self.single(
                action_key,
                pid,
                ActuatorDecisionOutcome::Blocked,
                "protected process at main-loop execution boundary",
            );
        }

        let (outcome, detail, effect) = if self.caps.can_taskpolicy {
            let effect = Effect::SetMachPolicy {
                pid,
                start_sec: identity.start_sec,
                policy: MachPolicyKind::Background,
            };
            let precondition = PreCondition {
                pid_identity: Some((pid, identity.start_sec)),
                require_not_protected: true,
                ..PreCondition::default()
            };
            let effector = MachPolicyEffector::new(self.state.mach_qos.clone());
            match mediate(&effect, &precondition, &effector) {
                Ok(receipt) if receipt.applied_count > 0 => (
                    ActuatorDecisionOutcome::Applied,
                    "Mach background tier applied".to_string(),
                    Some(AppliedEffect::MachTier { pid }),
                ),
                Ok(_) => (
                    ActuatorDecisionOutcome::NoOp,
                    "Mach background tier already active".to_string(),
                    None,
                ),
                Err(error) => (
                    mediator_outcome(&error),
                    format!("Mach background tier failed: {error:?}"),
                    None,
                ),
            }
        } else if apply_io_tier(pid, IOTier::Passive) {
            (
                ActuatorDecisionOutcome::Applied,
                "Darwin background priority applied".to_string(),
                Some(AppliedEffect::DarwinBg { pid }),
            )
        } else {
            (
                ActuatorDecisionOutcome::Failed,
                format!("setpriority failed: {}", std::io::Error::last_os_error()),
                None,
            )
        };

        if let Some(effect) = effect {
            record_global(effect, DEFAULT_TTL, identity.start_sec, MIGRATION_OWNER);
            self.state
                .resource_interrupt
                .interrupt_migrated_pids
                .lock_recover()
                .insert(pid);
            self.state
                .resource_interrupt
                .total_migrated
                .fetch_add(1, Ordering::Relaxed);
        }
        self.single(action_key, pid, outcome, detail)
    }

    fn freeze(
        &mut self,
        pid: u32,
        name: String,
        start_sec: u64,
        start_usec: u64,
    ) -> CycleDecisionEvents {
        const ACTION_KEY: &str = "thermal_interrupt:freeze";
        let identity = match self.checked_identity(ACTION_KEY, pid, &name, start_sec, start_usec) {
            Ok(identity) => identity,
            Err(events) => return events,
        };
        if self.state.frozen_state.lock_recover().contains_key(&pid) {
            return self.single(
                ACTION_KEY,
                pid,
                ActuatorDecisionOutcome::NoOp,
                "process already frozen",
            );
        }

        let mut frozen: HashSet<u32> = self
            .state
            .frozen_state
            .lock_recover()
            .keys()
            .copied()
            .collect();
        let (learned_protected, learned_interactive) = {
            let policy = self.state.policy.lock_recover();
            (
                policy.learned_policy.protected_patterns.clone(),
                policy.learned_policy.interactive_patterns.clone(),
            )
        };

        // Sentinel freeze historically meant one SIGSTOP. Keep that exact
        // surface while using the broker's safety, identity, and saga path;
        // scheduling migration has its own independently receipted proposal.
        let mut freeze_caps = self.caps.clone();
        freeze_caps.can_memorystatus = false;
        freeze_caps.can_taskpolicy = false;
        let execution =
            ActuationBroker::from_runtime(&freeze_caps, false).execute(ActuationRequest {
                actions: vec![RootAction::FreezeProcess {
                    pid,
                    name: identity.name.clone(),
                    reason: "resource interrupt emergency".to_string(),
                    decision_reason: DecisionReason::CriticalBypass,
                    start_sec: identity.start_sec,
                    start_usec: identity.start_usec,
                }],
                caps: &freeze_caps,
                journal_path: self.journal_path,
                frozen: &mut frozen,
                learned_protected: learned_protected.as_slice(),
                learned_interactive: learned_interactive.as_slice(),
                qos_mgr: None,
                async_commands: None,
                memory_pressure: self.memory_pressure,
                thrashing_score: 0.0,
                coalition_guard: None,
                cpu_pegged_fraction: 0.0,
            });
        let broker_event = execution.outcomes.decision_events.as_slice().first();
        let outcome = broker_event
            .map(|event| event.outcome)
            .unwrap_or(ActuatorDecisionOutcome::Failed);
        let detail = broker_event
            .map(|event| event.detail.clone())
            .unwrap_or_else(|| "actuation broker returned no freeze receipt".to_string());

        if execution.outcomes.newly_frozen_pids.contains(&pid) {
            let original_jetsam_priority = execution
                .outcomes
                .newly_frozen_identity
                .iter()
                .find(|(frozen_pid, _, _)| *frozen_pid == pid)
                .and_then(|(_, _, priority)| *priority);
            self.state
                .resource_interrupt
                .interrupt_frozen_pids
                .lock_recover()
                .insert(pid);
            let mut frozen_state = self.state.frozen_state.lock_recover();
            frozen_state.insert(
                pid,
                FrozenEntry {
                    frozen_at: Utc::now(),
                    source: FreezeSource::Sentinel,
                    pressure_at_freeze: self.memory_pressure,
                    process_name: Some(identity.name),
                    start_sec: identity.start_sec,
                    original_jetsam_priority,
                },
            );
            write_frozen_state(self.frozen_state_path, &frozen_state);
            self.state
                .resource_interrupt
                .total_frozen
                .fetch_add(1, Ordering::Relaxed);
            let mut metrics = self.state.metrics.lock_recover();
            metrics.metrics.freezes_applied = metrics.metrics.freezes_applied.saturating_add(1);
        }

        self.single(ACTION_KEY, pid, outcome, detail)
    }

    fn restore_scheduling(
        &mut self,
        pid: u32,
        name: String,
        start_sec: u64,
        start_usec: u64,
    ) -> CycleDecisionEvents {
        let mach = AppliedEffect::MachTier { pid };
        let darwin = AppliedEffect::DarwinBg { pid };
        let mach_owned = is_global_owner(&mach, MIGRATION_OWNER);
        let darwin_owned = is_global_owner(&darwin, MIGRATION_OWNER);
        let action_key = if mach_owned || is_global_tracked(&mach) {
            "thermal_interrupt:mach_tier_restore"
        } else if darwin_owned || is_global_tracked(&darwin) {
            "thermal_interrupt:darwin_bg_restore"
        } else {
            "thermal_interrupt:qos_restore"
        };

        if !self
            .state
            .resource_interrupt
            .interrupt_migrated_pids
            .lock_recover()
            .contains(&pid)
        {
            return self.single(
                action_key,
                pid,
                ActuatorDecisionOutcome::NoOp,
                "scheduling migration already resolved",
            );
        }

        let identity = match ProcessIdentity::from_pid(pid) {
            Some(identity)
                if identity.matches(
                    (!name.is_empty()).then_some(name.as_str()),
                    start_sec,
                    start_usec,
                ) =>
            {
                identity
            }
            Some(_) => {
                self.finish_migration(pid, &mach, &darwin, mach_owned, darwin_owned);
                return self.single(
                    action_key,
                    pid,
                    ActuatorDecisionOutcome::Blocked,
                    "stale process identity; old scheduling ownership discarded",
                );
            }
            None => {
                self.finish_migration(pid, &mach, &darwin, mach_owned, darwin_owned);
                return self.single(
                    action_key,
                    pid,
                    ActuatorDecisionOutcome::NoOp,
                    "process exited; scheduling ownership discarded",
                );
            }
        };

        let (outcome, detail, resolved) = if mach_owned {
            let effect = Effect::SetMachPolicy {
                pid,
                start_sec: identity.start_sec,
                policy: MachPolicyKind::Default,
            };
            let precondition = PreCondition {
                pid_identity: Some((pid, identity.start_sec)),
                ..PreCondition::default()
            };
            let effector = MachPolicyEffector::new(self.state.mach_qos.clone());
            match mediate(&effect, &precondition, &effector) {
                Ok(receipt) if receipt.applied_count > 0 => (
                    ActuatorDecisionOutcome::Reverted,
                    "Mach tier restored to normal".to_string(),
                    true,
                ),
                Ok(_) => (
                    ActuatorDecisionOutcome::NoOp,
                    "Mach tier already normal".to_string(),
                    true,
                ),
                Err(error) => (
                    mediator_outcome(&error),
                    format!("Mach normal-tier restoration failed: {error:?}"),
                    false,
                ),
            }
        } else if darwin_owned {
            if apply_io_tier(pid, IOTier::Standard) {
                (
                    ActuatorDecisionOutcome::Reverted,
                    "Darwin background priority cleared".to_string(),
                    true,
                )
            } else {
                (
                    ActuatorDecisionOutcome::Failed,
                    format!(
                        "setpriority restore failed: {}",
                        std::io::Error::last_os_error()
                    ),
                    false,
                )
            }
        } else if is_global_tracked(&mach) || is_global_tracked(&darwin) {
            (
                ActuatorDecisionOutcome::Blocked,
                "migration ownership transferred; restore suppressed".to_string(),
                true,
            )
        } else {
            (
                ActuatorDecisionOutcome::NoOp,
                "migration already resolved by TTL reconciliation".to_string(),
                true,
            )
        };

        if resolved {
            self.finish_migration(pid, &mach, &darwin, mach_owned, darwin_owned);
        }
        self.single(action_key, pid, outcome, detail)
    }

    fn finish_migration(
        &self,
        pid: u32,
        mach: &AppliedEffect,
        darwin: &AppliedEffect,
        mach_owned: bool,
        darwin_owned: bool,
    ) {
        self.state
            .resource_interrupt
            .interrupt_migrated_pids
            .lock_recover()
            .remove(&pid);
        if mach_owned {
            forget_global_if_justification(mach, MIGRATION_OWNER);
        }
        if darwin_owned {
            forget_global_if_justification(darwin, MIGRATION_OWNER);
        }
    }

    fn unfreeze(
        &mut self,
        pid: u32,
        name: String,
        start_sec: u64,
        start_usec: u64,
        reason: ResourceInterruptUnfreezeReason,
    ) -> CycleDecisionEvents {
        let action_key = match reason {
            ResourceInterruptUnfreezeReason::Foreground => "thermal_interrupt:foreground_sigcont",
            ResourceInterruptUnfreezeReason::Recovery => "thermal_interrupt:sigcont_recovery",
        };
        let entry = self.state.frozen_state.lock_recover().get(&pid).cloned();
        let interrupt_owned = self
            .state
            .resource_interrupt
            .interrupt_frozen_pids
            .lock_recover()
            .contains(&pid);
        if entry.is_none() && !interrupt_owned {
            return self.single(
                action_key,
                pid,
                ActuatorDecisionOutcome::NoOp,
                "freeze ownership already resolved",
            );
        }
        if reason == ResourceInterruptUnfreezeReason::Recovery
            && entry
                .as_ref()
                .is_some_and(|entry| entry.source != FreezeSource::Sentinel)
        {
            self.state
                .resource_interrupt
                .interrupt_frozen_pids
                .lock_recover()
                .remove(&pid);
            return self.single(
                action_key,
                pid,
                ActuatorDecisionOutcome::Blocked,
                "freeze ownership transferred; SIGCONT suppressed",
            );
        }

        let expected_name = entry
            .as_ref()
            .and_then(|entry| entry.process_name.as_deref())
            .filter(|name| !name.is_empty())
            .or_else(|| (!name.is_empty()).then_some(name.as_str()));
        let expected_start = entry
            .as_ref()
            .map(|entry| entry.start_sec)
            .filter(|start| *start > 0)
            .unwrap_or(start_sec);
        let identity = match ProcessIdentity::from_pid(pid) {
            Some(identity) if identity.matches(expected_name, expected_start, start_usec) => {
                identity
            }
            Some(_) => {
                self.finish_unfreeze(pid, entry.as_ref());
                return self.single(
                    action_key,
                    pid,
                    ActuatorDecisionOutcome::Blocked,
                    "stale process identity; freeze ownership discarded",
                );
            }
            None => {
                self.finish_unfreeze(pid, entry.as_ref());
                return self.single(
                    action_key,
                    pid,
                    ActuatorDecisionOutcome::NoOp,
                    "process exited; freeze ownership discarded",
                );
            }
        };

        let effect = Effect::SigCont {
            pid,
            start_sec: identity.start_sec,
        };
        let precondition = PreCondition {
            pid_identity: Some((pid, identity.start_sec)),
            ..PreCondition::default()
        };
        match mediate(&effect, &precondition, &SignalEffector) {
            Ok(receipt) if receipt.applied_count > 0 => {
                self.finish_unfreeze(pid, entry.as_ref());
                self.state
                    .resource_interrupt
                    .total_recoveries
                    .fetch_add(1, Ordering::Relaxed);
                let mut metrics = self.state.metrics.lock_recover();
                metrics.metrics.unfreezes_applied =
                    metrics.metrics.unfreezes_applied.saturating_add(1);
                self.single(
                    action_key,
                    pid,
                    ActuatorDecisionOutcome::Reverted,
                    "verified SIGCONT applied",
                )
            }
            Ok(_) => self.single(
                action_key,
                pid,
                ActuatorDecisionOutcome::NoOp,
                "SIGCONT produced no mutation",
            ),
            Err(error) => self.single(
                action_key,
                pid,
                mediator_outcome(&error),
                format!("SIGCONT failed: {error:?}"),
            ),
        }
    }

    fn finish_unfreeze(&self, pid: u32, entry: Option<&FrozenEntry>) {
        self.state
            .resource_interrupt
            .interrupt_frozen_pids
            .lock_recover()
            .remove(&pid);
        if entry.is_some() {
            let mut frozen_state = self.state.frozen_state.lock_recover();
            frozen_state.remove(&pid);
            write_frozen_state(self.frozen_state_path, &frozen_state);
        }
    }
}

fn mediator_outcome(error: &BlockReason) -> ActuatorDecisionOutcome {
    match error {
        BlockReason::NoOpDetected => ActuatorDecisionOutcome::NoOp,
        BlockReason::OsError { .. } => ActuatorDecisionOutcome::Failed,
        BlockReason::IdentityMismatch { .. }
        | BlockReason::PreconditionViolated { .. }
        | BlockReason::ProcessProtected { .. }
        | BlockReason::BudgetExhausted { .. }
        | BlockReason::GateRejected { .. } => ActuatorDecisionOutcome::Blocked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syscall_failure_maps_to_failed_receipt() {
        let error = BlockReason::OsError {
            errno: libc::EPERM,
            context: "test".to_string(),
        };
        assert_eq!(mediator_outcome(&error), ActuatorDecisionOutcome::Failed);
    }

    #[test]
    fn identity_veto_maps_to_blocked_receipt() {
        let error = BlockReason::IdentityMismatch {
            pid: 42,
            expected_start_sec: 1,
        };
        assert_eq!(mediator_outcome(&error), ActuatorDecisionOutcome::Blocked);
    }
}
