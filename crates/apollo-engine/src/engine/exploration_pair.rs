//! The one object that crosses from the exploration authority into the Lab.
//!
//! Everything the Lab needs to count an experiment as evidence lives here, and
//! nothing else crosses. The Lab does not reconstruct a pair from loose
//! endpoints — "I saw a treatment, later I found a compatible control" is
//! precisely the accidental matching this type exists to make impossible.
//!
//! # Identity
//!
//! Two levels, deliberately distinct:
//!
//! - [`ExperimentId`] names the **experiment**. It is minted once, before
//!   randomisation, and both arms carry it.
//! - `ProbeCorrelation` names **one arm**. It was already per-assignment in the
//!   scheduler (`arm_sequence` and `next_correlation` advance together), so
//!   reusing it as pair identity would have named an arm and claimed it named a
//!   pair — the same class of defect as a gate counting exposure.
//!
//! ```text
//! Experiment E42
//!   treatment  correlation C101
//!   control    correlation C102
//! ```
//!
//! Never `C101` on both sides. [`CompletedExplorationPair::assemble`] refuses
//! it, so the invalid shape cannot be constructed at all.
//!
//! # Authority
//!
//! ```text
//! scheduler  decides randomisation and holdout
//! ledger     records what happened
//! Lab        consumes a certified pair, read-only
//! gate       consumes evidence
//! ```
//!
//! The gate never learns how the experiment was obtained. That separation is
//! what stops a phase from being asked for evidence it has no way to produce.

use serde::{Deserialize, Serialize};

use crate::engine::exploration_scheduler::{ActionClass, ExplorationArm, ProbeCorrelation};
use crate::engine::telemetry_medallion::ActuatorFamily;

/// Identity of a whole experiment: one randomisation decision, two arms.
///
/// Carries the boot epoch in its high bits so an id minted before a restart can
/// never collide with one minted after. Without that, a late endpoint from a
/// previous boot could close an experiment on this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExperimentId(pub u128);

impl ExperimentId {
    pub fn new(boot_epoch: u64, sequence: u64) -> Option<Self> {
        if boot_epoch == 0 || sequence == 0 {
            return None;
        }
        Some(Self((u128::from(boot_epoch) << 64) | u128::from(sequence)))
    }

    pub fn boot_epoch(self) -> u64 {
        (self.0 >> 64) as u64
    }
}

/// How an arm ended. Terminal by construction: there is no "still running".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArmTerminalState {
    /// Treatment: the action ran and its effect was reverted.
    AppliedAndReverted,
    /// Treatment: the action ran and needs no revert (non-kernel effect).
    AppliedNoRevertNeeded,
    /// Control: the action was deliberately withheld by the scheduler.
    WithheldByHoldout,
}

impl ArmTerminalState {
    fn is_treatment(self) -> bool {
        matches!(self, Self::AppliedAndReverted | Self::AppliedNoRevertNeeded)
    }

    fn is_control(self) -> bool {
        matches!(self, Self::WithheldByHoldout)
    }
}

/// One arm's certified outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExplorationArmOutcome {
    pub correlation_id: ProbeCorrelation,
    pub arm: ExplorationArm,
    pub issued_cycle: u64,
    pub settled_cycle: u64,
    pub terminal_state: ArmTerminalState,
    /// Measured utility in micro-units. Signed: an arm may be worse.
    pub utility_micros: i64,
}

impl ExplorationArmOutcome {
    fn settled_in_window(&self, expires_after_cycle: u64) -> bool {
        self.settled_cycle >= self.issued_cycle && self.settled_cycle <= expires_after_cycle
    }
}

/// Why a candidate pair was refused. Every variant names a specific failed
/// invariant rather than a generic "invalid", because a refusal nobody can
/// diagnose is how a route goes quiet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairAssemblyError {
    /// Both arms carry the same correlation: they are one assignment, not two.
    SharedCorrelation,
    /// A correlation is unset.
    MissingCorrelation,
    /// The treatment slot does not hold a treatment terminal state, or the
    /// control slot does not hold a control one.
    ArmRoleMismatch,
    /// An arm settled outside its own window.
    OutsideWindow,
    /// The experiment id does not belong to the boot that is assembling it.
    ForeignBootEpoch,
    /// Identity fields are unset or inconsistent.
    IncoherentIdentity,
}

/// A completed, certified causal experiment.
///
/// The only object the Lab consumes. Build it through [`Self::assemble`]; the
/// fields are public to read and the constructor is the only way to create one,
/// so an unchecked pair cannot exist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletedExplorationPair {
    pub experiment_id: ExperimentId,
    pub family: ActuatorFamily,
    pub action_class: ActionClass,
    pub canonical_key: String,
    pub policy_version: u32,
    pub boot_epoch: u64,
    pub treatment: ExplorationArmOutcome,
    pub control: ExplorationArmOutcome,
}

impl CompletedExplorationPair {
    /// Certify a pair, or say exactly which invariant failed.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        experiment_id: ExperimentId,
        family: ActuatorFamily,
        action_class: ActionClass,
        canonical_key: String,
        policy_version: u32,
        boot_epoch: u64,
        treatment: ExplorationArmOutcome,
        control: ExplorationArmOutcome,
        treatment_expires_after_cycle: u64,
        control_expires_after_cycle: u64,
    ) -> Result<Self, PairAssemblyError> {
        if canonical_key.is_empty() || boot_epoch == 0 {
            return Err(PairAssemblyError::IncoherentIdentity);
        }
        if experiment_id.boot_epoch() != boot_epoch {
            return Err(PairAssemblyError::ForeignBootEpoch);
        }
        if treatment.correlation_id.0 == 0 || control.correlation_id.0 == 0 {
            return Err(PairAssemblyError::MissingCorrelation);
        }
        // The arms are two assignments of one experiment. Sharing a correlation
        // would mean one assignment counted twice.
        if treatment.correlation_id == control.correlation_id {
            return Err(PairAssemblyError::SharedCorrelation);
        }
        if !treatment.terminal_state.is_treatment() || !control.terminal_state.is_control() {
            return Err(PairAssemblyError::ArmRoleMismatch);
        }
        if !treatment.settled_in_window(treatment_expires_after_cycle)
            || !control.settled_in_window(control_expires_after_cycle)
        {
            return Err(PairAssemblyError::OutsideWindow);
        }
        Ok(Self {
            experiment_id,
            family,
            action_class,
            canonical_key,
            policy_version,
            boot_epoch,
            treatment,
            control,
        })
    }

    /// Treatment minus control, in micro-units. The only causal quantity this
    /// object asserts, and only because both arms came from one randomisation.
    pub fn effect_micros(&self) -> i64 {
        self.treatment
            .utility_micros
            .saturating_sub(self.control.utility_micros)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm(correlation: u64, terminal: ArmTerminalState) -> ExplorationArmOutcome {
        ExplorationArmOutcome {
            correlation_id: ProbeCorrelation(correlation),
            arm: ExplorationArm::MarkovCacheOnly,
            issued_cycle: 10,
            settled_cycle: 14,
            terminal_state: terminal,
            utility_micros: 0,
        }
    }

    fn assemble(
        treatment: ExplorationArmOutcome,
        control: ExplorationArmOutcome,
    ) -> Result<CompletedExplorationPair, PairAssemblyError> {
        CompletedExplorationPair::assemble(
            ExperimentId::new(7, 42).expect("id"),
            ActuatorFamily::MarkovPrewarm,
            ActionClass::MarkovPredictedApp,
            "markov_prewarm:predicted_app@cache_only".to_string(),
            3,
            7,
            treatment,
            control,
            20,
            20,
        )
    }

    #[test]
    fn a_valid_pair_assembles_and_reports_its_effect() {
        let mut t = arm(101, ArmTerminalState::AppliedAndReverted);
        t.utility_micros = 1_500;
        let mut c = arm(102, ArmTerminalState::WithheldByHoldout);
        c.utility_micros = 400;
        let pair = assemble(t, c).expect("valid pair");
        assert_eq!(pair.effect_micros(), 1_100);
        assert_eq!(pair.experiment_id.boot_epoch(), 7);
    }

    #[test]
    fn two_arms_may_never_share_one_correlation() {
        // ProbeCorrelation names an assignment. The same one on both sides is
        // one assignment counted twice, not a controlled comparison.
        let t = arm(101, ArmTerminalState::AppliedAndReverted);
        let c = arm(101, ArmTerminalState::WithheldByHoldout);
        assert_eq!(assemble(t, c), Err(PairAssemblyError::SharedCorrelation));
    }

    #[test]
    fn an_applied_action_can_never_occupy_the_control_slot() {
        let t = arm(101, ArmTerminalState::AppliedAndReverted);
        let c = arm(102, ArmTerminalState::AppliedNoRevertNeeded);
        assert_eq!(assemble(t, c), Err(PairAssemblyError::ArmRoleMismatch));
    }

    #[test]
    fn a_withheld_action_can_never_occupy_the_treatment_slot() {
        let t = arm(101, ArmTerminalState::WithheldByHoldout);
        let c = arm(102, ArmTerminalState::WithheldByHoldout);
        assert_eq!(assemble(t, c), Err(PairAssemblyError::ArmRoleMismatch));
    }

    #[test]
    fn an_arm_settling_outside_its_window_is_refused() {
        let mut t = arm(101, ArmTerminalState::AppliedAndReverted);
        t.settled_cycle = 999;
        let c = arm(102, ArmTerminalState::WithheldByHoldout);
        assert_eq!(assemble(t, c), Err(PairAssemblyError::OutsideWindow));
    }

    #[test]
    fn an_experiment_from_another_boot_cannot_be_assembled_here() {
        // A late endpoint from a previous boot must not close an experiment on
        // this one. The epoch lives inside the id, so the check cannot be
        // forgotten by a caller.
        let t = arm(101, ArmTerminalState::AppliedAndReverted);
        let c = arm(102, ArmTerminalState::WithheldByHoldout);
        let err = CompletedExplorationPair::assemble(
            ExperimentId::new(6, 42).expect("id"),
            ActuatorFamily::MarkovPrewarm,
            ActionClass::MarkovPredictedApp,
            "markov_prewarm:predicted_app@cache_only".to_string(),
            3,
            7,
            t,
            c,
            20,
            20,
        );
        assert_eq!(err, Err(PairAssemblyError::ForeignBootEpoch));
    }

    #[test]
    fn an_unset_correlation_is_refused() {
        let t = arm(0, ArmTerminalState::AppliedAndReverted);
        let c = arm(102, ArmTerminalState::WithheldByHoldout);
        assert_eq!(assemble(t, c), Err(PairAssemblyError::MissingCorrelation));
    }

    #[test]
    fn an_experiment_id_needs_both_an_epoch_and_a_sequence() {
        assert!(ExperimentId::new(0, 1).is_none());
        assert!(ExperimentId::new(1, 0).is_none());
        assert!(ExperimentId::new(1, 1).is_some());
    }
}
