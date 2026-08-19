//! Micro-canary: the smallest thing that can produce a causal pair.
//!
//! Shadow cannot produce causal evidence and never will: a control endpoint
//! requires an action to have been *deliberately withheld*, and a phase whose
//! contract is "change nothing" cannot withhold. That is not a defect in
//! Shadow — it is what Shadow means. So the withholding lives here, in a phase
//! named for the fact that it intervenes.
//!
//! What it does, and nothing more:
//!
//! ```text
//! eligible opportunity (already past every normal gate)
//!        ↓  sampled at most 1 in 100
//! experiment_id minted BEFORE randomisation
//!        ↓
//! random assignment ─┬─ treatment: the action runs
//!                    └─ control:   the action is withheld
//!        ↓
//! both arms settle → CompletedExplorationPair
//! ```
//!
//! # What it may never do
//!
//! - Create eligibility. A candidate that failed a normal gate is not offered
//!   here; the sample is drawn from opportunities Apollo was already going to
//!   act on.
//! - Exceed its budget. One open experiment per family, two globally, and at
//!   most one sampled opportunity in a hundred.
//! - Touch a family it was not enabled for. MarkovPrewarm is the only one
//!   enabled, on purpose: the first goal is to prove a lifecycle end to end,
//!   not to make Apollo learn faster.
//! - Survive a kill switch. `disable` stops sampling immediately and lets the
//!   open experiments drain.

use serde::{Deserialize, Serialize};

use crate::engine::exploration_pair::{
    ArmTerminalState, CompletedExplorationPair, ExperimentId, ExplorationArmOutcome,
    PairAssemblyError,
};
use crate::engine::exploration_scheduler::{ActionClass, ExplorationArm, ProbeCorrelation};
use crate::engine::telemetry_medallion::ActuatorFamily;

/// Sampled opportunities per thousand eligible. 10 ‰ = 1.0 %, the ceiling for
/// MarkovPrewarm. Deliberately not a knob: raising it to "learn faster" before
/// one pair has ever completed would be optimising a rate that has never once
/// produced a result.
pub const MARKOV_SAMPLE_PER_MILLE: u64 = 10;

/// Open experiments per family, and in total. One and two: enough to prove the
/// cycle, small enough that a mistake costs a single withheld prewarm.
pub const MAX_OPEN_PER_FAMILY: usize = 1;
pub const MAX_OPEN_GLOBAL: usize = 2;

/// Cycles an experiment may stay open before both arms are abandoned.
pub const EXPERIMENT_HORIZON_CYCLES: u64 = 240;

/// Why an experiment left the estimator without contributing an effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CensorCause {
    /// The measurement window was confounded.
    Confounded,
    /// An arm never produced a terminal observation.
    Incomplete,
    /// The two arms could not be measured on equal terms at all.
    NotComparable,
}

/// What the caller should do with an opportunity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArmDecision {
    /// Not sampled. Behave exactly as if the micro-canary did not exist — this
    /// is the answer for at least 99 opportunities in every 100.
    Proceed,
    /// Sampled into the treatment arm: perform the action, and report its
    /// endpoint under this correlation.
    Treatment {
        experiment_id: ExperimentId,
        correlation: ProbeCorrelation,
    },
    /// Sampled into the control arm: **withhold** the action. This is the only
    /// place in the system that asks for something not to happen, and it is
    /// why this phase is not called Shadow.
    WithholdAsControl {
        experiment_id: ExperimentId,
        correlation: ProbeCorrelation,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MicroCanaryMetrics {
    /// Opportunities offered, all of them already eligible.
    pub eligible_seen: u64,
    /// Opportunities drawn into an experiment.
    pub sampled: u64,
    /// Draws refused because a budget was full.
    pub refused_budget: u64,
    /// Draws refused because the family is not enabled.
    pub refused_family: u64,
    /// Draws refused because the kill switch is off.
    pub refused_disabled: u64,
    /// Experiments where both arms settled and the pair assembled.
    pub pairs_completed: u64,
    /// Experiments abandoned at their horizon with an arm missing.
    pub pairs_expired: u64,
    /// Endpoints for an experiment that is no longer open.
    pub endpoints_late: u64,
    /// Endpoints for an arm that already reported.
    pub endpoints_duplicate: u64,
    /// Pairs that failed assembly. Non-zero means the producer emitted
    /// something the contract refuses, which is a bug here and not there.
    pub assembly_refused: u64,
    /// Treatment arms handed to the caller.
    pub treatment_issued: u64,
    /// Control arms handed to the caller — each one is a real action withheld
    /// from the machine, so this is the number that measures the intervention.
    pub control_issued: u64,
    /// Controls the caller reported it actually honoured. It should track
    /// `control_issued`; a gap means something was asked to be withheld and
    /// went ahead anyway, which is the one outcome that would invalidate a
    /// pair without any counter noticing.
    pub control_honoured: u64,
    /// Arms reported in each terminal state.
    pub arms_applied_reverted: u64,
    pub arms_applied_no_revert: u64,
    pub arms_withheld: u64,
    /// Largest number of simultaneously open experiments seen this boot.
    pub open_high_watermark: u32,
    /// Experiments abandoned because the arms could not be compared on equal
    /// terms. Counted apart and never coerced to a zero-utility observation:
    /// "we did not measure" and "we measured nothing" are different facts.
    pub abandoned_not_comparable: u64,
    /// Attrition, **by arm**. Not measuring is correct, but if the treatment
    /// loses more windows than the control, excluding them quietly biases the
    /// effect upward: the surviving treatments would be the ones that went
    /// well. These make the loss visible so it can be shown non-differential.
    pub treatment_confounded: u64,
    pub control_confounded: u64,
    pub treatment_incomplete: u64,
    pub control_incomplete: u64,
    /// Experiments that left the estimator without contributing an effect,
    /// whatever the cause. The denominator for an attrition rate.
    pub pairs_censored: u64,
    /// Predictions that resolved, by arm. Diagnostic only — it shows whether
    /// the two arms drew equivalent opportunities. It is never the effect,
    /// because whether a prediction comes true is not something a pre-warm can
    /// change; treating it as the outcome would measure noise and call it zero.
    pub treatment_predictions_resolved: u64,
    pub control_predictions_resolved: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct OpenExperiment {
    experiment_id: ExperimentId,
    family: ActuatorFamily,
    action_class: ActionClass,
    arm: ExplorationArm,
    canonical_key: String,
    policy_version: u32,
    treatment_correlation: ProbeCorrelation,
    control_correlation: ProbeCorrelation,
    opened_cycle: u64,
    expires_after_cycle: u64,
    /// Which arm the caller was handed. Without it, the complement is unknown
    /// until something settles — which is exactly when the caller needs it,
    /// since it has to run the other half before either can settle.
    treatment_issued_first: bool,
    treatment: Option<ExplorationArmOutcome>,
    control: Option<ExplorationArmOutcome>,
}

impl OpenExperiment {
    fn is_complete(&self) -> bool {
        self.treatment.is_some() && self.control.is_some()
    }
}

/// The producer. Holds no actuator and performs no action: it answers
/// questions and records what the caller reports back.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroCanary {
    boot_epoch: u64,
    next_sequence: u64,
    next_correlation: u64,
    /// Deterministic stream, so a run can be replayed exactly.
    rng_state: u64,
    enabled: bool,
    open: Vec<OpenExperiment>,
    metrics: MicroCanaryMetrics,
}

impl MicroCanary {
    /// Construct **disabled**. The producer that ships in a binary must do
    /// nothing until someone turns it on: a wired-but-dormant canary and a
    /// baseline have to be indistinguishable, or the comparison B exists for
    /// is worthless.
    pub fn new_disabled(boot_epoch: u64) -> Self {
        let mut c = Self::new(boot_epoch);
        c.enabled = false;
        c
    }

    /// Start sampling. The moment B begins.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn new(boot_epoch: u64) -> Self {
        Self {
            boot_epoch,
            next_sequence: 0,
            next_correlation: 0,
            // Seeded from the boot epoch: reproducible within a boot, different
            // across boots, so a restart does not replay the same draws.
            rng_state: boot_epoch | 1,
            enabled: true,
            open: Vec::with_capacity(MAX_OPEN_GLOBAL),
            metrics: MicroCanaryMetrics::default(),
        }
    }

    pub fn metrics(&self) -> &MicroCanaryMetrics {
        &self.metrics
    }

    pub fn open_len(&self) -> usize {
        self.open.len()
    }

    /// Stop sampling. Open experiments are left to drain rather than dropped,
    /// so an arm already withheld still gets its pair.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Only MarkovPrewarm, and only for now. Widening this is a decision with
    /// evidence behind it, not a constant someone edits.
    fn family_enabled(family: ActuatorFamily) -> bool {
        matches!(family, ActuatorFamily::MarkovPrewarm)
    }

    fn next_u64(&mut self) -> u64 {
        // splitmix64: small, deterministic, adequate for a 1-in-100 draw.
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn budget_allows(&self, family: ActuatorFamily) -> bool {
        if self.open.len() >= MAX_OPEN_GLOBAL {
            return false;
        }
        self.open.iter().filter(|e| e.family == family).count() < MAX_OPEN_PER_FAMILY
    }

    /// Offer one already-eligible opportunity.
    ///
    /// `eligible` is the caller's assertion that this opportunity passed every
    /// normal gate and Apollo was going to act on it. The micro-canary never
    /// makes something eligible; it only decides whether to withhold something
    /// that was going to happen anyway.
    #[allow(clippy::too_many_arguments)]
    pub fn offer(
        &mut self,
        family: ActuatorFamily,
        action_class: ActionClass,
        arm: ExplorationArm,
        canonical_key: &str,
        policy_version: u32,
        cycle: u64,
    ) -> ArmDecision {
        self.metrics.eligible_seen = self.metrics.eligible_seen.saturating_add(1);
        if !self.enabled {
            self.metrics.refused_disabled = self.metrics.refused_disabled.saturating_add(1);
            return ArmDecision::Proceed;
        }
        if !Self::family_enabled(family) {
            self.metrics.refused_family = self.metrics.refused_family.saturating_add(1);
            return ArmDecision::Proceed;
        }
        // Draw first, budget second: a budget-refused draw must still consume
        // its share of the rate, or a full budget would silently raise the
        // sampling rate of everything that follows.
        if self.next_u64() % 1_000 >= MARKOV_SAMPLE_PER_MILLE {
            return ArmDecision::Proceed;
        }
        if !self.budget_allows(family) {
            self.metrics.refused_budget = self.metrics.refused_budget.saturating_add(1);
            return ArmDecision::Proceed;
        }

        // Identity is minted before randomisation, so which arm goes first
        // cannot influence what the experiment is called.
        let sequence = self.next_sequence.saturating_add(1);
        let Some(experiment_id) = ExperimentId::new(self.boot_epoch, sequence) else {
            return ArmDecision::Proceed;
        };
        self.next_sequence = sequence;
        let treatment_correlation = ProbeCorrelation(self.mint_correlation());
        let control_correlation = ProbeCorrelation(self.mint_correlation());
        let treatment_first = self.next_u64() & 1 == 0;

        self.open.push(OpenExperiment {
            experiment_id,
            family,
            action_class,
            arm,
            canonical_key: canonical_key.to_string(),
            policy_version,
            treatment_correlation,
            control_correlation,
            opened_cycle: cycle,
            expires_after_cycle: cycle.saturating_add(EXPERIMENT_HORIZON_CYCLES),
            treatment_issued_first: treatment_first,
            treatment: None,
            control: None,
        });
        self.metrics.sampled = self.metrics.sampled.saturating_add(1);

        let open_len = self.open.len() as u32;
        if open_len > self.metrics.open_high_watermark {
            self.metrics.open_high_watermark = open_len;
        }

        if treatment_first {
            self.metrics.treatment_issued = self.metrics.treatment_issued.saturating_add(1);
            ArmDecision::Treatment {
                experiment_id,
                correlation: treatment_correlation,
            }
        } else {
            self.metrics.control_issued = self.metrics.control_issued.saturating_add(1);
            ArmDecision::WithholdAsControl {
                experiment_id,
                correlation: control_correlation,
            }
        }
    }

    /// The caller confirms it actually withheld the action it was asked to
    /// withhold. Separate from issuing on purpose: a control that was issued
    /// and then quietly performed anyway produces a pair whose control arm is
    /// not a control, and no other counter would show it.
    pub fn confirm_control_honoured(&mut self) {
        self.metrics.control_honoured = self.metrics.control_honoured.saturating_add(1);
    }

    fn mint_correlation(&mut self) -> u64 {
        self.next_correlation = self.next_correlation.saturating_add(1);
        // Namespaced by boot so a correlation from a previous boot cannot be
        // mistaken for one of this boot's arms.
        (self.boot_epoch << 32) | (self.next_correlation & 0xFFFF_FFFF)
    }

    /// The complementary arm for an experiment already open, so the caller can
    /// run the half it has not run yet.
    pub fn complementary_arm(&self, experiment_id: ExperimentId) -> Option<ArmDecision> {
        let e = self
            .open
            .iter()
            .find(|e| e.experiment_id == experiment_id)?;
        let want_control = match (e.treatment.is_some(), e.control.is_some()) {
            (true, false) => true,
            (false, true) => false,
            // Nothing has settled yet: the complement is whichever arm was not
            // handed out at assignment time.
            (false, false) => e.treatment_issued_first,
            (true, true) => return None,
        };
        Some(if want_control {
            ArmDecision::WithholdAsControl {
                experiment_id,
                correlation: e.control_correlation,
            }
        } else {
            ArmDecision::Treatment {
                experiment_id,
                correlation: e.treatment_correlation,
            }
        })
    }

    /// Which experiment and arm a utility-window key belongs to.
    ///
    /// The key is `ProbeCorrelation::ledger_correlation_id`, already namespaced
    /// by the scheduler, so it cannot collide with a real decision id.
    pub fn arm_for_ledger_id(&self, ledger_id: u64) -> Option<(ExperimentId, bool)> {
        self.open.iter().find_map(|e| {
            if e.treatment_correlation.ledger_correlation_id() == ledger_id {
                Some((e.experiment_id, true))
            } else if e.control_correlation.ledger_correlation_id() == ledger_id {
                Some((e.experiment_id, false))
            } else {
                None
            }
        })
    }

    /// Both correlations of an open experiment, for opening the two identical
    /// utility windows.
    pub fn correlations(
        &self,
        experiment_id: ExperimentId,
    ) -> Option<(ProbeCorrelation, ProbeCorrelation)> {
        self.open
            .iter()
            .find(|e| e.experiment_id == experiment_id)
            .map(|e| (e.treatment_correlation, e.control_correlation))
    }

    /// Drop an experiment whose arms cannot be compared on equal terms.
    ///
    /// The case this exists for: an opportunity was assigned to treatment and
    /// the pre-warm then did not actuate. Its arm is not a treatment, and
    /// forcing it to a zero-utility observation would put a measurement that
    /// never happened into the estimator.
    pub fn abandon_not_comparable(&mut self, experiment_id: ExperimentId) -> bool {
        self.censor(experiment_id, None, CensorCause::NotComparable)
    }

    /// Censor an experiment, naming the arm responsible when one is.
    ///
    /// `arm_is_treatment` is `None` when the loss belongs to the experiment
    /// rather than to one side — both windows failing to open, for instance.
    pub fn censor(
        &mut self,
        experiment_id: ExperimentId,
        arm_is_treatment: Option<bool>,
        cause: CensorCause,
    ) -> bool {
        let before = self.open.len();
        self.open.retain(|e| e.experiment_id != experiment_id);
        let dropped = before != self.open.len();
        if !dropped {
            return false;
        }
        self.metrics.pairs_censored = self.metrics.pairs_censored.saturating_add(1);
        match (cause, arm_is_treatment) {
            (CensorCause::Confounded, Some(true)) => {
                self.metrics.treatment_confounded =
                    self.metrics.treatment_confounded.saturating_add(1)
            }
            (CensorCause::Confounded, Some(false)) => {
                self.metrics.control_confounded = self.metrics.control_confounded.saturating_add(1)
            }
            (CensorCause::Incomplete, Some(true)) => {
                self.metrics.treatment_incomplete =
                    self.metrics.treatment_incomplete.saturating_add(1)
            }
            (CensorCause::Incomplete, Some(false)) => {
                self.metrics.control_incomplete = self.metrics.control_incomplete.saturating_add(1)
            }
            _ => {}
        }
        self.metrics.abandoned_not_comparable =
            self.metrics.abandoned_not_comparable.saturating_add(1);
        dropped
    }

    /// Record that a prediction resolved for one arm. Diagnostic: it shows the
    /// two arms drew equivalent opportunities. Never an effect.
    pub fn note_prediction_resolved(&mut self, treatment: bool) {
        if treatment {
            self.metrics.treatment_predictions_resolved = self
                .metrics
                .treatment_predictions_resolved
                .saturating_add(1);
        } else {
            self.metrics.control_predictions_resolved =
                self.metrics.control_predictions_resolved.saturating_add(1);
        }
    }

    /// Report a settled arm. Returns a completed pair once both have landed.
    pub fn record_arm(
        &mut self,
        experiment_id: ExperimentId,
        correlation: ProbeCorrelation,
        settled_cycle: u64,
        terminal_state: ArmTerminalState,
        utility_micros: i64,
    ) -> Option<CompletedExplorationPair> {
        let Some(index) = self
            .open
            .iter()
            .position(|e| e.experiment_id == experiment_id)
        else {
            self.metrics.endpoints_late = self.metrics.endpoints_late.saturating_add(1);
            return None;
        };
        let expires = self.open[index].expires_after_cycle;
        let opened = self.open[index].opened_cycle;
        let outcome = ExplorationArmOutcome {
            correlation_id: correlation,
            arm: self.open[index].arm,
            issued_cycle: opened,
            settled_cycle,
            terminal_state,
            utility_micros,
        };
        let complete;
        {
            let e = &mut self.open[index];
            let slot = if correlation == e.treatment_correlation {
                &mut e.treatment
            } else if correlation == e.control_correlation {
                &mut e.control
            } else {
                // The experiment is open but this is not one of its two arms.
                self.metrics.endpoints_late = self.metrics.endpoints_late.saturating_add(1);
                return None;
            };
            if slot.is_some() {
                self.metrics.endpoints_duplicate =
                    self.metrics.endpoints_duplicate.saturating_add(1);
                return None;
            }
            *slot = Some(outcome);
            complete = e.is_complete();
        }
        // Counted where the arm is recorded, not where the pair completes: the
        // first arm of every experiment returns early, so counting after the
        // completeness check would have made half of them invisible.
        match terminal_state {
            ArmTerminalState::AppliedAndReverted => {
                self.metrics.arms_applied_reverted =
                    self.metrics.arms_applied_reverted.saturating_add(1)
            }
            ArmTerminalState::AppliedNoRevertNeeded => {
                self.metrics.arms_applied_no_revert =
                    self.metrics.arms_applied_no_revert.saturating_add(1)
            }
            ArmTerminalState::WithheldByHoldout => {
                self.metrics.arms_withheld = self.metrics.arms_withheld.saturating_add(1)
            }
        }
        if !complete {
            return None;
        }

        let e = self.open.swap_remove(index);
        let treatment = e.treatment.expect("complete");
        let control = e.control.expect("complete");
        match CompletedExplorationPair::assemble(
            e.experiment_id,
            e.family,
            e.action_class,
            e.canonical_key,
            e.policy_version,
            self.boot_epoch,
            treatment,
            control,
            expires,
            expires,
        ) {
            Ok(pair) => {
                self.metrics.pairs_completed = self.metrics.pairs_completed.saturating_add(1);
                Some(pair)
            }
            Err(error) => {
                // The contract refused something this producer built. That is a
                // defect here, and it is counted rather than retried, because a
                // retry would only produce the same refusal.
                let _: PairAssemblyError = error;
                self.metrics.assembly_refused = self.metrics.assembly_refused.saturating_add(1);
                None
            }
        }
    }

    /// Abandon experiments past their horizon. Call once per cycle.
    pub fn expire(&mut self, cycle: u64) -> u64 {
        // Attribute the loss to the arm that never reported, so attrition can
        // be shown non-differential rather than merely counted.
        let missing: Vec<Option<bool>> = self
            .open
            .iter()
            .filter(|e| cycle > e.expires_after_cycle)
            .map(|e| match (e.treatment.is_some(), e.control.is_some()) {
                (true, false) => Some(false),
                (false, true) => Some(true),
                _ => None,
            })
            .collect();
        let before = self.open.len();
        self.open.retain(|e| cycle <= e.expires_after_cycle);
        let dropped = (before - self.open.len()) as u64;
        self.metrics.pairs_expired = self.metrics.pairs_expired.saturating_add(dropped);
        self.metrics.pairs_censored = self.metrics.pairs_censored.saturating_add(dropped);
        for arm in missing {
            match arm {
                Some(true) => {
                    self.metrics.treatment_incomplete =
                        self.metrics.treatment_incomplete.saturating_add(1)
                }
                Some(false) => {
                    self.metrics.control_incomplete =
                        self.metrics.control_incomplete.saturating_add(1)
                }
                None => {}
            }
        }
        dropped
    }

    /// Copy this producer's accounting onto `RuntimeMetrics`.
    ///
    /// Lives here rather than in the daemon loop: three counters were added to
    /// the lab and never reached that struct, each time because the mapping sat
    /// inline where no test could walk it.
    pub fn publish(&self, out: &mut crate::engine::types::RuntimeMetrics) {
        let m = &self.metrics;
        out.canary_enabled = self.enabled;
        out.canary_eligible_seen = m.eligible_seen;
        out.canary_sampled = m.sampled;
        out.canary_refused_budget = m.refused_budget;
        out.canary_refused_family = m.refused_family;
        out.canary_refused_disabled = m.refused_disabled;
        out.canary_treatment_issued = m.treatment_issued;
        out.canary_control_issued = m.control_issued;
        out.canary_control_honoured = m.control_honoured;
        out.canary_arms_applied_reverted = m.arms_applied_reverted;
        out.canary_arms_applied_no_revert = m.arms_applied_no_revert;
        out.canary_arms_withheld = m.arms_withheld;
        out.canary_pairs_completed = m.pairs_completed;
        out.canary_pairs_expired = m.pairs_expired;
        out.canary_endpoints_late = m.endpoints_late;
        out.canary_endpoints_duplicate = m.endpoints_duplicate;
        out.canary_assembly_refused = m.assembly_refused;
        out.canary_open_experiments = self.open.len() as u64;
        out.canary_open_high_watermark = u64::from(m.open_high_watermark);
        out.canary_observed_per_mille = self.observed_per_mille();
        out.canary_treatment_confounded = m.treatment_confounded;
        out.canary_control_confounded = m.control_confounded;
        out.canary_treatment_incomplete = m.treatment_incomplete;
        out.canary_control_incomplete = m.control_incomplete;
        out.canary_pairs_censored = m.pairs_censored;
    }

    /// Observed sampling rate per thousand. For a kill condition that reads
    /// what happened rather than what was configured.
    pub fn observed_per_mille(&self) -> f64 {
        if self.metrics.eligible_seen == 0 {
            return 0.0;
        }
        (self.metrics.sampled as f64) * 1_000.0 / (self.metrics.eligible_seen as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOOT: u64 = 0x5EED;

    fn canary() -> MicroCanary {
        MicroCanary::new(BOOT)
    }

    /// Offer until one opportunity is sampled, or give up. Deterministic: the
    /// stream is seeded from the boot epoch.
    fn offer_until_sampled(c: &mut MicroCanary, cycle: u64) -> Option<ArmDecision> {
        for _ in 0..100_000 {
            match c.offer(
                ActuatorFamily::MarkovPrewarm,
                ActionClass::MarkovPredictedApp,
                ExplorationArm::MarkovCacheOnly,
                "markov_prewarm:predicted_app@cache_only",
                3,
                cycle,
            ) {
                ArmDecision::Proceed => continue,
                other => return Some(other),
            }
        }
        None
    }

    fn ids(d: &ArmDecision) -> (ExperimentId, ProbeCorrelation) {
        match d {
            ArmDecision::Treatment {
                experiment_id,
                correlation,
            }
            | ArmDecision::WithholdAsControl {
                experiment_id,
                correlation,
            } => (*experiment_id, *correlation),
            ArmDecision::Proceed => panic!("not sampled"),
        }
    }

    /// Drive one experiment to completion from whichever arm came first.
    fn complete_one(c: &mut MicroCanary, cycle: u64) -> Option<CompletedExplorationPair> {
        let first = offer_until_sampled(c, cycle)?;
        let (id, corr) = ids(&first);
        let first_state = match first {
            ArmDecision::Treatment { .. } => ArmTerminalState::AppliedNoRevertNeeded,
            _ => ArmTerminalState::WithheldByHoldout,
        };
        // Utility by role, never by order: which arm came first is random, so a
        // test that pays the first arm 100 asserts a coin flip.
        let first_utility = if matches!(first, ArmDecision::Treatment { .. }) {
            100
        } else {
            40
        };
        assert!(c
            .record_arm(id, corr, cycle + 1, first_state, first_utility)
            .is_none());
        let second = c.complementary_arm(id).expect("complementary arm");
        let (_, corr2) = ids(&second);
        let second_state = match second {
            ArmDecision::Treatment { .. } => ArmTerminalState::AppliedNoRevertNeeded,
            _ => ArmTerminalState::WithheldByHoldout,
        };
        let second_utility = if matches!(second, ArmDecision::Treatment { .. }) {
            100
        } else {
            40
        };
        c.record_arm(id, corr2, cycle + 2, second_state, second_utility)
    }

    // ── 1. assignment ───────────────────────────────────────────────────────

    #[test]
    fn an_assignment_mints_identity_before_choosing_the_arm() {
        let mut c = canary();
        let d = offer_until_sampled(&mut c, 10).expect("sampled");
        let (id, corr) = ids(&d);
        assert_eq!(id.boot_epoch(), BOOT, "identity carries the boot");
        assert_ne!(corr.0, 0);
        assert_eq!(c.open_len(), 1);
        // Whichever arm it chose, the complement is the other one.
        let other = c.complementary_arm(id).expect("complement");
        assert_ne!(
            std::mem::discriminant(&d),
            std::mem::discriminant(&other),
            "the two arms must differ"
        );
    }

    #[test]
    fn a_completed_pair_carries_two_distinct_correlations() {
        let mut c = canary();
        let pair = complete_one(&mut c, 10).expect("pair completes");
        assert_ne!(
            pair.treatment.correlation_id, pair.control.correlation_id,
            "one assignment counted twice is not a controlled comparison"
        );
        assert_eq!(pair.experiment_id.boot_epoch(), BOOT);
        assert_eq!(c.metrics().pairs_completed, 1);
        assert_eq!(c.metrics().assembly_refused, 0);
        assert_eq!(c.open_len(), 0, "a completed experiment frees its slot");
    }

    // ── 2. expiry ───────────────────────────────────────────────────────────

    #[test]
    fn an_experiment_past_its_horizon_is_abandoned_and_frees_its_slot() {
        let mut c = canary();
        offer_until_sampled(&mut c, 10).expect("sampled");
        assert_eq!(c.open_len(), 1);
        assert_eq!(c.expire(10 + EXPERIMENT_HORIZON_CYCLES), 0, "not yet");
        assert_eq!(c.expire(11 + EXPERIMENT_HORIZON_CYCLES), 1);
        assert_eq!(c.open_len(), 0);
        assert_eq!(c.metrics().pairs_expired, 1);
    }

    // ── 3. restart ──────────────────────────────────────────────────────────

    #[test]
    fn a_restart_cannot_reuse_a_previous_boots_identity() {
        let mut a = MicroCanary::new(7);
        let da = offer_until_sampled(&mut a, 1).expect("sampled");
        let (ida, _) = ids(&da);
        let mut b = MicroCanary::new(8);
        let db = offer_until_sampled(&mut b, 1).expect("sampled");
        let (idb, _) = ids(&db);
        assert_ne!(ida, idb, "identity must not repeat across boots");
        assert_eq!(ida.boot_epoch(), 7);
        assert_eq!(idb.boot_epoch(), 8);
        // And an arm from the old boot cannot land on the new one.
        let (_, corra) = ids(&da);
        assert!(b
            .record_arm(ida, corra, 5, ArmTerminalState::WithheldByHoldout, 0)
            .is_none());
        assert_eq!(b.metrics().endpoints_late, 1);
    }

    // ── 4. duplicate endpoint ───────────────────────────────────────────────

    #[test]
    fn the_same_arm_reporting_twice_pays_once() {
        let mut c = canary();
        let d = offer_until_sampled(&mut c, 10).expect("sampled");
        let (id, corr) = ids(&d);
        let state = match d {
            ArmDecision::Treatment { .. } => ArmTerminalState::AppliedNoRevertNeeded,
            _ => ArmTerminalState::WithheldByHoldout,
        };
        assert!(c.record_arm(id, corr, 11, state, 5).is_none());
        assert!(c.record_arm(id, corr, 12, state, 5).is_none());
        assert_eq!(c.metrics().endpoints_duplicate, 1);
        assert_eq!(c.metrics().pairs_completed, 0);
    }

    // ── 5. late endpoint ────────────────────────────────────────────────────

    #[test]
    fn an_endpoint_for_an_abandoned_experiment_closes_nothing() {
        let mut c = canary();
        let d = offer_until_sampled(&mut c, 10).expect("sampled");
        let (id, corr) = ids(&d);
        c.expire(11 + EXPERIMENT_HORIZON_CYCLES);
        assert!(c
            .record_arm(id, corr, 400, ArmTerminalState::WithheldByHoldout, 9)
            .is_none());
        assert_eq!(c.metrics().endpoints_late, 1);
        assert_eq!(c.metrics().pairs_completed, 0);
    }

    // ── 6. canonical key reuse ──────────────────────────────────────────────

    #[test]
    fn two_experiments_on_the_same_key_never_share_identity() {
        let mut c = canary();
        let first = complete_one(&mut c, 10).expect("first pair");
        let second = complete_one(&mut c, 100).expect("second pair");
        assert_eq!(
            first.canonical_key, second.canonical_key,
            "same key by design"
        );
        assert_ne!(
            first.experiment_id, second.experiment_id,
            "reusing a key must not reuse an experiment"
        );
        assert_ne!(
            first.treatment.correlation_id,
            second.treatment.correlation_id
        );
        assert_ne!(first.control.correlation_id, second.control.correlation_id);
    }

    // ── 7. budget ───────────────────────────────────────────────────────────

    #[test]
    fn the_budget_caps_open_experiments_per_family_and_globally() {
        let mut c = canary();
        offer_until_sampled(&mut c, 10).expect("first sampled");
        assert_eq!(c.open_len(), 1);
        // MarkovPrewarm allows one open experiment. Every further draw is
        // refused for budget, never silently admitted.
        for _ in 0..20_000 {
            let d = c.offer(
                ActuatorFamily::MarkovPrewarm,
                ActionClass::MarkovPredictedApp,
                ExplorationArm::MarkovCacheOnly,
                "markov_prewarm:predicted_app@cache_only",
                3,
                11,
            );
            assert_eq!(d, ArmDecision::Proceed, "no second open experiment");
        }
        assert_eq!(c.open_len(), 1);
        assert!(
            c.metrics().refused_budget > 0,
            "budget refusals must be visible, not silent"
        );
        assert!(c.open_len() <= MAX_OPEN_GLOBAL);
    }

    #[test]
    fn the_sampling_rate_stays_at_or_under_one_percent() {
        // Draws are counted even when the budget refuses them, so a full budget
        // cannot silently raise the rate of everything that follows.
        let mut c = canary();
        for i in 0..200_000u64 {
            c.offer(
                ActuatorFamily::MarkovPrewarm,
                ActionClass::MarkovPredictedApp,
                ExplorationArm::MarkovCacheOnly,
                "markov_prewarm:predicted_app@cache_only",
                3,
                i,
            );
            c.expire(i + EXPERIMENT_HORIZON_CYCLES + 1);
        }
        let rate = c.observed_per_mille();
        assert!(
            rate <= MARKOV_SAMPLE_PER_MILLE as f64 * 1.25,
            "observed {rate:.2}‰ over the 10‰ ceiling"
        );
    }

    // ── 8. starvation ───────────────────────────────────────────────────────

    #[test]
    fn a_blocked_family_never_starves_the_others() {
        // Only MarkovPrewarm is enabled, so every other family must pass
        // through untouched however long Markov holds its slot.
        let mut c = canary();
        offer_until_sampled(&mut c, 10).expect("markov holds its slot");
        for family in [ActuatorFamily::Boost, ActuatorFamily::InteractionQos] {
            for _ in 0..1_000 {
                assert_eq!(
                    c.offer(
                        family,
                        ActionClass::BoostBackground,
                        ExplorationArm::BoostOmission,
                        "boost:background@omission",
                        3,
                        11,
                    ),
                    ArmDecision::Proceed,
                    "a family that is not enabled is never withheld"
                );
            }
        }
        assert!(c.metrics().refused_family > 0);
        assert_eq!(c.open_len(), 1, "and no slot was taken from Markov");
    }

    // ── 9. kill switch ──────────────────────────────────────────────────────

    #[test]
    fn the_kill_switch_stops_sampling_and_lets_open_work_drain() {
        let mut c = canary();
        let d = offer_until_sampled(&mut c, 10).expect("sampled");
        let (id, corr) = ids(&d);
        c.disable();
        assert!(!c.is_enabled());
        for _ in 0..50_000 {
            assert_eq!(
                c.offer(
                    ActuatorFamily::MarkovPrewarm,
                    ActionClass::MarkovPredictedApp,
                    ExplorationArm::MarkovCacheOnly,
                    "markov_prewarm:predicted_app@cache_only",
                    3,
                    11,
                ),
                ArmDecision::Proceed
            );
        }
        assert_eq!(c.metrics().sampled, 1, "no new draw after the switch");
        // The experiment already withheld still completes: killing the switch
        // must not strand an arm that was already taken from the machine.
        let state = match d {
            ArmDecision::Treatment { .. } => ArmTerminalState::AppliedNoRevertNeeded,
            _ => ArmTerminalState::WithheldByHoldout,
        };
        assert!(c.record_arm(id, corr, 11, state, 1).is_none());
        let other = c.complementary_arm(id).expect("complement still available");
        let (_, corr2) = ids(&other);
        let state2 = match other {
            ArmDecision::Treatment { .. } => ArmTerminalState::AppliedNoRevertNeeded,
            _ => ArmTerminalState::WithheldByHoldout,
        };
        assert!(c.record_arm(id, corr2, 12, state2, 1).is_some());
    }

    #[test]
    fn an_issued_control_that_was_never_honoured_is_visible() {
        // The one failure no other counter would show: the caller is told to
        // withhold, goes ahead anyway, and the pair's control arm is not a
        // control. Issuing and honouring are counted separately so the gap
        // between them is readable.
        let mut c = canary();
        let mut issued_controls = 0;
        for cycle in 0..40_000u64 {
            if let ArmDecision::WithholdAsControl { .. } = c.offer(
                ActuatorFamily::MarkovPrewarm,
                ActionClass::MarkovPredictedApp,
                ExplorationArm::MarkovCacheOnly,
                "markov_prewarm:predicted_app@cache_only",
                3,
                cycle,
            ) {
                issued_controls += 1;
                if issued_controls > 1 {
                    // Deliberately do not confirm this one.
                    break;
                }
                c.confirm_control_honoured();
            }
            c.expire(cycle + EXPERIMENT_HORIZON_CYCLES + 1);
        }
        assert!(issued_controls >= 2, "the draw must reach two controls");
        assert_eq!(c.metrics().control_issued, issued_controls);
        assert_eq!(
            c.metrics().control_honoured,
            issued_controls - 1,
            "the unhonoured control shows as a gap, not as silence"
        );
    }

    #[test]
    fn terminal_states_and_the_open_watermark_are_counted() {
        let mut c = canary();
        let pair = complete_one(&mut c, 10).expect("pair");
        let m = c.metrics();
        assert_eq!(m.arms_withheld, 1, "exactly one control arm settled");
        assert_eq!(
            m.arms_applied_reverted + m.arms_applied_no_revert,
            1,
            "and exactly one treatment arm"
        );
        assert!(m.open_high_watermark >= 1, "the peak is remembered");
        assert_eq!(m.pairs_completed, 1);
        assert_eq!(
            pair.control.terminal_state,
            ArmTerminalState::WithheldByHoldout
        );
    }

    #[test]
    fn every_canary_counter_reaches_runtime_metrics() {
        // The same walk that caught three counters stopping short of
        // `RuntimeMetrics` in the lab. Drive one full experiment so nothing is
        // zero by accident, then require each field to arrive.
        let mut c = canary();
        complete_one(&mut c, 10).expect("pair");
        c.confirm_control_honoured();
        c.offer(
            ActuatorFamily::Boost,
            ActionClass::BoostBackground,
            ExplorationArm::BoostOmission,
            "boost:background@omission",
            3,
            11,
        );
        let mut out = crate::engine::types::RuntimeMetrics::default();
        c.publish(&mut out);

        assert!(out.canary_enabled);
        assert_eq!(out.canary_eligible_seen, c.metrics().eligible_seen);
        assert_eq!(out.canary_sampled, 1);
        assert_eq!(out.canary_refused_family, 1);
        assert_eq!(out.canary_control_honoured, 1);
        assert_eq!(out.canary_arms_withheld, 1);
        assert_eq!(
            out.canary_arms_applied_reverted + out.canary_arms_applied_no_revert,
            1
        );
        assert_eq!(out.canary_pairs_completed, 1);
        assert_eq!(out.canary_assembly_refused, 0);
        assert_eq!(out.canary_pairs_censored, 0);
        assert_eq!(out.canary_treatment_confounded, 0);
        assert_eq!(out.canary_control_incomplete, 0);
        assert!(out.canary_open_high_watermark >= 1);
        assert!(out.canary_observed_per_mille > 0.0);
        assert_eq!(
            out.canary_treatment_issued + out.canary_control_issued,
            1,
            "one arm was handed out for the one experiment"
        );
    }

    #[test]
    fn a_disabled_producer_is_indistinguishable_from_not_having_one() {
        // The property the whole pre-B baseline rests on. If a wired-but-off
        // canary differed from no canary at all, every comparison against the
        // baseline would be measuring the wiring instead of B.
        let mut c = MicroCanary::new_disabled(BOOT);
        assert!(!c.is_enabled());
        for cycle in 0..100_000u64 {
            assert_eq!(
                c.offer(
                    ActuatorFamily::MarkovPrewarm,
                    ActionClass::MarkovPredictedApp,
                    ExplorationArm::MarkovCacheOnly,
                    "markov_prewarm:predicted_app@cache_only",
                    1,
                    cycle,
                ),
                ArmDecision::Proceed,
                "a disabled producer never asks for anything to be withheld"
            );
        }
        let m = c.metrics();
        assert_eq!(m.sampled, 0);
        assert_eq!(
            m.control_issued, 0,
            "no action was withheld from the machine"
        );
        assert_eq!(m.control_honoured, 0);
        assert_eq!(m.treatment_issued, 0);
        assert_eq!(m.pairs_completed, 0);
        assert_eq!(c.open_len(), 0);
        assert_eq!(m.refused_disabled, 100_000, "and the refusals are visible");

        let mut out = crate::engine::types::RuntimeMetrics::default();
        c.publish(&mut out);
        assert!(!out.canary_enabled);
        assert_eq!(out.canary_control_issued, 0);
        assert_eq!(out.canary_observed_per_mille, 0.0);
    }

    #[test]
    fn enabling_is_the_only_thing_that_starts_it() {
        let mut c = MicroCanary::new_disabled(BOOT);
        assert!(offer_until_sampled(&mut c, 1).is_none(), "off means off");
        c.enable();
        assert!(
            offer_until_sampled(&mut c, 1).is_some(),
            "and on means the draw resumes"
        );
    }

    // ── The four measurement invariants ─────────────────────────────────────

    #[test]
    fn invariant_1_the_arm_is_assigned_before_anything_could_act() {
        // `offer` returns the assignment and takes no action itself. The caller
        // cannot have emitted a pre-warm before it knows which arm it is in,
        // because the answer is what tells it whether to emit one.
        let mut c = canary();
        let d = offer_until_sampled(&mut c, 10).expect("sampled");
        let (id, _) = ids(&d);
        assert_eq!(c.open_len(), 1, "the experiment exists at assignment time");
        assert!(
            c.correlations(id).is_some(),
            "both correlations are minted up front, before either arm runs"
        );
        assert_eq!(c.metrics().arms_withheld, 0, "and nothing has settled yet");
    }

    #[test]
    fn invariant_2_both_arms_are_measured_on_the_same_yardstick() {
        // Both keys go to the same `open_lab_utility_window(family, horizon)`,
        // so the objective and its components are identical by construction.
        // What this test can pin is that the two keys are distinct and both
        // namespaced, so neither arm can silently reuse the other's window.
        let mut c = canary();
        let d = offer_until_sampled(&mut c, 10).expect("sampled");
        let (id, _) = ids(&d);
        let (t, k) = c.correlations(id).expect("both correlations");
        assert_ne!(t.ledger_correlation_id(), k.ledger_correlation_id());
        assert_eq!(
            c.arm_for_ledger_id(t.ledger_correlation_id()),
            Some((id, true))
        );
        assert_eq!(
            c.arm_for_ledger_id(k.ledger_correlation_id()),
            Some((id, false))
        );
        assert_eq!(
            c.arm_for_ledger_id(999_999),
            None,
            "a foreign key matches nothing"
        );
    }

    #[test]
    fn invariant_3_a_treatment_that_never_actuated_is_abandoned_not_zeroed() {
        // The trap this closes: an opportunity assigned to treatment whose
        // pre-warm then did not run. Its arm is not a treatment. Recording it
        // as a zero-utility observation would put a measurement that never
        // happened into the estimator.
        let mut c = canary();
        let d = offer_until_sampled(&mut c, 10).expect("sampled");
        let (id, _) = ids(&d);
        assert!(c.abandon_not_comparable(id));
        assert_eq!(c.open_len(), 0);
        assert_eq!(c.metrics().abandoned_not_comparable, 1);
        assert_eq!(c.metrics().pairs_completed, 0, "and no pair was produced");
        assert!(!c.abandon_not_comparable(id), "abandoning twice is a no-op");
    }

    #[test]
    fn invariant_4_a_pair_closes_only_when_both_sides_have_a_terminal_observation() {
        let mut c = canary();
        let d = offer_until_sampled(&mut c, 10).expect("sampled");
        let (id, corr) = ids(&d);
        let state = match d {
            ArmDecision::Treatment { .. } => ArmTerminalState::AppliedNoRevertNeeded,
            _ => ArmTerminalState::WithheldByHoldout,
        };
        assert!(
            c.record_arm(id, corr, 11, state, 500).is_none(),
            "one side is not a pair"
        );
        assert_eq!(c.metrics().pairs_completed, 0);
        let other = c.complementary_arm(id).expect("complement");
        let (_, corr2) = ids(&other);
        let state2 = match other {
            ArmDecision::Treatment { .. } => ArmTerminalState::AppliedNoRevertNeeded,
            _ => ArmTerminalState::WithheldByHoldout,
        };
        assert!(
            c.record_arm(id, corr2, 12, state2, 200).is_some(),
            "both sides terminal: now it is comparable"
        );
    }

    #[test]
    fn prediction_resolution_is_diagnostic_and_never_the_effect() {
        // hit/miss shows the arms drew equivalent opportunities. It contributes
        // nothing to the effect, which is utility_treatment - utility_control.
        let mut c = canary();
        c.note_prediction_resolved(true);
        c.note_prediction_resolved(false);
        let m = c.metrics();
        assert_eq!(m.treatment_predictions_resolved, 1);
        assert_eq!(m.control_predictions_resolved, 1);
        assert_eq!(m.pairs_completed, 0, "diagnostics close no pair");

        let pair = complete_one(&mut c, 10).expect("pair");
        assert_eq!(
            pair.effect_micros(),
            60,
            "the effect is the utility difference, nothing else"
        );
    }

    #[test]
    fn a_confounded_arm_is_censored_by_arm_and_never_becomes_a_zero() {
        // Not measuring is correct. Measuring zero is a lie. And if one arm
        // loses more windows than the other, excluding them quietly would bias
        // the effect: the surviving treatments would be the ones that went
        // well. So the loss is named, and named by arm.
        let mut c = canary();
        let d = offer_until_sampled(&mut c, 10).expect("sampled");
        let (id, _) = ids(&d);
        let was_treatment = matches!(d, ArmDecision::Treatment { .. });

        assert!(c.censor(id, Some(was_treatment), CensorCause::Confounded));

        let m = c.metrics();
        assert_eq!(m.pairs_censored, 1, "the pair left the estimator");
        assert_eq!(m.pairs_completed, 0, "it contributed no effect");
        if was_treatment {
            assert_eq!(m.treatment_confounded, 1);
            assert_eq!(m.control_confounded, 0);
        } else {
            assert_eq!(m.control_confounded, 1);
            assert_eq!(m.treatment_confounded, 0);
        }
        // The utility counters are untouched: nothing was recorded as zero.
        assert_eq!(m.arms_withheld, 0);
        assert_eq!(m.arms_applied_reverted + m.arms_applied_no_revert, 0);
        assert_eq!(c.open_len(), 0);
        assert!(
            !c.censor(id, Some(was_treatment), CensorCause::Confounded),
            "censoring twice is a no-op"
        );
    }

    #[test]
    fn an_expired_experiment_names_the_arm_that_never_reported() {
        let mut c = canary();
        let d = offer_until_sampled(&mut c, 10).expect("sampled");
        let (id, corr) = ids(&d);
        let first_is_treatment = matches!(d, ArmDecision::Treatment { .. });
        let state = if first_is_treatment {
            ArmTerminalState::AppliedNoRevertNeeded
        } else {
            ArmTerminalState::WithheldByHoldout
        };
        // One side reports; the other never does.
        assert!(c.record_arm(id, corr, 11, state, 77).is_none());
        c.expire(11 + EXPERIMENT_HORIZON_CYCLES);

        let m = c.metrics();
        assert_eq!(m.pairs_expired, 1);
        assert_eq!(m.pairs_censored, 1);
        if first_is_treatment {
            assert_eq!(m.control_incomplete, 1, "the control never reported");
            assert_eq!(m.treatment_incomplete, 0);
        } else {
            assert_eq!(m.treatment_incomplete, 1);
            assert_eq!(m.control_incomplete, 0);
        }
        assert_eq!(m.pairs_completed, 0);
    }

    #[test]
    fn attrition_can_be_shown_non_differential() {
        // The property the counters exist to support: with censoring split by
        // arm, a reader can compare the two sides instead of taking a single
        // aggregate on trust.
        let mut c = canary();
        for seq in 0..6u64 {
            let d = offer_until_sampled(&mut c, seq * 100 + 1);
            let Some(d) = d else { break };
            let (id, _) = ids(&d);
            c.censor(id, Some(seq % 2 == 0), CensorCause::Confounded);
        }
        let m = c.metrics();
        assert_eq!(
            m.treatment_confounded + m.control_confounded,
            m.pairs_censored,
            "every censored pair is attributable to a side"
        );
        assert_eq!(m.pairs_completed, 0);
    }

    // ── 10. contractual reachability ────────────────────────────────────────

    #[test]
    fn the_enabled_configuration_can_actually_produce_the_evidence_it_owes() {
        // The test that would have caught both earlier designs: Shadow owed a
        // control endpoint no Shadow producer could emit, and hypothesis C
        // leaned on a scheduler that had minted six correlations in its whole
        // history. Assert the producer can reach every terminal state the
        // contract requires of it.
        let mut c = canary();
        let pair = complete_one(&mut c, 10).expect("the enabled family completes a pair");
        assert!(matches!(
            pair.treatment.terminal_state,
            ArmTerminalState::AppliedAndReverted | ArmTerminalState::AppliedNoRevertNeeded
        ));
        assert_eq!(
            pair.control.terminal_state,
            ArmTerminalState::WithheldByHoldout,
            "a control must come from a deliberate holdout"
        );
        assert_eq!(pair.effect_micros(), 60, "100 treatment - 40 control");
    }
}
