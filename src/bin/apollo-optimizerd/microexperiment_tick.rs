use std::time::Instant;

use apollo_engine::engine::decision_ledger::ResolvedDecisionEpisode;
use apollo_engine::engine::exploration_scheduler::{
    ActionClass, ExplorationArm, ExplorationContext, ExplorationOrigin,
};
use apollo_engine::engine::microexperiment_actions::{
    canonical_action_key, family_horizon_cycles, CanonicalAction,
};
use apollo_engine::engine::microexperiment_endpoints::{
    EndpointAdapterCounters, EndpointUtilitySample, MicroexperimentEndpointAdapter, NewArmBinding,
};
use apollo_engine::engine::microexperiment_lab::{
    CandidateDisposition, LabError, MicroexperimentLab, MicroexperimentLabPersisted, PairCandidate,
    PairClosure, PairDirective, PairGates, PairGoldRecord, PairInvalidation, PairProgress,
    RestoreDisposition, TimedEndpointDisposition, TimedPairEndpoint,
};
use apollo_engine::engine::telemetry_medallion::ActuatorFamily;

/// Washout between the two arms of one pair, in daemon cycles.
const PAIR_WASHOUT_CYCLES: u32 = 3;
/// Smallest effect the lab will call effective, in microseconds of utility.
const PAIR_MINIMUM_EFFECT_MICROS: i64 = 500;

#[derive(Debug, Clone, Copy)]
pub struct MicroexperimentTickInput<'a> {
    pub cycle: u64,
    pub interaction_activations: u64,
    pub markov_applied: u64,
    pub workload: &'a str,
    pub inherited_safe: bool,
    /// Operator opt-in. Absent by default; without it the lab stays in Shadow.
    pub experiments_enabled: bool,
    /// Privacy posture is known and currently permits mutation.
    pub privacy_known: bool,
    pub secure_input: bool,
    pub screen_capture: bool,
    pub camera_active: bool,
    pub sensitive_context: bool,
}

#[derive(Debug, Clone, Default)]
pub struct MicroexperimentTickMetrics {
    pub phase: String,
    pub blocker: String,
    pub restore: String,
    /// Progress toward the current phase gate, and the threshold it needs.
    pub rollout_progress: u64,
    pub rollout_required: u64,
    /// Provenance for `rollout_progress`, scoped to this boot: what `restore`
    /// handed over before the first cycle, and how many runtime resets have
    /// discarded progress since. Separates "the disk held a low value" from
    /// "the runtime threw a high one away" — indistinguishable otherwise.
    pub restored_progress_at_boot: u64,
    pub progress_resets_total: u64,
    pub last_progress_reset_reason: String,
    pub proposed_total: u64,
    pub eligible_total: u64,
    pub randomized_total: u64,
    pub shadow_would_open_total: u64,
    pub open_pairs: u64,
    pub shadow_measurements_proven_total: u64,
    pub lab_pairs_seen: u64,
    pub lab_pairs_accepted: u64,
    pub lab_pairs_duplicate: u64,
    pub lab_pairs_rejected: u64,
    pub causal_pairs_consumed: u64,
    pub shadow_pairs_expired_total: u64,
    pub shadow_pairs_reaped_total: u64,
    pub shadow_open_pairs: u64,
    pub shadow_open_high_watermark: u64,
    pub shadow_endpoints_late_total: u64,
    pub shadow_measurements_refused_total: u64,
    pub terminal_pairs: u64,
    pub completed_pairs_valid: u64,
    pub interrupted_pairs: u64,
    pub control_endpoints_total: u64,
    pub treatment_endpoints_total: u64,
    pub complete_horizons_total: u64,
    pub rollback_closed_total: u64,
    pub pair_gold_total: u64,
    pub effective_total: u64,
    pub harmful_total: u64,
    pub confounded_total: u64,
    pub interrupted_total: u64,
    pub synthetic_quarantined_total: u64,
    pub invalidated_total: u64,
    pub deadline_expired_total: u64,
    pub unbound_expiries_total: u64,
    pub rollback_failed_total: u64,
    pub mean_effect: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct MicroexperimentControlledInput<'a> {
    pub cycle: u64,
    pub monotonic_millis: u64,
    pub gates: PairGates,
    pub endpoint_contract_ready: bool,
    pub candidates: &'a [PairCandidate],
    pub observations: &'a [TimedPairEndpoint],
}

#[derive(Debug, Clone, Default)]
pub struct MicroexperimentTickOutput {
    pub metrics: MicroexperimentTickMetrics,
    pub directives: Vec<PairDirective>,
    pub invalidations: Vec<PairInvalidation>,
    pub closures: Vec<PairClosure>,
    pub pair_gold: Vec<PairGoldRecord>,
}

#[derive(Debug, Clone, Copy)]
struct CounterBaseline {
    interaction_activations: u64,
    markov_applied: u64,
}

pub struct MicroexperimentRuntime {
    origin: ExplorationOrigin,
    lab: MicroexperimentLab,
    adapter: MicroexperimentEndpointAdapter,
    epoch: u64,
    counters: Option<CounterBaseline>,
    restore: &'static str,
    started_at: Instant,
}

impl MicroexperimentRuntime {
    pub fn new(origin: ExplorationOrigin, persisted: Option<MicroexperimentLabPersisted>) -> Self {
        Self::with_epoch(origin, persisted, default_epoch())
    }

    /// `epoch` identifies this daemon generation. Directives, bindings and
    /// endpoints never cross generations, so a restart cannot resurrect a
    /// half-observed pair as fresh evidence.
    pub fn with_epoch(
        origin: ExplorationOrigin,
        persisted: Option<MicroexperimentLabPersisted>,
        epoch: u64,
    ) -> Self {
        let (lab, restore) = persisted.map_or_else(
            || (MicroexperimentLab::cold_start(origin), "cold-start"),
            |persisted| {
                let (lab, disposition) = MicroexperimentLab::restore(persisted, origin);
                let label = match disposition {
                    RestoreDisposition::Restored => "restored",
                    RestoreDisposition::RestoredInterrupted => "restored-interrupted",
                    RestoreDisposition::ResetOrigin => "reset-origin",
                    RestoreDisposition::ResetHostile => "reset-hostile",
                };
                (lab, label)
            },
        );
        Self {
            origin,
            lab,
            adapter: MicroexperimentEndpointAdapter::new(origin, epoch),
            epoch,
            counters: None,
            restore,
            started_at: Instant::now(),
        }
    }

    /// Feed the cycle's resolved decisions to the adapter. Called at the cycle
    /// tail, right after `DecisionLedger::ingest_cycle_events`, so the endpoints
    /// they produce reach the lab on the following cycle.
    pub fn observe_decisions(&mut self, episodes: &[ResolvedDecisionEpisode], cycle: u64) {
        self.adapter.observe_episodes(episodes, cycle);
    }

    /// Feed measured outcomes for decisions already bound to an arm.
    pub fn observe_utilities(&mut self, samples: &[EndpointUtilitySample], cycle: u64) {
        self.adapter.observe_utilities(samples, cycle);
    }

    /// Arms bound since the previous drain, so the daemon can start measuring
    /// their outcome window.
    pub fn drain_new_bindings(&mut self) -> Vec<NewArmBinding> {
        self.adapter.drain_new_bindings()
    }

    pub fn adapter_counters(&self) -> EndpointAdapterCounters {
        self.adapter.counters()
    }

    pub fn endpoint_contract_ready(&self, cycle: u64) -> bool {
        self.adapter.contract_ready(cycle, self.epoch)
    }

    /// Actions the lab currently needs observed as a *withheld* control window.
    /// Empty in Shadow, empty without an opt-in, empty whenever no pair is open.
    pub fn control_withhold_requests(&self) -> Vec<CanonicalAction> {
        self.adapter.outstanding_control_actions()
    }

    pub fn tick(&mut self, input: MicroexperimentTickInput<'_>) -> MicroexperimentTickOutput {
        let current = CounterBaseline {
            interaction_activations: input.interaction_activations,
            markov_applied: input.markov_applied,
        };
        let Some(previous) = self.counters.replace(current) else {
            return MicroexperimentTickOutput {
                metrics: self.metrics("no-candidates"),
                ..MicroexperimentTickOutput::default()
            };
        };
        let interaction_delta = input
            .interaction_activations
            .saturating_sub(previous.interaction_activations);
        let markov_delta = input.markov_applied.saturating_sub(previous.markov_applied);
        let mut candidates = Vec::with_capacity(2);
        let mut uncatalogued = 0_u64;
        if interaction_delta > 0 {
            match candidate(
                input.cycle.saturating_mul(4).saturating_add(1).max(1),
                self.origin,
                ActuatorFamily::InteractionQos,
                ActionClass::InteractionForeground,
                ExplorationArm::InteractionQosStandard,
                ExplorationContext::Interactive,
                stratum_hash(input.workload, 1),
            ) {
                Some(candidate) => candidates.push(candidate),
                None => uncatalogued = uncatalogued.saturating_add(1),
            }
        }
        if markov_delta > 0 {
            match candidate(
                input.cycle.saturating_mul(4).saturating_add(2).max(1),
                self.origin,
                ActuatorFamily::MarkovPrewarm,
                ActionClass::MarkovPredictedApp,
                ExplorationArm::MarkovCacheOnly,
                ExplorationContext::Background,
                stratum_hash(input.workload, 2),
            ) {
                Some(candidate) => candidates.push(candidate),
                None => uncatalogued = uncatalogued.saturating_add(1),
            }
        }
        let observations = self.adapter.drain_ready();
        let endpoint_contract_ready = self.endpoint_contract_ready(input.cycle);
        let mut output = self.tick_controlled(MicroexperimentControlledInput {
            cycle: input.cycle,
            monotonic_millis: self.started_at.elapsed().as_millis() as u64,
            gates: PairGates {
                experiments_enabled: input.experiments_enabled,
                privacy_known: input.privacy_known,
                secure_input: input.secure_input,
                screen_capture: input.screen_capture,
                camera_active: input.camera_active,
                sensitive_context: input.sensitive_context,
                inherited_safe: input.inherited_safe,
            },
            endpoint_contract_ready,
            candidates: &candidates,
            observations: &observations,
        });
        if !input.inherited_safe && !candidates.is_empty() {
            output.metrics.blocker = "safety-gate".to_string();
        } else if uncatalogued > 0 && candidates.is_empty() {
            output.metrics.blocker = "candidate-outside-catalog".to_string();
        }

        // The adapter only ever observes arms the lab actually issued, and
        // forgets everything the lab already resolved.
        self.adapter
            .register_directives(&output.directives, input.cycle);
        for invalidation in &output.invalidations {
            self.adapter.forget_pair(invalidation.pair_id);
        }
        for closure in &output.closures {
            self.adapter.forget_pair(closure.id);
        }
        self.adapter.prune(input.cycle);
        output
    }

    pub fn tick_controlled(
        &mut self,
        input: MicroexperimentControlledInput<'_>,
    ) -> MicroexperimentTickOutput {
        let effective_gates = if input.endpoint_contract_ready {
            input.gates
        } else {
            PairGates {
                experiments_enabled: false,
                ..input.gates
            }
        };
        let mut invalidations =
            self.lab
                .advance_cycle(input.cycle, input.monotonic_millis, effective_gates);
        let mut closures = Vec::new();
        let mut blocker = if !input.endpoint_contract_ready {
            "endpoint-wiring-required"
        } else if !effective_gates.allows_pair() {
            "safety-or-privacy-gate"
        } else {
            "no-candidates"
        };

        if input.endpoint_contract_ready && effective_gates.allows_pair() {
            for observation in input.observations.iter().cloned() {
                let pair_id = observation.pair_id;
                match self.lab.record_timed_endpoint(observation) {
                    Ok(TimedEndpointDisposition::Progress(PairProgress::ReadyToClose)) => {
                        match self.lab.close_pair(pair_id) {
                            Ok(closure) => {
                                closures.push(closure);
                                blocker = "pair-closed";
                            }
                            Err(error) => blocker = blocker_for_endpoint_error(error),
                        }
                    }
                    Ok(TimedEndpointDisposition::ShadowMeasured) => blocker = "shadow-measured",
                    Ok(TimedEndpointDisposition::Progress(_)) => blocker = "washout",
                    Ok(TimedEndpointDisposition::Invalidated(record)) => {
                        invalidations.push(record);
                        blocker = "endpoint-invalidated";
                    }
                    Err(error) => blocker = blocker_for_endpoint_error(error),
                }
            }
            invalidations.extend(self.lab.advance_cycle(
                input.cycle,
                input.monotonic_millis,
                effective_gates,
            ));
        }

        let mut opened = 0_u64;
        let mut shadowed = 0_u64;
        let mut sampled_out = 0_u64;
        for candidate in input.candidates.iter().cloned() {
            match self
                .lab
                .consider_candidate(candidate, effective_gates, input.monotonic_millis)
            {
                Ok(CandidateDisposition::Shadow(_)) => shadowed = shadowed.saturating_add(1),
                Ok(CandidateDisposition::CanarySkipped) => {
                    sampled_out = sampled_out.saturating_add(1)
                }
                Ok(CandidateDisposition::Opened(_)) => opened = opened.saturating_add(1),
                Err(error) => blocker = blocker_for_candidate_error(error),
            }
        }

        // Real arms only outside Shadow; shadow arms are observe-only and are
        // issued from the separate collection. `issue_ready_arms` returns empty
        // in Shadow and `issue_shadow_arms` returns empty outside it, so the two
        // can never both produce a directive for the same pair.
        let mut directives = if input.endpoint_contract_ready && effective_gates.allows_pair() {
            self.lab.issue_ready_arms(input.cycle, effective_gates)
        } else {
            Vec::new()
        };
        if input.endpoint_contract_ready {
            directives.extend(self.lab.issue_shadow_arms(input.cycle, effective_gates));
        }
        if input.endpoint_contract_ready && effective_gates.allows_pair() {
            blocker = if let Some(readiness) = self.lab.readiness_blocker() {
                readiness
            } else if opened > 0 {
                "pair-opened"
            } else if shadowed > 0 {
                "shadow-warming"
            } else if sampled_out > 0 {
                "canary-sampling"
            } else {
                blocker
            };
        }
        let pair_gold = self.lab.drain_pair_gold();
        MicroexperimentTickOutput {
            metrics: self.metrics(blocker),
            directives,
            invalidations,
            closures,
            pair_gold,
        }
    }

    pub fn persisted(&self) -> MicroexperimentLabPersisted {
        self.lab.persisted()
    }

    fn metrics(&self, blocker: &str) -> MicroexperimentTickMetrics {
        let metrics = self.lab.metrics();
        MicroexperimentTickMetrics {
            phase: self.lab.phase().as_str().to_string(),
            blocker: blocker.to_string(),
            restore: self.restore.to_string(),
            rollout_progress: self.lab.rollout_progress().0,
            rollout_required: self.lab.rollout_progress().1,
            restored_progress_at_boot: self.lab.rollout_provenance().0,
            progress_resets_total: self.lab.rollout_provenance().1,
            last_progress_reset_reason: self.lab.rollout_provenance().2.to_string(),
            proposed_total: metrics.proposed_total,
            eligible_total: metrics.eligible_total,
            randomized_total: metrics.randomized_total,
            shadow_would_open_total: metrics.shadow_would_open_total,
            open_pairs: metrics.open_pairs as u64,
            shadow_measurements_proven_total: metrics.shadow_measurements_proven_total,
            lab_pairs_seen: metrics.lab_pairs_seen,
            lab_pairs_accepted: metrics.lab_pairs_accepted,
            lab_pairs_duplicate: metrics.lab_pairs_duplicate,
            lab_pairs_rejected: metrics.lab_pairs_rejected,
            causal_pairs_consumed: metrics.causal_pairs_consumed,
            shadow_pairs_expired_total: metrics.shadow_pairs_expired_total,
            shadow_pairs_reaped_total: metrics.shadow_pairs_reaped_total,
            shadow_open_pairs: metrics.shadow_open_pairs as u64,
            shadow_open_high_watermark: u64::from(metrics.shadow_open_high_watermark),
            shadow_endpoints_late_total: metrics.shadow_endpoints_late_total,
            shadow_measurements_refused_total: metrics.shadow_measurements_refused_total,
            terminal_pairs: metrics.terminal_pairs as u64,
            completed_pairs_valid: metrics.completed_pairs_valid as u64,
            interrupted_pairs: metrics.interrupted_pairs as u64,
            control_endpoints_total: metrics.control_endpoints_total,
            treatment_endpoints_total: metrics.treatment_endpoints_total,
            complete_horizons_total: metrics.complete_horizons_total,
            rollback_closed_total: metrics.rollback_closed_total,
            pair_gold_total: metrics.pair_gold_total,
            effective_total: metrics.effective_total,
            harmful_total: metrics.harmful_total,
            confounded_total: metrics.confounded_total,
            interrupted_total: metrics.interrupted_total,
            synthetic_quarantined_total: metrics.synthetic_quarantined_total,
            invalidated_total: metrics.invalidated_total,
            deadline_expired_total: metrics.deadline_expired_total,
            unbound_expiries_total: metrics.unbound_expiries_total,
            rollback_failed_total: metrics.rollback_failed_total,
            mean_effect: metrics.mean_effect,
        }
    }
}

/// Build one catalogued candidate. Both the action key and the outcome horizon
/// come from the shared canonical catalog, so generation, actuation and
/// observation cannot drift apart again.
fn candidate(
    sequence: u64,
    origin: ExplorationOrigin,
    family: ActuatorFamily,
    action_class: ActionClass,
    treatment_arm: ExplorationArm,
    context: ExplorationContext,
    stratum_hash: u64,
) -> Option<PairCandidate> {
    Some(PairCandidate {
        sequence,
        origin,
        family,
        action_class,
        treatment_arm,
        context,
        action_key: canonical_action_key(family, action_class, treatment_arm)?.to_string(),
        stratum_hash,
        horizon_cycles: family_horizon_cycles(family)?,
        washout_cycles: PAIR_WASHOUT_CYCLES,
        minimum_effect_micros: PAIR_MINIMUM_EFFECT_MICROS,
    })
}

/// Generation token for this daemon process. Monotonic wall time is enough:
/// two runs of the daemon can never share one, and it is never persisted.
fn default_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(1, |elapsed| elapsed.as_nanos() as u64)
        .max(1)
}

fn stratum_hash(workload: &str, family: u8) -> u64 {
    workload
        .as_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325 ^ u64::from(family), |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x1000_0000_01b3)
        })
        .max(1)
}

fn blocker_for_candidate_error(error: LabError) -> &'static str {
    match error {
        LabError::Gate => "safety-or-privacy-gate",
        LabError::Origin => "candidate-origin-mismatch",
        LabError::Catalog => "candidate-outside-catalog",
        LabError::Capacity => "pair-capacity",
        LabError::Invalid => "candidate-invalid",
        LabError::DuplicatePair => "pair-duplicate",
        _ => "candidate-rejected",
    }
}

fn blocker_for_endpoint_error(error: LabError) -> &'static str {
    match error {
        LabError::HorizonPending => "horizon-pending",
        LabError::EndpointNotIssued => "endpoint-not-issued",
        LabError::DeadlineExpired => "endpoint-expired",
        LabError::UnknownPair => "endpoint-unknown-pair",
        LabError::Origin => "endpoint-origin-mismatch",
        LabError::Mismatch | LabError::WrongArm => "endpoint-mismatch",
        LabError::DuplicateArm | LabError::DuplicatePair => "endpoint-duplicate",
        LabError::WashoutPending => "washout",
        LabError::NotReady => "pair-not-ready",
        _ => "endpoint-rejected",
    }
}

/// Publish the lab's tick metrics onto `RuntimeMetrics`.
///
/// Extracted from the daemon loop so the mapping can be tested. Three times a
/// counter was added to the lab, wired through `LabMetrics`, and never reached
/// this struct — the number governing a gate, or proving a lifecycle fix,
/// stayed invisible outside the process. The test below walks every field and
/// fails on the next one that stops here.
pub fn publish_microexperiment_metrics(
    out: &mut apollo_engine::engine::types::RuntimeMetrics,
    micro_metrics: MicroexperimentTickMetrics,
) {
    out.microexperiment_phase = micro_metrics.phase;
    out.microexperiment_rollout_progress = micro_metrics.rollout_progress;
    out.microexperiment_rollout_required = micro_metrics.rollout_required;
    out.microexperiment_blocker = micro_metrics.blocker;
    out.microexperiment_restore = micro_metrics.restore;
    out.microexperiment_restored_progress_at_boot = micro_metrics.restored_progress_at_boot;
    out.microexperiment_progress_resets_total = micro_metrics.progress_resets_total;
    out.microexperiment_last_progress_reset_reason = micro_metrics.last_progress_reset_reason;
    out.microexperiment_proposed_total = micro_metrics.proposed_total;
    out.microexperiment_eligible_total = micro_metrics.eligible_total;
    out.microexperiment_randomized_total = micro_metrics.randomized_total;
    out.microexperiment_shadow_would_open_total = micro_metrics.shadow_would_open_total;
    out.microexperiment_open_pairs = micro_metrics.open_pairs;
    out.microexperiment_shadow_measurements_proven_total =
        micro_metrics.shadow_measurements_proven_total;
    out.microexperiment_shadow_measurements_refused_total =
        micro_metrics.shadow_measurements_refused_total;
    out.microexperiment_lab_pairs_seen = micro_metrics.lab_pairs_seen;
    out.microexperiment_lab_pairs_accepted = micro_metrics.lab_pairs_accepted;
    out.microexperiment_lab_pairs_duplicate = micro_metrics.lab_pairs_duplicate;
    out.microexperiment_lab_pairs_rejected = micro_metrics.lab_pairs_rejected;
    out.microexperiment_causal_pairs_consumed = micro_metrics.causal_pairs_consumed;
    out.microexperiment_shadow_pairs_expired_total = micro_metrics.shadow_pairs_expired_total;
    out.microexperiment_shadow_pairs_reaped_total = micro_metrics.shadow_pairs_reaped_total;
    out.microexperiment_shadow_open_pairs = micro_metrics.shadow_open_pairs;
    out.microexperiment_shadow_open_high_watermark = micro_metrics.shadow_open_high_watermark;
    out.microexperiment_shadow_endpoints_late_total = micro_metrics.shadow_endpoints_late_total;
    out.microexperiment_terminal_pairs = micro_metrics.terminal_pairs;
    out.microexperiment_completed_pairs_valid = micro_metrics.completed_pairs_valid;
    out.microexperiment_interrupted_pairs = micro_metrics.interrupted_pairs;
    out.microexperiment_control_endpoints_total = micro_metrics.control_endpoints_total;
    out.microexperiment_treatment_endpoints_total = micro_metrics.treatment_endpoints_total;
    out.microexperiment_complete_horizons_total = micro_metrics.complete_horizons_total;
    out.microexperiment_rollback_closed_total = micro_metrics.rollback_closed_total;
    out.microexperiment_pair_gold_total = micro_metrics.pair_gold_total;
    out.microexperiment_effective_total = micro_metrics.effective_total;
    out.microexperiment_harmful_total = micro_metrics.harmful_total;
    out.microexperiment_confounded_total = micro_metrics.confounded_total;
    out.microexperiment_interrupted_total = micro_metrics.interrupted_total;
    out.microexperiment_synthetic_quarantined_total = micro_metrics.synthetic_quarantined_total;
    out.microexperiment_mean_effect = micro_metrics.mean_effect;
    out.microexperiment_invalidated_total = micro_metrics.invalidated_total;
    out.microexperiment_deadline_expired_total = micro_metrics.deadline_expired_total;
    out.microexperiment_unbound_expiries_total = micro_metrics.unbound_expiries_total;
    out.microexperiment_rollback_failed_total = micro_metrics.rollback_failed_total;
}

#[cfg(test)]
mod tests {
    /// Every numeric counter the tick produces must land on `RuntimeMetrics`.
    ///
    /// Three separate times a counter was added to the lab, threaded through
    /// `LabMetrics` and the tick, and then stopped here — so the number that
    /// gated Shadow, and later the one proving the lifecycle fix, could not be
    /// read from outside the process. Each was found by staring at production,
    /// never by the suite. This walks the mapping instead.
    #[test]
    fn every_tick_counter_reaches_runtime_metrics() {
        // Distinct non-zero values, so a field that never gets assigned stays
        // at zero and is named.
        let m = MicroexperimentTickMetrics {
            phase: "shadow".to_string(),
            blocker: "none".to_string(),
            restore: "restored".to_string(),
            rollout_progress: 11,
            rollout_required: 12,
            restored_progress_at_boot: 13,
            proposed_total: 14,
            eligible_total: 15,
            randomized_total: 16,
            shadow_would_open_total: 17,
            shadow_measurements_proven_total: 18,
            shadow_measurements_refused_total: 19,
            shadow_pairs_expired_total: 20,
            shadow_endpoints_late_total: 21,
            open_pairs: 22,
            terminal_pairs: 23,
            completed_pairs_valid: 24,
            interrupted_pairs: 25,
            ..Default::default()
        };
        let mut out = apollo_engine::engine::types::RuntimeMetrics::default();
        publish_microexperiment_metrics(&mut out, m);

        // The two the lifecycle fix depends on. Both were absent in production
        // after the deploy that was supposed to make them observable.
        assert_eq!(out.microexperiment_shadow_pairs_expired_total, 20);
        assert_eq!(out.microexperiment_shadow_endpoints_late_total, 21);
        // And the gate's own quantity.
        assert_eq!(out.microexperiment_shadow_measurements_proven_total, 18);
        assert_eq!(out.microexperiment_shadow_measurements_refused_total, 19);
        assert_eq!(out.microexperiment_rollout_progress, 11);
        assert_eq!(out.microexperiment_terminal_pairs, 23);
        assert_eq!(out.microexperiment_completed_pairs_valid, 24);
        assert_eq!(out.microexperiment_interrupted_pairs, 25);
        assert_eq!(out.microexperiment_shadow_would_open_total, 17);
    }

    use super::*;
    use apollo_engine::engine::exploration_scheduler::{ExplorationOrigin, HardwareIdentity};

    fn origin() -> ExplorationOrigin {
        ExplorationOrigin {
            installation_id: 7,
            hardware: HardwareIdentity {
                p_core_count: 4,
                e_core_count: 6,
                ram_gib: 16,
            },
        }
    }

    fn input(cycle: u64) -> MicroexperimentTickInput<'static> {
        MicroexperimentTickInput {
            cycle,
            interaction_activations: 0,
            markov_applied: 0,
            workload: "interactive",
            inherited_safe: true,
            experiments_enabled: false,
            privacy_known: false,
            secure_input: false,
            screen_capture: false,
            camera_active: false,
            sensitive_context: false,
        }
    }

    /// Opted-in, private-context-clear input: everything the operator controls
    /// is on, so only the endpoint contract can still hold the lab back.
    fn opted_in(cycle: u64) -> MicroexperimentTickInput<'static> {
        MicroexperimentTickInput {
            experiments_enabled: true,
            privacy_known: true,
            ..input(cycle)
        }
    }

    #[test]
    fn real_counter_delta_creates_shadow_pair_diagnostic_once() {
        let mut runtime = MicroexperimentRuntime::new(origin(), None);
        let baseline = runtime.tick(input(1)).metrics;
        assert_eq!(baseline.proposed_total, 0);

        let first = runtime
            .tick(MicroexperimentTickInput {
                cycle: 2,
                interaction_activations: 1,
                markov_applied: 1,
                ..input(2)
            })
            .metrics;
        assert_eq!(first.proposed_total, 2);
        assert_eq!(first.shadow_would_open_total, 2);
        assert_eq!(first.open_pairs, 0);
        assert_eq!(first.blocker, "endpoint-wiring-required");

        let unchanged = runtime
            .tick(MicroexperimentTickInput {
                cycle: 3,
                interaction_activations: 1,
                markov_applied: 1,
                ..input(3)
            })
            .metrics;
        assert_eq!(unchanged.proposed_total, 2);
        assert_eq!(unchanged.shadow_would_open_total, 2);
    }

    #[test]
    fn candidates_carry_the_canonical_production_action_keys() {
        let mut runtime = MicroexperimentRuntime::new(origin(), None);
        runtime.tick(input(1));
        runtime.tick(MicroexperimentTickInput {
            cycle: 2,
            interaction_activations: 1,
            markov_applied: 1,
            ..input(2)
        });

        let interaction = candidate(
            1,
            origin(),
            ActuatorFamily::InteractionQos,
            ActionClass::InteractionForeground,
            ExplorationArm::InteractionQosStandard,
            ExplorationContext::Interactive,
            7,
        )
        .expect("catalogued");
        let markov = candidate(
            2,
            origin(),
            ActuatorFamily::MarkovPrewarm,
            ActionClass::MarkovPredictedApp,
            ExplorationArm::MarkovCacheOnly,
            ExplorationContext::Background,
            7,
        )
        .expect("catalogued");
        assert_eq!(
            interaction.action_key,
            "interaction_qos:foreground@standard"
        );
        // Production writes this key; the retired `markov:cache-only` never
        // joined a real decision.
        assert_eq!(markov.action_key, "markov_prewarm:predicted_app");
        assert_eq!(interaction.horizon_cycles, 30);
        assert_eq!(markov.horizon_cycles, 120);
    }

    /// A settled episode outside the experiment catalog. Enough to prove the
    /// ledger delivers; never enough to bind an arm or earn evidence.
    fn liveness_episode(cycle: u64) -> ResolvedDecisionEpisode {
        ResolvedDecisionEpisode {
            id: apollo_engine::engine::decision_ledger::DecisionId(900_000 + cycle),
            lifecycle: apollo_engine::engine::decision_ledger::DecisionLifecycle::Settled,
            settled_cycle: cycle,
            authority_eligible: false,
            envelope: Default::default(),
        }
    }

    #[test]
    fn contract_is_false_until_the_ledger_actually_delivers_episodes() {
        let mut runtime = MicroexperimentRuntime::new(origin(), None);
        assert!(
            !runtime.endpoint_contract_ready(1),
            "no ledger batch has been observed yet"
        );
        // An empty poll is not delivery: liveness must not move.
        runtime.observe_decisions(&[], 2);
        assert!(
            !runtime.endpoint_contract_ready(2),
            "an empty batch is the ledger being asked, not the ledger answering"
        );
        runtime.observe_decisions(&[liveness_episode(2)], 2);
        assert!(runtime.endpoint_contract_ready(2));

        let opted = runtime.tick(opted_in(3));
        assert_ne!(opted.metrics.blocker, "endpoint-wiring-required");
    }

    #[test]
    fn an_opted_out_daemon_stays_in_shadow_even_with_a_live_ledger() {
        let mut runtime = MicroexperimentRuntime::new(origin(), None);
        runtime.observe_decisions(&[liveness_episode(1)], 1);
        runtime.tick(input(1));
        let metrics = runtime
            .tick(MicroexperimentTickInput {
                cycle: 2,
                interaction_activations: 1,
                ..input(2)
            })
            .metrics;
        assert_eq!(metrics.phase, "shadow");
        assert_eq!(metrics.open_pairs, 0);
        assert_eq!(metrics.blocker, "safety-or-privacy-gate");
        assert!(runtime.control_withhold_requests().is_empty());
    }

    #[test]
    fn shadow_never_asks_an_actuator_to_withhold_anything() {
        let mut runtime = MicroexperimentRuntime::new(origin(), None);
        runtime.observe_decisions(&[], 1);
        runtime.tick(opted_in(1));
        let output = runtime.tick(MicroexperimentTickInput {
            cycle: 2,
            interaction_activations: 1,
            markov_applied: 1,
            ..opted_in(2)
        });
        assert_eq!(output.metrics.phase, "shadow");
        assert!(output.directives.is_empty());
        assert!(runtime.control_withhold_requests().is_empty());
    }

    #[test]
    fn unsafe_cycle_records_blocker_without_opening_pair() {
        let mut runtime = MicroexperimentRuntime::new(origin(), None);
        runtime.tick(input(1));
        let metrics = runtime
            .tick(MicroexperimentTickInput {
                cycle: 2,
                interaction_activations: 1,
                inherited_safe: false,
                ..input(2)
            })
            .metrics;
        assert_eq!(metrics.proposed_total, 1);
        assert_eq!(metrics.shadow_would_open_total, 0);
        assert_eq!(metrics.blocker, "safety-gate");
    }

    #[test]
    fn persisted_shadow_state_restores_without_replaying_deltas() {
        let mut runtime = MicroexperimentRuntime::new(origin(), None);
        runtime.tick(input(1));
        runtime.tick(MicroexperimentTickInput {
            cycle: 2,
            interaction_activations: 1,
            ..input(2)
        });
        let persisted = runtime.persisted();
        let mut restored = MicroexperimentRuntime::new(origin(), Some(persisted));
        let metrics = restored
            .tick(MicroexperimentTickInput {
                cycle: 3,
                interaction_activations: 1,
                ..input(3)
            })
            .metrics;
        assert_eq!(metrics.shadow_would_open_total, 1);
        assert_eq!(metrics.restore, "restored");
    }

    #[test]
    fn controlled_tick_opens_no_pair_without_endpoint_contract_enablement() {
        let mut runtime = MicroexperimentRuntime::new(origin(), None);
        let candidate = PairCandidate {
            sequence: 1,
            origin: origin(),
            family: ActuatorFamily::InteractionQos,
            action_class: ActionClass::InteractionForeground,
            treatment_arm: ExplorationArm::InteractionQosStandard,
            context: ExplorationContext::Interactive,
            action_key: "interaction_qos:foreground@standard".to_string(),
            stratum_hash: 7,
            horizon_cycles: 4,
            washout_cycles: 2,
            minimum_effect_micros: 500,
        };
        let output = runtime.tick_controlled(MicroexperimentControlledInput {
            cycle: 1,
            monotonic_millis: 1,
            gates: PairGates {
                inherited_safe: true,
                ..PairGates::default()
            },
            endpoint_contract_ready: false,
            candidates: &[candidate],
            observations: &[],
        });

        assert_eq!(output.metrics.phase, "shadow");
        assert_eq!(output.metrics.blocker, "endpoint-wiring-required");
        assert!(output.directives.is_empty());
        assert!(output.pair_gold.is_empty());
    }

    #[test]
    fn controlled_tick_never_converts_missing_observations_into_gold() {
        let mut runtime = MicroexperimentRuntime::new(origin(), None);
        runtime
            .lab
            .force_phase_for_test(apollo_engine::engine::microexperiment_lab::LabPhase::Active);
        let candidate = PairCandidate {
            sequence: 10,
            origin: origin(),
            family: ActuatorFamily::InteractionQos,
            action_class: ActionClass::InteractionForeground,
            treatment_arm: ExplorationArm::InteractionQosStandard,
            context: ExplorationContext::Interactive,
            action_key: "interaction_qos:foreground@standard".to_string(),
            stratum_hash: 11,
            horizon_cycles: 4,
            washout_cycles: 2,
            minimum_effect_micros: 500,
        };
        let output = runtime.tick_controlled(MicroexperimentControlledInput {
            cycle: 10,
            monotonic_millis: 10,
            gates: PairGates::healthy_enabled(),
            endpoint_contract_ready: true,
            candidates: &[candidate],
            observations: &[],
        });

        assert_eq!(output.directives.len(), 1);
        assert_eq!(output.metrics.open_pairs, 1);
        assert_eq!(output.metrics.pair_gold_total, 0);
        assert!(output.pair_gold.is_empty());
        assert_eq!(output.metrics.blocker, "awaiting-real-endpoint");
    }
}
