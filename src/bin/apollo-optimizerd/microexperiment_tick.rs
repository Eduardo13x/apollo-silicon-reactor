use std::time::Instant;

use apollo_engine::engine::exploration_scheduler::{
    ActionClass, ExplorationArm, ExplorationContext, ExplorationOrigin,
};
use apollo_engine::engine::microexperiment_lab::{
    CandidateDisposition, LabError, MicroexperimentLab, MicroexperimentLabPersisted, PairCandidate,
    PairClosure, PairDirective, PairGates, PairGoldRecord, PairInvalidation, PairProgress,
    RestoreDisposition, TimedEndpointDisposition, TimedPairEndpoint,
};
use apollo_engine::engine::telemetry_medallion::ActuatorFamily;

#[derive(Debug, Clone, Copy)]
pub struct MicroexperimentTickInput<'a> {
    pub cycle: u64,
    pub interaction_activations: u64,
    pub markov_applied: u64,
    pub workload: &'a str,
    pub inherited_safe: bool,
}

#[derive(Debug, Clone, Default)]
pub struct MicroexperimentTickMetrics {
    pub phase: String,
    pub blocker: String,
    pub restore: String,
    pub proposed_total: u64,
    pub eligible_total: u64,
    pub randomized_total: u64,
    pub shadow_would_open_total: u64,
    pub open_pairs: u64,
    pub completed_pairs: u64,
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
    counters: Option<CounterBaseline>,
    restore: &'static str,
    started_at: Instant,
}

impl MicroexperimentRuntime {
    pub fn new(origin: ExplorationOrigin, persisted: Option<MicroexperimentLabPersisted>) -> Self {
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
            counters: None,
            restore,
            started_at: Instant::now(),
        }
    }

    pub fn tick(&mut self, input: MicroexperimentTickInput<'_>) -> MicroexperimentTickMetrics {
        let current = CounterBaseline {
            interaction_activations: input.interaction_activations,
            markov_applied: input.markov_applied,
        };
        let Some(previous) = self.counters.replace(current) else {
            return self.metrics("no-candidates");
        };
        let interaction_delta = input
            .interaction_activations
            .saturating_sub(previous.interaction_activations);
        let markov_delta = input.markov_applied.saturating_sub(previous.markov_applied);
        let mut candidates = Vec::with_capacity(2);
        if interaction_delta > 0 {
            candidates.push(PairCandidate {
                sequence: input.cycle.saturating_mul(4).saturating_add(1).max(1),
                origin: self.origin,
                family: ActuatorFamily::InteractionQos,
                action_class: ActionClass::InteractionForeground,
                treatment_arm: ExplorationArm::InteractionQosStandard,
                context: ExplorationContext::Interactive,
                action_key: "interaction_qos:foreground@standard".to_string(),
                stratum_hash: stratum_hash(input.workload, 1),
                horizon_cycles: 12,
                washout_cycles: 3,
                minimum_effect_micros: 500,
            });
        }
        if markov_delta > 0 {
            candidates.push(PairCandidate {
                sequence: input.cycle.saturating_mul(4).saturating_add(2).max(1),
                origin: self.origin,
                family: ActuatorFamily::MarkovPrewarm,
                action_class: ActionClass::MarkovPredictedApp,
                treatment_arm: ExplorationArm::MarkovCacheOnly,
                context: ExplorationContext::Background,
                action_key: "markov:cache-only".to_string(),
                stratum_hash: stratum_hash(input.workload, 2),
                horizon_cycles: 120,
                washout_cycles: 3,
                minimum_effect_micros: 500,
            });
        }
        let mut output = self.tick_controlled(MicroexperimentControlledInput {
            cycle: input.cycle,
            monotonic_millis: self.started_at.elapsed().as_millis() as u64,
            gates: PairGates {
                inherited_safe: input.inherited_safe,
                ..PairGates::default()
            },
            endpoint_contract_ready: false,
            candidates: &candidates,
            observations: &[],
        });
        if !input.inherited_safe && !candidates.is_empty() {
            output.metrics.blocker = "safety-gate".to_string();
        }
        output.metrics
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

        let directives = if input.endpoint_contract_ready && effective_gates.allows_pair() {
            self.lab.issue_ready_arms(input.cycle, effective_gates)
        } else {
            Vec::new()
        };
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
            proposed_total: metrics.proposed_total,
            eligible_total: metrics.eligible_total,
            randomized_total: metrics.randomized_total,
            shadow_would_open_total: metrics.shadow_would_open_total,
            open_pairs: metrics.open_pairs as u64,
            completed_pairs: metrics.completed_pairs as u64,
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
            rollback_failed_total: metrics.rollback_failed_total,
            mean_effect: metrics.mean_effect,
        }
    }
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

#[cfg(test)]
mod tests {
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
        }
    }

    #[test]
    fn real_counter_delta_creates_shadow_pair_diagnostic_once() {
        let mut runtime = MicroexperimentRuntime::new(origin(), None);
        let baseline = runtime.tick(input(1));
        assert_eq!(baseline.proposed_total, 0);

        let first = runtime.tick(MicroexperimentTickInput {
            cycle: 2,
            interaction_activations: 1,
            markov_applied: 1,
            ..input(2)
        });
        assert_eq!(first.proposed_total, 2);
        assert_eq!(first.shadow_would_open_total, 2);
        assert_eq!(first.open_pairs, 0);
        assert_eq!(first.blocker, "endpoint-wiring-required");

        let unchanged = runtime.tick(MicroexperimentTickInput {
            cycle: 3,
            interaction_activations: 1,
            markov_applied: 1,
            ..input(3)
        });
        assert_eq!(unchanged.proposed_total, 2);
        assert_eq!(unchanged.shadow_would_open_total, 2);
    }

    #[test]
    fn unsafe_cycle_records_blocker_without_opening_pair() {
        let mut runtime = MicroexperimentRuntime::new(origin(), None);
        runtime.tick(input(1));
        let metrics = runtime.tick(MicroexperimentTickInput {
            cycle: 2,
            interaction_activations: 1,
            inherited_safe: false,
            ..input(2)
        });
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
        let metrics = restored.tick(MicroexperimentTickInput {
            cycle: 3,
            interaction_activations: 1,
            ..input(3)
        });
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
