use apollo_engine::engine::exploration_scheduler::{
    ActionClass, ExplorationArm, ExplorationContext, ExplorationOrigin,
};
use apollo_engine::engine::microexperiment_lab::{
    LabError, MicroexperimentLab, MicroexperimentLabPersisted, PairCandidate, PairGates,
    RestoreDisposition,
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
    pub mean_effect: f64,
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
        let gates = PairGates {
            inherited_safe: input.inherited_safe,
            ..PairGates::default()
        };
        let mut attempted = 0_u8;
        let mut admitted = 0_u8;
        if interaction_delta > 0 {
            attempted += 1;
            let candidate = PairCandidate {
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
            };
            admitted += u8::from(self.lab.evaluate_shadow(candidate, gates).is_ok());
        }
        if markov_delta > 0 {
            attempted += 1;
            let candidate = PairCandidate {
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
            };
            admitted += u8::from(self.lab.evaluate_shadow(candidate, gates).is_ok());
        }
        let blocker = match (attempted, admitted) {
            (0, _) => "no-candidates",
            (_, 0) if !input.inherited_safe => "safety-gate",
            (_, 0) => blocker_for_last_error(LabError::Gate),
            _ => "shadow-observed",
        };
        self.metrics(blocker)
    }

    pub fn persisted(&self) -> MicroexperimentLabPersisted {
        self.lab.persisted()
    }

    fn metrics(&self, blocker: &str) -> MicroexperimentTickMetrics {
        let metrics = self.lab.metrics();
        MicroexperimentTickMetrics {
            phase: "shadow".to_string(),
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
            mean_effect: 0.0,
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

fn blocker_for_last_error(_error: LabError) -> &'static str {
    "candidate-rejected"
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
}
