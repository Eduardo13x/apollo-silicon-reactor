//! Bounded temporal rollouts for the World Model.
//!
//! This is a receding-horizon ranker, not an actuator. It evaluates at most
//! `MAX_DISPATCHABLE + BEAM_WIDTH * (MAX_DISPATCHABLE + MAX_AMBIENT_FOLLOWUPS)`
//! trajectories and returns scores for actions that already exist in the
//! specialist proposal set.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::engine::installation_identity::InstallationId;
use crate::engine::telemetry_medallion::{
    ActionModelStats, HardwareRegime, TelemetryContextSummary, WorldStateDelta,
};

const TEMPORAL_MEMORY_CAPACITY: usize = 32;
const TEMPORAL_MIN_SAMPLES: usize = 4;
const MAX_DISPATCHABLE: usize = 8;
const MAX_AMBIENT_FOLLOWUPS: usize = 2;
const BEAM_WIDTH: usize = 4;
const STEP_CYCLES: f64 = 4.0;
const MIN_STATE_EVIDENCE: f64 = 4.0;
const MIN_MODEL_QUALITY: f64 = 0.85;
const MODEL_MAX_AGE_SECS: i64 = 14 * 24 * 60 * 60;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SequenceActionScore {
    pub expected_gain: f64,
    pub uncertainty: f64,
    pub exact_evidence: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TemporalSequencePlan {
    pub action_scores: HashMap<String, SequenceActionScore>,
    pub best_first: Option<String>,
    pub best_second: Option<String>,
    pub expected_gain: f64,
    pub uncertainty: f64,
    pub predicted_pressure_delta: f64,
    pub predicted_fluidity_delta: f64,
    pub predicted_energy_delta: f64,
    pub candidates: u64,
    pub sequences_evaluated: u64,
    pub memory_samples: u64,
    pub authoritative: bool,
    pub abstention_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, Default)]
struct JointState {
    pressure: f64,
    fluidity: f64,
    energy: f64,
    cpu: f64,
    thermal: f64,
    thrashing: f64,
    stall: f64,
}

impl JointState {
    fn from_context(context: &TelemetryContextSummary) -> Self {
        Self {
            pressure: context.memory_pressure.clamp(0.0, 1.0),
            fluidity: context.fluidity_score.clamp(0.0, 1.0),
            energy: (context.package_watts.unwrap_or(0.0) / 50.0).clamp(0.0, 1.0),
            cpu: context.cpu_max_busy.clamp(0.0, 1.0),
            thermal: context.thermal_score.clamp(0.0, 1.0),
            thrashing: (context.thrashing_score / 50_000.0).clamp(0.0, 1.0),
            stall: context.stall_fraction.clamp(0.0, 1.0),
        }
    }

    fn apply(self, delta: WorldStateDelta) -> Self {
        Self {
            pressure: (self.pressure + delta.pressure).clamp(0.0, 1.0),
            fluidity: (self.fluidity + delta.fluidity).clamp(0.0, 1.0),
            energy: (self.energy + delta.energy).clamp(0.0, 1.0),
            cpu: (self.cpu + delta.cpu).clamp(0.0, 1.0),
            thermal: (self.thermal + delta.thermal).clamp(0.0, 1.0),
            thrashing: (self.thrashing + delta.thrashing).clamp(0.0, 1.0),
            stall: (self.stall + delta.stall).clamp(0.0, 1.0),
        }
    }

    fn delta_from(self, baseline: Self) -> WorldStateDelta {
        WorldStateDelta {
            pressure: self.pressure - baseline.pressure,
            fluidity: self.fluidity - baseline.fluidity,
            energy: self.energy - baseline.energy,
            cpu: self.cpu - baseline.cpu,
            thermal: self.thermal - baseline.thermal,
            thrashing: self.thrashing - baseline.thrashing,
            stall: self.stall - baseline.stall,
        }
    }

    fn utility(self) -> f64 {
        (0.24 * (1.0 - self.pressure)
            + 0.25 * self.fluidity
            + 0.12 * (1.0 - self.energy)
            + 0.08 * (1.0 - self.cpu)
            + 0.11 * (1.0 - self.thermal)
            + 0.10 * (1.0 - self.thrashing)
            + 0.10 * (1.0 - self.stall))
            .clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Default)]
pub struct TemporalMemory {
    states: VecDeque<(u64, i64, JointState)>,
    dynamics_per_cycle: WorldStateDelta,
    variance_per_cycle: WorldStateDelta,
}

impl TemporalMemory {
    pub fn observe(&mut self, context: &TelemetryContextSummary) {
        let state = JointState::from_context(context);
        if let Some((last_cycle, last_timestamp, last_state)) = self.states.back().copied() {
            if context.cycle == last_cycle {
                return;
            }
            let discontinuity = context.cycle < last_cycle
                || context.timestamp_unix < last_timestamp
                || context.timestamp_unix.saturating_sub(last_timestamp) > 300
                || context.cycle.saturating_sub(last_cycle) > 600;
            if discontinuity {
                self.clear();
            } else {
                let cycle_gap = context.cycle.saturating_sub(last_cycle).max(1) as f64;
                let observed = state.delta_from(last_state).scaled(1.0 / cycle_gap);
                if self.states.len() == 1 {
                    self.dynamics_per_cycle = observed;
                } else {
                    let residual = observed.minus(self.dynamics_per_cycle);
                    self.dynamics_per_cycle = self.dynamics_per_cycle.ema(observed, 0.20);
                    self.variance_per_cycle = self
                        .variance_per_cycle
                        .variance_update(residual, 0.20)
                        .clamped(0.0, 1.0);
                }
            }
        }
        if self.states.len() >= TEMPORAL_MEMORY_CAPACITY {
            self.states.pop_front();
        }
        self.states
            .push_back((context.cycle, context.timestamp_unix, state));
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.dynamics_per_cycle = WorldStateDelta::default();
        self.variance_per_cycle = WorldStateDelta::default();
    }

    pub fn samples(&self) -> usize {
        self.states.len()
    }

    fn current(&self) -> Option<JointState> {
        self.states.back().map(|(_, _, state)| *state)
    }
}

#[derive(Clone)]
struct Candidate {
    key: String,
    effect: WorldStateDelta,
    utility: f64,
    uncertainty: f64,
    exact: bool,
    dispatchable: bool,
}

#[derive(Clone)]
struct Trajectory {
    first: usize,
    second: Option<usize>,
    score: f64,
    uncertainty: f64,
    final_state: JointState,
    exact: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn plan_temporal_sequence(
    memory: &TemporalMemory,
    context: Option<&TelemetryContextSummary>,
    installation_id: InstallationId,
    authority_trusted: bool,
    action_models: &HashMap<String, ActionModelStats>,
    action_keys: &[String],
    workload: &str,
) -> TemporalSequencePlan {
    let mut plan = TemporalSequencePlan {
        memory_samples: memory.samples() as u64,
        ..TemporalSequencePlan::default()
    };
    let Some(context) = context else {
        plan.abstention_reason = Some("no_current_gold");
        return plan;
    };
    let Some(current) = memory.current() else {
        plan.abstention_reason = Some("temporal_warmup");
        return plan;
    };
    if memory.samples() < TEMPORAL_MIN_SAMPLES {
        plan.abstention_reason = Some("temporal_warmup");
        return plan;
    }
    if action_keys.is_empty() {
        plan.abstention_reason = Some("idle_no_accelerator");
        return plan;
    }

    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    for key in action_keys.iter().take(MAX_DISPATCHABLE) {
        if seen.insert(key.clone()) {
            if let Some(candidate) =
                candidate_for_key(key, true, action_models, workload, context, installation_id)
            {
                candidates.push(candidate);
            }
        }
    }
    let ambient = ambient_followups(context);
    for key in ambient.iter().take(MAX_AMBIENT_FOLLOWUPS) {
        if seen.insert((*key).to_string()) {
            if let Some(candidate) = candidate_for_key(
                key,
                false,
                action_models,
                workload,
                context,
                installation_id,
            ) {
                candidates.push(candidate);
            }
        }
    }
    plan.candidates = candidates.len() as u64;
    if !candidates.iter().any(|candidate| candidate.dispatchable) {
        plan.abstention_reason = Some("transition_evidence");
        return plan;
    }

    let drift = memory.dynamics_per_cycle.scaled(STEP_CYCLES);
    let baseline_one = current.apply(drift);
    let baseline_two = baseline_one.apply(drift);
    let mut trajectories = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if !candidate.dispatchable {
            continue;
        }
        let state = baseline_one.apply(candidate.effect);
        let score = sequence_score(
            state.utility() - baseline_one.utility(),
            candidate.utility,
            action_cost(&candidate.key),
        );
        plan.sequences_evaluated = plan.sequences_evaluated.saturating_add(1);
        plan.action_scores.insert(
            candidate.key.clone(),
            SequenceActionScore {
                expected_gain: score,
                uncertainty: candidate.uncertainty,
                exact_evidence: candidate.exact,
            },
        );
        trajectories.push(Trajectory {
            first: index,
            second: None,
            score,
            uncertainty: candidate.uncertainty,
            final_state: state,
            exact: candidate.exact,
        });
    }
    trajectories.sort_by(|left, right| right.score.total_cmp(&left.score));
    trajectories.truncate(BEAM_WIDTH);

    let beams = trajectories.clone();
    for beam in beams {
        for (second_index, second) in candidates.iter().enumerate() {
            if second_index == beam.first {
                continue;
            }
            let first = &candidates[beam.first];
            let mut final_state = beam.final_state.apply(drift).apply(second.effect);
            let mut learned_utility = first.utility + 0.90 * second.utility;
            let mut exact = first.exact && second.exact;
            let mut uncertainty = (first.uncertainty + second.uncertainty) * 0.5;
            if let Some(synergy) = coordinated_effect(
                &first.key,
                &second.key,
                action_models,
                context,
                installation_id,
            ) {
                final_state = final_state.apply(synergy.effect);
                learned_utility += 0.50 * synergy.utility;
                uncertainty = (uncertainty + synergy.uncertainty) * 0.5;
                exact &= synergy.exact;
            }
            let state_gain = 0.45 * (beam.final_state.utility() - baseline_one.utility())
                + 0.55 * (final_state.utility() - baseline_two.utility());
            let score = sequence_score(
                state_gain,
                learned_utility,
                action_cost(&first.key) + 0.90 * action_cost(&second.key),
            );
            plan.sequences_evaluated = plan.sequences_evaluated.saturating_add(1);
            let entry = plan.action_scores.entry(first.key.clone()).or_default();
            if score > entry.expected_gain {
                *entry = SequenceActionScore {
                    expected_gain: score,
                    uncertainty,
                    exact_evidence: exact,
                };
            }
            trajectories.push(Trajectory {
                first: beam.first,
                second: Some(second_index),
                score,
                uncertainty,
                final_state,
                exact,
            });
        }
    }

    let Some(best) = trajectories
        .into_iter()
        .max_by(|left, right| left.score.total_cmp(&right.score))
    else {
        plan.abstention_reason = Some("transition_evidence");
        return plan;
    };
    let final_baseline = if best.second.is_some() {
        baseline_two
    } else {
        baseline_one
    };
    let predicted = best.final_state.delta_from(final_baseline);
    plan.best_first = Some(candidates[best.first].key.clone());
    plan.best_second = best.second.map(|index| candidates[index].key.clone());
    plan.expected_gain = best.score;
    plan.uncertainty = best.uncertainty.clamp(0.0, 1.0);
    plan.predicted_pressure_delta = predicted.pressure;
    plan.predicted_fluidity_delta = predicted.fluidity;
    plan.predicted_energy_delta = predicted.energy;
    plan.authoritative = authority_trusted && best.exact && best.score > 0.0;
    plan.abstention_reason = (!plan.authoritative).then_some("ranking_only");
    plan
}

fn sequence_score(state_gain: f64, learned_utility: f64, cost: f64) -> f64 {
    (0.65 * state_gain + 0.35 * learned_utility - cost).clamp(-1.0, 1.0)
}

fn candidate_for_key(
    key: &str,
    dispatchable: bool,
    models: &HashMap<String, ActionModelStats>,
    workload: &str,
    context: &TelemetryContextSummary,
    installation_id: InstallationId,
) -> Option<Candidate> {
    let workload_key = format!("{workload}|{key}");
    let family_key = key.split_once(':').map(|(family, _)| format!("{family}:*"));
    let selected = [
        (models.get(&workload_key), true),
        (models.get(key), true),
        (family_key.as_ref().and_then(|key| models.get(key)), false),
    ]
    .into_iter()
    .find_map(|(model, exact)| {
        model
            .filter(|model| transition_model_ready(model, context, installation_id))
            .map(|model| (model, exact))
    })?;
    let (model, exact) = selected;
    let evidence = model
        .effective_state_evidence_at(context.timestamp_unix)
        .max(1.0);
    let state_uncertainty = (model.state_variance_ema.mean_variance().max(0.0) / evidence).sqrt();
    let utility_uncertainty = (model.utility_variance_ema.max(0.0001) / evidence).sqrt();
    let prior_scale = if exact { 1.0 } else { 0.50 };
    Some(Candidate {
        key: key.to_string(),
        effect: model.state_delta_ema.scaled(prior_scale),
        utility: model.utility_ema * prior_scale,
        uncertainty: (state_uncertainty + utility_uncertainty + if exact { 0.0 } else { 0.25 })
            .clamp(0.0, 1.0),
        exact,
        dispatchable,
    })
}

fn coordinated_effect(
    first: &str,
    second: &str,
    models: &HashMap<String, ActionModelStats>,
    context: &TelemetryContextSummary,
    installation_id: InstallationId,
) -> Option<Candidate> {
    let mut families = [first.split_once(':')?.0, second.split_once(':')?.0];
    families.sort_unstable();
    let key = format!("coordinated:{}+{}", families[0], families[1]);
    candidate_for_key(
        &key,
        false,
        models,
        &context.workload,
        context,
        installation_id,
    )
}

fn transition_model_ready(
    model: &ActionModelStats,
    context: &TelemetryContextSummary,
    installation_id: InstallationId,
) -> bool {
    model.quality_ema >= MIN_MODEL_QUALITY
        && model.last_observed_unix > 0
        && context.timestamp_unix >= model.last_observed_unix
        && context.timestamp_unix - model.last_observed_unix <= MODEL_MAX_AGE_SECS
        && model.effective_state_evidence_at(context.timestamp_unix) >= MIN_STATE_EVIDENCE
        && model.state_delta_ema.is_finite()
        && model.state_variance_ema.is_finite()
        && installation_id.is_known()
        && model.installation_id == installation_id
        && HardwareRegime::from_context(context).is_known()
        && model.hardware_regime.matches_context(context)
}

fn ambient_followups(context: &TelemetryContextSummary) -> Vec<&'static str> {
    let mut followups = Vec::with_capacity(MAX_AMBIENT_FOLLOWUPS);
    if !context.markov_prewarm_active
        && context.markov_prediction_confidence >= 0.35
        && (0.0..=20.0).contains(&context.markov_prediction_eta_secs)
    {
        followups.push("markov_prewarm:predicted_app");
    }
    if context.app_launching || context.window_op_active || context.markov_prewarm_active {
        followups.push("interaction_qos:foreground");
    }
    followups
}

fn action_cost(key: &str) -> f64 {
    match key.split_once(':').map(|(family, _)| family) {
        Some("unfreeze") => 0.001,
        Some("boost" | "thread_qos" | "interaction_qos") => 0.003,
        Some("markov_prewarm") => 0.004,
        Some("throttle" | "memorystatus") => 0.006,
        Some("sysctl" | "predictive_threshold") => 0.008,
        Some("freeze" | "quarantine" | "spotlight") => 0.015,
        _ => 0.007,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCAL_ID: InstallationId = InstallationId(7);

    fn context(cycle: u64) -> TelemetryContextSummary {
        TelemetryContextSummary {
            cycle,
            timestamp_unix: 1_800_000_000 + cycle as i64,
            workload: "coding".to_string(),
            memory_pressure: 0.40,
            fluidity_score: 0.70,
            package_watts: Some(4.0),
            cpu_max_busy: 0.50,
            p_core_count: 4,
            e_core_count: 6,
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            markov_prediction_confidence: 0.60,
            markov_prediction_eta_secs: 4.0,
            ..TelemetryContextSummary::default()
        }
    }

    fn model(context: &TelemetryContextSummary, utility: f64, fluidity: f64) -> ActionModelStats {
        ActionModelStats {
            observations: 12,
            utility_ema: utility,
            evidence_mass: 12.0,
            utility_variance_ema: 0.0001,
            state_delta_ema: WorldStateDelta {
                fluidity,
                ..WorldStateDelta::default()
            },
            state_variance_ema: WorldStateDelta::default(),
            state_evidence_mass: 12.0,
            quality_ema: 1.0,
            last_cycle: context.cycle,
            last_observed_unix: context.timestamp_unix,
            hardware_regime: HardwareRegime::from_context(context),
            installation_id: LOCAL_ID,
            ..ActionModelStats::default()
        }
    }

    fn warmed_memory() -> (TemporalMemory, TelemetryContextSummary) {
        let mut memory = TemporalMemory::default();
        let mut latest = context(1);
        for cycle in 1..=6 {
            latest = context(cycle);
            latest.fluidity_score += cycle as f64 * 0.001;
            memory.observe(&latest);
        }
        (memory, latest)
    }

    #[test]
    fn temporal_memory_deduplicates_and_resets_discontinuities() {
        let mut memory = TemporalMemory::default();
        memory.observe(&context(10));
        memory.observe(&context(10));
        assert_eq!(memory.samples(), 1);
        memory.observe(&context(11));
        assert_eq!(memory.samples(), 2);
        memory.observe(&context(1));
        assert_eq!(memory.samples(), 1);
    }

    #[test]
    fn rollout_prefers_action_with_useful_followup() {
        let (memory, latest) = warmed_memory();
        let models = HashMap::from([
            ("boost:Editor".to_string(), model(&latest, 0.01, 0.01)),
            ("boost:Browser".to_string(), model(&latest, 0.02, 0.01)),
            (
                "markov_prewarm:predicted_app".to_string(),
                model(&latest, 0.04, 0.02),
            ),
            (
                "coordinated:boost+markov_prewarm".to_string(),
                model(&latest, 0.08, 0.05),
            ),
        ]);
        let plan = plan_temporal_sequence(
            &memory,
            Some(&latest),
            LOCAL_ID,
            true,
            &models,
            &["boost:Editor".to_string(), "boost:Browser".to_string()],
            "coding",
        );
        assert_eq!(plan.best_first.as_deref(), Some("boost:Browser"));
        assert_eq!(
            plan.best_second.as_deref(),
            Some("markov_prewarm:predicted_app")
        );
        assert!(plan.authoritative);
        assert!(plan.expected_gain > 0.0);
    }

    #[test]
    fn rollout_work_is_beam_bounded() {
        let (memory, latest) = warmed_memory();
        let mut models = HashMap::new();
        let keys: Vec<String> = (0..64).map(|index| format!("boost:p{index}")).collect();
        for key in &keys {
            models.insert(key.clone(), model(&latest, 0.01, 0.01));
        }
        let plan = plan_temporal_sequence(
            &memory,
            Some(&latest),
            LOCAL_ID,
            true,
            &models,
            &keys,
            "coding",
        );
        assert!(plan.candidates <= (MAX_DISPATCHABLE + MAX_AMBIENT_FOLLOWUPS) as u64);
        assert!(
            plan.sequences_evaluated
                <= (MAX_DISPATCHABLE + BEAM_WIDTH * (MAX_DISPATCHABLE + MAX_AMBIENT_FOLLOWUPS))
                    as u64
        );
    }

    #[test]
    fn legacy_scalar_model_cannot_drive_rollout() {
        let (memory, latest) = warmed_memory();
        let models = HashMap::from([(
            "boost:Editor".to_string(),
            ActionModelStats {
                utility_ema: 0.5,
                evidence_mass: 64.0,
                quality_ema: 1.0,
                last_observed_unix: latest.timestamp_unix,
                hardware_regime: HardwareRegime::from_context(&latest),
                installation_id: LOCAL_ID,
                ..ActionModelStats::default()
            },
        )]);
        let plan = plan_temporal_sequence(
            &memory,
            Some(&latest),
            LOCAL_ID,
            true,
            &models,
            &["boost:Editor".to_string()],
            "coding",
        );
        assert_eq!(plan.abstention_reason, Some("transition_evidence"));
        assert!(plan.action_scores.is_empty());
    }

    #[test]
    fn empty_specialist_set_is_reported_as_idle_not_missing_evidence() {
        let (memory, latest) = warmed_memory();
        let plan = plan_temporal_sequence(
            &memory,
            Some(&latest),
            LOCAL_ID,
            true,
            &HashMap::new(),
            &[],
            "coding",
        );
        assert_eq!(plan.abstention_reason, Some("idle_no_accelerator"));
        assert_eq!(plan.candidates, 0);
        assert_eq!(plan.sequences_evaluated, 0);
    }
}
