//! Bounded temporal rollouts for the World Model.
//!
//! This is a receding-horizon ranker, not an actuator. It evaluates at most
//! `MAX_DISPATCHABLE + BEAM_WIDTH * (MAX_DISPATCHABLE + MAX_AMBIENT_FOLLOWUPS)`
//! trajectories and returns scores for actions that already exist in the
//! specialist proposal set.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::engine::causal_dynamics::{CausalDynamicsModel, DynamicsState};
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
const MIN_AUTHORITATIVE_STATE_EVIDENCE: f64 = 8.0;
const MIN_AUTHORITATIVE_MODEL_QUALITY: f64 = 0.90;
const MAX_AUTHORITATIVE_UNCERTAINTY: f64 = 0.20;
const MODEL_MAX_AGE_SECS: i64 = 14 * 24 * 60 * 60;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SequenceActionScore {
    pub expected_gain: f64,
    pub uncertainty: f64,
    pub exact_evidence: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TemporalSequencePlan {
    /// Best-effort scores, including family priors and immature exact models.
    pub action_scores: HashMap<String, SequenceActionScore>,
    /// Scores backed only by mature, local, exact transition evidence.
    pub authoritative_action_scores: HashMap<String, SequenceActionScore>,
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
    pub authoritative_sequences_evaluated: u64,
    pub authoritative: bool,
    pub authoritative_best_first: Option<String>,
    pub authoritative_best_second: Option<String>,
    pub authoritative_expected_gain: f64,
    pub authoritative_uncertainty: f64,
    pub authoritative_pressure_delta: f64,
    pub authoritative_fluidity_delta: f64,
    pub authoritative_energy_delta: f64,
    pub dynamics_predictions: u64,
    pub dynamics_ranking_predictions: u64,
    pub dynamics_authoritative_predictions: u64,
    pub dynamics_baseline_used: bool,
    pub dynamics_mean_uncertainty: f64,
    pub abstention_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Default)]
pub struct TemporalMemory {
    states: VecDeque<(u64, i64, DynamicsState)>,
    dynamics_per_cycle: WorldStateDelta,
    variance_per_cycle: WorldStateDelta,
}

impl TemporalMemory {
    pub fn observe(&mut self, context: &TelemetryContextSummary) {
        let state = DynamicsState::from_context(context);
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

    fn current(&self) -> Option<DynamicsState> {
        self.states.back().map(|(_, _, state)| *state)
    }
}

#[derive(Clone)]
struct Candidate {
    key: String,
    legacy_effect: WorldStateDelta,
    effect: WorldStateDelta,
    utility: f64,
    legacy_uncertainty: f64,
    uncertainty: f64,
    exact: bool,
    authoritative: bool,
    legacy_authoritative: bool,
    dynamics_weight: f64,
    dynamics: bool,
    dynamics_ranking_eligible: bool,
    dynamics_authoritative: bool,
    dispatchable: bool,
}

#[derive(Clone)]
struct Trajectory {
    first: usize,
    second: Option<usize>,
    score: f64,
    uncertainty: f64,
    final_state: DynamicsState,
    authoritative: bool,
}

struct CandidateEvidence<'a> {
    models: &'a HashMap<String, ActionModelStats>,
    dynamics: Option<&'a CausalDynamicsModel>,
    workload: &'a str,
    context: &'a TelemetryContextSummary,
    installation_id: InstallationId,
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
    plan_temporal_sequence_with_dynamics(
        memory,
        context,
        installation_id,
        authority_trusted,
        action_models,
        None,
        action_keys,
        workload,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn plan_temporal_sequence_with_dynamics(
    memory: &TemporalMemory,
    context: Option<&TelemetryContextSummary>,
    installation_id: InstallationId,
    authority_trusted: bool,
    action_models: &HashMap<String, ActionModelStats>,
    dynamics: Option<&CausalDynamicsModel>,
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

    let evidence = CandidateEvidence {
        models: action_models,
        dynamics,
        workload,
        context,
        installation_id,
    };

    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    for key in action_keys.iter().take(MAX_DISPATCHABLE) {
        if seen.insert(key.clone()) {
            if let Some(candidate) = candidate_for_key(key, true, &evidence, current) {
                candidates.push(candidate);
            }
        }
    }
    let ambient = ambient_followups(context);
    for key in ambient.iter().take(MAX_AMBIENT_FOLLOWUPS) {
        if seen.insert((*key).to_string()) {
            if let Some(candidate) = candidate_for_key(key, false, &evidence, current) {
                candidates.push(candidate);
            }
        }
    }
    plan.candidates = candidates.len() as u64;
    plan.dynamics_predictions = candidates
        .iter()
        .filter(|candidate| candidate.dynamics)
        .count() as u64;
    plan.dynamics_authoritative_predictions = candidates
        .iter()
        .filter(|candidate| candidate.dynamics_authoritative)
        .count() as u64;
    plan.dynamics_ranking_predictions = candidates
        .iter()
        .filter(|candidate| candidate.dynamics_ranking_eligible)
        .count() as u64;
    if plan.dynamics_predictions > 0 {
        plan.dynamics_mean_uncertainty = candidates
            .iter()
            .filter(|candidate| candidate.dynamics)
            .map(|candidate| candidate.uncertainty)
            .sum::<f64>()
            / plan.dynamics_predictions as f64;
    }
    if !candidates.iter().any(|candidate| candidate.dispatchable) {
        plan.abstention_reason = Some("transition_evidence");
        return plan;
    }

    let fallback_drift = memory.dynamics_per_cycle.scaled(STEP_CYCLES);
    let (drift_one, learned_baseline_one) = baseline_drift(
        dynamics,
        workload,
        context,
        current,
        installation_id,
        fallback_drift,
    );
    let baseline_one = current.apply(drift_one);
    let (drift_two, learned_baseline_two) = baseline_drift(
        dynamics,
        workload,
        context,
        baseline_one,
        installation_id,
        fallback_drift,
    );
    let baseline_two = baseline_one.apply(drift_two);
    plan.dynamics_baseline_used = learned_baseline_one || learned_baseline_two;
    let mut one_step_trajectories = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if !candidate.dispatchable {
            continue;
        }
        let state = baseline_one.apply(candidate.effect);
        let score = sequence_score(
            state.utility() - baseline_one.utility(),
            candidate.utility,
            action_cost(&candidate.key),
            if candidate.dynamics_weight > 0.0 {
                0.05 * candidate.uncertainty
            } else {
                0.0
            },
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
        if candidate.authoritative {
            plan.authoritative_action_scores.insert(
                candidate.key.clone(),
                SequenceActionScore {
                    expected_gain: score,
                    uncertainty: candidate.uncertainty,
                    exact_evidence: true,
                },
            );
            plan.authoritative_sequences_evaluated =
                plan.authoritative_sequences_evaluated.saturating_add(1);
        }
        one_step_trajectories.push(Trajectory {
            first: index,
            second: None,
            score,
            uncertainty: candidate.uncertainty,
            final_state: state,
            authoritative: candidate.authoritative,
        });
    }
    one_step_trajectories.sort_by(|left, right| right.score.total_cmp(&left.score));
    let beams: Vec<_> = one_step_trajectories
        .iter()
        .take(BEAM_WIDTH)
        .cloned()
        .collect();
    // Keep every one-step candidate eligible for the authoritative lane even
    // when an exploratory prior occupies the bounded expansion beam.
    let mut trajectories = one_step_trajectories;
    for beam in beams {
        for (second_index, second) in candidates.iter().enumerate() {
            if second_index == beam.first {
                continue;
            }
            let first = &candidates[beam.first];
            let (second_effect, second_uncertainty, second_authoritative) = effect_at_state(
                second,
                dynamics,
                workload,
                context,
                beam.final_state,
                installation_id,
            );
            let (second_drift, learned_second_baseline) = baseline_drift(
                dynamics,
                workload,
                context,
                beam.final_state,
                installation_id,
                fallback_drift,
            );
            plan.dynamics_baseline_used |= learned_second_baseline;
            let mut final_state = beam.final_state.apply(second_drift).apply(second_effect);
            let mut learned_utility = first.utility + 0.90 * second.utility;
            let mut exact = first.exact && second.exact;
            let endpoints_authoritative = first.authoritative && second_authoritative;
            // A two-action sequence needs a mature coordinated model. Exact
            // endpoint models alone do not prove the interaction is safe.
            let mut authoritative = false;
            let mut uncertainty = (first.uncertainty + second_uncertainty) * 0.5;
            if let Some(synergy) =
                coordinated_effect(&first.key, &second.key, &evidence, beam.final_state)
            {
                final_state = final_state.apply(synergy.effect);
                learned_utility += 0.50 * synergy.utility;
                uncertainty = (uncertainty + synergy.uncertainty) * 0.5;
                exact &= synergy.exact;
                authoritative = endpoints_authoritative && synergy.authoritative;
            }
            let state_gain = 0.45 * (beam.final_state.utility() - baseline_one.utility())
                + 0.55 * (final_state.utility() - baseline_two.utility());
            let score = sequence_score(
                state_gain,
                learned_utility,
                action_cost(&first.key) + 0.90 * action_cost(&second.key),
                if first.dynamics_weight > 0.0 || second.dynamics_weight > 0.0 {
                    0.05 * uncertainty
                } else {
                    0.0
                },
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
            if authoritative {
                plan.authoritative_sequences_evaluated =
                    plan.authoritative_sequences_evaluated.saturating_add(1);
                let entry = plan
                    .authoritative_action_scores
                    .entry(first.key.clone())
                    .or_default();
                if score > entry.expected_gain {
                    *entry = SequenceActionScore {
                        expected_gain: score,
                        uncertainty,
                        exact_evidence: true,
                    };
                }
            }
            trajectories.push(Trajectory {
                first: beam.first,
                second: Some(second_index),
                score,
                uncertainty,
                final_state,
                authoritative,
            });
        }
    }

    let Some(best) = trajectories
        .iter()
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

    if let Some(best_authoritative) = trajectories
        .iter()
        .filter(|trajectory| trajectory.authoritative)
        .max_by(|left, right| left.score.total_cmp(&right.score))
    {
        let final_baseline = if best_authoritative.second.is_some() {
            baseline_two
        } else {
            baseline_one
        };
        let predicted = best_authoritative.final_state.delta_from(final_baseline);
        plan.authoritative_best_first = Some(candidates[best_authoritative.first].key.clone());
        plan.authoritative_best_second = best_authoritative
            .second
            .map(|index| candidates[index].key.clone());
        plan.authoritative_expected_gain = best_authoritative.score;
        plan.authoritative_uncertainty = best_authoritative.uncertainty.clamp(0.0, 1.0);
        plan.authoritative_pressure_delta = predicted.pressure;
        plan.authoritative_fluidity_delta = predicted.fluidity;
        plan.authoritative_energy_delta = predicted.energy;
        plan.authoritative = authority_trusted && best_authoritative.score > 0.0;
        plan.abstention_reason = if !authority_trusted {
            Some("authority_phase")
        } else if best_authoritative.score <= 0.0 {
            Some("nonpositive_authority")
        } else {
            None
        };
    } else {
        plan.abstention_reason = Some("ranking_only");
    }
    plan
}

fn sequence_score(state_gain: f64, learned_utility: f64, cost: f64, risk_penalty: f64) -> f64 {
    (0.65 * state_gain + 0.35 * learned_utility - cost - risk_penalty.clamp(0.0, 0.05))
        .clamp(-1.0, 1.0)
}

fn candidate_for_key(
    key: &str,
    dispatchable: bool,
    evidence: &CandidateEvidence<'_>,
    state: DynamicsState,
) -> Option<Candidate> {
    let workload_key = format!("{}|{key}", evidence.workload);
    let family_key = key.split_once(':').map(|(family, _)| format!("{family}:*"));
    let mut options = Vec::with_capacity(3);
    for (model, exact, contextual) in [
        (evidence.models.get(&workload_key), true, true),
        (evidence.models.get(key), true, false),
        (
            family_key.as_ref().and_then(|key| evidence.models.get(key)),
            false,
            false,
        ),
    ] {
        let Some(model) = model.filter(|model| {
            transition_model_ready(model, evidence.context, evidence.installation_id)
        }) else {
            continue;
        };
        let evidence = model
            .effective_state_evidence_at(evidence.context.timestamp_unix)
            .max(1.0);
        let uncertainty = model_uncertainty(model, evidence, exact);
        let authoritative = exact
            && evidence >= MIN_AUTHORITATIVE_STATE_EVIDENCE
            && model.quality_ema >= MIN_AUTHORITATIVE_MODEL_QUALITY
            && uncertainty <= MAX_AUTHORITATIVE_UNCERTAINTY;
        options.push((
            model,
            exact,
            contextual,
            evidence,
            uncertainty,
            authoritative,
        ));
    }
    let (model, exact, _, _, uncertainty, authoritative) =
        options.into_iter().max_by(|left, right| {
            left.5
                .cmp(&right.5)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| right.4.total_cmp(&left.4))
                .then_with(|| left.3.total_cmp(&right.3))
        })?;
    let prior_scale = if exact { 1.0 } else { 0.50 };
    let legacy_effect = model.state_delta_ema.scaled(prior_scale);
    let legacy_uncertainty = uncertainty;
    let legacy_authoritative = authoritative;
    let forecast = evidence.dynamics.and_then(|dynamics| {
        dynamics.predict_action_from_state(
            key,
            evidence.workload,
            evidence.context,
            state,
            evidence.installation_id,
        )
    });
    let dynamics_weight = forecast.map_or(0.0, |forecast| {
        if forecast.authoritative {
            0.70
        } else if forecast.ranking_eligible {
            (forecast.effective_evidence / (forecast.effective_evidence + 24.0) * 0.35)
                .clamp(0.05, 0.35)
        } else {
            0.0
        }
    });
    let effect = forecast.map_or(legacy_effect, |forecast| {
        legacy_effect
            .scaled(1.0 - dynamics_weight)
            .plus(forecast.mean_delta.scaled(dynamics_weight))
    });
    let uncertainty = forecast.map_or(uncertainty, |forecast| {
        ((1.0 - dynamics_weight) * uncertainty + dynamics_weight * forecast.uncertainty)
            .clamp(0.0, 1.0)
    });
    let authoritative = forecast.map_or(authoritative, |forecast| {
        if forecast.ranking_eligible {
            authoritative && forecast.authoritative
        } else {
            authoritative
        }
    });
    Some(Candidate {
        key: key.to_string(),
        legacy_effect,
        effect,
        utility: model.utility_ema * prior_scale,
        legacy_uncertainty,
        uncertainty,
        exact,
        authoritative,
        legacy_authoritative,
        dynamics_weight,
        dynamics: forecast.is_some(),
        dynamics_ranking_eligible: forecast.is_some_and(|forecast| forecast.ranking_eligible),
        dynamics_authoritative: forecast.is_some_and(|forecast| forecast.authoritative),
        dispatchable,
    })
}

fn model_uncertainty(model: &ActionModelStats, evidence: f64, exact: bool) -> f64 {
    let state_uncertainty = (model.state_variance_ema.mean_variance().max(0.0) / evidence).sqrt();
    let utility_uncertainty = (model.utility_variance_ema.max(0.0001) / evidence).sqrt();
    (state_uncertainty + utility_uncertainty + if exact { 0.0 } else { 0.25 }).clamp(0.0, 1.0)
}

fn coordinated_effect(
    first: &str,
    second: &str,
    evidence: &CandidateEvidence<'_>,
    state: DynamicsState,
) -> Option<Candidate> {
    let mut families = [first.split_once(':')?.0, second.split_once(':')?.0];
    families.sort_unstable();
    let key = format!("coordinated:{}+{}", families[0], families[1]);
    candidate_for_key(&key, false, evidence, state)
}

fn effect_at_state(
    candidate: &Candidate,
    dynamics: Option<&CausalDynamicsModel>,
    workload: &str,
    context: &TelemetryContextSummary,
    state: DynamicsState,
    installation_id: InstallationId,
) -> (WorldStateDelta, f64, bool) {
    let Some(forecast) = dynamics.and_then(|dynamics| {
        dynamics.predict_action_from_state(
            &candidate.key,
            workload,
            context,
            state,
            installation_id,
        )
    }) else {
        return (
            candidate.effect,
            candidate.uncertainty,
            candidate.authoritative,
        );
    };
    let effect = candidate
        .legacy_effect
        .scaled(1.0 - candidate.dynamics_weight)
        .plus(forecast.mean_delta.scaled(candidate.dynamics_weight));
    let uncertainty = ((1.0 - candidate.dynamics_weight) * candidate.legacy_uncertainty
        + candidate.dynamics_weight * forecast.uncertainty)
        .clamp(0.0, 1.0);
    (
        effect,
        uncertainty,
        candidate.legacy_authoritative && forecast.authoritative,
    )
}

fn baseline_drift(
    dynamics: Option<&CausalDynamicsModel>,
    workload: &str,
    context: &TelemetryContextSummary,
    state: DynamicsState,
    installation_id: InstallationId,
    fallback: WorldStateDelta,
) -> (WorldStateDelta, bool) {
    let Some(forecast) = dynamics.and_then(|dynamics| {
        dynamics.predict_baseline_from_state(workload, context, state, installation_id)
    }) else {
        return (fallback, false);
    };
    let learned = forecast.mean_delta.scaled(STEP_CYCLES);
    if !forecast.ranking_eligible {
        return (fallback, false);
    }
    let maturity = forecast.effective_evidence / (forecast.effective_evidence + 24.0);
    let confidence = (1.0 - forecast.uncertainty).clamp(0.0, 1.0) * maturity * 0.70;
    (
        fallback
            .scaled(1.0 - confidence)
            .plus(learned.scaled(confidence))
            .clamped(-0.20, 0.20),
        true,
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
        Some("io_shaping") => 0.002,
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
    fn causal_dynamics_re_ranks_only_existing_specialist_actions() {
        let mut dynamics = CausalDynamicsModel::new(LOCAL_ID);
        for sample in 1..=40 {
            let before = context(sample);
            let hardware = HardwareRegime::from_context(&before);
            dynamics.observe_action(
                "boost:Editor",
                "boost",
                "coding",
                &before,
                WorldStateDelta {
                    fluidity: 0.06,
                    pressure: -0.01,
                    ..WorldStateDelta::default()
                },
                4,
                0.98,
                before.timestamp_unix + 4,
                hardware,
                LOCAL_ID,
                sample,
            );
            dynamics.observe_action(
                "boost:Browser",
                "boost",
                "coding",
                &before,
                WorldStateDelta {
                    fluidity: -0.04,
                    pressure: 0.01,
                    ..WorldStateDelta::default()
                },
                4,
                0.98,
                before.timestamp_unix + 4,
                hardware,
                LOCAL_ID,
                1_000 + sample,
            );
        }
        let mut memory = TemporalMemory::default();
        let mut latest = context(100);
        for cycle in 95..=100 {
            latest = context(cycle);
            memory.observe(&latest);
        }
        let models = HashMap::from([
            ("boost:Editor".to_string(), model(&latest, 0.02, 0.01)),
            ("boost:Browser".to_string(), model(&latest, 0.02, 0.01)),
        ]);
        let proposed = ["boost:Editor".to_string(), "boost:Browser".to_string()];
        let plan = plan_temporal_sequence_with_dynamics(
            &memory,
            Some(&latest),
            LOCAL_ID,
            true,
            &models,
            Some(&dynamics),
            &proposed,
            "coding",
        );

        assert_eq!(plan.dynamics_predictions, 2);
        assert_eq!(plan.dynamics_ranking_predictions, 2);
        assert_eq!(plan.best_first.as_deref(), Some("boost:Editor"));
        assert!(plan.action_scores.keys().all(|key| proposed.contains(key)));
        assert!(plan.dynamics_mean_uncertainty.is_finite());
    }

    #[test]
    fn unvalidated_dynamics_is_observed_but_has_zero_planner_influence() {
        let mut dynamics = CausalDynamicsModel::new(LOCAL_ID);
        for sample in 1..=6 {
            let before = context(sample);
            dynamics.observe_action(
                "boost:Editor",
                "boost",
                "coding",
                &before,
                WorldStateDelta {
                    pressure: 0.50,
                    fluidity: -0.50,
                    ..WorldStateDelta::default()
                },
                4,
                0.98,
                before.timestamp_unix + 4,
                HardwareRegime::from_context(&before),
                LOCAL_ID,
                sample,
            );
        }
        let mut memory = TemporalMemory::default();
        let mut latest = context(100);
        for cycle in 95..=100 {
            latest = context(cycle);
            memory.observe(&latest);
        }
        let models = HashMap::from([("boost:Editor".to_string(), model(&latest, 0.04, 0.04))]);
        let actions = ["boost:Editor".to_string()];
        let legacy = plan_temporal_sequence(
            &memory,
            Some(&latest),
            LOCAL_ID,
            true,
            &models,
            &actions,
            "coding",
        );
        let shadow = plan_temporal_sequence_with_dynamics(
            &memory,
            Some(&latest),
            LOCAL_ID,
            true,
            &models,
            Some(&dynamics),
            &actions,
            "coding",
        );

        assert_eq!(shadow.dynamics_predictions, 1);
        assert_eq!(shadow.dynamics_ranking_predictions, 0);
        assert_eq!(shadow.dynamics_authoritative_predictions, 0);
        assert!(!shadow.dynamics_baseline_used);
        assert!((shadow.expected_gain - legacy.expected_gain).abs() < 1e-12);
        assert_eq!(shadow.authoritative, legacy.authoritative);
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
    fn immature_exact_transition_stays_in_exploratory_lane() {
        let (memory, latest) = warmed_memory();
        let mut immature = model(&latest, 0.08, 0.04);
        immature.observations = 4;
        immature.evidence_mass = 4.0;
        immature.state_evidence_mass = 4.0;
        let models = HashMap::from([("boost:Editor".to_string(), immature)]);
        let plan = plan_temporal_sequence(
            &memory,
            Some(&latest),
            LOCAL_ID,
            true,
            &models,
            &["boost:Editor".to_string()],
            "coding",
        );

        assert!(plan.action_scores.contains_key("boost:Editor"));
        assert!(plan.authoritative_action_scores.is_empty());
        assert!(!plan.authoritative);
        assert_eq!(plan.abstention_reason, Some("ranking_only"));
    }

    #[test]
    fn speculative_winner_does_not_hide_authoritative_one_step() {
        let (memory, latest) = warmed_memory();
        let models = HashMap::from([
            ("boost:Editor".to_string(), model(&latest, 0.03, 0.02)),
            ("markov_prewarm:*".to_string(), model(&latest, 0.80, 0.45)),
        ]);
        let plan = plan_temporal_sequence(
            &memory,
            Some(&latest),
            LOCAL_ID,
            true,
            &models,
            &["boost:Editor".to_string()],
            "coding",
        );

        assert_eq!(plan.best_first.as_deref(), Some("boost:Editor"));
        assert_eq!(
            plan.best_second.as_deref(),
            Some("markov_prewarm:predicted_app")
        );
        assert_eq!(
            plan.authoritative_best_first.as_deref(),
            Some("boost:Editor")
        );
        assert_eq!(plan.authoritative_best_second, None);
        assert!(plan.authoritative);
        assert!(plan.authoritative_expected_gain > 0.0);
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
