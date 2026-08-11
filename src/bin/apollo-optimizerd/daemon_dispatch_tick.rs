//! Dispatch Tick — final decision and execution phase of the daemon loop.
//!
//! Handles:
//! 1. Filter pipeline execution (circuit breaker, degradation, cognitive gates).
//! 2. Predictive thaw gate (model-informed control to prevent spikes).
//! 3. Action dispatch via `execute_actions`.
//! 4. Circuit breaker and degradation state updates.
//! 5. Frozen state persistence.

use chrono::Utc;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use apollo_engine::collector::{SystemCollector, SystemSnapshot};
use apollo_engine::engine::action_planner::{
    plan_actions, GpuRankInfluence, IntentEvidence, PlanReport, PlanningContext,
};
use apollo_engine::engine::actuation_broker::{ActuationBroker, ActuationRequest};
use apollo_engine::engine::audit_types::{DecisionReason, PolicyDecisionTrace};
use apollo_engine::engine::daemon_helpers::write_frozen_state;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::decision_ledger::{ActuatorDecisionOutcome, CycleDecisionEvents};
use apollo_engine::engine::degradation::DegradationInputs;
use apollo_engine::engine::execute_actions::{
    decision_event_for_root_action_from, ExecuteOutcomes,
};
use apollo_engine::engine::gpu_imagination::{
    GpuImaginationCandidate, GpuImaginationGate, GpuImaginationRequest, GpuImaginationWorker,
};
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::lse_counters::LockFreeMetrics;
use apollo_engine::engine::recently_applied::{CachedActionKind, RecentlyApplied};
use apollo_engine::engine::swap_reclaim::SwapRisk;
use apollo_engine::engine::telemetry_medallion::DecisionAttribution;
use apollo_engine::engine::types::{FreezeSource, FrozenEntry, RootAction};
use apollo_engine::engine::unfreeze_decay::UnfreezeDecayModel;

/// Action kinds tracked for per-PID dedup.
/// Variants without a target PID (SetSysctl, ToggleSpotlight, QuarantineDaemon)
/// bypass the consolidator and are kept verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DedupKind {
    SetMemorystatus,
    Throttle,
    Freeze,
    Unfreeze,
    Boost,
    SetThreadQoS,
}

/// Counts of duplicate actions dropped per kind in a single dispatch cycle.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DedupStats {
    pub set_memorystatus: u64,
    pub throttle: u64,
    pub freeze: u64,
    pub unfreeze: u64,
    pub boost: u64,
    pub set_thread_qos: u64,
}

impl DedupStats {
    pub fn total_dropped(&self) -> u64 {
        self.set_memorystatus
            + self.throttle
            + self.freeze
            + self.unfreeze
            + self.boost
            + self.set_thread_qos
    }
}

/// Extract `(pid, kind)` for actions targeting a specific process. Returns
/// `None` for actions without a PID target (sysctl, spotlight, quarantine).
///
/// SetThreadQoS uses a `(pid, thread_index)`-aware key. Conflicting tiers are
/// resolved separately so an interactive promotion wins over a stale demotion.
fn dedup_key(action: &RootAction) -> Option<(u32, DedupKind, u32)> {
    match action {
        RootAction::SetMemorystatus { pid, .. } => Some((*pid, DedupKind::SetMemorystatus, 0)),
        RootAction::ThrottleProcess { pid, .. } => Some((*pid, DedupKind::Throttle, 0)),
        RootAction::FreezeProcess { pid, .. } => Some((*pid, DedupKind::Freeze, 0)),
        RootAction::UnfreezeProcess { pid, .. } => Some((*pid, DedupKind::Unfreeze, 0)),
        RootAction::BoostProcess { pid, .. } => Some((*pid, DedupKind::Boost, 0)),
        RootAction::SetThreadQoS {
            pid, thread_index, ..
        } => Some((*pid, DedupKind::SetThreadQoS, *thread_index)),
        RootAction::SetSysctl(_)
        | RootAction::ToggleSpotlight { .. }
        | RootAction::QuarantineDaemon { .. } => None,
    }
}

fn thread_qos_rank(action: &RootAction) -> u8 {
    match action {
        RootAction::SetThreadQoS { tier, .. } if tier == "interactive" => 3,
        RootAction::SetThreadQoS { tier, .. } if tier == "background" => 1,
        RootAction::SetThreadQoS { .. } => 2,
        _ => 0,
    }
}

/// Consolidate per-PID actions: keep at most one action per `(pid, kind)`,
/// drop subsequent duplicates. Conflict resolution between different kinds
/// for the same PID is intentionally NOT performed here — those represent
/// distinct intents (e.g., Throttle and SetMemorystatus on the same PID
/// can coexist legitimately).
///
/// Closes the Critical gap from NotebookLM peer review (2026-05-06):
/// 14 emission paths constructed RootActions without per-PID dedup,
/// causing pid 65808 to receive SetMemorystatus 8× in same second.
///
/// [Saltzer & Schroeder 1975] Economy of Mechanism — single chokepoint
/// before execute_actions eliminates the bug class without touching
/// every emission site.
pub fn consolidate_actions_per_pid(actions: Vec<RootAction>) -> (Vec<RootAction>, DedupStats) {
    let (actions, stats, _) = consolidate_actions_per_pid_with_dropped(actions);
    (actions, stats)
}

fn consolidate_actions_per_pid_with_dropped(
    actions: Vec<RootAction>,
) -> (Vec<RootAction>, DedupStats, Vec<RootAction>) {
    let mut seen: HashMap<(u32, DedupKind, u32), usize> = HashMap::with_capacity(actions.len());
    let mut stats = DedupStats::default();
    let mut out: Vec<RootAction> = Vec::with_capacity(actions.len());
    let mut dropped = Vec::new();

    for action in actions {
        match dedup_key(&action) {
            Some(key) => {
                if let std::collections::hash_map::Entry::Vacant(entry) = seen.entry(key) {
                    entry.insert(out.len());
                    out.push(action);
                } else {
                    match key.1 {
                        DedupKind::SetMemorystatus => stats.set_memorystatus += 1,
                        DedupKind::Throttle => stats.throttle += 1,
                        DedupKind::Freeze => stats.freeze += 1,
                        DedupKind::Unfreeze => stats.unfreeze += 1,
                        DedupKind::Boost => stats.boost += 1,
                        DedupKind::SetThreadQoS => stats.set_thread_qos += 1,
                    }
                    if key.1 == DedupKind::SetThreadQoS {
                        let existing_idx = seen[&key];
                        if thread_qos_rank(&action) > thread_qos_rank(&out[existing_idx]) {
                            dropped.push(std::mem::replace(&mut out[existing_idx], action));
                        } else {
                            dropped.push(action);
                        }
                    } else {
                        dropped.push(action);
                    }
                }
            }
            None => out.push(action),
        }
    }
    (out, stats, dropped)
}

/// Increment lock-free dedup_drops counters from DedupStats.
/// Called by run_dispatch_tick after consolidate_actions_per_pid.
pub fn record_dedup_drops(lf: &LockFreeMetrics, stats: &DedupStats) {
    if stats.set_memorystatus > 0 {
        lf.add_dedup_drops_setmemorystatus(stats.set_memorystatus);
    }
    if stats.throttle > 0 {
        lf.add_dedup_drops_throttle(stats.throttle);
    }
    if stats.freeze > 0 {
        lf.add_dedup_drops_freeze(stats.freeze);
    }
    if stats.unfreeze > 0 {
        lf.add_dedup_drops_unfreeze(stats.unfreeze);
    }
    if stats.boost > 0 {
        lf.add_dedup_drops_boost(stats.boost);
    }
    if stats.set_thread_qos > 0 {
        lf.add_dedup_drops_thread_qos(stats.set_thread_qos);
    }
}

/// Commit only confirmed system mutations to the cross-cycle cache. Queue
/// admission, cognitive filtering, dry-run simulation and failed syscalls are
/// intentionally excluded so a blocked action can be retried next cycle.
pub fn record_applied_actions(
    traces: &[PolicyDecisionTrace],
    recently_applied: &mut RecentlyApplied,
) -> usize {
    let mut recorded = 0;
    for trace in traces.iter().filter(|trace| trace.applied) {
        if let Some((pid, kind, discriminator)) =
            CachedActionKind::from_root_action(&trace.intended_action)
        {
            recently_applied.record_scoped(pid, kind, discriminator);
            recorded += 1;
        }
    }
    recorded
}

use crate::{cognitive_tick, daemon_action_pipeline};

fn record_world_model_abstention(
    metrics: &mut apollo_engine::engine::types::RuntimeMetrics,
    reason: apollo_engine::engine::world_model::UtilityAbstentionReason,
    action_key: &str,
    workload: &str,
) {
    use apollo_engine::engine::world_model::UtilityAbstentionReason;

    metrics.world_model_abstentions_total = metrics.world_model_abstentions_total.saturating_add(1);
    metrics.world_model_last_abstention_reason = reason.as_str().to_string();
    metrics.world_model_last_abstention_action = action_key.to_string();
    metrics.world_model_last_abstention_workload = workload.to_string();
    let counter = match reason {
        UtilityAbstentionReason::NoCurrentGold => &mut metrics.world_model_abstention_no_gold_total,
        UtilityAbstentionReason::UnknownAction => &mut metrics.world_model_abstention_unknown_total,
        UtilityAbstentionReason::ImmatureEvidence => {
            &mut metrics.world_model_abstention_immature_total
        }
        UtilityAbstentionReason::LowQuality => &mut metrics.world_model_abstention_quality_total,
        UtilityAbstentionReason::StaleEvidence => &mut metrics.world_model_abstention_stale_total,
        UtilityAbstentionReason::ForeignInstallation => {
            &mut metrics.world_model_abstention_origin_total
        }
        UtilityAbstentionReason::HardwareMismatch => {
            &mut metrics.world_model_abstention_hardware_total
        }
        UtilityAbstentionReason::UncertainInterval => {
            &mut metrics.world_model_abstention_uncertain_total
        }
    };
    *counter = counter.saturating_add(1);
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorldModelInfluence {
    pub kind: String,
    pub action_key: String,
    pub workload: String,
    pub scope: String,
    pub utility: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub evidence: f64,
    pub quality: f64,
    pub margin: f64,
}

impl WorldModelInfluence {
    fn from_assessment(
        kind: &str,
        action_key: String,
        workload: &str,
        assessment: apollo_engine::engine::world_model::UtilityAssessment,
    ) -> Self {
        let margin = match assessment.verdict {
            apollo_engine::engine::world_model::UtilityImagined::ActWins { margin } => margin,
            apollo_engine::engine::world_model::UtilityImagined::DoNothingDominates {
                predicted_utility,
            } => predicted_utility,
            apollo_engine::engine::world_model::UtilityImagined::Unknown => 0.0,
        };
        Self {
            kind: kind.to_string(),
            action_key,
            workload: workload.to_string(),
            scope: assessment.scope.as_str().to_string(),
            utility: assessment.utility_ema,
            lower_bound: assessment.lower_bound,
            upper_bound: assessment.upper_bound,
            evidence: assessment.effective_evidence,
            quality: assessment.quality,
            margin,
        }
    }
}

fn decision_proposer(action: &RootAction) -> String {
    if action.reason().contains("predictive-agent") {
        return "predictive-agent".to_string();
    }
    match action.decision_reason() {
        DecisionReason::CausalInference => "causal-specialist",
        DecisionReason::InteractiveFocus
        | DecisionReason::MLWorkload
        | DecisionReason::DisplayPipeline
        | DecisionReason::CompositorPriority
        | DecisionReason::ThreadQoSRouting => "interaction-specialist",
        DecisionReason::WaitGraphBlocker => "wait-graph",
        DecisionReason::OutcomeIneffective => "outcome-model",
        DecisionReason::AnomalyDetected
        | DecisionReason::IoBurst
        | DecisionReason::WakeupVampire
        | DecisionReason::DramBandwidth => "signal-intelligence",
        DecisionReason::MemoryBudget => "memory-budget",
        DecisionReason::SwarmThrottling | DecisionReason::GraduatedIdle => "adaptive-governor",
        DecisionReason::PressureContext
        | DecisionReason::CriticalBypass
        | DecisionReason::HysteresisRecovery => "pressure-specialist",
        DecisionReason::IpcProtected
        | DecisionReason::UserActiveSkip
        | DecisionReason::HrpoSkip => "safety-policy",
        DecisionReason::Other(_) => "specialist",
    }
    .to_string()
}

fn decision_attribution(
    action: &RootAction,
    key: String,
    world_model: &apollo_engine::engine::world_model::WorldModel,
    workload: &str,
    evidence: IntentEvidence,
    temporal_gain: Option<f64>,
    gpu_support: Option<f64>,
) -> DecisionAttribution {
    use apollo_engine::engine::world_model::{Imagined, UtilityImagined};

    let mut supporters = Vec::with_capacity(6);
    let mut vetoes = Vec::with_capacity(4);
    if let Some(assessment) = world_model.assess_utility(&key, workload) {
        match assessment.verdict {
            UtilityImagined::ActWins { .. } => supporters.push("world-model".to_string()),
            UtilityImagined::DoNothingDominates { .. } => vetoes.push("world-model".to_string()),
            UtilityImagined::Unknown => {}
        }
        if assessment.counterfactual_support > f64::EPSILON {
            supporters.push("noop-control".to_string());
        } else if assessment.counterfactual_support < -f64::EPSILON {
            vetoes.push("noop-control".to_string());
        }
        if assessment.episodic_support > f64::EPSILON {
            supporters.push("episodic-memory".to_string());
        } else if assessment.episodic_support < -f64::EPSILON {
            vetoes.push("episodic-memory".to_string());
        }
    } else if world_model.assess_family_prior(&key).is_some() {
        supporters.push("family-prior".to_string());
    }
    match world_model.imagine(&key) {
        Imagined::ActWins { .. } => supporters.push("causal-model".to_string()),
        Imagined::DoNothingDominates { .. } => vetoes.push("causal-model".to_string()),
        Imagined::Unknown => {}
    }
    if let Some(gain) = temporal_gain {
        if gain > f64::EPSILON {
            supporters.push("world-sequence".to_string());
        } else if gain < -f64::EPSILON {
            vetoes.push("world-sequence".to_string());
        }
    }
    if let Some(support) = gpu_support {
        if support > f64::EPSILON {
            supporters.push("gpu-model".to_string());
        } else if support < -f64::EPSILON {
            vetoes.push("gpu-model".to_string());
        }
    }
    DecisionAttribution {
        action_key: key,
        proposer: decision_proposer(action),
        supporters,
        vetoes,
        predicted_gain: evidence.expected_benefit.clamp(-1.0, 1.0),
        uncertainty: evidence.uncertainty.clamp(0.0, 1.0),
    }
}

#[derive(Debug, Clone, PartialEq)]
enum UtilityGateDecision {
    Admit,
    Veto(WorldModelInfluence),
    Abstained {
        reason: apollo_engine::engine::world_model::UtilityAbstentionReason,
        action_key: String,
    },
}

fn evaluate_utility_gate(
    action: &RootAction,
    world_model: &apollo_engine::engine::world_model::WorldModel,
    workload: &str,
) -> UtilityGateDecision {
    if !apollo_engine::engine::telemetry_medallion::utility_veto_eligible(action) {
        return UtilityGateDecision::Admit;
    }
    let Some(key) = apollo_engine::engine::telemetry_medallion::actuator_action_key(action) else {
        return UtilityGateDecision::Admit;
    };
    match world_model.assess_utility_diagnostic(&key, workload) {
        apollo_engine::engine::world_model::UtilityAssessmentResult::Assessed(assessment)
            if matches!(
                assessment.verdict,
                apollo_engine::engine::world_model::UtilityImagined::DoNothingDominates { .. }
            ) =>
        {
            UtilityGateDecision::Veto(WorldModelInfluence::from_assessment(
                "veto", key, workload, assessment,
            ))
        }
        apollo_engine::engine::world_model::UtilityAssessmentResult::Abstained(abstention) => {
            UtilityGateDecision::Abstained {
                reason: abstention.reason,
                action_key: key,
            }
        }
        _ => UtilityGateDecision::Admit,
    }
}

/// Build a single intent plan from specialist proposals and mature World
/// Model evidence. The learned model contributes only utility and uncertainty;
/// the planner remains unable to manufacture actions or bypass safety gates.
pub fn plan_action_intents(
    actions: Vec<RootAction>,
    world_model: &apollo_engine::engine::world_model::WorldModel,
    workload: &str,
    memory_pressure: f64,
    app_launching: bool,
    fluidity_degraded: bool,
) -> (Vec<RootAction>, PlanReport) {
    plan_action_intents_inner(
        actions,
        world_model,
        workload,
        memory_pressure,
        app_launching,
        fluidity_degraded,
        None,
    )
}

pub struct GpuPlannerRuntime<'a> {
    pub cycle: u64,
    pub worker: &'a mut GpuImaginationWorker,
    pub gate: GpuImaginationGate,
    pub external_candidates: Vec<GpuImaginationCandidate>,
}

fn gpu_portfolio_candidate(
    world_model: &apollo_engine::engine::world_model::WorldModel,
    workload: &str,
    action_key: &str,
    specialist_gain: f64,
    specialist_uncertainty: f64,
) -> GpuImaginationCandidate {
    let (expected_gain, uncertainty) = if let Some(assessment) =
        world_model.assess_utility(action_key, workload)
    {
        (
            (specialist_gain * 0.35
                + assessment.utility_ema * 0.65
                + assessment.counterfactual_support
                + assessment.episodic_support)
                .clamp(-1.0, 1.0),
            ((assessment.upper_bound - assessment.lower_bound).abs() * 0.70
                + specialist_uncertainty * 0.30)
                .clamp(0.0, 1.0),
        )
    } else if let Some(prior) = world_model.assess_family_prior(action_key) {
        (
            (specialist_gain * 0.70 + prior.utility_ema * 0.30).clamp(-1.0, 1.0),
            ((prior.upper_bound - prior.lower_bound).abs() * 0.40 + specialist_uncertainty * 0.60)
                .clamp(0.0, 1.0),
        )
    } else {
        (specialist_gain, specialist_uncertainty)
    };
    GpuImaginationCandidate {
        action_key: action_key.to_string(),
        expected_gain,
        uncertainty,
    }
}

/// Build the live cross-engine candidate set. These are existing specialist
/// actions, not new commands: the GPU only estimates robustness so each
/// owning lane can tune its already-admitted decision through the World Model.
pub fn build_gpu_candidate_portfolio(
    world_model: &apollo_engine::engine::world_model::WorldModel,
    workload: &str,
    runtime: &apollo_engine::engine::types::RuntimeMetrics,
) -> Vec<GpuImaginationCandidate> {
    let mut candidates = Vec::with_capacity(12);
    let mut push = |action_key: &str, gain: f64, uncertainty: f64| {
        candidates.push(gpu_portfolio_candidate(
            world_model,
            workload,
            action_key,
            gain,
            uncertainty,
        ));
    };

    if !runtime.markov_prediction_app.is_empty() && runtime.markov_prediction_confidence > 0.0 {
        let confidence = runtime.markov_prediction_confidence.clamp(0.0, 1.0);
        push(
            "markov_prewarm:predicted_app",
            0.01 + confidence * 0.03,
            1.0 - confidence,
        );
    }
    if runtime.interaction_qos_active || runtime.app_launching || runtime.window_op_active {
        push("interaction_qos:foreground", 0.020, 0.30);
        push("io_shaping:interactive_release", 0.012, 0.40);
    }
    if runtime.chromium_renderers_total > 0 {
        push("chromium_ecore:background_renderer", 0.015, 0.35);
        push("chromium_purge:purgeable_renderer", 0.008, 0.55);
        push("chromium_jetsam:background_renderer", 0.006, 0.60);
    }
    if runtime.predictive_agent_active {
        push("predictive_threshold:tighten", 0.010, 0.50);
        push("predictive_profile:aggressive", 0.010, 0.50);
        push("predictive_prethrottle:noise", 0.006, 0.60);
        push("predictive_purge:kernel", 0.006, 0.60);
    }
    candidates
}

pub fn plan_action_intents_with_gpu(
    actions: Vec<RootAction>,
    world_model: &apollo_engine::engine::world_model::WorldModel,
    workload: &str,
    memory_pressure: f64,
    app_launching: bool,
    fluidity_degraded: bool,
    gpu: GpuPlannerRuntime<'_>,
) -> (Vec<RootAction>, PlanReport) {
    plan_action_intents_inner(
        actions,
        world_model,
        workload,
        memory_pressure,
        app_launching,
        fluidity_degraded,
        Some(gpu),
    )
}

fn plan_action_intents_inner(
    actions: Vec<RootAction>,
    world_model: &apollo_engine::engine::world_model::WorldModel,
    workload: &str,
    memory_pressure: f64,
    app_launching: bool,
    fluidity_degraded: bool,
    mut gpu: Option<GpuPlannerRuntime<'_>>,
) -> (Vec<RootAction>, PlanReport) {
    let mut context = PlanningContext {
        memory_pressure,
        app_launching,
        fluidity_degraded,
        ..PlanningContext::default()
    };
    let mut family_priors_used = 0_u64;
    let mut exact_positive_evidence = 0_u64;
    let mut counterfactual_ranked = 0_u64;
    let mut episodic_ranked = 0_u64;
    for action in &actions {
        let Some(key) = apollo_engine::engine::telemetry_medallion::actuator_action_key(action)
        else {
            continue;
        };
        let Some(assessment) = world_model.assess_utility(&key, workload) else {
            let prior = world_model.assess_family_prior(&key);
            let episodic = world_model.recall_similar_episodes(&key, workload);
            if prior.is_some() || episodic.is_some() {
                let prior_benefit = prior
                    .map(|prior| prior.lower_bound.max(0.0) * 0.50)
                    .unwrap_or(0.0);
                let episodic_benefit = episodic.map_or(0.0, |recall| recall.rank_support);
                let prior_uncertainty =
                    prior.map(|prior| (prior.upper_bound - prior.lower_bound).abs());
                let episodic_uncertainty = episodic.map(|recall| 1.0 - recall.mean_similarity);
                let uncertainty = match (prior_uncertainty, episodic_uncertainty) {
                    (Some(prior), Some(episode)) => prior.max(episode),
                    (Some(prior), None) => prior,
                    (None, Some(episode)) => episode,
                    (None, None) => 1.0,
                }
                .clamp(0.0, 1.0);
                context.utility_evidence.insert(
                    key,
                    IntentEvidence {
                        expected_benefit: prior_benefit + episodic_benefit,
                        uncertainty,
                    },
                );
            }
            if prior.is_some() {
                family_priors_used = family_priors_used.saturating_add(1);
            }
            if episodic.is_some_and(|recall| recall.rank_support.abs() > f64::EPSILON) {
                episodic_ranked = episodic_ranked.saturating_add(1);
            }
            continue;
        };
        let expected_benefit = match assessment.verdict {
            apollo_engine::engine::world_model::UtilityImagined::ActWins { margin } => {
                exact_positive_evidence = exact_positive_evidence.saturating_add(1);
                if assessment.counterfactual_observations > 0 {
                    counterfactual_ranked = counterfactual_ranked.saturating_add(1);
                }
                if assessment.episodic_observations > 0 {
                    episodic_ranked = episodic_ranked.saturating_add(1);
                }
                (margin + assessment.counterfactual_support + assessment.episodic_support).max(0.0)
            }
            apollo_engine::engine::world_model::UtilityImagined::DoNothingDominates {
                predicted_utility,
            } => predicted_utility.min(0.0),
            apollo_engine::engine::world_model::UtilityImagined::Unknown => {
                if assessment.episodic_observations > 0 {
                    episodic_ranked = episodic_ranked.saturating_add(1);
                }
                assessment.episodic_support
            }
        };
        context.utility_evidence.insert(
            key,
            IntentEvidence {
                expected_benefit,
                uncertainty: (assessment.upper_bound - assessment.lower_bound)
                    .abs()
                    .clamp(0.0, 1.0),
            },
        );
    }
    let action_keys: Vec<String> = actions
        .iter()
        // The central planner only reorders accelerator slots. Keeping the
        // rollout set identical prevents it from imagining an order that the
        // safety-priority dispatcher will intentionally refuse to execute.
        .filter(|action| {
            matches!(action, RootAction::BoostProcess { .. })
                || matches!(action, RootAction::SetThreadQoS { tier, .. } if tier == "interactive")
        })
        .filter_map(apollo_engine::engine::telemetry_medallion::actuator_action_key)
        .collect();
    let temporal_plan = world_model.plan_temporal_sequence(&action_keys, workload);
    let mut gpu_backend = String::new();
    let mut gpu_device = String::new();
    let mut gpu_submit_outcome = String::new();
    let mut gpu_completed = None;
    let mut gpu_initialization_error = None;
    let mut gpu_support_uses = 0_u64;
    let mut gpu_supported_actions = Vec::new();
    if let Some(runtime) = gpu.as_mut() {
        gpu_completed = runtime.worker.take_completed();
        let latest_result = runtime.worker.latest().cloned();
        let mut supported_keys = HashSet::new();
        for key in &action_keys {
            if !supported_keys.insert(key) {
                continue;
            }
            // Prefer the World Model's bounded cache: the async worker may
            // have already moved on to another specialist portfolio, while
            // this root action still has fresh advice from its own batch.
            let support = world_model.gpu_rank_support_for(key, workload).or_else(|| {
                latest_result
                    .as_ref()
                    .filter(|result| result.is_fresh_for(runtime.cycle, workload))
                    .and_then(|result| result.support_for(key))
                    .map(|raw| world_model.calibrate_gpu_rank_support(key, workload, raw))
                    .filter(|support| support.abs() > f64::EPSILON)
            });
            let Some(support) = support else {
                continue;
            };
            let evidence = context.utility_evidence.entry(key.clone()).or_default();
            evidence.expected_benefit += support;
            gpu_support_uses = gpu_support_uses.saturating_add(1);
            gpu_supported_actions.push(GpuRankInfluence {
                action_key: key.clone(),
                support,
            });
        }
        let mut gpu_candidates: Vec<_> = temporal_plan
            .action_scores
            .iter()
            .map(|(key, score)| GpuImaginationCandidate {
                action_key: key.clone(),
                expected_gain: score.expected_gain,
                uncertainty: score.uncertainty,
            })
            .collect();
        let temporal_keys: HashSet<_> = gpu_candidates
            .iter()
            .map(|candidate| candidate.action_key.clone())
            .collect();
        // Cold root actions do not yet have a temporal score. They still need
        // a chance to be imagined, otherwise the only GPU work is the
        // cross-engine portfolio and a root ranking can never get evidence.
        for key in &action_keys {
            if !temporal_keys.contains(key) {
                gpu_candidates.push(gpu_portfolio_candidate(
                    world_model,
                    workload,
                    key,
                    0.0,
                    1.0,
                ));
            }
        }
        gpu_candidates.sort_by(|left, right| {
            right
                .expected_gain
                .total_cmp(&left.expected_gain)
                .then_with(|| left.uncertainty.total_cmp(&right.uncertainty))
                .then_with(|| left.action_key.cmp(&right.action_key))
        });
        // Reserve half the bounded Metal batch for cross-engine specialists,
        // so a large process cohort cannot starve Markov/Chromium/QoS advice.
        gpu_candidates.truncate(12);
        gpu_candidates.extend(runtime.external_candidates.iter().cloned());
        gpu_submit_outcome = GpuImaginationRequest::new(runtime.cycle, workload, gpu_candidates)
            .map(|request| runtime.worker.try_submit(request, runtime.gate).as_str())
            .unwrap_or("no-candidates")
            .to_string();
        gpu_backend = runtime.worker.backend().as_str().to_string();
        gpu_device = runtime.worker.device_name().to_string();
        gpu_initialization_error = runtime.worker.initialization_error().map(str::to_string);
    }
    let mut temporal_promotions = 0_u64;
    for (key, score) in &temporal_plan.action_scores {
        let exploratory_gain = score.expected_gain.max(0.0) * 0.20;
        let authoritative = temporal_plan
            .authoritative
            .then(|| temporal_plan.authoritative_action_scores.get(key))
            .flatten();
        let authoritative_gain = authoritative
            .map(|score| score.expected_gain.max(0.0) * 0.50)
            .unwrap_or(0.0);
        let (gain, uncertainty) = if authoritative_gain > exploratory_gain {
            (
                authoritative_gain,
                authoritative.map_or(score.uncertainty, |score| score.uncertainty),
            )
        } else {
            (exploratory_gain, score.uncertainty)
        };
        if gain <= 0.0 {
            continue;
        }
        temporal_promotions = temporal_promotions.saturating_add(1);
        let evidence = context.utility_evidence.entry(key.clone()).or_default();
        evidence.expected_benefit += gain;
        evidence.uncertainty = evidence.uncertainty.max(uncertainty);
    }
    let (planned, mut report) = plan_actions(actions, &context);
    report.decision_attributions = planned
        .iter()
        .filter_map(|action| {
            let key = apollo_engine::engine::telemetry_medallion::actuator_action_key(action)?;
            let evidence = context
                .utility_evidence
                .get(&key)
                .copied()
                .unwrap_or_default();
            let temporal_gain = temporal_plan
                .action_scores
                .get(&key)
                .map(|score| score.expected_gain);
            let gpu_support = gpu_supported_actions
                .iter()
                .find(|influence| influence.action_key == key)
                .map(|influence| influence.support);
            Some(decision_attribution(
                action,
                key,
                world_model,
                workload,
                evidence,
                temporal_gain,
                gpu_support,
            ))
        })
        .collect();
    report.evidence_ranked = exact_positive_evidence;
    report.family_priors_used = family_priors_used;
    report.counterfactual_ranked = counterfactual_ranked;
    report.episodic_ranked = episodic_ranked;
    report.temporal_candidates = temporal_plan.candidates;
    report.temporal_rollouts = temporal_plan.sequences_evaluated;
    report.temporal_promotions = temporal_promotions;
    report.temporal_memory_samples = temporal_plan.memory_samples;
    report.temporal_expected_gain = temporal_plan.expected_gain;
    report.temporal_uncertainty = temporal_plan.uncertainty;
    report.temporal_pressure_delta = temporal_plan.predicted_pressure_delta;
    report.temporal_fluidity_delta = temporal_plan.predicted_fluidity_delta;
    report.temporal_energy_delta = temporal_plan.predicted_energy_delta;
    report.temporal_authoritative = temporal_plan.authoritative;
    report.temporal_authoritative_rollouts = temporal_plan.authoritative_sequences_evaluated;
    report.temporal_authoritative_expected_gain = temporal_plan.authoritative_expected_gain;
    report.temporal_authoritative_uncertainty = temporal_plan.authoritative_uncertainty;
    report.temporal_authoritative_pressure_delta = temporal_plan.authoritative_pressure_delta;
    report.temporal_authoritative_fluidity_delta = temporal_plan.authoritative_fluidity_delta;
    report.temporal_authoritative_energy_delta = temporal_plan.authoritative_energy_delta;
    report.dynamics_predictions = temporal_plan.dynamics_predictions;
    report.dynamics_ranking_predictions = temporal_plan.dynamics_ranking_predictions;
    report.dynamics_authoritative_predictions = temporal_plan.dynamics_authoritative_predictions;
    report.dynamics_baseline_used = temporal_plan.dynamics_baseline_used;
    report.dynamics_mean_uncertainty = temporal_plan.dynamics_mean_uncertainty;
    report.gpu_imagination_backend = gpu_backend;
    report.gpu_imagination_device = gpu_device;
    report.gpu_imagination_submit_outcome = gpu_submit_outcome;
    report.gpu_imagination_support_uses = gpu_support_uses;
    report.gpu_imagination_supported_actions = gpu_supported_actions;
    if let Some(result) = gpu_completed {
        report.gpu_imagination_result = Some(result.clone());
        report.gpu_imagination_completed = true;
        report.gpu_imagination_error = result.error.clone();
        report.gpu_imagination_samples = result.samples;
        report.gpu_imagination_gpu_time_ns = result.gpu_time_ns;
        report.gpu_imagination_wall_time_ns = result.wall_time_ns;
        if let Some(best) = result.best() {
            report.gpu_imagination_best_action = Some(best.action_key.clone());
            report.gpu_imagination_best_positive_probability = best.positive_probability;
            report.gpu_imagination_best_p10_gain = best.p10_gain;
        }
    } else {
        report.gpu_imagination_error = gpu_initialization_error;
    }
    report.temporal_best_first = temporal_plan.best_first;
    report.temporal_best_second = temporal_plan.best_second;
    report.temporal_authoritative_best_first = temporal_plan.authoritative_best_first;
    report.temporal_authoritative_best_second = temporal_plan.authoritative_best_second;
    report.temporal_abstention_reason = temporal_plan.abstention_reason.map(str::to_string);
    (planned, report)
}

pub fn record_world_model_influence(
    metrics: &mut apollo_engine::engine::types::RuntimeMetrics,
    influence: &WorldModelInfluence,
) {
    metrics.world_model_last_influence_kind = influence.kind.clone();
    metrics.world_model_last_influence_action = influence.action_key.clone();
    metrics.world_model_last_influence_workload = influence.workload.clone();
    metrics.world_model_last_influence_scope = influence.scope.clone();
    metrics.world_model_last_influence_utility = influence.utility;
    metrics.world_model_last_influence_lower_bound = influence.lower_bound;
    metrics.world_model_last_influence_upper_bound = influence.upper_bound;
    metrics.world_model_last_influence_evidence = influence.evidence;
    metrics.world_model_last_influence_quality = influence.quality;
    metrics.world_model_last_influence_margin = influence.margin;
}

fn is_responsiveness_accelerator(action: &RootAction) -> bool {
    matches!(action, RootAction::BoostProcess { .. })
        || matches!(action, RootAction::SetThreadQoS { tier, .. } if tier == "interactive")
}

const CONTROLLED_HOLDOUT_MODULUS: u64 = 64;

fn controlled_holdout_slot(cycle: u64, action_key: &str) -> bool {
    let hash = action_key
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            hash.wrapping_mul(0x100_0000_01b3) ^ byte as u64
        });
    (hash ^ cycle).is_multiple_of(CONTROLLED_HOLDOUT_MODULUS)
}

fn controlled_holdout_safe(pressure: f64, app_launching: bool, fluidity_degraded: bool) -> bool {
    pressure.is_finite() && pressure < 0.55 && !app_launching && !fluidity_degraded
}

#[derive(Debug, Clone)]
pub struct CounterfactualHoldout {
    pub action: RootAction,
    pub action_key: String,
}

/// Input dependencies for the dispatch tick.
pub struct DispatchTickInput<'a> {
    pub state: &'a SharedState,
    pub caps: &'a apollo_engine::engine::types::CapabilityReport,
    pub journal_path: &'a Path,
    pub frozen_state_path: &'a Path,
    pub final_actions: Vec<RootAction>,
    pub snapshot: &'a SystemSnapshot,
    pub prev_cog_decision: Option<&'a cognitive_tick::CognitiveDecision>,
    pub causal_qos_names: &'a HashSet<String>,
    pub reclaim_risk: SwapRisk,
    pub unfreeze_decay: &'a mut UnfreezeDecayModel,
    pub collector: &'a SystemCollector,
    pub dry_run: bool,
    /// Lock-free metrics for per-cycle dedup_drops accounting.
    /// Optional so legacy callers and unit tests can pass `None`.
    pub lf_metrics: Option<&'a LockFreeMetrics>,
    /// Coalition guard: tracker + recent-fg envelope. None opts out of
    /// coalition-aware skipping (legacy callers / tests).
    pub coalition_guard:
        Option<&'a apollo_engine::engine::active_coalition_envelope::CoalitionGuard<'a>>,
    /// Per-cycle fraction of CPU cores pegged ≥0.80 busy (from
    /// background_collectors.cpu_saturation.pegged_fraction). When this
    /// rises above 0.80 with memory pressure <0.75, freeze/throttle are
    /// gated as `BlockReason::CpuSaturated`.
    pub cpu_pegged_fraction: f64,
    /// Mature Gold-only utility predictions for discretionary actuators.
    pub world_model: &'a apollo_engine::engine::world_model::WorldModel,
    pub workload: &'a str,
    pub cycle_count: u64,
}

/// Output results from the dispatch tick.
pub struct DispatchTickOutput {
    pub outcomes: ExecuteOutcomes,
    pub causal_qos_upgrades: u32,
    /// Dedup statistics from this cycle's consolidation pass.
    /// Currently consumed only by lf_metrics counters; may be read by
    /// downstream observers (Phase 6 self-healing layer) in future.
    #[allow(dead_code)]
    pub dedup_stats: DedupStats,
    pub counterfactual_holdouts: Vec<CounterfactualHoldout>,
}

/// Runs the dispatch and execution orchestration logic.
pub fn run_dispatch_tick(input: DispatchTickInput) -> DispatchTickOutput {
    let DispatchTickInput {
        state,
        caps,
        journal_path,
        frozen_state_path,
        final_actions,
        snapshot,
        prev_cog_decision,
        causal_qos_names,
        reclaim_risk,
        unfreeze_decay,
        collector,
        dry_run,
        lf_metrics,
        coalition_guard,
        cpu_pegged_fraction,
        world_model,
        workload,
        cycle_count,
    } = input;

    // ── Filter pipeline ──────────────────────────────────────────────────────
    let filter_outcome = daemon_action_pipeline::run_filter_pipeline(
        final_actions,
        state,
        snapshot,
        prev_cog_decision,
        causal_qos_names,
        reclaim_risk,
    );
    let cb_is_open = filter_outcome.cb_is_open;
    let op_mode = filter_outcome.op_mode;
    let mut filtered_actions = filter_outcome.filtered_actions;
    let causal_qos_upgrades = filter_outcome.causal_qos_upgrades;
    let mut dispatch_decision_events = CycleDecisionEvents::default();
    for (action, reason) in filter_outcome.blocked_actions {
        dispatch_decision_events.push(decision_event_for_root_action_from(
            &action,
            ActuatorDecisionOutcome::Blocked,
            "dispatch-mode-filter",
            reason.to_string(),
        ));
    }

    // Universal world-model gate for discretionary actions. Pressure relief
    // and recovery remain governed by their specialist safety paths. An
    // immature model always abstains, so this cannot starve exploration.
    let mut utility_vetoes = 0_u64;
    let mut last_utility_veto = None;
    let mut utility_abstentions = Vec::new();
    let mut utility_admitted = Vec::with_capacity(filtered_actions.len());
    for action in filtered_actions {
        match evaluate_utility_gate(&action, world_model, workload) {
            UtilityGateDecision::Veto(influence) => {
                utility_vetoes = utility_vetoes.saturating_add(1);
                tracing::debug!(action_key = %influence.action_key, workload, "world model vetoed low-utility action");
                last_utility_veto = Some(influence);
                dispatch_decision_events.push(decision_event_for_root_action_from(
                    &action,
                    ActuatorDecisionOutcome::Vetoed,
                    "world-model-gate",
                    "world-model-utility-veto".to_string(),
                ));
            }
            UtilityGateDecision::Abstained { reason, action_key } => {
                utility_abstentions.push((reason, action_key));
                utility_admitted.push(action);
            }
            UtilityGateDecision::Admit => utility_admitted.push(action),
        }
    }
    filtered_actions = utility_admitted;
    if utility_vetoes > 0 || !utility_abstentions.is_empty() {
        let mut metrics = state.metrics.lock_recover();
        metrics.metrics.world_model_utility_vetoes_total = metrics
            .metrics
            .world_model_utility_vetoes_total
            .saturating_add(utility_vetoes);
        if let Some(influence) = last_utility_veto.as_ref() {
            record_world_model_influence(&mut metrics.metrics, influence);
        }
        for (reason, action_key) in utility_abstentions {
            record_world_model_abstention(&mut metrics.metrics, reason, &action_key, workload);
        }
    }
    // ── Per-PID dedup chokepoint ─────────────────────────────────────────────
    // Single consolidation pass before execute_actions. 14 upstream emission
    // paths (decide_actions, daemon_paging_hints, daemon_agent_actions,
    // process_enrichment, local policy, freeze-confirmation, etc.) push freely;
    // here we collapse duplicate (pid, kind) pairs. Without this, pid 65808
    // received SetMemorystatus 8× in the same second (prod observation).
    // [Saltzer & Schroeder 1975] Economy of Mechanism.
    let (deduped, dedup_stats, dedup_dropped) =
        consolidate_actions_per_pid_with_dropped(filtered_actions);
    filtered_actions = deduped;
    for action in dedup_dropped {
        dispatch_decision_events.push(decision_event_for_root_action_from(
            &action,
            ActuatorDecisionOutcome::NoOp,
            "dispatch-dedup",
            "same-cycle-dedup".to_string(),
        ));
    }
    if let Some(lf) = lf_metrics {
        record_dedup_drops(lf, &dedup_stats);
    }
    if dedup_stats.total_dropped() > 0 {
        tracing::debug!(
            target: "apollo.dispatch.dedup",
            dropped_total = dedup_stats.total_dropped(),
            sm_status = dedup_stats.set_memorystatus,
            throttle = dedup_stats.throttle,
            freeze = dedup_stats.freeze,
            unfreeze = dedup_stats.unfreeze,
            boost = dedup_stats.boost,
            thread_qos = dedup_stats.set_thread_qos,
            "consolidate_actions_per_pid: collapsed duplicates"
        );
    }

    // ── Predictive thaw gate ─────────────────────────────────────────────
    // [Strogatz 2015 §2.3] model-informed control;
    // [Nygard 2018 §5] backpressure by action refusal.
    {
        const PRED_GATE_PRESSURE: f64 = 0.80;
        const MAX_PRED_GROWTH_BYTES: u64 = 200 * 1024 * 1024; // 200 MB
        let pressure = snapshot.pressure.memory_pressure;
        if pressure > PRED_GATE_PRESSURE {
            let mut deferred = 0u32;
            let mut admitted = Vec::with_capacity(filtered_actions.len());
            for action in filtered_actions {
                if let RootAction::UnfreezeProcess { pid, name, .. } = &action {
                    let m_0 = collector
                        .system()
                        .process(sysinfo::Pid::from_u32(*pid))
                        .map(|p| p.memory())
                        .unwrap_or(0);
                    let predicted = unfreeze_decay.predict_rss(name, m_0, 5.0);
                    let growth = predicted.saturating_sub(m_0);
                    if growth > MAX_PRED_GROWTH_BYTES {
                        tracing::info!(
                            target: "apollo.unfreeze_decay",
                            pid = *pid,
                            name = %name,
                            pressure = %format!("{:.2}", pressure),
                            growth_mb = growth / (1024 * 1024),
                            "deferring thaw: predicted RSS growth exceeds headroom"
                        );
                        deferred += 1;
                        dispatch_decision_events.push(decision_event_for_root_action_from(
                            &action,
                            ActuatorDecisionOutcome::Blocked,
                            "predictive-thaw-gate",
                            "predictive-thaw-rss-growth".to_string(),
                        ));
                        continue;
                    }
                }
                admitted.push(action);
            }
            filtered_actions = admitted;
            if deferred > 0 {
                tracing::warn!(
                    target: "apollo.unfreeze_decay",
                    deferred,
                    active_thaws = unfreeze_decay.active_thaw_count(),
                    learned_apps = unfreeze_decay.learned_app_count(),
                    "predictive thaw gate dropped {} candidate(s)",
                    deferred
                );
            }
        }
    }

    // A tiny deterministic control arm for mature, positive accelerators.
    // Selection happens after dedup and every predictive gate so withholding
    // one action cannot be defeated by an equivalent duplicate downstream.
    let (app_launching, fluidity_degraded, speculation_allowed) = {
        let metrics = state.metrics.lock_recover();
        (
            metrics.metrics.app_launching,
            metrics.metrics.fluidity_degraded,
            metrics.metrics.apollo_overhead_speculation_allowed,
        )
    };
    let holdout_safe = !cb_is_open
        && speculation_allowed
        && controlled_holdout_safe(
            snapshot.pressure.memory_pressure,
            app_launching,
            fluidity_degraded,
        );
    let mut counterfactual_holdouts = Vec::with_capacity(1);
    let mut counterfactual_eligible = 0_u64;
    if holdout_safe {
        filtered_actions.retain(|action| {
            if !counterfactual_holdouts.is_empty() || !is_responsiveness_accelerator(action) {
                return true;
            }
            let Some(key) = apollo_engine::engine::telemetry_medallion::actuator_action_key(action)
            else {
                return true;
            };
            let Some(assessment) = world_model.assess_utility(&key, workload) else {
                return true;
            };
            if !matches!(
                assessment.verdict,
                apollo_engine::engine::world_model::UtilityImagined::ActWins { .. }
            ) {
                return true;
            }
            counterfactual_eligible = counterfactual_eligible.saturating_add(1);
            if !controlled_holdout_slot(cycle_count, &key) {
                return true;
            }
            counterfactual_holdouts.push(CounterfactualHoldout {
                action: action.clone(),
                action_key: key,
            });
            dispatch_decision_events.push(decision_event_for_root_action_from(
                action,
                ActuatorDecisionOutcome::Rejected,
                "controlled-holdout",
                "controlled-counterfactual-holdout".to_string(),
            ));
            false
        });
    }
    if counterfactual_eligible > 0 {
        let mut metrics = state.metrics.lock_recover();
        metrics.metrics.world_model_counterfactual_eligible_total = metrics
            .metrics
            .world_model_counterfactual_eligible_total
            .saturating_add(counterfactual_eligible);
    }

    // ── Circuit breaker + execute_actions ────────────────────
    let mut frozen_set: HashSet<u32> = state.frozen_state.lock_recover().keys().copied().collect();
    let frozen_before: HashSet<u32> = frozen_set.clone();

    let (learned_protected, learned_interactive) = {
        let pg = state.policy.lock_recover();
        (
            pg.learned_policy.protected_patterns.clone(),
            pg.learned_policy.interactive_patterns.clone(),
        )
    };
    // S4 cutover (2026-06-06): pass the Arc clone directly; effectors lock
    // inside each call site. Drop the outer prelocked guard — the old
    // pattern was OK while execute_actions took &mut MachQoSManager but
    // would now hold the lock across the entire cycle.
    let qos_arc = state.mach_qos.clone();

    let broker = ActuationBroker::from_runtime(caps, dry_run);
    let broker_execution = if cb_is_open {
        // Circuit Open: only dispatch unfreeze (always safe).
        tracing::warn!(
            op_mode = op_mode.as_str(),
            "circuit-breaker: open — skipping execute_actions, dispatching unfreeze only"
        );
        let mut safe_actions = Vec::with_capacity(filtered_actions.len());
        for action in filtered_actions {
            if matches!(action, RootAction::UnfreezeProcess { .. }) {
                safe_actions.push(action);
            } else {
                dispatch_decision_events.push(decision_event_for_root_action_from(
                    &action,
                    ActuatorDecisionOutcome::Blocked,
                    "circuit-breaker",
                    "circuit-breaker-open".to_string(),
                ));
            }
        }
        broker.execute(ActuationRequest {
            actions: safe_actions,
            caps,
            journal_path,
            frozen: &mut frozen_set,
            learned_protected: &learned_protected,
            learned_interactive: &learned_interactive,
            qos_mgr: Some(&qos_arc),
            memory_pressure: snapshot.pressure.memory_pressure,
            thrashing_score: snapshot.pressure.thrashing_score,
            coalition_guard,
            cpu_pegged_fraction,
        })
    } else {
        // Circuit Closed or HalfOpen: run normally, then report outcome.
        let out = broker.execute(ActuationRequest {
            actions: filtered_actions,
            caps,
            journal_path,
            frozen: &mut frozen_set,
            learned_protected: &learned_protected,
            learned_interactive: &learned_interactive,
            qos_mgr: Some(&qos_arc),
            memory_pressure: snapshot.pressure.memory_pressure,
            thrashing_score: snapshot.pressure.thrashing_score,
            coalition_guard,
            cpu_pegged_fraction,
        });
        // Report outcome to circuit breaker.
        {
            let mut pg = state.policy.lock_recover();
            if out.outcomes.failures == 0 {
                pg.circuit_breaker.record_success();
            } else {
                for _ in 0..out.outcomes.failures {
                    pg.circuit_breaker.record_failure();
                }
            }
        }
        out
    };
    {
        let mut metrics = state.metrics.lock_recover();
        metrics.metrics.privilege_boundary_mode = broker_execution.mode.as_str().to_string();
        metrics.metrics.privileged_action_requests_total = metrics
            .metrics
            .privileged_action_requests_total
            .saturating_add(broker_execution.requests);
        metrics.metrics.privileged_action_rejections_total = metrics
            .metrics
            .privileged_action_rejections_total
            .saturating_add(broker_execution.rejected);
    }
    let mut outcomes = broker_execution.outcomes;
    outcomes
        .decision_events
        .extend(dispatch_decision_events.as_slice().iter().cloned());

    // Update degradation controller with new failure count.
    if outcomes.failures > 0 {
        let mut pg = state.policy.lock_recover();
        let inp = DegradationInputs {
            new_failures: outcomes.failures,
            kernel_task_cpu_pct: 0.0,
            circuit_open: false,
            circuit_open_duration: None,
        };
        pg.degradation.update(&inp);
    }

    // Sync frozen state back and persist if changed.
    {
        let now = Utc::now();
        // Build identity map from this cycle's freeze results so FrozenEntry carries
        // the correct start_sec and original_jetsam_priority captured at SIGSTOP time.
        // [A5/D1 fix] Without this, the normal loop path always stored None for
        // original_jetsam_priority, preventing proper priority restoration on thaw.
        let identity_map: HashMap<u32, (u64, Option<i32>)> = outcomes
            .newly_frozen_identity
            .iter()
            .map(|(pid, start_sec, pri)| (*pid, (*start_sec, *pri)))
            .collect();
        let mut frozen_state = state.frozen_state.lock_recover();
        for pid in &frozen_set {
            frozen_state.entry(*pid).or_insert_with(|| {
                let name = apollo_engine::engine::process_identity::proc_name_for_pid(*pid);
                let (start_sec, original_jetsam_priority) =
                    identity_map.get(pid).copied().unwrap_or_else(|| {
                        let s = apollo_engine::engine::process_identity::ProcessIdentity::from_pid(
                            *pid,
                        )
                        .map(|pi| pi.start_sec)
                        .unwrap_or(0);
                        (s, None)
                    });
                FrozenEntry {
                    frozen_at: now,
                    source: FreezeSource::MainLoop,
                    pressure_at_freeze: snapshot.pressure.memory_pressure,
                    process_name: name,
                    start_sec,
                    original_jetsam_priority,
                }
            });
        }
        frozen_state.retain(|pid, _| frozen_set.contains(pid));
        if frozen_set != frozen_before {
            write_frozen_state(frozen_state_path, &frozen_state);
        }
    }

    DispatchTickOutput {
        outcomes,
        causal_qos_upgrades,
        dedup_stats,
        counterfactual_holdouts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::collector::{CpuStats, MemoryStats, PressureStats};
    use apollo_engine::engine::adaptive_governor::AdaptiveGovernor;
    use apollo_engine::engine::audit_types::DecisionReason;
    use apollo_engine::engine::circuit_breaker::{CircuitBreaker, CircuitState};
    use apollo_engine::engine::daemon_helpers::WakeRuntimeState;
    use apollo_engine::engine::daemon_state::{
        HardwareState, MetricsState, PolicyState, ProcessState, UsageDomainState,
    };
    use apollo_engine::engine::degradation::DegradationController;
    use apollo_engine::engine::mach_qos::MachQoSManager;
    use apollo_engine::engine::policy_store::LearnedPolicy;
    use apollo_engine::engine::sysctl_governor::SysctlGovernorStatus;
    use apollo_engine::engine::types::{
        CapabilityReport, LatencyTarget, OptimizationProfile, RuntimeMetrics,
    };
    use apollo_engine::engine::usage_model::UsageModel;
    use std::collections::{BTreeMap, HashMap, VecDeque};
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};

    fn create_test_state() -> SharedState {
        SharedState {
            policy: Arc::new(Mutex::new(PolicyState {
                profile: OptimizationProfile::BalancedRoot,
                latency_target: LatencyTarget::Normal,
                governor: apollo_engine::engine::profile_governor::ProfileGovernor::new(
                    OptimizationProfile::BalancedRoot,
                ),
                learned_policy: LearnedPolicy::default(),
                learned_policy_path: PathBuf::from("/tmp/apollo_test_lp"),
                feedback_path: PathBuf::from("/tmp/apollo_test_feedback"),
                adaptive_governor: AdaptiveGovernor::new(),
                timeline: std::collections::VecDeque::new(),
                circuit_breaker: CircuitBreaker::default(),
                degradation: DegradationController::default(),
            })),
            metrics: Arc::new(Mutex::new(MetricsState {
                metrics: RuntimeMetrics::default(),
                throttle_level: "balanced".to_string(),
                thermal_state: "nominal".to_string(),
                thermal_level_real: "unknown".to_string(),
                fast_tick_until: None,
                reactor_event_weight: 0.0,
                reactor_status: apollo_engine::engine::daemon_state::ReactorStatus::default(),
                survival_window:
                    apollo_engine::engine::survival_window::SurvivalActivationWindow::new(),
            })),
            frozen_state: Arc::new(Mutex::new(HashMap::new())),
            process: Arc::new(Mutex::new(ProcessState {
                last_blockers: Vec::new(),
                wake_state: WakeRuntimeState {
                    last_cycle_wallclock: chrono::Utc::now(),
                    last_wake_at: None,
                    post_wake_grace_until: None,
                    post_wake_policy: "normal".to_string(),
                    post_wake_reclaim_until: None,
                },
            })),
            stop: Arc::new(AtomicBool::new(false)),
            user_profile_path: PathBuf::from("/tmp/apollo_test_user_profile"),
            usage: Arc::new(Mutex::new(UsageDomainState {
                usage_model: UsageModel::default(),
                usage_model_path: PathBuf::from("/tmp/apollo_test_um"),
                usage_events_path: PathBuf::from("/tmp/apollo_test_ue"),
                usage_tracker: apollo_engine::engine::daemon_state::UsageTrackerState::default(),
            })),
            mach_qos: Arc::new(Mutex::new(MachQoSManager::new())),
            freeze_cooldown: Arc::new(Mutex::new(
                apollo_engine::engine::freeze_cooldown::FreezeCooldown::new(),
            )),
            effect_decay: Arc::new(Mutex::new(
                apollo_engine::engine::effect_decay::DecayWatchdog::new(),
            )),
            hardware: Arc::new(Mutex::new(HardwareState {
                last_hw_snapshot: None,
                sysctl_governor_status: SysctlGovernorStatus {
                    active: false,
                    current_values: HashMap::new(),
                    defaults: HashMap::new(),
                    total_writes: 0,
                    active_tunings: 0,
                    retransmission_rate: 0.0,
                    listen_drop_rate: 0.0,
                    last_tune_secs_ago: HashMap::new(),
                    tcp_consecutive_high: 0,
                    tcp_consecutive_low: 0,
                    tcp_last_scale_up_secs_ago: None,
                    ipc_consecutive_drops: 0,
                    ipc_consecutive_clean: 0,
                    vm_consecutive_high: 0,
                    vm_consecutive_low: 0,
                    fs_consecutive_high: 0,
                    fs_consecutive_low: 0,
                },
            })),
            revert_sysctls_requested: Arc::new(AtomicBool::new(false)),
            cycle_condvar: Arc::new((Mutex::new(false), std::sync::Condvar::new())),
            resource_interrupt: Arc::new(
                apollo_engine::engine::thermal_interrupt::ResourceInterruptState::new(),
            ),
            subscribers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    #[test]
    fn test_dispatch_tick_circuit_open() {
        let state = create_test_state();
        {
            let mut pg = state.policy.lock().unwrap();
            // Record enough failures to trip the circuit breaker.
            for _ in 0..10 {
                pg.circuit_breaker.record_failure();
            }
            assert_eq!(*pg.circuit_breaker.state(), CircuitState::Open);
        }

        let caps = CapabilityReport {
            can_taskpolicy: true,
            can_sysctl: true,
            can_memorystatus: true,
            can_memory_pressure_send: true,
            can_mdutil: true,
            can_tmutil: true,
            is_root: true,
            p_core_count: Some(8),
            e_core_count: Some(4),
            unavailable: Vec::new(),
            memorystatus_probe: None,
            task_for_pid_probe: None,
        };

        let mut unfreeze_decay = UnfreezeDecayModel::new();
        let collector = SystemCollector::new();
        let snapshot = SystemSnapshot {
            timestamp: Utc::now(),
            cpu: CpuStats {
                global_usage: 0.0,
                core_count: 1,
            },
            memory: MemoryStats {
                total_ram: 0,
                used_ram: 0,
                free_ram: 0,
                total_swap: 0,
                used_swap: 0,
            },
            pressure: PressureStats {
                memory_pressure: 0.0,
                swap_used_bytes: 0,
                swap_total_bytes: 0,
                swap_delta_bytes_per_sec: 0.0,
                thermal_level: "nominal".to_string(),
                compressor_pressure: 0.0,
                thrashing_score: 0.0,
                memory_pressure_raw: 0.0,
                refault_delta_per_sec: 0.0,
            },
            disks: Vec::new(),
            networks: Vec::new(),
            top_processes: Vec::new(),
        };
        let causal_qos = HashSet::new();
        let world_model = apollo_engine::engine::world_model::WorldModel::default();

        let input = DispatchTickInput {
            state: &state,
            caps: &caps,
            journal_path: Path::new("/tmp/apollo_test_journal"),
            frozen_state_path: Path::new("/tmp/apollo_test_frozen"),
            final_actions: vec![
                RootAction::throttle(1234, "test", true, "test", DecisionReason::PressureContext),
                RootAction::unfreeze(
                    5678,
                    "test_unfreeze",
                    "test",
                    DecisionReason::PressureContext,
                ),
            ],
            snapshot: &snapshot,
            prev_cog_decision: None,
            causal_qos_names: &causal_qos,
            reclaim_risk: SwapRisk::Safe,
            unfreeze_decay: &mut unfreeze_decay,
            collector: &collector,
            dry_run: true,
            lf_metrics: None,
            coalition_guard: None,
            cpu_pegged_fraction: 0.0,
            world_model: &world_model,
            workload: "idle",
            cycle_count: 1,
        };

        let output = run_dispatch_tick(input);

        // When circuit is open, only unfreeze actions should be dispatched.
        assert_eq!(output.outcomes.unfreezes_applied, 1);
        assert_eq!(output.outcomes.throttles_applied, 0);
    }

    // ── Per-PID dedup unit tests ─────────────────────────────────────────────

    fn sm_status(pid: u32) -> RootAction {
        RootAction::SetMemorystatus {
            pid,
            priority: -1,
            reason: format!("test pid {}", pid),
            decision_reason: DecisionReason::MemoryBudget,
        }
    }

    fn throttle(pid: u32) -> RootAction {
        RootAction::ThrottleProcess {
            pid,
            name: format!("p{}", pid),
            aggressive: false,
            reason: "test".to_string(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }
    }

    #[test]
    fn same_cycle_dedup_returns_noop_receipt_candidates() {
        let action = throttle(41);

        let (kept, stats, dropped) =
            consolidate_actions_per_pid_with_dropped(vec![action.clone(), action]);

        assert_eq!(kept.len(), 1);
        assert_eq!(stats.throttle, 1);
        assert!(matches!(
            dropped.as_slice(),
            [RootAction::ThrottleProcess { pid: 41, .. }]
        ));
    }

    fn freeze(pid: u32) -> RootAction {
        RootAction::FreezeProcess {
            pid,
            name: format!("p{}", pid),
            reason: "test".to_string(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }
    }

    fn boost(pid: u32) -> RootAction {
        RootAction::BoostProcess {
            pid,
            name: format!("p{}", pid),
            reason: "test".to_string(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }
    }

    fn thread_qos(pid: u32, thread_index: u32) -> RootAction {
        RootAction::SetThreadQoS {
            pid,
            name: format!("p{}", pid),
            thread_index,
            tier: "interactive".to_string(),
            reason: "test".to_string(),
            decision_reason: DecisionReason::ThreadQoSRouting,
            affinity_tag: None,
            start_sec: 0,
            start_usec: 0,
        }
    }

    #[test]
    fn consolidate_drops_4x_setmemorystatus_same_pid() {
        // Reproduces prod observation: pid 65808 SetMemorystatus 4× per cycle.
        let actions = vec![
            sm_status(65808),
            sm_status(65808),
            sm_status(65808),
            sm_status(65808),
        ];
        let (out, stats) = consolidate_actions_per_pid(actions);
        assert_eq!(out.len(), 1, "should keep first occurrence only");
        assert_eq!(stats.set_memorystatus, 3, "should drop 3 duplicates");
        assert_eq!(stats.total_dropped(), 3);
    }

    #[test]
    fn consolidate_keeps_distinct_pid_setmemorystatus() {
        let actions = vec![sm_status(100), sm_status(200), sm_status(300)];
        let (out, stats) = consolidate_actions_per_pid(actions);
        assert_eq!(out.len(), 3, "distinct PIDs not deduped");
        assert_eq!(stats.total_dropped(), 0);
    }

    #[test]
    fn consolidate_keeps_throttle_and_setmemorystatus_same_pid() {
        // Different kinds for same PID coexist legitimately.
        let actions = vec![throttle(100), sm_status(100)];
        let (out, stats) = consolidate_actions_per_pid(actions);
        assert_eq!(out.len(), 2, "different kinds for same PID coexist");
        assert_eq!(stats.total_dropped(), 0);
    }

    #[test]
    fn consolidate_drops_mixed_duplicates_per_kind() {
        // 3× SetMemorystatus + 2× Throttle + 1 Freeze for pid 100; 1 Freeze for pid 200.
        let actions = vec![
            sm_status(100),
            sm_status(100),
            sm_status(100),
            throttle(100),
            throttle(100),
            freeze(100),
            freeze(200),
        ];
        let (out, stats) = consolidate_actions_per_pid(actions);
        // Survivors: 1 SM(100), 1 Throttle(100), 1 Freeze(100), 1 Freeze(200) = 4
        assert_eq!(out.len(), 4);
        assert_eq!(stats.set_memorystatus, 2);
        assert_eq!(stats.throttle, 1);
        assert_eq!(stats.freeze, 0);
        assert_eq!(stats.total_dropped(), 3);
    }

    #[test]
    fn consolidate_preserves_action_order() {
        // First occurrence wins — order must be deterministic.
        let actions = vec![sm_status(1), sm_status(2), sm_status(1), sm_status(3)];
        let (out, stats) = consolidate_actions_per_pid(actions);
        assert_eq!(out.len(), 3);
        assert_eq!(stats.set_memorystatus, 1);
        // Verify pid order is 1, 2, 3 (not re-sorted).
        if let RootAction::SetMemorystatus { pid, .. } = &out[0] {
            assert_eq!(*pid, 1);
        } else {
            panic!("expected SetMemorystatus first");
        }
        if let RootAction::SetMemorystatus { pid, .. } = &out[1] {
            assert_eq!(*pid, 2);
        } else {
            panic!("expected SetMemorystatus second");
        }
        if let RootAction::SetMemorystatus { pid, .. } = &out[2] {
            assert_eq!(*pid, 3);
        } else {
            panic!("expected SetMemorystatus third");
        }
    }

    #[test]
    fn consolidate_passes_through_non_pid_actions() {
        // SetSysctl / ToggleSpotlight have no PID — never deduped, always pass through.
        let actions = vec![
            RootAction::ToggleSpotlight {
                enabled: false,
                reason: "test".to_string(),
                decision_reason: DecisionReason::PressureContext,
            },
            RootAction::ToggleSpotlight {
                enabled: false,
                reason: "test2".to_string(),
                decision_reason: DecisionReason::PressureContext,
            },
        ];
        let (out, stats) = consolidate_actions_per_pid(actions);
        assert_eq!(out.len(), 2, "non-PID actions pass through");
        assert_eq!(stats.total_dropped(), 0);
    }

    #[test]
    fn consolidate_scopes_thread_qos_by_thread_index() {
        let actions = vec![thread_qos(100, 1), thread_qos(100, 2), thread_qos(100, 1)];
        let (out, stats) = consolidate_actions_per_pid(actions);
        assert_eq!(out.len(), 2, "distinct threads must both survive");
        assert_eq!(stats.set_thread_qos, 1);
    }

    #[test]
    fn consolidate_prefers_interactive_thread_qos_regardless_of_input_order() {
        for actions in [
            vec![thread_qos(100, 1), {
                let mut action = thread_qos(100, 1);
                if let RootAction::SetThreadQoS { tier, .. } = &mut action {
                    *tier = "background".to_string();
                }
                action
            }],
            vec![
                {
                    let mut action = thread_qos(100, 1);
                    if let RootAction::SetThreadQoS { tier, .. } = &mut action {
                        *tier = "background".to_string();
                    }
                    action
                },
                thread_qos(100, 1),
            ],
        ] {
            let (out, stats) = consolidate_actions_per_pid(actions);
            assert_eq!(out.len(), 1);
            assert_eq!(stats.set_thread_qos, 1);
            assert!(matches!(
                &out[0],
                RootAction::SetThreadQoS { tier, .. } if tier == "interactive"
            ));
        }
    }

    #[test]
    fn record_dedup_drops_publishes_boost_and_thread_qos() {
        let lf = LockFreeMetrics::new();
        let (_, stats) = consolidate_actions_per_pid(vec![
            boost(100),
            boost(100),
            thread_qos(200, 1),
            thread_qos(200, 1),
        ]);

        record_dedup_drops(&lf, &stats);
        let snapshot = lf.snapshot();
        assert_eq!(snapshot.dedup_drops_boost, 1);
        assert_eq!(snapshot.dedup_drops_thread_qos, 1);
    }

    #[test]
    fn recently_applied_records_only_confirmed_mutations() {
        let applied = throttle(100);
        let blocked = boost(200);
        let traces = vec![
            PolicyDecisionTrace {
                t: Utc::now(),
                cycle: 1,
                intended_action: applied,
                decision_reason: DecisionReason::PressureContext,
                applied: true,
                block_reason: None,
                pressure: 0.5,
                swap_gb: 0.0,
                thrashing: 0.0,
            },
            PolicyDecisionTrace {
                t: Utc::now(),
                cycle: 1,
                intended_action: blocked,
                decision_reason: DecisionReason::PressureContext,
                applied: false,
                block_reason: Some(
                    apollo_engine::engine::audit_types::BlockReason::CircuitBreakerActive,
                ),
                pressure: 0.5,
                swap_gb: 0.0,
                thrashing: 0.0,
            },
        ];
        let mut cache = RecentlyApplied::new();

        assert_eq!(record_applied_actions(&traces, &mut cache), 1);
        assert!(cache.is_recent(100, CachedActionKind::Throttle));
        assert!(!cache.is_recent(200, CachedActionKind::Boost));
    }

    #[test]
    fn mature_negative_utility_vetoes_only_discretionary_actions() {
        use apollo_engine::engine::telemetry_medallion::{
            ActionModelStats, HardwareRegime, TelemetryContextSummary, TelemetryMedallionMetrics,
            TrustedTelemetryView,
        };
        use apollo_engine::engine::world_model::WorldModel;

        let installation_id = apollo_engine::engine::installation_identity::InstallationId(1);
        let now_unix = chrono::Utc::now().timestamp();
        let context = TelemetryContextSummary {
            timestamp_unix: now_unix,
            cpu_core_count: 10,
            p_core_count: 4,
            e_core_count: 6,
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        let action_models = [(
            "idle|boost:p100".to_string(),
            ActionModelStats {
                observations: 12,
                effective_observations: 1,
                utility_ema: -0.04,
                evidence_mass: 12.0,
                utility_variance_ema: 0.0001,
                quality_ema: 0.95,
                last_cycle: 100,
                last_observed_unix: now_unix,
                hardware_regime: HardwareRegime {
                    p_core_count: 4,
                    e_core_count: 6,
                    ram_gib: 16,
                },
                installation_id,
                ..ActionModelStats::default()
            },
        )]
        .into_iter()
        .collect();
        let mut model = WorldModel::default();
        model.attach_context(TrustedTelemetryView {
            current: Some(&context),
            installation_id,
            action_models: &action_models,
            action_models_revision: 1,
            controlled_models: &BTreeMap::new(),
            controlled_models_revision: 0,
            episodic_evidence: &VecDeque::new(),
            causal_dynamics: &apollo_engine::engine::causal_dynamics::CausalDynamicsModel::default(
            ),
            causal_dynamics_revision: 0,
            gpu_calibration_models: &BTreeMap::new(),
            gpu_calibration_revision: 0,
            metrics: TelemetryMedallionMetrics {
                bronze_total: 1,
                gold_total: 1,
                local_gold_total: 1,
                ..TelemetryMedallionMetrics::default()
            },
        });

        assert!(matches!(
            evaluate_utility_gate(&boost(100), &model, "idle"),
            UtilityGateDecision::Veto(WorldModelInfluence { action_key, .. })
                if action_key == "boost:p100"
        ));
        assert_eq!(
            evaluate_utility_gate(&throttle(100), &model, "idle"),
            UtilityGateDecision::Admit
        );
        assert!(matches!(
            evaluate_utility_gate(&boost(200), &model, "idle"),
            UtilityGateDecision::Abstained { .. }
        ));
    }

    #[test]
    fn mature_positive_utility_advances_only_accelerator_slots() {
        use apollo_engine::engine::telemetry_medallion::{
            ActionModelStats, HardwareRegime, TelemetryContextSummary, TelemetryMedallionMetrics,
            TrustedTelemetryView,
        };
        use apollo_engine::engine::world_model::WorldModel;

        let installation_id = apollo_engine::engine::installation_identity::InstallationId(1);
        let now_unix = chrono::Utc::now().timestamp();
        let context = TelemetryContextSummary {
            timestamp_unix: now_unix,
            cpu_core_count: 10,
            p_core_count: 4,
            e_core_count: 6,
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        let action_models = [(
            "build|boost:p200".to_string(),
            ActionModelStats {
                observations: 12,
                effective_observations: 10,
                utility_ema: 0.08,
                evidence_mass: 12.0,
                utility_variance_ema: 0.0001,
                quality_ema: 0.95,
                last_cycle: 100,
                last_observed_unix: now_unix,
                hardware_regime: HardwareRegime {
                    p_core_count: 4,
                    e_core_count: 6,
                    ram_gib: 16,
                },
                installation_id,
                ..ActionModelStats::default()
            },
        )]
        .into_iter()
        .collect();
        let mut model = WorldModel::default();
        model.attach_context(TrustedTelemetryView {
            current: Some(&context),
            installation_id,
            action_models: &action_models,
            action_models_revision: 1,
            controlled_models: &BTreeMap::new(),
            controlled_models_revision: 0,
            episodic_evidence: &VecDeque::new(),
            causal_dynamics: &apollo_engine::engine::causal_dynamics::CausalDynamicsModel::default(
            ),
            causal_dynamics_revision: 0,
            gpu_calibration_models: &BTreeMap::new(),
            gpu_calibration_revision: 0,
            metrics: TelemetryMedallionMetrics {
                bronze_total: 1,
                gold_total: 1,
                local_gold_total: 1,
                ..TelemetryMedallionMetrics::default()
            },
        });

        let (actions, report) = plan_action_intents(
            vec![boost(100), freeze(900), boost(200)],
            &model,
            "build",
            0.30,
            false,
            false,
        );

        assert_eq!(report.evidence_ranked, 1);
        assert_eq!(report.reordered, 2);
        assert!(matches!(
            actions[0],
            RootAction::BoostProcess { pid: 200, .. }
        ));
        assert!(matches!(
            actions[1],
            RootAction::FreezeProcess { pid: 900, .. }
        ));
        assert!(matches!(
            actions[2],
            RootAction::BoostProcess { pid: 100, .. }
        ));
    }

    #[test]
    fn exact_control_arm_breaks_accelerator_tie_without_granting_authority() {
        use apollo_engine::engine::telemetry_medallion::{
            ActionModelStats, ControlledCounterfactualStats, HardwareRegime,
            TelemetryContextSummary, TelemetryMedallionMetrics, TrustedTelemetryView,
        };
        use apollo_engine::engine::world_model::WorldModel;

        let installation_id = apollo_engine::engine::installation_identity::InstallationId(1);
        let now_unix = chrono::Utc::now().timestamp();
        let hardware_regime = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let context = TelemetryContextSummary {
            timestamp_unix: now_unix,
            cpu_core_count: 10,
            p_core_count: 4,
            e_core_count: 6,
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        let mature = || ActionModelStats {
            observations: 12,
            effective_observations: 10,
            utility_ema: 0.08,
            evidence_mass: 12.0,
            utility_variance_ema: 0.0001,
            quality_ema: 0.95,
            last_cycle: 100,
            last_observed_unix: now_unix,
            hardware_regime,
            installation_id,
            ..ActionModelStats::default()
        };
        let action_models = BTreeMap::from([
            ("build|boost:p100".to_string(), mature()),
            ("build|boost:p200".to_string(), mature()),
        ]);
        let controlled_models = BTreeMap::from([(
            "build|boost:p200".to_string(),
            ControlledCounterfactualStats {
                observations: 8,
                would_have_helped: 7,
                control_utility_ema: -0.04,
                quality_ema: 0.95,
                last_cycle: 100,
                last_observed_unix: now_unix,
                hardware_regime,
                installation_id,
            },
        )]);
        let mut model = WorldModel::default();
        model.attach_context(TrustedTelemetryView {
            current: Some(&context),
            installation_id,
            action_models: &action_models,
            action_models_revision: 1,
            controlled_models: &controlled_models,
            controlled_models_revision: 1,
            episodic_evidence: &VecDeque::new(),
            causal_dynamics: &apollo_engine::engine::causal_dynamics::CausalDynamicsModel::default(
            ),
            causal_dynamics_revision: 0,
            gpu_calibration_models: &BTreeMap::new(),
            gpu_calibration_revision: 0,
            metrics: TelemetryMedallionMetrics {
                local_gold_total: 1,
                ..TelemetryMedallionMetrics::default()
            },
        });

        let (actions, report) = plan_action_intents(
            vec![boost(100), boost(200)],
            &model,
            "build",
            0.30,
            false,
            false,
        );

        assert_eq!(report.counterfactual_ranked, 1);
        assert_eq!(report.evidence_ranked, 2);
        assert!(matches!(
            actions[0],
            RootAction::BoostProcess { pid: 200, .. }
        ));
    }

    #[test]
    fn contextual_episodes_rank_accelerators_without_authorizing_new_actions() {
        use apollo_engine::engine::telemetry_medallion::{
            ActuatorEpisodeContext, ActuatorFamily, ActuatorObjective, EvidenceTier,
            HardwareRegime, ResolvedActuatorEvidence, TelemetryContextSummary,
            TelemetryMedallionMetrics, TrustedTelemetryView, WorldStateDelta,
        };
        use apollo_engine::engine::world_model::WorldModel;

        let installation_id = apollo_engine::engine::installation_identity::InstallationId(1);
        let now_unix = chrono::Utc::now().timestamp();
        let context = TelemetryContextSummary {
            cycle: 100,
            timestamp_unix: now_unix,
            workload: "build".to_string(),
            cpu_core_count: 10,
            p_core_count: 4,
            e_core_count: 6,
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            fluidity_score: 0.82,
            ..Default::default()
        };
        let hardware = HardwareRegime::from_context(&context);
        let episodes = (1..=2)
            .map(|id| ResolvedActuatorEvidence {
                id,
                decision_id: None,
                family: ActuatorFamily::Boost,
                objective: ActuatorObjective::Responsiveness,
                action_key: "boost:p200".to_string(),
                target: "p200".to_string(),
                workload: "build".to_string(),
                issued_cycle: 90 + id,
                resolved_cycle: 93 + id,
                resolved_timestamp_unix: now_unix - id as i64,
                hardware_regime: hardware,
                installation_id,
                horizon_cycles: 3,
                tier: EvidenceTier::Gold,
                quality: 0.96,
                raw_utility_delta: 0.08,
                counterfactual_delta: 0.0,
                net_utility_delta: 0.08,
                attribution: Default::default(),
                utility: Default::default(),
                perceptual_latency_improvement: 0.0,
                net_state_delta: WorldStateDelta::default(),
                context_before: ActuatorEpisodeContext::from_telemetry(&context),
                effective: true,
                confounder_count: 0,
                target_present_after: Some(true),
            })
            .collect::<VecDeque<_>>();
        let mut model = WorldModel::default();
        model.attach_context(TrustedTelemetryView {
            current: Some(&context),
            installation_id,
            action_models: &BTreeMap::new(),
            action_models_revision: 1,
            controlled_models: &BTreeMap::new(),
            controlled_models_revision: 0,
            episodic_evidence: &episodes,
            causal_dynamics: &apollo_engine::engine::causal_dynamics::CausalDynamicsModel::default(
            ),
            causal_dynamics_revision: 0,
            gpu_calibration_models: &BTreeMap::new(),
            gpu_calibration_revision: 0,
            metrics: TelemetryMedallionMetrics {
                local_gold_total: 2,
                ..TelemetryMedallionMetrics::default()
            },
        });

        let (actions, report) = plan_action_intents(
            vec![boost(100), boost(200)],
            &model,
            "build",
            0.30,
            false,
            false,
        );

        assert_eq!(actions.len(), 2);
        assert!(matches!(
            actions[0],
            RootAction::BoostProcess { pid: 200, .. }
        ));
        assert_eq!(report.episodic_ranked, 2);
        assert_eq!(report.reordered, 2);
    }

    #[test]
    fn mature_positive_utility_reports_promotion_without_reordering() {
        use apollo_engine::engine::telemetry_medallion::{
            ActionModelStats, HardwareRegime, TelemetryContextSummary, TelemetryMedallionMetrics,
            TrustedTelemetryView,
        };
        use apollo_engine::engine::world_model::WorldModel;

        let installation_id = apollo_engine::engine::installation_identity::InstallationId(1);
        let now_unix = chrono::Utc::now().timestamp();
        let context = TelemetryContextSummary {
            timestamp_unix: now_unix,
            cpu_core_count: 10,
            p_core_count: 4,
            e_core_count: 6,
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        let action_models = [(
            "build|boost:p200".to_string(),
            ActionModelStats {
                observations: 12,
                effective_observations: 10,
                utility_ema: 0.08,
                evidence_mass: 12.0,
                utility_variance_ema: 0.0001,
                quality_ema: 0.95,
                last_cycle: 100,
                last_observed_unix: now_unix,
                hardware_regime: HardwareRegime {
                    p_core_count: 4,
                    e_core_count: 6,
                    ram_gib: 16,
                },
                installation_id,
                ..ActionModelStats::default()
            },
        )]
        .into_iter()
        .collect();
        let mut model = WorldModel::default();
        model.attach_context(TrustedTelemetryView {
            current: Some(&context),
            installation_id,
            action_models: &action_models,
            action_models_revision: 1,
            controlled_models: &BTreeMap::new(),
            controlled_models_revision: 0,
            episodic_evidence: &VecDeque::new(),
            causal_dynamics: &apollo_engine::engine::causal_dynamics::CausalDynamicsModel::default(
            ),
            causal_dynamics_revision: 0,
            gpu_calibration_models: &BTreeMap::new(),
            gpu_calibration_revision: 0,
            metrics: TelemetryMedallionMetrics {
                bronze_total: 1,
                gold_total: 1,
                local_gold_total: 1,
                ..TelemetryMedallionMetrics::default()
            },
        });

        let (actions, report) = plan_action_intents(
            vec![boost(200), boost(100)],
            &model,
            "build",
            0.30,
            false,
            false,
        );

        assert_eq!(report.evidence_ranked, 1);
        assert_eq!(report.reordered, 0);
        assert!(matches!(
            actions[0],
            RootAction::BoostProcess { pid: 200, .. }
        ));
    }

    #[test]
    fn temporal_rollout_ranks_accelerators_without_moving_safety_actions() {
        use apollo_engine::engine::telemetry_medallion::{
            ActionModelStats, HardwareRegime, TelemetryContextSummary, TelemetryMedallionMetrics,
            TrustedTelemetryView, WorldStateDelta,
        };
        use apollo_engine::engine::world_model::WorldModel;

        let installation_id = apollo_engine::engine::installation_identity::InstallationId(1);
        let now_unix = chrono::Utc::now().timestamp();
        let hardware_regime = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let action_models = [
            (
                "boost:p100".to_string(),
                ActionModelStats {
                    observations: 4,
                    state_delta_ema: WorldStateDelta::default(),
                    state_evidence_mass: 4.0,
                    quality_ema: 0.95,
                    last_cycle: 6,
                    last_observed_unix: now_unix,
                    hardware_regime,
                    installation_id,
                    ..ActionModelStats::default()
                },
            ),
            (
                "boost:p200".to_string(),
                ActionModelStats {
                    observations: 4,
                    utility_ema: 0.02,
                    state_delta_ema: WorldStateDelta {
                        fluidity: 0.10,
                        ..WorldStateDelta::default()
                    },
                    state_evidence_mass: 4.0,
                    quality_ema: 0.95,
                    last_cycle: 6,
                    last_observed_unix: now_unix,
                    hardware_regime,
                    installation_id,
                    ..ActionModelStats::default()
                },
            ),
            (
                "freeze:p900".to_string(),
                ActionModelStats {
                    observations: 20,
                    utility_ema: 0.90,
                    state_delta_ema: WorldStateDelta {
                        fluidity: 0.90,
                        ..WorldStateDelta::default()
                    },
                    state_evidence_mass: 20.0,
                    quality_ema: 0.99,
                    last_cycle: 6,
                    last_observed_unix: now_unix,
                    hardware_regime,
                    installation_id,
                    ..ActionModelStats::default()
                },
            ),
        ]
        .into_iter()
        .collect();
        let mut model = WorldModel::default();
        for cycle in 1..=6 {
            let context = TelemetryContextSummary {
                cycle,
                timestamp_unix: now_unix - (6 - cycle) as i64,
                workload: "build".to_string(),
                memory_pressure: 0.30,
                fluidity_score: 0.70,
                cpu_core_count: 10,
                p_core_count: 4,
                e_core_count: 6,
                total_ram_bytes: 16 * 1024 * 1024 * 1024,
                ..TelemetryContextSummary::default()
            };
            model.attach_context(TrustedTelemetryView {
                current: Some(&context),
                installation_id,
                action_models: &action_models,
                action_models_revision: 1,
                controlled_models: &BTreeMap::new(),
                controlled_models_revision: 0,
                episodic_evidence: &VecDeque::new(),
                causal_dynamics:
                    &apollo_engine::engine::causal_dynamics::CausalDynamicsModel::default(),
                causal_dynamics_revision: 0,
                gpu_calibration_models: &BTreeMap::new(),
                gpu_calibration_revision: 0,
                metrics: TelemetryMedallionMetrics {
                    bronze_total: cycle,
                    gold_total: cycle,
                    local_gold_total: cycle,
                    ..TelemetryMedallionMetrics::default()
                },
            });
        }

        let (actions, report) = plan_action_intents(
            vec![boost(100), freeze(900), boost(200)],
            &model,
            "build",
            0.30,
            false,
            false,
        );

        assert_eq!(report.temporal_memory_samples, 6);
        assert_eq!(report.temporal_candidates, 2);
        assert!(report.temporal_rollouts >= 2);
        assert!(report.temporal_promotions >= 1);
        assert_eq!(report.reordered, 2);
        assert!(matches!(
            actions[0],
            RootAction::BoostProcess { pid: 200, .. }
        ));
        assert!(matches!(
            actions[1],
            RootAction::FreezeProcess { pid: 900, .. }
        ));
        assert!(matches!(
            actions[2],
            RootAction::BoostProcess { pid: 100, .. }
        ));
    }

    #[test]
    fn controlled_holdout_is_sparse_deterministic_and_health_gated() {
        let selected = (0..CONTROLLED_HOLDOUT_MODULUS)
            .filter(|cycle| controlled_holdout_slot(*cycle, "boost:p200"))
            .count();
        assert_eq!(selected, 1);
        assert!(controlled_holdout_safe(0.30, false, false));
        assert!(!controlled_holdout_safe(0.55, false, false));
        assert!(!controlled_holdout_safe(0.30, true, false));
        assert!(!controlled_holdout_safe(0.30, false, true));
    }

    #[test]
    fn gpu_portfolio_covers_live_specialists_without_root_actions() {
        let runtime = apollo_engine::engine::types::RuntimeMetrics {
            markov_prediction_app: "Editor".to_string(),
            markov_prediction_confidence: 0.80,
            chromium_renderers_total: 6,
            interaction_qos_active: true,
            predictive_agent_active: true,
            ..apollo_engine::engine::types::RuntimeMetrics::default()
        };
        let model = apollo_engine::engine::world_model::WorldModel::default();
        let candidates = build_gpu_candidate_portfolio(&model, "build", &runtime);
        let keys: HashSet<&str> = candidates
            .iter()
            .map(|candidate| candidate.action_key.as_str())
            .collect();

        assert!(keys.contains("markov_prewarm:predicted_app"));
        assert!(keys.contains("interaction_qos:foreground"));
        assert!(keys.contains("io_shaping:interactive_release"));
        assert!(keys.contains("chromium_ecore:background_renderer"));
        assert!(keys.contains("chromium_purge:purgeable_renderer"));
        assert!(keys.contains("chromium_jetsam:background_renderer"));
        assert!(keys.contains("predictive_profile:aggressive"));
        assert!(keys.contains("predictive_purge:kernel"));
        assert_eq!(keys.len(), candidates.len());
    }

    #[test]
    fn decision_attribution_names_owner_and_advisory_gpu_support() {
        let model = apollo_engine::engine::world_model::WorldModel::default();
        let attribution = decision_attribution(
            &boost(100),
            "boost:p100".to_string(),
            &model,
            "build",
            IntentEvidence {
                expected_benefit: 0.04,
                uncertainty: 0.20,
            },
            Some(0.03),
            Some(0.02),
        );

        assert_eq!(attribution.proposer, "pressure-specialist");
        assert_eq!(attribution.predicted_gain, 0.04);
        assert!(attribution
            .supporters
            .contains(&"world-sequence".to_string()));
        assert!(attribution.supporters.contains(&"gpu-model".to_string()));
        assert!(attribution.vetoes.is_empty());
    }
}
