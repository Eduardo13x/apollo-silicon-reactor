//! World Model — Mode-2 imagination before acting.
//!
//! 2026-06-11. Apollo already owned every piece of a LeCun-style world
//! model — CausalGraph carries Gold-only per-action pressure-delta
//! predictions curated by the data medallion, OutcomeTracker carries the Rubin do-nothing
//! counterfactual (`natural_drift_ema`), hazard/Kalman estimate state —
//! but the pieces were scattered and `decide_actions` never ASKED them
//! anything before emitting an action. Apollo acted, then learned; it
//! never imagined first.
//!
//! This module is the missing harness: a per-cycle snapshot facade that
//! answers ONE question — *"if I take this action, does my own learned
//! model predict a better future than doing nothing?"* — and an
//! admission verdict ([`Imagined`]) the decision path can consult.
//!
//! The pressure veto remains deliberately one-step. A separate bounded
//! receding-horizon lane learns joint state transitions and evaluates two-step
//! action sequences for ranking; it cannot manufacture or authorize actions.
//!
//! ## References
//! - [LeCun 2022] "A Path Towards Autonomous Machine Intelligence" §4.2 —
//!   world-model-predictive action selection (MPC over learned model).
//! - [Sutton 1991] Dyna — planning as acting through learned model.
//! - [Rubin 1974] Potential Outcomes — the do-nothing counterfactual is
//!   the control arm every action must beat.
//! - [Camacho 2007] MPC — act only when the predicted trajectory under
//!   action improves on the free response.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::engine::causal_dynamics::{CausalDynamicsMetrics, CausalDynamicsModel};
use crate::engine::causal_graph::CausalGraph;
use crate::engine::gpu_imagination::{GpuCandidateAdvice, GpuImaginationResult};
use crate::engine::installation_identity::InstallationId;
use crate::engine::local_consolidation::LocalConsolidationView;
use crate::engine::outcome_tracker::OutcomeTracker;
use crate::engine::telemetry_medallion::{
    gpu_calibration_key, ActionModelStats, ActuatorEpisodeContext, ControlledCounterfactualStats,
    EvidenceTier, GpuCalibrationStats, ResolvedActuatorEvidence, TelemetryContextSummary,
    TrustedTelemetryView,
};
use crate::engine::world_model_sequence::{
    plan_temporal_sequence_with_dynamics, TemporalMemory, TemporalSequencePlan,
};

/// Minimum causal-edge evidence before a prediction is trusted enough to
/// VETO an action. Below this the model abstains ([`Imagined::Unknown`])
/// — an immature model must never block exploration (the same data-starve
/// guard as the HRPO cold-start admit).
const MIN_EVIDENCE: u32 = 10;

/// Minimum medallion quality for a veto-grade prediction. Effectiveness is
/// represented by predicted drop; epistemic trust comes from data quality and
/// sample maturity, so reliably ineffective actions can still be vetoed.
const MIN_DATA_QUALITY: f32 = 0.90;

/// Dominance margin: the action's predicted drop must beat the natural
/// drift by at least this much pressure to justify the side-effects.
/// 0.005 ≈ half a percent of pressure — below that the action is noise
/// relative to what the system does on its own.
const DOMINANCE_MARGIN: f64 = 0.005;
const MIN_UTILITY_EVIDENCE: f64 = 10.0;
const MIN_UTILITY_DATA_QUALITY: f64 = 0.85;
const UTILITY_CONFIDENCE_Z: f64 = 1.96;
const UTILITY_VARIANCE_FLOOR: f64 = 0.0001;
const UTILITY_MAX_AGE_SECS: i64 = 14 * 24 * 60 * 60;
const MIN_COUNTERFACTUAL_EVIDENCE: u32 = 3;
const MIN_COUNTERFACTUAL_QUALITY: f64 = 0.85;
const COUNTERFACTUAL_MAX_RANK_SUPPORT: f64 = 0.01;
const EPISODIC_MAX_AGE_SECS: i64 = 14 * 24 * 60 * 60;
const EPISODIC_MIN_QUALITY: f64 = 0.85;
const EPISODIC_MIN_SIMILARITY: f64 = 0.55;
const EPISODIC_NEIGHBORS: usize = 8;
const EPISODIC_MAX_RANK_SUPPORT: f64 = 0.012;
const GPU_ADVICE_MAX_AGE_CYCLES: u64 = 30;
/// Keep enough overlapping batches for asynchronous specialists to consume
/// their own advice, without turning the World Model into an unbounded cache.
const MAX_GPU_ADVICE_ENTRIES: usize = 96;
const GPU_COLD_START_TRUST: f64 = 0.25;
const GPU_MAX_RANK_SUPPORT: f64 = 0.005;
type EpisodicNeighbor = (f64, f64, f64, bool, f64);

/// The model's verdict for a candidate action.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Imagined {
    /// The learned model predicts the action beats doing nothing.
    ActWins {
        /// Predicted pressure-drop advantage over the natural drift.
        margin: f64,
    },
    /// The learned model predicts doing nothing is at least as good —
    /// the action's expected effect does not clear the natural drift
    /// plus margin. Acting would be side-effects for nothing.
    DoNothingDominates {
        predicted_drop: f64,
        natural_drift: f64,
    },
    /// Not enough evidence to imagine this action — caller must admit
    /// (exploration produces the evidence the model lacks).
    Unknown,
}

/// Utility-space verdict for actuators whose goal is not solely pressure
/// reduction (boost, QoS, prewarm, sysctl, recovery, and policy actions).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UtilityImagined {
    ActWins { margin: f64 },
    DoNothingDominates { predicted_utility: f64 },
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtilityEvidenceScope {
    Workload,
    Aggregate,
}

impl UtilityEvidenceScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Workload => "workload",
            Self::Aggregate => "aggregate",
        }
    }
}

/// Evidence behind one utility-space verdict. This is intentionally a small
/// value object so callers can explain a real decision without exposing the
/// medallion's private installation identity or copying model state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtilityAssessment {
    pub verdict: UtilityImagined,
    pub scope: UtilityEvidenceScope,
    pub utility_ema: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub effective_evidence: f64,
    pub quality: f64,
    /// Bounded ranking-only support from an exact same-workload control arm.
    /// It never changes the authoritative utility verdict.
    pub counterfactual_support: f64,
    pub counterfactual_observations: u32,
    /// Context-nearest outcomes from any universal actuator family. This is a
    /// ranking hint only and never changes the authoritative utility verdict.
    pub episodic_support: f64,
    pub episodic_observations: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EpisodicRecall {
    pub expected_utility: f64,
    pub rank_support: f64,
    pub observations: u32,
    pub exact_observations: u32,
    pub mean_similarity: f64,
}

/// Bounded, context-sensitive preference for an action already proposed by a
/// specialist. The score is deliberately unitless and capped to `[-1, 1]` so
/// external actuator lanes can map it into their own narrow parameter bands.
/// It never authorizes an action or bypasses a specialist gate.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ContextualActionBias {
    pub score: f64,
    pub model_observations: u32,
    pub episodic_observations: u32,
    pub gpu_predictions: u32,
    /// The bounded portion of `score` contributed by the fresh GPU forecast.
    pub gpu_context_support: f64,
    pub gpu_calibration_trust: f64,
    /// True only when a mature exact/workload utility model contributed.
    pub authoritative: bool,
}

impl ContextualActionBias {
    pub fn is_informative(self) -> bool {
        self.score.abs() > f64::EPSILON
            && (self.model_observations > 0
                || self.episodic_observations > 0
                || self.gpu_predictions > 0)
    }

    /// True when the bounded GPU forecast contributed to this bias. This is
    /// distinct from a generic World Model contribution so consumers can make
    /// the source visible without treating it as action authority.
    pub fn has_gpu_influence(self) -> bool {
        self.gpu_predictions > 0 && self.gpu_context_support.abs() > f64::EPSILON
    }
}

/// Compact evidence package used by the local deliberator. It describes the
/// evidence available to existing decision lanes; it is not a policy
/// recommendation and cannot authorize an action on its own.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeliberationEvidence {
    pub workload: String,
    pub authority_phase: String,
    pub local_gold: u64,
    pub data_quality: f64,
    pub utility_ready_actions: u64,
    pub local_consolidation_confidence: f64,
    pub local_consolidation_families: u32,
    pub gpu_fresh_predictions: u32,
    pub gpu_top_action: String,
    pub gpu_top_context_support: f64,
    pub gpu_top_rank_support: f64,
    pub gpu_calibration_trust: f64,
}

/// Local replacement for the former prompt/JSON teacher. It fuses System 1's
/// self-assessment, Dr Zero calibration, medallion quality and fresh GPU
/// forecasts into a bounded confidence scale for existing World Model advice.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SystemDeliberation {
    pub mode: String,
    pub confidence: f64,
    pub advisory_support_scale: f64,
    pub gpu_support_scale: f64,
    pub system1_struggling: bool,
    pub dr_zero_self_challenge: f64,
    pub evidence: DeliberationEvidence,
}

#[derive(Debug, Clone, PartialEq)]
struct GpuAdviceEntry {
    generation: u64,
    workload: String,
    advice: GpuCandidateAdvice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtilityAbstentionReason {
    NoCurrentGold,
    UnknownAction,
    ImmatureEvidence,
    LowQuality,
    StaleEvidence,
    ForeignInstallation,
    HardwareMismatch,
    UncertainInterval,
}

impl UtilityAbstentionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoCurrentGold => "no_current_gold",
            Self::UnknownAction => "unknown_action",
            Self::ImmatureEvidence => "immature_evidence",
            Self::LowQuality => "low_quality",
            Self::StaleEvidence => "stale_evidence",
            Self::ForeignInstallation => "foreign_installation",
            Self::HardwareMismatch => "hardware_mismatch",
            Self::UncertainInterval => "uncertain_interval",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UtilityAbstention {
    pub reason: UtilityAbstentionReason,
    pub scope: Option<UtilityEvidenceScope>,
}

/// Point-in-time inventory of every exact utility model. Unlike the event
/// abstention counters, this explains the full `ready / known` denominator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UtilityReadinessBreakdown {
    pub known: u64,
    pub ready: u64,
    pub no_current_gold: u64,
    pub immature: u64,
    pub low_quality: u64,
    pub stale: u64,
    pub foreign_installation: u64,
    pub hardware_mismatch: u64,
    pub uncertain_interval: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UtilityAssessmentResult {
    Assessed(UtilityAssessment),
    Abstained(UtilityAbstention),
}

/// Non-authoritative family evidence for ranking a previously unseen target.
/// Family priors can improve exploration order but can never veto an action.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtilityPrior {
    pub utility_ema: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub effective_evidence: f64,
    pub quality: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAuthorityPhase {
    #[default]
    Protected,
    Calibrating,
    Trusted,
    Suspended,
}

impl ModelAuthorityPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Protected => "protected",
            Self::Calibrating => "calibrating",
            Self::Trusted => "trusted",
            Self::Suspended => "suspended",
        }
    }
}

/// Per-cycle snapshot of the learned action-conditioned predictions plus
/// the do-nothing baseline. Built once per decision cycle from the live
/// CausalGraph + OutcomeTracker (O(edges), no allocation per query).
#[derive(Debug, Clone, Default)]
pub struct WorldModel {
    /// `"throttle:Name"` / `"freeze:Name"` → (Gold avg pressure drop,
    /// medallion quality, Gold evidence count).
    predicted: HashMap<String, (f64, f32, u32)>,
    /// Rubin counterfactual: EMA of pressure drift on no-action windows.
    /// Positive = pressure tends to drop by itself.
    pub natural_drift: f64,
    curated_observations: u64,
    contextual_actions: usize,
    mean_data_quality: f64,
    context_bronze: u64,
    context_silver: u64,
    context_gold: u64,
    context_local_gold: u64,
    context_quality: f64,
    latest_context: Option<TelemetryContextSummary>,
    current_installation_id: InstallationId,
    authority_phase: ModelAuthorityPhase,
    /// Action key -> (counterfactual-adjusted utility EMA, data quality,
    /// Gold observations). Populated from the universal actuator medallion.
    utility_predicted: HashMap<String, ActionModelStats>,
    controlled_predicted: HashMap<String, ControlledCounterfactualStats>,
    episodic_evidence: Vec<ResolvedActuatorEvidence>,
    episodic_revision: Option<(usize, u64)>,
    episodic_families: usize,
    causal_revision: Option<u64>,
    causal_workload: String,
    causal_debias_bits: u32,
    utility_revision: Option<u64>,
    controlled_revision: Option<u64>,
    causal_refreshes: u64,
    causal_cache_hits: u64,
    utility_refreshes: u64,
    utility_cache_hits: u64,
    temporal_memory: TemporalMemory,
    causal_dynamics: CausalDynamicsModel,
    causal_dynamics_revision: Option<u64>,
    gpu_advice: HashMap<String, GpuAdviceEntry>,
    gpu_advice_generation: Option<u64>,
    gpu_calibration: HashMap<String, GpuCalibrationStats>,
    gpu_calibration_revision: Option<u64>,
    local_consolidation_confidence: f64,
    local_consolidation_families: u32,
    local_family_scales: HashMap<String, f64>,
    deliberation_advisory_support_scale: f64,
    deliberation_gpu_support_scale: f64,
    deliberation_active: bool,
}

impl WorldModel {
    /// Attach the bounded reflex memory compiled from universal Gold outcomes.
    /// This affects ranking and parameter tuning only; model verdicts and
    /// actuator admission remain owned by their existing safety paths.
    pub fn attach_local_consolidation(&mut self, view: LocalConsolidationView) {
        self.local_consolidation_confidence = view.confidence.clamp(0.0, 1.0);
        self.local_consolidation_families = view.families_with_evidence;
        self.local_family_scales = view
            .family_scales
            .into_iter()
            .map(|(family, scale)| (family, scale.clamp(0.75, 1.0)))
            .collect();
    }

    /// Snapshot the live learned state into a query-cheap model.
    ///
    /// `prediction_debias` is the MetaCognition multiplier for the
    /// CausalGraph subsystem (`subsystem_debias_multiplier`, clamped
    /// [0.25, 1.5] at source) — the calibration loop-closure (87c342f)
    /// and the imagination layer MUST share one belief about how much
    /// the causal predictions over-promise. Without it the world model
    /// imagines through raw avg_delta values the system itself has
    /// measured as ~3x inflated (gap 0.256), making ActWins verdicts
    /// systematically optimistic. Pass 1.0 when meta-cognition is
    /// cold-starting.
    pub fn from_parts(
        causal: &CausalGraph,
        tracker: &OutcomeTracker,
        prediction_debias: f32,
    ) -> Self {
        Self::from_parts_for_workload(causal, tracker, prediction_debias, "any")
    }

    /// Build a model for the current workload. Same-workload Gold evidence is
    /// preferred once mature; otherwise the aggregate Gold stream provides a
    /// conservative fallback. Raw CausalGraph edges never enter this model.
    pub fn from_parts_for_workload(
        causal: &CausalGraph,
        tracker: &OutcomeTracker,
        prediction_debias: f32,
        workload: &str,
    ) -> Self {
        let mut model = Self::default();
        model.refresh_from_parts_for_workload(causal, tracker, prediction_debias, workload);
        model
    }

    /// Refresh the causal facade in place. Rebuild only when Gold evidence,
    /// workload, or calibration changed; natural drift remains live every cycle.
    pub fn refresh_from_parts_for_workload(
        &mut self,
        causal: &CausalGraph,
        tracker: &OutcomeTracker,
        prediction_debias: f32,
        workload: &str,
    ) {
        let debias = if prediction_debias.is_finite() && prediction_debias > 0.0 {
            prediction_debias as f64
        } else {
            1.0
        };
        self.natural_drift = tracker.natural_drift();
        let revision = causal.curated_revision();
        let debias_bits = (debias as f32).to_bits();
        if self.causal_revision == Some(revision)
            && self.causal_workload == workload
            && self.causal_debias_bits == debias_bits
        {
            self.causal_cache_hits = self.causal_cache_hits.saturating_add(1);
            return;
        }

        self.predicted.clear();
        let mut curated_observations = 0_u64;
        let mut contextual_actions = 0_usize;
        let mut quality_weighted_sum = 0.0_f64;
        for (action_key, evidence) in causal.curated_prediction_evidence(workload) {
            curated_observations =
                curated_observations.saturating_add(evidence.aggregate.observations as u64);
            quality_weighted_sum +=
                evidence.aggregate.data_quality as f64 * evidence.aggregate.observations as f64;
            let selected = evidence
                .contextual
                .filter(|stats| stats.observations >= MIN_EVIDENCE)
                .unwrap_or(evidence.aggregate);
            if evidence
                .contextual
                .is_some_and(|stats| stats.observations >= MIN_EVIDENCE)
            {
                contextual_actions += 1;
            }
            self.predicted.insert(
                action_key.to_string(),
                (
                    selected.avg_pressure_drop as f64 * debias,
                    selected.data_quality,
                    selected.observations,
                ),
            );
        }
        self.curated_observations = curated_observations;
        self.contextual_actions = contextual_actions;
        self.mean_data_quality = if curated_observations == 0 {
            0.0
        } else {
            (quality_weighted_sum / curated_observations as f64).clamp(0.0, 1.0)
        };
        self.causal_revision = Some(revision);
        self.causal_workload.clear();
        self.causal_workload.push_str(workload);
        self.causal_debias_bits = debias_bits;
        self.causal_refreshes = self.causal_refreshes.saturating_add(1);
    }

    /// Mode-2 step: imagine the action through the learned model and
    /// compare against the do-nothing counterfactual.
    pub fn imagine(&self, action_key: &str) -> Imagined {
        let Some(&(avg_delta, confidence, evidence)) = self.predicted.get(action_key) else {
            return Imagined::Unknown;
        };
        if evidence < MIN_EVIDENCE || confidence < MIN_DATA_QUALITY {
            return Imagined::Unknown;
        }
        // Both quantities are pressure deltas over the causal evaluation
        // window: avg_delta = drop attributed to the action (effective
        // observations EMA), natural_drift = drop with no action at all.
        let baseline = self.natural_drift.max(0.0);
        if avg_delta > baseline + DOMINANCE_MARGIN {
            Imagined::ActWins {
                margin: avg_delta - baseline,
            }
        } else {
            Imagined::DoNothingDominates {
                predicted_drop: avg_delta,
                natural_drift: baseline,
            }
        }
    }

    /// Number of action keys the model can currently imagine.
    pub fn known_actions(&self) -> usize {
        self.predicted.len()
    }

    pub fn ready_actions(&self) -> usize {
        self.predicted
            .values()
            .filter(|(_, quality, evidence)| {
                *evidence >= MIN_EVIDENCE && *quality >= MIN_DATA_QUALITY
            })
            .count()
    }

    pub fn curated_observations(&self) -> u64 {
        self.curated_observations
    }

    pub fn contextual_actions(&self) -> usize {
        self.contextual_actions
    }

    pub fn mean_data_quality(&self) -> f64 {
        self.mean_data_quality
    }

    /// Attach the current live-system context to the model facade. Context is
    /// available for future sequence imagination, but never changes the
    /// action-causal prediction map or its safety thresholds.
    pub fn attach_context(&mut self, view: TrustedTelemetryView<'_>) {
        let local_gold_total = view.metrics.local_gold_total;
        if self.current_installation_id.is_known()
            && view.installation_id.is_known()
            && self.current_installation_id != view.installation_id
        {
            self.temporal_memory.clear();
            self.episodic_evidence.clear();
            self.episodic_revision = None;
            self.causal_dynamics_revision = None;
            self.gpu_advice.clear();
            self.gpu_advice_generation = None;
            self.gpu_calibration.clear();
            self.gpu_calibration_revision = None;
        }
        self.context_bronze = view.metrics.bronze_total;
        self.context_silver = view.metrics.silver_total;
        self.context_gold = view.metrics.gold_total;
        self.context_local_gold = view.metrics.local_gold_total;
        self.context_quality = view.metrics.mean_quality;
        self.latest_context = view.current.cloned();
        self.current_installation_id = view.installation_id;
        if let Some(context) = view.current {
            self.temporal_memory.observe(context);
        }
        if self.causal_dynamics_revision != Some(view.causal_dynamics_revision) {
            self.causal_dynamics = view.causal_dynamics.clone();
            self.causal_dynamics_revision = Some(view.causal_dynamics_revision);
        }
        if self.utility_revision == Some(view.action_models_revision) {
            self.utility_cache_hits = self.utility_cache_hits.saturating_add(1);
        } else {
            self.utility_predicted.clear();
            self.utility_predicted.extend(
                view.action_models
                    .iter()
                    .map(|(key, stats)| (key.clone(), stats.clone())),
            );
            self.utility_revision = Some(view.action_models_revision);
            self.utility_refreshes = self.utility_refreshes.saturating_add(1);
        }
        let episodic_revision = (
            view.episodic_evidence.len(),
            view.episodic_evidence
                .back()
                .map_or(0, |evidence| evidence.id),
        );
        if self.episodic_revision != Some(episodic_revision) {
            self.episodic_evidence.clear();
            self.episodic_evidence.extend(
                view.episodic_evidence
                    .iter()
                    .filter(|evidence| evidence.tier != EvidenceTier::Bronze)
                    .cloned(),
            );
            self.episodic_families = self
                .episodic_evidence
                .iter()
                .map(|evidence| evidence.family)
                .collect::<HashSet<_>>()
                .len();
            self.episodic_revision = Some(episodic_revision);
        }
        if self.controlled_revision != Some(view.controlled_models_revision) {
            self.controlled_predicted.clear();
            self.controlled_predicted.extend(
                view.controlled_models
                    .iter()
                    .map(|(key, stats)| (key.clone(), stats.clone())),
            );
            self.controlled_revision = Some(view.controlled_models_revision);
        }
        if self.gpu_calibration_revision != Some(view.gpu_calibration_revision) {
            self.gpu_calibration.clear();
            self.gpu_calibration.extend(
                view.gpu_calibration_models
                    .iter()
                    .map(|(key, stats)| (key.clone(), stats.clone())),
            );
            self.gpu_calibration_revision = Some(view.gpu_calibration_revision);
        }
        self.authority_phase = if self.latest_context.is_none() {
            if local_gold_total == 0 {
                ModelAuthorityPhase::Protected
            } else {
                ModelAuthorityPhase::Suspended
            }
        } else if self.utility_ready_actions() == 0 {
            ModelAuthorityPhase::Calibrating
        } else {
            ModelAuthorityPhase::Trusted
        };
    }

    /// Attach one completed GPU batch as short-lived context. Candidates are
    /// still owned by their specialist lanes; this cache cannot manufacture
    /// or authorize an action.
    pub fn attach_gpu_imagination(&mut self, result: &GpuImaginationResult) -> u64 {
        if result.error.is_some()
            || result.workload.is_empty()
            || result.candidates.is_empty()
            || self
                .gpu_advice_generation
                .is_some_and(|generation| result.generation < generation)
        {
            return 0;
        }
        // Batches overlap in time: the next batch often contains a different
        // specialist portfolio. Clearing here used to erase a still-fresh
        // Markov/Chromium/root forecast before its owning lane could read it.
        // Retain only the same short horizon used by the consumer below.
        self.gpu_advice.retain(|_, entry| {
            result.generation.saturating_sub(entry.generation) <= GPU_ADVICE_MAX_AGE_CYCLES
        });
        for advice in &result.candidates {
            if advice.action_key.is_empty()
                || ![
                    advice.expected_gain,
                    advice.uncertainty,
                    advice.mean_gain,
                    advice.p10_gain,
                    advice.positive_probability,
                    advice.rank_support,
                    advice.context_score,
                ]
                .into_iter()
                .all(f64::is_finite)
            {
                continue;
            }
            self.gpu_advice.insert(
                gpu_calibration_key(&advice.action_key, &result.workload),
                GpuAdviceEntry {
                    generation: result.generation,
                    workload: result.workload.clone(),
                    advice: advice.clone(),
                },
            );
        }
        if self.gpu_advice.len() > MAX_GPU_ADVICE_ENTRIES {
            let mut stale_keys: Vec<_> = self
                .gpu_advice
                .iter()
                .map(|(key, entry)| (key.clone(), entry.generation))
                .collect();
            stale_keys
                .sort_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0)));
            for (key, _) in stale_keys
                .into_iter()
                .take(self.gpu_advice.len() - MAX_GPU_ADVICE_ENTRIES)
            {
                self.gpu_advice.remove(&key);
            }
        }
        self.gpu_advice_generation = Some(result.generation);
        self.gpu_advice.len() as u64
    }

    fn gpu_calibration_for(
        &self,
        action_key: &str,
        workload: &str,
    ) -> Option<&GpuCalibrationStats> {
        self.gpu_calibration
            .get(&gpu_calibration_key(action_key, workload))
            .or_else(|| {
                self.gpu_calibration
                    .get(&gpu_calibration_key(action_key, "*"))
            })
    }

    fn gpu_calibration_trust(&self, action_key: &str, workload: &str) -> f64 {
        let Some(context) = self.latest_context.as_ref() else {
            return 0.0;
        };
        let Some(stats) = self.gpu_calibration_for(action_key, workload) else {
            return GPU_COLD_START_TRUST;
        };
        let learned_trust = stats.trust(context, self.current_installation_id);
        let calibration_reliability = (1.0 - stats.absolute_error_ema / 0.25).clamp(0.10, 1.0)
            * (1.0 - stats.brier_ema).clamp(0.10, 1.0).sqrt();
        (GPU_COLD_START_TRUST
            + (1.0 - GPU_COLD_START_TRUST) * learned_trust * calibration_reliability)
            .clamp(GPU_COLD_START_TRUST, 1.0)
    }

    fn fresh_gpu_advice(&self, action_key: &str, workload: &str) -> Option<&GpuAdviceEntry> {
        let context = self.latest_context.as_ref()?;
        let entry = self
            .gpu_advice
            .get(&gpu_calibration_key(action_key, workload))?;
        if entry.workload != workload
            || context.cycle < entry.generation
            || context.cycle.saturating_sub(entry.generation) > GPU_ADVICE_MAX_AGE_CYCLES
        {
            return None;
        }
        Some(entry)
    }

    fn gpu_context_support(&self, action_key: &str, workload: &str) -> Option<(f64, f64)> {
        let context = self.latest_context.as_ref()?;
        let entry = self.fresh_gpu_advice(action_key, workload)?;
        let trust = self.gpu_calibration_trust(action_key, workload);
        let correction = self
            .gpu_calibration_for(action_key, workload)
            .filter(|stats| stats.trust(context, self.current_installation_id) > 0.0)
            .map_or(0.0, |stats| stats.signed_error_ema * 0.10);
        let support = ((entry.advice.context_score + correction).clamp(-0.08, 0.08)
            * trust
            * self.deliberation_gpu_support_scale()
            * self.deliberation_advisory_scale_for(action_key))
        .clamp(-0.08, 0.08);
        Some((support, trust))
    }

    /// Return the calibrated, ranking-only support for a still-fresh GPU
    /// forecast. Callers must already own and admit the action independently.
    pub fn gpu_rank_support_for(&self, action_key: &str, workload: &str) -> Option<f64> {
        let entry = self.fresh_gpu_advice(action_key, workload)?;
        let support =
            self.calibrate_gpu_rank_support(action_key, workload, entry.advice.rank_support);
        (support.abs() > f64::EPSILON).then_some(support)
    }

    /// Return the untouched output of a still-fresh GPU imagination. This is
    /// a read-only decision-time forecast; callers must not treat it as
    /// actuation authority.
    pub fn gpu_forecast_for(
        &self,
        action_key: &str,
        workload: &str,
    ) -> Option<&GpuCandidateAdvice> {
        self.fresh_gpu_advice(action_key, workload)
            .map(|entry| &entry.advice)
    }

    /// Summarize current causal and GPU evidence for local deliberation.
    pub fn deliberation_evidence(&self, workload: &str) -> DeliberationEvidence {
        let mut evidence = DeliberationEvidence {
            workload: workload.to_string(),
            authority_phase: self.authority_phase.as_str().to_string(),
            local_gold: self.context_local_gold,
            data_quality: self.context_quality.clamp(0.0, 1.0),
            utility_ready_actions: self.utility_ready_actions() as u64,
            local_consolidation_confidence: self.local_consolidation_confidence,
            local_consolidation_families: self.local_consolidation_families,
            ..DeliberationEvidence::default()
        };
        let Some(context) = self.latest_context.as_ref() else {
            return evidence;
        };
        let mut candidates: Vec<_> = self
            .gpu_advice
            .iter()
            .filter_map(|(key, entry)| {
                (entry.workload == workload
                    && context.cycle >= entry.generation
                    && context.cycle.saturating_sub(entry.generation) <= GPU_ADVICE_MAX_AGE_CYCLES)
                    .then_some((key, entry))
            })
            .collect();
        candidates.sort_by(|left, right| {
            right
                .1
                .advice
                .context_score
                .abs()
                .total_cmp(&left.1.advice.context_score.abs())
                .then_with(|| left.0.cmp(right.0))
        });
        evidence.gpu_fresh_predictions = candidates.len().min(u32::MAX as usize) as u32;
        if let Some((_, entry)) = candidates.first() {
            evidence.gpu_top_action = entry.advice.action_key.clone();
            if let Some((support, trust)) =
                self.gpu_context_support(&entry.advice.action_key, workload)
            {
                evidence.gpu_top_context_support = support;
                evidence.gpu_calibration_trust = trust;
            }
            evidence.gpu_top_rank_support = self
                .gpu_rank_support_for(&entry.advice.action_key, workload)
                .unwrap_or(0.0);
        }
        evidence
    }

    /// Fuse local model health into every bounded World Model advisory path.
    /// This replaces the former prompt/JSON teacher: it never proposes or
    /// authorizes an action, and cannot change an authoritative model verdict.
    pub fn synthesize_deliberation(
        &mut self,
        workload: &str,
        system1_struggling: bool,
        dr_zero_self_challenge: f64,
    ) -> SystemDeliberation {
        let evidence = self.deliberation_evidence(workload);
        let medallion_maturity = evidence.local_gold as f64 / (evidence.local_gold as f64 + 20.0);
        let gpu_confidence = if evidence.gpu_fresh_predictions > 0 {
            0.50 + evidence.gpu_calibration_trust * 0.50
        } else {
            0.50
        };
        let self_check = 1.0 - dr_zero_self_challenge.clamp(0.0, 1.0);
        let local_confidence = if evidence.local_consolidation_families > 0 {
            evidence.local_consolidation_confidence
        } else {
            0.50
        };
        let confidence = (evidence.data_quality * 0.25
            + medallion_maturity * 0.25
            + self_check * 0.20
            + local_confidence * 0.20
            + gpu_confidence * 0.10)
            * if system1_struggling { 0.85 } else { 1.0 };
        let confidence = confidence.clamp(0.0, 1.0);
        let advisory_support_scale = (0.80 + confidence * 0.20).clamp(0.80, 1.0);
        let gpu_support_scale = (0.35 + confidence * 0.65).clamp(0.35, 1.0);
        self.deliberation_advisory_support_scale = advisory_support_scale;
        self.deliberation_gpu_support_scale = gpu_support_scale;
        self.deliberation_active = true;
        let mode = if evidence.authority_phase == "trusted"
            && evidence.data_quality >= MIN_UTILITY_DATA_QUALITY
            && evidence.local_gold >= MIN_UTILITY_EVIDENCE as u64
            && self_check >= 0.70
        {
            "grounded"
        } else if evidence.local_gold > 0
            || evidence.gpu_fresh_predictions > 0
            || evidence.local_consolidation_families > 0
        {
            "calibrating"
        } else {
            "observing"
        };
        SystemDeliberation {
            mode: mode.to_string(),
            confidence,
            advisory_support_scale,
            gpu_support_scale,
            system1_struggling,
            dr_zero_self_challenge: dr_zero_self_challenge.clamp(0.0, 1.0),
            evidence,
        }
    }

    fn deliberation_gpu_support_scale(&self) -> f64 {
        if self.deliberation_active {
            self.deliberation_gpu_support_scale.clamp(0.35, 1.0)
        } else {
            1.0
        }
    }

    fn deliberation_advisory_scale_for(&self, action_key: &str) -> f64 {
        if !self.deliberation_active {
            return 1.0;
        }
        let global = self.deliberation_advisory_support_scale.clamp(0.80, 1.0);
        let family = action_key
            .split_once(':')
            .map(|(family, _)| family)
            .unwrap_or(action_key);
        let family_scale = self
            .local_family_scales
            .get(family)
            .copied()
            .unwrap_or(1.0)
            .clamp(0.75, 1.0);
        (global * family_scale).clamp(0.60, 1.0)
    }

    /// Calibrate the GPU's tiny central-planner ranking support against
    /// same-hardware Gold outcomes. The result remains ranking-only.
    pub fn calibrate_gpu_rank_support(
        &self,
        action_key: &str,
        workload: &str,
        raw_support: f64,
    ) -> f64 {
        if !raw_support.is_finite() {
            return 0.0;
        }
        (raw_support.clamp(-GPU_MAX_RANK_SUPPORT, GPU_MAX_RANK_SUPPORT)
            * self.gpu_calibration_trust(action_key, workload)
            * self.deliberation_gpu_support_scale()
            * self.deliberation_advisory_scale_for(action_key))
        .clamp(-GPU_MAX_RANK_SUPPORT, GPU_MAX_RANK_SUPPORT)
    }

    pub fn cache_stats(&self) -> (u64, u64, u64, u64) {
        (
            self.causal_refreshes,
            self.causal_cache_hits,
            self.utility_refreshes,
            self.utility_cache_hits,
        )
    }

    /// Imagine a non-pressure actuator in its own utility space. Workload-
    /// specific evidence wins when mature, then action-specific evidence is
    /// used. Family aggregates are deliberately not allowed to veto an unseen
    /// target: a new app or thread must retain an exploration path.
    pub fn imagine_utility(&self, action_key: &str, workload: &str) -> UtilityImagined {
        self.assess_utility(action_key, workload)
            .map(|assessment| assessment.verdict)
            .unwrap_or(UtilityImagined::Unknown)
    }

    pub fn assess_utility(&self, action_key: &str, workload: &str) -> Option<UtilityAssessment> {
        match self.assess_utility_diagnostic(action_key, workload) {
            UtilityAssessmentResult::Assessed(assessment) => Some(assessment),
            UtilityAssessmentResult::Abstained(_) => None,
        }
    }

    pub fn assess_utility_diagnostic(
        &self,
        action_key: &str,
        workload: &str,
    ) -> UtilityAssessmentResult {
        let Some(now_unix) = self.latest_context.as_ref().map(|ctx| ctx.timestamp_unix) else {
            return UtilityAssessmentResult::Abstained(UtilityAbstention {
                reason: UtilityAbstentionReason::NoCurrentGold,
                scope: None,
            });
        };
        let workload_key = format!("{workload}|{action_key}");
        // Prefer same-workload evidence only when it is mature. An immature
        // contextual bucket must not hide a mature aggregate model.
        let candidates = [
            (
                UtilityEvidenceScope::Workload,
                self.utility_predicted.get(&workload_key),
            ),
            (
                UtilityEvidenceScope::Aggregate,
                self.utility_predicted.get(action_key),
            ),
        ];
        let mut last_abstention = None;
        let mut selected = None;
        for (scope, stats) in candidates {
            let Some(stats) = stats else {
                continue;
            };
            match utility_model_status(
                stats,
                now_unix,
                self.latest_context.as_ref(),
                self.current_installation_id,
                true,
            ) {
                Ok(()) => {
                    selected = Some((scope, stats));
                    break;
                }
                Err(reason) => {
                    last_abstention = Some(UtilityAbstention {
                        reason,
                        scope: Some(scope),
                    });
                }
            }
        }
        let Some((scope, stats)) = selected else {
            return UtilityAssessmentResult::Abstained(last_abstention.unwrap_or(
                UtilityAbstention {
                    reason: UtilityAbstentionReason::UnknownAction,
                    scope: None,
                },
            ));
        };
        let effective_evidence = stats.effective_evidence_at(now_unix);
        let standard_error =
            (stats.utility_variance_ema.max(UTILITY_VARIANCE_FLOOR) / effective_evidence).sqrt();
        let lower = stats.utility_ema - UTILITY_CONFIDENCE_Z * standard_error;
        let upper = stats.utility_ema + UTILITY_CONFIDENCE_Z * standard_error;
        let verdict = if lower > DOMINANCE_MARGIN {
            UtilityImagined::ActWins {
                margin: lower - DOMINANCE_MARGIN,
            }
        } else if upper <= 0.0 {
            UtilityImagined::DoNothingDominates {
                predicted_utility: stats.utility_ema,
            }
        } else {
            UtilityImagined::Unknown
        };
        let advisory_scale = self.deliberation_advisory_scale_for(action_key);
        let (counterfactual_support, counterfactual_observations) = self
            .counterfactual_rank_support(action_key, workload, now_unix)
            .unwrap_or((0.0, 0));
        let episodic = self
            .recall_similar_episodes(action_key, workload)
            .unwrap_or_default();
        UtilityAssessmentResult::Assessed(UtilityAssessment {
            verdict,
            scope,
            utility_ema: stats.utility_ema,
            lower_bound: lower,
            upper_bound: upper,
            effective_evidence,
            quality: stats.quality_ema,
            counterfactual_support: counterfactual_support * advisory_scale,
            counterfactual_observations,
            episodic_support: episodic.rank_support * advisory_scale,
            episodic_observations: episodic.observations,
        })
    }

    /// Recall bounded, same-machine outcomes from contexts closest to the
    /// current one. Exact action episodes dominate; same-family episodes are
    /// deliberately shrunk and can only influence planner ordering.
    pub fn recall_similar_episodes(
        &self,
        action_key: &str,
        workload: &str,
    ) -> Option<EpisodicRecall> {
        let current = self.latest_context.as_ref()?;
        let current_episode = ActuatorEpisodeContext::from_telemetry(current);
        if !current_episode.valid || !self.current_installation_id.is_known() {
            return None;
        }
        let family = action_key.split_once(':')?.0;
        let now_unix = current.timestamp_unix;
        let mut neighbors: [Option<EpisodicNeighbor>; EPISODIC_NEIGHBORS] =
            [None; EPISODIC_NEIGHBORS];
        let mut neighbor_len = 0_usize;
        for evidence in &self.episodic_evidence {
            if evidence.tier == EvidenceTier::Bronze
                || evidence.quality < EPISODIC_MIN_QUALITY
                || evidence.confounder_count > 2
                || !evidence.net_utility_delta.is_finite()
                || !evidence.context_before.valid
                || evidence.installation_id != self.current_installation_id
                || !evidence.hardware_regime.matches_context(current)
                || evidence.resolved_timestamp_unix <= 0
                || now_unix < evidence.resolved_timestamp_unix
                || now_unix - evidence.resolved_timestamp_unix > EPISODIC_MAX_AGE_SECS
            {
                continue;
            }
            let exact = evidence.action_key == action_key;
            if !exact && evidence.family.as_str() != family {
                continue;
            }
            let similarity = episode_similarity(current_episode, evidence.context_before);
            if similarity < EPISODIC_MIN_SIMILARITY {
                continue;
            }
            let scope_weight = if exact { 1.0 } else { 0.30 };
            let tier_weight = if evidence.tier == EvidenceTier::Gold {
                1.0
            } else {
                0.35
            };
            let workload_weight = if evidence.workload == workload {
                1.0
            } else {
                0.55
            };
            let age_secs = (now_unix - evidence.resolved_timestamp_unix).max(0) as f64;
            let recency = 0.5_f64.powf(age_secs / (7.0 * 24.0 * 60.0 * 60.0));
            let weight = similarity.powi(2)
                * evidence.quality
                * scope_weight
                * tier_weight
                * workload_weight
                * recency;
            let candidate = (
                similarity,
                weight,
                evidence.net_utility_delta,
                exact,
                tier_weight,
            );
            let index = (0..neighbor_len)
                .find(|&index| neighbors[index].is_some_and(|neighbor| neighbor.1 < weight))
                .unwrap_or(neighbor_len);
            if index < EPISODIC_NEIGHBORS {
                let new_len = neighbor_len.saturating_add(1).min(EPISODIC_NEIGHBORS);
                for destination in (index + 1..new_len).rev() {
                    neighbors[destination] = neighbors[destination - 1];
                }
                neighbors[index] = Some(candidate);
                neighbor_len = new_len;
            }
        }
        if neighbor_len < 2 {
            return None;
        }
        let weight_sum = neighbors
            .iter()
            .take(neighbor_len)
            .flatten()
            .map(|(_, weight, _, _, _)| weight)
            .sum::<f64>();
        if weight_sum <= f64::EPSILON {
            return None;
        }
        let expected_utility = neighbors
            .iter()
            .take(neighbor_len)
            .flatten()
            .map(|(_, weight, utility, _, _)| weight * utility)
            .sum::<f64>()
            / weight_sum;
        let mean_similarity = neighbors
            .iter()
            .take(neighbor_len)
            .flatten()
            .map(|(similarity, weight, _, _, _)| similarity * weight)
            .sum::<f64>()
            / weight_sum;
        let maturity = neighbor_len as f64 / (neighbor_len as f64 + 3.0);
        let exact_observations = neighbors
            .iter()
            .take(neighbor_len)
            .flatten()
            .filter(|(_, _, _, exact, _)| *exact)
            .count() as u32;
        let exact_fraction = exact_observations as f64 / neighbor_len as f64;
        let scope_strength = 0.30 + 0.70 * exact_fraction;
        let tier_strength = neighbors
            .iter()
            .take(neighbor_len)
            .flatten()
            .map(|(_, weight, _, _, tier)| weight * tier)
            .sum::<f64>()
            / weight_sum;
        let rank_support =
            (expected_utility * mean_similarity * maturity * scope_strength * tier_strength)
                .clamp(-EPISODIC_MAX_RANK_SUPPORT, EPISODIC_MAX_RANK_SUPPORT);
        Some(EpisodicRecall {
            expected_utility,
            rank_support,
            observations: neighbor_len as u32,
            exact_observations,
            mean_similarity,
        })
    }

    /// Combine mature utility evidence with local context-nearest episodes
    /// into a small advisory signal. Callers may tune an action they already
    /// admitted, but must retain their physical, safety, and capability gates.
    pub fn contextual_action_bias(&self, action_key: &str, workload: &str) -> ContextualActionBias {
        let mut bias = if let Some(assessment) = self.assess_utility(action_key, workload) {
            let model_signal = match assessment.verdict {
                UtilityImagined::ActWins { margin } => (margin / 0.03).clamp(0.0, 1.0),
                UtilityImagined::DoNothingDominates { predicted_utility } => {
                    (predicted_utility / 0.03).clamp(-1.0, 0.0)
                }
                UtilityImagined::Unknown => 0.0,
            } * self.deliberation_advisory_scale_for(action_key);
            let model_weight = if assessment.verdict == UtilityImagined::Unknown {
                0.0
            } else {
                (assessment.effective_evidence / (assessment.effective_evidence + 10.0))
                    * assessment.quality.clamp(0.0, 1.0)
            };
            let episode_signal =
                (assessment.episodic_support / EPISODIC_MAX_RANK_SUPPORT).clamp(-1.0, 1.0);
            let episode_weight = if assessment.episodic_observations > 0 {
                let observations = f64::from(assessment.episodic_observations);
                0.35 * observations / (observations + 3.0)
            } else {
                0.0
            };
            let total_weight = model_weight + episode_weight;
            let score = if total_weight > f64::EPSILON {
                (model_signal * model_weight + episode_signal * episode_weight) / total_weight
            } else {
                0.0
            };
            ContextualActionBias {
                score: score.clamp(-1.0, 1.0),
                model_observations: assessment
                    .effective_evidence
                    .round()
                    .clamp(0.0, f64::from(u32::MAX)) as u32,
                episodic_observations: assessment.episodic_observations,
                authoritative: assessment.verdict != UtilityImagined::Unknown,
                ..ContextualActionBias::default()
            }
        } else if let Some(episodic) = self.recall_similar_episodes(action_key, workload) {
            ContextualActionBias {
                score: ((episodic.rank_support / EPISODIC_MAX_RANK_SUPPORT)
                    * self.deliberation_advisory_scale_for(action_key))
                .clamp(-1.0, 1.0),
                model_observations: 0,
                episodic_observations: episodic.observations,
                authoritative: false,
                ..ContextualActionBias::default()
            }
        } else {
            ContextualActionBias::default()
        };
        if let Some((support, trust)) = self.gpu_context_support(action_key, workload) {
            bias.score = (bias.score + support).clamp(-1.0, 1.0);
            bias.gpu_predictions = 1;
            bias.gpu_context_support = support;
            bias.gpu_calibration_trust = trust;
        }
        bias
    }

    fn counterfactual_rank_support(
        &self,
        action_key: &str,
        workload: &str,
        now_unix: i64,
    ) -> Option<(f64, u32)> {
        let context = self.latest_context.as_ref()?;
        let stats = self
            .controlled_predicted
            .get(&format!("{workload}|{action_key}"))?;
        if stats.observations < MIN_COUNTERFACTUAL_EVIDENCE
            || stats.quality_ema < MIN_COUNTERFACTUAL_QUALITY
            || stats.last_observed_unix <= 0
            || now_unix < stats.last_observed_unix
            || now_unix - stats.last_observed_unix > UTILITY_MAX_AGE_SECS
            || !self.current_installation_id.is_known()
            || stats.installation_id != self.current_installation_id
            || !stats.hardware_regime.matches_context(context)
        {
            return None;
        }
        let maturity = stats.observations as f64 / (stats.observations as f64 + 4.0);
        let help_rate = stats.would_have_helped as f64 / stats.observations as f64;
        let directional = (help_rate * 2.0 - 1.0) * DOMINANCE_MARGIN;
        let support = ((-stats.control_utility_ema * 0.80 + directional * 0.20) * maturity).clamp(
            -COUNTERFACTUAL_MAX_RANK_SUPPORT,
            COUNTERFACTUAL_MAX_RANK_SUPPORT,
        );
        Some((support, stats.observations))
    }

    /// Return a same-installation, same-hardware family prior for ranking.
    /// Unlike [`Self::assess_utility`], this deliberately accepts a confidence
    /// interval that crosses zero: its width becomes uncertainty, and callers
    /// are forbidden from using the prior as veto authority.
    pub fn assess_family_prior(&self, action_key: &str) -> Option<UtilityPrior> {
        let now_unix = self.latest_context.as_ref()?.timestamp_unix;
        let family = action_key.split_once(':')?.0;
        let stats = self.utility_predicted.get(&format!("{family}:*"))?;
        utility_model_status(
            stats,
            now_unix,
            self.latest_context.as_ref(),
            self.current_installation_id,
            false,
        )
        .ok()?;
        let effective_evidence = stats.effective_evidence_at(now_unix);
        let standard_error =
            (stats.utility_variance_ema.max(UTILITY_VARIANCE_FLOOR) / effective_evidence).sqrt();
        Some(UtilityPrior {
            utility_ema: stats.utility_ema,
            lower_bound: stats.utility_ema - UTILITY_CONFIDENCE_Z * standard_error,
            upper_bound: stats.utility_ema + UTILITY_CONFIDENCE_Z * standard_error,
            effective_evidence,
            quality: stats.quality_ema,
        })
    }

    pub fn utility_known_actions(&self) -> usize {
        self.utility_predicted
            .keys()
            .filter(|key| actionable_utility_key(key))
            .count()
    }

    pub fn utility_ready_actions(&self) -> usize {
        let Some(now_unix) = self.latest_context.as_ref().map(|ctx| ctx.timestamp_unix) else {
            return 0;
        };
        self.utility_predicted
            .iter()
            .filter(|(key, stats)| {
                actionable_utility_key(key)
                    && utility_model_ready(
                        stats,
                        now_unix,
                        self.latest_context.as_ref(),
                        self.current_installation_id,
                    )
            })
            .count()
    }

    pub fn utility_readiness_breakdown(&self) -> UtilityReadinessBreakdown {
        let now_unix = self
            .latest_context
            .as_ref()
            .map(|context| context.timestamp_unix)
            .unwrap_or(0);
        let mut breakdown = UtilityReadinessBreakdown::default();
        for (key, stats) in &self.utility_predicted {
            if !actionable_utility_key(key) {
                continue;
            }
            breakdown.known = breakdown.known.saturating_add(1);
            match utility_model_status(
                stats,
                now_unix,
                self.latest_context.as_ref(),
                self.current_installation_id,
                true,
            ) {
                Ok(()) => breakdown.ready = breakdown.ready.saturating_add(1),
                Err(UtilityAbstentionReason::NoCurrentGold) => {
                    breakdown.no_current_gold = breakdown.no_current_gold.saturating_add(1)
                }
                Err(UtilityAbstentionReason::ImmatureEvidence) => {
                    breakdown.immature = breakdown.immature.saturating_add(1)
                }
                Err(UtilityAbstentionReason::LowQuality) => {
                    breakdown.low_quality = breakdown.low_quality.saturating_add(1)
                }
                Err(UtilityAbstentionReason::StaleEvidence) => {
                    breakdown.stale = breakdown.stale.saturating_add(1)
                }
                Err(UtilityAbstentionReason::ForeignInstallation) => {
                    breakdown.foreign_installation =
                        breakdown.foreign_installation.saturating_add(1)
                }
                Err(UtilityAbstentionReason::HardwareMismatch) => {
                    breakdown.hardware_mismatch = breakdown.hardware_mismatch.saturating_add(1)
                }
                Err(UtilityAbstentionReason::UncertainInterval) => {
                    breakdown.uncertain_interval = breakdown.uncertain_interval.saturating_add(1)
                }
                Err(UtilityAbstentionReason::UnknownAction) => {}
            }
        }
        breakdown
    }

    pub fn utility_known_families(&self) -> usize {
        self.utility_predicted
            .keys()
            .filter(|key| key.ends_with(":*"))
            .count()
    }

    pub fn utility_ready_families(&self) -> usize {
        let Some(now_unix) = self.latest_context.as_ref().map(|ctx| ctx.timestamp_unix) else {
            return 0;
        };
        self.utility_predicted
            .iter()
            .filter(|(key, stats)| {
                key.ends_with(":*")
                    && utility_model_status(
                        stats,
                        now_unix,
                        self.latest_context.as_ref(),
                        self.current_installation_id,
                        false,
                    )
                    .is_ok()
            })
            .count()
    }

    pub fn context_bronze(&self) -> u64 {
        self.context_bronze
    }

    pub fn context_silver(&self) -> u64 {
        self.context_silver
    }

    pub fn context_gold(&self) -> u64 {
        self.context_gold
    }

    pub fn context_quality(&self) -> f64 {
        self.context_quality
    }

    pub fn latest_context(&self) -> Option<&TelemetryContextSummary> {
        self.latest_context.as_ref()
    }

    /// Evaluate a bounded two-step receding-horizon plan over actions already
    /// proposed by specialists. The returned scores are ranking evidence only;
    /// safety admission and root execution remain outside the World Model.
    pub fn plan_temporal_sequence(
        &self,
        action_keys: &[String],
        workload: &str,
    ) -> TemporalSequencePlan {
        plan_temporal_sequence_with_dynamics(
            &self.temporal_memory,
            self.latest_context.as_ref(),
            self.current_installation_id,
            self.authority_phase == ModelAuthorityPhase::Trusted,
            &self.utility_predicted,
            Some(&self.causal_dynamics),
            action_keys,
            workload,
        )
    }

    pub fn temporal_memory_samples(&self) -> usize {
        self.temporal_memory.samples()
    }

    pub fn episodic_memory_samples(&self) -> usize {
        self.episodic_evidence.len()
    }

    pub fn episodic_memory_families(&self) -> usize {
        self.episodic_families
    }

    pub fn causal_dynamics_metrics(&self) -> CausalDynamicsMetrics {
        self.causal_dynamics.metrics(self.latest_context.as_ref())
    }

    pub fn authority_phase(&self) -> ModelAuthorityPhase {
        self.authority_phase
    }

    /// Maximum predicted pressure-drop advantage over the natural drift,
    /// across all action keys the model can currently imagine. Empty model
    /// returns 0.0. Used by the per-cycle telemetry archive (Phase 1.5a,
    /// MLP router unblock) to expose f[12] in the 16-d feature vector.
    /// [LeCun 2022 §4.2] — the regime-level "max predicted gain" the
    /// offline trainer correlates against actual intervention outcomes.
    pub fn max_predicted_margin(&self) -> f64 {
        let baseline = self.natural_drift.max(0.0);
        self.predicted
            .values()
            .filter(|(_, quality, evidence)| {
                *evidence >= MIN_EVIDENCE && *quality >= MIN_DATA_QUALITY
            })
            .map(|(avg_delta, _, _)| (avg_delta - baseline).max(0.0))
            .fold(0.0_f64, f64::max)
    }
}

fn actionable_utility_key(key: &str) -> bool {
    !key.ends_with(":*")
}

fn utility_model_ready(
    stats: &ActionModelStats,
    now_unix: i64,
    context: Option<&TelemetryContextSummary>,
    installation_id: InstallationId,
) -> bool {
    utility_model_status(stats, now_unix, context, installation_id, true).is_ok()
}

fn episode_similarity(left: ActuatorEpisodeContext, right: ActuatorEpisodeContext) -> f64 {
    if !left.valid || !right.valid || !left.is_finite() || !right.is_finite() {
        return 0.0;
    }
    let scalar_distance = [
        (left.memory_pressure, right.memory_pressure, 2.0),
        (left.compressor_pressure, right.compressor_pressure, 1.25),
        (left.thrashing_score, right.thrashing_score, 1.5),
        (left.cpu_global_usage, right.cpu_global_usage, 1.0),
        (left.cpu_max_busy, right.cpu_max_busy, 1.0),
        (left.cpu_pegged_fraction, right.cpu_pegged_fraction, 1.25),
        (left.stall_fraction, right.stall_fraction, 1.25),
        (left.used_ram_fraction, right.used_ram_fraction, 0.75),
        (left.thermal_score, right.thermal_score, 1.0),
        (left.fluidity_score, right.fluidity_score, 2.0),
        (
            left.windowserver_cpu_fraction,
            right.windowserver_cpu_fraction,
            1.5,
        ),
        (left.arousal_level, right.arousal_level, 0.75),
        (
            left.markov_prediction_confidence,
            right.markov_prediction_confidence,
            0.75,
        ),
        (
            left.network_retransmit_fraction,
            right.network_retransmit_fraction,
            0.75,
        ),
        (left.network_drop_rate, right.network_drop_rate, 0.75),
        (
            left.package_power_fraction,
            right.package_power_fraction,
            0.75,
        ),
        (left.p_cluster_util, right.p_cluster_util, 0.75),
        (left.e_cluster_util, right.e_cluster_util, 0.75),
        (left.ane_util_fraction, right.ane_util_fraction, 0.50),
        (left.user_idle_fraction, right.user_idle_fraction, 0.75),
    ];
    let mut distance = 0.0;
    let mut weight = 0.0;
    for (a, b, dimension_weight) in scalar_distance {
        distance += (a - b).abs().min(1.0) * dimension_weight;
        weight += dimension_weight;
    }
    for (a, b, dimension_weight) in [
        (left.app_launching, right.app_launching, 0.75),
        (left.window_op_active, right.window_op_active, 0.75),
        (left.foreground_idle, right.foreground_idle, 0.50),
        (
            left.user_call_in_progress,
            right.user_call_in_progress,
            0.75,
        ),
        (left.user_audio_active, right.user_audio_active, 0.50),
        (
            left.coreaudio_direct_probe_available,
            right.coreaudio_direct_probe_available,
            0.10,
        ),
        (
            left.coreaudio_session_fallback,
            right.coreaudio_session_fallback,
            0.10,
        ),
        (
            left.markov_prewarm_active,
            right.markov_prewarm_active,
            0.50,
        ),
        (
            left.predictive_agent_active,
            right.predictive_agent_active,
            0.50,
        ),
    ] {
        distance += f64::from(a != b) * dimension_weight;
        weight += dimension_weight;
    }
    for (a, b, dimension_weight) in [
        (left.foreground_app_hash, right.foreground_app_hash, 1.25),
        (
            left.effective_profile_hash,
            right.effective_profile_hash,
            0.75,
        ),
    ] {
        if a != 0 && b != 0 {
            distance += f64::from(a != b) * dimension_weight;
            weight += dimension_weight;
        }
    }
    (1.0 - distance / weight.max(f64::EPSILON)).clamp(0.0, 1.0)
}

fn utility_model_status(
    stats: &ActionModelStats,
    now_unix: i64,
    context: Option<&TelemetryContextSummary>,
    installation_id: InstallationId,
    require_decisive_interval: bool,
) -> Result<(), UtilityAbstentionReason> {
    let Some(context) = context else {
        return Err(UtilityAbstentionReason::NoCurrentGold);
    };
    if stats.effective_evidence_at(now_unix) < MIN_UTILITY_EVIDENCE {
        return Err(UtilityAbstentionReason::ImmatureEvidence);
    }
    if stats.quality_ema < MIN_UTILITY_DATA_QUALITY {
        return Err(UtilityAbstentionReason::LowQuality);
    }
    if stats.last_observed_unix <= 0
        || now_unix < stats.last_observed_unix
        || now_unix - stats.last_observed_unix > UTILITY_MAX_AGE_SECS
    {
        return Err(UtilityAbstentionReason::StaleEvidence);
    }
    if !installation_id.is_known() || stats.installation_id != installation_id {
        return Err(UtilityAbstentionReason::ForeignInstallation);
    }
    if !stats.hardware_regime.matches_context(context) {
        return Err(UtilityAbstentionReason::HardwareMismatch);
    }
    if !require_decisive_interval {
        return Ok(());
    }
    let evidence = stats.effective_evidence_at(now_unix);
    let standard_error = (stats.utility_variance_ema.max(UTILITY_VARIANCE_FLOOR) / evidence).sqrt();
    let lower = stats.utility_ema - UTILITY_CONFIDENCE_Z * standard_error;
    let upper = stats.utility_ema + UTILITY_CONFIDENCE_Z * standard_error;
    if lower > DOMINANCE_MARGIN || upper <= 0.0 {
        Ok(())
    } else {
        Err(UtilityAbstentionReason::UncertainInterval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::installation_identity::InstallationId;
    use crate::engine::telemetry_medallion::{
        ActuatorFamily, ActuatorObjective, HardwareRegime, TelemetryMedallion,
        TelemetryMedallionMetrics, TrustedTelemetryView,
    };
    use chrono::Utc;
    use std::collections::{BTreeMap, VecDeque};
    use std::time::{Duration, Instant};

    const LOCAL_ID: InstallationId = InstallationId(0x1020_3040_5060_7080);

    fn m4_context(now_unix: i64) -> TelemetryContextSummary {
        TelemetryContextSummary {
            cycle: 100,
            timestamp_unix: now_unix,
            workload: "build".to_string(),
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            cpu_core_count: 10,
            p_core_count: 4,
            e_core_count: 6,
            reactor_healthy: true,
            collector_pressure_alive: true,
            ..TelemetryContextSummary::default()
        }
    }

    #[test]
    fn coreaudio_provenance_gently_discounts_cross_session_evidence() {
        let direct = ActuatorEpisodeContext {
            valid: true,
            coreaudio_direct_probe_available: true,
            ..ActuatorEpisodeContext::default()
        };
        let fallback = ActuatorEpisodeContext {
            valid: true,
            coreaudio_session_fallback: true,
            ..ActuatorEpisodeContext::default()
        };

        let similarity = episode_similarity(direct, fallback);
        assert!(similarity < 1.0);
        assert!(similarity > 0.98, "provenance must remain a weak feature");
    }

    fn mature_model(now_unix: i64, installation_id: InstallationId) -> ActionModelStats {
        ActionModelStats {
            observations: 20,
            effective_observations: 18,
            utility_ema: 0.08,
            evidence_mass: 20.0,
            utility_variance_ema: 0.0001,
            state_delta_ema: Default::default(),
            state_variance_ema: Default::default(),
            state_evidence_mass: 0.0,
            quality_ema: 0.95,
            last_cycle: 100,
            last_observed_unix: now_unix,
            hardware_regime: HardwareRegime {
                p_core_count: 4,
                e_core_count: 6,
                ram_gib: 16,
            },
            installation_id,
        }
    }

    fn episode(
        id: u64,
        family: ActuatorFamily,
        action_key: &str,
        utility: f64,
        context: &TelemetryContextSummary,
        installation_id: InstallationId,
    ) -> ResolvedActuatorEvidence {
        ResolvedActuatorEvidence {
            id,
            decision_id: None,
            family,
            objective: ActuatorObjective::Responsiveness,
            action_key: action_key.to_string(),
            target: action_key.to_string(),
            workload: context.workload.clone(),
            issued_cycle: context.cycle.saturating_sub(3),
            resolved_cycle: context.cycle,
            resolved_timestamp_unix: context.timestamp_unix.saturating_sub(1),
            hardware_regime: HardwareRegime::from_context(context),
            installation_id,
            horizon_cycles: 3,
            tier: EvidenceTier::Gold,
            quality: 0.96,
            raw_utility_delta: utility,
            counterfactual_delta: 0.0,
            net_utility_delta: utility,
            attribution: Default::default(),
            calibration_provenance: Default::default(),
            utility: Default::default(),
            perceptual_latency_improvement: 0.0,
            net_state_delta: Default::default(),
            context_before: ActuatorEpisodeContext::from_telemetry(context),
            effective: utility > 0.0,
            confounder_count: 0,
            target_present_after: Some(true),
        }
    }

    fn attach_view(
        model: &mut WorldModel,
        context: Option<&TelemetryContextSummary>,
        models: &BTreeMap<String, ActionModelStats>,
        local_gold_total: u64,
    ) {
        model.attach_context(TrustedTelemetryView {
            current: context,
            installation_id: LOCAL_ID,
            action_models: models,
            action_models_revision: 1,
            controlled_models: &BTreeMap::new(),
            controlled_models_revision: 0,
            episodic_evidence: &VecDeque::new(),
            causal_dynamics: &CausalDynamicsModel::default(),
            causal_dynamics_revision: 0,
            gpu_calibration_models: &BTreeMap::new(),
            gpu_calibration_revision: 0,
            metrics: TelemetryMedallionMetrics {
                bronze_total: local_gold_total,
                gold_total: local_gold_total,
                local_gold_total,
                ..TelemetryMedallionMetrics::default()
            },
        });
    }

    #[test]
    fn fresh_gpu_advice_is_bounded_and_never_authoritative_by_itself() {
        use crate::engine::gpu_imagination::{
            GpuCandidateAdvice, GpuImaginationBackend, GpuImaginationResult,
        };

        let context = m4_context(Utc::now().timestamp());
        let mut model = WorldModel::default();
        attach_view(&mut model, Some(&context), &BTreeMap::new(), 1);
        model.gpu_calibration.insert(
            gpu_calibration_key("markov_prewarm:predicted_app", "build"),
            GpuCalibrationStats {
                predictions: 20,
                used: 20,
                resolved: 20,
                gold: 20,
                signed_error_ema: -1.0,
                absolute_error_ema: 1.0,
                brier_ema: 1.0,
                p10_coverage_ema: 0.0,
                quality_ema: 1.0,
                evidence_mass: 20.0,
                last_cycle: 99,
                last_observed_unix: context.timestamp_unix,
                hardware_regime: HardwareRegime {
                    p_core_count: 4,
                    e_core_count: 4,
                    ram_gib: 8,
                },
                installation_id: InstallationId(99),
            },
        );
        assert_eq!(
            model.attach_gpu_imagination(&GpuImaginationResult {
                generation: 100,
                workload: "build".to_string(),
                backend: GpuImaginationBackend::Metal,
                device_name: "Apple M4".to_string(),
                samples: 4_096,
                gpu_time_ns: 8_000,
                wall_time_ns: 20_000,
                candidates: vec![GpuCandidateAdvice {
                    action_key: "markov_prewarm:predicted_app".to_string(),
                    expected_gain: 0.04,
                    uncertainty: 0.20,
                    mean_gain: 0.035,
                    p10_gain: 0.01,
                    positive_probability: 0.80,
                    rank_support: 0.003,
                    context_score: 0.08,
                }],
                error: None,
            }),
            1
        );

        let bias = model.contextual_action_bias("markov_prewarm:predicted_app", "build");
        assert!(bias.score > 0.0);
        assert!(bias.score <= 0.08);
        assert_eq!(bias.gpu_predictions, 1);
        assert_eq!(bias.gpu_calibration_trust, GPU_COLD_START_TRUST);
        assert!(!bias.authoritative);
    }

    #[test]
    fn fresh_gpu_batches_coexist_until_their_short_advice_window_expires() {
        use crate::engine::gpu_imagination::{
            GpuCandidateAdvice, GpuImaginationBackend, GpuImaginationResult,
        };

        let context = m4_context(Utc::now().timestamp());
        let mut model = WorldModel::default();
        attach_view(&mut model, Some(&context), &BTreeMap::new(), 3);
        let result = |generation: u64, action_key: &str, context_score: f64| GpuImaginationResult {
            generation,
            workload: "build".to_string(),
            backend: GpuImaginationBackend::Metal,
            device_name: "Apple M4".to_string(),
            samples: 4_096,
            gpu_time_ns: 8_000,
            wall_time_ns: 20_000,
            candidates: vec![GpuCandidateAdvice {
                action_key: action_key.to_string(),
                expected_gain: 0.04,
                uncertainty: 0.20,
                mean_gain: 0.035,
                p10_gain: 0.01,
                positive_probability: 0.80,
                rank_support: 0.003,
                context_score,
            }],
            error: None,
        };

        model.attach_gpu_imagination(&result(98, "markov_prewarm:predicted_app", 0.06));
        model.attach_gpu_imagination(&result(99, "chromium_ecore:background_renderer", 0.03));

        let markov = model.contextual_action_bias("markov_prewarm:predicted_app", "build");
        assert!(markov.has_gpu_influence());
        assert!(model
            .gpu_rank_support_for("markov_prewarm:predicted_app", "build")
            .is_some());
        let evidence = model.deliberation_evidence("build");
        assert_eq!(evidence.gpu_fresh_predictions, 2);
        assert_eq!(evidence.gpu_top_action, "markov_prewarm:predicted_app");
        assert!(evidence.gpu_top_context_support > 0.0);
        assert!(evidence.gpu_top_rank_support > 0.0);
        assert_eq!(evidence.local_gold, 3);
    }

    #[test]
    fn local_deliberation_scales_gpu_advice_without_authorizing_an_action() {
        let context = m4_context(Utc::now().timestamp());
        let mut model = WorldModel::default();
        attach_view(&mut model, Some(&context), &BTreeMap::new(), 2);

        let deliberation = model.synthesize_deliberation("build", true, 0.80);

        assert_eq!(deliberation.mode, "calibrating");
        assert!(deliberation.system1_struggling);
        assert_eq!(deliberation.dr_zero_self_challenge, 0.80);
        assert!(deliberation.confidence > 0.0);
        assert!(deliberation.gpu_support_scale >= 0.35);
        assert!(deliberation.gpu_support_scale < 1.0);
        assert_eq!(model.authority_phase(), ModelAuthorityPhase::Calibrating);
    }

    #[test]
    fn world_model_abstains_without_current_gold_even_with_mature_models() {
        let now = Utc::now().timestamp();
        let models = BTreeMap::from([("boost:Editor".to_string(), mature_model(now, LOCAL_ID))]);
        let mut model = WorldModel::default();
        attach_view(&mut model, None, &models, 1);
        assert_eq!(model.authority_phase(), ModelAuthorityPhase::Suspended);
        assert_eq!(model.utility_ready_actions(), 0);
        assert_eq!(
            model.imagine_utility("boost:Editor", "build"),
            UtilityImagined::Unknown
        );
    }

    #[test]
    fn authority_progresses_from_protected_to_calibrating_to_trusted() {
        let now = Utc::now().timestamp();
        let context = m4_context(now);
        let empty = BTreeMap::new();
        let mut model = WorldModel::default();
        attach_view(&mut model, None, &empty, 0);
        assert_eq!(model.authority_phase(), ModelAuthorityPhase::Protected);

        attach_view(&mut model, Some(&context), &empty, 1);
        assert_eq!(model.authority_phase(), ModelAuthorityPhase::Calibrating);

        let models = BTreeMap::from([("boost:Editor".to_string(), mature_model(now, LOCAL_ID))]);
        model.attach_context(TrustedTelemetryView {
            current: Some(&context),
            installation_id: LOCAL_ID,
            action_models: &models,
            action_models_revision: 2,
            controlled_models: &BTreeMap::new(),
            controlled_models_revision: 0,
            episodic_evidence: &VecDeque::new(),
            causal_dynamics: &CausalDynamicsModel::default(),
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
        assert_eq!(model.authority_phase(), ModelAuthorityPhase::Trusted);
    }

    #[test]
    fn episodic_recall_is_contextual_and_universal_across_actuator_families() {
        let now = Utc::now().timestamp();
        let context = m4_context(now);
        let mut episodes = VecDeque::from([
            episode(
                1,
                ActuatorFamily::InteractionQos,
                "interaction_qos:foreground",
                0.04,
                &context,
                LOCAL_ID,
            ),
            episode(
                2,
                ActuatorFamily::InteractionQos,
                "interaction_qos:foreground",
                0.06,
                &context,
                LOCAL_ID,
            ),
            episode(
                3,
                ActuatorFamily::InteractionQos,
                "interaction_qos:other",
                0.03,
                &context,
                LOCAL_ID,
            ),
            episode(
                4,
                ActuatorFamily::MarkovPrewarm,
                "markov_prewarm:predicted_app",
                -0.08,
                &context,
                LOCAL_ID,
            ),
            episode(
                5,
                ActuatorFamily::MarkovPrewarm,
                "markov_prewarm:predicted_app",
                -0.06,
                &context,
                LOCAL_ID,
            ),
        ]);
        for id in 6..=7 {
            let mut io = episode(
                id,
                ActuatorFamily::IoShaping,
                "io_shaping:foreground",
                0.04,
                &context,
                LOCAL_ID,
            );
            io.tier = EvidenceTier::Silver;
            io.confounder_count = 1;
            episodes.push_back(io);
        }
        let mut model = WorldModel::default();
        model.attach_context(TrustedTelemetryView {
            current: Some(&context),
            installation_id: LOCAL_ID,
            action_models: &BTreeMap::new(),
            action_models_revision: 1,
            controlled_models: &BTreeMap::new(),
            controlled_models_revision: 0,
            episodic_evidence: &episodes,
            causal_dynamics: &CausalDynamicsModel::default(),
            causal_dynamics_revision: 0,
            gpu_calibration_models: &BTreeMap::new(),
            gpu_calibration_revision: 0,
            metrics: TelemetryMedallionMetrics {
                local_gold_total: episodes.len() as u64,
                ..TelemetryMedallionMetrics::default()
            },
        });

        let qos = model
            .recall_similar_episodes("interaction_qos:foreground", "build")
            .expect("QoS episodes");
        assert_eq!(qos.observations, 3);
        assert_eq!(qos.exact_observations, 2);
        assert!(qos.expected_utility > 0.0);
        assert!(qos.rank_support > 0.0);
        assert!(qos.rank_support <= EPISODIC_MAX_RANK_SUPPORT);

        let prewarm = model
            .recall_similar_episodes("markov_prewarm:predicted_app", "build")
            .expect("prewarm episodes");
        assert_eq!(prewarm.observations, 2);
        assert!(prewarm.expected_utility < 0.0);
        assert!(prewarm.rank_support < 0.0);

        let io = model
            .recall_similar_episodes("io_shaping:foreground", "build")
            .expect("high-quality Silver I/O episodes");
        assert_eq!(io.observations, 2);
        assert!(io.rank_support > 0.0);
        assert!(io.rank_support < qos.rank_support);
    }

    #[test]
    fn contextual_bias_combines_mature_models_and_episode_only_advice() {
        let now = Utc::now().timestamp();
        let context = m4_context(now);
        let mut negative = mature_model(now, LOCAL_ID);
        negative.utility_ema = -0.08;
        let models = BTreeMap::from([
            (
                "interaction_qos:foreground".to_string(),
                mature_model(now, LOCAL_ID),
            ),
            ("markov_prewarm:predicted_app".to_string(), negative),
        ]);
        let episodes = VecDeque::from([
            episode(
                1,
                ActuatorFamily::IoShaping,
                "io_shaping:interactive_release",
                0.04,
                &context,
                LOCAL_ID,
            ),
            episode(
                2,
                ActuatorFamily::IoShaping,
                "io_shaping:interactive_release",
                0.05,
                &context,
                LOCAL_ID,
            ),
        ]);
        let mut model = WorldModel::default();
        model.attach_context(TrustedTelemetryView {
            current: Some(&context),
            installation_id: LOCAL_ID,
            action_models: &models,
            action_models_revision: 1,
            controlled_models: &BTreeMap::new(),
            controlled_models_revision: 0,
            episodic_evidence: &episodes,
            causal_dynamics: &CausalDynamicsModel::default(),
            causal_dynamics_revision: 0,
            gpu_calibration_models: &BTreeMap::new(),
            gpu_calibration_revision: 0,
            metrics: TelemetryMedallionMetrics {
                local_gold_total: 22,
                ..TelemetryMedallionMetrics::default()
            },
        });

        let qos = model.contextual_action_bias("interaction_qos:foreground", "build");
        assert!(qos.authoritative);
        assert!(qos.model_observations >= 10);
        assert!(qos.score > 0.0);

        let markov = model.contextual_action_bias("markov_prewarm:predicted_app", "build");
        assert!(markov.authoritative);
        assert!(markov.score < 0.0);

        let io = model.contextual_action_bias("io_shaping:interactive_release", "build");
        assert!(!io.authoritative);
        assert_eq!(io.model_observations, 0);
        assert_eq!(io.episodic_observations, 2);
        assert!(io.score > 0.0);

        assert_eq!(
            model.contextual_action_bias("predictive_profile:aggressive", "build"),
            ContextualActionBias::default()
        );
    }

    #[test]
    fn episodic_recall_rejects_foreign_and_dissimilar_state() {
        let now = Utc::now().timestamp();
        let stored = m4_context(now);
        let mut current = m4_context(now);
        current.memory_pressure = 1.0;
        current.compressor_pressure = 1.0;
        current.thrashing_score = 1.0;
        current.cpu_global_usage = 1.0;
        current.cpu_max_busy = 1.0;
        current.cpu_pegged_fraction = 1.0;
        current.stall_fraction = 1.0;
        current.used_ram_fraction = 1.0;
        current.thermal_score = 1.0;
        current.fluidity_score = 1.0;
        current.windowserver_cpu_fraction = 1.0;
        current.arousal_level = 1.0;
        current.markov_prediction_confidence = 1.0;
        current.app_launching = true;
        current.window_op_active = true;
        let episodes = VecDeque::from([
            episode(
                1,
                ActuatorFamily::IoShaping,
                "io_shaping:foreground",
                1.0,
                &stored,
                LOCAL_ID,
            ),
            episode(
                2,
                ActuatorFamily::IoShaping,
                "io_shaping:foreground",
                1.0,
                &stored,
                InstallationId(99),
            ),
        ]);
        let mut model = WorldModel::default();
        model.attach_context(TrustedTelemetryView {
            current: Some(&current),
            installation_id: LOCAL_ID,
            action_models: &BTreeMap::new(),
            action_models_revision: 1,
            controlled_models: &BTreeMap::new(),
            controlled_models_revision: 0,
            episodic_evidence: &episodes,
            causal_dynamics: &CausalDynamicsModel::default(),
            causal_dynamics_revision: 0,
            gpu_calibration_models: &BTreeMap::new(),
            gpu_calibration_revision: 0,
            metrics: TelemetryMedallionMetrics {
                local_gold_total: 2,
                ..TelemetryMedallionMetrics::default()
            },
        });

        assert!(model
            .recall_similar_episodes("io_shaping:foreground", "build")
            .is_none());
    }

    #[test]
    fn stale_variance_and_origin_change_revoke_trust() {
        let now = Utc::now().timestamp();
        let context = m4_context(now);
        let mut stale = mature_model(now, LOCAL_ID);
        stale.last_observed_unix = now - UTILITY_MAX_AGE_SECS - 1;
        let mut uncertain = mature_model(now, LOCAL_ID);
        uncertain.utility_variance_ema = 1.0;
        let foreign = mature_model(now, InstallationId(99));
        for stats in [stale, uncertain, foreign] {
            let models = BTreeMap::from([("boost:Editor".to_string(), stats)]);
            let mut model = WorldModel::default();
            attach_view(&mut model, Some(&context), &models, 1);
            assert_ne!(model.authority_phase(), ModelAuthorityPhase::Trusted);
            assert_eq!(model.utility_ready_actions(), 0);
        }
    }

    #[test]
    fn readiness_inventory_accounts_for_every_known_utility_model() {
        let now = Utc::now().timestamp();
        let context = m4_context(now);
        let mut ready = mature_model(now, LOCAL_ID);
        ready.utility_ema = 0.12;
        let mut immature = mature_model(now, LOCAL_ID);
        immature.evidence_mass = 2.0;
        let mut low_quality = mature_model(now, LOCAL_ID);
        low_quality.quality_ema = 0.40;
        let mut stale = mature_model(now, LOCAL_ID);
        stale.observations = 256;
        stale.evidence_mass = 256.0;
        stale.last_observed_unix = now - UTILITY_MAX_AGE_SECS - 1;
        let foreign = mature_model(now, InstallationId(99));
        let mut hardware = mature_model(now, LOCAL_ID);
        hardware.hardware_regime = HardwareRegime {
            p_core_count: 8,
            e_core_count: 2,
            ram_gib: 64,
        };
        let mut uncertain = mature_model(now, LOCAL_ID);
        uncertain.utility_ema = 0.0;
        uncertain.utility_variance_ema = 1.0;
        let models = BTreeMap::from([
            ("boost:Ready".to_string(), ready),
            ("boost:Immature".to_string(), immature),
            ("boost:LowQuality".to_string(), low_quality),
            ("boost:Stale".to_string(), stale),
            ("boost:Foreign".to_string(), foreign),
            ("boost:Hardware".to_string(), hardware),
            ("boost:Uncertain".to_string(), uncertain),
        ]);
        let mut model = WorldModel::default();
        attach_view(&mut model, Some(&context), &models, 1);

        let breakdown = model.utility_readiness_breakdown();
        assert_eq!(breakdown.known, 7);
        assert_eq!(breakdown.ready, 1);
        assert_eq!(breakdown.immature, 1);
        assert_eq!(breakdown.low_quality, 1);
        assert_eq!(breakdown.stale, 1);
        assert_eq!(breakdown.foreign_installation, 1);
        assert_eq!(breakdown.hardware_mismatch, 1);
        assert_eq!(breakdown.uncertain_interval, 1);
        assert_eq!(
            breakdown.ready
                + breakdown.no_current_gold
                + breakdown.immature
                + breakdown.low_quality
                + breakdown.stale
                + breakdown.foreign_installation
                + breakdown.hardware_mismatch
                + breakdown.uncertain_interval,
            breakdown.known
        );

        let mut no_context = WorldModel::default();
        attach_view(&mut no_context, None, &models, 1);
        let no_context = no_context.utility_readiness_breakdown();
        assert_eq!(no_context.no_current_gold, no_context.known);
    }

    fn model_with(key: &str, delta: f64, quality: f32, evidence: u32, drift: f64) -> WorldModel {
        let mut predicted = HashMap::new();
        predicted.insert(key.to_string(), (delta, quality, evidence));
        WorldModel {
            predicted,
            natural_drift: drift,
            ..WorldModel::default()
        }
    }

    #[test]
    fn unknown_action_and_immature_evidence_abstain() {
        let m = WorldModel::default();
        assert_eq!(
            m.imagine("freeze:Ghost"),
            Imagined::Unknown,
            "no edge → abstain"
        );

        let young = model_with("freeze:App", 0.08, 0.9, MIN_EVIDENCE - 1, 0.0);
        assert_eq!(
            young.imagine("freeze:App"),
            Imagined::Unknown,
            "immature evidence must never veto exploration"
        );

        let unsure = model_with("freeze:App", 0.08, MIN_DATA_QUALITY - 0.05, 50, 0.0);
        assert_eq!(unsure.imagine("freeze:App"), Imagined::Unknown);
    }

    #[test]
    fn act_wins_when_predicted_drop_beats_drift() {
        // Model predicts 6% drop; system drifts down only 1% alone.
        let m = model_with("freeze:Heavy", 0.06, 1.0, 40, 0.01);
        match m.imagine("freeze:Heavy") {
            Imagined::ActWins { margin } => {
                assert!((margin - 0.05).abs() < 1e-9, "margin = delta - drift");
            }
            other => panic!("expected ActWins, got {other:?}"),
        }
    }

    #[test]
    fn do_nothing_dominates_futile_action() {
        // Model has SOLID evidence the action barely moves pressure (0.4%)
        // while the system drops 1% by itself — acting is side-effects for
        // nothing. This is the imagined version of the Browser-0.27 lesson.
        let m = model_with("freeze:Futile", 0.004, 1.0, 60, 0.01);
        match m.imagine("freeze:Futile") {
            Imagined::DoNothingDominates {
                predicted_drop,
                natural_drift,
            } => {
                assert!(predicted_drop < natural_drift + DOMINANCE_MARGIN);
            }
            other => panic!("expected DoNothingDominates, got {other:?}"),
        }
    }

    #[test]
    fn negative_drift_clamps_to_zero_baseline() {
        // Pressure RISING on its own (drift negative): any solid positive
        // predicted drop above the margin must win — the baseline clamps
        // at 0 so a deteriorating system never suppresses relief actions.
        let m = model_with("throttle:Hog", 0.02, 1.0, 20, -0.03);
        assert!(matches!(
            m.imagine("throttle:Hog"),
            Imagined::ActWins { .. }
        ));
    }

    #[test]
    fn debias_deflates_inflated_imagination() {
        // Build through from_parts with a synthetic causal graph is heavy;
        // pin the semantics directly: an edge predicting 0.02 drop against
        // 0.012 drift wins raw, but at the prod CausalGraph debias (0.25x,
        // gap 0.256 regime) the calibrated prediction 0.005 loses — the
        // imagination must share the calibration layer's honesty.
        let raw = model_with("freeze:Inflated", 0.02, 1.0, 30, 0.012);
        assert!(matches!(
            raw.imagine("freeze:Inflated"),
            Imagined::ActWins { .. }
        ));

        let calibrated = model_with("freeze:Inflated", 0.02 * 0.25, 1.0, 30, 0.012);
        assert!(matches!(
            calibrated.imagine("freeze:Inflated"),
            Imagined::DoNothingDominates { .. }
        ));
    }

    #[test]
    fn raw_causal_edges_never_enter_curated_world_model() {
        let mut g = crate::engine::causal_graph::CausalGraph::new();
        for cycle in 0..50 {
            g.record_action("freeze:Legacy", 0.80, cycle * 4);
            g.evaluate(0.70, cycle * 4 + 3);
        }

        let tracker = crate::engine::outcome_tracker::OutcomeTracker::new();
        let m = WorldModel::from_parts(&g, &tracker, 1.0);
        assert_eq!(m.known_actions(), 0);
        assert_eq!(m.imagine("freeze:Legacy"), Imagined::Unknown);
    }

    #[test]
    fn mature_gold_evidence_drives_workload_specific_imagination() {
        let mut g = crate::engine::causal_graph::CausalGraph::new();
        let gold = crate::engine::data_medallion::CuratedLabel::trusted_legacy();
        for cycle in 0..10 {
            assert!(g.observe_curated_outcome("freeze:Editor", "build", 0.08, cycle, gold,));
            assert!(g.observe_curated_outcome(
                "freeze:Editor",
                "browsing",
                0.002,
                cycle + 10,
                gold,
            ));
        }

        let tracker = crate::engine::outcome_tracker::OutcomeTracker::new();
        let build = WorldModel::from_parts_for_workload(&g, &tracker, 1.0, "build");
        let browsing = WorldModel::from_parts_for_workload(&g, &tracker, 1.0, "browsing");

        assert!(matches!(
            build.imagine("freeze:Editor"),
            Imagined::ActWins { .. }
        ));
        assert!(matches!(
            browsing.imagine("freeze:Editor"),
            Imagined::DoNothingDominates { .. }
        ));
        assert_eq!(build.curated_observations(), 20);
        assert_eq!(build.contextual_actions(), 1);
        assert_eq!(build.ready_actions(), 1);
        assert_eq!(build.mean_data_quality(), 1.0);
    }

    #[test]
    fn max_margin_ignores_immature_predictions() {
        let immature = model_with("freeze:Young", 0.30, 1.0, MIN_EVIDENCE - 1, 0.0);
        assert_eq!(immature.max_predicted_margin(), 0.0);
    }

    #[test]
    fn universal_gold_actuator_models_drive_utility_imagination() {
        let now_unix = Utc::now().timestamp();
        let context = m4_context(now_unix);
        let models = [
            (
                "build|boost:Editor".to_string(),
                ActionModelStats {
                    observations: 12,
                    effective_observations: 10,
                    utility_ema: 0.08,
                    evidence_mass: 12.0,
                    utility_variance_ema: 0.0001,
                    state_delta_ema: Default::default(),
                    state_variance_ema: Default::default(),
                    state_evidence_mass: 0.0,
                    quality_ema: 0.95,
                    last_cycle: 100,
                    last_observed_unix: now_unix,
                    hardware_regime: HardwareRegime {
                        p_core_count: 4,
                        e_core_count: 6,
                        ram_gib: 16,
                    },
                    installation_id: LOCAL_ID,
                },
            ),
            (
                "thread_qos:Worker:background".to_string(),
                ActionModelStats {
                    observations: 15,
                    effective_observations: 2,
                    utility_ema: -0.03,
                    evidence_mass: 15.0,
                    utility_variance_ema: 0.0001,
                    state_delta_ema: Default::default(),
                    state_variance_ema: Default::default(),
                    state_evidence_mass: 0.0,
                    quality_ema: 0.93,
                    last_cycle: 101,
                    last_observed_unix: now_unix,
                    hardware_regime: HardwareRegime {
                        p_core_count: 4,
                        e_core_count: 6,
                        ram_gib: 16,
                    },
                    installation_id: LOCAL_ID,
                },
            ),
        ]
        .into_iter()
        .collect();
        let mut model = WorldModel::default();
        attach_view(&mut model, Some(&context), &models, 1);

        assert!(matches!(
            model.imagine_utility("boost:Editor", "build"),
            UtilityImagined::ActWins { .. }
        ));
        assert!(matches!(
            model.imagine_utility("thread_qos:Worker:background", "idle"),
            UtilityImagined::DoNothingDominates { .. }
        ));
        assert_eq!(model.utility_ready_actions(), 2);
        assert_eq!(
            model.imagine_utility("boost:Unknown", "build"),
            UtilityImagined::Unknown
        );
    }

    #[test]
    fn stale_or_uncertain_utility_models_abstain() {
        let now_unix = 2_000_000;
        let mut telemetry =
            TelemetryMedallion::new(crate::engine::installation_identity::InstallationId(1));
        let mut persisted =
            crate::engine::telemetry_medallion::TelemetryMedallionPersisted::default();
        persisted.actuator_evidence_schema_version = 2;
        persisted.latest = Some(TelemetryContextSummary {
            timestamp_unix: now_unix,
            ..TelemetryContextSummary::default()
        });
        persisted.action_models = [
            (
                "boost:Stale".to_string(),
                ActionModelStats {
                    observations: 100,
                    evidence_mass: 64.0,
                    utility_ema: 0.30,
                    utility_variance_ema: 0.0001,
                    quality_ema: 1.0,
                    last_observed_unix: now_unix - UTILITY_MAX_AGE_SECS - 1,
                    ..ActionModelStats::default()
                },
            ),
            (
                "boost:Uncertain".to_string(),
                ActionModelStats {
                    observations: 20,
                    evidence_mass: 20.0,
                    utility_ema: 0.01,
                    utility_variance_ema: 0.04,
                    quality_ema: 1.0,
                    last_observed_unix: now_unix,
                    ..ActionModelStats::default()
                },
            ),
        ]
        .into_iter()
        .collect();
        telemetry.restore(persisted);
        let mut model = WorldModel::default();
        model.attach_context(telemetry.trusted_view());

        assert_eq!(
            model.imagine_utility("boost:Stale", "idle"),
            UtilityImagined::Unknown
        );
        assert_eq!(
            model.imagine_utility("boost:Uncertain", "idle"),
            UtilityImagined::Unknown
        );
    }

    #[test]
    fn foreign_hardware_model_abstains_even_when_statistically_mature() {
        use crate::engine::telemetry_medallion::HardwareRegime;

        let now_unix = 2_500_000;
        let mut telemetry =
            TelemetryMedallion::new(crate::engine::installation_identity::InstallationId(1));
        let mut persisted =
            crate::engine::telemetry_medallion::TelemetryMedallionPersisted::default();
        persisted.actuator_evidence_schema_version = 2;
        persisted.latest = Some(TelemetryContextSummary {
            timestamp_unix: now_unix,
            p_core_count: 4,
            e_core_count: 6,
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            ..TelemetryContextSummary::default()
        });
        persisted.action_models.insert(
            "boost:Editor".to_string(),
            ActionModelStats {
                observations: 64,
                evidence_mass: 64.0,
                utility_ema: 0.30,
                utility_variance_ema: 0.0001,
                quality_ema: 1.0,
                last_observed_unix: now_unix,
                hardware_regime: HardwareRegime {
                    p_core_count: 4,
                    e_core_count: 4,
                    ram_gib: 8,
                },
                ..ActionModelStats::default()
            },
        );
        telemetry.restore(persisted);
        let mut model = WorldModel::default();
        model.attach_context(telemetry.trusted_view());

        assert_eq!(model.utility_known_actions(), 1);
        assert_eq!(model.utility_ready_actions(), 0);
        assert_eq!(
            model.imagine_utility("boost:Editor", "idle"),
            UtilityImagined::Unknown
        );
    }

    #[test]
    fn immature_workload_bucket_falls_back_to_mature_aggregate() {
        let now_unix = Utc::now().timestamp();
        let context = m4_context(now_unix);
        let mature = ActionModelStats {
            observations: 20,
            evidence_mass: 20.0,
            utility_ema: 0.08,
            utility_variance_ema: 0.0001,
            quality_ema: 0.95,
            last_observed_unix: now_unix,
            hardware_regime: HardwareRegime {
                p_core_count: 4,
                e_core_count: 6,
                ram_gib: 16,
            },
            installation_id: LOCAL_ID,
            ..ActionModelStats::default()
        };
        let immature = ActionModelStats {
            observations: 2,
            evidence_mass: 2.0,
            utility_ema: -0.50,
            utility_variance_ema: 0.0001,
            quality_ema: 0.95,
            last_observed_unix: now_unix,
            hardware_regime: HardwareRegime {
                p_core_count: 4,
                e_core_count: 6,
                ram_gib: 16,
            },
            installation_id: LOCAL_ID,
            ..ActionModelStats::default()
        };
        let models = BTreeMap::from([
            ("boost:Editor".to_string(), mature),
            ("build|boost:Editor".to_string(), immature),
        ]);
        let mut model = WorldModel::default();
        attach_view(&mut model, Some(&context), &models, 1);

        assert!(matches!(
            model.imagine_utility("boost:Editor", "build"),
            UtilityImagined::ActWins { .. }
        ));
    }

    #[test]
    fn incremental_refresh_skips_unchanged_models_and_invalidates_on_gold() {
        let mut graph = crate::engine::causal_graph::CausalGraph::new();
        let tracker = crate::engine::outcome_tracker::OutcomeTracker::new();
        let telemetry =
            TelemetryMedallion::new(crate::engine::installation_identity::InstallationId(1));
        let mut model = WorldModel::default();

        model.refresh_from_parts_for_workload(&graph, &tracker, 1.0, "build");
        model.attach_context(telemetry.trusted_view());
        model.refresh_from_parts_for_workload(&graph, &tracker, 1.0, "build");
        model.attach_context(telemetry.trusted_view());
        assert_eq!(model.cache_stats(), (1, 1, 1, 1));

        let gold = crate::engine::data_medallion::CuratedLabel::trusted_legacy();
        assert!(graph.observe_curated_outcome("freeze:Editor", "build", 0.08, 10, gold,));
        model.refresh_from_parts_for_workload(&graph, &tracker, 1.0, "build");
        assert_eq!(model.known_actions(), 1);
        assert_eq!(model.cache_stats(), (2, 1, 1, 1));
    }

    #[test]
    fn utility_abstention_explains_missing_and_immature_evidence() {
        let now = Utc::now().timestamp();
        let context = m4_context(now);
        let mut model = WorldModel::default();
        attach_view(&mut model, Some(&context), &BTreeMap::new(), 1);
        assert_eq!(
            model.assess_utility_diagnostic("boost:Unknown", "build"),
            UtilityAssessmentResult::Abstained(UtilityAbstention {
                reason: UtilityAbstentionReason::UnknownAction,
                scope: None,
            })
        );

        let mut immature = mature_model(now, LOCAL_ID);
        immature.evidence_mass = 2.0;
        immature.effective_observations = 2;
        let models = BTreeMap::from([("boost:Editor".to_string(), immature)]);
        model.attach_context(TrustedTelemetryView {
            current: Some(&context),
            installation_id: LOCAL_ID,
            action_models: &models,
            action_models_revision: 2,
            controlled_models: &BTreeMap::new(),
            controlled_models_revision: 0,
            episodic_evidence: &VecDeque::new(),
            causal_dynamics: &CausalDynamicsModel::default(),
            causal_dynamics_revision: 0,
            gpu_calibration_models: &BTreeMap::new(),
            gpu_calibration_revision: 0,
            metrics: TelemetryMedallionMetrics {
                local_gold_total: 1,
                ..TelemetryMedallionMetrics::default()
            },
        });
        assert!(matches!(
            model.assess_utility_diagnostic("boost:Editor", "build"),
            UtilityAssessmentResult::Abstained(UtilityAbstention {
                reason: UtilityAbstentionReason::ImmatureEvidence,
                ..
            })
        ));
    }

    #[test]
    fn local_control_arm_adds_bounded_ranking_support_without_changing_verdict() {
        let now = Utc::now().timestamp();
        let context = m4_context(now);
        let models = BTreeMap::from([(
            "build|boost:Editor".to_string(),
            mature_model(now, LOCAL_ID),
        )]);
        let controls = BTreeMap::from([(
            "build|boost:Editor".to_string(),
            ControlledCounterfactualStats {
                observations: 8,
                would_have_helped: 6,
                control_utility_ema: -0.04,
                quality_ema: 0.95,
                last_cycle: 100,
                last_observed_unix: now,
                hardware_regime: HardwareRegime {
                    p_core_count: 4,
                    e_core_count: 6,
                    ram_gib: 16,
                },
                installation_id: LOCAL_ID,
            },
        )]);
        let mut model = WorldModel::default();
        model.attach_context(TrustedTelemetryView {
            current: Some(&context),
            installation_id: LOCAL_ID,
            action_models: &models,
            action_models_revision: 1,
            controlled_models: &controls,
            controlled_models_revision: 1,
            episodic_evidence: &VecDeque::new(),
            causal_dynamics: &CausalDynamicsModel::default(),
            causal_dynamics_revision: 0,
            gpu_calibration_models: &BTreeMap::new(),
            gpu_calibration_revision: 0,
            metrics: TelemetryMedallionMetrics {
                local_gold_total: 1,
                ..TelemetryMedallionMetrics::default()
            },
        });

        let assessment = model
            .assess_utility("boost:Editor", "build")
            .expect("mature treatment model");
        assert!(matches!(
            assessment.verdict,
            UtilityImagined::ActWins { .. }
        ));
        assert_eq!(assessment.counterfactual_observations, 8);
        assert!(assessment.counterfactual_support > 0.0);
        assert!(assessment.counterfactual_support <= COUNTERFACTUAL_MAX_RANK_SUPPORT);
    }

    #[test]
    fn foreign_control_arm_never_influences_local_ranking() {
        let now = Utc::now().timestamp();
        let context = m4_context(now);
        let models = BTreeMap::from([(
            "build|boost:Editor".to_string(),
            mature_model(now, LOCAL_ID),
        )]);
        let controls = BTreeMap::from([(
            "build|boost:Editor".to_string(),
            ControlledCounterfactualStats {
                observations: 20,
                would_have_helped: 20,
                control_utility_ema: -1.0,
                quality_ema: 1.0,
                last_cycle: 100,
                last_observed_unix: now,
                hardware_regime: HardwareRegime {
                    p_core_count: 4,
                    e_core_count: 6,
                    ram_gib: 16,
                },
                installation_id: InstallationId(99),
            },
        )]);
        let mut model = WorldModel::default();
        model.attach_context(TrustedTelemetryView {
            current: Some(&context),
            installation_id: LOCAL_ID,
            action_models: &models,
            action_models_revision: 1,
            controlled_models: &controls,
            controlled_models_revision: 1,
            episodic_evidence: &VecDeque::new(),
            causal_dynamics: &CausalDynamicsModel::default(),
            causal_dynamics_revision: 0,
            gpu_calibration_models: &BTreeMap::new(),
            gpu_calibration_revision: 0,
            metrics: TelemetryMedallionMetrics {
                local_gold_total: 1,
                ..TelemetryMedallionMetrics::default()
            },
        });

        let assessment = model
            .assess_utility("boost:Editor", "build")
            .expect("mature treatment model");
        assert_eq!(assessment.counterfactual_observations, 0);
        assert_eq!(assessment.counterfactual_support, 0.0);
    }

    #[test]
    fn family_prior_ranks_unseen_target_without_granting_veto_authority() {
        let now = Utc::now().timestamp();
        let context = m4_context(now);
        let models = BTreeMap::from([("boost:*".to_string(), mature_model(now, LOCAL_ID))]);
        let mut model = WorldModel::default();
        attach_view(&mut model, Some(&context), &models, 1);

        let prior = model
            .assess_family_prior("boost:Unseen")
            .expect("mature local family prior");
        assert!(prior.utility_ema > 0.0);
        assert_eq!(
            model.assess_utility_diagnostic("boost:Unseen", "build"),
            UtilityAssessmentResult::Abstained(UtilityAbstention {
                reason: UtilityAbstentionReason::UnknownAction,
                scope: None,
            }),
            "family evidence must not become exact-action veto authority"
        );
    }

    #[test]
    fn local_consolidation_scales_advice_without_changing_authoritative_verdict() {
        let now = Utc::now().timestamp();
        let context = m4_context(now);
        let models = BTreeMap::from([(
            "build|boost:Editor".to_string(),
            mature_model(now, LOCAL_ID),
        )]);
        let mut model = WorldModel::default();
        attach_view(&mut model, Some(&context), &models, 24);

        let before_assessment = model
            .assess_utility("boost:Editor", "build")
            .expect("mature local utility model");
        let before_bias = model.contextual_action_bias("boost:Editor", "build");
        model.attach_local_consolidation(LocalConsolidationView {
            confidence: 0.40,
            families_with_evidence: 1,
            total_consolidations: 12,
            family_scales: BTreeMap::from([("boost".to_string(), 0.75)]),
        });
        let deliberation = model.synthesize_deliberation("build", false, 0.20);
        let after_assessment = model
            .assess_utility("boost:Editor", "build")
            .expect("authority must survive local advisory calibration");
        let after_bias = model.contextual_action_bias("boost:Editor", "build");

        assert_eq!(before_assessment.verdict, after_assessment.verdict);
        assert_eq!(before_assessment.lower_bound, after_assessment.lower_bound);
        assert_eq!(before_assessment.upper_bound, after_assessment.upper_bound);
        assert!(before_bias.authoritative && after_bias.authoritative);
        assert!(after_bias.score > 0.0 && after_bias.score < before_bias.score);
        assert_eq!(deliberation.evidence.local_consolidation_families, 1);
        assert!(deliberation.advisory_support_scale <= 1.0);
    }

    #[test]
    fn dr_zero_and_system1_struggle_reduce_local_deliberation_confidence() {
        let now = Utc::now().timestamp();
        let context = m4_context(now);
        let models = BTreeMap::from([(
            "build|boost:Editor".to_string(),
            mature_model(now, LOCAL_ID),
        )]);
        let mut grounded = WorldModel::default();
        attach_view(&mut grounded, Some(&context), &models, 24);
        grounded.attach_local_consolidation(LocalConsolidationView {
            confidence: 0.90,
            families_with_evidence: 1,
            total_consolidations: 24,
            family_scales: BTreeMap::from([("boost".to_string(), 0.95)]),
        });
        let mut challenged = grounded.clone();

        let healthy = grounded.synthesize_deliberation("build", false, 0.0);
        let guarded = challenged.synthesize_deliberation("build", true, 1.0);
        assert!(guarded.confidence < healthy.confidence);
        assert!(guarded.advisory_support_scale < healthy.advisory_support_scale);
        assert!(guarded.gpu_support_scale < healthy.gpu_support_scale);
    }

    #[test]
    fn curated_world_model_handles_500_outcomes_with_bounded_query_cost() {
        let mut graph = crate::engine::causal_graph::CausalGraph::new();
        let gold = crate::engine::data_medallion::CuratedLabel::trusted_legacy();
        let keys: Vec<String> = (0..50)
            .map(|action| format!("freeze:worker-{action}"))
            .collect();
        for (action, key) in keys.iter().enumerate() {
            let drop = if action % 2 == 0 { 0.04 } else { 0.001 };
            for sample in 0..10 {
                assert!(graph.observe_curated_outcome(
                    key,
                    "build",
                    drop,
                    (action * 10 + sample) as u64,
                    gold,
                ));
            }
        }

        let tracker = crate::engine::outcome_tracker::OutcomeTracker::new();
        let started = Instant::now();
        let model = WorldModel::from_parts_for_workload(&graph, &tracker, 1.0, "build");
        for _ in 0..200 {
            for key in &keys {
                std::hint::black_box(model.imagine(key));
            }
        }
        let elapsed = started.elapsed();

        assert_eq!(model.curated_observations(), 500);
        assert_eq!(model.known_actions(), 50);
        assert_eq!(model.ready_actions(), 50);
        assert_eq!(model.contextual_actions(), 50);
        assert!(matches!(model.imagine(&keys[0]), Imagined::ActWins { .. }));
        assert!(matches!(
            model.imagine(&keys[1]),
            Imagined::DoNothingDominates { .. }
        ));
        assert!(
            elapsed < Duration::from_millis(250),
            "500-outcome build plus 10k queries took {elapsed:?}"
        );
        eprintln!("curated_world_model_500: {elapsed:?}");
    }
}
