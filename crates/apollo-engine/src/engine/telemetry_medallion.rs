//! Live context and actuator-evidence medallion for the World Model.
//!
//! The original action medallion remains the specialist trust boundary for
//! pressure-relief learning. This module adds two complementary, persisted
//! lanes:
//! - complete descriptive machine context on every cycle;
//! - resolved outcomes for every confirmed Apollo actuator, evaluated against
//!   a no-action baseline with family-specific objectives.
//!
//! Context Gold remains descriptive. Actuator Gold requires an applied action,
//! a later observation, coherent telemetry, and low-confounding context.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt;
use std::marker::PhantomData;

use serde::de::{IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::collector::SystemSnapshot;
use crate::engine::causal_dynamics::CausalDynamicsModel;
use crate::engine::decision_ledger::{
    DecisionId, DecisionLifecycle, ReceiptAttribution, ResolvedDecisionEpisode,
};
use crate::engine::execute_actions::ExecuteOutcomes;
use crate::engine::gpu_imagination::GpuImaginationResult;
use crate::engine::installation_identity::InstallationId;
use crate::engine::iokit_sensors::HardwareSnapshot;
use crate::engine::learning_hierarchy::{HierarchyContext, HierarchyPath, ResolvedLearningDetails};
use crate::engine::model_calibration::{
    valid_forecast_deltas, CalibrationActionScope, CalibrationKey, CalibrationObservation,
    CalibrationProvenance, CalibrationUpdate, ForegroundContext, ModelCalibrationMetrics,
    ModelCalibrationPersisted, ModelCalibrationStore, ModelCalibrationSummary, PressureBand,
    ProcessClass, ProducerId, SeparabilityState, ThermalBand, TrustState,
};
use crate::engine::predictive_agent::Intervention;
use crate::engine::signal_intelligence::SignalDigest;
use crate::engine::telemetry_context_admission::{
    classify, ContextAdmission, ContextAdmissionInput, ContextField, ContextFieldViolation,
    ContextReasonCounters, ContextTier,
};
use crate::engine::types::{CapabilityReport, RootAction, RuntimeMetrics};
use chrono::Utc;

const MAX_PENDING_ACTIONS: usize = 192;
const MAX_RECENT_EVIDENCE: usize = 64;
const MAX_EPISODIC_EVIDENCE: usize = 128;
const MAX_EPISODES_PER_FAMILY: usize = 12;
const MAX_ACTION_MODELS: usize = 256;
const MAX_PENDING_CONTROLLED_HOLDOUTS: usize = 32;
const MAX_CONTROLLED_MODELS: usize = 256;
const MAX_GPU_PREDICTIONS: usize = 256;
const MAX_GPU_CALIBRATION_MODELS: usize = 256;
const MAX_DECISION_SOURCES: usize = 48;
const MAX_STAGED_ATTRIBUTIONS: usize = 64;
const MAX_STAGED_DECISION_EPISODES: usize = 640;
const GPU_PREDICTION_MATCH_MAX_AGE_CYCLES: u64 = 30;
const CONTROLLED_HOLDOUT_HORIZON_CYCLES: u64 = 30;
const ACTION_MODEL_EMA_ALPHA: f64 = 0.20;
const ACTION_MODEL_EVIDENCE_HALF_LIFE_SECS: f64 = 7.0 * 24.0 * 60.0 * 60.0;
const ACTION_MODEL_EVIDENCE_CAP: f64 = 64.0;
// Version 1 enrolled predictive recommendations before confirming that their
// operating-system side effect occurred. Do not carry that bias forward.
const ACTUATOR_EVIDENCE_SCHEMA_VERSION: u32 = 3;
const TELEMETRY_CONTEXT_SCHEMA_VERSION: u32 = 3;
const GPU_PREDICTION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(default)]
pub struct HardwareRegime {
    pub p_core_count: u32,
    pub e_core_count: u32,
    pub ram_gib: u32,
}

impl HardwareRegime {
    pub fn from_context(context: &TelemetryContextSummary) -> Self {
        const GIB: u64 = 1024 * 1024 * 1024;
        Self {
            p_core_count: context.p_core_count,
            e_core_count: context.e_core_count,
            ram_gib: context
                .total_ram_bytes
                .saturating_add(GIB / 2)
                .checked_div(GIB)
                .unwrap_or(0)
                .min(u32::MAX as u64) as u32,
        }
    }

    pub fn is_known(self) -> bool {
        self.p_core_count.saturating_add(self.e_core_count) > 0 && self.ram_gib > 0
    }

    pub fn matches_context(self, context: &TelemetryContextSummary) -> bool {
        let current = Self::from_context(context);
        !current.is_known() || (self.is_known() && self == current)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ActuatorFamily {
    Boost,
    Throttle,
    Freeze,
    Unfreeze,
    Memorystatus,
    Sysctl,
    Spotlight,
    Quarantine,
    ThreadQos,
    MarkovPrewarm,
    InteractionQos,
    IoShaping,
    PredictiveThreshold,
    PredictiveProfile,
    PredictivePreThrottle,
    PredictivePurge,
    ChromiumEcore,
    ChromiumPurge,
    ChromiumJetsam,
    Coordinated,
}

impl ActuatorFamily {
    pub const ALL: [Self; 20] = [
        Self::Boost,
        Self::Throttle,
        Self::Freeze,
        Self::Unfreeze,
        Self::Memorystatus,
        Self::Sysctl,
        Self::Spotlight,
        Self::Quarantine,
        Self::ThreadQos,
        Self::MarkovPrewarm,
        Self::InteractionQos,
        Self::IoShaping,
        Self::PredictiveThreshold,
        Self::PredictiveProfile,
        Self::PredictivePreThrottle,
        Self::PredictivePurge,
        Self::ChromiumEcore,
        Self::ChromiumPurge,
        Self::ChromiumJetsam,
        Self::Coordinated,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Boost => "boost",
            Self::Throttle => "throttle",
            Self::Freeze => "freeze",
            Self::Unfreeze => "unfreeze",
            Self::Memorystatus => "memorystatus",
            Self::Sysctl => "sysctl",
            Self::Spotlight => "spotlight",
            Self::Quarantine => "quarantine",
            Self::ThreadQos => "thread_qos",
            Self::MarkovPrewarm => "markov_prewarm",
            Self::InteractionQos => "interaction_qos",
            Self::IoShaping => "io_shaping",
            Self::PredictiveThreshold => "predictive_threshold",
            Self::PredictiveProfile => "predictive_profile",
            Self::PredictivePreThrottle => "predictive_prethrottle",
            Self::PredictivePurge => "predictive_purge",
            Self::ChromiumEcore => "chromium_ecore",
            Self::ChromiumPurge => "chromium_purge",
            Self::ChromiumJetsam => "chromium_jetsam",
            Self::Coordinated => "coordinated",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ActuatorObjective {
    PressureRelief,
    Responsiveness,
    Efficiency,
    Recovery,
    NetworkHealth,
    Availability,
    Prediction,
    BalancedUtility,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceTier {
    Bronze,
    Silver,
    Gold,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceProvenance {
    ObservedLocal,
    SyntheticCounter,
    GpuImagined,
    ModelCounterfactual,
    Advisory,
    #[default]
    LegacyUnknown,
}

pub fn quarantine_experimental_tier(
    provenance: EvidenceProvenance,
    proposed: EvidenceTier,
) -> EvidenceTier {
    if provenance == EvidenceProvenance::ObservedLocal {
        proposed
    } else {
        EvidenceTier::Bronze
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct ActuatorFamilyStats {
    pub issued_total: u64,
    pub resolved_total: u64,
    pub bronze_total: u64,
    pub silver_total: u64,
    pub gold_total: u64,
    pub effective_total: u64,
    pub rejected_total: u64,
    pub expired_total: u64,
    pub utility_sum: f64,
    pub quality_sum: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct ActionModelStats {
    pub observations: u32,
    pub effective_observations: u32,
    pub utility_ema: f64,
    /// Exponentially discounted effective sample size. Unlike `observations`,
    /// this loses authority when the machine or workload regime changes.
    pub evidence_mass: f64,
    /// Exponentially weighted utility variance used by WorldModel confidence
    /// bounds. Zero is valid for a genuinely stable freshly learned model.
    pub utility_variance_ema: f64,
    /// Counterfactual-adjusted transition of the joint machine state. These
    /// dimensions let the World Model roll actions forward without collapsing
    /// fluidity, energy, thermal and memory behavior into one scalar utility.
    pub state_delta_ema: WorldStateDelta,
    pub state_variance_ema: WorldStateDelta,
    /// Decayed authority for the transition vector. Kept separate from utility
    /// evidence so legacy scalar models cannot immediately drive rollouts.
    pub state_evidence_mass: f64,
    pub quality_ema: f64,
    pub last_cycle: u64,
    /// Wall-clock freshness survives daemon cycle resets and machine imports.
    /// Legacy persisted models deserialize as zero and remain weak priors until
    /// this Mac produces new Gold evidence.
    pub last_observed_unix: i64,
    /// Portable state may cross machines, but decision authority may not. The
    /// regime uses capabilities rather than a chip-name allowlist so future
    /// Apple Silicon generations remain supported without code changes.
    pub hardware_regime: HardwareRegime,
    /// Decision authority is local to one private Apollo installation.
    pub installation_id: InstallationId,
}

/// Bounded provenance for one executed decision. The owning specialist is the
/// proposer; other engines may support or oppose the same candidate without
/// gaining authority to manufacture a root action.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct DecisionAttribution {
    pub action_key: String,
    pub proposer: String,
    pub supporters: Vec<String>,
    pub vetoes: Vec<String>,
    pub predicted_gain: f64,
    pub uncertainty: f64,
}

impl DecisionAttribution {
    fn bounded(mut self) -> Self {
        self.action_key = bounded_text(&self.action_key, 320);
        self.proposer = bounded_text(&self.proposer, 48);
        self.supporters = bounded_sources(self.supporters);
        self.vetoes = bounded_sources(self.vetoes);
        self.predicted_gain = finite_or_zero(self.predicted_gain).clamp(-1.0, 1.0);
        self.uncertainty = finite_or_zero(self.uncertainty).clamp(0.0, 1.0);
        self
    }
}

/// Human-facing and machine-facing value remain separate so an energy win
/// cannot hide a responsiveness regression (or vice versa).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct UtilityDecomposition {
    pub system_gain: f64,
    pub human_gain: f64,
    pub intervention_cost: f64,
    pub apollo_utility: f64,
}

impl UtilityDecomposition {
    fn is_finite(self) -> bool {
        [
            self.system_gain,
            self.human_gain,
            self.intervention_cost,
            self.apollo_utility,
        ]
        .into_iter()
        .all(f64::is_finite)
    }
}

/// Online calibration of one decision source (S1 specialist, World Model,
/// GPU, causal model, etc.). Positive credit means its direction agreed with
/// later measured Apollo utility; veto credit is inverted by construction.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct DecisionSourceStats {
    pub observations: u32,
    pub supports: u32,
    pub vetoes: u32,
    pub correct: u32,
    pub credit_ema: f64,
    pub absolute_error_ema: f64,
    pub last_cycle: u64,
}

impl DecisionSourceStats {
    pub fn accuracy(&self) -> f64 {
        if self.observations == 0 {
            0.0
        } else {
            (self.correct as f64 / self.observations as f64).clamp(0.0, 1.0)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct ControlledCounterfactualStats {
    pub observations: u32,
    pub would_have_helped: u32,
    /// Utility change observed while the selected action was deliberately
    /// withheld. Negative means the no-action branch degraded.
    pub control_utility_ema: f64,
    pub quality_ema: f64,
    pub last_cycle: u64,
    #[serde(default)]
    pub last_observed_unix: i64,
    #[serde(default)]
    pub hardware_regime: HardwareRegime,
    #[serde(default)]
    pub installation_id: InstallationId,
}

#[derive(Debug, Clone)]
struct PendingControlledHoldout {
    action_key: String,
    workload: String,
    family: ActuatorFamily,
    objective: ActuatorObjective,
    issued_cycle: u64,
    family_issued_total_at_start: u64,
    before: TelemetryContextSummary,
}

impl ActionModelStats {
    pub fn effective_evidence_at(&self, now_unix: i64) -> f64 {
        if self.last_observed_unix <= 0 || now_unix < self.last_observed_unix {
            return 0.0;
        }
        let age_secs = (now_unix - self.last_observed_unix) as f64;
        let decay = 0.5_f64.powf(age_secs / ACTION_MODEL_EVIDENCE_HALF_LIFE_SECS);
        (self.evidence_mass * decay).clamp(0.0, ACTION_MODEL_EVIDENCE_CAP)
    }

    pub fn effective_state_evidence_at(&self, now_unix: i64) -> f64 {
        if self.last_observed_unix <= 0 || now_unix < self.last_observed_unix {
            return 0.0;
        }
        let age_secs = (now_unix - self.last_observed_unix) as f64;
        let decay = 0.5_f64.powf(age_secs / ACTION_MODEL_EVIDENCE_HALF_LIFE_SECS);
        (self.state_evidence_mass * decay).clamp(0.0, ACTION_MODEL_EVIDENCE_CAP)
    }
}

/// Normalized transition of the joint machine state. Lower is better for all
/// dimensions except `fluidity`, where positive is an improvement.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct WorldStateDelta {
    pub pressure: f64,
    pub fluidity: f64,
    /// Positive means perceptual latency worsened.
    #[serde(default)]
    pub latency: f64,
    pub energy: f64,
    pub cpu: f64,
    pub thermal: f64,
    pub thrashing: f64,
    pub stall: f64,
}

impl WorldStateDelta {
    pub fn between(before: &TelemetryContextSummary, after: &TelemetryContextSummary) -> Self {
        let energy = match (before.package_watts, after.package_watts) {
            (Some(before), Some(after)) => (after - before) / 50.0,
            _ => 0.0,
        };
        Self {
            pressure: after.memory_pressure - before.memory_pressure,
            fluidity: after.fluidity_score - before.fluidity_score,
            latency: after.perceptual_latency_score - before.perceptual_latency_score,
            energy,
            cpu: after.cpu_max_busy - before.cpu_max_busy,
            thermal: after.thermal_score - before.thermal_score,
            thrashing: (after.thrashing_score - before.thrashing_score) / 50_000.0,
            stall: after.stall_fraction - before.stall_fraction,
        }
        .clamped(-1.0, 1.0)
    }

    pub fn is_finite(self) -> bool {
        [
            self.pressure,
            self.fluidity,
            self.latency,
            self.energy,
            self.cpu,
            self.thermal,
            self.thrashing,
            self.stall,
        ]
        .into_iter()
        .all(f64::is_finite)
    }

    pub fn clamped(self, min: f64, max: f64) -> Self {
        Self {
            pressure: self.pressure.clamp(min, max),
            fluidity: self.fluidity.clamp(min, max),
            latency: self.latency.clamp(min, max),
            energy: self.energy.clamp(min, max),
            cpu: self.cpu.clamp(min, max),
            thermal: self.thermal.clamp(min, max),
            thrashing: self.thrashing.clamp(min, max),
            stall: self.stall.clamp(min, max),
        }
    }

    pub fn scaled(self, scale: f64) -> Self {
        Self {
            pressure: self.pressure * scale,
            fluidity: self.fluidity * scale,
            latency: self.latency * scale,
            energy: self.energy * scale,
            cpu: self.cpu * scale,
            thermal: self.thermal * scale,
            thrashing: self.thrashing * scale,
            stall: self.stall * scale,
        }
    }

    pub fn plus(self, other: Self) -> Self {
        Self {
            pressure: self.pressure + other.pressure,
            fluidity: self.fluidity + other.fluidity,
            latency: self.latency + other.latency,
            energy: self.energy + other.energy,
            cpu: self.cpu + other.cpu,
            thermal: self.thermal + other.thermal,
            thrashing: self.thrashing + other.thrashing,
            stall: self.stall + other.stall,
        }
    }

    pub fn minus(self, other: Self) -> Self {
        self.plus(other.scaled(-1.0))
    }

    pub(crate) fn ema(self, observation: Self, alpha: f64) -> Self {
        self.scaled(1.0 - alpha).plus(observation.scaled(alpha))
    }

    pub(crate) fn variance_update(self, residual: Self, alpha: f64) -> Self {
        let squared = Self {
            pressure: residual.pressure * residual.pressure,
            fluidity: residual.fluidity * residual.fluidity,
            latency: residual.latency * residual.latency,
            energy: residual.energy * residual.energy,
            cpu: residual.cpu * residual.cpu,
            thermal: residual.thermal * residual.thermal,
            thrashing: residual.thrashing * residual.thrashing,
            stall: residual.stall * residual.stall,
        };
        self.plus(squared.scaled(alpha)).scaled(1.0 - alpha)
    }

    pub fn mean_variance(self) -> f64 {
        (self.pressure
            + self.fluidity
            + self.latency
            + self.energy
            + self.cpu
            + self.thermal
            + self.thrashing
            + self.stall)
            / 8.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedActuatorEvidence {
    pub id: u64,
    /// Universal identity assigned by `DecisionLedger`. Aggregate fallback
    /// evidence and legacy state have no decision identity.
    #[serde(default)]
    pub decision_id: Option<DecisionId>,
    pub family: ActuatorFamily,
    pub objective: ActuatorObjective,
    pub action_key: String,
    pub target: String,
    pub workload: String,
    pub issued_cycle: u64,
    pub resolved_cycle: u64,
    #[serde(default)]
    pub resolved_timestamp_unix: i64,
    #[serde(default)]
    pub hardware_regime: HardwareRegime,
    #[serde(default)]
    pub installation_id: InstallationId,
    pub horizon_cycles: u64,
    pub tier: EvidenceTier,
    #[serde(default)]
    pub provenance: EvidenceProvenance,
    pub quality: f64,
    pub raw_utility_delta: f64,
    pub counterfactual_delta: f64,
    pub net_utility_delta: f64,
    /// Source-aware credit assignment captured before the action crossed the
    /// dispatcher. Legacy evidence deserializes as unknown provenance.
    #[serde(default)]
    pub attribution: DecisionAttribution,
    /// Full bounded decision-time forecast provenance. This is separate from
    /// the compatibility attribution projection so delayed Gold resolution
    /// cannot collapse to the first prediction.
    #[serde(default)]
    pub calibration_provenance: CalibrationProvenance,
    /// Immutable result of the one authoritative Task 3 admission. It is
    /// present only on locally authoritative Gold evidence.
    #[serde(default)]
    pub learning_details: Option<ResolvedLearningDetails>,
    /// Objective-independent utility used as Apollo's top-level score.
    #[serde(default)]
    pub utility: UtilityDecomposition,
    /// Positive means the local responsiveness proxy improved between the
    /// episode's admitted pre/post contexts.
    #[serde(default)]
    pub perceptual_latency_improvement: f64,
    #[serde(default)]
    pub net_state_delta: WorldStateDelta,
    /// Compact state at action emission. This preserves enough universal
    /// context for same-machine episodic recall without persisting the full
    /// telemetry frame for every outcome.
    #[serde(default)]
    pub context_before: ActuatorEpisodeContext,
    pub effective: bool,
    pub confounder_count: u8,
    pub target_present_after: Option<bool>,
}

/// Compact Bronze -> Silver -> Gold lineage for one GPU-imagined candidate.
/// Raw Monte Carlo samples remain ephemeral; only decision-relevant summaries
/// and their later measured outcome are persisted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuPredictionEvidence {
    pub generation: u64,
    pub action_key: String,
    pub workload: String,
    pub issued_cycle: u64,
    pub expected_gain: f64,
    pub uncertainty: f64,
    pub mean_gain: f64,
    pub p10_gain: f64,
    pub positive_probability: f64,
    pub rank_support: f64,
    pub context_score: f64,
    pub samples: u64,
    pub gpu_time_ns: u64,
    pub used_cycle: Option<u64>,
    pub resolved_cycle: Option<u64>,
    pub tier: EvidenceTier,
    pub actual_utility: Option<f64>,
    pub absolute_error: Option<f64>,
    pub brier_score: Option<f64>,
    pub p10_covered: Option<bool>,
    pub quality: f64,
    pub hardware_regime: HardwareRegime,
    pub installation_id: InstallationId,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct GpuCalibrationStats {
    pub predictions: u32,
    pub used: u32,
    pub resolved: u32,
    pub gold: u32,
    pub signed_error_ema: f64,
    pub absolute_error_ema: f64,
    pub brier_ema: f64,
    pub p10_coverage_ema: f64,
    pub quality_ema: f64,
    pub evidence_mass: f64,
    pub last_cycle: u64,
    pub last_observed_unix: i64,
    pub hardware_regime: HardwareRegime,
    pub installation_id: InstallationId,
}

impl GpuCalibrationStats {
    pub fn trust(&self, context: &TelemetryContextSummary, installation_id: InstallationId) -> f64 {
        if self.installation_id != installation_id
            || !self.hardware_regime.matches_context(context)
            || self.gold == 0
            || self.last_observed_unix <= 0
            || context.timestamp_unix < self.last_observed_unix
            || context.timestamp_unix - self.last_observed_unix
                > ACTION_MODEL_EVIDENCE_HALF_LIFE_SECS as i64 * 2
        {
            return 0.0;
        }
        let age_secs = (context.timestamp_unix - self.last_observed_unix) as f64;
        let recency = 0.5_f64.powf(age_secs / ACTION_MODEL_EVIDENCE_HALF_LIFE_SECS);
        let maturity = (self.evidence_mass * recency / 10.0).sqrt().clamp(0.0, 1.0);
        (maturity * self.quality_ema.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    }
}

pub fn gpu_calibration_key(action_key: &str, workload: &str) -> String {
    format!(
        "{}|{}",
        bounded_text(action_key, 320),
        bounded_text(workload, 64)
    )
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct ActuatorEpisodeContext {
    pub valid: bool,
    pub memory_pressure: f64,
    pub compressor_pressure: f64,
    pub thrashing_score: f64,
    pub cpu_global_usage: f64,
    pub cpu_max_busy: f64,
    pub cpu_pegged_fraction: f64,
    pub stall_fraction: f64,
    pub used_ram_fraction: f64,
    pub thermal_score: f64,
    pub fluidity_score: f64,
    /// Composite local responsiveness signal. Lower is better. This is kept
    /// beside fluidity so pre/post actuator evidence can explain user-facing
    /// benefit without confusing it with daemon cycle time.
    pub perceptual_latency_score: f64,
    pub scheduler_jitter_p95_ms: f64,
    pub windowserver_cpu_fraction: f64,
    pub arousal_level: f64,
    pub signal_urgency: f64,
    pub signal_entropy_anomaly: f64,
    pub nars_drift_score: f64,
    pub markov_prediction_confidence: f64,
    pub network_retransmit_fraction: f64,
    pub network_drop_rate: f64,
    pub package_power_fraction: f64,
    pub p_cluster_util: f64,
    pub e_cluster_util: f64,
    pub ane_util_fraction: f64,
    pub user_idle_fraction: f64,
    pub foreground_app_hash: u64,
    pub effective_profile_hash: u64,
    pub app_launching: bool,
    pub window_op_active: bool,
    pub foreground_idle: bool,
    pub user_call_in_progress: bool,
    pub user_audio_active: bool,
    pub coreaudio_direct_probe_available: bool,
    pub coreaudio_session_fallback: bool,
    pub markov_prewarm_active: bool,
    pub predictive_agent_active: bool,
    pub kpc_available: bool,
    pub kpc_memory_bound_score: f64,
    pub amx_available: bool,
    pub amx_cs_overhead_ns: u64,
}

impl ActuatorEpisodeContext {
    pub fn from_telemetry(context: &TelemetryContextSummary) -> Self {
        let mut episode = Self {
            valid: true,
            memory_pressure: context.memory_pressure,
            compressor_pressure: context.compressor_pressure,
            thrashing_score: context.thrashing_score,
            cpu_global_usage: context.cpu_global_usage,
            cpu_max_busy: context.cpu_max_busy,
            cpu_pegged_fraction: context.cpu_pegged_fraction,
            stall_fraction: context.stall_fraction,
            used_ram_fraction: context.used_ram_fraction,
            thermal_score: context.thermal_score,
            fluidity_score: context.fluidity_score,
            perceptual_latency_score: context.perceptual_latency_score,
            scheduler_jitter_p95_ms: context.scheduler_jitter_p95_ms,
            windowserver_cpu_fraction: context.windowserver_cpu_fraction,
            arousal_level: context.arousal_level,
            signal_urgency: context.signal_urgency,
            signal_entropy_anomaly: context.signal_entropy_anomaly,
            nars_drift_score: context.nars_drift_score,
            markov_prediction_confidence: context.markov_prediction_confidence,
            network_retransmit_fraction: (context.network_retransmits_per_k / 1_000.0)
                .clamp(0.0, 1.0),
            network_drop_rate: context.network_listen_drop_rate.clamp(0.0, 1.0),
            package_power_fraction: (context.package_watts.unwrap_or(0.0) / 100.0).clamp(0.0, 1.0),
            p_cluster_util: context.p_cluster_util.unwrap_or(context.cpu_global_usage),
            e_cluster_util: context.e_cluster_util.unwrap_or(context.cpu_global_usage),
            ane_util_fraction: (context.ane_util_pct.unwrap_or(0.0) / 100.0).clamp(0.0, 1.0),
            user_idle_fraction: (context.user_idle_secs / 300.0).clamp(0.0, 1.0),
            foreground_app_hash: stable_episode_tag(context.foreground_app.as_deref()),
            effective_profile_hash: stable_episode_tag(Some(&context.effective_profile)),
            app_launching: context.app_launching,
            window_op_active: context.window_op_active,
            foreground_idle: context.foreground_idle,
            user_call_in_progress: context.user_call_in_progress,
            user_audio_active: context.user_audio_active,
            coreaudio_direct_probe_available: context.coreaudio_direct_probe_available,
            coreaudio_session_fallback: context.coreaudio_session_fallback,
            markov_prewarm_active: context.markov_prewarm_active,
            predictive_agent_active: context.predictive_agent_active,
            kpc_available: context.kpc_available,
            kpc_memory_bound_score: context.kpc_memory_bound_score,
            amx_available: context.amx_available,
            amx_cs_overhead_ns: context.amx_cs_overhead_ns,
        };
        episode.valid = episode.is_finite();
        episode
    }

    pub fn is_finite(self) -> bool {
        [
            self.memory_pressure,
            self.compressor_pressure,
            self.thrashing_score,
            self.cpu_global_usage,
            self.cpu_max_busy,
            self.cpu_pegged_fraction,
            self.stall_fraction,
            self.used_ram_fraction,
            self.thermal_score,
            self.fluidity_score,
            self.perceptual_latency_score,
            self.scheduler_jitter_p95_ms,
            self.windowserver_cpu_fraction,
            self.arousal_level,
            self.signal_urgency,
            self.signal_entropy_anomaly,
            self.nars_drift_score,
            self.markov_prediction_confidence,
            self.network_retransmit_fraction,
            self.network_drop_rate,
            self.package_power_fraction,
            self.p_cluster_util,
            self.e_cluster_util,
            self.ane_util_fraction,
            self.user_idle_fraction,
            self.kpc_memory_bound_score,
        ]
        .into_iter()
        .all(f64::is_finite)
    }
}

fn stable_episode_tag(value: Option<&str>) -> u64 {
    value
        .filter(|value| !value.is_empty())
        .map(|value| {
            value.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
                (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3)
            })
        })
        .unwrap_or(0)
}

fn parameter_parent_action_key(action_key: &str) -> Option<&str> {
    let (parent, arm) = action_key.rsplit_once('@')?;
    matches!(arm, "short" | "standard" | "long").then_some(parent)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct PendingActuatorEvidence {
    id: u64,
    #[serde(default)]
    decision_id: Option<DecisionId>,
    family: ActuatorFamily,
    objective: ActuatorObjective,
    action_key: String,
    target: String,
    target_pid: Option<u32>,
    workload: String,
    issued_cycle: u64,
    horizon_cycles: u64,
    cohort_size: u16,
    issued_total_at_start: u64,
    purge_recent: bool,
    event_resolved: bool,
    #[serde(default)]
    gpu_prediction_generation: Option<u64>,
    #[serde(default)]
    attribution: DecisionAttribution,
    #[serde(default)]
    calibration_provenance: CalibrationProvenance,
    #[serde(default)]
    provenance: EvidenceProvenance,
    before: TelemetryContextSummary,
}

#[derive(Debug, Clone)]
struct StagedDecisionEpisode {
    episode: ResolvedDecisionEpisode,
    cohort_size: u16,
}

/// Most arms a microexperiment can have measured at once. Two per open pair.
pub const MAX_LAB_WINDOWS: usize = 64;
pub const MAX_LAB_SAMPLES: usize = 64;
/// Cycles past its horizon that an arm window waits for a closing context
/// before it is abandoned without a sample.
pub const LAB_WINDOW_GRACE_CYCLES: u64 = 12;

/// One open measurement window for a microexperiment arm.
///
/// This is intentionally *not* a `PendingActuatorEvidence`. A lab arm must be
/// measured without being admitted as evidence: it never touches
/// `actuator_issued_total`, `family_stats`, `recent_evidence`, the action
/// models, the calibration store or the causal dynamics model. It reads
/// telemetry and nothing else.
#[derive(Debug, Clone)]
struct LabUtilityWindow {
    decision_id: u64,
    objective: ActuatorObjective,
    opened_cycle: u64,
    horizon_cycles: u64,
    deadline_cycle: u64,
    before: TelemetryContextSummary,
}

/// Raw local utility observed across one arm's window.
///
/// The value is deliberately **not** counterfactual-adjusted. In a paired
/// design the complementary arm *is* the counterfactual, so subtracting a
/// modelled baseline here would discount it twice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabUtilitySample {
    pub decision_id: u64,
    pub utility_micros: i64,
    pub resolved_cycle: u64,
    pub confounded: bool,
    pub quality: f64,
}

/// Outcome objective for the two families the experiment catalog admits.
/// Mirrors `decision_episode_spec` so an arm is scored the same way the
/// corresponding production action would be.
fn lab_objective(family: ActuatorFamily) -> Option<ActuatorObjective> {
    match family {
        ActuatorFamily::InteractionQos => Some(ActuatorObjective::Responsiveness),
        ActuatorFamily::MarkovPrewarm => Some(ActuatorObjective::Prediction),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
struct ExternalActuatorCounters {
    markov_applied: u64,
    markov_hits: u64,
    markov_misses: u64,
    interaction_qos_activations: u64,
    interaction_qos_reverts: u64,
    acceleration_io_promotions: u64,
    chromium_ecore_demotions: u64,
    chromium_purge_hints: u64,
    chromium_jetsam_demotions: u64,
}

impl ExternalActuatorCounters {
    fn from_runtime(runtime: &RuntimeMetrics, _intervention: Intervention) -> Self {
        Self {
            markov_applied: runtime.markov_prewarm_applied,
            markov_hits: runtime.markov_prewarm_hits,
            markov_misses: runtime.markov_prewarm_misses,
            interaction_qos_activations: runtime.interaction_qos_activations,
            interaction_qos_reverts: runtime.interaction_qos_reverts,
            acceleration_io_promotions: runtime.acceleration_lease_io_promotions_total,
            chromium_ecore_demotions: runtime.chromium_ecore_demotions_total,
            chromium_purge_hints: runtime.chromium_purge_hints_total,
            chromium_jetsam_demotions: runtime.chromium_jetsam_demotions_total,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ExternalDeltas {
    markov_applied: u64,
    markov_hits: u64,
    markov_misses: u64,
    interaction_activations: u64,
    interaction_reverts: u64,
    io_promotions: u64,
    chromium_ecore_demotions: u64,
    chromium_purge_hints: u64,
    chromium_jetsam_demotions: u64,
}

impl ExternalDeltas {
    fn suppress(&mut self, family: ActuatorFamily) {
        let counter = match family {
            ActuatorFamily::MarkovPrewarm => &mut self.markov_applied,
            ActuatorFamily::InteractionQos => &mut self.interaction_activations,
            ActuatorFamily::IoShaping => &mut self.io_promotions,
            ActuatorFamily::ChromiumEcore => &mut self.chromium_ecore_demotions,
            ActuatorFamily::ChromiumPurge => &mut self.chromium_purge_hints,
            ActuatorFamily::ChromiumJetsam => &mut self.chromium_jetsam_demotions,
            _ => return,
        };
        *counter = counter.saturating_sub(1);
    }
}

#[derive(Debug)]
struct ActionSpec {
    family: ActuatorFamily,
    objective: ActuatorObjective,
    action_key: String,
    target: String,
    target_pid: Option<u32>,
    horizon_cycles: u64,
    provenance: EvidenceProvenance,
}

impl ActionSpec {
    fn synthetic(
        family: ActuatorFamily,
        objective: ActuatorObjective,
        action_key: &str,
        target: &str,
        horizon_cycles: u64,
    ) -> Self {
        Self {
            family,
            objective,
            action_key: bounded_text(action_key, 320),
            target: bounded_text(target, 256),
            target_pid: None,
            horizon_cycles,
            provenance: EvidenceProvenance::SyntheticCounter,
        }
    }
}

const ALL_OBJECTIVES: [ActuatorObjective; 8] = [
    ActuatorObjective::PressureRelief,
    ActuatorObjective::Responsiveness,
    ActuatorObjective::Efficiency,
    ActuatorObjective::Recovery,
    ActuatorObjective::NetworkHealth,
    ActuatorObjective::Availability,
    ActuatorObjective::Prediction,
    ActuatorObjective::BalancedUtility,
];

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct TelemetryContextSummary {
    pub cycle: u64,
    pub timestamp_unix: i64,
    pub workload: String,
    pub memory_pressure: f64,
    pub memory_pressure_raw: f64,
    pub compressor_pressure: f64,
    pub thrashing_score: f64,
    pub refault_delta_per_sec: f64,
    pub swap_used_bytes: u64,
    pub swap_delta_bytes_per_sec: f64,
    pub cpu_global_usage: f64,
    pub cpu_mean_busy: f64,
    pub cpu_max_busy: f64,
    pub cpu_pegged_fraction: f64,
    pub cpu_core_count: u32,
    pub stall_fraction: f64,
    pub used_ram_fraction: f64,
    pub total_ram_bytes: u64,
    pub used_ram_bytes: u64,
    pub free_ram_bytes: u64,
    pub swap_total_bytes: u64,
    pub process_count: u32,
    pub total_process_rss_bytes: u64,
    pub top_process_cpu: f64,
    pub top_process_rss_bytes: u64,
    pub disk_count: u32,
    pub disk_total_bytes: u64,
    pub disk_available_bytes: u64,
    pub network_count: u32,
    pub network_received_bytes: u64,
    pub network_transmitted_bytes: u64,
    pub thermal_score: f64,
    pub p_cluster_temp_c: Option<f64>,
    pub e_cluster_temp_c: Option<f64>,
    pub gpu_temp_c: Option<f64>,
    pub nand_temp_c: Option<f64>,
    pub temperatures_estimated: bool,
    pub p_cluster_util: Option<f64>,
    pub e_cluster_util: Option<f64>,
    pub package_watts: Option<f64>,
    pub cpu_watts: Option<f64>,
    pub gpu_watts: Option<f64>,
    pub dram_watts: Option<f64>,
    pub ane_watts: Option<f64>,
    pub ane_util_pct: Option<f64>,
    pub battery_percent: Option<u32>,
    pub battery_watts: Option<f64>,
    pub fluidity_score: f64,
    pub perceptual_latency_score: f64,
    pub scheduler_jitter_p95_ms: f64,
    pub windowserver_cpu_fraction: f64,
    pub network_retransmits_per_k: f64,
    pub network_listen_drop_rate: f64,
    pub foreground_app: Option<String>,
    pub foreground_idle: bool,
    pub app_launching: bool,
    pub window_op_active: bool,
    pub user_idle_secs: f64,
    pub user_call_in_progress: bool,
    pub user_audio_active: bool,
    pub coreaudio_direct_probe_available: bool,
    pub coreaudio_session_fallback: bool,
    pub user_has_sleep_assertion: bool,
    pub effective_profile: String,
    pub pressure_total_boost: f64,
    pub pressure_dominant_factor: String,
    pub collector_pressure_alive: bool,
    pub collector_smc_alive: bool,
    pub reactor_healthy: bool,
    pub operation_failures_total: u64,
    pub daemon_is_root: bool,
    pub kernel_taskpolicy_available: bool,
    pub kernel_sysctl_available: bool,
    pub kernel_memorystatus_available: bool,
    pub kernel_pressure_send_available: bool,
    pub p_core_count: u32,
    pub e_core_count: u32,
    pub unavailable_capability_count: u32,
    pub memorystatus_probe_ok: bool,
    pub task_for_pid_probe_ok: bool,
    pub signal_pressure_smooth: f64,
    pub signal_pressure_velocity: f64,
    pub signal_p_oom_30s: f64,
    pub signal_urgency: f64,
    pub signal_entropy_anomaly: f64,
    pub signal_transformer_anomaly: f64,
    pub nars_drift_score: f64,
    pub nars_beliefs_total: u64,
    pub natural_drift: f64,
    pub arousal_level: f64,
    pub boosts_applied: u64,
    pub throttles_applied: u64,
    pub freezes_applied: u64,
    pub paging_hints_applied: u64,
    pub sysctl_applied: u64,
    pub unfreezes_applied: u64,
    pub thread_qos_applied: u64,
    pub markov_prediction_confidence: f64,
    pub markov_prediction_eta_secs: f64,
    pub markov_prewarm_active: bool,
    pub predictive_agent_active: bool,
    pub predictive_intervention: String,
    pub kpc_available: bool,
    pub kpc_memory_bound_score: f64,
    pub amx_available: bool,
    pub amx_cs_overhead_ns: u64,
}

pub struct TelemetryObservation<'a> {
    pub snapshot: &'a SystemSnapshot,
    pub hardware: Option<&'a HardwareSnapshot>,
    pub runtime: &'a RuntimeMetrics,
    pub capabilities: Option<&'a CapabilityReport>,
    pub signal: &'a SignalDigest,
    pub workload: &'a str,
    pub cycle: u64,
    pub outcomes: &'a ExecuteOutcomes,
    pub intervention: Intervention,
    /// Only set when a predictive recommendation changed a threshold or
    /// profile. Process-level interventions are instead derived from their
    /// confirmed execution traces.
    pub applied_intervention: Option<Intervention>,
    pub purge_recent: bool,
    pub nars_drift_score: f64,
    pub nars_beliefs_total: u64,
    pub natural_drift: f64,
    pub arousal_level: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TelemetryMedallionMetrics {
    pub bronze_total: u64,
    pub silver_total: u64,
    pub gold_total: u64,
    pub local_gold_total: u64,
    pub rejected_total: u64,
    pub invalid_total: u64,
    pub non_finite_total: u64,
    pub range_total: u64,
    pub stale_total: u64,
    pub temporal_total: u64,
    pub foreign_total: u64,
    pub coherence_total: u64,
    pub top_rejected_field: Option<ContextField>,
    pub top_rejected_field_total: u64,
    pub last_field_violation: Option<ContextFieldViolation>,
    pub current_tier: ContextTier,
    pub mean_quality: f64,
    pub gold_rate: f64,
    pub actuator_issued_total: u64,
    pub actuator_pending_total: u64,
    pub actuator_bronze_total: u64,
    pub actuator_silver_total: u64,
    pub actuator_gold_total: u64,
    pub actuator_effective_total: u64,
    pub actuator_rejected_total: u64,
    pub actuator_expired_total: u64,
    pub actuator_mean_quality: f64,
    pub actuator_mean_utility: f64,
    /// Action-model capacity accounting. `capacity` is the hard ceiling, so a
    /// `len` that sits on it means every newly observed key destroys a learned
    /// one — the difference between "still maturing" and "repeatedly reborn".
    pub action_model_len: u64,
    pub action_model_capacity: u64,
    pub action_model_evictions_total: u64,
    pub action_model_births_total: u64,
    pub action_model_evidence_updates_total: u64,
    pub action_model_last_evidence_cycle: u64,
    pub apollo_utility_ema: f64,
    pub decision_credit_sources: u64,
    pub decision_credit_leader_score: f64,
    pub decision_credit_leader_accuracy: f64,
    pub decision_credit_leader_observations: u32,
    pub actuator_ready_models: u64,
    pub controlled_holdout_issued_total: u64,
    pub controlled_holdout_pending_total: u64,
    pub controlled_holdout_resolved_total: u64,
    pub controlled_holdout_rejected_total: u64,
    pub controlled_holdout_would_help_total: u64,
    pub controlled_holdout_mean_control_utility: f64,
    pub gpu_prediction_bronze_total: u64,
    pub gpu_prediction_silver_total: u64,
    pub gpu_prediction_gold_total: u64,
    pub gpu_prediction_rejected_total: u64,
    /// Breakdown of `gpu_prediction_rejected_total`, which conflates a
    /// capacity eviction, advice nobody consumed, and a Bronze-tier
    /// calibration refusal. Only the last is a model-quality signal.
    pub gpu_prediction_evicted_total: u64,
    pub gpu_prediction_unused_total: u64,
    pub gpu_prediction_bronze_rejected_total: u64,
    /// Rejections the breakdown above cannot account for.
    ///
    /// `gpu_prediction_rejected_total` is persisted and has accumulated since
    /// the lane existed. The three buckets were added later and, being
    /// `#[serde(default)]`, started from zero on the first restore after that
    /// change — they only classify rejections observed from then on. So the
    /// buckets provably cannot sum to the aggregate, and reporting them as if
    /// they did would misattribute the entire pre-breakdown history to
    /// whichever bucket a reader assumed.
    ///
    /// Publishing the remainder keeps the funnel closed by construction and
    /// makes the unclassifiable tail visible instead of silently absorbed. It
    /// stays flat while the classified buckets grow.
    pub gpu_prediction_unclassified_rejections: u64,
    pub gpu_prediction_pending_total: u64,
    pub gpu_prediction_calibrated_models: u64,
    pub gpu_prediction_mean_absolute_error: f64,
    pub gpu_prediction_mean_brier: f64,
    pub gpu_prediction_mean_quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TelemetryMedallionPersisted {
    #[serde(default)]
    pub actuator_evidence_schema_version: u32,
    #[serde(default)]
    pub context_schema_version: u32,
    #[serde(default)]
    pub installation_id: InstallationId,
    /// Highest locally assigned universal DecisionId observed before the
    /// snapshot, including non-authoritative terminal outcomes.
    #[serde(default)]
    pub decision_id_high_water: u64,
    #[serde(default)]
    pub bronze_total: u64,
    #[serde(default)]
    pub silver_total: u64,
    #[serde(default)]
    pub gold_total: u64,
    #[serde(default)]
    pub rejected_total: u64,
    #[serde(default)]
    pub invalid_total: u64,
    #[serde(default)]
    pub quality_sum: f64,
    #[serde(default)]
    pub last_cycle: u64,
    #[serde(default)]
    pub latest: Option<TelemetryContextSummary>,
    #[serde(default)]
    pending_actions: Vec<PendingActuatorEvidence>,
    #[serde(default)]
    pub family_stats: BTreeMap<ActuatorFamily, ActuatorFamilyStats>,
    #[serde(default)]
    pub action_models: BTreeMap<String, ActionModelStats>,
    #[serde(default)]
    pub action_model_evictions_total: u64,
    #[serde(default)]
    pub action_model_births_total: u64,
    #[serde(default)]
    pub action_model_evidence_updates_total: u64,
    #[serde(default)]
    pub action_model_last_evidence_cycle: u64,
    #[serde(default, deserialize_with = "deserialize_recent_evidence")]
    pub recent_evidence: Vec<ResolvedActuatorEvidence>,
    #[serde(default, deserialize_with = "deserialize_episodic_evidence")]
    pub episodic_evidence: Vec<ResolvedActuatorEvidence>,
    #[serde(default)]
    external_counters: ExternalActuatorCounters,
    #[serde(default)]
    pub next_action_id: u64,
    #[serde(default)]
    pub actuator_issued_total: u64,
    #[serde(default)]
    pub actuator_resolved_total: u64,
    #[serde(default)]
    pub actuator_silver_total: u64,
    #[serde(default)]
    pub actuator_gold_total: u64,
    #[serde(default)]
    pub actuator_effective_total: u64,
    #[serde(default)]
    pub actuator_rejected_total: u64,
    #[serde(default)]
    pub actuator_expired_total: u64,
    #[serde(default)]
    pub actuator_quality_sum: f64,
    #[serde(default)]
    pub actuator_utility_sum: f64,
    #[serde(default)]
    pub apollo_utility_ema: f64,
    #[serde(default)]
    pub apollo_utility_observations: u64,
    #[serde(default)]
    pub decision_source_stats: BTreeMap<String, DecisionSourceStats>,
    #[serde(default)]
    pub model_calibration: Option<ModelCalibrationPersisted>,
    #[serde(default)]
    pub no_action_delta_ema: BTreeMap<ActuatorObjective, f64>,
    #[serde(default)]
    pub no_action_state_delta_ema: WorldStateDelta,
    #[serde(default)]
    pub controlled_models: BTreeMap<String, ControlledCounterfactualStats>,
    #[serde(default)]
    pub controlled_holdout_issued_total: u64,
    #[serde(default)]
    pub controlled_holdout_resolved_total: u64,
    #[serde(default)]
    pub controlled_holdout_rejected_total: u64,
    #[serde(default)]
    pub controlled_holdout_pending_total: u64,
    #[serde(default)]
    pub causal_dynamics: CausalDynamicsModel,
    #[serde(default)]
    pub gpu_prediction_schema_version: u32,
    #[serde(default)]
    pub gpu_predictions: Vec<GpuPredictionEvidence>,
    #[serde(default)]
    pub gpu_calibration_models: BTreeMap<String, GpuCalibrationStats>,
    #[serde(default)]
    pub gpu_prediction_bronze_total: u64,
    #[serde(default)]
    pub gpu_prediction_silver_total: u64,
    #[serde(default)]
    pub gpu_prediction_gold_total: u64,
    #[serde(default)]
    pub gpu_prediction_rejected_total: u64,
    #[serde(default)]
    pub gpu_prediction_evicted_total: u64,
    #[serde(default)]
    pub gpu_prediction_unused_total: u64,
    #[serde(default)]
    pub gpu_prediction_bronze_rejected_total: u64,
}

struct BoundedEvidenceVisitor<const MAX: usize>(PhantomData<ResolvedActuatorEvidence>);

impl<'de, const MAX: usize> Visitor<'de> for BoundedEvidenceVisitor<MAX> {
    type Value = Vec<ResolvedActuatorEvidence>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "at most {MAX} retained actuator evidence records"
        )
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::with_capacity(MAX.min(sequence.size_hint().unwrap_or(MAX)));
        while values.len() < MAX {
            let Some(value) = sequence.next_element()? else {
                return Ok(values);
            };
            values.push(value);
        }
        while sequence.next_element::<IgnoredAny>()?.is_some() {}
        Ok(values)
    }
}

fn deserialize_recent_evidence<'de, D>(
    deserializer: D,
) -> Result<Vec<ResolvedActuatorEvidence>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(BoundedEvidenceVisitor::<MAX_RECENT_EVIDENCE>(PhantomData))
}

fn deserialize_episodic_evidence<'de, D>(
    deserializer: D,
) -> Result<Vec<ResolvedActuatorEvidence>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(BoundedEvidenceVisitor::<MAX_EPISODIC_EVIDENCE>(PhantomData))
}

#[derive(Debug)]
pub struct TelemetryMedallion {
    installation_id: InstallationId,
    live_hardware_regime: HardwareRegime,
    decision_id_high_water: u64,
    current_tier: ContextTier,
    last_admitted_live: Option<TelemetryContextSummary>,
    consecutive_gold: u32,
    local_gold_total: u64,
    reason_counters: ContextReasonCounters,
    field_rejection_counters: BTreeMap<ContextField, u64>,
    last_field_violation: Option<ContextFieldViolation>,
    bronze_total: u64,
    silver_total: u64,
    gold_total: u64,
    rejected_total: u64,
    invalid_total: u64,
    quality_sum: f64,
    last_cycle: u64,
    latest: Option<TelemetryContextSummary>,
    pending_actions: VecDeque<PendingActuatorEvidence>,
    /// Microexperiment arm windows. Deliberately separate from
    /// `pending_actions`: a lab arm is measured, never admitted as evidence.
    lab_windows: VecDeque<LabUtilityWindow>,
    lab_samples: VecDeque<LabUtilitySample>,
    lab_windows_expired_total: u64,
    family_stats: BTreeMap<ActuatorFamily, ActuatorFamilyStats>,
    action_models: BTreeMap<String, ActionModelStats>,
    action_models_revision: u64,
    /// Capacity accounting for `action_models`. Eviction at `MAX_ACTION_MODELS`
    /// destroys learned evidence, so it is counted rather than silent: a reborn
    /// key is otherwise indistinguishable from one patiently maturing.
    action_model_evictions_total: u64,
    action_model_births_total: u64,
    action_model_evidence_updates_total: u64,
    action_model_last_evidence_cycle: u64,
    recent_evidence: VecDeque<ResolvedActuatorEvidence>,
    episodic_evidence: VecDeque<ResolvedActuatorEvidence>,
    new_gold_evidence: VecDeque<ResolvedActuatorEvidence>,
    external_counters: ExternalActuatorCounters,
    next_action_id: u64,
    actuator_issued_total: u64,
    actuator_resolved_total: u64,
    actuator_silver_total: u64,
    actuator_gold_total: u64,
    actuator_effective_total: u64,
    actuator_rejected_total: u64,
    actuator_expired_total: u64,
    actuator_quality_sum: f64,
    actuator_utility_sum: f64,
    apollo_utility_ema: f64,
    apollo_utility_observations: u64,
    decision_source_stats: BTreeMap<String, DecisionSourceStats>,
    model_calibration: ModelCalibrationStore,
    quarantined_future_state: Option<Box<TelemetryMedallionPersisted>>,
    staged_attributions: BTreeMap<String, VecDeque<DecisionAttribution>>,
    staged_decision_episodes: VecDeque<StagedDecisionEpisode>,
    no_action_delta_ema: BTreeMap<ActuatorObjective, f64>,
    no_action_state_delta_ema: WorldStateDelta,
    pending_controlled_holdouts: VecDeque<PendingControlledHoldout>,
    controlled_models: BTreeMap<String, ControlledCounterfactualStats>,
    controlled_models_revision: u64,
    controlled_holdout_issued_total: u64,
    controlled_holdout_resolved_total: u64,
    controlled_holdout_rejected_total: u64,
    causal_dynamics: CausalDynamicsModel,
    gpu_predictions: VecDeque<GpuPredictionEvidence>,
    gpu_calibration_models: BTreeMap<String, GpuCalibrationStats>,
    gpu_same_cycle_consumed: BTreeSet<(u64, String, String)>,
    gpu_calibration_revision: u64,
    gpu_prediction_bronze_total: u64,
    gpu_prediction_silver_total: u64,
    gpu_prediction_gold_total: u64,
    gpu_prediction_rejected_total: u64,
    gpu_prediction_evicted_total: u64,
    gpu_prediction_unused_total: u64,
    gpu_prediction_bronze_rejected_total: u64,
}

impl Default for TelemetryMedallion {
    fn default() -> Self {
        Self {
            installation_id: InstallationId::UNKNOWN,
            live_hardware_regime: HardwareRegime::default(),
            decision_id_high_water: 0,
            current_tier: ContextTier::Rejected,
            last_admitted_live: None,
            consecutive_gold: 0,
            local_gold_total: 0,
            reason_counters: ContextReasonCounters::default(),
            field_rejection_counters: BTreeMap::new(),
            last_field_violation: None,
            bronze_total: 0,
            silver_total: 0,
            gold_total: 0,
            rejected_total: 0,
            invalid_total: 0,
            quality_sum: 0.0,
            last_cycle: 0,
            latest: None,
            pending_actions: VecDeque::new(),
            lab_windows: VecDeque::new(),
            lab_samples: VecDeque::new(),
            lab_windows_expired_total: 0,
            family_stats: BTreeMap::new(),
            action_models: BTreeMap::new(),
            action_models_revision: 0,
            action_model_evictions_total: 0,
            action_model_births_total: 0,
            action_model_evidence_updates_total: 0,
            action_model_last_evidence_cycle: 0,
            recent_evidence: VecDeque::new(),
            episodic_evidence: VecDeque::new(),
            new_gold_evidence: VecDeque::new(),
            external_counters: ExternalActuatorCounters::default(),
            next_action_id: 0,
            actuator_issued_total: 0,
            actuator_resolved_total: 0,
            actuator_silver_total: 0,
            actuator_gold_total: 0,
            actuator_effective_total: 0,
            actuator_rejected_total: 0,
            actuator_expired_total: 0,
            actuator_quality_sum: 0.0,
            actuator_utility_sum: 0.0,
            apollo_utility_ema: 0.0,
            apollo_utility_observations: 0,
            decision_source_stats: BTreeMap::new(),
            model_calibration: ModelCalibrationStore::new(InstallationId::UNKNOWN),
            quarantined_future_state: None,
            staged_attributions: BTreeMap::new(),
            staged_decision_episodes: VecDeque::new(),
            no_action_delta_ema: BTreeMap::new(),
            no_action_state_delta_ema: WorldStateDelta::default(),
            pending_controlled_holdouts: VecDeque::new(),
            controlled_models: BTreeMap::new(),
            controlled_models_revision: 0,
            controlled_holdout_issued_total: 0,
            controlled_holdout_resolved_total: 0,
            controlled_holdout_rejected_total: 0,
            causal_dynamics: CausalDynamicsModel::default(),
            gpu_predictions: VecDeque::new(),
            gpu_calibration_models: BTreeMap::new(),
            gpu_same_cycle_consumed: BTreeSet::new(),
            gpu_calibration_revision: 0,
            gpu_prediction_bronze_total: 0,
            gpu_prediction_silver_total: 0,
            gpu_prediction_gold_total: 0,
            gpu_prediction_rejected_total: 0,
            gpu_prediction_evicted_total: 0,
            gpu_prediction_unused_total: 0,
            gpu_prediction_bronze_rejected_total: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct TrustedTelemetryView<'a> {
    pub current: Option<&'a TelemetryContextSummary>,
    pub installation_id: InstallationId,
    pub action_models: &'a BTreeMap<String, ActionModelStats>,
    pub action_models_revision: u64,
    pub controlled_models: &'a BTreeMap<String, ControlledCounterfactualStats>,
    pub controlled_models_revision: u64,
    pub episodic_evidence: &'a VecDeque<ResolvedActuatorEvidence>,
    pub causal_dynamics: &'a CausalDynamicsModel,
    pub causal_dynamics_revision: u64,
    pub gpu_calibration_models: &'a BTreeMap<String, GpuCalibrationStats>,
    pub gpu_calibration_revision: u64,
    pub metrics: TelemetryMedallionMetrics,
}

impl TelemetryMedallion {
    pub fn new(installation_id: InstallationId) -> Self {
        Self {
            installation_id,
            decision_id_high_water: 0,
            causal_dynamics: CausalDynamicsModel::new(installation_id),
            model_calibration: ModelCalibrationStore::new(installation_id),
            ..Self::default()
        }
    }

    pub fn bind_live_hardware(&mut self, hardware_regime: HardwareRegime) {
        self.live_hardware_regime = hardware_regime;
    }

    /// Admit compact GPU forecasts as Bronze evidence. The Monte Carlo sample
    /// buffer is intentionally discarded by the worker; only summaries needed
    /// for later causal calibration cross the medallion boundary.
    pub fn observe_gpu_imagination(&mut self, result: &GpuImaginationResult, cycle: u64) -> u64 {
        self.expire_unused_gpu_predictions(cycle);
        if result.error.is_some() || result.candidates.is_empty() {
            return 0;
        }
        let hardware_regime = self
            .latest
            .as_ref()
            .map(HardwareRegime::from_context)
            .unwrap_or_default();
        let per_candidate_samples = result.samples / result.candidates.len() as u64;
        let per_candidate_gpu_ns = result.gpu_time_ns / result.candidates.len() as u64;
        let mut admitted = 0_u64;
        for candidate in &result.candidates {
            let finite = [
                candidate.expected_gain,
                candidate.uncertainty,
                candidate.mean_gain,
                candidate.p10_gain,
                candidate.positive_probability,
                candidate.rank_support,
                candidate.context_score,
            ]
            .into_iter()
            .all(f64::is_finite);
            if candidate.action_key.is_empty()
                || result.workload.is_empty()
                || !finite
                || self.gpu_predictions.iter().any(|prediction| {
                    prediction.generation == result.generation
                        && prediction.action_key == candidate.action_key
                        && prediction.workload == result.workload
                })
            {
                continue;
            }
            if self.gpu_predictions.len() >= MAX_GPU_PREDICTIONS
                && self
                    .gpu_predictions
                    .pop_front()
                    .is_some_and(|prediction| prediction.resolved_cycle.is_none())
            {
                self.gpu_prediction_rejected_total =
                    self.gpu_prediction_rejected_total.saturating_add(1);
                self.gpu_prediction_evicted_total =
                    self.gpu_prediction_evicted_total.saturating_add(1);
            }
            self.gpu_predictions.push_back(GpuPredictionEvidence {
                generation: result.generation,
                action_key: bounded_text(&candidate.action_key, 320),
                workload: bounded_text(&result.workload, 64),
                issued_cycle: cycle,
                expected_gain: candidate.expected_gain.clamp(-1.0, 1.0),
                uncertainty: candidate.uncertainty.clamp(0.0, 1.0),
                mean_gain: candidate.mean_gain.clamp(-1.0, 1.0),
                p10_gain: candidate.p10_gain.clamp(-1.0, 1.0),
                positive_probability: candidate.positive_probability.clamp(0.0, 1.0),
                rank_support: candidate.rank_support.clamp(-0.005, 0.005),
                context_score: candidate.context_score.clamp(-0.08, 0.08),
                samples: per_candidate_samples,
                gpu_time_ns: per_candidate_gpu_ns,
                used_cycle: None,
                resolved_cycle: None,
                tier: EvidenceTier::Bronze,
                actual_utility: None,
                absolute_error: None,
                brier_score: None,
                p10_covered: None,
                quality: 0.0,
                hardware_regime,
                installation_id: self.installation_id,
            });
            self.gpu_prediction_bronze_total = self.gpu_prediction_bronze_total.saturating_add(1);
            self.update_gpu_prediction_count(
                &candidate.action_key,
                &result.workload,
                hardware_regime,
                cycle,
            );
            admitted = admitted.saturating_add(1);
        }
        admitted
    }

    /// Rejections that predate the per-reason breakdown, so the funnel is
    /// closed by construction: evicted + unused + bronze_rejected +
    /// unclassified == rejected_total, always.
    ///
    /// Saturating rather than asserting: the restore path clamps each bucket
    /// to the aggregate independently, so a hostile or truncated checkpoint
    /// must not be able to panic the daemon here.
    fn gpu_prediction_unclassified_rejections(&self) -> u64 {
        self.gpu_prediction_rejected_total
            .saturating_sub(self.gpu_prediction_evicted_total)
            .saturating_sub(self.gpu_prediction_unused_total)
            .saturating_sub(self.gpu_prediction_bronze_rejected_total)
    }

    fn expire_unused_gpu_predictions(&mut self, cycle: u64) {
        self.gpu_same_cycle_consumed
            .retain(|(authorized_cycle, _, _)| *authorized_cycle >= cycle);
        for prediction in self.gpu_predictions.iter_mut().filter(|prediction| {
            prediction.used_cycle.is_none()
                && prediction.resolved_cycle.is_none()
                && cycle.saturating_sub(prediction.issued_cycle)
                    > GPU_PREDICTION_MATCH_MAX_AGE_CYCLES
        }) {
            prediction.resolved_cycle = Some(cycle);
            self.gpu_prediction_rejected_total =
                self.gpu_prediction_rejected_total.saturating_add(1);
            self.gpu_prediction_unused_total = self.gpu_prediction_unused_total.saturating_add(1);
        }
    }

    fn calibration_keys(action_key: &str, workload: &str) -> [String; 2] {
        [
            gpu_calibration_key(action_key, workload),
            gpu_calibration_key(action_key, "*"),
        ]
    }

    fn ensure_gpu_calibration_capacity(&mut self, keys: &[String]) {
        let missing = keys
            .iter()
            .filter(|key| !self.gpu_calibration_models.contains_key(*key))
            .count();
        while self.gpu_calibration_models.len().saturating_add(missing) > MAX_GPU_CALIBRATION_MODELS
        {
            let Some(evict) = self
                .gpu_calibration_models
                .iter()
                .filter(|(key, _)| !keys.contains(key))
                .min_by_key(|(_, stats)| (stats.gold, stats.last_cycle))
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            self.gpu_calibration_models.remove(&evict);
        }
    }

    fn update_gpu_prediction_count(
        &mut self,
        action_key: &str,
        workload: &str,
        hardware_regime: HardwareRegime,
        cycle: u64,
    ) {
        let keys = Self::calibration_keys(action_key, workload);
        self.ensure_gpu_calibration_capacity(&keys);
        for key in keys {
            let stats = self.gpu_calibration_models.entry(key).or_default();
            stats.predictions = stats.predictions.saturating_add(1);
            stats.last_cycle = cycle;
            stats.hardware_regime = hardware_regime;
            stats.installation_id = self.installation_id;
        }
        self.gpu_calibration_revision = self.gpu_calibration_revision.wrapping_add(1);
    }

    fn mark_gpu_prediction_used(
        &mut self,
        action_key: &str,
        workload: &str,
        cycle: u64,
    ) -> Option<u64> {
        let same_cycle_authorized = self.gpu_same_cycle_consumed.iter().any(
            |(authorized_cycle, authorized_action, authorized_workload)| {
                *authorized_cycle == cycle
                    && authorized_workload == workload
                    && gpu_action_matches(authorized_action, action_key)
            },
        );
        let prediction = self.gpu_predictions.iter_mut().rev().find(|prediction| {
            gpu_action_matches(&prediction.action_key, action_key)
                && prediction.workload == workload
                && prediction.resolved_cycle.is_none()
                && (cycle > prediction.issued_cycle || same_cycle_authorized)
                && cycle.saturating_sub(prediction.issued_cycle)
                    <= GPU_PREDICTION_MATCH_MAX_AGE_CYCLES
                && (prediction.used_cycle.is_none() || prediction.used_cycle == Some(cycle))
        })?;
        let newly_used = prediction.used_cycle.is_none();
        if newly_used {
            prediction.used_cycle = Some(cycle);
            prediction.tier = EvidenceTier::Silver;
        }
        let generation = prediction.generation;
        if newly_used {
            let calibration_action_key = prediction.action_key.clone();
            let keys = Self::calibration_keys(&calibration_action_key, workload);
            for key in keys {
                if let Some(stats) = self.gpu_calibration_models.get_mut(&key) {
                    stats.used = stats.used.saturating_add(1);
                    stats.last_cycle = cycle;
                }
            }
            self.gpu_prediction_silver_total = self.gpu_prediction_silver_total.saturating_add(1);
            self.gpu_calibration_revision = self.gpu_calibration_revision.wrapping_add(1);
            self.gpu_same_cycle_consumed.retain(
                |(authorized_cycle, authorized_action, authorized_workload)| {
                    !(*authorized_cycle == cycle
                        && authorized_workload == workload
                        && gpu_action_matches(authorized_action, action_key))
                },
            );
        }
        Some(generation)
    }

    /// Record an action key the central planner demonstrably ranked with the
    /// newly completed GPU batch. External lanes do not call this: they mark
    /// use naturally on a later cycle after reading World Model context.
    pub fn mark_gpu_prediction_consumed(
        &mut self,
        action_key: &str,
        workload: &str,
        cycle: u64,
    ) -> bool {
        let Some(_prediction) = self.gpu_predictions.iter().rev().find(|prediction| {
            gpu_action_matches(&prediction.action_key, action_key)
                && prediction.workload == workload
                && prediction.issued_cycle == cycle
                && prediction.resolved_cycle.is_none()
        }) else {
            return false;
        };
        self.gpu_same_cycle_consumed.insert((
            cycle,
            bounded_text(action_key, 320),
            bounded_text(workload, 64),
        ));
        true
    }

    fn resolve_gpu_prediction(
        &mut self,
        generation: Option<u64>,
        evidence: &ResolvedActuatorEvidence,
    ) {
        let Some(generation) = generation else {
            return;
        };
        let Some(index) = self.gpu_predictions.iter().position(|prediction| {
            prediction.generation == generation
                && gpu_action_matches(&prediction.action_key, &evidence.action_key)
                && prediction.workload == evidence.workload
                && prediction.resolved_cycle.is_none()
        }) else {
            return;
        };
        let prediction = &mut self.gpu_predictions[index];
        let calibration_action_key = prediction.action_key.clone();
        prediction.resolved_cycle = Some(evidence.resolved_cycle);
        prediction.actual_utility = Some(evidence.net_utility_delta);
        prediction.quality = evidence.quality;
        let calibration_tier = if evidence.provenance == EvidenceProvenance::SyntheticCounter
            && evidence.quality >= 0.85
            && evidence.confounder_count == 0
        {
            // A measured counter endpoint may calibrate bounded GPU advice,
            // but remains Bronze in the universal/action-authority lane.
            EvidenceTier::Gold
        } else {
            evidence.tier
        };
        if calibration_tier == EvidenceTier::Bronze {
            self.gpu_prediction_rejected_total =
                self.gpu_prediction_rejected_total.saturating_add(1);
            self.gpu_prediction_bronze_rejected_total =
                self.gpu_prediction_bronze_rejected_total.saturating_add(1);
            return;
        }
        let signed_error = evidence.net_utility_delta - prediction.mean_gain;
        let absolute_error = signed_error.abs().clamp(0.0, 2.0);
        let observed_positive = u8::from(evidence.net_utility_delta > 0.0) as f64;
        let brier = (prediction.positive_probability - observed_positive)
            .powi(2)
            .clamp(0.0, 1.0);
        let p10_covered = evidence.net_utility_delta >= prediction.p10_gain;
        prediction.absolute_error = Some(absolute_error);
        prediction.brier_score = Some(brier);
        prediction.p10_covered = Some(p10_covered);
        prediction.tier = calibration_tier;
        if calibration_tier != EvidenceTier::Gold {
            return;
        }
        self.gpu_prediction_gold_total = self.gpu_prediction_gold_total.saturating_add(1);
        let calibration_quality =
            (evidence.quality * (1.0 - absolute_error.min(1.0)) * (1.0 - brier)).clamp(0.0, 1.0);
        let keys = Self::calibration_keys(&calibration_action_key, &evidence.workload);
        for key in keys {
            let stats = self.gpu_calibration_models.entry(key).or_default();
            let alpha = if stats.resolved == 0 { 1.0 } else { 0.20 };
            stats.resolved = stats.resolved.saturating_add(1);
            stats.gold = stats.gold.saturating_add(1);
            stats.signed_error_ema = alpha * signed_error + (1.0 - alpha) * stats.signed_error_ema;
            stats.absolute_error_ema =
                alpha * absolute_error + (1.0 - alpha) * stats.absolute_error_ema;
            stats.brier_ema = alpha * brier + (1.0 - alpha) * stats.brier_ema;
            stats.p10_coverage_ema =
                alpha * u8::from(p10_covered) as f64 + (1.0 - alpha) * stats.p10_coverage_ema;
            stats.quality_ema = alpha * calibration_quality + (1.0 - alpha) * stats.quality_ema;
            stats.evidence_mass = (stats.evidence_mass + evidence.quality).clamp(0.0, 64.0);
            stats.last_cycle = evidence.resolved_cycle;
            stats.last_observed_unix = evidence.resolved_timestamp_unix;
            stats.hardware_regime = evidence.hardware_regime;
            stats.installation_id = evidence.installation_id;
        }
        self.gpu_calibration_revision = self.gpu_calibration_revision.wrapping_add(1);
    }

    /// Admit one complete live snapshot. Bronze always advances for a cycle;
    /// Silver/Gold only advance when the normalized measurements are usable.
    pub fn observe(&mut self, observation: TelemetryObservation<'_>) -> ContextAdmission {
        self.expire_unused_gpu_predictions(observation.cycle);
        let summary = summarize(&observation);
        self.model_calibration
            .validate_hardware(HardwareRegime::from_context(&summary));
        let TelemetryObservation {
            snapshot,
            runtime,
            workload: _,
            cycle,
            outcomes,
            intervention,
            applied_intervention,
            purge_recent,
            ..
        } = observation;
        self.bronze_total = self.bronze_total.saturating_add(1);
        let admission = classify(ContextAdmissionInput::live(
            &summary,
            self.last_admitted_live.as_ref(),
            Utc::now().timestamp(),
            self.installation_id,
        ));
        self.record_admission(admission);

        if admission.tier != ContextTier::Gold {
            self.current_tier = admission.tier;
            self.consecutive_gold = 0;
            if admission.tier == ContextTier::Silver {
                self.last_admitted_live = Some(summary);
            }
            self.staged_attributions.clear();
            self.staged_decision_episodes.clear();
            return admission;
        }

        self.current_tier = ContextTier::Gold;
        self.consecutive_gold = self.consecutive_gold.saturating_add(1);
        self.local_gold_total = self.local_gold_total.saturating_add(1);
        self.last_cycle = cycle;

        let applied_root_actions: Vec<&RootAction> = outcomes
            .audit_traces
            .iter()
            .filter(|trace| trace.applied)
            .map(|trace| &trace.intended_action)
            .collect();
        let applied_families: Vec<ActuatorFamily> = applied_root_actions
            .iter()
            .filter_map(|action| action_spec(action).map(|spec| spec.family))
            .collect();
        self.resolve_controlled_holdouts(
            &summary,
            cycle,
            purge_recent,
            &applied_families,
            admission.quality,
        );
        let mut external_deltas = self.external_deltas(runtime);
        let staged_before = self.latest.clone().unwrap_or_else(|| summary.clone());
        let staged_issued =
            self.issue_staged_decisions(&staged_before, cycle, purge_recent, &mut external_deltas);
        let resolved_this_cycle = self.resolve_pending(
            &summary,
            snapshot,
            cycle,
            external_deltas.markov_hits,
            external_deltas.markov_misses,
            external_deltas.interaction_reverts,
        );
        // Measured against the same admitted context as production evidence,
        // but kept out of every learning aggregate.
        self.resolve_lab_windows(&summary, cycle);

        let root_cohort_size = applied_root_actions.len();
        let mut issued_this_cycle = (root_cohort_size as u64).saturating_add(staged_issued);
        issued_this_cycle = issued_this_cycle
            .saturating_add(external_deltas.markov_applied)
            .saturating_add(external_deltas.interaction_activations)
            .saturating_add(external_deltas.io_promotions)
            .saturating_add(external_deltas.chromium_ecore_demotions)
            .saturating_add(external_deltas.chromium_purge_hints)
            .saturating_add(external_deltas.chromium_jetsam_demotions)
            .saturating_add(u64::from(applied_intervention.is_some()));

        if self.consecutive_gold >= 2
            && issued_this_cycle == 0
            && resolved_this_cycle == 0
            && self.pending_actions.is_empty()
        {
            if let Some(previous) = self.latest.as_ref() {
                self.causal_dynamics.observe_no_action(
                    previous,
                    &summary,
                    admission.quality,
                    self.installation_id,
                );
                self.no_action_state_delta_ema = self
                    .no_action_state_delta_ema
                    .ema(WorldStateDelta::between(previous, &summary), 0.05)
                    .clamped(-0.05, 0.05);
                for objective in ALL_OBJECTIVES {
                    let delta =
                        utility_score(objective, &summary) - utility_score(objective, previous);
                    let baseline = self.no_action_delta_ema.entry(objective).or_insert(delta);
                    *baseline = (0.05 * delta + 0.95 * *baseline).clamp(-0.05, 0.05);
                }
            }
        }

        let cohort_size = issued_this_cycle.min(u16::MAX as u64) as u16;
        let mut coordinated_members = Vec::with_capacity(root_cohort_size.saturating_add(5));
        for action in applied_root_actions {
            if let Some(spec) = action_spec(action) {
                coordinated_members.push(spec.action_key.clone());
                self.issue(
                    spec,
                    &summary,
                    cycle,
                    cohort_size.max(1),
                    purge_recent,
                    false,
                );
            }
        }
        if external_deltas.markov_applied > 0 {
            coordinated_members.push("markov_prewarm:predicted_app".to_string());
        }
        for _ in 0..external_deltas.markov_applied.min(16) {
            self.issue(
                ActionSpec::synthetic(
                    ActuatorFamily::MarkovPrewarm,
                    ActuatorObjective::Prediction,
                    "markov_prewarm:predicted_app",
                    runtime.markov_prediction_app.as_str(),
                    120,
                ),
                &summary,
                cycle,
                cohort_size.max(1),
                purge_recent,
                true,
            );
        }
        let interaction_action_key = match (
            runtime.interaction_qos_ttl_exploratory,
            runtime.interaction_qos_ttl_band.as_str(),
        ) {
            (true, band @ ("short" | "standard" | "long")) => {
                format!("interaction_qos:foreground@{band}")
            }
            _ => "interaction_qos:foreground".to_string(),
        };
        if external_deltas.interaction_activations > 0 {
            coordinated_members.push(interaction_action_key.clone());
        }
        for _ in 0..external_deltas.interaction_activations.min(16) {
            self.issue(
                ActionSpec::synthetic(
                    ActuatorFamily::InteractionQos,
                    ActuatorObjective::Responsiveness,
                    &interaction_action_key,
                    runtime.interaction_qos_reason.as_str(),
                    30,
                ),
                &summary,
                cycle,
                cohort_size.max(1),
                purge_recent,
                true,
            );
        }
        if external_deltas.io_promotions > 0 {
            coordinated_members.push("io_shaping:interactive_release".to_string());
        }
        for _ in 0..external_deltas.io_promotions.min(16) {
            self.issue(
                ActionSpec::synthetic(
                    ActuatorFamily::IoShaping,
                    ActuatorObjective::Responsiveness,
                    "io_shaping:interactive_release",
                    runtime.acceleration_lease_last_family.as_str(),
                    30,
                ),
                &summary,
                cycle,
                cohort_size.max(1),
                purge_recent,
                false,
            );
        }
        if external_deltas.chromium_ecore_demotions > 0 {
            coordinated_members.push("chromium_ecore:background_renderer".to_string());
        }
        for _ in 0..external_deltas.chromium_ecore_demotions.min(16) {
            self.issue(
                ActionSpec::synthetic(
                    ActuatorFamily::ChromiumEcore,
                    ActuatorObjective::Efficiency,
                    "chromium_ecore:background_renderer",
                    "background_renderer",
                    30,
                ),
                &summary,
                cycle,
                cohort_size.max(1),
                purge_recent,
                false,
            );
        }
        if external_deltas.chromium_purge_hints > 0 {
            coordinated_members.push("chromium_purge:purgeable_renderer".to_string());
        }
        for _ in 0..external_deltas.chromium_purge_hints.min(16) {
            self.issue(
                ActionSpec::synthetic(
                    ActuatorFamily::ChromiumPurge,
                    ActuatorObjective::PressureRelief,
                    "chromium_purge:purgeable_renderer",
                    "background_renderer",
                    12,
                ),
                &summary,
                cycle,
                cohort_size.max(1),
                purge_recent,
                false,
            );
        }
        if external_deltas.chromium_jetsam_demotions > 0 {
            coordinated_members.push("chromium_jetsam:background_renderer".to_string());
        }
        for _ in 0..external_deltas.chromium_jetsam_demotions.min(16) {
            self.issue(
                ActionSpec::synthetic(
                    ActuatorFamily::ChromiumJetsam,
                    ActuatorObjective::Efficiency,
                    "chromium_jetsam:background_renderer",
                    "background_renderer",
                    30,
                ),
                &summary,
                cycle,
                cohort_size.max(1),
                purge_recent,
                false,
            );
        }
        if let Some(spec) = applied_intervention.and_then(intervention_spec) {
            coordinated_members.push(spec.action_key.clone());
            self.issue(
                spec,
                &summary,
                cycle,
                cohort_size.max(1),
                purge_recent,
                true,
            );
        }
        coordinated_members.sort();
        coordinated_members.dedup();
        if coordinated_members.len() > 1 {
            let family_key = coordinated_members
                .iter()
                .filter_map(|key| key.split_once(':').map(|(family, _)| family))
                .collect::<Vec<_>>()
                .join("+");
            let target = bounded_text(&coordinated_members.join("|"), 256);
            let mut coordinated = ActionSpec::synthetic(
                ActuatorFamily::Coordinated,
                ActuatorObjective::BalancedUtility,
                &format!("coordinated:{family_key}"),
                &target,
                8,
            );
            if root_cohort_size > 0 {
                coordinated.provenance = EvidenceProvenance::ObservedLocal;
            }
            self.issue(coordinated, &summary, cycle, 1, purge_recent, false);
        }
        let cohort_end_total = self.actuator_issued_total;
        for pending in self.pending_actions.iter_mut().rev() {
            if pending.issued_cycle != cycle {
                break;
            }
            pending.issued_total_at_start = cohort_end_total;
        }
        self.external_counters = ExternalActuatorCounters::from_runtime(runtime, intervention);

        self.live_hardware_regime = HardwareRegime::from_context(&summary);
        self.latest = Some(summary.clone());
        self.last_admitted_live = Some(summary);
        self.staged_attributions.clear();
        admission
    }

    fn record_admission(&mut self, admission: ContextAdmission) {
        self.reason_counters.record(admission.reasons);
        for field in admission.violating_fields.iter() {
            let counter = self.field_rejection_counters.entry(field).or_default();
            *counter = counter.saturating_add(1);
        }
        if let Some(violation) = admission.primary_violation {
            self.last_field_violation = Some(violation);
        }
        match admission.tier {
            ContextTier::Rejected => {
                self.rejected_total = self.rejected_total.saturating_add(1);
                self.invalid_total = self.invalid_total.saturating_add(1);
            }
            ContextTier::Silver => {
                self.quality_sum = (self.quality_sum + admission.quality).max(0.0);
                self.silver_total = self.silver_total.saturating_add(1);
            }
            ContextTier::Gold => {
                self.quality_sum = (self.quality_sum + admission.quality).max(0.0);
                self.gold_total = self.gold_total.saturating_add(1);
            }
        }
    }

    /// Register a safe, dispatcher-selected control arm. The action is not
    /// executed; its next Gold context becomes targeted no-action evidence.
    pub fn issue_controlled_holdout(
        &mut self,
        action: &RootAction,
        workload: &str,
        cycle: u64,
    ) -> bool {
        if self.current_tier != ContextTier::Gold {
            return false;
        }
        let Some(before) = self.latest.clone() else {
            return false;
        };
        let Some(spec) = action_spec(action) else {
            return false;
        };
        if spec.family != ActuatorFamily::Boost {
            return false;
        }
        if self.pending_controlled_holdouts.len() >= MAX_PENDING_CONTROLLED_HOLDOUTS {
            self.pending_controlled_holdouts.pop_front();
            self.controlled_holdout_rejected_total =
                self.controlled_holdout_rejected_total.saturating_add(1);
        }
        self.pending_controlled_holdouts
            .push_back(PendingControlledHoldout {
                action_key: spec.action_key,
                workload: bounded_text(workload, 64),
                family: spec.family,
                objective: spec.objective,
                issued_cycle: cycle,
                family_issued_total_at_start: self
                    .family_stats
                    .get(&spec.family)
                    .map(|stats| stats.issued_total)
                    .unwrap_or(0),
                before,
            });
        self.controlled_holdout_issued_total =
            self.controlled_holdout_issued_total.saturating_add(1);
        true
    }

    fn resolve_controlled_holdouts(
        &mut self,
        after: &TelemetryContextSummary,
        cycle: u64,
        purge_recent: bool,
        current_applied_families: &[ActuatorFamily],
        quality: f64,
    ) {
        let mut retained = VecDeque::with_capacity(self.pending_controlled_holdouts.len());
        while let Some(pending) = self.pending_controlled_holdouts.pop_front() {
            let age = cycle.saturating_sub(pending.issued_cycle);
            if age < CONTROLLED_HOLDOUT_HORIZON_CYCLES {
                retained.push_back(pending);
                continue;
            }
            let family_issued_total = self
                .family_stats
                .get(&pending.family)
                .map(|stats| stats.issued_total)
                .unwrap_or(0);
            let confounded = purge_recent
                || current_applied_families.contains(&pending.family)
                || pending.family_issued_total_at_start != family_issued_total;
            if confounded {
                self.controlled_holdout_rejected_total =
                    self.controlled_holdout_rejected_total.saturating_add(1);
                continue;
            }

            let control_delta = (utility_score(pending.objective, after)
                - utility_score(pending.objective, &pending.before))
            .clamp(-1.0, 1.0);
            let per_cycle_delta = control_delta / CONTROLLED_HOLDOUT_HORIZON_CYCLES.max(1) as f64;
            let baseline = self
                .no_action_delta_ema
                .entry(pending.objective)
                .or_insert(per_cycle_delta);
            *baseline = (0.20 * per_cycle_delta + 0.80 * *baseline).clamp(-0.05, 0.05);

            let key = format!("{}|{}", pending.workload, pending.action_key);
            if self.controlled_models.len() >= MAX_CONTROLLED_MODELS
                && !self.controlled_models.contains_key(&key)
            {
                if let Some(weakest) = self
                    .controlled_models
                    .iter()
                    .min_by_key(|(_, stats)| stats.observations)
                    .map(|(key, _)| key.clone())
                {
                    self.controlled_models.remove(&weakest);
                }
            }
            let model = self.controlled_models.entry(key).or_default();
            model.observations = model.observations.saturating_add(1);
            model.would_have_helped = model.would_have_helped.saturating_add(u32::from(
                control_delta < -objective_effect_threshold(pending.objective),
            ));
            let alpha = 0.20;
            model.control_utility_ema = if model.observations == 1 {
                control_delta
            } else {
                alpha * control_delta + (1.0 - alpha) * model.control_utility_ema
            };
            model.quality_ema = if model.observations == 1 {
                quality
            } else {
                alpha * quality + (1.0 - alpha) * model.quality_ema
            };
            model.last_cycle = cycle;
            model.last_observed_unix = after.timestamp_unix;
            model.hardware_regime = HardwareRegime::from_context(after);
            model.installation_id = self.installation_id;
            self.controlled_models_revision = self.controlled_models_revision.wrapping_add(1);
            self.controlled_holdout_resolved_total =
                self.controlled_holdout_resolved_total.saturating_add(1);
        }
        self.pending_controlled_holdouts = retained;
    }

    fn external_deltas(&mut self, runtime: &RuntimeMetrics) -> ExternalDeltas {
        // RuntimeMetrics starts fresh after a daemon restart while this state
        // persists. Rebase monotonically decreasing sources before diffing.
        if runtime.markov_prewarm_applied < self.external_counters.markov_applied {
            self.external_counters.markov_applied = runtime.markov_prewarm_applied;
            self.external_counters.markov_hits = runtime.markov_prewarm_hits;
            self.external_counters.markov_misses = runtime.markov_prewarm_misses;
        }
        if runtime.interaction_qos_activations < self.external_counters.interaction_qos_activations
        {
            self.external_counters.interaction_qos_activations =
                runtime.interaction_qos_activations;
            self.external_counters.interaction_qos_reverts = runtime.interaction_qos_reverts;
        }
        if runtime.acceleration_lease_io_promotions_total
            < self.external_counters.acceleration_io_promotions
        {
            self.external_counters.acceleration_io_promotions =
                runtime.acceleration_lease_io_promotions_total;
        }
        if runtime.chromium_ecore_demotions_total < self.external_counters.chromium_ecore_demotions
        {
            self.external_counters.chromium_ecore_demotions =
                runtime.chromium_ecore_demotions_total;
        }
        if runtime.chromium_purge_hints_total < self.external_counters.chromium_purge_hints {
            self.external_counters.chromium_purge_hints = runtime.chromium_purge_hints_total;
        }
        if runtime.chromium_jetsam_demotions_total
            < self.external_counters.chromium_jetsam_demotions
        {
            self.external_counters.chromium_jetsam_demotions =
                runtime.chromium_jetsam_demotions_total;
        }
        ExternalDeltas {
            markov_applied: runtime
                .markov_prewarm_applied
                .saturating_sub(self.external_counters.markov_applied),
            markov_hits: runtime
                .markov_prewarm_hits
                .saturating_sub(self.external_counters.markov_hits),
            markov_misses: runtime
                .markov_prewarm_misses
                .saturating_sub(self.external_counters.markov_misses),
            interaction_activations: runtime
                .interaction_qos_activations
                .saturating_sub(self.external_counters.interaction_qos_activations),
            interaction_reverts: runtime
                .interaction_qos_reverts
                .saturating_sub(self.external_counters.interaction_qos_reverts),
            io_promotions: runtime
                .acceleration_lease_io_promotions_total
                .saturating_sub(self.external_counters.acceleration_io_promotions),
            chromium_ecore_demotions: runtime
                .chromium_ecore_demotions_total
                .saturating_sub(self.external_counters.chromium_ecore_demotions),
            chromium_purge_hints: runtime
                .chromium_purge_hints_total
                .saturating_sub(self.external_counters.chromium_purge_hints),
            chromium_jetsam_demotions: runtime
                .chromium_jetsam_demotions_total
                .saturating_sub(self.external_counters.chromium_jetsam_demotions),
        }
    }

    fn issue(
        &mut self,
        spec: ActionSpec,
        before: &TelemetryContextSummary,
        cycle: u64,
        cohort_size: u16,
        purge_recent: bool,
        event_resolved: bool,
    ) {
        self.issue_with_decision_id(
            spec,
            before,
            cycle,
            cohort_size,
            purge_recent,
            event_resolved,
            None,
            None,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn issue_with_decision_id(
        &mut self,
        spec: ActionSpec,
        before: &TelemetryContextSummary,
        cycle: u64,
        cohort_size: u16,
        purge_recent: bool,
        event_resolved: bool,
        decision_id: Option<DecisionId>,
        calibration_provenance: Option<CalibrationProvenance>,
    ) {
        if self.pending_actions.len() >= MAX_PENDING_ACTIONS {
            if let Some(evicted) = self.pending_actions.pop_front() {
                self.expire_unresolved(evicted.family);
            }
        }
        let gpu_prediction_generation =
            self.mark_gpu_prediction_used(&spec.action_key, &before.workload, cycle);
        let mut attribution = self
            .staged_attributions
            .get_mut(&spec.action_key)
            .and_then(VecDeque::pop_front)
            .unwrap_or_else(|| DecisionAttribution {
                action_key: spec.action_key.clone(),
                proposer: spec.family.as_str().to_string(),
                ..DecisionAttribution::default()
            });
        if attribution.proposer.is_empty() {
            attribution.proposer = spec.family.as_str().to_string();
        }
        attribution.action_key = spec.action_key.clone();
        if gpu_prediction_generation.is_some()
            && !attribution
                .supporters
                .iter()
                .any(|source| source == "gpu-model")
        {
            attribution.supporters.push("gpu-model".to_string());
        }
        attribution = attribution.bounded();
        self.next_action_id = self.next_action_id.saturating_add(1);
        self.actuator_issued_total = self.actuator_issued_total.saturating_add(1);
        let family_stats = self.family_stats.entry(spec.family).or_default();
        family_stats.issued_total = family_stats.issued_total.saturating_add(1);
        self.pending_actions.push_back(PendingActuatorEvidence {
            id: self.next_action_id,
            decision_id,
            family: spec.family,
            objective: spec.objective,
            action_key: spec.action_key,
            target: spec.target,
            target_pid: spec.target_pid,
            workload: before.workload.clone(),
            issued_cycle: cycle,
            horizon_cycles: spec.horizon_cycles,
            cohort_size,
            issued_total_at_start: self.actuator_issued_total,
            purge_recent,
            event_resolved,
            gpu_prediction_generation,
            attribution,
            calibration_provenance: calibration_provenance.unwrap_or_default().bounded(),
            provenance: spec.provenance,
            before: before.clone(),
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_pending(
        &mut self,
        after: &TelemetryContextSummary,
        snapshot: &SystemSnapshot,
        cycle: u64,
        mut markov_hits: u64,
        mut markov_misses: u64,
        mut interaction_reverts: u64,
    ) -> u64 {
        let mut resolved = 0_u64;
        let mut retained = VecDeque::with_capacity(self.pending_actions.len());
        while let Some(pending) = self.pending_actions.pop_front() {
            let age = cycle.saturating_sub(pending.issued_cycle);
            let (explicit_score, event_fired) = match pending.family {
                ActuatorFamily::MarkovPrewarm if markov_hits > 0 => {
                    markov_hits -= 1;
                    (Some(1.0), true)
                }
                ActuatorFamily::MarkovPrewarm if markov_misses > 0 => {
                    markov_misses -= 1;
                    (Some(0.0), true)
                }
                ActuatorFamily::InteractionQos if interaction_reverts > 0 => {
                    interaction_reverts -= 1;
                    (None, true)
                }
                _ => (None, false),
            };
            let timed_out = pending.event_resolved && age >= pending.horizon_cycles;
            let due = (!pending.event_resolved && age >= pending.horizon_cycles)
                || event_fired
                || timed_out;
            if due {
                self.resolve_one(pending, after, snapshot, cycle, explicit_score);
                resolved = resolved.saturating_add(1);
            } else {
                retained.push_back(pending);
            }
        }
        self.pending_actions = retained;
        resolved
    }

    fn resolve_one(
        &mut self,
        pending: PendingActuatorEvidence,
        after: &TelemetryContextSummary,
        snapshot: &SystemSnapshot,
        cycle: u64,
        explicit_score: Option<f64>,
    ) {
        let gpu_prediction_generation = pending.gpu_prediction_generation;
        let before_utility = utility_score(pending.objective, &pending.before);
        let after_utility =
            explicit_score.unwrap_or_else(|| utility_score(pending.objective, after));
        let raw_delta = explicit_score
            .map(|score| score * 2.0 - 1.0)
            .unwrap_or(after_utility - before_utility)
            .clamp(-1.0, 1.0);
        let per_cycle_baseline = self
            .no_action_delta_ema
            .get(&pending.objective)
            .copied()
            .unwrap_or(0.0);
        let counterfactual =
            (per_cycle_baseline * pending.horizon_cycles as f64).clamp(-0.25, 0.25);
        let net_delta = (raw_delta - counterfactual).clamp(-1.0, 1.0);
        let raw_state_delta = WorldStateDelta::between(&pending.before, after);
        let counterfactual_state_delta = self
            .no_action_state_delta_ema
            .scaled(pending.horizon_cycles as f64)
            .clamped(-0.25, 0.25);
        let net_state_delta = raw_state_delta
            .minus(counterfactual_state_delta)
            .clamped(-1.0, 1.0);
        let perceptual_latency_improvement = (-net_state_delta.latency).clamp(-1.0, 1.0);
        let utility = decompose_utility(pending.family, net_state_delta);
        let target_present_after = target_presence(&pending, snapshot);

        let finite = [
            before_utility,
            after_utility,
            raw_delta,
            counterfactual,
            net_delta,
        ]
        .into_iter()
        .all(f64::is_finite)
            && net_state_delta.is_finite();
        let mut confounders = u8::from(pending.purge_recent);
        confounders = confounders.saturating_add(u8::from(pending.workload != after.workload));
        confounders = confounders.saturating_add(u8::from(pending.cohort_size > 1));
        confounders = confounders.saturating_add(u8::from(
            self.actuator_issued_total > pending.issued_total_at_start,
        ));
        confounders = confounders.saturating_add(u8::from(
            (pending.before.thermal_score - after.thermal_score).abs() > 0.34,
        ));
        confounders = confounders.saturating_add(u8::from(
            after.operation_failures_total > pending.before.operation_failures_total,
        ));

        let telemetry_quality = context_quality(&pending.before).min(context_quality(after));
        let horizon_quality = if cycle
            <= pending
                .issued_cycle
                .saturating_add(pending.horizon_cycles.saturating_mul(2).max(2))
        {
            1.0
        } else {
            0.7
        };
        let quality = (0.70 * telemetry_quality
            + 0.20 * horizon_quality
            + 0.10 * (1.0 - confounders as f64 / 6.0))
            .clamp(0.0, 1.0);
        let proposed_tier = if !finite {
            EvidenceTier::Bronze
        } else if confounders == 0 && quality >= 0.85 {
            EvidenceTier::Gold
        } else {
            EvidenceTier::Silver
        };
        let tier = quarantine_experimental_tier(pending.provenance, proposed_tier);
        let effective = explicit_score.map_or_else(
            || {
                net_delta > objective_effect_threshold(pending.objective)
                    && !matches!(target_present_after, Some(false))
            },
            |score| score >= 0.5,
        );

        if tier == EvidenceTier::Gold {
            self.causal_dynamics.observe_action(
                &pending.action_key,
                pending.family.as_str(),
                &pending.workload,
                &pending.before,
                net_state_delta,
                pending.horizon_cycles,
                quality,
                after.timestamp_unix,
                HardwareRegime::from_context(after),
                self.installation_id,
                pending.id,
            );
        }

        let evidence = ResolvedActuatorEvidence {
            id: pending.id,
            decision_id: pending.decision_id,
            family: pending.family,
            objective: pending.objective,
            action_key: pending.action_key,
            target: pending.target,
            workload: pending.workload,
            issued_cycle: pending.issued_cycle,
            resolved_cycle: cycle,
            resolved_timestamp_unix: after.timestamp_unix,
            hardware_regime: HardwareRegime::from_context(after),
            installation_id: self.installation_id,
            horizon_cycles: pending.horizon_cycles,
            tier,
            provenance: pending.provenance,
            quality,
            raw_utility_delta: finite_or_zero(raw_delta),
            counterfactual_delta: finite_or_zero(counterfactual),
            net_utility_delta: finite_or_zero(net_delta),
            attribution: pending.attribution,
            calibration_provenance: pending.calibration_provenance,
            learning_details: None,
            utility,
            perceptual_latency_improvement: finite_or_zero(perceptual_latency_improvement),
            net_state_delta: if net_state_delta.is_finite() {
                net_state_delta
            } else {
                WorldStateDelta::default()
            },
            context_before: ActuatorEpisodeContext::from_telemetry(&pending.before),
            effective,
            confounder_count: confounders,
            target_present_after,
        };
        self.resolve_gpu_prediction(gpu_prediction_generation, &evidence);
        self.admit_resolved(evidence);
    }

    fn admit_resolved(&mut self, mut evidence: ResolvedActuatorEvidence) {
        self.actuator_resolved_total = self.actuator_resolved_total.saturating_add(1);
        self.actuator_quality_sum += evidence.quality;
        self.actuator_utility_sum += evidence.net_utility_delta;
        if evidence.tier != EvidenceTier::Bronze && evidence.utility.is_finite() {
            let alpha = if self.apollo_utility_observations == 0 {
                1.0
            } else {
                0.05
            };
            self.apollo_utility_ema =
                alpha * evidence.utility.apollo_utility + (1.0 - alpha) * self.apollo_utility_ema;
            self.apollo_utility_observations = self.apollo_utility_observations.saturating_add(1);
        }
        if evidence.effective {
            self.actuator_effective_total = self.actuator_effective_total.saturating_add(1);
        }
        if evidence.tier != EvidenceTier::Bronze {
            self.actuator_silver_total = self.actuator_silver_total.saturating_add(1);
        } else {
            self.actuator_rejected_total = self.actuator_rejected_total.saturating_add(1);
        }
        if evidence.tier == EvidenceTier::Gold {
            self.actuator_gold_total = self.actuator_gold_total.saturating_add(1);
            self.update_action_model(&evidence);
            self.update_decision_source_credit(&evidence);
            let observation = calibration_observation(&evidence);
            let calibration = self.model_calibration.observe_local_gold(&observation);
            if let CalibrationUpdate::Accepted { deltas, .. } = calibration {
                evidence.learning_details = learning_details_for_evidence(&evidence, deltas);
            }
            if self.new_gold_evidence.len() >= MAX_RECENT_EVIDENCE {
                self.new_gold_evidence.pop_front();
            }
            self.new_gold_evidence.push_back(evidence.clone());
        }
        if evidence.tier != EvidenceTier::Bronze
            && evidence.quality >= 0.85
            && evidence.context_before.valid
        {
            self.admit_episode(evidence.clone());
        }

        let stats = self.family_stats.entry(evidence.family).or_default();
        stats.resolved_total = stats.resolved_total.saturating_add(1);
        stats.bronze_total = stats.bronze_total.saturating_add(1);
        stats.utility_sum += evidence.net_utility_delta;
        stats.quality_sum += evidence.quality;
        if evidence.tier != EvidenceTier::Bronze {
            stats.silver_total = stats.silver_total.saturating_add(1);
        } else {
            stats.rejected_total = stats.rejected_total.saturating_add(1);
        }
        if evidence.tier == EvidenceTier::Gold {
            stats.gold_total = stats.gold_total.saturating_add(1);
        }
        if evidence.effective {
            stats.effective_total = stats.effective_total.saturating_add(1);
        }
        if self.recent_evidence.len() >= MAX_RECENT_EVIDENCE {
            self.recent_evidence.pop_front();
        }
        evidence.learning_details = None;
        self.recent_evidence.push_back(evidence);
    }

    fn update_decision_source_credit(&mut self, evidence: &ResolvedActuatorEvidence) {
        let attribution = &evidence.attribution;
        if attribution.proposer.is_empty() || !evidence.utility.is_finite() {
            return;
        }
        let mut directed_sources =
            Vec::with_capacity(1 + attribution.supporters.len() + attribution.vetoes.len());
        directed_sources.push((attribution.proposer.clone(), 1.0_f64, true));
        directed_sources.extend(
            attribution
                .supporters
                .iter()
                .cloned()
                .map(|source| (source, 1.0, true)),
        );
        directed_sources.extend(
            attribution
                .vetoes
                .iter()
                .cloned()
                .map(|source| (source, -1.0, false)),
        );
        let mut seen_sources = BTreeSet::new();
        for (source, direction, supported) in directed_sources {
            if source.is_empty() || !seen_sources.insert(source.clone()) {
                continue;
            }
            if self.decision_source_stats.len() >= MAX_DECISION_SOURCES
                && !self.decision_source_stats.contains_key(&source)
            {
                if let Some(weakest) = self
                    .decision_source_stats
                    .iter()
                    .min_by_key(|(_, stats)| stats.observations)
                    .map(|(source, _)| source.clone())
                {
                    self.decision_source_stats.remove(&weakest);
                }
            }
            let stats = self.decision_source_stats.entry(source).or_default();
            let alpha = if stats.observations == 0 { 1.0 } else { 0.15 };
            stats.observations = stats.observations.saturating_add(1);
            if supported {
                stats.supports = stats.supports.saturating_add(1);
            } else {
                stats.vetoes = stats.vetoes.saturating_add(1);
            }
            let measured_direction = if evidence.utility.apollo_utility > 0.005 {
                1.0
            } else if evidence.utility.apollo_utility < -0.005 {
                -1.0
            } else {
                0.0
            };
            let directional_agreement = direction * measured_direction;
            if directional_agreement > 0.0 {
                stats.correct = stats.correct.saturating_add(1);
            }
            stats.credit_ema = alpha * directional_agreement + (1.0 - alpha) * stats.credit_ema;
            let absolute_error = ((1.0 - directional_agreement) / 2.0).clamp(0.0, 1.0);
            stats.absolute_error_ema =
                alpha * absolute_error + (1.0 - alpha) * stats.absolute_error_ema;
            stats.last_cycle = evidence.resolved_cycle;
        }
    }

    fn admit_episode(&mut self, evidence: ResolvedActuatorEvidence) {
        if evidence.decision_id.is_some()
            && self
                .episodic_evidence
                .iter()
                .any(|existing| existing.decision_id == evidence.decision_id)
        {
            return;
        }
        let family_count = self
            .episodic_evidence
            .iter()
            .filter(|existing| existing.family == evidence.family)
            .count();
        let eviction = if family_count >= MAX_EPISODES_PER_FAMILY {
            self.episodic_evidence
                .iter()
                .position(|existing| existing.family == evidence.family)
        } else if self.episodic_evidence.len() >= MAX_EPISODIC_EVIDENCE {
            let mut family_counts = BTreeMap::new();
            for existing in &self.episodic_evidence {
                *family_counts.entry(existing.family).or_insert(0_usize) += 1;
            }
            let largest = family_counts.values().copied().max().unwrap_or(0);
            self.episodic_evidence.iter().position(|existing| {
                family_counts.get(&existing.family).copied().unwrap_or(0) == largest
            })
        } else {
            None
        };
        if let Some(index) = eviction {
            self.episodic_evidence.remove(index);
        }
        self.episodic_evidence.push_back(evidence);
    }

    fn update_action_model(&mut self, evidence: &ResolvedActuatorEvidence) {
        let mut keys = Vec::with_capacity(5);
        keys.push(evidence.action_key.clone());
        keys.push(format!("{}|{}", evidence.workload, evidence.action_key));
        // Parameterized arms remain independently learnable while also
        // updating the legacy parent model. This avoids splitting all prior
        // interaction evidence at upgrade time and prevents a short/long arm
        // from making the aggregate action disappear from the planner.
        if let Some(parent) = parameter_parent_action_key(&evidence.action_key) {
            keys.push(parent.to_string());
            keys.push(format!("{}|{}", evidence.workload, parent));
        }
        keys.push(format!("{}:*", evidence.family.as_str()));
        let now_unix = evidence.resolved_timestamp_unix;
        for key in keys {
            let unseen = !self.action_models.contains_key(&key);
            if unseen && self.action_models.len() >= MAX_ACTION_MODELS {
                // Rank eviction by decayed evidence — the same currency the
                // readiness gate spends. Raw `observations` never decay, so
                // ranking by them keeps long-dead models resident forever while
                // destroying the young ones that are actually accumulating.
                if let Some(evict) = self
                    .action_models
                    .iter()
                    .min_by(|left, right| {
                        left.1
                            .effective_evidence_at(now_unix)
                            .total_cmp(&right.1.effective_evidence_at(now_unix))
                            .then(left.1.last_cycle.cmp(&right.1.last_cycle))
                    })
                    .map(|(key, _)| key.clone())
                {
                    self.action_models.remove(&evict);
                    self.action_model_evictions_total =
                        self.action_model_evictions_total.saturating_add(1);
                }
            }
            if unseen {
                self.action_model_births_total = self.action_model_births_total.saturating_add(1);
            }
            let model = self.action_models.entry(key).or_default();
            let previous_mean = model.utility_ema;
            let previous_state_mean = model.state_delta_ema;
            let same_hardware = !evidence.hardware_regime.is_known()
                || (model.hardware_regime.is_known()
                    && model.hardware_regime == evidence.hardware_regime);
            let same_installation = model.installation_id.is_known()
                && model.installation_id == evidence.installation_id;
            let has_local_epoch = model.last_observed_unix > 0
                && evidence.resolved_timestamp_unix >= model.last_observed_unix
                && same_hardware
                && same_installation;
            let previous_mass = if has_local_epoch {
                model.effective_evidence_at(evidence.resolved_timestamp_unix)
            } else {
                0.0
            };
            let previous_state_mass = if has_local_epoch {
                model.effective_state_evidence_at(evidence.resolved_timestamp_unix)
            } else {
                0.0
            };
            model.observations = model.observations.saturating_add(1);
            model.effective_observations = model
                .effective_observations
                .saturating_add(u32::from(evidence.effective));
            if !has_local_epoch {
                model.utility_ema = evidence.net_utility_delta;
                model.utility_variance_ema = 0.0;
                model.quality_ema = evidence.quality;
            } else {
                model.utility_ema = ACTION_MODEL_EMA_ALPHA * evidence.net_utility_delta
                    + (1.0 - ACTION_MODEL_EMA_ALPHA) * model.utility_ema;
                let residual = evidence.net_utility_delta - previous_mean;
                model.utility_variance_ema = (1.0 - ACTION_MODEL_EMA_ALPHA)
                    * (model.utility_variance_ema + ACTION_MODEL_EMA_ALPHA * residual * residual);
                model.quality_ema = ACTION_MODEL_EMA_ALPHA * evidence.quality
                    + (1.0 - ACTION_MODEL_EMA_ALPHA) * model.quality_ema;
            }
            if previous_state_mass <= 0.0 {
                model.state_delta_ema = evidence.net_state_delta;
                model.state_variance_ema = WorldStateDelta::default();
            } else {
                model.state_delta_ema = model
                    .state_delta_ema
                    .ema(evidence.net_state_delta, ACTION_MODEL_EMA_ALPHA);
                model.state_variance_ema = model.state_variance_ema.variance_update(
                    evidence.net_state_delta.minus(previous_state_mean),
                    ACTION_MODEL_EMA_ALPHA,
                );
            }
            model.evidence_mass = (previous_mass + 1.0).min(ACTION_MODEL_EVIDENCE_CAP);
            model.state_evidence_mass = (previous_state_mass + 1.0).min(ACTION_MODEL_EVIDENCE_CAP);
            model.last_cycle = evidence.resolved_cycle;
            model.last_observed_unix = evidence.resolved_timestamp_unix;
            model.hardware_regime = evidence.hardware_regime;
            model.installation_id = evidence.installation_id;
        }
        self.action_model_evidence_updates_total =
            self.action_model_evidence_updates_total.saturating_add(1);
        self.action_model_last_evidence_cycle = evidence.resolved_cycle;
        self.action_models_revision = self.action_models_revision.wrapping_add(1);
    }

    fn expire_unresolved(&mut self, family: ActuatorFamily) {
        self.actuator_expired_total = self.actuator_expired_total.saturating_add(1);
        let stats = self.family_stats.entry(family).or_default();
        stats.expired_total = stats.expired_total.saturating_add(1);
    }

    pub fn metrics(&self) -> TelemetryMedallionMetrics {
        let admitted = self.silver_total.saturating_add(self.gold_total);
        let (top_rejected_field, top_rejected_field_total) = self
            .field_rejection_counters
            .iter()
            .max_by(|left, right| left.1.cmp(right.1).then_with(|| right.0.cmp(left.0)))
            .map(|(field, count)| (Some(*field), *count))
            .unwrap_or((None, 0));
        let controlled_would_help_total = self
            .controlled_models
            .values()
            .map(|stats| stats.would_have_helped as u64)
            .sum();
        let controlled_utility_sum: f64 = self
            .controlled_models
            .values()
            .map(|stats| stats.control_utility_ema * stats.observations as f64)
            .sum();
        let controlled_observations: u64 = self
            .controlled_models
            .values()
            .map(|stats| stats.observations as u64)
            .sum();
        let calibrated_gpu_models: Vec<&GpuCalibrationStats> = self
            .gpu_calibration_models
            .iter()
            .filter(|(key, stats)| {
                !key.ends_with("|*")
                    && stats.gold > 0
                    && stats.installation_id == self.installation_id
            })
            .map(|(_, stats)| stats)
            .collect();
        let gpu_gold_weight: f64 = calibrated_gpu_models
            .iter()
            .map(|stats| stats.gold as f64)
            .sum();
        let gpu_weighted_mean = |value: fn(&GpuCalibrationStats) -> f64| {
            if gpu_gold_weight <= f64::EPSILON {
                0.0
            } else {
                calibrated_gpu_models
                    .iter()
                    .map(|stats| value(stats) * stats.gold as f64)
                    .sum::<f64>()
                    / gpu_gold_weight
            }
        };
        let decision_credit_leader = self
            .decision_source_stats
            .iter()
            .filter(|(_, stats)| stats.observations > 0)
            .max_by(|left, right| {
                source_authority_score(left.1)
                    .total_cmp(&source_authority_score(right.1))
                    .then_with(|| right.0.cmp(left.0))
            });
        TelemetryMedallionMetrics {
            bronze_total: self.bronze_total,
            silver_total: self.silver_total,
            gold_total: self.gold_total,
            local_gold_total: self.local_gold_total,
            rejected_total: self.rejected_total,
            invalid_total: self.invalid_total,
            non_finite_total: self.reason_counters.non_finite,
            range_total: self.reason_counters.out_of_range,
            stale_total: self.reason_counters.stale,
            temporal_total: self.reason_counters.temporal,
            foreign_total: self.reason_counters.foreign_hardware,
            coherence_total: self.reason_counters.coherence,
            top_rejected_field,
            top_rejected_field_total,
            last_field_violation: self.last_field_violation,
            current_tier: self.current_tier,
            mean_quality: if admitted == 0 {
                0.0
            } else {
                (self.quality_sum / admitted as f64).clamp(0.0, 1.0)
            },
            gold_rate: if self.bronze_total == 0 {
                0.0
            } else {
                (self.gold_total as f64 / self.bronze_total as f64).clamp(0.0, 1.0)
            },
            actuator_issued_total: self.actuator_issued_total,
            actuator_pending_total: self.pending_actions.len() as u64,
            actuator_bronze_total: self.actuator_resolved_total,
            actuator_silver_total: self.actuator_silver_total,
            actuator_gold_total: self.actuator_gold_total,
            actuator_effective_total: self.actuator_effective_total,
            actuator_rejected_total: self.actuator_rejected_total,
            actuator_expired_total: self.actuator_expired_total,
            actuator_mean_quality: if self.actuator_resolved_total == 0 {
                0.0
            } else {
                (self.actuator_quality_sum / self.actuator_resolved_total as f64).clamp(0.0, 1.0)
            },
            actuator_mean_utility: if self.actuator_resolved_total == 0 {
                0.0
            } else {
                (self.actuator_utility_sum / self.actuator_resolved_total as f64).clamp(-1.0, 1.0)
            },
            action_model_len: self.action_models.len() as u64,
            action_model_capacity: MAX_ACTION_MODELS as u64,
            action_model_evictions_total: self.action_model_evictions_total,
            action_model_births_total: self.action_model_births_total,
            action_model_evidence_updates_total: self.action_model_evidence_updates_total,
            action_model_last_evidence_cycle: self.action_model_last_evidence_cycle,
            apollo_utility_ema: self.apollo_utility_ema.clamp(-1.0, 1.0),
            decision_credit_sources: self.decision_source_stats.len() as u64,
            decision_credit_leader_score: decision_credit_leader
                .map(|(_, stats)| stats.credit_ema)
                .unwrap_or(0.0),
            decision_credit_leader_accuracy: decision_credit_leader
                .map(|(_, stats)| stats.accuracy())
                .unwrap_or(0.0),
            decision_credit_leader_observations: decision_credit_leader
                .map(|(_, stats)| stats.observations)
                .unwrap_or(0),
            actuator_ready_models: self
                .action_models
                .iter()
                .filter(|(key, model)| {
                    !key.ends_with(":*")
                        && self.current_tier == ContextTier::Gold
                        && self.local_gold_total > 0
                        && model.installation_id == self.installation_id
                        && self.latest.as_ref().is_some_and(|latest| {
                            model.effective_evidence_at(latest.timestamp_unix) >= 10.0
                                && model.quality_ema >= 0.85
                                && model.hardware_regime.matches_context(latest)
                        })
                })
                .count() as u64,
            controlled_holdout_issued_total: self.controlled_holdout_issued_total,
            controlled_holdout_pending_total: self.pending_controlled_holdouts.len() as u64,
            controlled_holdout_resolved_total: self.controlled_holdout_resolved_total,
            controlled_holdout_rejected_total: self.controlled_holdout_rejected_total,
            controlled_holdout_would_help_total: controlled_would_help_total,
            controlled_holdout_mean_control_utility: if controlled_observations == 0 {
                0.0
            } else {
                (controlled_utility_sum / controlled_observations as f64).clamp(-1.0, 1.0)
            },
            gpu_prediction_bronze_total: self.gpu_prediction_bronze_total,
            gpu_prediction_silver_total: self.gpu_prediction_silver_total,
            gpu_prediction_gold_total: self.gpu_prediction_gold_total,
            gpu_prediction_rejected_total: self.gpu_prediction_rejected_total,
            gpu_prediction_evicted_total: self.gpu_prediction_evicted_total,
            gpu_prediction_unused_total: self.gpu_prediction_unused_total,
            gpu_prediction_bronze_rejected_total: self.gpu_prediction_bronze_rejected_total,
            gpu_prediction_unclassified_rejections: self.gpu_prediction_unclassified_rejections(),
            gpu_prediction_pending_total: self
                .gpu_predictions
                .iter()
                .filter(|prediction| prediction.resolved_cycle.is_none())
                .count() as u64,
            gpu_prediction_calibrated_models: calibrated_gpu_models.len() as u64,
            gpu_prediction_mean_absolute_error: gpu_weighted_mean(|stats| stats.absolute_error_ema)
                .clamp(0.0, 2.0),
            gpu_prediction_mean_brier: gpu_weighted_mean(|stats| stats.brier_ema).clamp(0.0, 1.0),
            gpu_prediction_mean_quality: gpu_weighted_mean(|stats| stats.quality_ema)
                .clamp(0.0, 1.0),
        }
    }

    pub fn latest(&self) -> Option<&TelemetryContextSummary> {
        self.latest.as_ref()
    }

    pub fn trusted_view(&self) -> TrustedTelemetryView<'_> {
        TrustedTelemetryView {
            current: (self.current_tier == ContextTier::Gold)
                .then_some(self.latest.as_ref())
                .flatten(),
            installation_id: self.installation_id,
            action_models: &self.action_models,
            action_models_revision: self.action_models_revision,
            controlled_models: &self.controlled_models,
            controlled_models_revision: self.controlled_models_revision,
            episodic_evidence: &self.episodic_evidence,
            causal_dynamics: &self.causal_dynamics,
            causal_dynamics_revision: self.causal_dynamics.publication_revision(),
            gpu_calibration_models: &self.gpu_calibration_models,
            gpu_calibration_revision: self.gpu_calibration_revision,
            metrics: self.metrics(),
        }
    }

    pub fn family_stats(&self) -> &BTreeMap<ActuatorFamily, ActuatorFamilyStats> {
        &self.family_stats
    }

    pub fn action_models(&self) -> &BTreeMap<String, ActionModelStats> {
        &self.action_models
    }

    pub fn action_models_revision(&self) -> u64 {
        self.action_models_revision
    }

    /// Constant-time change token for cached learning-health projections.
    pub fn learning_revision(&self) -> u64 {
        self.action_models_revision
            ^ self.gpu_calibration_revision.rotate_left(11)
            ^ self.controlled_models_revision.rotate_left(23)
            ^ self.causal_dynamics.publication_revision().rotate_left(37)
            ^ self
                .model_calibration
                .metrics()
                .accepted_forecasts_total
                .rotate_left(49)
            ^ self.local_gold_total.rotate_left(5)
    }

    /// Accept the ledger's bounded terminal batch. All locally attributed
    /// outcomes cross this interface, but only locally applied episodes may
    /// open actuator evidence. Existing counter/audit fallback episodes are
    /// matched in place so one kernel action cannot create two medallion
    /// episodes.
    pub fn stage_decision_episodes(&mut self, episodes: &[ResolvedDecisionEpisode]) {
        for episode in episodes {
            self.decision_id_high_water = self.decision_id_high_water.max(episode.id.0);
        }
        let mut cohort_sizes = BTreeMap::<u64, u16>::new();
        for episode in episodes {
            if episode.envelope.lifecycle == DecisionLifecycle::Applied
                && episode.authority_eligible
                && decision_episode_attribution(episode).is_some()
                && !episode.envelope.action_key.starts_with("coordinated:")
            {
                let count = cohort_sizes
                    .entry(episode.envelope.proposed_cycle)
                    .or_default();
                *count = count.saturating_add(1);
            }
        }
        let mut exact_pending = HashMap::<(u64, String), VecDeque<usize>>::new();
        let mut coordinated_pending = HashMap::<u64, VecDeque<usize>>::new();
        for (index, pending) in self.pending_actions.iter().enumerate() {
            if pending.decision_id.is_some() {
                continue;
            }
            if pending.family == ActuatorFamily::Coordinated {
                coordinated_pending
                    .entry(pending.issued_cycle)
                    .or_default()
                    .push_back(index);
            } else {
                exact_pending
                    .entry((pending.issued_cycle, pending.action_key.clone()))
                    .or_default()
                    .push_back(index);
            }
        }
        for episode in episodes {
            let Some(attribution) = decision_episode_attribution(episode) else {
                continue;
            };
            if episode.envelope.lifecycle != DecisionLifecycle::Applied
                || !episode.authority_eligible
            {
                continue;
            }
            let cohort_size = if episode.envelope.action_key.starts_with("coordinated:") {
                1
            } else {
                cohort_sizes
                    .get(&episode.envelope.proposed_cycle)
                    .copied()
                    .unwrap_or(1)
                    .max(1)
            };
            let calibration_provenance = calibration_provenance_for_episode(episode, cohort_size);

            let pending_index = if episode.envelope.action_key.starts_with("coordinated:") {
                coordinated_pending
                    .get_mut(&episode.envelope.proposed_cycle)
                    .and_then(VecDeque::pop_front)
            } else {
                exact_pending
                    .get_mut(&(
                        episode.envelope.proposed_cycle,
                        episode.envelope.action_key.clone(),
                    ))
                    .and_then(VecDeque::pop_front)
            };
            if let Some(pending) =
                pending_index.and_then(|index| self.pending_actions.get_mut(index))
            {
                pending.decision_id = Some(episode.id);
                pending.action_key = bounded_text(&episode.envelope.action_key, 320);
                pending.target = bounded_text(&episode.envelope.target, 256);
                pending.target_pid = target_pid_from_decision_target(&episode.envelope.target);
                pending.cohort_size = pending.cohort_size.max(cohort_size);
                pending.attribution = attribution;
                pending.calibration_provenance = calibration_provenance;
                continue;
            }

            // A non-Gold source context cannot later become authoritative by
            // waiting for a healthier cycle. Exact events from the latest Gold
            // cycle are retained for the next observation only.
            if self.current_tier != ContextTier::Gold
                || self.last_cycle != episode.envelope.proposed_cycle
            {
                continue;
            }
            if self.staged_decision_episodes.len() >= MAX_STAGED_DECISION_EPISODES {
                self.staged_decision_episodes.pop_front();
            }
            self.staged_decision_episodes
                .push_back(StagedDecisionEpisode {
                    episode: episode.clone(),
                    cohort_size,
                });
        }
    }

    pub fn stage_decision_attribution(&mut self, attribution: DecisionAttribution) {
        let attribution = attribution.bounded();
        if attribution.action_key.is_empty() {
            return;
        }
        let staged_len: usize = self.staged_attributions.values().map(VecDeque::len).sum();
        if staged_len >= MAX_STAGED_ATTRIBUTIONS {
            self.staged_attributions.clear();
        }
        self.staged_attributions
            .entry(attribution.action_key.clone())
            .or_default()
            .push_back(attribution);
    }

    fn issue_staged_decisions(
        &mut self,
        before: &TelemetryContextSummary,
        cycle: u64,
        purge_recent: bool,
        external_deltas: &mut ExternalDeltas,
    ) -> u64 {
        let mut issued = 0_u64;
        while let Some(staged) = self.staged_decision_episodes.pop_front() {
            let episode = staged.episode;
            if episode.envelope.proposed_cycle >= cycle {
                self.staged_decision_episodes
                    .push_front(StagedDecisionEpisode {
                        episode,
                        cohort_size: staged.cohort_size,
                    });
                break;
            }
            let Some(spec) = decision_episode_spec(&episode) else {
                continue;
            };
            let family = spec.family;
            if let Some(attribution) = decision_episode_attribution(&episode) {
                self.stage_decision_attribution(attribution);
            }
            self.issue_with_decision_id(
                spec,
                before,
                episode.envelope.proposed_cycle,
                staged.cohort_size,
                purge_recent,
                matches!(
                    family,
                    ActuatorFamily::MarkovPrewarm | ActuatorFamily::InteractionQos
                ),
                Some(episode.id),
                Some(calibration_provenance_for_episode(
                    &episode,
                    staged.cohort_size,
                )),
            );
            external_deltas.suppress(family);
            issued = issued.saturating_add(1);
        }
        issued
    }

    pub fn decision_source_stats(&self) -> &BTreeMap<String, DecisionSourceStats> {
        &self.decision_source_stats
    }

    pub fn model_calibration_metrics(&self) -> ModelCalibrationMetrics {
        self.model_calibration.metrics()
    }

    pub fn model_calibration(&self) -> &ModelCalibrationStore {
        &self.model_calibration
    }

    pub fn model_calibration_summary(&self) -> ModelCalibrationSummary {
        self.model_calibration.summary()
    }

    pub fn decision_id_high_water(&self) -> u64 {
        self.decision_id_high_water
    }

    pub fn decision_credit_leader(&self) -> Option<(&str, &DecisionSourceStats)> {
        self.decision_source_stats
            .iter()
            .filter(|(_, stats)| stats.observations > 0)
            .max_by(|left, right| {
                source_authority_score(left.1)
                    .total_cmp(&source_authority_score(right.1))
                    .then_with(|| right.0.cmp(left.0))
            })
            .map(|(source, stats)| (source.as_str(), stats))
    }

    pub fn recent_actuator_evidence(&self) -> &VecDeque<ResolvedActuatorEvidence> {
        &self.recent_evidence
    }

    pub fn current_machine_recent_evidence(
        &self,
    ) -> impl Iterator<Item = &ResolvedActuatorEvidence> {
        self.recent_evidence.iter().filter(|evidence| {
            self.installation_id.is_known()
                && evidence.installation_id == self.installation_id
                && (!self.live_hardware_regime.is_known()
                    || evidence.hardware_regime == self.live_hardware_regime)
        })
    }

    /// Distinct, current-machine Gold decisions admitted by the calibration
    /// pipeline. Context Gold samples and legacy aggregate evidence are not
    /// decision identities and must not contribute to learning maturity.
    pub fn authoritative_gold_decision_ids(&self) -> Vec<u64> {
        self.model_calibration
            .accepted_decision_ids()
            .map(|id| id.0)
            .collect()
    }

    pub fn authoritative_gold_decision_count(&self) -> usize {
        self.model_calibration.accepted_decision_count()
    }

    pub fn causal_dynamics(&self) -> &CausalDynamicsModel {
        &self.causal_dynamics
    }

    /// Drain only Gold outcomes resolved since the previous call. This queue
    /// is intentionally not persisted: consumers must never replay historical
    /// outcomes after a daemon restart.
    pub fn drain_new_gold_evidence(&mut self) -> Vec<ResolvedActuatorEvidence> {
        self.new_gold_evidence.drain(..).collect()
    }

    /// Start measuring one microexperiment arm over its horizon.
    ///
    /// Returns `false` — and records nothing — when there is no admitted
    /// context to measure against, the family is outside the experiment
    /// catalog, the decision is already being measured, or the bounded window
    /// set is full. Opening a window grants no authority and performs no
    /// action; it only remembers a telemetry snapshot.
    pub fn open_lab_utility_window(
        &mut self,
        decision_id: u64,
        family: ActuatorFamily,
        horizon_cycles: u64,
        cycle: u64,
    ) -> bool {
        if decision_id == 0 || horizon_cycles == 0 || self.lab_windows.len() >= MAX_LAB_WINDOWS {
            return false;
        }
        let Some(objective) = lab_objective(family) else {
            return false;
        };
        let Some(before) = self.latest.clone() else {
            return false;
        };
        if self
            .lab_windows
            .iter()
            .any(|window| window.decision_id == decision_id)
        {
            return false;
        }
        let deadline_cycle = cycle
            .saturating_add(horizon_cycles)
            .saturating_add(LAB_WINDOW_GRACE_CYCLES);
        self.lab_windows.push_back(LabUtilityWindow {
            decision_id,
            objective,
            opened_cycle: cycle,
            horizon_cycles,
            deadline_cycle,
            before,
        });
        true
    }

    /// Hand over arm measurements closed since the previous call. Like the Gold
    /// queue this is never persisted, so a restart cannot replay a stale
    /// measurement as fresh experimental evidence.
    pub fn drain_lab_utility(&mut self) -> Vec<LabUtilitySample> {
        self.lab_samples.drain(..).collect()
    }

    pub fn lab_windows_open(&self) -> usize {
        self.lab_windows.len()
    }

    pub fn lab_windows_expired_total(&self) -> u64 {
        self.lab_windows_expired_total
    }

    /// Close every arm window that reached its horizon, and drop the ones that
    /// ran past their grace deadline without a closing context.
    fn resolve_lab_windows(&mut self, after: &TelemetryContextSummary, cycle: u64) {
        if self.lab_windows.is_empty() {
            return;
        }
        let mut retained = VecDeque::with_capacity(self.lab_windows.len());
        while let Some(window) = self.lab_windows.pop_front() {
            if cycle > window.deadline_cycle {
                self.lab_windows_expired_total = self.lab_windows_expired_total.saturating_add(1);
                continue;
            }
            if cycle < window.opened_cycle.saturating_add(window.horizon_cycles) {
                retained.push_back(window);
                continue;
            }
            let before_utility = utility_score(window.objective, &window.before);
            let after_utility = utility_score(window.objective, after);
            let raw_delta = (after_utility - before_utility).clamp(-1.0, 1.0);
            let quality = context_quality(&window.before).min(context_quality(after));
            let confounded = !raw_delta.is_finite()
                || quality < 0.85
                || window.before.workload != after.workload
                || (window.before.thermal_score - after.thermal_score).abs() > 0.34
                || after.operation_failures_total > window.before.operation_failures_total;
            if self.lab_samples.len() >= MAX_LAB_SAMPLES {
                self.lab_samples.pop_front();
            }
            self.lab_samples.push_back(LabUtilitySample {
                decision_id: window.decision_id,
                utility_micros: (finite_or_zero(raw_delta) * 1_000_000.0) as i64,
                resolved_cycle: cycle,
                confounded,
                quality,
            });
        }
        self.lab_windows = retained;
    }

    pub fn snapshot(&self) -> TelemetryMedallionPersisted {
        if let Some(quarantined) = &self.quarantined_future_state {
            return quarantined.as_ref().clone();
        }
        TelemetryMedallionPersisted {
            actuator_evidence_schema_version: ACTUATOR_EVIDENCE_SCHEMA_VERSION,
            context_schema_version: TELEMETRY_CONTEXT_SCHEMA_VERSION,
            installation_id: self.installation_id,
            decision_id_high_water: self.decision_id_high_water,
            bronze_total: self.bronze_total,
            silver_total: self.silver_total,
            gold_total: self.gold_total,
            rejected_total: self.rejected_total,
            invalid_total: self.invalid_total,
            quality_sum: self.quality_sum,
            last_cycle: self.last_cycle,
            latest: self.latest.clone(),
            pending_actions: self.pending_actions.iter().cloned().collect(),
            family_stats: self.family_stats.clone(),
            action_models: self.action_models.clone(),
            action_model_evictions_total: self.action_model_evictions_total,
            action_model_births_total: self.action_model_births_total,
            action_model_evidence_updates_total: self.action_model_evidence_updates_total,
            action_model_last_evidence_cycle: self.action_model_last_evidence_cycle,
            recent_evidence: self.recent_evidence.iter().cloned().collect(),
            episodic_evidence: self.episodic_evidence.iter().cloned().collect(),
            external_counters: self.external_counters.clone(),
            next_action_id: self.next_action_id,
            actuator_issued_total: self.actuator_issued_total,
            actuator_resolved_total: self.actuator_resolved_total,
            actuator_silver_total: self.actuator_silver_total,
            actuator_gold_total: self.actuator_gold_total,
            actuator_effective_total: self.actuator_effective_total,
            actuator_rejected_total: self.actuator_rejected_total,
            actuator_expired_total: self.actuator_expired_total,
            actuator_quality_sum: self.actuator_quality_sum,
            actuator_utility_sum: self.actuator_utility_sum,
            apollo_utility_ema: self.apollo_utility_ema,
            apollo_utility_observations: self.apollo_utility_observations,
            decision_source_stats: self.decision_source_stats.clone(),
            model_calibration: Some(self.model_calibration.snapshot()),
            no_action_delta_ema: self.no_action_delta_ema.clone(),
            no_action_state_delta_ema: self.no_action_state_delta_ema,
            controlled_models: self.controlled_models.clone(),
            controlled_holdout_issued_total: self.controlled_holdout_issued_total,
            controlled_holdout_resolved_total: self.controlled_holdout_resolved_total,
            controlled_holdout_rejected_total: self.controlled_holdout_rejected_total,
            controlled_holdout_pending_total: self.pending_controlled_holdouts.len() as u64,
            causal_dynamics: self.causal_dynamics.clone(),
            gpu_prediction_schema_version: GPU_PREDICTION_SCHEMA_VERSION,
            gpu_predictions: self.gpu_predictions.iter().cloned().collect(),
            gpu_calibration_models: self.gpu_calibration_models.clone(),
            gpu_prediction_bronze_total: self.gpu_prediction_bronze_total,
            gpu_prediction_silver_total: self.gpu_prediction_silver_total,
            gpu_prediction_gold_total: self.gpu_prediction_gold_total,
            gpu_prediction_rejected_total: self.gpu_prediction_rejected_total,
            gpu_prediction_evicted_total: self.gpu_prediction_evicted_total,
            gpu_prediction_unused_total: self.gpu_prediction_unused_total,
            gpu_prediction_bronze_rejected_total: self.gpu_prediction_bronze_rejected_total,
        }
    }

    pub fn restore(&mut self, mut state: TelemetryMedallionPersisted) {
        let persisted_actuator_schema = state.actuator_evidence_schema_version;
        if persisted_actuator_schema > ACTUATOR_EVIDENCE_SCHEMA_VERSION {
            let installation_id = self.installation_id;
            let live_hardware_regime = self.live_hardware_regime;
            *self = Self::new(installation_id);
            self.live_hardware_regime = live_hardware_regime;
            self.quarantined_future_state = Some(Box::new(state));
            return;
        }
        self.quarantined_future_state = None;
        let observed_live_hardware = self
            .latest
            .as_ref()
            .map(HardwareRegime::from_context)
            .unwrap_or_default();
        let live_hardware = if self.live_hardware_regime.is_known() {
            self.live_hardware_regime
        } else {
            observed_live_hardware
        };
        let persisted_hardware = state
            .latest
            .as_ref()
            .map(HardwareRegime::from_context)
            .unwrap_or_default();
        let current_hardware = if live_hardware.is_known() {
            live_hardware
        } else {
            persisted_hardware
        };
        let model_calibration = state.model_calibration.take();
        let same_installation =
            state.installation_id.is_known() && state.installation_id == self.installation_id;
        let evidence_high_water = state
            .recent_evidence
            .iter()
            .chain(state.episodic_evidence.iter())
            .filter_map(|evidence| evidence.decision_id)
            .map(|id| id.0)
            .max()
            .unwrap_or(0);
        self.decision_id_high_water = if same_installation {
            state.decision_id_high_water.max(evidence_high_water)
        } else {
            0
        };
        let same_origin = state.context_schema_version == TELEMETRY_CONTEXT_SCHEMA_VERSION
            && state.installation_id.is_known()
            && state.installation_id == self.installation_id;
        let reset_actuator_evidence = persisted_actuator_schema < 2;
        let reset_model_calibration = persisted_actuator_schema != ACTUATOR_EVIDENCE_SCHEMA_VERSION;

        self.bronze_total = state.bronze_total;
        self.gold_total = state.gold_total.min(self.bronze_total);
        self.rejected_total = state
            .rejected_total
            .min(self.bronze_total.saturating_sub(self.gold_total));
        self.silver_total = state.silver_total.min(
            self.bronze_total
                .saturating_sub(self.gold_total)
                .saturating_sub(self.rejected_total),
        );
        self.invalid_total = state.invalid_total.min(self.rejected_total);
        self.reason_counters = ContextReasonCounters::default();
        self.field_rejection_counters.clear();
        self.last_field_violation = None;
        let admitted = self.silver_total.saturating_add(self.gold_total);
        self.quality_sum = if state.quality_sum.is_finite() {
            state.quality_sum.clamp(0.0, admitted as f64)
        } else {
            0.0
        };
        self.last_cycle = state.last_cycle;
        self.current_tier = ContextTier::Rejected;
        self.last_admitted_live = None;
        self.latest = None;
        self.consecutive_gold = 0;
        self.local_gold_total = 0;
        self.pending_actions.clear();
        self.staged_decision_episodes.clear();
        self.family_stats = state.family_stats;
        self.action_models = state
            .action_models
            .into_iter()
            .filter_map(|(key, mut model)| {
                let valid = key.len() <= 320
                    && model.utility_ema.is_finite()
                    && model.utility_variance_ema.is_finite()
                    && model.evidence_mass.is_finite()
                    && model.quality_ema.is_finite()
                    && model.state_delta_ema.is_finite()
                    && model.state_variance_ema.is_finite()
                    && model.state_evidence_mass.is_finite();
                if !valid {
                    return None;
                }
                model.utility_ema = model.utility_ema.clamp(-1.0, 1.0);
                model.utility_variance_ema = model.utility_variance_ema.clamp(0.0, 1.0);
                model.quality_ema = model.quality_ema.clamp(0.0, 1.0);
                if same_origin && model.installation_id == self.installation_id {
                    model.evidence_mass = model.evidence_mass.clamp(0.0, ACTION_MODEL_EVIDENCE_CAP);
                    model.state_evidence_mass = model
                        .state_evidence_mass
                        .clamp(0.0, ACTION_MODEL_EVIDENCE_CAP);
                } else {
                    model.evidence_mass = 0.0;
                    model.state_evidence_mass = 0.0;
                }
                model.state_delta_ema = model.state_delta_ema.clamped(-1.0, 1.0);
                model.state_variance_ema = model.state_variance_ema.clamped(0.0, 1.0);
                Some((key, model))
            })
            .take(MAX_ACTION_MODELS)
            .collect();
        self.action_model_evictions_total = state.action_model_evictions_total;
        self.action_model_births_total = state.action_model_births_total;
        self.action_model_evidence_updates_total = state.action_model_evidence_updates_total;
        self.action_model_last_evidence_cycle = state.action_model_last_evidence_cycle;
        self.recent_evidence = state
            .recent_evidence
            .into_iter()
            .filter_map(|mut evidence| {
                evidence.calibration_provenance = evidence.calibration_provenance.bounded();
                evidence.learning_details = None;
                (evidence.action_key.len() <= 320
                    && evidence.target.len() <= 256
                    && evidence.workload.len() <= 64
                    && evidence.quality.is_finite()
                    && evidence.raw_utility_delta.is_finite()
                    && evidence.counterfactual_delta.is_finite()
                    && evidence.net_utility_delta.is_finite()
                    && evidence.utility.is_finite()
                    && evidence.attribution.predicted_gain.is_finite()
                    && evidence.attribution.uncertainty.is_finite()
                    && (!evidence.context_before.valid || evidence.context_before.is_finite()))
                .then_some(evidence)
            })
            .take(MAX_RECENT_EVIDENCE)
            .collect();
        self.episodic_evidence.clear();
        let installation_id = self.installation_id;
        for evidence in state
            .episodic_evidence
            .into_iter()
            .filter_map(|mut evidence| {
                evidence.calibration_provenance = evidence.calibration_provenance.bounded();
                if !valid_learning_details(&evidence, installation_id, live_hardware) {
                    evidence.learning_details = None;
                }
                (evidence.tier != EvidenceTier::Bronze
                    && evidence.action_key.len() <= 320
                    && evidence.target.len() <= 256
                    && evidence.workload.len() <= 64
                    && evidence.quality.is_finite()
                    && evidence.net_utility_delta.is_finite()
                    && evidence.utility.is_finite()
                    && evidence.context_before.valid
                    && evidence.context_before.is_finite())
                .then_some(evidence)
            })
            .take(MAX_EPISODIC_EVIDENCE)
        {
            self.admit_episode(evidence);
        }
        self.external_counters = ExternalActuatorCounters::default();
        self.next_action_id = state.next_action_id;
        self.actuator_issued_total = state.actuator_issued_total;
        self.actuator_resolved_total = state.actuator_resolved_total;
        self.actuator_silver_total = state
            .actuator_silver_total
            .min(self.actuator_resolved_total);
        self.actuator_gold_total = state.actuator_gold_total.min(self.actuator_silver_total);
        self.actuator_effective_total = state
            .actuator_effective_total
            .min(self.actuator_resolved_total);
        self.actuator_rejected_total = state
            .actuator_rejected_total
            .min(self.actuator_resolved_total);
        self.actuator_expired_total = state.actuator_expired_total;
        self.actuator_quality_sum = finite_or_zero(state.actuator_quality_sum)
            .clamp(0.0, self.actuator_resolved_total as f64);
        self.actuator_utility_sum = finite_or_zero(state.actuator_utility_sum).clamp(
            -(self.actuator_resolved_total as f64),
            self.actuator_resolved_total as f64,
        );
        self.apollo_utility_ema = finite_or_zero(state.apollo_utility_ema).clamp(-1.0, 1.0);
        self.apollo_utility_observations = state.apollo_utility_observations;
        self.decision_source_stats = if same_origin {
            state
                .decision_source_stats
                .into_iter()
                .filter_map(|(source, mut stats)| {
                    if source.is_empty()
                        || source.len() > 48
                        || !stats.credit_ema.is_finite()
                        || !stats.absolute_error_ema.is_finite()
                    {
                        return None;
                    }
                    stats.correct = stats.correct.min(stats.observations);
                    stats.supports = stats.supports.min(stats.observations);
                    stats.vetoes = stats.vetoes.min(stats.observations);
                    stats.credit_ema = stats.credit_ema.clamp(-1.0, 1.0);
                    stats.absolute_error_ema = stats.absolute_error_ema.clamp(0.0, 2.0);
                    Some((source, stats))
                })
                .take(MAX_DECISION_SOURCES)
                .collect()
        } else {
            BTreeMap::new()
        };
        self.model_calibration = ModelCalibrationStore::new(self.installation_id);
        if same_origin && !reset_model_calibration {
            if let Some(model_calibration) = model_calibration {
                self.model_calibration
                    .restore(model_calibration, current_hardware);
            }
        }
        strip_invalid_restored_trust_chains(&mut self.episodic_evidence, &self.model_calibration);
        self.staged_attributions.clear();
        self.staged_decision_episodes.clear();
        self.no_action_delta_ema = if same_origin {
            state
                .no_action_delta_ema
                .into_iter()
                .filter(|(_, value)| value.is_finite())
                .map(|(key, value)| (key, value.clamp(-0.05, 0.05)))
                .collect()
        } else {
            BTreeMap::new()
        };
        self.no_action_state_delta_ema =
            if same_origin && state.no_action_state_delta_ema.is_finite() {
                state.no_action_state_delta_ema.clamped(-0.05, 0.05)
            } else {
                WorldStateDelta::default()
            };
        self.pending_controlled_holdouts.clear();
        self.controlled_models = if same_origin {
            state
                .controlled_models
                .into_iter()
                .filter_map(|(key, mut stats)| {
                    if key.len() > 384
                        || !stats.control_utility_ema.is_finite()
                        || !stats.quality_ema.is_finite()
                    {
                        return None;
                    }
                    stats.control_utility_ema = stats.control_utility_ema.clamp(-1.0, 1.0);
                    stats.quality_ema = stats.quality_ema.clamp(0.0, 1.0);
                    stats.would_have_helped = stats.would_have_helped.min(stats.observations);
                    if stats.last_observed_unix <= 0 {
                        stats.hardware_regime = HardwareRegime::default();
                        stats.installation_id = InstallationId::UNKNOWN;
                    }
                    Some((key, stats))
                })
                .take(MAX_CONTROLLED_MODELS)
                .collect()
        } else {
            BTreeMap::new()
        };
        if same_origin {
            self.controlled_holdout_issued_total = state.controlled_holdout_issued_total;
            self.controlled_holdout_resolved_total = state.controlled_holdout_resolved_total;
            self.controlled_holdout_rejected_total = state
                .controlled_holdout_rejected_total
                .saturating_add(state.controlled_holdout_pending_total);
        } else {
            self.controlled_holdout_issued_total = 0;
            self.controlled_holdout_resolved_total = 0;
            self.controlled_holdout_rejected_total = 0;
        }
        self.causal_dynamics = state
            .causal_dynamics
            .sanitized_for_restore(self.installation_id, same_origin);
        let reset_gpu_predictions =
            !same_origin || state.gpu_prediction_schema_version < GPU_PREDICTION_SCHEMA_VERSION;
        self.gpu_predictions.clear();
        self.gpu_calibration_models.clear();
        self.gpu_same_cycle_consumed.clear();
        if !reset_gpu_predictions {
            let abandoned = state
                .gpu_predictions
                .iter()
                .filter(|prediction| prediction.resolved_cycle.is_none())
                .count() as u64;
            self.gpu_predictions = state
                .gpu_predictions
                .into_iter()
                .filter(|prediction| {
                    prediction.resolved_cycle.is_some()
                        && prediction.action_key.len() <= 320
                        && prediction.workload.len() <= 64
                        && prediction.installation_id == self.installation_id
                        && [
                            prediction.expected_gain,
                            prediction.uncertainty,
                            prediction.mean_gain,
                            prediction.p10_gain,
                            prediction.positive_probability,
                            prediction.rank_support,
                            prediction.context_score,
                            prediction.quality,
                        ]
                        .into_iter()
                        .all(f64::is_finite)
                        && prediction.actual_utility.is_none_or(f64::is_finite)
                        && prediction.absolute_error.is_none_or(f64::is_finite)
                        && prediction.brier_score.is_none_or(f64::is_finite)
                })
                .take(MAX_GPU_PREDICTIONS)
                .collect();
            self.gpu_calibration_models = state
                .gpu_calibration_models
                .into_iter()
                .filter_map(|(key, mut stats)| {
                    if key.len() > 384
                        || stats.installation_id != self.installation_id
                        || ![
                            stats.signed_error_ema,
                            stats.absolute_error_ema,
                            stats.brier_ema,
                            stats.p10_coverage_ema,
                            stats.quality_ema,
                            stats.evidence_mass,
                        ]
                        .into_iter()
                        .all(f64::is_finite)
                    {
                        return None;
                    }
                    stats.used = stats.used.min(stats.predictions);
                    stats.resolved = stats.resolved.min(stats.used);
                    stats.gold = stats.gold.min(stats.resolved);
                    stats.signed_error_ema = stats.signed_error_ema.clamp(-2.0, 2.0);
                    stats.absolute_error_ema = stats.absolute_error_ema.clamp(0.0, 2.0);
                    stats.brier_ema = stats.brier_ema.clamp(0.0, 1.0);
                    stats.p10_coverage_ema = stats.p10_coverage_ema.clamp(0.0, 1.0);
                    stats.quality_ema = stats.quality_ema.clamp(0.0, 1.0);
                    stats.evidence_mass = stats.evidence_mass.clamp(0.0, 64.0);
                    Some((key, stats))
                })
                .take(MAX_GPU_CALIBRATION_MODELS)
                .collect();
            self.gpu_prediction_bronze_total = state.gpu_prediction_bronze_total;
            self.gpu_prediction_silver_total = state
                .gpu_prediction_silver_total
                .min(self.gpu_prediction_bronze_total);
            self.gpu_prediction_gold_total = state
                .gpu_prediction_gold_total
                .min(self.gpu_prediction_silver_total);
            self.gpu_prediction_rejected_total = state
                .gpu_prediction_rejected_total
                .saturating_add(abandoned)
                .min(self.gpu_prediction_bronze_total);
            // Predictions abandoned by the restart were never consumed, so
            // they belong to the unused bucket.
            self.gpu_prediction_evicted_total = state
                .gpu_prediction_evicted_total
                .min(self.gpu_prediction_rejected_total);
            self.gpu_prediction_unused_total = state
                .gpu_prediction_unused_total
                .saturating_add(abandoned)
                .min(self.gpu_prediction_rejected_total);
            self.gpu_prediction_bronze_rejected_total = state
                .gpu_prediction_bronze_rejected_total
                .min(self.gpu_prediction_rejected_total);
        } else {
            self.gpu_prediction_bronze_total = 0;
            self.gpu_prediction_silver_total = 0;
            self.gpu_prediction_gold_total = 0;
            self.gpu_prediction_rejected_total = 0;
            self.gpu_prediction_evicted_total = 0;
            self.gpu_prediction_unused_total = 0;
            self.gpu_prediction_bronze_rejected_total = 0;
        }
        if !same_origin {
            for evidence in &mut self.recent_evidence {
                evidence.installation_id = state.installation_id;
            }
            self.episodic_evidence.clear();
        }
        if reset_actuator_evidence {
            self.pending_actions.clear();
            self.family_stats.clear();
            self.action_models.clear();
            self.recent_evidence.clear();
            self.episodic_evidence.clear();
            self.new_gold_evidence.clear();
            self.external_counters = ExternalActuatorCounters::default();
            self.next_action_id = 0;
            self.actuator_issued_total = 0;
            self.actuator_resolved_total = 0;
            self.actuator_silver_total = 0;
            self.actuator_gold_total = 0;
            self.actuator_effective_total = 0;
            self.actuator_rejected_total = 0;
            self.actuator_expired_total = 0;
            self.actuator_quality_sum = 0.0;
            self.actuator_utility_sum = 0.0;
            self.apollo_utility_ema = 0.0;
            self.apollo_utility_observations = 0;
            self.decision_source_stats.clear();
            self.model_calibration = ModelCalibrationStore::new(self.installation_id);
            self.staged_attributions.clear();
            self.no_action_delta_ema.clear();
            self.no_action_state_delta_ema = WorldStateDelta::default();
            self.pending_controlled_holdouts.clear();
            self.controlled_models.clear();
            self.controlled_holdout_issued_total = 0;
            self.controlled_holdout_resolved_total = 0;
            self.controlled_holdout_rejected_total = 0;
            self.causal_dynamics = CausalDynamicsModel::new(self.installation_id);
            self.gpu_predictions.clear();
            self.gpu_calibration_models.clear();
            self.gpu_same_cycle_consumed.clear();
            self.gpu_prediction_bronze_total = 0;
            self.gpu_prediction_silver_total = 0;
            self.gpu_prediction_gold_total = 0;
            self.gpu_prediction_rejected_total = 0;
            self.gpu_prediction_evicted_total = 0;
            self.gpu_prediction_unused_total = 0;
            self.gpu_prediction_bronze_rejected_total = 0;
        }
        self.action_models_revision = self.action_models_revision.wrapping_add(1);
        self.controlled_models_revision = self.controlled_models_revision.wrapping_add(1);
        self.gpu_calibration_revision = self.gpu_calibration_revision.wrapping_add(1);
    }
}

fn bounded_text(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn bounded_sources(sources: Vec<String>) -> Vec<String> {
    let mut sources: Vec<String> = sources
        .into_iter()
        .map(|source| bounded_text(source.trim(), 48))
        .filter(|source| !source.is_empty())
        .collect();
    sources.sort();
    sources.dedup();
    sources.truncate(8);
    sources
}

fn source_authority_score(stats: &DecisionSourceStats) -> f64 {
    let maturity = stats.observations as f64 / (stats.observations as f64 + 8.0);
    stats.credit_ema * maturity * (0.5 + 0.5 * stats.accuracy())
}

fn gpu_action_matches(predicted: &str, observed: &str) -> bool {
    predicted == observed
        || (predicted == "interaction_qos:foreground"
            && observed.starts_with("interaction_qos:foreground@"))
        || (predicted == "predictive_prethrottle:noise"
            && observed.starts_with("predictive_prethrottle:"))
        || (predicted == "predictive_purge:kernel" && observed.starts_with("predictive_purge:"))
}

fn decision_episode_attribution(episode: &ResolvedDecisionEpisode) -> Option<DecisionAttribution> {
    let receipt_attribution = episode
        .envelope
        .receipt
        .as_ref()
        .and_then(|receipt| receipt.attribution.as_ref())
        .or(episode.envelope.terminal_attribution.as_ref());
    let ReceiptAttribution::Local { source } = receipt_attribution? else {
        return None;
    };
    if source.is_empty() {
        return None;
    }
    let mut attribution = DecisionAttribution {
        action_key: episode.envelope.action_key.clone(),
        proposer: source.clone(),
        ..DecisionAttribution::default()
    };
    for adviser in &episode.envelope.adviser_contributions {
        if adviser.support < 0.0 {
            attribution.vetoes.push(adviser.adviser.clone());
        } else {
            attribution.supporters.push(adviser.adviser.clone());
        }
    }
    if let Some(prediction) = episode.envelope.predictions.first() {
        attribution.predicted_gain = prediction.expected_utility;
        attribution.uncertainty = prediction.uncertainty;
        if prediction.source != *source {
            attribution.supporters.push(prediction.source.clone());
        }
    }
    Some(attribution.bounded())
}

fn calibration_provenance_for_episode(
    episode: &ResolvedDecisionEpisode,
    cohort_size: u16,
) -> CalibrationProvenance {
    let proposer = episode
        .envelope
        .receipt
        .as_ref()
        .and_then(|receipt| receipt.attribution.as_ref())
        .or(episode.envelope.terminal_attribution.as_ref())
        .and_then(|attribution| match attribution {
            ReceiptAttribution::Local { source } if !source.is_empty() => Some(source.clone()),
            _ => None,
        })
        .unwrap_or_default();
    let mut prediction_sources = BTreeSet::new();
    let predictions = episode
        .envelope
        .predictions
        .iter()
        .filter(|prediction| {
            !prediction.source.is_empty() && prediction_sources.insert(prediction.source.clone())
        })
        .take(8)
        .cloned()
        .collect();
    let mut advisers = BTreeSet::new();
    let adviser_contributions = episode
        .envelope
        .adviser_contributions
        .iter()
        .filter(|adviser| !adviser.adviser.is_empty() && advisers.insert(adviser.adviser.clone()))
        .take(8)
        .cloned()
        .collect();
    let separability = if episode.envelope.action_key.starts_with("coordinated:") {
        SeparabilityState::CoordinatedComposite
    } else if cohort_size > 1 {
        SeparabilityState::Confounded
    } else {
        SeparabilityState::Individual
    };
    CalibrationProvenance {
        local_authority_eligible: episode.authority_eligible
            && episode.envelope.lifecycle == DecisionLifecycle::Applied,
        proposer,
        alternatives: episode
            .envelope
            .alternatives
            .iter()
            .take(8)
            .cloned()
            .collect(),
        predictions,
        adviser_contributions,
        hierarchy: episode.envelope.hierarchy,
        cohort_size,
        separability,
    }
    .bounded()
}

fn learning_details_for_evidence(
    evidence: &ResolvedActuatorEvidence,
    calibration_deltas: Vec<crate::engine::model_calibration::ForecastCalibrationDelta>,
) -> Option<ResolvedLearningDetails> {
    let decision_id = evidence.decision_id?;
    let hierarchy = HierarchyPath::classify(evidence.family, &evidence.action_key)?;
    let context = HierarchyContext::classify(&evidence.workload, &evidence.context_before)?;
    let expected_utility = evidence
        .calibration_provenance
        .predictions
        .iter()
        .find(|prediction| prediction.source == evidence.calibration_provenance.proposer)
        .or_else(|| evidence.calibration_provenance.predictions.first())?
        .expected_utility;
    let details = ResolvedLearningDetails {
        decision_id,
        lifecycle: DecisionLifecycle::Applied,
        hierarchy,
        context,
        alternatives: evidence.calibration_provenance.alternatives.clone(),
        predictions: evidence.calibration_provenance.predictions.clone(),
        adviser_contributions: evidence
            .calibration_provenance
            .adviser_contributions
            .clone(),
        expected_utility,
        actual_utility: evidence.utility.apollo_utility,
        raw_utility_delta: evidence.raw_utility_delta,
        counterfactual_delta: evidence.counterfactual_delta,
        quality: evidence.quality,
        causal_quality: evidence.quality * (1.0 - f64::from(evidence.confounder_count) / 6.0),
        confounder_count: evidence.confounder_count,
        separability: evidence.calibration_provenance.separability,
        calibration_deltas,
        installation_id: evidence.installation_id,
        hardware_regime: evidence.hardware_regime,
        resolved_cycle: evidence.resolved_cycle,
        resolved_timestamp_unix: evidence.resolved_timestamp_unix,
    };
    details.is_authoritative().then_some(details)
}

pub(crate) fn valid_learning_details(
    evidence: &ResolvedActuatorEvidence,
    installation_id: InstallationId,
    live_hardware: HardwareRegime,
) -> bool {
    let Some(details) = evidence.learning_details.as_ref() else {
        return true;
    };
    let expected_utility = evidence
        .calibration_provenance
        .predictions
        .iter()
        .find(|prediction| prediction.source == evidence.calibration_provenance.proposer)
        .or_else(|| evidence.calibration_provenance.predictions.first())
        .map(|prediction| prediction.expected_utility);
    details.is_authoritative()
        && evidence.tier == EvidenceTier::Gold
        && evidence.decision_id == Some(details.decision_id)
        && evidence.calibration_provenance.local_authority_eligible
        && match evidence.family {
            ActuatorFamily::Coordinated => {
                evidence.calibration_provenance.cohort_size > 0
                    && evidence.calibration_provenance.separability
                        == SeparabilityState::CoordinatedComposite
            }
            _ => evidence.calibration_provenance.cohort_size == 1,
        }
        && details.installation_id == installation_id
        && details.installation_id == evidence.installation_id
        && live_hardware.is_known()
        && details.hardware_regime == live_hardware
        && details.hardware_regime == evidence.hardware_regime
        && details.resolved_cycle == evidence.resolved_cycle
        && details.resolved_timestamp_unix == evidence.resolved_timestamp_unix
        && HierarchyPath::classify(evidence.family, &evidence.action_key)
            .is_some_and(|path| path == details.hierarchy)
        && HierarchyContext::classify(&evidence.workload, &evidence.context_before)
            .is_some_and(|context| context == details.context)
        && details.alternatives == evidence.calibration_provenance.alternatives
        && details.predictions == evidence.calibration_provenance.predictions
        && details.adviser_contributions == evidence.calibration_provenance.adviser_contributions
        && details.separability == evidence.calibration_provenance.separability
        && expected_utility
            .is_some_and(|expected| (details.expected_utility - expected).abs() <= f64::EPSILON)
        && (details.actual_utility - evidence.utility.apollo_utility).abs() <= f64::EPSILON
        && (details.raw_utility_delta - evidence.raw_utility_delta).abs() <= f64::EPSILON
        && (details.counterfactual_delta - evidence.counterfactual_delta).abs() <= f64::EPSILON
        && (details.quality - evidence.quality).abs() <= f64::EPSILON
        && (details.causal_quality - evidence.quality).abs() <= f64::EPSILON
        && details.confounder_count == evidence.confounder_count
        && valid_forecast_deltas(
            &calibration_observation(evidence),
            &details.calibration_deltas,
        )
}

fn strip_invalid_restored_trust_chains(
    evidence: &mut VecDeque<ResolvedActuatorEvidence>,
    calibration: &ModelCalibrationStore,
) {
    let mut chains: BTreeMap<
        (ProducerId, CalibrationActionScope),
        Vec<(DecisionId, TrustState, TrustState, CalibrationKey)>,
    > = BTreeMap::new();
    for episode in evidence.iter() {
        let Some(details) = episode.learning_details.as_ref() else {
            continue;
        };
        for delta in &details.calibration_deltas {
            chains
                .entry((delta.key.producer, delta.key.action.clone()))
                .or_default()
                .push((
                    details.decision_id,
                    delta.trust_before,
                    delta.trust_after,
                    delta.key.clone(),
                ));
        }
    }

    let mut invalid = BTreeSet::new();
    for chain in chains.values() {
        let continuous = chain.windows(2).all(|pair| pair[0].2 == pair[1].1);
        let anchored = chain.last().is_some_and(|(_, _, trust_after, key)| {
            calibration.record(key).is_some() && calibration.trust_for(key) == *trust_after
        });
        if !continuous || !anchored {
            invalid.extend(chain.iter().map(|(decision_id, ..)| *decision_id));
        }
    }
    for episode in evidence {
        if episode
            .learning_details
            .as_ref()
            .is_some_and(|details| invalid.contains(&details.decision_id))
        {
            episode.learning_details = None;
        }
    }
}

fn calibration_observation(evidence: &ResolvedActuatorEvidence) -> CalibrationObservation<'_> {
    let pressure = PressureBand::from_fraction(evidence.context_before.memory_pressure);
    let thermal = ThermalBand::from_fraction(evidence.context_before.thermal_score);
    let context_valid = evidence.context_before.valid
        && evidence.context_before.is_finite()
        && pressure.is_some()
        && thermal.is_some();
    let foreground = if evidence.context_before.app_launching {
        ForegroundContext::Launching
    } else if evidence.context_before.foreground_idle {
        ForegroundContext::Idle
    } else if evidence.context_before.foreground_app_hash != 0 {
        ForegroundContext::Active
    } else {
        ForegroundContext::Unknown
    };
    let process_class = ProcessClass::from_target(
        evidence.family,
        &evidence.target,
        matches!(
            foreground,
            ForegroundContext::Active | ForegroundContext::Launching
        ),
    );
    CalibrationObservation {
        decision_id: evidence.decision_id,
        tier: evidence.tier,
        installation_id: evidence.installation_id,
        hardware_regime: evidence.hardware_regime,
        family: evidence.family,
        action_key: evidence.action_key.clone(),
        workload: evidence.workload.clone(),
        process_class,
        pressure: pressure.unwrap_or_default(),
        thermal: thermal.unwrap_or_default(),
        foreground,
        context_valid,
        quality: evidence.quality,
        actual_utility: evidence.utility.apollo_utility,
        effective: evidence.effective,
        provenance: &evidence.calibration_provenance,
    }
}

fn target_pid_from_decision_target(target: &str) -> Option<u32> {
    target
        .rsplit_once(":pid:")
        .and_then(|(_, pid)| pid.parse().ok())
        .or_else(|| target.strip_prefix("pid:").and_then(|pid| pid.parse().ok()))
}

fn decision_episode_spec(episode: &ResolvedDecisionEpisode) -> Option<ActionSpec> {
    if episode.envelope.lifecycle != DecisionLifecycle::Applied || !episode.authority_eligible {
        return None;
    }
    let key = episode.envelope.action_key.as_str();
    let family_name = key.split_once(':').map_or(key, |(family, _)| family);
    let (family, objective, horizon_cycles) = match family_name {
        "boost" => (ActuatorFamily::Boost, ActuatorObjective::Responsiveness, 3),
        "throttle" => (
            ActuatorFamily::Throttle,
            ActuatorObjective::PressureRelief,
            5,
        ),
        "freeze" => (ActuatorFamily::Freeze, ActuatorObjective::PressureRelief, 5),
        "unfreeze" => (ActuatorFamily::Unfreeze, ActuatorObjective::Recovery, 3),
        "memorystatus" => (
            ActuatorFamily::Memorystatus,
            ActuatorObjective::PressureRelief,
            8,
        ),
        "sysctl" => (ActuatorFamily::Sysctl, ActuatorObjective::NetworkHealth, 15),
        "spotlight" => (
            ActuatorFamily::Spotlight,
            ActuatorObjective::Availability,
            15,
        ),
        "quarantine" => (
            ActuatorFamily::Quarantine,
            ActuatorObjective::Efficiency,
            10,
        ),
        "thread_qos" => (
            ActuatorFamily::ThreadQos,
            ActuatorObjective::Responsiveness,
            3,
        ),
        "markov_prewarm" => (
            ActuatorFamily::MarkovPrewarm,
            ActuatorObjective::Prediction,
            120,
        ),
        "interaction_qos" => (
            ActuatorFamily::InteractionQos,
            ActuatorObjective::Responsiveness,
            30,
        ),
        "io_shaping" => (
            ActuatorFamily::IoShaping,
            ActuatorObjective::Responsiveness,
            30,
        ),
        "predictive_threshold" => (
            ActuatorFamily::PredictiveThreshold,
            ActuatorObjective::Prediction,
            12,
        ),
        "predictive_profile" => (
            ActuatorFamily::PredictiveProfile,
            ActuatorObjective::Prediction,
            12,
        ),
        "predictive_prethrottle" => (
            ActuatorFamily::PredictivePreThrottle,
            ActuatorObjective::PressureRelief,
            5,
        ),
        "predictive_purge" => (
            ActuatorFamily::PredictivePurge,
            ActuatorObjective::PressureRelief,
            8,
        ),
        "chromium_ecore" => (
            ActuatorFamily::ChromiumEcore,
            ActuatorObjective::Efficiency,
            30,
        ),
        "chromium_purge" => (
            ActuatorFamily::ChromiumPurge,
            ActuatorObjective::PressureRelief,
            12,
        ),
        "chromium_jetsam" => (
            ActuatorFamily::ChromiumJetsam,
            ActuatorObjective::Efficiency,
            30,
        ),
        "coordinated" => (
            ActuatorFamily::Coordinated,
            ActuatorObjective::BalancedUtility,
            8,
        ),
        _ => return None,
    };
    let horizon_cycles = episode
        .envelope
        .predictions
        .first()
        .map_or(horizon_cycles, |prediction| {
            prediction.horizon_cycles.max(1)
        });
    Some(ActionSpec {
        family,
        objective,
        action_key: bounded_text(key, 320),
        target: bounded_text(&episode.envelope.target, 256),
        target_pid: target_pid_from_decision_target(&episode.envelope.target),
        horizon_cycles,
        provenance: EvidenceProvenance::ObservedLocal,
    })
}

fn action_spec(action: &RootAction) -> Option<ActionSpec> {
    let (family, objective, target, target_pid, horizon_cycles) = match action {
        RootAction::BoostProcess { pid, name, .. } => (
            ActuatorFamily::Boost,
            ActuatorObjective::Responsiveness,
            name.clone(),
            Some(*pid),
            3,
        ),
        RootAction::ThrottleProcess {
            pid, name, reason, ..
        } => (
            if reason.contains("predictive-agent: pre-throttle noise") {
                ActuatorFamily::PredictivePreThrottle
            } else {
                ActuatorFamily::Throttle
            },
            ActuatorObjective::PressureRelief,
            name.clone(),
            Some(*pid),
            5,
        ),
        RootAction::FreezeProcess { pid, name, .. } => (
            ActuatorFamily::Freeze,
            ActuatorObjective::PressureRelief,
            name.clone(),
            Some(*pid),
            5,
        ),
        RootAction::UnfreezeProcess { pid, name, .. } => (
            ActuatorFamily::Unfreeze,
            ActuatorObjective::Recovery,
            name.clone(),
            Some(*pid),
            3,
        ),
        RootAction::SetMemorystatus {
            pid,
            priority,
            reason,
            ..
        } => (
            if reason.contains("predictive-agent: proactive purge") {
                ActuatorFamily::PredictivePurge
            } else {
                ActuatorFamily::Memorystatus
            },
            ActuatorObjective::PressureRelief,
            format!("pid:{pid}:priority:{priority}"),
            Some(*pid),
            8,
        ),
        RootAction::SetSysctl(action) => (
            ActuatorFamily::Sysctl,
            ActuatorObjective::NetworkHealth,
            format!("{}={}", action.key(), action.value()),
            None,
            15,
        ),
        RootAction::ToggleSpotlight { enabled, .. } => (
            ActuatorFamily::Spotlight,
            if *enabled {
                ActuatorObjective::Availability
            } else {
                ActuatorObjective::Efficiency
            },
            if *enabled { "enabled" } else { "disabled" }.to_string(),
            None,
            15,
        ),
        RootAction::QuarantineDaemon { daemon, active, .. } => (
            ActuatorFamily::Quarantine,
            if *active {
                ActuatorObjective::Efficiency
            } else {
                ActuatorObjective::Recovery
            },
            format!("{}:{}", daemon, if *active { "active" } else { "released" }),
            None,
            10,
        ),
        RootAction::SetThreadQoS {
            pid, name, tier, ..
        } => (
            ActuatorFamily::ThreadQos,
            if tier == "interactive" {
                ActuatorObjective::Responsiveness
            } else {
                ActuatorObjective::Efficiency
            },
            format!("{name}:{tier}"),
            Some(*pid),
            3,
        ),
    };
    let target = bounded_text(&target, 256);
    let model_target = stable_model_target(family, &target);
    Some(ActionSpec {
        family,
        objective,
        action_key: format!("{}:{}", family.as_str(), model_target),
        target,
        target_pid,
        horizon_cycles,
        provenance: EvidenceProvenance::ObservedLocal,
    })
}

fn stable_model_target(family: ActuatorFamily, target: &str) -> String {
    match family {
        // A PID is an execution identity, not a reusable learning identity.
        ActuatorFamily::Memorystatus => target
            .rsplit_once(":priority:")
            .map(|(_, priority)| format!("priority:{priority}"))
            .unwrap_or_else(|| bounded_text(target, 256)),
        ActuatorFamily::Boost
        | ActuatorFamily::Throttle
        | ActuatorFamily::Freeze
        | ActuatorFamily::Unfreeze => {
            let class = crate::engine::freeze_intelligence::FreezeIntelligence::classify(target);
            if class == "generic" {
                bounded_text(target, 256)
            } else {
                class.to_string()
            }
        }
        ActuatorFamily::ThreadQos => {
            let Some((name, tier)) = target.rsplit_once(':') else {
                return bounded_text(target, 256);
            };
            let class = crate::engine::freeze_intelligence::FreezeIntelligence::classify(name);
            if class == "generic" {
                bounded_text(target, 256)
            } else {
                format!("{class}:{tier}")
            }
        }
        _ => bounded_text(target, 256),
    }
}

pub fn actuator_action_key(action: &RootAction) -> Option<String> {
    action_spec(action).map(|spec| spec.action_key)
}

pub fn actuator_horizon_cycles(action: &RootAction) -> Option<u64> {
    action_spec(action).map(|spec| spec.horizon_cycles)
}

pub fn utility_veto_eligible(action: &RootAction) -> bool {
    matches!(
        action,
        RootAction::BoostProcess { .. }
            | RootAction::SetThreadQoS { .. }
            | RootAction::SetSysctl(_)
            | RootAction::ToggleSpotlight { .. }
            | RootAction::QuarantineDaemon { active: true, .. }
    )
}

fn intervention_spec(intervention: Intervention) -> Option<ActionSpec> {
    let (family, objective, key) = match intervention {
        Intervention::Observe => return None,
        Intervention::TightenThresholds => (
            ActuatorFamily::PredictiveThreshold,
            ActuatorObjective::Prediction,
            "predictive_threshold:tighten",
        ),
        Intervention::SuggestAggressive => (
            ActuatorFamily::PredictiveProfile,
            ActuatorObjective::Prediction,
            "predictive_profile:aggressive",
        ),
        Intervention::PreThrottleNoise => (
            ActuatorFamily::PredictivePreThrottle,
            ActuatorObjective::PressureRelief,
            "predictive_prethrottle:noise",
        ),
        Intervention::ProactivePurge => (
            ActuatorFamily::PredictivePurge,
            ActuatorObjective::PressureRelief,
            "predictive_purge:kernel",
        ),
    };
    Some(ActionSpec {
        family,
        objective,
        action_key: key.to_string(),
        target: key.to_string(),
        target_pid: None,
        horizon_cycles: 5,
        provenance: EvidenceProvenance::ObservedLocal,
    })
}

fn target_presence(pending: &PendingActuatorEvidence, snapshot: &SystemSnapshot) -> Option<bool> {
    if pending.objective != ActuatorObjective::Recovery {
        return None;
    }
    let pid = pending.target_pid?;
    Some(snapshot.top_processes.iter().any(|process| {
        process.pid == pid
            || (!pending.target.is_empty()
                && (process.name == pending.target || pending.target.starts_with(&process.name)))
    }))
}

fn objective_effect_threshold(objective: ActuatorObjective) -> f64 {
    match objective {
        ActuatorObjective::Prediction => 0.0,
        ActuatorObjective::Recovery
        | ActuatorObjective::Responsiveness
        | ActuatorObjective::BalancedUtility => 0.005,
        ActuatorObjective::PressureRelief
        | ActuatorObjective::Efficiency
        | ActuatorObjective::NetworkHealth
        | ActuatorObjective::Availability => 0.003,
    }
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn intervention_cost(family: ActuatorFamily) -> f64 {
    match family {
        ActuatorFamily::Freeze | ActuatorFamily::Quarantine => 0.020,
        ActuatorFamily::Sysctl => 0.015,
        ActuatorFamily::Throttle | ActuatorFamily::PredictivePreThrottle => 0.010,
        ActuatorFamily::Memorystatus | ActuatorFamily::PredictivePurge => 0.008,
        ActuatorFamily::Boost | ActuatorFamily::MarkovPrewarm | ActuatorFamily::ChromiumPurge => {
            0.004
        }
        ActuatorFamily::ThreadQos
        | ActuatorFamily::InteractionQos
        | ActuatorFamily::IoShaping
        | ActuatorFamily::ChromiumEcore
        | ActuatorFamily::ChromiumJetsam => 0.003,
        ActuatorFamily::Spotlight
        | ActuatorFamily::PredictiveThreshold
        | ActuatorFamily::PredictiveProfile
        | ActuatorFamily::Coordinated
        | ActuatorFamily::Unfreeze => 0.005,
    }
}

fn decompose_utility(family: ActuatorFamily, net_state: WorldStateDelta) -> UtilityDecomposition {
    // Positive gains are always improvements. WorldStateDelta uses positive
    // values for worsening resource dimensions and positive fluidity for UX.
    let system_gain = (-0.24 * net_state.pressure
        - 0.16 * net_state.energy
        - 0.14 * net_state.cpu
        - 0.14 * net_state.thermal
        - 0.16 * net_state.thrashing
        - 0.16 * net_state.stall)
        .clamp(-1.0, 1.0);
    let human_gain = (0.50 * net_state.fluidity
        - 0.30 * net_state.latency
        - 0.15 * net_state.stall
        - 0.05 * net_state.cpu)
        .clamp(-1.0, 1.0);
    let intervention_cost = intervention_cost(family);
    let apollo_utility =
        (0.60 * human_gain + 0.40 * system_gain - intervention_cost).clamp(-1.0, 1.0);
    UtilityDecomposition {
        system_gain,
        human_gain,
        intervention_cost,
        apollo_utility,
    }
}

fn utility_score(objective: ActuatorObjective, context: &TelemetryContextSummary) -> f64 {
    let pressure_health = 1.0 - context.memory_pressure.clamp(0.0, 1.0);
    let swap_health = 1.0
        - (context.swap_delta_bytes_per_sec.max(0.0) / (64.0 * 1024.0 * 1024.0)).clamp(0.0, 1.0);
    let thrash_health = 1.0 - (context.thrashing_score / 50_000.0).clamp(0.0, 1.0);
    let refault_health = 1.0 - (context.refault_delta_per_sec.max(0.0) / 10_000.0).clamp(0.0, 1.0);
    let cpu_headroom = 1.0 - context.cpu_max_busy.clamp(0.0, 1.0);
    let stall_health = 1.0 - context.stall_fraction.clamp(0.0, 1.0);
    let thermal_health = 1.0 - context.thermal_score.clamp(0.0, 1.0);
    let fluidity = context.fluidity_score.clamp(0.0, 1.0);
    let latency_health = 1.0 - context.perceptual_latency_score.clamp(0.0, 1.0);
    let ws_headroom = 1.0 - context.windowserver_cpu_fraction.clamp(0.0, 1.0);
    let energy_health = 1.0 - (context.package_watts.unwrap_or(0.0) / 50.0).clamp(0.0, 1.0);
    let network_health = 1.0
        - ((context.network_retransmits_per_k / 100.0).clamp(0.0, 1.0) * 0.7
            + context.network_listen_drop_rate.clamp(0.0, 1.0) * 0.3);
    let launch_health = if context.app_launching { 0.0 } else { 1.0 };

    match objective {
        ActuatorObjective::PressureRelief => {
            0.35 * pressure_health
                + 0.15 * swap_health
                + 0.15 * thrash_health
                + 0.10 * refault_health
                + 0.10 * cpu_headroom
                + 0.10 * thermal_health
                + 0.05 * fluidity
        }
        ActuatorObjective::Responsiveness => {
            0.30 * fluidity
                + 0.20 * latency_health
                + 0.15 * stall_health
                + 0.10 * refault_health
                + 0.10 * ws_headroom
                + 0.075 * pressure_health
                + 0.075 * thermal_health
        }
        ActuatorObjective::Efficiency => {
            0.25 * energy_health
                + 0.20 * cpu_headroom
                + 0.15 * thermal_health
                + 0.15 * pressure_health
                + 0.15 * fluidity
                + 0.10 * stall_health
        }
        ActuatorObjective::Recovery => {
            0.40 * fluidity
                + 0.20 * launch_health
                + 0.15 * refault_health
                + 0.10 * pressure_health
                + 0.10 * stall_health
                + 0.05 * thermal_health
        }
        ActuatorObjective::NetworkHealth => {
            0.45 * network_health
                + 0.20 * fluidity
                + 0.15 * cpu_headroom
                + 0.10 * pressure_health
                + 0.10 * thermal_health
        }
        ActuatorObjective::Availability => {
            0.35 * fluidity
                + 0.20 * launch_health
                + 0.15 * network_health
                + 0.10 * pressure_health
                + 0.10 * cpu_headroom
                + 0.10 * thermal_health
        }
        ActuatorObjective::Prediction => {
            0.35 * fluidity
                + 0.20 * launch_health
                + 0.15 * pressure_health
                + 0.10 * refault_health
                + 0.10 * stall_health
                + 0.10 * thermal_health
        }
        ActuatorObjective::BalancedUtility => {
            0.20 * pressure_health
                + 0.20 * fluidity
                + 0.15 * cpu_headroom
                + 0.10 * stall_health
                + 0.10 * refault_health
                + 0.10 * thermal_health
                + 0.10 * energy_health
                + 0.05 * network_health
        }
    }
    .clamp(0.0, 1.0)
}

fn thermal_score(level: &str) -> f64 {
    match level {
        "light" | "moderate" => 0.33,
        "serious" => 0.66,
        "critical" => 1.0,
        _ => 0.0,
    }
}

fn valid_runtime_fraction(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value.clamp(0.0, 1.0)
    } else {
        fallback.clamp(0.0, 1.0)
    }
}

fn summarize(observation: &TelemetryObservation<'_>) -> TelemetryContextSummary {
    let TelemetryObservation {
        snapshot,
        hardware,
        runtime,
        capabilities,
        signal,
        workload,
        cycle,
        outcomes,
        intervention,
        nars_drift_score,
        nars_beliefs_total,
        natural_drift,
        arousal_level,
        ..
    } = observation;
    let process_count = snapshot.top_processes.len() as u32;
    let total_process_rss_bytes = snapshot.top_processes.iter().map(|p| p.memory_usage).sum();
    let top = snapshot.top_processes.iter().max_by_key(|p| p.memory_usage);
    let cpu_global_usage = (snapshot.cpu.global_usage as f64 / 100.0).clamp(0.0, 1.0);
    let process_cpu_max = snapshot
        .top_processes
        .iter()
        .map(|p| p.cpu_usage as f64 / 100.0)
        .fold(cpu_global_usage, f64::max)
        .clamp(0.0, 1.0);
    let sampled_pegged_fraction = if process_count == 0 {
        0.0
    } else {
        snapshot
            .top_processes
            .iter()
            .filter(|p| p.cpu_usage >= 80.0)
            .count() as f64
            / process_count as f64
    };
    let total_ram = snapshot.memory.total_ram.max(1) as f64;
    let process_cpu_capacity_pct = snapshot.cpu.core_count.max(1) as f64 * 100.0;
    TelemetryContextSummary {
        cycle: *cycle,
        timestamp_unix: snapshot.timestamp.timestamp(),
        workload: (*workload).to_string(),
        memory_pressure: snapshot.pressure.memory_pressure,
        memory_pressure_raw: if snapshot.pressure.memory_pressure_raw > 0.0 {
            snapshot.pressure.memory_pressure_raw
        } else {
            snapshot.pressure.memory_pressure
        },
        compressor_pressure: snapshot.pressure.compressor_pressure,
        thrashing_score: snapshot.pressure.thrashing_score,
        refault_delta_per_sec: snapshot.pressure.refault_delta_per_sec,
        swap_used_bytes: snapshot.pressure.swap_used_bytes,
        swap_delta_bytes_per_sec: snapshot.pressure.swap_delta_bytes_per_sec,
        cpu_global_usage,
        cpu_mean_busy: valid_runtime_fraction(runtime.cpu_mean_busy, cpu_global_usage),
        cpu_max_busy: valid_runtime_fraction(runtime.cpu_max_busy, process_cpu_max),
        cpu_pegged_fraction: valid_runtime_fraction(
            runtime.cpu_pegged_fraction,
            sampled_pegged_fraction,
        ),
        cpu_core_count: snapshot.cpu.core_count.min(u32::MAX as usize) as u32,
        stall_fraction: runtime.stall_fraction.clamp(0.0, 1.0),
        used_ram_fraction: (snapshot.memory.used_ram as f64 / total_ram).clamp(0.0, 1.0),
        total_ram_bytes: snapshot.memory.total_ram,
        used_ram_bytes: snapshot.memory.used_ram,
        free_ram_bytes: snapshot.memory.free_ram,
        swap_total_bytes: snapshot
            .pressure
            .swap_total_bytes
            .max(snapshot.memory.total_swap),
        process_count,
        total_process_rss_bytes,
        top_process_cpu: top.map_or(0.0, |p| {
            (p.cpu_usage as f64 / process_cpu_capacity_pct).clamp(0.0, 1.0)
        }),
        top_process_rss_bytes: top.map_or(0, |p| p.memory_usage),
        disk_count: snapshot.disks.len() as u32,
        disk_total_bytes: snapshot.disks.iter().map(|d| d.total_space).sum(),
        disk_available_bytes: snapshot.disks.iter().map(|d| d.available_space).sum(),
        network_count: snapshot.networks.len() as u32,
        network_received_bytes: snapshot.networks.iter().map(|n| n.received).sum(),
        network_transmitted_bytes: snapshot.networks.iter().map(|n| n.transmitted).sum(),
        thermal_score: thermal_score(&snapshot.pressure.thermal_level),
        p_cluster_temp_c: hardware
            .and_then(|h| h.temps.p_cluster_celsius)
            .map(f64::from),
        e_cluster_temp_c: hardware
            .and_then(|h| h.temps.e_cluster_celsius)
            .map(f64::from),
        gpu_temp_c: hardware.and_then(|h| h.temps.gpu_celsius).map(f64::from),
        nand_temp_c: hardware.and_then(|h| h.temps.nand_celsius).map(f64::from),
        temperatures_estimated: hardware.is_some_and(|h| h.temps_estimated),
        p_cluster_util: hardware.and_then(|h| h.p_cluster_util).map(f64::from),
        e_cluster_util: hardware.and_then(|h| h.e_cluster_util).map(f64::from),
        package_watts: hardware
            .and_then(|h| h.power.package_watts)
            .map(f64::from)
            .or(runtime.energy_package_watts),
        cpu_watts: hardware
            .and_then(|h| h.power.cpu_watts)
            .map(f64::from)
            .or(runtime.energy_cpu_watts),
        gpu_watts: hardware
            .and_then(|h| h.power.gpu_watts)
            .map(f64::from)
            .or(runtime.energy_gpu_watts),
        dram_watts: hardware.and_then(|h| h.power.dram_watts).map(f64::from),
        ane_watts: hardware
            .and_then(|h| h.power.ane_watts)
            .map(f64::from)
            .or(runtime.energy_ane_watts),
        ane_util_pct: hardware
            .and_then(|h| h.power.ane_util_pct)
            .map(f64::from)
            .or(runtime.energy_ane_util_pct),
        battery_percent: hardware.and_then(|h| h.battery_percent),
        battery_watts: hardware.and_then(|h| h.battery_watts).map(f64::from),
        fluidity_score: (signal.fluidity_score as f64).clamp(0.0, 1.0),
        perceptual_latency_score: runtime.perceptual_latency_score.clamp(0.0, 1.0),
        scheduler_jitter_p95_ms: runtime.scheduler_jitter_p95_ms.max(0.0),
        windowserver_cpu_fraction: (runtime.windowserver_cpu_pct as f64 / 100.0).clamp(0.0, 1.0),
        network_retransmits_per_k: runtime.network_retransmit_ratio.max(0.0),
        network_listen_drop_rate: runtime.network_listen_drop_rate.max(0.0),
        foreground_app: runtime.foreground_app.as_ref().map(|app| app.name.clone()),
        foreground_idle: runtime.foreground_idle,
        app_launching: signal.app_launching || runtime.app_launching,
        window_op_active: signal.window_op_active || runtime.window_op_active,
        user_idle_secs: runtime.user_idle_secs.max(0.0),
        user_call_in_progress: runtime.user_call_in_progress,
        user_audio_active: runtime.user_audio_active,
        coreaudio_direct_probe_available: matches!(
            runtime.coreaudio_probe_state.as_str(),
            "direct" | "degraded"
        ),
        coreaudio_session_fallback: runtime.coreaudio_probe_state == "session-fallback",
        user_has_sleep_assertion: runtime.user_has_sleep_assertion,
        effective_profile: runtime.effective_profile.as_str().to_string(),
        pressure_total_boost: runtime.pressure_total_boost,
        pressure_dominant_factor: runtime.pressure_dominant_factor.clone(),
        collector_pressure_alive: runtime.collector_pressure_alive,
        collector_smc_alive: runtime.collector_smc_alive,
        reactor_healthy: matches!(runtime.reactor_health.as_str(), "ok" | "healthy"),
        operation_failures_total: runtime.failures,
        daemon_is_root: capabilities.is_some_and(|caps| caps.is_root),
        kernel_taskpolicy_available: capabilities.is_some_and(|caps| caps.can_taskpolicy),
        kernel_sysctl_available: capabilities.is_some_and(|caps| caps.can_sysctl),
        kernel_memorystatus_available: capabilities.is_some_and(|caps| caps.can_memorystatus),
        kernel_pressure_send_available: capabilities
            .is_some_and(|caps| caps.can_memory_pressure_send),
        p_core_count: capabilities.and_then(|caps| caps.p_core_count).unwrap_or(0),
        e_core_count: capabilities.and_then(|caps| caps.e_core_count).unwrap_or(0),
        unavailable_capability_count: capabilities.map_or(0, |caps| {
            caps.unavailable.len().min(u32::MAX as usize) as u32
        }),
        memorystatus_probe_ok: capabilities.and_then(|caps| caps.memorystatus_probe.as_deref())
            == Some("ok"),
        task_for_pid_probe_ok: capabilities.and_then(|caps| caps.task_for_pid_probe.as_deref())
            == Some("ok"),
        signal_pressure_smooth: signal.pressure_smooth,
        signal_pressure_velocity: signal.pressure_velocity,
        signal_p_oom_30s: signal.p_oom_30s,
        signal_urgency: signal.urgency,
        signal_entropy_anomaly: signal.entropy_anomaly,
        signal_transformer_anomaly: signal.transformer_anomaly,
        nars_drift_score: *nars_drift_score,
        nars_beliefs_total: *nars_beliefs_total,
        natural_drift: *natural_drift,
        arousal_level: *arousal_level,
        boosts_applied: outcomes.boosts_applied,
        throttles_applied: outcomes.throttles_applied,
        freezes_applied: outcomes.freezes_applied,
        paging_hints_applied: outcomes.paging_hints_applied,
        sysctl_applied: outcomes.sysctl_applied,
        unfreezes_applied: outcomes.unfreezes_applied,
        thread_qos_applied: outcomes.thread_qos_applied,
        markov_prediction_confidence: runtime.markov_prediction_confidence,
        markov_prediction_eta_secs: runtime.markov_prediction_eta_secs.max(0.0),
        markov_prewarm_active: runtime.markov_prewarm_active,
        predictive_agent_active: runtime.predictive_agent_active,
        predictive_intervention: format!("{intervention:?}"),
        kpc_available: runtime.kpc_available,
        kpc_memory_bound_score: runtime.kpc_memory_bound_score.clamp(0.0, 1.0),
        amx_available: runtime.amx_available,
        amx_cs_overhead_ns: runtime.amx_cs_overhead_ns,
    }
}

fn context_quality(summary: &TelemetryContextSummary) -> f64 {
    let finite = [
        summary.thrashing_score,
        summary.refault_delta_per_sec,
        summary.swap_delta_bytes_per_sec,
        summary.signal_pressure_smooth,
        summary.signal_pressure_velocity,
        summary.signal_p_oom_30s,
        summary.signal_urgency,
        summary.signal_entropy_anomaly,
        summary.signal_transformer_anomaly,
        summary.kpc_memory_bound_score,
        summary.fluidity_score,
        summary.perceptual_latency_score,
        summary.scheduler_jitter_p95_ms,
        summary.stall_fraction,
        summary.windowserver_cpu_fraction,
        summary.network_retransmits_per_k,
        summary.network_listen_drop_rate,
        summary.pressure_total_boost,
    ]
    .into_iter()
    .filter(|value| value.is_finite())
    .count();
    finite as f64 / 18.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::{CpuStats, MemoryStats, PressureStats, ProcessStats};
    use crate::engine::audit_types::{DecisionReason, PolicyDecisionTrace};
    use crate::engine::decision_ledger::{
        ActuatorDecisionEvent, ActuatorDecisionOutcome, AdviserContribution,
        BinaryPredictionTarget, CandidateAlternative, CycleDecisionEvents, DecisionId,
        DecisionLedger, HierarchyCoordinates, PredictionRecord,
    };
    use crate::engine::lotka_volterra::StabilityRegime;
    use chrono::Utc;

    const LOCAL_ID: InstallationId = InstallationId(0x1020_3040_5060_7080);

    /// Every GPU rejection must land in exactly one bucket, including the ones
    /// that predate the buckets existing.
    ///
    /// `gpu_prediction_rejected_total` is persisted and has counted since the
    /// lane shipped. The three per-reason buckets were added later as
    /// `#[serde(default)]` fields, so on the first restore after that change
    /// they began at zero against an aggregate already in the hundreds of
    /// thousands — production read 261,296 rejected against 16,253 classified.
    /// A reader summing the buckets silently attributes the 245,043-event gap
    /// to whichever bucket they assumed.
    ///
    /// The remainder is therefore published rather than derived by the reader,
    /// and the funnel closes by construction.
    #[test]
    fn gpu_rejection_buckets_and_the_pre_breakdown_remainder_close_the_funnel() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        // A checkpoint written before the breakdown existed: the aggregate
        // carries history the buckets never saw.
        medallion.gpu_prediction_rejected_total = 261_296;
        medallion.gpu_prediction_evicted_total = 0;
        medallion.gpu_prediction_unused_total = 16_049;
        medallion.gpu_prediction_bronze_rejected_total = 204;

        let unclassified = medallion.gpu_prediction_unclassified_rejections();

        assert_eq!(unclassified, 245_043);
        assert_eq!(
            medallion.gpu_prediction_evicted_total
                + medallion.gpu_prediction_unused_total
                + medallion.gpu_prediction_bronze_rejected_total
                + unclassified,
            medallion.gpu_prediction_rejected_total,
            "the four buckets must partition the aggregate exactly"
        );
    }

    /// The restore path clamps each bucket to the aggregate independently, so
    /// a truncated or hostile checkpoint can present buckets that oversum.
    /// That must saturate to zero, never underflow into a vast bogus count.
    #[test]
    fn oversummed_gpu_buckets_saturate_instead_of_wrapping() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.gpu_prediction_rejected_total = 10;
        medallion.gpu_prediction_evicted_total = 8;
        medallion.gpu_prediction_unused_total = 8;
        medallion.gpu_prediction_bronze_rejected_total = 8;

        assert_eq!(medallion.gpu_prediction_unclassified_rejections(), 0);
    }

    fn snapshot() -> SystemSnapshot {
        SystemSnapshot {
            timestamp: Utc::now(),
            cpu: CpuStats {
                global_usage: 20.0,
                core_count: 10,
            },
            memory: MemoryStats {
                total_ram: 16 * 1024 * 1024 * 1024,
                used_ram: 8 * 1024 * 1024 * 1024,
                free_ram: 8 * 1024 * 1024 * 1024,
                total_swap: 4 * 1024 * 1024 * 1024,
                used_swap: 0,
            },
            pressure: PressureStats {
                memory_pressure: 0.2,
                swap_used_bytes: 0,
                swap_total_bytes: 4 * 1024 * 1024 * 1024,
                swap_delta_bytes_per_sec: 0.0,
                thermal_level: "nominal".to_string(),
                compressor_pressure: 0.1,
                thrashing_score: 0.0,
                memory_pressure_raw: 0.2,
                refault_delta_per_sec: 0.0,
            },
            disks: vec![],
            networks: vec![],
            top_processes: vec![],
        }
    }

    fn signal() -> SignalDigest {
        SignalDigest {
            pressure_smooth: 0.2,
            pressure_velocity: 0.0,
            pressure_predicted_5s: 0.2,
            pressure_predicted_30s: 0.2,
            swap_velocity_smooth: 0.0,
            pressure_integral: 0.0,
            regime_shift_up: false,
            regime_shift_down: false,
            cusum_score: 0.0,
            entropy_anomaly: 0.0,
            p_oom_30s: 0.0,
            monopoly_risk: 0.0,
            stability_regime: StabilityRegime::Degenerate,
            mpc_recommendation: 0,
            urgency: 0.0,
            transformer_anomaly: 0.0,
            memory_scan_available: false,
            fluidity_score: 1.0,
            window_op_active: false,
            app_launching: false,
            swap_net_rate_volatility: 0.0,
            lyapunov_exponent: 0.0,
            cumulative_stress: 0.0,
            hw_seasonal_anomaly: 1.0,
        }
    }

    #[test]
    fn coreaudio_source_provenance_enters_actuator_episode_context() {
        let context = TelemetryContextSummary {
            user_audio_active: true,
            coreaudio_direct_probe_available: false,
            coreaudio_session_fallback: true,
            ..TelemetryContextSummary::default()
        };

        let episode = ActuatorEpisodeContext::from_telemetry(&context);

        assert!(episode.user_audio_active);
        assert!(!episode.coreaudio_direct_probe_available);
        assert!(episode.coreaudio_session_fallback);
    }

    fn healthy_runtime() -> RuntimeMetrics {
        RuntimeMetrics {
            collector_pressure_alive: true,
            reactor_health: "healthy".to_string(),
            pressure_dominant_factor: "memory".to_string(),
            ..RuntimeMetrics::default()
        }
    }

    fn m4_capabilities() -> CapabilityReport {
        CapabilityReport {
            can_taskpolicy: true,
            can_sysctl: true,
            can_memorystatus: true,
            can_memory_pressure_send: false,
            can_mdutil: true,
            can_tmutil: true,
            is_root: true,
            p_core_count: Some(4),
            e_core_count: Some(6),
            unavailable: Vec::new(),
            memorystatus_probe: Some("ok".to_string()),
            task_for_pid_probe: Some("ok".to_string()),
        }
    }

    fn trace(action: RootAction, applied: bool) -> PolicyDecisionTrace {
        PolicyDecisionTrace {
            t: Utc::now(),
            cycle: 1,
            intended_action: action,
            decision_reason: DecisionReason::PressureContext,
            applied,
            block_reason: None,
            pressure: 0.2,
            swap_gb: 0.0,
            thrashing: 0.0,
        }
    }

    fn observe(
        medallion: &mut TelemetryMedallion,
        cycle: u64,
        outcomes: &ExecuteOutcomes,
        runtime: &RuntimeMetrics,
    ) -> ContextAdmission {
        let snapshot = snapshot();
        let signal = signal();
        let capabilities = m4_capabilities();
        medallion.observe(TelemetryObservation {
            snapshot: &snapshot,
            hardware: None,
            runtime,
            capabilities: Some(&capabilities),
            signal: &signal,
            workload: "idle",
            cycle,
            outcomes,
            intervention: Intervention::Observe,
            applied_intervention: None,
            purge_recent: false,
            nars_drift_score: 0.0,
            nars_beliefs_total: 1,
            natural_drift: 0.0,
            arousal_level: 0.5,
        })
    }

    #[test]
    fn a_lab_arm_window_is_measured_without_touching_any_learning_aggregate() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let outcomes = ExecuteOutcomes::default();
        let runtime = healthy_runtime();
        // A window can only be opened against an admitted Gold context.
        observe(&mut medallion, 1, &outcomes, &runtime);

        let before = medallion.metrics();
        let before_resolved = medallion.actuator_resolved_total;
        let before_issued = medallion.actuator_issued_total;
        let before_rejected = medallion.actuator_rejected_total;
        let before_recent = medallion.recent_evidence.len();
        let before_family = medallion
            .family_stats
            .get(&ActuatorFamily::InteractionQos)
            .cloned()
            .unwrap_or_default();

        assert!(medallion.open_lab_utility_window(991, ActuatorFamily::InteractionQos, 3, 1));
        assert_eq!(medallion.lab_windows_open(), 1);

        // Before the horizon there is nothing to hand back.
        observe(&mut medallion, 2, &outcomes, &runtime);
        assert!(medallion.drain_lab_utility().is_empty());

        observe(&mut medallion, 4, &outcomes, &runtime);
        let samples = medallion.drain_lab_utility();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].decision_id, 991);
        assert_eq!(samples[0].resolved_cycle, 4);
        assert!(medallion.drain_lab_utility().is_empty(), "drains once");
        assert_eq!(medallion.lab_windows_open(), 0);

        // Nothing about the learning pipeline moved because of the arm.
        assert_eq!(medallion.actuator_issued_total, before_issued);
        assert_eq!(medallion.actuator_resolved_total, before_resolved);
        assert_eq!(medallion.actuator_rejected_total, before_rejected);
        assert_eq!(medallion.recent_evidence.len(), before_recent);
        assert_eq!(
            medallion
                .family_stats
                .get(&ActuatorFamily::InteractionQos)
                .cloned()
                .unwrap_or_default(),
            before_family
        );
        let after = medallion.metrics();
        assert_eq!(after.actuator_gold_total, before.actuator_gold_total);
        assert_eq!(after.actuator_silver_total, before.actuator_silver_total);
        assert!(medallion.drain_new_gold_evidence().is_empty());
        assert_eq!(medallion.authoritative_gold_decision_count(), 0);
    }

    #[test]
    fn lab_windows_are_bounded_deduplicated_and_catalogue_only() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        observe(
            &mut medallion,
            1,
            &ExecuteOutcomes::default(),
            &healthy_runtime(),
        );

        // Uncatalogued family, zero identity and zero horizon are refused.
        assert!(!medallion.open_lab_utility_window(1, ActuatorFamily::Boost, 3, 1));
        assert!(!medallion.open_lab_utility_window(0, ActuatorFamily::InteractionQos, 3, 1));
        assert!(!medallion.open_lab_utility_window(2, ActuatorFamily::InteractionQos, 0, 1));

        assert!(medallion.open_lab_utility_window(3, ActuatorFamily::MarkovPrewarm, 5, 1));
        assert!(
            !medallion.open_lab_utility_window(3, ActuatorFamily::MarkovPrewarm, 5, 1),
            "one window per decision identity"
        );

        for id in 100..(100 + MAX_LAB_WINDOWS as u64 + 8) {
            medallion.open_lab_utility_window(id, ActuatorFamily::InteractionQos, 5, 1);
        }
        assert!(medallion.lab_windows_open() <= MAX_LAB_WINDOWS);
    }

    #[test]
    fn a_lab_window_past_its_grace_deadline_expires_without_a_sample() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let outcomes = ExecuteOutcomes::default();
        let runtime = healthy_runtime();
        observe(&mut medallion, 1, &outcomes, &runtime);
        assert!(medallion.open_lab_utility_window(77, ActuatorFamily::InteractionQos, 3, 1));

        let past_deadline = 1 + 3 + LAB_WINDOW_GRACE_CYCLES + 1;
        observe(&mut medallion, past_deadline, &outcomes, &runtime);
        assert!(medallion.drain_lab_utility().is_empty());
        assert_eq!(medallion.lab_windows_open(), 0);
        assert_eq!(medallion.lab_windows_expired_total(), 1);
    }

    #[test]
    fn a_lab_window_without_an_admitted_context_is_refused() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        assert!(
            !medallion.open_lab_utility_window(5, ActuatorFamily::InteractionQos, 3, 1),
            "no admitted telemetry context yet"
        );
    }

    fn local_episode(
        action_key: &str,
        target: &str,
        cycle: u64,
        outcome: ActuatorDecisionOutcome,
    ) -> crate::engine::decision_ledger::ResolvedDecisionEpisode {
        let mut events = CycleDecisionEvents::default();
        events.push(ActuatorDecisionEvent::local(
            action_key,
            target,
            cycle,
            outcome,
            "test-actuator",
            "focused medallion handoff test",
        ));
        DecisionLedger::new()
            .ingest_cycle_events(&mut events)
            .pop()
            .expect("resolved ledger episode")
    }

    fn rich_restore_snapshot() -> TelemetryMedallionPersisted {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = healthy_runtime();
        observe(&mut medallion, 1, &ExecuteOutcomes::default(), &runtime);
        let event = ActuatorDecisionEvent::local(
            "predictive_threshold:tighten",
            "predictive_threshold:tighten",
            1,
            ActuatorDecisionOutcome::Applied,
            "predictive-agent",
            "Task 4 restore review fixture",
        )
        .with_prediction(PredictionRecord {
            source: "world-model".to_string(),
            expected_utility: 0.03,
            uncertainty: 0.25,
            horizon_cycles: 10,
            positive_probability: Some(0.6),
            binary_target: Some(BinaryPredictionTarget::Effective),
        });
        let mut events = CycleDecisionEvents::default();
        events.push(event);
        let episodes = DecisionLedger::new().ingest_cycle_events(&mut events);
        medallion.stage_decision_episodes(&episodes);
        for cycle in 2..=11 {
            observe(&mut medallion, cycle, &ExecuteOutcomes::default(), &runtime);
        }
        let snapshot = medallion.snapshot();
        assert!(snapshot.episodic_evidence.iter().any(|evidence| {
            evidence
                .learning_details
                .as_ref()
                .is_some_and(ResolvedLearningDetails::is_authoritative)
        }));
        snapshot
    }

    #[test]
    fn task4_review_forged_calibration_delta_is_stripped_on_restore() {
        let mut persisted = rich_restore_snapshot();
        let evidence = persisted
            .episodic_evidence
            .iter_mut()
            .find(|evidence| evidence.learning_details.is_some())
            .expect("rich evidence");
        let delta = evidence
            .learning_details
            .as_mut()
            .and_then(|details| details.calibration_deltas.first_mut())
            .expect("calibration delta");
        delta.uncertainty_covered = !delta.uncertainty_covered;

        let mut restored = TelemetryMedallion::new(LOCAL_ID);
        restored.latest = persisted.latest.clone();
        restored.restore(persisted);

        assert!(restored
            .episodic_evidence
            .iter()
            .all(|evidence| evidence.learning_details.is_none()));
    }

    #[test]
    fn task4_review_live_hardware_mismatch_strips_restored_rich_details() {
        let persisted = rich_restore_snapshot();
        let mut live_context = persisted.latest.clone().expect("persisted context");
        live_context.p_core_count = live_context.p_core_count.saturating_add(2);

        let mut restored = TelemetryMedallion::new(LOCAL_ID);
        restored.latest = Some(live_context);
        restored.restore(persisted);

        assert!(restored
            .episodic_evidence
            .iter()
            .all(|evidence| evidence.learning_details.is_none()));
    }

    #[test]
    fn authoritative_gold_ids_come_from_valid_measured_learning_details() {
        let persisted = rich_restore_snapshot();
        let mut restored = TelemetryMedallion::new(LOCAL_ID);
        restored.latest = persisted.latest.clone();
        restored.restore(persisted);

        let ids = restored.authoritative_gold_decision_ids();

        assert_eq!(ids.len(), 1);
        assert!(ids[0] > 0);
    }

    #[test]
    fn current_machine_recent_evidence_excludes_imported_installations() {
        let hardware = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let mut evidence = gold_evidence("boost:Editor", 0.1, 20_000_000, hardware);
        evidence.decision_id = Some(DecisionId(7));
        evidence.installation_id = InstallationId(99);
        let persisted = TelemetryMedallionPersisted {
            actuator_evidence_schema_version: ACTUATOR_EVIDENCE_SCHEMA_VERSION,
            context_schema_version: TELEMETRY_CONTEXT_SCHEMA_VERSION,
            installation_id: InstallationId(99),
            latest: Some(TelemetryContextSummary {
                p_core_count: 4,
                e_core_count: 6,
                total_ram_bytes: 16 * 1024 * 1024 * 1024,
                ..TelemetryContextSummary::default()
            }),
            recent_evidence: vec![evidence],
            ..TelemetryMedallionPersisted::default()
        };
        let mut restored = TelemetryMedallion::new(LOCAL_ID);
        restored.restore(persisted);

        assert_eq!(restored.current_machine_recent_evidence().count(), 0);
    }

    #[test]
    fn local_root_episode_attaches_decision_id_to_existing_medallion_episode() {
        let action = RootAction::BoostProcess {
            pid: 300,
            name: "Editor".to_string(),
            reason: "fixture".to_string(),
            decision_reason: DecisionReason::InteractiveFocus,
            start_sec: 12_345,
            start_usec: 678,
        };
        let action_key = actuator_action_key(&action).expect("root action key");
        let outcomes = ExecuteOutcomes {
            audit_traces: vec![trace(action, true)],
            ..ExecuteOutcomes::default()
        };
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        observe(&mut medallion, 1, &outcomes, &healthy_runtime());

        medallion.stage_decision_episodes(&[local_episode(
            &action_key,
            "Editor:pid:300",
            1,
            ActuatorDecisionOutcome::Applied,
        )]);

        assert_eq!(medallion.pending_actions.len(), 1);
        assert_eq!(
            medallion.pending_actions[0].decision_id,
            Some(DecisionId(1))
        );
        for cycle in 2..=4 {
            observe(
                &mut medallion,
                cycle,
                &ExecuteOutcomes::default(),
                &healthy_runtime(),
            );
        }
        assert_eq!(
            medallion
                .recent_actuator_evidence()
                .back()
                .and_then(|evidence| evidence.decision_id),
            Some(DecisionId(1))
        );
    }

    #[test]
    fn all_predictions_survive_delayed_resolution_and_calibrate_once_at_gold_admission() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = healthy_runtime();
        observe(&mut medallion, 1, &ExecuteOutcomes::default(), &runtime);
        let sources = [
            "world-model",
            "gpu-model",
            "markov",
            "causal-graph",
            "mpc",
            "nars",
            "policy-scorer",
            "predictive-agent",
        ];
        let mut event = ActuatorDecisionEvent::local(
            "predictive_threshold:tighten",
            "predictive_threshold:tighten",
            1,
            ActuatorDecisionOutcome::Applied,
            "predictive-agent",
            "delayed calibration fixture",
        )
        .with_hierarchy(HierarchyCoordinates {
            level: 2,
            parent: Some(DecisionId(77)),
            cohort: 0,
        });
        event.proposal.alternatives = vec![
            CandidateAlternative {
                action_key: "predictive_threshold:hold".to_string(),
                target: "predictive_threshold:hold".to_string(),
                expected_utility: 0.02,
                uncertainty: 0.20,
            },
            CandidateAlternative {
                action_key: "predictive_threshold:relax".to_string(),
                target: "predictive_threshold:relax".to_string(),
                expected_utility: -0.01,
                uncertainty: 0.30,
            },
        ];
        let expected_alternatives = event.proposal.alternatives.clone();
        event.proposal.adviser_contributions = vec![
            AdviserContribution {
                adviser: "gpu-model".to_string(),
                support: 0.7,
                uncertainty: 0.2,
            },
            AdviserContribution {
                adviser: "causal-graph".to_string(),
                support: -0.4,
                uncertainty: 0.3,
            },
        ];
        for (index, source) in sources.into_iter().enumerate() {
            event = event.with_prediction(PredictionRecord {
                source: source.to_string(),
                expected_utility: index as f64 / 100.0,
                uncertainty: 0.25,
                horizon_cycles: 10,
                positive_probability: Some(0.6),
                binary_target: Some(BinaryPredictionTarget::Effective),
            });
        }
        let mut events = CycleDecisionEvents::default();
        events.push(event);
        let episodes = DecisionLedger::new().ingest_cycle_events(&mut events);
        medallion.stage_decision_episodes(&episodes);

        for cycle in 2..=11 {
            observe(&mut medallion, cycle, &ExecuteOutcomes::default(), &runtime);
        }

        let evidence = medallion
            .recent_actuator_evidence()
            .back()
            .expect("resolved evidence");
        assert_eq!(evidence.calibration_provenance.predictions.len(), 8);
        assert_eq!(evidence.calibration_provenance.alternatives.len(), 2);
        assert!(evidence.calibration_provenance.local_authority_eligible);
        assert_eq!(evidence.calibration_provenance.proposer, "predictive-agent");
        assert_eq!(
            evidence.calibration_provenance.adviser_contributions.len(),
            2
        );
        assert_eq!(evidence.calibration_provenance.hierarchy.level, 2);
        assert_eq!(
            evidence.calibration_provenance.hierarchy.parent,
            Some(DecisionId(77))
        );
        assert_eq!(evidence.calibration_provenance.cohort_size, 1);
        assert_eq!(
            evidence.calibration_provenance.separability,
            SeparabilityState::Individual
        );
        assert_eq!(
            medallion
                .model_calibration_metrics()
                .accepted_forecasts_total,
            8
        );
        assert_eq!(medallion.model_calibration_metrics().record_count, 8);
        assert!(medallion.gpu_calibration_models.is_empty());
        assert!(evidence.learning_details.is_none());

        let gold = medallion.drain_new_gold_evidence();
        assert_eq!(gold.len(), 1);
        let rich = gold[0]
            .learning_details
            .as_ref()
            .expect("Gold must be enriched before entering the drain");
        assert_eq!(rich.decision_id, gold[0].decision_id.unwrap());
        assert_eq!(rich.alternatives, expected_alternatives);
        assert_eq!(rich.predictions.len(), 8);
        assert_eq!(rich.adviser_contributions.len(), 2);
        assert_eq!(rich.calibration_deltas.len(), 8);
        assert!(rich.calibration_deltas.iter().all(|delta| {
            (delta.actual_utility - gold[0].utility.apollo_utility).abs() < f64::EPSILON
        }));
        assert_eq!(
            medallion
                .episodic_evidence
                .back()
                .and_then(|episode| episode.learning_details.as_ref())
                .map(|details| details.decision_id),
            Some(rich.decision_id)
        );

        let snapshot = medallion.snapshot();
        let calibration_bytes = serde_json::to_vec(
            snapshot
                .model_calibration
                .as_ref()
                .expect("nested calibration snapshot"),
        )
        .unwrap()
        .len();
        let mut without_calibration = snapshot.clone();
        without_calibration.model_calibration = None;
        let full_growth = serde_json::to_vec(&snapshot).unwrap().len()
            - serde_json::to_vec(&without_calibration).unwrap().len();
        assert!(calibration_bytes <= crate::engine::model_calibration::MAX_CALIBRATION_STATE_BYTES);
        assert!(full_growth <= 2 * 1024 * 1024);

        let mut forged_snapshot = snapshot.clone();
        forged_snapshot.episodic_evidence[0]
            .learning_details
            .as_mut()
            .expect("rich episode")
            .raw_utility_delta += 0.25;
        let mut forged_restore = TelemetryMedallion::new(LOCAL_ID);
        forged_restore.restore(forged_snapshot);
        assert!(forged_restore
            .episodic_evidence
            .back()
            .and_then(|episode| episode.learning_details.as_ref())
            .is_none());

        let live_context = snapshot.latest.clone();
        let mut restored = TelemetryMedallion::new(LOCAL_ID);
        restored.latest = live_context;
        restored.restore(snapshot);
        assert_eq!(restored.model_calibration_metrics().record_count, 8);
        assert!(restored.drain_new_gold_evidence().is_empty());
        assert!(restored
            .episodic_evidence
            .back()
            .and_then(|episode| episode.learning_details.as_ref())
            .is_some());
    }

    #[test]
    fn decision_source_credit_tracks_direction_not_duplicated_treatment_utility() {
        let hardware = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let mut evidence = gold_evidence("boost:Editor", 0.4, 20_000_000, hardware);
        evidence.decision_id = Some(DecisionId(1));
        evidence.utility.apollo_utility = 0.4;
        evidence.context_before = ActuatorEpisodeContext {
            valid: true,
            memory_pressure: 0.4,
            thermal_score: 0.2,
            foreground_app_hash: 1,
            ..ActuatorEpisodeContext::default()
        };
        evidence.attribution = DecisionAttribution {
            action_key: "boost:Editor".to_string(),
            proposer: "world-model".to_string(),
            supporters: vec!["gpu-model".to_string(), "markov".to_string()],
            vetoes: vec!["causal-graph".to_string()],
            predicted_gain: 0.2,
            uncertainty: 0.3,
        };
        evidence.calibration_provenance = CalibrationProvenance {
            local_authority_eligible: true,
            proposer: "world-model".to_string(),
            predictions: vec![
                PredictionRecord {
                    source: "gpu-model".to_string(),
                    expected_utility: -0.4,
                    uncertainty: 0.2,
                    horizon_cycles: 10,
                    positive_probability: None,
                    binary_target: None,
                },
                PredictionRecord {
                    source: "causal-graph".to_string(),
                    expected_utility: -0.4,
                    uncertainty: 0.2,
                    horizon_cycles: 10,
                    positive_probability: None,
                    binary_target: None,
                },
            ],
            adviser_contributions: vec![
                AdviserContribution {
                    adviser: "gpu-model".to_string(),
                    support: 0.7,
                    uncertainty: 0.2,
                },
                AdviserContribution {
                    adviser: "causal-graph".to_string(),
                    support: -0.4,
                    uncertainty: 0.3,
                },
            ],
            cohort_size: 1,
            ..CalibrationProvenance::default()
        };
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.admit_resolved(evidence);

        assert!((medallion.actuator_utility_sum - 0.4).abs() < 1e-12);
        assert_eq!(medallion.action_models["boost:Editor"].observations, 1);
        assert_eq!(
            medallion.decision_source_stats["world-model"].credit_ema,
            1.0
        );
        assert_eq!(medallion.decision_source_stats["gpu-model"].credit_ema, 1.0);
        assert_eq!(medallion.decision_source_stats["markov"].credit_ema, 1.0);
        assert_eq!(
            medallion.decision_source_stats["causal-graph"].credit_ema,
            -1.0
        );
        assert_eq!(
            medallion
                .model_calibration_metrics()
                .accepted_forecasts_total,
            2
        );
        assert!(medallion.model_calibration().records().all(|record| {
            (record.signed_error_ema - 0.8).abs() < 1e-12
                && (record.normalized_mae_ema - 0.4).abs() < 1e-12
        }));
    }

    #[test]
    fn actuator_schema_v2_keeps_specialists_but_starts_calibration_cold() {
        let now_unix = 20_000_000;
        let hardware = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let mut persisted = TelemetryMedallionPersisted {
            actuator_evidence_schema_version: 2,
            context_schema_version: TELEMETRY_CONTEXT_SCHEMA_VERSION,
            installation_id: LOCAL_ID,
            latest: Some(TelemetryContextSummary {
                timestamp_unix: now_unix,
                p_core_count: 4,
                e_core_count: 6,
                total_ram_bytes: 16 * 1024 * 1024 * 1024,
                ..TelemetryContextSummary::default()
            }),
            model_calibration: Some(
                crate::engine::model_calibration::ModelCalibrationPersisted {
                    installation_id: LOCAL_ID,
                    hardware_regime: hardware,
                    records: vec![crate::engine::model_calibration::CalibrationRecord {
                        authority_gold_count: 500,
                        lifetime_forecast_count: 500,
                        trust: crate::engine::model_calibration::TrustState::Trusted,
                        ..crate::engine::model_calibration::CalibrationRecord::default()
                    }],
                    ..crate::engine::model_calibration::ModelCalibrationPersisted::default()
                },
            ),
            ..TelemetryMedallionPersisted::default()
        };
        persisted.action_models.insert(
            "boost:Editor".to_string(),
            ActionModelStats {
                observations: 20,
                utility_ema: 0.1,
                evidence_mass: 20.0,
                quality_ema: 0.95,
                last_observed_unix: now_unix,
                hardware_regime: hardware,
                installation_id: LOCAL_ID,
                ..ActionModelStats::default()
            },
        );

        let mut restored = TelemetryMedallion::new(LOCAL_ID);
        restored.restore(persisted);

        assert!(restored.action_models().contains_key("boost:Editor"));
        assert_eq!(restored.model_calibration_metrics().record_count, 0);
        assert_eq!(
            restored.snapshot().actuator_evidence_schema_version,
            ACTUATOR_EVIDENCE_SCHEMA_VERSION
        );
    }

    #[test]
    fn future_actuator_schema_is_quarantined_without_interpretation_or_downgrade() {
        let hardware = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let mut evidence = gold_evidence("boost:Editor", 0.2, 20_000_000, hardware);
        evidence.decision_id = Some(DecisionId(9));
        evidence.context_before = ActuatorEpisodeContext {
            valid: true,
            memory_pressure: 0.3,
            thermal_score: 0.2,
            foreground_app_hash: 1,
            ..ActuatorEpisodeContext::default()
        };
        evidence.calibration_provenance = CalibrationProvenance {
            local_authority_eligible: true,
            proposer: "actuation-broker".to_string(),
            predictions: vec![PredictionRecord {
                source: "world-model".to_string(),
                expected_utility: 0.2,
                uncertainty: 0.1,
                horizon_cycles: 3,
                positive_probability: None,
                binary_target: None,
            }],
            cohort_size: 1,
            ..CalibrationProvenance::default()
        };
        let mut source = TelemetryMedallion::new(LOCAL_ID);
        source.admit_resolved(evidence);
        let mut persisted = source.snapshot();
        persisted.actuator_evidence_schema_version = ACTUATOR_EVIDENCE_SCHEMA_VERSION + 1;

        let mut restored = TelemetryMedallion::new(LOCAL_ID);
        restored.restore(persisted);

        assert_eq!(restored.model_calibration_metrics().record_count, 0);
        assert!(restored.recent_actuator_evidence().is_empty());
        assert_eq!(
            restored.snapshot().actuator_evidence_schema_version,
            ACTUATOR_EVIDENCE_SCHEMA_VERSION + 1
        );
    }

    #[test]
    fn persisted_evidence_deserialization_bounds_nested_calibration_provenance() {
        let hardware = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let oversized_source = "s".repeat(2_048);
        let oversized_adviser = "a".repeat(2_048);
        let mut evidence = gold_evidence("boost:Editor", 0.2, 20_000_000, hardware);
        evidence.calibration_provenance = CalibrationProvenance {
            local_authority_eligible: true,
            proposer: "p".repeat(2_048),
            predictions: (0..64)
                .map(|_| PredictionRecord {
                    source: oversized_source.clone(),
                    expected_utility: 0.2,
                    uncertainty: 0.1,
                    horizon_cycles: 3,
                    positive_probability: None,
                    binary_target: None,
                })
                .collect(),
            adviser_contributions: (0..64)
                .map(|_| AdviserContribution {
                    adviser: oversized_adviser.clone(),
                    support: 0.1,
                    uncertainty: 0.2,
                })
                .collect(),
            ..CalibrationProvenance::default()
        };
        let mut state = TelemetryMedallionPersisted {
            actuator_evidence_schema_version: ACTUATOR_EVIDENCE_SCHEMA_VERSION,
            context_schema_version: TELEMETRY_CONTEXT_SCHEMA_VERSION,
            installation_id: LOCAL_ID,
            recent_evidence: vec![evidence.clone(); MAX_RECENT_EVIDENCE + 32],
            episodic_evidence: vec![evidence; MAX_EPISODIC_EVIDENCE + 32],
            ..TelemetryMedallionPersisted::default()
        };
        state.decision_id_high_water = 1;

        let encoded = serde_json::to_vec(&state).unwrap();
        let bounded: TelemetryMedallionPersisted = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(bounded.recent_evidence.len(), MAX_RECENT_EVIDENCE);
        assert_eq!(bounded.episodic_evidence.len(), MAX_EPISODIC_EVIDENCE);
        for evidence in bounded
            .recent_evidence
            .iter()
            .chain(bounded.episodic_evidence.iter())
        {
            let provenance = &evidence.calibration_provenance;
            assert!(provenance.proposer.chars().count() <= 48);
            assert!(provenance.predictions.len() <= 8);
            assert!(provenance.adviser_contributions.len() <= 8);
            assert!(provenance.predictions.iter().all(|prediction| prediction
                .source
                .chars()
                .count()
                <= 48));
            assert!(provenance
                .adviser_contributions
                .iter()
                .all(|adviser| adviser.adviser.chars().count() <= 48));
        }
    }

    #[test]
    fn exact_side_episode_suppresses_duplicate_counter_fallback() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        observe(
            &mut medallion,
            1,
            &ExecuteOutcomes::default(),
            &healthy_runtime(),
        );
        medallion.stage_decision_episodes(&[local_episode(
            "interaction_qos:foreground",
            "pid:300",
            1,
            ActuatorDecisionOutcome::Applied,
        )]);
        let runtime = RuntimeMetrics {
            interaction_qos_activations: 1,
            interaction_qos_reason: "input".to_string(),
            ..healthy_runtime()
        };

        observe(&mut medallion, 2, &ExecuteOutcomes::default(), &runtime);

        assert_eq!(medallion.pending_actions.len(), 1);
        assert_eq!(medallion.metrics().actuator_issued_total, 1);
        assert_eq!(
            medallion.pending_actions[0].decision_id,
            Some(DecisionId(1))
        );
        assert_eq!(medallion.pending_actions[0].target, "pid:300");
    }

    #[test]
    fn exact_concurrent_side_episodes_keep_members_confounded_and_cohort_separate() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        observe(
            &mut medallion,
            1,
            &ExecuteOutcomes::default(),
            &healthy_runtime(),
        );
        let mut events = CycleDecisionEvents::default();
        events.push(ActuatorDecisionEvent::local(
            "markov_prewarm:predicted_app",
            "Editor",
            1,
            ActuatorDecisionOutcome::Applied,
            "markov",
            "applied",
        ));
        events.push(ActuatorDecisionEvent::local(
            "predictive_purge:maintenance",
            "host",
            1,
            ActuatorDecisionOutcome::Applied,
            "maintenance",
            "applied",
        ));
        let coordinated = crate::engine::decision_ledger::coordinated_action_event(&events, 1)
            .expect("coordinated event");
        events.push(coordinated);
        let episodes = DecisionLedger::new().ingest_cycle_events(&mut events);
        medallion.stage_decision_episodes(&episodes);

        observe(
            &mut medallion,
            2,
            &ExecuteOutcomes::default(),
            &healthy_runtime(),
        );

        assert_eq!(medallion.pending_actions.len(), 3);
        assert!(medallion.pending_actions.iter().all(|pending| {
            if pending.family == ActuatorFamily::Coordinated {
                pending.cohort_size == 1
            } else {
                pending.cohort_size == 2
            }
        }));
    }

    #[test]
    fn decision_id_high_water_round_trips_even_for_non_authoritative_receipts() {
        let mut events = CycleDecisionEvents::default();
        events.push(ActuatorDecisionEvent::local(
            "sysctl:shutdown-rollback",
            "kern.test=0",
            31,
            ActuatorDecisionOutcome::Failed,
            "shutdown",
            "write failed",
        ));
        let episodes = DecisionLedger::new().ingest_cycle_events(&mut events);
        let prior_id = episodes[0].id.0;
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.stage_decision_episodes(&episodes);

        let encoded = serde_json::to_string(&medallion.snapshot()).expect("persist medallion");
        let persisted: TelemetryMedallionPersisted =
            serde_json::from_str(&encoded).expect("restore persisted medallion");
        let mut restored = TelemetryMedallion::new(LOCAL_ID);
        restored.restore(persisted);
        let mut restarted_ledger = DecisionLedger::new();
        restarted_ledger.seed_high_water(restored.decision_id_high_water());

        assert_eq!(restored.decision_id_high_water(), prior_id);
        assert!(
            restarted_ledger
                .propose(crate::engine::decision_ledger::DecisionProposal::default())
                .0
                > prior_id
        );
    }

    #[test]
    fn imported_and_non_applied_ledger_episodes_cannot_open_medallion_evidence() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        observe(
            &mut medallion,
            1,
            &ExecuteOutcomes::default(),
            &healthy_runtime(),
        );
        let mut imported_events = CycleDecisionEvents::default();
        imported_events.push(ActuatorDecisionEvent::imported(
            "markov_prewarm:predicted_app",
            "Editor",
            1,
            ActuatorDecisionOutcome::Applied,
            "foreign evidence",
        ));
        let mut ledger = DecisionLedger::new();
        let mut episodes = ledger.ingest_cycle_events(&mut imported_events);
        episodes.push(local_episode(
            "freeze:background",
            "pid:44",
            1,
            ActuatorDecisionOutcome::Blocked,
        ));

        medallion.stage_decision_episodes(&episodes);
        observe(
            &mut medallion,
            2,
            &ExecuteOutcomes::default(),
            &healthy_runtime(),
        );

        assert!(medallion.pending_actions.is_empty());
        assert_eq!(medallion.metrics().actuator_issued_total, 0);
        assert_eq!(medallion.model_calibration_metrics().record_count, 0);
    }

    fn gold_evidence(
        action_key: &str,
        utility: f64,
        timestamp_unix: i64,
        hardware_regime: HardwareRegime,
    ) -> ResolvedActuatorEvidence {
        ResolvedActuatorEvidence {
            id: 1,
            decision_id: None,
            family: ActuatorFamily::Boost,
            objective: ActuatorObjective::Responsiveness,
            action_key: action_key.to_string(),
            target: "Editor".to_string(),
            workload: "build".to_string(),
            issued_cycle: 97,
            resolved_cycle: 100,
            resolved_timestamp_unix: timestamp_unix,
            hardware_regime,
            installation_id: LOCAL_ID,
            horizon_cycles: 3,
            tier: EvidenceTier::Gold,
            provenance: EvidenceProvenance::ObservedLocal,
            quality: 0.95,
            raw_utility_delta: utility,
            counterfactual_delta: 0.0,
            net_utility_delta: utility,
            attribution: DecisionAttribution::default(),
            calibration_provenance: CalibrationProvenance::default(),
            learning_details: None,
            utility: UtilityDecomposition::default(),
            perceptual_latency_improvement: 0.0,
            net_state_delta: WorldStateDelta::default(),
            context_before: ActuatorEpisodeContext::default(),
            effective: utility > 0.0,
            confounder_count: 0,
            target_present_after: None,
        }
    }

    /// Fills `action_models` to the hard ceiling with recently useful models so
    /// the next unseen key is forced to evict something.
    fn saturate_action_models(
        medallion: &mut TelemetryMedallion,
        now_unix: i64,
        hardware: HardwareRegime,
    ) {
        while medallion.action_models.len() < MAX_ACTION_MODELS {
            let key = format!("filler:{}", medallion.action_models.len());
            medallion.action_models.insert(
                key,
                ActionModelStats {
                    observations: 40,
                    evidence_mass: 40.0,
                    quality_ema: 0.95,
                    last_cycle: 500,
                    last_observed_unix: now_unix,
                    hardware_regime: hardware,
                    installation_id: LOCAL_ID,
                    ..ActionModelStats::default()
                },
            );
        }
    }

    #[test]
    fn a_saturated_map_evicts_the_decayed_model_not_the_young_active_one() {
        let now_unix = 1_900_000_000;
        let hardware = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        // A model that was busy long ago and has since decayed to nothing. Its
        // raw `observations` stay high forever because they never decay.
        medallion.action_models.insert(
            "zombie:Ancient".to_string(),
            ActionModelStats {
                observations: 5_000,
                evidence_mass: ACTION_MODEL_EVIDENCE_CAP,
                quality_ema: 0.95,
                last_cycle: 1,
                last_observed_unix: now_unix - 60 * 24 * 60 * 60,
                hardware_regime: hardware,
                installation_id: LOCAL_ID,
                ..ActionModelStats::default()
            },
        );
        // A model observed moments ago that is genuinely accumulating evidence.
        medallion.action_models.insert(
            "young:Active".to_string(),
            ActionModelStats {
                observations: 3,
                evidence_mass: 3.0,
                quality_ema: 0.95,
                last_cycle: 999,
                last_observed_unix: now_unix,
                hardware_regime: hardware,
                installation_id: LOCAL_ID,
                ..ActionModelStats::default()
            },
        );
        saturate_action_models(&mut medallion, now_unix, hardware);
        assert_eq!(medallion.action_models.len(), MAX_ACTION_MODELS);

        let zombie_evidence = medallion
            .action_models
            .get("zombie:Ancient")
            .unwrap()
            .effective_evidence_at(now_unix);
        let young_evidence = medallion
            .action_models
            .get("young:Active")
            .unwrap()
            .effective_evidence_at(now_unix);
        assert!(
            zombie_evidence < young_evidence,
            "the decayed model must carry less usable evidence: {zombie_evidence} vs {young_evidence}"
        );

        let mut evidence = gold_evidence("boost:Fresh", 0.1, now_unix, hardware);
        evidence.workload = "general".to_string();
        medallion.update_action_model(&evidence);

        assert!(
            medallion.action_models.contains_key("young:Active"),
            "a young model that is actively accumulating evidence must not be \
             evicted in favour of one whose evidence has fully decayed"
        );
        assert!(
            !medallion.action_models.contains_key("zombie:Ancient"),
            "the decayed model is the correct eviction victim"
        );
    }

    #[test]
    fn evicting_a_learned_model_is_counted_rather_than_silent() {
        let now_unix = 1_900_000_000;
        let hardware = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        saturate_action_models(&mut medallion, now_unix, hardware);
        assert_eq!(medallion.metrics().action_model_evictions_total, 0);

        let mut evidence = gold_evidence("boost:Fresh", 0.1, now_unix, hardware);
        evidence.workload = "general".to_string();
        medallion.update_action_model(&evidence);

        let metrics = medallion.metrics();
        assert_eq!(
            metrics.action_model_len, metrics.action_model_capacity,
            "the map stays pinned at its ceiling"
        );
        assert!(
            metrics.action_model_evictions_total > 0,
            "destroying learned evidence must be observable, not silent"
        );
        assert!(
            metrics.action_model_births_total > 0,
            "the replacement key must be counted as a birth"
        );
    }

    #[test]
    fn evidence_updates_record_when_the_world_model_last_made_progress() {
        let now_unix = 1_900_000_000;
        let hardware = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        assert_eq!(medallion.metrics().action_model_evidence_updates_total, 0);
        assert_eq!(medallion.metrics().action_model_last_evidence_cycle, 0);

        let mut evidence = gold_evidence("boost:Editor", 0.1, now_unix, hardware);
        evidence.resolved_cycle = 4_242;
        medallion.update_action_model(&evidence);

        let metrics = medallion.metrics();
        assert_eq!(metrics.action_model_evidence_updates_total, 1);
        assert_eq!(metrics.action_model_last_evidence_cycle, 4_242);
    }

    #[test]
    fn capacity_accounting_survives_the_persistence_round_trip() {
        let now_unix = 1_900_000_000;
        let hardware = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        saturate_action_models(&mut medallion, now_unix, hardware);
        let mut evidence = gold_evidence("boost:Fresh", 0.1, now_unix, hardware);
        evidence.workload = "general".to_string();
        medallion.update_action_model(&evidence);
        let before = medallion.metrics();
        assert!(before.action_model_evictions_total > 0);

        let mut restored = TelemetryMedallion::new(LOCAL_ID);
        restored.restore(medallion.snapshot());
        let after = restored.metrics();

        assert_eq!(
            after.action_model_evictions_total, before.action_model_evictions_total,
            "capacity pressure history must survive restart"
        );
        assert_eq!(
            after.action_model_births_total,
            before.action_model_births_total
        );
        assert_eq!(
            after.action_model_evidence_updates_total,
            before.action_model_evidence_updates_total
        );
        assert_eq!(
            after.action_model_last_evidence_cycle,
            before.action_model_last_evidence_cycle
        );
    }

    #[test]
    fn gold_state_transitions_survive_only_on_the_local_installation() {
        let now_unix = Utc::now().timestamp();
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        for sample in 0..4 {
            let mut evidence = gold_evidence(
                "boost:Editor",
                0.08,
                now_unix + sample,
                HardwareRegime {
                    p_core_count: 4,
                    e_core_count: 6,
                    ram_gib: 16,
                },
            );
            evidence.resolved_cycle += sample as u64;
            evidence.net_state_delta = WorldStateDelta {
                pressure: -0.02,
                fluidity: 0.04,
                latency: -0.03,
                energy: 0.01,
                cpu: 0.02,
                thermal: 0.0,
                thrashing: -0.01,
                stall: -0.03,
            };
            medallion.update_action_model(&evidence);
        }

        let local_model = medallion.action_models().get("boost:Editor").unwrap();
        assert!(local_model.state_evidence_mass > 3.99);
        assert!((local_model.state_delta_ema.fluidity - 0.04).abs() < 1e-9);
        assert!((local_model.state_delta_ema.pressure + 0.02).abs() < 1e-9);

        let persisted = medallion.snapshot();
        let mut local_restore = TelemetryMedallion::new(LOCAL_ID);
        local_restore.restore(persisted.clone());
        assert!(
            local_restore
                .action_models()
                .get("boost:Editor")
                .unwrap()
                .effective_state_evidence_at(now_unix + 4)
                > 3.9
        );

        let mut foreign_restore = TelemetryMedallion::new(InstallationId(99));
        foreign_restore.restore(persisted);
        let foreign_model = foreign_restore.action_models().get("boost:Editor").unwrap();
        assert_eq!(foreign_model.state_evidence_mass, 0.0);
        assert_eq!(foreign_model.effective_state_evidence_at(now_unix + 4), 0.0);
    }

    #[test]
    fn parameterized_interaction_evidence_updates_arm_and_parent_models() {
        let now_unix = Utc::now().timestamp();
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let mut evidence = gold_evidence(
            "interaction_qos:foreground@long",
            0.04,
            now_unix,
            HardwareRegime {
                p_core_count: 4,
                e_core_count: 6,
                ram_gib: 16,
            },
        );
        evidence.family = ActuatorFamily::InteractionQos;
        medallion.update_action_model(&evidence);

        assert_eq!(
            medallion
                .action_models()
                .get("interaction_qos:foreground@long")
                .unwrap()
                .observations,
            1
        );
        assert_eq!(
            medallion
                .action_models()
                .get("interaction_qos:foreground")
                .unwrap()
                .observations,
            1
        );
        assert!(medallion
            .action_models()
            .contains_key("build|interaction_qos:foreground@long"));
        assert!(medallion
            .action_models()
            .contains_key("build|interaction_qos:foreground"));
        assert_eq!(
            parameter_parent_action_key("interaction_qos:foreground@unknown"),
            None
        );
    }

    #[test]
    fn episodic_reservoir_prevents_one_family_from_erasing_others() {
        let now_unix = Utc::now().timestamp();
        let hardware = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let mut next_id = 1_u64;
        for family in [
            ActuatorFamily::PredictiveThreshold,
            ActuatorFamily::InteractionQos,
            ActuatorFamily::PredictiveThreshold,
        ] {
            let count = if family == ActuatorFamily::InteractionQos {
                2
            } else {
                40
            };
            for _ in 0..count {
                let mut evidence = gold_evidence(
                    if family == ActuatorFamily::InteractionQos {
                        "interaction_qos:foreground"
                    } else {
                        "predictive_threshold:tighten"
                    },
                    0.04,
                    now_unix + next_id as i64,
                    hardware,
                );
                evidence.id = next_id;
                evidence.family = family;
                evidence.context_before.valid = true;
                medallion.admit_resolved(evidence);
                next_id += 1;
            }
        }

        let predictive = medallion
            .episodic_evidence
            .iter()
            .filter(|evidence| evidence.family == ActuatorFamily::PredictiveThreshold)
            .count();
        let qos = medallion
            .episodic_evidence
            .iter()
            .filter(|evidence| evidence.family == ActuatorFamily::InteractionQos)
            .count();
        assert_eq!(predictive, MAX_EPISODES_PER_FAMILY);
        assert_eq!(qos, 2);
        assert_eq!(medallion.episodic_evidence.len(), 14);
    }

    #[test]
    fn episodic_context_round_trips_only_for_the_same_installation() {
        let now_unix = Utc::now().timestamp();
        let hardware = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let mut evidence = gold_evidence("interaction_qos:foreground", 0.05, now_unix, hardware);
        evidence.family = ActuatorFamily::InteractionQos;
        evidence.context_before = ActuatorEpisodeContext {
            valid: true,
            fluidity_score: 0.82,
            foreground_app_hash: 42,
            ..ActuatorEpisodeContext::default()
        };
        medallion.admit_resolved(evidence);

        let encoded = serde_json::to_vec(&medallion.snapshot()).expect("serialize medallion");
        let persisted: TelemetryMedallionPersisted =
            serde_json::from_slice(&encoded).expect("deserialize medallion");
        let mut restored = TelemetryMedallion::new(LOCAL_ID);
        restored.restore(persisted.clone());
        assert_eq!(restored.episodic_evidence.len(), 1);
        assert_eq!(
            restored.episodic_evidence[0]
                .context_before
                .foreground_app_hash,
            42
        );

        let mut foreign = TelemetryMedallion::new(InstallationId(99));
        foreign.restore(persisted);
        assert!(foreign.episodic_evidence.is_empty());
    }

    #[test]
    fn every_live_cycle_advances_context_bronze_and_gold() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let signal = signal();
        let outcomes = ExecuteOutcomes::default();
        let snapshot = snapshot();
        let runtime = healthy_runtime();
        let capabilities = m4_capabilities();
        medallion.observe(TelemetryObservation {
            snapshot: &snapshot,
            hardware: None,
            runtime: &runtime,
            capabilities: Some(&capabilities),
            signal: &signal,
            workload: "idle",
            cycle: 1,
            outcomes: &outcomes,
            intervention: Intervention::Observe,
            applied_intervention: None,
            purge_recent: false,
            nars_drift_score: 0.0,
            nars_beliefs_total: 1,
            natural_drift: 0.0,
            arousal_level: 0.5,
        });
        let metrics = medallion.metrics();
        assert_eq!(metrics.bronze_total, 1);
        assert_eq!(metrics.silver_total, 0);
        assert_eq!(metrics.gold_total, 1);
        assert!(medallion.latest().is_some());
    }

    #[test]
    fn controlled_holdout_resolves_into_targeted_no_action_evidence() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = healthy_runtime();
        let outcomes = ExecuteOutcomes::default();
        observe(&mut medallion, 1, &outcomes, &runtime);
        let action = RootAction::BoostProcess {
            pid: 42,
            name: "Editor".to_string(),
            reason: "test".to_string(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        };
        assert!(medallion.issue_controlled_holdout(&action, "build", 1));

        for cycle in 2..=31 {
            observe(&mut medallion, cycle, &outcomes, &runtime);
        }

        let metrics = medallion.metrics();
        assert_eq!(metrics.controlled_holdout_issued_total, 1);
        assert_eq!(metrics.controlled_holdout_pending_total, 0);
        assert_eq!(metrics.controlled_holdout_resolved_total, 1);
        assert_eq!(metrics.controlled_holdout_rejected_total, 0);
        assert!(medallion
            .controlled_models
            .contains_key("build|boost:Editor"));
        let controlled = medallion
            .controlled_models
            .get("build|boost:Editor")
            .expect("resolved control model");
        assert!(controlled.last_observed_unix > 0);
        assert_eq!(controlled.installation_id, LOCAL_ID);
        assert!(controlled.hardware_regime.is_known());
        assert_eq!(medallion.model_calibration_metrics().record_count, 0);
    }

    #[test]
    fn controlled_holdout_rejects_raw_thread_qos() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = healthy_runtime();
        let outcomes = ExecuteOutcomes::default();
        observe(&mut medallion, 1, &outcomes, &runtime);
        let action = RootAction::SetThreadQoS {
            pid: 42,
            name: "Editor".to_string(),
            thread_index: 0,
            tier: "interactive".to_string(),
            reason: "test".to_string(),
            decision_reason: DecisionReason::PressureContext,
            affinity_tag: None,
            start_sec: 1,
            start_usec: 0,
        };

        assert!(!medallion.issue_controlled_holdout(&action, "build", 1));
        assert_eq!(medallion.metrics().controlled_holdout_issued_total, 0);
    }

    #[test]
    fn restore_rejects_pending_controlled_holdouts_instead_of_resuming_them() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = healthy_runtime();
        let outcomes = ExecuteOutcomes::default();
        observe(&mut medallion, 1, &outcomes, &runtime);
        let action = RootAction::BoostProcess {
            pid: 42,
            name: "Editor".to_string(),
            reason: "test".to_string(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        };
        assert!(medallion.issue_controlled_holdout(&action, "build", 1));

        let mut restored = TelemetryMedallion::new(LOCAL_ID);
        restored.restore(medallion.snapshot());

        let metrics = restored.metrics();
        assert_eq!(metrics.controlled_holdout_issued_total, 1);
        assert_eq!(metrics.controlled_holdout_pending_total, 0);
        assert_eq!(metrics.controlled_holdout_resolved_total, 0);
        assert_eq!(metrics.controlled_holdout_rejected_total, 1);
    }

    #[test]
    fn expired_markov_eta_is_normalized_before_context_admission() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let outcomes = ExecuteOutcomes::default();
        let mut runtime = healthy_runtime();
        runtime.markov_prediction_eta_secs = -148.25;

        let admission = observe(&mut medallion, 1, &outcomes, &runtime);

        assert_eq!(admission.tier, ContextTier::Gold);
    }

    #[test]
    fn multicore_process_cpu_is_normalized_by_machine_capacity() {
        let mut snapshot = snapshot();
        snapshot.top_processes.push(ProcessStats {
            pid: 42,
            name: "rustc".to_string(),
            cpu_usage: 350.0,
            memory_usage: 512 * 1024 * 1024,
            cpu_wall_ratio: Some(1.0),
        });
        let runtime = healthy_runtime();
        let signal = signal();
        let capabilities = m4_capabilities();
        let outcomes = ExecuteOutcomes::default();
        let observation = TelemetryObservation {
            snapshot: &snapshot,
            hardware: None,
            runtime: &runtime,
            capabilities: Some(&capabilities),
            signal: &signal,
            workload: "build",
            cycle: 1,
            outcomes: &outcomes,
            intervention: Intervention::Observe,
            applied_intervention: None,
            purge_recent: false,
            nars_drift_score: 0.0,
            nars_beliefs_total: 1,
            natural_drift: 0.0,
            arousal_level: 0.5,
        };

        let summary = summarize(&observation);

        assert!((summary.top_process_cpu - 0.35).abs() < 1e-6);
    }

    #[test]
    fn kernel_capabilities_and_apple_silicon_topology_enter_context() {
        let snapshot = snapshot();
        let signal = signal();
        let mut runtime = healthy_runtime();
        runtime.kpc_available = true;
        runtime.kpc_memory_bound_score = 0.42;
        runtime.amx_available = true;
        runtime.amx_cs_overhead_ns = 50;
        let outcomes = ExecuteOutcomes::default();
        let capabilities = CapabilityReport {
            can_taskpolicy: true,
            can_sysctl: true,
            can_memorystatus: true,
            can_memory_pressure_send: false,
            can_mdutil: true,
            can_tmutil: true,
            is_root: true,
            p_core_count: Some(4),
            e_core_count: Some(6),
            unavailable: vec!["memorystatus_pressure_send".to_string()],
            memorystatus_probe: Some("ok".to_string()),
            task_for_pid_probe: Some("ok".to_string()),
        };
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.observe(TelemetryObservation {
            snapshot: &snapshot,
            hardware: None,
            runtime: &runtime,
            capabilities: Some(&capabilities),
            signal: &signal,
            workload: "idle",
            cycle: 1,
            outcomes: &outcomes,
            intervention: Intervention::Observe,
            applied_intervention: None,
            purge_recent: false,
            nars_drift_score: 0.0,
            nars_beliefs_total: 1,
            natural_drift: 0.0,
            arousal_level: 0.5,
        });
        let context = medallion.latest().expect("context");
        assert!(context.daemon_is_root);
        assert!(context.kernel_taskpolicy_available);
        assert!(context.kernel_memorystatus_available);
        assert!(!context.kernel_pressure_send_available);
        assert_eq!((context.p_core_count, context.e_core_count), (4, 6));
        assert_eq!(context.unavailable_capability_count, 1);
        assert!(context.memorystatus_probe_ok);
        assert!(context.task_for_pid_probe_ok);
        assert!(context.kpc_available);
        assert!((context.kpc_memory_bound_score - 0.42).abs() < f64::EPSILON);
        assert!(context.amx_available);
        assert_eq!(context.amx_cs_overhead_ns, 50);

        let episode = ActuatorEpisodeContext::from_telemetry(context);
        assert!(episode.kpc_available);
        assert!(episode.amx_available);
        assert_eq!(episode.signal_entropy_anomaly, signal.entropy_anomaly);
    }

    #[test]
    fn rejected_and_silver_context_have_zero_learning_side_effects() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = healthy_runtime();
        assert_eq!(
            observe(&mut medallion, 1, &ExecuteOutcomes::default(), &runtime,).tier,
            ContextTier::Gold
        );
        let before_latest = medallion.latest.clone();
        let before_pending = medallion.pending_actions.len();
        let before_models = medallion.action_models.clone();
        let before_baseline = medallion.no_action_delta_ema.clone();

        let mut rejected_snapshot = snapshot();
        rejected_snapshot.pressure.memory_pressure = f64::NAN;
        let signal = signal();
        let capabilities = m4_capabilities();
        let outcomes = ExecuteOutcomes::default();
        let rejected = medallion.observe(TelemetryObservation {
            snapshot: &rejected_snapshot,
            hardware: None,
            runtime: &runtime,
            capabilities: Some(&capabilities),
            signal: &signal,
            workload: "idle",
            cycle: 2,
            outcomes: &outcomes,
            intervention: Intervention::Observe,
            applied_intervention: None,
            purge_recent: false,
            nars_drift_score: 0.0,
            nars_beliefs_total: 1,
            natural_drift: 0.0,
            arousal_level: 0.5,
        });
        assert_eq!(rejected.tier, ContextTier::Rejected);
        assert_eq!(medallion.latest, before_latest);
        assert_eq!(medallion.pending_actions.len(), before_pending);
        assert_eq!(medallion.action_models, before_models);
        assert_eq!(medallion.no_action_delta_ema, before_baseline);

        let mut silver_runtime = healthy_runtime();
        silver_runtime.collector_pressure_alive = false;
        assert_eq!(
            observe(
                &mut medallion,
                3,
                &ExecuteOutcomes::default(),
                &silver_runtime,
            )
            .tier,
            ContextTier::Silver
        );
        assert_eq!(medallion.latest, before_latest);
        assert_eq!(medallion.pending_actions.len(), before_pending);
        assert_eq!(medallion.action_models, before_models);
        assert_eq!(medallion.no_action_delta_ema, before_baseline);
        assert!(medallion.trusted_view().current.is_none());
    }

    #[test]
    fn baseline_requires_two_consecutive_gold_contexts() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = healthy_runtime();
        observe(&mut medallion, 1, &ExecuteOutcomes::default(), &runtime);
        let mut silver_runtime = healthy_runtime();
        silver_runtime.collector_pressure_alive = false;
        observe(
            &mut medallion,
            3,
            &ExecuteOutcomes::default(),
            &silver_runtime,
        );
        observe(&mut medallion, 3, &ExecuteOutcomes::default(), &runtime);
        assert!(medallion.no_action_delta_ema.is_empty());
        observe(&mut medallion, 4, &ExecuteOutcomes::default(), &runtime);
        assert!(!medallion.no_action_delta_ema.is_empty());
    }

    #[test]
    fn action_evidence_requires_gold_at_both_endpoints() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let outcomes = ExecuteOutcomes {
            audit_traces: vec![trace(
                RootAction::BoostProcess {
                    pid: 300,
                    name: "Editor".to_string(),
                    reason: "fixture".to_string(),
                    decision_reason: DecisionReason::InteractiveFocus,
                    start_sec: 12_345,
                    start_usec: 678,
                },
                true,
            )],
            ..ExecuteOutcomes::default()
        };
        let runtime = healthy_runtime();
        assert_eq!(
            observe(&mut medallion, 1, &outcomes, &runtime).tier,
            ContextTier::Gold
        );
        assert_eq!(medallion.metrics().actuator_pending_total, 1);

        let mut silver_runtime = healthy_runtime();
        silver_runtime.collector_pressure_alive = false;
        assert_eq!(
            observe(
                &mut medallion,
                4,
                &ExecuteOutcomes::default(),
                &silver_runtime,
            )
            .tier,
            ContextTier::Silver
        );
        assert!(medallion.action_models().is_empty());
        assert_eq!(medallion.metrics().actuator_pending_total, 1);

        observe(&mut medallion, 5, &ExecuteOutcomes::default(), &runtime);
        assert_eq!(medallion.metrics().actuator_bronze_total, 1);
    }

    #[test]
    fn restore_clamps_corrupt_tier_ordering() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.restore(TelemetryMedallionPersisted {
            bronze_total: 2,
            silver_total: 8,
            gold_total: 9,
            rejected_total: 4,
            invalid_total: 7,
            quality_sum: 99.0,
            ..TelemetryMedallionPersisted::default()
        });
        let metrics = medallion.metrics();
        assert_eq!(metrics.silver_total, 0);
        assert_eq!(metrics.gold_total, 2);
        assert_eq!(metrics.rejected_total, 0);
        assert_eq!(metrics.invalid_total, 0);
        assert_eq!(metrics.mean_quality, 1.0);
    }

    #[test]
    fn every_root_action_variant_has_a_universal_actuator_spec() {
        let reason = DecisionReason::PressureContext;
        let actions = vec![
            RootAction::BoostProcess {
                pid: 1,
                name: "Editor".to_string(),
                reason: "test".to_string(),
                decision_reason: reason.clone(),
                start_sec: 0,
                start_usec: 0,
            },
            RootAction::throttle(2, "Build", true, "test", reason.clone()),
            RootAction::freeze(3, "Worker", "test", reason.clone()),
            RootAction::unfreeze(4, "Browser", "test", reason.clone()),
            RootAction::set_memorystatus(5, 10, "test", reason.clone()),
            RootAction::set_sysctl("kern.ipc.somaxconn", "1024", "test", reason.clone()),
            RootAction::ToggleSpotlight {
                enabled: false,
                reason: "test".to_string(),
                decision_reason: reason.clone(),
            },
            RootAction::QuarantineDaemon {
                daemon: "indexer".to_string(),
                active: true,
                reason: "test".to_string(),
                decision_reason: reason.clone(),
            },
            RootAction::SetThreadQoS {
                pid: 6,
                name: "Editor".to_string(),
                thread_index: 0,
                tier: "interactive".to_string(),
                reason: "test".to_string(),
                decision_reason: reason,
                affinity_tag: Some(1),
                start_sec: 0,
                start_usec: 0,
            },
        ];
        let families: Vec<ActuatorFamily> = actions
            .iter()
            .map(|action| action_spec(action).expect("all variants covered").family)
            .collect();
        assert_eq!(families.len(), 9);
        assert!(families.contains(&ActuatorFamily::Boost));
        assert!(families.contains(&ActuatorFamily::ThreadQos));
        assert!(families.contains(&ActuatorFamily::Sysctl));
        assert!(families.contains(&ActuatorFamily::Unfreeze));
    }

    #[test]
    fn learning_identity_is_stable_across_pids_and_helper_names() {
        let reason = DecisionReason::PressureContext;
        let first = RootAction::set_memorystatus(11, 10, "test", reason.clone());
        let second = RootAction::set_memorystatus(99, 10, "test", reason.clone());
        assert_eq!(
            actuator_action_key(&first).as_deref(),
            Some("memorystatus:priority:10")
        );
        assert_eq!(actuator_action_key(&first), actuator_action_key(&second));

        let renderer = RootAction::BoostProcess {
            pid: 42,
            name: "Brave Browser Helper (Renderer)".to_string(),
            reason: "test".to_string(),
            decision_reason: reason,
            start_sec: 0,
            start_usec: 0,
        };
        let spec = action_spec(&renderer).expect("renderer actuator spec");
        assert_eq!(spec.action_key, "boost:chromium-renderer");
        assert_eq!(spec.target, "Brave Browser Helper (Renderer)");
        assert_eq!(spec.target_pid, Some(42));
    }

    #[test]
    fn only_confirmed_actions_resolve_into_causal_bronze() {
        let reason = DecisionReason::PressureContext;
        let outcomes = ExecuteOutcomes {
            audit_traces: vec![
                trace(
                    RootAction::BoostProcess {
                        pid: 10,
                        name: "Editor".to_string(),
                        reason: "test".to_string(),
                        decision_reason: reason.clone(),
                        start_sec: 0,
                        start_usec: 0,
                    },
                    true,
                ),
                trace(
                    RootAction::SetThreadQoS {
                        pid: 10,
                        name: "Editor".to_string(),
                        thread_index: 0,
                        tier: "interactive".to_string(),
                        reason: "test".to_string(),
                        decision_reason: reason,
                        affinity_tag: Some(1),
                        start_sec: 0,
                        start_usec: 0,
                    },
                    false,
                ),
            ],
            ..ExecuteOutcomes::default()
        };
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = healthy_runtime();
        observe(&mut medallion, 1, &outcomes, &runtime);
        assert_eq!(medallion.metrics().actuator_issued_total, 1);
        assert_eq!(medallion.metrics().actuator_bronze_total, 0);
        for cycle in 2..=4 {
            observe(&mut medallion, cycle, &ExecuteOutcomes::default(), &runtime);
        }
        let metrics = medallion.metrics();
        assert_eq!(metrics.actuator_bronze_total, 1);
        assert_eq!(metrics.actuator_silver_total, 1);
        assert_eq!(metrics.actuator_gold_total, 1);
        assert_eq!(medallion.recent_actuator_evidence().len(), 1);
        let dynamics = medallion.causal_dynamics().metrics(medallion.latest());
        assert_eq!(dynamics.gold_action_updates, 1);
        assert_eq!(dynamics.action_models, 3);
    }

    #[test]
    fn executed_decision_keeps_provenance_and_calibrates_every_source() {
        let action = RootAction::BoostProcess {
            pid: 300,
            name: "Editor".to_string(),
            reason: "fixture".to_string(),
            decision_reason: DecisionReason::InteractiveFocus,
            start_sec: 12_345,
            start_usec: 678,
        };
        let outcomes = ExecuteOutcomes {
            audit_traces: vec![trace(action, true)],
            ..ExecuteOutcomes::default()
        };
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.stage_decision_attribution(DecisionAttribution {
            action_key: "boost:Editor".to_string(),
            proposer: "interaction-specialist".to_string(),
            supporters: vec!["gpu-model".to_string()],
            vetoes: vec!["noop-control".to_string()],
            predicted_gain: 0.08,
            uncertainty: 0.15,
        });
        let before = RuntimeMetrics {
            perceptual_latency_score: 0.20,
            ..healthy_runtime()
        };
        observe(&mut medallion, 1, &outcomes, &before);
        for cycle in 2..=4 {
            observe(
                &mut medallion,
                cycle,
                &ExecuteOutcomes::default(),
                &healthy_runtime(),
            );
        }

        let evidence = medallion
            .recent_actuator_evidence()
            .back()
            .expect("resolved action evidence");
        assert_eq!(evidence.attribution.proposer, "interaction-specialist");
        assert_eq!(evidence.attribution.supporters, ["gpu-model"]);
        assert_eq!(evidence.attribution.vetoes, ["noop-control"]);
        assert!(evidence.perceptual_latency_improvement > 0.15);
        assert!(evidence.utility.human_gain > 0.0);
        assert!(evidence.utility.apollo_utility > 0.0);

        let sources = medallion.decision_source_stats();
        assert!(sources["interaction-specialist"].credit_ema > 0.0);
        assert!(sources["gpu-model"].credit_ema > 0.0);
        assert!(sources["noop-control"].credit_ema < 0.0);
        assert_eq!(medallion.metrics().decision_credit_sources, 3);

        let persisted = medallion.snapshot();
        let mut local = TelemetryMedallion::new(LOCAL_ID);
        local.restore(persisted.clone());
        assert_eq!(local.decision_source_stats().len(), 3);
        let mut foreign = TelemetryMedallion::new(InstallationId(99));
        foreign.restore(persisted);
        assert!(foreign.decision_source_stats().is_empty());
    }

    #[test]
    fn apollo_utility_keeps_human_and_system_outcomes_separate() {
        let healthy = decompose_utility(
            ActuatorFamily::InteractionQos,
            WorldStateDelta {
                pressure: -0.05,
                fluidity: 0.10,
                latency: -0.20,
                energy: -0.03,
                cpu: -0.02,
                thermal: 0.0,
                thrashing: -0.01,
                stall: -0.04,
            },
        );
        assert!(healthy.system_gain > 0.0);
        assert!(healthy.human_gain > 0.0);
        assert!(healthy.apollo_utility > 0.0);
        assert!((healthy.intervention_cost - 0.003).abs() < f64::EPSILON);

        let energy_win_but_ux_loss = decompose_utility(
            ActuatorFamily::Throttle,
            WorldStateDelta {
                fluidity: -0.30,
                latency: 0.40,
                energy: -0.50,
                cpu: -0.20,
                ..WorldStateDelta::default()
            },
        );
        assert!(energy_win_but_ux_loss.system_gain > 0.0);
        assert!(energy_win_but_ux_loss.human_gain < 0.0);
        assert!(energy_win_but_ux_loss.apollo_utility < 0.0);
    }

    #[test]
    fn uncontaminated_no_action_windows_train_causal_baseline() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = healthy_runtime();
        for cycle in 1..=8 {
            observe(&mut medallion, cycle, &ExecuteOutcomes::default(), &runtime);
        }
        let dynamics = medallion.causal_dynamics().metrics(medallion.latest());
        assert!(dynamics.no_action_updates >= 4);
        assert!(dynamics.baseline_models >= 1);
        assert!(dynamics.baseline_ready_models >= 1);
    }

    #[test]
    fn unpaired_markov_hit_is_effective_support_but_not_gold() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let mut runtime = RuntimeMetrics {
            markov_prewarm_applied: 1,
            markov_prediction_app: "Terminal".to_string(),
            ..healthy_runtime()
        };
        observe(&mut medallion, 1, &ExecuteOutcomes::default(), &runtime);
        assert_eq!(medallion.metrics().actuator_pending_total, 1);
        runtime.markov_prewarm_hits = 1;
        observe(&mut medallion, 2, &ExecuteOutcomes::default(), &runtime);
        let metrics = medallion.metrics();
        assert_eq!(metrics.actuator_bronze_total, 1);
        assert_eq!(metrics.actuator_gold_total, 0);
        assert_eq!(metrics.actuator_effective_total, 1);
        let evidence = medallion
            .recent_actuator_evidence()
            .back()
            .expect("resolved evidence");
        assert_eq!(evidence.family, ActuatorFamily::MarkovPrewarm);
        assert_eq!(evidence.provenance, EvidenceProvenance::SyntheticCounter);
        assert_eq!(evidence.tier, EvidenceTier::Bronze);
        assert_eq!(evidence.net_utility_delta, 1.0);
        assert!(medallion.drain_new_gold_evidence().is_empty());
    }

    #[test]
    fn io_promotion_gets_its_own_horizon_resolved_causal_evidence() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = RuntimeMetrics {
            acceleration_lease_io_promotions_total: 1,
            acceleration_lease_last_family: "chromium".to_string(),
            ..healthy_runtime()
        };
        observe(&mut medallion, 1, &ExecuteOutcomes::default(), &runtime);
        assert_eq!(medallion.metrics().actuator_pending_total, 1);

        for cycle in 2..=31 {
            observe(&mut medallion, cycle, &ExecuteOutcomes::default(), &runtime);
        }
        let evidence = medallion
            .recent_actuator_evidence()
            .back()
            .expect("resolved I/O evidence");
        assert_eq!(evidence.family, ActuatorFamily::IoShaping);
        assert_eq!(evidence.target, "chromium");
        assert_eq!(medallion.metrics().actuator_bronze_total, 1);
    }

    #[test]
    fn interaction_activation_records_the_actual_ttl_arm() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = RuntimeMetrics {
            interaction_qos_activations: 1,
            interaction_qos_reason: "input".to_string(),
            interaction_qos_ttl_band: "long".to_string(),
            interaction_qos_ttl_ms: 1_760,
            interaction_qos_ttl_exploratory: true,
            ..healthy_runtime()
        };
        observe(&mut medallion, 1, &ExecuteOutcomes::default(), &runtime);

        assert!(medallion.pending_actions.iter().any(|pending| {
            pending.family == ActuatorFamily::InteractionQos
                && pending.action_key == "interaction_qos:foreground@long"
        }));
    }

    #[test]
    fn policy_selected_interaction_keeps_parameter_models_causally_clean() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = RuntimeMetrics {
            interaction_qos_activations: 1,
            interaction_qos_reason: "input".to_string(),
            interaction_qos_ttl_band: "long".to_string(),
            interaction_qos_ttl_ms: 1_760,
            interaction_qos_ttl_exploratory: false,
            ..healthy_runtime()
        };
        observe(&mut medallion, 1, &ExecuteOutcomes::default(), &runtime);

        assert!(medallion.pending_actions.iter().any(|pending| {
            pending.family == ActuatorFamily::InteractionQos
                && pending.action_key == "interaction_qos:foreground"
        }));
        assert!(!medallion
            .pending_actions
            .iter()
            .any(|pending| pending.action_key.contains('@')));
    }

    #[test]
    fn confirmed_chromium_soft_actions_enter_the_universal_medallion() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = RuntimeMetrics {
            chromium_ecore_demotions_total: 1,
            chromium_purge_hints_total: 1,
            chromium_jetsam_demotions_total: 1,
            ..healthy_runtime()
        };
        observe(&mut medallion, 1, &ExecuteOutcomes::default(), &runtime);

        assert!(medallion.pending_actions.iter().any(|pending| {
            pending.family == ActuatorFamily::ChromiumEcore
                && pending.action_key == "chromium_ecore:background_renderer"
        }));
        assert!(medallion.pending_actions.iter().any(|pending| {
            pending.family == ActuatorFamily::ChromiumPurge
                && pending.action_key == "chromium_purge:purgeable_renderer"
        }));
        assert!(medallion.pending_actions.iter().any(|pending| {
            pending.family == ActuatorFamily::ChromiumJetsam
                && pending.action_key == "chromium_jetsam:background_renderer"
        }));
    }

    #[test]
    fn concurrent_actions_are_kept_as_silver_not_false_individual_gold() {
        let reason = DecisionReason::PressureContext;
        let outcomes = ExecuteOutcomes {
            audit_traces: vec![
                trace(
                    RootAction::throttle(1, "Build", true, "test", reason.clone()),
                    true,
                ),
                trace(RootAction::freeze(2, "Worker", "test", reason), true),
            ],
            ..ExecuteOutcomes::default()
        };
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = healthy_runtime();
        observe(&mut medallion, 1, &outcomes, &runtime);
        for cycle in 2..=6 {
            observe(&mut medallion, cycle, &ExecuteOutcomes::default(), &runtime);
        }
        let metrics = medallion.metrics();
        assert_eq!(metrics.actuator_bronze_total, 2);
        assert_eq!(metrics.actuator_silver_total, 2);
        assert_eq!(metrics.actuator_gold_total, 0);
        assert!(medallion
            .recent_actuator_evidence()
            .iter()
            .all(|evidence| evidence.confounder_count > 0));
        for cycle in 7..=9 {
            observe(&mut medallion, cycle, &ExecuteOutcomes::default(), &runtime);
        }
        let coordinated = medallion
            .recent_actuator_evidence()
            .iter()
            .find(|evidence| evidence.family == ActuatorFamily::Coordinated)
            .expect("coordinated treatment evidence");
        assert_eq!(coordinated.tier, EvidenceTier::Gold);
        assert_eq!(medallion.metrics().actuator_gold_total, 1);
    }

    #[test]
    fn old_context_only_state_deserializes_with_empty_actuator_lane() {
        let state: TelemetryMedallionPersisted = serde_json::from_str(
            r#"{"bronze_total":7,"silver_total":7,"gold_total":6,"last_cycle":40}"#,
        )
        .expect("backward-compatible state");
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.restore(state);
        let metrics = medallion.metrics();
        assert_eq!(metrics.bronze_total, 7);
        assert_eq!(metrics.actuator_issued_total, 0);
        assert_eq!(metrics.actuator_bronze_total, 0);
    }

    #[test]
    fn legacy_actuator_evidence_is_reset_but_context_is_preserved() {
        let mut legacy = TelemetryMedallionPersisted {
            bronze_total: 12,
            silver_total: 12,
            gold_total: 11,
            actuator_issued_total: 20,
            actuator_resolved_total: 19,
            actuator_silver_total: 19,
            actuator_gold_total: 18,
            actuator_effective_total: 12,
            actuator_quality_sum: 18.0,
            actuator_utility_sum: 0.4,
            ..TelemetryMedallionPersisted::default()
        };
        legacy.action_models.insert(
            "predictive_profile:aggressive".to_string(),
            ActionModelStats {
                observations: 12,
                effective_observations: 2,
                utility_ema: -0.1,
                quality_ema: 1.0,
                last_cycle: 9,
                ..ActionModelStats::default()
            },
        );
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.restore(legacy);
        let metrics = medallion.metrics();
        assert_eq!(metrics.bronze_total, 12);
        assert_eq!(metrics.actuator_issued_total, 0);
        assert_eq!(metrics.actuator_bronze_total, 0);
        assert!(medallion.action_models().is_empty());
    }

    #[test]
    fn action_model_evidence_decays_by_wall_clock_and_legacy_has_no_authority() {
        let now_unix = 10_000_000;
        let model = ActionModelStats {
            observations: 100,
            evidence_mass: 20.0,
            last_observed_unix: now_unix,
            ..ActionModelStats::default()
        };
        assert!((model.effective_evidence_at(now_unix) - 20.0).abs() < 1e-9);
        assert!((model.effective_evidence_at(now_unix + 7 * 24 * 60 * 60) - 10.0).abs() < 1e-9);
        assert!((model.effective_evidence_at(now_unix + 14 * 24 * 60 * 60) - 5.0).abs() < 1e-9);

        let legacy = ActionModelStats {
            observations: 10_000,
            evidence_mass: 64.0,
            last_observed_unix: 0,
            ..ActionModelStats::default()
        };
        assert_eq!(legacy.effective_evidence_at(now_unix), 0.0);
    }

    #[test]
    fn first_local_gold_resets_mixed_legacy_mean_and_hardware_regime() {
        let now_unix = 15_000_000;
        let m1 = HardwareRegime {
            p_core_count: 4,
            e_core_count: 4,
            ram_gib: 8,
        };
        let m4 = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.action_models.insert(
            "boost:Editor".to_string(),
            ActionModelStats {
                observations: 500,
                utility_ema: -0.9,
                evidence_mass: 64.0,
                quality_ema: 1.0,
                last_observed_unix: now_unix - 1,
                hardware_regime: m1,
                ..ActionModelStats::default()
            },
        );
        // A fresh model makes this a mixed-version restore. The old all-legacy
        // migration cannot hide the per-model M1 reset exercised below.
        medallion.action_models.insert(
            "boost:Terminal".to_string(),
            ActionModelStats {
                observations: 2,
                utility_ema: 0.02,
                evidence_mass: 2.0,
                quality_ema: 0.95,
                last_observed_unix: now_unix,
                hardware_regime: m4,
                ..ActionModelStats::default()
            },
        );

        medallion.admit_resolved(gold_evidence("boost:Editor", 0.08, now_unix, m4));

        let reset = medallion.action_models().get("boost:Editor").unwrap();
        assert_eq!(reset.observations, 501, "lifetime audit count is preserved");
        assert_eq!(reset.evidence_mass, 1.0);
        assert!((reset.utility_ema - 0.08).abs() < f64::EPSILON);
        assert_eq!(reset.utility_variance_ema, 0.0);
        assert_eq!(reset.hardware_regime, m4);
    }

    #[test]
    fn ready_metric_rejects_foreign_hardware_and_family_aggregates() {
        let now_unix = 16_000_000;
        let m1 = HardwareRegime {
            p_core_count: 4,
            e_core_count: 4,
            ram_gib: 8,
        };
        let m4_context = TelemetryContextSummary {
            timestamp_unix: now_unix,
            p_core_count: 4,
            e_core_count: 6,
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            ..TelemetryContextSummary::default()
        };
        let mature = ActionModelStats {
            observations: 20,
            utility_ema: 0.08,
            evidence_mass: 20.0,
            quality_ema: 0.95,
            last_observed_unix: now_unix,
            hardware_regime: m1,
            ..ActionModelStats::default()
        };
        let mut persisted = TelemetryMedallionPersisted {
            actuator_evidence_schema_version: ACTUATOR_EVIDENCE_SCHEMA_VERSION,
            latest: Some(m4_context),
            ..TelemetryMedallionPersisted::default()
        };
        persisted
            .action_models
            .insert("boost:Editor".to_string(), mature.clone());
        let mut local_family = mature;
        local_family.hardware_regime = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        persisted
            .action_models
            .insert("boost:*".to_string(), local_family);

        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.restore(persisted);
        assert_eq!(medallion.metrics().actuator_ready_models, 0);
    }

    #[test]
    fn legacy_models_remain_audit_only_without_installation_origin() {
        let now_unix = 20_000_000;
        let mut persisted = TelemetryMedallionPersisted {
            actuator_evidence_schema_version: ACTUATOR_EVIDENCE_SCHEMA_VERSION,
            last_cycle: 1_000,
            latest: Some(TelemetryContextSummary {
                timestamp_unix: now_unix,
                ..TelemetryContextSummary::default()
            }),
            ..TelemetryMedallionPersisted::default()
        };
        persisted.action_models.insert(
            "boost:Editor".to_string(),
            ActionModelStats {
                observations: 500,
                utility_ema: -0.9,
                quality_ema: 1.0,
                last_cycle: 500_000,
                ..ActionModelStats::default()
            },
        );
        for id in 1..=11 {
            persisted.recent_evidence.push(ResolvedActuatorEvidence {
                id,
                decision_id: None,
                family: ActuatorFamily::Boost,
                objective: ActuatorObjective::Responsiveness,
                action_key: "boost:Editor".to_string(),
                target: "Editor".to_string(),
                workload: "build".to_string(),
                issued_cycle: 997,
                resolved_cycle: 1_000,
                resolved_timestamp_unix: 0,
                hardware_regime: HardwareRegime::default(),
                installation_id: InstallationId::UNKNOWN,
                horizon_cycles: 3,
                tier: EvidenceTier::Gold,
                provenance: EvidenceProvenance::LegacyUnknown,
                quality: 0.95,
                raw_utility_delta: 0.08,
                counterfactual_delta: 0.0,
                net_utility_delta: 0.08,
                attribution: DecisionAttribution::default(),
                calibration_provenance: CalibrationProvenance::default(),
                learning_details: None,
                utility: UtilityDecomposition::default(),
                perceptual_latency_improvement: 0.0,
                net_state_delta: WorldStateDelta::default(),
                context_before: ActuatorEpisodeContext::default(),
                effective: true,
                confounder_count: 0,
                target_present_after: None,
            });
        }
        // Imported evidence from a prior daemon epoch is ahead of last_cycle
        // and must not enter the rebuilt M4 model.
        persisted.recent_evidence.push(ResolvedActuatorEvidence {
            id: 99,
            decision_id: None,
            family: ActuatorFamily::Boost,
            objective: ActuatorObjective::Responsiveness,
            action_key: "boost:ImportedM1".to_string(),
            target: "ImportedM1".to_string(),
            workload: "build".to_string(),
            issued_cycle: 499_997,
            resolved_cycle: 500_000,
            resolved_timestamp_unix: 0,
            hardware_regime: HardwareRegime::default(),
            installation_id: InstallationId::UNKNOWN,
            horizon_cycles: 3,
            tier: EvidenceTier::Gold,
            provenance: EvidenceProvenance::LegacyUnknown,
            quality: 1.0,
            raw_utility_delta: 1.0,
            counterfactual_delta: 0.0,
            net_utility_delta: 1.0,
            attribution: DecisionAttribution::default(),
            calibration_provenance: CalibrationProvenance::default(),
            learning_details: None,
            utility: UtilityDecomposition::default(),
            perceptual_latency_improvement: 0.0,
            net_state_delta: WorldStateDelta::default(),
            context_before: ActuatorEpisodeContext::default(),
            effective: true,
            confounder_count: 0,
            target_present_after: None,
        });

        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.restore(persisted);
        let legacy = medallion.action_models().get("boost:Editor").unwrap();
        assert_eq!(legacy.observations, 500);
        assert_eq!(legacy.evidence_mass, 0.0);
        assert_eq!(legacy.effective_evidence_at(now_unix), 0.0);
        assert!(!medallion.action_models().contains_key("boost:ImportedM1"));
        assert!(medallion.trusted_view().current.is_none());
        assert_eq!(medallion.metrics().actuator_ready_models, 0);
    }

    #[test]
    fn restore_discards_pending_action_endpoint_continuity() {
        let reason = DecisionReason::PressureContext;
        let outcomes = ExecuteOutcomes {
            audit_traces: vec![trace(
                RootAction::BoostProcess {
                    pid: 10,
                    name: "Editor".to_string(),
                    reason: "test".to_string(),
                    decision_reason: reason,
                    start_sec: 0,
                    start_usec: 0,
                },
                true,
            )],
            ..ExecuteOutcomes::default()
        };
        let runtime = healthy_runtime();
        let mut original = TelemetryMedallion::new(LOCAL_ID);
        observe(&mut original, 1, &outcomes, &runtime);
        let wire = serde_json::to_vec(&original.snapshot()).expect("serialize state");
        let persisted: TelemetryMedallionPersisted =
            serde_json::from_slice(&wire).expect("deserialize state");
        let mut restored = TelemetryMedallion::new(LOCAL_ID);
        restored.restore(persisted);
        assert!(restored.trusted_view().current.is_none());
        assert_eq!(restored.metrics().actuator_pending_total, 0);
        assert_eq!(restored.metrics().actuator_bronze_total, 0);
    }

    fn persisted_with_ready_model(installation_id: InstallationId) -> TelemetryMedallionPersisted {
        let now_unix = Utc::now().timestamp();
        let mut persisted = TelemetryMedallionPersisted {
            actuator_evidence_schema_version: ACTUATOR_EVIDENCE_SCHEMA_VERSION,
            context_schema_version: TELEMETRY_CONTEXT_SCHEMA_VERSION,
            installation_id,
            latest: Some(TelemetryContextSummary {
                timestamp_unix: now_unix,
                p_core_count: 4,
                e_core_count: 6,
                total_ram_bytes: 16 * 1024 * 1024 * 1024,
                ..TelemetryContextSummary::default()
            }),
            ..TelemetryMedallionPersisted::default()
        };
        persisted.action_models.insert(
            "boost:Editor".to_string(),
            ActionModelStats {
                observations: 20,
                effective_observations: 18,
                utility_ema: 0.08,
                evidence_mass: 20.0,
                utility_variance_ema: 0.0001,
                state_delta_ema: WorldStateDelta::default(),
                state_variance_ema: WorldStateDelta::default(),
                state_evidence_mass: 0.0,
                quality_ema: 0.95,
                last_cycle: 1,
                last_observed_unix: now_unix,
                hardware_regime: HardwareRegime {
                    p_core_count: 4,
                    e_core_count: 6,
                    ram_gib: 16,
                },
                installation_id,
            },
        );
        persisted
    }

    #[test]
    fn restore_never_restores_live_context_or_pending_endpoints() {
        let mut persisted = persisted_with_ready_model(LOCAL_ID);
        persisted.pending_actions.push(PendingActuatorEvidence {
            id: 1,
            decision_id: None,
            family: ActuatorFamily::Boost,
            objective: ActuatorObjective::Responsiveness,
            action_key: "boost:Editor".to_string(),
            target: "Editor".to_string(),
            target_pid: Some(300),
            workload: "idle".to_string(),
            issued_cycle: 1,
            horizon_cycles: 3,
            cohort_size: 1,
            issued_total_at_start: 1,
            purge_recent: false,
            event_resolved: false,
            gpu_prediction_generation: None,
            attribution: DecisionAttribution::default(),
            calibration_provenance: CalibrationProvenance::default(),
            provenance: EvidenceProvenance::LegacyUnknown,
            before: TelemetryContextSummary::default(),
        });
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.restore(persisted);
        assert!(medallion.trusted_view().current.is_none());
        assert_eq!(medallion.metrics().actuator_pending_total, 0);
        assert_eq!(medallion.metrics().actuator_ready_models, 0);
    }

    #[test]
    fn foreign_installation_cannot_regain_authority_from_one_local_context() {
        let persisted = persisted_with_ready_model(InstallationId(99));
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.restore(persisted);
        let runtime = healthy_runtime();
        observe(&mut medallion, 1, &ExecuteOutcomes::default(), &runtime);
        assert_eq!(medallion.metrics().actuator_ready_models, 0);
        assert!(medallion
            .action_models()
            .values()
            .all(|model| model.installation_id != LOCAL_ID || model.evidence_mass == 0.0));
    }

    #[test]
    fn same_installation_fresh_evidence_survives_after_new_local_gold() {
        let persisted = persisted_with_ready_model(LOCAL_ID);
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        medallion.restore(persisted);
        assert_eq!(medallion.metrics().actuator_ready_models, 0);
        let runtime = healthy_runtime();
        observe(&mut medallion, 1, &ExecuteOutcomes::default(), &runtime);
        assert_eq!(medallion.metrics().actuator_ready_models, 1);
    }

    #[test]
    fn causal_dynamics_persists_only_for_the_same_installation() {
        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        for sample in 1..=8 {
            let before = TelemetryContextSummary {
                cycle: sample,
                timestamp_unix: 1_800_000_000 + sample as i64,
                workload: "coding".to_string(),
                memory_pressure: 0.4,
                fluidity_score: 0.8,
                package_watts: Some(8.0),
                cpu_max_busy: 0.4,
                total_ram_bytes: 16 * 1024 * 1024 * 1024,
                p_core_count: 4,
                e_core_count: 6,
                ..TelemetryContextSummary::default()
            };
            medallion.causal_dynamics.observe_action(
                "boost:Editor",
                "boost",
                "coding",
                &before,
                WorldStateDelta {
                    pressure: -0.02,
                    fluidity: 0.03,
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
        let persisted = medallion.snapshot();
        let mut local = TelemetryMedallion::new(LOCAL_ID);
        local.restore(persisted.clone());
        assert!(local.causal_dynamics.metrics(None).action_models > 0);

        let mut foreign = TelemetryMedallion::new(InstallationId(99));
        foreign.restore(persisted);
        assert_eq!(foreign.causal_dynamics.metrics(None).action_models, 0);
    }

    #[test]
    fn gpu_prediction_advances_bronze_silver_gold_and_persists_calibration() {
        use crate::engine::gpu_imagination::{
            GpuCandidateAdvice, GpuImaginationBackend, GpuImaginationResult,
        };

        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = healthy_runtime();
        observe(&mut medallion, 1, &ExecuteOutcomes::default(), &runtime);
        observe(&mut medallion, 2, &ExecuteOutcomes::default(), &runtime);
        let before = medallion.latest().cloned().expect("live context");
        let result = GpuImaginationResult {
            generation: 2,
            workload: "idle".to_string(),
            context_revision: 0,
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
                context_score: 0.04,
            }],
            error: None,
        };
        assert_eq!(medallion.observe_gpu_imagination(&result, 2), 1);
        assert_eq!(medallion.metrics().gpu_prediction_bronze_total, 1);

        medallion.issue(
            ActionSpec::synthetic(
                ActuatorFamily::MarkovPrewarm,
                ActuatorObjective::Prediction,
                "markov_prewarm:predicted_app",
                "Editor",
                1,
            ),
            &before,
            3,
            1,
            false,
            true,
        );
        assert_eq!(medallion.metrics().gpu_prediction_silver_total, 1);
        let pending = medallion
            .pending_actions
            .pop_front()
            .expect("pending action");
        let mut after = before.clone();
        after.cycle = 4;
        after.timestamp_unix = after.timestamp_unix.saturating_add(1);
        medallion.resolve_one(pending, &after, &snapshot(), 4, Some(0.75));

        let metrics = medallion.metrics();
        assert_eq!(metrics.gpu_prediction_gold_total, 1);
        assert_eq!(metrics.gpu_prediction_pending_total, 0);
        assert_eq!(metrics.gpu_prediction_calibrated_models, 1);
        assert!(metrics.gpu_prediction_mean_absolute_error.is_finite());
        assert!(metrics.gpu_prediction_mean_brier.is_finite());

        let persisted = medallion.snapshot();
        let mut restored = TelemetryMedallion::new(LOCAL_ID);
        restored.restore(persisted);
        assert_eq!(restored.metrics().gpu_prediction_gold_total, 1);
        assert_eq!(restored.metrics().gpu_prediction_calibrated_models, 1);
        assert_eq!(restored.metrics().gpu_prediction_pending_total, 0);
    }

    #[test]
    fn same_cycle_gpu_advice_becomes_silver_only_after_confirmed_issue() {
        use crate::engine::gpu_imagination::{
            GpuCandidateAdvice, GpuImaginationBackend, GpuImaginationResult,
        };

        let mut medallion = TelemetryMedallion::new(LOCAL_ID);
        let runtime = healthy_runtime();
        observe(&mut medallion, 1, &ExecuteOutcomes::default(), &runtime);
        observe(&mut medallion, 2, &ExecuteOutcomes::default(), &runtime);
        let before = medallion.latest().cloned().expect("live context");
        let result = GpuImaginationResult {
            generation: 2,
            workload: "idle".to_string(),
            context_revision: 0,
            backend: GpuImaginationBackend::Metal,
            candidates: vec![GpuCandidateAdvice {
                action_key: "boost:Editor".to_string(),
                expected_gain: 0.03,
                uncertainty: 0.20,
                mean_gain: 0.025,
                p10_gain: 0.005,
                positive_probability: 0.75,
                rank_support: 0.002,
                context_score: 0.03,
            }],
            samples: 4_096,
            ..GpuImaginationResult::default()
        };
        medallion.observe_gpu_imagination(&result, 2);
        assert!(medallion.mark_gpu_prediction_consumed("boost:Editor", "idle", 2));
        assert_eq!(medallion.metrics().gpu_prediction_silver_total, 0);

        medallion.issue(
            ActionSpec::synthetic(
                ActuatorFamily::Boost,
                ActuatorObjective::Responsiveness,
                "boost:Editor",
                "Editor",
                3,
            ),
            &before,
            2,
            1,
            false,
            false,
        );
        assert_eq!(medallion.metrics().gpu_prediction_silver_total, 1);
        assert_eq!(
            medallion
                .pending_actions
                .back()
                .and_then(|pending| pending.gpu_prediction_generation),
            Some(2)
        );
    }

    #[test]
    fn responsiveness_utility_rewards_measured_latency_improvement() {
        let mut before = TelemetryContextSummary {
            fluidity_score: 0.80,
            perceptual_latency_score: 0.70,
            ..TelemetryContextSummary::default()
        };
        let before_utility = utility_score(ActuatorObjective::Responsiveness, &before);
        before.perceptual_latency_score = 0.10;
        let after_utility = utility_score(ActuatorObjective::Responsiveness, &before);

        assert!(after_utility > before_utility);
        assert!((after_utility - before_utility - 0.12).abs() < 1e-9);
    }
}
