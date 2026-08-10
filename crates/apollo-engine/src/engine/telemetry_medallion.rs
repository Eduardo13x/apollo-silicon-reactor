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

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::collector::SystemSnapshot;
use crate::engine::causal_dynamics::CausalDynamicsModel;
use crate::engine::execute_actions::ExecuteOutcomes;
use crate::engine::gpu_imagination::GpuImaginationResult;
use crate::engine::installation_identity::InstallationId;
use crate::engine::iokit_sensors::HardwareSnapshot;
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
const GPU_PREDICTION_MATCH_MAX_AGE_CYCLES: u64 = 30;
const CONTROLLED_HOLDOUT_HORIZON_CYCLES: u64 = 30;
const ACTION_MODEL_EMA_ALPHA: f64 = 0.20;
const ACTION_MODEL_EVIDENCE_HALF_LIFE_SECS: f64 = 7.0 * 24.0 * 60.0 * 60.0;
const ACTION_MODEL_EVIDENCE_CAP: f64 = 64.0;
// Version 1 enrolled predictive recommendations before confirming that their
// operating-system side effect occurred. Do not carry that bias forward.
const ACTUATOR_EVIDENCE_SCHEMA_VERSION: u32 = 2;
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
            + self.energy
            + self.cpu
            + self.thermal
            + self.thrashing
            + self.stall)
            / 7.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedActuatorEvidence {
    pub id: u64,
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
    pub quality: f64,
    pub raw_utility_delta: f64,
    pub counterfactual_delta: f64,
    pub net_utility_delta: f64,
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
    pub windowserver_cpu_fraction: f64,
    pub arousal_level: f64,
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
    pub markov_prewarm_active: bool,
    pub predictive_agent_active: bool,
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
            windowserver_cpu_fraction: context.windowserver_cpu_fraction,
            arousal_level: context.arousal_level,
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
            markov_prewarm_active: context.markov_prewarm_active,
            predictive_agent_active: context.predictive_agent_active,
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
            self.windowserver_cpu_fraction,
            self.arousal_level,
            self.markov_prediction_confidence,
            self.network_retransmit_fraction,
            self.network_drop_rate,
            self.package_power_fraction,
            self.p_cluster_util,
            self.e_cluster_util,
            self.ane_util_fraction,
            self.user_idle_fraction,
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
    before: TelemetryContextSummary,
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

#[derive(Debug)]
struct ActionSpec {
    family: ActuatorFamily,
    objective: ActuatorObjective,
    action_key: String,
    target: String,
    target_pid: Option<u32>,
    horizon_cycles: u64,
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
    pub recent_evidence: Vec<ResolvedActuatorEvidence>,
    #[serde(default)]
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
}

#[derive(Debug)]
pub struct TelemetryMedallion {
    installation_id: InstallationId,
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
    family_stats: BTreeMap<ActuatorFamily, ActuatorFamilyStats>,
    action_models: BTreeMap<String, ActionModelStats>,
    action_models_revision: u64,
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
}

impl Default for TelemetryMedallion {
    fn default() -> Self {
        Self {
            installation_id: InstallationId::UNKNOWN,
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
            family_stats: BTreeMap::new(),
            action_models: BTreeMap::new(),
            action_models_revision: 0,
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
            causal_dynamics: CausalDynamicsModel::new(installation_id),
            ..Self::default()
        }
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
        if evidence.tier == EvidenceTier::Bronze {
            self.gpu_prediction_rejected_total =
                self.gpu_prediction_rejected_total.saturating_add(1);
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
        prediction.tier = evidence.tier;
        if evidence.tier != EvidenceTier::Gold {
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
        let external_deltas = self.external_deltas(runtime);
        let resolved_this_cycle = self.resolve_pending(
            &summary,
            snapshot,
            cycle,
            external_deltas.markov_hits,
            external_deltas.markov_misses,
            external_deltas.interaction_reverts,
        );

        let root_cohort_size = applied_root_actions.len();
        let mut issued_this_cycle = root_cohort_size as u64;
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
            self.issue(
                ActionSpec::synthetic(
                    ActuatorFamily::Coordinated,
                    ActuatorObjective::BalancedUtility,
                    &format!("coordinated:{family_key}"),
                    &target,
                    8,
                ),
                &summary,
                cycle,
                1,
                purge_recent,
                false,
            );
        }
        let cohort_end_total = self.actuator_issued_total;
        for pending in self.pending_actions.iter_mut().rev() {
            if pending.issued_cycle != cycle {
                break;
            }
            pending.issued_total_at_start = cohort_end_total;
        }
        self.external_counters = ExternalActuatorCounters::from_runtime(runtime, intervention);

        self.latest = Some(summary.clone());
        self.last_admitted_live = Some(summary);
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
        if !matches!(
            spec.family,
            ActuatorFamily::Boost | ActuatorFamily::ThreadQos
        ) {
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
        if self.pending_actions.len() >= MAX_PENDING_ACTIONS {
            if let Some(evicted) = self.pending_actions.pop_front() {
                self.expire_unresolved(evicted.family);
            }
        }
        let gpu_prediction_generation =
            self.mark_gpu_prediction_used(&spec.action_key, &before.workload, cycle);
        self.next_action_id = self.next_action_id.saturating_add(1);
        self.actuator_issued_total = self.actuator_issued_total.saturating_add(1);
        let family_stats = self.family_stats.entry(spec.family).or_default();
        family_stats.issued_total = family_stats.issued_total.saturating_add(1);
        self.pending_actions.push_back(PendingActuatorEvidence {
            id: self.next_action_id,
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
        let tier = if !finite {
            EvidenceTier::Bronze
        } else if confounders == 0 && quality >= 0.85 {
            EvidenceTier::Gold
        } else {
            EvidenceTier::Silver
        };
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
            quality,
            raw_utility_delta: finite_or_zero(raw_delta),
            counterfactual_delta: finite_or_zero(counterfactual),
            net_utility_delta: finite_or_zero(net_delta),
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

    fn admit_resolved(&mut self, evidence: ResolvedActuatorEvidence) {
        self.actuator_resolved_total = self.actuator_resolved_total.saturating_add(1);
        self.actuator_quality_sum += evidence.quality;
        self.actuator_utility_sum += evidence.net_utility_delta;
        if evidence.effective {
            self.actuator_effective_total = self.actuator_effective_total.saturating_add(1);
        }
        if evidence.tier != EvidenceTier::Bronze {
            self.actuator_silver_total = self.actuator_silver_total.saturating_add(1);
        } else {
            self.actuator_rejected_total = self.actuator_rejected_total.saturating_add(1);
        }
        if evidence.tier != EvidenceTier::Bronze
            && evidence.quality >= 0.85
            && evidence.context_before.valid
        {
            self.admit_episode(evidence.clone());
        }
        if evidence.tier == EvidenceTier::Gold {
            self.actuator_gold_total = self.actuator_gold_total.saturating_add(1);
            self.update_action_model(&evidence);
            if self.new_gold_evidence.len() >= MAX_RECENT_EVIDENCE {
                self.new_gold_evidence.pop_front();
            }
            self.new_gold_evidence.push_back(evidence.clone());
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
        self.recent_evidence.push_back(evidence);
    }

    fn admit_episode(&mut self, evidence: ResolvedActuatorEvidence) {
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
        for key in keys {
            if !self.action_models.contains_key(&key)
                && self.action_models.len() >= MAX_ACTION_MODELS
            {
                if let Some(evict) = self
                    .action_models
                    .iter()
                    .min_by_key(|(_, stats)| (stats.observations, stats.last_cycle))
                    .map(|(key, _)| key.clone())
                {
                    self.action_models.remove(&evict);
                }
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

    pub fn recent_actuator_evidence(&self) -> &VecDeque<ResolvedActuatorEvidence> {
        &self.recent_evidence
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

    pub fn snapshot(&self) -> TelemetryMedallionPersisted {
        TelemetryMedallionPersisted {
            actuator_evidence_schema_version: ACTUATOR_EVIDENCE_SCHEMA_VERSION,
            context_schema_version: TELEMETRY_CONTEXT_SCHEMA_VERSION,
            installation_id: self.installation_id,
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
        }
    }

    pub fn restore(&mut self, state: TelemetryMedallionPersisted) {
        let same_origin = state.context_schema_version == TELEMETRY_CONTEXT_SCHEMA_VERSION
            && state.installation_id.is_known()
            && state.installation_id == self.installation_id;
        let reset_actuator_evidence =
            state.actuator_evidence_schema_version < ACTUATOR_EVIDENCE_SCHEMA_VERSION;

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
        self.recent_evidence = state
            .recent_evidence
            .into_iter()
            .filter(|evidence| {
                evidence.action_key.len() <= 320
                    && evidence.target.len() <= 256
                    && evidence.workload.len() <= 64
                    && evidence.quality.is_finite()
                    && evidence.raw_utility_delta.is_finite()
                    && evidence.counterfactual_delta.is_finite()
                    && evidence.net_utility_delta.is_finite()
                    && (!evidence.context_before.valid || evidence.context_before.is_finite())
            })
            .take(MAX_RECENT_EVIDENCE)
            .collect();
        self.episodic_evidence.clear();
        for evidence in state
            .episodic_evidence
            .into_iter()
            .filter(|evidence| {
                evidence.tier != EvidenceTier::Bronze
                    && evidence.action_key.len() <= 320
                    && evidence.target.len() <= 256
                    && evidence.workload.len() <= 64
                    && evidence.quality.is_finite()
                    && evidence.net_utility_delta.is_finite()
                    && evidence.context_before.valid
                    && evidence.context_before.is_finite()
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
        } else {
            self.gpu_prediction_bronze_total = 0;
            self.gpu_prediction_silver_total = 0;
            self.gpu_prediction_gold_total = 0;
            self.gpu_prediction_rejected_total = 0;
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
        }
        self.action_models_revision = self.action_models_revision.wrapping_add(1);
        self.controlled_models_revision = self.controlled_models_revision.wrapping_add(1);
        self.gpu_calibration_revision = self.gpu_calibration_revision.wrapping_add(1);
    }
}

fn bounded_text(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn gpu_action_matches(predicted: &str, observed: &str) -> bool {
    predicted == observed
        || (predicted == "interaction_qos:foreground"
            && observed.starts_with("interaction_qos:foreground@"))
        || (predicted == "predictive_prethrottle:noise"
            && observed.starts_with("predictive_prethrottle:"))
        || (predicted == "predictive_purge:kernel" && observed.starts_with("predictive_purge:"))
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
    Some(ActionSpec {
        family,
        objective,
        action_key: format!("{}:{}", family.as_str(), target),
        target,
        target_pid,
        horizon_cycles,
    })
}

pub fn actuator_action_key(action: &RootAction) -> Option<String> {
    action_spec(action).map(|spec| spec.action_key)
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
    Some(ActionSpec::synthetic(family, objective, key, key, 5))
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
            0.40 * fluidity
                + 0.15 * stall_health
                + 0.15 * refault_health
                + 0.10 * ws_headroom
                + 0.10 * pressure_health
                + 0.10 * thermal_health
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
        summary.fluidity_score,
        summary.stall_fraction,
        summary.windowserver_cpu_fraction,
        summary.network_retransmits_per_k,
        summary.network_listen_drop_rate,
        summary.pressure_total_boost,
    ]
    .into_iter()
    .filter(|value| value.is_finite())
    .count();
    finite as f64 / 15.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::{CpuStats, MemoryStats, PressureStats, ProcessStats};
    use crate::engine::audit_types::{DecisionReason, PolicyDecisionTrace};
    use crate::engine::lotka_volterra::StabilityRegime;
    use chrono::Utc;

    const LOCAL_ID: InstallationId = InstallationId(0x1020_3040_5060_7080);

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

    fn gold_evidence(
        action_key: &str,
        utility: f64,
        timestamp_unix: i64,
        hardware_regime: HardwareRegime,
    ) -> ResolvedActuatorEvidence {
        ResolvedActuatorEvidence {
            id: 1,
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
            quality: 0.95,
            raw_utility_delta: utility,
            counterfactual_delta: 0.0,
            net_utility_delta: utility,
            net_state_delta: WorldStateDelta::default(),
            context_before: ActuatorEpisodeContext::default(),
            effective: utility > 0.0,
            confounder_count: 0,
            target_present_after: None,
        }
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
        let runtime = healthy_runtime();
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
    fn markov_hit_is_resolved_as_effective_gold_evidence() {
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
        assert_eq!(metrics.actuator_gold_total, 1);
        assert_eq!(metrics.actuator_effective_total, 1);
        let evidence = medallion
            .recent_actuator_evidence()
            .back()
            .expect("resolved evidence");
        assert_eq!(evidence.family, ActuatorFamily::MarkovPrewarm);
        assert_eq!(evidence.net_utility_delta, 1.0);
        assert_eq!(medallion.drain_new_gold_evidence().len(), 1);
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
                quality: 0.95,
                raw_utility_delta: 0.08,
                counterfactual_delta: 0.0,
                net_utility_delta: 0.08,
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
            quality: 1.0,
            raw_utility_delta: 1.0,
            counterfactual_delta: 0.0,
            net_utility_delta: 1.0,
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
}
