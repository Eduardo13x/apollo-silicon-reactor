//! Compact causal latent-dynamics ensemble for the World Model.
//!
//! The model learns Gold-only, counterfactual-adjusted state transitions. A
//! bounded deterministic ensemble predicts in a normalized joint-state space;
//! prequential validation (predict before update) decides whether its output is
//! exploratory or mature enough to join the authoritative ranking lane.
//!
//! This is intentionally a small online model rather than a general neural
//! runtime. It provides the useful JEPA/RSSM properties for Apollo's domain:
//! predicting a target-state embedding, conditioning on the current state and
//! action, maintaining uncertainty, and rolling the model forward under MPC.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::engine::installation_identity::InstallationId;
use crate::engine::telemetry_medallion::{
    HardwareRegime, TelemetryContextSummary, WorldStateDelta,
};

const SCHEMA_VERSION: u32 = 1;
const STATE_DIM: usize = 7;
const FEATURE_DIM: usize = 12;
const ENSEMBLE_SIZE: usize = 5;
const MAX_ACTION_MODELS: usize = 256;
const MAX_BASELINE_MODELS: usize = 16;
const MAX_MODEL_KEY_BYTES: usize = 2_048;
const MAX_WORKLOAD_KEY_BYTES: usize = 256;
const PUBLICATION_CADENCE: u64 = 8;
const MODEL_HALF_LIFE_SECS: f64 = 7.0 * 24.0 * 60.0 * 60.0;
const EVIDENCE_CAP: f64 = 256.0;
const MIN_PREDICTION_EVIDENCE: f64 = 4.0;
const MIN_PREDICTION_QUALITY: f64 = 0.85;
const MIN_AUTHORITATIVE_EVIDENCE: f64 = 16.0;
const MIN_AUTHORITATIVE_VALIDATIONS: u32 = 12;
const MIN_AUTHORITATIVE_QUALITY: f64 = 0.90;
const MAX_AUTHORITATIVE_MAE: f64 = 0.08;
const MIN_AUTHORITATIVE_COVERAGE: f64 = 0.60;
const MAX_AUTHORITATIVE_UNCERTAINTY: f64 = 0.12;
const MIN_RANKING_EVIDENCE: f64 = 8.0;
const MIN_RANKING_VALIDATIONS: u32 = 4;
const MAX_RANKING_MAE: f64 = 0.15;
const MIN_RANKING_COVERAGE: f64 = 0.40;
const MAX_RANKING_UNCERTAINTY: f64 = 0.25;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DynamicsAuthorityPhase {
    #[default]
    Protected,
    Calibrating,
    Shadow,
    Trusted,
}

impl DynamicsAuthorityPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Protected => "protected",
            Self::Calibrating => "calibrating",
            Self::Shadow => "shadow",
            Self::Trusted => "trusted",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DynamicsState {
    values: [f64; STATE_DIM],
}

impl DynamicsState {
    pub(crate) fn from_context(context: &TelemetryContextSummary) -> Self {
        Self {
            values: [
                context.memory_pressure.clamp(0.0, 1.0),
                context.fluidity_score.clamp(0.0, 1.0),
                (context.package_watts.unwrap_or(0.0) / 50.0).clamp(0.0, 1.0),
                context.cpu_max_busy.clamp(0.0, 1.0),
                context.thermal_score.clamp(0.0, 1.0),
                (context.thrashing_score / 50_000.0).clamp(0.0, 1.0),
                context.stall_fraction.clamp(0.0, 1.0),
            ],
        }
    }

    pub(crate) fn apply(self, delta: WorldStateDelta) -> Self {
        let delta = delta_to_array(delta);
        let mut values = self.values;
        for (value, change) in values.iter_mut().zip(delta) {
            *value = (*value + change).clamp(0.0, 1.0);
        }
        Self { values }
    }

    pub(crate) fn delta_from(self, baseline: Self) -> WorldStateDelta {
        array_to_delta(std::array::from_fn(|index| {
            self.values[index] - baseline.values[index]
        }))
    }

    pub(crate) fn utility(self) -> f64 {
        let [pressure, fluidity, energy, cpu, thermal, thrashing, stall] = self.values;
        (0.24 * (1.0 - pressure)
            + 0.25 * fluidity
            + 0.12 * (1.0 - energy)
            + 0.08 * (1.0 - cpu)
            + 0.11 * (1.0 - thermal)
            + 0.10 * (1.0 - thrashing)
            + 0.10 * (1.0 - stall))
            .clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicsEvidenceScope {
    Workload,
    Action,
    Family,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicsForecast {
    pub mean_delta: WorldStateDelta,
    pub member_deltas: [WorldStateDelta; ENSEMBLE_SIZE],
    pub uncertainty: f64,
    pub effective_evidence: f64,
    pub validation_samples: u32,
    pub validation_mae: f64,
    pub validation_coverage: f64,
    pub scope: DynamicsEvidenceScope,
    pub ranking_eligible: bool,
    pub authoritative: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CausalDynamicsMetrics {
    pub phase: DynamicsAuthorityPhase,
    pub action_models: u64,
    pub ready_models: u64,
    pub ranking_eligible_models: u64,
    pub authoritative_models: u64,
    pub baseline_models: u64,
    pub baseline_ready_models: u64,
    pub gold_action_updates: u64,
    pub no_action_updates: u64,
    pub validation_samples: u64,
    pub validation_mae: f64,
    pub validation_coverage: f64,
    pub publication_revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct EnsembleMember {
    weights: [[f64; FEATURE_DIM]; STATE_DIM],
    updates: u32,
}

impl Default for EnsembleMember {
    fn default() -> Self {
        Self {
            weights: [[0.0; FEATURE_DIM]; STATE_DIM],
            updates: 0,
        }
    }
}

impl EnsembleMember {
    fn predict(&self, features: &[f64; FEATURE_DIM]) -> [f64; STATE_DIM] {
        std::array::from_fn(|output| {
            self.weights[output]
                .iter()
                .zip(features)
                .map(|(weight, feature)| weight * feature)
                .sum::<f64>()
                .clamp(-1.0, 1.0)
        })
    }

    fn update(
        &mut self,
        features: &[f64; FEATURE_DIM],
        target: &[f64; STATE_DIM],
        bootstrap_weight: f64,
    ) {
        let prediction = self.predict(features);
        let norm = features.iter().map(|value| value * value).sum::<f64>() + 0.25;
        let anneal = 1.0 / (1.0 + self.updates as f64 / 256.0).sqrt();
        let step = (0.16 * anneal * bootstrap_weight / norm).clamp(0.002, 0.20);
        for output in 0..STATE_DIM {
            let error = (target[output] - prediction[output]).clamp(-1.0, 1.0);
            for (weight, feature) in self.weights[output].iter_mut().zip(features) {
                *weight = (*weight + step * error * feature).clamp(-2.0, 2.0);
            }
        }
        self.updates = self.updates.saturating_add(1);
    }

    fn sanitize(&mut self) -> bool {
        if self
            .weights
            .iter()
            .flatten()
            .any(|weight| !weight.is_finite())
        {
            return false;
        }
        for weight in self.weights.iter_mut().flatten() {
            *weight = weight.clamp(-2.0, 2.0);
        }
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct DynamicsRegressor {
    observations: u32,
    evidence_mass: f64,
    quality_ema: f64,
    validation_samples: u32,
    validation_mae_ema: f64,
    validation_coverage_ema: f64,
    residual_variance_ema: [f64; STATE_DIM],
    mean_horizon_cycles: f64,
    last_observed_unix: i64,
    hardware_regime: HardwareRegime,
    installation_id: InstallationId,
    members: [EnsembleMember; ENSEMBLE_SIZE],
}

impl Default for DynamicsRegressor {
    fn default() -> Self {
        Self {
            observations: 0,
            evidence_mass: 0.0,
            quality_ema: 0.0,
            validation_samples: 0,
            validation_mae_ema: 0.0,
            validation_coverage_ema: 0.0,
            residual_variance_ema: [0.0; STATE_DIM],
            mean_horizon_cycles: 1.0,
            last_observed_unix: 0,
            hardware_regime: HardwareRegime::default(),
            installation_id: InstallationId::UNKNOWN,
            members: std::array::from_fn(|_| EnsembleMember::default()),
        }
    }
}

impl DynamicsRegressor {
    #[allow(clippy::too_many_arguments)]
    fn observe(
        &mut self,
        features: &[f64; FEATURE_DIM],
        target: WorldStateDelta,
        horizon_cycles: u64,
        quality: f64,
        timestamp_unix: i64,
        hardware_regime: HardwareRegime,
        installation_id: InstallationId,
        sample_id: u64,
    ) {
        let same_epoch = self.installation_id.is_known()
            && self.installation_id == installation_id
            && self.hardware_regime.is_known()
            && self.hardware_regime == hardware_regime
            && timestamp_unix >= self.last_observed_unix;
        if !same_epoch {
            *self = Self::default();
        }

        let target = delta_to_array(target.clamped(-1.0, 1.0));
        if self.observations >= 4 {
            let prediction = self.predict_arrays(features);
            let mut mae = 0.0;
            let mut covered = 0_u32;
            for (output, target_value) in target.iter().copied().enumerate() {
                let error = (target_value - prediction.mean[output]).abs();
                mae += error;
                let radius = 2.0
                    * (prediction.variance[output] + self.residual_variance_ema[output] + 1e-5)
                        .sqrt();
                covered = covered.saturating_add(u32::from(error <= radius.max(0.01)));
                self.residual_variance_ema[output] = (0.90 * self.residual_variance_ema[output]
                    + 0.10 * error * error)
                    .clamp(0.0, 1.0);
            }
            mae /= STATE_DIM as f64;
            let coverage = covered as f64 / STATE_DIM as f64;
            if self.validation_samples == 0 {
                self.validation_mae_ema = mae;
                self.validation_coverage_ema = coverage;
            } else {
                self.validation_mae_ema = 0.90 * self.validation_mae_ema + 0.10 * mae;
                self.validation_coverage_ema =
                    0.90 * self.validation_coverage_ema + 0.10 * coverage;
            }
            self.validation_samples = self.validation_samples.saturating_add(1);
        }

        for (index, member) in self.members.iter_mut().enumerate() {
            member.update(features, &target, bootstrap_weight(sample_id, index as u64));
        }
        self.observations = self.observations.saturating_add(1);
        self.evidence_mass =
            (if same_epoch { self.evidence_mass } else { 0.0 } + 1.0).min(EVIDENCE_CAP);
        self.quality_ema = if self.observations == 1 {
            quality
        } else {
            0.90 * self.quality_ema + 0.10 * quality
        }
        .clamp(0.0, 1.0);
        self.mean_horizon_cycles = if self.observations == 1 {
            horizon_cycles.max(1) as f64
        } else {
            0.90 * self.mean_horizon_cycles + 0.10 * horizon_cycles.max(1) as f64
        };
        self.last_observed_unix = timestamp_unix;
        self.hardware_regime = hardware_regime;
        self.installation_id = installation_id;
    }

    fn effective_evidence_at(&self, now_unix: i64) -> f64 {
        if self.last_observed_unix <= 0 || now_unix < self.last_observed_unix {
            return 0.0;
        }
        let age = (now_unix - self.last_observed_unix) as f64;
        (self.evidence_mass * 0.5_f64.powf(age / MODEL_HALF_LIFE_SECS)).clamp(0.0, EVIDENCE_CAP)
    }

    fn ready_for(
        &self,
        context: &TelemetryContextSummary,
        installation_id: InstallationId,
    ) -> bool {
        installation_id.is_known()
            && self.installation_id == installation_id
            && self.hardware_regime.matches_context(context)
            && self.quality_ema >= MIN_PREDICTION_QUALITY
            && self.effective_evidence_at(context.timestamp_unix) >= MIN_PREDICTION_EVIDENCE
    }

    fn forecast(
        &self,
        features: &[f64; FEATURE_DIM],
        context: &TelemetryContextSummary,
        installation_id: InstallationId,
        scope: DynamicsEvidenceScope,
    ) -> Option<DynamicsForecast> {
        if !self.ready_for(context, installation_id) {
            return None;
        }
        let prediction = self.predict_arrays(features);
        let evidence = self.effective_evidence_at(context.timestamp_unix).max(1.0);
        let ensemble_variance = prediction.variance.iter().sum::<f64>() / STATE_DIM as f64;
        let residual_variance = self.residual_variance_ema.iter().sum::<f64>() / STATE_DIM as f64;
        let cold_start = 0.01 / evidence.sqrt();
        let uncertainty = (ensemble_variance + residual_variance).sqrt() + cold_start;
        let uncertainty = uncertainty.clamp(0.0, 1.0);
        let ranking_eligible = evidence >= MIN_RANKING_EVIDENCE
            && self.validation_samples >= MIN_RANKING_VALIDATIONS
            && self.quality_ema >= MIN_AUTHORITATIVE_QUALITY
            && self.validation_mae_ema <= MAX_RANKING_MAE
            && self.validation_coverage_ema >= MIN_RANKING_COVERAGE
            && uncertainty <= MAX_RANKING_UNCERTAINTY;
        let authoritative = scope != DynamicsEvidenceScope::Family
            && ranking_eligible
            && evidence >= MIN_AUTHORITATIVE_EVIDENCE
            && self.validation_samples >= MIN_AUTHORITATIVE_VALIDATIONS
            && self.quality_ema >= MIN_AUTHORITATIVE_QUALITY
            && self.validation_mae_ema <= MAX_AUTHORITATIVE_MAE
            && self.validation_coverage_ema >= MIN_AUTHORITATIVE_COVERAGE
            && uncertainty <= MAX_AUTHORITATIVE_UNCERTAINTY;
        Some(DynamicsForecast {
            mean_delta: array_to_delta(prediction.mean).clamped(-1.0, 1.0),
            member_deltas: prediction.members.map(array_to_delta),
            uncertainty,
            effective_evidence: evidence,
            validation_samples: self.validation_samples,
            validation_mae: self.validation_mae_ema,
            validation_coverage: self.validation_coverage_ema,
            scope,
            ranking_eligible,
            authoritative,
        })
    }

    fn predict_arrays(&self, features: &[f64; FEATURE_DIM]) -> EnsembleArrays {
        let members = self
            .members
            .each_ref()
            .map(|member| member.predict(features));
        let mean = std::array::from_fn(|output| {
            members.iter().map(|member| member[output]).sum::<f64>() / ENSEMBLE_SIZE as f64
        });
        let variance = std::array::from_fn(|output| {
            members
                .iter()
                .map(|member| {
                    let residual = member[output] - mean[output];
                    residual * residual
                })
                .sum::<f64>()
                / ENSEMBLE_SIZE as f64
        });
        EnsembleArrays {
            mean,
            variance,
            members,
        }
    }

    fn sanitize(&mut self) -> bool {
        if !self.evidence_mass.is_finite()
            || !self.quality_ema.is_finite()
            || !self.validation_mae_ema.is_finite()
            || !self.validation_coverage_ema.is_finite()
            || !self.mean_horizon_cycles.is_finite()
            || self
                .residual_variance_ema
                .iter()
                .any(|value| !value.is_finite())
            || self.members.iter_mut().any(|member| !member.sanitize())
        {
            return false;
        }
        self.evidence_mass = self.evidence_mass.clamp(0.0, EVIDENCE_CAP);
        self.quality_ema = self.quality_ema.clamp(0.0, 1.0);
        self.validation_mae_ema = self.validation_mae_ema.clamp(0.0, 1.0);
        self.validation_coverage_ema = self.validation_coverage_ema.clamp(0.0, 1.0);
        self.mean_horizon_cycles = self.mean_horizon_cycles.clamp(1.0, 600.0);
        for variance in &mut self.residual_variance_ema {
            *variance = variance.clamp(0.0, 1.0);
        }
        true
    }
}

struct EnsembleArrays {
    mean: [f64; STATE_DIM],
    variance: [f64; STATE_DIM],
    members: [[f64; STATE_DIM]; ENSEMBLE_SIZE],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CausalDynamicsModel {
    schema_version: u32,
    installation_id: InstallationId,
    action_models: BTreeMap<String, DynamicsRegressor>,
    baseline_models: BTreeMap<String, DynamicsRegressor>,
    revision: u64,
    gold_action_updates: u64,
    no_action_updates: u64,
}

impl Default for CausalDynamicsModel {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            installation_id: InstallationId::UNKNOWN,
            action_models: BTreeMap::new(),
            baseline_models: BTreeMap::new(),
            revision: 0,
            gold_action_updates: 0,
            no_action_updates: 0,
        }
    }
}

impl CausalDynamicsModel {
    pub fn new(installation_id: InstallationId) -> Self {
        Self {
            installation_id,
            ..Self::default()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn observe_action(
        &mut self,
        action_key: &str,
        family: &str,
        workload: &str,
        before: &TelemetryContextSummary,
        target: WorldStateDelta,
        horizon_cycles: u64,
        quality: f64,
        timestamp_unix: i64,
        hardware_regime: HardwareRegime,
        installation_id: InstallationId,
        sample_id: u64,
    ) {
        if !observation_valid(
            before,
            target,
            quality,
            timestamp_unix,
            hardware_regime,
            installation_id,
        ) || action_key.chars().count() > 320
            || workload.chars().count() > 64
            || family.chars().count() > 64
        {
            return;
        }
        self.ensure_origin(installation_id);
        let features = encode(before, DynamicsState::from_context(before));
        let keys = [
            action_key.to_string(),
            format!("{workload}|{action_key}"),
            format!("{family}:*"),
        ];
        for key in keys {
            evict_if_full(&mut self.action_models, MAX_ACTION_MODELS, &key);
            self.action_models.entry(key).or_default().observe(
                &features,
                target,
                horizon_cycles,
                quality,
                timestamp_unix,
                hardware_regime,
                installation_id,
                sample_id,
            );
        }
        self.gold_action_updates = self.gold_action_updates.saturating_add(1);
        self.revision = self.revision.wrapping_add(1);
    }

    pub fn observe_no_action(
        &mut self,
        before: &TelemetryContextSummary,
        after: &TelemetryContextSummary,
        quality: f64,
        installation_id: InstallationId,
    ) {
        let cycle_gap = after.cycle.saturating_sub(before.cycle);
        if cycle_gap == 0
            || cycle_gap > 120
            || before.workload != after.workload
            || after.timestamp_unix < before.timestamp_unix
            || after.timestamp_unix.saturating_sub(before.timestamp_unix) > 120
        {
            return;
        }
        let hardware = HardwareRegime::from_context(after);
        let target = WorldStateDelta::between(before, after)
            .scaled(1.0 / cycle_gap as f64)
            .clamped(-0.05, 0.05);
        if !observation_valid(
            before,
            target,
            quality,
            after.timestamp_unix,
            hardware,
            installation_id,
        ) {
            return;
        }
        self.ensure_origin(installation_id);
        let features = encode(before, DynamicsState::from_context(before));
        for key in ["any".to_string(), after.workload.clone()] {
            evict_if_full(&mut self.baseline_models, MAX_BASELINE_MODELS, &key);
            self.baseline_models.entry(key).or_default().observe(
                &features,
                target,
                1,
                quality,
                after.timestamp_unix,
                hardware,
                installation_id,
                after.cycle,
            );
        }
        self.no_action_updates = self.no_action_updates.saturating_add(1);
        self.revision = self.revision.wrapping_add(1);
    }

    pub fn predict_action(
        &self,
        action_key: &str,
        workload: &str,
        context: &TelemetryContextSummary,
        installation_id: InstallationId,
    ) -> Option<DynamicsForecast> {
        self.predict_action_from_state(
            action_key,
            workload,
            context,
            DynamicsState::from_context(context),
            installation_id,
        )
    }

    pub(crate) fn predict_action_from_state(
        &self,
        action_key: &str,
        workload: &str,
        context: &TelemetryContextSummary,
        state: DynamicsState,
        installation_id: InstallationId,
    ) -> Option<DynamicsForecast> {
        if self.installation_id != installation_id || !installation_id.is_known() {
            return None;
        }
        let features = encode(context, state);
        let workload_key = format!("{workload}|{action_key}");
        let family_key = action_key
            .split_once(':')
            .map(|(family, _)| format!("{family}:*"));
        for (key, scope) in [
            (Some(workload_key.as_str()), DynamicsEvidenceScope::Workload),
            (Some(action_key), DynamicsEvidenceScope::Action),
            (family_key.as_deref(), DynamicsEvidenceScope::Family),
        ] {
            let Some(model) = key.and_then(|key| self.action_models.get(key)) else {
                continue;
            };
            if let Some(forecast) = model.forecast(&features, context, installation_id, scope) {
                return Some(forecast);
            }
        }
        None
    }

    pub(crate) fn predict_baseline_from_state(
        &self,
        workload: &str,
        context: &TelemetryContextSummary,
        state: DynamicsState,
        installation_id: InstallationId,
    ) -> Option<DynamicsForecast> {
        if self.installation_id != installation_id || !installation_id.is_known() {
            return None;
        }
        let features = encode(context, state);
        for key in [workload, "any"] {
            let Some(model) = self.baseline_models.get(key) else {
                continue;
            };
            if let Some(forecast) = model.forecast(
                &features,
                context,
                installation_id,
                DynamicsEvidenceScope::Workload,
            ) {
                return Some(forecast);
            }
        }
        None
    }

    pub fn publication_revision(&self) -> u64 {
        self.revision.saturating_add(PUBLICATION_CADENCE - 1) / PUBLICATION_CADENCE
    }

    pub fn metrics(&self, context: Option<&TelemetryContextSummary>) -> CausalDynamicsMetrics {
        let mut metrics = CausalDynamicsMetrics {
            action_models: self.action_models.len() as u64,
            baseline_models: self.baseline_models.len() as u64,
            gold_action_updates: self.gold_action_updates,
            no_action_updates: self.no_action_updates,
            publication_revision: self.publication_revision(),
            ..CausalDynamicsMetrics::default()
        };
        let Some(context) = context else {
            metrics.phase = if self.action_models.is_empty() {
                DynamicsAuthorityPhase::Protected
            } else {
                DynamicsAuthorityPhase::Calibrating
            };
            return metrics;
        };
        let mut weighted_mae = 0.0;
        let mut weighted_coverage = 0.0;
        for (key, model) in &self.action_models {
            metrics.validation_samples = metrics
                .validation_samples
                .saturating_add(model.validation_samples as u64);
            weighted_mae += model.validation_mae_ema * model.validation_samples as f64;
            weighted_coverage += model.validation_coverage_ema * model.validation_samples as f64;
            if model.ready_for(context, self.installation_id) {
                metrics.ready_models = metrics.ready_models.saturating_add(1);
                let features = encode(context, DynamicsState::from_context(context));
                let scope = if key.ends_with(":*") {
                    DynamicsEvidenceScope::Family
                } else if key.contains('|') {
                    DynamicsEvidenceScope::Workload
                } else {
                    DynamicsEvidenceScope::Action
                };
                if let Some(forecast) =
                    model.forecast(&features, context, self.installation_id, scope)
                {
                    if forecast.ranking_eligible {
                        metrics.ranking_eligible_models =
                            metrics.ranking_eligible_models.saturating_add(1);
                    }
                    if forecast.authoritative {
                        metrics.authoritative_models =
                            metrics.authoritative_models.saturating_add(1);
                    }
                }
            }
        }
        metrics.baseline_ready_models = self
            .baseline_models
            .values()
            .filter(|model| model.ready_for(context, self.installation_id))
            .count() as u64;
        if metrics.validation_samples > 0 {
            metrics.validation_mae = weighted_mae / metrics.validation_samples as f64;
            metrics.validation_coverage = weighted_coverage / metrics.validation_samples as f64;
        }
        metrics.phase = if self.action_models.is_empty() {
            DynamicsAuthorityPhase::Protected
        } else if metrics.ready_models == 0 {
            DynamicsAuthorityPhase::Calibrating
        } else if metrics.authoritative_models == 0 {
            DynamicsAuthorityPhase::Shadow
        } else {
            DynamicsAuthorityPhase::Trusted
        };
        metrics
    }

    pub fn sanitized_for_restore(
        mut self,
        installation_id: InstallationId,
        same_origin: bool,
    ) -> Self {
        if !same_origin
            || self.schema_version != SCHEMA_VERSION
            || !installation_id.is_known()
            || self.installation_id != installation_id
        {
            return Self::new(installation_id);
        }
        self.action_models.retain(|key, model| {
            key.len() <= MAX_MODEL_KEY_BYTES
                && model.installation_id == installation_id
                && model.sanitize()
        });
        self.baseline_models.retain(|key, model| {
            key.len() <= MAX_WORKLOAD_KEY_BYTES
                && model.installation_id == installation_id
                && model.sanitize()
        });
        while self.action_models.len() > MAX_ACTION_MODELS {
            evict_weakest(&mut self.action_models);
        }
        while self.baseline_models.len() > MAX_BASELINE_MODELS {
            evict_weakest(&mut self.baseline_models);
        }
        self.schema_version = SCHEMA_VERSION;
        self.installation_id = installation_id;
        self
    }

    fn ensure_origin(&mut self, installation_id: InstallationId) {
        if self.installation_id != installation_id {
            *self = Self::new(installation_id);
        }
    }
}

fn encode(context: &TelemetryContextSummary, state: DynamicsState) -> [f64; FEATURE_DIM] {
    let [pressure, fluidity, energy, cpu, thermal, thrashing, stall] = state.values;
    [
        1.0,
        pressure * 2.0 - 1.0,
        fluidity * 2.0 - 1.0,
        energy * 2.0 - 1.0,
        cpu * 2.0 - 1.0,
        thermal * 2.0 - 1.0,
        thrashing * 2.0 - 1.0,
        stall * 2.0 - 1.0,
        (pressure * thrashing) * 2.0 - 1.0,
        (cpu * thermal) * 2.0 - 1.0,
        (context.signal_pressure_velocity * 5.0).clamp(-1.0, 1.0),
        if context.app_launching || context.window_op_active {
            1.0
        } else if context.foreground_idle {
            -1.0
        } else {
            0.0
        },
    ]
}

fn observation_valid(
    before: &TelemetryContextSummary,
    target: WorldStateDelta,
    quality: f64,
    timestamp_unix: i64,
    hardware_regime: HardwareRegime,
    installation_id: InstallationId,
) -> bool {
    target.is_finite()
        && quality.is_finite()
        && quality >= MIN_PREDICTION_QUALITY
        && timestamp_unix >= before.timestamp_unix
        && installation_id.is_known()
        && hardware_regime.is_known()
        && hardware_regime.matches_context(before)
}

fn evict_if_full(
    models: &mut BTreeMap<String, DynamicsRegressor>,
    capacity: usize,
    incoming: &str,
) {
    if !models.contains_key(incoming) && models.len() >= capacity {
        evict_weakest(models);
    }
}

fn evict_weakest(models: &mut BTreeMap<String, DynamicsRegressor>) {
    if let Some(key) = models
        .iter()
        .min_by_key(|(_, model)| (model.last_observed_unix, model.observations))
        .map(|(key, _)| key.clone())
    {
        models.remove(&key);
    }
}

fn bootstrap_weight(sample_id: u64, member: u64) -> f64 {
    let mut value = sample_id.wrapping_add(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(member + 1));
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    0.70 + (value % 61) as f64 / 100.0
}

fn delta_to_array(delta: WorldStateDelta) -> [f64; STATE_DIM] {
    [
        delta.pressure,
        delta.fluidity,
        delta.energy,
        delta.cpu,
        delta.thermal,
        delta.thrashing,
        delta.stall,
    ]
}

fn array_to_delta(values: [f64; STATE_DIM]) -> WorldStateDelta {
    WorldStateDelta {
        pressure: values[0],
        fluidity: values[1],
        energy: values[2],
        cpu: values[3],
        thermal: values[4],
        thrashing: values[5],
        stall: values[6],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCAL_ID: InstallationId = InstallationId(7);
    const FOREIGN_ID: InstallationId = InstallationId(8);

    fn context(cycle: u64, pressure: f64, workload: &str) -> TelemetryContextSummary {
        TelemetryContextSummary {
            cycle,
            timestamp_unix: 1_800_000_000 + cycle as i64,
            workload: workload.to_string(),
            memory_pressure: pressure,
            fluidity_score: 0.80 - pressure * 0.10,
            package_watts: Some(8.0 + pressure * 4.0),
            cpu_max_busy: 0.30 + pressure * 0.20,
            thermal_score: 0.20,
            thrashing_score: pressure * 1_000.0,
            stall_fraction: pressure * 0.02,
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            p_core_count: 4,
            e_core_count: 6,
            collector_pressure_alive: true,
            reactor_healthy: true,
            daemon_is_root: true,
            ..TelemetryContextSummary::default()
        }
    }

    fn target(pressure: f64, fluidity: f64) -> WorldStateDelta {
        WorldStateDelta {
            pressure,
            fluidity,
            cpu: pressure * 0.4,
            ..WorldStateDelta::default()
        }
    }

    #[test]
    fn learns_state_conditioned_gold_transition_prequentially() {
        let mut model = CausalDynamicsModel::new(LOCAL_ID);
        for sample in 0..48 {
            let pressure = 0.20 + (sample % 12) as f64 * 0.04;
            let before = context(sample, pressure, "coding");
            model.observe_action(
                "boost:Editor",
                "boost",
                "coding",
                &before,
                target(-0.02 - pressure * 0.05, 0.03 + pressure * 0.02),
                4,
                0.98,
                before.timestamp_unix + 4,
                HardwareRegime::from_context(&before),
                LOCAL_ID,
                sample,
            );
        }
        let now = context(60, 0.60, "coding");
        let forecast = model
            .predict_action("boost:Editor", "coding", &now, LOCAL_ID)
            .expect("mature local forecast");
        assert!(forecast.mean_delta.pressure < -0.025);
        assert!(forecast.mean_delta.fluidity > 0.015);
        assert!(forecast.validation_mae < 0.08);
        assert!(forecast.authoritative);
        let metrics = model.metrics(Some(&now));
        assert_eq!(metrics.ranking_eligible_models, 3);
        assert_eq!(metrics.authoritative_models, 2);
    }

    #[test]
    fn foreign_installation_never_gets_a_forecast() {
        let mut model = CausalDynamicsModel::new(LOCAL_ID);
        for sample in 0..8 {
            let before = context(sample, 0.4, "coding");
            model.observe_action(
                "boost:Editor",
                "boost",
                "coding",
                &before,
                target(-0.03, 0.02),
                4,
                1.0,
                before.timestamp_unix + 4,
                HardwareRegime::from_context(&before),
                LOCAL_ID,
                sample,
            );
        }
        let now = context(12, 0.4, "coding");
        assert!(model
            .predict_action("boost:Editor", "coding", &now, FOREIGN_ID)
            .is_none());
        let restored = model.sanitized_for_restore(FOREIGN_ID, false);
        assert_eq!(restored.metrics(Some(&now)).action_models, 0);
    }

    #[test]
    fn baseline_learns_only_contiguous_same_workload_windows() {
        let mut model = CausalDynamicsModel::new(LOCAL_ID);
        for cycle in 1..24 {
            let before = context(cycle, 0.40, "coding");
            let after = context(cycle + 1, 0.39, "coding");
            model.observe_no_action(&before, &after, 0.98, LOCAL_ID);
        }
        let now = context(30, 0.50, "coding");
        let forecast = model
            .predict_baseline_from_state(
                "coding",
                &now,
                DynamicsState::from_context(&now),
                LOCAL_ID,
            )
            .expect("baseline forecast");
        assert!(forecast.mean_delta.pressure < 0.0);

        let before = context(31, 0.40, "coding");
        let after = context(32, 0.90, "gaming");
        let updates = model.no_action_updates;
        model.observe_no_action(&before, &after, 1.0, LOCAL_ID);
        assert_eq!(model.no_action_updates, updates);
    }

    #[test]
    fn immature_model_remains_shadow_only() {
        let mut model = CausalDynamicsModel::new(LOCAL_ID);
        for sample in 0..6 {
            let before = context(sample, 0.4, "coding");
            model.observe_action(
                "boost:Editor",
                "boost",
                "coding",
                &before,
                target(-0.03, 0.02),
                4,
                1.0,
                before.timestamp_unix + 4,
                HardwareRegime::from_context(&before),
                LOCAL_ID,
                sample,
            );
        }
        let now = context(10, 0.4, "coding");
        let forecast = model
            .predict_action("boost:Editor", "coding", &now, LOCAL_ID)
            .expect("exploratory forecast");
        assert!(!forecast.authoritative);
        let metrics = model.metrics(Some(&now));
        assert_eq!(metrics.phase, DynamicsAuthorityPhase::Shadow);
        assert_eq!(metrics.ranking_eligible_models, 0);
        assert_eq!(metrics.authoritative_models, 0);
    }

    #[test]
    fn noisy_model_cannot_earn_authority_from_sample_count_alone() {
        let mut model = CausalDynamicsModel::new(LOCAL_ID);
        for sample in 0..80 {
            let before = context(sample, 0.4, "coding");
            let direction = if sample % 2 == 0 { 0.35 } else { -0.35 };
            model.observe_action(
                "boost:Editor",
                "boost",
                "coding",
                &before,
                target(direction, -direction),
                4,
                1.0,
                before.timestamp_unix + 4,
                HardwareRegime::from_context(&before),
                LOCAL_ID,
                sample,
            );
        }
        let now = context(100, 0.4, "coding");
        let forecast = model
            .predict_action("boost:Editor", "coding", &now, LOCAL_ID)
            .expect("mature but noisy forecast");
        assert!(forecast.effective_evidence >= MIN_AUTHORITATIVE_EVIDENCE);
        assert!(forecast.validation_mae > MAX_AUTHORITATIVE_MAE);
        assert!(!forecast.authoritative);
    }

    #[test]
    fn hardware_regime_change_resets_action_authority() {
        let mut model = CausalDynamicsModel::new(LOCAL_ID);
        for sample in 0..24 {
            let before = context(sample, 0.4, "coding");
            model.observe_action(
                "boost:Editor",
                "boost",
                "coding",
                &before,
                target(-0.03, 0.02),
                4,
                1.0,
                before.timestamp_unix + 4,
                HardwareRegime::from_context(&before),
                LOCAL_ID,
                sample,
            );
        }
        let mut changed = context(30, 0.4, "coding");
        changed.p_core_count = 2;
        changed.e_core_count = 6;
        model.observe_action(
            "boost:Editor",
            "boost",
            "coding",
            &changed,
            target(-0.01, 0.01),
            4,
            1.0,
            changed.timestamp_unix + 4,
            HardwareRegime::from_context(&changed),
            LOCAL_ID,
            99,
        );
        assert!(model
            .predict_action("boost:Editor", "coding", &changed, LOCAL_ID)
            .is_none());
    }

    #[test]
    fn corrupt_or_oversized_persisted_models_are_sanitized() {
        let mut model = CausalDynamicsModel::new(LOCAL_ID);
        for sample in 0..(MAX_ACTION_MODELS as u64 + 20) {
            let before = context(sample, 0.4, "coding");
            model.observe_action(
                &format!("boost:p{sample}"),
                "boost",
                "coding",
                &before,
                target(-0.01, 0.01),
                4,
                1.0,
                before.timestamp_unix + 4,
                HardwareRegime::from_context(&before),
                LOCAL_ID,
                sample,
            );
        }
        assert!(model.action_models.len() <= MAX_ACTION_MODELS);
        let persisted_bytes = serde_json::to_vec(&model).expect("serialize bounded model");
        assert!(persisted_bytes.len() < 8 * 1024 * 1024);
        if let Some(regressor) = model.action_models.values_mut().next() {
            regressor.members[0].weights[0][0] = f64::NAN;
        }
        let restored = model.sanitized_for_restore(LOCAL_ID, true);
        assert!(restored.action_models.len() < MAX_ACTION_MODELS);
        assert!(restored
            .action_models
            .values()
            .flat_map(|regressor| regressor.members.iter())
            .flat_map(|member| member.weights.iter().flatten())
            .all(|weight| weight.is_finite()));
    }

    #[test]
    fn contextual_keys_survive_persistence_at_the_documented_bounds() {
        let mut model = CausalDynamicsModel::new(LOCAL_ID);
        let action = "a".repeat(320);
        let workload = "w".repeat(64);
        for sample in 0..4 {
            let before = context(sample, 0.4, &workload);
            model.observe_action(
                &action,
                "boost",
                &workload,
                &before,
                target(-0.01, 0.01),
                4,
                1.0,
                before.timestamp_unix + 4,
                HardwareRegime::from_context(&before),
                LOCAL_ID,
                sample,
            );
        }
        let contextual_key = format!("{workload}|{action}");
        assert!(contextual_key.len() > 320);
        let restored = model.sanitized_for_restore(LOCAL_ID, true);
        assert!(restored.action_models.contains_key(&contextual_key));
    }

    #[test]
    fn ten_thousand_ensemble_queries_remain_bounded() {
        let mut model = CausalDynamicsModel::new(LOCAL_ID);
        for sample in 0..40 {
            let before = context(sample, 0.4, "coding");
            model.observe_action(
                "boost:Editor",
                "boost",
                "coding",
                &before,
                target(-0.03, 0.02),
                4,
                1.0,
                before.timestamp_unix + 4,
                HardwareRegime::from_context(&before),
                LOCAL_ID,
                sample,
            );
        }
        let now = context(100, 0.4, "coding");
        let started = std::time::Instant::now();
        let mut checksum = 0.0;
        for _ in 0..10_000 {
            checksum += model
                .predict_action("boost:Editor", "coding", &now, LOCAL_ID)
                .expect("forecast")
                .mean_delta
                .fluidity;
        }
        let elapsed = started.elapsed();
        eprintln!("causal_dynamics_10k: {elapsed:?}");
        assert!(checksum.is_finite());
        assert!(elapsed < std::time::Duration::from_secs(2));
    }

    #[test]
    fn publication_revision_is_bounded() {
        let mut model = CausalDynamicsModel::new(LOCAL_ID);
        for sample in 0..17 {
            let before = context(sample, 0.4, "coding");
            model.observe_action(
                "boost:Editor",
                "boost",
                "coding",
                &before,
                target(-0.03, 0.02),
                4,
                1.0,
                before.timestamp_unix + 4,
                HardwareRegime::from_context(&before),
                LOCAL_ID,
                sample,
            );
        }
        assert_eq!(model.publication_revision(), 3);
    }
}
