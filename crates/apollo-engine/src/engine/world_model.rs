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
//! Deliberately one-step (predict Δpressure over the causal-evaluation
//! horizon, compare against the no-action drift). Multi-step rollouts /
//! hierarchical planning [LeCun 2022 §4.3] stay future work; the
//! dominance check alone closes the act-blind gap.
//!
//! ## References
//! - [LeCun 2022] "A Path Towards Autonomous Machine Intelligence" §4.2 —
//!   world-model-predictive action selection (MPC over learned model).
//! - [Sutton 1991] Dyna — planning as acting through learned model.
//! - [Rubin 1974] Potential Outcomes — the do-nothing counterfactual is
//!   the control arm every action must beat.
//! - [Camacho 2007] MPC — act only when the predicted trajectory under
//!   action improves on the free response.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::engine::causal_graph::CausalGraph;
use crate::engine::installation_identity::InstallationId;
use crate::engine::outcome_tracker::OutcomeTracker;
use crate::engine::telemetry_medallion::{
    ActionModelStats, TelemetryContextSummary, TrustedTelemetryView,
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
    context_quality: f64,
    latest_context: Option<TelemetryContextSummary>,
    current_installation_id: InstallationId,
    authority_phase: ModelAuthorityPhase,
    /// Action key -> (counterfactual-adjusted utility EMA, data quality,
    /// Gold observations). Populated from the universal actuator medallion.
    utility_predicted: HashMap<String, ActionModelStats>,
    causal_revision: Option<u64>,
    causal_workload: String,
    causal_debias_bits: u32,
    utility_revision: Option<u64>,
    causal_refreshes: u64,
    causal_cache_hits: u64,
    utility_refreshes: u64,
    utility_cache_hits: u64,
}

impl WorldModel {
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
        self.context_bronze = view.metrics.bronze_total;
        self.context_silver = view.metrics.silver_total;
        self.context_gold = view.metrics.gold_total;
        self.context_quality = view.metrics.mean_quality;
        self.latest_context = view.current.cloned();
        self.current_installation_id = view.installation_id;
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
        let Some(now_unix) = self.latest_context.as_ref().map(|ctx| ctx.timestamp_unix) else {
            return None;
        };
        let workload_key = format!("{workload}|{action_key}");
        // Prefer same-workload evidence only when it is mature. An immature
        // contextual bucket must not hide a mature aggregate model.
        let selected = [
            (
                UtilityEvidenceScope::Workload,
                self.utility_predicted.get(&workload_key),
            ),
            (
                UtilityEvidenceScope::Aggregate,
                self.utility_predicted.get(action_key),
            ),
        ]
        .into_iter()
        .filter_map(|(scope, stats)| stats.map(|stats| (scope, stats)))
        .find(|(_, stats)| {
            utility_model_ready(
                stats,
                now_unix,
                self.latest_context.as_ref(),
                self.current_installation_id,
            )
        });
        let Some((scope, stats)) = selected else {
            return None;
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
        Some(UtilityAssessment {
            verdict,
            scope,
            utility_ema: stats.utility_ema,
            lower_bound: lower,
            upper_bound: upper,
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
    let base_ready = stats.quality_ema >= MIN_UTILITY_DATA_QUALITY
        && stats.last_observed_unix > 0
        && now_unix >= stats.last_observed_unix
        && now_unix - stats.last_observed_unix <= UTILITY_MAX_AGE_SECS
        && stats.effective_evidence_at(now_unix) >= MIN_UTILITY_EVIDENCE
        && context.is_some_and(|context| stats.hardware_regime.matches_context(context))
        && installation_id.is_known()
        && stats.installation_id == installation_id;
    if !base_ready {
        return false;
    }
    let evidence = stats.effective_evidence_at(now_unix);
    let standard_error = (stats.utility_variance_ema.max(UTILITY_VARIANCE_FLOOR) / evidence).sqrt();
    let lower = stats.utility_ema - UTILITY_CONFIDENCE_Z * standard_error;
    let upper = stats.utility_ema + UTILITY_CONFIDENCE_Z * standard_error;
    lower > DOMINANCE_MARGIN || upper <= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::installation_identity::InstallationId;
    use crate::engine::telemetry_medallion::{
        HardwareRegime, TelemetryMedallion, TelemetryMedallionMetrics, TrustedTelemetryView,
    };
    use chrono::Utc;
    use std::collections::BTreeMap;
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

    fn mature_model(now_unix: i64, installation_id: InstallationId) -> ActionModelStats {
        ActionModelStats {
            observations: 20,
            effective_observations: 18,
            utility_ema: 0.08,
            evidence_mass: 20.0,
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
            metrics: TelemetryMedallionMetrics {
                bronze_total: local_gold_total,
                gold_total: local_gold_total,
                local_gold_total,
                ..TelemetryMedallionMetrics::default()
            },
        });
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
