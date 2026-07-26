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

use crate::engine::causal_graph::CausalGraph;
use crate::engine::outcome_tracker::OutcomeTracker;

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
        let debias = if prediction_debias.is_finite() && prediction_debias > 0.0 {
            prediction_debias as f64
        } else {
            1.0
        };
        let mut predicted = HashMap::new();
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
            predicted.insert(
                action_key.to_string(),
                (
                    selected.avg_pressure_drop as f64 * debias,
                    selected.data_quality,
                    selected.observations,
                ),
            );
        }
        Self {
            predicted,
            natural_drift: tracker.natural_drift(),
            curated_observations,
            contextual_actions,
            mean_data_quality: if curated_observations == 0 {
                0.0
            } else {
                (quality_weighted_sum / curated_observations as f64).clamp(0.0, 1.0)
            },
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

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
