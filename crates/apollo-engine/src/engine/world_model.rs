//! World Model — Mode-2 imagination before acting.
//!
//! 2026-06-11. Apollo already owned every piece of a LeCun-style world
//! model — CausalGraph edges carry a learned per-action pressure-delta
//! prediction (`avg_delta`), OutcomeTracker carries the Rubin do-nothing
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

/// Minimum edge confidence for a veto-grade prediction.
const MIN_CONFIDENCE: f32 = 0.30;

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
    /// `"throttle:Name"` / `"freeze:Name"` → (avg pressure delta when the
    /// edge fired, confidence, evidence count).
    predicted: HashMap<String, (f64, f32, u32)>,
    /// Rubin counterfactual: EMA of pressure drift on no-action windows.
    /// Positive = pressure tends to drop by itself.
    pub natural_drift: f64,
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
        let debias = if prediction_debias.is_finite() && prediction_debias > 0.0 {
            prediction_debias as f64
        } else {
            1.0
        };
        let mut predicted = HashMap::new();
        // ALL pressure-drop edges, not just is_solid() ones (2026-06-12 fix):
        // solid_edges() pre-filters at confidence > 0.7, which silently
        // reduced the model to the rare super-solid edges and made
        // imagine()'s own MIN_CONFIDENCE = 0.30 gate dead code. Prod had 5
        // imaginable actions; the model saw 1. imagine() applies the real
        // admission thresholds.
        for edge in causal.pressure_drop_edges() {
            predicted.insert(
                edge.cause.clone(),
                (
                    edge.avg_delta as f64 * debias,
                    edge.confidence,
                    edge.evidence_count,
                ),
            );
        }
        Self {
            predicted,
            natural_drift: tracker.natural_drift(),
        }
    }

    /// Mode-2 step: imagine the action through the learned model and
    /// compare against the do-nothing counterfactual.
    pub fn imagine(&self, action_key: &str) -> Imagined {
        let Some(&(avg_delta, confidence, evidence)) = self.predicted.get(action_key) else {
            return Imagined::Unknown;
        };
        if evidence < MIN_EVIDENCE || confidence < MIN_CONFIDENCE {
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
            .map(|(avg_delta, _conf, _ev)| (avg_delta - baseline).max(0.0))
            .fold(0.0_f64, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_with(key: &str, delta: f64, conf: f32, evidence: u32, drift: f64) -> WorldModel {
        let mut predicted = HashMap::new();
        predicted.insert(key.to_string(), (delta, conf, evidence));
        WorldModel {
            predicted,
            natural_drift: drift,
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

        let unsure = model_with("freeze:App", 0.08, MIN_CONFIDENCE - 0.05, 50, 0.0);
        assert_eq!(unsure.imagine("freeze:App"), Imagined::Unknown);
    }

    #[test]
    fn act_wins_when_predicted_drop_beats_drift() {
        // Model predicts 6% drop; system drifts down only 1% alone.
        let m = model_with("freeze:Heavy", 0.06, 0.8, 40, 0.01);
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
        let m = model_with("freeze:Futile", 0.004, 0.9, 60, 0.01);
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
        let m = model_with("throttle:Hog", 0.02, 0.6, 20, -0.03);
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
        let raw = model_with("freeze:Inflated", 0.02, 0.8, 30, 0.012);
        assert!(matches!(
            raw.imagine("freeze:Inflated"),
            Imagined::ActWins { .. }
        ));

        let calibrated = model_with("freeze:Inflated", 0.02 * 0.25, 0.8, 30, 0.012);
        assert!(matches!(
            calibrated.imagine("freeze:Inflated"),
            Imagined::DoNothingDominates { .. }
        ));
    }

    #[test]
    fn from_parts_includes_sub_solid_edges_for_imagination() {
        // Prod 2026-06-12 finding: solid_edges() pre-filtered at conf>0.7,
        // starving the model to 1 of 5 imaginable actions and making the
        // MIN_CONFIDENCE=0.30 gate dead code. Pin the fix: a conf-0.39
        // edge with 338 obs (the live freeze:Hermes case) MUST enter the
        // model and be judged by imagine()'s own gates, while a conf-0.20
        // edge still abstains.
        let mut g = crate::engine::causal_graph::CausalGraph::new();
        let mut hermes =
            crate::engine::causal_graph::CausalEdge::new("freeze:Hermes", "pressure_drop");
        hermes.confidence = 0.39;
        hermes.evidence_count = 338;
        hermes.avg_delta = 0.0232;
        let mut weak = crate::engine::causal_graph::CausalEdge::new("freeze:Weak", "pressure_drop");
        weak.confidence = 0.20;
        weak.evidence_count = 50;
        weak.avg_delta = 0.09;
        g.restore(vec![
            (
                ("freeze:Hermes".to_string(), "pressure_drop".to_string()),
                hermes,
            ),
            (
                ("freeze:Weak".to_string(), "pressure_drop".to_string()),
                weak,
            ),
        ]);

        let tracker = crate::engine::outcome_tracker::OutcomeTracker::new();
        let m = WorldModel::from_parts(&g, &tracker, 1.0);
        assert_eq!(m.known_actions(), 2, "both pressure_drop edges enter");
        assert!(
            !matches!(m.imagine("freeze:Hermes"), Imagined::Unknown),
            "conf 0.39 >= 0.30 gate must be judged, not starved"
        );
        assert_eq!(
            m.imagine("freeze:Weak"),
            Imagined::Unknown,
            "conf 0.20 < 0.30 still abstains via imagine()'s own gate"
        );
    }
}
