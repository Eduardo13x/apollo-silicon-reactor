//! Self-Rewarding Evaluator — Apollo judges its own past decisions.
//!
//! ## Problem solved
//! Apollo only generates training signal on OOM events (sparse reward).
//! Between OOM events, subsystems receive no learning signal about whether
//! their decisions were good or bad.
//!
//! ## DR-ZERO insight [Yuan 2024]
//! A model can generate its own training signal without an external oracle.
//! Meta AI's Self-Rewarding Language Models use the model itself as judge.
//!
//! ## Apollo adaptation
//! Instead of an LLM judge, Apollo uses its CausalGraph as internal judge:
//! - Log every decision with predicted outcome
//! - N cycles later, evaluate using causal evidence
//! - JuicyScore = causal_confidence × pressure_improvement / cycles_to_effect
//! - Feed (predicted - actual) back to CognitiveRewardBus
//!
//! ## References
//! - [Yuan 2024] "Self-Rewarding Language Models" arXiv:2401.10020 §3
//! - [Pearl 2009] "Causality" — using causal graph as oracle

use std::collections::VecDeque;

use super::neon_ema::ema_f32;

use serde::{Deserialize, Serialize};

/// Number of cycles to wait before evaluating a past decision.
const EVAL_DELAY_CYCLES: u64 = 10;

/// Maximum decisions in the log.
const MAX_DECISION_LOG: usize = 50;

/// EMA alpha for self-evaluation accuracy tracking.
const SELF_EVAL_ALPHA: f32 = 0.05;

/// Minimum causal confidence to count as "informative evaluation".
const MIN_CAUSAL_CONFIDENCE: f32 = 0.10;

// ── Types ──────────────────────────────────────────────────────────────────────

/// A logged decision awaiting retroactive evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// Daemon cycle when the decision was made.
    pub cycle: u64,
    /// Action that was taken (e.g., "throttle:Firefox", "freeze:Slack Helper (Renderer)").
    pub action: String,
    /// Predicted outcome quality [0,1] at decision time (LinUCB confidence or similar).
    pub predicted_score: f32,
    /// Memory pressure at the time of decision.
    pub pressure_at_decision: f64,
    /// Retroactive evaluation score (filled in after EVAL_DELAY_CYCLES).
    pub actual_score: Option<f32>,
    /// Whether this decision has been evaluated.
    pub evaluated: bool,
}

/// Result of evaluating a past decision.
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// The decision that was evaluated.
    pub cycle: u64,
    pub action: String,
    /// JuicyScore: composite quality metric.
    pub juicy_score: f32,
    /// Prediction error: actual_score - predicted_score.
    pub prediction_error: f32,
    /// Whether causal evidence was strong enough to be informative.
    pub informative: bool,
}

/// Self-Rewarding Evaluator — generates dense training signal
/// by retroactively judging past decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfRewardingEvaluator {
    /// Log of recent decisions awaiting evaluation.
    decision_log: VecDeque<DecisionRecord>,
    /// EMA of JuicyScore — tracks overall decision quality.
    pub reward_ema: f32,
    /// EMA of |predicted - actual| — self-evaluation calibration.
    /// Low = the evaluator itself is well-calibrated.
    pub self_eval_accuracy: f32,
    /// Number of decisions evaluated so far.
    pub eval_count: u64,
    /// Number of informative evaluations (had causal evidence).
    pub informative_count: u64,
}

impl Default for SelfRewardingEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfRewardingEvaluator {
    pub fn new() -> Self {
        Self {
            decision_log: VecDeque::with_capacity(MAX_DECISION_LOG),
            reward_ema: 0.5,
            self_eval_accuracy: 0.0,
            eval_count: 0,
            informative_count: 0,
        }
    }

    /// Log a decision for future retroactive evaluation.
    pub fn log_decision(
        &mut self,
        cycle: u64,
        action: String,
        predicted_score: f32,
        pressure: f64,
    ) {
        if self.decision_log.len() >= MAX_DECISION_LOG {
            self.decision_log.pop_front();
        }
        self.decision_log.push_back(DecisionRecord {
            cycle,
            action,
            predicted_score: predicted_score.clamp(0.0, 1.0),
            pressure_at_decision: pressure,
            actual_score: None,
            evaluated: false,
        });
    }

    /// Evaluate past decisions that are old enough.
    ///
    /// `current_cycle`: the current daemon cycle
    /// `current_pressure`: current memory pressure
    /// `causal_confidence_fn`: closure that returns CausalGraph confidence for an action [0,1]
    ///
    /// Returns list of evaluation results for this tick.
    pub fn evaluate_past<F>(
        &mut self,
        current_cycle: u64,
        current_pressure: f64,
        causal_confidence_fn: F,
    ) -> Vec<EvalResult>
    where
        F: Fn(&str) -> f32,
    {
        let mut results = Vec::new();

        for record in self.decision_log.iter_mut() {
            if record.evaluated {
                continue;
            }
            if current_cycle.saturating_sub(record.cycle) < EVAL_DELAY_CYCLES {
                continue;
            }

            // Compute JuicyScore using CausalGraph as internal judge [Yuan 2024 §3]
            let causal_conf = causal_confidence_fn(&record.action);
            let pressure_improvement =
                (record.pressure_at_decision - current_pressure).max(0.0) as f32;
            let cycles_elapsed = (current_cycle - record.cycle).max(1) as f32;

            // JuicyScore = causal_confidence × pressure_improvement / cycles_to_effect
            // Bounded to [0, 1]
            let juicy =
                (causal_conf * pressure_improvement / (cycles_elapsed * 0.1 + 1.0)).clamp(0.0, 1.0);

            let informative = causal_conf >= MIN_CAUSAL_CONFIDENCE;
            let prediction_error = juicy - record.predicted_score;

            record.actual_score = Some(juicy);
            record.evaluated = true;
            self.eval_count += 1;

            if informative {
                self.informative_count += 1;
                // Update EMAs only with informative evaluations
                self.reward_ema = ema_f32(self.reward_ema, juicy, SELF_EVAL_ALPHA);
                self.self_eval_accuracy = ema_f32(
                    self.self_eval_accuracy,
                    prediction_error.abs(),
                    SELF_EVAL_ALPHA,
                );
            }

            results.push(EvalResult {
                cycle: record.cycle,
                action: record.action.clone(),
                juicy_score: juicy,
                prediction_error,
                informative,
            });
        }

        // Prune very old evaluated records
        while self.decision_log.len() > MAX_DECISION_LOG / 2 {
            if let Some(front) = self.decision_log.front() {
                if front.evaluated {
                    self.decision_log.pop_front();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        results
    }

    /// Is the self-evaluator well-calibrated?
    /// True if |predicted - actual| is consistently < 0.20 (accurate judge).
    pub fn is_well_calibrated(&self) -> bool {
        self.informative_count >= 10 && self.self_eval_accuracy < 0.20
    }

    /// How trustworthy is the evaluator's signal? [0,1]
    /// Higher = more evaluations AND better calibration.
    pub fn evaluator_trust(&self) -> f32 {
        if self.informative_count < 5 {
            return 0.0;
        }
        let calibration_bonus = (1.0 - self.self_eval_accuracy).max(0.0);
        let volume_bonus = ((self.informative_count as f32).ln() / 5.0).min(1.0);
        (calibration_bonus * 0.6 + volume_bonus * 0.4).clamp(0.0, 1.0)
    }

    /// Current mean decision quality (reward_ema).
    pub fn mean_quality(&self) -> f32 {
        self.reward_ema
    }

    /// Pending decisions (not yet evaluated).
    pub fn pending_count(&self) -> usize {
        self.decision_log.iter().filter(|d| !d.evaluated).count()
    }

    /// Total decisions logged.
    pub fn total_logged(&self) -> usize {
        self.decision_log.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_defaults() {
        let se = SelfRewardingEvaluator::new();
        assert_eq!(se.eval_count, 0);
        assert_eq!(se.informative_count, 0);
        assert_eq!(se.pending_count(), 0);
        assert!((se.reward_ema - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_log_decision() {
        let mut se = SelfRewardingEvaluator::new();
        se.log_decision(1, "throttle:Firefox".into(), 0.8, 0.75);
        assert_eq!(se.total_logged(), 1);
        assert_eq!(se.pending_count(), 1);
    }

    #[test]
    fn test_log_decision_overflow() {
        let mut se = SelfRewardingEvaluator::new();
        for i in 0..MAX_DECISION_LOG + 10 {
            se.log_decision(i as u64, format!("action:{i}"), 0.5, 0.5);
        }
        assert!(se.total_logged() <= MAX_DECISION_LOG);
    }

    #[test]
    fn test_evaluate_not_ready_yet() {
        let mut se = SelfRewardingEvaluator::new();
        se.log_decision(1, "throttle:Firefox".into(), 0.8, 0.75);
        // Only 5 cycles later — not ready (needs EVAL_DELAY_CYCLES=10)
        let results = se.evaluate_past(6, 0.60, |_| 0.5);
        assert!(results.is_empty());
        assert_eq!(se.pending_count(), 1);
    }

    #[test]
    fn test_evaluate_after_delay() {
        let mut se = SelfRewardingEvaluator::new();
        se.log_decision(1, "throttle:Firefox".into(), 0.8, 0.75);
        // 15 cycles later, pressure dropped, good causal evidence
        let results = se.evaluate_past(15, 0.55, |_| 0.85);
        assert_eq!(results.len(), 1);
        assert!(results[0].informative);
        assert!(results[0].juicy_score > 0.0);
        assert_eq!(se.eval_count, 1);
        assert_eq!(se.informative_count, 1);
    }

    #[test]
    fn test_evaluate_no_improvement() {
        let mut se = SelfRewardingEvaluator::new();
        se.log_decision(1, "throttle:contactsd".into(), 0.6, 0.75);
        // Pressure didn't drop → JuicyScore ≈ 0
        let results = se.evaluate_past(15, 0.80, |_| 0.70);
        assert_eq!(results.len(), 1);
        assert!(results[0].juicy_score < 0.01, "No improvement → low score");
    }

    #[test]
    fn test_evaluate_low_causal_confidence() {
        let mut se = SelfRewardingEvaluator::new();
        se.log_decision(1, "throttle:noise".into(), 0.5, 0.80);
        // Good pressure drop but no causal evidence
        let results = se.evaluate_past(15, 0.50, |_| 0.05);
        assert_eq!(results.len(), 1);
        assert!(
            !results[0].informative,
            "No causal evidence → not informative"
        );
    }

    #[test]
    fn test_prediction_error_overconfident() {
        let mut se = SelfRewardingEvaluator::new();
        se.log_decision(1, "throttle:X".into(), 0.90, 0.70);
        // Reality: not much happened → low juicy
        let results = se.evaluate_past(15, 0.68, |_| 0.50);
        assert!(
            results[0].prediction_error < 0.0,
            "Predicted high, got low = overconfident"
        );
    }

    #[test]
    fn test_prediction_error_underconfident() {
        let mut se = SelfRewardingEvaluator::new();
        se.log_decision(1, "throttle:Y".into(), 0.10, 0.90);
        // Reality: great improvement
        let results = se.evaluate_past(15, 0.50, |_| 0.90);
        assert!(
            results[0].prediction_error > 0.0,
            "Predicted low, got good = underconfident"
        );
    }

    #[test]
    fn test_evaluator_trust_cold_start() {
        let se = SelfRewardingEvaluator::new();
        assert_eq!(se.evaluator_trust(), 0.0, "No data → no trust");
    }

    #[test]
    fn test_evaluator_trust_grows_with_data() {
        let mut se = SelfRewardingEvaluator::new();
        for i in 0..30 {
            se.log_decision(i, format!("a:{i}"), 0.5, 0.70);
        }
        for i in 30..60 {
            let _ = se.evaluate_past(i, 0.55, |_| 0.60);
        }
        assert!(se.evaluator_trust() > 0.0, "With data → trust > 0");
    }

    #[test]
    fn test_well_calibrated_requires_data() {
        let se = SelfRewardingEvaluator::new();
        assert!(!se.is_well_calibrated(), "Needs ≥10 informative evals");
    }

    #[test]
    fn test_multiple_decisions_batch_evaluate() {
        let mut se = SelfRewardingEvaluator::new();
        se.log_decision(1, "A".into(), 0.5, 0.80);
        se.log_decision(2, "B".into(), 0.6, 0.85);
        se.log_decision(3, "C".into(), 0.7, 0.90);

        let results = se.evaluate_past(20, 0.60, |_| 0.70);
        assert_eq!(results.len(), 3, "All 3 should be evaluated");
    }

    #[test]
    fn test_double_evaluate_is_idempotent() {
        let mut se = SelfRewardingEvaluator::new();
        se.log_decision(1, "A".into(), 0.5, 0.80);

        let r1 = se.evaluate_past(15, 0.60, |_| 0.70);
        assert_eq!(r1.len(), 1);

        let r2 = se.evaluate_past(20, 0.55, |_| 0.70);
        assert!(r2.is_empty(), "Already evaluated — should not re-evaluate");
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut se = SelfRewardingEvaluator::new();
        se.log_decision(1, "throttle:X".into(), 0.7, 0.80);
        let _ = se.evaluate_past(15, 0.60, |_| 0.60);

        let json = serde_json::to_string(&se).expect("serialize");
        let restored: SelfRewardingEvaluator = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.eval_count, se.eval_count);
        assert!((restored.reward_ema - se.reward_ema).abs() < 1e-6);
    }
}
