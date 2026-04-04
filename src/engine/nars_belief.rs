//! NARS-inspired belief revision for concept drift detection.
//!
//! Implements the core TruthValue and Revision Rule from Pei Wang's
//! Non-Axiomatic Reasoning System (NARS, 2013), stripped to what Apollo needs:
//! detecting when learned effectiveness beliefs have drifted from reality.
//!
//! # Concept Drift Signal
//! A belief tracks: "action X → good outcome" with (frequency, confidence).
//! - frequency ∈ [0,1]: how often X actually produced a good outcome
//! - confidence ∈ [0,1): evidence weight — grows as more observations arrive
//!
//! When we re-observe X with new evidence, Revision updates the belief.
//! If `|f_after - f_before| > DRIFT_THRESHOLD`, the model has drifted —
//! the old learned pattern no longer matches current system behavior.
//!
//! # Revision Rule (Pei Wang 2013, §3.3.3)
//! w = c / (1 - c)
//! f_new = (w1·f1 + w2·f2) / (w1 + w2)
//! c_new = (w1 + w2) / (w1 + w2 + 1)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Constants ────────────────────────────────────────────────────────────────

/// Frequency shift that triggers a drift alert for a single belief.
/// A 20pp shift means the action's effectiveness profile has materially changed.
/// Inspired by Population Stability Index threshold (PSI ≥ 0.20 = major shift).
const DRIFT_THRESHOLD: f32 = 0.20;

/// Minimum confidence before drift can be declared (need enough evidence).
const MIN_CONFIDENCE_FOR_DRIFT: f32 = 0.30;

/// EMA alpha for aggregate drift score (slow-decaying: half-life ≈ 69 ticks).
const DRIFT_SCORE_ALPHA: f64 = 0.01;

// ── TruthValue ───────────────────────────────────────────────────────────────

/// NARS TruthValue: (frequency, confidence).
///
/// frequency ∈ [0,1]: P(proposition is true | all evidence)
/// confidence ∈ [0,1): evidence weight; approaches 1 asymptotically
///
/// [Pei Wang 2013] "Non-Axiomatic Reasoning System", §3.3
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct TruthValue {
    /// How often this proposition was true. 0 = never, 1 = always.
    pub frequency: f32,
    /// Evidence weight. Starts near 0, grows toward 1 with each observation.
    pub confidence: f32,
}

impl Default for TruthValue {
    /// Ignorance prior: no evidence either way.
    fn default() -> Self {
        Self { frequency: 0.5, confidence: 0.0 }
    }
}

impl TruthValue {
    pub fn new(frequency: f32, confidence: f32) -> Self {
        Self {
            frequency: frequency.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 0.9999),
        }
    }

    /// Expected value: P(true) weighted by confidence.
    /// Unconfident beliefs regress toward 0.5 (maximum uncertainty).
    /// [Pei Wang 2013] §3.3.1 — expectation = f·c + 0.5·(1-c)
    pub fn expectation(&self) -> f32 {
        self.frequency * self.confidence + 0.5 * (1.0 - self.confidence)
    }

    /// Apply the NARS Revision Rule: merge two independent observations.
    ///
    /// Returns updated TruthValue after incorporating new evidence.
    /// Revision is symmetric and commutative.
    ///
    /// [Pei Wang 2013] §3.3.3 — Revision
    pub fn revise(self, new_evidence: TruthValue) -> TruthValue {
        let eps = 1e-6_f32;
        let w1 = self.confidence / (1.0 - self.confidence + eps);
        let w2 = new_evidence.confidence / (1.0 - new_evidence.confidence + eps);
        let w = w1 + w2;
        if w < eps {
            return self;
        }
        let f_new = (w1 * self.frequency + w2 * new_evidence.frequency) / w;
        let c_new = w / (w + 1.0);
        TruthValue::new(f_new, c_new)
    }

    /// Confidence from evidence count n using the NARS formula: c = n / (n + k)
    /// where k = 1 (Laplace-like prior strength).
    pub fn confidence_from_count(n: u32) -> f32 {
        n as f32 / (n as f32 + 1.0)
    }
}

// ── BeliefEntry ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BeliefEntry {
    tv: TruthValue,
    /// Frequency value before the last revision — used to compute drift delta.
    freq_before_last_revision: f32,
    /// Total observations that fed this belief.
    observations: u32,
}

impl BeliefEntry {
    fn new(initial_freq: f32) -> Self {
        Self {
            tv: TruthValue::new(initial_freq, TruthValue::confidence_from_count(1)),
            freq_before_last_revision: initial_freq,
            observations: 1,
        }
    }

    /// Incorporate a new observation (true/false) and return the frequency delta.
    fn observe(&mut self, success: bool) -> f32 {
        self.observations += 1;
        let new_freq = if success { 1.0_f32 } else { 0.0_f32 };
        let new_conf = TruthValue::confidence_from_count(1); // single observation
        let new_evidence = TruthValue::new(new_freq, new_conf);
        self.freq_before_last_revision = self.tv.frequency;
        self.tv = self.tv.revise(new_evidence);
        (self.tv.frequency - self.freq_before_last_revision).abs()
    }

    /// True if this belief has shifted significantly since last calibration.
    fn is_drifted(&self) -> bool {
        let delta = (self.tv.frequency - self.freq_before_last_revision).abs();
        self.tv.confidence >= MIN_CONFIDENCE_FOR_DRIFT && delta >= DRIFT_THRESHOLD
    }
}

// ── DriftDetector ────────────────────────────────────────────────────────────

/// Tracks effectiveness beliefs for a set of named actions/specialists.
/// Detects concept drift via NARS Revision: when frequency shifts ≥20pp
/// with sufficient confidence, the learned model no longer matches reality.
///
/// Drift score ∈ [0,1]: 0 = stable, 1 = total model invalidation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DriftDetector {
    beliefs: HashMap<String, BeliefEntry>,
    /// EMA of per-belief drift deltas. High = model is drifting.
    pub drift_score: f64,
    /// Number of beliefs currently in a drifted state.
    pub drifted_count: usize,
}

impl DriftDetector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an observation for a named action/specialist.
    /// `success` = did it produce a good outcome?
    /// Returns the local frequency delta from revision.
    pub fn observe(&mut self, key: &str, success: bool) -> f32 {
        let delta = if let Some(entry) = self.beliefs.get_mut(key) {
            entry.observe(success)
        } else {
            let initial_freq = if success { 1.0 } else { 0.0 };
            self.beliefs.insert(key.to_string(), BeliefEntry::new(initial_freq));
            0.0 // first observation: no drift yet
        };

        // Update aggregate drift score via EMA
        self.drift_score = DRIFT_SCORE_ALPHA * delta as f64
            + (1.0 - DRIFT_SCORE_ALPHA) * self.drift_score;

        // Recount drifted beliefs
        self.drifted_count = self.beliefs.values().filter(|e| e.is_drifted()).count();

        delta
    }

    /// Overall drift score ∈ [0,1]. Threshold: > 0.05 = notable drift.
    pub fn score(&self) -> f64 {
        self.drift_score
    }

    /// True if model drift is significant enough to warrant recalibration.
    /// Threshold: ≥2 beliefs drifted OR aggregate EMA score > 0.08.
    pub fn needs_recalibration(&self) -> bool {
        self.drifted_count >= 2 || self.drift_score > 0.08
    }

    /// Get current TruthValue for a key (for diagnostics).
    pub fn belief(&self, key: &str) -> Option<TruthValue> {
        self.beliefs.get(key).map(|e| e.tv)
    }

    /// Reset drift signals after recalibration has been applied.
    /// Does NOT reset the beliefs themselves — keeps accumulated evidence.
    pub fn acknowledge_recalibration(&mut self) {
        self.drift_score *= 0.1; // decay but don't erase
        for entry in self.beliefs.values_mut() {
            entry.freq_before_last_revision = entry.tv.frequency;
        }
        self.drifted_count = 0;
    }

    /// Number of tracked beliefs.
    pub fn len(&self) -> usize {
        self.beliefs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.beliefs.is_empty()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TruthValue ────────────────────────────────────────────────────────────

    #[test]
    fn truth_value_defaults_to_ignorance_prior() {
        let tv = TruthValue::default();
        assert_eq!(tv.frequency, 0.5);
        assert_eq!(tv.confidence, 0.0);
        // Expectation of ignorance = 0.5
        assert!((tv.expectation() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn revision_symmetric_equal_evidence() {
        // Two equal observations → same result
        let tv1 = TruthValue::new(0.8, 0.6);
        let tv2 = TruthValue::new(0.2, 0.6);
        let r1 = tv1.revise(tv2);
        let r2 = tv2.revise(tv1);
        // Symmetric
        assert!((r1.frequency - r2.frequency).abs() < 1e-5, "revision should be symmetric");
        assert!((r1.confidence - r2.confidence).abs() < 1e-5);
        // Midpoint
        assert!((r1.frequency - 0.5).abs() < 0.01, "equal evidence → midpoint");
    }

    #[test]
    fn revision_higher_confidence_dominates() {
        // High-confidence belief should pull result toward its frequency
        let strong = TruthValue::new(0.9, 0.9);
        let weak = TruthValue::new(0.1, 0.1);
        let result = strong.revise(weak);
        assert!(result.frequency > 0.7, "strong belief should dominate: got {}", result.frequency);
        assert!(result.confidence > strong.confidence, "confidence should grow after revision");
    }

    #[test]
    fn revision_confidence_grows_monotonically() {
        let mut tv = TruthValue::new(0.5, 0.3);
        for _ in 0..10 {
            let prev_conf = tv.confidence;
            tv = tv.revise(TruthValue::new(0.5, 0.1));
            assert!(tv.confidence > prev_conf, "confidence must grow with each observation");
        }
    }

    #[test]
    fn confidence_from_count_approaches_one() {
        assert!((TruthValue::confidence_from_count(1) - 0.5).abs() < 1e-5);
        assert!((TruthValue::confidence_from_count(9) - 0.9).abs() < 1e-5);
        assert!(TruthValue::confidence_from_count(999) > 0.99);
        assert!(TruthValue::confidence_from_count(9999) > 0.999);
    }

    #[test]
    fn expectation_regresses_toward_half_for_low_confidence() {
        let tv = TruthValue::new(1.0, 0.0);
        // With zero confidence, expectation = 0.5 regardless of frequency
        assert!((tv.expectation() - 0.5).abs() < 1e-5);
    }

    // ── DriftDetector ─────────────────────────────────────────────────────────

    #[test]
    fn drift_detector_no_drift_on_consistent_outcomes() {
        let mut dd = DriftDetector::new();
        // 20 consistent successes → stable model
        for _ in 0..20 {
            dd.observe("proc_A", true);
        }
        assert!(!dd.needs_recalibration(), "consistent outcomes → no drift");
        assert!(dd.drift_score < 0.05);
    }

    #[test]
    fn drift_detector_detects_regime_change() {
        let mut dd = DriftDetector::new();
        // Phase 1: process always effective
        for _ in 0..30 {
            dd.observe("proc_X", true);
        }
        let score_before = dd.drift_score;
        // Phase 2: suddenly never effective (regime change)
        for _ in 0..30 {
            dd.observe("proc_X", false);
        }
        // Drift score should increase
        assert!(
            dd.drift_score > score_before || dd.drifted_count >= 1,
            "regime change should increase drift signal"
        );
    }

    #[test]
    fn drift_detector_acknowledge_resets_signal() {
        let mut dd = DriftDetector::new();
        for _ in 0..30 {
            dd.observe("proc_A", true);
        }
        for _ in 0..30 {
            dd.observe("proc_A", false);
        }
        let drift_before = dd.drift_score;
        dd.acknowledge_recalibration();
        assert!(dd.drift_score < drift_before * 0.5, "acknowledge should reduce drift score");
        assert_eq!(dd.drifted_count, 0);
    }

    #[test]
    fn drift_detector_multiple_beliefs_tracked_independently() {
        let mut dd = DriftDetector::new();
        // proc_A: stable
        for _ in 0..20 {
            dd.observe("proc_A", true);
        }
        // proc_B: unstable
        for _ in 0..10 {
            dd.observe("proc_B", true);
        }
        for _ in 0..20 {
            dd.observe("proc_B", false);
        }
        assert_eq!(dd.len(), 2);
        let tv_a = dd.belief("proc_A").unwrap();
        let tv_b = dd.belief("proc_B").unwrap();
        assert!(tv_a.frequency > 0.7, "proc_A should have high frequency");
        assert!(tv_b.frequency < 0.5, "proc_B should have lower frequency after failures");
    }

    #[test]
    fn drift_detector_first_observation_no_drift() {
        let mut dd = DriftDetector::new();
        let delta = dd.observe("new_process", true);
        assert_eq!(delta, 0.0, "first observation produces no drift delta");
        assert!(!dd.needs_recalibration());
    }

    #[test]
    fn revision_rule_math_from_paper() {
        // [Pei Wang 2013] §3.3.3 example: two beliefs with same frequency
        // f1=0.8, c1=0.6; f2=0.8, c2=0.6
        // w1 = 0.6/0.4 = 1.5; w2 = 1.5; w = 3.0
        // f_new = (1.5*0.8 + 1.5*0.8) / 3.0 = 0.8
        // c_new = 3.0 / (3.0 + 1.0) = 0.75
        let tv1 = TruthValue::new(0.8, 0.6);
        let tv2 = TruthValue::new(0.8, 0.6);
        let result = tv1.revise(tv2);
        assert!((result.frequency - 0.8).abs() < 0.001, "same freq → no change: {}", result.frequency);
        assert!((result.confidence - 0.75).abs() < 0.001, "c_new=0.75: {}", result.confidence);
    }
}
