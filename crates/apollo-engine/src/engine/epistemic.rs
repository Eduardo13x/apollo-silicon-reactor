//! Epistemic Uncertainty — per-decision composite uncertainty quantification.
//!
//! ## Problem solved
//! LinUCB exploration bonus only measures "I don't know this arm well".
//! Apollo lacks a TOTAL uncertainty signal across all cognitive subsystems
//! to gate risky actions when the system is genuinely unsure.
//!
//! ## Design
//! Composite uncertainty from 5 independent sources:
//! - RL Q-value variance (how spread are Q-values in current state)
//! - LinUCB exploration bonus (√(x'A⁻¹x) for chosen arm)
//! - NARS confidence spread (1 - min confidence across relevant beliefs)
//! - Drift score (DriftDetector.score() — model-reality divergence)
//! - MetaCognition calibration error (predicted-vs-actual gap across subsystems)
//!
//! Why calibration is a 5th component (added 2026-04-30):
//! Without it, a system with low Q-variance + few NARS observations + no drift
//! reads epistemic=LOW even when MetaCognition has measured a >0.20 calibration
//! error and activated humble_mode. The two signals would contradict each other
//! ("I'm 99% sure" vs "your predictions don't match reality"). Calibration error
//! is the most authoritative confidence signal — it directly compares prediction
//! to outcome — so it deserves explicit weight in the composite.
//!
//! When composite > 0.70 → block aggressive freezes.
//! When composite > 0.85 → force Observe arm only (zero side effects).
//!
//! ## References
//! - [Lakshminarayanan 2017] "Simple and Scalable Predictive Uncertainty
//!   Estimation using Deep Ensembles" NeurIPS §3
//! - [Guo 2017] "On Calibration of Modern Neural Networks" ICML §3
//!   ECE > 0.20 indicates miscalibration — overconfident predictions

use serde::{Deserialize, Serialize};

/// Threshold for high-uncertainty mode (block aggressive freezes).
const HIGH_UNCERTAINTY_THRESHOLD: f32 = 0.70;

/// Threshold for observe-only mode (force Observe arm, zero side effects).
const OBSERVE_ONLY_THRESHOLD: f32 = 0.85;

/// Weights for composite uncertainty formula.
/// Sum = 1.0. Tuned for Apollo's multi-agent architecture.
/// Calibration carries 0.25 (highest single weight) because it's the only
/// signal that directly compares predictions to actual outcomes — the others
/// measure spread/exploration, not correctness.
const W_RL: f32 = 0.25;
const W_LINUCB: f32 = 0.20;
const W_NARS: f32 = 0.20;
const W_DRIFT: f32 = 0.10;
const W_CALIB: f32 = 0.25;

// ── Types ──────────────────────────────────────────────────────────────────────

/// Per-decision epistemic uncertainty state.
///
/// Computed once per daemon cycle before apply_actions.
/// Gates risky actions when the cognitive system is genuinely unsure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpistemicUncertainty {
    /// RL Q-value variance in the current state [0, 1].
    /// High = Q-values are spread → agent is unsure which action is best.
    pub rl_q_variance: f32,

    /// LinUCB exploration bonus for the chosen arm [0, 1].
    /// High = arm has few observations → uncertain about its reward.
    pub linucb_exploration: f32,

    /// NARS confidence spread: 1 - min(confidence) across relevant beliefs [0, 1].
    /// High = at least one belief has very low confidence → model is uncertain.
    pub nars_confidence_spread: f32,

    /// DriftDetector score [0, 1].
    /// High = model has drifted from reality → past learning is unreliable.
    pub drift_score: f32,

    /// MetaCognition aggregate calibration error [0, 1].
    /// High = predicted confidence ≠ actual outcomes across subsystems.
    /// The most authoritative confidence signal (compares prediction to truth).
    pub meta_calibration_error: f32,

    /// Composite uncertainty [0, 1].
    /// Weighted combination of all 5 sources.
    pub composite: f32,

    /// Whether high uncertainty mode is active (composite > 0.70).
    pub high_uncertainty_mode: bool,

    /// Whether observe-only mode is active (composite > 0.85).
    pub observe_only_mode: bool,
}

impl EpistemicUncertainty {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update all uncertainty components and recompute composite.
    ///
    /// All inputs should be in [0, 1]. Out-of-range values are clamped.
    /// `meta_calibration_error` should come from `MetaCognition.calibration_error`
    /// after `meta_cognition.tick()` has run in the same cycle.
    pub fn update(
        &mut self,
        rl_q_variance: f32,
        linucb_exploration: f32,
        nars_confidence_spread: f32,
        drift_score: f32,
        meta_calibration_error: f32,
    ) {
        self.rl_q_variance = rl_q_variance.clamp(0.0, 1.0);
        self.linucb_exploration = linucb_exploration.clamp(0.0, 1.0);
        self.nars_confidence_spread = nars_confidence_spread.clamp(0.0, 1.0);
        self.drift_score = drift_score.clamp(0.0, 1.0);
        self.meta_calibration_error = meta_calibration_error.clamp(0.0, 1.0);

        // Composite: weighted sum [Lakshminarayanan 2017 §3 predictive entropy
        // + Guo 2017 §3 ECE calibration]
        self.composite = W_RL * self.rl_q_variance
            + W_LINUCB * self.linucb_exploration
            + W_NARS * self.nars_confidence_spread
            + W_DRIFT * self.drift_score
            + W_CALIB * self.meta_calibration_error;
        self.composite = self.composite.clamp(0.0, 1.0);

        // Mode transitions
        self.observe_only_mode = self.composite > OBSERVE_ONLY_THRESHOLD;
        self.high_uncertainty_mode = self.composite > HIGH_UNCERTAINTY_THRESHOLD;
    }

    /// Should aggressive actions (SIGSTOP, e-core demotion) be blocked?
    pub fn should_block_aggressive(&self) -> bool {
        self.high_uncertainty_mode
    }

    /// Should the system only observe (no actions at all)?
    pub fn should_observe_only(&self) -> bool {
        self.observe_only_mode
    }

    /// Uncertainty level label for dashboard.
    pub fn level_label(&self) -> &'static str {
        if self.observe_only_mode {
            "OBSERVE-ONLY"
        } else if self.high_uncertainty_mode {
            "HIGH"
        } else if self.composite > 0.40 {
            "MODERATE"
        } else {
            "LOW"
        }
    }

    /// Dominant uncertainty source (which component contributes most).
    pub fn dominant_source(&self) -> &'static str {
        let components = [
            (self.rl_q_variance * W_RL, "RL-QVar"),
            (self.linucb_exploration * W_LINUCB, "LinUCB-Explore"),
            (self.nars_confidence_spread * W_NARS, "NARS-Spread"),
            (self.drift_score * W_DRIFT, "Drift"),
            (self.meta_calibration_error * W_CALIB, "Calibration"),
        ];
        components
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, name)| *name)
            .unwrap_or("Unknown")
    }

    /// Per-component breakdown as (name, value, weighted_contribution).
    pub fn breakdown(&self) -> Vec<(&'static str, f32, f32)> {
        vec![
            ("RL-QVar", self.rl_q_variance, self.rl_q_variance * W_RL),
            (
                "LinUCB-Explore",
                self.linucb_exploration,
                self.linucb_exploration * W_LINUCB,
            ),
            (
                "NARS-Spread",
                self.nars_confidence_spread,
                self.nars_confidence_spread * W_NARS,
            ),
            ("Drift", self.drift_score, self.drift_score * W_DRIFT),
            (
                "Calibration",
                self.meta_calibration_error,
                self.meta_calibration_error * W_CALIB,
            ),
        ]
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_zero() {
        let eu = EpistemicUncertainty::new();
        assert_eq!(eu.composite, 0.0);
        assert!(!eu.high_uncertainty_mode);
        assert!(!eu.observe_only_mode);
    }

    #[test]
    fn test_all_low_uncertainty() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.1, 0.1, 0.1, 0.05, 0.05);
        assert!(eu.composite < 0.15);
        assert!(!eu.should_block_aggressive());
        assert!(!eu.should_observe_only());
        assert_eq!(eu.level_label(), "LOW");
    }

    #[test]
    fn test_all_high_uncertainty() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(1.0, 1.0, 1.0, 1.0, 1.0);
        assert!(eu.composite > 0.95);
        assert!(eu.should_block_aggressive());
        assert!(eu.should_observe_only());
        assert_eq!(eu.level_label(), "OBSERVE-ONLY");
    }

    #[test]
    fn test_high_mode_threshold() {
        let mut eu = EpistemicUncertainty::new();
        // Composite just above 0.70 — all components elevated
        eu.update(0.80, 0.80, 0.70, 0.30, 0.70);
        assert!(eu.should_block_aggressive());
        assert!(!eu.should_observe_only());
        assert_eq!(eu.level_label(), "HIGH");
    }

    #[test]
    fn test_observe_only_threshold() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.95, 0.95, 0.90, 0.60, 0.95);
        assert!(eu.should_observe_only());
    }

    #[test]
    fn test_moderate_level() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.50, 0.50, 0.50, 0.30, 0.50);
        assert_eq!(eu.level_label(), "MODERATE");
    }

    #[test]
    fn test_clamping_out_of_range() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(5.0, -2.0, 1.5, 3.0, 4.0);
        assert!(eu.rl_q_variance <= 1.0);
        assert!(eu.linucb_exploration >= 0.0);
        assert!(eu.meta_calibration_error <= 1.0);
        assert!(eu.composite <= 1.0);
        assert!(eu.composite >= 0.0);
    }

    #[test]
    fn test_dominant_source_rl() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(1.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(eu.dominant_source(), "RL-QVar");
    }

    #[test]
    fn test_dominant_source_drift() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.0, 0.0, 0.0, 1.0, 0.0);
        // W_CALIB=0.25 > W_DRIFT=0.10, so drift alone wins only when drift_score=1.0
        // and calibration is 0.0. Verify drift wins.
        assert_eq!(eu.dominant_source(), "Drift");
    }

    #[test]
    fn test_dominant_source_calibration() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.0, 0.0, 0.0, 0.0, 1.0);
        assert_eq!(eu.dominant_source(), "Calibration");
    }

    #[test]
    fn test_calibration_inflates_composite() {
        // The bug NotebookLM caught: epistemic could read LOW while
        // humble_mode was true. With calibration as a 5th component,
        // a high calibration error must drag composite up.
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.0, 0.0, 0.0, 0.0, 0.0);
        let baseline = eu.composite;
        eu.update(0.0, 0.0, 0.0, 0.0, 0.80);
        assert!(
            eu.composite > baseline + 0.15,
            "calibration=0.80 should add ≥0.15 to composite (W_CALIB=0.25), got {} → {}",
            baseline,
            eu.composite
        );
    }

    #[test]
    fn test_breakdown_sums_to_composite() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.6, 0.4, 0.3, 0.2, 0.5);
        let bd = eu.breakdown();
        let sum: f32 = bd.iter().map(|(_, _, w)| w).sum();
        assert!(
            (sum - eu.composite).abs() < 0.001,
            "sum={sum} composite={}",
            eu.composite
        );
    }

    #[test]
    fn test_weights_sum_to_one() {
        let sum = W_RL + W_LINUCB + W_NARS + W_DRIFT + W_CALIB;
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Weights should sum to 1.0: {sum}"
        );
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.5, 0.3, 0.7, 0.1, 0.4);

        let json = serde_json::to_string(&eu).expect("serialize");
        let restored: EpistemicUncertainty = serde_json::from_str(&json).expect("deserialize");

        assert!((restored.composite - eu.composite).abs() < 1e-6);
        assert_eq!(restored.high_uncertainty_mode, eu.high_uncertainty_mode);
    }

    #[test]
    fn test_mode_transitions_hysteresis() {
        let mut eu = EpistemicUncertainty::new();
        // Enter high mode
        eu.update(0.9, 0.9, 0.8, 0.5, 0.5);
        assert!(eu.high_uncertainty_mode);

        // Drop below → should exit
        eu.update(0.2, 0.2, 0.1, 0.1, 0.1);
        assert!(!eu.high_uncertainty_mode);
    }
}
