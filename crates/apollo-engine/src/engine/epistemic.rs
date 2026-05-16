//! Epistemic Uncertainty — per-decision composite uncertainty quantification.
//!
//! ## Problem solved
//! LinUCB exploration bonus only measures "I don't know this arm well".
//! Apollo lacks a TOTAL uncertainty signal across all cognitive subsystems
//! to gate risky actions when the system is genuinely unsure.
//!
//! ## Design
//! Composite uncertainty from 6 independent sources:
//! - RL Q-value variance (how spread are Q-values in current state)
//! - LinUCB exploration bonus (√(x'A⁻¹x) for chosen arm)
//! - NARS confidence spread (1 - min confidence across relevant beliefs)
//! - Drift score (DriftDetector.score() — model-reality divergence)
//! - MetaCognition calibration error (predicted-vs-actual gap across subsystems)
//! - Guard-tower over-protection (mean blocked-action effectiveness across mature
//!   patterns; high = blocks "would have helped" → policy is over-protective)
//!
//! Why calibration is a 5th component (added 2026-04-30):
//! Without it, a system with low Q-variance + few NARS observations + no drift
//! reads epistemic=LOW even when MetaCognition has measured a >0.20 calibration
//! error and activated humble_mode.
//!
//! Why guard-overprotection is a 6th component (added 2026-05-10):
//! After Sprint Coalition + CompanionGraph, ProactivePurge is gated by an 8-layer
//! filter and execute_actions runs a 6-layer guard. Each gate fires unilaterally
//! at high confidence; without a composed-uncertainty channel, the system can
//! over-protect for hours without anyone noticing. OutcomeTracker's
//! mean_blocked_overprotection() Bayesian-Laplace aggregate across mature blocked
//! patterns is the empirical signal: if blocks repeatedly "would have helped"
//! (Rubin 1974 counterfactual via tick_blocked), the guard tower is wrong and
//! cumulative confidence must drop. Three-Pillar Theorem (apollo_agi_paper_draft.md)
//! requires this composition under bounded rationality.
//!
//! When composite > 0.85 → block aggressive freezes.
//! When composite > 0.95 → force Observe arm only (zero side effects).
//!
//! ## Threshold calibration (2026-05-16, NotebookLM GAP 3)
//!
//! Both thresholds were rebalanced from the pre-Phase 2 linear-sum era
//! (HIGH=0.70, OBSERVE_ONLY=0.85) to the post-Phase 2 RSS-composition era
//! (HIGH=0.85, OBSERVE_ONLY=0.95). Under linear-sum, a single input at 1.0
//! with W=0.20 produced `composite ≈ 0.20`; under RSS, the same input
//! produces `composite ≈ 0.45` — more than double, due to the
//! `raw_rss / max_rss_possible` normalization that amplifies isolated
//! strong signals. Holding the thresholds at 0.70/0.85 would have made
//! HIGH/OBSERVE-ONLY mode trip on a single noisy source — exactly the
//! "regression paralysis" pattern Phase 2 set out to prevent.
//!
//! ## References
//! - [Lakshminarayanan 2017] "Simple and Scalable Predictive Uncertainty
//!   Estimation using Deep Ensembles" NeurIPS §3 — predictive uncertainty
//!   calibration: an ensemble's effective threshold must scale with the
//!   composition method, not the components.
//! - [Guo 2017] "On Calibration of Modern Neural Networks" ICML §3
//!   ECE > 0.20 indicates miscalibration — overconfident predictions

use serde::{Deserialize, Serialize};

/// Threshold for high-uncertainty mode (block aggressive freezes).
///
/// Recalibrated 2026-05-16 from 0.70 → 0.85 to compensate for the
/// linear→RSS composition change. Under RSS, isolated strong signals
/// reach ~0.45 (W=0.20) where the linear sum reached only 0.20; the
/// threshold must rise proportionally or HIGH triggers on any noisy
/// single input.
const HIGH_UNCERTAINTY_THRESHOLD: f32 = 0.85;

/// Threshold for observe-only mode (force Observe arm, zero side effects).
///
/// Recalibrated 2026-05-16 from 0.85 → 0.95 in lockstep with
/// `HIGH_UNCERTAINTY_THRESHOLD`. OBSERVE_ONLY is reserved for the
/// degenerate "system is maximally uncertain across all six sources"
/// regime — at composite >= 0.95, the only safe action is no action.
const OBSERVE_ONLY_THRESHOLD: f32 = 0.95;

/// Weights for composite uncertainty formula.
/// Sum = 1.0. Tuned for Apollo's multi-agent architecture.
/// Calibration and Guard-overprotection carry the largest single weights
/// because they're the two signals that directly compare predictions /
/// policies to actual outcomes — the others measure spread / exploration,
/// not correctness. Rebalanced 2026-05-10 to admit Guard-overprotection.
const W_RL: f32 = 0.20;
const W_LINUCB: f32 = 0.15;
const W_NARS: f32 = 0.15;
const W_DRIFT: f32 = 0.10;
const W_CALIB: f32 = 0.20;
const W_GUARD: f32 = 0.20;

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

    /// Guard-tower over-protection signal from `OutcomeTracker::
    /// mean_blocked_overprotection()` [0, 1].
    /// High = blocked actions repeatedly "would have helped" per Rubin 1974
    /// counterfactual → guard policy is wrong → cumulative confidence drops.
    #[serde(default)]
    pub guard_overprotection: f32,

    /// Composite uncertainty [0, 1].
    /// Weighted combination of all 6 sources.
    pub composite: f32,

    /// Whether high uncertainty mode is active (composite > 0.85).
    /// Threshold recalibrated 2026-05-16 from 0.70 for RSS composition.
    pub high_uncertainty_mode: bool,

    /// Whether observe-only mode is active (composite > 0.95).
    /// Threshold recalibrated 2026-05-16 from 0.85 for RSS composition.
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
    /// `guard_overprotection` should come from
    /// `OutcomeTracker.mean_blocked_overprotection()` after `tick_blocked` has
    /// resolved any patterns that crossed their 30 s evaluation window.
    pub fn update(
        &mut self,
        rl_q_variance: f32,
        linucb_exploration: f32,
        nars_confidence_spread: f32,
        drift_score: f32,
        meta_calibration_error: f32,
        guard_overprotection: f32,
    ) {
        self.rl_q_variance = rl_q_variance.clamp(0.0, 1.0);
        self.linucb_exploration = linucb_exploration.clamp(0.0, 1.0);
        self.nars_confidence_spread = nars_confidence_spread.clamp(0.0, 1.0);
        self.drift_score = drift_score.clamp(0.0, 1.0);
        self.meta_calibration_error = meta_calibration_error.clamp(0.0, 1.0);
        self.guard_overprotection = guard_overprotection.clamp(0.0, 1.0);

        // Composite: Root-Sum-Square (RSS) composition to prevent regression paralysis
        // from multiple weak noise signals, normalized to [0, 1].
        let sq_rl = (W_RL * self.rl_q_variance).powi(2);
        let sq_linucb = (W_LINUCB * self.linucb_exploration).powi(2);
        let sq_nars = (W_NARS * self.nars_confidence_spread).powi(2);
        let sq_drift = (W_DRIFT * self.drift_score).powi(2);
        let sq_calib = (W_CALIB * self.meta_calibration_error).powi(2);
        let sq_guard = (W_GUARD * self.guard_overprotection).powi(2);

        let sum_sq = sq_rl + sq_linucb + sq_nars + sq_drift + sq_calib + sq_guard;
        let raw_rss = sum_sq.sqrt();
        
        let max_rss_possible = (
            W_RL.powi(2) + W_LINUCB.powi(2) + W_NARS.powi(2) + 
            W_DRIFT.powi(2) + W_CALIB.powi(2) + W_GUARD.powi(2)
        ).sqrt();

        self.composite = raw_rss / max_rss_possible;
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
            (self.guard_overprotection * W_GUARD, "Guard-Overprotect"),
        ];
        components
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, name)| *name)
            .unwrap_or("Unknown")
    }

    /// Per-component breakdown as (name, value, weighted_contribution).
    pub fn breakdown(&self) -> Vec<(&'static str, f32, f32)> {
        let sq_rl = (self.rl_q_variance * W_RL).powi(2);
        let sq_linucb = (self.linucb_exploration * W_LINUCB).powi(2);
        let sq_nars = (self.nars_confidence_spread * W_NARS).powi(2);
        let sq_drift = (self.drift_score * W_DRIFT).powi(2);
        let sq_calib = (self.meta_calibration_error * W_CALIB).powi(2);
        let sq_guard = (self.guard_overprotection * W_GUARD).powi(2);

        let sum_sq = sq_rl + sq_linucb + sq_nars + sq_drift + sq_calib + sq_guard;
        if sum_sq == 0.0 {
            return vec![
                ("RL-QVar", self.rl_q_variance, 0.0),
                ("LinUCB-Explore", self.linucb_exploration, 0.0),
                ("NARS-Spread", self.nars_confidence_spread, 0.0),
                ("Drift", self.drift_score, 0.0),
                ("Calibration", self.meta_calibration_error, 0.0),
                ("Guard-Overprotect", self.guard_overprotection, 0.0),
            ];
        }

        let f = self.composite / sum_sq;
        vec![
            ("RL-QVar", self.rl_q_variance, sq_rl * f),
            ("LinUCB-Explore", self.linucb_exploration, sq_linucb * f),
            ("NARS-Spread", self.nars_confidence_spread, sq_nars * f),
            ("Drift", self.drift_score, sq_drift * f),
            ("Calibration", self.meta_calibration_error, sq_calib * f),
            ("Guard-Overprotect", self.guard_overprotection, sq_guard * f),
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
        eu.update(0.1, 0.1, 0.1, 0.05, 0.05, 0.05);
        assert!(eu.composite < 0.15);
        assert!(!eu.should_block_aggressive());
        assert!(!eu.should_observe_only());
        assert_eq!(eu.level_label(), "LOW");
    }

    #[test]
    fn test_all_high_uncertainty() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert!(eu.composite > 0.95);
        assert!(eu.should_block_aggressive());
        assert!(eu.should_observe_only());
        assert_eq!(eu.level_label(), "OBSERVE-ONLY");
    }

    #[test]
    fn test_high_mode_threshold() {
        let mut eu = EpistemicUncertainty::new();
        // Composite just above 0.85 — all heavy-weight components saturated
        // so the RSS-composed 6-component formula crosses the recalibrated
        // HIGH=0.85 threshold without also crossing OBSERVE_ONLY=0.95.
        // (Inputs updated 2026-05-16 alongside the threshold recalibration.)
        eu.update(1.0, 1.0, 0.50, 0.50, 1.0, 1.0);
        assert!(eu.should_block_aggressive());
        assert!(!eu.should_observe_only());
        assert_eq!(eu.level_label(), "HIGH");
    }

    #[test]
    fn test_observe_only_threshold() {
        let mut eu = EpistemicUncertainty::new();
        // Inputs updated 2026-05-16 to cross the recalibrated
        // OBSERVE_ONLY=0.95 threshold.
        eu.update(1.0, 1.0, 1.0, 0.90, 1.0, 1.0);
        assert!(eu.should_observe_only());
    }

    #[test]
    fn test_moderate_level() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.50, 0.50, 0.50, 0.30, 0.50, 0.50);
        assert_eq!(eu.level_label(), "MODERATE");
    }

    #[test]
    fn test_clamping_out_of_range() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(5.0, -2.0, 1.5, 3.0, 4.0, 9.9);
        assert!(eu.rl_q_variance <= 1.0);
        assert!(eu.linucb_exploration >= 0.0);
        assert!(eu.meta_calibration_error <= 1.0);
        assert!(eu.composite <= 1.0);
        assert!(eu.composite >= 0.0);
    }

    #[test]
    fn test_dominant_source_rl() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(eu.dominant_source(), "RL-QVar");
    }

    #[test]
    fn test_dominant_source_drift() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        // Drift weighted contribution = 0.10 (W_DRIFT). Other components 0.
        // Drift wins by default.
        assert_eq!(eu.dominant_source(), "Drift");
    }

    #[test]
    fn test_dominant_source_calibration() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        assert_eq!(eu.dominant_source(), "Calibration");
    }

    #[test]
    fn test_dominant_source_guard() {
        // Guard-overprotection alone with everything else 0 → Guard wins.
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        assert_eq!(eu.dominant_source(), "Guard-Overprotect");
    }

    #[test]
    fn test_guard_inflates_composite() {
        // The Three-Pillar gap NotebookLM caught 2026-05-10: an 8-layer guard
        // tower could over-protect for hours with epistemic still reading LOW.
        // With guard_overprotection as a 6th component (W_GUARD=0.20), a high
        // empirical over-protection signal must drag composite up.
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let baseline = eu.composite;
        eu.update(0.0, 0.0, 0.0, 0.0, 0.0, 0.80);
        assert!(
            eu.composite > baseline + 0.10,
            "guard=0.80 should add ≥0.10 to composite (W_GUARD=0.20), got {} → {}",
            baseline,
            eu.composite
        );
    }

    #[test]
    fn test_calibration_inflates_composite() {
        // The bug NotebookLM caught: epistemic could read LOW while
        // humble_mode was true. With calibration as a 5th component,
        // a high calibration error must drag composite up.
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let baseline = eu.composite;
        eu.update(0.0, 0.0, 0.0, 0.0, 0.80, 0.0);
        assert!(
            eu.composite > baseline + 0.10,
            "calibration=0.80 should add ≥0.10 to composite (W_CALIB=0.20), got {} → {}",
            baseline,
            eu.composite
        );
    }

    #[test]
    fn test_breakdown_sums_to_composite() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.6, 0.4, 0.3, 0.2, 0.5, 0.7);
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
        let sum = W_RL + W_LINUCB + W_NARS + W_DRIFT + W_CALIB + W_GUARD;
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Weights should sum to 1.0: {sum}"
        );
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.5, 0.3, 0.7, 0.1, 0.4, 0.6);

        let json = serde_json::to_string(&eu).expect("serialize");
        let restored: EpistemicUncertainty = serde_json::from_str(&json).expect("deserialize");

        assert!((restored.composite - eu.composite).abs() < 1e-6);
        assert_eq!(restored.high_uncertainty_mode, eu.high_uncertainty_mode);
    }

    #[test]
    fn test_mode_transitions_hysteresis() {
        let mut eu = EpistemicUncertainty::new();
        // Enter high mode — heavy-weight components saturated to cross the
        // recalibrated HIGH=0.85 threshold under RSS composition. Inputs
        // updated 2026-05-16 alongside the threshold change.
        eu.update(1.0, 1.0, 0.95, 0.80, 1.0, 1.0);
        assert!(eu.high_uncertainty_mode);

        // Drop below → should exit
        eu.update(0.2, 0.2, 0.1, 0.1, 0.1, 0.1);
        assert!(!eu.high_uncertainty_mode);
    }

    // ── RSS threshold recalibration (NotebookLM 2026-05-16, GAP 3) ───────────

    /// Under RSS composition, a single strong input no longer dominates the
    /// composite the way a linear-sum would: rl_q_variance=0.99 gives
    /// `composite ≈ 0.47` (well below the new HIGH=0.85 threshold). Under the
    /// pre-RSS HIGH=0.70 threshold this scenario would not have triggered HIGH
    /// either, but the test enforces the post-recalibration invariant
    /// "single strong source alone never trips HIGH" — preventing a future
    /// regression that re-narrows the threshold back below the RSS-feasible
    /// single-source ceiling.
    /// [Lakshminarayanan 2017] predictive uncertainty calibration.
    #[test]
    fn rss_composite_single_strong_input_below_new_high_threshold() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(0.99, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(
            eu.composite < 0.85,
            "single strong RL input should not reach HIGH=0.85 alone, got composite={}",
            eu.composite
        );
        assert!(
            !eu.high_uncertainty_mode,
            "high_uncertainty_mode must stay false for a single strong input"
        );
    }

    /// Genuine multi-source high uncertainty MUST still cross the
    /// recalibrated HIGH=0.85 threshold. Four heavy-weight inputs at 1.0
    /// (rl + linucb + calib + guard, all weight ≥0.15) produce composite
    /// ≈ 0.90 under RSS, which exceeds 0.85 and triggers
    /// `high_uncertainty_mode`. Without this regression test, a future
    /// over-widening of the threshold (e.g. to 0.95 for HIGH) would silently
    /// disable the guard for legitimate compound uncertainty.
    #[test]
    fn rss_composite_multiple_inputs_can_reach_new_high_threshold() {
        let mut eu = EpistemicUncertainty::new();
        // 4 heavy-weight components saturated; nars + drift left at 0.
        eu.update(1.0, 1.0, 0.0, 0.0, 1.0, 1.0);
        assert!(
            eu.composite > 0.85,
            "4 heavy inputs at 1.0 must exceed HIGH=0.85 under RSS, got composite={}",
            eu.composite
        );
        assert!(
            eu.high_uncertainty_mode,
            "high_uncertainty_mode must trigger when composite > 0.85"
        );
    }

    /// At maximum saturation across all six sources, composite is exactly the
    /// RSS ceiling (1.0 by construction of the normalization), which must
    /// reach the recalibrated OBSERVE-ONLY=0.95 threshold. This is the
    /// degenerate "system is maximally uncertain" case the OBSERVE-ONLY mode
    /// exists for: zero side effects until evidence accumulates.
    #[test]
    fn rss_composite_all_inputs_max_reaches_observe_only() {
        let mut eu = EpistemicUncertainty::new();
        eu.update(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert!(
            eu.composite >= 0.95,
            "all 6 inputs at max should reach OBSERVE-ONLY=0.95, got composite={}",
            eu.composite
        );
        assert!(
            eu.observe_only_mode,
            "observe_only_mode must trigger at composite >= 0.95"
        );
    }
}
