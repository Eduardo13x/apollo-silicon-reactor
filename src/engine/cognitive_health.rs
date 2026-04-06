//! Cognitive Health Score (UCHS) — unified 6-dimension health metric.
//!
//! ## Problem solved
//! AIS measures how well Apollo optimizes the system.
//! UCHS measures how well Apollo LEARNS — is the cognitive engine healthy?
//!
//! ## Design
//! 6 dimensions (mirrors AIS structure):
//! D1: Calibration — MetaCognition.meta_confidence
//! D2: Reward Quality — CognitiveRewardBus.signal_to_noise
//! D3: Belief Stability — 1 - DriftDetector.score()
//! D4: Self-Awareness — SelfRewardingEvaluator.evaluator_trust
//! D5: Adaptability — ReptileMeta.adaptation_quality
//! D6: Safety — AdversarialProbe.pass_rate_ema
//!
//! ## References
//! - [Doncieux 2018] "Open-ended Learning: Conceptual Framework" Front. Neurorobotics
//! - [Yuan 2024] "Self-Rewarding LMs" §5 — emergent self-regulation

use serde::{Deserialize, Serialize};

/// Weights for UCHS composite score. Sum = 1.0.
const W_D1: f32 = 0.20; // Calibration
const W_D2: f32 = 0.20; // Reward Quality
const W_D3: f32 = 0.15; // Belief Stability
const W_D4: f32 = 0.20; // Self-Awareness
const W_D5: f32 = 0.10; // Adaptability
const W_D6: f32 = 0.15; // Safety

/// Recovery mode threshold: UCHS < 0.40 → pause learning.
const RECOVERY_THRESHOLD: f32 = 0.40;

/// Recovery mode duration (cycles).
const RECOVERY_DURATION: u32 = 10;

/// S-tier threshold.
const S_TIER: f32 = 0.80;

// ── Types ──────────────────────────────────────────────────────────────────────

/// Individual dimension input for UCHS computation.
#[derive(Debug, Clone, Default)]
pub struct CognitiveInputs {
    /// MetaCognition.meta_confidence [0,1].
    pub calibration: f32,
    /// CognitiveRewardBus.signal_to_noise [0, ~10] → normalized to [0,1].
    pub reward_snr: f64,
    /// DriftDetector.score() [0,1] (inverted: 1 - score = stability).
    pub drift_score: f64,
    /// SelfRewardingEvaluator.evaluator_trust() [0,1].
    pub self_eval_trust: f32,
    /// ReptileMeta.adaptation_quality [0,1].
    pub adaptation_quality: f32,
    /// AdversarialProbe.safety_score() [0,1].
    pub safety_score: f32,
}

/// Unified Cognitive Health Score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveHealthScore {
    /// D1: Calibration quality.
    pub d1_calibration: f32,
    /// D2: Reward signal quality.
    pub d2_reward_quality: f32,
    /// D3: Belief stability.
    pub d3_belief_stability: f32,
    /// D4: Self-awareness / evaluator trust.
    pub d4_self_awareness: f32,
    /// D5: Adaptation / meta-learning quality.
    pub d5_adaptability: f32,
    /// D6: Cognitive safety (adversarial pass rate).
    pub d6_safety: f32,
    /// Weighted composite [0, 1].
    pub composite: f32,
    /// Grade label.
    pub grade: String,
    /// Whether recovery mode is active (all learning paused).
    pub recovery_mode: bool,
    /// Remaining cycles in recovery mode.
    recovery_cycles_remaining: u32,
}

impl Default for CognitiveHealthScore {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveHealthScore {
    pub fn new() -> Self {
        Self {
            d1_calibration: 1.0,
            d2_reward_quality: 0.5,
            d3_belief_stability: 1.0,
            d4_self_awareness: 0.0,
            d5_adaptability: 0.5,
            d6_safety: 1.0,
            composite: 0.5,
            grade: "B".into(),
            recovery_mode: false,
            recovery_cycles_remaining: 0,
        }
    }

    /// Update all dimensions from current cognitive state.
    pub fn update(&mut self, inputs: &CognitiveInputs) {
        self.d1_calibration = inputs.calibration.clamp(0.0, 1.0);

        // Normalize SNR: map [0, 5+] → [0, 1] via sigmoid (clamp negative)
        self.d2_reward_quality = (inputs.reward_snr.max(0.0) as f32 / 3.0).tanh();

        // Invert drift: low drift = high stability
        self.d3_belief_stability = (1.0 - inputs.drift_score as f32).clamp(0.0, 1.0);

        self.d4_self_awareness = inputs.self_eval_trust.clamp(0.0, 1.0);
        self.d5_adaptability = inputs.adaptation_quality.clamp(0.0, 1.0);
        self.d6_safety = inputs.safety_score.clamp(0.0, 1.0);

        // Composite
        self.composite = W_D1 * self.d1_calibration
            + W_D2 * self.d2_reward_quality
            + W_D3 * self.d3_belief_stability
            + W_D4 * self.d4_self_awareness
            + W_D5 * self.d5_adaptability
            + W_D6 * self.d6_safety;
        self.composite = self.composite.clamp(0.0, 1.0);

        // Grade
        self.grade = grade_label(self.composite);

        // Recovery mode management
        if self.recovery_mode {
            if self.recovery_cycles_remaining > 0 {
                self.recovery_cycles_remaining -= 1;
            }
            if self.recovery_cycles_remaining == 0 && self.composite >= RECOVERY_THRESHOLD {
                self.recovery_mode = false;
            }
        } else if self.composite < RECOVERY_THRESHOLD {
            self.recovery_mode = true;
            self.recovery_cycles_remaining = RECOVERY_DURATION;
        }
    }

    /// Should learning be paused? (Recovery mode active)
    pub fn should_pause_learning(&self) -> bool {
        self.recovery_mode
    }

    /// Is the cognitive engine in S-tier? (composite ≥ 0.80)
    pub fn is_s_tier(&self) -> bool {
        self.composite >= S_TIER
    }

    /// Per-dimension breakdown for dashboard display.
    pub fn breakdown(&self) -> Vec<(&'static str, f32)> {
        vec![
            ("Calibration", self.d1_calibration),
            ("RewardQuality", self.d2_reward_quality),
            ("BeliefStability", self.d3_belief_stability),
            ("SelfAwareness", self.d4_self_awareness),
            ("Adaptability", self.d5_adaptability),
            ("Safety", self.d6_safety),
        ]
    }

    /// Weakest dimension (for targeted improvement).
    pub fn weakest_dimension(&self) -> (&'static str, f32) {
        self.breakdown()
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(("Unknown", 0.0))
    }

    /// Composite as percentage string (e.g., "87.3%").
    pub fn as_pct_string(&self) -> String {
        format!("{:.1}%", self.composite * 100.0)
    }
}

fn grade_label(score: f32) -> String {
    match score {
        s if s >= 0.95 => "S+".into(),
        s if s >= 0.90 => "S".into(),
        s if s >= 0.80 => "A".into(),
        s if s >= 0.70 => "B".into(),
        s if s >= 0.60 => "C".into(),
        s if s >= 0.40 => "D".into(),
        _ => "F".into(),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn good_inputs() -> CognitiveInputs {
        CognitiveInputs {
            calibration: 0.92,
            reward_snr: 3.5,
            drift_score: 0.02,
            self_eval_trust: 0.85,
            adaptation_quality: 0.78,
            safety_score: 0.95,
        }
    }

    fn bad_inputs() -> CognitiveInputs {
        CognitiveInputs {
            calibration: 0.20,
            reward_snr: 0.1,
            drift_score: 0.90,
            self_eval_trust: 0.0,
            adaptation_quality: 0.10,
            safety_score: 0.30,
        }
    }

    #[test]
    fn test_new_defaults() {
        let chs = CognitiveHealthScore::new();
        assert!(!chs.recovery_mode);
        assert!(!chs.should_pause_learning());
    }

    #[test]
    fn test_good_inputs_high_score() {
        let mut chs = CognitiveHealthScore::new();
        chs.update(&good_inputs());
        assert!(chs.composite > 0.70, "Good inputs → high score: {}", chs.composite);
        assert!(!chs.recovery_mode);
    }

    #[test]
    fn test_bad_inputs_low_score() {
        let mut chs = CognitiveHealthScore::new();
        chs.update(&bad_inputs());
        assert!(chs.composite < 0.40, "Bad inputs → low score: {}", chs.composite);
    }

    #[test]
    fn test_recovery_mode_triggers() {
        let mut chs = CognitiveHealthScore::new();
        chs.update(&bad_inputs());
        assert!(chs.recovery_mode, "Low composite → recovery mode");
        assert!(chs.should_pause_learning());
    }

    #[test]
    fn test_recovery_mode_exits_after_improvement() {
        let mut chs = CognitiveHealthScore::new();
        chs.update(&bad_inputs());
        assert!(chs.recovery_mode);

        // Feed good inputs for recovery duration + 1
        for _ in 0..RECOVERY_DURATION + 1 {
            chs.update(&good_inputs());
        }
        assert!(!chs.recovery_mode, "Should exit recovery after good inputs");
    }

    #[test]
    fn test_recovery_persists_if_still_bad() {
        let mut chs = CognitiveHealthScore::new();
        chs.update(&bad_inputs());
        assert!(chs.recovery_mode);

        for _ in 0..RECOVERY_DURATION + 1 {
            chs.update(&bad_inputs());
        }
        assert!(chs.recovery_mode, "Should stay in recovery while bad");
    }

    #[test]
    fn test_grade_labels() {
        assert_eq!(grade_label(0.96), "S+");
        assert_eq!(grade_label(0.91), "S");
        assert_eq!(grade_label(0.85), "A");
        assert_eq!(grade_label(0.75), "B");
        assert_eq!(grade_label(0.65), "C");
        assert_eq!(grade_label(0.45), "D");
        assert_eq!(grade_label(0.20), "F");
    }

    #[test]
    fn test_is_s_tier() {
        let mut chs = CognitiveHealthScore::new();
        chs.update(&good_inputs());
        // May or may not be S-tier depending on exact inputs
        let expected_s = chs.composite >= 0.80;
        assert_eq!(chs.is_s_tier(), expected_s);
    }

    #[test]
    fn test_breakdown_has_6_dimensions() {
        let chs = CognitiveHealthScore::new();
        assert_eq!(chs.breakdown().len(), 6);
    }

    #[test]
    fn test_weakest_dimension() {
        let mut chs = CognitiveHealthScore::new();
        chs.d1_calibration = 0.90;
        chs.d2_reward_quality = 0.10; // weakest
        chs.d3_belief_stability = 0.80;
        chs.d4_self_awareness = 0.70;
        chs.d5_adaptability = 0.60;
        chs.d6_safety = 0.95;
        let (name, val) = chs.weakest_dimension();
        assert_eq!(name, "RewardQuality");
        assert!((val - 0.10).abs() < 0.001);
    }

    #[test]
    fn test_weights_sum_to_one() {
        let sum = W_D1 + W_D2 + W_D3 + W_D4 + W_D5 + W_D6;
        assert!((sum - 1.0).abs() < 0.001, "Weights sum: {sum}");
    }

    #[test]
    fn test_drift_inversion() {
        let mut chs = CognitiveHealthScore::new();
        let inputs = CognitiveInputs {
            drift_score: 0.80,
            ..Default::default()
        };
        chs.update(&inputs);
        assert!((chs.d3_belief_stability - 0.20).abs() < 0.01, "High drift → low stability");
    }

    #[test]
    fn test_snr_normalization() {
        let mut chs = CognitiveHealthScore::new();
        // High SNR → high reward quality
        let inputs = CognitiveInputs {
            reward_snr: 5.0,
            ..Default::default()
        };
        chs.update(&inputs);
        assert!(chs.d2_reward_quality > 0.8, "High SNR → high D2: {}", chs.d2_reward_quality);

        // Zero SNR → low reward quality
        let inputs_low = CognitiveInputs {
            reward_snr: 0.0,
            ..Default::default()
        };
        chs.update(&inputs_low);
        assert!(chs.d2_reward_quality < 0.1, "Zero SNR → low D2: {}", chs.d2_reward_quality);
    }

    #[test]
    fn test_as_pct_string() {
        let mut chs = CognitiveHealthScore::new();
        chs.composite = 0.873;
        assert_eq!(chs.as_pct_string(), "87.3%");
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut chs = CognitiveHealthScore::new();
        chs.update(&good_inputs());

        let json = serde_json::to_string(&chs).expect("serialize");
        let restored: CognitiveHealthScore = serde_json::from_str(&json).expect("deserialize");

        assert!((restored.composite - chs.composite).abs() < 1e-6);
        assert_eq!(restored.grade, chs.grade);
        assert_eq!(restored.recovery_mode, chs.recovery_mode);
    }

    #[test]
    fn test_clamping_extreme_inputs() {
        let mut chs = CognitiveHealthScore::new();
        let inputs = CognitiveInputs {
            calibration: 5.0,
            reward_snr: -10.0,
            drift_score: 2.0,
            self_eval_trust: -1.0,
            adaptation_quality: 99.0,
            safety_score: -5.0,
        };
        chs.update(&inputs);
        assert!(chs.composite >= 0.0 && chs.composite <= 1.0);
        for (_, val) in chs.breakdown() {
            assert!(val >= 0.0 && val <= 1.0, "Dimension out of range: {val}");
        }
    }
}
