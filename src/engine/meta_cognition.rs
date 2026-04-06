//! MetaCognition — second-order confidence calibration layer.
//!
//! ## Problem solved
//! Apollo decides with equal "confidence" whether it has 5 or 500 observations.
//! No subsystem tracks "am I learning correctly or spinning in circles?"
//!
//! ## Design
//! MetaCognition tracks per-subsystem accuracy via Expected Calibration Error (ECE)
//! proxy [Guo 2017]. When calibration degrades, it activates "humble mode" —
//! more exploration, softer thresholds, conservative actions.
//!
//! ## References
//! - [Guo 2017] "On Calibration of Modern Neural Networks" ICML
//!   §3: Expected Calibration Error, temperature scaling
//! - [Lakshminarayanan 2017] "Simple and Scalable Predictive Uncertainty" NeurIPS
//!   §2.1: predictive entropy as uncertainty proxy

use serde::{Deserialize, Serialize};

/// Calibration error threshold to trigger humble mode.
const HUMBLE_THRESHOLD: f32 = 0.20;

/// Number of cycles in humble mode before re-evaluating.
const HUMBLE_DURATION: u32 = 50;

/// EMA alpha for per-subsystem accuracy tracking.
const ACCURACY_EMA_ALPHA: f32 = 0.05;

/// Minimum observations before ECE is meaningful.
const MIN_OBS_FOR_ECE: u32 = 10;

// ── Types ──────────────────────────────────────────────────────────────────────

/// Subsystem identity for metacognitive tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubsystemId {
    RlAgent,
    LinUcb,
    NarsBelief,
    CausalGraph,
    SignalKalman,
    FreezeIntelligence,
}

/// Per-subsystem accuracy tracker.
///
/// Tracks the gap between predicted confidence and actual outcome.
/// Large gap = poorly calibrated = "thinks it knows but doesn't".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccuracyEma {
    /// EMA of predicted confidence values [0,1].
    pub predicted_ema: f32,
    /// EMA of actual outcome values [0,1].
    pub actual_ema: f32,
    /// EMA of |predicted - actual| — the calibration gap.
    pub calibration_gap: f32,
    /// Total observations fed.
    pub observations: u32,
}

impl AccuracyEma {
    /// Record a (predicted_confidence, actual_outcome) pair.
    pub fn observe(&mut self, predicted: f32, actual: f32) {
        let predicted = predicted.clamp(0.0, 1.0);
        let actual = actual.clamp(0.0, 1.0);
        let gap = (predicted - actual).abs();

        self.predicted_ema = ema_f32(self.predicted_ema, predicted, ACCURACY_EMA_ALPHA);
        self.actual_ema = ema_f32(self.actual_ema, actual, ACCURACY_EMA_ALPHA);
        self.calibration_gap = ema_f32(self.calibration_gap, gap, ACCURACY_EMA_ALPHA);
        self.observations += 1;
    }

    /// Is this subsystem well-calibrated? (gap < threshold)
    pub fn is_calibrated(&self, threshold: f32) -> bool {
        self.observations >= MIN_OBS_FOR_ECE && self.calibration_gap < threshold
    }

    /// Direction of miscalibration: positive = overconfident, negative = underconfident.
    pub fn miscalibration_direction(&self) -> f32 {
        self.predicted_ema - self.actual_ema
    }
}

/// MetaCognition layer — orchestrates second-order confidence.
///
/// Tracks per-subsystem calibration quality and activates protective
/// "humble mode" when the cognitive system is poorly calibrated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaCognition {
    /// Per-subsystem accuracy trackers.
    subsystems: Vec<(SubsystemId, AccuracyEma)>,
    /// Aggregate ECE proxy: weighted mean of calibration gaps.
    pub calibration_error: f32,
    /// Second-order confidence: how much to trust the system's confidence.
    /// meta_confidence = 1.0 - calibration_error
    pub meta_confidence: f32,
    /// Whether humble mode is active.
    pub humble_mode: bool,
    /// Remaining cycles in humble mode.
    humble_cycles_remaining: u32,
    /// Total observations across all subsystems.
    total_observations: u64,
}

impl Default for MetaCognition {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaCognition {
    pub fn new() -> Self {
        Self {
            subsystems: Vec::new(),
            calibration_error: 0.0,
            meta_confidence: 1.0,
            humble_mode: false,
            humble_cycles_remaining: 0,
            total_observations: 0,
        }
    }

    /// Record a calibration observation from a subsystem.
    ///
    /// `predicted`: what the subsystem thought would happen [0,1]
    /// `actual`: what actually happened [0,1]
    pub fn observe(&mut self, subsystem: SubsystemId, predicted: f32, actual: f32) {
        // Find or create subsystem tracker
        let tracker = match self.subsystems.iter_mut().find(|(id, _)| *id == subsystem) {
            Some((_, tracker)) => tracker,
            None => {
                self.subsystems.push((subsystem, AccuracyEma::default()));
                &mut self.subsystems.last_mut().unwrap().1
            }
        };
        tracker.observe(predicted, actual);
        self.total_observations += 1;
    }

    /// Update aggregate calibration error and humble mode state.
    ///
    /// Call once per daemon cycle after all subsystems have reported.
    pub fn tick(&mut self) {
        self.update_calibration_error();

        // Humble mode management
        if self.humble_mode {
            if self.humble_cycles_remaining > 0 {
                self.humble_cycles_remaining -= 1;
            }
            // Exit humble mode if calibration has improved AND duration expired
            if self.humble_cycles_remaining == 0 && self.calibration_error < HUMBLE_THRESHOLD {
                self.humble_mode = false;
            }
        } else if self.calibration_error > HUMBLE_THRESHOLD && self.total_observations >= MIN_OBS_FOR_ECE as u64 {
            // Enter humble mode
            self.humble_mode = true;
            self.humble_cycles_remaining = HUMBLE_DURATION;
        }
    }

    /// Recompute aggregate ECE from all subsystem trackers.
    fn update_calibration_error(&mut self) {
        if self.subsystems.is_empty() {
            return;
        }

        let mut total_gap = 0.0f32;
        let mut weight_sum = 0.0f32;

        for (_, tracker) in &self.subsystems {
            if tracker.observations >= MIN_OBS_FOR_ECE {
                // Weight by observation count (more data = more trustworthy ECE)
                let w = (tracker.observations as f32).sqrt();
                total_gap += tracker.calibration_gap * w;
                weight_sum += w;
            }
        }

        self.calibration_error = if weight_sum > 0.0 {
            (total_gap / weight_sum).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.meta_confidence = 1.0 - self.calibration_error;
    }

    /// Exploration multiplier when humble mode is active.
    ///
    /// RL: epsilon *= humble_exploration_mult()
    /// LinUCB: alpha *= humble_exploration_mult()
    pub fn humble_exploration_mult(&self) -> f32 {
        if self.humble_mode {
            2.0
        } else {
            1.0
        }
    }

    /// Freeze confidence floor when humble mode is active.
    ///
    /// Normal: 0.35 (MIN_FREEZE_CONFIDENCE)
    /// Humble: 0.45 (more conservative)
    pub fn humble_freeze_confidence_floor(&self) -> f32 {
        if self.humble_mode {
            0.45
        } else {
            0.35
        }
    }

    /// Query calibration gap for a specific subsystem.
    pub fn subsystem_gap(&self, subsystem: SubsystemId) -> Option<f32> {
        self.subsystems
            .iter()
            .find(|(id, _)| *id == subsystem)
            .map(|(_, tracker)| tracker.calibration_gap)
    }

    /// Query observation count for a specific subsystem.
    pub fn subsystem_observations(&self, subsystem: SubsystemId) -> u32 {
        self.subsystems
            .iter()
            .find(|(id, _)| *id == subsystem)
            .map(|(_, tracker)| tracker.observations)
            .unwrap_or(0)
    }

    /// Is a specific subsystem well-calibrated?
    pub fn subsystem_is_calibrated(&self, subsystem: SubsystemId) -> bool {
        self.subsystems
            .iter()
            .find(|(id, _)| *id == subsystem)
            .map(|(_, tracker)| tracker.is_calibrated(HUMBLE_THRESHOLD))
            .unwrap_or(false)
    }

    /// Number of tracked subsystems.
    pub fn tracked_subsystems(&self) -> usize {
        self.subsystems.len()
    }

    /// Total observations across all subsystems.
    pub fn total_observations(&self) -> u64 {
        self.total_observations
    }

    /// Miscalibration direction for a specific subsystem.
    /// Positive = overconfident, negative = underconfident.
    pub fn subsystem_miscalibration(&self, subsystem: SubsystemId) -> Option<f32> {
        self.subsystems
            .iter()
            .find(|(id, _)| *id == subsystem)
            .map(|(_, tracker)| tracker.miscalibration_direction())
    }
}

fn ema_f32(prev: f32, new: f32, alpha: f32) -> f32 {
    prev + alpha * (new - prev)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_meta_defaults() {
        let m = MetaCognition::new();
        assert_eq!(m.calibration_error, 0.0);
        assert_eq!(m.meta_confidence, 1.0);
        assert!(!m.humble_mode);
        assert_eq!(m.tracked_subsystems(), 0);
    }

    #[test]
    fn test_observe_creates_subsystem() {
        let mut m = MetaCognition::new();
        m.observe(SubsystemId::RlAgent, 0.8, 0.7);
        assert_eq!(m.tracked_subsystems(), 1);
        assert_eq!(m.subsystem_observations(SubsystemId::RlAgent), 1);
    }

    #[test]
    fn test_well_calibrated_subsystem() {
        let mut m = MetaCognition::new();
        // Predicted ~= actual → well calibrated
        for _ in 0..20 {
            m.observe(SubsystemId::LinUcb, 0.70, 0.72);
        }
        m.tick();
        let gap = m.subsystem_gap(SubsystemId::LinUcb).unwrap();
        assert!(gap < 0.05, "Small prediction gap: {gap}");
        assert!(m.subsystem_is_calibrated(SubsystemId::LinUcb));
    }

    #[test]
    fn test_poorly_calibrated_subsystem() {
        let mut m = MetaCognition::new();
        // Predicted 0.9 but actual 0.2 → poorly calibrated
        for _ in 0..20 {
            m.observe(SubsystemId::CausalGraph, 0.90, 0.20);
        }
        m.tick();
        let gap = m.subsystem_gap(SubsystemId::CausalGraph).unwrap();
        assert!(gap > 0.3, "Large prediction gap: {gap}");
        assert!(!m.subsystem_is_calibrated(SubsystemId::CausalGraph));
    }

    #[test]
    fn test_humble_mode_triggers_on_high_ece() {
        let mut m = MetaCognition::new();
        // All subsystems very poorly calibrated
        for _ in 0..30 {
            m.observe(SubsystemId::RlAgent, 0.95, 0.10);
            m.observe(SubsystemId::LinUcb, 0.85, 0.15);
        }
        m.tick();
        assert!(m.humble_mode, "Should enter humble mode with high ECE");
        assert!(m.calibration_error > HUMBLE_THRESHOLD);
    }

    #[test]
    fn test_humble_mode_exploration_mult() {
        let m_normal = MetaCognition::new();
        assert_eq!(m_normal.humble_exploration_mult(), 1.0);

        let mut m = MetaCognition::new();
        for _ in 0..30 {
            m.observe(SubsystemId::RlAgent, 0.95, 0.10);
        }
        m.tick();
        assert!(m.humble_mode);
        assert_eq!(m.humble_exploration_mult(), 2.0);
    }

    #[test]
    fn test_humble_mode_freeze_floor() {
        let m_normal = MetaCognition::new();
        assert_eq!(m_normal.humble_freeze_confidence_floor(), 0.35);

        let mut m = MetaCognition::new();
        for _ in 0..30 {
            m.observe(SubsystemId::NarsBelief, 0.90, 0.10);
        }
        m.tick();
        assert!(m.humble_mode);
        assert_eq!(m.humble_freeze_confidence_floor(), 0.45);
    }

    #[test]
    fn test_humble_mode_exits_after_duration_and_improvement() {
        let mut m = MetaCognition::new();
        // Enter humble mode
        for _ in 0..30 {
            m.observe(SubsystemId::RlAgent, 0.95, 0.10);
        }
        m.tick();
        assert!(m.humble_mode);

        // Now feed well-calibrated observations and tick through duration
        for _ in 0..HUMBLE_DURATION + 1 {
            m.observe(SubsystemId::RlAgent, 0.50, 0.50);
            m.tick();
        }
        // Should eventually exit
        assert!(!m.humble_mode, "Should exit humble after duration + good calibration");
    }

    #[test]
    fn test_humble_mode_persists_if_still_miscalibrated() {
        let mut m = MetaCognition::new();
        // Enter humble mode
        for _ in 0..30 {
            m.observe(SubsystemId::RlAgent, 0.95, 0.10);
        }
        m.tick();
        assert!(m.humble_mode);

        // Tick through duration but keep miscalibrating
        for _ in 0..HUMBLE_DURATION + 1 {
            m.observe(SubsystemId::RlAgent, 0.90, 0.10);
            m.tick();
        }
        // Should stay in humble mode
        assert!(m.humble_mode, "Should stay humble while still miscalibrated");
    }

    #[test]
    fn test_meta_confidence_tracks_ece() {
        let mut m = MetaCognition::new();
        for _ in 0..20 {
            m.observe(SubsystemId::RlAgent, 0.50, 0.50);
        }
        m.tick();
        assert!(m.meta_confidence > 0.9, "Well calibrated → high meta: {}", m.meta_confidence);

        let mut m2 = MetaCognition::new();
        for _ in 0..20 {
            m2.observe(SubsystemId::RlAgent, 0.95, 0.10);
        }
        m2.tick();
        assert!(m2.meta_confidence < 0.7, "Poorly calibrated → low meta: {}", m2.meta_confidence);
    }

    #[test]
    fn test_multiple_subsystems_aggregate() {
        let mut m = MetaCognition::new();
        // One well-calibrated, one poorly calibrated
        for _ in 0..20 {
            m.observe(SubsystemId::RlAgent, 0.50, 0.50);       // good
            m.observe(SubsystemId::CausalGraph, 0.90, 0.10);   // bad
        }
        m.tick();
        // Aggregate should be somewhere in between
        assert!(m.calibration_error > 0.05);
        assert!(m.calibration_error < 0.90);
    }

    #[test]
    fn test_not_enough_observations_no_humble() {
        let mut m = MetaCognition::new();
        // Few observations with terrible calibration
        for _ in 0..3 {
            m.observe(SubsystemId::RlAgent, 0.99, 0.01);
        }
        m.tick();
        assert!(!m.humble_mode, "Not enough data to trigger humble mode");
    }

    #[test]
    fn test_miscalibration_direction_overconfident() {
        let mut m = MetaCognition::new();
        for _ in 0..20 {
            m.observe(SubsystemId::SignalKalman, 0.90, 0.30);
        }
        let dir = m.subsystem_miscalibration(SubsystemId::SignalKalman).unwrap();
        assert!(dir > 0.0, "Predicted > actual = overconfident: {dir}");
    }

    #[test]
    fn test_miscalibration_direction_underconfident() {
        let mut m = MetaCognition::new();
        for _ in 0..20 {
            m.observe(SubsystemId::SignalKalman, 0.20, 0.80);
        }
        let dir = m.subsystem_miscalibration(SubsystemId::SignalKalman).unwrap();
        assert!(dir < 0.0, "Predicted < actual = underconfident: {dir}");
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut m = MetaCognition::new();
        for _ in 0..15 {
            m.observe(SubsystemId::RlAgent, 0.80, 0.60);
            m.observe(SubsystemId::LinUcb, 0.40, 0.45);
        }
        m.tick();

        let json = serde_json::to_string(&m).expect("serialize");
        let restored: MetaCognition = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.tracked_subsystems(), m.tracked_subsystems());
        assert!((restored.calibration_error - m.calibration_error).abs() < 1e-6);
        assert_eq!(restored.humble_mode, m.humble_mode);
    }

    #[test]
    fn test_accuracy_ema_clamping() {
        let mut tracker = AccuracyEma::default();
        // Out-of-range values should be clamped
        tracker.observe(5.0, -3.0);
        assert!(tracker.predicted_ema <= 1.0);
        assert!(tracker.actual_ema >= 0.0);
    }
}
