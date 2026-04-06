//! Cognitive Reward Bus — unified cross-subsystem reward signal.
//!
//! ## Problem solved
//! RL, NARS, LinUCB, and CausalGraph each learn in isolation with independent
//! reward signals. This causes conflicting policies (RL wants X, LinUCB wants Y)
//! and slow convergence (each subsystem sees only partial signal).
//!
//! ## Design
//! CognitiveRewardBus collects reward signals from all learning subsystems,
//! normalizes them using PPO-style tanh scaling [Schulman 2017], and broadcasts
//! integrated reward to downstream consumers.
//!
//! ## References
//! - [Schulman 2017] "Proximal Policy Optimization Algorithms" arXiv:1707.06347
//!   §3.2: reward normalization via running statistics
//! - [Yuan 2024] "Self-Rewarding Language Models" arXiv:2401.10020
//!   §3: self-generated training signal without external oracle

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Maximum number of reward signals retained in the bus buffer.
const BUS_CAPACITY: usize = 200;

/// EMA smoothing factor for per-source reward tracking.
const REWARD_EMA_ALPHA: f64 = 0.05;

/// Minimum std_dev for normalization (prevents division by near-zero).
const MIN_STD_DEV: f64 = 0.01;

// ── Types ──────────────────────────────────────────────────────────────────────

/// Source of a reward signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RewardSource {
    /// OutcomeTracker: Bayesian effectiveness feedback.
    Outcome,
    /// RlThresholdAgent: Q-value improvement signal.
    RlAgent,
    /// CausalGraph: causal confidence change.
    CausalGraph,
    /// SelfRewardingEvaluator: retroactive decision quality.
    SelfEval,
    /// MetaCognition: calibration improvement signal.
    MetaCognition,
}

/// A single reward signal emitted by a learning subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardSignal {
    /// Which subsystem produced this signal.
    pub source: RewardSource,
    /// Raw reward value in [-1, +1].
    pub value: f64,
    /// Confidence weight of the emitting subsystem [0, 1].
    pub confidence: f32,
    /// Daemon cycle when this signal was produced.
    pub cycle: u64,
}

/// Running statistics for PPO-style normalization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RunningStats {
    mean: f64,
    var: f64,
    count: u64,
}

impl RunningStats {
    /// Welford online update [Welford 1962].
    fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.var += delta * delta2;
    }

    fn std_dev(&self) -> f64 {
        if self.count < 2 {
            return 1.0;
        }
        (self.var / (self.count - 1) as f64).sqrt().max(MIN_STD_DEV)
    }

    fn normalize(&self, x: f64) -> f64 {
        if self.count < 5 {
            // Cold-start: not enough samples for meaningful statistics.
            // Pass through raw value (clamped to [-1,1] upstream).
            return x;
        }
        (x - self.mean) / self.std_dev()
    }
}

/// Unified cross-subsystem reward bus.
///
/// Collects reward signals from all learning subsystems, normalizes them,
/// and provides integrated reward EMAs per downstream consumer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveRewardBus {
    /// Ring buffer of recent reward signals.
    signals: VecDeque<RewardSignal>,
    /// Running statistics for PPO-style normalization.
    stats: RunningStats,
    /// EMA of integrated reward for RL agent.
    rl_reward_ema: f64,
    /// EMA of integrated reward for LinUCB agent.
    linucb_reward_ema: f64,
    /// EMA of integrated reward for NARS beliefs.
    nars_reward_ema: f64,
    /// Weighted sum of all signals this cycle (before EMA).
    current_cycle_sum: f64,
    /// Number of signals received in current cycle.
    current_cycle_count: u32,
    /// Total signals ever processed.
    total_signals: u64,
}

impl Default for CognitiveRewardBus {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveRewardBus {
    pub fn new() -> Self {
        Self {
            signals: VecDeque::with_capacity(BUS_CAPACITY),
            stats: RunningStats::default(),
            rl_reward_ema: 0.0,
            linucb_reward_ema: 0.0,
            nars_reward_ema: 0.0,
            current_cycle_sum: 0.0,
            current_cycle_count: 0,
            total_signals: 0,
        }
    }

    /// Publish a reward signal to the bus.
    ///
    /// The signal is normalized via running statistics and added to the buffer.
    /// Call `flush_cycle()` at the end of each daemon cycle to propagate to EMAs.
    pub fn publish(&mut self, signal: RewardSignal) {
        let raw = signal.value.clamp(-1.0, 1.0);
        let conf = signal.confidence.clamp(0.0, 1.0) as f64;

        // Update running stats for normalization
        self.stats.update(raw);

        // PPO-style tanh normalization [Schulman 2017 §3.2]
        let normalized = (self.stats.normalize(raw)).tanh();
        let weighted = normalized * conf;

        self.current_cycle_sum += weighted;
        self.current_cycle_count += 1;
        self.total_signals += 1;

        // Store in ring buffer
        if self.signals.len() >= BUS_CAPACITY {
            self.signals.pop_front();
        }
        self.signals.push_back(signal);
    }

    /// Flush the current cycle's accumulated signals into downstream EMAs.
    ///
    /// Call this once per daemon cycle after all subsystems have published.
    /// The integrated reward is the confidence-weighted mean of all signals
    /// received this cycle, then distributed to per-consumer EMAs.
    pub fn flush_cycle(&mut self) {
        if self.current_cycle_count == 0 {
            return;
        }

        let integrated = self.current_cycle_sum / self.current_cycle_count as f64;

        // Update per-consumer EMAs
        self.rl_reward_ema = ema(self.rl_reward_ema, integrated, REWARD_EMA_ALPHA);
        self.linucb_reward_ema = ema(self.linucb_reward_ema, integrated, REWARD_EMA_ALPHA);
        self.nars_reward_ema = ema(self.nars_reward_ema, integrated, REWARD_EMA_ALPHA);

        // Reset cycle accumulator
        self.current_cycle_sum = 0.0;
        self.current_cycle_count = 0;
    }

    /// Integrated reward EMA for the RL threshold agent.
    /// Positive = system improving, negative = degrading.
    pub fn rl_reward(&self) -> f64 {
        self.rl_reward_ema
    }

    /// Integrated reward EMA for the LinUCB predictive agent.
    pub fn linucb_reward(&self) -> f64 {
        self.linucb_reward_ema
    }

    /// Integrated reward EMA for NARS belief weighting.
    pub fn nars_reward(&self) -> f64 {
        self.nars_reward_ema
    }

    /// Total number of signals ever processed.
    pub fn total_signals(&self) -> u64 {
        self.total_signals
    }

    /// Number of signals in the ring buffer.
    pub fn buffer_len(&self) -> usize {
        self.signals.len()
    }

    /// Signal-to-noise ratio: |mean| / std_dev.
    /// High SNR = consistent reward direction. Low SNR = conflicting signals.
    pub fn signal_to_noise(&self) -> f64 {
        let sd = self.stats.std_dev();
        if sd < MIN_STD_DEV {
            return 0.0;
        }
        (self.stats.mean.abs() / sd).min(10.0)
    }

    /// Mean reward value across all observed signals.
    pub fn mean_reward(&self) -> f64 {
        self.stats.mean
    }

    /// Recent signals from a specific source (newest first), limited to `n`.
    pub fn recent_from(&self, source: RewardSource, n: usize) -> Vec<&RewardSignal> {
        self.signals
            .iter()
            .rev()
            .filter(|s| s.source == source)
            .take(n)
            .collect()
    }

    /// Prune signals older than `max_cycle` cycles from current.
    pub fn prune_old(&mut self, current_cycle: u64, max_age: u64) {
        let cutoff = current_cycle.saturating_sub(max_age);
        while let Some(front) = self.signals.front() {
            if front.cycle < cutoff {
                self.signals.pop_front();
            } else {
                break;
            }
        }
    }
}

/// Simple EMA helper.
fn ema(prev: f64, new: f64, alpha: f64) -> f64 {
    prev + alpha * (new - prev)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(source: RewardSource, value: f64, confidence: f32, cycle: u64) -> RewardSignal {
        RewardSignal {
            source,
            value,
            confidence,
            cycle,
        }
    }

    #[test]
    fn test_new_bus_defaults() {
        let bus = CognitiveRewardBus::new();
        assert_eq!(bus.total_signals(), 0);
        assert_eq!(bus.buffer_len(), 0);
        assert_eq!(bus.rl_reward(), 0.0);
        assert_eq!(bus.linucb_reward(), 0.0);
        assert_eq!(bus.nars_reward(), 0.0);
    }

    #[test]
    fn test_publish_and_flush_positive_reward() {
        let mut bus = CognitiveRewardBus::new();
        bus.publish(sig(RewardSource::Outcome, 0.8, 1.0, 1));
        bus.flush_cycle();

        assert_eq!(bus.total_signals(), 1);
        // After one positive signal, all EMAs should be positive
        assert!(bus.rl_reward() > 0.0, "rl_reward={}", bus.rl_reward());
        assert!(
            bus.linucb_reward() > 0.0,
            "linucb_reward={}",
            bus.linucb_reward()
        );
        assert!(
            bus.nars_reward() > 0.0,
            "nars_reward={}",
            bus.nars_reward()
        );
    }

    #[test]
    fn test_publish_and_flush_negative_reward() {
        let mut bus = CognitiveRewardBus::new();
        bus.publish(sig(RewardSource::RlAgent, -0.5, 0.9, 1));
        bus.flush_cycle();

        // Negative signal → negative EMAs
        assert!(bus.rl_reward() < 0.0);
    }

    #[test]
    fn test_multiple_signals_weighted_by_confidence() {
        let mut bus = CognitiveRewardBus::new();
        // High confidence positive
        bus.publish(sig(RewardSource::Outcome, 0.9, 1.0, 1));
        // Low confidence negative (should be dominated by positive)
        bus.publish(sig(RewardSource::CausalGraph, -0.3, 0.1, 1));
        bus.flush_cycle();

        assert!(bus.rl_reward() > 0.0, "High-confidence positive should dominate");
    }

    #[test]
    fn test_flush_without_signals_is_noop() {
        let mut bus = CognitiveRewardBus::new();
        bus.flush_cycle();
        assert_eq!(bus.rl_reward(), 0.0);
        assert_eq!(bus.total_signals(), 0);
    }

    #[test]
    fn test_ring_buffer_capacity() {
        let mut bus = CognitiveRewardBus::new();
        for i in 0..BUS_CAPACITY + 50 {
            bus.publish(sig(RewardSource::Outcome, 0.5, 1.0, i as u64));
        }
        assert_eq!(bus.buffer_len(), BUS_CAPACITY);
        assert_eq!(bus.total_signals(), (BUS_CAPACITY + 50) as u64);
    }

    #[test]
    fn test_signal_clamping() {
        let mut bus = CognitiveRewardBus::new();
        // Value > 1.0 should clamp to 1.0
        bus.publish(sig(RewardSource::Outcome, 5.0, 2.0, 1));
        bus.flush_cycle();
        // Should not panic or produce NaN
        assert!(bus.rl_reward().is_finite());
    }

    #[test]
    fn test_ema_convergence_over_many_cycles() {
        let mut bus = CognitiveRewardBus::new();
        for i in 0..100 {
            bus.publish(sig(RewardSource::Outcome, 0.5, 1.0, i));
            bus.flush_cycle();
        }
        // EMA should converge toward the normalized tanh of 0.5
        let r = bus.rl_reward();
        assert!(r > 0.0 && r < 1.0, "Should converge to positive: {r}");
    }

    #[test]
    fn test_signal_to_noise_consistent_signals() {
        let mut bus = CognitiveRewardBus::new();
        for i in 0..50 {
            bus.publish(sig(RewardSource::Outcome, 0.7, 1.0, i));
        }
        // Consistent positive signals → high SNR
        let snr = bus.signal_to_noise();
        assert!(snr > 1.0, "Consistent signals should yield high SNR: {snr}");
    }

    #[test]
    fn test_signal_to_noise_conflicting_signals() {
        let mut bus = CognitiveRewardBus::new();
        for i in 0..50 {
            let val = if i % 2 == 0 { 0.8 } else { -0.8 };
            bus.publish(sig(RewardSource::Outcome, val, 1.0, i));
        }
        // Conflicting signals → low SNR (mean near 0, high variance)
        let snr = bus.signal_to_noise();
        assert!(snr < 1.0, "Conflicting signals should yield low SNR: {snr}");
    }

    #[test]
    fn test_recent_from_filter() {
        let mut bus = CognitiveRewardBus::new();
        bus.publish(sig(RewardSource::Outcome, 0.5, 1.0, 1));
        bus.publish(sig(RewardSource::RlAgent, 0.3, 0.8, 2));
        bus.publish(sig(RewardSource::Outcome, 0.7, 1.0, 3));

        let outcome_sigs = bus.recent_from(RewardSource::Outcome, 10);
        assert_eq!(outcome_sigs.len(), 2);
        // Most recent first
        assert!((outcome_sigs[0].value - 0.7).abs() < 0.001);

        let rl_sigs = bus.recent_from(RewardSource::RlAgent, 10);
        assert_eq!(rl_sigs.len(), 1);
    }

    #[test]
    fn test_prune_old_signals() {
        let mut bus = CognitiveRewardBus::new();
        for i in 0..20 {
            bus.publish(sig(RewardSource::Outcome, 0.5, 1.0, i));
        }
        assert_eq!(bus.buffer_len(), 20);

        bus.prune_old(19, 10);
        // Should keep cycles 10..=19 (10 signals)
        assert!(bus.buffer_len() <= 11, "After prune: {}", bus.buffer_len());
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut bus = CognitiveRewardBus::new();
        bus.publish(sig(RewardSource::Outcome, 0.5, 1.0, 1));
        bus.publish(sig(RewardSource::CausalGraph, -0.3, 0.8, 2));
        bus.flush_cycle();

        let json = serde_json::to_string(&bus).expect("serialize");
        let restored: CognitiveRewardBus = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.total_signals(), bus.total_signals());
        assert!((restored.rl_reward() - bus.rl_reward()).abs() < 1e-10);
        assert_eq!(restored.buffer_len(), bus.buffer_len());
    }

    #[test]
    fn test_cross_feed_outcome_to_rl() {
        let mut bus = CognitiveRewardBus::new();
        // Simulate OutcomeTracker publishing effective results
        for i in 0..20 {
            bus.publish(sig(RewardSource::Outcome, 0.6, 0.9, i));
            bus.flush_cycle();
        }
        // RL agent should see positive reward from outcome feedback
        assert!(bus.rl_reward() > 0.0);
        // And it should have moved meaningfully from 0
        assert!(bus.rl_reward() > 0.01, "rl_reward={}", bus.rl_reward());
    }

    #[test]
    fn test_mean_reward_tracks_direction() {
        let mut bus = CognitiveRewardBus::new();
        for i in 0..30 {
            bus.publish(sig(RewardSource::Outcome, 0.8, 1.0, i));
        }
        assert!(bus.mean_reward() > 0.5, "mean={}", bus.mean_reward());

        let mut bus_neg = CognitiveRewardBus::new();
        for i in 0..30 {
            bus_neg.publish(sig(RewardSource::RlAgent, -0.6, 1.0, i));
        }
        assert!(bus_neg.mean_reward() < -0.3, "mean={}", bus_neg.mean_reward());
    }
}
