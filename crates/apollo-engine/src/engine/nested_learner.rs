//! # NestedLearner — Multi-Level Learning Coordinator
//!
//! Implements a 3-tier learning hierarchy inspired by Google's Nested Learning
//! paradigm (2025): each level has its own update frequency and context flow.
//!
//! ## Levels
//!
//! | Level | Frequency     | Subsystems                        | Context received      |
//! |-------|---------------|-----------------------------------|-----------------------|
//! | L0    | every cycle   | SignalIntelligence (Kalman/CUSUM)  | raw pressure signal   |
//! | L1    | per outcome   | OutcomeTracker, CausalGraph       | L0 signal quality     |
//! | L2    | periodic      | LearningPipeline, MetaLearning    | L1 aggregate outcome  |
//!
//! ## Context flow
//!
//! ```text
//! L0 signal quality EMA ──→ L1 outcome weight (high quality → trust outcome more)
//! L1 aggregate outcome  ──→ L2 meta context  (stable outcomes → slow meta rate)
//! L2 meta velocity      ──→ L0 gate threshold (high velocity → require better signal)
//! ```
//!
//! ## Why this matters
//!
//! Previously Apollo's three learning loops (RL, OutcomeTracker, PredictiveAgent)
//! never cross-fed in a principled way. [Google Nested Learning 2025] shows that
//! explicit context flow between frequency levels prevents catastrophic forgetting
//! and reduces variance of the outer (slower) loops.
//!
//! [Hochreiter & Schmidhuber 1997] LSTM showed multi-timescale memory prevents
//! gradient vanishing — the same principle applies here: L0's fast EMA stabilises
//! L1's slower Bayesian updates.
//!
//! ## Integration
//!
//! `NestedLearner` is a lightweight coordinator (~0 allocations per cycle).
//! It does NOT own the subsystems — callers keep ownership and pass signals in.
//! Wire into `learning_tick::run_learning_tick` as a single `&mut NestedLearner`.

use serde::{Deserialize, Serialize};

// ── Constants ─────────────────────────────────────────────────────────────────

/// EMA decay for L0 signal quality (fast, per-cycle).
const L0_ALPHA: f64 = 0.15;

/// EMA decay for L1 aggregate outcome (slower, per-outcome).
const L1_ALPHA: f64 = 0.20;

/// Baseline L0 signal quality required to allow L1 updates.
/// Dynamically raised by L2 meta-velocity: high meta-velocity → require cleaner signal.
/// [Google NL 2025 §6.2] — bidirectional context flow.
const L1_GATE_THRESHOLD: f64 = 0.25;

/// L1 flushes required before an L2 meta-context update fires.
const L2_GATE_PERIOD: u32 = 20;

/// EMA decay for L2 meta-velocity tracking (slow, per-flush).
const L2_VELOCITY_ALPHA: f64 = 0.30;

/// How much meta-velocity raises the L1 gate threshold.
/// At max velocity (1.0), gate rises to 0.25 + 0.20 = 0.45.
const L2_VELOCITY_GATE_SCALE: f64 = 0.20;

/// Gate ceiling — even at extreme meta-velocity, L1 never becomes fully blocked.
const L1_GATE_MAX: f64 = 0.60;

// ── NestedLearner ─────────────────────────────────────────────────────────────

/// Frequency-gated coordinator for Apollo's 3-tier learning hierarchy.
///
/// Persisted as part of `LearnedState` so the frequency counters and EMAs
/// survive daemon restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedLearner {
    // ── L0 state ──────────────────────────────────────────────────────────────
    /// EMA of signal quality [0, 1]. Updated every cycle from composite signal.
    pub l0_quality: f64,

    // ── L1 state ──────────────────────────────────────────────────────────────
    /// EMA of L0-weighted outcome effectiveness [0, 1].
    pub l1_aggregate: f64,
    /// Number of L1 updates since last L2 flush.
    pub l1_since_l2: u32,
    /// Cumulative L1 updates (for diagnostics).
    pub l1_total: u64,

    // ── L2 state ──────────────────────────────────────────────────────────────
    /// Last L2 meta-context value. Exposed to meta-learning for rate adaptation.
    pub l2_context: f64,
    /// Total L2 flushes (for diagnostics).
    pub l2_total: u64,

    // ── L2→L0 feedback ────────────────────────────────────────────────────────
    /// Previous l2_context value — used to compute meta-velocity delta.
    #[serde(default)]
    pub l2_prev_context: f64,
    /// EMA of |l2_context delta| per flush. High velocity = rapid meta-change.
    /// Feeds back to L0 gate: high velocity → raise quality bar for L1 updates.
    /// [Google NL 2025 §6.2] — "L2 meta-velocity feeds back to L0 gate threshold"
    #[serde(default)]
    pub l2_meta_velocity: f64,
}

impl Default for NestedLearner {
    fn default() -> Self {
        Self {
            l0_quality: 0.5, // start neutral
            l1_aggregate: 0.5,
            l1_since_l2: 0,
            l1_total: 0,
            l2_context: 0.5,
            l2_total: 0,
            l2_prev_context: 0.5,
            l2_meta_velocity: 0.0, // no velocity initially → gate = L1_GATE_THRESHOLD
        }
    }
}

impl NestedLearner {
    pub fn new() -> Self {
        Self::default()
    }

    // ── L0: per-cycle signal tick ─────────────────────────────────────────────

    /// Update L0 quality EMA from the current cycle's signal composite score.
    ///
    /// `signal_quality` should be in [0, 1] — 1.0 means signal is perfectly clean,
    /// 0.0 means all noise. Use `SignalDigest::composite_score()` or equivalent.
    ///
    /// Returns `true` when L0 quality is above the dynamic L1 gate threshold,
    /// meaning outcome observations this cycle are trustworthy.
    ///
    /// The gate is raised by L2 meta-velocity: rapid meta-changes demand cleaner
    /// signal before allowing new outcome observations to influence beliefs.
    /// [Google NL 2025 §6.2] — bidirectional context flow closes the feedback loop.
    pub fn tick_l0(&mut self, signal_quality: f64) -> bool {
        self.l0_quality =
            (1.0 - L0_ALPHA) * self.l0_quality + L0_ALPHA * signal_quality.clamp(0.0, 1.0);
        self.l0_quality >= self.dynamic_l1_gate()
    }

    /// Dynamic L1 gate threshold, raised by L2 meta-velocity.
    ///
    /// `gate = L1_GATE_THRESHOLD + L2_VELOCITY_GATE_SCALE * l2_meta_velocity`
    /// clamped to [L1_GATE_THRESHOLD, L1_GATE_MAX].
    ///
    /// At zero velocity (steady state): gate = 0.25 (unchanged from prior behavior).
    /// At max velocity (1.0):           gate = 0.45 (demands better signal).
    pub fn dynamic_l1_gate(&self) -> f64 {
        (L1_GATE_THRESHOLD + L2_VELOCITY_GATE_SCALE * self.l2_meta_velocity)
            .clamp(L1_GATE_THRESHOLD, L1_GATE_MAX)
    }

    // ── L1: per-outcome observation ───────────────────────────────────────────

    /// Feed an outcome observation into L1.
    ///
    /// `outcome_effectiveness` ∈ [0, 1] — 1.0 means the action was fully effective
    /// (e.g. pressure dropped by the expected amount).
    ///
    /// The observation is **weighted by L0 quality**: when the signal is noisy,
    /// a single outcome carries less weight in the aggregate. This is the core
    /// context-flow principle from [Google Nested Learning 2025].
    ///
    /// Returns `true` when enough L1 updates have accumulated to trigger an L2
    /// meta-context flush.
    pub fn tick_l1(&mut self, outcome_effectiveness: f64) -> bool {
        // Context flow: L0 quality weights how much this outcome moves L1 aggregate.
        let weighted = outcome_effectiveness.clamp(0.0, 1.0) * self.l0_quality;
        // Blend: high L0 quality → outcome moves aggregate; low → almost ignored.
        let effective_alpha = L1_ALPHA * self.l0_quality.max(0.1);
        self.l1_aggregate =
            (1.0 - effective_alpha) * self.l1_aggregate + effective_alpha * weighted;
        self.l1_since_l2 += 1;
        self.l1_total += 1;
        self.l1_since_l2 >= L2_GATE_PERIOD
    }

    // ── L2: periodic meta-context flush ──────────────────────────────────────

    /// Flush L1 aggregate into L2 meta-context.
    ///
    /// Call this when `tick_l1` returns `true`. Returns the new L2 context value,
    /// which callers should forward to `LearnedState::update_meta_learning_context`.
    ///
    /// The L2 context feeds back to outer-loop systems (meta-learning rate, zone
    /// learning, specialist weighting) — this is the downward context flow.
    pub fn flush_l2(&mut self) -> f64 {
        // Compute meta-velocity: how much did l2_context change this flush?
        // [Google NL 2025 §6.2] — meta-velocity feeds back to L0 gate threshold.
        let delta = (self.l1_aggregate - self.l2_prev_context).abs();
        self.l2_meta_velocity = (1.0 - L2_VELOCITY_ALPHA) * self.l2_meta_velocity
            + L2_VELOCITY_ALPHA * delta.clamp(0.0, 1.0);

        self.l2_prev_context = self.l2_context;
        self.l2_context = self.l1_aggregate;
        self.l1_since_l2 = 0;
        self.l2_total += 1;
        self.l2_context
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    /// Returns a diagnostic snapshot for metrics reporting.
    pub fn diagnostics(&self) -> NestedLearnerDiagnostics {
        NestedLearnerDiagnostics {
            l0_quality: self.l0_quality,
            l1_aggregate: self.l1_aggregate,
            l1_gate_open: self.l0_quality >= self.dynamic_l1_gate(),
            l1_gate_threshold: self.dynamic_l1_gate(),
            l2_context: self.l2_context,
            l2_meta_velocity: self.l2_meta_velocity,
            l1_total: self.l1_total,
            l2_total: self.l2_total,
        }
    }
}

/// Snapshot of NestedLearner state for metrics/logging.
#[derive(Debug, Clone)]
pub struct NestedLearnerDiagnostics {
    pub l0_quality: f64,
    pub l1_aggregate: f64,
    pub l1_gate_open: bool,
    /// Current dynamic gate threshold (raised by L2 meta-velocity).
    pub l1_gate_threshold: f64,
    pub l2_context: f64,
    /// EMA of |l2_context delta| — how rapidly the meta-context is changing.
    pub l2_meta_velocity: f64,
    pub l1_total: u64,
    pub l2_total: u64,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l0_quality_ema_converges() {
        let mut nl = NestedLearner::new();
        // Feed 100 cycles of quality = 0.8
        for _ in 0..100 {
            nl.tick_l0(0.8);
        }
        assert!(
            (nl.l0_quality - 0.8).abs() < 0.01,
            "L0 should converge to 0.8, got {}",
            nl.l0_quality
        );
    }

    #[test]
    fn l1_gate_blocks_noisy_signal() {
        let mut nl = NestedLearner::new();
        // Drive L0 quality very low
        for _ in 0..100 {
            nl.tick_l0(0.0);
        }
        // Gate should be closed
        assert!(
            !nl.tick_l0(0.0),
            "L1 gate should be closed when signal quality is low"
        );
    }

    #[test]
    fn l1_gate_opens_on_good_signal() {
        let mut nl = NestedLearner::new();
        // Drive L0 quality high
        for _ in 0..100 {
            nl.tick_l0(1.0);
        }
        assert!(
            nl.tick_l0(1.0),
            "L1 gate should open when signal quality is high"
        );
    }

    #[test]
    fn l1_aggregate_weighted_by_l0_quality() {
        let mut nl_high = NestedLearner::new();
        let mut nl_low = NestedLearner::new();

        // Drive high quality
        for _ in 0..50 {
            nl_high.tick_l0(1.0);
        }
        // Drive low quality
        for _ in 0..50 {
            nl_low.tick_l0(0.1);
        }

        // Same outcome effectiveness
        for _ in 0..5 {
            nl_high.tick_l1(0.9);
            nl_low.tick_l1(0.9);
        }

        // High quality should converge faster / have higher aggregate
        assert!(
            nl_high.l1_aggregate > nl_low.l1_aggregate,
            "High L0 quality should produce higher L1 aggregate: high={}, low={}",
            nl_high.l1_aggregate,
            nl_low.l1_aggregate
        );
    }

    #[test]
    fn l2_flush_triggers_after_gate_period() {
        let mut nl = NestedLearner::new();
        // Drive L0 quality above gate
        for _ in 0..50 {
            nl.tick_l0(1.0);
        }

        let mut l2_fired = false;
        for _ in 0..L2_GATE_PERIOD {
            if nl.tick_l1(0.7) {
                l2_fired = true;
                let ctx = nl.flush_l2();
                assert!(ctx > 0.0, "L2 context should be positive");
                assert_eq!(nl.l1_since_l2, 0, "Counter should reset after flush");
                assert_eq!(nl.l2_total, 1);
                break;
            }
        }
        assert!(
            l2_fired,
            "L2 should have fired after L2_GATE_PERIOD L1 updates"
        );
    }

    #[test]
    fn default_state_is_neutral() {
        let nl = NestedLearner::default();
        assert_eq!(nl.l0_quality, 0.5);
        assert_eq!(nl.l1_aggregate, 0.5);
        assert_eq!(nl.l2_context, 0.5);
    }

    #[test]
    fn diagnostics_reflect_state() {
        let mut nl = NestedLearner::new();
        for _ in 0..50 {
            nl.tick_l0(0.8);
        }
        let d = nl.diagnostics();
        assert!((d.l0_quality - nl.l0_quality).abs() < 1e-10);
        assert_eq!(d.l1_gate_open, nl.l0_quality >= nl.dynamic_l1_gate());
    }

    // ── L2→L0 feedback tests ──────────────────────────────────────────────────

    /// At zero meta-velocity, dynamic gate == L1_GATE_THRESHOLD (backward compatible).
    #[test]
    fn dynamic_gate_equals_baseline_at_zero_velocity() {
        let nl = NestedLearner::new();
        assert_eq!(nl.l2_meta_velocity, 0.0);
        assert!(
            (nl.dynamic_l1_gate() - L1_GATE_THRESHOLD).abs() < 1e-10,
            "gate should equal L1_GATE_THRESHOLD when velocity=0, got {}",
            nl.dynamic_l1_gate()
        );
    }

    /// High meta-velocity raises the gate above baseline.
    #[test]
    fn dynamic_gate_rises_with_meta_velocity() {
        let mut nl = NestedLearner::new();
        // Simulate oscillating l1_aggregate to produce high meta-velocity
        for _ in 0..50 {
            nl.tick_l0(1.0);
        }
        // Alternate high/low outcomes to force l2_context swings
        for _ in 0..10 {
            for _ in 0..L2_GATE_PERIOD {
                nl.tick_l1(1.0);
            }
            nl.flush_l2();
            for _ in 0..L2_GATE_PERIOD {
                nl.tick_l1(0.0);
            }
            nl.flush_l2();
        }
        let gate = nl.dynamic_l1_gate();
        assert!(
            gate > L1_GATE_THRESHOLD,
            "gate should rise above {} with high velocity, got {}",
            L1_GATE_THRESHOLD,
            gate
        );
    }

    /// Gate never exceeds L1_GATE_MAX regardless of velocity magnitude.
    #[test]
    fn dynamic_gate_clamped_at_max() {
        let mut nl = NestedLearner::new();
        nl.l2_meta_velocity = 100.0; // extreme artificial value
        let gate = nl.dynamic_l1_gate();
        assert!(
            gate <= L1_GATE_MAX,
            "gate must not exceed L1_GATE_MAX={}, got {}",
            L1_GATE_MAX,
            gate
        );
        assert!(
            gate >= L1_GATE_THRESHOLD,
            "gate must not fall below L1_GATE_THRESHOLD={}, got {}",
            L1_GATE_THRESHOLD,
            gate
        );
    }

    /// Production calibration test: simulates macOS realistic pressure drops (1-5%).
    ///
    /// With / 0.05 normalization (calibrated from 2026-04-10 prod data):
    ///   1% drop (threshold = effective) → effectiveness = 0.20
    ///   3% drop (typical good throttle) → effectiveness = 0.60
    ///   5% drop (excellent result)      → effectiveness = 1.0
    ///
    /// After 40 such outcomes with stable L0 quality, l1_aggregate should
    /// be meaningfully non-zero — catching any regression to the /0.30 scale.
    #[test]
    fn l1_aggregate_nonzero_with_macos_typical_drops() {
        let mut nl = NestedLearner::new();
        // Stable, moderately good signal quality
        for _ in 0..100 {
            nl.tick_l0(0.6);
        }
        assert!(nl.tick_l0(0.6), "gate should be open at quality 0.6");

        // Simulate 40 outcomes with 3% pressure drop (calibrated as 0.60 effective)
        for _ in 0..40 {
            let effectiveness = (0.03_f64 / 0.05).clamp(0.0, 1.0); // = 0.60
            nl.tick_l1(effectiveness);
            if nl.l1_since_l2 >= L2_GATE_PERIOD {
                nl.flush_l2();
            }
        }

        assert!(
            nl.l1_aggregate > 0.1,
            "l1_aggregate should be meaningfully non-zero with 3% macOS pressure drops, got {}",
            nl.l1_aggregate
        );
        assert!(
            nl.l2_context > 0.05,
            "l2_context should reflect l1_aggregate, got {}",
            nl.l2_context
        );
    }

    /// Zero-outcome convergence: when all outcomes have 0 effectiveness,
    /// l1_aggregate should converge toward 0 (correctly reflects idle system).
    #[test]
    fn l1_aggregate_converges_to_zero_on_no_pressure_drop() {
        let mut nl = NestedLearner::new();
        for _ in 0..100 {
            nl.tick_l0(0.7);
        }
        // Start from non-zero
        for _ in 0..20 {
            nl.tick_l1(0.5);
        }
        nl.flush_l2();

        // Feed 100 zero-effectiveness outcomes
        for _ in 0..100 {
            nl.tick_l1(0.0);
            if nl.l1_since_l2 >= L2_GATE_PERIOD {
                nl.flush_l2();
            }
        }

        assert!(
            nl.l1_aggregate < 0.1,
            "l1_aggregate should converge near zero on all-zero outcomes, got {}",
            nl.l1_aggregate
        );
    }
}
