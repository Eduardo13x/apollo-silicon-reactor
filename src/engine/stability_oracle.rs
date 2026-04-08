//! StabilityOracle — aggregate perceptual stability signals into a composite
//! negative reward for the RL threshold learner.
//!
//! Three independent signals measure stability from the user's perspective:
//! 1. **Display jank** — did the display freeze/unfreeze cycle this tick?
//!    (measured from `DisplayTurbo::tick()` returning `DeactivateTurbo`)
//! 2. **Zombie rate** — how many zombie processes appeared this cycle?
//!    (from `heuristic_stats.zombies_detected`)
//! 3. **Swap spike** — did swap usage jump by ≥512 MB in a single cycle?
//!    (from `snapshot.pressure.swap_used_bytes` delta)
//!
//! These are aggregated via EMAs into a stability score [0, 1]:
//! - 1.0 = fully stable (no jank, no zombies, no swap spikes)
//! - 0.0 = completely unstable
//!
//! The `instability_penalty()` (= 1 − score, scaled to [-1, 0]) is injected
//! into the RL system via `NeuroSignals::outcome_penalty` each cycle.
//!
//! ## References
//!
//! - Schulman et al. 2017, "Proximal Policy Optimization" — per-cycle reward
//!   signal guides policy toward stable system states.
//! - Nygard 2018, "Release It!" Ch.4 — stability anti-patterns: cascading
//!   failures, integration points, resource pools. Jank + zombie + swap spike
//!   together signal cascading instability, not just isolated noise.
//! - Kuncheva 2004 — EMA for non-stationary signal tracking.

/// EMA smoothing factor for stability signals (α ≈ 0.05 → ~20-sample window).
const ALPHA: f64 = 0.05;

/// Swap spike threshold: 512 MB jump in a single cycle.
const SWAP_SPIKE_THRESHOLD_BYTES: u64 = 512 * 1024 * 1024;

/// StabilityOracle aggregates three perceptual stability signals.
pub struct StabilityOracle {
    /// EMA of display-jank events (0=no jank, 1=jank).
    jank_ema: f64,
    /// EMA of zombie count (normalised: count / 5, capped at 1).
    zombie_rate_ema: f64,
    /// EMA of swap spike events (0=no spike, 1=spike).
    swap_spike_ema: f64,
    /// Previous cycle's swap usage (bytes) for spike detection.
    prev_swap_bytes: Option<u64>,
}

impl StabilityOracle {
    pub fn new() -> Self {
        Self {
            jank_ema: 0.0,
            zombie_rate_ema: 0.0,
            swap_spike_ema: 0.0,
            prev_swap_bytes: None,
        }
    }

    /// Record whether display jank occurred this cycle.
    ///
    /// Call after `DisplayTurbo::tick()`.  Pass `true` when the turbo just
    /// deactivated (display came back on): rapid freeze/unfreeze events imply
    /// that daemon was too aggressive and the user noticed.
    pub fn record_display_jank(&mut self, had_jank: bool) {
        let signal = if had_jank { 1.0 } else { 0.0 };
        self.jank_ema = ALPHA * signal + (1.0 - ALPHA) * self.jank_ema;
    }

    /// Record the zombie count this cycle.
    ///
    /// Normalised as `count / 5.0` capped at 1.  Five zombie processes per
    /// cycle is treated as fully unstable from a zombie-rate perspective.
    pub fn record_zombie_count(&mut self, count: usize) {
        let signal = (count as f64 / 5.0).min(1.0);
        self.zombie_rate_ema = ALPHA * signal + (1.0 - ALPHA) * self.zombie_rate_ema;
    }

    /// Record current swap usage and detect spikes.
    ///
    /// A spike is a single-cycle increase ≥ `SWAP_SPIKE_THRESHOLD_BYTES`.
    /// This distinguishes gradual swap growth (expected) from sudden spikes
    /// (process exploded or OOM-adjacent).
    pub fn record_swap_bytes(&mut self, swap_used: u64) {
        let had_spike = match self.prev_swap_bytes {
            Some(prev) => swap_used.saturating_sub(prev) >= SWAP_SPIKE_THRESHOLD_BYTES,
            None => false,
        };
        self.prev_swap_bytes = Some(swap_used);
        let signal = if had_spike { 1.0 } else { 0.0 };
        self.swap_spike_ema = ALPHA * signal + (1.0 - ALPHA) * self.swap_spike_ema;
    }

    /// Composite stability score ∈ [0, 1].  Higher = more stable.
    ///
    /// Equal-weight average of three inverted-EMA signals.
    pub fn stability_score(&self) -> f64 {
        let instability = (self.jank_ema + self.zombie_rate_ema + self.swap_spike_ema) / 3.0;
        (1.0 - instability).clamp(0.0, 1.0)
    }

    /// Instability penalty ∈ [0, 1].  0 = stable, 1 = maximally unstable.
    ///
    /// Suitable for injection into RL as a negative addend.  Scale chosen so
    /// that a single jank event (EMA ~0.05) adds ≈-0.017 to the reward —
    /// mild enough not to override the overflow penalty (-10), but persistent
    /// enough to steer over time.
    pub fn instability_penalty(&self) -> f64 {
        1.0 - self.stability_score()
    }

    /// Current jank EMA (for diagnostics/metrics).
    pub fn jank_ema(&self) -> f64 {
        self.jank_ema
    }

    /// Current zombie-rate EMA (for diagnostics/metrics).
    pub fn zombie_rate_ema(&self) -> f64 {
        self.zombie_rate_ema
    }

    /// Current swap-spike EMA (for diagnostics/metrics).
    pub fn swap_spike_ema(&self) -> f64 {
        self.swap_spike_ema
    }
}

impl Default for StabilityOracle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_system_has_score_one() {
        let mut oracle = StabilityOracle::new();
        for _ in 0..50 {
            oracle.record_display_jank(false);
            oracle.record_zombie_count(0);
            oracle.record_swap_bytes(100 * 1024 * 1024);
        }
        assert!(oracle.stability_score() > 0.99);
        assert!(oracle.instability_penalty() < 0.01);
    }

    #[test]
    fn jank_raises_instability() {
        let mut oracle = StabilityOracle::new();
        oracle.record_display_jank(true);
        oracle.record_zombie_count(0);
        oracle.record_swap_bytes(0);
        // After one jank event, EMA = ALPHA = 0.05, instability = 0.05/3 ≈ 0.017
        assert!(oracle.instability_penalty() > 0.0);
        assert!(oracle.instability_penalty() < 0.1);
    }

    #[test]
    fn zombie_spike_raises_instability() {
        let mut oracle = StabilityOracle::new();
        oracle.record_zombie_count(5); // fully unstable zombie signal
        oracle.record_display_jank(false);
        oracle.record_swap_bytes(0);
        assert!(oracle.zombie_rate_ema() > 0.0);
        assert!(oracle.instability_penalty() > 0.0);
    }

    #[test]
    fn swap_spike_detected() {
        let mut oracle = StabilityOracle::new();
        oracle.record_swap_bytes(100 * 1024 * 1024); // baseline
        oracle.record_swap_bytes(800 * 1024 * 1024); // +700 MB spike
        assert!(oracle.swap_spike_ema() > 0.0);
    }

    #[test]
    fn gradual_swap_growth_not_a_spike() {
        let mut oracle = StabilityOracle::new();
        // 50 MB increments — gradual growth, not a spike.
        for i in 0..20u64 {
            oracle.record_swap_bytes(i * 50 * 1024 * 1024);
        }
        assert!(oracle.swap_spike_ema() < 0.01);
    }

    #[test]
    fn persistent_instability_degrades_score() {
        let mut oracle = StabilityOracle::new();
        // Simulate 40 consecutive events on ALL three signals simultaneously.
        // With alpha=0.05, each EMA after 40 events ≈ 1 - 0.95^40 ≈ 0.87.
        // instability = (0.87 + 0.87 + 0.87) / 3 ≈ 0.87 → score ≈ 0.13 < 0.4.
        let mut swap = 0u64;
        for _ in 0..40 {
            oracle.record_display_jank(true);
            oracle.record_zombie_count(5);
            swap += 600 * 1024 * 1024; // spike every cycle
            oracle.record_swap_bytes(swap);
        }
        assert!(oracle.jank_ema() > 0.8);
        assert!(oracle.stability_score() < 0.4);
    }

    #[test]
    fn recovery_improves_score() {
        let mut oracle = StabilityOracle::new();
        // Destabilise.
        for _ in 0..30 {
            oracle.record_display_jank(true);
            oracle.record_zombie_count(3);
            oracle.record_swap_bytes(600 * 1024 * 1024);
        }
        let low_score = oracle.stability_score();
        // Recover.
        for i in 0..100u64 {
            oracle.record_display_jank(false);
            oracle.record_zombie_count(0);
            oracle.record_swap_bytes((i % 2) * 100 * 1024 * 1024);
        }
        let high_score = oracle.stability_score();
        assert!(high_score > low_score);
    }
}
