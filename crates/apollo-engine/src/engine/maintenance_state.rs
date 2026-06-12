// crates/apollo-engine/src/engine/maintenance_state.rs
//! Maintenance Purge Gate state — opportunistic non-crisis purge orchestration.
//!
//! See docs/superpowers/specs/2026-05-10-maintenance-purge-design.md
//!
//! Asymmetric cooldown: survival_tick writes last_any_purge_at but does not
//! read it (survival is physical-crisis sovereign). maintenance_tick reads
//! and writes (yields to anything recent).

use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

/// Window after a purge during which predictors must inhibit swap-derived
/// updates. Per Hellerstein 2004 §9 "Feedback Control of Computing Systems"
/// — an exogenous disturbance (here: forced `vm_purge`) must not be learned
/// as a load improvement by closed-loop predictors. 5 seconds covers the
/// kernel-side compressor flush + the daemon's next cycle window so the
/// Kalman/Hazard/MPC stack sees the post-purge state once it stabilises,
/// not the artificial dip mid-flush.
pub const PURGE_INHIBITION_WINDOW_SECS: u64 = 5;

/// Extended window when compressor hasn't stabilized. Covers slow flushes
/// on high-memory-pressure systems (M1 8GB under LLM workloads).
pub const PURGE_INHIBITION_WINDOW_MAX: u64 = 12;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaintenanceState {
    #[serde(skip)]
    pub swap_delta_window: SwapDeltaWindow,

    #[serde(default)]
    pub last_any_purge_at: Option<SystemTime>,

    #[serde(default)]
    pub last_cli_purge_at: Option<SystemTime>,

    #[serde(skip)]
    pub last_wake_at: Option<Instant>,

    /// 2026-05-12 Gate F: count of consecutive cycles with thrashing_score
    /// above the "sustained flow crisis" threshold. Increments when thrash
    /// > 15_000, resets to 0 below. Reaches ≥3 → emergency purge bypass.
    /// > Skipped from persistence — re-initializes to 0 per daemon restart.
    #[serde(skip)]
    pub consecutive_thrash_cycles: u32,

    /// 2026-06-09 Gate F bypass: count of consecutive cycles with thrashing_score
    /// above CRITICAL_THRASHING_PURGE_SCORE (50k). Used to bypass the
    /// MediaActive gate when thrashing is sustained — a 10-cycle streak at 50k+
    /// means the system is in a real flow crisis, not a transient audio glitch.
    /// Skipped from persistence — re-initializes to 0 per daemon restart.
    #[serde(skip)]
    pub consecutive_thrash_50k_cycles: u32,

    /// 2026-05-28: Tracks whether the compressor is still flushing after
    /// a purge. When true, the inhibition window extends to MAX (12s) instead
    /// of the base (5s) to avoid learning the artificial dip.
    #[serde(skip)]
    pub compressor_still_flushing: bool,

    /// 2026-05-30: Consecutive observations of non-negative swap delta
    /// (stationary or recovering). When ≥2 (≥4s on 2s tick), the
    /// `compressor_still_flushing` latch is auto-cleared and the
    /// inhibition window collapses back to the 5s base. Without this
    /// counter the latch is monotonic-set (every purge re-asserts true
    /// when delta < 0) and never falls — pinning the window to 12s
    /// permanently under sustained pressure.
    /// [Hellerstein 2004 §9] disturbance settled when control signal
    /// stationary for two consecutive samples.
    #[serde(skip)]
    pub positive_delta_samples: u8,

    /// B.4 purge band (2026-06-10): Schmitt-trigger eligibility for the
    /// rising-edge zone [0.70, 0.75). Set while pressure sits in the safe
    /// band [0.55, 0.70); cleared at ≥0.75 (crisis ramp) or <0.50 (calm).
    /// A purge at 0.70-0.75 is allowed only when we were ALREADY eligible
    /// below 0.70 — fresh entry at the top of the band is likely a fast
    /// ramp toward crisis where purge would add jank. [Hellerstein 2004
    /// §9 hysteresis bands over binary thresholds.]
    ///
    /// serde(skip): re-initializes false on daemon restart. If pressure is
    /// already in [0.70, 0.75) at boot the rising-edge skip holds until
    /// one dip below 0.70 — acceptable cold-start conservatism.
    #[serde(skip)]
    pub purge_band_eligible: bool,
}

impl MaintenanceState {
    /// B.4 purge band (2026-06-10): advance the Schmitt trigger. Call once
    /// per maintenance tick BEFORE should_fire reads the flag.
    pub fn tick_pressure_band(&mut self, pressure: f64) {
        if (0.55..0.70).contains(&pressure) {
            self.purge_band_eligible = true;
        } else if !(0.50..0.75).contains(&pressure) {
            self.purge_band_eligible = false;
        }
        // [0.70, 0.75) and [0.50, 0.55): hold previous state (hysteresis).
    }

    /// 2026-05-28: marks compressor as actively flushing. Call this when
    /// post-purge swap velocity is still negative (compressor feeding VM).
    pub fn mark_compressor_flushing(&mut self, active: bool) {
        self.compressor_still_flushing = active;
        if active {
            // A fresh flush-start invalidates any prior streak.
            self.positive_delta_samples = 0;
        }
    }

    /// 2026-05-30: per-cycle tick. Counts consecutive non-negative
    /// `swap_delta_bps` observations. Once two have been seen in a row
    /// (≥4s on a 2s daemon tick), clears `compressor_still_flushing`
    /// so the inhibition window collapses from MAX (12s) back to the
    /// base (5s). A single negative delta resets the streak.
    pub fn tick_compressor_status(&mut self, swap_delta_bps: f64) {
        if !self.compressor_still_flushing {
            return;
        }
        if swap_delta_bps >= 0.0 {
            self.positive_delta_samples = self.positive_delta_samples.saturating_add(1);
            if self.positive_delta_samples >= 2 {
                self.compressor_still_flushing = false;
                self.positive_delta_samples = 0;
            }
        } else {
            self.positive_delta_samples = 0;
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn secs_since_any_purge(&self) -> u64 {
        match self.last_any_purge_at {
            None => u64::MAX,
            Some(t) => SystemTime::now()
                .duration_since(t)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    pub fn is_purge_recent(&self, window_secs: u64) -> bool {
        self.secs_since_any_purge() < window_secs
    }

    /// Returns true while predictors should inhibit swap-derived updates
    /// because a recent purge perturbed the swap signal. See
    /// [`PURGE_INHIBITION_WINDOW_SECS`] for the base window rationale.
    ///
    /// Adaptive extension: if `compressor_still_flushing` is true, extends
    /// the window to [`PURGE_INHIBITION_WINDOW_MAX`] (12s) to cover slow
    /// compressor flushes on high-memory-pressure systems (M1 8GB under LLM).
    /// [Hellerstein 2004 §9] disturbance rejection in closed-loop systems.
    pub fn is_in_purge_inhibition_window(&self) -> bool {
        let window = if self.compressor_still_flushing {
            PURGE_INHIBITION_WINDOW_MAX
        } else {
            PURGE_INHIBITION_WINDOW_SECS
        };
        self.is_purge_recent(window)
    }

    pub fn secs_since_cli_purge(&self) -> u64 {
        match self.last_cli_purge_at {
            None => u64::MAX,
            Some(t) => SystemTime::now()
                .duration_since(t)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    pub fn secs_since_wake(&self) -> u64 {
        match self.last_wake_at {
            None => u64::MAX,
            Some(t) => t.elapsed().as_secs(),
        }
    }

    pub fn push_swap_delta(&mut self, delta_bps: f64) {
        self.swap_delta_window.push(SystemTime::now(), delta_bps);
    }

    /// 2026-05-12 Gate F: track sustained thrashing via consecutive-cycle
    /// streak (matches the "≥3 cycles" gating in run_maintenance_tick).
    pub fn push_thrashing(&mut self, thrash: f64) {
        if thrash > 15_000.0 {
            self.consecutive_thrash_cycles = self.consecutive_thrash_cycles.saturating_add(1);
        } else {
            self.consecutive_thrash_cycles = 0;
        }
        if thrash > 50_000.0 {
            self.consecutive_thrash_50k_cycles =
                self.consecutive_thrash_50k_cycles.saturating_add(1);
        } else {
            self.consecutive_thrash_50k_cycles = 0;
        }
    }

    pub fn thrashing_streak_above(&self, _threshold: f64, min_cycles: u32) -> bool {
        // _threshold parameter retained for API extensibility but the actual
        // streak threshold is fixed at 15k in push_thrashing — caller passes
        // the same value for documentation clarity.
        self.consecutive_thrash_cycles >= min_cycles
    }

    pub fn mark_purged(&mut self) {
        self.last_any_purge_at = Some(SystemTime::now());
    }

    pub fn mark_cli_purged(&mut self) {
        let now = SystemTime::now();
        self.last_cli_purge_at = Some(now);
        self.last_any_purge_at = Some(now);
    }

    pub fn observe_wake(&mut self) {
        self.last_wake_at = Some(Instant::now());
    }
}

#[derive(Debug, Clone, Default)]
pub struct SwapDeltaWindow {
    samples: VecDeque<(SystemTime, f64)>,
}

impl SwapDeltaWindow {
    pub const CAP: usize = 45;

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn push(&mut self, t: SystemTime, delta_bps: f64) {
        if self.samples.len() >= Self::CAP {
            self.samples.pop_front();
        }
        self.samples.push_back((t, delta_bps));
    }

    pub fn sustained_below(&self, threshold_bps: f64, secs: u64) -> bool {
        let cutoff = match SystemTime::now().checked_sub(Duration::from_secs(secs)) {
            Some(t) => t,
            None => return false,
        };

        let recent: Vec<&(SystemTime, f64)> =
            self.samples.iter().filter(|(t, _)| *t >= cutoff).collect();

        let min_samples = (secs / 2).max(1) as usize;
        if recent.len() < min_samples {
            return false;
        }

        recent.iter().all(|(_, bps)| *bps < threshold_bps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_delta_window_drops_oldest_at_capacity() {
        let mut w = SwapDeltaWindow::default();
        let t = SystemTime::now();
        for i in 0..50 {
            w.push(t + Duration::from_secs(i as u64), i as f64);
        }
        assert_eq!(w.len(), SwapDeltaWindow::CAP);
        // First sample retained should be sample index 5 (50 - 45)
        assert_eq!(w.samples.front().unwrap().1, 5.0);
    }

    #[test]
    fn swap_delta_window_sustained_below_with_full_window_returns_true() {
        let mut w = SwapDeltaWindow::default();
        let now = SystemTime::now();
        for i in 0..45 {
            let t = now - Duration::from_secs(89) + Duration::from_secs(i * 2);
            w.push(t, 50_000.0);
        }
        assert!(w.sustained_below(256_000.0, 90));
    }

    #[test]
    fn swap_delta_window_sustained_below_with_one_spike_returns_false() {
        let mut w = SwapDeltaWindow::default();
        let now = SystemTime::now();
        for i in 0..30 {
            let t = now - Duration::from_secs(89) + Duration::from_secs(i * 2);
            w.push(t, 50_000.0);
        }
        w.push(now - Duration::from_secs(10), 500_000.0);
        assert!(!w.sustained_below(256_000.0, 90));
    }

    #[test]
    fn swap_delta_window_sustained_below_empty_returns_false() {
        let w = SwapDeltaWindow::default();
        assert!(!w.sustained_below(256_000.0, 90));
    }

    #[test]
    fn swap_delta_window_sustained_below_partial_window_returns_false() {
        let mut w = SwapDeltaWindow::default();
        let now = SystemTime::now();
        for i in 0..10 {
            let t = now - Duration::from_secs(20) + Duration::from_secs(i * 2);
            w.push(t, 50_000.0);
        }
        assert!(!w.sustained_below(256_000.0, 90));
    }

    #[test]
    fn secs_since_any_purge_none_returns_max() {
        let s = MaintenanceState::default();
        assert_eq!(s.secs_since_any_purge(), u64::MAX);
    }

    #[test]
    fn secs_since_any_purge_clock_backwards_returns_zero() {
        let mut s = MaintenanceState::default();
        s.last_any_purge_at = Some(SystemTime::now() + Duration::from_secs(60));
        assert_eq!(s.secs_since_any_purge(), 0);
    }

    #[test]
    fn mark_cli_purged_updates_both_timestamps() {
        let mut s = MaintenanceState::default();
        s.mark_cli_purged();
        assert!(s.last_cli_purge_at.is_some());
        assert!(s.last_any_purge_at.is_some());
    }

    #[test]
    fn mark_purged_only_updates_any_not_cli() {
        let mut s = MaintenanceState::default();
        s.mark_purged();
        assert!(s.last_any_purge_at.is_some());
        assert!(s.last_cli_purge_at.is_none());
    }

    #[test]
    fn is_in_purge_inhibition_window_false_without_purge() {
        let s = MaintenanceState::default();
        assert!(
            !s.is_in_purge_inhibition_window(),
            "no purge → never in inhibition window"
        );
    }

    #[test]
    fn is_in_purge_inhibition_window_true_immediately_after_purge() {
        let mut s = MaintenanceState::default();
        s.mark_purged();
        assert!(
            s.is_in_purge_inhibition_window(),
            "fresh purge → in inhibition window"
        );
    }

    #[test]
    fn is_in_purge_inhibition_window_false_after_window_expires() {
        let mut s = MaintenanceState::default();
        let past = SystemTime::now() - Duration::from_secs(PURGE_INHIBITION_WINDOW_SECS + 1);
        s.last_any_purge_at = Some(past);
        assert!(
            !s.is_in_purge_inhibition_window(),
            "purge older than window → not inhibited"
        );
    }

    /// 2026-05-30: under two stationary positive-delta observations the
    /// `compressor_still_flushing` latch must clear and the window
    /// must collapse from MAX (12s) back to the 5s base. Verifies the
    /// fix that prevents the latch becoming monotonic-set under
    /// sustained pressure.
    #[test]
    fn test_compressor_flushing_clears_under_stationary_positive_delta() {
        let mut s = MaintenanceState::default();
        // Simulate a purge that fired between 6s and 11s ago — past the
        // 5s base window but inside the 12s extended window.
        let past = SystemTime::now() - Duration::from_secs(8);
        s.last_any_purge_at = Some(past);
        s.mark_compressor_flushing(true);
        assert!(
            s.is_in_purge_inhibition_window(),
            "with latch set, 8s-old purge still inside 12s extended window"
        );

        // Two consecutive non-negative deltas should clear the latch.
        s.tick_compressor_status(100.0);
        s.tick_compressor_status(100.0);
        assert!(
            !s.compressor_still_flushing,
            "latch must clear after 2 ticks"
        );
        assert!(
            !s.is_in_purge_inhibition_window(),
            "window collapses to 5s base — 8s-old purge no longer inhibited"
        );
    }

    /// Property test: under N stationary positive-delta samples the
    /// latch must clear within 2 ticks (≥4s on a 2s daemon cadence).
    #[test]
    fn tick_compressor_status_clears_within_two_ticks_under_positive_stream() {
        for n in 2..100u32 {
            let mut s = MaintenanceState::default();
            s.mark_compressor_flushing(true);
            for _ in 0..n {
                s.tick_compressor_status(50_000.0);
            }
            assert!(
                !s.compressor_still_flushing,
                "after {} positive ticks the latch should be cleared",
                n
            );
        }
    }

    /// A single negative delta inside the streak resets the counter —
    /// the latch must remain set until two consecutive non-negative
    /// observations are seen.
    #[test]
    fn tick_compressor_status_negative_delta_resets_streak() {
        let mut s = MaintenanceState::default();
        s.mark_compressor_flushing(true);
        s.tick_compressor_status(100.0);
        assert_eq!(s.positive_delta_samples, 1);
        s.tick_compressor_status(-200.0);
        assert_eq!(s.positive_delta_samples, 0);
        assert!(
            s.compressor_still_flushing,
            "single positive sample then negative must NOT clear the latch"
        );
    }

    /// Calling `tick_compressor_status` when the latch is already
    /// false is a no-op (no underflow, no spurious state writes).
    #[test]
    fn tick_compressor_status_noop_when_latch_clear() {
        let mut s = MaintenanceState::default();
        s.tick_compressor_status(100.0);
        s.tick_compressor_status(-100.0);
        assert!(!s.compressor_still_flushing);
        assert_eq!(s.positive_delta_samples, 0);
    }
}
