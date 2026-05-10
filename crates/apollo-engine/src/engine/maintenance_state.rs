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
}

impl MaintenanceState {
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
}
