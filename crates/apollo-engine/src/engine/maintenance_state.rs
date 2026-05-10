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
        // Add 1-second grace to the cutoff to avoid boundary jitter at exact window edges.
        let cutoff = match SystemTime::now().checked_sub(Duration::from_secs(secs + 1)) {
            Some(t) => t,
            None => return false,
        };

        let recent: Vec<&(SystemTime, f64)> = self
            .samples
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .collect();

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
            let t = now - Duration::from_secs(90) + Duration::from_secs(i * 2);
            w.push(t, 50_000.0);
        }
        assert!(w.sustained_below(256_000.0, 90));
    }
}
