//! 24h rolling window of survival-mode cycle observations.
//!
//! D5 fix: AIS `safety_compliance()` previously read the lifetime cumulative
//! counter, which monotonically tainted the score after any sustained crisis.
//! This module provides a wall-clock 24h window so chronic-load detection
//! reflects *recent* operational state, not since-boot history.
//!
//! Mirrors the `VecDeque<SystemTime>` shape used by `SwapDeltaWindow` in
//! `maintenance_state.rs` and persists wall-clock seconds per the
//! `RecentlyApplied` B1 pattern (commit 386382f).
//!
//! See `CLAUDE.md` Sprint 3 doctrine entry #5 — sticky-as-live-state-flag
//! anti-pattern remediation in the numerical-score domain.
//!
//! Papers / doctrine:
//! - [Beyer & Jones 2016 SRE Ch.3] graduated error budgets.
//! - [Welford 1962] — N/A (count-only window, no streaming variance).

use std::collections::VecDeque;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// 24h window — aligned with overflow_guard `dynamic_offset_recovers_after_24h_calm`.
pub const WINDOW_DURATION: Duration = Duration::from_secs(24 * 3600);

/// Hard cap on stored entries. 24h × 1800 cycles/h (2s cadence) + 600 jitter.
pub const CAP: usize = 43_800;

#[derive(Debug, Default, Clone)]
pub struct SurvivalActivationWindow {
    entries: VecDeque<SystemTime>,
}

impl SurvivalActivationWindow {
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
        }
    }

    /// Record one cycle observed while survival mode active.
    pub fn record(&mut self, now: SystemTime) {
        if self.entries.len() >= CAP {
            self.entries.pop_front();
        }
        self.entries.push_back(now);
    }

    /// Drop entries older than WINDOW_DURATION. Clock-backwards entries
    /// (future SystemTime) are kept (treated as age=0, defensive idiom from
    /// `secs_since_any_purge_clock_backwards_returns_zero`).
    pub fn prune(&mut self, now: SystemTime) {
        while let Some(front) = self.entries.front() {
            match now.duration_since(*front) {
                Ok(d) if d > WINDOW_DURATION => {
                    self.entries.pop_front();
                }
                _ => break, // either within window or clock-backwards → keep
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Persistence: collect as Unix seconds for cross-restart durability
    /// (B1 RecentlyApplied pattern, commit 386382f).
    pub fn to_unix_secs(&self) -> Vec<u64> {
        self.entries
            .iter()
            .filter_map(|t| t.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs()))
            .collect()
    }

    /// Restore from Unix seconds, pruning entries already outside window.
    pub fn from_unix_secs(secs: Vec<u64>, now: SystemTime) -> Self {
        let mut w = Self::new();
        for s in secs {
            let t = UNIX_EPOCH + Duration::from_secs(s);
            w.entries.push_back(t);
        }
        w.prune(now);
        while w.entries.len() > CAP {
            w.entries.pop_front();
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn survival_window_empty_returns_healthy_bucket() {
        let w = SurvivalActivationWindow::new();
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());
    }

    #[test]
    fn survival_window_prune_drops_entries_older_than_24h() {
        let mut w = SurvivalActivationWindow::new();
        let now = SystemTime::now();
        w.entries.push_back(now - Duration::from_secs(25 * 3600));
        w.entries.push_back(now - Duration::from_secs(23 * 3600));
        w.entries.push_back(now - Duration::from_secs(3600));
        w.prune(now);
        assert_eq!(w.len(), 2, "only 25h entry should be dropped");
    }

    #[test]
    fn survival_window_bucket_boundary_exact_300_and_301() {
        // The bucket boundary check lives in intelligence_score.rs; this test
        // guards the window count semantics that feed it. See
        // `crates/apollo-engine/src/engine/intelligence_score.rs` tests for
        // the bucket-score assertion.
        let mut w = SurvivalActivationWindow::new();
        let now = SystemTime::now();
        for i in 0..300 {
            w.record(now - Duration::from_secs(i));
        }
        assert_eq!(w.len(), 300, "transient boundary");
        w.record(now);
        assert_eq!(w.len(), 301, "crosses into sustained bucket");
    }

    #[test]
    fn survival_window_clock_backwards_keeps_entry() {
        let mut w = SurvivalActivationWindow::new();
        let now = SystemTime::now();
        // Push entry from the "future".
        w.entries.push_back(now + Duration::from_secs(3600));
        w.prune(now);
        assert_eq!(w.len(), 1, "future entry must be kept (age=0)");
    }

    #[test]
    fn survival_window_at_cap_drops_oldest_on_push() {
        let mut w = SurvivalActivationWindow::new();
        let base = SystemTime::now();
        // Use the unique "oldest" timestamp as a marker for the drop test.
        let oldest_marker = base - Duration::from_secs(CAP as u64 + 10);
        w.entries.push_back(oldest_marker);
        // Fill to CAP - 1 more.
        for i in 0..(CAP - 1) {
            w.entries
                .push_back(base - Duration::from_secs((CAP - i) as u64));
        }
        assert_eq!(w.len(), CAP, "filled to cap");
        // Record one more — should evict oldest_marker.
        let newest_marker = base + Duration::from_secs(1);
        w.record(newest_marker);
        assert_eq!(w.len(), CAP, "still at cap");
        assert!(
            !w.entries.iter().any(|t| *t == oldest_marker),
            "oldest dropped"
        );
        assert!(
            w.entries.iter().any(|t| *t == newest_marker),
            "newest present"
        );
    }

    #[test]
    fn survival_window_roundtrip_persists_unix_seconds_not_instant() {
        let mut w = SurvivalActivationWindow::new();
        let now = SystemTime::now();
        w.record(now - Duration::from_secs(3600));
        w.record(now - Duration::from_secs(60));
        w.record(now);
        let serialized = w.to_unix_secs();
        assert_eq!(serialized.len(), 3);
        let restored = SurvivalActivationWindow::from_unix_secs(serialized, now);
        assert_eq!(restored.len(), 3, "all three within 24h window");
        // Sanity: within 1s of source timestamps.
        let original_secs: Vec<u64> = w.to_unix_secs();
        let restored_secs: Vec<u64> = restored.to_unix_secs();
        for (a, b) in original_secs.iter().zip(restored_secs.iter()) {
            assert!(a.abs_diff(*b) <= 1, "timestamp drift within 1s");
        }
    }
}
