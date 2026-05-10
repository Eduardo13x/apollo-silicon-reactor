//! Time-decayed envelope of recently-active app coalitions.
//!
//! `daemon_agent_actions` only protected the *currently* foreground
//! coalition. That left a gap during rapid app switching: tabbing from
//! Antigravity to Terminal for a 3-second `git status` instantly stripped
//! Antigravity's helpers of coalition protection. If pressure spiked
//! during those seconds, helpers were eligible victims.
//!
//! This envelope keeps the last `MAX_TRACKED` coalitions seen in
//! foreground, each with a wall-clock `last_seen`. A coalition stays in
//! the protected set for `GRACE_SECS` after it was last fg, so
//! micro-switches between apps don't strip protection mid-task.
//!
//! ## Guarantees
//!
//! - Bounded memory: `MAX_TRACKED` (3) entries.
//! - O(1) update; O(MAX_TRACKED) `is_active`.
//! - SystemTime not Instant — survives sleep/wake cycles correctly.
//!
//! ## Caller contract
//!
//! Call `record_foreground(coalition_id)` once per cycle when a fg PID
//! and its coalition_id are available. Coalition_id 0 (kernel return for
//! "unknown") is ignored — we never protect the unknown set.
//!
//! Read with `is_active(coalition_id)` from any filter.

use std::time::{Duration, SystemTime};

/// Number of recent coalitions tracked. 3 covers the typical
/// "Antigravity ↔ Terminal ↔ Brave" rotation without bloating the
/// protected set.
const MAX_TRACKED: usize = 3;
/// Grace window: a coalition stays protected this long after it was
/// last foreground.
const GRACE_SECS: u64 = 300; // 5 minutes

#[derive(Debug, Clone, Copy)]
struct Entry {
    coalition_id: u64,
    last_seen: SystemTime,
}

#[derive(Debug, Default)]
pub struct ActiveCoalitionEnvelope {
    entries: Vec<Entry>,
}

impl ActiveCoalitionEnvelope {
    pub fn new() -> Self {
        Self {
            entries: Vec::with_capacity(MAX_TRACKED),
        }
    }

    /// Record that `coalition_id` was just observed in foreground.
    /// Refreshes `last_seen` if already tracked; otherwise inserts and
    /// evicts the oldest entry when the set is full.
    pub fn record_foreground(&mut self, coalition_id: u64) {
        if coalition_id == 0 {
            return;
        }
        let now = SystemTime::now();
        if let Some(e) = self.entries.iter_mut().find(|e| e.coalition_id == coalition_id) {
            e.last_seen = now;
            return;
        }
        if self.entries.len() >= MAX_TRACKED {
            // Evict the oldest (smallest last_seen).
            let oldest_idx = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_seen)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.entries.swap_remove(oldest_idx);
        }
        self.entries.push(Entry {
            coalition_id,
            last_seen: now,
        });
    }

    /// True if `coalition_id` was foreground within the grace window.
    pub fn is_active(&self, coalition_id: u64) -> bool {
        if coalition_id == 0 {
            return false;
        }
        let now = SystemTime::now();
        let grace = Duration::from_secs(GRACE_SECS);
        self.entries.iter().any(|e| {
            e.coalition_id == coalition_id
                && now
                    .duration_since(e.last_seen)
                    .map(|d| d <= grace)
                    .unwrap_or(true)
        })
    }

    /// Drop entries older than GRACE_SECS. Cheap to call each cycle but
    /// not strictly required — `is_active` already checks staleness.
    pub fn evict_stale(&mut self) {
        let now = SystemTime::now();
        let grace = Duration::from_secs(GRACE_SECS);
        self.entries.retain(|e| {
            now.duration_since(e.last_seen)
                .map(|d| d <= grace)
                .unwrap_or(true)
        });
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Borrow bundle: tracker + envelope. Pass this single struct into action
/// pipelines so a `CoalitionGuard::is_protected(pid)` query is one
/// chain-call, and adding the guard at a new emission site only adds one
/// parameter to the signature instead of two.
pub struct CoalitionGuard<'a> {
    tracker: &'a crate::engine::coalition::CoalitionTracker,
    envelope: &'a ActiveCoalitionEnvelope,
}

impl<'a> CoalitionGuard<'a> {
    pub fn new(
        tracker: &'a crate::engine::coalition::CoalitionTracker,
        envelope: &'a ActiveCoalitionEnvelope,
    ) -> Self {
        Self { tracker, envelope }
    }

    /// True when `pid` belongs to a coalition that was foreground in the
    /// last grace window. Use to skip destructive actions against
    /// subprocesses of the user's active workflow.
    pub fn is_protected(&self, pid: u32) -> bool {
        self.envelope
            .is_active(self.tracker.get_coalition_id(pid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn record_and_is_active_within_grace() {
        let mut env = ActiveCoalitionEnvelope::new();
        env.record_foreground(42);
        assert!(env.is_active(42));
        assert!(!env.is_active(99));
    }

    #[test]
    fn coalition_id_zero_is_ignored() {
        let mut env = ActiveCoalitionEnvelope::new();
        env.record_foreground(0);
        assert!(!env.is_active(0));
        assert_eq!(env.len(), 0);
    }

    #[test]
    fn capacity_evicts_oldest() {
        let mut env = ActiveCoalitionEnvelope::new();
        env.record_foreground(1);
        sleep(Duration::from_millis(10));
        env.record_foreground(2);
        sleep(Duration::from_millis(10));
        env.record_foreground(3);
        sleep(Duration::from_millis(10));
        // 4th push should evict #1 (oldest).
        env.record_foreground(4);
        assert_eq!(env.len(), MAX_TRACKED);
        assert!(!env.is_active(1));
        assert!(env.is_active(2));
        assert!(env.is_active(3));
        assert!(env.is_active(4));
    }

    #[test]
    fn refresh_updates_last_seen() {
        let mut env = ActiveCoalitionEnvelope::new();
        env.record_foreground(1);
        env.record_foreground(2);
        env.record_foreground(3);
        sleep(Duration::from_millis(20));
        // Refresh #1 — its last_seen should now be the newest.
        env.record_foreground(1);
        sleep(Duration::from_millis(10));
        // Push #5 — should evict the OLDEST, which is now #2 (1 was refreshed).
        env.record_foreground(5);
        assert!(env.is_active(1));
        assert!(!env.is_active(2));
        assert!(env.is_active(3));
        assert!(env.is_active(5));
    }
}
