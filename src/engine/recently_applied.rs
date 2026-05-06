//! # Recently-Applied Cache — cross-cycle governor state memory
//!
//! Closes NotebookLM Critical gap (2026-05-06):
//! > "Governor padece **Falta de Memoria de Estado** operativa.
//! > Sigue emitiendo la misma decisión para mismo PID ciclo tras ciclo si
//! > la condición de presión persiste."
//!
//! Empirical evidence: 87.5% of journal entries are `success: false` post-Phase 1
//! dedup. Most are "kernel says no-op (PID already in target state)". The
//! within-cycle chokepoint (commit 18f749d) catches multi-module dups but not
//! cross-cycle re-emissions when conditions don't change.
//!
//! ## Design
//!
//! Per-PID, per-action-kind, TTL-bounded cache. Filter governor decisions
//! before they become RootActions:
//! - Decision `(pid=X, kind=Throttle)` recorded → next 30s of Throttle proposals
//!   for that PID are suppressed
//! - When pressure regime shifts OR TTL expires, the entry is invalidated
//!   and Apollo can re-evaluate freshly
//!
//! ## Why per-kind, not just per-PID
//!
//! Throttle and Freeze are different physical state targets. A PID may be
//! throttled (renice +10) AND later need freeze (SIGSTOP). Per-kind cache
//! lets the upgrade path through.
//!
//! ## Bounds
//!
//! - TTL: 30s default — short enough that workload changes get fresh evaluation,
//!   long enough to suppress same-second-and-same-cycle redundancy
//! - Capacity: 5000 entries — eviction when exceeded (oldest first)
//! - cleanup_expired() called periodically by daemon (every 60 cycles)
//!
//! [Hellerstein 2004 §9] state-aware feedback control — controller must
//! remember its own actions to avoid redundant emission.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::engine::adaptive_governor::GovernorDecision;

/// Recently-applied decisions per (PID, kind) with TTL.
pub struct RecentlyApplied {
    map: HashMap<(u32, GovernorDecision), Instant>,
    ttl: Duration,
    capacity: usize,
}

impl RecentlyApplied {
    /// New cache with default 30s TTL and 5000-entry capacity.
    pub fn new() -> Self {
        Self::with_ttl(Duration::from_secs(30))
    }

    /// New cache with custom TTL.
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            map: HashMap::with_capacity(512),
            ttl,
            capacity: 5000,
        }
    }

    /// Record that `decision` was just applied to `pid`.
    /// Caller invokes this AFTER a successful action emission.
    pub fn record(&mut self, pid: u32, decision: GovernorDecision) {
        // Evict oldest if at capacity (cheap O(n) sweep, only when full).
        if self.map.len() >= self.capacity {
            self.evict_oldest();
        }
        self.map.insert((pid, decision), Instant::now());
    }

    /// Returns true if this PID had `decision` applied within the TTL window.
    /// Caller skips emitting when this returns true.
    pub fn is_recent(&self, pid: u32, decision: GovernorDecision) -> bool {
        match self.map.get(&(pid, decision)) {
            Some(t) => t.elapsed() <= self.ttl,
            None => false,
        }
    }

    /// Sweep expired entries. O(n) — call every 60 cycles to amortize.
    pub fn cleanup_expired(&mut self) -> usize {
        let ttl = self.ttl;
        let before = self.map.len();
        self.map.retain(|_, t| t.elapsed() <= ttl);
        before - self.map.len()
    }

    /// Forget all entries. Used on regime shift (pressure crosses threshold)
    /// when prior decisions may no longer be valid.
    pub fn invalidate_all(&mut self) {
        self.map.clear();
    }

    /// Forget entries for a specific PID — used when PID dies or is recycled.
    pub fn invalidate_pid(&mut self, pid: u32) {
        self.map.retain(|(p, _), _| *p != pid);
    }

    /// Current number of entries (post-cleanup).
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.len() == 0
    }

    /// Evict oldest entry (capacity overflow).
    fn evict_oldest(&mut self) {
        if let Some((oldest_key, _)) = self
            .map
            .iter()
            .min_by_key(|(_, t)| **t)
            .map(|(k, t)| (*k, *t))
        {
            self.map.remove(&oldest_key);
        }
    }
}

impl Default for RecentlyApplied {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cache_returns_false() {
        let cache = RecentlyApplied::new();
        assert!(!cache.is_recent(123, GovernorDecision::Throttle));
        assert!(cache.is_empty());
    }

    #[test]
    fn record_then_is_recent_returns_true() {
        let mut cache = RecentlyApplied::new();
        cache.record(123, GovernorDecision::Throttle);
        assert!(cache.is_recent(123, GovernorDecision::Throttle));
        assert!(!cache.is_recent(123, GovernorDecision::Freeze));
        assert!(!cache.is_recent(456, GovernorDecision::Throttle));
    }

    #[test]
    fn ttl_expiry_returns_false() {
        let mut cache = RecentlyApplied::with_ttl(Duration::from_millis(50));
        cache.record(123, GovernorDecision::Throttle);
        assert!(cache.is_recent(123, GovernorDecision::Throttle));
        std::thread::sleep(Duration::from_millis(80));
        assert!(!cache.is_recent(123, GovernorDecision::Throttle));
    }

    #[test]
    fn different_kinds_for_same_pid_coexist() {
        // Throttle + Freeze for same PID are distinct cache entries —
        // a process can be throttled then upgraded to frozen.
        let mut cache = RecentlyApplied::new();
        cache.record(123, GovernorDecision::Throttle);
        cache.record(123, GovernorDecision::Freeze);
        assert!(cache.is_recent(123, GovernorDecision::Throttle));
        assert!(cache.is_recent(123, GovernorDecision::Freeze));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cleanup_expired_returns_count_drained() {
        let mut cache = RecentlyApplied::with_ttl(Duration::from_millis(40));
        cache.record(1, GovernorDecision::Throttle);
        cache.record(2, GovernorDecision::Throttle);
        cache.record(3, GovernorDecision::Freeze);
        std::thread::sleep(Duration::from_millis(70));
        cache.record(4, GovernorDecision::Throttle); // Fresh
        let drained = cache.cleanup_expired();
        assert_eq!(drained, 3, "should drain 3 expired");
        assert_eq!(cache.len(), 1);
        assert!(cache.is_recent(4, GovernorDecision::Throttle));
    }

    #[test]
    fn invalidate_all_clears_cache() {
        let mut cache = RecentlyApplied::new();
        cache.record(1, GovernorDecision::Throttle);
        cache.record(2, GovernorDecision::Freeze);
        cache.invalidate_all();
        assert!(cache.is_empty());
    }

    #[test]
    fn invalidate_pid_removes_only_that_pid() {
        let mut cache = RecentlyApplied::new();
        cache.record(1, GovernorDecision::Throttle);
        cache.record(1, GovernorDecision::Freeze);
        cache.record(2, GovernorDecision::Throttle);
        cache.invalidate_pid(1);
        assert!(!cache.is_recent(1, GovernorDecision::Throttle));
        assert!(!cache.is_recent(1, GovernorDecision::Freeze));
        assert!(cache.is_recent(2, GovernorDecision::Throttle));
    }

    #[test]
    fn capacity_overflow_evicts_oldest() {
        let mut cache = RecentlyApplied::new();
        cache.capacity = 3; // shrink for test
        cache.record(1, GovernorDecision::Throttle);
        std::thread::sleep(Duration::from_millis(2));
        cache.record(2, GovernorDecision::Throttle);
        std::thread::sleep(Duration::from_millis(2));
        cache.record(3, GovernorDecision::Throttle);
        std::thread::sleep(Duration::from_millis(2));
        cache.record(4, GovernorDecision::Throttle);
        assert_eq!(cache.len(), 3);
        // PID 1 is oldest, should be evicted.
        assert!(!cache.is_recent(1, GovernorDecision::Throttle));
        assert!(cache.is_recent(4, GovernorDecision::Throttle));
    }

    #[test]
    fn allow_decision_treated_same_as_others() {
        // Sanity: GovernorDecision::Allow should NOT be recorded normally
        // (allow = inaction). But if caller does record it, cache treats
        // it as any other key.
        let mut cache = RecentlyApplied::new();
        cache.record(123, GovernorDecision::Allow);
        assert!(cache.is_recent(123, GovernorDecision::Allow));
    }

    #[test]
    fn high_traffic_bounded_growth() {
        // Stress: 10000 records with 5000 capacity → never grows past 5000.
        let mut cache = RecentlyApplied::new();
        for i in 0..10000u32 {
            cache.record(i, GovernorDecision::Throttle);
        }
        assert!(
            cache.len() <= cache.capacity,
            "len {} > capacity {}",
            cache.len(),
            cache.capacity
        );
    }
}
