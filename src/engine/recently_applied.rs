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
use crate::engine::types::RootAction;

/// Action kind for cache keying — broader than GovernorDecision because
/// it covers paths that emit RootAction directly (paging_hints, deep-scan,
/// llm_daemon, decide_actions) without going through the heuristic governor.
///
/// Distinct from `GovernorDecision` because:
/// - SetMemorystatus has no GovernorDecision equivalent (it's an OS-level hint)
/// - Boost is a RootAction but not a heuristic decision
/// - Maps Kill→Freeze (Apollo always downgrades Kill, never executes it)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CachedActionKind {
    Throttle,
    Freeze,
    Boost,
    Unfreeze,
    SetMemorystatus,
    SetThreadQoS,
}

impl CachedActionKind {
    /// Map a `GovernorDecision` to a cache kind.
    /// `Allow` returns None (no action emitted, nothing to cache).
    /// `Kill` maps to Freeze (Apollo's safety downgrade).
    pub fn from_governor(d: GovernorDecision) -> Option<Self> {
        match d {
            GovernorDecision::Throttle => Some(Self::Throttle),
            GovernorDecision::Freeze | GovernorDecision::Kill => Some(Self::Freeze),
            GovernorDecision::Allow => None,
        }
    }

    /// Map a `RootAction` to its cache kind, if it's a per-PID action.
    /// Non-PID actions (SetSysctl, ToggleSpotlight, QuarantineDaemon) → None.
    pub fn from_root_action(action: &RootAction) -> Option<(u32, Self)> {
        match action {
            RootAction::ThrottleProcess { pid, .. } => Some((*pid, Self::Throttle)),
            RootAction::FreezeProcess { pid, .. } => Some((*pid, Self::Freeze)),
            RootAction::UnfreezeProcess { pid, .. } => Some((*pid, Self::Unfreeze)),
            RootAction::BoostProcess { pid, .. } => Some((*pid, Self::Boost)),
            RootAction::SetMemorystatus { pid, .. } => Some((*pid, Self::SetMemorystatus)),
            RootAction::SetThreadQoS { pid, .. } => Some((*pid, Self::SetThreadQoS)),
            _ => None,
        }
    }
}

/// Recently-applied decisions per (PID, kind) with TTL.
pub struct RecentlyApplied {
    map: HashMap<(u32, CachedActionKind), Instant>,
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

    /// Record that `kind` was just applied to `pid`.
    /// Caller invokes this AFTER a successful action emission.
    pub fn record(&mut self, pid: u32, kind: CachedActionKind) {
        // Evict oldest if at capacity (cheap O(n) sweep, only when full).
        if self.map.len() >= self.capacity {
            self.evict_oldest();
        }
        self.map.insert((pid, kind), Instant::now());
    }

    /// Returns true if this PID had `kind` applied within the TTL window.
    /// Caller skips emitting when this returns true.
    pub fn is_recent(&self, pid: u32, kind: CachedActionKind) -> bool {
        match self.map.get(&(pid, kind)) {
            Some(t) => t.elapsed() <= self.ttl,
            None => false,
        }
    }

    /// Compatibility helper: takes GovernorDecision, maps to CachedActionKind.
    /// Used by governor heuristic path (process_enrichment).
    pub fn record_governor(&mut self, pid: u32, decision: GovernorDecision) {
        if let Some(kind) = CachedActionKind::from_governor(decision) {
            self.record(pid, kind);
        }
    }

    /// Compatibility helper: governor-decision query.
    pub fn is_recent_governor(&self, pid: u32, decision: GovernorDecision) -> bool {
        match CachedActionKind::from_governor(decision) {
            Some(kind) => self.is_recent(pid, kind),
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
        assert!(!cache.is_recent(123, CachedActionKind::Throttle));
        assert!(cache.is_empty());
    }

    #[test]
    fn record_then_is_recent_returns_true() {
        let mut cache = RecentlyApplied::new();
        cache.record(123, CachedActionKind::Throttle);
        assert!(cache.is_recent(123, CachedActionKind::Throttle));
        assert!(!cache.is_recent(123, CachedActionKind::Freeze));
        assert!(!cache.is_recent(456, CachedActionKind::Throttle));
    }

    #[test]
    fn ttl_expiry_returns_false() {
        let mut cache = RecentlyApplied::with_ttl(Duration::from_millis(50));
        cache.record(123, CachedActionKind::Throttle);
        assert!(cache.is_recent(123, CachedActionKind::Throttle));
        std::thread::sleep(Duration::from_millis(80));
        assert!(!cache.is_recent(123, CachedActionKind::Throttle));
    }

    #[test]
    fn different_kinds_for_same_pid_coexist() {
        let mut cache = RecentlyApplied::new();
        cache.record(123, CachedActionKind::Throttle);
        cache.record(123, CachedActionKind::Freeze);
        cache.record(123, CachedActionKind::SetMemorystatus);
        assert!(cache.is_recent(123, CachedActionKind::Throttle));
        assert!(cache.is_recent(123, CachedActionKind::Freeze));
        assert!(cache.is_recent(123, CachedActionKind::SetMemorystatus));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn cleanup_expired_returns_count_drained() {
        let mut cache = RecentlyApplied::with_ttl(Duration::from_millis(40));
        cache.record(1, CachedActionKind::Throttle);
        cache.record(2, CachedActionKind::Throttle);
        cache.record(3, CachedActionKind::Freeze);
        std::thread::sleep(Duration::from_millis(70));
        cache.record(4, CachedActionKind::Throttle);
        let drained = cache.cleanup_expired();
        assert_eq!(drained, 3);
        assert_eq!(cache.len(), 1);
        assert!(cache.is_recent(4, CachedActionKind::Throttle));
    }

    #[test]
    fn invalidate_all_clears_cache() {
        let mut cache = RecentlyApplied::new();
        cache.record(1, CachedActionKind::Throttle);
        cache.record(2, CachedActionKind::Freeze);
        cache.invalidate_all();
        assert!(cache.is_empty());
    }

    #[test]
    fn invalidate_pid_removes_only_that_pid() {
        let mut cache = RecentlyApplied::new();
        cache.record(1, CachedActionKind::Throttle);
        cache.record(1, CachedActionKind::Freeze);
        cache.record(2, CachedActionKind::Throttle);
        cache.invalidate_pid(1);
        assert!(!cache.is_recent(1, CachedActionKind::Throttle));
        assert!(!cache.is_recent(1, CachedActionKind::Freeze));
        assert!(cache.is_recent(2, CachedActionKind::Throttle));
    }

    #[test]
    fn capacity_overflow_evicts_oldest() {
        let mut cache = RecentlyApplied::new();
        cache.capacity = 3;
        cache.record(1, CachedActionKind::Throttle);
        std::thread::sleep(Duration::from_millis(2));
        cache.record(2, CachedActionKind::Throttle);
        std::thread::sleep(Duration::from_millis(2));
        cache.record(3, CachedActionKind::Throttle);
        std::thread::sleep(Duration::from_millis(2));
        cache.record(4, CachedActionKind::Throttle);
        assert_eq!(cache.len(), 3);
        assert!(!cache.is_recent(1, CachedActionKind::Throttle));
        assert!(cache.is_recent(4, CachedActionKind::Throttle));
    }

    #[test]
    fn from_governor_maps_correctly() {
        assert_eq!(
            CachedActionKind::from_governor(GovernorDecision::Throttle),
            Some(CachedActionKind::Throttle)
        );
        assert_eq!(
            CachedActionKind::from_governor(GovernorDecision::Freeze),
            Some(CachedActionKind::Freeze)
        );
        // Kill maps to Freeze (Apollo downgrades).
        assert_eq!(
            CachedActionKind::from_governor(GovernorDecision::Kill),
            Some(CachedActionKind::Freeze)
        );
        // Allow → None (no action to cache).
        assert_eq!(CachedActionKind::from_governor(GovernorDecision::Allow), None);
    }

    #[test]
    fn from_root_action_maps_per_pid_actions() {
        use crate::engine::audit_types::DecisionReason;
        let throttle = RootAction::ThrottleProcess {
            pid: 100,
            name: "p".to_string(),
            aggressive: false,
            reason: "r".to_string(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        };
        assert_eq!(
            CachedActionKind::from_root_action(&throttle),
            Some((100, CachedActionKind::Throttle))
        );

        let setmem = RootAction::SetMemorystatus {
            pid: 200,
            priority: -1,
            reason: "r".to_string(),
            decision_reason: DecisionReason::MemoryBudget,
        };
        assert_eq!(
            CachedActionKind::from_root_action(&setmem),
            Some((200, CachedActionKind::SetMemorystatus))
        );

        // Non-PID action → None.
        let toggle = RootAction::ToggleSpotlight {
            enabled: false,
            reason: "r".to_string(),
            decision_reason: DecisionReason::PressureContext,
        };
        assert_eq!(CachedActionKind::from_root_action(&toggle), None);
    }

    #[test]
    fn high_traffic_bounded_growth() {
        let mut cache = RecentlyApplied::new();
        for i in 0..10000u32 {
            cache.record(i, CachedActionKind::Throttle);
        }
        assert!(cache.len() <= cache.capacity);
    }

    #[test]
    fn record_governor_compatibility_helper() {
        let mut cache = RecentlyApplied::new();
        cache.record_governor(123, GovernorDecision::Throttle);
        assert!(cache.is_recent_governor(123, GovernorDecision::Throttle));
        assert!(cache.is_recent(123, CachedActionKind::Throttle));
        // Allow is no-op.
        cache.record_governor(456, GovernorDecision::Allow);
        assert!(!cache.is_recent_governor(456, GovernorDecision::Allow));
    }
}
