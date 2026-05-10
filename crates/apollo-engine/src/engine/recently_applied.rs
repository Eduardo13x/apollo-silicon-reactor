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

use serde::{Deserialize, Serialize};

use crate::engine::adaptive_governor::GovernorDecision;
use crate::engine::types::RootAction;

/// Persistable record (single entry in `/var/lib/apollo/recently_applied.jsonl`).
///
/// Wall-clock timestamp is recorded so post-restart staleness check can
/// drop entries older than TTL. We do NOT persist `Instant` directly because
/// `Instant` is monotonic-clock based and meaningless across reboots.
///
/// **Fail-empty restore policy** (Sprint 2 spec §B1):
/// - File missing → start empty (normal first boot)
/// - Parse error / malformed JSON → start empty + DELETE corrupt file
/// - Wall-clock delta write→read > 15s → discard all entries
/// - Per-entry wall-clock > 30s old → drop that entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistRecord {
    pub pid: u32,
    pub kind: CachedActionKind,
    /// Unix wall-clock seconds at write time.
    pub wall_unix_sec: u64,
}

/// Restore-status telemetry. Five mutually-exclusive states reported in
/// `runtime_metrics.json` for NotebookLM debrief observability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestoreStatus {
    /// File did not exist on startup (first boot or never persisted).
    Missing,
    /// Successful restore of N entries.
    RestoredN(u32),
    /// File existed but parse error / malformed JSON.
    DiscardedCorrupt,
    /// Wall-clock delta between write and read exceeded 15s.
    DiscardedClockDelta,
    /// Boot-time crossed (uptime is less than file age).
    DiscardedBootCrossed,
}

/// Action kind for cache keying — broader than GovernorDecision because
/// it covers paths that emit RootAction directly (paging_hints, deep-scan,
/// llm_daemon, decide_actions) without going through the heuristic governor.
///
/// Distinct from `GovernorDecision` because:
/// - SetMemorystatus has no GovernorDecision equivalent (it's an OS-level hint)
/// - Boost is a RootAction but not a heuristic decision
/// - Maps Kill→Freeze (Apollo always downgrades Kill, never executes it)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

    /// Snapshot current cache as persistable records. Caller serializes
    /// to disk on graceful shutdown.
    pub fn to_persist_records(&self) -> Vec<PersistRecord> {
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let now_instant = Instant::now();
        self.map
            .iter()
            .map(|((pid, kind), instant)| {
                // Map Instant elapsed → wall-clock seconds ago.
                let age_secs = now_instant.duration_since(*instant).as_secs();
                PersistRecord {
                    pid: *pid,
                    kind: *kind,
                    wall_unix_sec: now_unix.saturating_sub(age_secs),
                }
            })
            .collect()
    }

    /// Restore cache from records loaded from disk. Applies fail-empty
    /// policy: stale entries (>TTL old) are dropped per-entry.
    /// Returns the number of records actually restored.
    ///
    /// Caller is responsible for the GLOBAL fail-empty checks (parse error,
    /// clock delta, boot crossing) — this method only does per-entry checks.
    pub fn restore_from_records(&mut self, records: Vec<PersistRecord>) -> u32 {
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let ttl_secs = self.ttl.as_secs();
        let now_instant = Instant::now();
        let mut restored = 0u32;
        for r in records {
            // Per-entry staleness check.
            let age = now_unix.saturating_sub(r.wall_unix_sec);
            if age > ttl_secs {
                continue; // drop only this entry
            }
            // Compute monotonic Instant for this entry's timestamp.
            let entry_instant = now_instant
                .checked_sub(std::time::Duration::from_secs(age))
                .unwrap_or(now_instant);
            self.map.insert((r.pid, r.kind), entry_instant);
            restored += 1;
        }
        restored
    }

    /// Save the cache to disk using the provided path.
    pub fn save_to_disk(&mut self, path: &std::path::Path) {
        self.cleanup_expired();
        if self.is_empty() {
            let _ = std::fs::remove_file(path);
            return;
        }
        let records = self.to_persist_records();
        if let Ok(json) = serde_json::to_string(&records) {
            let _ = std::fs::write(path, json.as_bytes());
        }
    }

    /// Load the cache from disk, applying global fail-empty checks.
    /// Returns the instantiated cache and the outcome status.
    pub fn load_from_disk(path: &std::path::Path) -> (Self, RestoreStatus) {
        let mut cache = Self::new();
        let json = match std::fs::read_to_string(path) {
            Ok(j) => j,
            Err(_) => return (cache, RestoreStatus::Missing),
        };

        let records: Vec<PersistRecord> = match serde_json::from_str(&json) {
            Ok(r) => r,
            Err(_) => {
                let _ = std::fs::remove_file(path);
                return (cache, RestoreStatus::DiscardedCorrupt);
            }
        };

        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Global clock delta check
        for r in &records {
            if r.wall_unix_sec > now_unix + 15 {
                return (cache, RestoreStatus::DiscardedClockDelta);
            }
        }

        // Boot-time crossing check: if the oldest record is older than uptime,
        // it belongs to a previous boot session. Instant is invalid across boots.
        let uptime = crate::engine::daemon_helpers::system_uptime_secs();
        let oldest_record = records
            .iter()
            .map(|r| r.wall_unix_sec)
            .min()
            .unwrap_or(now_unix);
        let record_age = now_unix.saturating_sub(oldest_record);
        if uptime > 0 && record_age > uptime {
            return (cache, RestoreStatus::DiscardedBootCrossed);
        }

        let count = cache.restore_from_records(records);
        (cache, RestoreStatus::RestoredN(count))
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
    use std::fs;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

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
        assert_eq!(
            CachedActionKind::from_governor(GovernorDecision::Allow),
            None
        );
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

    #[test]
    fn load_missing_returns_empty() {
        let path = Path::new("/tmp/apollo_test_missing_cache.jsonl");
        let _ = fs::remove_file(path);
        let (cache, status) = RecentlyApplied::load_from_disk(path);
        assert!(cache.is_empty());
        assert_eq!(status, RestoreStatus::Missing);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let path = Path::new("/tmp/apollo_test_cache.jsonl");
        let _ = fs::remove_file(path);

        let mut cache = RecentlyApplied::new();
        cache.record(123, CachedActionKind::Throttle);
        cache.record(456, CachedActionKind::Freeze);

        cache.save_to_disk(path);
        assert!(path.exists());

        let (loaded, status) = RecentlyApplied::load_from_disk(path);
        assert_eq!(loaded.len(), 2, "expected 2 entries, status={status:?}");
        assert!(loaded.is_recent(123, CachedActionKind::Throttle));
        assert!(loaded.is_recent(456, CachedActionKind::Freeze));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn load_stale_data_fails_empty() {
        let path = Path::new("/tmp/apollo_test_stale_cache.jsonl");
        let _ = fs::remove_file(path);

        // Manually write stale data (2 minutes old, beyond 30s TTL)
        let now_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let stale_unix = now_unix - 120;

        let records = vec![PersistRecord {
            pid: 123,
            kind: CachedActionKind::Throttle,
            wall_unix_sec: stale_unix,
        }];
        fs::write(path, serde_json::to_string(&records).unwrap()).unwrap();

        let (loaded, _status) = RecentlyApplied::load_from_disk(path);
        assert!(loaded.is_empty(), "Stale data must result in empty cache");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn to_persist_records_yields_all_entries() {
        let mut cache = RecentlyApplied::new();
        cache.record(100, CachedActionKind::Throttle);
        cache.record(200, CachedActionKind::Freeze);
        let records = cache.to_persist_records();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn restore_from_records_skips_stale_entries() {
        let mut cache = RecentlyApplied::with_ttl(Duration::from_secs(30));
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let records = vec![
            PersistRecord {
                pid: 1,
                kind: CachedActionKind::Throttle,
                wall_unix_sec: now_unix - 10,
            }, // fresh
            PersistRecord {
                pid: 2,
                kind: CachedActionKind::Throttle,
                wall_unix_sec: now_unix - 60,
            }, // stale
        ];
        let restored = cache.restore_from_records(records);
        assert_eq!(restored, 1, "only fresh entry should restore");
        assert!(cache.is_recent(1, CachedActionKind::Throttle));
        assert!(!cache.is_recent(2, CachedActionKind::Throttle));
    }

    #[test]
    fn restore_then_persist_roundtrip() {
        let mut cache = RecentlyApplied::new();
        cache.record(42, CachedActionKind::SetMemorystatus);
        let records = cache.to_persist_records();
        let json: Vec<String> = records
            .iter()
            .map(|r| serde_json::to_string(r).unwrap())
            .collect();
        let parsed: Vec<PersistRecord> = json
            .iter()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        let mut cache2 = RecentlyApplied::new();
        let restored = cache2.restore_from_records(parsed);
        assert_eq!(restored, 1);
        assert!(cache2.is_recent(42, CachedActionKind::SetMemorystatus));
    }

    #[test]
    fn restore_empty_records_yields_zero() {
        let mut cache = RecentlyApplied::new();
        let restored = cache.restore_from_records(vec![]);
        assert_eq!(restored, 0);
        assert!(cache.is_empty());
    }
}
