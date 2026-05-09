//! IdentityCache — memoize process identity validation to skip redundant
//! `proc_pidpath` + `csops` syscalls on the hot path.
//!
//! Sprint 3 (2026-05-07) cost-recovery layer. Sprint 2 introduced
//! per-action `ProcessIdentity::from_pid()` calls in the universal filter
//! (commit 984f565). On a daemon cycle with N candidate actions, that's
//! N×~3-7µs of csops + proc_pidpath syscalls — accumulated to +64ms p95
//! regression. This cache amortizes the cost.
//!
//! Architectural frontier:
//! - `RecentlyApplied`: "did I act recently?" (action dedup)
//! - `IdentityCache`: "is this still the same entity?" (identity validation)
//!
//! TTL is 180s (calibrated 2026-05-07 from prod gaps: predictive-agent
//! retries same PID at 57-184s cadence; 30s TTL never caught repeats).
//! Invalidation is conservative:
//! - TTL expiry (per-entry, lazy on lookup)
//! - Periodic `cleanup_expired()` from daemon main loop
//! - `invalidate_pid()` callable on process exit observation
//!
//! [Cache-Aside (Lazy Loading) Pattern — 1001 patterns slide 11]

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Composite key — never trust pid alone. Mirrors verify_pid_identity at
/// execute_actions.rs:220. start_sec is monotonic kernel boot ticks;
/// start_usec is microsecond resolution within that second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IdentityKey {
    pub pid: u32,
    pub start_sec: u64,
    pub start_usec: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct IdentityCacheEntry {
    /// Hash of the proc_pidpath string. Changes if executable path changes
    /// (rare but possible on app updates / dlopen).
    pub path_hash: u64,
    pub validated_at: Instant,
    pub expires_at: Instant,
}

/// Result of a validate_or_refresh call. Telemetry path differs by variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityValidation {
    /// Cache hit — entry within TTL, no syscall.
    CachedValid,
    /// Cache miss → fresh validation succeeded (syscall happened).
    Validated,
    /// Cache hit OR fresh validation rejected — process changed identity.
    Invalid,
    /// Process is dead (proc_pidpath returned None).
    Dead,
}

/// LRU-bounded TTL cache for process identity validation.
pub struct IdentityCache {
    ttl: Duration,
    entries: Mutex<HashMap<IdentityKey, IdentityCacheEntry>>,
    capacity: usize,
}

impl IdentityCache {
    /// Default 180s TTL, 5000-entry capacity.
    pub fn new() -> Self {
        Self::with_ttl(Duration::from_secs(180))
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            ttl,
            entries: Mutex::new(HashMap::with_capacity(512)),
            capacity: 5000,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.lock().map(|g| g.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Validate process identity, using cache when fresh.
    /// Caller provides current path hash from `proc_pidpath` if cache misses.
    ///
    /// Behavior:
    /// - start_sec == 0 → forces validation, NEVER caches (legacy actions, no identity proof)
    /// - Cache hit + path_hash match (or fresh_path_hash None) + within TTL → CachedValid (no syscall)
    /// - Cache hit + path_hash mismatch → evict + return Invalid
    /// - Cache miss + Some(path_hash) → insert and return Validated
    /// - Cache miss + None → return Dead
    /// - Cache hit + expired → fall through to refresh path
    pub fn validate_or_refresh(
        &self,
        key: IdentityKey,
        fresh_path_hash: Option<u64>,
    ) -> IdentityValidation {
        // start_sec == 0 means caller doesn't have identity proof. Force refresh,
        // do NOT insert into cache (would lock the wrong entity).
        if key.start_sec == 0 {
            return match fresh_path_hash {
                Some(_) => IdentityValidation::Validated,
                None => IdentityValidation::Dead,
            };
        }

        let now = Instant::now();
        let mut entries = match self.entries.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };

        if let Some(entry) = entries.get(&key) {
            if now < entry.expires_at {
                // Path-hash mismatch on refresh forces eviction (rare).
                if let Some(fresh) = fresh_path_hash {
                    if fresh != entry.path_hash {
                        entries.remove(&key);
                        return IdentityValidation::Invalid;
                    }
                }
                return IdentityValidation::CachedValid;
            } else {
                // Expired — fall through to refresh.
                entries.remove(&key);
            }
        }

        // Cache miss OR expired. Caller's fresh_path_hash drives result.
        match fresh_path_hash {
            Some(hash) => {
                // Capacity check (cheap O(n) eviction when full).
                if entries.len() >= self.capacity {
                    if let Some((oldest_k, _)) = entries
                        .iter()
                        .min_by_key(|(_, e)| e.validated_at)
                        .map(|(k, e)| (*k, *e))
                    {
                        entries.remove(&oldest_k);
                    }
                }
                entries.insert(
                    key,
                    IdentityCacheEntry {
                        path_hash: hash,
                        validated_at: now,
                        expires_at: now + self.ttl,
                    },
                );
                IdentityValidation::Validated
            }
            None => IdentityValidation::Dead,
        }
    }

    /// PID-only lookup for callers without start_sec/start_usec proof
    /// (SetMemorystatus, Boost, Unfreeze, SetThreadQoS — actions that don't
    /// carry identity tuples). Returns Some(CachedValid) if any non-expired
    /// entry exists for this PID, else None. Caller MUST verify identity via
    /// syscall on None and insert with full key (Sprint 3 calibration fix).
    pub fn lookup_by_pid(&self, pid: u32) -> Option<IdentityValidation> {
        if pid == 0 {
            return None;
        }
        let now = Instant::now();
        let entries = match self.entries.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        for (k, e) in entries.iter() {
            if k.pid == pid && now < e.expires_at {
                return Some(IdentityValidation::CachedValid);
            }
        }
        None
    }

    /// Forget all entries for a PID. Call when caller knows process exited.
    /// Returns number of entries evicted.
    pub fn invalidate_pid(&self, pid: u32) -> usize {
        let mut entries = match self.entries.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let before = entries.len();
        entries.retain(|k, _| k.pid != pid);
        before - entries.len()
    }

    /// Sweep expired entries. O(n). Call periodically (e.g., every 60 cycles).
    /// Returns number of entries evicted.
    pub fn cleanup_expired(&self) -> usize {
        let mut entries = match self.entries.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let now = Instant::now();
        let before = entries.len();
        entries.retain(|_, e| now < e.expires_at);
        before - entries.len()
    }
}

impl Default for IdentityCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(pid: u32) -> IdentityKey {
        IdentityKey { pid, start_sec: 100, start_usec: 200 }
    }

    #[test]
    fn empty_cache_returns_validated_on_first_call() {
        let cache = IdentityCache::new();
        let r = cache.validate_or_refresh(key(123), Some(0xdead));
        assert_eq!(r, IdentityValidation::Validated);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_hit_within_ttl_returns_cached_valid() {
        let cache = IdentityCache::new();
        // Prime
        let r1 = cache.validate_or_refresh(key(123), Some(0xdead));
        assert_eq!(r1, IdentityValidation::Validated);
        // Hit
        let r2 = cache.validate_or_refresh(key(123), None);
        assert_eq!(r2, IdentityValidation::CachedValid);
    }

    #[test]
    fn ttl_expiry_forces_refresh() {
        let cache = IdentityCache::with_ttl(Duration::from_millis(40));
        cache.validate_or_refresh(key(123), Some(0xdead));
        std::thread::sleep(Duration::from_millis(70));
        // Now expired — should refresh on miss
        let r = cache.validate_or_refresh(key(123), Some(0xdead));
        assert_eq!(r, IdentityValidation::Validated);
    }

    #[test]
    fn invalidate_pid_evicts_all_keys_for_pid() {
        let cache = IdentityCache::new();
        cache.validate_or_refresh(
            IdentityKey { pid: 100, start_sec: 1, start_usec: 0 },
            Some(0xa),
        );
        cache.validate_or_refresh(
            IdentityKey { pid: 100, start_sec: 2, start_usec: 0 },
            Some(0xb),
        );
        cache.validate_or_refresh(
            IdentityKey { pid: 200, start_sec: 1, start_usec: 0 },
            Some(0xc),
        );
        let evicted = cache.invalidate_pid(100);
        assert_eq!(evicted, 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn dead_process_returns_dead() {
        let cache = IdentityCache::new();
        let r = cache.validate_or_refresh(key(99_999), None);
        assert_eq!(r, IdentityValidation::Dead);
        assert_eq!(cache.len(), 0, "dead process must not be cached");
    }

    #[test]
    fn path_hash_mismatch_forces_invalid() {
        let cache = IdentityCache::new();
        cache.validate_or_refresh(key(123), Some(0xdead));
        // Same key but different path hash → identity changed (e.g., exec replaced binary)
        let r = cache.validate_or_refresh(key(123), Some(0xbeef));
        assert_eq!(r, IdentityValidation::Invalid);
        // Must have been evicted.
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn start_sec_zero_forces_validation_no_cache_insert() {
        // Legacy actions with start_sec=0 must not be cached — no identity
        // proof. validate_or_refresh treats fresh_path_hash=Some as Validated
        // (caller did the syscall) but does NOT insert into cache.
        let cache = IdentityCache::new();
        let legacy_key = IdentityKey { pid: 123, start_sec: 0, start_usec: 0 };
        let r = cache.validate_or_refresh(legacy_key, Some(0xdead));
        assert_eq!(r, IdentityValidation::Validated);
        assert_eq!(cache.len(), 0, "start_sec=0 must NOT be cached");
    }

    #[test]
    fn start_sec_zero_with_no_path_returns_dead() {
        let cache = IdentityCache::new();
        let legacy_key = IdentityKey { pid: 123, start_sec: 0, start_usec: 0 };
        let r = cache.validate_or_refresh(legacy_key, None);
        assert_eq!(r, IdentityValidation::Dead);
    }

    #[test]
    fn cleanup_expired_drops_only_stale_entries() {
        let cache = IdentityCache::with_ttl(Duration::from_millis(40));
        cache.validate_or_refresh(key(1), Some(0xa));
        cache.validate_or_refresh(key(2), Some(0xb));
        std::thread::sleep(Duration::from_millis(70));
        cache.validate_or_refresh(key(3), Some(0xc)); // fresh
        let drained = cache.cleanup_expired();
        assert_eq!(drained, 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn invalidate_pid_zero_evicts_zero() {
        let cache = IdentityCache::new();
        cache.validate_or_refresh(key(123), Some(0xdead));
        let evicted = cache.invalidate_pid(0);
        assert_eq!(evicted, 0);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn lookup_by_pid_finds_any_fresh_entry_for_pid() {
        let cache = IdentityCache::new();
        cache.validate_or_refresh(
            IdentityKey { pid: 4242, start_sec: 1000, start_usec: 500 },
            Some(0xabcd),
        );
        assert_eq!(
            cache.lookup_by_pid(4242),
            Some(IdentityValidation::CachedValid)
        );
        assert_eq!(cache.lookup_by_pid(9999), None);
        assert_eq!(cache.lookup_by_pid(0), None);
    }

    #[test]
    fn lookup_by_pid_returns_none_after_ttl_expiry() {
        let cache = IdentityCache::with_ttl(Duration::from_millis(40));
        cache.validate_or_refresh(
            IdentityKey { pid: 5050, start_sec: 1000, start_usec: 500 },
            Some(0xbeef),
        );
        assert_eq!(
            cache.lookup_by_pid(5050),
            Some(IdentityValidation::CachedValid)
        );
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(cache.lookup_by_pid(5050), None);
    }
}
