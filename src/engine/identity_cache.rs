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
//! TTL is 30s (matches RecentlyApplied). Invalidation is conservative:
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
    /// Default 30s TTL, 5000-entry capacity.
    pub fn new() -> Self {
        Self::with_ttl(Duration::from_secs(30))
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
}

impl Default for IdentityCache {
    fn default() -> Self {
        Self::new()
    }
}
