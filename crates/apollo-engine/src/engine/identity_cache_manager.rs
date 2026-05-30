//! IdentityCacheManager — single owner of the process-identity cache lifecycle.
//!
//! Sprint 4 Fase 2 (2026-05-07): consolidates the cache lifecycle that was
//! previously split across 4 sites:
//! - `main.rs::main()` constructed the bare `IdentityCache`
//! - `main.rs::pid_identity_still_valid` did lookup + syscall + insert inline
//! - `main.rs` periodic loop called `cleanup_expired()` every 60 cycles
//! - `daemon_kqueue_tick.rs` called `invalidate_pid()` on `ProcessExited`
//!
//! With more PID-exit observation sources possible (wait4 reaper, audit),
//! the lifecycle needs a single owner that concentrates the invariants:
//! "cache only ever holds entries for live processes; expiry is the safety
//! net when the eager invalidation path drops an event".
//!
//! The manager wraps `IdentityCache` (kept as a private field — direct API
//! tests live with `IdentityCache` in `identity_cache.rs`). Callers no
//! longer reach into the cache directly; they speak the manager's verbs:
//! - `verify(pid, name, start_sec, start_usec, &lf_metrics) -> bool` —
//!   cache-aware identity check, mirrors the old `pid_identity_still_valid`
//!   body but with the action-extraction step removed (callers use
//!   `RootAction::identity_fields()` first).
//! - `notify_exited(pid)` — eager invalidation hook for any PID-exit
//!   observation source (today: kqueue NOTE_EXIT).
//! - `tick_cleanup()` — periodic safety-net pass; caller controls cadence.
//!
//! Semantics-preserving over Fase 1: same TTL (180s default), same
//! capacity (5000), same prefix-tolerance, same start_sec=0 fallback for
//! validation, same cache-key behavior. Only the *owner* changes.

use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::engine::identity_cache::{IdentityCache, IdentityKey, IdentityValidation};
use crate::engine::lse_counters::{LockFreeMetrics, LSE_COUNTERS};
use crate::engine::process_identity::ProcessIdentity;

pub struct IdentityCacheManager {
    cache: IdentityCache,
}

impl IdentityCacheManager {
    pub fn new() -> Self {
        Self {
            cache: IdentityCache::new(),
        }
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            cache: IdentityCache::with_ttl(ttl),
        }
    }

    /// Cache-aware identity verification. Returns `true` iff the process at
    /// `pid` still matches the supplied identity tuple.
    ///
    /// Mirrors the old `pid_identity_still_valid` semantics exactly:
    /// 1. Composite-key cache lookup for actions carrying `start_sec`.
    /// 2. Cache miss → `ProcessIdentity::from_pid` + `matches` (Fase 1
    ///    single source of truth) + insert under cacheable key.
    ///
    /// Actions with `start_sec == 0` must take the syscall path every time:
    /// there is no birth timestamp to prove a cached PID still names the same
    /// process if an exit notification was missed.
    pub fn verify(
        &self,
        pid: u32,
        expected_name: Option<&str>,
        start_sec: u64,
        start_usec: u64,
        lf_metrics: &LockFreeMetrics,
    ) -> bool {
        let key = IdentityKey {
            pid,
            start_sec,
            start_usec,
        };

        match self.cache.validate_or_refresh(key, None) {
            IdentityValidation::CachedValid | IdentityValidation::Validated => {
                lf_metrics
                    .identity_cache_hits
                    .fetch_add(1, Ordering::Relaxed);
                return true;
            }
            IdentityValidation::Invalid => return false,
            IdentityValidation::Dead => { /* fall through to syscall */ }
        }

        lf_metrics
            .identity_cache_misses
            .fetch_add(1, Ordering::Relaxed);
        lf_metrics
            .identity_proc_pidpath_calls
            .fetch_add(1, Ordering::Relaxed);

        let current = match ProcessIdentity::from_pid(pid) {
            Some(id) => id,
            None => return false,
        };
        if !current.matches(expected_name, start_sec, start_usec) {
            return false;
        }

        // Insert under a cacheable key. start_sec=0 actions get the
        // freshly-fetched start_sec/start_usec so future PID-only lookups
        // succeed (Sprint 3 calibration fix preserved).
        let path_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            current.name.hash(&mut hasher);
            hasher.finish()
        };
        let cacheable_key = if start_sec == 0 {
            IdentityKey {
                pid,
                start_sec: current.start_sec,
                start_usec: current.start_usec,
            }
        } else {
            key
        };
        self.cache
            .validate_or_refresh(cacheable_key, Some(path_hash));
        true
    }

    /// Eager invalidation on observed process exit. Currently called from
    /// `daemon_kqueue_tick` on `NOTE_EXIT`; any future PID-exit source
    /// (wait4 reaper, audit feed) should call this same entry point to
    /// keep the cache honest before TTL expiry catches it.
    /// Returns the count of evicted entries. Bumps the global
    /// `identity_cache_exit_invalidations` LSE counter by that amount
    /// when n > 0 so dashboards can distinguish "exit-driven invalidation
    /// healthy" from "exit-driven invalidation dead" — visibility into
    /// the dead-PID guard fire rate (ffa0b29).
    pub fn notify_exited(&self, pid: u32) -> usize {
        let n = self.cache.invalidate_pid(pid);
        if n > 0 {
            LSE_COUNTERS
                .identity_cache_exit_invalidations
                .fetch_add(n as u64, Ordering::Relaxed);
        }
        n
    }

    /// Periodic safety-net cleanup of expired entries. Caller decides
    /// cadence (today: every 60 cycles in main loop). Returns drained count.
    /// Bumps the global `identity_cache_ttl_expired` LSE counter by the
    /// drained amount when n > 0 so dashboards can distinguish "TTL
    /// expiration healthy" from "TTL expiration dead".
    pub fn tick_cleanup(&self) -> usize {
        let n = self.cache.cleanup_expired();
        if n > 0 {
            LSE_COUNTERS
                .identity_cache_ttl_expired
                .fetch_add(n as u64, Ordering::Relaxed);
        }
        n
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Default for IdentityCacheManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty() {
        let m = IdentityCacheManager::new();
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
    }

    #[test]
    fn verify_dead_pid_returns_false() {
        let m = IdentityCacheManager::new();
        let lf = LockFreeMetrics::new();
        assert!(!m.verify(99_999, Some("anything"), 0, 0, &lf));
        // Counters: 1 miss + 1 proc_pidpath call, no hits.
        assert_eq!(lf.identity_cache_hits.load(Ordering::Relaxed), 0);
        assert_eq!(lf.identity_cache_misses.load(Ordering::Relaxed), 1);
        assert_eq!(lf.identity_proc_pidpath_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn verify_self_caches_for_subsequent_full_identity_lookup() {
        // Self pid is alive; first verify misses + inserts; second verify
        // with the full identity tuple should hit without another syscall.
        let m = IdentityCacheManager::new();
        let lf = LockFreeMetrics::new();
        let me = std::process::id();
        let id = ProcessIdentity::from_pid(me).unwrap();

        assert!(m.verify(me, Some(&id.name), id.start_sec, id.start_usec, &lf));
        assert_eq!(lf.identity_cache_hits.load(Ordering::Relaxed), 0);
        assert_eq!(lf.identity_cache_misses.load(Ordering::Relaxed), 1);

        assert!(m.verify(me, Some(&id.name), id.start_sec, id.start_usec, &lf));
        assert_eq!(lf.identity_cache_hits.load(Ordering::Relaxed), 1);
        assert_eq!(
            lf.identity_proc_pidpath_calls.load(Ordering::Relaxed),
            1,
            "second verify must not call proc_pidpath"
        );
    }

    #[test]
    fn start_sec_zero_does_not_reuse_pid_only_cache_hit() {
        let m = IdentityCacheManager::new();
        let lf = LockFreeMetrics::new();
        let me = std::process::id();
        let id = ProcessIdentity::from_pid(me).unwrap();

        assert!(m.verify(me, Some(&id.name), id.start_sec, id.start_usec, &lf));
        let calls_after_seed = lf.identity_proc_pidpath_calls.load(Ordering::Relaxed);

        assert!(m.verify(me, Some(&id.name), 0, 0, &lf));
        assert!(
            lf.identity_proc_pidpath_calls.load(Ordering::Relaxed) > calls_after_seed,
            "start_sec=0 must not trust a cached PID-only entry"
        );
    }

    #[test]
    fn notify_exited_evicts_entry_so_next_verify_is_a_miss() {
        let m = IdentityCacheManager::new();
        let lf = LockFreeMetrics::new();
        let me = std::process::id();
        let id = ProcessIdentity::from_pid(me).unwrap();

        // Seed the cache.
        assert!(m.verify(me, Some(&id.name), id.start_sec, id.start_usec, &lf));
        assert_eq!(m.len(), 1);

        let evicted = m.notify_exited(me);
        assert_eq!(evicted, 1);
        assert_eq!(m.len(), 0);

        // Next verify must take the miss path again.
        let misses_before = lf.identity_cache_misses.load(Ordering::Relaxed);
        let _ = m.verify(me, Some(&id.name), id.start_sec, id.start_usec, &lf);
        assert!(
            lf.identity_cache_misses.load(Ordering::Relaxed) > misses_before,
            "post-eviction verify must miss"
        );
    }

    #[test]
    fn notify_exited_unknown_pid_returns_zero() {
        let m = IdentityCacheManager::new();
        assert_eq!(m.notify_exited(123_456), 0);
        assert_eq!(m.notify_exited(0), 0);
    }

    #[test]
    fn tick_cleanup_drops_expired_entries() {
        let m = IdentityCacheManager::with_ttl(Duration::from_millis(40));
        let lf = LockFreeMetrics::new();
        let me = std::process::id();
        let id = ProcessIdentity::from_pid(me).unwrap();

        assert!(m.verify(me, Some(&id.name), id.start_sec, id.start_usec, &lf));
        assert_eq!(m.len(), 1);

        std::thread::sleep(Duration::from_millis(60));
        let drained = m.tick_cleanup();
        assert_eq!(drained, 1);
        assert_eq!(m.len(), 0);
    }

    /// Round-trip test: populate cache with 3 entries, call
    /// `notify_exited(pid)` on 1 known PID, assert the global
    /// `LSE_COUNTERS.identity_cache_exit_invalidations` bumped by 1, then
    /// snapshot LF and confirm the counter propagates into
    /// `RuntimeMetrics` via `sync_from_lockfree`.
    ///
    /// Closes the structural-zero hole: before this fix the counter was
    /// pinned at 0 despite 21+ freezes_applied — dashboards could not
    /// distinguish "exit-driven invalidation healthy" from "dead".
    #[test]
    fn test_invalidate_cached_enrich_bumps_counter() {
        use crate::engine::daemon_state::MetricsState;
        use crate::engine::lse_counters::LSE_COUNTERS;

        let m = IdentityCacheManager::new();
        let lf = LockFreeMetrics::new();

        // Seed 3 distinct identity entries — one to invalidate, two
        // controls that must survive.
        let target_pid: u32 = 4_242_001;
        let other_pid_a: u32 = 4_242_002;
        let other_pid_b: u32 = 4_242_003;
        // Use unique start_sec values so we don't collide with whatever
        // the host PID space is doing.
        let key_target = IdentityKey {
            pid: target_pid,
            start_sec: 17_000_001,
            start_usec: 1,
        };
        let key_a = IdentityKey {
            pid: other_pid_a,
            start_sec: 17_000_002,
            start_usec: 2,
        };
        let key_b = IdentityKey {
            pid: other_pid_b,
            start_sec: 17_000_003,
            start_usec: 3,
        };
        // Seed each via the same Validated path used in prod.
        assert_eq!(
            m.cache.validate_or_refresh(key_target, Some(0xa)),
            IdentityValidation::Validated
        );
        assert_eq!(
            m.cache.validate_or_refresh(key_a, Some(0xb)),
            IdentityValidation::Validated
        );
        assert_eq!(
            m.cache.validate_or_refresh(key_b, Some(0xc)),
            IdentityValidation::Validated
        );
        assert_eq!(m.len(), 3);

        // Snapshot the global LSE counter BEFORE invalidation — other
        // tests sharing this process may have bumped it.
        let before = LSE_COUNTERS
            .identity_cache_exit_invalidations
            .load(Ordering::Relaxed);

        // Act: simulate NOTE_EXIT for the target PID only.
        let evicted = m.notify_exited(target_pid);
        assert_eq!(evicted, 1, "exactly one entry must be evicted");

        // Global LSE counter must have bumped by exactly 1.
        let after = LSE_COUNTERS
            .identity_cache_exit_invalidations
            .load(Ordering::Relaxed);
        assert_eq!(
            after - before,
            1,
            "identity_cache_exit_invalidations must bump by 1"
        );

        // Control entries untouched.
        assert_eq!(m.len(), 2, "neighbour PIDs must survive");

        // Round-trip: snapshot → MetricsState → RuntimeMetrics.
        let snap = lf.snapshot();
        let _ = snap; // local LF is empty by design; the counter is global.

        // The flush path reads from a snapshot of the GLOBAL counters
        // (that's what the daemon does in main loop). Verify propagation:
        let global_snap = LSE_COUNTERS.snapshot();
        let mut metrics_state = MetricsState::default();
        metrics_state.sync_from_lockfree(&global_snap);
        assert_eq!(
            metrics_state.metrics.identity_cache_exit_invalidations,
            after,
            "RuntimeMetrics.identity_cache_exit_invalidations must mirror the global LSE counter"
        );
        assert!(
            metrics_state.metrics.identity_cache_exit_invalidations >= 1,
            "post-eviction RuntimeMetrics counter must be non-zero"
        );
    }
}
