# Sprint 3 Implementation Plan — Cost Recovery: IdentityCache + Sync Flush + Sysctl Reconcile

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reclaim hot-path budget after Sprint 2 PidRecycled fix: drop p95 cycle ms from 139 to ≤80 by amortizing per-action `proc_pidpath`/`csops` syscalls behind a new `IdentityCache`, plus close two minor architectural gaps (`sync_from_lockfree` flush + Governor↔Safety sysctl range reconcile).

**Architecture:** New `src/engine/identity_cache.rs` module memoizes `pid_identity_still_valid()` results for 30s. The Sprint 2 helper at `main.rs` becomes a thin wrapper calling `IdentityCache::validate_or_refresh()`. Periodic `cleanup_expired()` replaces NOTE_EXIT hook (kqueue_pressure doesn't emit per-process exits). Phase B adds 5 `u64` fields to `RuntimeMetrics` mirrored from `lse_counters`. Phase C adds Governor pre-emit clamp against `safety::allowlisted_sysctls_with_ranges()`.

**Tech Stack:** Rust 1.x, sysinfo, libc (csops, proc_pidpath), atomic counters, serde, existing apollo-optimizer engine modules.

---

## File Structure

| File | Phase | Purpose |
|------|-------|---------|
| `src/engine/identity_cache.rs` (NEW) | A1 | Module + types + tests |
| `src/engine/mod.rs` | A1 | Add `pub mod identity_cache;` |
| `src/bin/apollo-optimizerd/daemon_init.rs` | A2 | Add `identity_cache: IdentityCache` to `DaemonSubsystems` |
| `src/bin/apollo-optimizerd/main.rs` | A2 | Modify `pid_identity_still_valid()` to take `&IdentityCache`; thread cache through call sites |
| `src/bin/apollo-optimizerd/main.rs` | A3 | Periodic `cleanup_expired()` call (every 60 cycles) |
| `src/engine/lse_counters.rs` | A4 | 6 new atomic counters + MetricsSnapshot fields |
| `src/engine/identity_cache.rs` | A5 | Guardrail tests (start_sec=0 force-refresh, etc.) |
| `src/engine/types.rs` | B | 5 `u64` fields in `RuntimeMetrics` |
| `src/engine/daemon_state.rs` | B | 5 `sync_from_lockfree` mapping lines |
| `src/engine/sysctl_governor.rs` | C | Pre-emit clamp against allowlisted ranges |
| `evolve/2026-05-07-sprint3/baseline.tsv` | D | Pre/post measurement |
| `evolve/2026-05-07-sprint3/final-debrief.md` | D | NotebookLM debrief |

---

## Task 1: Phase A1.1 — IdentityCache module skeleton + types

**Files:**
- Create: `src/engine/identity_cache.rs`
- Modify: `src/engine/mod.rs`

- [ ] **Step 1.1: Create new module file with types**

Write to `src/engine/identity_cache.rs`:

```rust
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
```

Then add `pub mod identity_cache;` to `src/engine/mod.rs` after the existing `pub mod recently_applied;` line.

- [ ] **Step 1.2: Verify build is still clean**

Run: `cargo check --all-targets 2>&1 | tail -3`
Expected: `Finished dev profile` no errors.

- [ ] **Step 1.3: Run lib tests for regression**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: `1907 passed; 0 failed; 1 ignored`.

- [ ] **Step 1.4: Commit**

```bash
git add src/engine/identity_cache.rs src/engine/mod.rs
git commit -m "$(cat <<'EOF'
feat(identity_cache): module skeleton + types

Sprint 3 Phase A1.1 — foundation for hot-path syscall amortization.
New module src/engine/identity_cache.rs introduces:
- IdentityKey (pid, start_sec, start_usec) — composite key, never pid alone
- IdentityCacheEntry (path_hash, validated_at, expires_at)
- IdentityValidation enum (CachedValid, Validated, Invalid, Dead)
- IdentityCache with Mutex<HashMap>, 30s TTL, 5000 capacity

[Cache-Aside Pattern — 1001 patterns slide 11]

Architectural frontier vs RecentlyApplied:
- RecentlyApplied: "did I act recently?" (action dedup)
- IdentityCache: "is this still the same entity?" (identity validation)

OPENS: 1 (validate_or_refresh + tests pending in Task 2)
CLOSES: 0

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Phase A1.2 — `validate_or_refresh()` method + 6 tests

**Files:**
- Modify: `src/engine/identity_cache.rs`

- [ ] **Step 2.1: Add `validate_or_refresh()` and `invalidate_pid()` and `cleanup_expired()` methods**

Add inside `impl IdentityCache { ... }` block:

```rust
    /// Validate process identity, using cache when fresh.
    /// Caller provides current (start_sec, start_usec, path) snapshot from
    /// `ProcessIdentity::from_pid()` if cache misses.
    ///
    /// Behavior:
    /// - Cache hit + path_hash match + within TTL → CachedValid (no syscall)
    /// - Cache hit + path_hash mismatch → evict + Validated with new entry
    /// - Cache miss → Validated, insert new entry
    /// - Returns Dead if caller signals process is dead via fresh_path = None
    /// - start_sec == 0 forces refresh (legacy actions, no identity to lock)
    pub fn validate_or_refresh(
        &self,
        key: IdentityKey,
        fresh_path_hash: Option<u64>,
    ) -> IdentityValidation {
        // start_sec == 0 means caller doesn't have identity proof. Force refresh.
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
                        // Caller observed a fresh path but it differs — treat as
                        // identity change (e.g., exec replaced binary).
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

    /// Forget all entries for a PID. Call when caller knows process exited.
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
```

- [ ] **Step 2.2: Add 6 unit tests**

Add at the bottom of `src/engine/identity_cache.rs`:

```rust
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
}
```

- [ ] **Step 2.3: Run tests**

Run: `cargo test --lib identity_cache 2>&1 | tail -10`
Expected: 6 tests pass.

- [ ] **Step 2.4: Run full lib regression**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: `1913 passed; 0 failed; 1 ignored` (1907 + 6 new).

- [ ] **Step 2.5: Commit**

```bash
git add src/engine/identity_cache.rs
git commit -m "$(cat <<'EOF'
feat(identity_cache): validate_or_refresh + invalidate_pid + cleanup_expired

Sprint 3 Phase A1.2 — caching primitive complete.

## API

- validate_or_refresh(key, Option<fresh_path_hash>) -> IdentityValidation
  - CachedValid: hit within TTL with matching path_hash (no syscall)
  - Validated: miss/expired, fresh syscall result inserted
  - Invalid: cache hit but path_hash mismatch (binary replaced via exec)
  - Dead: cache miss + caller signaled None path
- invalidate_pid(pid) -> usize: evicts all entries for PID
- cleanup_expired() -> usize: O(n) sweep of stale entries

## Invariants

- start_sec == 0 forces refresh (legacy actions have no identity proof)
- path_hash mismatch on refresh evicts + returns Invalid
- TTL expiry handled lazily on lookup
- Capacity 5000, oldest-first eviction on overflow
- Mutex poisoning recovered via into_inner()

## Tests (6 new)

- empty_cache_returns_validated_on_first_call
- cache_hit_within_ttl_returns_cached_valid
- ttl_expiry_forces_refresh
- invalidate_pid_evicts_all_keys_for_pid
- dead_process_returns_dead
- path_hash_mismatch_forces_invalid

cargo test --lib: 1913 passed (was 1907; +6)

OPENS: 0
CLOSES: 1 (Task 1 OPENS — module is now functional)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Phase A2 — Wire IdentityCache through DaemonSubsystems + filter

**Files:**
- Modify: `src/bin/apollo-optimizerd/daemon_init.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs`

- [ ] **Step 3.1: Add `identity_cache` field to `DaemonSubsystems`**

In `src/bin/apollo-optimizerd/daemon_init.rs`, find `pub(super) struct DaemonSubsystems`. Add field after `recently_applied`:

```rust
    /// Identity validation cache (Sprint 3 cost recovery).
    /// Memoizes proc_pidpath/csops syscalls per (pid, start_sec, start_usec)
    /// for 30s. Drops Sprint 2 +64ms p95 regression.
    pub identity_cache: apollo_optimizer::engine::identity_cache::IdentityCache,
```

In `DaemonSubsystems::new()`, add to struct literal:

```rust
            identity_cache: apollo_optimizer::engine::identity_cache::IdentityCache::new(),
```

- [ ] **Step 3.2: Modify `pid_identity_still_valid` to take `&IdentityCache`**

In `src/bin/apollo-optimizerd/main.rs`, locate the helper added in Sprint 2 (commit 984f565). Search for `fn pid_identity_still_valid`. Replace its signature and body with:

```rust
/// Verify a RootAction's target PID still has the same identity at filter time.
///
/// Returns `true` if the action is safe to emit/dispatch, `false` if the PID is
/// dead or has been recycled (different process at same numeric PID).
///
/// Mirrors `execute_actions::verify_pid_identity` exactly:
/// - For per-PID actions: start_sec match + start_usec match (when both >0) +
///   name match (always evaluated as defense-in-depth).
/// - For non-PID actions (SetSysctl/ToggleSpotlight/QuarantineDaemon): always
///   returns true (no PID to verify).
///
/// Sprint 3 cost recovery: results memoized in `IdentityCache` for 30s.
/// Cache hit skips proc_pidpath/csops syscalls. Cache miss does the full
/// verify_pid_identity-equivalent check then inserts.
///
/// [Idempotency Pattern — 1001 patterns slide 7]
/// [Cache-Aside Pattern — 1001 patterns slide 11]
fn pid_identity_still_valid(
    action: &apollo_optimizer::engine::types::RootAction,
    cache: &apollo_optimizer::engine::identity_cache::IdentityCache,
    lf_metrics: &apollo_optimizer::engine::lse_counters::LockFreeMetrics,
) -> bool {
    use apollo_optimizer::engine::types::RootAction;
    use apollo_optimizer::engine::identity_cache::{IdentityCache, IdentityKey, IdentityValidation};

    let pid_opt = match action {
        RootAction::ThrottleProcess { pid, .. }
        | RootAction::FreezeProcess { pid, .. }
        | RootAction::UnfreezeProcess { pid, .. }
        | RootAction::BoostProcess { pid, .. }
        | RootAction::SetMemorystatus { pid, .. }
        | RootAction::SetThreadQoS { pid, .. } => Some(*pid),
        _ => None,
    };
    let pid = match pid_opt {
        Some(p) => p,
        None => return true, // non-PID actions always pass
    };
    let (action_start_sec, action_start_usec) = match action {
        RootAction::ThrottleProcess { start_sec, start_usec, .. }
        | RootAction::FreezeProcess { start_sec, start_usec, .. } => (*start_sec, *start_usec),
        _ => (0u64, 0u64),
    };
    let action_name: Option<&str> = match action {
        RootAction::ThrottleProcess { name, .. }
        | RootAction::FreezeProcess { name, .. }
        | RootAction::UnfreezeProcess { name, .. }
        | RootAction::BoostProcess { name, .. }
        | RootAction::SetThreadQoS { name, .. } => Some(name.as_str()),
        _ => None,
    };

    let key = IdentityKey {
        pid,
        start_sec: action_start_sec,
        start_usec: action_start_usec,
    };

    // Try cache first (no syscall on hit).
    let cached_probe = cache.validate_or_refresh(key, None);
    match cached_probe {
        IdentityValidation::CachedValid => {
            lf_metrics.identity_cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }
        IdentityValidation::Invalid | IdentityValidation::Dead => {
            // start_sec == 0 path or cached invalid — fall through to full check
            // below (we may still validate via syscall for legacy start_sec=0 actions).
        }
        IdentityValidation::Validated => {
            // Shouldn't happen with None fresh_path_hash, but be safe.
            lf_metrics.identity_cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }
    }

    // Cache miss → do the full verify_pid_identity-equivalent check.
    lf_metrics.identity_cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    lf_metrics.identity_proc_pidpath_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let current = match apollo_optimizer::engine::process_identity::ProcessIdentity::from_pid(pid) {
        Some(id) => id,
        None => return false, // dead process
    };
    if action_start_sec > 0 && current.start_sec != action_start_sec {
        return false; // PID recycled
    }
    if action_start_sec > 0
        && action_start_usec > 0
        && current.start_usec != action_start_usec
    {
        return false; // sub-second recycle
    }
    if let Some(expected) = action_name {
        let name_ok = current.name == expected
            || (current.name.len() >= 6 && expected.starts_with(&current.name))
            || (expected.len() >= 6 && current.name.starts_with(expected));
        if !name_ok {
            return false;
        }
    }

    // Insert into cache for next-time hit.
    let path_hash = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        current.name.hash(&mut hasher);
        hasher.finish()
    };
    cache.validate_or_refresh(key, Some(path_hash));
    true
}
```

- [ ] **Step 3.3: Update both call sites of `pid_identity_still_valid`**

Search for `pid_identity_still_valid(&action)` and `pid_identity_still_valid(a)` in main.rs. Both must now pass `&identity_cache, &lf_metrics`.

Update universal filter call site (around line ~3915):
```rust
                            if !pid_identity_still_valid(&action, &identity_cache, &lf_metrics) {
                                continue;
                            }
```

Update post-drain filter (around line ~3975):
```rust
                let final_actions: Vec<RootAction> = action_queue
                    .drain_cycle()
                    .into_iter()
                    .filter(|a| pid_identity_still_valid(a, &identity_cache, &lf_metrics))
                    .collect();
```

Note: `identity_cache` and `lf_metrics` must both be in scope at these call sites. `lf_metrics` already is (per Sprint 2). `identity_cache` is destructured from `DaemonSubsystems` — verify by reading the destructure block in main.rs.

- [ ] **Step 3.4: Run cargo check**

Run: `cargo check --all-targets 2>&1 | tail -5`
Expected: build clean. If errors about `identity_cache` not in scope, add it to the destructure block.

- [ ] **Step 3.5: Run lib tests**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: `1913 passed`.

- [ ] **Step 3.6: Commit**

```bash
git add src/bin/apollo-optimizerd/daemon_init.rs src/bin/apollo-optimizerd/main.rs
git commit -m "$(cat <<'EOF'
feat(filter): wire IdentityCache through pid_identity_still_valid helper

Sprint 3 Phase A2 — cost recovery wire-in. Sprint 2 helper at main.rs
(commit 984f565) now consults IdentityCache before doing proc_pidpath/csops
syscall.

## Path

1. validate_or_refresh(key, None) — probe cache only
2. CachedValid → return true (no syscall, lf_metrics.identity_cache_hits++)
3. Miss → identity_proc_pidpath_calls++ + ProcessIdentity::from_pid (full check)
4. On success → validate_or_refresh(key, Some(path_hash)) inserts entry
5. Subsequent ticks within TTL hit the cache

## Wired

- DaemonSubsystems gains `identity_cache: IdentityCache` field
- Universal filter (main.rs:~3915): cache-aware
- Post-drain filter (main.rs:~3975): cache-aware

## Hypothesis

Per-action proc_pidpath calls drop ~95% (high reuse expected for stable
foreground apps + frozen renderers). p95 cycle ms recovers from 139 →
~80ms (Sprint 1 baseline). PidRecycled stays at 0 (helper logic
unchanged).

OPENS: 1 (telemetry counters not yet defined — Task 5)
CLOSES: 1 (Task 2 OPENS — cache is now consumed)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Phase A3 — Periodic cleanup_expired() in main loop

**Files:**
- Modify: `src/bin/apollo-optimizerd/main.rs`

- [ ] **Step 4.1: Add cleanup call near existing recently_applied cleanup**

Search in main.rs for the pattern `recently_applied.cleanup_expired()` (added in Sprint 1 commit `c4acfd6`). It runs every 60 cycles. Add identity_cache cleanup adjacent:

```rust
                // Phase A3 (Sprint 3 2026-05-07) — periodic IdentityCache cleanup.
                // Lazy expiry on lookup is sufficient for correctness, but a
                // periodic sweep keeps memory bounded under sustained load.
                if cycle_count % 60 == 0 {
                    let drained = identity_cache.cleanup_expired();
                    if drained > 0 {
                        tracing::debug!(
                            target: "apollo.identity_cache",
                            drained,
                            remaining = identity_cache.len(),
                            "cache cleanup expired entries"
                        );
                    }
                }
```

Place this immediately after the existing `recently_applied.cleanup_expired()` block.

- [ ] **Step 4.2: Run cargo check**

Run: `cargo check --all-targets 2>&1 | tail -3`
Expected: clean.

- [ ] **Step 4.3: Run lib tests**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: `1913 passed`.

- [ ] **Step 4.4: Commit**

```bash
git add src/bin/apollo-optimizerd/main.rs
git commit -m "$(cat <<'EOF'
feat(daemon): periodic IdentityCache cleanup_expired every 60 cycles

Sprint 3 Phase A3 — lazy TTL expiry is sufficient for correctness, but
a periodic sweep keeps cache memory bounded under sustained load. Mirrors
RecentlyApplied cleanup pattern (commit c4acfd6) for consistency.

NOTE: original spec called for kqueue NOTE_EXIT hook to invalidate dead
PIDs, but daemon_kqueue_tick uses kqueue_pressure events which don't
emit per-process exits. EVFILT_PROC subscription per tracked PID is
expensive. Periodic sweep + lazy TTL covers the same goal at lower cost.

OPENS: 0
CLOSES: 0

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Phase A4 — IdentityCache telemetry counters

**Files:**
- Modify: `src/engine/lse_counters.rs`

- [ ] **Step 5.1: Add 6 atomic counters to LockFreeMetrics**

Locate `pub struct LockFreeMetrics` in `src/engine/lse_counters.rs`. Add near other counters:

```rust
    /// IdentityCache telemetry (Phase A4 — Sprint 3 cost recovery).
    /// Lets NotebookLM debrief verify the cache hit ratio and quantify
    /// proc_pidpath syscall amortization.
    pub identity_cache_hits: AtomicU64,
    pub identity_cache_misses: AtomicU64,
    pub identity_cache_evictions: AtomicU64,
    pub identity_cache_ttl_expired: AtomicU64,
    pub identity_cache_exit_invalidations: AtomicU64,
    pub identity_proc_pidpath_calls: AtomicU64,
```

- [ ] **Step 5.2: Initialize in `LockFreeMetrics::new()`**

Find the `impl LockFreeMetrics { pub const fn new() -> Self { Self { ... } }` block and add to struct literal:

```rust
            identity_cache_hits: AtomicU64::new(0),
            identity_cache_misses: AtomicU64::new(0),
            identity_cache_evictions: AtomicU64::new(0),
            identity_cache_ttl_expired: AtomicU64::new(0),
            identity_cache_exit_invalidations: AtomicU64::new(0),
            identity_proc_pidpath_calls: AtomicU64::new(0),
```

- [ ] **Step 5.3: Add to MetricsSnapshot struct**

Locate `pub struct MetricsSnapshot { ... }`. Add 6 fields:

```rust
    pub identity_cache_hits: u64,
    pub identity_cache_misses: u64,
    pub identity_cache_evictions: u64,
    pub identity_cache_ttl_expired: u64,
    pub identity_cache_exit_invalidations: u64,
    pub identity_proc_pidpath_calls: u64,
```

- [ ] **Step 5.4: Add load() calls in snapshot()**

Locate `pub fn snapshot(&self) -> MetricsSnapshot` and add 6 lines:

```rust
            identity_cache_hits: self.identity_cache_hits.load(Ordering::Relaxed),
            identity_cache_misses: self.identity_cache_misses.load(Ordering::Relaxed),
            identity_cache_evictions: self.identity_cache_evictions.load(Ordering::Relaxed),
            identity_cache_ttl_expired: self.identity_cache_ttl_expired.load(Ordering::Relaxed),
            identity_cache_exit_invalidations: self.identity_cache_exit_invalidations.load(Ordering::Relaxed),
            identity_proc_pidpath_calls: self.identity_proc_pidpath_calls.load(Ordering::Relaxed),
```

- [ ] **Step 5.5: Run cargo check**

Run: `cargo check --all-targets 2>&1 | tail -3`
Expected: clean.

- [ ] **Step 5.6: Run lib tests**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: `1913 passed`.

- [ ] **Step 5.7: Commit**

```bash
git add src/engine/lse_counters.rs
git commit -m "$(cat <<'EOF'
feat(lse_counters): identity_cache_* telemetry counters

Sprint 3 Phase A4 — 6 new atomic counters for NotebookLM observability:
- identity_cache_hits / misses / evictions / ttl_expired / exit_invalidations
- identity_proc_pidpath_calls (raw syscall count)

Hit ratio derivable as hits / (hits + misses). Validates that p95
recovery is genuinely from amortization, not measurement noise.

OPENS: 0
CLOSES: 1 (Task 3 OPENS — telemetry counters now defined)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Phase A5 — Guardrail tests

**Files:**
- Modify: `src/engine/identity_cache.rs`

- [ ] **Step 6.1: Add 4 guardrail tests**

Append to existing `mod tests` in `src/engine/identity_cache.rs`:

```rust
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
```

- [ ] **Step 6.2: Run new tests**

Run: `cargo test --lib identity_cache 2>&1 | tail -10`
Expected: 10 tests pass (6 prior + 4 guardrail).

- [ ] **Step 6.3: Run full lib regression**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: `1917 passed`.

- [ ] **Step 6.4: Commit**

```bash
git add src/engine/identity_cache.rs
git commit -m "$(cat <<'EOF'
test(identity_cache): guardrail tests for risk signals

Sprint 3 Phase A5 — 4 new tests pin behavioral guarantees that prevent
silent cache poisoning:

- start_sec_zero_forces_validation_no_cache_insert: legacy start_sec=0
  actions never enter the cache (no identity proof to lock against).
- start_sec_zero_with_no_path_returns_dead: missing fresh_path_hash on
  start_sec=0 path returns Dead correctly.
- cleanup_expired_drops_only_stale_entries: per-entry expiry, fresh
  entries preserved.
- invalidate_pid_zero_evicts_zero: invalidating PID 0 (kernel_task) is
  a safe no-op (no entries to evict).

cargo test --lib: 1917 passed (was 1913; +4)

OPENS: 0
CLOSES: 0

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Phase B — sync_from_lockfree 5-field flush

**Files:**
- Modify: `src/engine/types.rs` (add 5 fields to RuntimeMetrics)
- Modify: `src/engine/daemon_state.rs` (add 5 mapping lines)

- [ ] **Step 7.1: Add 5 u64 fields to RuntimeMetrics**

In `src/engine/types.rs`, locate `pub struct RuntimeMetrics`. Find the section near `pub refresh_duration_ms: f64` (around line 589). Add 5 new fields after the existing `recently_applied_restore_status: Option<RestoreStatus>` field (around line 804):

```rust
    /// Restore status telemetry (Sprint 3 Phase B — flushed from lf_metrics).
    /// Mutually-exclusive: at most one is non-zero per startup.
    /// Replaces/parallels the legacy `recently_applied_restore_status` Option.
    #[serde(default)]
    pub restore_status_missing: u64,
    #[serde(default)]
    pub restore_status_restored_n: u64,
    #[serde(default)]
    pub restore_status_discarded_corrupt: u64,
    #[serde(default)]
    pub restore_status_discarded_clock_delta: u64,
    #[serde(default)]
    pub restore_status_discarded_boot_crossed: u64,
```

The `#[serde(default)]` is required for backward-compat with existing JSON files that lack these fields.

- [ ] **Step 7.2: Add 5 mapping lines in sync_from_lockfree**

In `src/engine/daemon_state.rs`, locate `pub fn sync_from_lockfree(&mut self, lf: &crate::engine::lse_counters::MetricsSnapshot)` (around line 57). After the existing `self.metrics.refresh_duration_ms = lf.refresh_duration_us as f64 / 1000.0;` line, add:

```rust
        // Sprint 3 Phase B — flush restore_status_* counters from lf to runtime metrics.
        self.metrics.restore_status_missing = lf.restore_status_missing;
        self.metrics.restore_status_restored_n = lf.restore_status_restored_n;
        self.metrics.restore_status_discarded_corrupt = lf.restore_status_discarded_corrupt;
        self.metrics.restore_status_discarded_clock_delta = lf.restore_status_discarded_clock_delta;
        self.metrics.restore_status_discarded_boot_crossed = lf.restore_status_discarded_boot_crossed;
```

- [ ] **Step 7.3: Run cargo check**

Run: `cargo check --all-targets 2>&1 | tail -3`
Expected: clean.

- [ ] **Step 7.4: Run lib tests**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: `1917 passed`.

- [ ] **Step 7.5: Commit**

```bash
git add src/engine/types.rs src/engine/daemon_state.rs
git commit -m "$(cat <<'EOF'
feat(metrics): Phase B — flush restore_status_* to runtime_metrics.json

Sprint 3 Phase B closes Sprint 2 telemetry gap: lf_metrics counters
(commit 407f717) had no path to runtime_metrics.json because
sync_from_lockfree didn't include them.

## Changes

- types.rs: 5 new u64 fields on RuntimeMetrics (#[serde(default)])
  - restore_status_missing
  - restore_status_restored_n
  - restore_status_discarded_corrupt
  - restore_status_discarded_clock_delta
  - restore_status_discarded_boot_crossed
- daemon_state.rs: 5 mapping lines in sync_from_lockfree()

NotebookLM debrief can now distinguish "persistence helps" from "always
starts empty" via runtime_metrics.json, closing the architectural gap
NotebookLM flagged 🟠 High in Sprint 2 final review.

OPENS: 0
CLOSES: 1 (Sprint 2 sync_from_lockfree gap)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Phase C — Governor pre-emit clamp against safety ranges

**Files:**
- Modify: `src/engine/sysctl_governor.rs`

- [ ] **Step 8.1: Read sysctl_governor proposal sites**

Run: `grep -n "RootAction::SetSysctl" src/engine/sysctl_governor.rs`
Expected: lines 338, 355, 467, 1038 (per pre-grep). Identify the 4 emission sites.

- [ ] **Step 8.2: Add private helper `clamp_to_allowed_range`**

Add near the top of `src/engine/sysctl_governor.rs` (after imports, before existing public API):

```rust
/// Clamp a proposed sysctl value to the allowed range from
/// `safety::allowlisted_sysctls_with_ranges()`.
///
/// Sprint 3 Phase C — Governor↔Safety contract reconcile. Sprint 2
/// measurement showed 35% of journal failures were `BlockReason::SysctlOutOfRange`.
/// Preventing emission for out-of-range values eliminates wasted journal
/// entries and audit log noise.
///
/// Returns clamped value (always within allowed range), or the original
/// if no range exists for this key.
fn clamp_to_allowed_range(key: &str, proposed: i64) -> i64 {
    let ranges = crate::engine::safety::allowlisted_sysctls_with_ranges();
    if let Some(r) = ranges.iter().find(|r| r.key == key) {
        proposed.clamp(r.min, r.max)
    } else {
        // Key not in allowlist — execute_actions will reject with
        // InvalidSysctl. Pass through unchanged so the failure surfaces.
        proposed
    }
}
```

- [ ] **Step 8.3: Wrap each `RootAction::SetSysctl` emission to clamp values**

For EACH of the 4 emission sites identified in Step 8.1, wrap the `value` field with `clamp_to_allowed_range`. The current pattern is approximately:

```rust
RootAction::SetSysctl { key, value, reason, decision_reason }
```

where `value` is something like `format!("{}", numeric)`. Convert to: parse → clamp → re-stringify.

For each site, replace:
```rust
.map(|(key, value)| RootAction::SetSysctl {
    key,
    value,
    ...
```

with:
```rust
.map(|(key, value)| {
    let clamped_value = match value.parse::<i64>() {
        Ok(n) => format!("{}", clamp_to_allowed_range(&key, n)),
        Err(_) => value, // non-numeric or already string — pass through
    };
    RootAction::SetSysctl {
        key,
        value: clamped_value,
        ...
    }
})
```

This is structurally identical at each site; the inner closure body is the only change. Apply at lines 338, 355, 467, 1038. The exact tuple shape may differ (some sites have `(key, value, reason)` tuples) — adapt the pattern.

- [ ] **Step 8.4: Add unit test for clamp**

Add at the bottom of `src/engine/sysctl_governor.rs` (inside existing `mod tests` block):

```rust
    #[test]
    fn clamp_value_within_range_passes_through() {
        // kern.maxproc has range [256, 65536] in safety.rs.
        let v = clamp_to_allowed_range("kern.maxproc", 1024);
        assert_eq!(v, 1024);
    }

    #[test]
    fn clamp_value_above_max_clamps_to_max() {
        // kern.maxproc max is 65536.
        let v = clamp_to_allowed_range("kern.maxproc", 999_999);
        assert!(v <= 65536, "clamped value must not exceed max: got {}", v);
    }

    #[test]
    fn clamp_value_below_min_clamps_to_min() {
        let v = clamp_to_allowed_range("kern.maxproc", 0);
        assert!(v >= 256, "clamped value must not fall below min: got {}", v);
    }

    #[test]
    fn clamp_unknown_key_passes_through() {
        let v = clamp_to_allowed_range("not.in.allowlist", 12345);
        assert_eq!(v, 12345);
    }
```

(Adjust min/max numbers in the test body to match the actual values in `safety::allowlisted_sysctls_with_ranges()` for `kern.maxproc` — read that function once and pin the assertions.)

- [ ] **Step 8.5: Run new tests**

Run: `cargo test --lib clamp_ 2>&1 | tail -10`
Expected: 4 tests pass.

- [ ] **Step 8.6: Run full lib regression**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: `1921 passed`.

- [ ] **Step 8.7: Commit**

```bash
git add src/engine/sysctl_governor.rs
git commit -m "$(cat <<'EOF'
fix(governor): clamp proposed sysctl values to safety allowed ranges

Sprint 3 Phase C — Governor↔Safety contract reconcile. Sprint 2
measurement showed 11/31 = 35% of journal failures were
BlockReason::SysctlOutOfRange. Cause: Governor proposed values outside
safety::allowlisted_sysctls_with_ranges() and execute_actions rejected
at the safety layer.

## Fix

New private helper clamp_to_allowed_range(key, proposed) that reads
safety ranges once and clamps proposed values to [min, max]. Applied at
all 4 RootAction::SetSysctl emission sites in sysctl_governor.rs.

Unknown keys pass through (execute_actions still rejects with
InvalidSysctl — the failure remains visible).

## Tests (4 new)

- clamp_value_within_range_passes_through
- clamp_value_above_max_clamps_to_max
- clamp_value_below_min_clamps_to_min
- clamp_unknown_key_passes_through

cargo test --lib: 1921 passed (was 1917; +4)

[Anti-Corruption Layer Pattern — 1001 patterns slide 48]

OPENS: 0
CLOSES: 1 (Sprint 2 SysctlOutOfRange gap class — 35% → expected ≤2%)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Phase D1 — Build, deploy, 500-cycle soak

**Files:**
- Run: cargo build, deploy script, Monitor tool

- [ ] **Step 9.1: Pre-deploy gates**

Run:
```bash
cargo test --lib 2>&1 | tail -3
cargo test --bin apollo-optimizerd 2>&1 | tail -3
cargo clippy --all-targets 2>&1 | grep -c "warning"
```
Expected: 1921 lib + ~75 daemon, warnings count ≤ baseline.

- [ ] **Step 9.2: Build release**

Run: `cargo build --release --bin apollo-optimizerd 2>&1 | tail -3`
Expected: `Finished release profile` clean.

- [ ] **Step 9.3: Capture pre-deploy baseline**

```bash
mkdir -p evolve/2026-05-07-sprint3
sudo apollo-optimizerctl status 2>/dev/null > /tmp/sprint3_pre_status.json
sudo tail -500 /var/lib/apollo/policy_audit.jsonl > /tmp/sprint3_pre_audit.jsonl
sudo tail -500 /var/lib/apollo/journal.jsonl > /tmp/sprint3_pre_journal.jsonl
sudo cat /var/lib/apollo/runtime_metrics.json > /tmp/sprint3_pre_runtime.json
```

Write `evolve/2026-05-07-sprint3/baseline.tsv` capturing key Sprint 2 metrics before deploy.

- [ ] **Step 9.4: Deploy + restart**

```bash
sudo cp target/release/apollo-optimizerd /usr/local/libexec/apollo-optimizerd
sudo launchctl bootout system/com.eduardocortez.systemoptimizerd 2>&1; sleep 3
sudo launchctl bootstrap system /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist
sleep 5
ps aux | grep apollo-optimizerd | grep -v grep
DEPLOY_TS=$(date -u "+%Y-%m-%dT%H:%M:%SZ")
echo "DEPLOY_TS=$DEPLOY_TS" > /tmp/sprint3_deploy_marker.txt
```

Expected: new daemon PID running, started within last 10s.

- [ ] **Step 9.5: Monitor 500 cycles**

Use the Monitor tool with this command:

```bash
while true; do
  c=$(sudo apollo-optimizerctl status 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin).get("metrics",{}).get("cycles",0))')
  if [ "$c" -ge 500 ]; then echo "READY cycles=$c"; break; fi
  sleep 15
done
```

Expected: notification with `READY cycles=500+`.

- [ ] **Step 9.6: Capture post-soak measurement**

```bash
sudo apollo-optimizerctl status 2>/dev/null > /tmp/sprint3_post_status.json
sudo tail -500 /var/lib/apollo/policy_audit.jsonl > /tmp/sprint3_post_audit.jsonl
sudo tail -500 /var/lib/apollo/journal.jsonl > /tmp/sprint3_post_journal.jsonl
sudo cat /var/lib/apollo/runtime_metrics.json > /tmp/sprint3_post_runtime.json
```

Write a small Python analysis script at `/tmp/sprint3_measure.py` that filters fresh-daemon-only entries (timestamp > DEPLOY_TS) and reports:
- p95_cycle_ms (target ≤80)
- failures (must be 0)
- PidRecycled audit blocks (must remain 0)
- identity_cache_hits, _misses, _evictions, _ttl_expired (computed hit ratio ≥ 0.85)
- identity_proc_pidpath_calls / cycles (target ≤5)
- SysctlOutOfRange share of fails (target ≤2%)
- restore_status_* in runtime_metrics.json (one of 5 should be non-zero)
- journal success rate over fresh sample (target ≥73.7%)

Run: `python3 /tmp/sprint3_measure.py`

Append output to `evolve/2026-05-07-sprint3/baseline.tsv`.

- [ ] **Step 9.7: Commit measurement**

```bash
git add evolve/2026-05-07-sprint3/baseline.tsv
git commit -m "$(cat <<'EOF'
chore(deploy): Sprint 3 deploy + 500-cycle soak measurement

Pre/post comparison captured. Mechanical metric targets:
- p95_cycle_ms ≤ 80ms (recovery from Sprint 2 139ms regression)
- failures = 0
- PidRecycled remains 0 (no regression)
- identity_cache hit ratio ≥ 0.85
- proc_pidpath calls / cycle ≤ 5 (drastic drop from per-action)
- SysctlOutOfRange ≤ 2% of fails
- restore_status_* visible in runtime_metrics.json

[Verification per Sprint 3 spec §Verification]

OPENS: 0
CLOSES: 0

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Phase D2 — NotebookLM debrief

**Files:**
- Create: `evolve/2026-05-07-sprint3/final-debrief.md`
- MCP: `notebook_query` to project notebook `8344b94c-a014-4803-abea-076a55753cfd`

- [ ] **Step 10.1: Push session deltas to NotebookLM**

Use `mcp__notebooklm-mcp__notebook_query` with a comprehensive query containing:
- Sprint 3 commit list (Tasks 1–9)
- Headline metrics (p95 recovery, hit ratio, failures, PidRecycled)
- Compared vs Sprint 2 baseline
- restore_status_* counter results
- SysctlOutOfRange before/after
- Open question: any newly-exposed gap class?

Wait for response. Capture the gap-sweep result.

- [ ] **Step 10.2: Write final debrief markdown**

Create `evolve/2026-05-07-sprint3/final-debrief.md` with sections:

```markdown
# Sprint 3 Final Debrief — Cost Recovery

**Date:** 2026-05-07
**Sprint commits:** 10 atomic over Sprint 2 base `4618c02`
**Frameworks:** apollo-evolve + apollo-nars + superpowers TDD + autoresearch + subagent-driven-development

## Headline metrics

| Metric | Sprint 2 baseline | Sprint 3 final | Δ | Verdict |
|---|---|---|---|---|
| p95_cycle_ms | 139 | <RESULT> | | <PASS/FAIL target ≤80> |
| failures | 0 | 0 | flat | ✅ |
| PidRecycled blocks | 0 | <RESULT> | | <flat = ✅> |
| identity_cache hit ratio | n/a | <RESULT> | | <≥0.85 = ✅> |
| proc_pidpath / cycle | many | <RESULT> | | <≤5 = ✅> |
| SysctlOutOfRange share | 35% | <RESULT> | | <≤2% = ✅> |
| restore_status_* in JSON | absent | <RESULT> | | <visible = ✅> |
| Tests passing | 1907 | 1921 | +14 | ✅ |

## Iteration log (10 commits)

(List Tasks 1–10 with their commit SHAs and one-line descriptions.)

## NotebookLM peer-review final

(Paste gap-sweep response.)

## What Sprint 3 delivered

- IdentityCache module (10 tests)
- Hot-path syscall amortization wired
- Periodic cleanup_expired
- 6 telemetry counters
- sync_from_lockfree 5-field flush
- Governor↔Safety sysctl range reconcile

## What Sprint 3 did NOT deliver

(Honest list of any missed targets.)

## Sprint 3 frameworks evaluation

(One paragraph each on apollo-evolve, NARS, TDD, autoresearch, subagent-driven.)

## Next-session priorities

(Per NotebookLM, ranked Critical/High/Medium/Low.)

## Verdict

(Closing line.)

**Sprint 3 closed. <N> commits total over original `7f2aae7` (sprint arc accumulator).**
```

Replace `<RESULT>` placeholders with the actual measurements from Task 9.6 output.

- [ ] **Step 10.3: Commit final debrief**

```bash
git add evolve/2026-05-07-sprint3/final-debrief.md
git commit -m "$(cat <<'EOF'
docs(evolve): Sprint 3 final debrief — cost recovery via IdentityCache

NotebookLM-driven peer review final. Sprint 3 closed:
- Phase A (IdentityCache module) — A1+A2+A3+A4+A5 wired
- Phase B (sync_from_lockfree flush) — restore_status_* in JSON
- Phase C (Governor↔Safety) — sysctl range reconcile
- Phase D (deploy + measure) — 500-cycle soak + NotebookLM debrief

[Sprint 3 closed per spec §Verification and §Stop Rules]

OPENS: 0
CLOSES: 1 (Sprint 3 wrap)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

After plan written, verified:

**1. Spec coverage** — every spec section maps to a task:
- ✓ Phase A1 → Tasks 1, 2 (skeleton + types, validate_or_refresh + tests)
- ✓ Phase A2 → Task 3 (wire DaemonSubsystems + filter)
- ✓ Phase A3 → Task 4 (periodic cleanup_expired) — kqueue NOTE_EXIT replaced with sweep per pre-grep finding
- ✓ Phase A4 → Task 5 (telemetry counters)
- ✓ Phase A5 → Task 6 (guardrail tests)
- ✓ Phase B → Task 7 (RuntimeMetrics fields + sync flush)
- ✓ Phase C → Task 8 (Governor clamp helper + 4-site wrap + tests)
- ✓ Phase D → Tasks 9, 10 (deploy + debrief)

**2. Placeholder scan** — no "TBD", no "implement later". Phase D2 has `<RESULT>` placeholders intentionally left for measurement substitution.

**3. Type consistency** — `IdentityKey`, `IdentityCacheEntry`, `IdentityValidation`, `IdentityCache` references consistent across Tasks 1–6.

**4. Stop rules respected** — Tasks 4 (cleanup), Tasks 7-8 (Phase B/C) are small additions; Task 9 has hard p95 ≤80 target with stop rules.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-07-sprint3-cost-recovery-identity-cache.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task; review between tasks; fast iteration with isolated context per task.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`; batch execution with checkpoints for review.

Which approach?
