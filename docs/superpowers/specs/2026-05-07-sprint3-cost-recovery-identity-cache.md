# Sprint 3 — Cost Recovery: Identity Cache + Sync Flush + Sysctl Reconcile

**Date:** 2026-05-07
**Frameworks:** apollo-evolve + apollo-nars + superpowers TDD + autoresearch + subagent-driven-development

## Context

Sprint 2 closed PidRecycled gap (67 → 0 audit blocks) by re-verifying
process identity at filter chokepoints (Phase A1+A2). Cost: per-action
`csops` + `proc_pidpath` syscalls accumulated, regressing p95 cycle ms
from 75.45 (Sprint 1 final) to 139 (+64ms).

NotebookLM peer review verdict: *"Sprint 2 reached Identity Honesty but
must reclaim hot-path budget."*

Sprint 3 = cost recovery. NOT new correctness. Three ranked tasks:

1. 🔴 **Identity-check cache** (Critical) — memoize `pid_identity_still_valid()`
   results to skip redundant `proc_pidpath` syscalls.
2. 🟠 **`sync_from_lockfree` 5-field flush** (High) — restore_status_*
   counters to runtime_metrics.json.
3. 🟡 **Governor ↔ Safety sysctl range reconcile** (Medium) — drop
   SysctlOutOfRange share from 35% → ≤2%.

## Goals

1. p95 cycle ms 139 → **≤80ms** (recovery to Sprint 1 baseline)
2. PidRecycled audit blocks remain at 0 (no regression)
3. failures remain at 0
4. journal success rate ≥73.7% over 500-event sample (recover to Sprint 1)
5. SysctlOutOfRange share ≤2% of fails
6. `restore_status_*` counters visible in `runtime_metrics.json`

## Non-Goals (out of scope)

- New emission paths (Sprint 1 + 2 covered all)
- New DecisionReason variants
- main.rs Strangler Fig wave 41+ (defer)
- LLM teacher policy reset
- Schema migration

## Architectural decision

> **RecentlyApplied evita actuar dos veces.**
> **IdentityCache evita dudar dos veces de la misma identidad ya verificada.**

Frontera clara: identity validation ≠ action dedup. Separating them
prevents future regressions where someone clears the action cache and
inadvertently affects identity semantics.

| Module | Responsibility |
|---|---|
| `RecentlyApplied` | Action dedup / cooldown ("did I act recently?") |
| `IdentityCache` (NEW) | Process identity validation / syscall amortization ("is this still the same entity?") |

## Phase A — IdentityCache module (Critical)

### A1 — Module skeleton + types

**File:** `src/engine/identity_cache.rs` (NEW)

```rust
//! IdentityCache — memoize process identity validation to skip redundant
//! `proc_pidpath` + `csops` syscalls on the hot path.
//!
//! Sprint 3 (2026-05-07) cost-recovery layer. Phase A1+A2 of Sprint 2
//! introduced per-action `ProcessIdentity::from_pid()` calls in the
//! universal filter and post-drain re-verify. On a daemon cycle with
//! N candidate actions, that's N×~3-7µs of csops + proc_pidpath
//! syscalls — accumulated to +64ms p95 regression.
//!
//! Architectural frontier:
//! - `RecentlyApplied`: "did I act recently?" (action dedup)
//! - `IdentityCache`: "is this still the same entity?" (identity validation)
//!
//! TTL is 30s, aligned with action dedup TTL. Invalidation is conservative:
//! - TTL expiry (per-entry)
//! - kqueue NOTE_EXIT for tracked PID
//! - path mismatch on refresh
//! - any caller-supplied force-refresh signal

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Composite key — never trust pid alone. Mirrors `verify_pid_identity`
/// at execute_actions.rs:220. start_sec is monotonic kernel boot ticks;
/// start_usec is microsecond resolution within that second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IdentityKey {
    pub pid: u32,
    pub start_sec: u64,
    pub start_usec: u64,
}

#[derive(Debug, Clone)]
pub struct IdentityCacheEntry {
    pub path_hash: u64,
    pub validated_at: Instant,
    pub expires_at: Instant,
}

/// Result of a validate_or_refresh call. Caller can act on each variant
/// distinctly for telemetry.
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

pub struct IdentityCache {
    ttl: Duration,
    entries: Mutex<HashMap<IdentityKey, IdentityCacheEntry>>,
    capacity: usize,
}

impl IdentityCache {
    pub fn new() -> Self { Self::with_ttl(Duration::from_secs(30)) }
    pub fn with_ttl(ttl: Duration) -> Self { ... }
    pub fn validate_or_refresh(&self, key: IdentityKey) -> IdentityValidation { ... }
    pub fn invalidate(&self, key: &IdentityKey) { ... }
    pub fn invalidate_pid(&self, pid: u32) { ... }
    pub fn cleanup_expired(&self) -> usize { ... }
    pub fn len(&self) -> usize { ... }
}
```

Tests (≥6):
- empty_cache_returns_validated_on_first_call
- cache_hit_within_ttl_returns_cached_valid
- ttl_expiry_forces_refresh
- invalidate_pid_evicts_all_keys_for_pid
- capacity_overflow_evicts_oldest
- dead_process_returns_dead

### A2 — Wire into `pid_identity_still_valid` helper

**File:** `src/bin/apollo-optimizerd/main.rs`

Modify `pid_identity_still_valid()` helper (added in Sprint 2 commit
`984f565`) to take `&IdentityCache` and short-circuit on cache hit.
Caller threads the cache via existing DaemonSubsystems.

```rust
fn pid_identity_still_valid(
    action: &RootAction,
    cache: &IdentityCache,
) -> bool {
    let (pid, start_sec, start_usec, name) = extract_identity(action);
    if pid == 0 { return true; } // non-PID action.
    let key = IdentityKey { pid, start_sec, start_usec };
    match cache.validate_or_refresh(key) {
        IdentityValidation::CachedValid => true,
        IdentityValidation::Validated => true,
        IdentityValidation::Invalid | IdentityValidation::Dead => false,
    }
}
```

The `validate_or_refresh()` internally calls `ProcessIdentity::from_pid()`
on miss (current path) and skips it on hit. Path-hash mismatch on refresh
forces eviction + revalidate.

### A3 — Hook kqueue NOTE_EXIT to invalidate

**File:** `src/bin/apollo-optimizerd/daemon_kqueue_tick.rs` (existing)

When kqueue reports `NOTE_EXIT` for a tracked PID, call
`identity_cache.invalidate_pid(pid)`. This ensures recycled PIDs do not
get a stale cache hit.

### A4 — Telemetry counters

**File:** `src/engine/lse_counters.rs` (extend)

7 new atomic counters per peer-review:
- `identity_cache_hits`
- `identity_cache_misses`
- `identity_cache_evictions`
- `identity_cache_ttl_expired`
- `identity_cache_exit_invalidations`
- `identity_proc_pidpath_calls`
- (computed: `identity_cache_hit_ratio` derivable from hits/(hits+misses))

Plus matching MetricsSnapshot fields and snapshot() loads.

### A5 — Guardrail tests

Tests asserting force-refresh in risk-signal cases:
- start_sec missing (=0) → no cache hit, force validation
- protected process classification changed → invalidate
- NOTE_EXIT observed → invalidate
- path missing on refresh → evict + return Dead

## Phase B — sync_from_lockfree 5-field flush (High)

**File:** `src/engine/daemon_state.rs`

Locate `pub fn sync_from_lockfree(&mut self, snap: &MetricsSnapshot)`.
Add 5 lines mirroring the Sprint 2 restore_status_* counter snapshot
fields (commit `407f717`):

```rust
        self.metrics.restore_status_missing = snap.restore_status_missing;
        self.metrics.restore_status_restored_n = snap.restore_status_restored_n;
        self.metrics.restore_status_discarded_corrupt = snap.restore_status_discarded_corrupt;
        self.metrics.restore_status_discarded_clock_delta = snap.restore_status_discarded_clock_delta;
        self.metrics.restore_status_discarded_boot_crossed = snap.restore_status_discarded_boot_crossed;
```

Plus matching `RuntimeMetrics` struct fields if not yet present.

Tests:
- `sync_from_lockfree_flushes_restore_status_fields`

## Phase C — Governor ↔ Safety sysctl range reconcile (Medium)

**File:** `src/engine/safety.rs` + `src/engine/sysctl_governor.rs` (or wherever Governor proposes sysctl values)

Sprint 2 measurement showed 11/31 = 35% of fails are `BlockReason::SysctlOutOfRange`.
This means Governor proposes values outside `safety::sysctl_allowed_range()`.

Two possible fixes (chose during implementation based on grep findings):

**Option C-1 — Narrow Governor output clamping**
Governor reads `safety::sysctl_allowed_range(key)` and clamps proposed
value to range before emitting `RootAction::SetSysctl`. Prevents emission
of doomed actions.

**Option C-2 — Widen safety ranges with kernel validation**
If Governor's proposed values are kernel-acceptable but safety overly
restrictive, expand allowed range with explicit kernel-acceptance test.

Likely Option C-1 wins (Governor is the producer; it should respect
consumer constraints). Audit during implementation.

Tests:
- `governor_clamps_proposed_value_to_safety_range`
- `governor_emits_no_action_when_proposed_equals_current`

## Verification

Mechanical checks:
1. `cargo test --lib` — ≥1907 + new tests
2. `cargo clippy --all-targets` — warnings flat or lower
3. Build release clean
4. Post-deploy 500-cycle soak:
   - p95 cycle ms ≤ 80ms (target)
   - failures = 0
   - PidRecycled = 0 (no regression)
   - identity_cache_hit_ratio ≥ 0.85 (high reuse)
   - identity_proc_pidpath_calls / cycle ≤ 5 (drastic drop from per-action)
   - SysctlOutOfRange ≤ 2% of fails
   - restore_status_* visible in runtime_metrics.json

## Stop rules

Per apollo-evolve discipline:
- 2 commits OPENS > CLOSES → STOP
- Cumulative OPENS − CLOSES > 5 → STOP
- p95 cycle ms NOT improved or regressed >10% → STOP and investigate
- Test count regression → STOP

## Implementation order (per peer-review)

1. Create `identity_cache.rs` (Phase A1)
2. Integrate around `pid_identity_still_valid()` (Phase A2)
3. Add cache invalidation on NOTE_EXIT (Phase A3)
4. Add 7 telemetry counters (Phase A4)
5. Add guardrail tests (Phase A5)
6. Build + canary deploy + verify p95 drops + PidRecycled stays 0 + failures=0
7. Phase B: sync_from_lockfree 5-field flush
8. Phase C: sysctl range reconcile
9. Final deploy + 500-cycle soak + NotebookLM debrief

## References

- Sprint 1 spec: `docs/superpowers/specs/2026-05-06-superplan-governor-state-memory.md`
- Sprint 2 spec: `docs/superpowers/specs/2026-05-07-sprint2-pidrecycled-patterns-dedup-design.md`
- Sprint 2 final debrief: `evolve/2026-05-07-sprint2/final-debrief.md`
- 1001 patterns: `/Users/eduardocortez/Downloads/1001_patrones_final.pptx`
- NotebookLM project: `8344b94c-a014-4803-abea-076a55753cfd`
- `pid_identity_still_valid` helper: commit `984f565`
- `verify_pid_identity` reference: `src/engine/execute_actions.rs:220-242`
