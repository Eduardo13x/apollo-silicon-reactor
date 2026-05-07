# Sprint 2 — PidRecycled Mitigation + 1001 Pattern Audit + Dedup/Homologate

**Date:** 2026-05-07
**Frameworks:** apollo-evolve + apollo-nars + superpowers TDD + autoresearch
**Source:** `/Users/eduardocortez/Downloads/1001_patrones_final.pptx` (39 patterns + 8 anti-patterns)

## Context

Prior sprint (`5cf08fd`) achieved 73.7% journal success rate (10x baseline)
via universal chokepoint filter + ApplePlatform/ProtectedProcess pre-filters.
Remaining 26.3% is dominated by `PidRecycled` (race condition between
snapshot and execute).

User goal: attack the noise floor + apply 1001 patterns where applicable
+ wire unconnected modules + homologate redundant code, in a single
unified sprint.

NotebookLM-driven peer review confirmed previous iterations work. This
sprint extends that foundation.

## Goals

1. Reduce `PidRecycled` BlockReason audit-entry count by ≥80% via
   pre-emit identity re-verify (filter drops actions before they reach
   safety layer, so they don't appear as `block_reason: PidRecycled`)
2. Apply Idempotency, Inbox, Retry+jitter, Compensating Tx, ACL patterns
   from 1001 deck where missing/incomplete
3. Detect and unify duplicate logic (dedup/homologate)
4. Lift `journal_success_rate` from 73.7% → ≥85%
5. Maintain failures=0, p95 ≤ 80ms, no anti-pattern regressions

## Non-Goals (out of scope)

- main.rs Strangler Fig wave 41+ (God Service anti-pattern; defer)
- New DecisionReason variants (4/6 already firing)
- Phase B SetThreadQoS workload-trigger validation (SIP-bound)
- Schema migration for new persisted state files (no versioning yet)
- LLM teacher policy reset (NotebookLM Medium gap; defer)

## Phases

### Phase A — Idempotency Hardening (3-4 commits)

**A1 — Pre-emit identity re-verify** (`main.rs:3826` universal filter)
- For each action with `(pid, kind)`, call
  `process_identity::ProcessIdentity::from_pid(pid)`.
- If `None` → skip (process already dead)
- If `start_sec` mismatch vs action's `start_sec` → skip (PID recycled)
- Cost: ~3µs × ~50 actions/cycle = 150µs negligible
- Race window: ~10-100ms (snapshot→execute) → ~1ms (filter→execute)

**A2 — Re-verify at action_queue drain** (`main.rs:3832`)
- Actions queued cycle N may dispatch cycle N+1 (priority); PID can die
  during the gap.
- Apply same filter post-`drain_cycle()`.

**A3 — Test invariant**
- Unit: `pre_emit_identity_filter_drops_dead_pid`
- Unit: `pre_emit_identity_filter_keeps_alive_match`
- Integration: PidRecycled drop rate ≥80% in synthetic harness

### Phase B — 1001 Pattern Audit (4-5 commits)

**B1 — Inbox pattern persistence (FAIL-EMPTY restore)**
- `RecentlyApplied` is in-memory only; lost on daemon restart.
- Persist last N entries to `/var/lib/apollo/recently_applied.jsonl`
  on graceful shutdown.
- Persist record format: `{ pid, kind, instant_ns_since_boot, wall_clock }`.

**FAIL-EMPTY restore policy** (per peer-review consensus 2026-05-07):

Because TTL is 30s and idempotency layer (Phase A) already prevents
double-action, a stale cache after restart is more dangerous than
starting fresh. Apply extreme conservatism:

| Condition | Action |
|-----------|--------|
| File missing | Start empty (normal first boot) |
| File parse error / malformed JSON | Start empty + DELETE corrupt file (no panic) |
| Wall-clock delta write→read > 15s | Discard all entries (clock drift / long restart) |
| Boot-time crossed (monotonic resets) | Discard all entries |
| Per-entry wall-clock > 30s old | Drop that entry only |

**Trade-off**: allow at most ONE redundant action per process in the
first cycle after restart. Phase A idempotency layer + universal filter's
`is_recent` check absorb it harmlessly. Better than a ghost-cache false
positive starving a critical process of resources.

**Inline code contract** (per peer-review 2026-05-07):
```rust
// restore_recently_applied should be fail-empty:
//   any global integrity doubt => empty cache
//   per-entry staleness         => drop only that entry
```

**Telemetry metric** `recently_applied_restore_status` with 5 mutually-
exclusive states (added to `lf_metrics` + reported in `runtime_metrics.json`):
- `missing` — file did not exist on startup
- `restored_n` — successful restore (N entries)
- `discarded_corrupt` — parse error / malformed JSON
- `discarded_clock_delta` — wall-clock delta > 15s
- `discarded_boot_crossed` — monotonic time reset detected

This lets `NotebookLM` debrief distinguish "persistence is providing value"
from "always starts empty" — without this metric, B1's effectiveness is
unobservable.

**Why TTL is 30s and Inbox value is short-lived**: at 1-2s cycle time,
30s = 15-30 cycles, which is enough to suppress steady-state cross-cycle
dups but short enough that workload regime shifts get fresh evaluation.

**B2 — Retry+jitter audit**
- Search: `for _ in 0..N { ... }` retry loops without backoff.
- Targets: `csops`, `task_for_pid`, `memorystatus_control` retries.
- Apply: exponential backoff + ±25% jitter
  `[Anti-pattern: Retry Storm — 1001 patterns slide 57]`.

**B3 — Compensating Tx audit**
- Verify shutdown handler unfreezes ALL frozen PIDs.
- Locate in `daemon_init.rs` or shutdown path.
- Compensating tx for: SIGSTOP→SIGCONT, throttle→untrottle, sysctl→default.
- Add test: `shutdown_unfreezes_all`.

**B4 — ACL hygiene**
- `classify_protection()` should be the SINGLE source of safety truth.
- Grep direct calls to `is_protected_name`, `is_apple_platform_process`
  outside the filter chokepoint.
- Migrate non-chokepoint callers to `classify_protection()` where appropriate.

**B5 — Anti-pattern scan**
- No-Timeout: search `recv()`, `lock()`, `wait()` without timeout.
- Retry-Storm: post-B2 verify.
- Ignoring-Idempotency: post-A1+A2 verify.

### Phase C — Dedup/Homologate (3-4 commits)

**C1 — Find duplicate protection checks**
- `grep -rn "is_protected_name\|is_apple_platform_process\|classify_protection" src/`
- Expected: many call sites. Goal: most should call `classify_protection()`.

**C2 — Find duplicate dedup logic**
- `HashSet<u32>` in: `daemon_paging_hints`, `daemon_dispatch_tick`, others?
- Decide: replace with `RecentlyApplied` OR keep as orthogonal within-cycle layer.

**C3 — Unify**
- One commit per consolidation.
- All existing tests must pass (refactor preserves behavior).

### Phase D — Deploy + Measure (1-2 commits)

**D1 — Build + deploy + 250-cycle soak**
- `cargo build --release` + `sudo cp` + `launchctl bootout/bootstrap`
- Wait 250 cycles via Monitor tool
- Capture `journal_success_rate` (target ≥85%, baseline 73.7%)

**D2 — NotebookLM debrief**
- Push commits + metrics to notebook `8344b94c-a014-4803-abea-076a55753cfd`
- Gap sweep
- Write `evolve/2026-05-07-sprint2/final-debrief.md`

## Architecture

### Data Flow

```
[decide_actions] → actions
       ↓
[universal filter @ main.rs:3826]
   ├ classify_protection (block protected)
   ├ is_apple_platform (block SIP)
   ├ recently_applied dedup (cross-cycle)
   ├ NEW Phase A1: identity-still-valid? (block PidRecycled at emit)
   └ record(pid, kind) → cache
       ↓
[action_queue.push_all]
       ↓
[action_queue.drain_cycle]
   └ NEW Phase A2: re-verify identity (queued >1 cycle)
       ↓
[dispatch_tick → execute_actions]
       ↓
[verify_pid_identity] (existing last-line defense)
       ↓
[journal write + audit]
       ↓
[graceful shutdown]
   └ NEW Phase B1: persist recently_applied last-N entries
```

### Error Handling

- Per-phase rollback: `git revert` on failed guard (preserve history).
- Max 2 rework attempts per phase before discard.
- Stop sprint if Σ(OPENS) − Σ(CLOSES) > 5 (apollo-evolve divergence).
- Runtime: `from_pid` None → silent skip (expected, not error).
- Persist failures: best-effort, log warn, don't crash daemon.
- Retry exhaustion: log + continue.
- Shutdown hook errors: log + continue (don't block shutdown).
- Mutex poisoning: existing `lock_recover()` pattern.
- No new panic sites: `?` propagation or `Option::map`/`Result::map_err`.

### Self-diagnosis hooks (Phase 6 layer reused)

- A1/A2 increment `dedup_drops_*` counters tracked by `self_diagnosis`.
- B1 add new metric `recently_applied_persisted_entries` to `lf_metrics`.
- C: no new metrics.

## Testing

### Unit tests (Phase A)
- `pre_emit_identity_filter_drops_dead_pid` — `from_pid` returns None
- `pre_emit_identity_filter_drops_recycled_pid` — `start_sec` mismatch
- `pre_emit_identity_filter_keeps_alive_match` — happy path

### Unit tests (Phase B)
- B1: `recently_applied_persists_and_restores`
- B2: `retry_with_jitter_no_storm` (variance across 100 retries)
- B3: `shutdown_unfreezes_all` (N pids frozen → all unfrozen on shutdown)
- B5: each grep finding gets a regression test

### Phase C
- Zero new tests (refactor preserves behavior).
- Existing 1897 lib + 75 daemon tests must pass.

### Phase D (deploy verification)
- 250-cycle prod soak.
- `journal_success_rate ≥ 85%` (baseline 73.7%).
- `failures = 0` throughout.

## Mechanical Metrics

```
score = 60 * (success_rate - 0.737) / (0.95 - 0.737)
      + 20 * (PidRecycled_drop_pct / 100)
      + 10 * (anti_patterns_eliminated)
      + 10 * (LOC_reduced_via_dedup / 100)
```

Target ≥ 60.

## Stop Rules

- Apollo-evolve divergence: 2 consecutive commits OPENS>CLOSES → STOP
- Cumulative OPENS−CLOSES > 5 → STOP
- Test count regression → STOP (unless explicit refactor justification)
- p95 cycle ms increase > 20% → STOP and investigate

## Critical files to modify

| File | Phase | Reason |
|------|-------|--------|
| `src/bin/apollo-optimizerd/main.rs` (line 3826, 3832) | A1, A2 | Universal filter pre-emit + post-drain re-verify |
| `src/engine/recently_applied.rs` | B1 | Persist/restore methods |
| `src/bin/apollo-optimizerd/daemon_init.rs` | B1, B3 | Restore on startup; shutdown unfreeze hook |
| Various retry sites (TBD by B2 grep) | B2 | Backoff+jitter |
| Various protection-check sites (TBD by C1 grep) | C1, C3 | Migrate to `classify_protection` |
| `src/engine/lse_counters.rs` | B1 | New metric `recently_applied_persisted_entries` |

## References

- 1001 Patrones de Arquitectura (Spanish, 64 slides) — Idempotency,
  Inbox, Retry+Jitter, Compensating Tx, ACL, anti-patterns
- `[Saltzer & Kaashoek 2009 §3.3]` Complete Mediation
- `[Hellerstein 2004 §9]` State-aware feedback control
- `[Nygard 2018]` Release It! Ch. 5 (resilience patterns)
- `[Pei Wang]` NARS belief revision under multi-signal evidence
- Prior sprint: 21 commits over `7f2aae7`, success rate 7.4% → 73.7%
- NotebookLM project ID: `8344b94c-a014-4803-abea-076a55753cfd`
