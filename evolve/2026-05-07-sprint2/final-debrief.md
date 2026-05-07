# Sprint 2 Final Debrief — PidRecycled + 1001 Patterns + Dedup

**Date:** 2026-05-07
**Sprint commits:** 15 atomic over Sprint 1 base `5cf08fd`
**Frameworks:** apollo-evolve + apollo-nars + superpowers TDD + autoresearch + subagent-driven-development

## Headline metrics

| Metric | Sprint 1 baseline | Sprint 2 final | Δ | Verdict |
|---|---|---|---|---|
| **PidRecycled audit blocks** | 67 | **0** | **-100%** | ✅ Target ≥80% drop MET |
| **journal pid-recycled skips** | non-zero | **0** | -100% | ✅ |
| **failures** | 0 | 0 | flat | ✅ |
| **Tests passing** | 1885 | 1907 | +22 | ✅ |
| journal success rate | 73.7% | 64.5% | -9pp | ⚠ small-N noise (n=31 vs 500) |
| p95 cycle ms | 75.45 | 139 | **+64** | 🔴 regression |
| restore_status_* in JSON | n/a | all-zero | gap | 🟠 sync_from_lockfree gap |

## Iteration log (15 commits)

### Phase A — Idempotency Hardening
- `17d81d5` Phase A1 pre-emit identity re-verify (universal filter)
- `57d1e9c` Fix: include start_usec + name fallback (mirror verify_pid_identity)
- `8bd2d36` Phase A2 post-drain re-verify
- `984f565` Helper extraction `pid_identity_still_valid` (eliminates A1+A2 duplication)
- `bc1e0ff` 3 process_identity unit tests

### Phase B — 1001 Pattern Audit
- `386382f` B1.1 PersistRecord + RestoreStatus types (serde derives)
- `6ee329e` B1.2 persist/restore methods + 4 tests + RecentlyApplied::save/load
- `407f717` B1.3 restore_status_* atomic counters (5 fields)
- `fb31be9` B1.4 wire load_from_disk on startup + save on shutdown
- `58a0b59` B1.5 wire restore_status to lf_metrics
- `cc964d3` B2 retry+jitter audit PASS
- `14579d2` B3 compensating-tx audit PASS
- `da8f42c` B4 ACL hygiene audit PASS
- `10bdf5a` B5 anti-pattern scan PASS

### Phase C — Dedup
- `73c561e` HashSet<u32> usages distinct semantics, 0 redundancies

### Phase D — Deploy + Measure
- `2b4aa66` 267-cycle prod soak measurement

## NotebookLM peer-review final

**Verdict:** "Functional success in hardening process identity, but architectural warning regarding hot-path latency. The system has reached **Identity Honesty** but must reclaim performance budget to stay within the 130ms Hellerstein target."

### Severity-ranked findings

| Priority | Gap | Severity | Action |
|---|---|---|---|
| 1 | p95 latency investigation | 🔴 Critical | Move `proc_pidpath` identity checks behind `RecentlyApplied` cache to avoid redundant syscalls (per-action `csops` + `proc_pidpath` accumulating ~3µs × N) |
| 2 | sync_from_lockfree gap | 🟠 High | Update `daemon_state.rs` to flush `restore_status_*` atomic counters during periodic sync |
| 3 | SysctlOutOfRange (35% of fails) | 🟡 Medium | Align Governor output clamping with safety layer's allowed sysctl ranges |

### Confirmed analyses

- **Small-N noise**: minimum N for fair comparison is 500 events. Current 31-event window after fresh deploy is below the threshold. Sprint 1 saw same pattern: 90% at fresh deploy → 14.5% at large sample → 73.7% after extended soak.
- **SysctlOutOfRange = NEW gap class**: was statistically drowned out by PidRecycled noise (67 events) before. Now exposed. Indicates Governor's internal "safe ranges" model out of sync with `safety.rs` enforced limits.
- **B1 telemetry path**: lf_metrics → snapshot → MetricsState → JSON. Restore status counters are "sticky" one-shot startup metrics that must be included in `sync_from_lockfree` to persist into runtime_metrics.json.

## What Sprint 2 delivered

1. **PidRecycled fully closed** — both audit blocks (67 → 0) and journal skips. Phase A1+A2 idempotency hardening at universal filter and post-drain chokepoints work as designed.
2. **`pid_identity_still_valid` helper** — single source of truth mirroring `execute_actions::verify_pid_identity` exactly (start_sec + start_usec + unconditional name match).
3. **RecentlyApplied persistence** — Inbox Pattern with FAIL-EMPTY restore (4-tier integrity check: file missing / parse error / clock delta / boot crossed).
4. **5 telemetry counters** — wired to RestoreStatus enum, ready for runtime_metrics.json once sync_from_lockfree is updated.
5. **5 audit confirmations** — B2/B3/B4/B5/C verified zero refactor needed for clean codebase.
6. **22 new tests** — 1885 → 1907.

## What Sprint 2 did NOT deliver

1. **journal_success_rate ≥85% target** — small-N (31 events) prevents fair measurement. Honest finding: target needs longer soak.
2. **p95 ≤80ms target** — regressed from 75.45 → 139 ms due to per-action ProcessIdentity::from_pid syscall accumulation. Critical issue exposed.
3. **restore_status_* in runtime_metrics.json** — counters set in lf_metrics but `sync_from_lockfree` doesn't include the new fields. Architectural gap.

## Sprint 2 frameworks evaluation

- **apollo-evolve**: 15 atomic commits, OPENS/CLOSES tracked. Stop rules respected (no divergence triggered).
- **apollo-nars**: belief revision identified PidRecycled gap class correctly. Now NARS belief: f=1.0, c=0.97 (proven idempotency works).
- **superpowers TDD**: 22 new tests (15+ TDD discipline). Helper extraction caught reviewer issue I-1 (start_usec missing).
- **autoresearch**: mechanical metrics revealed real p95 regression (+64ms). Without autoresearch discipline would have shipped silent regression.
- **subagent-driven-development**: 15 tasks, two-stage review (spec + quality), implementer concerns surfaced (Task 5 absorbed parallel-agent code, Task 7 hooks pre-modified files). All tasks completed.

## Next-session priorities

Per NotebookLM:
1. 🔴 **p95 latency investigation**: instrument A1/A2 filter stages, move identity checks behind RecentlyApplied cache to skip redundant syscalls when entry already valid in cache.
2. 🟠 **sync_from_lockfree fix**: 5-field flush update in daemon_state.rs.
3. 🟡 **SysctlOutOfRange handling**: Governor ↔ safety layer range alignment.

## Verdict

Sprint 2 achieved its **primary goal** (PidRecycled elimination) and **all audit-only verifications passed**. The price was a hot-path latency regression that NotebookLM correctly identified as architectural priority for next session.

Apollo is now **identity-honest**: actions for dead/recycled PIDs are dropped at filter time, not logged as block_reason at safety layer. The trade-off is per-action syscall cost which next session must amortize via cache.

**Sprint 2 closed. 21 + 15 = 36 commits total over original `7f2aae7`.**
