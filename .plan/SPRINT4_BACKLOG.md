# Sprint 4 — Backlog (deferred fases 6-9)

Closed at v0.6.1 (2026-05-08, commit `bca18e1` final). 5/9 fases shipped.
Remaining 4 deferred. **NOT obsolete** — valid architectural debt items
that pay off if Apollo development resumes.

This file is for future you. If you revisit Apollo in N months and don't
remember what these were, this file is your re-onboarding.

## Sprint 4 Fases — what shipped

| Fase | Commit | What | Why it mattered |
|---|---|---|---|
| 1 | `0af6fd1` | IdentityVerifier merge | 2 source-of-truth → 1 (`ProcessIdentity::matches`) |
| 2 | `c99aaff` | IdentityCacheManager | 4 lifecycle owners → 1 |
| 3 | `93041a0` | Sticky counter rename | `survival_mode_entry_count` + helper |
| 4 | `e594774` | SetSysctl seal | Type-system invariant — Bug 6 impossible |
| 5 | `bca18e1` | ActionAccumulator + 11 telemetry counters | 15 emit sites → 1 builder |

## Backlog — Fases 6-9

### Fase 6 — ProtectionValidator

**Files**: `decide_actions.rs:455-478`, `daemon_action_safety.rs:120-121`,
`execute_actions.rs:728`, `process_enrichment.rs`.

**Problem**: Protection invariant ("which processes are safe to throttle/freeze")
appears in 4 sites with implicit ordering ("Must run AFTER signal_digest...").

**Deepening**: `ProtectionValidator::can_modify(pid, action_kind, ctx) -> bool`
uniform predicate. Single source of truth.

**Trigger to re-engage**:
- Adding a new emit site and wanting the same protection rules to apply uniformly without copy-paste.
- A bug like "process X was protected in decide_actions but not blocked in execute_actions".

**Effort**: ~1.5-2h with 2-agent pattern.

**Deletion test**: would removing the validator concentrate complexity (yes — single predicate) or just shuffle it (no).

---

### Fase 7 — Pipeline + Safety merge

**Files**: `daemon_action_pipeline.rs` (263L), `daemon_action_safety.rs` (304L).

**Problem**: Both modules sit between decide_actions and execute_actions
with overlapping responsibilities and unclear ordering. Each is a
pass-through (`Vec<RootAction>` in, `Vec<RootAction>` out) with side
effects (lock acquisitions, behavioral decisions). Combined ~600 LoC.

**Deepening**: Single `ActionFilterPipeline` with phases (degradation →
cognitive_gates → heuristic_pass → protection_override). Each phase a
method, logging why each action was filtered (audit trail).

**Trigger to re-engage**:
- Debugging "why was this action filtered?" takes >1h because you don't
  know which of the two modules dropped it.
- A new filter requirement is needed and the choice between
  daemon_action_pipeline vs daemon_action_safety is unclear.

**Effort**: ~1.5h with 2-agent pattern.

**Deletion test**: combined module makes ordering explicit (yes — single
phase enum) without changing behavior.

---

### Fase 8 — SharedState god-lock split (continuation of v0.9.0 plan)

**Files**: `daemon_state.rs:200-291` (`MetricsState` struct + `SharedState`).

**Problem**: `MetricsState` holds 20+ heterogeneous fields under a single
`Mutex` — `thermal_state`, `throttle_level`, `fast_tick_until`,
`reactor_event_weight`, `reactor_status` (counters), plus the embedded
`RuntimeMetrics` (100+ counters). Reactor thread + dashboard reader +
llm_daemon all serialize on this single Mutex.

Sprint 4 inherited from earlier v0.9.0 cleanup pass that consolidated
20 → 10 sub-locks. The next pass splits MetricsState itself.

**Deepening**: Split MetricsState into:
- `ReactorState` (thermal_state, reactor_status)
- `PolicyDisplayState` (throttle_level, thermal_level_real, fast_tick_until)
- `InstrumentationMetrics` (RuntimeMetrics counters → lock-free snapshot
  built each cycle)

`MetricsState` becomes a read-only aggregate computed at sync points.

**Trigger to re-engage**:
- `p95_cycle_ms` sustained >200ms under normal load (not warmup or
  small-N). NotebookLM's prior verdict cited this as Critical for
  Hellerstein 130ms target.
- Profiler shows `state.metrics.lock_recover()` as the hot path
  bottleneck.
- Adding a new MetricsState consumer that competes with reactor thread
  for the lock.

**Effort**: ~3-4h. Touches concurrency. High risk — needs careful soak.

**Pre-existing plan**: `.plan/V090_SHARED_STATE.md` if it still exists,
otherwise this Fase rebuilds from scratch.

---

### Fase 9 — Seal all 8 PID-bearing variants (extending Fase 4 pattern)

**Files**: `types.rs::RootAction` enum + ~100 match-arm touch points
across the codebase.

**Problem**: Fase 4 sealed only `SetSysctl` via private `SetSysctlAction`
struct. The other 8 PID-bearing variants (`ThrottleProcess`,
`FreezeProcess`, `BoostProcess`, `UnfreezeProcess`, `SetMemorystatus`,
`SetThreadQoS`, `ToggleSpotlight`, `QuarantineDaemon`) still have `pub`
fields and can be struct-literal constructed — Bug 6 redux is
mechanically possible for any of them.

**Deepening**: Convert each variant to wrap a private action struct:
```rust
pub enum RootAction {
    ThrottleProcess(ThrottleProcessAction),  // private fields
    FreezeProcess(FreezeProcessAction),       // private fields
    ...
}
```
Each with typed factory methods. Mirrors the SetSysctlAction pattern
exactly.

**Trigger to re-engage**:
- A new bug surfaces in production where a struct-literal action
  bypassed validation (analogous to Bug 6 but for a non-Sysctl variant).
- Adding a new action variant and wanting the type-system invariant
  for it.
- Adversarial review concludes "Fase 4 pattern is incomplete".

**Effort**: ~3h. Mechanical (Rust compiler catches all match arms),
but blast wide. Same pattern as Fase 4, just 8x.

**Deletion test**: same as Fase 4 — sealed variant makes Bug 6 class
impossible, unsealed allows mechanical regression.

---

## Re-engagement criteria summary

Don't re-engage Sprint 4 unless ONE of these is true:
1. A real bug surfaces that one of these fases would have prevented.
2. A new feature is being added that touches one of the affected
   subsystems and the cleanup unblocks it.
3. Performance metric (p95) regresses past CLAUDE.md threshold under
   normal load.

Otherwise, **leave it alone**. The shipped 5 fases + 5 bug fixes
already pay back any near-term development value.

## Adversarial-pattern lessons captured (re-read before resuming)

- **NotebookLM is not a final gatekeeper** (CLAUDE.md top section).
  Validate sprint closure with prod metrics + adversarial diff re-read
  + N≥500 events.
- **Sticky-counter-as-state pattern** caused Bug 7. Don't use cumulative
  counters as live state flags. Use `*_entry_count` for counters,
  `is_*_active()` predicates for state.
- **Telemetry sync chain** is fragile. Adding a counter to
  `LockFreeMetrics` requires matching field in `MetricsSnapshot`,
  `RuntimeMetrics`, plus a flush line in
  `daemon_state.rs::sync_from_lockfree`. Sprint 3 hit this twice;
  Fase 5 reviewer caught it before commit. The pattern is mechanical —
  consider a macro to enforce it.
- **Scaffolding without wiring** (Fase 4 review): a factory without
  sealing is theatre. Either seal the variant or accept that future
  emit sites will bypass.
- **Two-agent adversarial review** (pragmatic + skeptic + reviewer)
  caught real defects each round. Time well spent.

## Final state at v0.6.1

- Daemon: PID at-time-of-tag corriendo, failures=0
- Tests: 1907 → 1967 (+60)
- Bugs closed today: 5
- Architectural fases: 5/9
- Backups: `~/backups/apollo-v0.6.1/` (3 binaries)
- Tag: `v0.6.1`
