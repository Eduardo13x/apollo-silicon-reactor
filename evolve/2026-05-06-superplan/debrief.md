# SuperPlan Debrief — 2026-05-06

**Frameworks combined:** apollo-evolve + apollo-nars + superpowers TDD + autoresearch

## Goal (NARS belief revision)

NotebookLM Critical residual gap from prior debrief:
> "Governor padece **Falta de Memoria de Estado** — 87.5% journal `success: false`"

NARS revision over 4 converging signals → priority 0.847 → top of queue.

## Iterations executed

| Iter | Commit | Mutation | Tests | Status |
|------|--------|----------|-------|--------|
| 1 | `0cf386a` | RecentlyApplied module (RED+GREEN, 220 LoC) | 10 unit | KEEP |
| 2+3 | `c4acfd6` | Wire-in to governor heuristic path + lifecycle | 4 wire-in | KEEP |
| 4 | n/a | Deploy + 250-cycle measurement | runtime | partial |

## Empirical results

| Metric | Baseline | SuperPlan | Target | Verdict |
|--------|----------|-----------|--------|---------|
| ThrottleProcess cross-cycle dups | n/a | **0** | 0 | ✅ MET |
| PressureContext catch-all share | 78% | **34.5%** | ≤45% | ✅ MET |
| journal success rate | 12.5% | 15.3% | ≥35% | ❌ partial |
| SetMemorystatus dups distinct | 0 | 6 | 0 | ❌ regression |
| failures | 0 | 0 | 0 | ✅ |
| Tests | 1885 | 1895 | +N | ✅ +10 |
| Self-diagnosis false-positives | 0 | 0 | 0 | ✅ |

## Why partial success on journal rate

RecentlyApplied wired ONLY to `convert_and_merge_heuristic_decisions` (governor
heuristic pass). Journal entries come from FIVE emission paths:

| Path | Wired? | Dominant action kind |
|------|--------|----------------------|
| Governor heuristic (Night mode etc.) | ✅ | Throttle |
| decide_actions direct (multi-thread + heuristic) | ❌ | Throttle/Freeze |
| daemon_paging_hints (pressure + ODE) | ❌ | SetMemorystatus |
| main.rs deep-scan (compressor-aware) | ❌ | SetMemorystatus |
| llm_daemon (Gemma suggestions) | ❌ | Throttle/Freeze |

Governor path now suppresses cross-cycle dups perfectly (ThrottleProcess dups = 0).
Other paths still emit redundant decisions = 85% remaining `success: false` rate.

## Closed gaps

- 🔴 **NotebookLM Critical: PressureContext catch-all** — closed.
  Foundation Phase 5 + SuperPlan combined: 78% → 34.5% (-43.5pp).
  MLWorkload + MemoryBudget + SwarmThrottling + CompositorPriority all firing.
- 🟠 **Cross-cycle ThrottleProcess dups** — fully eliminated for governor path.
- ⚪ **Test discipline (superpowers TDD)** — 14 new tests, all green.

## Open work for next session

| Priority | Work | LoC est. |
|----------|------|----------|
| 🟠 High | Extend RecentlyApplied to daemon_paging_hints | ~40 |
| 🟠 High | Extend to main.rs deep-scan SetMemorystatus | ~30 |
| 🟡 Medium | Extend to decide_actions direct emissions | ~80 |
| 🟡 Medium | Extend to llm_daemon emissions | ~40 |
| ⚪ Low | NARS bridge for self_diagnosis.jsonl | ~150 |
| ⚪ Low | Investigate 6 SetMemorystatus dups (root cause) | research |

## Process learnings

1. **NARS belief revision** correctly identified single highest-priority work
   when 4 weaker signals converged (priority 0.847 vs. ~0.6 individually).
2. **superpowers TDD discipline** caught cache key bug (Kill→Freeze mapping)
   before deploy via the `convert_and_merge_kill_caches_as_freeze` test.
3. **autoresearch metric** revealed partial success — without measuring journal
   success rate post-deploy, would have claimed full closure.
4. **apollo-evolve commit-per-phase** kept blast radius bounded; if Iter 4
   measurement showed regression, Iter 1+2 are revertable independently.
5. **Five emission paths uncovered** — design originally assumed governor was
   the dominant emitter; runtime data showed other paths matter equally.

## Verdict

SuperPlan delivered:
- **Catch-all problem closed** (PressureContext 78%→34.5%)
- **Cross-cycle Throttle dups eliminated** (governor path)
- **Cache infrastructure ready** for extension to remaining 4 paths

Did NOT deliver journal success rate ≥35% — that requires multi-path wiring
(estimated 4 more iterations / ~190 LoC).

The cache architecture is the right hammer; need to swing it on more nails.
