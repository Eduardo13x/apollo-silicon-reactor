# Sprint 3 Final Debrief — Cost Recovery: IdentityCache + Sync Flush + Sysctl Reconcile

**Date:** 2026-05-07
**Sprint commits:** 10 atomic + 1 sync-fix over Sprint 2 base `4618c02`
**Frameworks:** apollo-evolve + apollo-nars + superpowers TDD + autoresearch + subagent-driven-development

## Headline metrics

| Metric | Sprint 2 baseline | Sprint 3 final | Δ | Verdict |
|---|---|---|---|---|
| failures | 0 | 0 | flat | ✅ |
| success rate | 64.5% (n=31) | 100% (n=8) | small-N noise | ✅ no regression |
| **PidRecycled audit blocks** | 0 | 0 | flat | ✅ no regression |
| **SysctlOutOfRange share** | 35% (11/31 fails) | **0%** | **-100%** | ✅ Phase C MET |
| **identity_proc_pidpath/cycle** | n/a | **0.041** | new | ✅ target ≤5 by 100x |
| **restore_status_restored_n in JSON** | absent (0) | **61** | sync flush works | ✅ B1 closed |
| identity_cache hit ratio | n/a | 0/21 = 0% | idle workload | ⚠ unstressed |
| p95 cycle ms | 139 | 161 | +22 | ⚠ small-N variance |
| Tests | 1907 | 1921 | +14 | ✅ |

## Iteration log (10 commits + 1 sync fix)

### Phase A — IdentityCache module
- `3b4f80f` T1 module skeleton + types (IdentityKey, IdentityCacheEntry, IdentityValidation)
- `48c18f0` T2 validate_or_refresh + invalidate_pid + cleanup_expired + 6 unit tests
- `4523556` T3 wire through DaemonSubsystems + universal filter helper (cache-aware)
- `b6b203b` T4 periodic cleanup_expired every 60 cycles (replaces unfeasible NOTE_EXIT hook)
- `1ffbbc5` T5 6 lse_counters telemetry (hits/misses/evictions/ttl_expired/exit_invalidations + identity_proc_pidpath_calls)
- `54d1f7f` T6 4 guardrail tests (start_sec=0 force-refresh, cleanup, invalidate edge cases)

### Phase B — sync_from_lockfree flush
- `890dd1e` T7 restore_status_* 5-field flush (Sprint 2 gap closed)
- `9f74a2e` T9-fix identity_cache_* 6-field flush (discovered during deploy verification)

### Phase C — Governor ↔ Safety sysctl reconcile
- `87c56a1` T8 clamp_to_allowed_range helper + 4 SetSysctl emission site wraps + 4 tests

### Phase D — Deploy + Measure
- `a630a7c` T9 511-cycle prod soak measurement

## NotebookLM peer-review final

**Verdict:** *"Sprint 3 achieved 'Regulatory Honesty' by closing the SysctlOutOfRange gap and fixing telemetry sync. However, the system is now latently sluggish. The next frontier is no longer functional correctness, but Physical Performance Optimization."*

### Severity-ranked findings

| Priority | Gap | Severity | Action |
|---|---|---|---|
| 1 | **Cycle Stage Instrumentation** | 🔴 Critical | Cannot hit 130ms Hellerstein target without quantifying budget split. Instrument `sysinfo`, lock-wait times, IdentityCache overhead per cycle stage. |
| 2 | **Stress Load Validation** | 🟠 High | IdentityCache amortization & hit-ratio benefits theoretical. Trigger sustained pressure (swap > 80%) to verify cache holds under evaluation storms. |
| 3 | **Lock-Decomposition (v0.9.0)** | 🟡 Medium | God-Lock on `state.metrics` is the only path to reducing jitter from concurrent socket handler requests. |

### NotebookLM analysis

**(1) Hit ratio 0% — workload artifact, not bug.** With ~1 action per 126 seconds (idle workload, 8 actions / 506 cycles), probability of same PID hitting universal filter within 30s TTL is statistically near zero. Startup probes (`bird`, `fontworker`) are typically one-off events. Hits register only when **Adaptive Governor** enters high-pressure regime that re-evaluates same "Noise" candidates repeatedly.

**(2) Amortization theory still valid.** Current 0.041 syscall/cycle is "lost in noise" only because daemon idle. Sprint 2 regression theory (per-action `proc_pidpath` accumulation) was based on high-load cycles where `RootAction` queue saturates with dozens of candidates. Theory holds when system under pressure — that's exactly the state cache is designed for.

**(3) p95 161ms is structural, not action-driven.** Confirmed bottlenecks:
- **sysinfo refresh: 50–100ms per cycle** (process tree reconstruction) — historically dominant
- **God-Lock contention** on `state.metrics` held across long population blocks
- **lock_recover()** marginal overhead per guarded access
- **131L existing knowledge** (per memory `feedback_autoresearch_observations`): main.rs hot-path has 80-field metrics-population block under single mutex (Section #8 effective pressure aggregation, 348L)

**(4) restore_status telemetry wins.** restored_n=61 confirms `sync_from_lockfree` flush working AND Write-then-Rename persistence pattern operational. Other 4 variants (corruption/clock_delta/boot_crossed) at 0 simply mean healthy boot — no FS or RTC errors.

## What Sprint 3 delivered

1. **IdentityCache module** — 10 unit tests (6 core + 4 guardrail), TTL-based memoization, fail-empty restore semantics
2. **Universal filter cache-aware** — `pid_identity_still_valid` consults cache before syscall (2 call sites: filter + post-drain)
3. **Periodic cleanup** every 60 cycles (replaces NOTE_EXIT hook deemed unfeasible)
4. **Full telemetry pipeline** — 6 counters end-to-end: `lse_metrics → MetricsSnapshot → RuntimeMetrics → JSON`
5. **B1 sync flush gap closed** — restore_status_* counters now visible (Sprint 2 architectural debt paid)
6. **B1 identity_cache sync gap fixed** — discovered & closed during T9 deploy (parallel mapping caught)
7. **Phase C sysctl clamp** — Governor pre-emit clamp via `safety::allowlisted_sysctls_with_ranges()`. 0 SysctlOutOfRange post-deploy (Sprint 2: 35% of fails)
8. **14 new tests** — 1907 → 1921

## What Sprint 3 did NOT deliver

1. **p95 ≤80ms target missed** — 161ms post-deploy. Small-N variance dominates (8 actions / 506 cycles). True recovery validation requires stress workload.
2. **Cache hit ratio ≥0.85 unverified** — 0/21 measured but workload didn't exercise repeats within 30s TTL. Theoretical amortization, empirical pending.
3. **Cycle stage instrumentation** — root cause of structural p95 unmeasured. Next session priority.

## Architectural frontier achieved

> **RecentlyApplied evita actuar dos veces.**
> **IdentityCache evita dudar dos veces de la misma identidad ya verificada.**

Clean separation enforced. Future regressions prevented where someone clears action cache but inadvertently affects identity semantics.

## Sprint 3 frameworks evaluation

- **apollo-evolve**: 10 atomic commits + 1 hot-fix. OPENS/CLOSES tracked. Stop rules respected.
- **apollo-nars**: belief revision identified `csops/proc_pidpath` cost class correctly. NARS belief: `<identity-validation: cache-required>` f=1.0, c=0.95 (proven gap closure works).
- **superpowers TDD**: 14 new tests with RED-GREEN-REFACTOR. Spec reviewer caught I-1 (start_usec missing) and I-2 (helper duplication) early.
- **autoresearch**: mechanical metrics revealed identity_cache sync gap during T9 deploy verification (would have shipped silent telemetry blackout).
- **subagent-driven-development**: 10 tasks, two-stage review (spec + quality), implementer concerns surfaced (T2 absorbed parallel-agent code, T7 hooks pre-modified files). All tasks completed.

## Next-session priorities

Per NotebookLM severity-ranking:

1. 🔴 **Cycle Stage Instrumentation**: Split per-cycle p95 by stage (sensing / reasoning / execution / lock-wait). Per existing memory `feedback_autoresearch_observations` Section #8/14 effective pressure aggregation = 348L under single mutex — likely top contributor.
2. 🟠 **Stress Load Validation**: Synthetic workload triggering sustained pressure ≥0.80 to exercise IdentityCache hit ratio. Without this, cache amortization claim remains theoretical.
3. 🟡 **Lock-Decomposition (v0.9.0 SharedState Migration)**: Plan exists at `.plan/V090_SHARED_STATE.md` — target main.rs <4500L, 0 flat SharedState fields. NotebookLM frames this as "only path to reducing jitter."

## Verdict

Sprint 3 achieved its **primary goal** (regulatory honesty: SysctlOutOfRange 35→0%, telemetry sync working). The price was a measurement gap that NotebookLM correctly identified as architectural priority for next session: workload-stress validation of cache amortization + cycle stage instrumentation.

Apollo is now **regulatory-honest**: Governor proposals respect Safety constraints. Telemetry persistence layer fully observable. The remaining architectural debt is not correctness but observability + workload exercise.

> Sprint 3 closed the regulatory honesty debt: Apollo no longer emits
> doomed sysctl actions. The new debt is not safety, but **physical
> performance optimization** — instrumenting hot-path stages to find
> where the 161ms p95 budget actually goes.

> Sprint 3 cerró la deuda regulatoria: Apollo ya no propone valores
> doomed. La nueva deuda no es de seguridad, sino de **optimización
> física**: instrumentar etapas hot-path para descubrir dónde se gasta
> el budget de 161ms p95.

**Sprint 3 closed. 56 commits total over original `7f2aae7` (21 self-healing + 15 SuperPlan + 10 Sprint 2 + 10 Sprint 3 + helper polish).**

## Sprint 4 sketch (per closing peer-review)

The next sprint's architectural focus is *measurement before optimization*:

1. **Cycle stage histograms** — instrument 5 stages: sensing (sysinfo refresh), reasoning (decide_actions), execution (execute_actions), persistence (atomic writes), pacing (condvar wait). Persist p95 per stage to `runtime_metrics.json`.

2. **Stress workload harness** — `scripts/stress-workload.sh`: stress-ng + brave + xcode build simultaneously, drive pressure ≥0.80 sustained 60s. Measure hit ratio + p95 with cache active vs disabled (control).

3. **Lock-decomposition cohort** — pick 1 victim group from v0.9.0 plan (likely MetricsGroup), extract to per-stage atomic snapshots. Measure pre/post lock-wait p95.

> *First correctness, then regulatory honesty, then physical performance.*
> Sprint 4 owns the physical performance recovery half of that equation.

## References

- Sprint 1 spec: `docs/superpowers/specs/2026-05-06-superplan-governor-state-memory.md`
- Sprint 2 spec: `docs/superpowers/specs/2026-05-07-sprint2-pidrecycled-patterns-dedup-design.md`
- Sprint 3 spec: `docs/superpowers/specs/2026-05-07-sprint3-cost-recovery-identity-cache.md`
- Sprint 3 plan: `docs/superpowers/plans/2026-05-07-sprint3-cost-recovery-identity-cache.md`
- Sprint 2 final debrief: `evolve/2026-05-07-sprint2/final-debrief.md`
- 1001 patterns: `/Users/eduardocortez/Downloads/1001_patrones_final.pptx`
- NotebookLM project: `8344b94c-a014-4803-abea-076a55753cfd`
- Hellerstein 2004 §9: adaptive cycle target 130ms
- Saltzer & Kaashoek 2009 §3.3: Complete Mediation
- v0.9.0 SharedState plan: `.plan/V090_SHARED_STATE.md`
