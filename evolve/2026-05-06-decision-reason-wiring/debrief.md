# Apollo-Evolve Debrief — 2026-05-06

## Loop summary

5 iterations completed. Goal: close DecisionReason 0%-adoption gap on
MemoryBudget/CriticalBypass/HysteresisRecovery variants.

| Iter | Commit | Mutation | Status |
|------|--------|----------|--------|
| 0 (foundation) | e55c0bd | lock-free metrics + memory budget hysteresis + reactor weights halved + sysctl P/E cores | KEEP |
| 1 | ba84257 | wire MemoryBudget at 3 paging-hint sites | KEEP |
| 2 | e220476 | wire CriticalBypass when pressure ≥ 0.80 | KEEP |
| 3 | 2da70e4 | wire HysteresisRecovery in 30s post-Critical window | KEEP |
| 4 | 018cc35 | recovery window invariant test | KEEP |
| 5 (deploy) | prod | release build + restart + measure adoption | KEEP |

## Empirical metric

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| PressureContext share | 97% | 62.5% | -34.5pp |
| MemoryBudget share | 0% | 20.5% | +20.5pp |
| Distinct variants | 4 | 4 | 0 (CriticalBypass + HysteresisRecovery require pressure ≥ 0.80; not crossed) |
| p95 cycle ms | 121 | 86.97 | -28% |
| Failures | 0 | 0 | flat |

## NotebookLM peer review (severity-ranked)

**Critical**:
- **Global Action Deduplicator** — PID reapply spam (SetMemorystatus 4× same cycle)
  is back. Prior fix `0a84c11` (2026-04-15) addressed it once. Multiple modules
  (ChromiumManager, MemoryBudget, PagingHints) emit Jetsam intentions without
  final consolidation. Need filter at end of dispatch_tick: ≤1 memory action +
  ≤1 QoS action per PID per cycle.

**High**:
- **Thread-Level Scheduling Refinement** — Phase 1 ARM64 thread-level scheduling
  is fully implemented. With new P/E core sysctl counts wired (foundation commit),
  can route hot threads to P-cores within helper processes that today get
  blanket-throttled (Brave, Chrome helpers).
- **sysinfo Cache & Cadence** — process refresh dominates cycle cost.
  *Status note*: Foundation commit ALREADY added staggered refresh (Normal=8 cycles,
  Elevated=4, Critical=1). NotebookLM corpus pre-dates this change; verify
  effectiveness in production.

**Medium**:
- **Reactor weight halved consequences** — under regime shift, may take 50% more
  cycles to reach defensive saturation. Monitor AIS D5 in next swap storm.

**Low**:
- HysteresisRecovery + CriticalBypass 0% — correct, workload-dependent.
- MLWorkload spike 4→31 — expected from teach apply (eligibilityd routing).
- PressureContext catch-all 62.5% — next candidates: SwarmThrottling,
  GraduatedIdle, ThreadQoSRouting variants.
- Test coverage end-to-end policy_audit.jsonl assertion missing.

## AIS state

S-tier 90.27 in production (post-deploy).

## Top 3 priorities next session (per NotebookLM)

1. **[Critical]** Global Action Deduplicator (per-PID consolidation in dispatch_tick)
2. **[High]** Thread-Level Scheduling refinement using new sysctl P/E core counts
3. **[High]** sysinfo cache verification (foundation commit may have already
   addressed this — needs measurement)

## Outstanding observations from session

- `apollo_optimizer-*` ghost binaries showing in noise candidates
  (apollo_optimizer-89e6a224badf89d0, apollo_optimizer-009114652f920025,
   apollo_optimizer-255918a2d3abb5c4) — likely orphan cargo test artifacts that
  auto-promoted to noise-tracked. Not addressed this session, defer.

- DecisionReason adoption could push higher with simulated stress test:
  `pressure ≥ 0.80` workload would exercise CriticalBypass + HysteresisRecovery
  paths that natural 0.62 traffic doesn't trigger.

- Foundation commit polluted with untracked plan files
  (`docs/superpowers/plans/2026-05-03-*.md`) and `.plist.tmp`. Cosmetic, not
  worth amend per skill rules. Add `.plist.tmp` to .gitignore next session.
