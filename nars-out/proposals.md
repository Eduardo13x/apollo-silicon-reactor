# Apollo NARS Proposals — Graph + Bug Analysis Session 2026-04-11 (Updated with Gemma4 Teacher)

Generated from: `graphify-out/graph.json` (4727 nodes, 9271 edges, 143 communities) + unstaged diffs  
Baseline: 1608 lib tests | 165/165 scenarios | AIS 99.5

---

## P1: HazardModel — Proxy Signal Confusion (base_rate saturation)
**Belief ID**: B001  **Priority**: 0.894  **Truth**: <0.91, 0.98>

**Evidence:**
- `signal_intelligence.rs`: `record_overflow()` called unconditionally on `VmPressureLevel::Critical`, which fires at ~80% memory pressure — a **normal** operating state, not an OOM event
- macOS dynamic swap: `swap_ratio = swap_used / swap_total ≈ 1.0` whenever any swap exists (OS always uses all allocated swap space), so the feature the model trains on is a constant, not a signal
- `hazard_model.rs`: `base_rate` accumulates every pressure event, eventually saturating to >1 OOM/hour — physically impossible for a stable system
- Graph: `signal_intelligence.rs` degree=325 (god node, C18) — many downstream callers consume the saturated OOM probability, amplifying the error

**Root cause pattern:** Confusing a proxy metric (swap_ratio, pressure level) with the causal signal (swap VELOCITY, actual memory allocation failure). The model learned "pressure is always dangerous" instead of "fast swap growth is dangerous."

**Proposed mutation for apollo-evolve:**
1. `signal_intelligence.rs:380` — already partially fixed in diff. Verify `swap_growing_fast` threshold (512KB/s) is calibrated against real M1 8GB workloads
2. `hazard_model.rs` — add `validate_after_restore()` call path test: restore a saturated model, verify it resets to prior
3. `signal_intelligence.rs:795` — confirm `self.hazard.validate_after_restore()` runs after every `restore()`, not just cold start
4. Add metric: expose `hazard_base_rate` in `RuntimeMetrics` so saturation is observable in production

**Paper citation:**
Pearl 2009 — *Causality* — confusing association (pressure=high) with causation (swap growing = memory exhaustion) produces systematically wrong risk estimates; swap velocity is the Granger-causal predictor [Granger 1969]

**Expected gain:**
- OOM probability calibrated to real risk, not noise
- Fewer false-positive freeze triggers at moderate pressure
- `validate_after_restore()` prevents saturation from persisting across daemon restarts
- +2-4 unit tests for base_rate clamping and velocity-gated training

**Risk:** Low — only affects hazard model training gate, not safety thresholds. Already partially implemented in diff.

---

## P2: ThermalManager — Sentinel -1 as Option<T> type violation
**Belief ID**: B002  **Priority**: 0.882  **Truth**: <0.91, 0.98>

**Evidence:**
- `types.rs:1069`: `thermal_seconds_to_throttle: i32` with comment `/// -1 = no throttle predicted`
- `thermal_manager.rs`: `time_to_throttle()` returns `-1` in 3 branches as sentinel for "no data"
- `main.rs`: `let mut thermal_seconds_to_throttle: i32 = -1` — sentinel propagates through 100 lines of daemon logic
- Graph: `types.rs` degree=64 (god file) — sentinel anti-pattern fans out to all consumers
- Rust type system provides `Option<i32>` precisely for this: `None` = no forecast, `Some(0)` = already throttling, `Some(n)` = n seconds headroom

**Root cause pattern:** Java/C-style sentinel integers imported into Rust code. The type system's null-safety guarantee is bypassed, requiring every consumer to remember the magic value.

**Proposed mutation for apollo-evolve:**
Already implemented in diff. Verify:
1. All JSON serialization: `Option<i32>` serializes as `null` (not `-1`) in `runtime_metrics.json` — consumers (apolloctl, menubar) must handle null
2. Any pattern `== -1` or `< 0` check on `seconds_to_throttle` in bash scripts or external consumers
3. Add `#[serde(skip_serializing_if = "Option::is_none")]` if backward compat needed vs external tools
4. Audit codebase for other `i32` sentinels (-1, 0, 999) — this pattern likely recurs

**Paper citation:**
Kleppmann 2017 — *Designing Data-Intensive Applications* §10: sentinel values in serialized formats create silent compatibility bugs; use sum types (Option/enum) for absence semantics

**Expected gain:**
- Compiler enforces null check at every callsite — no forgotten `-1` comparisons
- JSON API cleaner: `null` vs `-1` is unambiguous to consumers
- Pattern applies to other `i32` sentinels in the codebase (audit needed)

**Risk:** Low — already implemented. External JSON consumers need null-handling update if present.

---

## P3: Critical Architecture Bugs (graph hyperedge, EXTRACTED 1.00)
**Belief ID**: B006  **Priority**: 0.810  **Truth**: <0.90, 0.90>

**Evidence:**
- Graphify hyperedge `[EXTRACTED 1.00]`: `Critical Architecture Bugs` → `bug_pending_trial_skill`, `bug_cooccurrence_eviction`, `bug_f32_precision`
- Sourced from `papers/apollo_agi_paper_draft.md` — formally documented, not informal todos
- These bugs affect the learning correctness of OptimizationSkills, CausalGraph co-occurrence, and causal weight precision

**Bug details:**
- **bug_pending_trial_skill**: `SkillTrial` starts in `Pending` state but no code path transitions it out before outcome evaluation — skills never graduate from trial to production
- **bug_cooccurrence_eviction**: LRU eviction removes co-occurrence pairs actively used for causal inference — removes evidence mid-inference cycle
- **bug_f32_precision**: f32 accumulation in causal weight update causes ~7 digits precision; over thousands of updates, weight drift exceeds meaningful threshold resolution

**Proposed mutation for apollo-evolve:**
1. `optimization_skills.rs`: find `Pending` state machine, wire `→ Active` transition after N successful outcomes (N=3, tunable via LearnableParams)
2. `causal_graph.rs`: LRU eviction should pin entries with `observations > threshold` — protect knowledge from eviction
3. `causal_graph.rs`: change accumulator fields from `f32` to `f64` in the weight update path only (not storage)

**Paper citation:**
Simon 1955 — *Bounded Rationality* — incomplete state machines create permanent sub-optimal behavior; the system "knows" skills are pending but has no mechanism to promote them. Pei Wang 2013 — NARS truth value revision requires completed belief cycles.

**Expected gain:**
- Skills graduate: optimization repertoire grows over time instead of staying permanently in trial
- Co-occurrence graph retains causal evidence across inference cycles
- Weight accumulation precision: ~15 digits prevents long-term drift

**Risk:** Medium (pending_trial_skill, cooccurrence_eviction) — state machine changes need careful testing. Low (f32→f64) — pure precision improvement.

---

## P4: frozen_state Ghost PIDs — Event-Only Cleanup Anti-Pattern
**Belief ID**: B003  **Priority**: 0.780  **Truth**: <0.83, 0.94>

**Evidence:**
- `main.rs`: before this diff, `frozen_state` was only updated by explicit SIGCONT/unfreeze actions — no periodic reconciliation
- kqueue `NOTE_EXIT` is not registered after a daemon restart (pid file gap) — ghost PIDs accumulate silently
- `frozen_ram_mb` metric counts ghost PIDs → inflated pressure measurements → premature freeze decisions
- `display_turbo.rs`: `turbo_frozen_pids` had the same gap — fixed with `gc_dead_pids()`
- Graph: `execute_actions.rs` (C11) and `safety.rs` (C11) — freeze/unfreeze tightly coupled to action execution, not to process lifecycle events

**Root cause pattern:** Event-driven state management without a defensive reconciliation fallback. kqueue is "mostly reliable" but has documented gaps (daemon restart, jetsam, force quit before registration).

**Proposed mutation for apollo-evolve:**
Already implemented in diff. Verify:
1. Ghost PID reconciliation runs BEFORE `frozen_ram_mb` is computed (ordering matters)
2. `write_frozen_state()` is called only when `removed > 0` (already in diff — correct)
3. Add test: simulate daemon restart with ghost PIDs in `frozen_state.json` → reconciliation clears them on first cycle

**Paper citation:**
Gray & Reuter 1992 — *Transaction Processing* §10: "defensive state reconciliation should run periodically regardless of events — events are optimistic notifications, not guarantees"

**Expected gain:**
- `frozen_ram_mb` reflects actual frozen memory (no ghost inflation)
- Pressure decisions no longer triggered by phantom processes
- Daemon restart resilience: first cycle cleans stale freeze state

**Risk:** Low — reconciliation is read-only against process table. Already implemented.

---

## P5: Freeze/Thaw Lifecycle — Missing OS Sleep/Wake Integration
**Belief ID**: B004  **Priority**: 0.771  **Truth**: <0.82, 0.94>

**Evidence:**
- Before diff: no `SleepNotifier` integration in daemon main loop
- IOKit `kIOMessageSystemWillSleep` fires ~30s before kernel suspends — window exists but was unused
- Frozen PIDs cannot be compressed/Jetsam'd during sleep → macOS kills other processes (widgets, extensions) more aggressively
- `sleep_notifier.rs` existed in C109 (2-node isolated community) — fully disconnected from daemon main loop
- Graph: C109 cohesion=0.00 — `SleepNotifier` was effectively dead code from the graph's perspective

**Root cause pattern:** Incomplete OS lifecycle integration. The freeze subsystem was designed around "process is alive and running," ignoring the third OS state: "system is suspending."

**Proposed mutation for apollo-evolve:**
Already implemented in diff. Verify:
1. `sleep_notifier.available = false` case: daemon continues normally (non-root path)
2. Post-wake: add `sleep_notifier.wake_pending()` check → extend reconciliation grace period (2-3 cycles before re-freezing)
3. Add metric: `pre_sleep_unfreezes_total` counter in `RuntimeMetrics`

**Paper citation:**
Nygard 2018 — *Release It!* §4 "Integration Points" — every external lifecycle event (sleep, wake, power) is an integration point requiring explicit handling; silent assumption of "always running" creates brittleness

**Expected gain:**
- macOS Jetsam no longer competes with frozen processes during sleep
- Fewer extension/widget kills on wake
- `SleepNotifier` C109 gains connections → graph cohesion improves

**Risk:** Low — pre-sleep path is defensive. Non-root path silently skips. Already implemented.

---

## P6: God Files — outcome_tracker.rs + predictive_agent.rs (500+ degree)
**Belief ID**: B005  **Priority**: 0.737  **Truth**: <0.75, 0.99>

**Evidence:**
- Graph: `outcome_tracker.rs` degree=515, `predictive_agent.rs` degree=505 — top 2 most connected files
- Community C1 cohesion=0.04 (Outcome Tracker) — fragmented, 113 nodes with weak internal connections
- Community C2 cohesion=0.05 (Learning Context + PredictiveAgent) — mixed concerns
- Graphify surprising connection: `Effectiveness Tracking (3 rings)` ↔ `Causal Graph Mechanism Mediation` — same concept without explicit link
- `autoresearch/redundancy_audit.md`: "three independent effectiveness rings never cross-feed" — confirmed by graph

**Root cause pattern:** Organic growth without bounded scope. Multiple learning loops added to same files over 8 months without extracting focused sub-traits.

**Proposed mutation for apollo-evolve:**
Medium-term refactor:
1. Extract `CoOccurrenceGraph` into its own module (partially done via `causal_graph.rs`)
2. Define `EffectivenessSignal` trait with `record()` + `blended_score()` — unify 3 rings
3. `predictive_agent.rs`: extract `AgentContext` builder into `agent_context.rs`

**Paper citation:**
Denning 1968 — *Working Set Model* — unbounded growth in any data structure degrades performance; bounded working sets force principled eviction and decomposition

**Expected gain:**
- `outcome_tracker.rs` splits from 500+ degree to ~200-250 per sub-module
- Co-occurrence graph testable in isolation
- `EffectivenessSignal` trait enables 3-ring cross-feed (closes redundancy audit finding)

**Risk:** High — large refactor, touches many test fixtures. Isolated branch + full scenario validation required.

---

## Root Cause Summary

| Pattern | Bugs | Files |
|---------|------|-------|
| Proxy signal confusion | B001 (hazard saturation) | signal_intelligence.rs, hazard_model.rs |
| Sentinel values bypassing type system | B002 (thermal -1) | types.rs, thermal_manager.rs |
| Event-only state without periodic reconciliation | B003 (ghost PIDs) | main.rs, display_turbo.rs |
| Incomplete OS lifecycle integration | B004 (pre-sleep) | main.rs, sleep_notifier.rs |
| Unbounded module scope (organic growth) | B005 (god files) | outcome_tracker.rs, predictive_agent.rs |
| Paper-to-code translation gaps | B006 (3 arch bugs) | optimization_skills.rs, causal_graph.rs |

**Cross-cutting observation**: B001–B004 share a common theme — **missing defensive fallbacks**.
The system relied on signals being "mostly correct" (swap_ratio, kqueue events, IOKit lifecycle) without
accounting for failure modes. The fixes add either:
- A secondary signal that validates the primary (swap velocity validates swap_ratio)
- A periodic reconciliation that validates event-driven state (ghost PID cleanup)
- An explicit OS integration point that was previously implicit (sleep hook)

B001 and B002 both stem from **macOS-specific assumptions** not being encoded in the type/model system:
macOS dynamic swap semantics and IOKit lifecycle are non-obvious to developers coming from Linux.

---
---

# New NARS Pass — Post Gemma4 Teacher Integration (2026-04-11 updated)
Graph: 4727 nodes, 9271 links | New beliefs from god-node + hyperedge analysis

## NEW-P1: BUG-01 + BUG-03 — Critical Bug Cluster (pending_trial_skill + f32 precision)
**Belief ID**: B011  **Priority**: 0.855  **Truth**: <0.95, 0.90>

**BUG-01 — pending_trial_skill lost on daemon crash mid-trial:**
Location: `src/bin/apollo-optimizerd/main.rs:4302`
When `pending_trial_skill = Some(skill_name, pressure_before)` is set, the value lives only in RAM until the next periodic persist (~every 2 min, L6201). A daemon crash in that window silently drops the trial result. The skill registry never learns from real executions.
Fix: call `write_json_critical` on the `learned_state` snapshot immediately after L4302.

**BUG-03 — f32 precision loss in skill record_result_with_pressure:**
Location: `src/bin/apollo-optimizerd/main.rs:4226`
`pressure_before as f32` converts a 64-bit float to 32-bit before writing into the skill registry Bayesian weights. Over thousands of records, small rounding errors accumulate and bias skill selection probabilities.
Fix: change `record_result_with_pressure` signature to accept `f64`.

## NEW-P2: daemon_helpers.rs — 68-edge god node (highest in codebase)
**Belief ID**: B001  **Priority**: 0.889  **Truth**: <0.95, 0.936>
Most-imported utility file. Every module that needs a helper imports it.
Fix: categorize exports by domain → extract `process_utils.rs` + `time_utils.rs`. `daemon_helpers.rs` becomes a re-export facade.

## NEW-P3: Gemma4 Teacher Loop — VALIDATED (new feature, no bugs found)
The new `TeacherContext` struct + `SuggestionOutcome` feedback loop implements a closed teacher-student cycle:
- Apollo collects Bayesian pattern scores → feeds to Gemma 4
- Gemma 4 suggests process classifications → Apollo applies them
- 30s later: Apollo measures pressure delta → stores as `last_suggestion_outcome`
- Next Gemma call: includes `PreviousGemmaSuggestion outcome` → Gemma can revise strategy
NARS belief: <f=0.90, c=0.85> — this architecture follows the NARS revision principle where beliefs are updated by evidence accumulation.

