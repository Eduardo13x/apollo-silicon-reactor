# NARS-Guided Improvement Proposals
# Generated: 2026-04-10 (post-autoresearch session)
# Graph: 4513 nodes · 8988 edges · 121 communities
# NARS beliefs: 25 · Top 7 proposals below (priority > 0.75)

---

## P1: prepare.rs + decide() — Over-coupled God Nodes [B002, B008, priority=0.842]

**Belief**: `prepare.rs` and `decide()` are the most-connected nodes — 67 and 55 edges respectively. Delegation is incomplete.
**Truth**: <f=0.85, c=0.99>

**Evidence:**
- Graph: `prepare.rs` module (67 edges) — central dispatch point for decision pipeline
- Graph: `decide()` function (55 INFERRED edges) — called by nearly every test module
- Session: 185 LOC eliminated from `decide_actions.rs` via `call_decide()` helper — pattern validated
- Same delegation pattern not yet applied to `prepare.rs` helpers

**Proposed mutation for apollo-evolve:**
Extract `fn make_snap(pid, name, rss, cpu, ...)` and `fn make_hunt(...)` builder helpers in `tests/` files that call `prepare::*` — consolidate into a shared `tests/helpers/prepare_helpers.rs` or inline into `call_decide`-style builder. Reduces test boilerplate that currently duplicates 8-12 fields per test site.

**Paper citation:** [McIlroy 1969] "Mass Produced Software Components" — composable small helpers outperform monolithic parameter lists

**Expected gain:** -40 to -80 LOC in test files, reduced coupling to `prepare.rs` internals

**Risk:** Low — test-only refactor, behavior unchanged

---

## P2: predictive_agent.rs — Decompose or Delegate [B003, priority=0.838]

**Belief**: `predictive_agent.rs` (67 edges) is as connected as `prepare.rs` — evidence of God Object.
**Truth**: <f=0.85, c=0.986>

**Evidence:**
- Graph: 67 edges from predictive_agent — cross-cutting dependency from learning, signal, outcome, NARS modules
- NARS belief: prior session noted "three independent learning loops never cross-feed" — partially fixed in v0.6.0 cables A/B/C
- community C20 (Decision Preparation) has cohesion 0.04 — fragmented

**Proposed mutation for apollo-evolve:**
Verify `PredictiveAgent::predict()` is only called via `LearningContext` (Cable B wired in v0.6.0). If direct callers in daemon main loop exist outside learning_tick, introduce a trait `PredictorTrait` and reduce `predictive_agent.rs` to its core Markov chain — extracting QoS wiring to `mach_qos.rs` (Cable C).

**Paper citation:** [Parnas 1972] "On the Criteria To Be Used in Decomposing Systems into Modules" — decompose by information hiding, not by convenience

**Expected gain:** Reduced coupling; predictive_agent.rs from 67 to ≤40 edges; Cable C consolidation

**Risk:** Medium — touches hot path in learning_tick

---

## P3: chromium_manager.rs — Continued Inline Refactor [B006, priority=0.840]

**Belief**: `chromium_manager.rs` (64 edges) — 6th most-connected node, autoresearch stopped at iteration 10.
**Truth**: <f=0.85, c=0.99>

**Evidence:**
- Graph: 64 edges — high coupling density for a focused module
- Autoresearch: iterations 6-9 removed 150 LOC (2125→1975), stopped at 10 iterations
- LOC target not yet reached: 50-100 more LOC available via test macro inlining

**Proposed mutation for apollo-evolve:**
Continue iteration 11+ on chromium_manager.rs. Target: multi-assert blocks in `test_thaw_logic`, `test_arousal_*`, `test_nars_*`. Apply same inline pattern used in iterations 6-9. Verify 3656 tests still pass.

**Paper citation:** [Saltzer 1975] "The Protection of Information in Computer Systems" §economy of mechanism — minimal surface area reduces defect probability

**Expected gain:** -30 to -50 LOC, reduces chromium_manager.rs below 1950 lines

**Risk:** Low — mechanical inline, zero behavioral change

---

## P4: Critical Architecture Bugs — Fix BUG-01/02/03 [B024, priority=0.765]

**Belief**: Hyperedge "Critical Architecture Bugs" is EXTRACTED (confidence=1.0) — three confirmed bugs await fixes.
**Truth**: <f=0.90, c=0.85>

**Evidence:**
- Hyperedge nodes: `bug_pending_trial_skill`, `bug_cooccurrence_eviction`, `bug_f32_precision`
- BUG-01: `pending_trial_skill` in `OutcomeTracker` not persisted → lost on daemon restart → learning regression
- BUG-02: `co_occurrence` graph under-eviction → unbounded growth → memory pressure over time
- BUG-03: f32 precision loss in pressure thresholds → flapping decisions near boundary values

**Proposed mutations for apollo-evolve (3 separate commits):**

1. **BUG-01**: Add `pending_trial_skill: Option<String>` to `LearnedState` → serialize/deserialize via `collect()`/`apply()`. Test: restart daemon mid-trial, verify skill resumes.

2. **BUG-02**: In `OutcomeTracker::self_improve()`, add co_occurrence eviction: cap at 1000 entries, prune by lowest weight. Test: inject 2000 pairs, verify cap holds.

3. **BUG-03**: Promote pressure threshold comparisons to f64: `let threshold = params.pressure_freeze_threshold as f64` before comparison. Test: threshold=0.80, pressure=0.800001 → must trigger.

**Paper citation:** [Gray & Reuter 1992] "Transaction Processing" §crash recovery — state that survives crashes must be explicitly persisted

**Expected gain:** Correct restart behavior (BUG-01), bounded memory (BUG-02), stable decisions near thresholds (BUG-03)

**Risk:** Low-Medium — BUG-01 touches persistence layer (needs `#[serde(default)]`)

---

## P5: ARM64 Optimization Phase 3 — RwLock Migration [B023, priority=0.765]

**Belief**: Hyperedge "ARM64 Optimization Phases" is EXTRACTED — documented plan awaits execution.
**Truth**: <f=0.90, c=0.85>

**Evidence:**
- Hyperedge nodes: `phase1_thread_scheduling`, `phase2_direct_mach`, `phase3_rwlock`
- PLAN_ARM64_OPTIMIZATIONS.md documents 5 phases; Phase 3 (RwLock) is lowest risk of remaining
- Collector hot path is read-dominant: 8 readers per 1 writer under normal load

**Proposed mutation for apollo-evolve:**
Migrate `Arc<Mutex<>>` to `Arc<RwLock<>>` in read-dominant paths in collector and daemon_state. Metric: `cargo bench` latency on `bench_collect_pressure_facts_latency`. Guard: `cargo test`.

**Paper citation:** [Bos 2022] "Rust Atomics and Locks" Ch.4 — RwLock optimal for read-dominant shared state on multi-core

**Expected gain:** -20% lock contention on M1, improved throughput on efficiency cores

**Risk:** Medium — requires audit of all write sites to prevent write starvation

---

## P6: outcome_tracker.rs — Extract CausalGraph [B004, priority=0.817]

**Belief**: `outcome_tracker.rs` (66 edges) + C1 low cohesion (0.04) — God Object with separable concerns.
**Truth**: <f=0.83, c=0.99>

**Evidence:**
- Graph: 66 edges cross-feeding NARS, causal graph, RL, signal intelligence
- BUG-02 is in this module (co_occurrence eviction) — easier to fix after extraction
- C1 (Outcome Tracker & Adaptive Wait) cohesion=0.04 — fragmented

**Proposed mutation for apollo-evolve:**
After fixing BUG-02, extract `CausalGraph` and `CoOccurrenceGraph` into separate files. Reduces outcome_tracker.rs by ~150-200 LOC and makes causal graph independently testable.

**Paper citation:** [Parnas 1972] "On the Criteria To Be Used in Decomposing Systems into Modules" — module = secret, not layer

**Expected gain:** outcome_tracker.rs from 66 to ≤40 edges; causal_graph.rs gains independent test coverage

**Risk:** Medium — mechanical move, updates imports across 6-8 call sites

---

## P7: safety.rs — Single Source of Truth for Protection [B007, priority=0.822]

**Belief**: `safety.rs` (59 edges) — 3 diverging protection lists across modules.
**Truth**: <f=0.83, c=0.99>

**Evidence:**
- 3 separate lists: `safety.rs::protected_processes()`, `decide_actions.rs::INTERACTIVE_APPS`, `thermal_interrupt.rs::sentinel_buffers`
- Cross-module invariant tests added in prior session confirm the divergence risk
- Graph: safety.rs has 59 edges — high fan-in consistent with authority module

**Proposed mutation for apollo-evolve:**
Expose `safety::is_interactive_protected(name: &str) -> bool` as single source of truth, replacing `INTERACTIVE_APPS` constant in `decide_actions.rs`. Thermal interrupt consults `safety::is_protected_static()`.

**Paper citation:** [Lampson 1974] "Hints for Computer System Design" §6 — single source of truth for protection policy

**Expected gain:** Eliminates protection list divergence permanently; -15 LOC duplicated constants

**Risk:** Low — safety.rs already owns this domain; mechanical substitution

---

## Summary Table

| # | Subject | Priority | f | c | Risk | Delta |
|---|---------|----------|---|---|------|-------|
| P1 | prepare.rs + decide() | 0.842 | 0.85 | 0.99 | Low | -40 to -80 LOC |
| P3 | chromium_manager.rs | 0.840 | 0.85 | 0.99 | Low | -30 to -50 LOC |
| P2 | predictive_agent.rs | 0.838 | 0.85 | 0.99 | Medium | -30 LOC |
| P7 | safety.rs single source | 0.822 | 0.83 | 0.99 | Low | -15 LOC |
| P6 | outcome_tracker decompose | 0.817 | 0.83 | 0.99 | Medium | -150 to -200 LOC |
| P4 | Critical Bugs BUG-01/02/03 | 0.765 | 0.90 | 0.85 | Low-Med | +5 tests |
| P5 | ARM64 Phase 3 RwLock | 0.765 | 0.90 | 0.85 | Medium | -20% latency |

**Recommended sequencing for apollo-evolve:**
P4 (bugs first) → P7 (safety consolidation) → P3 (chromium inline) → P1 (prepare helpers) → P6 (outcome_tracker) → P5 (ARM64)
