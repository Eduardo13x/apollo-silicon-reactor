# Apollo Optimizer — Redundancy & Homologation Audit

**Date**: 2026-04-02 | **Codebase**: ~54K lines, 105 modules (Rust 2021)

---

## Executive Summary

**7 major redundancy clusters** spanning process protection, pressure signals, effectiveness tracking, persistence, frozen state, and skill prefix filtering. Consolidating these would reduce code surface ~8%, eliminate divergent behavior bugs, and make the daemon easier to reason about.

**Top 5 by impact:**
1. Unify process protection checks (5 sites, 4 matching strategies)
2. Expose pressure boosts to `decide_actions` (currently invisible)
3. Merge outcome/effectiveness tracking (3 independent rings)
4. Unify persistence (4 independent .json files)
5. Batch frozen state writes (5 write sites, no atomicity)

---

## Section 1: Duplicated Logic

### 1.1 Process Protection Checks — 5 Divergent Sites

| Site | File:Line | Strategy | Lists Used |
|------|-----------|----------|------------|
| A — safety.rs exact | safety.rs:64 | Exact HashSet `.contains(name)` | `protected_processes()` |
| B — display_turbo | main.rs:1267 | Substring `name.contains(p)` | `protected_pats`, `critical_pats`, `policy_protected` |
| C — trial loop | main.rs:3092 | Substring `target.contains(p)` | `hard_protected`, `policy_prot` |
| D — heuristic_critical_pids | main.rs:3362-3389 | Substring + behavioral gate via `is_user_interactive_app()` | All above + behavior score |
| E — execute_actions | execute_actions.rs:191 | Substring `target.contains(p)` | `protected_processes()`, `learned_protected` |

**Key divergence**: Site A uses exact match; Sites B–E use substring — no word-boundary checks. If `"Box"` ever enters the protected list, `"Dropbox"`, `"Sandbox"`, and `"Mailbox"` get silently protected at Sites B–E but not at Site A.

**Site D is the most correct**: it has the foreground-conditional behavioral gate (`is_user_interactive_app`). Sites B and C don't. Display_turbo (Site B) could freeze a foreground user app that Site D would conditionally protect.

---

### 1.2 Pressure Signals — 3 Independent Formulas

| Site | Formula | Who uses it |
|------|---------|-------------|
| `collector.rs` | Raw `memory_pressure` from sysinfo | Base source |
| `main.rs:2150-2182` | `pressure_ram = (base + hw_boost + batt_boost + thermal + llm + charging + mem_bw + smc + battery_overheat).clamp(0,1)` — 10 factors | Main loop decision engine |
| `decide_actions.rs:89` | `ram_pressure = snapshot.pressure.memory_pressure` | decide_actions context |

**Critical gap**: `decide_actions` sees raw pressure. The main loop sees pressure boosted by up to +0.40 (thermal + LLM + hardware combined). This means decide_actions systematically underestimates system stress at margin thresholds (0.55–0.70 range).

---

### 1.3 Effectiveness Tracking — 3 Independent Rings

| System | File | Metric | Update Mechanism |
|--------|------|--------|-----------------|
| `OutcomeTracker.weights` | outcome_tracker.rs | `effective_count / throttle_count` (Bayesian) | `record_outcome()` on pressure delta |
| `SkillRegistry.success_rate` | optimization_skills.rs | `success_count / apply_count` (EMA on record_result) | `record_result()` in trial loop |
| `CausalGraph.confidence` | causal_graph.rs | EMA confidence per edge | `evaluate()` on action→outcome delay |

Same process can have 3 divergent effectiveness ratings with no reconciliation. Currently: SkillRegistry wins in trial loop; CausalGraph is computed but only partially consumed (main.rs:3134 uses `top_causal_pairs` for coordinated freeze, not `solid_edges_by_impact`).

---

### 1.4 Co-occurrence Tracking — 2 Overlapping Systems

| System | File | Tracks | Method |
|--------|------|--------|--------|
| `outcome_tracker.co_occurrence` | outcome_tracker.rs:469 | Process pairs co-spiking: `HashMap<(String,String), u32>` | `record_co_occurrence(active_procs)` every cycle |
| `causal_graph` edges | causal_graph.rs | Action→outcome causal edges with confidence | `record_action()` + `evaluate()` |

OutcomeTracker tracks correlation (A and B both ran during spike). CausalGraph tracks causation (A caused pressure drop with 0.8 confidence). Both are fed from the same events but neither informs the other. Rule inducer uses only OutcomeTracker correlations → generates group skills based on co-occurrence, not causal evidence. Result: some induced skills target processes that co-occur but don't causally drive pressure.

---

### 1.5 Frozen State Management — 5 Write Sites

| Site | File:Line | Operation |
|------|-----------|-----------|
| Init | main.rs:581 | `load_frozen_state()` |
| display_turbo freeze | main.rs:1258-1294 | lock → insert → `write_frozen_state()` |
| display_turbo unfreeze | main.rs:1303-1307 | lock → remove → `write_frozen_state()` |
| foreground-switch unfreeze | main.rs:1364-1367 | lock → remove → `write_frozen_state()` |
| execute_actions result | (execute_actions.rs) | modify + write |
| shutdown cleanup | main.rs:760, 1220 | lock → clear → `write_frozen_state()` |

Every individual freeze/unfreeze triggers a disk write. No batching. Not part of unified `learned_state.json` → frozen state can be out-of-sync with other learned state on crash.

---

### 1.6 Skill Prefix Filtering — 2 Conflicting Filter Sites

| Site | File:Line | Filter |
|------|-----------|--------|
| `next_trial_skill` | optimization_skills.rs:138-149 | `starts_with("group:") \|\| starts_with("batch:")` — only induced skills |
| `purge_unexecutable` | optimization_skills.rs:154-164 | `starts_with("group:") \|\| starts_with("batch:")` — only removes induced |

**Gap**: Individual `throttle:X` and `induced:X` skills are never re-trialed even if they become unreliable over time. Only auto-induced skills go through the trial loop. If `throttle:Safari` drops to 20% success_rate, it never gets re-evaluated.

---

### 1.7 Persistence — 4 Independent .json Files

| File | Module | In LearnedState? |
|------|--------|-----------------|
| `learned_state.json` | learned_state.rs | YES (canonical) |
| `optimization_skills.json` | optimization_skills.rs | NO (own persist) |
| `learned_policy.json` | llm.rs | NO (own async load) |
| `overflow_history.json` | overflow_guard.rs | NO (own persist) |
| `frozen_state.json` | main.rs | NO (5 manual write sites) |

If daemon crashes mid-cycle, these 4 files can be in inconsistent states with each other. No transaction semantics.

---

## Section 2: Consolidation Opportunities (Ranked by Impact)

### C-01 🔴 Unify Protection Checks → `safety::check_protected()`
**Current**: 5 sites, 4 strategies, divergent behavior.
**Proposal**:
```rust
pub enum ProtectionLevel {
    Unconditional,          // OS/infra essentials — never touch
    ConditionalForeground,  // User apps — protect only when active
    Unprotected,
}

pub fn check_protected(
    name: &str,
    hard: &HashSet<&str>,
    policy: &[String],
    behavior: Option<&BehaviorScore>,
) -> ProtectionLevel
```
Single function used by display_turbo, trial loop, heuristic_critical_pids, execute_actions.
**Savings**: ~150 lines. **Risk**: Medium (must preserve foreground-conditional behavior of Site D).

---

### C-02 🔴 Expose Pressure Boosts to decide_actions
**Current**: 10-factor pressure_ram computed in main.rs, invisible to decide_actions.
**Proposal**: Extract to `effective_pressure::compute(snapshot, smc, power) -> (f64, PressureComponents)`. Pass result to decide_actions instead of raw snapshot.
**Impact**: decide_actions sees true system stress. Decisions at 0.55–0.70 pressure range become more accurate.
**Savings**: ~80 lines (deduplicated). **Risk**: Low (pure extraction).

---

### C-03 🟡 Merge OutcomeTracker + CausalGraph → EffectivenessTracker
**Current**: 3 independent rings with divergent ratings.
**Proposal**: Single `EffectivenessTracker` blending Bayesian weight + causal confidence. SkillRegistry reads from it instead of tracking independently.
**Impact**: Single truth for "did this work". CausalGraph confidence now gates coordinated freeze.
**Savings**: ~200 lines. **Risk**: Medium (state migration, test coverage needed).

---

### C-04 🟡 Unify Persistence → Extended LearnedState
**Current**: 4 independent .json files, no cross-file atomicity.
**Proposal**: Add to LearnedState:
```rust
pub skill_registry: HashMap<String, OptimizationSkill>,
pub overflow_guard_state: OverflowGuardState,
pub frozen_pids: HashMap<u32, FrozenEntry>,
```
`learned_policy` stays separate (hot-reloaded by LLM async).
Include migration: on startup, detect old .json files → merge into unified state.
**Impact**: All learned state consistent on crash. Single `persist()` call.
**Savings**: ~120 lines. **Risk**: Medium (backward compat migration).

---

### C-05 🟡 Batch Frozen State Writes
**Current**: Every individual freeze/unfreeze triggers a disk write (5 sites).
**Proposal**: Collect all freeze/unfreeze operations per cycle, single write at cycle end:
```rust
let mut frozen_changed = false;
// ... all modifications set frozen_changed = true ...
if frozen_changed {
    write_frozen_state(&path, &frozen_state);
}
```
**Impact**: -95% disk I/O for freeze operations on high-pressure cycles.
**Savings**: ~40 lines. **Risk**: Very low.

---

### C-06 🟡 Wire CausalGraph into Coordinated Freeze Logic
**Current**: `causal_graph.solid_edges_by_impact()` computed but not used in coordinated freeze (main.rs:3130-3180).
**Proposal**: Add causal confidence filter to coordinated freeze: only act on co-occurrence pairs where at least one process has causal confidence > 0.4.
**Impact**: Fewer false co-occurrence-based freezes (correlation ≠ causation).
**Risk**: Low (additive filter; degrades gracefully when causal data is sparse).

---

### C-07 🟢 Standardize Skill Origin Taxonomy
**Current**: Prefix strings hardcoded in 2+ places (`starts_with("group:") || starts_with("batch:")`).
**Proposal**:
```rust
pub enum SkillOrigin { Individual, Induced }
impl SkillOrigin {
    pub fn from_name(name: &str) -> Self {
        if name.starts_with("group:") || name.starts_with("batch:") {
            SkillOrigin::Induced
        } else {
            SkillOrigin::Individual
        }
    }
}
```
**Impact**: Type-safe origin filtering. Enables re-trialing individual skills (pass `&[Individual]`).
**Savings**: ~20 lines. **Risk**: Very low.

---

### C-08 🟢 Throttle Action Constructor
**Current**: `RootAction::ThrottleProcess { pid, name, aggressive: false, reason: format!("..."), start_sec: 0, start_usec: 0 }` repeated 5+ times.
**Proposal**:
```rust
impl RootAction {
    pub fn throttle(pid: u32, name: impl Into<String>, aggressive: bool, reason: impl Into<String>) -> Self {
        RootAction::ThrottleProcess { pid, name: name.into(), aggressive, reason: reason.into(), start_sec: 0, start_usec: 0 }
    }
}
```
**Savings**: ~30 lines. **Risk**: Very low.

---

## Section 3: Homologation Proposals

### H-01: Protection Check API
```rust
// BEFORE (5 variants in 5 files):
if hard_protected.iter().any(|p| target.contains(p)) { continue; }

// AFTER (one call everywhere):
if safety::check_protected(&name, &hard, &policy, behavior) != ProtectionLevel::Unprotected {
    continue;
}
```

### H-02: Pressure API
```rust
// BEFORE: raw in decide_actions, boosted in main.rs
let ram_pressure = snapshot.pressure.memory_pressure;

// AFTER: both get same effective pressure
let (pressure, _components) = effective_pressure::compute(&snapshot, &smc, &power_mgr);
decide_actions(ctx, pressure);
```

### H-03: Effectiveness API
```rust
// BEFORE:
outcome_tracker.record_co_occurrence(&procs);
causal_graph.record_action(&name, pressure, cycle);
skill_registry.record_result(&name, was_effective);

// AFTER:
effectiveness_tracker.record_action(&name, pressure, &procs, cycle);
effectiveness_tracker.evaluate_outcomes(current_pressure, cycle);
```

### H-04: Frozen State Lifecycle
```rust
// BEFORE: 5 independent write calls spread across code
frozen_state.insert(pid, entry);
write_frozen_state(&path, &frozen_state);  // immediately

// AFTER: mutations collected, single write per cycle
frozen_mutations.push(FrozenMutation::Freeze(pid, entry));
// ...end of cycle:
if !frozen_mutations.is_empty() {
    apply_mutations(&mut frozen_state, &frozen_mutations);
    write_frozen_state(&path, &frozen_state);
}
```

---

## Implementation Roadmap

| Phase | Items | Risk |
|-------|-------|------|
| **P1** (trivial wins) | BUG-02 (co_occurrence fix), BUG-09 (unfreeze on startup), C-08 (throttle constructor), C-05 (batch frozen writes) | Very low |
| **P2** (API foundation) | C-01 (unified protection check), C-02 (effective_pressure module) | Medium |
| **P3** (migration) | C-04 (unified persistence), C-07 (skill origin enum) | Medium |
| **P4** (major refactor) | C-03 (merge effectiveness tracking), C-06 (wire CausalGraph) | High |
