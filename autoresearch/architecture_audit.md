# Apollo Optimizer — Architecture Audit & Bug Map

**Date**: 2026-04-02 | **Codebase**: ~54K lines, 105 modules (Rust 2021)

---

## Architecture Map

### Key Data Flow
```
collector.rs → SystemSnapshot
    → signal_intelligence.rs (zone routing, pressure boosting)
        → decide_actions.rs (what to do)
            → execute_actions.rs (apply SIGSTOP/sysctl)
                → outcome_tracker.rs (did it work?)
                    → rule_inducer.rs (every 100 cycles → new skills)
                        → skill_registry / optimization_skills.json
                            → matching_skills() (next cycle)
```

### Daemon Main Loop (`src/bin/apollo-optimizerd/main.rs`, ~4700 lines)
- Cycle duration: min 300 ms
- Key in-memory state: `pending_trial_skill: Option<(String, f64)>` (line ~1052) — **NOT PERSISTED**
- Frequency gates: `% 100` (rule_inducer), `% 500` (GC), `% 7200` (hourly housekeeping)
- Reactor thread: parallel kqueue-based pressure monitoring

### Persistence Layer
| File | Module | Trigger |
|------|--------|---------|
| `learned_state.json` | learned_state.rs | every cycle (persist_improved) |
| `optimization_skills.json` | optimization_skills.rs | rule induction + GC |
| `frozen_state.json` | main.rs (5 sites) | every freeze/unfreeze change |
| `learned_policy.json` | llm.rs | LLM async task |
| `overflow_history.json` | overflow_guard.rs | policy update |

---

## Bug Map

### 🔴 BUG-01 — `pending_trial_skill` lost on daemon restart
**File**: `main.rs:1052, 3064-3122`
**Severity**: Critical — learned skills never stabilize across restarts
**Mechanism**: `pending_trial_skill` is a stack `Option<(String, f64)>`. If the daemon crashes or is restarted between the cycle that applies a trial skill and the next cycle that measures its result, the trial is recorded as neither effective nor ineffective. The skill's `apply_count` never gets a success observation → eventually GC'd as ineffective.
**Fix**: Add `pending_trial_skill` to `LearnedState` (trivial: one field + collect/apply lines).
**Complexity**: Trivial (~10 lines).

---

### 🔴 BUG-02 — `co_occurrence` under-eviction (off-by-one in retain)
**File**: `src/engine/outcome_tracker.rs:483-488`
**Severity**: Critical — memory bloat + rule induction quality degradation over time
**Mechanism**:
```rust
let cutoff = counts[counts.len().saturating_sub(100)];
self.co_occurrence.retain(|_, &mut v| v > cutoff);  // strict > evicts entries == cutoff
```
When multiple pairs share the cutoff count, ALL of them are evicted. The map can grow back to 151 before next GC fires. Over days/weeks, low-signal "ghost pairs" accumulate and are never evicted.
**Fix**: Change `v > cutoff` → `v >= cutoff`.
**Complexity**: Trivial (1 character).

---

### 🔴 BUG-03 — f32 precision loss in pressure thresholds
**Files**: `main.rs:3069, 3075` / `optimization_skills.rs:40, 62-63`
**Severity**: Critical — skill gates misfire silently
**Mechanism**: `snapshot.pressure.memory_pressure` is `f64`. All calls to `next_trial_skill()` and `record_result_with_pressure()` cast to `f32`. f32 only has ~7 significant decimal digits vs f64's ~15. At pressure=0.5500, f32 rounding adds ±2.4e-8 jitter. Skills near threshold boundary trigger inconsistently → success_rate stays low → premature GC.
**Fix**: Store `min_pressure` as `f64`; convert to `f32` only at persist time.
**Complexity**: Moderate (type change through OptimizationSkill + callers).

---

### 🟡 BUG-04 — `outcome_tracker.weights` HashMap unbounded growth
**File**: `src/engine/outcome_tracker.rs:336`
**Severity**: Medium — slow memory leak over days/weeks
**Mechanism**: `self.weights.entry(name).or_default()` auto-vivifies entries. Transient build processes, test runners, etc. each create a permanent entry. No GC. After 30 days: 500–1000 dead entries (~100 KB, but violates daemon's bounded-memory invariant).
**Fix**: Add `gc_weights()` called every 500 cycles: `self.weights.retain(|_, w| w.throttle_count >= 5)`.
**Complexity**: Trivial.

---

### 🟡 BUG-05 — Trial skill skipped when target is foreground, recorded as ineffective
**File**: `main.rs:3099-3125`
**Severity**: Medium — foreground apps never get reliable trials
**Mechanism**: If a trial skill's target is the foreground PID, the throttle is skipped but `trialed` stays `false`, causing `record_result(&name, false)` to be called. Skills targeting frequently-active foreground apps (browsers, editors) accumulate "ineffective" counts even though they were never actually tested. After 10 such cycles, `should_retire()` fires.
**Fix**: Track `targets_found_but_skipped` separately; only record ineffective if target is truly absent from process list (not just foreground-skipped).
**Complexity**: Moderate.

---

### 🟡 BUG-06 — Cycle counter scheduling inconsistency
**File**: `main.rs:1182, 1626, 3898, 4703`
**Severity**: Medium — inconsistent first-cycle vs steady-state behavior
**Mechanism**: Mix of `% N == 0` and `% N == 1` patterns throughout:
- `cycle_count % 10 == 1` → fires on cycles 1, 11, 21
- `cycle_count % 60 == 0 || cycle_count == 1` → fires on 1, 60, 120
- `cycle_count % 7200 == 1` → fires on 1, 7201, 14401
The special-case `cycle_count == 1` on some checks creates divergent first-cycle behavior that doesn't repeat in steady state.
**Fix**: Standardize all to `% N == 0` pattern. Remove `|| cycle_count == 1` special cases.
**Complexity**: Trivial (but tedious).

---

### 🟡 BUG-07 — Protected process check inconsistency (5 divergent sites)
**Files**: `safety.rs:64` vs `main.rs:3092` vs `main.rs:1267` vs `execute_actions.rs:191`
**Severity**: Medium — wrong processes may be throttled or incorrectly protected
**Mechanism**:
- `safety.rs:64`: exact HashSet membership (`protected_processes().contains(name)`)
- `main.rs:3092` (trial loop): `target.contains(p)` — substring match
- `main.rs:1267` (display_turbo): `name.contains(p)` — substring match, no word boundary
- `execute_actions.rs`: own protection check (unclear variant)

A policy pattern like `"box"` would protect `Dropbox`, `Sandbox`, `Mailbox` via substring, but only the exact entry `"box"` via HashSet. Sites apply different strategies to same process.
**Fix**: Centralize into `safety::is_protected(name, &protected_list) -> bool` with consistent word-boundary logic.
**Complexity**: Moderate.

---

### 🟡 BUG-08 — `skill_registry` and `outcome_tracker` effectiveness diverge
**Files**: `optimization_skills.rs`, `outcome_tracker.rs`, `causal_graph.rs`
**Severity**: Medium — triple-tracking same concept with divergent results
**Mechanism**: Three independent systems each track "did throttling process X work?":
1. `OutcomeTracker.weights` — Bayesian `effective_count / throttle_count`
2. `SkillRegistry.success_rate` — EMA via `record_result()`
3. `CausalGraph` — confidence EMA via `evaluate()`

These never cross-feed. A process can be 70% effective in OutcomeTracker and 30% in CausalGraph simultaneously. No reconciliation mechanism.
**Fix**: Either feed OutcomeTracker results into SkillRegistry, or merge into a single EffectivenessTracker.
**Complexity**: Moderate to major.

---

### 🟡 BUG-09 — Frozen processes not proactively unfrozen on startup after crash
**File**: `main.rs:581, 5083-5091`
**Severity**: Medium — user sees hung processes if daemon crashes
**Mechanism**: On clean shutdown (SIGTERM), daemon unfreezes all tracked processes (line 5083). On crash, it doesn't. On restart, frozen_state.json is loaded, but unfreezing only happens if the next decision cycle's `should_unfreeze()` fires — which requires the pressure condition to normalize, which may take multiple cycles.
**Fix**: On startup, immediately SIGCONT all PIDs loaded from frozen_state.json before entering main loop.
**Complexity**: Trivial.

---

### 🟢 BUG-10 — `pending VecDeque` silent data loss
**File**: `src/engine/outcome_tracker.rs:347-349`
**Severity**: Low — rare, capped, but silent
**Mechanism**: `if self.pending.len() > 300 { self.pending.drain(..100); }` — 100 pending outcomes discarded silently. No log, no metric. During high-pressure spikes with many simultaneous throttles, oldest pending outcomes lose their effectiveness measurement.
**Fix**: Add `tracing::warn!(drained, "pending outcomes discarded (cap reached)")`.
**Complexity**: Trivial.

---

## Dead Code (Safe to Remove)

| Symbol | File | Status |
|--------|------|--------|
| `optimizer.rs::optimize()` | src/optimizer.rs | Never called; modern path uses decide_actions |
| `TransformerPredictor` | src/engine/signal_intelligence.rs | Disabled in code |
| `TelemetryLogger` | (unknown) | Disabled per project notes |

---

## Summary Table

| ID | Sev | Title | File | Fix Complexity |
|----|-----|-------|------|---------------|
| BUG-01 | 🔴 | pending_trial_skill lost on restart | main.rs:1052 | Trivial |
| BUG-02 | 🔴 | co_occurrence under-eviction | outcome_tracker.rs:488 | Trivial |
| BUG-03 | 🔴 | f32 precision in pressure thresholds | main.rs:3069 | Moderate |
| BUG-04 | 🟡 | weights HashMap unbounded growth | outcome_tracker.rs:336 | Trivial |
| BUG-05 | 🟡 | foreground trial skip → false ineffective | main.rs:3099 | Moderate |
| BUG-06 | 🟡 | cycle counter scheduling inconsistency | main.rs:1182 | Trivial |
| BUG-07 | 🟡 | 5 divergent protection check sites | safety.rs, main.rs | Moderate |
| BUG-08 | 🟡 | skill_registry ≠ outcome_tracker sync | opt_skills.rs | Moderate |
| BUG-09 | 🟡 | frozen processes not cleared on crash | main.rs:581 | Trivial |
| BUG-10 | 🟢 | pending VecDeque silent drain | outcome_tracker.rs:347 | Trivial |
