# Apollo NARS Proposals — Paper-Gap Session 2026-04-10

Generated from: `papers/apollo_agi_paper_draft.md` + code grep analysis  
Baseline: 1602 tests | 165/165 scenarios | AIS 99.5

---

## P1: nested_learner.rs — Wire L2→L0 Dynamic Gate Feedback
**Belief ID**: B031  **Priority**: 0.810  **Truth**: <0.90, 0.90>

**Evidence:**
- `nested_learner.rs:51`: `L1_GATE_THRESHOLD: f64 = 0.25` — hardcoded constant
- `flush_l2()` exposes `l2_context` but zero callers use it to adjust the gate
- Paper §6.2: "L2 meta-velocity feeds back to L0's gate threshold: if L2 detects rapid meta-changes, it raises L0's quality requirement"
- Paper claim is architecturally described in Definition 1 (`σ` cycle function) but unimplemented

**Proposed mutation for apollo-evolve:**

In `src/engine/nested_learner.rs`:

1. Add `l2_prev_context: f64` and `l2_meta_velocity: f64` fields to `NestedLearner`
2. In `flush_l2()`: compute `l2_meta_velocity = EMA(|l2_context - l2_prev_context|)`, update `l2_prev_context`
3. Add `fn dynamic_l1_gate(&self) -> f64` returning `L1_GATE_THRESHOLD + 0.20 * self.l2_meta_velocity`  
   (clamp [0.10, 0.60]) — high meta-velocity → raise quality bar
4. Replace `self.l0_quality >= L1_GATE_THRESHOLD` in `tick_l0()` with `self.l0_quality >= self.dynamic_l1_gate()`
5. Add 3 unit tests: low velocity → gate≈0.25, high velocity → gate rises, gate clamps at 0.60

**Paper citation:**
Google Nested Learning 2025 §6.2 — bidirectional context flow prevents catastrophic forgetting;
Hochreiter & Schmidhuber 1997 — multi-timescale memory prevents gradient vanishing

**Expected gain:**
- Closes the last architectural claim from §6.2 that was described but unimplemented
- Under rapid workload changes (rustc → LLM → browser), meta-velocity rises → L1 gate tightens → fewer noisy beliefs revise in unstable regimes
- Paper's §8.3 L2 limitation becomes less severe (benchmark detects this dynamic behavior)
- +3-5 unit tests, ~30 LOC

**Risk:** Low — gate only gets stricter, never bypasses safety. Default behavior (velocity=0) matches current constant gate exactly.

---

## P2: causal_graph.mechanism() → QoS vs SIGSTOP Routing
**Belief ID**: B032  **Priority**: 0.774  **Truth**: <0.88, 0.88>

**Evidence:**
- `causal_graph.rs:415`: `pub fn mechanism(&self, action_key: &str) -> Option<(&str, f32, f32, f32)>` — API exists
- `causal_graph.rs:418`: requires `observations >= 3` before returning data — safe fallback built in
- `main.rs`: all `set_tier(pid, SchedulingTier::Background)` calls use hardcoded tier regardless of mechanism
- Zero matches for `mechanism.*set_tier` or `causal.*qos` in entire codebase
- Paper §5.2: "if a process's causal effect operates primarily through CPU reduction, Apollo can use QoS tiering rather than SIGSTOP, preserving the process's ability to respond to events"

**Proposed mutation for apollo-evolve:**

In the throttle decision path in `src/bin/apollo-optimizerd/main.rs` (or `execute_actions.rs`):

When deciding how to throttle a non-protected process `pid` with `name`:
```
let action_key = format!("throttle:{}", name);
let use_qos = if let Some((primary, _, _, _)) = causal_graph.mechanism(&action_key) {
    primary == "cpu_reduction"   // CPU-dominant → QoS tier (gentler, process stays responsive)
} else {
    false  // no mechanism data yet → default SIGSTOP (conservative)
};

if use_qos {
    qos.set_tier(pid, SchedulingTier::Background);
} else {
    // existing SIGSTOP path
}
```

Add integration test: mock causal graph with cpu_dominant edge → verify QoS path taken; rss_dominant edge → verify SIGSTOP path.

**Paper citation:**
Pearl 2009 Ch.3 — mediation analysis: identify causal pathway (mechanism), not just causal effect;
Nygard 2018 — bulkhead: least-invasive intervention first

**Expected gain:**
- CPU-dominant processes (background daemons, Electron apps doing JS) get QoS throttle instead of SIGSTOP → still respond to user events, less jank
- Closes paper §5.2 claim completely
- +2 integration tests, ~50 LOC

**Risk:** Medium — affects daemon hot path. Safe because: (a) requires ≥3 causal observations before routing (cold processes default to SIGSTOP), (b) QoS Background is less aggressive than SIGSTOP.

---

## P3: Continuous Workload Benchmark Generator
**Belief ID**: B033  **Priority**: 0.792  **Truth**: <0.90, 0.88>

**Evidence:**
- Paper §8.3 L2: "does not capture adversarial workloads, multi-hour gradual memory leaks, or hardware-fault conditions"
- Paper §8.3 Future Work: "continuous workload simulation benchmark replacing fixed scenarios"
- 165 scenarios are deterministic snapshots — no temporal evolution or drift
- NARS revision: B018 (Prediction & Forecasting, hyperedge_6_nodes) × paper L2 gap

**Proposed mutation for apollo-evolve:**

Add `tests/continuous_workload.rs` (or extend existing benchmark harness):

Define 4 workload sequences as pressure/swap time-series (50 steps each):
1. `compilation_spike`: linear rise 0.3→0.85, plateau 30 steps, decay
2. `browser_accumulation`: slow drift 0.5→0.75 over 50 steps (memory leak pattern)  
3. `llm_steady`: constant 0.72 ± 0.03 noise (LLM inference)
4. `mixed_adversarial`: alternating compile+browse, rapid regime changes every 10 steps

Feed each sequence through `SignalIntelligence` + `NestedLearner` + `NarsBeliefs`.

Assert invariants at each step:
- Signal quality EMA converges within 20 steps
- L1 gate responds to regime changes (velocity > threshold → gate tightens)
- NARS confidence grows monotonically under stable regimes
- No panics / no safety invariant violations

**Paper citation:**
Page 1954 — CUSUM designed for continuous regime detection, not snapshot testing;
Kuncheva 2004 — concept drift requires streaming validation

**Expected gain:**
- Addresses §8.3 L2 limitation explicitly
- Validates P1 (dynamic gate) in realistic temporal sequences
- +4 scenario sequences × ~50 steps = 200 signal evaluations, ~80 LOC
- Paper can claim "continuous workload validation" in §7.3

**Risk:** Low — test-only, no production code changes.

---

## Execution Order for apollo-evolve

```
Iter 1: P1 (nested_learner L2→L0 feedback) — 30 LOC, Low risk
Iter 2: P3 (continuous benchmark) — validates P1 with temporal data
Iter 3: P2 (mechanism → QoS routing) — 50 LOC, Medium risk, guarded by causal evidence count
```

**NARS feedback rule for each iteration:**
- Test pass + scenario ↑ → `f += 0.05, c += 0.10` on the belief
- Test fail → `f -= 0.15, c += 0.05` (negative evidence accumulated)
- Reverted → `f -= 0.10, c += 0.03`
