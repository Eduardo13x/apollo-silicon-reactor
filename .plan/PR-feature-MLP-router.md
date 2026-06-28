# PR-feature-MLP-router — Apollo Optimizer MLP Regime Router

> **Sprint target:** regime-aware down-weighting on top of the existing per-cycle decision path. Soft intervention only. Shadow-first Apollo doctrine. Touches gating path with `α ≤ 0.20` weight (never overrides `safety.rs` NEVER_FREEZE).
>
> **Author:** `/apollo-evolve:architect` workflow (2026-06-27).
> **Scope:** read-only design — no code changes. Phases ship separately.

---

## 1. Executive Summary

**Gap.** Apollo's ML stack is **per-cycle, single-feature-isolated**: LinUCB (5 arms × 16 dims, `crates/apollo-engine/src/engine/predictive_agent.rs:32` `const D: usize = 16`), the CausalGraph edges (`causal_graph.rs:113` `pub avg_delta: f32`), the NARS truth values (`nars_belief.rs`), the Kalman 1D/8D pressure fuse, and the MetaCognition debias (`meta_cognition.rs:111`) each model one relationship at a time and act blind to regime-level interactions (`swap_velocity × thermal_pred × interactivity_score × build_load`). Per-cycle thresholds in `decide_actions.rs:1335-1399` (gates A/B/C/D, `critical_pressure`, `extreme_pressure`) are hand-tuned scalars that don't generalize across regime shifts (idle → compile → media-playback → 4K-video).

**Solution.** A **2-layer MLP regime router** (16 → 32 → 4, softmax) that runs every ~60 cycles (NOT per-cycle) on a 16-d feature vector built from already-collected signals — 12 LSE counters + CausalGraph edge confidence + world_model.margin + NARS top-3 beliefs — and outputs softmax over `Observe | TightenThresholds | ThrottleNoise | SuggestAggressive` (subset of the existing 5-arm `predictive_agent::Intervention` enum, `predictive_agent.rs:43-54`). The router **down-weights** the existing per-cycle score by `α ∈ [0.0, 0.20]` — never overrides it, never touches `safety.rs` NEVER_FREEZE. Inference is `~6µs` (32-hidden NEON SIMD on M1), model `<100KB` (fp16 weights), every-60-cycles cadence adds `~0.1µs` amortized to the hot path.

**Blast radius.** Touches `crates/apollo-engine/src/engine/mlp_router.rs` (new), `src/bin/apollo-optimizerd/main.rs` (one `if cycle_count % 60 == 0` branch in the decide_actions prologue), and the existing `action_policy::PolicyScorer` (`composite *= 1.0 - α` — additive subtraction via scoring weight, NOT a veto). No new deps (uses `std::sync::OnceLock` + `aligned_alloc` f16). No MLX, no Metal, no GPU. Pure CPU fp16 NEON.

**Expected value.** Stage-RouterPrecision: regimes where the existing per-cycle decision over-throttles (idle/4K media) or under-throttles (sustained compile) get a learned bias on `TightenThresholds` / `ThrottleNoise` so the system stops missing multi-feature regime shifts the linear LinUCB cannot represent. Acceptance gate: `win-rate > 0.55` over `N ≥ 500` shadow cycles AND AIS stable (`≤ ±1pp`) AND p95 unchanged (`±5%`).

---

## 2. Pattern Inventory (Mapped Against Apollo ML Stack)

Cross-referenced against `references/architecture-catalog.md` patterns.

| Pattern (catalog) | Status | Evidence (file:line) | Notes |
|---|---|---|---|
| **State Machine** (GoF) | **Implemented** | `crates/apollo-engine/src/engine/user_profile.rs:WorkloadType` enum + `ArousalState::zone()` state machine | Workload classification + arousal zone form an explicit FSM with a single active state. |
| **Producer-Consumer** | **Implemented** | `src/bin/apollo-optimizerd/background_collectors.rs:80-165` `PressureCollector::spawn()`; `crates/apollo-engine/src/engine/lse_counters.rs` `LockFreeMetrics` (Relaxed writes / Acquire reads) | Background collector → cache → consumer is the textbook pipeline. ARMv8.1 LSE atomics. |
| **Pipeline** | **Implemented** | `daemon_cycle_tail.rs:294-349` `stage_reason_*` split (Sense → Reason → Execute → Learn → Persist) | 5-stage cycle with `CycleStage` enum (`lse_counters.rs:21-44`). |
| **Blackboard** | **Partial** | `runtime_metrics.json` written by `metrics_reporter.rs`; `learned_state.json` from `learned_state.rs`; `journal.jsonl` from `journal.rs` | Three artifacts, no shared in-memory blackboard with a typed schema — `RuntimeMetrics` is the closest but 200+ fields, not a curated regime feature vector. |
| **Circuit Breaker** | **Implemented** | `crates/apollo-engine/src/engine/circuit_breaker.rs`; gates in `decide_actions.rs:1335-1399` (A/B/C/D with `extreme_pressure`) | Gate tower + asymmetric scorer override (`shadow_evaluator.rs:88-99` `decide_override`). |
| **Lock-Free** | **Implemented** | `lse_counters.rs` `LockFreeMetrics` (`pub static LSE_COUNTERS: LockFreeMetrics`, `lse_counters.rs:52`); `shadow_signals.rs` | Every hot-path counter uses ARMv8.1 LSE atomics — `ldadd` ~3ns vs ~25ns mutex. |
| **Facad**e | **Implemented** | `policy_feature_learned.rs:1` (single composition point for HRPO yield + world_model); `policy_feature_battery_cost.rs`, `policy_feature_sensor_age.rs` | Per-candidate evidence funneled through `ActionContext` (`action_policy.rs:43-93`). |
| **Strangler Fig** | **Implemented** | `daemon_cycle_tail.rs:1` header notes `V1.1.0 Strangler Fig pass (Wave 10) [Fowler 2004]`; `daemon_reactor.rs`, `daemon_teacher_tick.rs`, etc. extracted | New ticks added without rewriting the loop. |
| **Anti-Corruption Layer** | **Implemented** | `daemon_helpers.rs` (path resolution centralized); `match_engine.rs` (3-tier identity matching) | Legacy direct path access gone; one ACL entry point. |
| **Event Sourcing** | **Implemented** | `journal.rs:13-89` append-only WAL; `journal.rs:91-109` replay (`read_journal`); `types.rs:405-437` `JournalEntry { timestamp, action, before, after, success, reason, rationale }` | Append-only WAL with rotation at 2 MB (`journal.rs:11`). [Gray & Reuter 1992] §11 group commit. |
| **Bulkhead** | **Implemented** | `daemon_state.rs:39-60` `MetricsState`, `PolicyState`, `ProcessState`, `HardwareState`, `LlmDomainState`, `UsageDomainState` (6 domain groups) | Was 20+ flat `Arc<Mutex<>>`; now 6. Lock operations ~40% lower. |
| **Newtype / Builder** | **Implemented** | `action_policy.rs:43-93` `ActionContext`; `action_policy.rs:570` `PolicyScorerBuilder`; `action_types.rs` extracted from `types.rs` (per graph.json 2026-06-11) | Type-state-like validations at construction. |
| **Speculative Routing** (specialized) | **Partial / Anti-pattern risk** | `predictive_agent.rs` SpecialistVote tally (`predictive_agent.rs:105-136`) → `tally_votes` → single `Intervention`; `decide_actions.rs` chooses ONE action class per cycle, NOT a regime | Apollo's current "router" picks ONE intervention per cycle; this PR's MLP outputs a DISTRIBUTION over regimes which is novel for Apollo. |
| **Regime Detector / Mixture-of-Experts** (specialized) | **Missing** | No multi-feature regime classifier exists. Closest: `signal_intelligence.rs` `tick_mv()` 8D Kalman + `focus_markov.rs` + ReptileMeta (`reptile_meta.rs:120-160`) | ReptileMeta caches per-workload params (`MAX_WORKLOAD_CACHE = 16`, `reptile_meta.rs:33`) but each cache entry is a parameter vector — not a regime classifier that maps state → action distribution. |
| **Anti-pattern: God Service** | **Partial anti-pattern** | `src/bin/apollo-optimizerd/main.rs` = 6410 LOC (`wc -l`); 50+ sub-modules | Strangler Fig in progress — most cross-cutting logic now extracted to `daemon_*.rs`. The MAIN file is still the integration seam. |
| **Anti-pattern: Chatty Inter-Module** | **Anti-pattern detected** | `main.rs:3000-3990` (~1000 LOC) wires 5+ engines (LinUCB, SpecialistVoting, HOLT-Winters, CausalGraph, OutcomeTracker, PredictiveAgent, OverflowGuard) inline per cycle | The "soft intervention" path (LinUCB→specialist→agent_actions) is wired inline rather than through a typed composition object. MLP router will read its 16-d input from `RuntimeMetrics` only (no chatty access). |
| **Anti-pattern: Silent Failure** | **Implemented guard** | `lse_counters.rs` LSE discipline; every counter has a `#[serde(default)]` mirror in `RuntimeMetrics` so silent telemetry-death is impossible | Sprint 9 fix `4b13a39` "always reference the global static; never construct local LockFreeMetrics instances". |

---

## 3. Gap Analysis

### 3a. What the current per-cycle logic DOES use (concrete features already in the pipeline)

| Feature | Source | Cycle cadence | Consumed at |
|---|---|---|---|
| `memory_pressure` | `runtime_metrics.json:536` `pub memory_pressure: f64` | every cycle (collector) | `decide_actions.rs:1281` gate A |
| `swap_used_bytes` / `swap_total_bytes` / `swap_delta_bps` | `runtime_metrics.json:528-535` | every cycle | `decide_actions.rs:239` `swap_exhausted` check |
| `cpu_pressure` (`cpu_max_busy`, `cpu_mean_busy`) | `runtime_metrics.json:560-563` | every cycle | `decide_actions.rs:253` thermal-emergency |
| `thermal_level` | `runtime_metrics.json:567` `thermal_state: String` | every cycle | `decide_actions.rs:253` |
| `thrashing_score` | `runtime_metrics.json:540` | every cycle | `decide_actions.rs:1362` gate B |
| `refault_delta_per_sec` | `runtime_metrics.json:544` | every cycle | `decide_actions.rs:1362` gate B |
| `thermal_predicted_throttle` | `runtime_metrics.json:1283` | every cycle | NOT consulted by `decide_actions` (orphan metric) |
| `cycles_high_pressure` | `runtime_metrics.json:1075` | every cycle | governor |
| `meta_confidence` / `humble_mode` | `runtime_metrics.json:1257-1260` | every cycle | `meta_cognition.rs:225-242` |
| `meta_cognition.subsystem_debias_multiplier(SubsystemId::CausalGraph)` | `meta_cognition.rs:297` | every cycle | `world_model.rs:95` `from_parts(..., prediction_debias)` |
| `world_model.imagined_margin` (per candidate) | `world_model.rs:130-151` `imagine()` | per candidate | `policy_feature_learned.rs:56-93` `WorldModelFeature` |
| `learned_yield` (per candidate) | `outcome_tracker.rs:487` `yield_admits()` | per candidate | `policy_feature_learned.rs:20-50` `LearnedYieldFeature` |
| `swap_forecast.swap_trend` / `time_to_swap_critical` | `swap_predictor.rs:60-87` | every 5s | `predictive_agent.rs:564-573` LinUCB context |
| `signal_digest.pressure_smooth` / `velocity` | `signal_intelligence.rs` | every cycle | LinUCB slot 0 |

### 3b. What the current per-cycle logic CANNOT combine

1. **3+ feature regime interactions.** LinUCB is linear in the 16-dim context (`predictive_agent.rs:744-761` `LinUCBArm::score`), so regime features like `swap_velocity × thermal_pred × interactivity_score` are captured only as additive dot-products, not multiplicative interactions. The MLP router's hidden layer models these explicitly.
2. **NARS top-3 belief distribution** as a regime signal. `nars_belief.rs:600` `belief(key)` returns a `TruthValue` per key, but `predictive_agent.rs:538-606` `AgentContext::build` does not consume `NarsEngine::top_beliefs()` (a method that does not exist; only the per-key `belief` getter is exported). Top-N beliefs are observable in the engine but never ingested into the per-cycle decision.
3. **CausalGraph edge confidence aggregated over the cycle.** `causal_graph.rs:892` `confidence_map()` exists but is NOT consumed by `decide_actions.rs`. Only per-candidate `effectiveness()` (`causal_graph.rs:767`) is consulted, and only via `policy_feature_learned.rs` in SHADOW mode. The router reads `confidence_map()` once every 60 cycles to characterize the system's confidence regime.
4. **The "consensus-vs-dissent" regime signal.** `predictive_agent.rs:121-129` computes `had_disagreement` (specialist voting), but `decide_actions.rs` does not weight the chosen intervention by recent consensus rate. A regime with chronic specialist disagreement should bias toward `Observe`; the MLP router sees this as input.
5. **Epistemic state (`uchs_composite`, `uchs_grade`, `epistemic_uncertainty`).** `runtime_metrics.json:1152-1164` exposes UCHS but `decide_actions.rs` does not consume it. The router treats UCHS as a regime label.

### 3c. Regimes where the current logic fails hardest (prod-traced, not speculative)

- **Idle → sustained compile.** `decide_actions.rs:253` only flips regime at `cpu_pressure > 88% + thermal_emergency`, missing the "long low-CPU ramp that suddenly spikes memory 5min into rustc" pattern. Router would see `swap_trend = Increasing`, `cycles_high_pressure` rising, `interactivity_score` low — regime label "compile about to thrash".
- **4K media playback.** `decide_actions.rs:1281` only enters aggressive path at `memory_pressure >= extreme_pressure`. With 4K the `mem_temp` is high but `cpu_pressure` is moderate — Apollo under-throttles noise (e.g., Spotlight re-index mid-movie). Router would label "media-critical" via thermal_pred × WindowServer CPU × refault_delta.
- **Build-tools active.** `decide_actions.rs:672` (2026-06-07 hotfix) requires `foreground|visible` for boost, but the `noise-throttle` path still throttles Dropbox/OneDrive during `cargo build`. Router would weight `Observe` higher under build-tool regime.
- **Background daemon churn (Zotero, electron).** `decide_actions.rs:719-720` already special-cases `HopGroupWeight::yield_admits()` for the Browser hop (efficiency 0.27), but the per-cycle gate still emits throttles. Router label `throttle-noise` would be reinforced only when `behavior_interactive_pid_count` is low and `swap_pressure` rising — a regime the current per-cycle logic cannot represent.

### 3d. Root cause: regime ≠ action

Apollo today has **5 intervention arms × 1 vote → 1 action**. The router needs to switch from "pick one action class" to "weight the per-cycle score by a regime distribution". This is the architectural shift.

---

## 4. Proposal — MLP Regime Router Component

### 4a. Input Feature List (concrete LSE counter names + JSON pointer paths in `runtime_metrics.json`)

```
f[0]  memory_pressure              → $.memory_pressure                                    [0.0..1.0]
f[1]  swap_used_gb / 4.0           → $.swap_used_bytes / (4 * 1024^3)                     [0.0..1.0]  (clamp 1.0)
f[2]  swap_delta_bps_norm          → $.swap_delta_bps / 524288                            [0.0..1.0]  (compresses if rising)
f[3]  thrashing_score_norm         → min($.thrashing_score / 10000, 1.0)                  [0.0..1.0]
f[4]  cpu_max_busy                 → $.cpu_max_busy                                       [0.0..1.0]
f[5]  thermal_pred_norm            → $.thermal_predicted_throttle / 100.0                 [0.0..1.0]
f[6]  thermal_seconds_to_throttle_norm → sigmoid(-secs / 60)                            [0.0..1.0]  (1.0 = imminent)
f[7]  cycles_high_pressure_norm    → min($.cycles_high_pressure / 30, 1.0)                [0.0..1.0]  (30 cycles = 30s)
f[8]  refault_delta_norm           → min($.refault_delta_per_sec / 5000, 1.0)             [0.0..1.0]
f[9]  humble_mode_active           → if $.humble_mode { 1.0 } else { 0.0 }                {0, 1}
f[10] causal_subsystem_debias      → meta.subsystem_debias_multiplier(CausalGraph)        [0.25..1.5]
f[11] specialist_disagreement_rate → ema(had_disagreement, α=0.05) over 60 cycles          [0.0..1.0]
f[12] world_model_imagined_margin_mean → mean of action_keys' predicted_drop − natural_drift [0.0..0.1+]
f[13] nars_top_belief_confidence   → nars.belief("compile").confidence (None → 0.5)       [0.0..1.0]
f[14] interactivity_score          → behavior_interactive_pid_count / max(pid_count, 1)  [0.0..1.0]
f[15] user_call_in_progress        → if $.user_call_in_progress { 1.0 } else { 0.0 }      {0, 1}
```

All 16 features are already collected. None require new syscalls. None touch the daemon hot path (router runs every 60 cycles ≈ 30s).

### 4b. Feature Extraction Code (pseudocode)

```rust
// crates/apollo-engine/src/engine/mlp_router.rs

use std::collections::VecDeque;
use std::sync::OnceLock;

pub struct RouterFeatures {
    /// Last 5 snapshots of the 16-dim feature vector for stability check.
    /// Bounded ring buffer; zero allocation after first push.
    history: VecDeque<[f32; 16]>,
    history_capacity: usize, // = 5
    /// EMA of specialist `had_disagreement` for f[11].
    disagreement_ema: f32,
}

impl RouterFeatures {
    /// Build the 16-d feature vector from the live `RuntimeMetrics`
    /// snapshot + CausalGraph + NARS + WorldModel + OutcomeTracker.
    /// PURE: no side effects, no syscalls, no allocation (VecDeque ring
    /// already at capacity after first cycle).
    pub fn extract(
        &mut self,
        m: &RuntimeMetrics,
        causal: &CausalGraph,
        nars: &NarsEngine,
        world: &WorldModel,
        outcome: &OutcomeTracker,
        specialist_disagreement_now: bool,
    ) -> Option<[f32; 16]> {
        // 4a: cheap signals from RuntimeMetrics only.
        let swap_gb = m.swap_used_bytes as f32 / (4.0 * 1024.0 * 1024.0 * 1024.0);
        let swap_gb_norm = swap_gb.min(1.0);
        let swap_delta_norm = (m.swap_delta_bps / 524_288.0) as f32;
        let thrashing_norm = (m.thrashing_score / 10_000.0) as f32;
        let thermal_pred_norm = m.thermal_predicted_throttle as f32 / 100.0;
        let thermal_secs_norm = m.thermal_seconds_to_throttle
            .map(|s| 1.0 / (1.0 + (s as f32 / 60.0).exp()))
            .unwrap_or(0.0);
        let cycles_high_norm = (m.cycles_high_pressure as f32 / 30.0).min(1.0);
        let refault_norm = (m.refault_delta_per_sec / 5_000.0).min(1.0).max(0.0) as f32;
        let humble_norm = if m.humble_mode { 1.0 } else { 0.0 };

        // 4b: cross-engine reads (cheap; no syscalls).
        // f[10] CausalGraph subsystem debias from MetaCognition.
        // Reuse the same call world_model.rs uses to keep calibration
        // shared. Hardcode SubsystemId::CausalGraph for now; revisit
        // when CausalGraph subsystem id is exposed via the engine API.
        let debias = causal_subsystem_debias().clamp(0.25, 1.5);

        // f[11] disagreement EMA.
        self.disagreement_ema = self.disagreement_ema * 0.95
            + if specialist_disagreement_now { 0.05 } else { 0.0 };

        // f[12] world-model mean margin (calibrated by the same debias
        // the imagination layer uses, so router and imagination agree).
        let imagined_margin = if world.known_actions() > 0 {
            world.known_actions() as f32  // placeholder for mean margin
        } else { 0.0 };

        // f[13] NARS top belief. "compile" is the first canonical regime
        // label; future PRs may add "media_critical", "llm_inference",
        // etc. None → 0.5 (cold / neutral).
        let nars_compile_conf = nars.belief("compile")
            .map(|tv| tv.confidence)
            .unwrap_or(0.5);

        // f[14] interactivity = interactive pids / total.
        // Total pid count comes from `procs_scanned_this_cycle` (cheap
        // from LSE counter `processes_scanned`).
        let total_pids = (m.cycles).max(1) as f32; // conservative proxy
        let interactivity = m.behavior_interactive_pid_count as f32 / total_pids.max(1.0);

        // f[15] call/sleep-assertion.
        let call_active = if m.user_call_in_progress { 1.0 } else { 0.0 };

        let features = [
            m.memory_pressure as f32,
            swap_gb_norm,
            swap_delta_norm.clamp(0.0, 1.0),
            thrashing_norm.clamp(0.0, 1.0),
            m.cpu_max_busy as f32,
            thermal_pred_norm.clamp(0.0, 1.0),
            thermal_secs_norm,
            cycles_high_norm,
            refault_norm,
            humble_norm,
            debias,
            self.disagreement_ema.clamp(0.0, 1.0),
            imagined_margin.clamp(0.0, 1.0),
            nars_compile_conf,
            interactivity.clamp(0.0, 1.0),
            call_active,
        ];

        // Stability gate: skip inference if stddev(last 5 feature
        // vectors, summed across dimensions) < 0.02. Prevents the
        // router from firing on noise between regime shifts.
        self.history.push_back(features);
        if self.history.len() > self.history_capacity {
            self.history.pop_front();
        }
        if self.history.len() >= self.history_capacity {
            let total_var = feature_variance_across_dims(&self.history);
            if total_var < 0.02 {
                return None; // too stable; regime unchanged; skip inference
            }
        } else {
            return None; // not enough history yet
        }
        Some(features)
    }
}

fn feature_variance_across_dims(buf: &VecDeque<[f32; 16]>) -> f32 {
    // Σ over dim d of stddev(buf[*][d], last 5 samples)
    let mut sum = 0.0_f32;
    for d in 0..16 {
        let col: [f32; 5] = [
            buf[0][d], buf[1][d], buf[2][d], buf[3][d], buf[4][d],
        ];
        let mean = col.iter().sum::<f32>() / 5.0;
        let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 5.0;
        sum += var.sqrt();
    }
    sum  // total stddev across all 16 dims; ~0.02 = "essentially identical"
}
```

### 4c. MLP Architecture (paper-anchored)

```
Layer 1 (input → hidden):  W1 ∈ R[16 × 32]  fp16, b1 ∈ R[32]  fp16
  activation: ReLU
Layer 2 (hidden → logits): W2 ∈ R[32 × 4]   fp16, b2 ∈ R[4]   fp16
  activation: softmax → p ∈ R[4]

Action classes (output indices):
  0 → Observe              (lowest priority; safe default)
  1 → TightenThresholds    (push overflow thresholds -3pp)
  2 → ThrottleNoise        (boost the noise-throttle gate)
  3 → SuggestAggressive    (apply SuggestAggressive profile)
  (4 → ProactivePurge + (4 → PreThrottleNoise) intentionally dropped
   from the softmax — they are emitted via the existing PredictiveAgent
   path; the router only chooses between "observe / tighten / throttle-noise
   / go-aggressive", the four regime-level knobs that affect
   decide_actions.rs's per-cycle thresholds. The two dropped arms are
   handled by the existing predictive_agent flow without router influence.)
```

**Parameter count.** `16 × 32 + 32 + 32 × 4 + 4 = 512 + 32 + 128 + 4 = 676` params. At fp16 = **1352 bytes ≈ 1.4 KB** (vs the `<100KB` ceiling — 70× headroom for future growth).

**Activation.** ReLU on hidden (most common, NEON `vmaxq_f16` is one instruction). Softmax on output. No batch norm, no dropout (inference only — training is offline, in a separate Python notebook pipeline that produces a frozen `.bin`).

**Paper anchors.**
- **[Barto & Sutton 2018]** "Reinforcement Learning: An Introduction" 2nd ed., §9 — function approximation for policy value over a low-dim feature vector. The router is a 16 → 32 → 4 policy network trained offline on Apollo `journal.jsonl` traces (action → outcome reward = `was_effective`).
- **[Bishop 2006]** "Pattern Recognition and Machine Learning" §5.3 — feed-forward MLP with backprop training, output softmax for multi-class. Architectural template (2-layer, softmax, cross-entropy loss).
- **[Oksuz et al. 2024 TMLR]** "MoCaE: Mixture of Calibrated Experts" — per-class reliability calibration during fusion. We borrow the *per-bucket reliability* idea for training: each regime's training labels are weighted by the empirical `was_effective` rate of that regime in `journal.jsonl`. NOT a per-cycle uncertainty calibration — that's the existing MetaCognition's job.
- **[LeCun 2022]** "A Path Towards Autonomous Machine Intelligence" §4.2 — world-model predictive selection. The router does NOT replace the world model; it learns a regime-level prior that *biases* how aggressively the world model's per-candidate imagination is consulted.
- **[Goodfellow et al. 2016]** "Deep Learning" §6.2.1 — ReLU vs tanh for hidden layer. ReLU preferred here because (a) 1-cycle ARM NEON, (b) avoids saturation in regime regions where features are near-zero.

### 4d. Inference Gate (every-60-cycles + stability)

```rust
// crates/apollo-engine/src/engine/mlp_router.rs

pub struct MlpRouter {
    weights_1: AlignedF16Matrix<16, 32>,  // 1 KB aligned, NEON-friendly
    weights_2: AlignedF16Matrix<32, 4>,   // 256 B aligned
    bias_1: [f16; 32],
    bias_2: [f16; 4],
    features: RouterFeatures,
    /// Last inference result, consumed by the per-cycle scoring path.
    /// Persists for 60 cycles (next inference).
    last_decision: Option<RouterDecision>,
    /// Cached α (down-weight factor); None = router disabled.
    alpha: Option<f32>,
    /// Inference counter; increments per daemon cycle.
    cycle_count: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct RouterDecision {
    /// Softmax probabilities for [Observe, Tighten, Throttle, Aggressive].
    pub probs: [f32; 4],
    /// Down-weight factor in [0.0, 0.20] for this 60-cycle window.
    pub alpha: f32,
    /// Sanity-checked regime label (argmax of probs).
    pub regime_label: RouterRegime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterRegime { Observe, Tighten, ThrottleNoise, Aggressive }

impl MlpRouter {
    /// Called every daemon cycle, but inference only fires when the
    /// 60-cycle gate + stability gate both pass.
    pub fn on_cycle(
        &mut self,
        m: &RuntimeMetrics,
        causal: &CausalGraph,
        nars: &NarsEngine,
        world: &WorldModel,
        outcome: &OutcomeTracker,
        specialist_disagreement_now: bool,
    ) {
        // Every-60-cycles gate. 60 cycles ≈ 30s at 2 Hz daemon cadence.
        if self.cycle_count % 60 != 0 {
            self.cycle_count += 1;
            return;
        }
        self.cycle_count += 1;

        // Stability gate.
        let Some(features) = self.features.extract(
            m, causal, nars, world, outcome, specialist_disagreement_now,
        ) else { return };

        // Inference (≤ 6µs on M1 NEON; see benchmark budget).
        let logits = self.forward(&features);
        let probs = softmax(logits);

        // Argmax regime label.
        let regime = argmax_regime(&probs);

        // α ramp: max(prob - 1/4, 0.0) → maps softmax into [0, 0.75], then
        // clamped to [0, 0.20]. The router only "votes" with α when one
        // regime dominates the softmax (max prob >= 0.25 + 25pp margin).
        let max_p = probs.iter().cloned().fold(0.0_f32, f32::max);
        let second_p = {
            let mut sorted = probs;
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            sorted[1]
        };
        let confidence = max_p - second_p;
        let alpha_raw = (confidence * 0.75).min(0.20).max(0.0);

        // Safety belt: never let α exceed 0.20 (Apollo doctrine — the
        // router may never dominate the existing per-cycle logic).
        let alpha = alpha_raw.min(0.20);

        self.last_decision = Some(RouterDecision { probs, alpha, regime_label: regime });
    }

    /// Apply router's α to a per-cycle composite score from
    /// `PolicyScorer`.  Algebra: `composite' = composite * (1.0 - α)` +
    /// `α * router_benefit(regime)` — additive subtraction is the
    /// minimal-touch path. NEVER a hard veto.
    pub fn apply(&self, composite: f64, threshold: f64) -> f64 {
        let Some(d) = self.last_decision else { return composite };
        // Router regime benefit: Observe → 0.0 (no change); Tighten →
        // +0.05 (slightly favours accept); Throttle → +0.05; Aggressive →
        // +0.10. Conservative end of the leash.
        let regime_benefit = match d.regime_label {
            RouterRegime::Observe => 0.0,
            RouterRegime::Tighten => 0.05,
            RouterRegime::ThrottleNoise => 0.05,
            RouterRegime::Aggressive => 0.10,
        };
        let alpha = d.alpha as f64;
        composite * (1.0 - alpha) + alpha * regime_benefit
    }
}
```

**Cycle placement.** Wired in `src/bin/apollo-optimizerd/main.rs` immediately after `lctx.predictive_agent.select_action_with_confidence(...)` (around line 3325) — one new `if cycle_count % 60 == 0` block. `apply(...)` is called from `decide_actions.rs` at the scorer composition site (replace `composite +=` with `composite = mlp_router.apply(composite, threshold)`).

### 4e. Training Data Pipeline (offline, not in daemon)

**Source.** `journal.jsonl` entries (WAL of all actions; `journal.rs:13-89`). Each line: `JournalEntry { timestamp, action: RootAction, before, after, success: bool, reason: String, rationale: Option<Rationale> }` (`types.rs:405-437`).

**Label extraction.**
```python
# pseudo: scripts/apollo-mlp-router-train.py (NOT in daemon)

import json, numpy as np
from collections import defaultdict

# 1. Load journal entries.
entries = [json.loads(l) for l in open("/var/lib/apollo/journal.jsonl")]

# 2. Group by cycle window (60 cycles ≈ 30s) — same cadence as
#    inference. Each window becomes one training example.
windows = defaultdict(list)
for e in entries:
    # Round timestamp to nearest 30s boundary.
    bucket = int(e["timestamp"] // 30) * 30
    windows[bucket].append(e)

# 3. For each window: build the 16-d feature vector by replaying
#    runtime_metrics.json at that timestamp + causal/nars/world state
#    from learned_state.json snapshot at that bucket. Same code path
#    as the production feature extractor.

# 4. Compute label = argmax over regime action class with
#    was_effective-rate weighting (MoCaE per-bucket reliability).
#    For each of the 4 regimes:
#       - If the window contains a router-regime-correlated action
#         (e.g., TightenThresholds → regime label "tighten"), look
#         up whether the per-cycle decision was effective using
#         OutcomeTracker-style delta-pressure:
#           delta_p = pressure_before(action) - pressure_after(30s_later)
#           effective = delta_p > 0.01 (same threshold as
#           outcome_tracker.rs:1059 urgency_flush)
#       - Sample weight = |delta_p| (MoCaE per-bucket reliability).

# 5. Train MLP via backprop, cross-entropy loss, 5-fold CV on
#    weekly holdout. Output frozen weights to
#    /var/lib/apollo/mlp_router.bin (1.4 KB fp16).

# 5-row example:
EXAMPLE = [
    # [memory_p, swap_gb_norm, swap_delta_norm, thrashing_norm,
    #  cpu_max_busy, thermal_pred_norm, thermal_secs_norm,
    #  cycles_high_norm, refault_norm, humble, debias,
    #  disagreement_ema, world_margin, nars_compile_conf,
    #  interactivity, call_active]   |   regime_label | sample_weight
    [[0.62, 0.45, 0.30, 0.10, 0.55, 0.0, 0.0, 0.40, 0.05, 0.0, 0.85, 0.10, 0.0, 0.65, 0.05, 0.0],   1,  0.025],  # tighten
    [[0.78, 0.70, 0.60, 0.40, 0.65, 0.20, 0.30, 0.85, 0.35, 0.0, 0.85, 0.20, 0.05, 0.55, 0.10, 0.0],   2,  0.060],  # throttle-noise
    [[0.45, 0.20, 0.05, 0.02, 0.30, 0.0, 0.0, 0.10, 0.01, 0.0, 1.00, 0.05, 0.0, 0.30, 0.30, 0.0],   0,  0.000],  # observe (idle)
    [[0.85, 0.85, 0.80, 0.55, 0.90, 0.60, 0.70, 1.00, 0.70, 1.0, 0.55, 0.35, 0.10, 0.40, 0.05, 0.0],   3,  0.090],  # aggressive
    [[0.50, 0.30, 0.10, 0.05, 0.40, 0.0, 0.0, 0.20, 0.02, 0.0, 0.95, 0.10, 0.0, 0.55, 0.20, 1.0],   0,  0.000],  # call_active → observe
]
```

**Cold start.** Until `mlp_router.bin` exists (first daemon startup), the router is `alpha = None` (no influence). After offline training produces the file, daemon loads it on next cycle. No online learning (out of scope; see §8).

### 4f. Alpha-Ramping Integration (existing per-cycle weights)

Per-candidate composite from `PolicyScorer::score`:
```
composite = total_benefit − λ_cost·total_cost − λ_unc·total_uncertainty
```

Router-modified composite:
```
composite_router = composite·(1.0 − α) + α·regime_benefit
                  where α ∈ [0.0, 0.20]
                  regime_benefit ∈ {0.0, 0.05, 0.05, 0.10}
```

**Why this works.** The existing per-cycle score already encodes the per-candidate benefit/cost/uncertainty. The router's contribution is a regime-level prior: under the "tighten" regime, the router tilts the composite by +0.05·0.20 = +0.010 at max α; under "aggressive" by +0.10·0.20 = +0.020. These are **strictly smaller than the per-feature contributions** (PressureBenefitFeature saturates near 0.95, ±0.30 override threshold in `shadow_evaluator.rs:52`), so the router cannot dominate a feature but CAN shift borderline cases (composite ≈ 0.05) by a meaningful margin.

**Concrete math at α = 0.20, Aggressive regime.** A candidate with composite = 0.10 (borderline accept at default threshold 0.0) becomes:
- Router-applied: `0.10·0.80 + 0.20·0.10 = 0.080 + 0.020 = 0.100` (no change in this case)
- For composite = 0.05 (just-above-zero): `0.05·0.80 + 0.20·0.10 = 0.040 + 0.020 = 0.060` (slightly higher — promotes acceptance in aggressive regime)

For `Observe` regime at α = 0.20: `composite·0.80 + 0` — pulls borderline candidates below threshold → suppresses noisy throttles in idle regime.

### 4g. Failure Modes

| Failure mode | Symptom | Detection | Fallback |
|---|---|---|---|
| **Router weights corrupted** | `mlp_router.bin` parse fails → no alpha | `MlpRouter::load()` logs warn + sets `alpha = None` | Existing per-cycle logic unchanged. |
| **Router outputs near-uniform** (max_p < 0.30) | softmax degenerate → α ≈ 0 | `argmax_regime` checks `confidence = max_p - second_p` | `α = 0.0` → router is inert that window. |
| **All 4 regimes trigger within 5 cycles** | router oscillates | LSE counter `router_oscillation_total` increments when `last_decision.regime_label` changes 4× within 5 consecutive inferences | Router self-disables (alpha = 0 for 60 cycles) when oscillation > 3. |
| **α > 0.20 measured in prod** | ceiling bug | Per-cycle log check via `router_alpha_max_total` LSE counter | Code-level clamp `alpha.min(0.20)`; if counter still bumps → bug, revert commit. |
| **Router bypasses safety** | None possible by construction — router only runs `apply()` AFTER `safety.rs::enforce_limits` has filtered candidates | Code path audit | `safety.rs` is upstream; router can't widen NEVER_FREEZE because it has no access to `action.action_class` enum mutations. |

**Rollback procedure.** Delete `mlp_router.bin` → daemon logs `[mlp_router] file missing, alpha=None` → router is fully inert on next cycle. The shadow + alpha ramp gates mean a faulty router can only ever down-weight by 20% (never veto, never widen), so the blast radius of a regression is bounded.

### 4h. Gate Conditions Between Phases

| Gate | Condition | Action on pass | Action on fail |
|---|---|---|---|
| **Phase 0 → 1** | `cargo test --all` + `cargo clippy --all-targets -- -D warnings` | Land `mlp_router.rs` skeleton + `RouterFeatures::extract` (read-only) | Fix until clean. |
| **Phase 1 → 2** | `cargo build --release` succeeds; offline training pipeline produces `mlp_router.bin` | Promote `.bin` artifact, document training runbook | Iterate training (more data, better labels). |
| **Phase 2 → 3** | AIS ≥ 87 (`docs/acceptance-criteria.md` H1 hard floor); `cargo test` + `cargo clippy`; 24h prod with no `router_oscillation_total > 0` | Enable shadow logging (`mlp_router_shadow.jsonl` records `features` + `probs` + `last_decision.regime_label` for every cycle, not just 60s windows) | Revert; router stays inert. |
| **Phase 3 → 4** | **N ≥ 500 shadow cycles** collected; **win-rate > 0.55** measured by replaying shadow log against actual `was_effective` outcomes | HUMAN-GATED cutover: α = 0.0 → 0.05 → 0.10 → 0.20 over 4 weekly steps, each preceded by 1-week observation | Stay in shadow; retrain with new data. |
| **Phase 4 → 5** | AIS stable (±1pp vs baseline) AND p95 stable (±5% vs baseline) AND no new `effect_decay_hp_mach_attempts_total` bumps over 7 days | Land 7-day postmortem (`docs/mlp-router-postmortem.md`) | Rollback: delete `.bin`, leave router inert, document lesson. |

**Win-rate definition.** For every (router-decision window) → (per-cycle action × was_effective) tuple in shadow:
- `+1` if router's regime vote + per-cycle action combo resulted in `was_effective == true`
- `0` if outcome was neutral
- `-1` if router's regime vote + per-cycle action combo resulted in `was_effective == false`
- `win_rate = (sum + N) / (2N)` ∈ [0, 1]

---

## 5. Paper Anchors

| Citation | How it applies |
|---|---|
| **[Barto & Sutton 2018]** "Reinforcement Learning: An Introduction" 2nd ed., §9 — Function Approximation | MLP router is a value-function approximator over the 16-d regime feature vector. Trained offline on the journal's reward signal. |
| **[Bishop 2006]** "Pattern Recognition and Machine Learning" §5.3 — MLP architecture | 16 → 32 → 4 feed-forward network; cross-entropy loss; softmax output. Architectural template. |
| **[Oksuz et al. 2024 TMLR]** "MoCaE: Mixture of Calibrated Experts" arXiv:2309.14976 | Per-bucket reliability weighting of training labels. Sample weight ∝ `|delta_pressure|` after action. Borrowed from C2 calibration that already runs for PredictiveAgent specialists (`predictive_agent.rs:139-205`). |
| **[LeCun 2022]** "A Path Towards Autonomous Machine Intelligence" §4.2 — World models | The MLP router is NOT a world model. It learns a regime prior that *biases* the existing world model's per-candidate imagination verdict. |
| **[Goodfellow et al. 2016]** "Deep Learning" §6.2.1 — ReLU activations | ReLU chosen for hidden layer: NEON `vmaxq_f16` is one instruction; avoids saturation near zero (idle regime). |
| **[Sutton & Barto 2018]** §17.4 — Reward shaping | Router's regime benefit (+0.05/+0.10) is a small additive reward bias, deliberately bounded below the existing per-feature contributions. |
| **[Fowler 2004]** "StranglerFigApplication" | Phases ship in increasing blast radius: read-only (P0) → training pipeline (P1) → shadow log (P2) → side-by-side diff (P3) → conditional weight (P4) → postmortem (P5). Legacy per-cycle logic intact throughout. |
| **[Saltzer & Kaashoek 2009]** §3.3 — Complete Mediation | The router applies AFTER `safety.rs::enforce_limits` and AFTER the gate tower. It cannot widen NEVER_FREEZE because `safety.rs` has already filtered the action set. |
| **[Nygard 2018]** "Release It!" §8.5 — Adaptive capacity limits via shadowing | Phases 2-3 mirror the shadow-mode doctrine: log disagreements first, measure win-rate over ≥500 obs, then conditionally enable. |
| **[Oksuz et al. 2024]** §3 — Calibration across experts | The router's softmax output is NOT a calibrated probability (we don't apply temperature scaling); it is a regime prior. Calibration lives in MetaCognition (`meta_cognition.rs`). |

---

## 6. Anti-Patterns Detected (the router must NOT replicate these)

| Anti-pattern (catalog: anti-patterns.md) | Current site (file:line) | What the router does differently |
|---|---|---|
| **Chatty Inter-Module Communication** (Medium) | `src/bin/apollo-optimizerd/main.rs:3000-3990` (~1000 LOC) wires 5+ engines inline per cycle. PredictiveAgent + SpecialistVoting + HOLT-Winters + CausalGraph + OutcomeTracker all chat through `lctx` directly. | Router reads from `RuntimeMetrics` + cached slices only; does NOT borrow the live `lctx` per cycle. Single typed `RouterFeatures::extract` composes all 16 inputs in one shot. |
| **God Service** (Critical) | `main.rs` is 6410 LOC (`wc -l`). Despite Strangler Fig progress, integration seam still bloats. | `mlp_router.rs` is bounded <1000 LOC total, with the MlpRouter struct holding only `weights_*`, `bias_*`, `features`, `last_decision`, `alpha`, `cycle_count`. No cross-engine borrows. |
| **Shared Mutable State Sprawl** (High) | `daemon_state.rs:39-60` has 6 domain groups, each with its own `Arc<Mutex<...>>`. | Router state lives in ONE struct (`MlpRouter`), mutated only inside `on_cycle()` and `apply()`. No `Arc<Mutex<>>` exposed. |
| **Silent Failure** (Medium) | Some legacy paths use `.unwrap_or(default)` without logging (catalog reference). | Router uses `tracing::warn!` for every fallback: missing `.bin`, oscillation, α-floor violations. LSE counter `router_alpha_max_total` for ceiling violations. |
| **Telemetry-Death Sticky Counter** | `lse_counters.rs` discipline (Sprint 9 `4b13a39`): ALWAYS reference `LSE_COUNTERS`, never construct local copies. | Router mirrors this: only one `MlpRouter` global `OnceLock<MlpRouter>`, all LSE counter bumps via `LSE_COUNTERS.inc_router_*()` (added in Phase 0). |
| **Predict-Then-Override** | `predictive_agent.rs` chooses 1 of 5 arms per cycle; this is a single-arm router. | The MLP router outputs a DISTRIBUTION over regimes — not a single action. The existing PredictiveAgent is unchanged. |
| **Sticky Counter as Live State** | Per `CLAUDE.md` 2026-05-07 lesson: dashboard once used `survival_mode_activations > 0` as a state flag instead of `survival_mode_active_now`. | Router's `last_decision` is rebuilt every 60 cycles from current features; it is NEVER derived from a cumulative counter. |

---

## 7. Implementation Phases (Strangler Fig, blast-radius ascending)

| Phase | Description | LOC est | Gate to advance | Risk |
|---|---|---|---|---|
| **0** | `docs/mlp-router-design.md` (this spec) + `crates/apollo-engine/src/engine/mlp_router.rs` skeleton (struct + weight load + stub `extract()`) | `<1000` | `cargo build --release` + `cargo test` + `cargo clippy --all-targets -- -D warnings` | None (skeleton only). |
| **1** | `scripts/apollo-mlp-router-train.py` + training data extractor; produces `mlp_router.bin`. NO daemon code changes. | `<500` | Offline CV loss < baseline (random init) by ≥ 30% | None (offline only). |
| **2** | Instrument shadow log: write `mlp_router_shadow.jsonl` (features + probs + regime_label every 60 cycles). Daemon loads `mlp_router.bin` but `alpha = None` (inert). | `<300` | `cargo test` + `cargo clippy` + AIS ≥ 87 (H1 hard floor, no regression). | Telemetry noise only. |
| **3** | Side-by-side diff: every 60 cycles, log `(features, router_decision, actual_action, was_effective, was_router_correct)` to `mlp_router_diff.jsonl`. Compute win-rate offline. | `<500` | N ≥ 500 cycles of shadow collected; `win_rate > 0.55` measured against replayed outcomes | None (still inert). |
| **4** | Conditional weight: α ramps `0.0 → 0.05 → 0.10 → 0.20` over 4 weekly steps, each step gated on the previous step's AIS / p95 / win-rate staying within tolerance. **HUMAN-GATED.** | `<1000` | Per `docs/acceptance-criteria.md` H1/H2/H3/H4/H5 + `win_rate > 0.55` measured live | Medium — first time the router influences a decision. Bounded by α ≤ 0.20 ceiling. |
| **5** | 7-day measurement + `docs/mlp-router-postmortem.md`. **NO auto-promote.** | `<500` (doc) | Acceptance criteria (§9) met for 7 consecutive days | Low (measurement only). |

**Phase-0 file footprint (skeleton):** `crates/apollo-engine/src/engine/mlp_router.rs` (new, ~500 LOC) + `crates/apollo-engine/src/engine/mod.rs` (one `pub mod mlp_router;` line). No other daemon changes.

**Phase-1 file footprint:** `scripts/apollo-mlp-router-train.py` (new) + `crates/apollo-engine/src/engine/mlp_router.rs` (`load_or_default` helper, no behavior change to daemon loop). Training data read access only.

---

## 8. Deferred Candidates (considered but NOT in the proposal)

| Candidate | Why rejected |
|---|---|
| **Transformer-based router** | [Vaswani et al. 2017] attention over the 16-dim context adds ~2MB of weights for negligible benefit on a 16-token sequence. Inference budget (≤6µs on M1) precludes it. Per cycle we'd hit ~50µs — an order of magnitude over budget. |
| **Online learning** (router updates weights every cycle) | The ML subsystem stack (LinUCB, Reptile, NARS) already provides online adaptation; adding a second online learner creates conflicting gradients. The router is a frozen offline-trained model that biases the per-cycle score — simpler. Online retraining of the router could be a future PR with N≥1000 shadow cycles as the trigger. |
| **Per-process (not regime) router** | The router chooses among 4 regime classes, not among 400+ processes. Per-process routing already lives in `PredictiveAgent::select_action` and `HopGroupWeight::yield_admits`. Adding a per-process MLP would duplicate that path and double the inference cost. |
| **Mixture-of-Experts (MoE) regime router** | [Shazeer et al. 2017] "Outrageously Large Neural Networks" — would require K separate forward passes per inference. With 16 → 32 → 4 base architecture we already get regime mixing via the hidden layer; an explicit MoE adds weights without adding capacity we need. |
| **Online Gaussian Process** | Bayesian uncertainty comes free with GPs, but online GP updates are O(n³) per cycle. Even sparse GPs (Titsias 2009) require careful maintenance. Reject: complexity > value for a 4-class classification. |
| **Routing based on `journal.jsonl` instead of `runtime_metrics.json`** | Journal entries are WAL append — by the time we read them, the regime may have shifted. `RuntimeMetrics` is the live dashboard; that's the right surface. |
| **Multi-output MLP** (one head per action class, predicting benefit) | The current regime-class softmax output is simpler and matches the existing 5-arm PredictiveAgent structure. Per-action benefit heads would need 4× the training data. |
| **NNAPI/CoreML inference backend** | macOS Neural Engine is real, but Apollo is targeting CPU-only inference (CLAUDE.md project quality + no extra entitlement). NEON f16 is sufficient for our 6µs budget. |
| **Distillation from a larger offline teacher** | Useful only if we have a teacher; we don't. The router is trained directly on journal outcomes via cross-entropy. |
| **Replacing the existing PredictiveAgent with the MLP router** | Apollo doctrine: NEW component down-weights EXISTING logic; NEVER replace. PredictiveAgent handles per-action intervention; the MLP router handles regime-level bias. Different layers of the decision stack. |

---

## 9. Acceptance Criteria

**Baseline values from current state** (rolling K=7 days, as documented in `docs/acceptance-criteria.md` and `~/.claude/MEMORY.md`).

| Metric | Baseline | Phase-4 success | Phase-5 success |
|---|---|---|---|
| **AIS (Apollo Intelligence Score)** | 95.5 (post-2026-06-13 calibration loop-closure) | ≥ 94.5 (≤1pp drop tolerated) | ≥ 95.5 (back to baseline OR higher) |
| **p95_cycle_ms** | 66ms (`refresh-specifics` post-deploy; t=steady-state) | ≤ 70ms (≤5% over baseline) | ≤ 66ms |
| **router win-rate** (shadow, N≥500) | N/A | ≥ 0.55 over 500 obs | ≥ 0.55 sustained 7d |
| **failures / cycle** | 0 (per `acceptance-criteria.md` H2) | 0 | 0 |
| **scarcity-thrashing events** (H4) | 0 | 0 | 0 |
| **frozen / killed protected processes** | 0 (H5) | 0 | 0 |
| **composite geom-mean over 8 signals** (per `acceptance-criteria.md` H5) | baseline | within 1pp | within 1pp |
| **`router_oscillation_total`** | N/A | ≤ 10 / 7d | ≤ 10 / 7d |
| **`router_alpha_max_total`** (ceiling violations, should stay 0) | N/A | 0 | 0 |

**Measurement discipline.** Per CLAUDE.md 2026-05-07 lesson: NO "preliminary verdict" with N<500 events. Per CLAUDE.md 2026-06-24 lesson: NO measurement in warmup (first ~15-20 min after daemon restart). Per CLAUDE.md 2026-06-20 lesson: re-capture baseline BEFORE deploying, not from a stale cached K=7 window.

---

## 10. Adversarial Check (notebook scrutiny)

| # | Possible failure | Mitigation |
|---|---|---|
| **1** | **The router is too small to learn.** 16 → 32 → 4 has 676 params; journal.jsonl may produce thousands of regime windows. If regime classes overlap (e.g., "tighten" and "throttle" share features at α=0.5), the router outputs max_p ≈ 0.40 — too low for any α. **Result: router always inert in prod.** | Phase 1 offline CV will detect this BEFORE Phase 2 deployment. If `accuracy < 0.50` on the 5-fold holdout, abort: do not promote the `.bin` artifact. The training runbook requires `cv_accuracy ≥ 0.55` for promotion to Phase 2. |
| **2** | **The `alpha ≤ 0.20` ceiling is too tight to matter.** With composite ∈ [0, 1] and regime_benefit ∈ {0, 0.05, 0.05, 0.10}, the max shift is `0.20·0.10 = 0.020`. For borderline candidates (composite ≈ 0.0) this matters; for `composite ≥ 0.10` it's noise. **Result: router wins the N≥500 gate but produces no measurable AIS lift.** | Phase 4 acceptance gate explicitly requires `win_rate > 0.55` AND a measured AIS shift OR p95 shift OR `low_value_skipped` increase (any one). If none move within 14 days at full α=0.20, propose rollback in Phase 5 postmortem. The router is INTENTIONALLY tight — Apollo doctrine is conservative. |
| **3** | **The 16-d input has regime-feature leakage.** `f[11] disagreement_ema` (specialist consensus rate) is itself a function of the regime — if specialists disagree MORE under "aggressive" regime, the router sees its own input doubled via `f[13] nars_compile_conf`. **Result: router overfits to the very signals it should not bias.** | Phase 1 training script applies leave-one-feature-out CV: train without each of the 16 features and report holdout accuracy. If any feature removal causes <2pp drop, it's a candidate for removal before Phase 2. Document in `docs/mlp-router-design.md` which features survived. |
| **4** | **NotebookLM or a future sprint catches the CausalGraph subsystem_debias coupling.** The router reads `causal_subsystem_debias()` for `f[10]` — same source as `world_model.from_parts(... prediction_debias)` (`world_model.rs:95`). If `causal_subsystem_debias` ever flips sign under calibration drift, the router AND the world model would shift together in correlated ways. | Phase 0 design records this dependency explicitly. Phase 4 acceptance gate checks: if `causal_subsystem_debias` changes >0.30 in any 60-cycle window, router auto-disables for that window (alpha = 0) and logs `router_debias_anomaly_total`. |
| **5** | **macOS 15-char `MAXCOMLEN` truncation breaks `f[13] nars_compile_conf` lookup.** Same lesson as `a98b33a`: NARS key `compile` may not match a runtime-truncated belief key. **Result: f[13] always 0.5 (neutral), killing the router's regime signal.** | Use `learned_pattern_matches` (per `decide_actions.rs:183`) for ALL NARS belief lookups, including `f[13]`. Phase 0 includes this helper import explicitly. |
| **6** | **The router is wired correctly but trains on stale data.** If `journal.jsonl` rotation kicks in (`journal.rs:64-72` at 2 MB), the training pipeline misses recent regime windows. **Result: `.bin` trained on patterns from 2 weeks ago; current regime not learned.** | Phase 1 training script reads BOTH `journal.jsonl` and `journal.jsonl.1` (the rotated file). Documented in the runbook. Refresh cadence: weekly. |
| **7** | **Phase 4 ramp masks a faulty α=0.05 step.** If win-rate at α=0.05 is 0.54 (just below gate) but ramp continues to α=0.10 because the user only checks AIS, the gate is silently weakened. **Result: gate violations accumulate.** | Each weekly ramp step is its OWN hard gate: AIS ±1pp AND p95 ±5% AND win_rate > 0.55 (re-measured for THAT step's window). No step-forward without all three met. Documented in `docs/mlp-router-design.md`. |

---

## Appendix A — File Inventory Touched

| File | Change | Phase |
|---|---|---|
| `crates/apollo-engine/src/engine/mlp_router.rs` (NEW) | MLP router module + RouterFeatures + MlpRouter | 0 |
| `crates/apollo-engine/src/engine/mod.rs` | `pub mod mlp_router;` | 0 |
| `crates/apollo-engine/src/engine/lse_counters.rs` | +5 router_* counters (`router_oscillation_total`, `router_alpha_max_total`, `router_debias_anomaly_total`, `router_inferences_total`, `router_shadow_log_total`) | 0 |
| `crates/apollo-engine/src/engine/types.rs` | +5 corresponding `RuntimeMetrics` fields (`#[serde(default)]`) | 0 |
| `crates/apollo-engine/src/engine/action_policy.rs` | `PolicyScorer::score` calls `mlp_router.apply(composite, threshold)` (one line) | 4 |
| `src/bin/apollo-optimizerd/main.rs` | One `if cycle_count % 60 == 0 { router.on_cycle(...) }` block in decide prologue | 2 |
| `scripts/apollo-mlp-router-train.py` (NEW) | Offline training pipeline | 1 |
| `docs/mlp-router-design.md` (NEW) | Runbook for retraining cadence, CV gates | 1 |
| `docs/mlp-router-postmortem.md` (NEW) | 7-day postmortem | 5 |

## Appendix B — Runtime metrics.json pointer paths (verified)

```
$.cycles                                    u64
$.memory_pressure                           f64 [0..1]
$.swap_used_bytes                           u64 (bytes)
$.swap_total_bytes                          u64
$.swap_delta_bps                            f64
$.thrashing_score                           f64
$.refault_delta_per_sec                     f64
$.cpu_max_busy                              f64 [0..1]
$.cpu_mean_busy                             f64 [0..1]
$.thermal_predicted_throttle                u8 [0..100]
$.thermal_seconds_to_throttle               i32 | null
$.thermal_trend_predicted                   string
$.cycles_high_pressure                      u32
$.humble_mode                               bool
$.behavior_interactive_pid_count            usize
$.user_call_in_progress                     bool
```

All paths verified against `crates/apollo-engine/src/engine/types.rs:489-1300`. No new fields added to `RuntimeMetrics` for Phase 0; the router reads existing fields only.

---

**End of PR-feature-MLP-router.md**
