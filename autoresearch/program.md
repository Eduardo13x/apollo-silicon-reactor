# Apollo AutoResearch — Program

> Inspired by Karpathy's AutoResearch (2026).
> "The human's job shifts from writing code to writing research directions."

## The Contract

You are an autonomous research agent improving Apollo, a macOS system optimizer.

### Rules

1. **Modify only `src/`** — engine modules, daemon, CLI. Never touch `autoresearch/evaluate.sh`.
2. **One experiment per iteration** — a single focused change, not a sprawling refactor.
3. **Run `./autoresearch/evaluate.sh`** after every change. The output is your ground truth.
4. **PASS=1 is mandatory** — if PASS=0, revert immediately (`git checkout -- .`). No exceptions.
5. **SCORE must improve** — if SCORE ≤ previous best, revert. Equal score is only kept if the change simplifies code (fewer lines for same score).
6. **Log everything** in `autoresearch/results.tsv` — kept, discarded, AND crashed experiments.
7. **Commit kept experiments** with a descriptive message. Discarded experiments leave no git trace.
8. **Never stop** — if you run out of ideas, re-read the code, re-read results.tsv for patterns, try combining near-misses, try the opposite of what failed.
9. **Simplicity bias** — "A 0.001 improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome."

### The Metric

```
score = tests_passed
      - clippy_warnings * 5
      - max(0, binary_bloat_kb) * 0.01
      + new_tests * 0.5
```

**Lower clippy = better. More tests = better. Smaller binary = better. All tests must pass.**

### Time Budget

Each experiment should complete evaluation in < 3 minutes (build + test + clippy). If an experiment requires architectural changes that take longer to validate, break it into smaller steps.

## Research Directions

### Tier 0: Research-Driven Optimization (make Apollo smarter)

These directions improve the daemon's actual optimization intelligence — better decisions, fewer false positives, more responsive. Each is grounded in published research. **New tests that validate improved behavior count toward score.**

#### 0A. Compression-Aware Freeze Scoring
**Paper**: ZipNN (arXiv:2411.05239) — entropy classification predicts compressibility.
**Gap**: `decide_actions()` freezes only 4 hardcoded apps at pressure > 0.90. But `compressor_aware.rs` already has `scan_regions()` + compression ratio data that's **unused in freeze decisions**.
**Experiment**: Score each freeze candidate by `(rss_bytes * compression_ratio) / recency_weight`. Freeze the top-N by score instead of matching names. Processes with high compression ratio are cheap to freeze (pages already compressed); low ratio = encrypted/media = expensive (causes swap I/O on thaw).
**Validate**: Test that a process with 3:1 compression scores higher than one with 1.1:1. Test that the top-N ranking changes with different RSS values.

#### 0B. ProcessTree Cascade (Group Decisions)
**Paper**: cgroups v2 design (kernel.org) — hierarchical resource control.
**Gap**: `process_tree.rs` builds parent→child relationships but `decide_actions()` treats every process independently. Chrome spawns 20+ helpers; throttling the parent without children is pointless.
**Experiment**: When deciding to throttle/boost a process, cascade the action to its children via `ProcessTree`. When freezing a parent, freeze all children. When unfreezing, unfreeze the group.
**Validate**: Test that throttling "Chrome" also throttles "Chrome Helper (Renderer)". Test that orphan processes (no parent in tree) are unaffected.

#### 0C. Adaptive Pressure Thresholds via Shaped RL Reward
**Paper**: Potential-Based Reward Shaping (Ng et al. 1999, ICML) — provably preserves optimal policy while accelerating learning.
**Gap**: `rl_threshold.rs` uses sparse reward: +1 per stable tick, ±10 on overflow. A throttle that reduces pressure by 15% gets the same +1 as doing nothing. The RL agent can't distinguish "good throttle" from "lucky idle."
**Experiment**: Replace sparse reward with shaped reward: `R = Φ(s') - Φ(s)` where `Φ(pressure) = -pressure². This gives continuous feedback: any pressure reduction yields positive reward proportional to magnitude. Feed `OutcomeTracker.pressure_drop` as the shaping signal.
**Validate**: Test that reducing pressure from 0.80→0.60 yields higher reward than 0.80→0.75. Test that shaped reward + baseline reward sum correctly.

#### 0D. Online Bayesian Process Classification
**Paper**: Dirichlet-Multinomial (Murphy 2012, "Machine Learning: A Probabilistic Perspective") — conjugate prior for categorical data with online updates.
**Gap**: Process classification uses hardcoded lists (22 apps) + one-shot LLM patterns. New apps (Obsidian, Cursor, Warp) are unknown until manually added. No learning from outcomes.
**Experiment**: Maintain a per-process-name Dirichlet prior over categories {interactive, noise, background, system}. Initialize from hardcoded lists (strong prior α=10). Update posterior each cycle from behavioral signals (CPU pattern, window ownership, wakeups/sec). After 20 observations, posterior dominates prior — new apps get classified automatically.
**Validate**: Test that an unknown process with high CPU + GUI window converges to "interactive" within 20 cycles. Test that a known "noise" process stays classified despite occasional CPU spikes.

#### 0E. Cost-Benefit Action Selection (Harm Budget)
**Paper**: Constrained MDPs (Altman 1999) — optimize reward subject to cumulative cost constraints.
**Gap**: No cross-cycle harm tracking. The daemon can emit 50 freezes per minute with no penalty. Freeze has ~200ms UI stutter on thaw; throttle has ~1ms overhead. No cost weighting.
**Experiment**: Assign UI-cost per action: freeze=10, throttle=1, sysctl=0.5, boost=0. Track cumulative cost over a 60s sliding window. Cap at 30 points/minute. When budget exhausted, only allow zero-cost actions (boost, monitoring). Prefer throttle over freeze when both achieve similar pressure reduction.
**Validate**: Test that 3 freezes (30pts) exhausts the budget. Test that throttles are preferred when freeze budget is low. Test that budget resets after 60s window slides.

#### 0F. Learned Ensemble Weights (Specialist Voting)
**Paper**: Stacking / Super Learner (van der Laan et al. 2007, Statistical Applications in Genetics and Molecular Biology) — optimal convex combination of base learners.
**Gap**: Specialist voting weights are hardcoded (`hazard * 1.0`, `monopoly * 0.7`, `entropy * 0.5`). Why 0.7? No one knows. Different pressure regimes need different specialist weights.
**Experiment**: Track each specialist's prediction accuracy against actual outcomes (did pressure spike? did throttle help?). Maintain per-specialist EMA accuracy. Use accuracy as voting weight: `vote_weight = ema_accuracy / sum(all_accuracies)`. Specialists that consistently predict well get amplified; unreliable ones fade.
**Validate**: Test that a specialist with 90% accuracy gets ~3× the weight of one with 30%. Test that weights sum to 1.0. Test that a newly-added specialist starts at uniform weight.

#### 0G. Adaptive Unfreeze (Pressure-Driven Thaw)
**Paper**: MEMTIS (SOSP'23) — inflection-point tiering for memory management.
**Gap**: Frozen processes stay frozen for fixed 10-minute TTL regardless of pressure. If pressure drops to 0.20 after 30 seconds, user waits 9.5 minutes for no reason.
**Experiment**: Replace fixed TTL with pressure-driven unfreeze: `unfreeze_when(pressure < freeze_pressure * 0.6 && time_frozen > 30s)`. Unfreeze in priority order (most recently used first). Keep a minimum freeze duration (30s) to avoid thrashing.
**Validate**: Test that pressure drop from 0.90→0.40 triggers unfreeze after 30s. Test that pressure staying at 0.85 keeps processes frozen. Test that rapid freeze/unfreeze cycling is prevented by 30s minimum.

#### 0H. Device-Aware Baseline Calibration
**Paper**: DAMON (arXiv:2303.05919) — adaptive monitoring regions based on hardware characteristics.
**Gap**: Pressure thresholds (0.58/0.78) are identical on M1 Air (8GB, fanless, throttles at 95°C) and M1 Max (64GB, fans, throttles at 105°C). An M1 Air at 0.55 pressure is closer to crisis than an M1 Max at 0.70.
**Experiment**: At daemon startup, query `hw.memsize` and `hw.ncpu` via sysctl. Compute `memory_headroom = total_ram / 8GB`. Shift thresholds: `adjusted = baseline - (1.0 - memory_headroom) * 0.10`. On 8GB: thresholds drop by 0.10 (more aggressive). On 32GB: thresholds stay or rise (more relaxed).
**Validate**: Test that 8GB device gets lower thresholds than 32GB. Test that threshold shift is bounded (max ±0.15). Test that runtime behavior changes with different headroom values.

---

#### 0I. Predictive Coding — Error-Only Processing
**Paper**: Salvatori et al. (arXiv:2308.07870, 2023) — "Brain-Inspired Computational Intelligence via Predictive Coding." Foundational: Rao & Ballard 1999.
**Insight**: The brain doesn't process every raw sensory input. It maintains a generative model and only propagates *prediction errors* — the delta between expected and actual. Stable states consume zero cognitive bandwidth.
**Gap**: Apollo runs the full signal pipeline (Kalman, CUSUM, entropy, DAMON, causal graph) every 5s cycle regardless of stability. During 80% of cycles (low pressure, stable workload), most computation is wasted.
**Experiment**: Maintain a `PredictedState` with EMA-predicted values for top 5 metrics (pressure, swap_velocity, cpu_total, entropy, dominant_share). Each cycle compute `residual = max(abs(actual - predicted) / max(predicted, 0.01))`. If max residual < 0.05 for 3 consecutive cycles, skip heavy subsystems (DAMON estimator, causal graph, deep scan). Reset skip counter on any residual spike. This is the router's pressure-zone logic extended with a *stability dimension*.
**Validate**: Test that stable state (residual < 0.05 × 3) triggers skip mode. Test that a sudden pressure spike immediately exits skip mode. Test that skipped cycles produce same final actions as non-skipped when state is truly stable.

#### 0J. Allostatic Regulation — Anticipatory Resource Prep
**Paper**: Sterling, Physiology & Behavior, 2012 — "Allostasis: A model of predictive regulation." Computational extension: arXiv:2503.16085 (2025).
**Insight**: Unlike homeostasis (react to deviations), allostasis *anticipates* needs and pre-positions resources before demand arrives. The body raises cortisol before waking, not after stress hits.
**Gap**: Apollo reacts to pressure crossing thresholds. The Markov predictor estimates P(high_pressure | current_state) but this prediction is never used to *prepare* actions in advance. By the time pressure spikes, there's a 5-10s decision latency.
**Experiment**: When `PredictiveAgent` estimates P(high_pressure) > 0.60, pre-compute a "ready list" of freeze-eligible processes (run classification + compression scoring but DON'T act). Cache the list. When pressure actually crosses threshold, use the cached list instead of computing fresh. This reduces time-to-first-action from ~200ms to ~5ms during pressure spikes.
**Validate**: Test that ready list is built when P > 0.60 and empty when P < 0.30. Test that cached list is consumed (not recomputed) on actual threshold crossing. Test that stale cached lists (> 30s old) are discarded.

#### 0K. Cerebellar Forward Model — Predict Before Acting
**Paper**: Bhatt et al., Nature Reviews Neuroscience, 2025 — "Cerebellar circuit computations for predictive motor control."
**Insight**: The cerebellum predicts the sensory consequences of an action *before* executing it. If the predicted outcome is bad, the action is aborted without waiting for real feedback. This enables fast correction at near-zero cost.
**Gap**: Apollo freezes/throttles and then waits 30s to observe the outcome via OutcomeTracker. No pre-commit check: "will this action actually help?" Many freezes are no-ops (process had small RSS, or was already mostly compressed).
**Experiment**: Build `forward_model(action, process, pressure) -> PredictedOutcome` using OutcomeTracker's experience memory. For each candidate freeze: lookup similar past experiences (`query_similar(process, pressure)`). If predicted `pressure_drop < 0.01` OR predicted user-impact > threshold, skip the action entirely. Only execute actions with positive predicted benefit.
**Validate**: Test that a process with 5 past freezes and 0 pressure drops gets blocked. Test that a process with no history passes through (optimism under uncertainty). Test that forward model updates after real outcomes arrive.

#### 0L. Dopaminergic RPE — Surprise-Scaled Learning Rate
**Paper**: Bhatt et al., Nature Communications, 2024 — "Learning to express reward prediction error-like dopaminergic activity." Babayan et al., PMC, 2024.
**Insight**: Dopamine fires proportionally to *surprise* (reward prediction error). Expected rewards produce no dopamine → no learning needed. Unexpected outcomes trigger large learning updates. The RPE signal is decoupled from the base learning rate, allowing separate modulation of *what to learn* vs *how fast*.
**Gap**: `rl_threshold.rs` decays alpha on a fixed schedule (0.20→0.02 over 200 ticks). But a workload shift (user switches from coding to video editing) should trigger a learning rate spike — the old Q-values are suddenly stale. Currently, the RL agent adapts slowly because alpha is time-based, not surprise-based.
**Experiment**: Replace `alpha = max(0.02, 0.20 / (1 + ticks/200))` with `alpha = base_alpha * clamp(abs(RPE) / rpe_ema, 0.5, 5.0)` where `base_alpha = 0.05`, `RPE = actual_reward - Q[s,a]`, and `rpe_ema` is a running average of |RPE|. When an action produces a surprising result (RPE >> rpe_ema), alpha spikes temporarily. Steady state: alpha ≈ base_alpha. Shock: alpha → 5× base for rapid adaptation.
**Validate**: Test that steady-state alpha stays near base_alpha. Test that a large RPE spike (10× average) causes alpha to hit 5× ceiling. Test that after shock, alpha decays back to base within 20 ticks.

#### 0M. Default Mode Network — Productive Idle Cycles
**Paper**: Buckner & DiNicola, Neuron, 2023 — "20 years of the default mode network: A review and synthesis."
**Insight**: The brain's DMN activates during idle time but is *not truly idle* — it consolidates memories, plans futures, and extracts patterns. It's the brain doing maintenance and learning when no external task demands attention.
**Gap**: When pressure < 0.20, the router skips heavy subsystems. These idle cycles are wasted. The RL agent doesn't learn, experience memory isn't consolidated, and workload fingerprints aren't pre-computed.
**Experiment**: In the router's idle zone (pressure < 0.20), instead of skipping everything, run a `consolidate()` function: (1) replay last 50 experiences through RL Q-update (offline learning), (2) prune experience memory entries older than 24h with effectiveness < 0.30, (3) pre-classify memory layout of top-5 RSS processes (cache for future freeze decisions). This turns dead cycles into learning cycles.
**Validate**: Test that consolidation only runs in idle zone. Test that experience replay updates Q-values (before ≠ after). Test that pruning reduces memory entries. Test that pre-classified processes have cached RegionSummary.

#### 0N. Global Workspace Theory — Broadcast Attention
**Paper**: Baars 1988; computational: arXiv:2505.13969 (2025) — "Hypothesis on the Functional Advantages of the Selection-Broadcast Cycle Structure."
**Insight**: The brain's specialized modules (vision, language, motor) compete for access to a limited-capacity "global workspace." The winner gets broadcast to ALL modules simultaneously, enabling system-wide coordination without O(n²) point-to-point wiring.
**Gap**: Apollo's subsystems are wired with specific cables (A: OutcomeTracker→RL, B: OutcomeTracker→PredictiveAgent, C: Markov→QoS). Adding a new subsystem requires N new cables. The `SignalDigest` is a flat struct, not a prioritized workspace.
**Experiment**: Add a `workspace_signal: Option<(f64, WorkspaceEntry)>` to SignalDigest. Each subsystem submits its most urgent finding with a priority score. After all subsystems run, the highest-priority entry becomes the broadcast. All subsystems receive it next cycle as input context. Replace point-to-point cables with broadcast consumption. WorkspaceEntry: enum {PressureSpike(f64), AnomalyDetected(score), OverflowImminent(eta_s), WorkloadShift(from, to)}.
**Validate**: Test that highest-priority signal wins the workspace. Test that all subsystems can read the broadcast. Test that a PressureSpike broadcast causes RL to shift to defensive action.

---

#### 0O. HARP — IPC-Aware P-core/E-core Routing
**Paper**: Khasanov et al., ACM Middleware 2025 (Best Paper Honorable Mention) — "HARP: Energy-Aware and Adaptive Management of Heterogeneous Processors." 12% faster, 28% less energy.
**Insight**: On big.LITTLE, the resource manager should learn which processes benefit from P-cores (compute-bound, high IPC) vs those equally served by E-cores (I/O-bound, memory-bound). Reactive tier assignment wastes P-core cycles on I/O-wait processes.
**Gap**: `mach_qos.rs` assigns tiers reactively based on current thread state. No learning of "this process historically benefits from P-cores" or "this process is I/O-bound and E-core is fine." `analyze_threads()` detects patterns but doesn't persist them.
**Experiment**: Track per-process `p_core_benefit` EMA: after boosting to P-core tier, measure CPU efficiency gain (IPC proxy via `ri_instructions/ri_cycles` from proc_taskinfo). If efficiency gain < 10% on P-core vs E-core, mark process as "E-core-optimal" and stop wasting P-core slots on it. Maintain a HashMap<String, f64> of learned affinities.
**Validate**: Test that an I/O-bound process (low IPC) converges to E-core preference. Test that a compute-bound process (high IPC) stays on P-core. Test that affinity persists across cycles.

#### 0P. DVFS Efficiency Curves — Power-Proportional Throttling
**Paper**: Hunold et al., arXiv:2502.05317 (2025) — "Apple vs. Oranges: Evaluating Apple Silicon M-Series for HPC." Thomas Kaiser DVFS exploration (2024).
**Insight**: M1 Firestorm at max frequency uses 33% more power for only 17% more throughput. The efficiency curve is strongly non-linear — the last 20% of frequency costs 50% more energy. E-cores at full speed are 3-4× more power-efficient per instruction than P-cores at max.
**Gap**: Apollo's throttling is binary: SIGSTOP (100% throttle) or QoS tier change. No awareness that moving a process from P-core to E-core saves 3× power while retaining 60% throughput. For medium-priority processes, E-core migration is strictly better than stop/start cycling.
**Experiment**: For processes in the "throttle" tier (not freeze-worthy), replace SIGSTOP cycling with QoS_CLASS_UTILITY migration (E-core preference). Track power savings via energy_pid `ri_pkg_energy` delta. Only SIGSTOP processes that are both non-essential AND high-RSS (freeze candidates). This creates a 3-tier response: boost (P-core) → migrate (E-core) → freeze (SIGSTOP).
**Validate**: Test that medium-priority processes get E-core migration instead of SIGSTOP. Test that energy delta is tracked before/after migration. Test that freeze is only applied when RSS > threshold.

#### 0Q. Cache Contention Detection on Heterogeneous Cores
**Paper**: ARM-sponsored case study, arXiv:2304.13110 (2023) — "Analysis and Mitigation of Shared Resource Contention on Heterogeneous Multicore."
**Insight**: On big.LITTLE, a background build process thrashing the shared L2 can degrade interactive latency even when CPU utilization looks fine. Cache contention is invisible to CPU-only monitoring.
**Gap**: Apollo protects interactive apps (Brave, Claude) but doesn't detect cache contention. `hw_predictor.rs` measures `cache_latency_us` and `l1_latency_us` but these aren't correlated with specific offending processes.
**Experiment**: During build mode (≥2 compilers), measure page probe latency for the frontmost application (using existing `probe_page_temp`). If probe latency increases >2× while system CPU < 70%, this indicates cache contention — migrate compiler processes to E-core QoS. Track `cache_contention_events` counter for observability.
**Validate**: Test that high cache latency + low CPU triggers contention detection. Test that contention detection doesn't fire when CPU is >80% (that's just load, not contention). Test that migration reduces probe latency.

---

#### 0R. TMO — Lost-Work Memory Metric
**Paper**: Weiner et al., Meta, ASPLOS 2022 — "TMO: Transparent Memory Offloading in Datacenters." Follow-up: "Tiered Memory Management Beyond Hotness" OSDI 2025.
**Insight**: TMO measures the *actual lost work* (refault rate) caused by memory reclaim, not just pressure levels. It auto-tunes aggressiveness based on application sensitivity to slowdown. The key metric is "how much real work is lost" not "how much memory is freed."
**Gap**: Apollo tracks pressure and swap velocity but not the collateral damage of its own actions. Freezing process X might cause Y's pages to get evicted by the compressor (because freed pages get reused). No measurement of this second-order effect.
**Experiment**: After each freeze action, monitor system-wide major page fault rate (via `host_statistics64`) for 30s. If major faults increase >20% vs pre-freeze baseline, record a negative outcome ("collateral damage freeze"). Add the frozen process to a penalty list that reduces its freeze score in future decisions. This teaches Apollo which freezes are "clean" vs "dirty."
**Validate**: Test that post-freeze fault spike is detected. Test that penalty list reduces freeze score. Test that processes not causing collateral retain normal scores.

#### 0S. PARTIES — QoS-Aware Hill Climbing
**Paper**: Chen, Delimitrou, Martinez, ASPLOS 2019 — "PARTIES: QoS-Aware Resource Partitioning." 61% throughput improvement under QoS constraints.
**Insight**: PARTIES dynamically partitions resources across latency-sensitive services using gradient-free hill climbing: try small resource adjustments, measure QoS impact, iterate. No model needed — just observe the effect of each perturbation.
**Gap**: Apollo protects foreground apps as binary (protected or not). No gradient: it doesn't measure whether protection is *sufficient* or *excessive*. A protected app getting 95% responsiveness wastes resources that could help background throughput.
**Experiment**: Define a responsiveness metric for the frontmost app: page probe latency (from existing `probe_page_temp`). Set target: <5ms. Each cycle, if target violated → demote one background process from "throttle" to "freeze" (give more resources to foreground). If target easily met (< 2ms) → promote one frozen process to "throttle" (recover throughput). Hill climbing on the throttle/freeze boundary.
**Validate**: Test that QoS violation triggers one demotion. Test that QoS surplus triggers one promotion. Test that oscillation is damped (no promote→demote→promote in 3 consecutive cycles).

#### 0T. CXL Tiering Semantics for Compressor
**Paper**: NeoMem (arXiv:2403.18702, 2024) — "Hardware/Software Co-Design for CXL-Native Memory Tiering." OSDI 2024 follow-up.
**Insight**: CXL memory tiering tracks page "hotness" and promotes/demotes pages between fast DRAM and slower CXL memory. Key insight: even within single-tier DRAM, Apple's compressor creates an effective two-tier system (uncompressed=fast, compressed=slow).
**Gap**: `decide_enhanced()` in `compressor_aware.rs` already considers `pct_compressed`, but treats it as a binary threshold (>0.60 → skip). CXL tiering suggests a continuous cost model: each byte has a `tier_cost` based on its current state.
**Experiment**: Define `tier_cost = uncompressed_bytes * 1.0 + compressed_bytes * 0.3 + swapped_bytes * 0.05`. Rank freeze candidates by `tier_cost / total_rss` — higher ratio means more "real" (uncompressed) memory freed per byte. Processes with mostly compressed pages have low tier_cost/rss and are poor freeze candidates.
**Validate**: Test that a process with 80% uncompressed scores higher than one with 80% compressed. Test that tier_cost ranking differs from raw RSS ranking. Test edge case: process with 100% compressed → near-zero freeze priority.

---

#### 0U. Budgeted Multi-Armed Bandit for Freeze Selection
**Paper**: arXiv:2505.02640 (2025) — "Adaptive Budgeted Multi-Armed Bandits for IoT with Dynamic Resource Constraints."
**Insight**: Budgeted MAB adds cost constraints to exploration/exploitation: each arm pull has an uncertain reward AND a cost. Budgeted UCB balances reward optimization with cost budgets, achieving sublinear regret.
**Gap**: When Apollo needs to free N MB, it picks freeze candidates heuristically. Each "freeze arm" has uncertain reward (actual memory freed depends on compressor) and cost (user-impact risk). No principled exploration of which processes are most cost-effective.
**Experiment**: Model each freezeable process as a bandit arm. `reward_estimate = EMA(actual_memory_freed)`. `cost = co_occurrence_centrality * (1 - pct_compressed)`. `ucb_score = reward_estimate + sqrt(ln(total_freezes) / times_frozen_this_process)`. Select greedily by `ucb_score / cost` until memory target met or cost budget exhausted.
**Validate**: Test that high-reward/low-cost arms are selected first. Test that UCB exploration term decreases with more pulls. Test that cost budget prevents selecting expensive arms when cheaper alternatives exist.

#### 0V. Model-Based RL with Dyna Architecture
**Paper**: Dyna (Sutton 1991); modern: "Dyna-style Model-Based RL with Model-Free Policy Optimization," Knowledge-Based Systems, 2024.
**Insight**: Model-based RL builds a "world model" predicting next-state from (state, action). The agent does *imagined rollouts* — planning without real interaction. 2-9× better sample efficiency than model-free RL. Critical for systems where each real sample takes 5 seconds.
**Gap**: `rl_threshold.rs` learns from 1 real observation per 5s cycle. After a workload shift, it takes ~200 cycles (16 minutes!) to converge. A world model could simulate 50 experiences per real cycle, converging in ~60s.
**Experiment**: Build a lookup table `(pressure_band, action) -> (delta_pressure_mean, delta_pressure_var)` populated from OutcomeTracker's experience memory. Each real cycle, do 5 imagined rollouts: sample random states, apply random actions, predict outcome via world model, update Q-values with discount 0.5 (vs 1.0 for real). This is Dyna-Q.
**Validate**: Test that world model predictions improve with more data. Test that imagined rollouts update Q-values. Test that convergence after workload shift is >2× faster with Dyna vs pure Q-learning.

#### 0W. Stigmergy — Pheromone-Based Freeze Memory
**Paper**: Dorigo et al. (ACO, 1996); modern: Smith, "Collective Stigmergic Optimization" (2024); "SwarmFabSim" Springer 2023.
**Insight**: Ants coordinate not by direct communication but by depositing pheromones in the environment. Good paths get reinforced; pheromone evaporates so stale knowledge auto-expires. Indirect coordination through shared state — no central planner needed.
**Gap**: Apollo's freeze decisions don't remember cross-session "which processes are good freeze targets." OutcomeTracker has per-process effectiveness, but it's a complex Bayesian system. A simpler pheromone signal could complement it with fast, intuitive "this worked before" memory.
**Experiment**: Add `pheromone: HashMap<String, f64>` to daemon state. Successful freeze (memory freed > threshold): deposit +1.0 pheromone on process name. Failed freeze: deposit -0.5. Every cycle: decay all by 0.95×. Use pheromone as a multiplier in freeze candidate scoring: `score *= (1.0 + pheromone.get(name).unwrap_or(0.0)).max(0.1)`. Pheromone auto-expires (half-life ≈ 14 cycles ≈ 70s).
**Validate**: Test that successful freezes increase pheromone. Test that pheromone decays to near-zero after 50 cycles of no reinforcement. Test that negative pheromone reduces freeze priority.

#### 0X. Danger Theory — Multi-Signal Crisis Detection
**Paper**: Hosseini et al., Wiley Journal of Engineering, 2025 — "Artificial Immune Systems for Industrial Intrusion Detection." Jia et al., Future Generation Computer Systems, 2023. Foundational: Matzinger's Danger Theory, 2002.
**Insight**: The immune system responds to *danger signals* from stressed cells, not to "foreign" entities per se. This prevents false positives on benign anomalies. Key: requires **multiple concurrent danger signals** before escalating (like dendritic cells needing 2+ signals to activate T-cells).
**Gap**: Apollo escalates to aggressive freeze on a single condition: `pressure >= 0.90 && swap_delta > 5MB/s`. But pressure can spike transiently (app launch) without being dangerous. Single-signal triggers cause false positives.
**Experiment**: Define 4 danger signals: (1) compressor_ratio spike > 2× in 5s, (2) thermal_pressure > moderate, (3) pressure_state change to critical (vm_pressure kernel notification), (4) frontmost app probe latency > 3× baseline. Track `danger_level = count(active_signals)`. Require danger_level >= 2 for aggressive freeze (currently danger_level >= 1 via single threshold). This reduces false positives while maintaining response to real crises.
**Validate**: Test that single danger signal does NOT trigger aggressive freeze. Test that 2+ concurrent signals DO trigger. Test that transient pressure spike (1 signal, resolves in 5s) doesn't escalate. Test that real crisis (3+ signals) escalates faster than current system.

#### 0Y. Interpretable RL via Decision Tree Distillation
**Paper**: arXiv:2405.19131 (2024) — "Learning Interpretable Scheduling Algorithms for Data Processing Clusters."
**Insight**: Complex RL schedulers are black boxes. This paper trains RL, then distills the learned policy into a decision tree that is interpretable, fast to execute, and auditable. The tree captures >90% of the RL policy's behavior in human-readable rules.
**Gap**: Apollo's Q-table produces opaque decisions. The journal logs *what* Apollo did but not *why*. Users can't audit "why did you freeze Slack?" without reverse-engineering Q-values. No explainability.
**Experiment**: Every 1000 cycles, export the Q-table as human-readable rules: for each state, output the dominant action and its Q-advantage. Format: `"pressure=high, compressor=mid, overflows=0 → Lower5pp (Q=2.3 vs Hold=1.1)"`. Log to journal as `policy_snapshot` event. Add `--explain` to `apolloctl status` that prints the current distilled policy.
**Validate**: Test that policy export produces valid rules for all 36 states. Test that distilled rules agree with Q-table argmax >95% of the time. Test that rules are human-readable (parseable format).

### Tier 1: High-Value (likely to improve score)

1. **Dead code removal** — Find functions/structs never called from any binary. Remove them. Score improves via smaller binary + potential clippy reduction. Candidates:
   - `optimizer.rs:optimize()` (confirmed dead)
   - `TelemetryLogger` (confirmed disabled)
   - Any `pub` function with 0 callers outside its own module

2. **Test coverage gaps** — Modules with 0 tests. Each new passing test adds +1 to score. Priority:
   - `src/engine/wait_graph.rs` — has functions but no unit tests
   - `src/engine/gpu_manager.rs` — only has Default impl test
   - `src/engine/page_reclaim.rs` — check if tested
   - `src/engine/process_tree.rs` — check if tested
   - `src/engine/coalition.rs` — check if tested

3. **Clippy fixes** — Each warning eliminated is +5 to score. Run `cargo clippy --all-targets` and fix everything it flags.

### Tier 2: Medium-Value (structural improvements)

4. **Redundant computation elimination** — Profile the hot path in `apollo-optimizerd.rs` for repeated work:
   - Multiple `lock_recover()` calls on the same mutex in the same scope
   - Repeated string formatting of the same process names
   - `collector.system().processes()` iterated multiple times

5. **Const promotion** — Find `let` bindings of literal values that should be `const`. Compiler can optimize better.

6. **Allocation reduction** — Find `Vec::new()` or `String::new()` in the hot loop that could be pre-allocated or reused across cycles.

### Tier 3: Exploratory (may or may not improve score)

7. **Algorithm improvements** — Better heuristics in existing decision code:
   - `behavioral_protection_score()` — can the formula be tighter?
   - `overflow_guard` thresholds — are they well-calibrated?
   - `rl_threshold` learning rate schedule — does EMA α converge?

8. **Error handling audit** — Find `.unwrap()` calls that should be `.unwrap_or()` or `?`. Not for score, but for daemon stability.

9. **Module consolidation** — If two small modules (<50 lines each) serve related purposes, merging them reduces cognitive overhead and may eliminate unused `pub` exports.

### Tier 4: Maintenance

10. **Dependency audit** — Are all Cargo.toml dependencies actually used? Unused deps increase compile time and binary size.

11. **Feature flag cleanup** — `tract-onnx` is optional and disabled. Are there other dead features?

## Anti-Patterns (DO NOT)

- Do NOT add features the user didn't ask for
- Do NOT refactor working code just because it's "ugly"
- Do NOT add comments, docstrings, or type annotations to code you didn't functionally change
- Do NOT add error handling for impossible scenarios
- Do NOT create new modules — extend existing ones
- Do NOT add dependencies
- Do NOT modify `autoresearch/evaluate.sh`
- Do NOT modify tests to make them pass (fix the code, not the test)
- Do NOT make multiple unrelated changes in one experiment

## How to Read results.tsv

```
experiment  branch  score_before  score_after  delta  status  description
```

- `kept` = score improved, committed
- `discarded` = score same or worse, reverted
- `crash` = build/test failed, reverted
- Look for patterns: which directions yield improvements? Which are dead ends?
