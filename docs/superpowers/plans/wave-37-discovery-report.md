# Wave 37+ Discovery Report

Decision: C (Mixed - Fluidity Intelligence is cohesive and extractable, others are mostly downstream/orchestration or wrapped).
Top-1 candidate: Fluidity Intelligence (lines 2291-2521)

## Section Inventory (main.rs loop body 1229-4630)

| Section | Lines | LOC | Class | Extractable? | Notes |
|---------|-------|-----|-------|--------------|-------|
| Feature 4: Post-Wake Suppression | 1245-1349 | 105 | WRAPPED | No | Calls `daemon_feature_gates::apply_post_wake_suppression` |
| Dry-run ultra-fast-path | 1350-1671 | 322 | INLINE_COHESIVE | Yes | Highly tangled with telemetry writing |
| LearningContext | 1672-1884 | 213 | ORCHESTRATION_GLUE | No | Context building for downstream modules |
| IOReport / Per-cycle sensor telemetry pass | 1885-1920 | 36 | WRAPPED | No | Calls `daemon_sensor_tick::SensorTickOutput` |
| Feature 1: LLM Inference Mode | 1921-1932 | 12 | WRAPPED | No | Calls `daemon_feature_gates::run_llm_inference_mode_tick` |
| Feature 3: RT Boost for Foreground | 1933-1942 | 10 | WRAPPED | No | Calls `daemon_feature_gates::apply_rt_boost_foreground` |
| Effective pressure aggregation | 1943-2290 | 348 | WRAPPED/GLUE | No | Core logic wrapped in `daemon_pressure_aggregator`, remainder is context_switch_burst inline glue |
| Fluidity Intelligence | 2291-2521 | 231 | INLINE_COHESIVE | Yes | Target for Wave 37 |
| Signal intelligence | 2522-3187 | 666 | DOWNSTREAM_CONSUMER | Yes (Partial) | `run_signal_tick` wrapped, but lines 2546+ are inline consumers. Target for Wave 39 |
| Heuristic pass: AdaptiveGovernor | 3188-3245 | 58 | WRAPPED | No | Calls `daemon_action_safety::run_heuristic_pass` |
| Neuromodulator: bio-inspired | 3246-3319 | 74 | INLINE_COHESIVE | Yes | Small cohesive block |
| Feature 5: Wakeup Budget Enforcer | 3320-3330 | 11 | WRAPPED | No | Calls `daemon_feature_gates::enforce_wakeup_budget` |
| Feature 2 + 4: App Nap | 3331-3524 | 194 | WRAPPED | No | Calls `daemon_feature_gates::apply_app_nap_scheduling` |
| Chromium Renderer Manager | 3525-3940 | 416 | WRAPPED/GLUE | No | Core logic wrapped in `daemon_chromium_tick::run_chromium_tick`, rest is post-tick execution |
| Filter pipeline | 3941-3959 | 19 | WRAPPED | No | Calls `daemon_action_pipeline::run_filter_pipeline` |
| Predictive thaw gate | 3960-4025 | 66 | INLINE_COHESIVE | Yes | Small gate logic |
| Circuit breaker + execute_actions | 4026-4200 | 175 | INLINE_COHESIVE | Yes | Target for Wave 40 |
| Cognitive gate: pause_learning | 4201-4272 | 72 | INLINE_COHESIVE | Yes | Small inline gate |
| Neurocognitive tick | 4273-4360 | 88 | WRAPPED | No | Calls `daemon_neuro_tick::run_neurocognitive_tick` |
| Fluidity QoS elevation | 4361-4389 | 29 | WRAPPED | No | Calls `daemon_cycle_tail::apply_fluidity_qos` |
| Enriched telemetry | 4390-4411 | 22 | WRAPPED | No | Calls `daemon_cycle_tail::wire_enriched_telemetry` |
| Periodic stage: GC | 4412-4630 | 219 | WRAPPED | No | Calls `daemon_cycle_tail::run_periodic_stage` |

## Real Extraction Candidates (INLINE_COHESIVE only)

| Candidate | Lines | LOC | Why cohesive | State accessed | Risk |
|-----------|-------|-----|--------------|----------------|------|
| Fluidity Intelligence | 2291-2521 | 231 | Self-contained state update and signal generation | proc_snaps, hw_snap, fluidity_state | Low - state is passed by mutable ref, output is pure value |
| Circuit breaker + execute_actions | 4026-4200 | 175 | Clearly defined responsibility (action dispatch) | cb state, action list | High - touches execution pipeline |
| Signal Intel Downstream | 2546-3187 | 641 | Downstream consumers of signal intel | many context states | High - NaN bomb risk |

## Sections that LOOK extractable but aren't

- Effective pressure aggregation (348 LOC): Mostly wrapped in `aggregate_cycle_pressure`.
- Chromium Renderer Manager (416 LOC): Mostly wrapped in `run_chromium_tick`.
- Signal intelligence (666 LOC): The core tick is wrapped in `run_signal_tick`, though downstream consumers form a large block.

## Recommendation

Top-1 candidate is **Fluidity Intelligence**. It represents a 231 LOC cohesive block that can be easily extracted without interfering with core execution paths or carrying NaN bomb risks. Circuit breaker and Signal Intel downstream are secondary targets.
