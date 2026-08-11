# Apollo Unified Learning Architecture Implementation Plan

**Goal:** Close Apollo's state -> decision -> action -> outcome -> credit -> learning loop across every actuator path without granting new kernel authority to advisory models.

**Architecture:** Add a bounded universal decision ledger inside the engine. Existing specialists continue to propose actions, existing safety gates and `ActuationBroker` remain authoritative, and daemon tick outputs feed receipts back to the medallion. Gold evidence alone updates model authority, NARS, hierarchy, and AIS.

**Global constraints:** Stability and fluidity first; no LLM or prompt teacher; capability-derived M1-M4 behavior; GPU/World Model/MPC remain advisory; no new direct kernel path; no blocking I/O or unbounded collection in the control loop; no quadratic engine comparisons or full sorting where bounded top-k works; preserve all existing serialized and metrics fields; imported evidence cannot grant local authority.

## Task 0: Preserve the verified base

- Commit the already deployed attribution, utility decomposition, latency dynamics, GPU calibration, Markov horizons, metrics, and dashboard changes.
- Exclude `.DS_Store`, manual-teach policy files, and generated caches.
- Baseline gate: `cargo test --workspace --release` and `cargo fmt --all -- --check`.

## Task 1: Universal decision ledger

- Add `decision_ledger.rs` with bounded `DecisionId`, `DecisionEnvelope`, candidate alternatives, hierarchy coordinates, prediction records, execution receipts, lifecycle states, and resolved episodes.
- Keep 192 pending, 64 recent, 128 episodic records, at most 8 alternatives and 8 adviser contributions per decision.
- Provide O(1) or bounded-linear APIs for propose, reject/veto/block, record execution, expire, and settle.
- Reject unattributed applied receipts from authority-producing learning and count them explicitly.
- TDD: lifecycle completion, duplicate/idempotent receipts, eviction, expiry, missing attribution, concurrent actions, serialization defaults.

## Task 2: Wire every actuator path

- Root dispatch, Markov prewarm, interaction QoS, Chromium e-core/jetsam/purge, maintenance purge, memorystatus, sysctl, freeze/throttle, recovery, and coordinated actions emit decision/receipt events.
- Extend tick outputs or use cycle-local buffers; do not add a hot-path global mutex.
- Emergency actions may bypass model ranking but must still be audited and reversible.
- TDD: applied, blocked, failed, no-op, reverted, expired, and side-channel actions all close with a `DecisionId`.

## Task 3: Calibration, causal credit, and trust

- Add `model_calibration.rs` with bounded keys for producer, action/family, workload, process class, 5s/30s/2m/10m horizon, pressure band, thermal band, and foreground context.
- Track signed error, MAE, uncertainty coverage, and Brier where the target is binary; cap at 512 keys with family fallback.
- Assign causal credit to the actuator using matched no-action drift and controlled evidence. Give advisers calibration credit, never duplicate the actuator's utility across supporters.
- Confounded coordinated cohorts update only the coordinated model unless individual effects are separable.
- Per-model trust: Immature (<10 local Gold), Candidate (>=10, quality >=0.85), Validated (>=20, >=3 contexts, normalized error <=0.15), Trusted (>=50, >=5 contexts, error <=0.10, decisive 95% interval, three stable windows), Degraded after two bad windows or immediately on origin mismatch.
- TDD: natural recovery earns no Apollo credit, bad predictions lose trust, promotion/demotion hysteresis, hardware/install reset, and v2 -> v3 persistence.

## Task 4: Hierarchical memory and autonomous consolidation

- Add `learning_hierarchy.rs`: Goal -> Strategy -> ActuatorFamily tactic -> concrete action.
- Goals: stability, responsiveness, memory headroom, thermal safety, energy efficiency.
- Strategies: protect foreground, predict next use, relieve pressure, shift background work, recover state, reduce energy.
- Persist rich resolved episodes with alternatives, predicted/actual outcomes, causal evidence, calibration delta, and trust transition.
- Consolidate Gold into bounded prototypes and NARS beliefs by goal/strategy/workload/context. Apply decay, deduplication, and weak/stale eviction.
- Remove any remaining runtime dependency on manual teach or prompt-teacher artifacts.

## Task 5: Safe counterfactual exploration

- Prefer observational matched baselines, then sparse reversible parameter probes for Boost, interactive QoS, and Markov prewarm.
- Prefer TTL/cache-only variants to omitting an acceleration.
- Global budget: one probe per 15 minutes; same action/context cooldown 24 hours.
- Gate out audio/calls, launches, window operations, degraded fluidity, pressure >=0.55, non-nominal thermal state, elevated hazard, open circuit, and invalid process identity.
- Freeze, throttle, purge, sysctl, and recovery are never exploratory.
- TDD every gate, deterministic scheduling, cancellation, cooldown, and persistence behavior.

## Task 6: World Model, AIS, metrics, and dashboard

- Feed unified calibration/trust/episodes into World Model, MPC, GPU, Markov, and planner ranking only.
- AIS may reward closure coverage, calibrated accuracy, causal resolution, and trusted breadth only after local Gold maturity. Action count, imported evidence, and dormant opportunity-dependent actuators earn no score.
- Preserve existing JSON fields and add ledger closure, trust inventory, horizon calibration, exploration, and latest resolved episode fields with serde defaults.
- Dashboard shows human-readable collecting/no-evidence states, closure %, trusted/degraded count, worst active predictor/horizon, and latest expected vs measured effect.

## Task 7: Verification, review, and rollout

- Run focused RED/GREEN tests per task, then `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets`, `cargo test --workspace --release`, and `cargo build --workspace --bins --release`.
- Performance acceptance: cycle p95 regression <=5%, daemon RSS delta <=10 MiB, persisted-state delta <=2 MiB, and bounded O(candidates + advisers + fixed horizons) work.
- Graphify audit must prove every applied effect reaches the ledger, every local Gold reaches consolidation, and no advisory model reaches a kernel mutation outside existing authority.
- Deploy as risky/structural through `scripts/apollo-accept-gate.sh --risky`, with pre-deploy state/binary backups and existing rollback behavior.
- Production checks: launchd running, cycles advancing, failures zero, fresh journal attribution, ledger closure >=95% after horizons, no unattributed applied effects, and no p95/AIS/safety regression. Claims remain preliminary until >=500 post-deploy Gold events.

## Commit sequence

1. `feat: add bounded universal decision ledger`
2. `feat: connect all Apollo actuator outcomes`
3. `feat: calibrate predictions and model trust`
4. `feat: consolidate hierarchical Apollo learning`
5. `feat: add safe counterfactual exploration`
6. `feat: expose unified learning health`
7. `docs: record unified learning verification`
