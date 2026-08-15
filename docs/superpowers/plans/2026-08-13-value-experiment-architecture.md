# Apollo Option C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy a bounded value-per-cost scheduler, versioned cycle context, truthful heterogeneous runtime, and paired microexperiment evidence path without changing Apollo's existing safety or actuation authority.

**Architecture:** The main loop publishes one immutable cycle identity and remains the only control-plane owner. A fixed value scheduler admits optional advisory work under the existing adaptive-overhead budget; existing models and Metal only rank. A pure microexperiment state machine joins real control/treatment endpoints and is the only new path to experimental Gold or AIS attribution.

**Tech Stack:** Rust 2021, serde, standard-library synchronization/channels, Rayon only where already compiled, Objective-C++ Metal bridge, Cargo workspace tests, launchd guarded deployment.

## Global Constraints

- Safety, rollback, urgent release, kill-switch, sleep/wake recovery, identity recheck, `ActionQueue`, and `ActuationBroker` are never value-scheduled.
- `MAX_JOBS=64`, `MAX_SELECTED_PER_CYCLE=16`, `MAX_DEPENDENCIES_PER_JOB=4`, `MAX_IN_FLIGHT_OPTIONAL=4`, and `MAX_COMPLETIONS_PER_CYCLE=64`.
- Optional budgets remain `150_000/100_000/60_000 us` for Nominal/Guarded/Constrained.
- Scheduler and lab begin in Shadow and require 500 valid post-warmup cycles before automatic Active eligibility.
- Metal is advisory and non-blocking; synthetic/model/GPU evidence never becomes Pair Gold or AIS credit.
- macOS lanes are named `efficiency` and `latency`; no P-core/E-core residency claim is permitted.
- Core ML/ANE and raw AMX participation are deferred.
- AIS weights remain unchanged.
- One Cargo command at a time; broad release verification runs once at the integration gate.
- Preserve all existing dirty-worktree changes and never reset unrelated work.

---

### Task 1: Immutable Cycle Identity and Fixed Value Scheduler

**Files:**
- Create: `crates/apollo-engine/src/engine/cycle_snapshot.rs`
- Create: `crates/apollo-engine/src/engine/value_scheduler.rs`
- Create: `crates/apollo-engine/tests/value_scheduler_contract.rs`
- Modify: `crates/apollo-engine/src/engine/mod.rs`

**Interfaces:**
- Produces: `SnapshotId`, `ObservationStatus`, `CycleContextSnapshot`, `JobId`, `JobDescriptor`, `SchedulerLevel`, `SchedulerPhase`, `SchedulerInputs`, `JobPermit`, `JobCompletion`, `ValueScheduler`, and `ValueSchedulerMetrics`.
- Consumes later: daemon integration supplies the current cycle context, adaptive-overhead level, fixed descriptors, and completions.

- [ ] **Step 1: Write failing scheduler and snapshot contract tests**

Cover exact registry bounds, deterministic ranking, invalid floating-point sanitization, budget/capacity limits, latest-wins generation behavior, stale identity drops, monotonic revision overflow, immutable publication, and the all-caps selection latency invariant.

```rust
#[test]
fn predicted_plan_never_exceeds_budget_or_sixteen_jobs() {
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Shadow);
    let plan = scheduler.plan(&fixture_snapshot(7), SchedulerInputs::nominal_all_due());
    assert!(plan.permits.len() <= MAX_SELECTED_PER_CYCLE);
    assert!(plan.predicted_us <= NOMINAL_BUDGET_US);
}

#[test]
fn stale_wrong_epoch_wrong_revision_and_wrong_identity_completions_drop() {
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Shadow);
    let permit = scheduler.plan(&fixture_snapshot(9), SchedulerInputs::nominal_all_due())
        .permits.into_iter().next().unwrap();
    let wrong = JobCompletion { snapshot_id: SnapshotId { daemon_epoch: 8, sequence: 9 }, ..completion_for(&permit) };
    assert_eq!(scheduler.complete(wrong), CompletionDisposition::Stale);
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `cargo test -p apollo-engine --test value_scheduler_contract`

Expected: compilation fails because `cycle_snapshot` and `value_scheduler` do not exist.

- [ ] **Step 3: Implement the pure bounded types and algorithms**

`CycleContextSnapshot` contains only finite compact scalar/context fields and identity revisions; no live process object, mutex, mutable cache, action, or model. `ValueScheduler` uses a fixed `JobId::ALL` registry, bounded arrays/vectors, checked generations, deterministic integer scoring, and terminal accounting. Shadow emits permits as recommendations but exposes `should_execute=false`.

```rust
pub fn plan(
    &mut self,
    snapshot: &CycleContextSnapshot,
    inputs: SchedulerInputs,
) -> SchedulerPlan;

pub fn complete(&mut self, completion: JobCompletion) -> CompletionDisposition;

pub fn observe_legacy_run(
    &mut self,
    job: JobId,
    snapshot_id: SnapshotId,
    elapsed_us: u32,
    succeeded: bool,
);
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `cargo test -p apollo-engine --test value_scheduler_contract`

Expected: all scheduler/snapshot contract tests pass.

- [ ] **Step 5: Update the progress ledger**

Record Task 1 implementation complete, the exact focused test output, and Task 2 as the next gate. Do not commit yet; this repository batches the integrated Option C feature on the current dirty branch.

### Task 2: Daemon Shadow Integration and Honest Scheduler Metrics

**Files:**
- Modify: `crates/apollo-engine/src/engine/policy_store.rs`
- Modify: `crates/apollo-engine/src/engine/types.rs`
- Modify: `src/bin/apollo-optimizerd/adaptive_overhead.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs`
- Modify: `src/bin/apollo-optimizerd/metrics_reporter.rs`
- Modify: `src/bin/apollo-optimizerd/socket_handler.rs`
- Create: `src/bin/apollo-optimizerd/value_scheduler_tick.rs`
- Test: `src/bin/apollo-optimizerd/value_scheduler_tick.rs`

**Interfaces:**
- Consumes: Task 1 scheduler and snapshot types.
- Produces: one per-cycle `CycleContextSnapshot`, scheduler Shadow recommendations, legacy-run cost observations, rollout readiness/blocker, and globally serialized metrics.

- [ ] **Step 1: Write failing daemon tests**

Test exact conversion from `OverheadLevel` to scheduler budgets, one snapshot sequence per cycle, Shadow never changing cadence, source unknowns remaining unavailable, 500-cycle readiness gates, kill-switch/sleep cancellation, and no impact on Reflex decisions.

```rust
#[test]
fn shadow_plan_records_without_changing_legacy_execution() {
    let result = run_value_scheduler_tick(fixture_tick(SchedulerPhase::Shadow));
    assert!(!result.execute_permits);
    assert!(result.metrics.selected_total > 0);
    assert_eq!(result.legacy_due_jobs, result.legacy_executed_jobs);
}
```

- [ ] **Step 2: Run the focused daemon test and verify RED**

Run: `cargo test --bin apollo-optimizerd value_scheduler_tick`

Expected: compile failure for the missing daemon integration module/config/metrics.

- [ ] **Step 3: Add configuration and runtime metrics**

Add serde-defaulted `ValueSchedulerConfig { enabled, phase, shadow_cycles: 500 }`. Extend `RuntimeMetrics` with bounded aggregate scheduler/snapshot fields and a fixed-size recent-job view. Keep all legacy fields unchanged.

- [ ] **Step 4: Wire one snapshot and observe existing jobs**

Build `CycleContextSnapshot` immediately after current source/context values are available. Register current Holt-Winters, Page Reclaim, GPU Imagination, Reflex Reasoning, World Model refresh, AIS refresh, hardware prediction, planner advice, periodic maintenance, and telemetry flush. In Shadow, preserve legacy cadence and call `observe_legacy_run`; do not double-run anything.

- [ ] **Step 5: Implement rollout readiness**

Readiness requires 500 valid cycles, profile match, p95 below 75 ms and within 10% baseline, no cycle/rollback/protected-action failures, no oscillation regression above 10%, and scheduler selection p95/max within 250 us/1 ms. Publish one exact blocker string/enum.

- [ ] **Step 6: Run the focused daemon test and verify GREEN**

Run: `cargo test --bin apollo-optimizerd value_scheduler_tick`

Expected: all integration tests pass and existing Reflex tests remain unchanged.

### Task 3: Paired Microexperiment State Machine

**Files:**
- Create: `crates/apollo-engine/src/engine/microexperiment_lab.rs`
- Create: `crates/apollo-engine/tests/microexperiment_lab_contract.rs`
- Modify: `crates/apollo-engine/src/engine/mod.rs`
- Modify: `crates/apollo-engine/src/engine/learned_state.rs`

**Interfaces:**
- Consumes: existing `ExplorationArm`, `ExplorationFamily`, `ExplorationMetadata`, installation/hardware origin, and bounded endpoint facts.
- Produces: `PairId`, `PairAssignment`, `PairEndpoint`, orthogonal execution/horizon/rollback closures, `PairGoldRecord`, bounded persisted lab state, and funnel metrics.

- [ ] **Step 1: Write the failing T01-T18 and T37-T45 pure contract tests**

Cover exact allowlist, one control/one treatment, deterministic balanced AB/BA assignment independent of model score, complement lock, safety/privacy gates, full-horizon invalidation, pair capacity, washout, no third-arm oscillation, persistence bounds/privacy, hostile restore, origin mismatch, clock anomalies, and bounded scans.

```rust
#[test]
fn pair_requires_exactly_one_control_and_one_treatment() {
    let mut lab = MicroexperimentLab::cold_start(local_origin());
    let pair = lab.open_pair(valid_qos_candidate()).unwrap();
    assert!(lab.record_endpoint(pair.id, control_endpoint()).is_ok());
    assert!(matches!(lab.record_endpoint(pair.id, control_endpoint()), Err(LabError::DuplicateArm)));
    assert!(lab.record_endpoint(pair.id, treatment_endpoint()).is_ok());
}
```

- [ ] **Step 2: Run the focused contract test and verify RED**

Run: `cargo test -p apollo-engine --test microexperiment_lab_contract`

Expected: missing module/types.

- [ ] **Step 3: Implement the pure lab and bounded restore**

The lab owns no I/O, PID mutation, actuator, model update, or AIS call. It stores only coarse hashed/action identity, assignment, closure facts, deadlines, bounded summaries, and dedup IDs. Open pairs become interrupted on restart and never resume.

```rust
pub fn propose(&mut self, candidate: PairCandidate, gates: PairGates) -> Result<PairAssignment, LabError>;
pub fn record_endpoint(&mut self, pair_id: PairId, endpoint: PairEndpoint) -> Result<PairProgress, LabError>;
pub fn close_pair(&mut self, pair_id: PairId) -> Result<PairClosure, LabError>;
pub fn drain_pair_gold(&mut self) -> Vec<PairGoldRecord>;
pub fn persisted(&self) -> MicroexperimentLabPersisted;
```

- [ ] **Step 4: Wire serde-defaulted persistence**

Add one `Option<MicroexperimentLabPersisted>` field to `LearnedStateSupplement`; enforce 64 KiB, 32 open, 128 completed, 128 dedup, string bounds, schema/origin/hardware checks, no process names/paths/titles/raw samples, and interrupted-open behavior.

- [ ] **Step 5: Run the focused contract test and verify GREEN**

Run: `cargo test -p apollo-engine --test microexperiment_lab_contract`

Expected: all pure lifecycle/capacity/persistence tests pass.

### Task 4: Evidence Provenance, Closure, and AIS Honesty

**Files:**
- Modify: `crates/apollo-engine/src/engine/decision_ledger.rs`
- Modify: `crates/apollo-engine/src/engine/telemetry_medallion.rs`
- Modify: `crates/apollo-engine/src/engine/causal_graph.rs`
- Modify: `crates/apollo-engine/src/engine/intelligence_score.rs`
- Modify: `crates/apollo-engine/src/engine/unified_learning_health.rs`
- Modify: `src/bin/apollo-optimizerd/metrics_reporter.rs`
- Create: `crates/apollo-engine/tests/pair_gold_contract.rs`
- Modify: `crates/apollo-engine/tests/unified_learning_health_contract.rs`

**Interfaces:**
- Consumes: Task 3 closure facts and Pair Gold records.
- Produces: explicit evidence provenance, one-pair/one-Gold fanout, corrected closure health, and AIS inputs based on distinct authoritative evidence.

- [ ] **Step 1: Write failing T19, T26-T36, and T46-T49 tests**

Assert that applied receipt survives rollback as a separate fact; wrapper `Settled` does not hide envelope terminal state; synthetic/external/GPU/model endpoints stay Bronze; Pair Gold requires two real local matched endpoints; null/harmful Gold is not effective; duplicate/out-of-order callbacks emit once; and rollback alone never counts as adaptation.

```rust
#[test]
fn rollback_success_alone_never_counts_as_verified_adaptation() {
    let before = ais_projection(&fixture_metrics());
    let after = ais_projection(&fixture_metrics().with_qos_rollback_success());
    assert_eq!(before.verified_adaptations_correct, after.verified_adaptations_correct);
}
```

- [ ] **Step 2: Run focused evidence tests and verify RED**

Run: `cargo test -p apollo-engine --test pair_gold_contract --test unified_learning_health_contract`

Expected: failures demonstrate current Bronze attribution, rollback-credit, or lifecycle gaps.

- [ ] **Step 3: Add provenance and quarantine rules**

Add fixed `EvidenceProvenance` variants. Existing runtime-counter, GPU, model, and advisory observations remain useful Bronze support but cannot update experimental action authority, experimental causal fanout, Pair Gold, or AIS attribution.

- [ ] **Step 4: Correct closure and AIS projection**

Read `episode.envelope.lifecycle` for terminal closure. Count distinct local authoritative Pair Gold IDs for attributed evidence and the beneficial subset for effective evidence. Remove raw Interaction QoS rollback and unpaired Markov hit/miss from verified-adaptation correctness. Do not alter AIS weights.

- [ ] **Step 5: Run focused evidence tests and verify GREEN**

Run: `cargo test -p apollo-engine --test pair_gold_contract --test unified_learning_health_contract`

Expected: all provenance, closure, dedup, and AIS honesty tests pass.

### Task 5: Daemon Experiment Wiring and Rollback Safety

**Files:**
- Modify: `src/bin/apollo-optimizerd/main.rs`
- Modify: `src/bin/apollo-optimizerd/daemon_dispatch_tick.rs`
- Modify: `src/bin/apollo-optimizerd/daemon_cycle_tail.rs`
- Modify: `src/bin/apollo-optimizerd/daemon_markov_tick.rs`
- Modify: `src/bin/apollo-optimizerd/learning_tick.rs`
- Modify: `src/bin/apollo-optimizerd/metrics_reporter.rs`
- Modify: `crates/apollo-engine/src/engine/policy_store.rs`

**Interfaces:**
- Consumes: existing safe candidates and ExplorationScheduler reservations plus Task 3 lab.
- Produces: Shadow pair funnel now; mutable experiments only after explicit opt-in and readiness; treatment/control endpoints bound to real receipts and the same snapshot stratum.

- [ ] **Step 1: Write failing daemon wiring tests T08-T25**

Use fake identities/effect backends. Cover real interaction-triggered lease only, Markov cache-only closure, background Boost restrictions, media/build/privacy/full-horizon cancellation, immediate identity recheck, recycled-PID rollback, shutdown ordering, restart interruption, and exact rollback journal matching.

- [ ] **Step 2: Run the focused daemon tests and verify RED**

Run: `cargo test --bin apollo-optimizerd microexperiment`

Expected: missing lab wiring and closure propagation.

- [ ] **Step 3: Add Shadow-only orchestration**

Create proposals only from candidates that already reached the normal pipeline. Feed model/GPU/NARS/MPC values only as bounded rank hints. In Shadow, record would-open/would-assign diagnostics and leave current actuation unchanged.

- [ ] **Step 4: Add explicit opt-in and continuous gates**

Add `MicroexperimentConfig { enabled: false, phase: Shadow, shadow_cycles: 500 }`. Unknown privacy/secure-input/share/camera state blocks mutable experimentation. Re-evaluate all existing media/build/pressure/hazard/thermal/fluidity/circuit/pause gates throughout the endpoint horizon.

- [ ] **Step 5: Persist and restore lab state at existing checkpoints**

Use the existing learned-state supplement path. Shutdown order is rollback/release, terminalize pair, then persist. Restart interrupts open pairs; no endpoint resumes and no Gold is emitted.

- [ ] **Step 6: Run focused daemon tests and verify GREEN**

Run: `cargo test --bin apollo-optimizerd microexperiment`

Expected: all candidate, lifecycle, and safety tests pass without real kernel mutation.

### Task 6: Truthful Heterogeneous Lanes and Metal Deadline

**Files:**
- Modify: `src/bin/apollo-optimizerd/daemon_init.rs`
- Modify: `crates/apollo-engine/src/engine/gpu_imagination.rs`
- Modify: `src/bin/apollo-optimizerd/daemon_dispatch_tick.rs`
- Modify: `crates/apollo-engine/src/engine/types.rs`
- Modify: `src/bin/apollo-optimizerd/metrics_reporter.rs`
- Modify: `src/bin/apollo-optimizerctl/dashboard.rs`
- Modify: `scripts/hardware-build-profile.sh`
- Modify: `scripts/build-release.sh`
- Modify: `scripts/deploy.sh`
- Modify: `scripts/apollo-deploy-gate.sh`
- Test: `crates/apollo-engine/src/engine/gpu_imagination.rs`
- Test: `tests/scripts/hardware-build-profile-test.sh`

**Interfaces:**
- Consumes: Task 1 snapshot identity and scheduler permits.
- Produces: efficiency/latency lane truth, non-panicking worker startup, Metal request/result deadlines, explicit unknown telemetry, quarantine metrics, and pre-swap build-profile rejection.

- [ ] **Step 1: Write failing lane, Metal, and manifest tests**

Cover M1/validated-M4/unknown worker policy, QoS intent names, spawn/QoS failure, no core-residency claims, missing watts not becoming zero, capacity-one busy behavior, deadline/workload/revision mismatch, non-finite/shape rejection, timeout quarantine, sleep/wake invalidation, portable fallback, and mismatched manifest rejection before swap.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `cargo test --bin apollo-optimizerd daemon_init && cargo test -p apollo-engine gpu_imagination && bash tests/scripts/hardware-build-profile-test.sh`

Expected: current user-initiated global pool, missing deadline identity, or manifest schema fails new assertions.

- [ ] **Step 3: Implement truthful CPU lane reporting**

Rename claims to efficiency/latency intent. Efficiency workers request utility QoS and record the return status. Latency work remains default unless tied to current measured foreground interaction. Worker spawn failure degrades the optional lane without panicking the daemon. Never emit E/P residency metrics.

- [ ] **Step 4: Harden Metal advisory lifecycle**

Bind request/result to snapshot/workload/capability/thermal revisions and a 250 ms monotonic deadline. Preserve one in-flight request and existing support bound. Validate finite shape. A timeout or repeated command failure quarantines the lane; the main loop continues and never falls back to CPU Monte Carlo in production. Preserve unknown GPU watts as unknown and gate using public state plus owned GPU time.

- [ ] **Step 5: Extend build/deploy capability manifest**

Record explicit CPU baseline, target triple, feature set, deployment mode, daemon/ctl hashes, and Metal source mode/hash. Candidate preflight rejects unsupported baseline/profile/capability before replacing production. Portable mode permits one-worker/no-Metal fallback; heterogeneous-required mode rejects a missing declared lane.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the same three focused commands once. Expected: lane, Metal, and script contracts pass.

### Task 7: Dashboard, Adversarial Review, Integrated Verification, and Deploy

**Files:**
- Modify: `src/bin/apollo-optimizerctl/dashboard.rs`
- Modify: `.superpowers/sdd/2026-08-13-value-experiment-architecture/progress.md`
- Verify: all files touched by Tasks 1-6

**Interfaces:**
- Consumes: all integrated runtime metrics.
- Produces: a compact readable dashboard, one batched review fix pass, one release candidate, guarded deployment, and fresh production evidence.

- [ ] **Step 1: Render compact dashboard truth**

Show:

```text
Value  shadow 327/500  jobs 4/10  68/150ms  block samples
Lanes  efficiency 4 utility ok | latency 1 default | Metal ready
Lab    shadow pairs 0/1  Gold 0 effective 0  synthetic 14q
```

Keep support/advisory counts visibly separate from action and Pair Gold. Do not imply that zero experiments or Gold is unhealthy when no safe opportunity exists.

- [ ] **Step 2: Run one batched adversarial review**

Review the entire diff against the frozen scenario matrix: every job/arm/outcome, startup/shutdown/wake/kill, corrupt/imported/foreign state, capacity/overflow/dedup/order, protected/PID-reuse paths, direct actuation bypasses, stale completion, GPU timeout, and AIS provenance. Apply one coherent correction pass for demonstrated P0/P1 gaps only.

- [ ] **Step 3: Run task-end verification**

Run, sequentially:

```bash
cargo test -p apollo-engine
cargo test --bin apollo-optimizerd
cargo test --bin apollo-optimizerctl
cargo fmt --all -- --check
git diff --check
```

Expected: all affected crate/binary tests pass; formatting and whitespace checks are clean.

- [ ] **Step 4: Run the integration release gate once**

Run, sequentially and without another Cargo process:

```bash
cargo test --workspace --all-targets --features adaptive-multicore
cargo clippy --workspace --all-targets --features adaptive-multicore -- -D warnings
./scripts/build-release.sh adaptive-multicore
./scripts/e2e-test.sh
```

Expected: workspace, Clippy, release build, profile manifest, and E2E all pass. Preserve raw output for the user.

- [ ] **Step 5: Show the exact deployment command and wait for approval**

Present:

```bash
./scripts/apollo-deploy-gate.sh --profile adaptive-multicore
```

Do not execute until the user explicitly approves that exact shared-state command, even though general deployment permission was granted earlier.

- [ ] **Step 6: Deploy and verify production mechanically**

Run the approved guarded deploy. Then show:

```bash
sudo launchctl print system/com.eduardocortez.systemoptimizerd
/usr/local/bin/apollo-optimizerctl status --json
shasum -a 256 /usr/local/libexec/apollo-optimizerd /usr/local/bin/apollo-optimizerctl
```

Verify launchd `state = running`, installed hashes match the candidate manifest, the effective M4 profile and workers are correct, the scheduler/lab are Shadow with honest blockers, Metal is non-blocking, p95 remains within gate, and no fresh protected-action/rollback/daemon failures appear.

- [ ] **Step 7: Record preliminary production verdict**

Label fewer than 500 new cycles preliminary. Report exactly what is deployed, current sample count, readiness blockers, p95, lane health, Pair Gold funnel, and any uncertainty. Automatic Active eligibility remains a daemon decision under the frozen gates; do not manually inflate evidence or AIS.
