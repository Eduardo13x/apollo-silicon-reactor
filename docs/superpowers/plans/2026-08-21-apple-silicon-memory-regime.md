# Apple Silicon Memory Regime Implementation Plan

**Spec:** `docs/superpowers/specs/2026-08-21-apple-silicon-memory-regime-design.md`

## Verification Discipline

- One Cargo process at a time against the existing shared target.
- RED tests precede production changes for each task.
- Focused tests during implementation.
- At task end: affected engine and daemon tests, `cargo fmt --all -- --check`,
  and `git diff --check` once.
- No deployment or daemon restart in this plan.

## Task 1: Shared Memory Evidence Model

**Files:**
- Create `crates/apollo-engine/src/engine/memory_regime.rs`
- Modify `crates/apollo-engine/src/engine/mod.rs`

1. Add RED tests for capability validation, RAM scaling equivalence, native
   page conversion, non-finite rejection, stale evidence, and every regime
   transition/hysteresis path.
2. Implement `MemoryCapabilities`, `MemoryObservation`,
   `NormalizedMemoryState`, `MemoryRegime`, `MemoryRegimeEvidence`, and a
   time-based `MemoryRegimeDetector`.
3. Keep policy constants dimensionless and private to one policy type.
4. Run the focused `memory_regime` tests once.

## Task 2: Collector Generation And Cached Derivation

**Files:**
- Modify `crates/apollo-engine/src/engine/background_collectors.rs`
- Modify `src/bin/apollo-optimizerd/daemon_init.rs`
- Modify `src/bin/apollo-optimizerd/main.rs`

1. Add RED tests proving each successful collection increments one generation
   and duplicate reads retain the same generation.
2. Add `generation: u64` to `PressureData` and increment it in the existing
   collector thread without adding another atomic, lock, or poll.
3. Detect immutable memory capabilities at daemon startup using existing
   hardware data.
4. Derive and cache normalized state only when generation changes.
5. Add a duplicate-generation test proving predictor/regime state is unchanged.

## Task 3: Time-Aware Swap Predictor

**Files:**
- Modify `crates/apollo-engine/src/engine/swap_predictor.rs`
- Modify `tests/level5_tier1_extended.rs`
- Modify `src/bin/apollo-optimizerd/main.rs`

1. Replace cycle-based tests with timestamped scenarios: 500 ms, 2 s, burst
   then flat, falling swap, dynamic swap-total changes, and sleep-sized gaps.
2. Store RAM-normalized samples in a preallocated fixed ring and maintain
   bounded running sums for O(1) slope updates.
3. Forecast by elapsed seconds and physical RAM, not sample count or dynamic
   swap capacity.
4. Preserve existing forecast fields for protocol compatibility.
5. Run focused predictor tests once.

## Task 4: Page-Accurate Reclaim Estimator

**Files:**
- Modify `crates/apollo-engine/src/engine/swap_reclaim.rs`
- Modify `src/bin/apollo-optimizerd/main.rs`
- Modify `crates/apollo-engine/src/engine/adversarial_probe.rs` as required by
  the input contract

1. Add RED tests for 4/16/64 KiB pages, equal normalized load across 8/16 GiB,
   duplicate generation, cadence-independent EMA, and invalid page size.
2. Add page size, physical RAM, generation, and elapsed time to the input.
3. Convert and normalize once; return the cached forecast on duplicate input.
4. Remove the hard-coded page-size policy constant from calculations.
5. Run focused reclaim tests once.

## Task 5: Governor And Survival Consumption

**Files:**
- Modify `crates/apollo-engine/src/engine/profile_governor.rs`
- Modify `crates/apollo-engine/src/engine/safety.rs`
- Modify `src/bin/apollo-optimizerd/daemon_survival_tick.rs`
- Modify `src/bin/apollo-optimizerd/main.rs`

1. Add RED matrix tests proving equivalent normalized evidence produces the
   same result on 8/16/24/32 GiB hosts.
2. Add tests that `Unknown`, one burst, stable swap, thermal constraint,
   anti-thrash lock, build mode, and kill-switch paths do not gain aggression.
3. Replace duplicated absolute swap formulas with shared regime/evidence.
4. Retain independent survival hard gates and all protected-process rules.
5. Run focused governor, safety, and survival tests once.

## Task 6: Hazard Episode Integrity

**Files:**
- Modify `crates/apollo-engine/src/engine/hazard_model.rs`
- Modify `crates/apollo-engine/src/engine/signal_intelligence.rs`
- Modify hazard callers only where return/evidence plumbing requires it

1. Add RED tests proving repeated observations within one episode count once,
   a later episode counts again, batch replay leaves event/time counts stable,
   and burst-then-flat output loses physical corroboration.
2. Separate physical-event accounting from beta-only replay.
3. Add feature schema metadata with backward-compatible defaults.
4. Preserve restored legacy weights and history; do not reset learned files.
5. Run focused hazard and signal-intelligence tests once.

## Task 7: Override Provenance And Lease

**Files:**
- Modify `crates/apollo-engine/src/engine/types.rs`
- Modify `crates/apollo-engine/src/engine/profile_governor.rs`
- Modify `src/bin/apollo-optimizerd/daemon_cognitive_tick.rs`
- Modify `src/bin/apollo-optimizerd/socket_handler.rs`
- Modify `src/bin/apollo-optimizerctl/dashboard.rs`
- Modify `src/bin/apollo-menubar.rs`
- Modify affected protocol fixtures and E2E tests

1. Add RED tests for legacy deserialization, operator precedence, predictive
   renewal, early release, expiry, status serialization, and UI labels.
2. Add `OverrideOrigin` with legacy default `Operator`.
3. Give predictive overrides a bounded renewable lease and release them when
   physical corroboration disappears.
4. Keep operator override behavior unchanged.
5. Run focused governor, cognitive, protocol, dashboard, and menubar tests.

## Task 8: Typed Sysctl Postconditions

**Files:**
- Modify `crates/apollo-engine/src/engine/sysctl_direct.rs`
- Modify `crates/apollo-engine/src/engine/execute_actions.rs`
- Modify `crates/apollo-engine/src/engine/mediator.rs` only for shared typed
  helpers/tests, without broad mediator migration

1. Add RED tests for binary 100 decoding as decimal 100, 4/8-byte widths,
   exact apply, no-op, coercion, unreadable post-state, timeout, and mismatch.
2. Add typed numeric read/write requests to the existing serial worker.
3. Mark an action applied only after exact requested-value confirmation.
4. Enroll delayed postcondition observation only after confirmed application.
5. Run focused sysctl, mediator, and execute-action tests once.

## Task 9: Batched Adversarial Review And Performance Gate

1. Re-read the complete diff against the acceptance matrix: variants, restart,
   stale state, bypass paths, capacity, root/non-root, and protected processes.
2. Batch any demonstrated correctness fixes into one pass.
3. Run the affected engine and daemon test passes once.
4. Run a focused release performance probe comparing duplicate and fresh sample
   paths; report raw p50/p95/max and allocation behavior.
5. Run `cargo fmt --all -- --check` and `git diff --check` once.
6. Show raw verification output to the user. Deployment remains a separate,
   explicitly approved phase.
