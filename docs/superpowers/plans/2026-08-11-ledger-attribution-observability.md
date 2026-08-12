# Ledger Attribution Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish Apollo's existing unattributed-applied receipt counter in runtime metrics and the dashboard.

**Architecture:** Carry one bounded `u64` from `DecisionLedger` through the cached `UnifiedLearningHealth` projection into `RuntimeMetrics`. Render the scalar as an audit line without feeding it into any decision, score, or authority path.

**Tech Stack:** Rust, serde, Cargo tests, macOS launchd deployment scripts.

## Global Constraints

- Preserve all existing JSON fields and default missing historical values to zero.
- Keep GPU, World Model, MPC, NARS, and learning health advisory semantics unchanged.
- Add no kernel action, persistence writer, blocking I/O, collection, or unbounded work.
- Keep publication O(1) and dashboard text bounded to `CW`.

---

### Task 1: Publish The Ledger Audit Counter

**Files:**
- Modify: `crates/apollo-engine/src/engine/types.rs`
- Modify: `crates/apollo-engine/src/engine/unified_learning_health.rs`
- Modify: `crates/apollo-engine/tests/unified_learning_health_contract.rs`
- Modify: `src/bin/apollo-optimizerd/metrics_reporter.rs`

**Interfaces:**
- Consumes: `DecisionLedger::unattributed_applied_total() -> u64`
- Produces: `RuntimeMetrics::decision_ledger_unattributed_applied_total: u64`

- [ ] **Step 1: Write failing compatibility and publication tests**

Add literal assertions that legacy JSON defaults the field to `0`, and that a
`UnifiedLearningInput` containing `7` publishes exactly `7` to
`RuntimeMetrics`.

- [ ] **Step 2: Run the focused contract test and verify RED**

Run: `cargo test -p apollo-engine --test unified_learning_health_contract`

Expected: compilation fails because the new input and runtime fields do not
exist yet.

- [ ] **Step 3: Add the minimal projection**

Add the scalar to `RuntimeMetrics`, `UnifiedLearningInput`, and
`UnifiedLearningHealth`; apply `#[serde(default)]` only to the `RuntimeMetrics`
field for historical JSON compatibility. Copy it in `from_input` and
`publish_to`. In `build_unified_learning_health`, set it from
`ledger.unattributed_applied_total()`.

- [ ] **Step 4: Run focused engine and daemon tests**

Run:
`cargo test -p apollo-engine --test unified_learning_health_contract`

Run:
`cargo test --bin apollo-optimizerd metrics_reporter::tests`

Expected: all selected tests pass.

### Task 2: Render The Audit Counter

**Files:**
- Modify: `src/bin/apollo-optimizerctl/dashboard.rs`

**Interfaces:**
- Consumes: `RuntimeMetrics::decision_ledger_unattributed_applied_total`
- Produces: fourth learning-band line `Ledger huérfanos N` for schema 2+, or
  `Ledger huérfanos n/d` for older/uninitialized metrics

- [ ] **Step 1: Write the failing dashboard test**

Cover schema 1 as unavailable, schema 2 with literal `3`, and `u64::MAX`; assert
the fourth line has the expected value and every line has display width at most
`CW`.

- [ ] **Step 2: Run the focused dashboard test and verify RED**

Run:
`cargo test --bin apollo-optimizerctl unified_learning_band_`

Expected: the fourth line is absent or reports a false numeric zero for schema
1 metrics.

- [ ] **Step 3: Add the bounded audit line**

Gate on `unified_learning_schema_version`: render the scalar only for schema 2+
and render `Ledger huérfanos n/d` otherwise, before applying
`bounded_learning_line`.

- [ ] **Step 4: Run dashboard tests**

Run: `cargo test --bin apollo-optimizerctl`

Expected: all dashboard and CLI tests pass.

### Task 3: Verify, Commit, Publish, And Deploy

**Files:**
- Verify all modified files from Tasks 1 and 2.

**Interfaces:**
- Consumes: signed `adaptive-multicore` release binaries.
- Produces: matching Git `main`, installed hashes, advancing daemon, and live zero/nonzero audit metric.

- [ ] **Step 1: Run the serial verification lane**

Run `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets`,
`cargo test --workspace --release --no-fail-fast`, and
`cargo build --workspace --bins --release --features adaptive-multicore`.

- [ ] **Step 2: Commit and push**

Commit the spec/plan separately, commit the tested implementation, then push
the feature branch and fast-forward `origin/main`.

- [ ] **Step 3: Deploy with backup**

Sign fixed-path candidates, capture pre-deploy status and installed hashes,
invoke the scoped root helper, and retain its emitted backup path.

- [ ] **Step 4: Verify production**

Require launchd running, cycles advancing, installed hashes matching signed
candidates, `failures == 0`, `last_error == null`, acceptance gate PASS, and a
live `decision_ledger_unattributed_applied_total` value.
