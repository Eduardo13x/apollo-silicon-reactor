# Runtime Metrics Honesty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the confirmed runtime metric contradictions without weakening Apollo's safety gates, then build and deploy the corrected daemon and dashboard.

**Architecture:** Keep actuation policy unchanged except for removing the false foreground-starvation trigger. Add explicit evidence fields at existing boundaries so dashboard and doctor distinguish observations from mutations, contextual telemetry from authoritative learning, and partial capability from effective capability. Feed unified trust health into deliberation as a bounded confidence ceiling, never as new actuation authority.

**Tech Stack:** Rust 2021, serde runtime metrics, Cargo tests, launchd deployment.

## Global Constraints

- Preserve Chromium SIGSTOP disable and every protected-process/safety gate.
- Do not add blocking I/O or unbounded work to the daemon hot path.
- Preserve persistence compatibility with serde defaults.
- Do not increase NARS capacity or synthesize learning evidence.
- Use the existing guarded deployment helper and verify installed hashes.

---

### Task 1: Perceptual Latency Validity

**Files:**
- Modify: `crates/apollo-engine/src/engine/latency_monitor.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs`
- Modify: `crates/apollo-engine/src/engine/types.rs`
- Modify: `src/bin/apollo-optimizerctl/dashboard.rs`

**Interfaces:**
- Extend `LatencySignals` with explicit recent-user-activity and launch/window-operation evidence.
- Extend `LatencyScore` with `measured: bool` and component evidence.
- Publish `perceptual_latency_measured` and avoid boosting from idle-low-CPU alone.

- [ ] Add failing tests: quiet foreground without interaction is not starvation; launch plus low CPU remains actionable; M4-normal WindowServer load does not alone imply sluggishness.
- [ ] Run the focused latency tests and confirm RED.
- [ ] Implement validity gating and hardware-aware thresholds without changing pressure or safety policy.
- [ ] Run focused tests and confirm GREEN.

### Task 2: Honest QoS Accounting and Effective Capability

**Files:**
- Modify: `src/bin/apollo-optimizerd/metrics_reporter.rs`
- Modify: `crates/apollo-engine/src/engine/types.rs`
- Modify: `src/bin/apollo-optimizerd/socket_handler.rs`

**Interfaces:**
- Count `qos_*_count` only when `QoSOutcome::mutated` is true.
- Add cumulative QoS request/no-op/blocked telemetry using audited dispositions.
- Doctor reports effective task-QoS mutations, nice fallbacks, capability skips, and errors.

- [ ] Add failing tests proving cache no-ops cannot increment applied counters.
- [ ] Run focused daemon tests and confirm RED.
- [ ] Implement audited batch accounting and doctor projection.
- [ ] Run focused tests and confirm GREEN.

### Task 3: Unified Learning Projection and Deliberation Confidence

**Files:**
- Modify: `crates/apollo-engine/src/engine/unified_learning_health.rs`
- Modify: `src/bin/apollo-optimizerd/metrics_reporter.rs`
- Modify: `crates/apollo-engine/src/engine/world_model.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs`

**Interfaces:**
- `AdviceRecord` carries MAE, coverage, authoritative Gold count, and model identity.
- Trust inventory counts unique producer/action models, reports recovery progress, and fills worst coverage.
- Deliberation consumes a bounded unified-trust summary and cannot call itself grounded with zero validated/trusted models or degraded-only advice.

- [ ] Add failing tests for unique model counting, worst coverage, recovery progress, and trust-aware deliberation.
- [ ] Run focused engine tests and confirm RED.
- [ ] Implement the projection and confidence ceiling.
- [ ] Run focused tests and confirm GREEN.

### Task 4: Dashboard Semantics

**Files:**
- Modify: `src/bin/apollo-optimizerctl/dashboard.rs`

**Interfaces:**
- Label contextual Gold as `Ctx-local`, authoritative decisions as `Learn Gold`, and degraded models as `relearning` with progress.
- Render UX as `proxy unavailable` when no interaction/launch evidence exists.
- Label kernel thermal state and Chromium freeze policy explicitly.

- [ ] Add failing dashboard contract tests for every label.
- [ ] Run dashboard tests and confirm RED.
- [ ] Implement compact labels that fit existing width.
- [ ] Run dashboard tests and confirm GREEN.

### Task 5: Verification and Deployment

**Files:**
- Update: `.superpowers/sdd/2026-08-12-runtime-metrics-honesty/progress.md`

- [ ] Run affected engine and binary test suites in the single Cargo lane.
- [ ] Run `cargo fmt --all -- --check` and `git diff --check`.
- [ ] Perform one adversarial scenario review against this contract.
- [ ] Build release binaries once, sign staged candidates, and deploy with `/usr/local/sbin/apollo-deploy deploy`.
- [ ] Verify launchd state, installed hashes, cycles, failures, and a live before/after metric sample.
