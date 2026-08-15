# Apollo Heterogeneous Capability Fabric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Apollo uses one Cargo lane and one production-code owner for shared contracts.

**Goal:** Make Apollo runtime-capability-driven and able to route bounded optional work across CPU, Metal, and Core ML without delaying reflexes or inflating AIS.

**Architecture:** Platform adapters publish versioned capabilities and bounded events into immutable world snapshots. A heterogeneous compute fabric executes optional jobs asynchronously; the existing arbiter and ReflexBroker remain the only action-authority path.

**Tech stack:** Rust 2021, macOS Mach/kqueue/IOKit/CoreAudio, Objective-C++ Metal/Core ML bridges, launchd, serde, bounded channels.

## Global Constraints

- Preserve the existing dirty worktree and all conservative action gates.
- No kext, DriverKit, private write API, raw screen/audio persistence, or content semantics.
- The control loop never waits for optional work; all queues and histories are bounded.
- Hardware policy uses capabilities, not chip names; macOS is the first adapter.
- GPU/Core ML/model output has advisory authority only; Pair Gold remains the only new AIS attribution.
- One portable Apple Silicon release build and one guarded deployment after integration.

## Frozen Contract

### Ownership and flow

1. `PlatformAdapter` owns capability discovery, platform events, fallback sensing, and actuator descriptions.
2. `EventMesh` owns ordering, deduplication, bounded overflow accounting, and source health.
3. `WorldStatePublisher` owns immutable epoch/revision snapshots and feature-vector publication.
4. `ComputeFabric` owns backend routing, deadlines, latest-wins replacement, circuit state, and result identity validation.
5. Models consume one snapshot and emit proposals; the existing arbiter, safety gates, mediator, and ReflexBroker retain authority.
6. Medallions observe all outcomes, but only local authoritative Pair Gold may affect AIS attribution.

### Bounds

- Event queue: 256 envelopes; coalesce replaceable source updates, count nonreplaceable drops.
- Snapshot feature vector: schema-versioned, finite `f32`, at most 256 values, three retained revisions.
- Compute registry: at most 32 job classes, four active optional jobs, one latest-wins slot per class/backend.
- Result: at most 64 KiB, exact epoch/revision/workload identity, finite values, declared deadline and maximum age.
- Context telemetry: numeric aggregates only; no frame, sample, title, text, path, bundle document, or media metadata persists.
- Core ML model: versioned feature schema, CPU oracle, bounded input/output, advisory only.

### Failure behavior

- Missing permission/capability becomes `Unavailable`; it never becomes a nominal zero.
- Sleep, wake, session change, capability revision, PID identity change, kill switch, thermal pressure, or low-power invalidates optional work.
- Worker spawn/disconnect/panic, stale completion, invalid shape, NaN/inf, timeout, or queue saturation cannot delay the control loop or create an action.
- Backend failures open a per-backend circuit; CPU/reflex remains available. Metal/Core ML use half-open probes after cooldown.
- A context-agent disconnect marks its observations stale and then unavailable; root behavior continues.
- Restore rejects future/corrupt/oversized state and downgrades unverifiable authority.

### Compatibility

- Existing `CapabilityReport`, `CycleContextSnapshot`, protocol fields, dashboard fields, and learned-state fields remain readable during migration.
- New fields use serde defaults; old clients continue to receive legacy fields until the final compatibility cleanup.
- The build baseline remains M1-compatible; worker count and backend availability are runtime decisions.

## Acceptance Matrix

### Deploy health

- Fresh AIS warmup cannot independently trigger rollback.
- Crash, zero progress, failures, last error, hash mismatch, or sustained post-warmup regression still blocks deployment.
- `/var` and `/private/var` backup aliases validate to the same scoped direct child.

### Capabilities and platform

- M1, M4, future simulated Apple Silicon, no-accelerator, non-root, unsupported OS, partial sensor, and denied-permission fixtures.
- Startup/reprobe/session/wake revision changes invalidate dependent work exactly once.

### Event and snapshot lifecycle

- Ordered, duplicate, out-of-order, overflow, stale, invalid, truncated, shutdown, restart, wake, and source-disconnect cases.
- Concurrent readers see a whole old or new snapshot, never mixed revisions.

### Compute lifecycle

- Every job class covers submitted, replaced, busy, completed, stale, timed-out, cancelled, failed, invalid, and fallback outcomes.
- Out-of-order results, recycled identity, old capability revision, backend death, circuit open/half-open/closed, and thermal cancellation.
- CPU, Metal, and Core ML oracle parity; no backend may invent an action or evidence closure.

### Privacy and persistence

- Raw frames/audio/titles/text/paths cannot enter structs marked serializable, logs, status, medallions, or learned state.
- Restart never resumes an in-flight perception or compute job.

### Promotion and performance

- Shadow: at least 500 eligible jobs and 15 minutes. Canary: 10% for another 500 eligible jobs.
- Promotion requires zero safety/rollback failures, 99% deadlines, oracle accuracy within one percentage point, and either 10% lower latency or 15% lower energy.
- Control-loop p95 is at most 75 ms and at most 10% worse than baseline; idle CPU is below 1%; RSS increase is at most 64 MiB.
- A backend rolls back independently; fresh AIS is not a deployment health signal.

## Implementation Tasks

1. Make deployment health warmup-aware and canonicalize rollback paths.
2. Add `CapabilityGraph`, `PlatformAdapter`, macOS and simulated adapters.
3. Add `EventMesh`, `WorldStateSnapshot`, bounded `FeatureStore`, and lifecycle invalidation.
4. Add authenticated numeric-only `apollo-context-agent` transport and launchd installation.
5. Add `ComputeBackend`, CPU lanes, router, promotion state, and bounded queues.
6. Replace permanent Metal quarantine with a circuit breaker, warmup, crossover calibration, and shared buffers.
7. Add Core ML temporal predictor bridge, CPU oracle, fallback, and truthful ANE evidence.
8. Feed snapshots/results into existing models, medallions, metrics, and dashboard without changing action/AIS authority.
9. Run focused and affected suites, one adversarial review/fix pass, workspace release verification, deploy once, and observe at least 1,000 fresh cycles.
