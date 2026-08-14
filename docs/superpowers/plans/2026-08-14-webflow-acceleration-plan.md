# Apollo WebFlow Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add privacy-preserving, exact Chromium navigation sensing and use it to drive Apollo's existing reversible acceleration path with measurable causal outcomes.

**Architecture:** A Manifest V3 extension emits bounded numeric lifecycle/vitals messages to an unprivileged native bridge. The user context agent authenticates and forwards those events to the daemon, where `WebFlowController` reconciles them with daemon inference, publishes a coherent web observation in `WorldStateSnapshot`, and supplies a bounded `WebNavigation` trigger to the existing reflex/lease authority.

**Tech Stack:** Rust 2021, serde/serde_json, Unix domain sockets, Chromium Native Messaging, Manifest V3 vanilla JavaScript, existing Apollo Event Mesh, World State, ReflexBroker, metrics/dashboard, shell deploy tests.

## Global Constraints

- No proxy, TLS interception, packet rewriting, DNS replacement, content cache, resource prefetch, or browser automation.
- No page text, DOM content, forms, credentials, cookies, titles, raw URL/origin, request/response content, image, or audio persistence/logging.
- Native messages are at most 16 KiB; active tabs and navigations are each capped at 64; site buckets at 256; WebFlow events drained per cycle at 128.
- Initial lease is 2,000 ms and total continuous WebFlow acceleration is capped at 15,000 ms.
- The browser and bridge run unprivileged. The browser never connects to a root endpoint.
- `ReflexBroker` remains the acceleration authority and `SysctlGovernor` remains the sole TCP sysctl writer.
- Missing extension/model evidence degrades to deterministic daemon inference and never blocks the main cycle.
- Browser extension installation requires explicit browser/user confirmation.
- Preserve all existing dirty-tree changes; never reset or overwrite unrelated work.
- Use one Cargo lane. Run focused tests during tasks and broad verification once at the final integration gate.

---

## File Structure

### New production files

- `crates/apollo-engine/src/engine/webflow_types.rs`: versioned wire types, bounds, validation, privacy-safe metrics, process-wide bounded ingress queue.
- `crates/apollo-engine/src/engine/webflow_controller.rs`: navigation state machine, deduplication, inference reconciliation, intents, rollout, counters, and bounded histories.
- `crates/apollo-engine/src/engine/webflow_native.rs`: native-message framing, token bucket, user socket path, and peer-validation helpers.
- `src/bin/apollo-web-bridge.rs`: thin stdin/stdout Native Messaging host forwarding validated frames to the user agent.
- `src/bin/apollo-optimizerd/webflow_tick.rs`: daemon-cycle adapter from ingress/events/system context to snapshot observation and acceleration hint.
- `extensions/apollo-webflow-chromium/manifest.json`: stable-ID Manifest V3 package with lifecycle and optional host permissions.
- `extensions/apollo-webflow-chromium/background.js`: lifecycle, origin HMAC bucket, native connection, bounded reconnect.
- `extensions/apollo-webflow-chromium/content.js`: numeric PerformanceObserver aggregation only.
- `extensions/apollo-webflow-chromium/protocol.js`: closed schema builders, range clamps, and message-size guard.
- `scripts/install-webflow-extension.sh`: install native-host manifests and print the explicit browser registration path.

### New test files

- `crates/apollo-engine/tests/webflow_types_contract.rs`
- `crates/apollo-engine/tests/webflow_controller_contract.rs`
- `crates/apollo-engine/tests/webflow_native_contract.rs`
- `tests/webflow_extension_contract.js`
- `tests/scripts/install-webflow-extension-test.sh`

### Existing files modified

- `crates/apollo-engine/src/engine/mod.rs`
- `crates/apollo-engine/src/engine/protocol.rs`
- `crates/apollo-engine/src/engine/event_mesh.rs`
- `crates/apollo-engine/src/engine/world_state.rs`
- `crates/apollo-engine/src/engine/context_agent.rs`
- `crates/apollo-engine/src/engine/reflex.rs`
- `crates/apollo-engine/src/engine/types.rs`
- `src/bin/apollo-context-agent.rs`
- `src/bin/apollo-optimizerd/main.rs`
- `src/bin/apollo-optimizerd/socket_handler.rs`
- `src/bin/apollo-optimizerd/value_scheduler_tick.rs`
- `src/bin/apollo-optimizerd/daemon_cycle_tail.rs`
- `src/bin/apollo-optimizerd/metrics_reporter.rs`
- `src/bin/apollo-optimizerctl/dashboard.rs`
- `scripts/build-release.sh`
- `scripts/hardware-build-profile.sh`
- `scripts/deploy.sh`
- `scripts/install-root-daemon.sh`
- `scripts/uninstall-root-daemon.sh`
- `scripts/apollo-accept-gate.sh`
- `scripts/apollo-deploy-gate.sh`
- `tests/scripts/hardware-build-profile-test.sh`
- `tests/scripts/apollo-deploy-test.sh`
- `tests/scripts/apollo-accept-gate-test.sh`

---

### Task 1: Versioned WebFlow Contracts and Controller

**Files:**
- Create: `crates/apollo-engine/src/engine/webflow_types.rs`
- Create: `crates/apollo-engine/src/engine/webflow_controller.rs`
- Create: `crates/apollo-engine/tests/webflow_types_contract.rs`
- Create: `crates/apollo-engine/tests/webflow_controller_contract.rs`
- Modify: `crates/apollo-engine/src/engine/mod.rs`

**Interfaces:**
- Produces: `WebFlowEvent::validate()`, `WebFlowIngress::accept/drain`, `WebFlowController::tick`, `WebFlowIntent`, `WebWorldObservation`, `WebFlowCounters`.
- Consumes: no daemon globals or macOS APIs; all time enters explicitly as monotonic milliseconds.

- [ ] **Step 1: Write failing contract tests**

```rust
#[test]
fn valid_started_event_produces_a_two_second_intent() {
    let mut controller = WebFlowController::new(WebFlowRolloutPhase::Shadow);
    let output = controller.tick(WebFlowTickInput::foreground_browser(1000), [started(1)]);
    assert_eq!(output.intent.unwrap().ttl_ms, 2_000);
}

#[test]
fn newer_navigation_abandons_the_previous_tab_episode() {
    let mut controller = WebFlowController::new(WebFlowRolloutPhase::Shadow);
    controller.tick(WebFlowTickInput::foreground_browser(1000), [started(1)]);
    let output = controller.tick(WebFlowTickInput::foreground_browser(1100), [started(2)]);
    assert_eq!(output.closed[0].closure, WebFlowClosure::Abandoned);
}
```

Cover every phase, lifecycle-only grace, vitals settle, failure, abandonment, tab/browser/session change, duplicate, out-of-order, stale, non-finite, overflow, lease renewal, 15-second hard cap, pressure/thermal/low-power cancellation, and extension-vs-inference deduplication.

- [ ] **Step 2: Verify tests fail for missing modules**

Run: `cargo test -p apollo-engine --test webflow_types_contract --test webflow_controller_contract`

Expected: compilation fails because WebFlow modules and types do not exist.

- [ ] **Step 3: Implement closed types and validation**

```rust
pub const WEBFLOW_SCHEMA_VERSION: u16 = 1;
pub const MAX_WEBFLOW_MESSAGE_BYTES: usize = 16 * 1024;
pub const MAX_ACTIVE_NAVIGATIONS: usize = 64;

pub struct WebFlowEvent {
    pub schema_version: u16,
    pub browser_session_id: OpaqueId,
    pub tab_session_id: OpaqueId,
    pub navigation_id: OpaqueId,
    pub sequence: u64,
    pub phase: WebFlowPhase,
    pub source: WebFlowSource,
    pub site_bucket: Option<OpaqueBucket>,
    pub metrics: WebFlowMetrics,
}
pub enum WebFlowPhase { Started, Committed, DomReady, Loaded, Settled, Failed, Abandoned }
pub enum WebFlowSource { ExtensionLifecycle, ExtensionVitals, DaemonInference }
pub struct WebFlowIntent {
    pub navigation_id: Option<OpaqueId>,
    pub confidence_q: u16,
    pub intensity_q: u16,
    pub ttl_ms: u32,
    pub reason: WebFlowReason,
}
```

Use fixed-size `[u8; 16]` opaque IDs, integer/fixed-point metrics, `#[serde(deny_unknown_fields)]`, explicit kebab-case names, and distinct `Option` values for unavailable measurements. `WebFlowIngress` uses `VecDeque` capacity 256 and drains at most 128 events.

- [ ] **Step 4: Implement deterministic state machine and rollout**

Use a keyed map bounded to 64 active navigations plus an expiry min-heap/timer ordering. Exact extension evidence replaces a matching inferred episode. `Started` opens 2,000 ms; progress renews; no episode exceeds 15,000 ms. Shadow creates intents but marks them non-admitted. Canary admission is deterministic from opaque navigation ID modulo ten.

- [ ] **Step 5: Run focused tests**

Run: `cargo test -p apollo-engine --test webflow_types_contract --test webflow_controller_contract`

Expected: all WebFlow contract tests pass.

- [ ] **Step 6: Commit task-level behavior**

```bash
git add crates/apollo-engine/src/engine/mod.rs crates/apollo-engine/src/engine/webflow_types.rs crates/apollo-engine/src/engine/webflow_controller.rs crates/apollo-engine/tests/webflow_types_contract.rs crates/apollo-engine/tests/webflow_controller_contract.rs
git commit -m "feat: add bounded WebFlow controller"
```

### Task 2: Event Mesh, World Snapshot, and Daemon Ingress

**Files:**
- Modify: `crates/apollo-engine/src/engine/protocol.rs`
- Modify: `crates/apollo-engine/src/engine/event_mesh.rs`
- Modify: `crates/apollo-engine/src/engine/world_state.rs`
- Modify: `src/bin/apollo-optimizerd/socket_handler.rs`
- Modify: `src/bin/apollo-optimizerd/value_scheduler_tick.rs`
- Create: `src/bin/apollo-optimizerd/webflow_tick.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs`
- Modify tests: `crates/apollo-engine/tests/world_state_contract.rs`

**Interfaces:**
- Consumes: `WebFlowEvent`, `WebFlowIngress`, `WebFlowController` from Task 1.
- Produces: `DaemonRequest::SubmitWebFlow`, `EventSource::WebFlow`, `WorldStateSnapshot.web`, `WebFlowCycleOutput`.

- [ ] **Step 1: Add failing protocol and snapshot tests**

```rust
#[test]
fn webflow_request_roundtrips_and_is_unprivileged_ingress() {
    let req = DaemonRequest::SubmitWebFlow { event: started(7) };
    assert!(matches!(roundtrip(&req), DaemonRequest::SubmitWebFlow { .. }));
    assert!(!req.is_privileged());
}

#[test]
fn wake_invalidates_web_observation() {
    let next = publish_after_wake(snapshot_with_web());
    assert!(next.web.is_none());
}
```

- [ ] **Step 2: Verify focused failures**

Run: `cargo test -p apollo-engine protocol world_state_contract`

Expected: missing request/source/snapshot fields fail compilation.

- [ ] **Step 3: Wire bounded socket ingress and Event Mesh payload**

Add `SubmitWebFlow { event: WebFlowEvent }`. The socket handler validates encoded length and event fields before `process_webflow_ingress().accept(event)`. Add `EventSource::WebFlow` and a bounded web payload. Web events are nonreplaceable except a newer metrics update for the same navigation/phase may coalesce.

- [ ] **Step 4: Publish coherent web observation**

Extend `WorldStateSnapshot` with `pub web: Option<WebWorldObservation>` and a `with_web` constructor. `ValueSchedulerTickInput` accepts the current observation and drains WebFlow Event Mesh input before publishing. Wake/session/capability revision clears the observation.

- [ ] **Step 5: Add one cycle adapter**

`webflow_tick::WebFlowRuntime::tick` drains at most 128 ingress events, adds low-confidence daemon inference only for a foreground Chromium family plus recent interaction/socket activity, runs the controller, and returns one observation plus at most one intent. It never scans all processes itself.

- [ ] **Step 6: Run focused tests and commit**

Run: `cargo test -p apollo-engine protocol world_state_contract webflow`

```bash
git add crates/apollo-engine/src/engine/protocol.rs crates/apollo-engine/src/engine/event_mesh.rs crates/apollo-engine/src/engine/world_state.rs crates/apollo-engine/tests/world_state_contract.rs src/bin/apollo-optimizerd/socket_handler.rs src/bin/apollo-optimizerd/value_scheduler_tick.rs src/bin/apollo-optimizerd/webflow_tick.rs src/bin/apollo-optimizerd/main.rs
git commit -m "feat: publish WebFlow world observations"
```

### Task 3: Native Bridge and Authenticated User-Agent Socket

**Files:**
- Create: `crates/apollo-engine/src/engine/webflow_native.rs`
- Create: `crates/apollo-engine/tests/webflow_native_contract.rs`
- Create: `src/bin/apollo-web-bridge.rs`
- Modify: `crates/apollo-engine/src/engine/context_agent.rs`
- Modify: `src/bin/apollo-context-agent.rs`
- Modify: `crates/apollo-engine/src/engine/mod.rs`

**Interfaces:**
- Consumes: validated `WebFlowEvent` and daemon `SubmitWebFlow` from Tasks 1-2.
- Produces: `read_native_frame`, `write_native_frame`, `webflow_agent_socket_path`, `ContextWebFlowServer`.

- [ ] **Step 1: Write failing framing, rate, disconnect, and peer tests**

```rust
#[test]
fn native_frame_rejects_oversize_before_allocating() {
    let header = ((MAX_WEBFLOW_MESSAGE_BYTES + 1) as u32).to_ne_bytes();
    assert_eq!(read_native_frame(&mut header.as_slice()).unwrap_err().kind(), ErrorKind::InvalidData);
}

#[test]
fn token_bucket_caps_sixty_four_events_per_second() {
    let mut bucket = EventTokenBucket::new(0);
    assert_eq!((0..65).filter(|_| bucket.admit(0)).count(), 64);
}
```

- [ ] **Step 2: Implement allocation-safe framing and local socket security**

Read exactly four native-endian length bytes, reject zero/oversize, then allocate only the accepted length. Resolve the per-user socket below `TMPDIR`, verify parent ownership, create mode `0600`, reject unsafe stale nodes, and verify peer UID with `getpeereid` on macOS.

- [ ] **Step 3: Extend context agent without blocking its sampler**

Run a nonblocking `UnixListener` beside the existing one-second sampler. Accept bounded frames, validate them again, forward each accepted event to the daemon, and retain no replay file or payload log. Browser disconnect and daemon absence drop bounded events.

- [ ] **Step 4: Implement thin native host**

`apollo-web-bridge` reads stdin frames, validates JSON, enforces 64 events/second, connects only to the user-agent socket, forwards one event, and returns a small numeric acknowledgment. It owns no state or actuator.

- [ ] **Step 5: Run focused tests and commit**

Run: `cargo test -p apollo-engine --test webflow_native_contract context_agent`

```bash
git add crates/apollo-engine/src/engine/mod.rs crates/apollo-engine/src/engine/context_agent.rs crates/apollo-engine/src/engine/webflow_native.rs crates/apollo-engine/tests/webflow_native_contract.rs src/bin/apollo-context-agent.rs src/bin/apollo-web-bridge.rs
git commit -m "feat: add WebFlow native bridge"
```

### Task 4: Chromium Extension with Lifecycle and Optional Vitals

**Files:**
- Create: `extensions/apollo-webflow-chromium/manifest.json`
- Create: `extensions/apollo-webflow-chromium/protocol.js`
- Create: `extensions/apollo-webflow-chromium/background.js`
- Create: `extensions/apollo-webflow-chromium/content.js`
- Create: `tests/webflow_extension_contract.js`

**Interfaces:**
- Produces native host messages matching Rust `WebFlowEvent` schema version 1.
- Consumes browser `webNavigation`, `storage`, `runtime.connectNative`, Navigation Timing, and PerformanceObserver only.

- [ ] **Step 1: Write Node contract tests**

Test schema rejection, clamping, 16 KiB cap, opaque IDs, lifecycle ordering, 30-day secret rotation, 256-bucket cap, reconnect backoff, aggregate-only resource timing, and absence of forbidden keys (`url`, `title`, `text`, `cookie`, `header`, `body`, `dom`).

Run: `node --test tests/webflow_extension_contract.js`

Expected: fail because extension modules do not exist.

- [ ] **Step 2: Implement lifecycle-only mode**

Manifest permissions are `webNavigation`, `storage`, and `nativeMessaging`; `<all_urls>` is optional host permission. The service worker emits only top-frame lifecycle phases, random opaque IDs, sequence, source, and bounded numeric error class. Raw URL is discarded after optional bucket derivation.

- [ ] **Step 3: Implement optional vitals mode**

The content script uses `PerformanceObserver` for LCP, event/INP, layout-shift/CLS, long tasks, and aggregate resource count/bytes. It never traverses DOM nodes or includes resource names. `Settled` requires 750 ms without a new resource entry or long task after load, capped at 10 seconds.

- [ ] **Step 4: Implement rotating site buckets and native reconnect**

Generate a nonextractable HMAC-SHA-256 key, rotate every 30 days, expose only 16 bytes of digest, cap local buckets at 256, and clear old bucket metadata at rotation. Use bounded exponential native reconnect from 250 ms to 10 seconds with no disk queue.

- [ ] **Step 5: Run tests and commit**

Run: `node --test tests/webflow_extension_contract.js`

```bash
git add extensions/apollo-webflow-chromium tests/webflow_extension_contract.js
git commit -m "feat: add privacy-safe WebFlow extension"
```

### Task 5: Reflex/Lease Actuation and Model Support

**Files:**
- Modify: `crates/apollo-engine/src/engine/reflex.rs`
- Modify: `src/bin/apollo-optimizerd/daemon_cycle_tail.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs`
- Modify: `crates/apollo-engine/src/engine/types.rs`
- Add focused tests in existing `daemon_cycle_tail.rs` test module and `crates/apollo-engine/tests/reflex_contract.rs`

**Interfaces:**
- Consumes: one fresh `WebFlowIntent` from Task 2.
- Produces: `ReflexTrigger::WebNavigation`, existing lease receipts with `webflow:` action-key attribution, WebFlow counters.

- [ ] **Step 1: Write failing authority and safety tests**

```rust
#[test]
fn web_navigation_uses_existing_closed_catalog() {
    let intents = decide_webflow_lanes(valid_intent(), safe_context());
    assert!(intents.iter().all(|lane| matches!(lane, TaskQos | Nice | IoRelease)));
}

#[test]
fn media_and_protected_targets_never_receive_harmful_web_actions() {
    let media = safety_context().with_media_active(true);
    let protected = safety_context().with_target_protected(true);
    for context in [media, protected] {
        let lanes = decide_webflow_lanes(valid_intent(), context);
        assert!(!lanes.iter().any(|lane| matches!(lane, Freeze | Throttle | Purge)));
    }
}
```

Cover PID reuse, Apple/protected target, active/minimized media, pressure, thermal, low-power, kill switch, existing owner conflict, no-op, renewal, release, rollback, restart, duplicate and 15-second hard cap.

- [ ] **Step 2: Add WebNavigation trigger to existing broker**

Extend `InteractionReason` and `ReflexTrigger` with `WebNavigation`. Pass the fresh controller intent into `update_acceleration_lease`; foreground Chromium target selection still uses current coalition/process-tree identity. The intent may tune TTL/intensity but cannot bypass `decide_acceleration_lanes`, target gates, effect ledger, live identity recheck, exploration reservation, or rollback.

- [ ] **Step 3: Preserve single TCP authority and protection behavior**

WebFlow may request a `SysctlGovernor` context reevaluation flag only. Add a mediator test proving no WebFlow module calls `sysctlbyname`, `sysctl -w`, freeze, throttle, purge, or memorystatus methods. Existing audio/video coalition protection remains stronger than WebFlow acceleration.

- [ ] **Step 4: Attach model support without authority inflation**

Use the same `ReasoningSnapshot` identity and maximum age of two cycles. World Model/NARS/Markov/MPC/GPU outputs can bound intensity/TTL/priority. Missing or stale advice is neutral. Record support separately from applied receipts.

- [ ] **Step 5: Run focused tests and commit**

Run: `cargo test -p apollo-engine --test reflex_contract webflow`

Run: `cargo test --bin apollo-optimizerd webflow`

```bash
git add crates/apollo-engine/src/engine/reflex.rs crates/apollo-engine/src/engine/types.rs src/bin/apollo-optimizerd/daemon_cycle_tail.rs src/bin/apollo-optimizerd/main.rs
git commit -m "feat: drive reflex acceleration from WebFlow"
```

### Task 6: Honest Metrics, Causal Closure, and Dashboard

**Files:**
- Modify: `crates/apollo-engine/src/engine/types.rs`
- Modify: `src/bin/apollo-optimizerd/metrics_reporter.rs`
- Modify: `src/bin/apollo-optimizerd/daemon_cycle_tail.rs`
- Modify: `src/bin/apollo-optimizerctl/dashboard.rs`
- Add tests in `crates/apollo-engine/tests/pair_gold_contract.rs`
- Add tests in the existing `#[cfg(test)]` modules of `crates/apollo-engine/src/engine/types.rs`, `src/bin/apollo-optimizerd/metrics_reporter.rs`, `src/bin/apollo-optimizerd/daemon_cycle_tail.rs`, and `src/bin/apollo-optimizerctl/dashboard.rs`

**Interfaces:**
- Consumes: controller counters, lifecycle outcomes, real effect receipts, existing Pair Gold pipeline.
- Produces: backward-compatible `RuntimeMetrics` WebFlow fields and three compact dashboard lines.

- [ ] **Step 1: Write failing legacy/default and dashboard tests**

Verify old JSON defaults every WebFlow field, no NaN serializes, observed/proposed/supported/applied/improved remain distinct, and dashboard renders `Web`, `Web+`, and `WebC` without line overflow.

- [ ] **Step 2: Add bounded metrics**

Add mode/connectivity/freshness, accepted/invalid/stale/duplicate/out-of-order/rate-limit/drop counters, proposal/admission/apply/skip/veto/revert/fail counters, timing distributions, causal pair/Gold quality, phase and exact blocker. Histories remain fixed-size.

- [ ] **Step 3: Close causal outcomes honestly**

Only paired real applied/control episodes with matching browser family, available site bucket, pressure/power/cache/resource bands and complete endpoints can close Gold. Confounded, interrupted, media-heavy, background, stale, inferred-only, model-only, or no-op episodes cannot affect Pair Gold or AIS.

- [ ] **Step 4: Run focused tests and commit**

Run: `cargo test -p apollo-engine --test pair_gold_contract runtime_metrics`

Run: `cargo test --bin apollo-optimizerctl webflow`

```bash
git add crates/apollo-engine/src/engine/types.rs crates/apollo-engine/tests/pair_gold_contract.rs src/bin/apollo-optimizerd/metrics_reporter.rs src/bin/apollo-optimizerd/daemon_cycle_tail.rs src/bin/apollo-optimizerctl/dashboard.rs
git commit -m "feat: report WebFlow outcomes honestly"
```

### Task 7: Packaging, Installation, Deploy Gates, and Uninstall

**Files:**
- Create: `scripts/install-webflow-extension.sh`
- Create: `tests/scripts/install-webflow-extension-test.sh`
- Modify: `scripts/build-release.sh`
- Modify: `scripts/hardware-build-profile.sh`
- Modify: `scripts/deploy.sh`
- Modify: `scripts/install-root-daemon.sh`
- Modify: `scripts/uninstall-root-daemon.sh`
- Modify: `scripts/apollo-accept-gate.sh`
- Modify: `scripts/apollo-deploy-gate.sh`
- Modify: `tests/scripts/hardware-build-profile-test.sh`
- Modify: `tests/scripts/apollo-deploy-test.sh`
- Modify: `tests/scripts/apollo-accept-gate-test.sh`

**Interfaces:**
- Produces: signed `apollo-web-bridge`, stable extension bundle, Native Messaging manifests for Brave/Chrome/Edge, manifest hashes, guarded install/uninstall.

- [ ] **Step 1: Write failing shell contract tests**

Verify artifact absence fails preflight, hashes cover bridge/extension/native manifests, install destinations are user/browser correct, rollback is per artifact, uninstall removes only Apollo-owned files, and scripts never silently enable/load the extension.

- [ ] **Step 2: Add portable build artifacts**

Build/sign `apollo-web-bridge`, copy extension files unchanged, derive the stable extension ID from the committed manifest key, and generate each browser Native Messaging manifest with only `chrome-extension://<derived-id>/` in `allowed_origins`.

- [ ] **Step 3: Add guarded install and deploy behavior**

Install the bridge in `/usr/local/libexec`, native manifests in each detected browser's per-user NativeMessagingHosts directory, and extension bundle under Apollo's shared assets. Print the local extension path requiring explicit confirmation. Do not modify browser preference databases.

- [ ] **Step 4: Extend deploy evidence and rollback scope**

Preflight verifies codesign, SHA-256, schema compatibility, agent socket health, and native bridge self-check. A WebFlow failure rolls back WebFlow artifacts/config first; the daemon rolls back only under existing crash/corruption/unsafe-actuation policy.

- [ ] **Step 5: Run script tests and commit**

Run: `bash tests/scripts/install-webflow-extension-test.sh`

Run: `bash tests/scripts/hardware-build-profile-test.sh`

Run: `bash tests/scripts/apollo-deploy-test.sh`

```bash
git add scripts tests/scripts
git commit -m "build: package WebFlow browser integration"
```

### Task 8: Integrated Adversarial Verification and Production Shadow Deploy

**Files:**
- Update: `.superpowers/sdd/2026-08-14-webflow-acceleration/progress.md`
- Modify only demonstrated P0/P1 gaps from the single adversarial review.

**Interfaces:**
- Consumes every prior task.
- Produces one verified release candidate and production Shadow evidence.

- [ ] **Step 1: Run one batched adversarial review**

Check all phase variants, browser matrices, permissions, lifecycle invalidations, queues, bounds, PID identity, media, bypass paths, model staleness, causal inflation, installation ownership, restart, sleep/wake, corrupt inputs, and rollback. Batch demonstrated findings into one fix pass.

- [ ] **Step 2: Run affected-crate and binary verification once**

Run: `cargo test -p apollo-engine`

Run: `cargo test --bin apollo-optimizerd --bin apollo-optimizerctl --bin apollo-context-agent --bin apollo-web-bridge`

Run: `node --test tests/webflow_extension_contract.js`

Run: `cargo fmt --all -- --check`

Run: `git diff --check`

Expected: every command exits zero and no Cargo command overlaps another.

- [ ] **Step 3: Build one portable release**

Run: `./scripts/build-release.sh`

Expected: daemon, ctl, context agent, web bridge, extension bundle, native manifests, model assets, and hashes exist in the isolated release directory with an M1-compatible Apple Silicon baseline.

- [ ] **Step 4: Present the exact deploy command and execute after the existing explicit production gate**

Run: `./scripts/apollo-deploy-gate.sh`

Do not roll back solely because AIS is fresh or preliminary. Do roll back under existing crash, corruption, unsafe actuation, identity, media interruption, or sustained control-loop regression gates.

- [ ] **Step 5: Verify production Shadow evidence**

Show raw `launchctl print` state for daemon/context agent, bridge self-check, extension connection mode, fresh runtime metrics, and journal excerpts. The initial verdict remains preliminary until at least 500 eligible Shadow navigations and 15 minutes exist.

- [ ] **Step 6: Commit any final mechanically required fix and stop**

Do not expand into Safari, Firefox, proxying, prefetch, or semantic page understanding.
