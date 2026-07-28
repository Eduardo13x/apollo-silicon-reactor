# M4 Evolutionary Context Trust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bounded, installation-local telemetry trust boundary that blocks dirty data before it reaches Apollo's World Model and lets reversible M4 acceleration gain or lose authority from fresh evidence without sacrificing stability or fluidity.

**Architecture:** Add a one-time installation identity and a pure `O(F)` context-admission classifier. `TelemetryMedallion` becomes the sole owner of Rejected/Silver/Gold authority, persistence sanitization, and Gold-only action endpoints; `WorldModel` consumes an explicit trusted view and falls back to the baseline whenever the current context or utility evidence is immature. Metrics and the dashboard expose context health and evolutionary phase separately from action readiness.

**Tech Stack:** Rust 2021, serde, existing Apollo engine/daemon/CLI crates, Cargo tests, Graphify, macOS launchd deployment gate.

---

## Scope And Working-Tree Discipline

The current worktree contains deployed but uncommitted M4/World Model work,
including the currently untracked `telemetry_medallion.rs`. Treat it as the
authoritative baseline. Never reset or discard it. Do not stage `.DS_Store`,
`manual-teach-m4-policy.json`, or `scripts/__pycache__/`.

The implementation is one integrated subsystem: admission, persistence,
World Model authority, observability, and deployment all enforce the same trust
contract. Splitting them into independent releases would temporarily leave a
bypass path.

## File Map

- Create `crates/apollo-engine/src/engine/installation_identity.rs`: persistent,
  random installation identity with no hardware serial dependency.
- Create `crates/apollo-engine/src/engine/telemetry_context_admission.rs`: pure
  context validation, tiering, fixed reason set, and quality calculation.
- Modify `crates/apollo-engine/src/engine/mod.rs`: register both focused modules.
- Modify `crates/apollo-engine/src/engine/daemon_helpers.rs`: expose the
  environment-aware installation identity path.
- Modify `crates/apollo-engine/src/engine/telemetry_medallion.rs`: integrate
  admission, Gold-only side effects, origin-bound evidence, bounded metrics,
  and sanitized persistence.
- Modify `crates/apollo-engine/src/engine/world_model.rs`: consume a trusted
  medallion view and expose evolutionary authority phase.
- Modify `crates/apollo-engine/src/engine/types.rs`: serialize runtime admission
  and authority metrics.
- Modify `src/bin/apollo-optimizerd/main.rs`: load identity once, pass it to
  restore/observation, and attach trusted views.
- Modify `src/bin/apollo-optimizerd/metrics_reporter.rs`: publish new metrics.
- Modify `src/bin/apollo-optimizerctl/dashboard.rs`: render honest context and
  calibration state without ambiguous `0/0` output.
- Modify `docs/acceptance-criteria.md`: replace M1-only interpretation with
  capability-derived M4 canary criteria for this feature.
- Modify `README.md`: document the new runtime metric meanings.

### Task 0: Freeze The Authoritative Baseline

**Files:**
- Inspect: all currently modified Rust files
- Exclude: `.DS_Store`, `manual-teach-m4-policy.json`, `scripts/__pycache__/`

- [ ] **Step 1: Record the baseline status and diff summary**

Run:

```bash
git status --short
git diff --stat
git ls-files --others --exclude-standard
```

Expected: the known M4/World Model source changes are present; the three
excluded artifact paths remain unstaged.

- [ ] **Step 2: Verify the authoritative baseline before committing it**

Run serially:

```bash
cargo fmt --all -- --check
cargo test --workspace
```

Expected: formatting passes and the full baseline suite passes. If a baseline
test fails, use `superpowers:systematic-debugging`; do not proceed by weakening
or deleting the failing test.

- [ ] **Step 3: Commit only the existing source baseline**

Stage the current Rust source changes explicitly, including the new medallion,
then inspect the index:

```bash
git add crates/apollo-engine/src/engine/causal_graph.rs crates/apollo-engine/src/engine/daemon_metrics_history.rs crates/apollo-engine/src/engine/data_medallion.rs crates/apollo-engine/src/engine/effect_ledger.rs crates/apollo-engine/src/engine/fluidity.rs crates/apollo-engine/src/engine/intelligence_score.rs crates/apollo-engine/src/engine/learned_state.rs crates/apollo-engine/src/engine/learning_pipeline.rs crates/apollo-engine/src/engine/mod.rs crates/apollo-engine/src/engine/predictive_agent.rs crates/apollo-engine/src/engine/profile_governor.rs crates/apollo-engine/src/engine/sysctl_governor.rs crates/apollo-engine/src/engine/telemetry_logger.rs crates/apollo-engine/src/engine/telemetry_medallion.rs crates/apollo-engine/src/engine/types.rs crates/apollo-engine/src/engine/usage_model.rs crates/apollo-engine/src/engine/world_model.rs crates/apollo-engine/tests/level3_arousal_decay_stress.rs src/bin/apollo-optimizerctl/dashboard.rs src/bin/apollo-optimizerd/daemon_agent_actions.rs src/bin/apollo-optimizerd/daemon_cognitive_tick.rs src/bin/apollo-optimizerd/daemon_cycle_tail.rs src/bin/apollo-optimizerd/daemon_dispatch_tick.rs src/bin/apollo-optimizerd/daemon_markov_tick.rs src/bin/apollo-optimizerd/learning_tick.rs src/bin/apollo-optimizerd/main.rs src/bin/apollo-optimizerd/metrics_reporter.rs
git diff --cached --check
git diff --cached --stat
git commit -m "feat: preserve current M4 world model baseline"
```

Expected: one baseline commit containing only source/test files. The excluded
artifacts remain visible in `git status --short` and uncommitted.

### Task 1: Add A Private Installation Identity

**Files:**
- Create: `crates/apollo-engine/src/engine/installation_identity.rs`
- Modify: `crates/apollo-engine/src/engine/mod.rs`
- Modify: `crates/apollo-engine/src/engine/daemon_helpers.rs`
- Test: inline tests in `installation_identity.rs`

- [ ] **Step 1: Write failing identity tests**

Add tests proving creation, stable reload, nonzero enforcement, and owner-only
permissions:

```rust
#[test]
fn creates_once_and_reloads_the_same_nonzero_id() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("installation_id");
    let mut first_entropy = &0x1020_3040_5060_7080_u64.to_le_bytes()[..];
    let first = load_or_create_from(&path, &mut first_entropy).unwrap();
    let mut ignored_entropy = &0x8877_6655_4433_2211_u64.to_le_bytes()[..];
    let second = load_or_create_from(&path, &mut ignored_entropy).unwrap();
    assert_eq!(first, InstallationId(0x1020_3040_5060_7080));
    assert_eq!(second, first);
    #[cfg(unix)]
    assert_eq!(std::fs::metadata(path).unwrap().permissions().mode() & 0o777, 0o600);
}

#[test]
fn zero_entropy_is_rejected_instead_of_creating_portable_authority() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("installation_id");
    let mut zero = &[0_u8; 8][..];
    assert_eq!(
        load_or_create_from(&path, &mut zero).unwrap_err().kind(),
        std::io::ErrorKind::InvalidData
    );
    assert!(!path.exists());
}
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cargo test -p apollo-engine installation_identity::tests -- --nocapture
```

Expected: FAIL because the module and symbols do not exist.

- [ ] **Step 3: Implement the identity module**

Use this public contract and a private injectable entropy reader:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InstallationId(pub u64);

impl InstallationId {
    pub const UNKNOWN: Self = Self(0);
    pub fn is_known(self) -> bool { self.0 != 0 }
}

pub fn load_or_create(path: &Path) -> io::Result<InstallationId> {
    let mut entropy = File::open("/dev/urandom")?;
    load_or_create_from(path, &mut entropy)
}
```

`load_or_create_from` must:

1. Parse an existing file as exactly 16 lowercase/uppercase hexadecimal digits.
2. Reject zero and malformed values.
3. Read exactly eight bytes from entropy for a missing file.
4. Create the parent directory if needed.
5. Use `OpenOptionsExt::mode(0o600)` and `create_new(true)`.
6. Write `format!("{:016x}\n", id.0)`, call `sync_all`, and handle an
   `AlreadyExists` race by reading the winner's file.

Add to `daemon_helpers.rs`:

```rust
pub fn installation_id_path() -> &'static str {
    if cfg!(test) { "/tmp/apollo-installation-id-test" }
    else { "/var/lib/apollo/installation_id" }
}
```

- [ ] **Step 4: Run identity tests and formatting**

Run:

```bash
cargo test -p apollo-engine installation_identity::tests -- --nocapture
cargo fmt --all -- --check
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/apollo-engine/src/engine/installation_identity.rs crates/apollo-engine/src/engine/mod.rs crates/apollo-engine/src/engine/daemon_helpers.rs
git commit -m "feat(learning): bind model authority to installation"
```

### Task 2: Build The Pure Context Admission Classifier

**Files:**
- Create: `crates/apollo-engine/src/engine/telemetry_context_admission.rs`
- Modify: `crates/apollo-engine/src/engine/mod.rs`
- Test: inline tests in `telemetry_context_admission.rs`

- [ ] **Step 1: Write failing tests for malformed and impossible context**

Create a `clean_context()` fixture representing this Mac's 4P/6E/16 GiB
regime, then add table-driven mutations:

```rust
const LOCAL_ID: InstallationId = InstallationId(0x1020_3040_5060_7080);

fn clean_context() -> TelemetryContextSummary {
    TelemetryContextSummary {
        cycle: 10,
        timestamp_unix: Utc::now().timestamp(),
        workload: "idle".to_string(),
        memory_pressure: 0.25,
        memory_pressure_raw: 0.25,
        compressor_pressure: 0.10,
        cpu_global_usage: 0.20,
        cpu_mean_busy: 0.20,
        cpu_max_busy: 0.40,
        cpu_pegged_fraction: 0.0,
        cpu_core_count: 10,
        used_ram_fraction: 0.50,
        total_ram_bytes: 16 * 1024 * 1024 * 1024,
        used_ram_bytes: 8 * 1024 * 1024 * 1024,
        free_ram_bytes: 8 * 1024 * 1024 * 1024,
        swap_total_bytes: 4 * 1024 * 1024 * 1024,
        disk_count: 1,
        disk_total_bytes: 1_000_000_000_000,
        disk_available_bytes: 500_000_000_000,
        thermal_score: 0.0,
        fluidity_score: 0.95,
        effective_profile: "balanced-root".to_string(),
        pressure_dominant_factor: "memory".to_string(),
        collector_pressure_alive: true,
        reactor_healthy: true,
        p_core_count: 4,
        e_core_count: 6,
        signal_pressure_smooth: 0.25,
        signal_p_oom_30s: 0.01,
        signal_urgency: 0.10,
        signal_entropy_anomaly: 0.0,
        signal_transformer_anomaly: 0.0,
        arousal_level: 0.10,
        markov_prediction_confidence: 0.0,
        markov_prediction_eta_secs: 0.0,
        predictive_intervention: "Observe".to_string(),
        ..TelemetryContextSummary::default()
    }
}

fn classify_live(context: &TelemetryContextSummary) -> ContextAdmission {
    classify(ContextAdmissionInput::live(
        context,
        None,
        context.timestamp_unix,
        LOCAL_ID,
    ))
}

fn assert_rejected(context: TelemetryContextSummary, reason: ContextReason) {
    let admission = classify_live(&context);
    assert_eq!(admission.tier, ContextTier::Rejected);
    assert!(admission.reasons.contains(reason));
}

#[test]
fn rejects_every_non_finite_required_family() {
    let mutations: &[fn(&mut TelemetryContextSummary)] = &[
        |c| c.memory_pressure = f64::NAN,
        |c| c.cpu_global_usage = f64::INFINITY,
        |c| c.fluidity_score = f64::NEG_INFINITY,
        |c| c.signal_pressure_smooth = f64::NAN,
        |c| c.natural_drift = f64::NAN,
        |c| c.markov_prediction_confidence = f64::NAN,
    ];
    for mutate in mutations {
        let mut context = clean_context();
        mutate(&mut context);
        let admission = classify(ContextAdmissionInput::live(
            &context, None, context.timestamp_unix, LOCAL_ID,
        ));
        assert_eq!(admission.tier, ContextTier::Rejected);
        assert!(admission.reasons.contains(ContextReason::NonFinite));
    }
}

#[test]
fn rejects_cross_signal_and_temporal_contradictions() {
    let mut context = clean_context();
    context.used_ram_bytes = context.total_ram_bytes + 1;
    assert_rejected(context, ContextReason::Coherence);

    let mut context = clean_context();
    context.swap_used_bytes = context.swap_total_bytes + 1;
    assert_rejected(context, ContextReason::Coherence);

    let previous = clean_context();
    let mut regressed = clean_context();
    regressed.cycle = previous.cycle - 1;
    let admission = classify(ContextAdmissionInput::live(
        &regressed, Some(&previous), regressed.timestamp_unix, LOCAL_ID,
    ));
    assert_eq!(admission.tier, ContextTier::Rejected);
    assert!(admission.reasons.contains(ContextReason::Temporal));
}

#[test]
fn optional_sensor_absence_does_not_block_gold() {
    let mut context = clean_context();
    context.p_cluster_temp_c = None;
    context.e_cluster_temp_c = None;
    context.gpu_temp_c = None;
    context.ane_watts = None;
    context.ane_util_pct = None;
    context.collector_smc_alive = false;
    assert_eq!(classify_live(&context).tier, ContextTier::Gold);
}

#[test]
fn missing_required_live_authority_is_silver() {
    let mut context = clean_context();
    context.collector_pressure_alive = false;
    assert_eq!(classify_live(&context).tier, ContextTier::Silver);

    let context = clean_context();
    let admission = classify(ContextAdmissionInput::live(
        &context,
        None,
        context.timestamp_unix,
        InstallationId::UNKNOWN,
    ));
    assert_eq!(admission.tier, ContextTier::Silver);
}
```

- [ ] **Step 2: Run classifier tests and verify RED**

Run:

```bash
cargo test -p apollo-engine telemetry_context_admission::tests -- --nocapture
```

Expected: FAIL because admission types and `classify` are absent.

- [ ] **Step 3: Implement fixed admission types**

Use fixed, serializable types with no reason strings or growable collections:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextTier { #[default] Rejected, Silver, Gold }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ContextReason {
    NonFinite = 0,
    OutOfRange = 1,
    InvalidIdentity = 2,
    Stale = 3,
    FutureTimestamp = 4,
    Temporal = 5,
    ForeignHardware = 6,
    Coherence = 7,
    RequiredCollector = 8,
    UnknownInstallation = 9,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContextReasonSet(u16);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextAdmission {
    pub tier: ContextTier,
    pub quality: f64,
    pub reasons: ContextReasonSet,
    pub hardware_regime: HardwareRegime,
    pub installation_id: InstallationId,
    pub local_epoch: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextReasonCounters {
    pub non_finite: u64,
    pub out_of_range: u64,
    pub identity: u64,
    pub stale: u64,
    pub temporal: u64,
    pub foreign_hardware: u64,
    pub coherence: u64,
    pub collector: u64,
    pub unknown_installation: u64,
}
```

Provide `insert`, `contains`, and category predicates on `ContextReasonSet`.
Provide `ContextReasonCounters::record(ContextReasonSet)` using one saturating
increment per category; one sample may increment multiple truthful categories.
`classify` must use fixed arrays/slices and helper predicates, never a `Vec`,
map, sort, regex, subprocess, or I/O.

- [ ] **Step 4: Implement exact validation policy**

Apply these rules:

```rust
const MAX_LIVE_AGE_SECS: i64 = 30;
const MAX_FUTURE_SKEW_SECS: i64 = 5;
const MAX_CONTEXT_TEXT_BYTES: usize = 64;
const MIN_TEMP_C: f64 = -20.0;
const MAX_TEMP_C: f64 = 150.0;
const MAX_COMPONENT_WATTS: f64 = 500.0;
```

- Required finite fractions in `[0,1]`: memory pressure/raw/compressor,
  CPU global/mean/max/pegged, stall, used RAM, thermal, fluidity,
  top-process CPU, WindowServer CPU, smooth pressure, OOM probability, urgency,
  entropy anomaly, transformer anomaly, pressure boost, arousal, and Markov
  confidence.
- Required finite signed values: pressure velocity, swap delta, natural drift,
  and NARS drift. Thrashing, refault rate, retransmits-per-k, listen-drop rate,
  Markov ETA, and user idle must be finite and non-negative.
- Optional temperatures are finite and within `[-20,150]` when present.
- Optional component watts are finite and within `[0,500]` when present;
  battery watts remains finite signed data.
- Optional utilization is within `[0,1]` for cluster utilization and `[0,100]`
  for ANE percentage.
- Battery percentage is at most 100 when present; battery watts is finite signed
  data because charging direction may be represented by sign.
- Workload/effective profile/dominant factor are nonempty and at most 64 bytes;
  foreground app is at most 256 bytes when present.
- Total RAM and total core count are nonzero; used/free RAM do not exceed total;
  swap used does not exceed a nonzero swap total; disk available does not exceed
  disk total; top-process RSS does not exceed aggregate process RSS.
- Capability P+E count equals snapshot CPU count when both are known.
- Live timestamp is no more than 30 seconds old or 5 seconds in the future.
- Within one epoch, cycle cannot decrease and timestamp cannot decrease.
- Optional sensor absence and dead SMC do not block Gold.
- Dead pressure collector, unhealthy reactor, unknown installation ID, or
  unknown hardware regime yields Silver after structural validation.
- Any hard reason yields Rejected. Otherwise unresolved degradation reasons
  yield Silver. Otherwise yield Gold.

Quality is a finite weighted score: structural `0.40`, temporal `0.20`, hardware
and origin `0.20`, required collector/reactor health `0.15`, optional coverage
`0.05`. Clamp to `[0,1]`; rejected quality cannot exceed `0.40`.

- [ ] **Step 5: Add boundary and complexity tests**

Add tests for exact accepted boundaries, stale/future time, hardware mismatch,
same-second timestamps, optional values, reason-bit stability, and one million
classifications with invariant fixed reason storage:

```rust
#[test]
fn classifier_has_fixed_storage_and_no_history_growth() {
    assert_eq!(std::mem::size_of::<ContextReasonSet>(), 2);
    let context = clean_context();
    for _ in 0..1_000_000 {
        assert_eq!(classify_live(&context).tier, ContextTier::Gold);
    }
}
```

Do not assert a wall-clock duration in unit tests; the structural no-allocation
contract and live p95 canary provide stable performance evidence.

- [ ] **Step 6: Run classifier tests and commit**

```bash
cargo test -p apollo-engine telemetry_context_admission::tests -- --nocapture
cargo fmt --all -- --check
git add crates/apollo-engine/src/engine/telemetry_context_admission.rs crates/apollo-engine/src/engine/mod.rs
git commit -m "feat(world-model): classify trusted live context"
```

Expected: PASS.

### Task 3: Make Medallion Side Effects Gold-Only

**Files:**
- Modify: `crates/apollo-engine/src/engine/telemetry_medallion.rs`
- Test: inline `telemetry_medallion::tests`

- [ ] **Step 1: Write failing authority tests**

Extend the existing test helpers with an owned fixture so fields can be mutated
before the observation borrows them:

```rust
struct ObservationFixture {
    snapshot: SystemSnapshot,
    runtime: RuntimeMetrics,
    capabilities: CapabilityReport,
    signal: SignalDigest,
    outcomes: ExecuteOutcomes,
    cycle: u64,
}

impl ObservationFixture {
    fn clean(cycle: u64) -> Self {
        let mut runtime = RuntimeMetrics::default();
        runtime.collector_pressure_alive = true;
        runtime.reactor_health = "healthy".to_string();
        Self {
            snapshot: snapshot(),
            runtime,
            capabilities: CapabilityReport {
                can_taskpolicy: true,
                can_sysctl: true,
                can_memorystatus: true,
                can_memory_pressure_send: false,
                can_mdutil: true,
                can_tmutil: true,
                is_root: true,
                p_core_count: Some(4),
                e_core_count: Some(6),
                unavailable: Vec::new(),
                memorystatus_probe: Some("ok".to_string()),
                task_for_pid_probe: Some("ok".to_string()),
            },
            signal: signal(),
            outcomes: ExecuteOutcomes::default(),
            cycle,
        }
    }

    fn observe(&self, medallion: &mut TelemetryMedallion) -> ContextAdmission {
        medallion.observe(TelemetryObservation {
            snapshot: &self.snapshot,
            hardware: None,
            runtime: &self.runtime,
            capabilities: Some(&self.capabilities),
            signal: &self.signal,
            workload: "idle",
            cycle: self.cycle,
            outcomes: &self.outcomes,
            intervention: Intervention::Observe,
            applied_intervention: None,
            purge_recent: false,
            nars_drift_score: 0.0,
            nars_beliefs_total: 1,
            natural_drift: 0.0,
            arousal_level: 0.5,
        })
    }
}

fn boost_action() -> RootAction {
    RootAction::BoostProcess {
        pid: 300,
        name: "Editor".to_string(),
        reason: "fixture".to_string(),
        decision_reason: DecisionReason::InteractiveFocus,
        start_sec: 12_345,
        start_usec: 678,
    }
}
```

`ObservationFixture::clean` must use a current timestamp, 4P/6E capabilities,
16 GiB RAM, healthy pressure/reactor collectors, finite normalized signals, and
the existing default `ExecuteOutcomes`. Add tests that snapshot all
authority-bearing state before injecting bad and Silver observations:

```rust
#[test]
fn rejected_and_silver_context_have_zero_learning_side_effects() {
    let mut medallion = TelemetryMedallion::new(LOCAL_ID);
    ObservationFixture::clean(1).observe(&mut medallion);
    let before_latest = medallion.latest_gold.clone();
    let before_pending = medallion.pending_actions.clone();
    let before_models = medallion.action_models.clone();
    let before_baseline = medallion.no_action_delta_ema.clone();

    let mut rejected = ObservationFixture::clean(2);
    rejected.snapshot.pressure.memory_pressure = f64::NAN;
    assert_eq!(rejected.observe(&mut medallion).tier, ContextTier::Rejected);
    assert_eq!(medallion.latest_gold, before_latest);
    assert_eq!(medallion.pending_actions, before_pending);
    assert_eq!(medallion.action_models, before_models);
    assert_eq!(medallion.no_action_delta_ema, before_baseline);

    let mut silver = ObservationFixture::clean(3);
    silver.runtime.collector_pressure_alive = false;
    assert_eq!(silver.observe(&mut medallion).tier, ContextTier::Silver);
    assert_eq!(medallion.latest_gold, before_latest);
    assert_eq!(medallion.pending_actions, before_pending);
    assert_eq!(medallion.action_models, before_models);
    assert_eq!(medallion.no_action_delta_ema, before_baseline);
    assert!(medallion.trusted_view().current.is_none());
}

#[test]
fn baseline_requires_two_consecutive_gold_contexts() {
    let mut medallion = TelemetryMedallion::new(LOCAL_ID);
    ObservationFixture::clean(1).observe(&mut medallion);
    let mut silver = ObservationFixture::clean(2);
    silver.runtime.collector_pressure_alive = false;
    silver.observe(&mut medallion);
    ObservationFixture::clean(3).observe(&mut medallion);
    assert!(medallion.no_action_delta_ema.is_empty());
    ObservationFixture::clean(4).observe(&mut medallion);
    assert!(!medallion.no_action_delta_ema.is_empty());
}

#[test]
fn action_evidence_requires_gold_at_both_endpoints() {
    let mut medallion = TelemetryMedallion::new(LOCAL_ID);
    let mut applied = ObservationFixture::clean(1);
    applied.outcomes.audit_traces.push(trace(boost_action(), true));
    applied.observe(&mut medallion);
    let mut silver = ObservationFixture::clean(2);
    silver.runtime.collector_pressure_alive = false;
    silver.observe(&mut medallion);
    assert_eq!(medallion.action_models().len(), 0);
    assert_eq!(medallion.metrics().actuator_pending_total, 1);
}
```

- [ ] **Step 2: Run tests and verify RED**

```bash
cargo test -p apollo-engine telemetry_medallion::tests -- --nocapture
```

Expected: the new tests fail because current `observe` updates authority after
the shallow validity check and reuses `latest` after degraded samples.

- [ ] **Step 3: Integrate installation and admission state**

Change the core shape to:

```rust
pub struct TelemetryMedallion {
    installation_id: InstallationId,
    current_tier: ContextTier,
    last_admitted_live: Option<TelemetryContextSummary>,
    latest_gold: Option<TelemetryContextSummary>,
    consecutive_gold: u32,
    local_gold_total: u64,
    reason_counters: ContextReasonCounters,
}

pub fn new(installation_id: InstallationId) -> Self;
pub fn observe(&mut self, observation: TelemetryObservation<'_>) -> ContextAdmission;
```

Add `installation_id` to `ResolvedActuatorEvidence` and `ActionModelStats`.
Keep it out of `TelemetryContextSummary` so descriptive telemetry cannot expose
installation identity. The medallion owns the ID and passes it separately to
admission, persistence, trusted views, and actuator evidence; callers cannot
supply or forge it per observation.

- [ ] **Step 4: Reorder `observe` around the trust gate**

The method order must be exact:

```rust
let summary = summarize(&observation);
self.bronze_total = self.bronze_total.saturating_add(1);
let admission = classify(ContextAdmissionInput::live(
    &summary,
    self.last_admitted_live.as_ref(),
    Utc::now().timestamp(),
    self.installation_id,
));
self.record_admission(admission);

if admission.tier != ContextTier::Gold {
    self.current_tier = admission.tier;
    self.consecutive_gold = 0;
    if admission.tier == ContextTier::Silver {
        self.last_admitted_live = Some(summary);
    }
    return admission;
}

self.current_tier = ContextTier::Gold;
self.consecutive_gold = self.consecutive_gold.saturating_add(1);
self.local_gold_total = self.local_gold_total.saturating_add(1);
self.latest_gold = Some(summary.clone());
self.last_admitted_live = Some(summary);
admission
```

Do not allocate rejected reason strings or persist rejected summaries. Baseline
updates require `consecutive_gold >= 2`. Pending issuance and resolution occur
only inside the Gold branch. Move the current applied-action extraction,
`external_deltas`, `resolve_pending`, no-action EMA update, `issue`, coordinated
action enrollment, external-counter update, quality accumulation, and latest
context assignment into that branch in their current relative order.

- [ ] **Step 5: Add a borrowed trusted-view contract**

```rust
#[derive(Clone, Copy)]
pub struct TrustedTelemetryView<'a> {
    pub current: Option<&'a TelemetryContextSummary>,
    pub installation_id: InstallationId,
    pub action_models: &'a BTreeMap<String, ActionModelStats>,
    pub action_models_revision: u64,
    pub metrics: TelemetryMedallionMetrics,
}

pub fn trusted_view(&self) -> TrustedTelemetryView<'_> {
    TrustedTelemetryView {
        current: (self.current_tier == ContextTier::Gold)
            .then_some(self.latest_gold.as_ref()).flatten(),
        installation_id: self.installation_id,
        action_models: &self.action_models,
        action_models_revision: self.action_models_revision,
        metrics: self.metrics(),
    }
}
```

- [ ] **Step 6: Run medallion tests and commit**

```bash
cargo test -p apollo-engine telemetry_medallion::tests -- --nocapture
cargo fmt --all -- --check
git add crates/apollo-engine/src/engine/telemetry_medallion.rs
git commit -m "feat(world-model): gate learning side effects on Gold context"
```

Expected: PASS.

### Task 4: Sanitize Restore And Bind Evidence Authority

**Files:**
- Modify: `crates/apollo-engine/src/engine/telemetry_medallion.rs`
- Modify: `crates/apollo-engine/src/engine/learned_state.rs` only if serde
  fixture construction requires the new schema defaults
- Modify: `src/bin/apollo-optimizerd/main.rs`
- Test: medallion restore tests and daemon construction tests

- [ ] **Step 1: Write failing migration/restore tests**

```rust
fn persisted_with_ready_model(installation_id: InstallationId) -> TelemetryMedallionPersisted {
    let mut source = TelemetryMedallion::new(installation_id);
    let mut applied = ObservationFixture::clean(1);
    applied.outcomes.audit_traces.push(trace(boost_action(), true));
    applied.observe(&mut source);
    source.action_models.insert(
        "boost:Editor".to_string(),
        ActionModelStats {
            observations: 20,
            effective_observations: 18,
            utility_ema: 0.08,
            evidence_mass: 20.0,
            utility_variance_ema: 0.0001,
            quality_ema: 0.95,
            last_cycle: 1,
            last_observed_unix: Utc::now().timestamp(),
            hardware_regime: HardwareRegime {
                p_core_count: 4,
                e_core_count: 6,
                ram_gib: 16,
            },
            installation_id,
        },
    );
    source.action_models_revision = source.action_models_revision.wrapping_add(1);
    source.snapshot()
}

#[test]
fn restore_never_restores_live_context_or_pending_endpoints() {
    let persisted = persisted_with_ready_model(LOCAL_ID);
    let mut medallion = TelemetryMedallion::new(LOCAL_ID);
    medallion.restore(persisted);
    assert!(medallion.trusted_view().current.is_none());
    assert_eq!(medallion.metrics().actuator_pending_total, 0);
    assert_eq!(medallion.metrics().actuator_ready_models, 0);
}

#[test]
fn foreign_installation_cannot_regain_authority_from_one_local_context() {
    let mut persisted = persisted_with_ready_model(InstallationId(99));
    persisted.installation_id = InstallationId(99);
    let mut medallion = TelemetryMedallion::new(LOCAL_ID);
    medallion.restore(persisted);
    ObservationFixture::clean(1).observe(&mut medallion);
    assert_eq!(medallion.metrics().actuator_ready_models, 0);
    assert!(medallion
        .action_models()
        .values()
        .all(|model| model.installation_id != LOCAL_ID || model.evidence_mass == 0.0));
}

#[test]
fn same_installation_fresh_evidence_survives_restart_after_local_gold() {
    let persisted = persisted_with_ready_model(LOCAL_ID);
    let mut medallion = TelemetryMedallion::new(LOCAL_ID);
    medallion.restore(persisted);
    assert_eq!(medallion.metrics().actuator_ready_models, 0);
    ObservationFixture::clean(1).observe(&mut medallion);
    assert_eq!(medallion.metrics().actuator_ready_models, 1);
}
```

- [ ] **Step 2: Run restore tests and verify RED**

```bash
cargo test -p apollo-engine telemetry_medallion::tests::restore -- --nocapture
cargo test -p apollo-engine telemetry_medallion::tests::foreign_installation -- --nocapture
```

Expected: FAIL because restored `latest`, pending endpoints, and origin authority
are currently accepted.

- [ ] **Step 3: Advance and sanitize the persisted schema**

Add independently versioned context schema and origin:

```rust
const TELEMETRY_CONTEXT_SCHEMA_VERSION: u32 = 3;

#[serde(default)]
pub context_schema_version: u32,
#[serde(default)]
pub installation_id: InstallationId,
```

`snapshot` records the current schema and ID. `restore` must always clear live
context, `last_admitted_live`, current tier, consecutive Gold, and pending
actions. It restores a no-action baseline and action-model evidence only when:

```rust
let same_origin = state.context_schema_version == TELEMETRY_CONTEXT_SCHEMA_VERSION
    && state.installation_id.is_known()
    && state.installation_id == self.installation_id;
```

For foreign/legacy origin, retain bounded lifetime audit counts but clear the
baseline, recent endpoint evidence, and external deltas. Keep model observation
counts for diagnostics, set effective evidence to zero, and leave the model's
foreign `installation_id` intact so it cannot match local context. Remove the
legacy path that rebuilds confidence from unauthenticated recent Gold records.

When new Gold evidence updates a model whose installation ID or hardware regime
differs, reset its EMA, variance, quality, and evidence mass before applying the
new observation; preserve lifetime `observations` as an audit counter.

- [ ] **Step 4: Load identity once in the daemon**

Before learned state restore:

```rust
let installation_id = installation_identity::load_or_create(Path::new(
    daemon_helpers::installation_id_path(),
))
.unwrap_or_else(|error| {
    tracing::error!(%error, "installation identity unavailable; World Model authority disabled");
    InstallationId::UNKNOWN
});
let mut telemetry_medallion = TelemetryMedallion::new(installation_id);
```

Change `restore` and every test constructor to the new API. Do not silently
generate an in-memory ID on failure: unknown origin must degrade to Silver and
leave the baseline policy operational.

- [ ] **Step 5: Run persistence and learned-state suites**

```bash
cargo test -p apollo-engine telemetry_medallion::tests -- --nocapture
cargo test -p apollo-engine learned_state::tests -- --nocapture
cargo test --bin apollo-optimizerd -- --nocapture
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/apollo-engine/src/engine/telemetry_medallion.rs crates/apollo-engine/src/engine/learned_state.rs src/bin/apollo-optimizerd/main.rs
git commit -m "feat(learning): quarantine foreign and legacy model authority"
```

### Task 5: Make World Model Trust Explicit And Evolutionary

**Files:**
- Modify: `crates/apollo-engine/src/engine/world_model.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs`
- Modify: `src/bin/apollo-optimizerd/daemon_dispatch_tick.rs`
- Test: world-model and dispatch tests

- [ ] **Step 1: Write failing trust and phase tests**

```rust
const LOCAL_ID: InstallationId = InstallationId(0x1020_3040_5060_7080);

fn m4_context(now_unix: i64) -> TelemetryContextSummary {
    TelemetryContextSummary {
        cycle: 100,
        timestamp_unix: now_unix,
        workload: "build".to_string(),
        total_ram_bytes: 16 * 1024 * 1024 * 1024,
        cpu_core_count: 10,
        p_core_count: 4,
        e_core_count: 6,
        reactor_healthy: true,
        collector_pressure_alive: true,
        ..TelemetryContextSummary::default()
    }
}

fn mature_model(now_unix: i64, installation_id: InstallationId) -> ActionModelStats {
    ActionModelStats {
        observations: 20,
        effective_observations: 18,
        utility_ema: 0.08,
        evidence_mass: 20.0,
        utility_variance_ema: 0.0001,
        quality_ema: 0.95,
        last_cycle: 100,
        last_observed_unix: now_unix,
        hardware_regime: HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        },
        installation_id,
    }
}

fn attach_view(
    model: &mut WorldModel,
    context: Option<&TelemetryContextSummary>,
    models: &BTreeMap<String, ActionModelStats>,
    local_gold_total: u64,
) {
    model.attach_context(TrustedTelemetryView {
        current: context,
        installation_id: LOCAL_ID,
        action_models: models,
        action_models_revision: 1,
        metrics: TelemetryMedallionMetrics {
            bronze_total: local_gold_total,
            gold_total: local_gold_total,
            local_gold_total,
            ..TelemetryMedallionMetrics::default()
        },
    });
}

#[test]
fn world_model_abstains_without_current_gold_even_with_mature_models() {
    let now = Utc::now().timestamp();
    let models = BTreeMap::from([(
        "boost:Editor".to_string(),
        mature_model(now, LOCAL_ID),
    )]);
    let mut model = WorldModel::default();
    attach_view(&mut model, None, &models, 1);
    assert_eq!(model.authority_phase(), ModelAuthorityPhase::Suspended);
    assert_eq!(model.utility_ready_actions(), 0);
    assert_eq!(model.imagine_utility("boost:Editor", "build"), UtilityImagined::Unknown);
}

#[test]
fn authority_progresses_from_protected_to_calibrating_to_trusted() {
    let now = Utc::now().timestamp();
    let context = m4_context(now);
    let empty = BTreeMap::new();
    let mut model = WorldModel::default();
    attach_view(&mut model, None, &empty, 0);
    assert_eq!(model.authority_phase(), ModelAuthorityPhase::Protected);

    attach_view(&mut model, Some(&context), &empty, 1);
    assert_eq!(model.authority_phase(), ModelAuthorityPhase::Calibrating);

    let models = BTreeMap::from([(
        "boost:Editor".to_string(),
        mature_model(now, LOCAL_ID),
    )]);
    model.attach_context(TrustedTelemetryView {
        current: Some(&context),
        installation_id: LOCAL_ID,
        action_models: &models,
        action_models_revision: 2,
        metrics: TelemetryMedallionMetrics {
            bronze_total: 1,
            gold_total: 1,
            local_gold_total: 1,
            ..TelemetryMedallionMetrics::default()
        },
    });
    assert_eq!(model.authority_phase(), ModelAuthorityPhase::Trusted);
}

#[test]
fn stale_variance_and_origin_change_revoke_trust() {
    let now = Utc::now().timestamp();
    let context = m4_context(now);
    let mut stale = mature_model(now, LOCAL_ID);
    stale.last_observed_unix = now - UTILITY_MAX_AGE_SECS - 1;
    let mut uncertain = mature_model(now, LOCAL_ID);
    uncertain.utility_variance_ema = 1.0;
    let foreign = mature_model(now, InstallationId(99));
    for stats in [stale, uncertain, foreign] {
        let models = BTreeMap::from([("boost:Editor".to_string(), stats)]);
        let mut model = WorldModel::default();
        attach_view(&mut model, Some(&context), &models, 1);
        assert_ne!(model.authority_phase(), ModelAuthorityPhase::Trusted);
        assert_eq!(model.utility_ready_actions(), 0);
    }
}
```

- [ ] **Step 2: Run tests and verify RED**

```bash
cargo test -p apollo-engine world_model::tests -- --nocapture
```

Expected: FAIL because `attach_context` accepts the whole medallion and there is
no explicit authority phase.

- [ ] **Step 3: Implement the trusted-view contract**

Change the API and phase:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAuthorityPhase {
    #[default] Protected,
    Calibrating,
    Trusted,
    Suspended,
}

pub fn attach_context(&mut self, view: TrustedTelemetryView<'_>) {
    let local_gold_total = view.metrics.local_gold_total;
    self.context_bronze = view.metrics.bronze_total;
    self.context_silver = view.metrics.silver_total;
    self.context_gold = view.metrics.gold_total;
    self.context_quality = view.metrics.mean_quality;
    self.latest_context = view.current.cloned();
    self.current_installation_id = view.installation_id;
    if self.utility_revision != Some(view.action_models_revision) {
        self.utility_predicted.clear();
        self.utility_predicted.extend(
            view.action_models.iter().map(|(key, stats)| (key.clone(), stats.clone())),
        );
        self.utility_revision = Some(view.action_models_revision);
        self.utility_refreshes = self.utility_refreshes.saturating_add(1);
    } else {
        self.utility_cache_hits = self.utility_cache_hits.saturating_add(1);
    }
    self.authority_phase = if self.latest_context.is_none() {
        if local_gold_total == 0 {
            ModelAuthorityPhase::Protected
        } else {
            ModelAuthorityPhase::Suspended
        }
    } else if self.utility_ready_actions() == 0 {
        ModelAuthorityPhase::Calibrating
    } else {
        ModelAuthorityPhase::Trusted
    };
}
```

Phase rules:

- no current Gold and no local Gold history: Protected;
- no current Gold with local Gold history: Suspended;
- current Gold with zero locally ready utility models: Calibrating;
- current Gold with at least one locally ready model: Trusted.

`utility_model_ready` must require minimum decayed evidence, quality, freshness,
hardware match, installation-ID match, and a decisive 95% confidence interval.
The same interval drives the verdict: a positive lower bound promotes; a
nonpositive upper bound vetoes; overlap returns Unknown and is not counted as
ready. No new action family is generated.

- [ ] **Step 4: Update all call sites and dispatch tests**

Replace:

```rust
world_model.attach_context(&telemetry_medallion);
```

with:

```rust
world_model.attach_context(telemetry_medallion.trusted_view());
```

Update fixture model origins. Add dispatch regression tests proving Suspended and
Calibrating phases neither veto nor promote, while Trusted can only reorder the
existing allowlisted Boost/interactive QoS candidates.

- [ ] **Step 5: Run model and dispatch suites**

```bash
cargo test -p apollo-engine world_model::tests -- --nocapture
cargo test --bin apollo-optimizerd daemon_dispatch_tick::tests -- --nocapture
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/apollo-engine/src/engine/world_model.rs src/bin/apollo-optimizerd/main.rs src/bin/apollo-optimizerd/daemon_dispatch_tick.rs
git commit -m "feat(world-model): evolve authority from local confidence"
```

### Task 6: Publish Honest Metrics And Dashboard State

**Files:**
- Modify: `crates/apollo-engine/src/engine/types.rs`
- Modify: `src/bin/apollo-optimizerd/metrics_reporter.rs`
- Modify: `src/bin/apollo-optimizerctl/dashboard.rs`
- Test: metrics reporter and dashboard inline tests

- [ ] **Step 1: Write failing metrics/dashboard tests**

Extend the dashboard fixture and assert exact, width-safe lines:

```rust
status.metrics.world_model_context_bronze_total = 500;
status.metrics.world_model_context_silver_total = 7;
status.metrics.world_model_context_gold_total = 490;
status.metrics.world_model_context_rejected_total = 3;
status.metrics.world_model_context_stale_total = 1;
status.metrics.world_model_context_quality = 0.98;
status.metrics.world_model_context_authority_phase = "calibrating".into();
status.metrics.world_model_actuator_known_models = 24;
status.metrics.world_model_actuator_ready_models = 0;

let think = render_think_q(&status);
assert!(think.iter().any(|line| line == "Ctx    G490 S7 R3 q98%"));
assert!(think.iter().any(|line| line == "WM-U   calibrating 0/24"));
assert!(think.iter().all(|line| display_width(line) <= QW));
```

Add a zero-model test expecting `WM-U   protected · no evidence`, not `0/0`.

- [ ] **Step 2: Run dashboard tests and verify RED**

```bash
cargo test --bin apollo-optimizerctl dashboard::tests -- --nocapture
```

Expected: FAIL because fields and labels are absent.

- [ ] **Step 3: Add bounded runtime fields**

Add serde-defaulted fields to `RuntimeMetrics`:

```rust
pub world_model_context_rejected_total: u64,
pub world_model_context_non_finite_total: u64,
pub world_model_context_range_total: u64,
pub world_model_context_stale_total: u64,
pub world_model_context_temporal_total: u64,
pub world_model_context_foreign_total: u64,
pub world_model_context_coherence_total: u64,
pub world_model_context_local_gold_total: u64,
pub world_model_context_current_tier: String,
pub world_model_context_authority_phase: String,
```

Map fixed reason counters in `metrics_reporter.rs`; do not serialize a dynamic
map. `world_model_context_quality` is admitted Silver+Gold quality, while Gold
rate is Gold/Bronze and rejections remain visible.

- [ ] **Step 4: Render context, phase, and readiness separately**

Rules:

- `Ctx G<gold> S<silver> R<rejected> q<quality>%` always reports admission.
- `WM-U protected · no evidence` when known is zero.
- `WM-U calibrating <ready>/<known>` when Gold is healthy but evidence immature.
- `WM-U suspended <ready>/<known>` when current context is not Gold.
- `WM-U trusted <ready>/<known>` when locally ready.
- Keep veto/promotion totals on a following line only when nonzero.
- Every line must satisfy `display_width(line) <= QW`.

- [ ] **Step 5: Run type, reporter, dashboard, and protocol tests**

```bash
cargo test -p apollo-engine types::tests -- --nocapture
cargo test --bin apollo-optimizerd metrics_reporter::tests -- --nocapture
cargo test --bin apollo-optimizerctl dashboard::tests -- --nocapture
cargo test --test level2_integration -- --nocapture
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/apollo-engine/src/engine/types.rs src/bin/apollo-optimizerd/metrics_reporter.rs src/bin/apollo-optimizerctl/dashboard.rs
git commit -m "feat(dashboard): expose World Model trust and calibration"
```

### Task 7: Audit Graph Reachability, Documentation, And Regression Coverage

**Files:**
- Modify: `docs/acceptance-criteria.md`
- Modify: `README.md`
- Test: Graphify artifacts and complete non-deploy gate

- [ ] **Step 1: Rebuild or refresh Graphify analysis**

Run with the installed Graphify environment:

```bash
env PYTHONPATH=/private/tmp/apollo-graphify-env /Users/edcrtz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m graphify update .
env PYTHONPATH=/private/tmp/apollo-graphify-env /Users/edcrtz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m graphify affected ContextAdmission --depth 4
env PYTHONPATH=/private/tmp/apollo-graphify-env /Users/edcrtz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m graphify affected TrustedTelemetryView --depth 4
```

Expected: every decision path reaches raw context only through
`TelemetryMedallion::observe`, and every World Model utility path receives
`TrustedTelemetryView`. Any direct `TelemetryContextSummary` decision consumer
is a review finding and must be fixed before proceeding.

- [ ] **Step 2: Run static bypass searches**

```bash
rg -n "attach_context\(&telemetry|latest\(\).*Telemetry|state\.latest|TelemetryContextSummary" crates/apollo-engine/src src/bin/apollo-optimizerd
rg -n "sort\(|sort_by|collect::<Vec" crates/apollo-engine/src/engine/telemetry_context_admission.rs
```

Expected: no old attach call, no raw/Silver decision path, and no sorting or
growable validation collection.

- [ ] **Step 3: Update documentation**

In `docs/acceptance-criteria.md`, label legacy M1 numbers as historical and add
the capability-derived M4 trust canary:

- hardware regime detected as 4P/6E/16 GiB on this host, without chip-name
  branching;
- rejected injected fixtures have zero authority changes;
- current tier reaches Gold with available collectors;
- installation ID is nonzero and not emitted in runtime metrics/dashboard;
- failures remain zero;
- post-deploy p95 does not regress more than 25% against the pre-snapshot;
- fluidity does not regress more than 0.03 absolute after warmup;
- pressure/thermal safety gates remain active.

Update README metric examples from ambiguous `World 0/0` language to context
admission plus World Model authority phase.

- [ ] **Step 4: Run the full pre-deploy verification**

Run serially:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --doc
./scripts/apollo-deploy-gate.sh --dry-run
```

Expected: all commands succeed. Record exact test totals and any ignored tests;
do not summarize an old run.

- [ ] **Step 5: Commit docs and any Graphify-driven fixes**

```bash
git add docs/acceptance-criteria.md README.md
git diff --cached --check
git commit -m "docs: define M4 World Model trust acceptance"
```

Do not commit `graphify-out` unless it is already repository policy to track
generated graph artifacts.

### Task 8: Guarded Deployment And Live M4 Canary

**Files:**
- Runtime: `/var/lib/apollo/runtime_metrics.json`
- Runtime: `/var/lib/apollo/learned_state.json`
- Installed: `/usr/local/sbin/apollo-optimizerd`
- Installed: `/usr/local/bin/apollo-optimizerctl`

- [ ] **Step 1: Capture a fresh pre-deploy baseline**

Use the repository gate's snapshot plus an explicit readable copy:

```bash
./scripts/apollo-deploy-gate.sh --dry-run
sudo -n cp /var/lib/apollo/runtime_metrics.json /private/tmp/apollo-context-trust-pre.json
shasum -a 256 /usr/local/sbin/apollo-optimizerd /usr/local/bin/apollo-optimizerctl
```

Expected: dry-run gates pass and baseline includes cycles, failures, p95,
fluidity, pressure, thermal state, context totals, readiness, and AIS.

- [ ] **Step 2: Deploy through the scoped guarded deployer**

```bash
./scripts/apollo-deploy-gate.sh
```

Expected: adaptive-multicore release build for 10-core Apple Silicon, backup
created under `/var/lib/apollo/backups`, launchd bootstrap succeeds, and the
daemon becomes responsive.

- [ ] **Step 3: Verify binary identity and daemon health**

```bash
shasum -a 256 target/release/apollo-optimizerd /usr/local/sbin/apollo-optimizerd
shasum -a 256 target/release/apollo-optimizerctl /usr/local/bin/apollo-optimizerctl
/usr/local/bin/apollo-optimizerctl status
```

Expected: build/installed hash pairs match, daemon reports running, cycles
advance, and operation failures remain zero.

- [ ] **Step 4: Run a bounded warmup canary**

Wait for at least 500 new cycles using the existing supervision loop, then
evaluate admission and performance:

```bash
./scripts/apollo-supervision-loop.sh --iterations 5 --sleep 60
./scripts/apollo-accept-gate.sh
```

Expected after warmup:

- current context is Gold under healthy collectors;
- local Gold count increases;
- rejection reasons remain explainable and bounded;
- foreign/legacy state creates no ready model;
- known models may remain nonzero while readiness truthfully calibrates;
- failures are zero;
- p95 is within 25% of pre-deploy baseline;
- fluidity is no worse than 0.03 below baseline;
- memory pressure and thermal state remain healthy;
- AIS is reported but is not used as the sole acceptance gate.

- [ ] **Step 5: Exercise fail-closed behavior without corrupting production state**

Run the engine's fixture-based corruption tests against the release build; do
not inject NaN or foreign IDs into `/var/lib/apollo/learned_state.json`:

```bash
cargo test --release -p apollo-engine telemetry_context_admission::tests -- --nocapture
cargo test --release -p apollo-engine telemetry_medallion::tests -- --nocapture
```

Expected: all corruption and authority tests pass.

- [ ] **Step 6: Roll back on any failed canary criterion**

Use the backup path printed by `apollo-deploy`; restore both binaries and the
pre-deploy learned state through the existing scoped deployment/backup process,
restart launchd, and re-run status. Do not leave a partially accepted daemon.

- [ ] **Step 7: Final completion audit**

Verify each acceptance criterion in the design spec against fresh evidence:

```bash
git status --short
git log --oneline -8
cargo test --workspace
/usr/local/bin/apollo-optimizerctl status
```

Also retain the Graphify affected output and pre/post metric summaries in the
task report. Completion requires all criteria, matching installed hashes, and a
healthy live canary; passing unit tests alone is insufficient.
