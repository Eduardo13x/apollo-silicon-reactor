# M4 Evolutionary Context Trust Design

**Status:** Approved for planning on 2026-07-27

## Summary

Apollo will treat telemetry validity, statistical confidence, and permission to
act as three separate concerns. A bounded admission gate will reject malformed
or impossible context before it can update the no-action baseline, create or
resolve actuator evidence, become the World Model's current context, or acquire
persisted decision authority. Valid but incomplete context may be retained as
Silver diagnostics. Only fresh, coherent, locally observed Gold context can
influence the World Model.

The operating policy is stability and user-perceived fluidity first, with
performance aggressiveness earned progressively by clean evidence from the
current Apple Silicon hardware regime. This is capability-based rather than an
`M4` chip-name allowlist so the same trust rules remain safe on other Apple
Silicon generations.

## Problem

`TelemetryMedallion::observe` currently promotes almost every finite snapshot
to contextual Gold. Its hard validation covers workload length and a small
subset of pressure, CPU, and thermal fractions, while
`TelemetryContextSummary` carries many more model-consumed fields. A sample can
therefore contain non-finite, out-of-range, stale, temporally regressive, or
cross-signal-incoherent values and still become `latest`, update the no-action
baseline, resolve pending actions, and later be persisted.

The persisted `latest` context is restored without full revalidation. That can
let legacy or foreign-machine context appear current before this Mac has
produced local evidence. Current metrics also make a near-100% Gold rate look
healthy even when the admission test is too permissive.

The result is not necessarily an immediate unsafe actuator command, because
Apollo has downstream guards, but it weakens the World Model's epistemic
boundary. A model that cannot distinguish "observed" from "trusted" can become
confident for the wrong reasons.

## Goals

1. Block malformed, impossible, stale, and temporally regressive context before
   it can influence any learning or decision path.
2. Preserve useful but incomplete observations as diagnostics without granting
   them decision authority.
3. Require local, hardware-compatible evidence before imported state can affect
   World Model decisions.
4. Let reversible acceleration become more assertive only when confidence and
   measured user benefit improve on this machine.
5. Fall back to the existing stable policy or no-op whenever context or outcome
   evidence is uncertain.
6. Keep the hot path linear in the fixed number of telemetry fields, allocation
   free after summary construction, and bounded in memory and persistence.
7. Expose truthful dashboard and metrics state: admission tier, rejection
   reasons, local calibration progress, model readiness, vetoes, and
   promotions.

## Non-Goals

- Apollo will not modify the XNU kernel or invent new privileged actuators.
- The World Model will not bypass the existing action safety, capability,
  thermal, memory-pressure, cooldown, audit, or rollback gates.
- The design will not hard-code thresholds for one marketing chip name.
- Silver data will not be probabilistically downweighted into decision models;
  it has zero decision authority.
- This change will not replace every specialist learner in Apollo. It secures
  the shared context and World Model authority boundary first.

## Evidence And Design Basis

Graphify's bidirectional affected graph for `TelemetryMedallion` reaches
`learning_tick`, `metrics_reporter`, `WorldModel::attach_context`, learned-state
persistence, historical feature extraction, and daemon dispatch. The admission
decision must therefore be made once inside the medallion and carried
explicitly to consumers rather than independently reconstructed downstream.

The uncertainty policy follows three useful results:

- [Safe Policy Improvement with Baseline Bootstrapping](https://proceedings.mlr.press/v97/laroche19a.html): use the baseline policy where evidence is insufficient.
- [Online Learning with Off-Policy Feedback](https://proceedings.mlr.press/v238/guo24b.html): persistent context measurement error must be represented rather than treated as exact context.
- [Corruption-Robust Algorithms with Uncertainty Weighting](https://proceedings.mlr.press/v202/ye23d.html): uncertainty should reduce authority under contaminated observations.

Apollo applies these conservatively: structurally bad context is rejected;
incomplete context is diagnostic-only; only coherent Gold context participates
in policy evidence; and uncertain policies bootstrap to the existing baseline.

## Architecture

### 1. One Admission Boundary

`telemetry_medallion.rs` will own a pure, deterministic context validator. It
will return a `ContextAdmission` containing:

- `tier`: Rejected, Silver, or Gold;
- `quality`: finite value in `[0, 1]`;
- `reasons`: a bounded bitset of stable reason codes;
- `hardware_regime`: capability-derived P-core, E-core, and RAM class;
- `local_epoch`: whether the context was observed in the current daemon epoch.

The validator receives the candidate summary, the last admitted live summary,
the current wall-clock time, and the live capability-derived hardware regime.
It performs one pass over a fixed set of fields. It does not retain rejected
payloads, allocate per-reason strings, sort collections, or grow a history.

### 2. Hard Rejection

A rejected observation increments Bronze/rejection/reason counters and returns
before any learning side effect. Rejection includes:

- non-finite required scalar values;
- fractions or probabilities outside `[0, 1]`;
- negative rates, power, utilization, idle time, or impossible optional sensor
  values where the field's semantics are non-negative;
- empty or overlong required identity fields;
- zero or implausible total RAM/core identity for a context claiming local
  hardware authority;
- used RAM, free RAM, swap, disk, or process aggregates that contradict their
  enclosing totals beyond an explicit tolerance;
- timestamp regression, excessive future skew, or stale live snapshots;
- cycle regression within the same daemon epoch;
- a hardware regime that contradicts live capabilities.

Rejected context must not:

- replace the latest trusted context;
- update the no-action baseline;
- resolve pending actuator evidence;
- seed a pending actuator observation;
- update action models or recent evidence;
- attach to `WorldModel`;
- be serialized as trusted `latest` context.

### 3. Silver Diagnostic Context

Silver is structurally valid but lacks enough contemporaneous signal quality
for policy authority. Examples include a temporarily dead optional collector,
unknown hardware regime during startup, estimated-only thermal data, a large
but still physically possible discontinuity, or a missing optional energy
sensor.

Silver may update bounded diagnostics and rejection/degradation counters. It
may be displayed and used to report collector health. It cannot update any
baseline, pending outcome, action model, World Model decision context, or
readiness count.

SMC and other optional sensor availability are not permanent Gold requirements.
The validator scores only fields whose collectors claim a usable sample, so an
M4 without a particular optional sensor can still reach Gold using coherent
macOS-normalized signals.

### 4. Gold Decision Context

Gold requires:

- all structural checks to pass;
- a fresh, monotonic timestamp and cycle;
- a known hardware regime derived from live capabilities;
- coherent required pressure, CPU, RAM, fluidity, workload, reactor-health, and
  foreground/context signals;
- sufficient collector health for the fields used by that sample;
- no hard rejection and no unresolved Silver degradation reason.

Only Gold can become `latest_gold`. Baseline deltas require two consecutive
local Gold contexts. Pending actuator evidence requires Gold at issuance and
Gold at resolution. A missing trustworthy endpoint rejects or expires the
evidence instead of manufacturing a low-quality label.

### 5. Persistence And Migration

The persisted telemetry schema version will advance. Restore is audit-only
until the daemon observes the first local Gold context:

- persisted `latest` never becomes live decision context;
- pending actions are discarded because their endpoint continuity cannot
  survive restart or machine transfer;
- malformed model entries and recent evidence are dropped;
- lifetime counters may be retained after saturation and consistency checks;
- action-model observations may remain for audit, but effective evidence is
  zero when schema, freshness, wall clock, or hardware regime is unknown or
  incompatible;
- imported M1 models cannot define the current M4 hardware regime;
- the first local Gold sample establishes the live regime, after which matching
  fresh priors may be evaluated under existing confidence rules.

This preserves portable descriptive history without granting foreign history
decision authority.

### 6. Evolutionary Authority

Evolution is per action/workload model, not one global aggressive switch:

1. **Protected:** no trustworthy local outcome evidence. Apollo runs the
   existing stable policy; the World Model may observe Gold context but cannot
   promote an accelerator.
2. **Calibrating:** local Gold context and outcomes accumulate. Candidate
   recommendations are evaluated in shadow; no imported evidence satisfies the
   local evidence requirement.
3. **Trusted:** effective local evidence, quality, freshness, and a positive
   lower confidence bound permit promotion of an already-safe reversible
   accelerator such as Boost or interactive QoS.
4. **Regressed:** negative upper confidence bound, rising p95/stall/pressure,
   thermal stress, stale evidence, or regime change immediately removes extra
   authority and returns to the baseline policy.

Readiness must continue to use decayed effective evidence and confidence
bounds, never raw lifetime observation counts. More clean evidence can increase
authority; age, variance, hardware change, or measured regressions reduce it.

### 7. World Model Contract

`WorldModel::attach_context` will consume an explicit Gold-only trusted view,
not infer trust from the presence of `TelemetryContextSummary`. Its imagination
API remains advisory:

- promote only existing, reversible, safety-approved accelerator candidates;
- veto only discretionary actions when the no-op confidence bound dominates;
- return Unknown when context is absent, Silver, stale, foreign, or statistically
  immature;
- never generate throttle, freeze, sysctl, memorystatus, or other privileged
  actions outside their owning policy and safety pipelines.

The medallion remains the evidence owner; the World Model remains a bounded
consumer. This avoids parallel trust logic and prevents N+1 validation work in
dispatch loops.

### 8. Observability

Runtime metrics will expose:

- Bronze, Silver, Gold, rejected, stale, foreign-hardware, temporal, range, and
  coherence totals;
- Gold admission rate and finite mean quality;
- current authority phase and local Gold calibration count;
- known versus locally ready action models;
- utility vetoes and promotions.

The dashboard will label context separately from actuator utility readiness.
It will not render an ambiguous `0/0` as failure. A protected but healthy M4
should read as healthy context plus "calibrating" or "no local action evidence",
while `0 ready` remains truthful.

### 9. Complexity And Storage Bounds

- Validation is `O(F)` for a compile-time-bounded telemetry field count.
- Reason storage is a fixed bitset plus saturating counters.
- No rejected context payload is persisted.
- Existing pending-action, recent-evidence, and action-model limits remain hard
  bounds.
- No full collection sort is added to the cycle hot path.
- Validation executes once per cycle and consumers reuse its result.

## Data Flow

```text
macOS collectors
      |
      v
TelemetryContextSummary
      |
      v
ContextAdmission (single bounded pass)
      |
      +-- Rejected -> counters only
      |
      +-- Silver   -> diagnostics only
      |
      `-- Gold     -> latest_gold
                       |-- no-action baseline
                       |-- pending action endpoints
                       |-- Gold outcome models
                       `-- WorldModel trusted context
                                  |
                                  v
                        advisory veto/promotion
                                  |
                                  v
                     existing safety/action pipeline
```

## Failure Handling

- Counter overflow uses saturating arithmetic.
- Non-finite persisted aggregates become zero and cannot create readiness.
- A daemon cycle reset starts a new local epoch and clears endpoint continuity.
- A clock regression invalidates freshness until a new monotonic live sample is
  observed.
- Collector failure degrades to Silver or Unknown rather than fabricating a
  normal value.
- Hardware regime change revokes action-model authority immediately but keeps
  bounded audit counters.
- Validation bugs fail closed for World Model authority while leaving Apollo's
  existing baseline policy operational.

## Verification Strategy

Implementation follows strict TDD. Each behavior begins with a failing test.

1. **Validator unit tests:** every required scalar receives NaN, infinity, and
   boundary violations; optional fields are checked only when present.
2. **Invariant tests:** contradictory RAM/swap/disk/core aggregates, temporal
   regressions, stale/future timestamps, and live hardware mismatch are blocked.
3. **Authority tests:** rejected and Silver samples cannot modify latest Gold,
   baselines, pending actions, action models, World Model readiness, or snapshot
   authority.
4. **Restore tests:** corrupt JSON-compatible state, legacy schema, M1 regime,
   future timestamps, and pending endpoints cannot create M4 readiness.
5. **Evolution tests:** local clean evidence progresses Protected -> Calibrating
   -> Trusted; variance, age, negative utility, or hardware change revokes it.
6. **Graphify audit:** inspect every affected path from `TelemetryMedallion`,
   `ContextAdmission`, and `WorldModel::attach_context`; no consumer may bypass
   the Gold-only contract.
7. **Regression suite:** engine, daemon, optimizerctl, workspace, and existing
   end-to-end tests all pass.
8. **Runtime canary:** deploy with backup, confirm daemon health/failure count,
   validate nonzero clean admission, and compare p95/fluidity/pressure against a
   pre-deploy rolling baseline before leaving the new build active.

## Acceptance Criteria

1. All injected NaN, infinity, out-of-range, stale, cycle-regressive,
   timestamp-regressive, and live-hardware-mismatch samples are denied decision
   authority.
2. Rejected and Silver samples produce zero changes to the no-action baseline,
   pending evidence endpoints, action models, and World Model trusted context.
3. No restored state can create a ready M4 action model before at least one
   compatible local Gold context and sufficient fresh local Gold outcomes.
4. Clean M4 fixtures covering optional collector combinations are admitted
   without requiring unavailable optional sensors.
5. Authority increases only from fresh local Gold evidence with positive
   confidence bounds and drops on regression, staleness, or regime mismatch.
6. Dashboard metrics distinguish context health, calibration, known models, and
   locally ready models without ambiguous `0/0` output.
7. Graphify shows no path from raw or Silver context to World Model decisions.
8. The validator uses bounded storage and one `O(F)` pass per cycle.
9. The complete test suite passes, the installed binaries match the verified
   release build, daemon failures remain zero, and the runtime canary shows no
   material p95 or fluidity regression.

## Rollout And Rollback

Build and test before installation. Deployment will back up the installed
binaries and `/var/lib/apollo` learned state, stop the LaunchDaemon, atomically
install the verified release artifacts, restart it, and run the canary. If
health, failures, admission, p95, or fluidity violate acceptance thresholds,
restore the previous binaries and learned state and restart the prior daemon.

