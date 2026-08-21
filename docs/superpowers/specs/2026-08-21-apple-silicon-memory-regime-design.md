# Apple Silicon Memory Regime Design

**Date:** 2026-08-21
**Status:** Approved for implementation
**Scope:** Apple Silicon only; RAM- and chip-agnostic

## Goal

Replace cycle-counted and capacity-specific memory decisions with one passive,
dimensionless memory evidence model shared by prediction, reclaim estimation,
profile governance, survival safety, and predictive profile overrides.

The implementation must improve resource efficiency and responsiveness on an
M1 with 8 GiB and an M4 with 16 GiB without naming either chip in production
policy and without resetting Apollo's learned state.

## Non-Negotiable Performance Contract

The feature is observational. It must not introduce a new actuator or make an
existing actuator fire more often merely because the feature exists.

- No new kernel polls, subprocesses, threads, timers, sleeps, file writes, or
  lock acquisitions on the daemon cycle path.
- Normalize only when `PressureCollector` publishes a new sample. Repeated
  daemon cycles reuse the last derived state.
- All retained histories are fixed-capacity and allocate only at startup.
- Per-sample work is bounded and independent of daemon uptime.
- No Cargo commands run concurrently.
- No new purge, freeze, throttle, QoS, jetsam, or sysctl action is added.
- Existing hard safety gates remain authoritative.
- Target added work: <= 50 us p95 per fresh pressure sample on the M1 baseline
  and no measurable increase in daemon cycle p95 above normal run-to-run noise.

## Ownership And Data Flow

```text
PressureCollector (existing 500 ms kernel sample)
    |
    v
MemoryObservation + MemoryCapabilities
    |
    v
MemoryNormalizer (pure, once per collector generation)
    |
    v
NormalizedMemoryState
    |
    +--> MemoryRegimeDetector ----> governor / survival / cognitive corroboration
    +--> SwapPredictor -----------> trend / projection telemetry
    +--> SwapReclaimModel --------> compressor balance / saturation estimate
    +--> HazardModel -------------> learned advisory evidence
```

`NormalizedMemoryState` is the sole owner of unit conversion. Consumers must
not independently divide bytes by hard-coded GiB values or infer elapsed time
from cycle counts.

## Capability Model

`MemoryCapabilities` is immutable after startup and contains:

- physical memory bytes;
- native VM page bytes;
- whether physical memory and page size were kernel-observed or defaulted;
- Apple Silicon support state.

Physical memory comes from the existing capability/sysctl discovery. Page size
comes from `host_statistics64`/`sysconf` through the existing pressure sample.
Invalid values are rejected. The safe Apple Silicon fallback is 8 GiB and
16 KiB, marked low-confidence so policy can remain conservative.

No production branch may match chip names, generations, or marketing models.

## Observation And Normalization

`MemoryObservation` contains a collector generation, monotonic timestamp,
pressure, swap level and velocity, VM flow rates, and page size.

`NormalizedMemoryState` derives:

- `swap_fraction_of_ram`;
- `swap_growth_fraction_per_minute`;
- `compression_fraction_per_second`;
- `reclaim_fraction_per_second`;
- `swapout_fraction_per_second`;
- pressure level and pressure velocity;
- sample age, validity, and confidence;
- projected 30-second swap fraction using a bounded temporal slope.

Every floating-point input is checked for finiteness. Invalid or stale input
produces `Unknown` evidence and cannot escalate policy.

`vm.swapusage.xsu_total` is dynamic capacity on macOS. It remains telemetry,
but it is not used as the primary denominator for control decisions.

## Memory Regime

The detector is a time-based state machine:

- `Unknown`: missing, stale, non-finite, or unsupported evidence;
- `Calm`: low level and no meaningful positive flow;
- `Building`: sustained positive normalized growth;
- `Contended`: elevated level corroborated by compressor or swap I/O flow;
- `Crisis`: high level plus sustained adverse flow, or exhausted RAM-relative
  swap envelope under an existing hard safety condition;
- `Recovering`: negative flow after contention, held long enough to prevent
  an immediate re-escalation.

Transitions use monotonic durations and hysteresis, never cycle counts. A
single burst can raise evidence confidence but cannot by itself establish a
sustained regime. `Unknown` fails closed to no escalation.

Thresholds are dimensionless ratios and durations kept in one policy struct.
They are not duplicated in consumers. Existing learned bands remain inputs to
the detector where their semantics already match normalized pressure.

## Predictor And Reclaim Estimator

`SwapPredictor` stores timestamped RAM-normalized observations in a fixed ring.
It maintains bounded running sums for slope estimation, making insertion and
eviction O(1). Forecast horizons are wall-clock durations. A long sleep or
collector gap invalidates the slope rather than compressing hours into one
daemon cycle.

`SwapReclaimModel` converts pages/sec with the observed page size and normalizes
flows by physical RAM. Its EMA is elapsed-time-aware and is updated only for a
new collector generation. Duplicate daemon cycles return the cached forecast.

Neither estimator performs I/O or directly emits an action.

## Learned Hazard Compatibility

The persisted hazard model remains intact and advisory.

- Existing weights, event history, cycles, and learned state are not deleted.
- New feature semantics carry an explicit schema version.
- Legacy hazard output remains observable while normalized physical evidence
  runs in shadow.
- Authority moves only through existing quality/evidence gates; there is no
  time-based forced promotion.
- Repeated observations from one pressure episode are deduplicated with a
  monotonic refractory window.
- Batch replay updates model weights only. It must not increment the physical
  event count or elapsed observation time.
- A high learned `p_oom` cannot establish `Crisis` or a predictive override
  without current physical corroboration.

## Profile And Survival Consumers

`ProfileGovernor` receives the shared regime and normalized score. Its own
absolute swap-GiB formula is removed. Existing CPU, interactivity, thermal,
development, anti-thrash, and cooldown behavior remains unchanged.

Survival keeps an independent hard safety path. It consumes the normalized
state but never delegates its final safety decision to a learned model. Legacy
functions remain as compatibility wrappers where needed; production call sites
with capabilities use the new API.

No relaxed threshold may expand the set of processes eligible for freezing,
throttling, jetsam changes, or purge.

## Override Provenance

The persisted override gains a backward-compatible origin:

- `Operator`: explicit user request; highest precedence and existing TTL;
- `Predictive`: automatic advisory lease; renewable only while physical
  corroboration remains valid and releasable early when it disappears.

Legacy persisted overrides default to `Operator`. Status clients receive the
origin explicitly. Dashboard and menubar must not label predictive control as
manual.

## Sysctl Receipt Correctness

Allowlisted sysctls are handled as typed numeric values using their observed
kernel width. Binary integer bytes are decoded before any text interpretation.

A write is `Applied` only when:

1. the pre-read succeeds and matches the expected value;
2. the write syscall succeeds;
3. the immediate post-read succeeds;
4. the post-read equals the requested value.

Coercion, unreadable post-state, timeout, or disagreement is not applied and
must not enroll an effect-decay observation. A later kernel reversion remains
the responsibility of the existing delayed postcondition watchdog.

## Persistence And Compatibility

- Every new persisted field uses `#[serde(default)]`.
- Existing `learned_state.json`, `governor_state.json`, journals, counters, and
  timelines remain readable.
- No migration deletes or rewrites historical evidence destructively.
- Corrupt, oversized, non-finite, and future-version input degrades to passive
  behavior.
- Restart, sleep/wake, kill switch, and root/non-root paths cannot reuse stale
  temporal slopes.

## Acceptance Matrix

| Dimension | Required scenarios |
|---|---|
| Hardware | 8/16/24/32+ GiB Apple Silicon; detected and fallback capabilities |
| Page size | observed 16 KiB; synthetic 4/64 KiB; zero and invalid values |
| Cadence | 500 ms, 2 s, duplicate generation, delayed sample, sleep-sized gap |
| Swap | zero, stable, rising, falling, burst-then-flat, dynamic total growth/shrink |
| Numeric safety | NaN, infinity, negative rates, overflow-sized counters |
| Regime | every state, hysteresis, recovery, stale/unknown fail-closed behavior |
| Lifecycle | cold start, persisted restore, restart, wake, kill switch, shutdown |
| Overrides | legacy, operator, predictive, expiry, early release, precedence |
| Sysctl | 100 decodes as 100, 4/8-byte values, no-op, coercion, timeout, mismatch |
| Learning | episode dedup, batch replay count stability, old-state round trip |
| Performance | no duplicate-sample recompute, fixed capacities, focused p95 probe |

## Deferred Scope

- Intel Mac support.
- Replacing the complete hazard model or resetting its learned parameters.
- New memory actuators or broader process eligibility.
- Full mediator cutover for all syscall families.
- Chip-specific tuning or benchmark-selected per-model constants.
