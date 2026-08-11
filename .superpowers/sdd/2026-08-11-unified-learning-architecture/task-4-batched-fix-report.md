# Task 4 Batched Review Fix Report

Date: 2026-08-11
Base: `b27d250 feat: consolidate hierarchical Apollo learning`

## Status

Implemented the one batched Task 4 review fix. The change remains advisory-only:
it adds no actuator path, kernel authority, safety bypass, sidecar, persistence
writer, deployment action, or Task 5 behavior.

## Root Causes

### 1. Rich-detail strings were bounded only after ordinary `String` decoding

`ResolvedLearningDetails` capped its nested vectors, but each retained element
was decoded before the vector visitor could reject a ninth item.
`CandidateAlternative.action_key`, `CandidateAlternative.target`,
`CalibrationActionScope::Exact`, and `CalibrationKey.workload` used derived
`String` serde. Their later `is_authoritative` length checks therefore were the
only field-level defense and happened after the values had entered the rich
detail object.

The fix gives those four persisted string carriers explicit field-level serde
visitors and matching serializers. Deserialization rejects the first character
beyond the named cap, and serialization refuses to emit an over-cap in-memory
value. Existing canonical limits remain 320 action characters, 256 target
characters, and 64 workload characters.

### 2. Restored deltas were not exact Task 3 projections

The old `valid_deltas` checked producer uniqueness, action scope, error
arithmetic, and Brier range only. It did not bind a retained delta one-for-one
to the accepted canonical prediction set or verify workload, process class,
horizon, pressure, thermal, foreground state, uncertainty coverage, exact Brier
outcome, or the persisted trust transition.

The fix extracts Task 3's pure `project_forecast_delta` calculation and uses it
for both live admission output and restore validation. `valid_forecast_deltas`
replays no observation and mutates no store; it projects the immutable result
from the evidence/prediction context and requires exact equality. The hierarchy
validator also enforces the detail-contained portions of that projection.
Restored trust transitions must form a continuous retained chain per
producer/action and the latest transition must match an exact retained
calibration record and its restored trust state. A broken chain strips the rich
detail for every affected retained decision.

### 3. Restore treated persisted hardware as live hardware

`TelemetryMedallion::restore` derived `current_hardware` from
`state.latest`, so a moved checkpoint compared its nested detail only with its
own persisted outer evidence. The daemon did not provide its detected startup
hardware until later, when restoring `LocalConsolidator`.

The fix adds a transient, non-persisted live hardware binding to
`TelemetryMedallion`. The daemon computes the existing hierarchy hardware once,
binds it before `LearnedState` restore, and reuses the same regime for
`LocalConsolidator`. Unknown or mismatched live hardware strips rich details
before episodic evidence is exposed. Existing schema migration, installation
origin, future-version quarantine, current/previous recovery, and the single
checkpoint writer remain unchanged.

## TDD RED Evidence

Initial shell attempt:

```text
$ cargo test -p apollo-engine --lib task4_review_ -- --nocapture
zsh:1: command not found: cargo
exit 127
```

This was an environment-path failure, not behavioral RED evidence. Cargo was
then invoked through `/Users/edcrtz/.cargo/bin/cargo` for every Rust command.

Required single focused RED filter against `b27d250` plus the three new tests:

```text
$ /Users/edcrtz/.cargo/bin/cargo test -p apollo-engine --lib task4_review_ -- --nocapture
running 3 tests
task4_review_hostile_oversized_rich_strings_fail_serde ... FAILED
task4_review_forged_calibration_delta_is_stripped_on_restore ... FAILED
task4_review_live_hardware_mismatch_strips_restored_rich_details ... FAILED
test result: FAILED. 0 passed; 3 failed; 0 ignored; 2660 filtered out
exit 101
```

Each failed at its intended final assertion: oversized nested strings decoded,
the forged coverage bit survived restore, and rich evidence survived a different
pre-restored live hardware context.

## Files Changed

- `crates/apollo-engine/src/engine/decision_ledger.rs`
  - Bounded serde for candidate action and target strings.
- `crates/apollo-engine/src/engine/model_calibration.rs`
  - Bounded serde for exact calibration actions and workloads.
  - Pure shared forecast-delta projection and exact observation validator.
  - Live Task 3 admission now uses the same projector as restore validation.
- `crates/apollo-engine/src/engine/learning_hierarchy.rs`
  - Exact one-to-one prediction/delta checks for detail-contained context and
    derived fields.
  - Oversized rich-string regression test.
- `crates/apollo-engine/src/engine/telemetry_medallion.rs`
  - Transient live hardware binding and mismatch stripping.
  - Full outer evidence/delta projection validation.
  - Restored trust-chain continuity and final calibration-store anchoring.
  - Forged-delta and hardware-mismatch regression tests.
- `crates/apollo-engine/src/engine/local_consolidation.rs`
  - Reuses the centralized rich-detail/evidence validator.
  - Synthetic tests now construct canonical Task 3 delta context.
- `src/bin/apollo-optimizerd/main.rs`
  - Binds detected live hardware before telemetry restore and reuses it for the
    hierarchy origin gate.
- `.superpowers/sdd/2026-08-11-unified-learning-architecture/progress.md`
  - Records the batched-fix phase and gates; this local ledger is ignored by the
    repository.
- `.superpowers/sdd/2026-08-11-unified-learning-architecture/task-4-batched-fix-report.md`
  - This report.

## GREEN Evidence

The first focused GREEN attempt found one test-only import removed during
centralization:

```text
$ /Users/edcrtz/.cargo/bin/cargo test -p apollo-engine --lib task4_review_ -- --nocapture
error[E0433]: cannot find type `EvidenceTier` in this scope
error: could not compile `apollo-engine` (lib test) due to 3 previous errors
exit 101
```

After restoring that test import, the unchanged focused filter passed:

```text
$ /Users/edcrtz/.cargo/bin/cargo test -p apollo-engine --lib task4_review_ -- --nocapture
running 3 tests
task4_review_hostile_oversized_rich_strings_fail_serde ... ok
task4_review_forged_calibration_delta_is_stripped_on_restore ... ok
task4_review_live_hardware_mismatch_strips_restored_rich_details ... ok
test result: ok. 3 passed; 0 failed; 0 ignored; 2660 filtered out
exit 0
```

## Required Affected Passes

The first engine-wide attempt exposed eight `local_consolidation` fixtures that
still fabricated default calibration context rather than a canonical Task 3
projection:

```text
$ /Users/edcrtz/.cargo/bin/cargo test -p apollo-engine --lib
test result: FAILED. 2654 passed; 8 failed; 1 ignored
exit 101
```

Only the shared synthetic fixture was corrected, using the production pure
projector. The fresh affected engine pass was then clean:

```text
$ /Users/edcrtz/.cargo/bin/cargo test -p apollo-engine --lib
test result: ok. 2662 passed; 0 failed; 1 ignored
exit 0
```

The one affected daemon pass was zero-failure, including the previously
transient live-process census test:

```text
$ /Users/edcrtz/.cargo/bin/cargo test --bin apollo-optimizerd
test result: ok. 250 passed; 0 failed; 0 ignored
exit 0
```

Formatting and diff checks:

```text
$ /Users/edcrtz/.cargo/bin/cargo fmt --all -- --check
exit 0 (no output)

$ git diff --check
exit 0 (no output)
```

The compiler continued to report only pre-existing unrelated warnings in
`holt_winters.rs`, `mach_qos.rs`, `intelligence_score.rs`, daemon maintenance,
and daemon startup bookkeeping. No warning originated in this fix.

## Invariant Review

- No calibration observation is replayed and no calibration/trust record is
  rewritten during validation.
- Delta matching is bounded by the existing eight predictions/deltas.
- Trust validation is bounded by the 128 retained episodic records and uses
  ordered maps/sets only during restore, never in a control cycle.
- Hardware binding is transient and adds no persisted field or schema bump.
- Rich details are removed before `TrustedTelemetryView` or
  `LocalConsolidator` can observe a hardware mismatch.
- Hierarchy, NARS, prototypes, World Model, and calibration remain advisory.
- No safety, actuator, kernel, persistence transaction, or deployment behavior
  changed.

## Remaining Uncertainty

- A current aggregate calibration store cannot independently reconstruct the
  `trust_before` value of the oldest retained transition after earlier episodes
  have been evicted. Restore therefore validates all retained transition
  continuity and anchors the newest `trust_after` to the exact restored model;
  any discontinuity fails closed by stripping the affected rich chain.
- The requested one engine affected pass required one rerun after the first
  attempt correctly rejected noncanonical synthetic fixtures. Both raw outcomes
  are recorded above; no production behavior was weakened to preserve the old
  fixtures.
- Release workspace tests, Clippy, release build, complete persistence-size
  measurement, p95/RSS comparison, E2E, production sampling, and deployment
  remain deferred to Tasks 7/8 as frozen. No deployment was attempted.
