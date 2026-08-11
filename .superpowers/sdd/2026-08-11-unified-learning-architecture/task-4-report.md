# Task 4 Report: Hierarchical Apollo Learning

Date: 2026-08-11

## Result

Implemented the complete frozen Task 4 contract on `feat/unified-learning-architecture`.
The hierarchy is advisory only, consumes the immutable result of the single Task 3
Gold calibration admission, and persists through the existing learned-state
transaction. No deployment, kernel authority, AIS/ranking change, sidecar, second
queue, or per-episode I/O was added.

## Scenario Matrix

- [x] All 20 `ActuatorFamily::ALL` values have an exhaustive stable
  Goal/Strategy mapping; hierarchy/context enums and paths round-trip through serde.
- [x] The one Task 3 admission returns at most eight exact forecast deltas with
  prediction/error/coverage/Brier and `trust_before`/`trust_after`; no second
  observation or store rescan occurs.
- [x] Gold receives rich identity, alternatives, forecasts, advisers, expected and
  actual utility, causal state, calibration deltas, trust transitions, and origin
  before episodic retention and the live Gold drain. Lightweight recent evidence
  remains detail-free.
- [x] Missing ID/detail, non-Applied lifecycle, unattributed input, Bronze/Silver,
  confounding, nonfinite or invalid context, unknown/foreign origin, hardware
  mismatch, and missing calibration all fail closed independently.
- [x] Coordinated cohorts consolidate once as `CoordinatedComposite`; no member
  family receives utility, prototype, or NARS credit.
- [x] Same-ID drain/fanout/restart/conflicting replay is idempotent; distinct IDs
  for identical actions update independently. Dedup is a persisted 128-ID window.
- [x] Positive and negative Gold update one prototype and exactly four canonical
  NARS propositions; neutral Gold updates the prototype and emits zero. No target,
  PID, raw process, legacy action, or teacher proposition is emitted.
- [x] Prototype state enforces 256 total, 8 family-context variants, 4
  representatives, top-4 retrieval, lazy seven-day half-life, deterministic weak
  eviction, 14-day weak/stale cleanup, finite normalization, and negative-evidence
  retention.
- [x] Restore rejects oversized nested vectors/actions, duplicate or invalid
  records, forged rich detail, and nonfinite prototypes; private family indices are
  rebuilt before runtime use.
- [x] Old/default state and teacher JSON remain descriptive/ignored; v2 calibration
  starts cold, future evidence/top-level schemas fail closed, and malformed current
  state recovers `.previous`.
- [x] A production broker forecast traverses dispatcher -> ledger -> medallion ->
  causal fanout/episodic retention/local hierarchy with one `DecisionId`; duplicate
  local delivery is inert.
- [x] Static audit found no hierarchy/local-consolidation I/O, mutex, sort,
  all-pairs operation, actuator call, safety bypass, or runtime teacher read.

## RED / GREEN Evidence

RED was established before production implementation:

- `task4_hierarchy_contract` failed to compile because `learning_hierarchy` and
  `ForecastCalibrationDelta` did not exist.
- Telemetry enrichment tests failed because alternatives and rich learning details
  were absent.
- Local consolidation tests failed because origin restore, hierarchy ownership,
  and hierarchy-only NARS cleanup did not exist.
- The production fanout test initially exposed raw action identity as unsuitable
  for authority; normalization to Task 3's reusable action class made it green
  without weakening validation.

Final focused GREEN summaries:

- Hierarchy unit filter: `7 passed; 0 failed; 2653 filtered out`.
- Local consolidation filter: `11 passed; 0 failed; 2649 filtered out`.
- Task 4 integration target: `5 passed; 0 failed`.
- Single calibration delta/trust filter: `1 passed; 0 failed; 2658 filtered out`.
- Rich medallion admission/restore filter: `1 passed; 0 failed; 2658 filtered out`.
- Learned-state local compatibility filter: `3 passed; 0 failed; 2656 filtered out`.
- v2 cold migration, future evidence quarantine, nested hostile evidence, future
  top-level refusal, and `.previous` recovery: each `1 passed; 0 failed`.
- NARS saturated insert cap: `1 passed; 0 failed; 2658 filtered out`.
- NARS hierarchy cleanup: `1 passed; 0 failed; 2659 filtered out`.
- Production dispatcher/fanout filter: `1 passed; 0 failed; 249 filtered out`.

Task-end broad evidence:

- `cargo test -p apollo-engine --lib`:
  `2659 passed; 0 failed; 1 ignored; finished in 2.48s`.
- `cargo test --bin apollo-optimizerd`:
  `249 passed; 1 failed`; the unchanged live-process census test
  `warm_enrichment_cache_preserves_live_snapshot_shape` observed process churn.
  Its immediate isolated rerun was
  `1 passed; 0 failed; 249 filtered out; finished in 0.02s`.

## Files

- `crates/apollo-engine/src/engine/learning_hierarchy.rs` (new)
- `crates/apollo-engine/tests/task4_hierarchy_contract.rs` (new)
- `crates/apollo-engine/src/engine/model_calibration.rs`
- `crates/apollo-engine/src/engine/telemetry_medallion.rs`
- `crates/apollo-engine/src/engine/local_consolidation.rs`
- `crates/apollo-engine/src/engine/nars_belief.rs`
- `crates/apollo-engine/src/engine/decision_ledger.rs`
- `crates/apollo-engine/src/engine/mod.rs`
- `crates/apollo-engine/src/engine/causal_graph.rs`
- `crates/apollo-engine/src/engine/world_model.rs`
- `src/bin/apollo-optimizerd/daemon_dispatch_tick.rs`
- `src/bin/apollo-optimizerd/learning_tick.rs`
- `src/bin/apollo-optimizerd/main.rs`

## Bounds And Persistence

- Common prototype lookup is a map lookup plus a private family index containing
  at most eight keys. Retrieval scans only those keys and maintains an insertion
  buffer of four. Global eviction scans at most 256 entries. No full-store sort is
  used.
- NARS contextual eviction is one deterministic O(200) weakest scan; global
  eviction is a deterministic capped O(3000) scan with lexical ties and protected
  seeds excluded.
- Deserializers allocate only named cap-sized vectors and reject the first excess
  element. A 257-prototype hostile payload is rejected; a 256-prototype saturated
  serialized hierarchy is asserted below the 2 MiB Task 7 delta budget and is
  normalized to at most eight variants per current family.
- Checkpoint snapshots perform one bounded decay/cleanup sweep and remain inside
  the existing `LearnedStateSupplement` transaction at both periodic and shutdown
  call sites. Rich raw episodes exist only in the medallion's capped episodic store.

## Remaining Uncertainty

- The one daemon-wide pass had a transient failure in an unchanged test that
  queries live process identity twice; the focused rerun passed. No Task 4 file
  participates in that test.
- With 20 current families and eight contexts each, normal runtime occupancy is
  at most 160 prototypes; the 256 global cap is exercised structurally and by
  hostile serialization, leaving the frozen future-family headroom unused today.
- Release workspace tests, Clippy, release build, p95/RSS measurement, external
  persistence-size measurement, E2E, and deployment remain deferred to Tasks 7/8
  as required.
