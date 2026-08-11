# Task 3 Batched Correctness Fix Report

Date: 2026-08-11
Branch: `feat/unified-learning-architecture`
Base: `ff33fa33c8e079ced2c9ee519dfd1ecb91fe2cc0`
Commit subject: `fix: make calibration authoritative and bounded`

## Findings Closed

- [x] Real decision-time World Model and fresh GPU forecasts are captured before broker execution, bounded to two producer records, attached to matching root-action receipts, and preserved through ledger, pending evidence, resolved Gold evidence, and calibration. The temporal Markov cache-warm path captures the same model outputs before warming.
- [x] Restore derives a bounded per-`ModelKey` evidence aggregate from retained calibration records and cross-validates lifetime/authority counts, Welford count/mean/M2, retained contexts, update sequence, completed-window cardinality, current-window cardinality, and retained MAE mass before recomputing trust. Mismatch cold-resets the model and its records.
- [x] Structured serde visitors cap calibration records (384 exact + 128 family), model records (512), decision IDs (128), model contexts (16), windows (3), provenance predictions/advisers (8 each), source strings (48 chars), recent evidence (64), and episodic evidence (128). Restore selection uses bounded ordered indices with `O(n log cap)` work and no repeated retained-map victim scan.
- [x] Calibration authority requires actuator evidence schema exactly v3. A future nested schema is quarantined, never restored into live authority, and snapshots retain the future schema/state rather than downgrade it to v3.
- [x] Snapshot size enforcement uses deterministic exact-before-family batch retention and at most three full serializations. A final metadata-only fallback guarantees the 1 MiB bound without a per-record serialize/rescan loop.

## Changed Files

- `crates/apollo-engine/src/engine/decision_ledger.rs`
- `crates/apollo-engine/src/engine/model_calibration.rs`
- `crates/apollo-engine/src/engine/telemetry_medallion.rs`
- `crates/apollo-engine/src/engine/world_model.rs`
- `src/bin/apollo-optimizerd/daemon_dispatch_tick.rs`
- `src/bin/apollo-optimizerd/daemon_markov_tick.rs`
- `.superpowers/sdd/2026-08-11-unified-learning-architecture/task-3-batched-fix-report.md`

Controller-owned `AGENTS.md`, the shared progress ledger, and `docs/superpowers/plans/2026-08-11-unified-learning-architecture.md` were not staged.

## RED / GREEN Evidence

RED:

- Focused model-calibration compile failed on the missing `bound_snapshot_state` batching primitive.
- The initial production integration exposed a missing explicit signal fixture, then showed that an unstamped cycle prevented ledger provenance from matching the already-issued pending action. The test now mirrors the daemon merge point's cycle stamp.

GREEN focused results:

- Model calibration filter: `15 passed; 0 failed`.
- Production World Model + GPU event through ledger and Gold calibration: `1 passed; 0 failed`; two real producer forecasts became two calibration records.
- Nested provenance hostile deserialization: `1 passed; 0 failed`.
- Future v4 actuator schema quarantine: `1 passed; 0 failed`.
- Delayed eight-prediction provenance: `1 passed; 0 failed`.
- Directional decision-source credit: `1 passed; 0 failed`.
- v2-to-v3 cold calibration migration: `1 passed; 0 failed`.

## Final Raw Summary

- `cargo test -p apollo-engine --lib`: `2644 passed; 0 failed; 1 ignored`.
- `cargo test --bin apollo-optimizerd`: `250 passed; 0 failed; 0 ignored`.
- `cargo fmt --all -- --check`: exit 0, no output.
- `git diff --check`: exit 0, no output.
- Existing unrelated warnings remained in Holt-Winters, Mach QoS, intelligence score, daemon maintenance tests, planner hint initialization, and Markov release fields.

## Bounds and Persistence Evidence

- The hostile persisted-state test serializes 700 calibration records/models, 500 decision IDs, 40 contexts per model, and 12 windows per model; deserialization retains no more than the fixed caps above.
- The saturated snapshot test first asserts the fixture exceeds `1_048_576` bytes, then asserts the bounded snapshot is at most `1_048_576` bytes and used at most three serialization operations.
- Restore processes each candidate once with bounded `BTreeMap`/`BTreeSet` indices. No global mutex, I/O, actuation authority, specialist-store write, all-pairs comparison, or per-record serialization loop was added.

## Remaining Uncertainty

- The fixed serialization operation count is covered, but wall-clock checkpoint benchmarking remains intentionally deferred to Task 7.
- Forecast availability still correctly depends on mature same-origin World Model evidence or a fresh GPU imagination. Cold/abstaining production decisions remain calibration-inert rather than receiving fabricated forecasts.
- No deployment, workspace release suite, Clippy, or release build was run.
