# PR-feature-MLP-router — DEFERRED (Phase 1 CV failed 0.55 gate)

- **Date**: 2026-06-27
- **Verdict**: ABORT — mean CV accuracy 0.4990 < 0.55
- **Recommendation**: defer MLP router entirely; do NOT promote `mlp_router.bin`; revisit when per-cycle runtime telemetry is available historically.

## Root-cause analysis

### Why < 0.55

All 16 feature columns are constant across the dataset (`nonzero_dim_count = 0 / 16`). Phase 1a (`extract.py`) had to impute every feature to 0.0 because `runtime_metrics.json` is a single current snapshot, not a per-cycle time series. `journal.jsonl` carries action records with no paired runtime state. There is no instrumented `mlp_router_shadow.jsonl` yet (Phase 2 ships that), and `journal.jsonl` rotation prevents backfilling. Result: the MLP receives a constant input vector and can only learn a single output class — the majority class. Predicted accuracy collapses to the majority-class baseline (~0.50 here, which is below the 0.55 gate).

### Why the SPEC's CV gate caught it (this is the gate working as intended)

`.plan/PR-feature-MLP-router.md §10 adversarial check #1` explicitly anticipates this: 'The router is too small to learn. If `accuracy < 0.50` on the 5-fold holdout, abort.' The 0.55 gate is calibrated to catch exactly this kind of input-degeneracy failure before a bad `.bin` is promoted to Phase 2. The CV gate worked: it prevented a corrupt artifact from reaching the daemon.

## Dataset summary

- Rows: **5413** (target >1000 — PASS)
- NaN cells: 0
- Label distribution: {"2": 2701, "3": 2003, "1": 709}
- Features with non-zero std: **0 / 16**

### Per-fold accuracies

| Fold | n_train | n_val | Accuracy | F1 (macro) |
|---|---|---|---|---|
| 1 | 4330 | 1083 | 0.4986 | 0.1664 |
| 2 | 4330 | 1083 | 0.4986 | 0.1664 |
| 3 | 4330 | 1083 | 0.4995 | 0.1666 |
| 4 | 4331 | 1082 | 0.4991 | 0.1665 |
| 5 | 4331 | 1082 | 0.4991 | 0.1665 |

- **Mean**: 0.4990 ± 0.0004
- **Aggregate confusion matrix** (true x pred, 0..3):

```
       pred0  pred1  pred2  pred3
true0      0      0      0      0
true1      0      0    709      0
true2      0      0   2701      0
true3      0      0   2003      0
```

## What would unblock this

1. **Ship Phase 2 first** (shadow instrumentation per `.plan/PR-feature-MLP-router.md §7 phase 2`) so a historical per-cycle feature stream exists. Then re-run `extract.py` against `mlp_router_shadow.jsonl` joined with `journal.jsonl`.
2. **Add a structured per-cycle `runtime_metrics_history.jsonl` append** in `daemon_cycle_tail.rs` so the training pipeline can replay runtime state at each cycle (one line per cycle, rotation at 5 MB). This makes the historical feature vector recoverable without rebuilding shadow infra.
3. **Re-evaluate the CV threshold.** 0.55 is calibrated for a 16 -> 32 -> 4 router on a 4-class regime-classification problem where adjacent regimes overlap. If the regime taxonomy collapses to 3 classes after observing the data, re-anchor the gate (3-class random baseline = 0.33; a 0.45 gate is still informative).
4. **Revisit label extraction.** 2701 of 5413 rows are `ThrottleNoise` (49.9%) — the label distribution itself is moderately skewed. If the regime taxonomy under-samples `Observe` (label 0 = 0 rows), the router cannot learn it. Consider: (a) emitting synthetic Observe windows from no-action intervals in journal, or (b) collapsing Observe into `TightenThresholds` so the 4-class becomes 3-class.

## Paper anchors (still apply; not blocked by CV failure)

- [Barto & Sutton 2018, §9.5] — function approximation: what function approximators CAN and CANNOT learn from limited data. The architecture choice (16 -> 32 -> 4, ReLU, softmax) is sound; the failure is **data**, not model — 676 parameters against a degenerate (constant) input vector is unrecoverable from any function-approximation standpoint.
- [Bishop 2006, §5.3] — MLP template. sklearn's `MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam')` matches the template.

## What this report does NOT do

- It does NOT write `/var/lib/apollo/mlp_router.bin`. The artifact is not produced.
- It does NOT modify the daemon. No Rust changes, no LSE counters added, no shadow logging.
- It does NOT close the MLP router PR. The PR remains open in `Phase 1` status pending the dataset fix above.
