# Ledger Attribution Observability Design

## Goal

Make Apollo's existing count of applied execution receipts without usable
attribution visible in runtime metrics and the dashboard, so production can
prove the Task 7 invariant instead of relying on private persisted state or a
root-only journal.

## Design

The existing `DecisionLedger::unattributed_applied_total()` remains the single
source of truth. `build_unified_learning_health` copies that scalar into
`UnifiedLearningInput`; `UnifiedLearningHealth` caches it alongside the other
ledger projections and publishes it as
`RuntimeMetrics::decision_ledger_unattributed_applied_total`.

The `RuntimeMetrics` field is `u64` with `serde(default)`, preserving
compatibility with old runtime JSON. It is diagnostic only: it does not
contribute to AIS, trust, World Model advice, ranking, exploration, or kernel
authority. Reading and publishing it is O(1). Because
`DecisionLedger::revision()` omits this counter, the scalar is an explicit
`UnifiedLearningRevision` cache-key component so unattributed applied receipts
invalidate the cached health view.

The dashboard adds one bounded line to the learning band:
`Ledger huérfanos N`. Schema version 2 marks the counter as available; older
schemas render `Ledger huérfanos n/d` because a defaulted zero is ambiguous.
For schema 2, zero is shown explicitly because it is the passing audit
condition, and nonzero values remain visible rather than being normalized away.

## Verification

- A legacy `RuntimeMetrics` JSON payload defaults the new field to zero.
- A real `UnifiedLearningHealth` input publishes a nonzero literal into
  `RuntimeMetrics`.
- The daemon builder copies a nonzero ledger counter into the health view.
- The dashboard renders the literal counter and remains within its fixed width.
- Focused tests run RED before production code, then GREEN.
- Workspace formatting, Clippy, release tests, release build, signed deployment,
  and live status verify the complete path.
