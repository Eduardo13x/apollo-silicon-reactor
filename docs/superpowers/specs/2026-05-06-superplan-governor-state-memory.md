# SuperPlan: Governor State Memory (Self-Healing Sprint Phase 2)

**Date:** 2026-05-06
**Frameworks:** apollo-evolve + apollo-nars + superpowers (TDD) + autoresearch (goal-metric-iterate)

---

## 1. Context (autoresearch Goal section)

Sprint 1 (self-healing 10 commits) closed:
- Critical: PID reapply spam → 0
- Critical: Sysinfo cache → 511x faster
- High: Thread affinity FFI scaffolding + consumer wired
- Meta: self_diagnosis observability layer operational

**Residual NotebookLM Critical gap (untouched):**
> "Governor padece **Falta de Memoria de Estado** operativa.
> Sigue emitiendo la misma decisión para mismo PID ciclo tras ciclo si la
> condición de presión persiste."

Empirical: 87.5% journal `success: false` (post-Phase 1 dedup) — most are
"PID already in target state" no-ops. Apollo evaluates 21 governor rules
per process per cycle for processes already throttled/frozen.

## 2. NARS Belief Set (prior-data based)

| Belief ID | Subject | Predicate | f | c | Priority |
|-----------|---------|-----------|---|---|----------|
| B-G1 | adaptive_governor::decide_one | re-emits identical decision for already-throttled PIDs | 0.92 | 0.85 | 0.782 |
| B-G2 | process_enrichment::convert_and_merge | passes governor decision through without state-aware filter | 0.88 | 0.80 | 0.704 |
| B-G3 | dispatch chokepoint dedup | catches same-cycle dups, NOT cross-cycle | 0.95 | 0.90 | 0.855 |
| B-G4 | journal `success: false` rate ≈ 87.5% | dominant class is "kernel says no-op (already applied)" | 0.85 | 0.75 | 0.638 |

**Revision (NARS rule):** B-G1 + B-G2 + B-G3 + B-G4 all converge on:
> **Apollo lacks per-PID applied-state cache; needs cross-cycle state memory.**
> Truth: f=0.91, c=0.93 → priority 0.847 (top-of-queue).

## 3. superpowers TDD Discipline

Phase order:
1. **RED** — write test that fails: governor's `decide_one` returns `Allow`
   when PID was throttled <30s ago AND conditions unchanged
2. **GREEN** — minimal code to pass: add `recently_applied: HashMap<u32, (GovernorDecision, Instant)>`
   to `AdaptiveGovernor`, check before re-emitting same decision
3. **REFACTOR** — TTL aging, cleanup loop, integrate with FreezeCooldown

## 4. apollo-evolve Loop Structure

| Iter | Mutation | Metric | Direction | Verify | Guard |
|------|----------|--------|-----------|--------|-------|
| 1 | RED test (new) | tests passing | higher | `cargo test adaptive_governor::tests::recently_applied` | `cargo test --lib` |
| 2 | GREEN minimal impl | tests passing | higher | same | `cargo clippy --all-targets` |
| 3 | Wire to process_enrichment | tests + journal `success_rate` | higher | post-deploy 200-event sample | tests pass |
| 4 | TTL + cleanup | journal `success_rate` | higher | post-deploy 200-event sample | tests pass |
| 5 | Deploy + measure | journal `success_rate` | higher | post-deploy 200-event sample | failures=0 |

**Composite score (autoresearch metric):**
```
score = (success_rate_pct * 0.6)
      + (cargo_clippy_warnings_delta * -5)
      + (tests_added * 2)
      + (LOC_delta_negative * 0.1)
```

## 5. Architecture (superpowers Design section)

### Component: `RecentlyAppliedCache`

```rust
pub struct RecentlyAppliedCache {
    map: HashMap<u32, (GovernorDecision, Instant)>,
    ttl: Duration,  // 30s default
}

impl RecentlyAppliedCache {
    pub fn record(&mut self, pid: u32, decision: GovernorDecision);
    pub fn was_recently(&self, pid: u32, decision: GovernorDecision) -> bool;
    pub fn cleanup_expired(&mut self);  // O(n) sweep, called every 60 cycles
}
```

### Wire-in points

1. `AdaptiveGovernor::decide_all_with_hw` — at end of decide_one's match,
   if same `decision` recently applied → return `Allow` instead of repeating
2. `process_enrichment::convert_and_merge_heuristic_decisions` — when emitting
   ThrottleProcess/FreezeProcess, call `recently_applied.record()`
3. New `daemon_init.rs` field `recently_applied: RecentlyAppliedCache`

### Why this beats per-emission-site dedup
- 30s TTL captures the "still throttled, kernel says no-op" window
- Doesn't permanently block — when conditions change (CPU drops, GUI returns),
  recently_applied entry expires and Apollo can re-evaluate freshly
- Surface area: 3 files, ~100 LoC

## 6. Risk Analysis (NARS confidence weighting)

| Risk | Severity | Mitigation |
|------|----------|------------|
| TTL too short → still emit dups | Low | Start 30s, observable via dedup_drops metric (Phase 6 already wired) |
| TTL too long → miss legit re-throttle | Medium | When pressure regime changes, bypass cache (gate on signal_digest.regime_shift) |
| Cache memory grows unbounded | Low | cleanup_expired sweep + cap 5000 entries |
| Tests catch only happy path | Medium | superpowers TDD discipline: write 4+ adversarial tests before GREEN |

## 7. Verification Plan (autoresearch Verify section)

Mechanical checks:
1. `cargo test --lib` — 1885 → ≥1885 pass (no regression)
2. `cargo test adaptive_governor` — at least 4 new tests pass
3. `cargo clippy --all-targets 2>&1 | grep -c "warning"` — flat or lower
4. Post-deploy 200-event sample: `journal_success_rate_pct` rises from
   12.5% → ≥35% (3x improvement)
5. `lf_metrics.dedup_drops_throttle` daily total (cumulative delta over 24h)
   baseline → drop ≥50%

## 8. Stop Conditions

Per apollo-evolve `references/darwinian-loop.md` divergence stop rule:
- 2 consecutive commits with `OPENS > CLOSES` → STOP
- Cumulative `Σ(OPENS) - Σ(CLOSES) > 5` → STOP
- Metric plateaus 3 commits AND debt positive → STOP

## 9. Out-of-scope (explicit)

- IpcProtected / AnomalyDetected / DisplayPipeline variant wiring (Medium gap, defer)
- NARS bridge for self_diagnosis.jsonl (Medium gap, defer)
- Phase B trigger debugging (SIP-bound, working as designed)
