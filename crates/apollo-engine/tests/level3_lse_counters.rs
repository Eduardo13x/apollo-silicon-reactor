//! Sprint 12 leftover — ReasonDecide stage producer wiring.
//!
//! Verifies the invariant that **every declared `CycleStage` MUST have ≥1
//! producer**, otherwise the consumer side (`runtime_metrics.json`) reads
//! zero forever. Sprint 9 silent-telemetry-death (`4b13a39`) +
//! Hellerstein 2004 §3 observability invariant.
//!
//! The ReasonDecide producer was missing in `main.rs` despite the enum
//! variant existing in `lse_counters.rs` and the dispatch wired in
//! `record_stage` / `drain_stage_max_ns`. The 12-of-13 staging gap was
//! visible in the cycle-tail avg/max table (every other Reason* substage
//! propagated except this one).

use std::sync::atomic::Ordering;

use apollo_engine::engine::lse_counters::{CycleStage, LockFreeMetrics};

#[test]
fn reason_decide_stage_emits_after_decision() {
    let lf = LockFreeMetrics::new();

    // Pre-condition: counter is dark.
    assert_eq!(
        lf.stage_reason_decide_total_ns.load(Ordering::Relaxed),
        0,
        "stage_reason_decide_total_ns must start at 0"
    );
    assert_eq!(
        lf.stage_reason_decide_max_ns.load(Ordering::Relaxed),
        0,
        "stage_reason_decide_max_ns must start at 0"
    );
    assert_eq!(
        lf.stage_count.load(Ordering::Relaxed),
        0,
        "stage_count must start at 0"
    );

    // Drive what the daemon does around the decide_actions call site:
    // record_stage(ReasonDecide, elapsed_ns) at the end of the decision
    // block. We use a deterministic synthetic ns value so the assertions
    // are stable across CI noise.
    const SYNTHETIC_NS: u64 = 12_345_678;
    lf.record_stage(CycleStage::ReasonDecide, SYNTHETIC_NS);
    lf.finish_stage_cycle();

    // Post-condition: the counter dispatch reached the ReasonDecide arms
    // of `record_stage` (not e.g. ReasonNeuro or Reason umbrella) and the
    // stage_count tick advanced.
    assert_eq!(
        lf.stage_reason_decide_total_ns.load(Ordering::Relaxed),
        SYNTHETIC_NS,
        "stage_reason_decide_total_ns must equal recorded ns"
    );
    assert_eq!(
        lf.stage_reason_decide_max_ns.load(Ordering::Relaxed),
        SYNTHETIC_NS,
        "stage_reason_decide_max_ns must equal recorded ns"
    );
    assert_eq!(
        lf.stage_count.load(Ordering::Relaxed),
        1,
        "stage_count must increment exactly once per finish_stage_cycle()"
    );

    // Cross-check: no unrelated Reason* substage was clobbered (catches
    // the case where the match arm dispatches to the wrong total/max
    // pair — the Sprint 3 redux pattern).
    assert_eq!(
        lf.stage_reason_neuro_total_ns.load(Ordering::Relaxed),
        0,
        "ReasonDecide must not write into ReasonNeuro counters"
    );
    assert_eq!(
        lf.stage_reason_holtwinters_total_ns.load(Ordering::Relaxed),
        0,
        "ReasonDecide must not write into ReasonHoltWinters counters"
    );
    assert_eq!(
        lf.stage_reason_total_ns.load(Ordering::Relaxed),
        0,
        "ReasonDecide (substage) must not write into Reason umbrella counter"
    );

    // Drain semantics: max value drains to 0 after read but total stays
    // cumulative for long-run averages.
    let drained = lf.drain_stage_max_ns(CycleStage::ReasonDecide);
    assert_eq!(
        drained, SYNTHETIC_NS,
        "drain_stage_max_ns must return the last recorded max"
    );
    assert_eq!(
        lf.stage_reason_decide_max_ns.load(Ordering::Relaxed),
        0,
        "stage_reason_decide_max_ns must reset to 0 after drain"
    );
    assert_eq!(
        lf.stage_reason_decide_total_ns.load(Ordering::Relaxed),
        SYNTHETIC_NS,
        "stage_reason_decide_total_ns must stay cumulative after drain"
    );
}
