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

// ── Sprint 13 — windowed stage avg/max invariant ────────────────────────────
//
// Locks the fix for the structural `avg_ms > max_ms` artifact previously
// observed on tail-light stages (especially Persist). The bug was a
// horizon mismatch:
//
//   * lifetime `stage_count` + lifetime cumulative `stage_*_total_ns`
//     produced an average over *all observations since boot*,
//   * while `drain_stage_max_ns` swapped a per-interval max to 0
//     every publish.
//
// Result: after a stall, the lifetime avg drifted upward and stayed
// above the freshly-drained max for ages.
//
// Fix: `drain_stage_total_ns(stage)` + `drain_stage_count_window()`
// mirror `drain_stage_max_ns` semantics — both producer (record_stage)
// and consumer (windowed avg in `daemon_cycle_tail`) agree on the
// same time horizon. [Welford 1962] online statistics windowing,
// Sprint 9 `4b13a39` rule (producer + consumer same horizon).

/// Burst-then-outlier scenario: record `n` short observations and one
/// large outlier into the same stage. Across the drained window, the
/// computed avg MUST be <= the drained max — otherwise the dashboard
/// surfaces an impossible "avg > max" telemetry pair.
#[test]
fn stage_avg_never_exceeds_stage_max_in_window() {
    let lf = LockFreeMetrics::new();

    // 99 short observations + 1 large outlier per stage, then bump the
    // per-cycle stage_count by 100. Each cycle records once per stage
    // and is finalized by `finish_stage_cycle`.
    const N_CYCLES: u64 = 100;
    const SHORT_NS: u64 = 50_000; // 50 µs
    const OUTLIER_NS: u64 = 25_000_000; // 25 ms

    let stages = [
        CycleStage::Sense,
        CycleStage::Reason,
        CycleStage::Execute,
        CycleStage::Learn,
        CycleStage::Persist,
        CycleStage::ReasonSignalTick,
        CycleStage::ReasonNeuro,
        CycleStage::ReasonUserContext,
        CycleStage::ReasonHoltWinters,
        CycleStage::ReasonPageReclaim,
        CycleStage::ReasonChromium,
        CycleStage::ReasonEnrich,
    ];

    for i in 0..N_CYCLES {
        for s in &stages {
            let ns = if i == N_CYCLES - 1 {
                OUTLIER_NS
            } else {
                SHORT_NS
            };
            lf.record_stage(*s, ns);
        }
        lf.finish_stage_cycle();
    }

    // Drain the count once for the window divisor, then per-stage
    // total + max — same horizon as the consumer in
    // `daemon_cycle_tail::populate_diagnostic_metrics`.
    let sc_window = lf.drain_stage_count_window();
    assert_eq!(
        sc_window, N_CYCLES,
        "drained count must equal recorded cycles"
    );

    for s in &stages {
        let total_window_ns = lf.drain_stage_total_ns(*s);
        let max_window_ns = lf.drain_stage_max_ns(*s);

        // ms-domain math identical to to_avg_ms / ns_to_ms.
        let avg_ms = (total_window_ns as f64 / sc_window as f64) / 1_000_000.0;
        let max_ms = max_window_ns as f64 / 1_000_000.0;

        assert!(
            avg_ms <= max_ms + 1e-9,
            "stage {:?}: windowed avg {:.6} ms exceeds drained max {:.6} ms \
             (total_window_ns={} max_window_ns={} sc_window={})",
            s,
            avg_ms,
            max_ms,
            total_window_ns,
            max_window_ns,
            sc_window,
        );
    }
}

/// Confirm both new helpers honour the swap-to-0 contract: a second
/// drain immediately after the first returns 0. Same contract as
/// `drain_stage_max_ns`. Prevents future regressions where someone
/// switches to a `load` and re-introduces the horizon mismatch.
#[test]
fn drain_helpers_swap_to_zero_on_read() {
    let lf = LockFreeMetrics::new();

    // Single record-and-finalize cycle.
    lf.record_stage(CycleStage::Persist, 1_500_000);
    lf.record_stage(CycleStage::Sense, 700_000);
    lf.finish_stage_cycle();

    assert_eq!(lf.drain_stage_count_window(), 1);
    assert_eq!(lf.drain_stage_count_window(), 0);

    assert_eq!(lf.drain_stage_total_ns(CycleStage::Persist), 1_500_000);
    assert_eq!(lf.drain_stage_total_ns(CycleStage::Persist), 0);

    assert_eq!(lf.drain_stage_total_ns(CycleStage::Sense), 700_000);
    assert_eq!(lf.drain_stage_total_ns(CycleStage::Sense), 0);
}

/// Calibration loop-closure (2026-06-11): silent-telemetry-death guard for
/// `prediction_debias_applied_total`. Producer bump → snapshot →
/// sync_from_lockfree → serialized RuntimeMetrics JSON, plus serde-default
/// survival when deserializing an older payload missing the field.
#[test]
fn prediction_debias_counter_round_trips_and_survives_old_payload() {
    use apollo_engine::engine::daemon_state::{MetricsState, ReactorStatus};
    use apollo_engine::engine::types::RuntimeMetrics;

    let lf = LockFreeMetrics::new();
    lf.inc_prediction_debias_applied();
    lf.inc_prediction_debias_applied();
    lf.commit();

    let snap = lf.snapshot();
    let mut state = MetricsState {
        metrics: RuntimeMetrics::default(),
        throttle_level: "balanced".to_string(),
        thermal_state: "nominal".to_string(),
        thermal_level_real: "unknown".to_string(),
        fast_tick_until: None,
        reactor_event_weight: 0.0,
        reactor_status: ReactorStatus::default(),
        survival_window: apollo_engine::engine::survival_window::SurvivalActivationWindow::new(),
    };
    state.sync_from_lockfree(&snap);
    let json = serde_json::to_string(&state.metrics).expect("serialize");
    assert!(
        json.contains("\"prediction_debias_applied_total\":2"),
        "counter absent or wrong in runtime_metrics JSON: {json}"
    );

    // Older payload without the field must deserialize to 0, not error.
    let mut v = serde_json::to_value(RuntimeMetrics::default()).expect("to_value");
    v.as_object_mut()
        .expect("object")
        .remove("prediction_debias_applied_total");
    let old: RuntimeMetrics = serde_json::from_value(v).expect("old payload deserializes");
    assert_eq!(old.prediction_debias_applied_total, 0);
}
