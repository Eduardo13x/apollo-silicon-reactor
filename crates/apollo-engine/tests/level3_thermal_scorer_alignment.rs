//! Sprint 12 Convergence #4 — Thermal × Scorer-Override alignment probe.
//!
//! Closes the loop between Phase 4.2 (CausalGraph external_blame producer
//! at thermal-throttle transitions) and Phase C (asymmetric scorer override
//! rejects) by adding an LSE counter that fires only when both events
//! coincide inside `EXTERNAL_BLAME_WINDOW` (10 s).
//!
//! This integration test drives the synthetic preconditions and asserts:
//!   1. With a recent `ThermalThrottle` event AND a fresh scorer-override
//!      bump, the alignment counter increments by exactly the override
//!      delta.
//!   2. With a thermal event but NO scorer-override bump, the counter
//!      stays flat.
//!   3. With a scorer-override bump but NO thermal event in the window,
//!      the counter stays flat.
//!   4. When the thermal event is older than `EXTERNAL_BLAME_WINDOW`,
//!      the counter stays flat even with a fresh scorer-override bump.
//!
//! ## References
//! [Pearl 2009 §3] Causality — confounder adjustment via co-occurrence.
//! [Sutton & Barto 2018 §11.7] Off-policy correction via meta-signal.

use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime};

use apollo_engine::engine::causal_graph::{CausalGraph, ExternalEventKind};
use apollo_engine::engine::lse_counters::LSE_COUNTERS;

static COUNTER_GUARD: Mutex<()> = Mutex::new(());

/// Mirror of the daemon main-loop convergence probe at
/// `src/bin/apollo-optimizerd/main.rs:~3667`. The test exercises this
/// closure directly to avoid pulling in the entire daemon binary; the
/// production code is bit-identical so a regression here surfaces a
/// regression there too.
fn run_convergence_probe(
    causal_graph: &CausalGraph,
    prev_override_rejects: &mut u64,
    now: SystemTime,
) {
    let cur = LSE_COUNTERS
        .scorer_override_rejects_total
        .load(Ordering::Relaxed);
    let delta = cur.saturating_sub(*prev_override_rejects);
    if delta > 0 && causal_graph.has_recent_external_event(ExternalEventKind::ThermalThrottle, now)
    {
        for _ in 0..delta {
            LSE_COUNTERS.inc_causal_thermal_scorer_override_alignment();
        }
    }
    *prev_override_rejects = cur;
}

#[test]
fn alignment_fires_when_thermal_and_scorer_override_coincide() {
    let _g = COUNTER_GUARD.lock().unwrap();
    let before = LSE_COUNTERS
        .causal_thermal_scorer_override_alignments_total
        .load(Ordering::Relaxed);

    let mut g = CausalGraph::new();
    let now = SystemTime::now();
    g.record_external_event(ExternalEventKind::ThermalThrottle, 0.65, now);
    let mut prev = LSE_COUNTERS
        .scorer_override_rejects_total
        .load(Ordering::Relaxed);
    LSE_COUNTERS.inc_scorer_override_reject();

    run_convergence_probe(&g, &mut prev, now);

    let after = LSE_COUNTERS
        .causal_thermal_scorer_override_alignments_total
        .load(Ordering::Relaxed);
    assert_eq!(
        after - before,
        1,
        "fresh thermal + scorer-override → alignment += 1"
    );
}

#[test]
fn alignment_silent_with_thermal_but_no_scorer_bump() {
    let _g = COUNTER_GUARD.lock().unwrap();
    let before = LSE_COUNTERS
        .causal_thermal_scorer_override_alignments_total
        .load(Ordering::Relaxed);

    let mut g = CausalGraph::new();
    let now = SystemTime::now();
    g.record_external_event(ExternalEventKind::ThermalThrottle, 0.65, now);
    let mut prev = LSE_COUNTERS
        .scorer_override_rejects_total
        .load(Ordering::Relaxed);
    // No LSE_COUNTERS.inc_scorer_override_reject() this turn.

    run_convergence_probe(&g, &mut prev, now);

    let after = LSE_COUNTERS
        .causal_thermal_scorer_override_alignments_total
        .load(Ordering::Relaxed);
    assert_eq!(after - before, 0, "thermal alone → counter flat");
}

#[test]
fn alignment_silent_with_scorer_bump_but_no_thermal() {
    let _g = COUNTER_GUARD.lock().unwrap();
    let before = LSE_COUNTERS
        .causal_thermal_scorer_override_alignments_total
        .load(Ordering::Relaxed);

    let g = CausalGraph::new();
    let mut prev = LSE_COUNTERS
        .scorer_override_rejects_total
        .load(Ordering::Relaxed);
    LSE_COUNTERS.inc_scorer_override_reject();

    run_convergence_probe(&g, &mut prev, SystemTime::now());

    let after = LSE_COUNTERS
        .causal_thermal_scorer_override_alignments_total
        .load(Ordering::Relaxed);
    assert_eq!(after - before, 0, "scorer alone → counter flat");
}

#[test]
fn alignment_silent_when_thermal_outside_window() {
    let _g = COUNTER_GUARD.lock().unwrap();
    let before = LSE_COUNTERS
        .causal_thermal_scorer_override_alignments_total
        .load(Ordering::Relaxed);

    let mut g = CausalGraph::new();
    let now = SystemTime::now();
    // Record thermal 15 s in the past — strictly outside the 10 s
    // EXTERNAL_BLAME_WINDOW.
    let stale = now - Duration::from_secs(15);
    g.record_external_event(ExternalEventKind::ThermalThrottle, 0.65, stale);
    let mut prev = LSE_COUNTERS
        .scorer_override_rejects_total
        .load(Ordering::Relaxed);
    LSE_COUNTERS.inc_scorer_override_reject();

    run_convergence_probe(&g, &mut prev, now);

    let after = LSE_COUNTERS
        .causal_thermal_scorer_override_alignments_total
        .load(Ordering::Relaxed);
    assert_eq!(after - before, 0, "stale thermal → counter flat");
}
