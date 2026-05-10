// crates/apollo-engine/tests/level3_maintenance_purge.rs
//! Integration tests for Maintenance Purge Gate.
//! This file's first test is the Sprint 3 telemetry-death safeguard:
//! it round-trips all 7 maintenance counters through the full chain
//! (LockFreeMetrics → MetricsSnapshot → sync_from_lockfree → RuntimeMetrics → JSON)
//! and asserts literal substrings appear in serialized JSON.

use std::sync::atomic::Ordering;

use apollo_engine::engine::daemon_state::MetricsState;
use apollo_engine::engine::lse_counters::LockFreeMetrics;

#[test]
fn maintenance_counters_round_trip_to_runtime_metrics_json() {
    let lf = LockFreeMetrics::new();
    lf.maintenance_purge_total.fetch_add(1, Ordering::Relaxed);
    lf.maintenance_purge_skipped_pressure_total
        .fetch_add(2, Ordering::Relaxed);
    lf.maintenance_purge_skipped_swap_floor_total
        .fetch_add(3, Ordering::Relaxed);
    lf.maintenance_purge_skipped_growing_total
        .fetch_add(5, Ordering::Relaxed);
    lf.maintenance_purge_skipped_idle_total
        .fetch_add(7, Ordering::Relaxed);
    lf.maintenance_purge_skipped_build_mode_total
        .fetch_add(11, Ordering::Relaxed);
    lf.maintenance_purge_skipped_rate_limit_total
        .fetch_add(13, Ordering::Relaxed);

    let snap = lf.snapshot();
    let mut state = MetricsState::default();
    state.sync_from_lockfree(&snap);

    let json = serde_json::to_string(&state.metrics).expect("serialize RuntimeMetrics");

    assert!(
        json.contains(r#""maintenance_purge_total":1"#),
        "missing maintenance_purge_total in JSON: {json}"
    );
    assert!(
        json.contains(r#""maintenance_purge_skipped_pressure_total":2"#),
        "missing pressure counter in JSON"
    );
    assert!(
        json.contains(r#""maintenance_purge_skipped_swap_floor_total":3"#),
        "missing swap_floor counter"
    );
    assert!(
        json.contains(r#""maintenance_purge_skipped_growing_total":5"#),
        "missing growing counter"
    );
    assert!(
        json.contains(r#""maintenance_purge_skipped_idle_total":7"#),
        "missing idle counter"
    );
    assert!(
        json.contains(r#""maintenance_purge_skipped_build_mode_total":11"#),
        "missing build_mode counter"
    );
    assert!(
        json.contains(r#""maintenance_purge_skipped_rate_limit_total":13"#),
        "missing rate_limit counter"
    );
}
