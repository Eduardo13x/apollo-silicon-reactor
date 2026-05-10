// crates/apollo-engine/tests/level3_maintenance_purge.rs
//! Integration tests for Maintenance Purge Gate.
//! This file's first test is the Sprint 3 telemetry-death safeguard:
//! it round-trips all 7 maintenance counters through the full chain
//! (LockFreeMetrics → MetricsSnapshot → sync_from_lockfree → RuntimeMetrics → JSON)
//! and asserts literal substrings appear in serialized JSON.

use std::sync::atomic::Ordering;

use apollo_engine::engine::daemon_state::MetricsState;
use apollo_engine::engine::lse_counters::LockFreeMetrics;
use apollo_engine::engine::maintenance_state::SwapDeltaWindow;

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

#[test]
fn maintenance_state_swap_floor_blocks_m1_cold_boot() {
    // Verify the swap_floor calculation matches spec.
    // M1 cold boot: swap_total = 800 MB, swap_used = 500 MB.
    let swap_total: u64 = 800 * 1024 * 1024;
    let swap_used: u64 = 500 * 1024 * 1024;
    let swap_floor = std::cmp::max(1_536u64 * 1024 * 1024, swap_total / 2);
    assert_eq!(swap_floor, 1_536 * 1024 * 1024);
    assert!(swap_used < swap_floor, "M1 cold boot should not trigger maintenance");
}

#[test]
fn maintenance_state_swap_floor_passes_for_typical_m1_8gb() {
    // Typical loaded M1 8GB: swap_total = 4 GB, swap_used = 2.5 GB.
    let swap_total: u64 = 4 * 1024 * 1024 * 1024;
    let swap_used: u64 = 2_560 * 1024 * 1024;
    let swap_floor = std::cmp::max(1_536u64 * 1024 * 1024, swap_total / 2);
    assert_eq!(swap_floor, 2 * 1024 * 1024 * 1024);
    assert!(swap_used > swap_floor, "loaded M1 should pass swap_floor");
}

#[test]
fn maintenance_window_requires_90s_sustained() {
    let mut w = SwapDeltaWindow::default();
    let now = std::time::SystemTime::now();
    for i in 0..30 {
        let t = now - std::time::Duration::from_secs(60)
            + std::time::Duration::from_secs(i * 2);
        w.push(t, 50_000.0);
    }
    assert!(!w.sustained_below(256_000.0, 90), "60s history should fail 90s requirement");

    for i in 0..15 {
        let t = now - std::time::Duration::from_secs(30)
            + std::time::Duration::from_secs(i * 2);
        w.push(t, 50_000.0);
    }
    assert!(w.sustained_below(256_000.0, 90), "90s history should pass");
}
