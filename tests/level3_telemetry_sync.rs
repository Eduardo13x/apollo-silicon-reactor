//! Sprint 4 Fase 5 reviewer fix — telemetry sync chain integration test.
//!
//! Proves that ActionAccumulator counters survive the full chain:
//!
//!   ActionAccumulator (Atomic increment)
//!     -> LockFreeMetrics::snapshot() (consistent point-in-time read)
//!     -> MetricsState::sync_from_lockfree()  (flush into RuntimeMetrics)
//!     -> serde_json::to_string()              (visible to dashboard)
//!
//! This is the integration test that would have caught the Sprint 3 redux:
//! atomic counters incremented on the hot path but never flushed end-to-end.

use std::sync::atomic::Ordering;

use apollo_optimizer::engine::daemon_state::{MetricsState, ReactorStatus};
use apollo_optimizer::engine::lse_counters::LockFreeMetrics;
use apollo_optimizer::engine::types::RuntimeMetrics;

fn fresh_metrics_state() -> MetricsState {
    MetricsState {
        metrics: RuntimeMetrics::default(),
        throttle_level: "balanced".to_string(),
        thermal_state: "nominal".to_string(),
        thermal_level_real: "unknown".to_string(),
        fast_tick_until: None,
        reactor_event_weight: 0.0,
        reactor_status: ReactorStatus::default(),
    }
}

#[test]
fn fase5_counters_reach_runtime_metrics_json() {
    let lf = LockFreeMetrics::new();
    // Bump a couple of counters to non-zero, distinct values. These need to
    // appear in the final serialized JSON for the dashboard to ever see them.
    lf.actions_pushed_throttle_total
        .fetch_add(7, Ordering::Relaxed);
    lf.actions_rejected_shape_total
        .fetch_add(2, Ordering::Relaxed);
    lf.actions_pushed_freeze_total
        .fetch_add(3, Ordering::Relaxed);
    lf.actions_pushed_set_sysctl_total
        .fetch_add(11, Ordering::Relaxed);
    lf.actions_pushed_raw_total.fetch_add(5, Ordering::Relaxed);
    lf.commit();

    let snap = lf.snapshot();
    let mut state = fresh_metrics_state();
    state.sync_from_lockfree(&snap);

    let json = serde_json::to_string(&state.metrics).expect("serialize RuntimeMetrics");

    // Print the relevant slice for human verification (run with --nocapture).
    for field in [
        "actions_pushed_throttle_total",
        "actions_pushed_freeze_total",
        "actions_pushed_set_sysctl_total",
        "actions_pushed_raw_total",
        "actions_rejected_shape_total",
    ] {
        if let Some(idx) = json.find(field) {
            let end = (idx + field.len() + 16).min(json.len());
            println!("[smoke] {}: ...{}...", field, &json[idx..end]);
        } else {
            println!("[smoke] {}: NOT FOUND", field);
        }
    }

    // Each Fase 5 counter must round-trip into the JSON dashboard payload.
    assert!(
        json.contains("\"actions_pushed_throttle_total\":7"),
        "actions_pushed_throttle_total absent or wrong: {}",
        json
    );
    assert!(
        json.contains("\"actions_rejected_shape_total\":2"),
        "actions_rejected_shape_total absent or wrong: {}",
        json
    );
    assert!(
        json.contains("\"actions_pushed_freeze_total\":3"),
        "actions_pushed_freeze_total absent or wrong: {}",
        json
    );
    assert!(
        json.contains("\"actions_pushed_set_sysctl_total\":11"),
        "actions_pushed_set_sysctl_total absent or wrong: {}",
        json
    );
    assert!(
        json.contains("\"actions_pushed_raw_total\":5"),
        "actions_pushed_raw_total absent or wrong: {}",
        json
    );
}

#[test]
fn fase5_all_eleven_action_counters_reach_runtime_metrics() {
    // Belt-and-braces: every one of the 11 Fase 5 counters must be wired
    // into the sync flush + serialized to JSON. This catches a typo or
    // omission in `sync_from_lockfree`.
    let lf = LockFreeMetrics::new();
    lf.actions_pushed_throttle_total
        .fetch_add(1, Ordering::Relaxed);
    lf.actions_pushed_freeze_total
        .fetch_add(2, Ordering::Relaxed);
    lf.actions_pushed_unfreeze_total
        .fetch_add(3, Ordering::Relaxed);
    lf.actions_pushed_boost_total.fetch_add(4, Ordering::Relaxed);
    lf.actions_pushed_set_memorystatus_total
        .fetch_add(5, Ordering::Relaxed);
    lf.actions_pushed_set_thread_qos_total
        .fetch_add(6, Ordering::Relaxed);
    lf.actions_pushed_set_sysctl_total
        .fetch_add(7, Ordering::Relaxed);
    lf.actions_pushed_toggle_spotlight_total
        .fetch_add(8, Ordering::Relaxed);
    lf.actions_pushed_quarantine_daemon_total
        .fetch_add(9, Ordering::Relaxed);
    lf.actions_pushed_raw_total.fetch_add(10, Ordering::Relaxed);
    lf.actions_rejected_shape_total
        .fetch_add(11, Ordering::Relaxed);
    lf.commit();

    let snap = lf.snapshot();
    let mut state = fresh_metrics_state();
    state.sync_from_lockfree(&snap);

    // Validate via the strongly-typed RuntimeMetrics fields too — this
    // catches "field exists in serde JSON but reads default zero" cases.
    assert_eq!(state.metrics.actions_pushed_throttle_total, 1);
    assert_eq!(state.metrics.actions_pushed_freeze_total, 2);
    assert_eq!(state.metrics.actions_pushed_unfreeze_total, 3);
    assert_eq!(state.metrics.actions_pushed_boost_total, 4);
    assert_eq!(state.metrics.actions_pushed_set_memorystatus_total, 5);
    assert_eq!(state.metrics.actions_pushed_set_thread_qos_total, 6);
    assert_eq!(state.metrics.actions_pushed_set_sysctl_total, 7);
    assert_eq!(state.metrics.actions_pushed_toggle_spotlight_total, 8);
    assert_eq!(state.metrics.actions_pushed_quarantine_daemon_total, 9);
    assert_eq!(state.metrics.actions_pushed_raw_total, 10);
    assert_eq!(state.metrics.actions_rejected_shape_total, 11);

    // And one final round-trip via JSON to prove the serde derives are wired.
    let json = serde_json::to_string(&state.metrics).expect("serialize RuntimeMetrics");
    for (field, expected) in [
        ("actions_pushed_throttle_total", 1u64),
        ("actions_pushed_freeze_total", 2),
        ("actions_pushed_unfreeze_total", 3),
        ("actions_pushed_boost_total", 4),
        ("actions_pushed_set_memorystatus_total", 5),
        ("actions_pushed_set_thread_qos_total", 6),
        ("actions_pushed_set_sysctl_total", 7),
        ("actions_pushed_toggle_spotlight_total", 8),
        ("actions_pushed_quarantine_daemon_total", 9),
        ("actions_pushed_raw_total", 10),
        ("actions_rejected_shape_total", 11),
    ] {
        let needle = format!("\"{}\":{}", field, expected);
        assert!(
            json.contains(&needle),
            "field '{}' missing or wrong value (expected {}): {}",
            field,
            expected,
            json
        );
    }
}
