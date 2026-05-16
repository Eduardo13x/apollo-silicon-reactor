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

use apollo_engine::engine::daemon_state::{MetricsState, ReactorStatus};
use apollo_engine::engine::lse_counters::LockFreeMetrics;
use apollo_engine::engine::types::RuntimeMetrics;

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
    lf.actions_pushed_boost_total
        .fetch_add(4, Ordering::Relaxed);
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

/// Batch 1 (Sprint 6/7/8) telemetry round-trip: the 7 counters added by
/// phases 3.2 / 4.2 / 4.3 / 5.2 must all survive the full chain
/// LockFreeMetrics → snapshot → sync_from_lockfree → RuntimeMetrics → JSON.
///
/// Sprint 3 telemetry-death scar: counter increments on the hot path but
/// never reaches the dashboard. NotebookLM 2026-05-16 flagged ~12 manual
/// merge resolutions in the metric sync chain as a potential reintroduction
/// of that bug. This test pins each new counter to a distinct literal so
/// regressions surface as test failures, not as silent dashboard zeros.
#[test]
fn batch1_phases_32_42_43_52_counters_reach_runtime_metrics_json() {
    let lf = LockFreeMetrics::new();

    // Phase 3.2 — arousal-modulated NARS decay
    lf.add_arousal_decay_accelerations(13);

    // Phase 4.2 — external-event causal attribution
    lf.inc_causal_external_thermal_blame();
    lf.inc_causal_external_thermal_blame();
    lf.inc_causal_external_disk_blame();
    lf.inc_causal_external_disk_blame();
    lf.inc_causal_external_disk_blame();
    for _ in 0..4 {
        lf.inc_causal_external_net_blame();
    }

    // Phase 4.3 — policy rollback guard
    for _ in 0..5 {
        lf.inc_policy_rollback_evaluation();
    }
    lf.inc_policy_rollback_execution();
    lf.inc_policy_rollback_execution();

    // Phase 5.2 — battery-aware cost penalty
    for _ in 0..6 {
        lf.inc_battery_aware_penalty_emission();
    }

    let snap = lf.snapshot();
    let mut state = fresh_metrics_state();
    state.sync_from_lockfree(&snap);

    let json = serde_json::to_string(&state.metrics).expect("serialize RuntimeMetrics");

    for (field, expected) in [
        ("arousal_decay_accelerations_total", 13u64),
        ("causal_external_thermal_blames_total", 2),
        ("causal_external_disk_blames_total", 3),
        ("causal_external_net_blames_total", 4),
        ("policy_rollback_evaluations_total", 5),
        ("policy_rollback_executions_total", 2),
        ("battery_aware_penalty_emissions_total", 6),
    ] {
        let needle = format!("\"{}\":{}", field, expected);
        assert!(
            json.contains(&needle),
            "Batch 1 counter '{}' missing or wrong value (expected {}): {}",
            field,
            expected,
            &json[..json.len().min(500)]
        );
    }
}

/// Phase 3.3 — Cross-Group Companion Attention counter round-trip.
#[test]
fn phase33_companion_cross_group_inferences_reach_runtime_metrics_json() {
    let lf = LockFreeMetrics::new();
    lf.add_companion_cross_group_inferences(37);
    lf.commit();

    let snap = lf.snapshot();
    let mut state = fresh_metrics_state();
    state.sync_from_lockfree(&snap);

    assert_eq!(
        state.metrics.companion_cross_group_inferences_total, 37,
        "counter must round-trip into RuntimeMetrics via sync_from_lockfree"
    );

    let json = serde_json::to_string(&state.metrics).expect("serialize RuntimeMetrics");
    assert!(
        json.contains("\"companion_cross_group_inferences_total\":37"),
        "field absent or wrong value in JSON: {}",
        json
    );
}

/// Phase 4.1 — Adaptive Drift Threshold counter round-trip.
#[test]
fn phase41_adaptive_drift_threshold_raises_reach_runtime_metrics_json() {
    let lf = LockFreeMetrics::new();
    lf.add_adaptive_drift_threshold_raises(23);
    lf.commit();

    let snap = lf.snapshot();
    let mut state = fresh_metrics_state();
    state.sync_from_lockfree(&snap);

    assert_eq!(
        state.metrics.adaptive_drift_threshold_raises_total, 23,
        "counter must round-trip into RuntimeMetrics via sync_from_lockfree"
    );

    let json = serde_json::to_string(&state.metrics).expect("serialize RuntimeMetrics");
    assert!(
        json.contains("\"adaptive_drift_threshold_raises_total\":23"),
        "field absent or wrong value in JSON: {}",
        json
    );
}

/// Phase 5.1 — User-Presence Suppression counter round-trip.
#[test]
fn phase51_user_presence_suppressions_reach_runtime_metrics_json() {
    let lf = LockFreeMetrics::new();
    lf.add_user_presence_suppressions(19);
    lf.commit();
    let snap = lf.snapshot();
    let mut state = fresh_metrics_state();
    state.sync_from_lockfree(&snap);
    assert_eq!(state.metrics.user_presence_suppressions_total, 19);
    let json = serde_json::to_string(&state.metrics).unwrap();
    assert!(json.contains("\"user_presence_suppressions_total\":19"));
}
