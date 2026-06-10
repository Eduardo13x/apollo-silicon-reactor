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
        survival_window: apollo_engine::engine::survival_window::SurvivalActivationWindow::new(),
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

#[test]
fn sync_from_lockfree_does_not_clobber_executor_action_totals() {
    let lf = LockFreeMetrics::new();
    lf.inc_cycles();
    lf.commit();

    let snap = lf.snapshot();
    let mut state = fresh_metrics_state();
    state.metrics.boosts_applied = 17;
    state.metrics.throttles_applied = 11;
    state.metrics.freezes_applied = 7;
    state.metrics.unfreezes_applied = 5;
    state.metrics.throttle_reverted = 3;

    state.sync_from_lockfree(&snap);

    assert_eq!(state.metrics.cycles, 1);
    assert_eq!(state.metrics.boosts_applied, 17);
    assert_eq!(state.metrics.throttles_applied, 11);
    assert_eq!(state.metrics.freezes_applied, 7);
    assert_eq!(state.metrics.unfreezes_applied, 5);
    assert_eq!(state.metrics.throttle_reverted, 3);
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

/// Phase 5.3 — Journal Rationale attachment counter round-trip.
#[test]
fn phase53_journal_rationales_attached_reach_runtime_metrics_json() {
    let lf = LockFreeMetrics::new();
    for _ in 0..29 {
        lf.inc_journal_rationale_attached();
    }
    lf.commit();
    let snap = lf.snapshot();
    let mut state = fresh_metrics_state();
    state.sync_from_lockfree(&snap);
    assert_eq!(state.metrics.journal_rationales_attached_total, 29);
    let json = serde_json::to_string(&state.metrics).unwrap();
    assert!(json.contains("\"journal_rationales_attached_total\":29"));
}

/// Phase 2 god-lock decomposition (Sprint 8, 2026-05-16) — round-trip
/// test for the migration of `habituation_skips` from a `state.metrics`
/// mutex write to the LSE counter. Verifies:
///   1. The new `LSE_COUNTERS.add_habituation_skips(N)` writes to
///      `LockFreeMetrics::habituation_skips_total`.
///   2. `MetricsSnapshot.habituation_skips_total` surfaces the value.
///   3. `MetricsState::sync_from_lockfree` populates the legacy
///      `RuntimeMetrics.habituation_skips` field from the atomic (single
///      source of truth — AIS reads `rm_u("habituation_skips")` and the
///      JSON dashboard reads the same key).
#[test]
fn phase_2_habituation_skips_round_trip() {
    let lf = LockFreeMetrics::new();
    lf.add_habituation_skips(17);
    lf.commit();

    let snap = lf.snapshot();
    assert_eq!(
        snap.habituation_skips_total, 17,
        "lock-free snapshot did not surface habituation_skips_total"
    );

    let mut state = fresh_metrics_state();
    state.sync_from_lockfree(&snap);
    assert_eq!(
        state.metrics.habituation_skips, 17,
        "sync_from_lockfree did not populate legacy habituation_skips field"
    );

    let json = serde_json::to_string(&state.metrics).expect("serialize RuntimeMetrics");
    assert!(
        json.contains("\"habituation_skips\":17"),
        "habituation_skips absent or wrong in JSON: {}",
        json
    );
}

/// Phase 4.3.1 (Sprint 8, 2026-05-16) — round-trip test for the
/// `specialist_accuracy_purge_inhibitions_total` observability counter.
///
/// Same shape as `phase5_1_user_presence_suppressions_round_trip`: bump
/// the lock-free counter, snapshot, sync into `RuntimeMetrics`, and
/// prove the value survives to the serialized JSON the dashboard reads.
#[test]
fn phase4_3_1_specialist_accuracy_purge_inhibitions_round_trip() {
    let lf = LockFreeMetrics::new();
    // Bump three times so a single-increment-only bug would also fail.
    lf.inc_specialist_accuracy_purge_inhibitions();
    lf.inc_specialist_accuracy_purge_inhibitions();
    lf.inc_specialist_accuracy_purge_inhibitions();
    lf.commit();

    let snap = lf.snapshot();
    assert_eq!(
        snap.specialist_accuracy_purge_inhibitions_total, 3,
        "lock-free snapshot did not surface the counter"
    );

    let mut state = fresh_metrics_state();
    state.sync_from_lockfree(&snap);
    assert_eq!(
        state.metrics.specialist_accuracy_purge_inhibitions_total, 3,
        "sync_from_lockfree did not flush specialist_accuracy_purge_inhibitions_total"
    );

    let json = serde_json::to_string(&state.metrics).expect("serialize RuntimeMetrics");
    assert!(
        json.contains("\"specialist_accuracy_purge_inhibitions_total\":3"),
        "specialist_accuracy_purge_inhibitions_total absent or wrong in JSON: {}",
        json
    );
}

/// Phase C SCORER-OVERRIDE (Sprint 11 finale, 2026-05-16) — round-trip
/// test for the asymmetric scorer/gate disagreement counters. Mirrors the
/// Phase 5.1 / 5.3 pattern: bump the lock-free counter, snapshot, sync into
/// `RuntimeMetrics`, and prove the value survives to the serialized JSON
/// the operator reads.
///
/// Tests both counters in one #[test] (single LockFreeMetrics instance)
/// since they share the same wiring shape and skipping one would still
/// pass an "additive" wiring bug — interleaving here catches the case
/// where someone copy-pasted a sync line and forgot to rename the field.
#[test]
fn phase_c_scorer_override_counters_reach_runtime_metrics_json() {
    let lf = LockFreeMetrics::new();
    // Distinct, non-trivial values so a swap or off-by-one bug fails loudly.
    for _ in 0..7 {
        lf.inc_scorer_override_reject();
    }
    for _ in 0..13 {
        lf.inc_scorer_disagreement_strong_accept();
    }
    lf.commit();

    let snap = lf.snapshot();
    assert_eq!(
        snap.scorer_override_rejects_total, 7,
        "lock-free snapshot dropped scorer_override_rejects_total"
    );
    assert_eq!(
        snap.scorer_disagreement_strong_accepts_total, 13,
        "lock-free snapshot dropped scorer_disagreement_strong_accepts_total"
    );

    let mut state = fresh_metrics_state();
    state.sync_from_lockfree(&snap);
    assert_eq!(
        state.metrics.scorer_override_rejects_total, 7,
        "sync_from_lockfree did not flush scorer_override_rejects_total"
    );
    assert_eq!(
        state.metrics.scorer_disagreement_strong_accepts_total, 13,
        "sync_from_lockfree did not flush scorer_disagreement_strong_accepts_total"
    );

    let json = serde_json::to_string(&state.metrics).expect("serialize RuntimeMetrics");
    assert!(
        json.contains("\"scorer_override_rejects_total\":7"),
        "scorer_override_rejects_total absent/wrong in JSON: {}",
        json
    );
    assert!(
        json.contains("\"scorer_disagreement_strong_accepts_total\":13"),
        "scorer_disagreement_strong_accepts_total absent/wrong in JSON: {}",
        json
    );
}

/// Sprint 13 Pressure-Router Gate (2026-05-30) — closes the loop on the
/// `companion_observe_router_skips_total` counter wired into the daemon
/// main loop. Drives a 200-cycle synthetic pressure stream at p=0.10
/// (below the default 0.30 `learned_mid_entry`) and asserts:
///
///   1. The skip counter accumulates ~150 (out of 200) — matching the
///      modulo-4 forced-exploration fallback (every 4th cycle goes through
///      regardless, so 50 hits + 150 skips = 75% skip ratio).
///   2. The companion graph `edge_count()` still grows monotonically across
///      the run, proving the modulo-4 fallback keeps the Lift denominator
///      updating instead of letting the graph go stale.
///   3. The counter round-trips end-to-end through `sync_from_lockfree`
///      into `RuntimeMetrics` and the serialized JSON the operator reads.
///
/// Paper backing:
/// - Adaptive router gate pattern: [signal_intelligence.rs:404-424]
///   (MoR-style conditional compute)
/// - Forced-exploration fallback: [Sutton & Barto §2.7]
/// - Skip-with-counter telemetry shape: Sprint 12 G12 `bus_saturated`
///   precedent (commit `5f1c984`).
#[test]
fn companion_observe_pressure_gated() {
    use apollo_engine::engine::companion_graph::CompanionGraph;
    use apollo_engine::engine::signal_intelligence::SignalIntelligence;

    let si = SignalIntelligence::new();
    // Default zone — no workload offsets persisted → mid_entry = 0.30.
    let (mid_entry, _) = si.effective_zones(0);
    assert!(
        mid_entry > 0.15,
        "sanity: default mid_entry should be > 0.15, got {mid_entry}"
    );

    let lf = LockFreeMetrics::new();
    let mut companion_graph = CompanionGraph::new();
    let mut edge_counts: Vec<usize> = Vec::with_capacity(50);

    // Synthetic alive set with names that exceed the SSO threshold (≈23
    // bytes) so the `alive: Vec<String>` clones we skip in prod are real
    // heap allocations, not inline strings.
    let alive_set: Vec<String> = vec![
        "com.apple.WebKit.WebContent.AnchorOne".to_string(),
        "com.apple.WebKit.WebContent.AnchorTwo".to_string(),
        "com.apple.Music.WidgetExtension".to_string(),
        "com.apple.coreservices.uiagent".to_string(),
    ];
    const FG_APP: &str = "com.apple.Safari.Anchor";

    // Synthetic pressure floor below the mid_entry → gate closes except
    // on the modulo-4 cycles.
    let synthetic_pressure = 0.10_f64;

    for cycle in 0..200u64 {
        let pressure_router_open = synthetic_pressure >= mid_entry || cycle.is_multiple_of(4);
        if pressure_router_open {
            companion_graph.observe_cycle(Some(FG_APP), &alive_set, cycle);
        } else {
            lf.inc_companion_observe_router_skip();
        }
        // Sample edge_count() at every observe to verify monotonicity.
        if pressure_router_open && cycle.is_multiple_of(8) {
            edge_counts.push(companion_graph.edge_count());
        }
    }

    lf.commit();
    let snap = lf.snapshot();

    // Cycles 0,4,8,...,196 are multiples of 4 → 50 allowed, 150 skipped.
    assert_eq!(
        snap.companion_observe_router_skips_total, 150,
        "expected 150 skips out of 200 cycles (75% skip ratio under pressure < mid_entry, \
         modulo-4 fallback every 4th cycle), got {}",
        snap.companion_observe_router_skips_total
    );

    // Modulo-4 fallback must keep observing → edges grow monotonically.
    assert!(
        edge_counts.len() >= 2,
        "expected at least two edge_count samples, got {}",
        edge_counts.len()
    );
    for win in edge_counts.windows(2) {
        assert!(
            win[1] >= win[0],
            "companion_graph.edge_count() decreased between samples ({} -> {}) — \
             modulo-4 fallback must keep the Lift denominator updating",
            win[0],
            win[1]
        );
    }
    assert!(
        companion_graph.edge_count() > 0,
        "modulo-4 fallback fired 50 times but edge_count() == 0 — observe_cycle never recorded"
    );

    // End-to-end round-trip into RuntimeMetrics JSON, mirroring the Phase
    // 3.3 / Phase 4.1 / Phase C tests above.
    let mut state = fresh_metrics_state();
    state.sync_from_lockfree(&snap);
    assert_eq!(
        state.metrics.companion_observe_router_skips_total, 150,
        "sync_from_lockfree did not flush companion_observe_router_skips_total"
    );
    let json = serde_json::to_string(&state.metrics).expect("serialize RuntimeMetrics");
    assert!(
        json.contains("\"companion_observe_router_skips_total\":150"),
        "companion_observe_router_skips_total absent/wrong in JSON: {}",
        json
    );
}
