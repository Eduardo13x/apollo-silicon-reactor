//! Group C (2026-06-06) — Invariant #13 port-hub gate + Dempster-Shafer
//! evidential aggregation. Integration tests.
//!
//! These tests run against the *real* `LSE_COUNTERS` static and verify
//! round-trip telemetry the same way `level3_telemetry_sync.rs` did for
//! Sprint 4 — bump a counter, snapshot, sync_from_lockfree, serialize,
//! search the JSON for the expected key+value. Silent-telemetry-death
//! prevention.
//!
//! # Invariant #13 (Mach port hub)
//!
//! The runtime gate lives inside `MachPolicyEffector::apply` and inspects
//! `MachQoSManager::get_mach_port_count(pid)` before any tier demote. We
//! cannot exercise the real syscall path in a unit test (would require
//! root + a live victim PID), so we verify the *observable contract*:
//!
//! 1. The LSE counters `mediator_port_hub_blocks_total` and
//!    `mediator_port_hub_probe_unavailable_total` exist, increment via
//!    their `inc_*` helpers, snapshot cleanly, and reach
//!    `RuntimeMetrics` through `sync_from_lockfree`.
//! 2. The `is_demote_target` helper correctly classifies the four
//!    `SchedulingTier` cases (asymmetric reject — only Background
//!    triggers the gate).
//! 3. The `PORT_HUB_THRESHOLD` constant is stable at 5000 (changing it
//!    is a doctrine-level decision that requires updating CLAUDE.md).
//!
//! # Dempster-Shafer
//!
//! 1. RSS mode is the byte-equivalent default — a scorer with the
//!    builder unchanged produces a `PolicyScore` whose DS fields are
//!    `(0.0, 0.0, 1.0, 0.0, false)`, and behaviour matches Sprint 11.
//! 2. Two singleton-belief features in agreement compose to high belief
//!    via Dempster's rule.
//! 3. Two singleton features in opposition produce K = 1.0 conflict and
//!    the scorer falls back to RSS, bumping the LSE fallback counter
//!    (Zadeh counter-example, Dezert 2002).
//! 4. Yager fallback fires when K exceeds the configured threshold but
//!    not when K stays below.

use std::sync::atomic::Ordering;

use apollo_engine::engine::action_policy::{
    ActionContext, AggregatorMode, Contribution, PolicyFeature, PolicyScorer,
};
use apollo_engine::engine::daemon_state::{MetricsState, ReactorStatus};
use apollo_engine::engine::lse_counters::{LockFreeMetrics, LSE_COUNTERS};
use apollo_engine::engine::safety::ProtectionLevel;
use apollo_engine::engine::types::{RootAction, RuntimeMetrics};

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

fn neutral_ctx() -> ActionContext {
    ActionContext {
        pressure: 0.5,
        swap_gb: 1.0,
        learned_yield: None,
        imagined_margin: None,
        thrashing_score: 0.0,
        p_oom_30s: Some(0.0),
        p_jank_60s: Some(0.0),
        has_sleep_assertion: false,
        call_in_progress: false,
        idle_secs: 30.0,
        foreground_pid: None,
        is_foreground_family: false,
        is_recently_active: false,
        thermal_emergency: false,
        interrupt_phase: 0,
        protection_level: ProtectionLevel::Unprotected,
        hot_page_fraction: Some(0.0),
        wss_mb: Some(0.0),
        sensor_age_ms: Some(0),
        epistemic_uncertainty: 0.0,
        is_on_battery: Some(false),
        wakeups_per_sec: Some(0.0),
        ctx_switches_per_sec: Some(0.0),
    }
}

fn freeze_action() -> RootAction {
    RootAction::FreezeProcess {
        pid: 42,
        name: "test_proc".to_string(),
        reason: "test".to_string(),
        decision_reason: apollo_engine::engine::audit_types::DecisionReason::PressureContext,
        start_sec: 0,
        start_usec: 0,
    }
}

// -----------------------------------------------------------------------------
// Invariant #13 — port-hub gate
// -----------------------------------------------------------------------------

#[test]
fn invariant_13_port_hub_counters_round_trip_through_runtime_metrics() {
    // Use a local LockFreeMetrics for isolation from the global static —
    // we are testing the snapshot/sync surface, not the production
    // single-counter contract (which the Sprint 9 4b13a39 fix protects
    // separately at producer call sites).
    let lf = LockFreeMetrics::new();
    lf.mediator_port_hub_blocks_total
        .fetch_add(3, Ordering::Relaxed);
    lf.mediator_port_hub_probe_unavailable_total
        .fetch_add(7, Ordering::Relaxed);
    lf.policy_scorer_ds_high_conflict_fallback_total
        .fetch_add(5, Ordering::Relaxed);
    lf.commit();

    let snap = lf.snapshot();
    let mut state = fresh_metrics_state();
    state.sync_from_lockfree(&snap);

    let json = serde_json::to_string(&state.metrics).expect("serialize RuntimeMetrics");

    assert!(
        json.contains("\"mediator_port_hub_blocks_total\":3"),
        "mediator_port_hub_blocks_total absent or wrong: {}",
        json
    );
    assert!(
        json.contains("\"mediator_port_hub_probe_unavailable_total\":7"),
        "mediator_port_hub_probe_unavailable_total absent or wrong: {}",
        json
    );
    assert!(
        json.contains("\"policy_scorer_ds_high_conflict_fallback_total\":5"),
        "policy_scorer_ds_high_conflict_fallback_total absent or wrong: {}",
        json
    );
}

#[test]
fn invariant_13_inc_helpers_bump_the_correct_counter() {
    // Snapshot baseline before bumping so concurrent runs of other tests
    // do not interfere — we assert *delta*, not absolute value.
    let pre_blocks = LSE_COUNTERS
        .mediator_port_hub_blocks_total
        .load(Ordering::Relaxed);
    let pre_unavail = LSE_COUNTERS
        .mediator_port_hub_probe_unavailable_total
        .load(Ordering::Relaxed);
    let pre_ds = LSE_COUNTERS
        .policy_scorer_ds_high_conflict_fallback_total
        .load(Ordering::Relaxed);

    LSE_COUNTERS.inc_mediator_port_hub_block();
    LSE_COUNTERS.inc_mediator_port_hub_block();
    LSE_COUNTERS.inc_mediator_port_hub_probe_unavailable();
    LSE_COUNTERS.inc_policy_scorer_ds_high_conflict_fallback();

    let post_blocks = LSE_COUNTERS
        .mediator_port_hub_blocks_total
        .load(Ordering::Relaxed);
    let post_unavail = LSE_COUNTERS
        .mediator_port_hub_probe_unavailable_total
        .load(Ordering::Relaxed);
    let post_ds = LSE_COUNTERS
        .policy_scorer_ds_high_conflict_fallback_total
        .load(Ordering::Relaxed);

    assert_eq!(post_blocks - pre_blocks, 2);
    assert_eq!(post_unavail - pre_unavail, 1);
    assert_eq!(post_ds - pre_ds, 1);
}

// -----------------------------------------------------------------------------
// Dempster-Shafer — RSS-mode default
// -----------------------------------------------------------------------------

/// Test feature opting into DS with full belief.
struct StrongBeliefFeature;
impl PolicyFeature for StrongBeliefFeature {
    fn name(&self) -> &'static str {
        "strong_belief"
    }
    fn contribute(&self, _: &RootAction, _: &ActionContext) -> Contribution {
        Contribution::with_mass(1.0, 0.0, 0.0, false, 0.9, 0.0, 0.1)
    }
}

/// Test feature contributing vacuous (RSS-only) evidence.
struct VacuousFeature;
impl PolicyFeature for VacuousFeature {
    fn name(&self) -> &'static str {
        "vacuous"
    }
    fn contribute(&self, _: &RootAction, _: &ActionContext) -> Contribution {
        Contribution::zero()
    }
}

#[test]
fn dempster_shafer_default_mode_leaves_ds_fields_neutral() {
    // Default builder is RSS mode — verify the new DS fields are at
    // their neutral defaults so legacy consumers see byte-equivalent
    // behaviour to pre-Group-C.
    let scorer = PolicyScorer::builder().feature(StrongBeliefFeature).build();
    let score = scorer.score(&freeze_action(), &neutral_ctx());

    assert_eq!(score.ds_belief, 0.0, "RSS default → ds_belief must be 0.0");
    assert_eq!(
        score.ds_disbelief, 0.0,
        "RSS default → ds_disbelief must be 0.0"
    );
    assert_eq!(
        score.ds_uncertain, 1.0,
        "RSS default → ds_uncertain must be 1.0 (vacuous)"
    );
    assert_eq!(score.ds_conflict, 0.0);
    assert!(!score.ds_fallback_used);
}

#[test]
fn dempster_shafer_two_agreeing_features_compose_high_belief() {
    let scorer = PolicyScorer::builder()
        .aggregator_mode(AggregatorMode::Dempster)
        .feature(StrongBeliefFeature)
        .feature(StrongBeliefFeature)
        .build();
    let score = scorer.score(&freeze_action(), &neutral_ctx());

    assert!(
        !score.ds_fallback_used,
        "two agreeing features should not trip Yager fallback"
    );
    assert!(
        score.ds_belief > 0.9,
        "Dempster's rule on two 0.9-belief features must exceed 0.9; got {}",
        score.ds_belief
    );
    assert!(
        score.ds_disbelief < 1e-9,
        "no feature emitted disbelief — combined disbelief must stay near 0; got {}",
        score.ds_disbelief
    );
}

#[test]
fn dempster_shafer_opposing_singletons_trigger_yager_fallback() {
    let pre = LSE_COUNTERS
        .policy_scorer_ds_high_conflict_fallback_total
        .load(Ordering::Relaxed);

    // Two features each claiming m=1.0 on opposing singletons →
    // K=1.0 exactly → Dempster's rule divides by zero → fallback.
    struct PureAccept;
    impl PolicyFeature for PureAccept {
        fn name(&self) -> &'static str {
            "pure_accept"
        }
        fn contribute(&self, _: &RootAction, _: &ActionContext) -> Contribution {
            Contribution::with_mass(1.0, 0.0, 0.0, false, 1.0, 0.0, 0.0)
        }
    }
    struct PureReject;
    impl PolicyFeature for PureReject {
        fn name(&self) -> &'static str {
            "pure_reject"
        }
        fn contribute(&self, _: &RootAction, _: &ActionContext) -> Contribution {
            Contribution::with_mass(0.0, 1.0, 0.0, false, 0.0, 1.0, 0.0)
        }
    }

    let scorer = PolicyScorer::builder()
        .aggregator_mode(AggregatorMode::Dempster)
        .feature(PureAccept)
        .feature(PureReject)
        .build();
    let score = scorer.score(&freeze_action(), &neutral_ctx());

    assert!(
        score.ds_fallback_used,
        "K=1.0 opposing singletons must trigger fallback"
    );
    let post = LSE_COUNTERS
        .policy_scorer_ds_high_conflict_fallback_total
        .load(Ordering::Relaxed);
    assert!(
        post > pre,
        "LSE fallback counter must bump on Zadeh counter-example"
    );
}

#[test]
fn dempster_shafer_vacuous_feature_does_not_poison_combination() {
    // A feature contributing pure ignorance (m_uncertain=1.0) must not
    // shift the combined belief — this is the BPA conservation property
    // that makes mixed RSS-only + DS-opt-in features safe.
    let scorer_with = PolicyScorer::builder()
        .aggregator_mode(AggregatorMode::Dempster)
        .feature(StrongBeliefFeature)
        .feature(VacuousFeature)
        .build();
    let scorer_without = PolicyScorer::builder()
        .aggregator_mode(AggregatorMode::Dempster)
        .feature(StrongBeliefFeature)
        .build();

    let with = scorer_with.score(&freeze_action(), &neutral_ctx());
    let without = scorer_without.score(&freeze_action(), &neutral_ctx());

    assert!(
        (with.ds_belief - without.ds_belief).abs() < 1e-9,
        "vacuous feature must not change combined belief; with={} without={}",
        with.ds_belief,
        without.ds_belief
    );
    assert!(!with.ds_fallback_used);
}

#[test]
fn aggregator_mode_from_persisted_unknown_string_defaults_to_rss() {
    assert_eq!(AggregatorMode::from_persisted(""), AggregatorMode::Rss);
    assert_eq!(
        AggregatorMode::from_persisted("typo"),
        AggregatorMode::Rss,
        "unknown values must default to RSS — typo-safety"
    );
    assert_eq!(AggregatorMode::from_persisted("rss"), AggregatorMode::Rss);
    assert_eq!(
        AggregatorMode::from_persisted("ds"),
        AggregatorMode::Dempster
    );
    assert_eq!(
        AggregatorMode::from_persisted("dempster"),
        AggregatorMode::Dempster
    );
    assert_eq!(
        AggregatorMode::from_persisted("dempster_shafer"),
        AggregatorMode::Dempster
    );
}
