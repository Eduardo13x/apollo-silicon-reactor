use apollo_engine::engine::reflex::{
    LatestReasoningWorker, ReasoningIdentity, ReasoningLookup, ReasoningSnapshot, ReflexActionKind,
    ReflexAdvice, ReflexBlocker, ReflexBroker, ReflexDecision, ReflexHealthSample, ReflexIntent,
    ReflexRolloutPhase, ReflexSafetyContext, ReflexSource, ReflexTarget, ReflexTrigger,
};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

fn target(pid: u32) -> ReflexTarget {
    ReflexTarget {
        pid,
        start_sec: 100,
        start_usec: 7,
        name: "Editor".to_string(),
    }
}

fn intent(cycle: u64) -> ReflexIntent {
    ReflexIntent::new(
        ReflexActionKind::InteractionQos,
        Some(target(42)),
        ReflexTrigger::Input,
        ReflexSource::Deterministic,
        cycle,
        1_600,
    )
    .expect("valid deterministic intent")
}

fn safe() -> ReflexSafetyContext {
    ReflexSafetyContext {
        identity_present: true,
        identity_start_nonzero: true,
        identity_recheck_ok: true,
        capability_available: true,
        ..ReflexSafetyContext::default()
    }
}

#[test]
fn closed_catalog_rejects_invalid_intents_and_bounds_ttl() {
    assert!(ReflexIntent::new(
        ReflexActionKind::InteractionQos,
        None,
        ReflexTrigger::Input,
        ReflexSource::Deterministic,
        1,
        1_600,
    )
    .is_err());
    assert!(ReflexIntent::new(
        ReflexActionKind::TemporalBoost,
        Some(target(42)),
        ReflexTrigger::Input,
        ReflexSource::Deterministic,
        1,
        12_001,
    )
    .is_err());
    assert!(ReflexIntent::new(
        ReflexActionKind::MarkovPrewarm,
        None,
        ReflexTrigger::Prediction,
        ReflexSource::Markov,
        1,
        5_000,
    )
    .is_err());
    assert!(ReflexIntent::new(
        ReflexActionKind::MarkovPrewarm,
        Some(target(42)),
        ReflexTrigger::Prediction,
        ReflexSource::Markov,
        1,
        5_000,
    )
    .is_ok());
}

#[test]
fn web_and_universal_network_are_typed_triggers_not_new_actuators() {
    for trigger in [ReflexTrigger::WebNavigation, ReflexTrigger::NetworkActivity] {
        let candidate = ReflexIntent::new(
            ReflexActionKind::InteractionQos,
            Some(target(42)),
            trigger,
            ReflexSource::Deterministic,
            1,
            1_200,
        )
        .expect("flow trigger uses the existing closed catalog");
        assert_eq!(candidate.action, ReflexActionKind::InteractionQos);
        assert!(candidate.reversible);
    }
}

#[test]
fn missing_or_uncertain_advice_is_neutral_but_decisive_negative_vetoes() {
    let mut broker = ReflexBroker::active_for_test();
    assert_eq!(broker.decide(&intent(1), safe()), ReflexDecision::Admit);

    let uncertain = intent(2).with_advice(ReflexAdvice {
        score: -1.0,
        lower_bound: -1.0,
        upper_bound: -0.5,
        confidence: 0.2,
        authoritative: false,
        sources: vec![ReflexSource::WorldModel],
    });
    assert_eq!(broker.decide(&uncertain, safe()), ReflexDecision::Admit);

    let decisive = intent(3).with_advice(ReflexAdvice {
        score: -0.8,
        lower_bound: -0.9,
        upper_bound: -0.1,
        confidence: 0.95,
        authoritative: true,
        sources: vec![ReflexSource::WorldModel, ReflexSource::Causal],
    });
    assert_eq!(
        broker.decide(&decisive, safe()),
        ReflexDecision::Veto(ReflexBlocker::DecisiveModelVeto)
    );
}

#[test]
fn safety_identity_and_dedup_precede_admission() {
    let mut broker = ReflexBroker::active_for_test();
    let mut protected = safe();
    protected.target_protected = true;
    assert_eq!(
        broker.decide(&intent(1), protected),
        ReflexDecision::Skipped(ReflexBlocker::ProtectedTarget)
    );
    assert_eq!(broker.counters().protected_blocked, 1);
    assert_eq!(broker.counters().protected_admitted, 0);

    let mut recycled = safe();
    recycled.identity_recheck_ok = false;
    assert_eq!(
        broker.decide(&intent(2), recycled),
        ReflexDecision::Skipped(ReflexBlocker::IdentityMismatch)
    );
    assert_eq!(broker.counters().omitted, 2);
    assert_eq!(broker.counters().no_op, 0);

    assert_eq!(broker.decide(&intent(3), safe()), ReflexDecision::Admit);
    assert_eq!(
        broker.decide(&intent(3), safe()),
        ReflexDecision::Skipped(ReflexBlocker::Duplicate)
    );
    assert_eq!(broker.counters().omitted, 3);
    assert!(broker.counters().last_decision_latency_us < 10_000);
}

fn healthy(cycle: u64) -> ReflexHealthSample {
    ReflexHealthSample {
        cycle,
        p95_cycle_ms: 40.0,
        applied_total: cycle,
        reverted_total: cycle / 20,
        protected_admissions_total: 0,
        failures_total: 0,
        rollback_failures_total: 0,
        expected_profile: "adaptive-multicore".to_string(),
        compiled_profile: "adaptive-multicore".to_string(),
        effective_profile: "adaptive-multicore".to_string(),
        paused: false,
    }
}

#[test]
fn shadow_activates_on_exact_boundary_and_exposes_blocker() {
    let mut broker = ReflexBroker::new(true, 500, "adaptive-multicore");
    for cycle in 1..500 {
        broker.observe_health(healthy(cycle));
    }
    assert_eq!(broker.rollout().phase, ReflexRolloutPhase::Shadow);
    assert_eq!(broker.rollout().valid_cycles, 499);
    broker.observe_health(healthy(500));
    assert_eq!(broker.rollout().phase, ReflexRolloutPhase::Active);
    assert_eq!(broker.rollout().blocker, "ready");

    let mut mismatch = ReflexBroker::new(true, 500, "adaptive-multicore");
    for cycle in 1..=500 {
        let mut sample = healthy(cycle);
        sample.compiled_profile = "sequential".to_string();
        sample.effective_profile = "sequential".to_string();
        mismatch.observe_health(sample);
    }
    assert_eq!(mismatch.rollout().phase, ReflexRolloutPhase::Shadow);
    assert_eq!(mismatch.rollout().blocker, "compiled-profile-mismatch");
}

#[test]
fn rollout_never_shortens_the_required_500_cycle_shadow_window() {
    let mut broker = ReflexBroker::new(true, 1, "adaptive-multicore");
    assert_eq!(broker.rollout().shadow_cycles, 500);
    for cycle in 1..500 {
        broker.observe_health(healthy(cycle));
    }
    assert_eq!(broker.rollout().phase, ReflexRolloutPhase::Shadow);
    broker.observe_health(healthy(500));
    assert_eq!(broker.rollout().phase, ReflexRolloutPhase::Active);
}

#[test]
fn active_rollout_returns_to_shadow_when_effective_profile_drifts() {
    let mut broker = ReflexBroker::active_for_test();
    let mut sample = healthy(501);
    sample.expected_profile = "test".to_string();
    sample.compiled_profile = "test".to_string();
    sample.effective_profile = "sequential".to_string();

    broker.observe_health(sample);

    assert_eq!(broker.rollout().phase, ReflexRolloutPhase::Shadow);
    assert_eq!(broker.rollout().blocker, "effective-profile-mismatch");
}

#[test]
fn invalid_health_samples_do_not_advance_shadow() {
    let mut broker = ReflexBroker::new(true, 500, "adaptive-multicore");
    let mut invalid = healthy(1);
    invalid.p95_cycle_ms = f64::NAN;
    broker.observe_health(invalid);
    assert_eq!(broker.rollout().valid_cycles, 0);
    assert_eq!(broker.rollout().invalid_samples, 1);
}

#[test]
fn rollout_json_is_backward_compatible_and_corruption_falls_back_to_shadow() {
    let restored = ReflexBroker::restore_json(
        r#"{"schema_version":1,"enabled":true,"shadow_cycles":500,"build_profile":"adaptive-multicore","valid_cycles":12,"failures_seen":3}"#,
        "adaptive-multicore",
    );
    assert_eq!(restored.rollout().valid_cycles, 12);
    assert_eq!(restored.rollout().phase, ReflexRolloutPhase::Shadow);
    assert_eq!(restored.rollout().schema_version, 2);

    let corrupt = ReflexBroker::restore_json("{nope", "adaptive-multicore");
    assert_eq!(corrupt.rollout().valid_cycles, 0);
    assert_eq!(corrupt.rollout().blocker, "state-corrupt");
}

#[test]
fn restored_rollout_rebases_process_local_action_counters() {
    let json = r#"{
        "schema_version": 1,
        "enabled": true,
        "shadow_cycles": 500,
        "build_profile": "adaptive-multicore",
        "phase": "shadow",
        "valid_cycles": 0,
        "previous_applied": 99,
        "previous_reverted": 99
    }"#;
    let mut broker = ReflexBroker::restore_json(json, "adaptive-multicore");
    let mut sample = healthy(1);
    sample.applied_total = 1;
    sample.reverted_total = 1;
    broker.observe_health(sample);
    assert_eq!(broker.rollout().baseline_churn(), 1.0);
}

fn reasoning_identity(pid: u32, start_sec: u64) -> ReasoningIdentity {
    ReasoningIdentity {
        pid,
        start_sec,
        start_usec: 7,
    }
}

#[test]
fn reasoning_mailbox_overwrites_pending_work_with_latest_snapshot() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let worker_gate = Arc::clone(&gate);
    let worker = LatestReasoningWorker::spawn("reflex-contract", move |value: u64| {
        if value == 1 {
            let (lock, condvar) = &*worker_gate;
            let mut released = lock.lock().unwrap_or_else(|error| error.into_inner());
            while !*released {
                released = condvar
                    .wait(released)
                    .unwrap_or_else(|error| error.into_inner());
            }
        }
        value * 10
    })
    .expect("worker starts");
    let identity = reasoning_identity(42, 100);
    worker.submit(ReasoningSnapshot::new(1, identity, 1));
    std::thread::sleep(Duration::from_millis(10));
    worker.submit(ReasoningSnapshot::new(2, identity, 2));
    worker.submit(ReasoningSnapshot::new(3, identity, 3));

    {
        let (lock, condvar) = &*gate;
        *lock.lock().unwrap_or_else(|error| error.into_inner()) = true;
        condvar.notify_all();
    }
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        if let ReasoningLookup::Fresh(result) = worker.latest_for(3, identity) {
            if result.cycle == 3 {
                assert_eq!(result.payload, 30);
                break;
            }
        }
        assert!(
            Instant::now() < deadline,
            "latest snapshot was not processed"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(worker.stats().dropped >= 1);
}

#[test]
fn reasoning_results_accept_age_two_and_reject_age_three_or_other_identity() {
    let worker = LatestReasoningWorker::spawn("reflex-freshness", |value: u64| value + 1)
        .expect("worker starts");
    let identity = reasoning_identity(42, 100);
    worker.submit(ReasoningSnapshot::new(8, identity, 9));
    let deadline = Instant::now() + Duration::from_secs(1);
    while !matches!(worker.latest_for(8, identity), ReasoningLookup::Fresh(_)) {
        assert!(Instant::now() < deadline, "reasoning result not published");
        std::thread::sleep(Duration::from_millis(2));
    }

    assert!(matches!(
        worker.latest_for(10, identity),
        ReasoningLookup::Fresh(_)
    ));
    assert_eq!(
        worker.latest_for(11, identity),
        ReasoningLookup::Stale { age_cycles: 3 }
    );
    assert_eq!(
        worker.latest_for(9, reasoning_identity(42, 101)),
        ReasoningLookup::Pending
    );
}

#[test]
fn rejected_reasoning_results_are_discarded_after_one_observation() {
    let stale_worker = LatestReasoningWorker::spawn("reflex-stale-discard", |value: u64| value)
        .expect("worker starts");
    let identity = reasoning_identity(42, 100);
    stale_worker.submit(ReasoningSnapshot::new(1, identity, 9));
    let deadline = Instant::now() + Duration::from_secs(1);
    while !matches!(
        stale_worker.latest_for(1, identity),
        ReasoningLookup::Fresh(_)
    ) {
        assert!(Instant::now() < deadline, "reasoning result not published");
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(
        stale_worker.latest_for(4, identity),
        ReasoningLookup::Stale { age_cycles: 3 }
    );
    assert_eq!(
        stale_worker.latest_for(4, identity),
        ReasoningLookup::Pending
    );
    assert_eq!(stale_worker.stats().deadline_misses, 1);

    let mismatch_worker =
        LatestReasoningWorker::spawn("reflex-identity-discard", |value: u64| value)
            .expect("worker starts");
    mismatch_worker.submit(ReasoningSnapshot::new(8, identity, 9));
    let deadline = Instant::now() + Duration::from_secs(1);
    while mismatch_worker.stats().completed == 0 {
        assert!(Instant::now() < deadline, "reasoning result not published");
        std::thread::sleep(Duration::from_millis(2));
    }
    let other_identity = reasoning_identity(42, 101);
    assert_eq!(
        mismatch_worker.latest_for(8, other_identity),
        ReasoningLookup::IdentityMismatch
    );
    assert_eq!(
        mismatch_worker.latest_for(8, other_identity),
        ReasoningLookup::Pending
    );
    assert_eq!(mismatch_worker.stats().identity_mismatches, 1);
}

#[test]
fn admitted_and_applied_are_distinct_counters() {
    let mut broker = ReflexBroker::active_for_test();
    assert_eq!(broker.decide(&intent(1), safe()), ReflexDecision::Admit);
    assert_eq!(broker.counters().admitted, 1);
    assert_eq!(broker.counters().applied, 0);
    broker.record_applied(2);
    broker.record_reverted(1);
    broker.record_failed(1);
    assert_eq!(broker.counters().applied, 2);
    assert_eq!(broker.counters().reverted, 1);
    assert_eq!(broker.counters().failed, 1);
}
