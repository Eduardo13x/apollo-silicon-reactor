use std::collections::HashSet;

use apollo_engine::engine::decision_ledger::{
    ActuatorDecisionEvent, ActuatorDecisionOutcome, CycleDecisionEvents, DecisionLedger,
};
use apollo_engine::engine::exploration_scheduler::{
    ActionClass, CommitEvidence, CommitResult, ExplorationArm, ExplorationCandidate,
    ExplorationContext, ExplorationGateBlocker, ExplorationGates, ExplorationMode,
    ExplorationOrigin, ExplorationScheduler, ExplorationSchedulerPersisted, HardwareIdentity,
    ProbeCorrelation, RestoreContext, RestoreDisposition, TerminalDiagnostic, TimePoint,
    MAX_CANDIDATES_PER_CYCLE, MAX_COOLDOWNS, MAX_SERIALIZED_BYTES, MAX_TERMINAL_DEDUP,
};
use apollo_engine::engine::installation_identity::InstallationId;
use apollo_engine::engine::learned_state::LearnedState;
use apollo_engine::engine::telemetry_medallion::{ActuatorFamily, TelemetryMedallion};

fn local_origin() -> ExplorationOrigin {
    ExplorationOrigin {
        installation_id: 7,
        hardware: HardwareIdentity {
            p_core_count: 4,
            e_core_count: 4,
            ram_gib: 16,
        },
    }
}

fn now(wall: i64, monotonic: u64) -> TimePoint {
    TimePoint {
        wall_unix_secs: wall,
        monotonic_secs: monotonic,
        boot_id: 11,
    }
}

fn healthy() -> ExplorationGates {
    ExplorationGates::healthy()
}

fn candidate(family: ActuatorFamily, arm: ExplorationArm) -> ExplorationCandidate {
    let (mode, action) = match arm {
        ExplorationArm::NaturalObservation => (ExplorationMode::Natural, ActionClass::Natural),
        ExplorationArm::BoostOmission => (ExplorationMode::Control, ActionClass::BoostBackground),
        ExplorationArm::MarkovCacheOnly => {
            (ExplorationMode::Treatment, ActionClass::MarkovPredictedApp)
        }
        ExplorationArm::InteractionQosShort
        | ExplorationArm::InteractionQosStandard
        | ExplorationArm::InteractionQosLong => (
            ExplorationMode::Treatment,
            ActionClass::InteractionForeground,
        ),
    };
    ExplorationCandidate::new(
        family,
        mode,
        arm,
        action,
        ExplorationContext::General,
        local_origin(),
    )
    .expect("valid fixture")
}

fn scheduler() -> ExplorationScheduler {
    ExplorationScheduler::cold_start(local_origin())
}

#[test]
fn exhaustive_allowlist_is_exactly_three_of_twenty() {
    let allowed: HashSet<_> = ActuatorFamily::ALL
        .into_iter()
        .filter(|family| ExplorationScheduler::family_allowed(*family))
        .collect();
    assert_eq!(
        allowed,
        HashSet::from([
            ActuatorFamily::Boost,
            ActuatorFamily::InteractionQos,
            ActuatorFamily::MarkovPrewarm,
        ])
    );
    assert_eq!(ActuatorFamily::ALL.len(), 20);
}

#[test]
fn forged_family_action_and_raw_qos_aliases_fail_before_reservation() {
    let scheduler = scheduler();
    for (family, action, arm) in [
        (
            ActuatorFamily::ThreadQos,
            ActionClass::InteractionForeground,
            ExplorationArm::InteractionQosShort,
        ),
        (
            ActuatorFamily::IoShaping,
            ActionClass::InteractionForeground,
            ExplorationArm::InteractionQosShort,
        ),
        (
            ActuatorFamily::Boost,
            ActionClass::MarkovPredictedApp,
            ExplorationArm::BoostOmission,
        ),
    ] {
        assert!(ExplorationCandidate::new(
            family,
            ExplorationMode::Treatment,
            arm,
            action,
            ExplorationContext::General,
            local_origin(),
        )
        .is_err());
    }
    assert!(!scheduler.has_active_reservation());
    assert_eq!(scheduler.committed_count(), 0);
}

#[test]
fn gates_reject_independently_in_the_frozen_order() {
    let baseline = healthy();
    let cases: Vec<(ExplorationGateBlocker, Box<dyn Fn(&mut ExplorationGates)>)> = vec![
        (
            ExplorationGateBlocker::Lifecycle,
            Box::new(|g| g.daemon_shutdown = true),
        ),
        (
            ExplorationGateBlocker::Lifecycle,
            Box::new(|g| g.kill_switch = true),
        ),
        (
            ExplorationGateBlocker::Lifecycle,
            Box::new(|g| g.cognitive_paused = true),
        ),
        (
            ExplorationGateBlocker::Media,
            Box::new(|g| g.audio_output_active = true),
        ),
        (
            ExplorationGateBlocker::Media,
            Box::new(|g| g.audio_input_active = true),
        ),
        (
            ExplorationGateBlocker::Media,
            Box::new(|g| g.call_active = true),
        ),
        (
            ExplorationGateBlocker::Media,
            Box::new(|g| g.sleep_assertion = true),
        ),
        (
            ExplorationGateBlocker::Media,
            Box::new(|g| g.media_available = false),
        ),
        (
            ExplorationGateBlocker::UserInteraction,
            Box::new(|g| g.app_launching = true),
        ),
        (
            ExplorationGateBlocker::UserInteraction,
            Box::new(|g| g.window_operation = true),
        ),
        (
            ExplorationGateBlocker::Fluidity,
            Box::new(|g| g.fluidity_degraded = true),
        ),
        (
            ExplorationGateBlocker::Fluidity,
            Box::new(|g| g.predicted_fluidity_degraded = true),
        ),
        (
            ExplorationGateBlocker::Pressure,
            Box::new(|g| g.memory_pressure = 0.55),
        ),
        (
            ExplorationGateBlocker::Thermal,
            Box::new(|g| g.thermal_available = false),
        ),
        (
            ExplorationGateBlocker::Thermal,
            Box::new(|g| g.thermal_nominal = false),
        ),
        (
            ExplorationGateBlocker::Hazard,
            Box::new(|g| g.hazard_available = false),
        ),
        (
            ExplorationGateBlocker::Hazard,
            Box::new(|g| g.p_oom_30s = 0.30),
        ),
        (
            ExplorationGateBlocker::Circuit,
            Box::new(|g| g.circuit_closed = false),
        ),
        (
            ExplorationGateBlocker::Circuit,
            Box::new(|g| g.speculation_allowed = false),
        ),
        (
            ExplorationGateBlocker::Build,
            Box::new(|g| g.build_workload = true),
        ),
        (
            ExplorationGateBlocker::Build,
            Box::new(|g| g.build_phase_idle = false),
        ),
        (
            ExplorationGateBlocker::Build,
            Box::new(|g| g.compiler_protection_active = true),
        ),
        (
            ExplorationGateBlocker::Identity,
            Box::new(|g| g.identity_present = false),
        ),
        (
            ExplorationGateBlocker::Identity,
            Box::new(|g| g.identity_start_nonzero = false),
        ),
        (
            ExplorationGateBlocker::Identity,
            Box::new(|g| g.identity_stale = true),
        ),
        (
            ExplorationGateBlocker::Identity,
            Box::new(|g| g.identity_recycled = true),
        ),
        (
            ExplorationGateBlocker::Identity,
            Box::new(|g| g.target_protected = true),
        ),
        (
            ExplorationGateBlocker::Identity,
            Box::new(|g| g.target_apple_owned = true),
        ),
        (
            ExplorationGateBlocker::Identity,
            Box::new(|g| g.identity_recheck_ok = false),
        ),
        (
            ExplorationGateBlocker::Ownership,
            Box::new(|g| g.markov_quarantined = true),
        ),
        (
            ExplorationGateBlocker::Ownership,
            Box::new(|g| g.effect_owner_conflict = true),
        ),
    ];
    for (expected, mutate) in cases {
        let mut gates = baseline;
        mutate(&mut gates);
        let mut scheduler = scheduler();
        let rejection = scheduler
            .request(
                &candidate(
                    ActuatorFamily::MarkovPrewarm,
                    ExplorationArm::MarkovCacheOnly,
                ),
                &gates,
                now(1_000, 1_000),
            )
            .unwrap_err();
        assert_eq!(rejection, expected);
        assert!(!scheduler.has_active_reservation());
    }

    let mut all_bad = baseline;
    all_bad.cognitive_paused = true;
    all_bad.audio_output_active = true;
    all_bad.memory_pressure = f64::NAN;
    assert_eq!(
        scheduler()
            .request(
                &candidate(
                    ActuatorFamily::MarkovPrewarm,
                    ExplorationArm::MarkovCacheOnly
                ),
                &all_bad,
                now(1_000, 1_000)
            )
            .unwrap_err(),
        ExplorationGateBlocker::Lifecycle
    );
}

#[test]
fn exact_pressure_and_hazard_boundaries_fail_closed() {
    for pressure in [f64::NAN, f64::INFINITY, 0.55, 1.0] {
        let mut gates = healthy();
        gates.memory_pressure = pressure;
        assert_eq!(
            scheduler()
                .request(
                    &candidate(
                        ActuatorFamily::MarkovPrewarm,
                        ExplorationArm::MarkovCacheOnly
                    ),
                    &gates,
                    now(1_000, 1_000)
                )
                .unwrap_err(),
            ExplorationGateBlocker::Pressure
        );
    }
    for hazard in [f64::NAN, f64::INFINITY, 0.30, 1.0] {
        let mut gates = healthy();
        gates.p_oom_30s = hazard;
        assert_eq!(
            scheduler()
                .request(
                    &candidate(
                        ActuatorFamily::MarkovPrewarm,
                        ExplorationArm::MarkovCacheOnly
                    ),
                    &gates,
                    now(1_000, 1_000)
                )
                .unwrap_err(),
            ExplorationGateBlocker::Hazard
        );
    }
    let mut gates = healthy();
    gates.memory_pressure = 0.55 - f64::EPSILON;
    gates.p_oom_30s = 0.30 - f64::EPSILON;
    assert!(scheduler()
        .request(
            &candidate(
                ActuatorFamily::MarkovPrewarm,
                ExplorationArm::MarkovCacheOnly
            ),
            &gates,
            now(1_000, 1_000)
        )
        .is_ok());
}

fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    if items.is_empty() {
        return vec![Vec::new()];
    }
    let mut output = Vec::new();
    for index in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(index);
        for mut tail in permutations(&rest) {
            tail.insert(0, head.clone());
            output.push(tail);
        }
    }
    output
}

#[test]
fn selection_is_deterministic_across_every_candidate_permutation() {
    let candidates = vec![
        candidate(ActuatorFamily::Boost, ExplorationArm::BoostOmission),
        candidate(
            ActuatorFamily::InteractionQos,
            ExplorationArm::InteractionQosLong,
        ),
        candidate(
            ActuatorFamily::InteractionQos,
            ExplorationArm::InteractionQosShort,
        ),
        candidate(
            ActuatorFamily::MarkovPrewarm,
            ExplorationArm::MarkovCacheOnly,
        ),
        candidate(ActuatorFamily::Boost, ExplorationArm::NaturalObservation),
    ];
    for permutation in permutations(&candidates) {
        let selected = scheduler()
            .select(&permutation, &healthy(), now(1_000, 1_000))
            .expect("selection");
        assert_eq!(selected.metadata.arm, ExplorationArm::NaturalObservation);
        assert_eq!(selected.metadata.family, ActuatorFamily::Boost);
    }
}

#[test]
fn candidate_caps_and_single_reservation_are_enforced() {
    let one = candidate(
        ActuatorFamily::InteractionQos,
        ExplorationArm::InteractionQosShort,
    );
    let too_many = vec![one.clone(); MAX_CANDIDATES_PER_CYCLE + 1];
    assert_eq!(
        scheduler()
            .select(&too_many, &healthy(), now(1_000, 1_000))
            .unwrap_err(),
        ExplorationGateBlocker::Capacity
    );
    let five_same_family = vec![one.clone(); 5];
    assert_eq!(
        scheduler()
            .select(&five_same_family, &healthy(), now(1_000, 1_000))
            .unwrap_err(),
        ExplorationGateBlocker::Capacity
    );

    let mut scheduler = scheduler();
    scheduler
        .request(&one, &healthy(), now(1_000, 1_000))
        .unwrap();
    assert_eq!(
        scheduler
            .request(&one, &healthy(), now(1_000, 1_000))
            .unwrap_err(),
        ExplorationGateBlocker::Ownership
    );
}

#[test]
fn natural_observation_is_preferred_and_consumes_no_budget_or_cooldown() {
    let mut scheduler = scheduler();
    let approval = scheduler
        .request(
            &candidate(ActuatorFamily::Boost, ExplorationArm::NaturalObservation),
            &healthy(),
            now(1_000, 1_000),
        )
        .unwrap();
    assert!(!approval.metadata.treatment);
    assert!(!scheduler.has_active_reservation());
    assert_eq!(scheduler.committed_count(), 0);
    assert_eq!(scheduler.cooldown_count(), 0);
}

#[test]
fn markov_and_qos_arms_are_closed_and_bounded() {
    assert!(!ExplorationArm::MarkovCacheOnly.allows_kernel_acceleration());
    for arm in [
        ExplorationArm::InteractionQosShort,
        ExplorationArm::InteractionQosStandard,
        ExplorationArm::InteractionQosLong,
    ] {
        assert!(arm.ttl_millis().unwrap() <= 12_000);
        assert!(candidate(ActuatorFamily::InteractionQos, arm).is_mutable());
    }
}

#[test]
fn boost_omission_requires_the_complete_background_capability() {
    let mut gates = healthy();
    gates.target_foreground = true;
    assert_eq!(
        scheduler()
            .request(
                &candidate(ActuatorFamily::Boost, ExplorationArm::BoostOmission),
                &gates,
                now(1_000, 1_000)
            )
            .unwrap_err(),
        ExplorationGateBlocker::Ownership
    );
    for mutate in [
        |g: &mut ExplorationGates| g.target_launching = true,
        |g: &mut ExplorationGates| g.interactive_lease_active = true,
        |g: &mut ExplorationGates| g.coalition_conflict = true,
        |g: &mut ExplorationGates| g.recovery_required = true,
    ] {
        let mut gates = healthy();
        mutate(&mut gates);
        assert!(scheduler()
            .request(
                &candidate(ActuatorFamily::Boost, ExplorationArm::BoostOmission),
                &gates,
                now(1_000, 1_000)
            )
            .is_err());
    }
}

#[test]
fn global_899_900_and_key_86399_86400_boundaries_are_exact() {
    let probe = candidate(
        ActuatorFamily::MarkovPrewarm,
        ExplorationArm::MarkovCacheOnly,
    );
    let mut scheduler = scheduler();
    let first = scheduler
        .request(&probe, &healthy(), now(1_000, 1_000))
        .unwrap();
    assert_eq!(
        scheduler.commit(
            first.metadata.correlation,
            now(1_000, 1_000),
            CommitEvidence::MutationApplied
        ),
        CommitResult::Committed
    );
    assert_eq!(
        scheduler
            .request(
                &candidate(
                    ActuatorFamily::InteractionQos,
                    ExplorationArm::InteractionQosShort
                ),
                &healthy(),
                now(1_899, 1_899)
            )
            .unwrap_err(),
        ExplorationGateBlocker::GlobalBudget
    );
    let second = scheduler
        .request(
            &candidate(
                ActuatorFamily::InteractionQos,
                ExplorationArm::InteractionQosShort,
            ),
            &healthy(),
            now(1_900, 1_900),
        )
        .unwrap();
    assert_eq!(
        scheduler.commit(
            second.metadata.correlation,
            now(1_900, 1_900),
            CommitEvidence::MutationApplied
        ),
        CommitResult::Committed
    );
    assert_eq!(
        scheduler
            .request(&probe, &healthy(), now(87_399, 87_399))
            .unwrap_err(),
        ExplorationGateBlocker::KeyCooldown
    );
    assert!(scheduler
        .request(&probe, &healthy(), now(87_400, 87_400))
        .is_ok());
}

#[test]
fn wall_rollback_and_forward_jump_cannot_bypass_same_boot_monotonic_time() {
    let probe = candidate(
        ActuatorFamily::MarkovPrewarm,
        ExplorationArm::MarkovCacheOnly,
    );
    let mut scheduler = scheduler();
    let first = scheduler
        .request(&probe, &healthy(), now(10_000, 10_000))
        .unwrap();
    scheduler.commit(
        first.metadata.correlation,
        now(10_000, 10_000),
        CommitEvidence::MutationApplied,
    );
    assert_eq!(
        scheduler
            .request(
                &candidate(
                    ActuatorFamily::InteractionQos,
                    ExplorationArm::InteractionQosShort
                ),
                &healthy(),
                now(9_999, 11_000)
            )
            .unwrap_err(),
        ExplorationGateBlocker::GlobalBudget
    );
    assert_eq!(
        scheduler
            .request(
                &candidate(
                    ActuatorFamily::InteractionQos,
                    ExplorationArm::InteractionQosShort
                ),
                &healthy(),
                now(99_999, 10_899)
            )
            .unwrap_err(),
        ExplorationGateBlocker::GlobalBudget
    );
}

#[test]
fn no_op_failure_and_dry_run_clear_without_committing() {
    for evidence in [
        CommitEvidence::NoOp,
        CommitEvidence::Failed,
        CommitEvidence::DryRun,
    ] {
        let mut scheduler = scheduler();
        let approval = scheduler
            .request(
                &candidate(
                    ActuatorFamily::InteractionQos,
                    ExplorationArm::InteractionQosShort,
                ),
                &healthy(),
                now(1_000, 1_000),
            )
            .unwrap();
        assert_eq!(
            scheduler.commit(approval.metadata.correlation, now(1_000, 1_000), evidence),
            CommitResult::ReleasedWithoutCommit
        );
        assert_eq!(scheduler.committed_count(), 0);
        assert_eq!(scheduler.cooldown_count(), 0);
        assert!(!scheduler.has_active_reservation());
    }
}

#[test]
fn omission_commits_only_after_endpoint_opened() {
    let mut scheduler = scheduler();
    let approval = scheduler
        .request(
            &candidate(ActuatorFamily::Boost, ExplorationArm::BoostOmission),
            &healthy(),
            now(1_000, 1_000),
        )
        .unwrap();
    assert_eq!(
        scheduler.commit(
            approval.metadata.correlation,
            now(1_000, 1_000),
            CommitEvidence::MutationApplied
        ),
        CommitResult::ReleasedWithoutCommit
    );
    let approval = scheduler
        .request(
            &candidate(ActuatorFamily::Boost, ExplorationArm::BoostOmission),
            &healthy(),
            now(1_000, 1_000),
        )
        .unwrap();
    assert_eq!(
        scheduler.commit(
            approval.metadata.correlation,
            now(1_000, 1_000),
            CommitEvidence::OmissionEndpointOpened
        ),
        CommitResult::Committed
    );
}

#[test]
fn cancellation_and_terminal_callbacks_are_bounded_and_idempotent() {
    let mut scheduler = scheduler();
    let approval = scheduler
        .request(
            &candidate(
                ActuatorFamily::InteractionQos,
                ExplorationArm::InteractionQosLong,
            ),
            &healthy(),
            now(1_000, 1_000),
        )
        .unwrap();
    assert!(scheduler.cancel(
        approval.metadata.correlation,
        TerminalDiagnostic::KillSwitch
    ));
    assert!(!scheduler.cancel(approval.metadata.correlation, TerminalDiagnostic::Shutdown));
    for value in 1..=(MAX_TERMINAL_DEDUP as u64 + 30) {
        scheduler.record_terminal(ProbeCorrelation(value), TerminalDiagnostic::Expired);
    }
    assert_eq!(scheduler.terminal_dedup_count(), MAX_TERMINAL_DEDUP);
}

#[test]
fn matching_restart_restores_but_origin_hardware_and_imported_state_reset() {
    let probe = candidate(
        ActuatorFamily::MarkovPrewarm,
        ExplorationArm::MarkovCacheOnly,
    );
    let mut scheduler = scheduler();
    let approval = scheduler
        .request(&probe, &healthy(), now(1_000, 1_000))
        .unwrap();
    scheduler.commit(
        approval.metadata.correlation,
        now(1_000, 1_000),
        CommitEvidence::MutationApplied,
    );
    let persisted = scheduler.persisted();

    let (restored, disposition) = ExplorationScheduler::restore(
        persisted.clone(),
        RestoreContext::local(local_origin(), now(1_100, 5)),
    );
    assert_eq!(disposition, RestoreDisposition::Restored);
    assert_eq!(restored.committed_count(), 1);
    assert!(!restored.has_active_reservation());

    let changed_hardware = ExplorationOrigin {
        hardware: HardwareIdentity {
            ram_gib: 32,
            ..local_origin().hardware
        },
        ..local_origin()
    };
    for context in [
        RestoreContext::local(changed_hardware, now(1_100, 5)),
        RestoreContext::unknown(local_origin(), now(1_100, 5)),
        RestoreContext::imported_m1(local_origin(), now(1_100, 5)),
    ] {
        let (reset, disposition) = ExplorationScheduler::restore(persisted.clone(), context);
        assert_eq!(disposition, RestoreDisposition::ResetOrigin);
        assert_eq!(reset.committed_count(), 0);
        assert_eq!(reset.cooldown_count(), 0);
    }
}

#[test]
fn hostile_future_over_cap_and_oversize_state_fail_closed() {
    let mut persisted = ExplorationSchedulerPersisted::default_for(local_origin());
    persisted.schema_version += 1;
    assert_eq!(
        ExplorationScheduler::restore(
            persisted,
            RestoreContext::local(local_origin(), now(1_000, 1))
        )
        .1,
        RestoreDisposition::ResetHostile
    );

    let mut persisted = ExplorationSchedulerPersisted::default_for(local_origin());
    persisted.inject_hostile_cooldowns_for_test(MAX_COOLDOWNS + 1);
    assert_eq!(
        ExplorationScheduler::restore(
            persisted,
            RestoreContext::local(local_origin(), now(1_000, 1))
        )
        .1,
        RestoreDisposition::ResetHostile
    );

    let persisted =
        ExplorationSchedulerPersisted::oversized_for_test(local_origin(), MAX_SERIALIZED_BYTES + 1);
    assert_eq!(
        ExplorationScheduler::restore(
            persisted,
            RestoreContext::local(local_origin(), now(1_000, 1))
        )
        .1,
        RestoreDisposition::ResetHostile
    );
}

#[test]
fn full_live_cooldown_map_rejects_new_key_but_expired_entries_are_pruned() {
    let mut persisted = ExplorationSchedulerPersisted::default_for(local_origin());
    persisted.inject_live_cooldowns_for_test(MAX_COOLDOWNS, 100_000);
    let (mut scheduler, _) = ExplorationScheduler::restore(
        persisted,
        RestoreContext::local(local_origin(), now(1_000, 1)),
    );
    assert_eq!(
        scheduler
            .request(
                &candidate(ActuatorFamily::Boost, ExplorationArm::BoostOmission),
                &healthy(),
                now(1_000, 1)
            )
            .unwrap_err(),
        ExplorationGateBlocker::Capacity
    );

    let mut persisted = ExplorationSchedulerPersisted::default_for(local_origin());
    persisted.inject_live_cooldowns_for_test(MAX_COOLDOWNS, 999);
    let (mut scheduler, _) = ExplorationScheduler::restore(
        persisted,
        RestoreContext::local(local_origin(), now(1_000, 1)),
    );
    assert!(scheduler
        .request(
            &candidate(ActuatorFamily::Boost, ExplorationArm::BoostOmission),
            &healthy(),
            now(1_000, 1)
        )
        .is_ok());
    assert!(scheduler.cooldown_count() < MAX_COOLDOWNS);
}

#[test]
fn one_probe_correlation_maps_to_one_normal_decision_id_and_episode() {
    let mut scheduler = scheduler();
    let approval = scheduler
        .request(
            &candidate(
                ActuatorFamily::InteractionQos,
                ExplorationArm::InteractionQosStandard,
            ),
            &healthy(),
            now(1_000, 1_000),
        )
        .unwrap();
    let metadata = scheduler
        .commit_metadata(
            approval.metadata.correlation,
            now(1_000, 1_000),
            CommitEvidence::MutationApplied,
        )
        .expect("committed metadata");
    let mut events = CycleDecisionEvents::default();
    events.push(
        ActuatorDecisionEvent::local(
            "interaction_qos:foreground@standard",
            "general",
            1,
            ActuatorDecisionOutcome::Pending,
            "acceleration-lease",
            "lease started",
        )
        .with_correlation(metadata.correlation.0)
        .with_exploration(metadata.clone()),
    );
    let mut ledger = DecisionLedger::new();
    assert!(ledger.ingest_cycle_events(&mut events).is_empty());
    events.push(
        ActuatorDecisionEvent::local(
            "interaction_qos:foreground@standard",
            "general",
            2,
            ActuatorDecisionOutcome::Reverted,
            "acceleration-lease",
            "lease released",
        )
        .with_correlation(metadata.correlation.0)
        .with_exploration(metadata),
    );
    let episodes = ledger.ingest_cycle_events(&mut events);
    assert_eq!(episodes.len(), 1);
    assert_ne!(episodes[0].id.0, 0);
    assert_eq!(
        episodes[0]
            .envelope
            .exploration
            .as_ref()
            .unwrap()
            .correlation,
        approval.metadata.correlation
    );
    let mut medallion = TelemetryMedallion::new(InstallationId(7));
    medallion.stage_decision_episodes(&episodes);
    assert_eq!(medallion.decision_id_high_water(), episodes[0].id.0);
}

#[test]
fn controls_cancelled_and_non_applied_experiments_never_gain_authority() {
    for (arm, outcome) in [
        (
            ExplorationArm::BoostOmission,
            ActuatorDecisionOutcome::Rejected,
        ),
        (
            ExplorationArm::MarkovCacheOnly,
            ActuatorDecisionOutcome::NoOp,
        ),
    ] {
        let family = if arm == ExplorationArm::BoostOmission {
            ActuatorFamily::Boost
        } else {
            ActuatorFamily::MarkovPrewarm
        };
        let mut scheduler = scheduler();
        let approval = scheduler
            .request(&candidate(family, arm), &healthy(), now(1_000, 1_000))
            .unwrap();
        let mut metadata = approval.metadata;
        metadata.cancelled = Some(TerminalDiagnostic::Wake);
        let mut events = CycleDecisionEvents::default();
        events.push(
            ActuatorDecisionEvent::local("task5:test", "bounded", 1, outcome, "test", "terminal")
                .with_exploration(metadata),
        );
        let episode = DecisionLedger::new()
            .ingest_cycle_events(&mut events)
            .remove(0);
        assert!(!episode.authority_eligible);
    }
}

#[test]
fn learned_state_has_one_serde_defaulted_scheduler_field() {
    let mut legacy: LearnedState = serde_json::from_value(serde_json::json!({})).unwrap();
    assert!(legacy.exploration_scheduler.is_none());
    legacy.exploration_scheduler = Some(scheduler().persisted());
    let serialized = serde_json::to_value(legacy).unwrap();
    assert_eq!(
        serialized
            .as_object()
            .unwrap()
            .keys()
            .filter(|key| key.contains("exploration_scheduler"))
            .count(),
        1
    );
    let restored: LearnedState = serde_json::from_value(serialized).unwrap();
    assert!(restored.exploration_scheduler.is_some());
}

#[test]
fn saturated_selection_reports_at_most_twelve_plus_two_hundred_fifty_six_work() {
    let mut persisted = ExplorationSchedulerPersisted::default_for(local_origin());
    persisted.inject_live_cooldowns_for_test(MAX_COOLDOWNS, 100_000);
    let (mut scheduler, _) = ExplorationScheduler::restore(
        persisted,
        RestoreContext::local(local_origin(), now(1_000, 1)),
    );
    let candidates = vec![
        candidate(
            ActuatorFamily::InteractionQos,
            ExplorationArm::InteractionQosShort,
        ),
        candidate(
            ActuatorFamily::InteractionQos,
            ExplorationArm::InteractionQosStandard,
        ),
        candidate(
            ActuatorFamily::InteractionQos,
            ExplorationArm::InteractionQosLong,
        ),
        candidate(
            ActuatorFamily::InteractionQos,
            ExplorationArm::InteractionQosShort,
        ),
        candidate(
            ActuatorFamily::MarkovPrewarm,
            ExplorationArm::MarkovCacheOnly,
        ),
        candidate(
            ActuatorFamily::MarkovPrewarm,
            ExplorationArm::MarkovCacheOnly,
        ),
        candidate(
            ActuatorFamily::MarkovPrewarm,
            ExplorationArm::MarkovCacheOnly,
        ),
        candidate(
            ActuatorFamily::MarkovPrewarm,
            ExplorationArm::MarkovCacheOnly,
        ),
        candidate(ActuatorFamily::Boost, ExplorationArm::BoostOmission),
        candidate(ActuatorFamily::Boost, ExplorationArm::BoostOmission),
        candidate(ActuatorFamily::Boost, ExplorationArm::BoostOmission),
        candidate(ActuatorFamily::Boost, ExplorationArm::BoostOmission),
    ];
    assert_eq!(
        scheduler
            .select(&candidates, &healthy(), now(1_000, 1))
            .unwrap_err(),
        ExplorationGateBlocker::Capacity
    );
    let work = scheduler.last_work();
    assert_eq!(work.candidates_examined, MAX_CANDIDATES_PER_CYCLE);
    assert_eq!(work.cooldowns_examined, MAX_COOLDOWNS);
    let source = include_str!("../src/engine/exploration_scheduler.rs");
    assert!(!source.contains(".sort("));
    assert!(!source.contains("execute_actions"));
    assert!(!source.contains("std::process::Command"));
}
