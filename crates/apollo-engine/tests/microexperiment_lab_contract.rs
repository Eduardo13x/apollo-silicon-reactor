use apollo_engine::engine::exploration_scheduler::{
    ActionClass, ExplorationArm, ExplorationContext, ExplorationOrigin, HardwareIdentity,
};
use apollo_engine::engine::microexperiment_lab::{
    ArmKind, EvidenceClosure, ExecutionClosure, HorizonClosure, LabError, MicroexperimentLab,
    PairCandidate, PairEndpoint, PairGates, PairOrder, PairProgress, RestoreDisposition,
    RollbackClosure, MAX_COMPLETED_PAIRS, MAX_GOLD_DEDUP, MAX_OPEN_PAIRS, MAX_SERIALIZED_BYTES,
};
use apollo_engine::engine::telemetry_medallion::ActuatorFamily;

fn origin() -> ExplorationOrigin {
    ExplorationOrigin {
        installation_id: 0xA110,
        hardware: HardwareIdentity {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        },
    }
}

fn qos_candidate(sequence: u64) -> PairCandidate {
    PairCandidate {
        sequence,
        origin: origin(),
        family: ActuatorFamily::InteractionQos,
        action_class: ActionClass::InteractionForeground,
        treatment_arm: ExplorationArm::InteractionQosStandard,
        context: ExplorationContext::Interactive,
        action_key: "interaction_qos:foreground@standard".to_string(),
        stratum_hash: 0x51,
        horizon_cycles: 12,
        washout_cycles: 3,
        minimum_effect_micros: 500,
    }
}

fn endpoint(candidate: &PairCandidate, arm: ArmKind, utility_micros: i64) -> PairEndpoint {
    PairEndpoint {
        arm,
        origin: candidate.origin,
        family: candidate.family,
        action_class: candidate.action_class,
        context: candidate.context,
        action_key: candidate.action_key.clone(),
        stratum_hash: candidate.stratum_hash,
        horizon_cycles: candidate.horizon_cycles,
        decision_id: match arm {
            ArmKind::Control => 101,
            ArmKind::Treatment => 202,
        },
        observed_local: true,
        synthetic: false,
        execution: match arm {
            ArmKind::Control => ExecutionClosure::NoOp,
            ArmKind::Treatment => ExecutionClosure::Applied,
        },
        horizon: HorizonClosure::Complete,
        rollback: RollbackClosure::Succeeded,
        utility_micros,
    }
}

fn record_in_assignment_order(
    lab: &mut MicroexperimentLab,
    candidate: &PairCandidate,
    assignment: apollo_engine::engine::microexperiment_lab::PairAssignment,
    control_utility: i64,
    treatment_utility: i64,
) {
    let first_utility = match assignment.first {
        ArmKind::Control => control_utility,
        ArmKind::Treatment => treatment_utility,
    };
    assert_eq!(
        lab.record_endpoint(
            assignment.id,
            endpoint(candidate, assignment.first, first_utility)
        )
        .unwrap(),
        PairProgress::Washout
    );
    assert!(matches!(
        lab.record_endpoint(
            assignment.id,
            endpoint(candidate, assignment.second, treatment_utility)
        ),
        Err(LabError::WashoutPending)
    ));
    assert_eq!(
        lab.advance_washout(assignment.id, candidate.washout_cycles)
            .unwrap(),
        PairProgress::AwaitingComplement
    );
    let second_utility = match assignment.second {
        ArmKind::Control => control_utility,
        ArmKind::Treatment => treatment_utility,
    };
    assert_eq!(
        lab.record_endpoint(
            assignment.id,
            endpoint(candidate, assignment.second, second_utility)
        )
        .unwrap(),
        PairProgress::ReadyToClose
    );
}

#[test]
fn exact_closed_catalog_is_admitted() {
    let mut lab = MicroexperimentLab::cold_start(origin());
    let gates = PairGates::healthy_enabled();
    assert!(lab.propose(qos_candidate(1), gates).is_ok());

    let mut markov = qos_candidate(2);
    markov.family = ActuatorFamily::MarkovPrewarm;
    markov.action_class = ActionClass::MarkovPredictedApp;
    markov.treatment_arm = ExplorationArm::MarkovCacheOnly;
    markov.context = ExplorationContext::Background;
    markov.action_key = "markov:cache-only".to_string();
    assert!(lab.propose(markov, gates).is_ok());

    let mut boost = qos_candidate(3);
    boost.family = ActuatorFamily::Boost;
    boost.action_class = ActionClass::BoostBackground;
    boost.treatment_arm = ExplorationArm::BoostOmission;
    boost.context = ExplorationContext::Background;
    boost.action_key = "boost:background".to_string();
    assert!(lab.propose(boost, gates).is_ok());
}

#[test]
fn unknown_family_or_arm_combination_is_rejected() {
    let mut lab = MicroexperimentLab::cold_start(origin());
    let mut candidate = qos_candidate(1);
    candidate.treatment_arm = ExplorationArm::MarkovCacheOnly;
    assert_eq!(
        lab.propose(candidate, PairGates::healthy_enabled()),
        Err(LabError::Catalog)
    );
}

#[test]
fn assignment_is_balanced_and_independent_of_model_score() {
    let mut lab = MicroexperimentLab::cold_start(origin());
    let first = lab
        .propose(qos_candidate(10), PairGates::healthy_enabled())
        .unwrap();
    let second = lab
        .propose(qos_candidate(11), PairGates::healthy_enabled())
        .unwrap();
    assert_eq!(first.order, PairOrder::ControlThenTreatment);
    assert_eq!(second.order, PairOrder::TreatmentThenControl);
}

#[test]
fn pair_requires_exactly_one_control_and_one_treatment() {
    let candidate = qos_candidate(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    let assignment = lab
        .propose(candidate.clone(), PairGates::healthy_enabled())
        .unwrap();
    let first = endpoint(&candidate, assignment.first, 1_000);
    assert_eq!(
        lab.record_endpoint(assignment.id, first.clone()).unwrap(),
        PairProgress::Washout
    );
    assert_eq!(
        lab.record_endpoint(assignment.id, first),
        Err(LabError::DuplicateArm)
    );
}

#[test]
fn complement_is_locked_until_washout_completes() {
    let candidate = qos_candidate(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    let assignment = lab
        .propose(candidate.clone(), PairGates::healthy_enabled())
        .unwrap();
    lab.record_endpoint(assignment.id, endpoint(&candidate, assignment.first, 1_000))
        .unwrap();
    assert_eq!(
        lab.advance_washout(assignment.id, candidate.washout_cycles - 1),
        Ok(PairProgress::Washout)
    );
    assert_eq!(
        lab.record_endpoint(
            assignment.id,
            endpoint(&candidate, assignment.second, 2_000)
        ),
        Err(LabError::WashoutPending)
    );
}

#[test]
fn safety_and_privacy_are_fail_closed() {
    let blockers = [
        PairGates::default(),
        PairGates {
            secure_input: true,
            ..PairGates::healthy_enabled()
        },
        PairGates {
            screen_capture: true,
            ..PairGates::healthy_enabled()
        },
        PairGates {
            camera_active: true,
            ..PairGates::healthy_enabled()
        },
        PairGates {
            sensitive_context: true,
            ..PairGates::healthy_enabled()
        },
        PairGates {
            inherited_safe: false,
            ..PairGates::healthy_enabled()
        },
    ];
    for gates in blockers {
        let mut lab = MicroexperimentLab::cold_start(origin());
        assert_eq!(lab.propose(qos_candidate(1), gates), Err(LabError::Gate));
    }
}

#[test]
fn origin_and_identity_mismatch_are_rejected() {
    let candidate = qos_candidate(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    let assignment = lab
        .propose(candidate.clone(), PairGates::healthy_enabled())
        .unwrap();
    let mut foreign = endpoint(&candidate, assignment.first, 1_000);
    foreign.origin.installation_id += 1;
    assert_eq!(
        lab.record_endpoint(assignment.id, foreign),
        Err(LabError::Origin)
    );
}

#[test]
fn endpoint_must_match_family_action_context_stratum_and_horizon() {
    let candidate = qos_candidate(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    let assignment = lab
        .propose(candidate.clone(), PairGates::healthy_enabled())
        .unwrap();
    let mut mismatched = endpoint(&candidate, assignment.first, 1_000);
    mismatched.stratum_hash += 1;
    assert_eq!(
        lab.record_endpoint(assignment.id, mismatched),
        Err(LabError::Mismatch)
    );
}

#[test]
fn synthetic_or_unattributed_endpoint_never_reaches_pair_gold() {
    for mutate in [0_u8, 1] {
        let candidate = qos_candidate(u64::from(mutate) + 1);
        let mut lab = MicroexperimentLab::cold_start(origin());
        let assignment = lab
            .propose(candidate.clone(), PairGates::healthy_enabled())
            .unwrap();
        let mut first = endpoint(&candidate, assignment.first, 1_000);
        if mutate == 0 {
            first.synthetic = true;
        } else {
            first.observed_local = false;
        }
        lab.record_endpoint(assignment.id, first).unwrap();
        lab.advance_washout(assignment.id, candidate.washout_cycles)
            .unwrap();
        lab.record_endpoint(
            assignment.id,
            endpoint(&candidate, assignment.second, 2_000),
        )
        .unwrap();
        let closure = lab.close_pair(assignment.id).unwrap();
        assert_eq!(closure.evidence, EvidenceClosure::Silver);
        assert!(lab.drain_pair_gold().is_empty());
    }
}

#[test]
fn incomplete_confounded_or_failed_rollback_never_reaches_pair_gold() {
    for case in 0..3_u8 {
        let candidate = qos_candidate(u64::from(case) + 1);
        let mut lab = MicroexperimentLab::cold_start(origin());
        let assignment = lab
            .propose(candidate.clone(), PairGates::healthy_enabled())
            .unwrap();
        let mut first = endpoint(&candidate, assignment.first, 1_000);
        match case {
            0 => first.horizon = HorizonClosure::Incomplete,
            1 => first.horizon = HorizonClosure::Confounded,
            _ => first.rollback = RollbackClosure::Failed,
        }
        lab.record_endpoint(assignment.id, first).unwrap();
        lab.advance_washout(assignment.id, candidate.washout_cycles)
            .unwrap();
        lab.record_endpoint(
            assignment.id,
            endpoint(&candidate, assignment.second, 2_000),
        )
        .unwrap();
        assert_ne!(
            lab.close_pair(assignment.id).unwrap().evidence,
            EvidenceClosure::PairGold
        );
    }
}

#[test]
fn rollback_is_orthogonal_to_applied_execution() {
    let candidate = qos_candidate(1);
    let mut treatment = endpoint(&candidate, ArmKind::Treatment, 2_000);
    treatment.rollback = RollbackClosure::Succeeded;
    assert_eq!(treatment.execution, ExecutionClosure::Applied);
    assert_eq!(treatment.rollback, RollbackClosure::Succeeded);
}

#[test]
fn markov_non_kernel_closure_can_create_pair_gold() {
    let mut candidate = qos_candidate(1);
    candidate.family = ActuatorFamily::MarkovPrewarm;
    candidate.action_class = ActionClass::MarkovPredictedApp;
    candidate.treatment_arm = ExplorationArm::MarkovCacheOnly;
    candidate.context = ExplorationContext::Background;
    candidate.action_key = "markov:cache-only".to_string();
    let mut lab = MicroexperimentLab::cold_start(origin());
    let assignment = lab
        .propose(candidate.clone(), PairGates::healthy_enabled())
        .unwrap();
    let first_utility = if assignment.first == ArmKind::Treatment {
        3_000
    } else {
        1_000
    };
    let mut first = endpoint(&candidate, assignment.first, first_utility);
    first.rollback = RollbackClosure::NotRequiredNonKernel;
    lab.record_endpoint(assignment.id, first).unwrap();
    lab.advance_washout(assignment.id, candidate.washout_cycles)
        .unwrap();
    let second_utility = if assignment.second == ArmKind::Treatment {
        3_000
    } else {
        1_000
    };
    let mut second = endpoint(&candidate, assignment.second, second_utility);
    second.rollback = RollbackClosure::NotRequiredNonKernel;
    lab.record_endpoint(assignment.id, second).unwrap();
    assert_eq!(
        lab.close_pair(assignment.id).unwrap().evidence,
        EvidenceClosure::PairGold
    );
}

#[test]
fn beneficial_null_and_harmful_results_are_gold_but_only_beneficial_is_effective() {
    for (sequence, treatment, effective, harmful) in [
        (1, 2_000, true, false),
        (2, 1_200, false, false),
        (3, 0, false, true),
    ] {
        let candidate = qos_candidate(sequence);
        let mut lab = MicroexperimentLab::cold_start(origin());
        let assignment = lab
            .propose(candidate.clone(), PairGates::healthy_enabled())
            .unwrap();
        record_in_assignment_order(&mut lab, &candidate, assignment, 1_000, treatment);
        let closure = lab.close_pair(assignment.id).unwrap();
        assert_eq!(closure.evidence, EvidenceClosure::PairGold);
        assert_eq!(closure.effective, effective);
        assert_eq!(closure.harmful, harmful);
    }
}

#[test]
fn pair_gold_emits_exactly_once_even_after_duplicate_close() {
    let candidate = qos_candidate(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    let assignment = lab
        .propose(candidate.clone(), PairGates::healthy_enabled())
        .unwrap();
    record_in_assignment_order(&mut lab, &candidate, assignment, 1_000, 2_000);
    lab.close_pair(assignment.id).unwrap();
    assert_eq!(lab.drain_pair_gold().len(), 1);
    assert_eq!(lab.close_pair(assignment.id), Err(LabError::DuplicatePair));
    assert!(lab.drain_pair_gold().is_empty());
}

#[test]
fn open_completed_and_dedup_state_are_bounded() {
    let mut lab = MicroexperimentLab::cold_start(origin());
    for sequence in 0..MAX_OPEN_PAIRS as u64 {
        assert!(lab
            .propose(qos_candidate(sequence + 1), PairGates::healthy_enabled())
            .is_ok());
    }
    assert_eq!(
        lab.propose(qos_candidate(99_999), PairGates::healthy_enabled()),
        Err(LabError::Capacity)
    );
    assert!(MAX_COMPLETED_PAIRS >= MAX_OPEN_PAIRS);
    assert!(MAX_GOLD_DEDUP >= MAX_OPEN_PAIRS);
}

#[test]
fn action_and_stratum_bounds_reject_oversized_or_unknown_data() {
    let mut lab = MicroexperimentLab::cold_start(origin());
    let mut candidate = qos_candidate(1);
    candidate.action_key = "x".repeat(97);
    assert_eq!(
        lab.propose(candidate, PairGates::healthy_enabled()),
        Err(LabError::Invalid)
    );

    let mut candidate = qos_candidate(2);
    candidate.stratum_hash = 0;
    assert_eq!(
        lab.propose(candidate, PairGates::healthy_enabled()),
        Err(LabError::Invalid)
    );
}

#[test]
fn persisted_state_is_bounded_and_contains_no_raw_private_fields() {
    let mut lab = MicroexperimentLab::cold_start(origin());
    for sequence in 0..MAX_OPEN_PAIRS as u64 {
        lab.propose(qos_candidate(sequence + 1), PairGates::healthy_enabled())
            .unwrap();
    }
    let json = serde_json::to_vec(&lab.persisted()).unwrap();
    assert!(json.len() <= MAX_SERIALIZED_BYTES);
    let text = String::from_utf8(json).unwrap();
    for forbidden in [
        "process_name",
        "executable_path",
        "window_title",
        "media_metadata",
    ] {
        assert!(!text.contains(forbidden));
    }
}

#[test]
fn restore_interrupts_open_pairs_and_never_emits_gold() {
    let mut lab = MicroexperimentLab::cold_start(origin());
    lab.propose(qos_candidate(1), PairGates::healthy_enabled())
        .unwrap();
    let (mut restored, disposition) = MicroexperimentLab::restore(lab.persisted(), origin());
    assert_eq!(disposition, RestoreDisposition::RestoredInterrupted);
    assert_eq!(restored.metrics().interrupted_total, 1);
    assert_eq!(restored.metrics().open_pairs, 0);
    assert!(restored.drain_pair_gold().is_empty());
}

#[test]
fn foreign_or_hostile_restore_cold_starts() {
    let lab = MicroexperimentLab::cold_start(origin());
    let mut foreign_origin = origin();
    foreign_origin.installation_id += 1;
    let (foreign, disposition) = MicroexperimentLab::restore(lab.persisted(), foreign_origin);
    assert_eq!(disposition, RestoreDisposition::ResetOrigin);
    assert_eq!(foreign.metrics().open_pairs, 0);

    let hostile =
        apollo_engine::engine::microexperiment_lab::MicroexperimentLabPersisted::oversized_for_test(
            origin(),
            MAX_SERIALIZED_BYTES + 1,
        );
    let (reset, disposition) = MicroexperimentLab::restore(hostile, origin());
    assert_eq!(disposition, RestoreDisposition::ResetHostile);
    assert_eq!(reset.metrics().open_pairs, 0);
}

#[test]
fn restore_reconciles_unbacked_pair_gold_counters_before_ais_can_read_them() {
    let persisted = MicroexperimentLab::cold_start(origin())
        .persisted()
        .with_claimed_gold_for_test(9_999, 8_888, 7_777);
    let (restored, disposition) = MicroexperimentLab::restore(persisted, origin());
    assert_eq!(disposition, RestoreDisposition::Restored);
    assert_eq!(restored.metrics().pair_gold_total, 0);
    assert_eq!(restored.metrics().effective_total, 0);
    assert_eq!(restored.metrics().harmful_total, 0);
}

#[test]
fn restart_downgrades_persisted_gold_until_local_endpoints_are_reverified() {
    let persisted = MicroexperimentLab::cold_start(origin())
        .persisted()
        .with_forged_gold_for_test();
    let (restored, disposition) = MicroexperimentLab::restore(persisted, origin());
    assert_eq!(disposition, RestoreDisposition::Restored);
    assert_eq!(restored.metrics().pair_gold_total, 0);
    assert_eq!(restored.metrics().effective_total, 0);
    assert_eq!(restored.metrics().harmful_total, 0);
}

#[test]
fn bounded_work_never_scans_beyond_open_pair_cap() {
    let mut lab = MicroexperimentLab::cold_start(origin());
    for sequence in 0..MAX_OPEN_PAIRS as u64 {
        lab.propose(qos_candidate(sequence + 1), PairGates::healthy_enabled())
            .unwrap();
    }
    let missing = apollo_engine::engine::microexperiment_lab::PairId(u128::MAX);
    assert_eq!(
        lab.advance_washout(missing, u32::MAX),
        Err(LabError::UnknownPair)
    );
    assert!(lab.last_work().pairs_examined <= MAX_OPEN_PAIRS);
}

#[test]
fn learned_state_round_trips_bounded_lab_checkpoint() {
    let mut lab = MicroexperimentLab::cold_start(origin());
    lab.propose(qos_candidate(1), PairGates::healthy_enabled())
        .unwrap();
    let value = serde_json::json!({
        "version": 2,
        "microexperiment_lab": lab.persisted(),
    });
    let state: apollo_engine::engine::learned_state::LearnedState =
        serde_json::from_value(value).unwrap();
    assert!(state.microexperiment_lab.is_some());
    let encoded = serde_json::to_value(state).unwrap();
    assert!(encoded["microexperiment_lab"].is_object());
}

#[test]
fn shadow_evaluation_never_opens_or_executes_a_pair() {
    let mut lab = MicroexperimentLab::cold_start(origin());
    let assignment = lab
        .evaluate_shadow(
            qos_candidate(1),
            PairGates {
                inherited_safe: true,
                ..PairGates::default()
            },
        )
        .unwrap();
    assert_eq!(assignment.order, PairOrder::ControlThenTreatment);
    assert_eq!(lab.metrics().open_pairs, 0);
    assert_eq!(lab.metrics().shadow_would_open_total, 1);
    assert!(lab.drain_pair_gold().is_empty());
}

#[test]
fn shadow_evaluation_still_respects_inherited_safety() {
    let mut lab = MicroexperimentLab::cold_start(origin());
    let gates = PairGates {
        inherited_safe: false,
        ..PairGates::healthy_enabled()
    };
    assert_eq!(
        lab.evaluate_shadow(qos_candidate(1), gates),
        Err(LabError::Gate)
    );
    assert_eq!(lab.metrics().shadow_would_open_total, 0);
}
