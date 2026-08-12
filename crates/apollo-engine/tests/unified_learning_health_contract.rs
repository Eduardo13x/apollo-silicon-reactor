use apollo_engine::engine::model_calibration::{
    CalibrationActionScope, CalibrationHorizon, CalibrationKey, ForegroundContext, PressureBand,
    ProcessClass, ProducerId, ThermalBand, TrustState,
};
use apollo_engine::engine::types::RuntimeMetrics;
use apollo_engine::engine::unified_learning_health::{
    adjust_gpu_candidate, bound_display_text, combine_gpu_support, markov_rank,
    mpc_confidence_weight, planner_rank, world_model_rank_contribution, AdviceBatch,
    AdviceCandidate, AdviceRecord, ClosureObservation, ClosureOutcome, HorizonCalibrationInput,
    LatestResolvedEpisodeSnapshot, LearningEvidenceState, UnifiedLearningHealth,
    UnifiedLearningHealthCache, UnifiedLearningInput, UnifiedLearningRevision, MAX_ADVICE_REQUESTS,
};

fn key(action: &str) -> CalibrationKey {
    CalibrationKey {
        producer: ProducerId::WorldModel,
        action: CalibrationActionScope::Exact(action.to_string()),
        workload: "build".to_string(),
        process_class: ProcessClass::Compiler,
        horizon: CalibrationHorizon::Sec5,
        pressure: PressureBand::Moderate,
        thermal: ThermalBand::Nominal,
        foreground: ForegroundContext::Active,
    }
}

#[test]
fn unchanged_revision_reuses_one_immutable_health_publication() {
    let mut cache = UnifiedLearningHealthCache::default();
    let revision = UnifiedLearningRevision {
        ledger: 7,
        calibration: 11,
        hierarchy: 3,
        exploration: 2,
        causal: 5,
    };
    let mut builds = 0;
    let first = cache.refresh(revision, || {
        builds += 1;
        UnifiedLearningHealth::from_input(UnifiedLearningInput {
            local_gold_decisions: 20,
            ..Default::default()
        })
    });
    let second = cache.refresh(revision, || {
        builds += 1;
        UnifiedLearningHealth::default()
    });

    assert!(first);
    assert!(!second);
    assert_eq!(builds, 1);
    assert_eq!(cache.health().trust_inventory.local_gold_decisions, 20);
    assert_eq!(cache.rebuilds(), 1);
}

#[test]
fn changed_revision_publishes_once_and_advice_is_signed_and_trust_capped() {
    let mut health = UnifiedLearningHealth::from_input(UnifiedLearningInput {
        local_gold_decisions: 50,
        advice_records: vec![
            AdviceRecord {
                key: key("boost:action"),
                trust: TrustState::Validated,
                signed_error: 1.0,
                current_epoch: true,
            },
            AdviceRecord {
                key: key("thread_qos:interactive"),
                trust: TrustState::Trusted,
                signed_error: -1.0,
                current_epoch: true,
            },
        ],
        ..Default::default()
    });
    let candidates = [
        AdviceCandidate::new(key("boost:action")),
        AdviceCandidate::new(key("thread_qos:interactive")),
    ];
    let batch = AdviceBatch::build(&mut health, &candidates);

    assert_eq!(batch.support(&candidates[0]), 0.00125);
    assert_eq!(batch.support(&candidates[1]), -0.005);
    assert_eq!(batch.len(), 2);
}

#[test]
fn advice_batch_deduplicates_and_the_49th_candidate_is_neutral() {
    let mut health = UnifiedLearningHealth::default();
    let candidates: Vec<_> = (0..=MAX_ADVICE_REQUESTS)
        .map(|index| AdviceCandidate::new(key(&format!("boost:action-{index:02}"))))
        .collect();
    let mut duplicated = candidates.clone();
    duplicated.insert(0, candidates[0].clone());
    let batch = AdviceBatch::build(&mut health, &duplicated);

    assert_eq!(batch.len(), MAX_ADVICE_REQUESTS);
    assert_eq!(batch.support(&candidates[MAX_ADVICE_REQUESTS]), 0.0);
    assert_eq!(health.advice_overflow_total, 1);
}

#[test]
fn advice_overflow_is_one_bounded_event_per_cycle() {
    let mut health = UnifiedLearningHealth::default();
    let candidates: Vec<_> = (0..=MAX_ADVICE_REQUESTS + 3)
        .map(|index| AdviceCandidate::new(key(&format!("boost:overflow-{index:02}"))))
        .collect();

    let batch = AdviceBatch::build(&mut health, &candidates);

    assert_eq!(batch.len(), MAX_ADVICE_REQUESTS);
    assert_eq!(health.advice_overflow_total, 1);
}

#[test]
fn ais_uses_only_mature_local_evidence_and_zero_evidence_is_unavailable() {
    let empty = UnifiedLearningHealth::from_input(UnifiedLearningInput::default());
    assert_eq!(empty.ais.local_learning_maturity, 0.0);
    assert_eq!(empty.ais.learning, 0.0);
    assert_eq!(empty.ais.wisdom, 0.0);
    assert_eq!(
        empty.ledger_closure.evidence_state,
        LearningEvidenceState::Collecting
    );

    let immature = UnifiedLearningHealth::from_input(UnifiedLearningInput {
        local_gold_decisions: 10,
        closure: Some((10, 10)),
        calibrated_accuracy: Some(1.0),
        causal_resolution: Some(1.0),
        trusted_active_models: 5,
        active_models: 5,
        ..Default::default()
    });
    assert_eq!(immature.ais.learning, 0.0);
    assert_eq!(immature.ais.wisdom, 0.0);

    let mature = UnifiedLearningHealth::from_input(UnifiedLearningInput {
        local_gold_decisions: 50,
        closure: Some((40, 36)),
        calibrated_accuracy: Some(0.8),
        causal_resolution: Some(0.75),
        trusted_active_models: 5,
        active_models: 5,
        ..Default::default()
    });
    assert!((mature.ais.learning - 0.855).abs() < 1e-12);
    assert!((mature.ais.wisdom - 0.825).abs() < 1e-12);
    assert_eq!(mature.ais.local_learning_maturity, 1.0);
}

#[test]
fn imported_dormant_and_raw_activity_cannot_raise_ais() {
    let baseline = UnifiedLearningHealth::from_input(UnifiedLearningInput {
        local_gold_decisions: 50,
        imported_gold_decisions: u64::MAX,
        raw_action_count: u64::MAX,
        trusted_models: 100,
        trusted_active_models: 0,
        active_models: 0,
        ..Default::default()
    });
    assert_eq!(baseline.ais.learning, 0.0);
    assert_eq!(baseline.ais.wisdom, 0.0);
}

#[test]
fn runtime_metrics_legacy_json_defaults_every_unified_learning_field() {
    let legacy: RuntimeMetrics = serde_json::from_str(r#"{"cycles":17,"failures":2}"#).unwrap();
    assert_eq!(legacy.cycles, 17);
    assert_eq!(legacy.failures, 2);
    assert_eq!(legacy.unified_learning_schema_version, 0);
    assert_eq!(legacy.ledger_closure.local_due, 0);
    assert!(legacy.horizon_calibration.is_empty());
    assert!(!legacy.latest_resolved_episode.present);

    let encoded = serde_json::to_value(&legacy).unwrap();
    assert_eq!(encoded["cycles"], 17);
    assert_eq!(encoded["unified_learning_schema_version"], 0);
    assert!(encoded.get("ledger_closure").is_some());
}

#[test]
fn closure_counts_only_due_local_decisions_and_all_immediate_terminals() {
    let terminal = [
        ClosureOutcome::Rejected,
        ClosureOutcome::Vetoed,
        ClosureOutcome::Blocked,
        ClosureOutcome::Failed,
        ClosureOutcome::NoOp,
        ClosureOutcome::Reverted,
        ClosureOutcome::Expired,
    ];
    let mut observations: Vec<_> = terminal
        .into_iter()
        .enumerate()
        .map(|(index, outcome)| ClosureObservation {
            decision_id: index as u64 + 1,
            local: true,
            outcome,
            ..Default::default()
        })
        .collect();
    observations.extend([
        ClosureObservation {
            decision_id: 20,
            local: true,
            outcome: ClosureOutcome::Applied,
            issued_cycle: 100,
            horizon_cycles: 30,
            now_cycle: 120,
            resolved_evidence: true,
            ..Default::default()
        },
        ClosureObservation {
            decision_id: 21,
            local: true,
            outcome: ClosureOutcome::Applied,
            issued_cycle: 100,
            horizon_cycles: 30,
            now_cycle: 130,
            resolved_evidence: false,
            ..Default::default()
        },
        ClosureObservation {
            decision_id: 22,
            local: true,
            outcome: ClosureOutcome::Applied,
            issued_cycle: 100,
            horizon_cycles: 30,
            now_cycle: 130,
            resolved_evidence: true,
            ..Default::default()
        },
        ClosureObservation {
            decision_id: 23,
            local: false,
            outcome: ClosureOutcome::Rejected,
            ..Default::default()
        },
        ClosureObservation {
            decision_id: 1,
            local: true,
            outcome: ClosureOutcome::Rejected,
            duplicate: true,
            ..Default::default()
        },
    ]);

    let health = UnifiedLearningHealth::from_input(UnifiedLearningInput {
        closure_observations: observations,
        ..Default::default()
    });
    assert_eq!(health.ledger_closure.local_due, 9);
    assert_eq!(health.ledger_closure.local_closed, 8);
    assert_eq!(health.ledger_closure.open_due, 1);
    assert_eq!(health.ledger_closure.closure_coverage, Some(8.0 / 9.0));
    assert_eq!(
        health.ledger_closure.evidence_state,
        LearningEvidenceState::Available
    );
}

#[test]
fn closure_keeps_zero_id_synthetic_overflow_failures_visible() {
    let health = UnifiedLearningHealth::from_input(UnifiedLearningInput {
        closure_observations: vec![
            ClosureObservation {
                outcome: ClosureOutcome::Failed,
                local: true,
                ..Default::default()
            },
            ClosureObservation {
                outcome: ClosureOutcome::Failed,
                local: true,
                synthetic_overflow: true,
                ..Default::default()
            },
            ClosureObservation {
                outcome: ClosureOutcome::Failed,
                local: true,
                synthetic_overflow: true,
                ..Default::default()
            },
        ],
        ..Default::default()
    });

    assert_eq!(health.ledger_closure.local_due, 3);
    assert_eq!(health.ledger_closure.local_closed, 3);
}

#[test]
fn ais_maturity_uses_distinct_current_epoch_gold_decisions() {
    let health = UnifiedLearningHealth::from_input(UnifiedLearningInput {
        local_gold_decisions: 100,
        authoritative_gold_decision_ids: vec![7, 7, 8],
        closure: Some((2, 2)),
        calibrated_accuracy: Some(1.0),
        causal_resolution: Some(1.0),
        trusted_active_models: 5,
        active_models: 5,
        ..Default::default()
    });

    assert_eq!(health.trust_inventory.local_gold_decisions, 2);
    assert_eq!(health.ais.local_learning_maturity, 0.0);
    assert_eq!(health.ais.learning, 0.0);
    assert_eq!(health.ais.wisdom, 0.0);
}

#[test]
fn latest_resolved_episode_uses_newest_cycle_then_id() {
    let health = UnifiedLearningHealth::from_input(UnifiedLearningInput {
        latest_resolved_episodes: vec![
            LatestResolvedEpisodeSnapshot {
                present: true,
                id: 4,
                resolved_cycle: 20,
                action: "older-tie".into(),
                ..Default::default()
            },
            LatestResolvedEpisodeSnapshot {
                present: true,
                id: 5,
                resolved_cycle: 20,
                action: "newer-tie".into(),
                ..Default::default()
            },
            LatestResolvedEpisodeSnapshot {
                present: true,
                id: 99,
                resolved_cycle: 1,
                action: "stale".into(),
                ..Default::default()
            },
        ],
        ..Default::default()
    });

    assert_eq!(health.latest_resolved_episode.id, 5);
    assert_eq!(health.latest_resolved_episode.action, "newer-tie");
}

#[test]
fn horizon_projection_is_fixed_order_finite_and_display_text_is_ascii_bounded() {
    let health = UnifiedLearningHealth::from_input(UnifiedLearningInput {
        horizon_calibration: vec![
            HorizonCalibrationInput::new(CalibrationHorizon::Min10, 2, 0.4, 0.8, None),
            HorizonCalibrationInput::new(
                CalibrationHorizon::Sec5,
                3,
                f64::NAN,
                2.0,
                Some(f64::INFINITY),
            ),
        ],
        ..Default::default()
    });
    assert_eq!(
        health
            .horizon_calibration
            .iter()
            .map(|summary| summary.horizon.as_str())
            .collect::<Vec<_>>(),
        ["5s", "30s", "2m", "10m"]
    );
    assert_eq!(health.horizon_calibration[0].normalized_mae, None);
    assert_eq!(health.horizon_calibration[0].coverage, Some(1.0));
    assert_eq!(health.horizon_calibration[0].brier, None);
    assert_eq!(health.horizon_calibration[3].normalized_mae, Some(0.4));

    let bounded = bound_display_text("world-🚀-model-name-that-is-deliberately-long", 16);
    assert!(bounded.is_ascii());
    assert!(bounded.len() <= 16);
}

#[test]
fn ranking_seams_are_finite_bounded_and_neutral_when_advice_is_missing() {
    assert_eq!(world_model_rank_contribution(0.4, 0.0), 0.4);
    assert!(world_model_rank_contribution(0.4, 0.005) > 0.4);
    assert!(mpc_confidence_weight(0.8, -0.005) < 0.8);
    let adjusted = adjust_gpu_candidate(0.2, 0.3, 0.005);
    assert!(adjusted.expected_gain > 0.2);
    assert!(adjusted.uncertainty < 0.3);
    assert_eq!(combine_gpu_support(0.005, 0.005), 0.005);
    assert_eq!(combine_gpu_support(f64::NAN, 0.005), 0.005);
    assert_eq!(markov_rank(0.6, 0.0), 0.6);
    assert_eq!(planner_rank(0.6, 0.0), 0.6);
    assert!(markov_rank(0.6, -0.005) < 0.6);
    assert!(planner_rank(0.6, 0.005) > 0.6);
}
