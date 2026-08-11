use apollo_engine::engine::decision_ledger::{
    AdviserContribution, CandidateAlternative, DecisionId, DecisionLifecycle, PredictionRecord,
};
use apollo_engine::engine::installation_identity::InstallationId;
use apollo_engine::engine::learning_hierarchy::{
    classify_family, ForegroundBand, Goal, HierarchyContext, HierarchyPath, LearningHierarchy,
    MediaState, ResolvedLearningDetails, Strategy, WorkloadClass, MAX_CONTEXTS_PER_FAMILY,
    MAX_PROCESSED_DECISION_IDS, MAX_PROTOTYPES, MAX_REPRESENTATIVE_ACTIONS, RETRIEVAL_TOP_K,
};
use apollo_engine::engine::model_calibration::{
    CalibrationActionScope, CalibrationHorizon, CalibrationKey, ForecastCalibrationDelta,
    ForegroundContext, PressureBand, ProcessClass, ProducerId, SeparabilityState, ThermalBand,
    TrustState,
};
use apollo_engine::engine::telemetry_medallion::{
    ActuatorEpisodeContext, ActuatorFamily, HardwareRegime,
};

fn hardware() -> HardwareRegime {
    HardwareRegime {
        p_core_count: 4,
        e_core_count: 6,
        ram_gib: 16,
    }
}

fn canonical_action(family: ActuatorFamily, suffix: &str) -> String {
    match family {
        ActuatorFamily::Coordinated => "coordinated:boost+throttle".to_string(),
        ActuatorFamily::Sysctl => format!("sysctl:{suffix}=1"),
        _ => format!("{}:{suffix}", family.as_str()),
    }
}

fn detail(id: u64, family: ActuatorFamily, action: &str, utility: f64) -> ResolvedLearningDetails {
    let context = ActuatorEpisodeContext {
        valid: true,
        memory_pressure: 0.62,
        thermal_score: 0.40,
        foreground_app_hash: 9,
        user_audio_active: true,
        ..ActuatorEpisodeContext::default()
    };
    let path = HierarchyPath::classify(family, action).expect("canonical action");
    let hierarchy_context =
        HierarchyContext::classify("build", &context).expect("canonical context");
    let prediction = PredictionRecord {
        source: "world-model".to_string(),
        expected_utility: 0.08,
        uncertainty: 0.10,
        horizon_cycles: 30,
        positive_probability: Some(0.75),
        binary_target: None,
    };
    ResolvedLearningDetails {
        decision_id: DecisionId(id),
        lifecycle: DecisionLifecycle::Applied,
        hierarchy: path,
        context: hierarchy_context,
        alternatives: vec![CandidateAlternative {
            action_key: format!("{}:alternative", family.as_str()),
            target: "background".to_string(),
            expected_utility: 0.04,
            uncertainty: 0.15,
        }],
        predictions: vec![prediction.clone()],
        adviser_contributions: vec![AdviserContribution {
            adviser: "policy-scorer".to_string(),
            support: 0.7,
            uncertainty: 0.1,
        }],
        expected_utility: prediction.expected_utility,
        actual_utility: utility,
        raw_utility_delta: utility,
        counterfactual_delta: 0.0,
        quality: 0.95,
        causal_quality: 0.95,
        confounder_count: 0,
        separability: if family == ActuatorFamily::Coordinated {
            SeparabilityState::CoordinatedComposite
        } else {
            SeparabilityState::Individual
        },
        calibration_deltas: vec![ForecastCalibrationDelta {
            key: CalibrationKey {
                producer: ProducerId::WorldModel,
                action: CalibrationActionScope::Family(family),
                workload: "build".to_string(),
                process_class: ProcessClass::Background,
                horizon: CalibrationHorizon::Sec30,
                pressure: PressureBand::High,
                thermal: ThermalBand::Nominal,
                foreground: ForegroundContext::Active,
            },
            predicted_utility: prediction.expected_utility,
            actual_utility: utility,
            signed_error: utility - prediction.expected_utility,
            normalized_absolute_error: (utility - prediction.expected_utility).abs() / 2.0,
            uncertainty_covered: true,
            brier: None,
            trust_before: TrustState::Immature,
            trust_after: TrustState::Candidate,
        }],
        installation_id: InstallationId(7),
        hardware_regime: hardware(),
        resolved_cycle: id.saturating_add(100),
        resolved_timestamp_unix: 1_700_000_000 + id as i64,
    }
}

#[test]
fn all_twenty_families_have_stable_total_hierarchy_and_serde() {
    use ActuatorFamily::*;
    use Goal::*;
    use Strategy::*;

    let expected = [
        (Boost, Responsiveness, ProtectForeground),
        (Throttle, MemoryHeadroom, RelievePressure),
        (Freeze, MemoryHeadroom, RelievePressure),
        (Unfreeze, Stability, RecoverState),
        (Memorystatus, MemoryHeadroom, RelievePressure),
        (Sysctl, Stability, RecoverState),
        (Spotlight, EnergyEfficiency, ShiftBackgroundWork),
        (Quarantine, EnergyEfficiency, ShiftBackgroundWork),
        (ThreadQos, Responsiveness, ProtectForeground),
        (MarkovPrewarm, Responsiveness, PredictNextUse),
        (InteractionQos, Responsiveness, ProtectForeground),
        (IoShaping, Responsiveness, ProtectForeground),
        (PredictiveThreshold, Stability, PredictNextUse),
        (PredictiveProfile, Responsiveness, PredictNextUse),
        (PredictivePreThrottle, MemoryHeadroom, RelievePressure),
        (PredictivePurge, MemoryHeadroom, RelievePressure),
        (ChromiumEcore, EnergyEfficiency, ReduceEnergy),
        (ChromiumPurge, MemoryHeadroom, RelievePressure),
        (ChromiumJetsam, MemoryHeadroom, RelievePressure),
        (Coordinated, Stability, RecoverState),
    ];

    assert_eq!(ActuatorFamily::ALL.len(), expected.len());
    for (family, goal, strategy) in expected {
        assert_eq!(classify_family(family), (goal, strategy));
        let path = HierarchyPath::classify(family, &canonical_action(family, "canonical"))
            .expect("known family has a canonical path");
        assert_eq!(
            (path.goal, path.strategy, path.family),
            (goal, strategy, family)
        );
        assert_eq!(
            serde_json::from_str::<HierarchyPath>(&serde_json::to_string(&path).unwrap()).unwrap(),
            path
        );
    }
    for goal in [
        Stability,
        Responsiveness,
        MemoryHeadroom,
        ThermalSafety,
        EnergyEfficiency,
    ] {
        assert_eq!(
            serde_json::from_str::<Goal>(&serde_json::to_string(&goal).unwrap()).unwrap(),
            goal
        );
    }
    for strategy in [
        ProtectForeground,
        PredictNextUse,
        RelievePressure,
        ShiftBackgroundWork,
        RecoverState,
        ReduceEnergy,
    ] {
        assert_eq!(
            serde_json::from_str::<Strategy>(&serde_json::to_string(&strategy).unwrap()).unwrap(),
            strategy
        );
    }
    for workload in [
        WorkloadClass::Build,
        WorkloadClass::LlmInference,
        WorkloadClass::Browsing,
        WorkloadClass::Idle,
        WorkloadClass::Unknown,
    ] {
        assert_eq!(
            serde_json::from_str::<WorkloadClass>(&serde_json::to_string(&workload).unwrap())
                .unwrap(),
            workload
        );
    }
    for foreground in [ForegroundBand::Foreground, ForegroundBand::Background] {
        assert_eq!(
            serde_json::from_str::<ForegroundBand>(&serde_json::to_string(&foreground).unwrap())
                .unwrap(),
            foreground
        );
    }
    for media in [MediaState::Quiet, MediaState::Audio, MediaState::Call] {
        assert_eq!(
            serde_json::from_str::<MediaState>(&serde_json::to_string(&media).unwrap()).unwrap(),
            media
        );
    }
}

#[test]
fn rich_detail_round_trip_preserves_one_identity_and_full_bounded_provenance() {
    let original = detail(44, ActuatorFamily::Boost, "boost:Editor", 0.06);
    let encoded = serde_json::to_vec(&original).unwrap();
    let restored: ResolvedLearningDetails = serde_json::from_slice(&encoded).unwrap();

    assert_eq!(restored, original);
    assert_eq!(restored.decision_id, DecisionId(44));
    assert_eq!(restored.alternatives.len(), 1);
    assert_eq!(restored.predictions.len(), 1);
    assert_eq!(restored.adviser_contributions.len(), 1);
    assert_eq!(restored.calibration_deltas.len(), 1);
    assert_eq!(
        restored.calibration_deltas[0].trust_before,
        TrustState::Immature
    );
    assert_eq!(
        restored.calibration_deltas[0].trust_after,
        TrustState::Candidate
    );
}

#[test]
fn hierarchy_context_and_propositions_are_bounded_and_never_target_derived() {
    let details = detail(45, ActuatorFamily::Boost, "boost:Editor", 0.06);
    assert_eq!(details.hierarchy.action, "boost:action");
    assert_eq!(details.context.workload, WorkloadClass::Build);
    assert_eq!(details.context.media, MediaState::Audio);
    let propositions = details.hierarchy.propositions(details.context);

    assert_eq!(propositions.len(), 4);
    assert_eq!(propositions[0], "goal:responsiveness");
    assert_eq!(
        propositions[1],
        "strategy:responsiveness:protect-foreground"
    );
    assert_eq!(
        propositions[2],
        "tactic:responsiveness:protect-foreground:boost"
    );
    assert!(propositions[3].starts_with(
        "context:responsiveness:protect-foreground:boost:build:high:nominal:foreground:audio"
    ));
    assert!(propositions.iter().all(|key| !key.contains("Editor")));
    assert!(HierarchyPath::classify(ActuatorFamily::Boost, "").is_none());
    assert!(HierarchyPath::classify(ActuatorFamily::Boost, "throttle:Editor").is_none());
    assert!(HierarchyPath::classify(ActuatorFamily::Boost, "boost:pid:123").is_none());
    assert!(HierarchyPath::classify(ActuatorFamily::Boost, &"x".repeat(321)).is_none());
}

#[test]
fn one_decision_updates_once_distinct_ids_update_and_caps_are_fixed() {
    let mut memory = LearningHierarchy::new(InstallationId(7), hardware());
    let first = detail(1, ActuatorFamily::Boost, "boost:Editor", 0.06);
    let duplicate = first.clone();
    let distinct = detail(2, ActuatorFamily::Boost, "boost:Editor", -0.04);

    assert!(memory.consolidate(&first).accepted());
    assert!(memory.consolidate(&duplicate).duplicate());
    assert!(memory.consolidate(&distinct).accepted());
    assert_eq!(memory.prototype_count(), 1);
    assert_eq!(memory.processed_decision_count(), 2);
    assert_eq!(memory.prototype_for(&first).unwrap().observations, 2);

    for id in 3..260 {
        let family = ActuatorFamily::ALL[(id as usize) % ActuatorFamily::ALL.len()];
        let mut item = detail(
            id,
            family,
            &canonical_action(family, &format!("action-{id}")),
            0.02,
        );
        item.context.pressure = match id % 4 {
            0 => PressureBand::Low,
            1 => PressureBand::Moderate,
            2 => PressureBand::High,
            _ => PressureBand::Critical,
        };
        item.context.thermal = match (id / 4) % 4 {
            0 => ThermalBand::Cool,
            1 => ThermalBand::Nominal,
            2 => ThermalBand::Warm,
            _ => ThermalBand::Hot,
        };
        let _ = memory.consolidate(&item);
    }

    assert!(memory.prototype_count() <= MAX_PROTOTYPES);
    assert!(memory.processed_decision_count() <= MAX_PROCESSED_DECISION_IDS);
    for prototype in memory.prototypes() {
        assert!(prototype.representative_actions.len() <= MAX_REPRESENTATIVE_ACTIONS);
        assert!(memory.variant_count(prototype.key.family) <= MAX_CONTEXTS_PER_FAMILY);
    }
    assert!(
        memory
            .retrieve(&first.context, first.hierarchy.family, 1_800_000_000)
            .len()
            <= RETRIEVAL_TOP_K
    );
}

#[test]
fn coordinated_and_invalid_authority_are_inert_and_restore_is_origin_gated() {
    let mut memory = LearningHierarchy::new(InstallationId(7), hardware());
    let mut invalid = detail(80, ActuatorFamily::Boost, "boost:Editor", 0.04);
    invalid.lifecycle = DecisionLifecycle::NoOp;
    assert!(memory.consolidate(&invalid).rejected());
    invalid.lifecycle = DecisionLifecycle::Applied;
    invalid.causal_quality = f64::NAN;
    assert!(memory.consolidate(&invalid).rejected());

    let coordinated = detail(
        81,
        ActuatorFamily::Coordinated,
        "coordinated:boost+throttle",
        0.05,
    );
    assert!(memory.consolidate(&coordinated).accepted());
    assert_eq!(memory.prototype_count(), 1);

    let bytes = serde_json::to_vec(&memory).unwrap();
    let mut restored: LearningHierarchy = serde_json::from_slice(&bytes).unwrap();
    assert!(!restored.restore_for_origin(InstallationId(7), hardware()));
    assert_eq!(restored.prototype_count(), 1);
    assert!(restored.restore_for_origin(
        InstallationId(8),
        HardwareRegime {
            ram_gib: 24,
            ..hardware()
        }
    ));
    assert_eq!(restored.prototype_count(), 0);
    assert_eq!(restored.processed_decision_count(), 0);
}
