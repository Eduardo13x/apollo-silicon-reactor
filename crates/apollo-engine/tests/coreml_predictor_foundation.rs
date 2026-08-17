use apollo_engine::engine::coreml_predictor::{
    cpu_oracle_predict, AneObservation, CoreMlBackend, CoreMlPredictor, Prediction,
    PredictorBackend, PredictorStatus, TemporalFeatureVector, MAX_TEMPORAL_FEATURES, MODEL_HASH,
    SCHEMA_VERSION, TEMPORAL_FEATURE_COUNT, TEMPORAL_SCHEMA_HASH,
};

#[test]
fn temporal_schema_is_versioned_bounded_and_finite() {
    assert_eq!(SCHEMA_VERSION, 1);
    assert!(TEMPORAL_FEATURE_COUNT <= MAX_TEMPORAL_FEATURES);
    assert!(TEMPORAL_SCHEMA_HASH != 0);
    assert!(MODEL_HASH != 0);

    let mut values = [0.0_f32; TEMPORAL_FEATURE_COUNT];
    values[0] = f32::NAN;
    values[1] = f32::INFINITY;
    let sanitized = TemporalFeatureVector::new(values);

    assert!(sanitized.as_slice().iter().all(|value| value.is_finite()));
    assert_eq!(sanitized.schema_version(), SCHEMA_VERSION);
}

/// Core ML accepts a compute-unit request at model load and then decides per
/// inference where work actually runs, without exposing that decision. The
/// bridge is honest about it — `coreml_predictor_bridge.mm:222` assigns 0 with
/// the comment "a compute-unit request is not measured proof of ANE use" — but
/// a `bool` cannot carry that. Downstream, "we never implemented the
/// observation" and "we measured it and the ANE stayed idle" collapse onto the
/// same `false`.
///
/// The distinction decides real actions: the first says "go build a probe", the
/// second says "stop paying for this lane". Pin them apart.
#[test]
fn an_unimplemented_ane_observation_is_not_evidence_that_the_ane_stayed_idle() {
    let unobservable = PredictorStatus {
        backend: PredictorBackend::CoreMl,
        requested_backend: CoreMlBackend::CpuAndNeuralEngine,
        configured_backend: Some(CoreMlBackend::CpuAndNeuralEngine),
        model_available: true,
        ane_observation: AneObservation::Unsupported,
        schema_hash: TEMPORAL_SCHEMA_HASH,
        model_hash: MODEL_HASH,
        reason: None,
    };
    let measured_idle = PredictorStatus {
        ane_observation: AneObservation::Measured(false),
        ..unobservable.clone()
    };

    assert_ne!(unobservable.ane_observation, measured_idle.ane_observation);
    assert!(!unobservable.ane_observation.is_measurement());
    assert!(measured_idle.ane_observation.is_measurement());
    // Neither may be reported as the ANE having run.
    assert!(!unobservable.ane_observation.ane_active());
    assert!(!measured_idle.ane_observation.ane_active());
    // And they must not render identically.
    assert_ne!(
        unobservable.ane_observation.as_str(),
        measured_idle.ane_observation.as_str()
    );
}

/// A configured backend is what Core ML accepted at load time. Reporting it
/// under a name like "effective" invites reading it as where inference ran,
/// which is the one thing the platform does not tell us.
#[test]
fn a_configured_backend_is_never_promoted_to_an_observed_one() {
    let status = PredictorStatus {
        backend: PredictorBackend::CoreMl,
        requested_backend: CoreMlBackend::CpuAndNeuralEngine,
        configured_backend: Some(CoreMlBackend::CpuAndNeuralEngine),
        model_available: true,
        ane_observation: AneObservation::Unsupported,
        schema_hash: TEMPORAL_SCHEMA_HASH,
        model_hash: MODEL_HASH,
        reason: None,
    };

    // The accelerator *configuration* is available...
    assert!(status.accelerator_backend_available());
    // ...which says nothing about the accelerator having executed.
    assert!(!status.ane_observation.ane_active());
    assert!(!status.ane_observation.is_measurement());
}

#[test]
fn cpu_only_coreml_is_not_accelerator_evidence() {
    let status = PredictorStatus {
        backend: PredictorBackend::CoreMl,
        requested_backend: CoreMlBackend::CpuAndNeuralEngine,
        configured_backend: Some(CoreMlBackend::CpuOnly),
        model_available: true,
        ane_observation: AneObservation::Unsupported,
        schema_hash: TEMPORAL_SCHEMA_HASH,
        model_hash: MODEL_HASH,
        reason: None,
    };

    assert!(!status.accelerator_backend_available());
}

#[test]
fn temporal_schema_rejects_wrong_length_and_non_finite_input() {
    let wrong_length = vec![0.0_f32; TEMPORAL_FEATURE_COUNT - 1];
    assert!(TemporalFeatureVector::try_from_slice(&wrong_length).is_err());

    let mut non_finite = vec![0.0_f32; TEMPORAL_FEATURE_COUNT];
    non_finite[3] = f32::NEG_INFINITY;
    assert!(TemporalFeatureVector::try_from_slice(&non_finite).is_err());
}

#[test]
fn cpu_oracle_is_deterministic_and_bounded() {
    let features = TemporalFeatureVector::new([
        0.82, 0.12, 0.67, 0.10, 0.74, 0.18, 0.59, 0.21, 0.80, 0.64, 0.31, 0.28, 0.16, 0.43, 0.72,
        0.05,
    ]);

    let first = cpu_oracle_predict(&features);
    let second = cpu_oracle_predict(&features);

    assert_eq!(first, second);
    assert!(first.is_finite());
    assert!(first
        .as_array()
        .iter()
        .all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn prediction_constructor_keeps_outputs_finite_and_bounded() {
    let prediction = Prediction::from_array([f32::NAN, f32::INFINITY, -4.0, 7.0]);

    assert!(prediction.is_finite());
    assert_eq!(prediction.as_array(), [0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn cpu_oracle_predictor_reports_no_unmeasured_ane_execution() {
    let predictor = CoreMlPredictor::cpu_oracle();
    let status = predictor.status();
    let features = TemporalFeatureVector::new([0.0; TEMPORAL_FEATURE_COUNT]);

    assert_eq!(status.backend, PredictorBackend::CpuOracle);
    assert_eq!(status.requested_backend, CoreMlBackend::CpuAndNeuralEngine);
    assert_eq!(status.configured_backend, None);
    assert!(!status.ane_observation.is_measurement());
    assert_eq!(predictor.predict(&features), cpu_oracle_predict(&features));
}

#[test]
fn configured_coreml_model_matches_the_cpu_oracle() {
    if std::env::var_os("APOLLO_COREML_MODEL_PATH").is_none() {
        return;
    }
    let features = TemporalFeatureVector::new([0.1; TEMPORAL_FEATURE_COUNT]);
    let oracle = cpu_oracle_predict(&features);
    let predictor = CoreMlPredictor::new();
    let status = predictor.status();
    assert!(status.model_available, "{:?}", status.reason);
    let predicted = predictor.predict(&features);
    for (left, right) in oracle.as_array().into_iter().zip(predicted.as_array()) {
        assert!(
            (left - right).abs() <= 0.000_1,
            "oracle={left} coreml={right}"
        );
    }
    assert!(!predictor.status().ane_observation.ane_active());
}
