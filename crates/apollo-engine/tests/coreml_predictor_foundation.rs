use apollo_engine::engine::coreml_predictor::{
    cpu_oracle_predict, CoreMlBackend, CoreMlPredictor, Prediction, PredictorBackend,
    PredictorStatus, TemporalFeatureVector, MAX_TEMPORAL_FEATURES, MODEL_HASH, SCHEMA_VERSION,
    TEMPORAL_FEATURE_COUNT, TEMPORAL_SCHEMA_HASH,
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

#[test]
fn cpu_only_coreml_is_not_accelerator_evidence() {
    let status = PredictorStatus {
        backend: PredictorBackend::CoreMl,
        requested_backend: CoreMlBackend::CpuAndNeuralEngine,
        effective_backend: Some(CoreMlBackend::CpuOnly),
        model_available: true,
        ane_execution_measured: false,
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
    assert_eq!(status.effective_backend, None);
    assert!(!status.ane_execution_measured);
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
    assert!(!predictor.status().ane_execution_measured);
}
