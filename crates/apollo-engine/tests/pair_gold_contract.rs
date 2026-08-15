use apollo_engine::engine::intelligence_score::project_pair_evidence;
use apollo_engine::engine::telemetry_medallion::{
    quarantine_experimental_tier, EvidenceProvenance, EvidenceTier,
};

#[test]
fn bronze_rollback_and_unpaired_markov_never_receive_ais_credit() {
    let metrics = serde_json::json!({
        "world_model_actuator_bronze_total": 50_000,
        "world_model_actuator_effective_total": 40_000,
        "interaction_qos_activations": 300,
        "interaction_qos_reverts": 300,
        "markov_prewarm_hits": 900,
        "markov_prewarm_misses": 100,
        "microexperiment_pair_gold_total": 0,
        "microexperiment_effective_total": 0,
    });
    let projection = project_pair_evidence(&metrics);
    assert_eq!(projection.attributed_observations, 0);
    assert_eq!(projection.effective_observations, 0);
    assert_eq!(projection.verified_adaptations_correct, 0);
    assert_eq!(projection.verified_adaptations_total, 0);
}

#[test]
fn distinct_pair_gold_is_the_only_attributed_adaptation_source() {
    let metrics = serde_json::json!({
        "microexperiment_pair_gold_total": 10,
        "microexperiment_effective_total": 4,
        "microexperiment_harmful_total": 2,
        "microexperiment_mean_effect": 0.125,
        "interaction_qos_reverts": 9_999,
        "markov_prewarm_hits": 9_999,
    });
    let projection = project_pair_evidence(&metrics);
    assert_eq!(projection.attributed_observations, 10);
    assert_eq!(projection.effective_observations, 4);
    assert_eq!(projection.verified_adaptations_correct, 4);
    assert_eq!(projection.verified_adaptations_total, 10);
    assert_eq!(projection.harmful_observations, 2);
    assert!((projection.mean_effect - 0.125).abs() < f64::EPSILON);
}

#[test]
fn malformed_pair_metrics_are_clamped_conservatively() {
    let metrics = serde_json::json!({
        "microexperiment_pair_gold_total": 3,
        "microexperiment_effective_total": 30,
        "microexperiment_harmful_total": 20,
        "microexperiment_mean_effect": "nan",
    });
    let projection = project_pair_evidence(&metrics);
    assert_eq!(projection.attributed_observations, 3);
    assert_eq!(projection.effective_observations, 3);
    assert_eq!(projection.harmful_observations, 0);
    assert_eq!(projection.mean_effect, 0.0);
}

#[test]
fn synthetic_gpu_model_and_advisory_evidence_are_quarantined_at_bronze() {
    for provenance in [
        EvidenceProvenance::SyntheticCounter,
        EvidenceProvenance::GpuImagined,
        EvidenceProvenance::ModelCounterfactual,
        EvidenceProvenance::Advisory,
        EvidenceProvenance::LegacyUnknown,
    ] {
        assert_eq!(
            quarantine_experimental_tier(provenance, EvidenceTier::Gold),
            EvidenceTier::Bronze
        );
    }
    assert_eq!(
        quarantine_experimental_tier(EvidenceProvenance::ObservedLocal, EvidenceTier::Gold),
        EvidenceTier::Gold
    );
}
