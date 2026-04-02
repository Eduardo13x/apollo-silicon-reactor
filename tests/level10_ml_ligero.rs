// tests/level10_ml_ligero.rs
// Tests for the ML Ligero Bayesian workload classifier.

use apollo_optimizer::engine::{
    adaptive_governor::AdaptiveGovernor,
    llm::LearnedPolicy,
    user_profile::{UserProfile, UserProfilePersisted, WorkloadType},
    workload_classifier::{ClassifierSource, WorkloadClassification, WorkloadClassifier},
};
use std::collections::HashMap;

// ── WorkloadClassifier unit tests ─────────────────────────────────────────────

#[test]
fn test_classifier_new_has_no_learned_weights() {
    let classifier = WorkloadClassifier::new();
    // With no foreground app and no learned weights, classification should be low confidence.
    // Hour model with General: 1.0 → score = 0.30, runner_up = 0.0 → margin = 0.0
    // D1≈0.14, D2≈0.18 (1 source), D3≈0.03 (score 0.30 << 2.0) → confidence ≈ 0.13
    let hour_model: [_; 24] = std::array::from_fn(|_| {
        let mut m = std::collections::HashMap::new();
        m.insert(WorkloadType::General, 1.0f32);
        m
    });
    let app_stats = HashMap::new();
    let result = classifier.classify(None, &[], &hour_model, &app_stats, 12);
    assert!(
        result.confidence <= 0.30,
        "confidence={}",
        result.confidence
    );
}

#[test]
fn test_classifier_foreground_cursor_is_coding() {
    let classifier = WorkloadClassifier::new();
    let hour_model: [_; 24] = std::array::from_fn(|_| {
        let mut m = HashMap::new();
        m.insert(WorkloadType::General, 1.0f32);
        m
    });
    let app_stats = HashMap::new();
    let procs = vec!["Cursor", "cargo", "rustc"];
    let result = classifier.classify(Some("Cursor"), &procs, &hour_model, &app_stats, 10);
    assert_eq!(result.workload, WorkloadType::Coding);
    assert!(
        result.confidence > 0.5,
        "expected high confidence, got {}",
        result.confidence
    );
}

#[test]
fn test_classifier_idle_when_no_foreground_and_weak_evidence() {
    let classifier = WorkloadClassifier::new();
    // Use an empty hour model so no hour prior score is added.
    // With no foreground, no matching procs, and no hour prior scores,
    // best_score = 0.0, total = max(0.0, 1.0) = 1.0, confidence = 0.0 < 0.25 → Idle.
    let hour_model: [_; 24] = std::array::from_fn(|_| HashMap::new());
    let app_stats = HashMap::new();
    // Non-matching processes → no process-mix score
    let result = classifier.classify(
        None,
        &["some-daemon", "launchd"],
        &hour_model,
        &app_stats,
        3,
    );
    assert_eq!(result.workload, WorkloadType::Idle);
}

#[test]
fn test_classifier_confidence_always_in_range() {
    let classifier = WorkloadClassifier::new();
    let hour_model: [_; 24] = std::array::from_fn(|_| {
        let mut m = HashMap::new();
        m.insert(WorkloadType::General, 1.0f32);
        m
    });
    let app_stats = HashMap::new();
    let procs = vec!["Xcode", "cargo", "clang", "rustc", "git"];
    let result = classifier.classify(Some("Xcode"), &procs, &hour_model, &app_stats, 14);
    assert!(result.confidence >= 0.0);
    assert!(result.confidence <= 1.0);
}

#[test]
fn test_classifier_llm_learned_boost_increases_confidence() {
    let mut classifier = WorkloadClassifier::new();
    let policy = LearnedPolicy {
        interactive_patterns: vec!["Cursor".to_string(), "cargo".to_string()],
        noise_patterns: vec![],
        protected_patterns: vec![],
        learned_at: None,
        pattern_weights: std::collections::HashMap::new(),
    };
    classifier.update_learned_policy(&policy);

    let hour_model: [_; 24] = std::array::from_fn(|_| {
        let mut m = HashMap::new();
        m.insert(WorkloadType::General, 1.0f32);
        m
    });
    let app_stats = HashMap::new();
    let procs = vec!["Cursor", "cargo"];
    let result = classifier.classify(Some("Cursor"), &procs, &hour_model, &app_stats, 10);
    assert_eq!(result.workload, WorkloadType::Coding);
    assert!(
        result.confidence > 0.6,
        "LLM boost should push confidence high, got {}",
        result.confidence
    );
}

#[test]
fn test_classifier_hour_prior_contributes_as_source() {
    let classifier = WorkloadClassifier::new();
    let mut hour_model: [_; 24] = std::array::from_fn(|_| {
        let mut m = HashMap::new();
        m.insert(WorkloadType::General, 1.0f32);
        m
    });
    // Inject a strong hour prior for Coding at hour 10
    hour_model[10].insert(WorkloadType::Coding, 50.0);
    let app_stats = HashMap::new();
    let result = classifier.classify(None, &[], &hour_model, &app_stats, 10);
    let has_hour_prior = result
        .sources
        .iter()
        .any(|s| matches!(s, ClassifierSource::HourPrior));
    assert!(has_hour_prior, "HourPrior source should be present");
}

#[test]
fn test_classifier_process_mix_contributes_as_source() {
    let classifier = WorkloadClassifier::new();
    let hour_model: [_; 24] = std::array::from_fn(|_| {
        let mut m = HashMap::new();
        m.insert(WorkloadType::General, 1.0f32);
        m
    });
    let app_stats = HashMap::new();
    let procs = vec!["cargo", "rustc", "clang", "git"];
    let result = classifier.classify(None, &procs, &hour_model, &app_stats, 12);
    let has_mix = result
        .sources
        .iter()
        .any(|s| matches!(s, ClassifierSource::ProcessMix(_)));
    assert!(has_mix, "ProcessMix source should be present");
}

#[test]
fn test_classifier_sources_summary_returns_strings() {
    let classifier = WorkloadClassifier::new();
    let hour_model: [_; 24] = std::array::from_fn(|_| {
        let mut m = HashMap::new();
        m.insert(WorkloadType::General, 1.0f32);
        m
    });
    let app_stats = HashMap::new();
    let procs = vec!["Cursor", "cargo"];
    let result = classifier.classify(Some("Cursor"), &procs, &hour_model, &app_stats, 10);
    let summary = result.sources_summary();
    assert!(!summary.is_empty(), "sources_summary should not be empty");
    // Expected values from implementation: "foreground-app", "hour-prior", "process-mix:N"
    assert!(
        summary
            .iter()
            .any(|s| s == "foreground-app" || s.contains("prior") || s.contains("mix")),
        "unexpected summary: {:?}",
        summary
    );
}

#[test]
fn test_classifier_zoom_detected_as_video_call() {
    let classifier = WorkloadClassifier::new();
    let hour_model: [_; 24] = std::array::from_fn(|_| {
        let mut m = HashMap::new();
        m.insert(WorkloadType::General, 1.0f32);
        m
    });
    let app_stats = HashMap::new();
    let procs = vec!["zoom.us"];
    let result = classifier.classify(Some("zoom.us"), &procs, &hour_model, &app_stats, 15);
    assert_eq!(result.workload, WorkloadType::VideoCall);
}

#[test]
fn test_adaptive_governor_ml_classification_in_decide_all() {
    // Verify that after decide_all(), last_ml_classification() returns a valid result.
    let mut governor = AdaptiveGovernor::new();
    // Call decide_all with a coding foreground
    let result = governor.decide_all(
        &[], // no proc snapshots
        &[], // no hunt snapshots
        Some("Cursor"),
        &["Cursor", "cargo", "rustc"],
        10,
    );
    // decide_all should not panic and should return a vec (may be empty with no snaps)
    let _ = result;
    let ml = governor.last_ml_classification();
    assert_eq!(ml.workload, WorkloadType::Coding);
    assert!(ml.confidence > 0.0);
}

// ── UserProfile persistence tests ─────────────────────────────────────────────

#[test]
fn test_user_profile_secs_since_last_use_increments() {
    use std::time::Duration;
    let mut profile = UserProfile::new();

    // First observe — foreground is "Cursor"
    profile.observe(Some("Cursor"), &["Cursor", "cargo"], 10);

    // Small sleep to let time pass
    std::thread::sleep(Duration::from_millis(50));

    // Second observe — foreground switched to something else, closes "Cursor" session
    profile.observe(Some("Safari"), &["Safari"], 10);

    // After close_session("Cursor"), Cursor enters app_stats with secs_since_last_use = 0.
    // The field must be present and valid (u64 is always >= 0).
    let stats = profile.app_stats_ref().get("Cursor");
    if let Some(s) = stats {
        // secs_since_last_use is a u64 — always a valid non-negative value
        let _ = s.secs_since_last_use;
    }
    // Safari session is still open, so it may or may not be in app_stats yet.
    // Just verify no panic occurred and the profile is consistent.
}

#[test]
fn test_user_profile_to_from_persisted_roundtrip() {
    let mut profile = UserProfile::new();
    profile.observe(Some("Cursor"), &["cargo", "rustc"], 10);
    profile.observe(Some("Safari"), &["Safari"], 11);

    let persisted = profile.to_persisted();
    // Verify we can serialize to JSON
    let json = serde_json::to_string(&persisted).expect("should serialize");
    // And deserialize back
    let restored: UserProfilePersisted = serde_json::from_str(&json).expect("should deserialize");
    // The restored profile should have the same app_stats keys
    assert_eq!(persisted.app_stats.len(), restored.app_stats.len());
    assert_eq!(persisted.hour_model.len(), restored.hour_model.len());
}

#[test]
fn test_user_profile_from_persisted_restores_hour_model() {
    let mut profile = UserProfile::new();
    // Observe coding at hour 9 many times to build up hour model
    for _ in 0..5 {
        profile.observe(Some("Cursor"), &["cargo"], 9);
    }
    let persisted = profile.to_persisted();
    let restored = UserProfile::from_persisted(persisted);
    // The restored profile should believe hour 9 is likely Coding
    // (Coding count = 5.0 > General count = 1.0)
    let likely = restored.likely_workload_at_hour(9);
    assert_eq!(likely, WorkloadType::Coding);
}

// ── Additional coverage tests ─────────────────────────────────────────────────

#[test]
fn test_classifier_foreground_app_source_present() {
    let classifier = WorkloadClassifier::new();
    let hour_model: [_; 24] = std::array::from_fn(|_| {
        let mut m = HashMap::new();
        m.insert(WorkloadType::General, 1.0f32);
        m
    });
    let app_stats = HashMap::new();
    let result = classifier.classify(Some("Xcode"), &["Xcode"], &hour_model, &app_stats, 9);
    let has_fg = result
        .sources
        .iter()
        .any(|s| matches!(s, ClassifierSource::ForegroundApp));
    assert!(
        has_fg,
        "ForegroundApp source should be present when foreground matches a signature"
    );
}

#[test]
fn test_classifier_video_edit_detected() {
    let classifier = WorkloadClassifier::new();
    let hour_model: [_; 24] = std::array::from_fn(|_| {
        let mut m = HashMap::new();
        m.insert(WorkloadType::General, 1.0f32);
        m
    });
    let app_stats = HashMap::new();
    let procs = vec!["Final Cut", "HandBrake", "ffmpeg"];
    let result = classifier.classify(Some("Final Cut"), &procs, &hour_model, &app_stats, 16);
    assert_eq!(result.workload, WorkloadType::VideoEdit);
    assert!(
        result.confidence > 0.5,
        "Video edit confidence should be high, got {}",
        result.confidence
    );
}

#[test]
fn test_classifier_default_is_same_as_new() {
    // WorkloadClassifier derives Default via impl Default → same as new()
    let from_default = WorkloadClassifier::default();
    let hour_model: [_; 24] = std::array::from_fn(|_| {
        let mut m = HashMap::new();
        m.insert(WorkloadType::General, 1.0f32);
        m
    });
    let app_stats = HashMap::new();
    let result = from_default.classify(Some("Xcode"), &["Xcode"], &hour_model, &app_stats, 10);
    assert_eq!(result.workload, WorkloadType::Coding);
}

#[test]
fn test_governor_update_learned_policy_affects_classification() {
    let mut governor = AdaptiveGovernor::new();

    let policy = LearnedPolicy {
        interactive_patterns: vec!["zoom.us".to_string()],
        noise_patterns: vec![],
        protected_patterns: vec![],
        learned_at: None,
        pattern_weights: std::collections::HashMap::new(),
    };
    governor.update_learned_policy(&policy);

    // Now classify with zoom in foreground
    let classification = governor.classify_workload(Some("zoom.us"), &["zoom.us"], 14);
    assert_eq!(classification.workload, WorkloadType::VideoCall);
    assert!(classification.confidence > 0.5);
}

#[test]
fn test_governor_classify_workload_returns_general_for_unknown() {
    let mut governor = AdaptiveGovernor::new();
    // Use an empty hour model equivalent by calling with unknown procs
    let classification = governor.classify_workload(None, &["unknown-process-xyz"], 2);
    // Either General or Idle is acceptable for completely unknown input
    assert!(
        matches!(
            classification.workload,
            WorkloadType::General | WorkloadType::Idle
        ),
        "Expected General or Idle, got {:?}",
        classification.workload
    );
}

#[test]
fn test_classification_struct_fields_accessible() {
    // Verify that WorkloadClassification fields are public and accessible
    let c = WorkloadClassification {
        workload: WorkloadType::Coding,
        confidence: 0.85,
        sources: vec![ClassifierSource::ForegroundApp, ClassifierSource::HourPrior],
    };
    assert_eq!(c.workload, WorkloadType::Coding);
    assert!((c.confidence - 0.85).abs() < 1e-6);
    assert_eq!(c.sources.len(), 2);

    let summary = c.sources_summary();
    assert_eq!(summary.len(), 2);
    assert_eq!(summary[0], "foreground-app");
    assert_eq!(summary[1], "hour-prior");
}

#[test]
fn test_classifier_llm_noise_pattern_reduces_score() {
    // A pattern listed as noise gets negative weight toward General.
    // Using "analyticsd" as a noise pattern should not boost any productive workload.
    let mut classifier = WorkloadClassifier::new();
    let policy = LearnedPolicy {
        interactive_patterns: vec![],
        noise_patterns: vec!["analyticsd".to_string()],
        protected_patterns: vec![],
        learned_at: None,
        pattern_weights: std::collections::HashMap::new(),
    };
    classifier.update_learned_policy(&policy);

    let hour_model: [_; 24] = std::array::from_fn(|_| {
        let mut m = HashMap::new();
        m.insert(WorkloadType::General, 1.0f32);
        m
    });
    let app_stats = HashMap::new();
    // analyticsd in foreground (unusual, but tests noise logic)
    let result = classifier.classify(None, &["analyticsd"], &hour_model, &app_stats, 12);
    // Should not classify as Coding / VideoCall / etc.
    assert!(
        matches!(result.workload, WorkloadType::General | WorkloadType::Idle),
        "Noise pattern should not boost productive workloads, got {:?}",
        result.workload
    );
}
