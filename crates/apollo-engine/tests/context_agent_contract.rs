use std::time::Duration;

use apollo_engine::engine::context_agent::{
    validate_context_payload, AntiReplayStore, ContextAgentState, ContextPermissions,
    ContextSummary, ContextValidationError, PermissionState, TriState, MAX_CONTEXT_PAYLOAD_BYTES,
};
use apollo_engine::engine::protocol::DaemonRequest;

fn summary(sequence: u64) -> ContextSummary {
    ContextSummary {
        schema_version: 1,
        daemon_epoch: 7,
        sequence,
        monotonic_ns: 10_000 + sequence,
        audio_output: TriState::Yes,
        audio_input: TriState::Unknown,
        visual_change_q: 0.25,
        interaction_q: 0.75,
        permissions: ContextPermissions {
            screen_capture: PermissionState::Granted,
            microphone: PermissionState::Unknown,
            accessibility: PermissionState::Denied,
            input_monitoring: PermissionState::Unknown,
        },
    }
}

#[test]
fn submit_context_is_numeric_and_not_privileged() {
    let request = DaemonRequest::SubmitContext {
        summary: summary(1),
    };
    assert!(!request.is_privileged());

    let value: serde_json::Value = serde_json::to_value(request).expect("request serializes");
    assert!(value["payload"]["summary"]["audio_output"].is_number());
    assert!(value["payload"]["summary"]["permissions"]["screen_capture"].is_number());
    assert!(value["payload"]["summary"]["visual_change_q"].is_number());
}

#[test]
fn valid_summary_roundtrips_and_stays_below_wire_limit() {
    let value = summary(1);
    value.validate().expect("fixture is valid");
    let bytes = serde_json::to_vec(&value).expect("summary serializes");
    assert!(bytes.len() <= MAX_CONTEXT_PAYLOAD_BYTES);
    let restored: ContextSummary = serde_json::from_slice(&bytes).expect("summary parses");
    assert_eq!(restored, value);
}

#[test]
fn validation_rejects_non_finite_and_out_of_range_quality() {
    let mut nan = summary(1);
    nan.visual_change_q = f64::NAN;
    assert!(nan.validate().is_err());

    let mut too_high = summary(1);
    too_high.interaction_q = 1.000_1;
    assert!(too_high.validate().is_err());
}

#[test]
fn deserialization_rejects_non_numeric_or_unknown_fields() {
    let mut value = serde_json::to_value(summary(1)).expect("fixture serializes");
    value["audio_output"] = serde_json::json!("yes");
    assert!(serde_json::from_value::<ContextSummary>(value).is_err());

    let mut value = serde_json::to_value(summary(1)).expect("fixture serializes");
    value["raw_text"] = serde_json::json!("must not enter the contract");
    assert!(serde_json::from_value::<ContextSummary>(value).is_err());

    let request = serde_json::json!({
        "type": "SubmitContext",
        "payload": { "summary": serde_json::to_value(summary(1)).expect("fixture serializes") },
        "raw_path": "/private/user/document.txt"
    });
    let bytes = serde_json::to_vec(&request).expect("request serializes");
    assert!(validate_context_payload(&bytes).is_err());
}

#[test]
fn anti_replay_store_rejects_duplicates_and_regressions() {
    let mut store = AntiReplayStore::default();
    assert!(store.accept(summary(1)).is_ok());
    assert!(store.accept(summary(1)).is_err());
    assert!(store.accept(summary(0)).is_err());

    let mut regressed_time = summary(2);
    regressed_time.monotonic_ns = 1;
    assert!(store.accept(regressed_time).is_err());
}

#[test]
fn newer_epoch_restarts_sequence_continuity() {
    let mut store = AntiReplayStore::default();
    assert!(store.accept(summary(9)).is_ok());

    let mut next_epoch = summary(1);
    next_epoch.daemon_epoch = 8;
    next_epoch.monotonic_ns = 1;
    assert!(store.accept(next_epoch).is_ok());
}

#[test]
fn payload_parser_enforces_wire_bound_before_deserialization() {
    let request = DaemonRequest::SubmitContext {
        summary: summary(1),
    };
    let bytes = serde_json::to_vec(&request).expect("request serializes");
    assert!(validate_context_payload(&bytes).is_ok());

    let oversized = vec![b' '; MAX_CONTEXT_PAYLOAD_BYTES + 1];
    assert_eq!(
        validate_context_payload(&oversized),
        Err(ContextValidationError::PayloadTooLarge)
    );
}

#[test]
fn disconnected_context_expires_instead_of_becoming_a_permanent_signal() {
    let mut state = ContextAgentState::default();
    state.accept(summary(1)).expect("fresh context");
    assert!(state.latest_fresh(Duration::from_secs(1)).is_some());
    std::thread::sleep(Duration::from_millis(2));
    assert!(state.latest_fresh(Duration::from_millis(1)).is_none());
}
