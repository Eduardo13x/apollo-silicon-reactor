use apollo_engine::engine::webflow_types::{
    OpaqueBucket, OpaqueId, WebFlowErrorClass, WebFlowEvent, WebFlowIngress, WebFlowMetrics,
    WebFlowPhase, WebFlowSource, MAX_WEBFLOW_INGRESS_EVENTS, MAX_WEBFLOW_MESSAGE_BYTES,
    WEBFLOW_SCHEMA_VERSION,
};

fn id(value: u8) -> OpaqueId {
    OpaqueId::new([value; 16]).expect("nonzero opaque id")
}

fn valid_event(sequence: u64) -> WebFlowEvent {
    WebFlowEvent {
        schema_version: WEBFLOW_SCHEMA_VERSION,
        browser_session_id: id(1),
        tab_session_id: id(2),
        navigation_id: id(3),
        sequence,
        phase: WebFlowPhase::Started,
        source: WebFlowSource::ExtensionLifecycle,
        site_bucket: Some(OpaqueBucket::new([4; 16]).expect("nonzero bucket")),
        metrics: WebFlowMetrics::default(),
    }
}

#[test]
fn valid_event_roundtrips_below_wire_limit_without_content_fields() {
    let event = valid_event(1);
    let bytes = event.bounded_json().expect("valid event");
    assert!(bytes.len() < MAX_WEBFLOW_MESSAGE_BYTES);
    let json = String::from_utf8(bytes.clone()).expect("json");
    for forbidden in [
        "url", "title", "text", "cookie", "header", "body", "dom", "origin",
    ] {
        assert!(!json.to_ascii_lowercase().contains(forbidden), "{forbidden}");
    }
    let decoded = WebFlowEvent::from_bounded_json(&bytes).expect("roundtrip");
    assert_eq!(decoded, event);
}

#[test]
fn validation_rejects_zero_identity_sequence_and_unknown_fields() {
    let mut event = valid_event(1);
    event.sequence = 0;
    assert!(event.validate().is_err());

    let json = br#"{"schema_version":1,"browser_session_id":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"tab_session_id":[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],"navigation_id":[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],"sequence":1,"phase":"started","source":"extension-lifecycle","site_bucket":null,"metrics":{},"url":"https://example.invalid"}"#;
    assert!(WebFlowEvent::from_bounded_json(json).is_err());
    assert!(OpaqueId::new([0; 16]).is_err());
}

#[test]
fn validation_rejects_oversized_and_out_of_range_numeric_metrics() {
    let oversized = vec![b' '; MAX_WEBFLOW_MESSAGE_BYTES + 1];
    assert!(WebFlowEvent::from_bounded_json(&oversized).is_err());

    let mut event = valid_event(1);
    event.metrics.lcp_ms = Some(120_001);
    assert!(event.validate().is_err());
    event.metrics.lcp_ms = Some(1_200);
    event.metrics.error_class = Some(WebFlowErrorClass::Network);
    assert!(event.validate().is_ok());
}

#[test]
fn ingress_is_bounded_and_drains_at_most_requested_events() {
    let mut ingress = WebFlowIngress::new();
    for sequence in 1..=(MAX_WEBFLOW_INGRESS_EVENTS as u64 + 4) {
        ingress.accept_at(valid_event(sequence), sequence * 10);
    }
    assert_eq!(ingress.len(), MAX_WEBFLOW_INGRESS_EVENTS);
    assert_eq!(ingress.counters().dropped, 4);
    assert_eq!(ingress.drain(7).len(), 7);
    assert_eq!(ingress.len(), MAX_WEBFLOW_INGRESS_EVENTS - 7);
}

#[test]
fn legacy_metrics_fields_remain_optional_not_zero_filled() {
    let event = valid_event(1);
    assert_eq!(event.metrics.ttfb_ms, None);
    assert_eq!(event.metrics.lcp_ms, None);
    assert_eq!(event.metrics.transfer_bytes, None);
}
