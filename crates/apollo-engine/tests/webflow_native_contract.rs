use std::io::{Cursor, ErrorKind};

use apollo_engine::engine::webflow_native::{
    process_bridge_payload, read_native_frame, send_event_to_context_agent_at,
    webflow_agent_socket_path_for, write_native_frame, BridgeAck, ContextWebFlowServer,
    EventTokenBucket,
};
use apollo_engine::engine::webflow_types::{
    OpaqueId, WebFlowEvent, WebFlowMetrics, WebFlowPhase, WebFlowSource, MAX_WEBFLOW_MESSAGE_BYTES,
    WEBFLOW_SCHEMA_VERSION,
};

fn event(sequence: u64) -> WebFlowEvent {
    WebFlowEvent {
        schema_version: WEBFLOW_SCHEMA_VERSION,
        browser_session_id: OpaqueId::new([1; 16]).unwrap(),
        tab_session_id: OpaqueId::new([2; 16]).unwrap(),
        navigation_id: OpaqueId::new([3; 16]).unwrap(),
        sequence,
        phase: WebFlowPhase::Started,
        source: WebFlowSource::ExtensionLifecycle,
        site_bucket: None,
        metrics: WebFlowMetrics::default(),
    }
}

#[test]
fn native_frame_roundtrips_with_native_endian_length() {
    let payload = br#"{"schema_version":1}"#;
    let mut wire = Vec::new();
    write_native_frame(&mut wire, payload).expect("write frame");
    assert_eq!(
        u32::from_ne_bytes(wire[..4].try_into().unwrap()) as usize,
        payload.len()
    );
    let decoded = read_native_frame(&mut Cursor::new(wire))
        .expect("read frame")
        .expect("one frame");
    assert_eq!(decoded, payload);
}

#[test]
fn native_frame_rejects_zero_and_oversize_before_allocating() {
    let zero = 0u32.to_ne_bytes();
    let error = read_native_frame(&mut Cursor::new(zero)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidData);

    let oversize = ((MAX_WEBFLOW_MESSAGE_BYTES + 1) as u32).to_ne_bytes();
    let error = read_native_frame(&mut Cursor::new(oversize)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidData);
}

#[test]
fn native_frame_distinguishes_clean_eof_from_truncation() {
    assert!(read_native_frame(&mut Cursor::new(Vec::<u8>::new()))
        .unwrap()
        .is_none());
    let error = read_native_frame(&mut Cursor::new(vec![1, 0])).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
}

#[test]
fn token_bucket_caps_sixty_four_events_per_second() {
    let mut bucket = EventTokenBucket::new(0);
    assert_eq!((0..65).filter(|_| bucket.admit(0)).count(), 64);
    assert!(bucket.admit(1_000), "one-second boundary refills capacity");
}

#[test]
fn socket_path_is_uid_scoped_and_stays_below_unix_limit() {
    let path = webflow_agent_socket_path_for("/tmp/user-private", 501).unwrap();
    let text = path.to_string_lossy();
    assert!(text.contains("501"));
    assert!(text.ends_with(".sock"));
    assert!(text.len() < 104);
}

#[test]
fn user_socket_delivers_one_validated_event_and_acknowledges_it() {
    let directory = tempfile::tempdir_in("/private/tmp").unwrap();
    let path = directory.path().join("webflow.sock");
    let server = ContextWebFlowServer::bind_at(path.clone()).unwrap();
    let handle = std::thread::spawn(move || {
        let mut received = None;
        server
            .serve_once(|event| {
                received = Some(event);
                Ok(())
            })
            .unwrap();
        received.expect("one event")
    });

    let ack = send_event_to_context_agent_at(&path, &event(7)).unwrap();
    assert_eq!(ack, BridgeAck::ACCEPTED);
    assert_eq!(handle.join().unwrap(), event(7));
}

#[test]
fn bridge_processing_rejects_invalid_json_and_rate_limited_events() {
    let mut bucket = EventTokenBucket::new(0);
    assert_eq!(
        process_bridge_payload(b"not-json", &mut bucket, 0, |_| Ok(())),
        BridgeAck::REJECTED
    );

    let payload = event(1).bounded_json().unwrap();
    let mut forwarded = 0;
    for _ in 0..64 {
        let ack = process_bridge_payload(&payload, &mut bucket, 0, |_| {
            forwarded += 1;
            Ok(())
        });
        assert_eq!(ack, BridgeAck::ACCEPTED);
    }
    assert_eq!(forwarded, 64);
    assert_eq!(
        process_bridge_payload(&payload, &mut bucket, 0, |_| Ok(())),
        BridgeAck::REJECTED
    );
}
