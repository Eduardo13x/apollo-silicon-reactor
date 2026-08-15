use std::sync::Arc;

use apollo_engine::engine::event_mesh::{
    EventEnvelope, EventIngestOutcome, EventMesh, EventPayload, EventSource, LifecycleEvent,
    EVENT_MESH_CAPACITY,
};
use apollo_engine::engine::network_flow::NetworkWorldObservation;
use apollo_engine::engine::webflow_controller::WebWorldObservation;
use apollo_engine::engine::webflow_types::{WebFlowPhase, WebFlowSource};
use apollo_engine::engine::world_state::{
    FeatureStore, WorldIdentity, WorldStatePublisher, WorldStateSnapshot, FEATURE_CAPACITY,
    RETAINED_WORLD_REVISIONS,
};

fn scalar(source: EventSource, generation: u64, sequence: u64, value: f32) -> EventEnvelope {
    EventEnvelope::scalar(
        7,
        source,
        generation,
        sequence,
        sequence * 10,
        10_000,
        value,
    )
    .unwrap()
}

fn web_observation() -> WebWorldObservation {
    WebWorldObservation {
        accepted_events: 1,
        active_navigations: 1,
        last_phase: Some(WebFlowPhase::Committed),
        source: Some(WebFlowSource::ExtensionLifecycle),
        confidence_q: 9_000,
        last_event_age_ms: Some(4),
        vitals_available: false,
    }
}

#[test]
fn duplicate_and_out_of_order_events_cannot_rewind_a_source() {
    let mut mesh = EventMesh::new(7);
    assert_eq!(
        mesh.ingest(scalar(EventSource::Pressure, 1, 1, 0.2)),
        EventIngestOutcome::Accepted
    );
    assert_eq!(
        mesh.ingest(scalar(EventSource::Pressure, 1, 1, 0.4)),
        EventIngestOutcome::Duplicate
    );
    assert_eq!(
        mesh.ingest(scalar(EventSource::Pressure, 1, 0, 0.1)),
        EventIngestOutcome::OutOfOrder
    );
    assert_eq!(mesh.len(), 1);
}

#[test]
fn new_source_generation_accepts_sequence_zero_and_rejects_old_generation() {
    let mut mesh = EventMesh::new(7);
    mesh.ingest(scalar(EventSource::Thermal, 2, 9, 0.3));
    assert_eq!(
        mesh.ingest(scalar(EventSource::Thermal, 3, 0, 0.2)),
        EventIngestOutcome::Accepted
    );
    assert_eq!(
        mesh.ingest(scalar(EventSource::Thermal, 2, 10, 0.4)),
        EventIngestOutcome::OutOfOrder
    );
}

#[test]
fn replaceable_overflow_coalesces_while_lifecycle_overflow_degrades() {
    let mut replaceable = EventMesh::new(7);
    for sequence in 0..EVENT_MESH_CAPACITY as u64 {
        let source = if sequence == 0 {
            EventSource::Pressure
        } else {
            EventSource::Process
        };
        replaceable.ingest(scalar(source, 1, sequence, 0.2));
    }
    assert_eq!(
        replaceable.ingest(scalar(EventSource::Pressure, 1, 999, 0.9)),
        EventIngestOutcome::Coalesced
    );
    assert_eq!(replaceable.len(), EVENT_MESH_CAPACITY);
    assert_eq!(replaceable.metrics().coalesced_total, 1);

    let mut nonreplaceable = EventMesh::new(7);
    for sequence in 0..EVENT_MESH_CAPACITY as u64 {
        nonreplaceable.ingest(scalar(EventSource::Process, 1, sequence, 0.1));
    }
    let wake = EventEnvelope::lifecycle(7, 1, 999, 20_000, LifecycleEvent::Wake);
    assert_eq!(nonreplaceable.ingest(wake), EventIngestOutcome::Dropped);
    assert_eq!(nonreplaceable.metrics().dropped_total, 1);
    assert!(nonreplaceable.source_degraded(EventSource::Lifecycle));
}

#[test]
fn event_mesh_has_exact_fixed_capacity() {
    assert_eq!(EventMesh::new(1).capacity(), EVENT_MESH_CAPACITY);
}

#[test]
fn scalar_event_rejects_nonfinite_values_and_bad_confidence() {
    assert!(EventEnvelope::scalar(1, EventSource::Power, 1, 1, 1, 1, f32::NAN).is_err());
    assert!(EventEnvelope::scalar(1, EventSource::Power, 1, 1, 1, 10_001, 1.0).is_err());
}

#[test]
fn event_mesh_accepts_a_bounded_numeric_webflow_phase() {
    let event = EventEnvelope::webflow(
        7,
        1,
        1,
        10,
        WebFlowPhase::Committed,
        WebFlowSource::ExtensionLifecycle,
        1,
    )
    .expect("valid webflow event");
    assert_eq!(event.source, EventSource::WebFlow);
    let encoded = serde_json::to_string(&event).unwrap();
    assert!(!encoded.contains("url"));
    assert!(!encoded.contains("title"));
}

#[test]
fn feature_store_rejects_nonfinite_and_the_257th_feature() {
    let valid = vec![0.25; FEATURE_CAPACITY];
    assert!(FeatureStore::try_new(1, valid).is_ok());
    assert!(FeatureStore::try_new(1, vec![0.0; FEATURE_CAPACITY + 1]).is_err());
    assert!(FeatureStore::try_new(1, vec![f32::INFINITY]).is_err());
}

fn snapshot(revision: u64) -> WorldStateSnapshot {
    WorldStateSnapshot::new(
        WorldIdentity {
            daemon_epoch: 7,
            revision,
            workload_id: 11,
            capability_revision: 2,
            thermal_revision: 3,
            process_revision: 4,
            session_revision: 5,
            kill_switch: false,
            sleeping: false,
        },
        revision,
        FeatureStore::try_new(1, vec![revision as f32]).unwrap(),
    )
    .unwrap()
}

#[test]
fn publisher_retains_three_whole_revisions() {
    let publisher = WorldStatePublisher::new(snapshot(1));
    for revision in 2..=4 {
        publisher.publish(snapshot(revision)).unwrap();
    }
    assert_eq!(publisher.retained_len(), RETAINED_WORLD_REVISIONS);
    assert!(publisher.by_revision(1).is_none());
    assert_eq!(publisher.by_revision(2).unwrap().identity.revision, 2);
    assert_eq!(publisher.latest().identity.revision, 4);
}

#[test]
fn publisher_rejects_epoch_change_and_nonmonotonic_revision() {
    let publisher = WorldStatePublisher::new(snapshot(1));
    assert!(publisher.publish(snapshot(1)).is_err());
    let mut wrong_epoch = snapshot(2);
    wrong_epoch.identity.daemon_epoch = 8;
    assert!(publisher.publish(wrong_epoch).is_err());
}

#[test]
fn readers_observe_a_complete_old_or_new_arc() {
    let publisher = Arc::new(WorldStatePublisher::new(snapshot(1)));
    let reader = Arc::clone(&publisher);
    let handle = std::thread::spawn(move || {
        for _ in 0..1000 {
            let world = reader.latest();
            let feature = world.features.values()[0] as u64;
            assert_eq!(world.identity.revision, feature);
        }
    });
    publisher.publish(snapshot(2)).unwrap();
    handle.join().unwrap();
}

#[test]
fn lifecycle_identity_change_invalidates_an_old_result_identity() {
    let old = snapshot(1).identity;
    let mut wake = old;
    wake.revision += 1;
    wake.session_revision += 1;
    assert!(!old.accepts_result_for(wake));
    assert!(wake.accepts_result_for(wake));
}

#[test]
fn session_revision_change_clears_web_observation_before_publication() {
    let publisher = WorldStatePublisher::new(snapshot(1).with_web(Some(web_observation())));
    assert!(publisher.latest().web.is_some());
    let mut next = snapshot(2).with_web(Some(web_observation()));
    next.identity.session_revision += 1;
    publisher.publish(next).unwrap();
    assert!(publisher.latest().web.is_none());
}

#[test]
fn session_revision_change_clears_universal_network_observation() {
    let network = NetworkWorldObservation {
        active: true,
        inferred: true,
        target_available: true,
        socket_active: true,
        sample_fresh: true,
        interaction_active: true,
        traffic_bps: 250_000,
        confidence_q: 5_000,
    };
    let publisher = WorldStatePublisher::new(snapshot(1).with_network(Some(network)));
    assert!(publisher.latest().network.is_some());
    let mut next = snapshot(2).with_network(Some(network));
    next.identity.session_revision += 1;
    publisher.publish(next).unwrap();
    assert!(publisher.latest().network.is_none());
}

#[test]
fn universal_network_observation_contains_no_app_or_destination_fields() {
    let world = snapshot(1).with_network(Some(NetworkWorldObservation {
        active: true,
        inferred: true,
        target_available: true,
        socket_active: true,
        sample_fresh: true,
        interaction_active: true,
        traffic_bps: 64_000,
        confidence_q: 4_500,
    }));
    let encoded = serde_json::to_string(&world).unwrap();
    for forbidden in ["app_name", "url", "host", "port", "destination"] {
        assert!(
            !encoded.contains(forbidden),
            "found forbidden field {forbidden}"
        );
    }
}

#[test]
fn recent_results_allow_two_cycles_but_never_cross_semantic_revisions() {
    let old = snapshot(10).identity;
    let mut current = old;
    current.revision = 12;
    assert!(old.accepts_recent_result_for(current, 2));

    current.revision = 13;
    assert!(!old.accepts_recent_result_for(current, 2));

    for mutate in [
        |identity: &mut WorldIdentity| identity.daemon_epoch += 1,
        |identity: &mut WorldIdentity| identity.workload_id += 1,
        |identity: &mut WorldIdentity| identity.capability_revision += 1,
        |identity: &mut WorldIdentity| identity.thermal_revision += 1,
        |identity: &mut WorldIdentity| identity.process_revision += 1,
        |identity: &mut WorldIdentity| identity.session_revision += 1,
    ] {
        let mut changed = old;
        changed.revision = 11;
        mutate(&mut changed);
        assert!(!old.accepts_recent_result_for(changed, 2));
    }

    let mut sleeping = old;
    sleeping.revision = 11;
    sleeping.sleeping = true;
    assert!(!old.accepts_recent_result_for(sleeping, 2));

    let mut killed = old;
    killed.revision = 11;
    killed.kill_switch = true;
    assert!(!old.accepts_recent_result_for(killed, 2));
}

#[test]
fn serialized_event_payload_has_no_free_form_content_variant() {
    let event = scalar(EventSource::VisualActivity, 1, 1, 0.7);
    let encoded = serde_json::to_string(&event).unwrap();
    assert!(matches!(event.payload, EventPayload::Scalar { .. }));
    assert!(!encoded.contains("title"));
    assert!(!encoded.contains("path"));
    assert!(!encoded.contains("text"));
}
