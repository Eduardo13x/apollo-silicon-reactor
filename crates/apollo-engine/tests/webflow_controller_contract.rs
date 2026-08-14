use apollo_engine::engine::webflow_controller::{
    WebFlowClosure, WebFlowController, WebFlowRolloutPhase, WebFlowTickInput,
};
use apollo_engine::engine::webflow_types::{
    OpaqueId, ReceivedWebFlowEvent, WebFlowEvent, WebFlowMetrics, WebFlowPhase, WebFlowSource,
    WEBFLOW_SCHEMA_VERSION,
};

fn id(value: u8) -> OpaqueId {
    OpaqueId::new([value; 16]).expect("nonzero opaque id")
}

fn event(tab: u8, navigation: u8, sequence: u64, phase: WebFlowPhase) -> ReceivedWebFlowEvent {
    ReceivedWebFlowEvent {
        event: WebFlowEvent {
            schema_version: WEBFLOW_SCHEMA_VERSION,
            browser_session_id: id(1),
            tab_session_id: id(tab),
            navigation_id: id(navigation),
            sequence,
            phase,
            source: WebFlowSource::ExtensionLifecycle,
            site_bucket: None,
            metrics: WebFlowMetrics::default(),
        },
        received_at_ms: 1_000 + sequence * 10,
    }
}

fn input(now_ms: u64) -> WebFlowTickInput {
    WebFlowTickInput {
        now_ms,
        foreground_browser: true,
        identity_available: true,
        pressure_constrained: false,
        thermal_constrained: false,
        low_power: false,
        sleeping: false,
        kill_switch: false,
        session_revision: 1,
    }
}

#[test]
fn valid_started_event_produces_a_two_second_shadow_intent() {
    let mut controller = WebFlowController::new(WebFlowRolloutPhase::Shadow);
    let output = controller.tick(input(1_020), [event(2, 3, 1, WebFlowPhase::Started)]);
    let intent = output.intent.expect("safe deterministic intent");
    assert_eq!(intent.ttl_ms, 2_000);
    assert!(!output.admitted, "shadow must not add actuation");
    assert_eq!(output.observation.active_navigations, 1);
}

#[test]
fn newer_navigation_abandons_the_previous_tab_episode() {
    let mut controller = WebFlowController::new(WebFlowRolloutPhase::Shadow);
    controller.tick(input(1_020), [event(2, 3, 1, WebFlowPhase::Started)]);
    let output = controller.tick(input(1_040), [event(2, 4, 2, WebFlowPhase::Started)]);
    assert_eq!(output.closed.len(), 1);
    assert_eq!(output.closed[0].closure, WebFlowClosure::Abandoned);
    assert_eq!(output.observation.active_navigations, 1);
}

#[test]
fn duplicate_and_out_of_order_events_never_renew_the_lease() {
    let mut controller = WebFlowController::new(WebFlowRolloutPhase::Active);
    controller.tick(input(1_020), [event(2, 3, 2, WebFlowPhase::Started)]);
    let duplicate = controller.tick(input(1_030), [event(2, 3, 2, WebFlowPhase::Committed)]);
    let old = controller.tick(input(1_040), [event(2, 3, 1, WebFlowPhase::Committed)]);
    assert_eq!(duplicate.counters.duplicate, 1);
    assert_eq!(old.counters.out_of_order, 1);
    assert_eq!(duplicate.observation.last_phase, Some(WebFlowPhase::Started));
    assert_eq!(old.observation.last_phase, Some(WebFlowPhase::Started));
}

#[test]
fn lifecycle_loaded_uses_fixed_grace_and_never_claims_settle() {
    let mut controller = WebFlowController::new(WebFlowRolloutPhase::Active);
    controller.tick(input(1_020), [event(2, 3, 1, WebFlowPhase::Started)]);
    controller.tick(input(1_040), [event(2, 3, 2, WebFlowPhase::Loaded)]);
    let output = controller.tick(input(1_541), std::iter::empty());
    assert_eq!(output.closed[0].closure, WebFlowClosure::Expired);
    assert_eq!(output.counters.settled, 0);
}

#[test]
fn hard_deadline_caps_continuous_renewal_at_fifteen_seconds() {
    let mut controller = WebFlowController::new(WebFlowRolloutPhase::Active);
    controller.tick(input(1_020), [event(2, 3, 1, WebFlowPhase::Started)]);
    for sequence in 2..12 {
        let now = 1_020 + (sequence - 1) * 1_300;
        controller.tick(
            input(now),
            [ReceivedWebFlowEvent {
                received_at_ms: now,
                ..event(2, 3, sequence, WebFlowPhase::Committed)
            }],
        );
    }
    let output = controller.tick(input(16_021), std::iter::empty());
    assert!(output.intent.is_none());
    assert_eq!(output.closed[0].closure, WebFlowClosure::Expired);
}

#[test]
fn constraints_remove_actuation_but_keep_numeric_observation() {
    for constrained in ["pressure", "thermal", "power", "sleep", "kill"] {
        let mut controller = WebFlowController::new(WebFlowRolloutPhase::Active);
        let mut tick = input(1_020);
        match constrained {
            "pressure" => tick.pressure_constrained = true,
            "thermal" => tick.thermal_constrained = true,
            "power" => tick.low_power = true,
            "sleep" => tick.sleeping = true,
            "kill" => tick.kill_switch = true,
            _ => unreachable!(),
        }
        let output = controller.tick(tick, [event(2, 3, 1, WebFlowPhase::Started)]);
        assert!(output.intent.is_none(), "{constrained}");
        assert_eq!(output.observation.accepted_events, 1, "{constrained}");
    }
}

#[test]
fn exact_extension_event_supersedes_inferred_event_for_same_navigation() {
    let mut controller = WebFlowController::new(WebFlowRolloutPhase::Active);
    let mut inferred = event(2, 3, 1, WebFlowPhase::Started);
    inferred.event.source = WebFlowSource::DaemonInference;
    controller.tick(input(1_020), [inferred]);
    let exact = controller.tick(input(1_040), [event(2, 3, 2, WebFlowPhase::Committed)]);
    assert_eq!(exact.observation.source, Some(WebFlowSource::ExtensionLifecycle));
    assert_eq!(exact.observation.active_navigations, 1);
}

#[test]
fn session_revision_change_invalidates_prior_navigation() {
    let mut controller = WebFlowController::new(WebFlowRolloutPhase::Active);
    controller.tick(input(1_020), [event(2, 3, 1, WebFlowPhase::Started)]);
    let mut next = input(1_040);
    next.session_revision = 2;
    let output = controller.tick(next, std::iter::empty());
    assert!(output.intent.is_none());
    assert_eq!(output.closed[0].closure, WebFlowClosure::Invalidated);
}
