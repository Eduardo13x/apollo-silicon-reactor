use apollo_engine::engine::network_flow::{
    NetworkFlowController, NetworkFlowTickInput, GENERIC_FLOW_HARD_CAP_MS,
    GENERIC_FLOW_INITIAL_TTL_MS,
};
use apollo_engine::engine::network_monitor::TcpStats;
use std::time::Duration;

fn qualifying(now_ms: u64) -> NetworkFlowTickInput {
    NetworkFlowTickInput {
        now_ms,
        session_revision: 7,
        foreground_pid: Some(42),
        identity_available: true,
        interaction_active: true,
        foreground_socket_active: true,
        tcp_sample_age_ms: 100,
        send_bps: 16 * 1024,
        recv_bps: 96 * 1024,
        new_connections: 1,
        exact_web_active: false,
        pressure_constrained: false,
        thermal_constrained: false,
        low_power: false,
        sleeping: false,
        kill_switch: false,
    }
}

#[test]
fn any_foreground_app_can_produce_a_short_inferred_intent() {
    let mut controller = NetworkFlowController::new();
    let output = controller.tick(qualifying(1_000));
    let intent = output.intent.expect("qualifying generic flow");
    assert_eq!(intent.target_pid, 42);
    assert_eq!(intent.ttl_ms, GENERIC_FLOW_INITIAL_TTL_MS as u32);
    assert!(output.observation.active);
    assert!(output.observation.inferred);
}

#[test]
fn sockets_traffic_and_interaction_are_all_required() {
    let mut controller = NetworkFlowController::new();
    let mut sockets_only = qualifying(1_000);
    sockets_only.send_bps = 0;
    sockets_only.recv_bps = 0;
    sockets_only.new_connections = 0;
    assert!(controller.tick(sockets_only).intent.is_none());

    let mut traffic_without_interaction = qualifying(2_000);
    traffic_without_interaction.interaction_active = false;
    assert!(controller
        .tick(traffic_without_interaction)
        .intent
        .is_none());

    let mut traffic_without_foreground_socket = qualifying(3_000);
    traffic_without_foreground_socket.foreground_socket_active = false;
    assert!(controller
        .tick(traffic_without_foreground_socket)
        .intent
        .is_none());
}

#[test]
fn exact_web_evidence_suppresses_generic_intent() {
    let mut controller = NetworkFlowController::new();
    assert!(controller.tick(qualifying(1_000)).intent.is_some());
    let mut exact = qualifying(1_100);
    exact.exact_web_active = true;
    let output = controller.tick(exact);
    assert!(output.intent.is_none());
    assert!(!output.observation.active);
    assert_eq!(output.counters.suppressed_exact, 1);
}

#[test]
fn stale_tcp_sample_never_starts_or_renews_a_flow() {
    let mut controller = NetworkFlowController::new();
    let mut stale = qualifying(1_000);
    stale.tcp_sample_age_ms = 2_001;
    assert!(controller.tick(stale).intent.is_none());
}

#[test]
fn constraints_and_session_changes_close_active_flow() {
    for mutate in [
        |input: &mut NetworkFlowTickInput| input.pressure_constrained = true,
        |input: &mut NetworkFlowTickInput| input.thermal_constrained = true,
        |input: &mut NetworkFlowTickInput| input.low_power = true,
        |input: &mut NetworkFlowTickInput| input.sleeping = true,
        |input: &mut NetworkFlowTickInput| input.kill_switch = true,
    ] {
        let mut controller = NetworkFlowController::new();
        assert!(controller.tick(qualifying(1_000)).intent.is_some());
        let mut blocked = qualifying(1_100);
        mutate(&mut blocked);
        let output = controller.tick(blocked);
        assert!(output.intent.is_none());
        assert!(!output.observation.active);
    }

    let mut controller = NetworkFlowController::new();
    controller.tick(qualifying(1_000));
    let mut changed = qualifying(1_100);
    changed.session_revision = 8;
    assert!(controller.tick(changed).intent.is_none());
}

#[test]
fn continuous_generic_flow_hits_hard_cap_and_cooldown() {
    let mut controller = NetworkFlowController::new();
    assert!(controller.tick(qualifying(1_000)).intent.is_some());
    let capped_at = 1_000 + GENERIC_FLOW_HARD_CAP_MS;
    let output = controller.tick(qualifying(capped_at));
    assert!(output.intent.is_none());
    assert_eq!(output.counters.hard_cap_expirations, 1);
    assert!(controller.tick(qualifying(capped_at + 1)).intent.is_none());
}

#[test]
fn foreground_identity_change_starts_a_new_bounded_episode() {
    let mut controller = NetworkFlowController::new();
    controller.tick(qualifying(1_000));
    let mut other = qualifying(1_100);
    other.foreground_pid = Some(99);
    let output = controller.tick(other);
    assert_eq!(output.intent.expect("new target").target_pid, 99);
    assert_eq!(output.counters.target_changes, 1);
}

#[test]
fn tcp_delta_exposes_bounded_per_second_rates_without_another_probe() {
    let stats = TcpStats {
        bytes_sent: 40_000,
        bytes_recv: 120_000,
        connections: 3,
        elapsed: Duration::from_millis(500),
        ..TcpStats::default()
    };
    let sample = stats.flow_sample();
    assert_eq!(sample.send_bps, 80_000);
    assert_eq!(sample.recv_bps, 240_000);
    assert_eq!(sample.new_connections, 3);

    let zero_elapsed = TcpStats {
        elapsed: Duration::ZERO,
        ..stats
    };
    assert_eq!(zero_elapsed.flow_sample().send_bps, 0);
}
