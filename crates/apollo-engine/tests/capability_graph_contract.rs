use apollo_engine::engine::capability_graph::{
    CapabilityGraph, CapabilityId, CapabilityNode, CapabilityState, ComputeClass, MAX_CAPABILITIES,
};
use apollo_engine::engine::platform::{PlatformAdapter, PlatformProbe, SimulatedPlatformAdapter};

fn probe(p_cores: u32, e_cores: u32, gpu: bool, core_ml: bool) -> PlatformProbe {
    PlatformProbe {
        logical_cpus: p_cores + e_cores,
        performance_cpus: Some(p_cores),
        efficiency_cpus: Some(e_cores),
        memory_bytes: Some(16 * 1024 * 1024 * 1024),
        gpu_compute: gpu,
        core_ml,
        unified_memory: true,
        is_root: true,
        task_policy: true,
        sysctl: true,
        memorystatus: true,
        memory_pressure_send: false,
        spotlight: true,
        time_machine: true,
    }
}

#[test]
fn capability_policy_depends_on_resources_not_chip_name() {
    let mut current = SimulatedPlatformAdapter::new("current", probe(4, 6, true, true));
    let mut future = SimulatedPlatformAdapter::new("future-unknown", probe(4, 6, true, true));

    let current_graph = current.probe();
    let future_graph = future.probe();

    assert_eq!(
        current_graph.compute_classes(),
        future_graph.compute_classes()
    );
    assert_eq!(current_graph.recommended_cpu_workers(), 4);
    assert_eq!(future_graph.recommended_cpu_workers(), 4);
}

#[test]
fn no_accelerator_fixture_keeps_cpu_and_marks_accelerators_unavailable() {
    let mut adapter = SimulatedPlatformAdapter::new("portable", probe(4, 4, false, false));
    let graph = adapter.probe();

    assert!(graph.supports_compute(ComputeClass::CpuInteractive));
    assert!(graph.supports_compute(ComputeClass::CpuUtility));
    assert!(!graph.supports_compute(ComputeClass::Metal));
    assert!(!graph.supports_compute(ComputeClass::CoreMl));
    assert_eq!(
        graph.state(CapabilityId::GpuCompute),
        CapabilityState::Unavailable
    );
}

#[test]
fn unknown_capacity_is_not_projected_as_zero_or_available() {
    let mut fixture = probe(4, 6, true, true);
    fixture.memory_bytes = None;
    let mut adapter = SimulatedPlatformAdapter::new("unknown-memory", fixture);
    let graph = adapter.probe();

    let memory = graph.node(CapabilityId::UnifiedMemory).unwrap();
    assert_eq!(memory.state, CapabilityState::Unavailable);
    assert_eq!(memory.capacity, None);
}

#[test]
fn reprobe_increments_revision_only_for_semantic_change() {
    let mut adapter = SimulatedPlatformAdapter::new("fixture", probe(4, 6, true, true));
    let first = adapter.probe();
    let identical = adapter.probe();
    adapter.set_probe(probe(4, 6, false, true));
    let changed = adapter.probe();

    assert_eq!(first.revision, identical.revision);
    assert!(changed.revision > identical.revision);
    assert_eq!(
        changed.state(CapabilityId::GpuCompute),
        CapabilityState::Unavailable
    );
}

#[test]
fn legacy_projection_preserves_existing_capability_report_contract() {
    let mut adapter = SimulatedPlatformAdapter::new("fixture", probe(4, 6, true, true));
    let legacy = adapter.probe().legacy_report();

    assert!(legacy.can_taskpolicy);
    assert!(legacy.can_sysctl);
    assert!(legacy.can_memorystatus);
    assert_eq!(legacy.p_core_count, Some(4));
    assert_eq!(legacy.e_core_count, Some(6));
    assert!(legacy.memorystatus_probe.is_none());
    assert!(legacy.task_for_pid_probe.is_none());
}

#[test]
fn legacy_json_deserializes_without_graph_fields() {
    let old = r#"{
        "can_taskpolicy":true,"can_sysctl":true,"can_memorystatus":false,
        "can_memory_pressure_send":false,"can_mdutil":true,"can_tmutil":true,
        "is_root":false,"p_core_count":4,"e_core_count":4,"unavailable":[],
        "memorystatus_probe":null,"task_for_pid_probe":null
    }"#;
    let report: apollo_engine::engine::types::CapabilityReport = serde_json::from_str(old).unwrap();
    assert_eq!(report.p_core_count, Some(4));
}

#[test]
fn graph_enforces_a_fixed_capability_bound() {
    let node = CapabilityNode::unavailable(CapabilityId::SensorPressure);
    let oversized = vec![node; MAX_CAPABILITIES + 1];
    assert!(CapabilityGraph::try_new(1, "simulated", oversized).is_err());
}

#[test]
fn serialized_graph_has_a_version_and_no_chip_name_authority() {
    let mut adapter = SimulatedPlatformAdapter::new("future-unknown", probe(8, 8, true, true));
    let encoded = serde_json::to_value(adapter.probe()).unwrap();
    assert_eq!(encoded["schema_version"], 1);
    assert!(encoded.get("chip_name").is_none());
}
