//! Platform boundary for capability discovery and future event/actuator adapters.

use super::capability_graph::{
    CapabilityGraph, CapabilityId, CapabilityNode, CapabilityState, CapacityUnit,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformProbe {
    pub logical_cpus: u32,
    pub performance_cpus: Option<u32>,
    pub efficiency_cpus: Option<u32>,
    pub memory_bytes: Option<u64>,
    pub gpu_compute: bool,
    pub core_ml: bool,
    pub unified_memory: bool,
    pub is_root: bool,
    pub task_policy: bool,
    pub sysctl: bool,
    pub memorystatus: bool,
    pub memory_pressure_send: bool,
    pub spotlight: bool,
    pub time_machine: bool,
}

impl PlatformProbe {
    fn nodes(&self) -> Vec<CapabilityNode> {
        let cpu = |id, value: Option<u32>| match value.filter(|value| *value > 0) {
            Some(value) => CapabilityNode::capacity(
                id,
                CapabilityState::Verified,
                Some(value as u64),
                CapacityUnit::Workers,
            ),
            None => CapabilityNode::unavailable(id),
        };
        let memory = match (self.unified_memory, self.memory_bytes) {
            (true, Some(bytes)) if bytes > 0 => CapabilityNode::capacity(
                CapabilityId::UnifiedMemory,
                CapabilityState::Detected,
                Some(bytes),
                CapacityUnit::Bytes,
            ),
            _ => CapabilityNode::unavailable(CapabilityId::UnifiedMemory),
        };
        vec![
            cpu(CapabilityId::CpuInteractive, self.performance_cpus),
            cpu(CapabilityId::CpuUtility, self.efficiency_cpus),
            CapabilityNode::boolean(CapabilityId::GpuCompute, self.gpu_compute),
            CapabilityNode::boolean(CapabilityId::MlInference, self.core_ml),
            memory,
            CapabilityNode::boolean(CapabilityId::SensorPressure, self.sysctl),
            CapabilityNode::boolean(CapabilityId::SensorThermal, cfg!(target_os = "macos")),
            CapabilityNode::boolean(CapabilityId::SensorPower, cfg!(target_os = "macos")),
            CapabilityNode::unavailable(CapabilityId::SensorAudioActivity),
            CapabilityNode::unavailable(CapabilityId::SensorVisualActivity),
            CapabilityNode::boolean(CapabilityId::ActuatorTaskPolicy, self.task_policy),
            CapabilityNode::boolean(CapabilityId::ActuatorSysctl, self.sysctl),
            CapabilityNode::boolean(CapabilityId::ActuatorMemoryStatus, self.memorystatus),
            CapabilityNode::boolean(
                CapabilityId::ActuatorMemoryPressureSend,
                self.memory_pressure_send,
            ),
            CapabilityNode::boolean(CapabilityId::ActuatorSpotlight, self.spotlight),
            CapabilityNode::boolean(CapabilityId::ActuatorTimeMachine, self.time_machine),
            CapabilityNode::boolean(CapabilityId::PrivilegeRoot, self.is_root),
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformSubscription {
    Lifecycle,
    Pressure,
    Thermal,
    Power,
    Process,
    Session,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlatformSample {
    pub pressure: Option<f32>,
    pub thermal: Option<f32>,
    pub power_watts: Option<f32>,
}

pub trait PlatformAdapter {
    fn probe(&mut self) -> CapabilityGraph;
    fn subscribe(&self) -> Vec<PlatformSubscription>;
    fn sample_fallback(&mut self) -> PlatformSample;
    fn actuators(&self) -> Vec<CapabilityId>;
}

#[derive(Debug, Clone)]
pub struct SimulatedPlatformAdapter {
    platform: String,
    probe: PlatformProbe,
    last_probe: Option<PlatformProbe>,
    revision: u64,
}

impl SimulatedPlatformAdapter {
    pub fn new(platform: impl Into<String>, probe: PlatformProbe) -> Self {
        Self {
            platform: platform.into(),
            probe,
            last_probe: None,
            revision: 0,
        }
    }

    pub fn set_probe(&mut self, probe: PlatformProbe) {
        self.probe = probe;
    }
}

impl PlatformAdapter for SimulatedPlatformAdapter {
    fn probe(&mut self) -> CapabilityGraph {
        if self.last_probe.as_ref() != Some(&self.probe) {
            self.revision = self.revision.saturating_add(1).max(1);
            self.last_probe = Some(self.probe.clone());
        }
        CapabilityGraph::try_new(self.revision, &self.platform, self.probe.nodes())
            .expect("fixed platform probe must fit the capability graph")
    }

    fn subscribe(&self) -> Vec<PlatformSubscription> {
        vec![
            PlatformSubscription::Lifecycle,
            PlatformSubscription::Pressure,
        ]
    }

    fn sample_fallback(&mut self) -> PlatformSample {
        PlatformSample::default()
    }

    fn actuators(&self) -> Vec<CapabilityId> {
        self.probe
            .nodes()
            .into_iter()
            .filter(|node| {
                matches!(
                    node.id,
                    CapabilityId::ActuatorTaskPolicy
                        | CapabilityId::ActuatorSysctl
                        | CapabilityId::ActuatorMemoryStatus
                        | CapabilityId::ActuatorMemoryPressureSend
                        | CapabilityId::ActuatorSpotlight
                        | CapabilityId::ActuatorTimeMachine
                ) && node.state.usable()
            })
            .map(|node| node.id)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct MacOsPlatformAdapter {
    last_probe: Option<PlatformProbe>,
    revision: u64,
}

impl MacOsPlatformAdapter {
    pub fn detect() -> Self {
        Self {
            last_probe: None,
            revision: 0,
        }
    }

    fn detect_probe() -> PlatformProbe {
        let read_u32 = |name| super::sysctl_direct::read_u32_val(name);
        let p = read_u32("hw.perflevel0.logicalcpu");
        let e = read_u32("hw.perflevel1.logicalcpu");
        let logical = read_u32("hw.logicalcpu").unwrap_or_else(|| p.unwrap_or(0) + e.unwrap_or(0));
        let memory_bytes = super::sysctl_direct::read_u64("hw.memsize");
        let is_root = unsafe { libc::geteuid() == 0 };
        let macos = cfg!(target_os = "macos");
        PlatformProbe {
            logical_cpus: logical,
            performance_cpus: p,
            efficiency_cpus: e,
            memory_bytes,
            gpu_compute: macos
                && std::path::Path::new("/System/Library/Frameworks/Metal.framework").exists(),
            core_ml: macos
                && std::path::Path::new("/System/Library/Frameworks/CoreML.framework").exists(),
            unified_memory: cfg!(all(target_os = "macos", target_arch = "aarch64")),
            is_root,
            task_policy: macos,
            sysctl: super::sysctl_direct::exists("kern.ostype"),
            memorystatus: is_root && macos,
            memory_pressure_send: is_root
                && super::sysctl_direct::exists("kern.memorystatus_vm_pressure_send"),
            spotlight: std::path::Path::new("/usr/bin/mdutil").exists(),
            time_machine: std::path::Path::new("/usr/bin/tmutil").exists(),
        }
    }
}

impl PlatformAdapter for MacOsPlatformAdapter {
    fn probe(&mut self) -> CapabilityGraph {
        let probe = Self::detect_probe();
        if self.last_probe.as_ref() != Some(&probe) {
            self.revision = self.revision.saturating_add(1).max(1);
            self.last_probe = Some(probe.clone());
        }
        CapabilityGraph::try_new(
            self.revision,
            if cfg!(target_os = "macos") {
                "macos"
            } else {
                "unsupported"
            },
            probe.nodes(),
        )
        .expect("fixed macOS probe must fit the capability graph")
    }
    fn subscribe(&self) -> Vec<PlatformSubscription> {
        if cfg!(target_os = "macos") {
            vec![
                PlatformSubscription::Lifecycle,
                PlatformSubscription::Pressure,
                PlatformSubscription::Thermal,
                PlatformSubscription::Power,
                PlatformSubscription::Process,
                PlatformSubscription::Session,
            ]
        } else {
            Vec::new()
        }
    }
    fn sample_fallback(&mut self) -> PlatformSample {
        PlatformSample::default()
    }
    fn actuators(&self) -> Vec<CapabilityId> {
        self.last_probe
            .clone()
            .unwrap_or_else(Self::detect_probe)
            .nodes()
            .into_iter()
            .filter(|node| {
                matches!(
                    node.id,
                    CapabilityId::ActuatorTaskPolicy
                        | CapabilityId::ActuatorSysctl
                        | CapabilityId::ActuatorMemoryStatus
                        | CapabilityId::ActuatorMemoryPressureSend
                        | CapabilityId::ActuatorSpotlight
                        | CapabilityId::ActuatorTimeMachine
                ) && node.state.usable()
            })
            .map(|node| node.id)
            .collect()
    }
}
