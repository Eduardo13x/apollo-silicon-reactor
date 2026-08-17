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
            // Core ML loads and returns correct output, so the capability is
            // real and stays usable. It is Degraded rather than Detected
            // because it has not earned promotion on the model Apollo actually
            // ships: a bounded probe measured the lane at ~400x the
            // deterministic CPU oracle that produces the same four outputs to
            // 1e-4, and Core ML publishes no per-inference dispatch target, so
            // no accelerator use can be confirmed either.
            //
            // Degraded is the honest middle: the lane remains available for a
            // model large enough to amortise dispatch overhead, while the graph
            // records that on a 16-feature model it demonstrated no benefit.
            // Promotion back to Detected belongs to new evidence, not to a
            // larger request.
            CapabilityNode {
                id: CapabilityId::MlInference,
                state: if self.core_ml {
                    CapabilityState::Degraded
                } else {
                    CapabilityState::Unavailable
                },
                capacity: if self.core_ml { Some(1) } else { None },
                unit: CapacityUnit::Boolean,
            },
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

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(core_ml: bool) -> PlatformProbe {
        PlatformProbe {
            logical_cpus: 8,
            performance_cpus: Some(4),
            efficiency_cpus: Some(4),
            memory_bytes: Some(8 * 1024 * 1024 * 1024),
            gpu_compute: true,
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

    /// The Core ML lane stays usable but must not present itself as a
    /// promoted capability.
    ///
    /// It produces the same four outputs as `cpu_oracle_predict` (pinned to
    /// 1e-4 by `configured_coreml_model_matches_the_cpu_oracle`) at roughly
    /// 400x the latency, measured by the bounded probe in
    /// `tests/coreml_backend_probe.rs`, and Core ML publishes no
    /// per-inference dispatch target, so no accelerator use can be confirmed.
    ///
    /// Degraded records that without disabling the lane: a larger future model
    /// may well amortise the dispatch cost. Flipping this back to Detected is
    /// a claim that new evidence exists, so it should require editing this
    /// test.
    #[test]
    fn core_ml_is_usable_but_not_promoted_while_it_shows_no_benefit() {
        let nodes = probe(true).nodes();
        let ml = nodes
            .iter()
            .find(|node| node.id == CapabilityId::MlInference)
            .expect("the graph always carries an ML inference node");

        assert_eq!(ml.state, CapabilityState::Degraded);
        assert!(
            ml.state.usable(),
            "degraded must keep the lane selectable, not disable it"
        );
    }

    #[test]
    fn absent_core_ml_is_unavailable_rather_than_degraded() {
        let nodes = probe(false).nodes();
        let ml = nodes
            .iter()
            .find(|node| node.id == CapabilityId::MlInference)
            .expect("the graph always carries an ML inference node");

        assert_eq!(ml.state, CapabilityState::Unavailable);
        assert!(!ml.state.usable());
    }
}
