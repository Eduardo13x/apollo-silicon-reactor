//! Versioned, platform-neutral description of runtime capabilities.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::types::CapabilityReport;

pub const CAPABILITY_GRAPH_SCHEMA_VERSION: u16 = 1;
pub const MAX_CAPABILITIES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CapabilityId {
    CpuInteractive,
    CpuUtility,
    GpuCompute,
    MlInference,
    UnifiedMemory,
    SensorPressure,
    SensorThermal,
    SensorPower,
    SensorAudioActivity,
    SensorVisualActivity,
    ActuatorTaskPolicy,
    ActuatorSysctl,
    ActuatorMemoryStatus,
    ActuatorMemoryPressureSend,
    ActuatorSpotlight,
    ActuatorTimeMachine,
    PrivilegeRoot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CapabilityState {
    Unsupported,
    Unavailable,
    PermissionDenied,
    Detected,
    Verified,
    Degraded,
}

impl CapabilityState {
    pub const fn usable(self) -> bool {
        matches!(self, Self::Detected | Self::Verified | Self::Degraded)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CapacityUnit {
    Workers,
    Bytes,
    Boolean,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityNode {
    pub id: CapabilityId,
    pub state: CapabilityState,
    pub capacity: Option<u64>,
    pub unit: CapacityUnit,
}

impl CapabilityNode {
    pub const fn unavailable(id: CapabilityId) -> Self {
        Self {
            id,
            state: CapabilityState::Unavailable,
            capacity: None,
            unit: CapacityUnit::Boolean,
        }
    }

    pub const fn boolean(id: CapabilityId, available: bool) -> Self {
        Self {
            id,
            state: if available {
                CapabilityState::Detected
            } else {
                CapabilityState::Unavailable
            },
            capacity: if available { Some(1) } else { None },
            unit: CapacityUnit::Boolean,
        }
    }

    pub const fn capacity(
        id: CapabilityId,
        state: CapabilityState,
        value: Option<u64>,
        unit: CapacityUnit,
    ) -> Self {
        Self {
            id,
            state,
            capacity: value,
            unit,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ComputeClass {
    CpuInteractive,
    CpuUtility,
    Metal,
    CoreMl,
}

impl ComputeClass {
    const fn capability(self) -> CapabilityId {
        match self {
            Self::CpuInteractive => CapabilityId::CpuInteractive,
            Self::CpuUtility => CapabilityId::CpuUtility,
            Self::Metal => CapabilityId::GpuCompute,
            Self::CoreMl => CapabilityId::MlInference,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityGraph {
    pub schema_version: u16,
    pub revision: u64,
    pub platform: String,
    nodes: Vec<CapabilityNode>,
}

impl CapabilityGraph {
    pub fn try_new(
        revision: u64,
        platform: impl Into<String>,
        mut nodes: Vec<CapabilityNode>,
    ) -> Result<Self, &'static str> {
        if revision == 0 {
            return Err("capability revision must be nonzero");
        }
        if nodes.len() > MAX_CAPABILITIES {
            return Err("capability graph exceeds fixed capacity");
        }
        nodes.sort_by_key(|node| node.id);
        let unique = nodes.iter().map(|node| node.id).collect::<BTreeSet<_>>();
        if unique.len() != nodes.len() {
            return Err("duplicate capability id");
        }
        Ok(Self {
            schema_version: CAPABILITY_GRAPH_SCHEMA_VERSION,
            revision,
            platform: platform.into().chars().take(32).collect(),
            nodes,
        })
    }

    pub fn node(&self, id: CapabilityId) -> Option<&CapabilityNode> {
        self.nodes
            .binary_search_by_key(&id, |node| node.id)
            .ok()
            .map(|index| &self.nodes[index])
    }

    pub fn state(&self, id: CapabilityId) -> CapabilityState {
        self.node(id)
            .map(|node| node.state)
            .unwrap_or(CapabilityState::Unavailable)
    }

    pub fn supports_compute(&self, class: ComputeClass) -> bool {
        self.state(class.capability()).usable()
    }

    pub fn compute_classes(&self) -> Vec<ComputeClass> {
        [
            ComputeClass::CpuInteractive,
            ComputeClass::CpuUtility,
            ComputeClass::Metal,
            ComputeClass::CoreMl,
        ]
        .into_iter()
        .filter(|class| self.supports_compute(*class))
        .collect()
    }

    pub fn recommended_cpu_workers(&self) -> usize {
        self.node(CapabilityId::CpuInteractive)
            .and_then(|node| node.capacity)
            .unwrap_or(1)
            .clamp(1, 4) as usize
    }

    pub fn legacy_report(&self) -> CapabilityReport {
        let available = |id| self.state(id).usable();
        let count = |id| {
            self.node(id)
                .and_then(|node| node.capacity)
                .and_then(|value| u32::try_from(value).ok())
        };
        let mut unavailable = Vec::new();
        for (id, name) in [
            (CapabilityId::ActuatorTaskPolicy, "taskpolicy"),
            (CapabilityId::ActuatorSysctl, "sysctl"),
            (CapabilityId::ActuatorMemoryStatus, "memorystatus"),
            (CapabilityId::ActuatorSpotlight, "mdutil"),
            (CapabilityId::ActuatorTimeMachine, "tmutil"),
        ] {
            if !available(id) {
                unavailable.push(name.to_string());
            }
        }
        CapabilityReport {
            can_taskpolicy: available(CapabilityId::ActuatorTaskPolicy),
            can_sysctl: available(CapabilityId::ActuatorSysctl),
            can_memorystatus: available(CapabilityId::ActuatorMemoryStatus),
            can_memory_pressure_send: available(CapabilityId::ActuatorMemoryPressureSend),
            can_mdutil: available(CapabilityId::ActuatorSpotlight),
            can_tmutil: available(CapabilityId::ActuatorTimeMachine),
            is_root: available(CapabilityId::PrivilegeRoot),
            p_core_count: count(CapabilityId::CpuInteractive),
            e_core_count: count(CapabilityId::CpuUtility),
            unavailable,
            memorystatus_probe: None,
            task_for_pid_probe: None,
        }
    }
}
