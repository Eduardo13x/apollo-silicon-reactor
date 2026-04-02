use serde::{Deserialize, Serialize};

use crate::engine::llm::{LearnedPolicy, LlmSuggestion};

/// Wire protocol version.  Bump when adding variants that older clients/daemons
/// cannot understand.  Both apollo-optimizerd and apollo-optimizerctl expose
/// this at runtime so a version mismatch can be reported cleanly.
pub const PROTOCOL_VERSION: u32 = 1;
use crate::engine::types::{
    BlockerScore, CapabilityReport, DaemonStatus, LatencyTarget, LlmStatus, OptimizationProfile,
    ProfileTransition, RuntimeMetrics, UsageResponse,
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum DaemonRequest {
    GetStatus,
    GetMetrics,
    GetTopBlockers,
    GetCapabilities,
    SetProfile {
        profile: OptimizationProfile,
        ttl_minutes: Option<u64>,
    },
    SetLatencyTarget {
        target: LatencyTarget,
    },
    SetAutoProfile {
        enabled: bool,
    },
    ClearProfileOverride,
    GetProfileTimeline,
    Restore,
    PanicRestore,
    Doctor,
    GetLlmStatus,
    GetLearnedPolicy,
    LlmSetKey {
        api_key: String,
        ttl_days: u64,
    },
    LlmDisable,
    LlmTest,
    UsageTop {
        limit: Option<usize>,
    },
    UsageExplain {
        name: String,
    },
    Feedback {
        rating: String,
        note: Option<String>,
    },
    SetLearnedPolicy {
        policy: LearnedPolicy,
    },
    GetSysctlGovernor,
    /// Suscripcion push: el daemon enviara StatusPush en cada ciclo de optimizacion.
    /// La conexion se mantiene abierta indefinidamente.
    Subscribe,
    /// Returns protocol version and build string for compatibility checks.
    GetVersion,
}

impl DaemonRequest {
    pub fn is_privileged(&self) -> bool {
        match self {
            Self::GetStatus
            | Self::GetMetrics
            | Self::GetTopBlockers
            | Self::GetCapabilities
            | Self::GetProfileTimeline
            | Self::Doctor
            | Self::GetLlmStatus
            | Self::UsageTop { .. }
            | Self::UsageExplain { .. }
            | Self::GetLearnedPolicy
            | Self::GetSysctlGovernor
            | Self::Subscribe
            | Self::GetVersion => false,

            Self::SetProfile { .. }
            | Self::SetLatencyTarget { .. }
            | Self::SetAutoProfile { .. }
            | Self::ClearProfileOverride
            | Self::Restore
            | Self::PanicRestore
            | Self::LlmSetKey { .. }
            | Self::LlmDisable
            | Self::LlmTest
            | Self::Feedback { .. }
            | Self::SetLearnedPolicy { .. } => true,
        }
    }

    pub fn sanitize(&mut self) {
        match self {
            Self::LlmSetKey { api_key, .. } => {
                if api_key.len() > 1024 {
                    api_key.truncate(1024);
                }
            }
            Self::UsageExplain { name } => {
                if name.len() > 256 {
                    name.truncate(256);
                }
            }
            Self::Feedback { rating, note } => {
                if rating.len() > 32 {
                    rating.truncate(32);
                }
                if let Some(n) = note {
                    if n.len() > 1024 {
                        n.truncate(1024);
                    }
                }
            }
            _ => {}
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
#[allow(clippy::large_enum_variant)]
pub enum DaemonResponse {
    Ok,
    Status(DaemonStatus),
    Metrics(RuntimeMetrics),
    TopBlockers(Vec<BlockerScore>),
    ProfileTimeline(Vec<ProfileTransition>),
    Capabilities(CapabilityReport),
    Doctor {
        checks: Vec<String>,
    },
    LlmStatus(LlmStatus),
    LearnedPolicy(LearnedPolicy),
    LlmTestResult {
        ok: bool,
        http_status: Option<u16>,
        error: Option<String>,
        suggestion: Option<LlmSuggestion>,
    },
    Usage(UsageResponse),
    SysctlGovernor(crate::engine::sysctl_governor::SysctlGovernorStatus),
    /// Evento push enviado por el daemon a los suscriptores en cada ciclo.
    StatusPush(DaemonStatus),
    /// Response to GetVersion.
    VersionInfo {
        protocol: u32,
        build: String,
    },
    Error {
        message: String,
    },
}
