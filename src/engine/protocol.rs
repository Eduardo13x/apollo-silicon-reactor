use serde::{Deserialize, Serialize};

use crate::engine::llm::{LearnedPolicy, LlmSuggestion};
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
    Error {
        message: String,
    },
}
