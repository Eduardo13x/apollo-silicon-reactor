use serde::{Deserialize, Serialize};

use crate::engine::llm::{LearnedPolicy, LlmSuggestion};

/// Wire protocol version.  Bump when adding variants that older clients/daemons
/// cannot understand.  Both apollo-optimizerd and apollo-optimizerctl expose
/// this at runtime so a version mismatch can be reported cleanly.
///
/// Cross-crate visibility: read by apollo-optimizerctl to detect daemon version mismatches.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
pub const PROTOCOL_VERSION: u32 = 1;
use crate::engine::types::{
    BlockerScore, CapabilityReport, DaemonStatus, HealthReport, LatencyTarget, LlmStatus,
    OptimizationProfile, ProfileTransition, RuntimeMetrics, UsageResponse,
};

/// IPC request type.
///
/// Cross-crate visibility: all bins that communicate with the daemon (apollo-optimizerctl,
/// apollo-menubar, apollo-optimizerd socket_handler) construct and match on this type.
/// Must remain `pub`. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
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
    /// Revert all sysctl changes made by the daemon to their startup defaults.
    RevertSysctls,
    /// Trigger an immediate maintenance purge through the daemon.
    /// Subject to MaintenanceState rate-limits (5 min CLI + 1 min auto spacing).
    Purge,
    /// Suscripcion push: el daemon enviara StatusPush en cada ciclo de optimizacion.
    /// La conexion se mantiene abierta indefinidamente.
    Subscribe,
    /// Returns protocol version and build string for compatibility checks.
    GetVersion,
    /// Returns circuit breaker and degradation health summary.
    GetHealth,
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
            | Self::GetVersion
            | Self::GetHealth => false,

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
            | Self::SetLearnedPolicy { .. }
            | Self::RevertSysctls
            | Self::Purge => true,
        }
    }

    pub fn sanitize(&mut self) {
        match self {
            Self::LlmSetKey { api_key, .. }
                if api_key.len() > 1024 => {
                    api_key.truncate(1024);
                }
            Self::UsageExplain { name }
                if name.len() > 256 => {
                    name.truncate(256);
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

/// IPC response type.
///
/// Cross-crate visibility: all IPC clients (apollo-optimizerctl, apollo-menubar) match on
/// variants of this type; socket_handler.rs in apollo-optimizerd constructs them.
/// Must remain `pub`. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
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
    /// Response to GetHealth.
    Health(HealthReport),
    PurgeResult {
        fired: bool,
        reason: String,
    },
    Error {
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Serde roundtrip helpers ───────────────────────────────────────────────

    fn roundtrip(req: &DaemonRequest) -> DaemonRequest {
        let json = serde_json::to_string(req).expect("serialize DaemonRequest");
        serde_json::from_str(&json).expect("deserialize DaemonRequest")
    }

    // ── Roundtrip tests ───────────────────────────────────────────────────────

    #[test]
    fn roundtrip_get_status() {
        let rt = roundtrip(&DaemonRequest::GetStatus);
        assert!(matches!(rt, DaemonRequest::GetStatus));
    }

    #[test]
    fn roundtrip_get_metrics() {
        let rt = roundtrip(&DaemonRequest::GetMetrics);
        assert!(matches!(rt, DaemonRequest::GetMetrics));
    }

    #[test]
    fn roundtrip_subscribe() {
        let rt = roundtrip(&DaemonRequest::Subscribe);
        assert!(matches!(rt, DaemonRequest::Subscribe));
    }

    #[test]
    fn roundtrip_get_version() {
        let rt = roundtrip(&DaemonRequest::GetVersion);
        assert!(matches!(rt, DaemonRequest::GetVersion));
    }

    #[test]
    fn roundtrip_set_profile_fields() {
        let req = DaemonRequest::SetProfile {
            profile: OptimizationProfile::BalancedRoot,
            ttl_minutes: None,
        };
        let rt = roundtrip(&req);
        match rt {
            DaemonRequest::SetProfile {
                profile,
                ttl_minutes,
            } => {
                assert_eq!(profile, OptimizationProfile::BalancedRoot);
                assert_eq!(ttl_minutes, None);
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }

    #[test]
    fn roundtrip_llm_set_key_fields() {
        let req = DaemonRequest::LlmSetKey {
            api_key: "sk-test".to_string(),
            ttl_days: 7,
        };
        let rt = roundtrip(&req);
        match rt {
            DaemonRequest::LlmSetKey { api_key, ttl_days } => {
                assert_eq!(api_key, "sk-test");
                assert_eq!(ttl_days, 7);
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }

    #[test]
    fn roundtrip_usage_explain_fields() {
        let req = DaemonRequest::UsageExplain {
            name: "Brave".to_string(),
        };
        let rt = roundtrip(&req);
        match rt {
            DaemonRequest::UsageExplain { name } => assert_eq!(name, "Brave"),
            other => panic!("unexpected variant: {:?}", other),
        }
    }

    #[test]
    fn roundtrip_feedback_fields() {
        let req = DaemonRequest::Feedback {
            rating: "good".to_string(),
            note: None,
        };
        let rt = roundtrip(&req);
        match rt {
            DaemonRequest::Feedback { rating, note } => {
                assert_eq!(rating, "good");
                assert_eq!(note, None);
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }

    // ── is_privileged tests ───────────────────────────────────────────────────

    #[test]
    fn not_privileged_get_status() {
        assert!(!DaemonRequest::GetStatus.is_privileged());
    }

    #[test]
    fn not_privileged_get_metrics() {
        assert!(!DaemonRequest::GetMetrics.is_privileged());
    }

    #[test]
    fn not_privileged_get_version() {
        assert!(!DaemonRequest::GetVersion.is_privileged());
    }

    #[test]
    fn privileged_restore() {
        assert!(DaemonRequest::Restore.is_privileged());
    }

    #[test]
    fn privileged_llm_disable() {
        assert!(DaemonRequest::LlmDisable.is_privileged());
    }

    #[test]
    fn privileged_panic_restore() {
        assert!(DaemonRequest::PanicRestore.is_privileged());
    }

    #[test]
    fn privileged_set_profile() {
        let req = DaemonRequest::SetProfile {
            profile: OptimizationProfile::BalancedRoot,
            ttl_minutes: None,
        };
        assert!(req.is_privileged());
    }

    // ── sanitize tests ────────────────────────────────────────────────────────

    #[test]
    fn sanitize_llm_set_key_does_not_panic() {
        let mut req = DaemonRequest::LlmSetKey {
            api_key: "sk-test".to_string(),
            ttl_days: 7,
        };
        req.sanitize();
        match req {
            DaemonRequest::LlmSetKey { api_key, .. } => {
                assert!(!api_key.is_empty());
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }

    #[test]
    fn sanitize_truncates_overlong_api_key() {
        let long_key = "x".repeat(2000);
        let mut req = DaemonRequest::LlmSetKey {
            api_key: long_key,
            ttl_days: 1,
        };
        req.sanitize();
        match req {
            DaemonRequest::LlmSetKey { api_key, .. } => {
                assert!(api_key.len() <= 1024);
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }

    // ── PROTOCOL_VERSION test ─────────────────────────────────────────────────

    #[test]
    fn protocol_version_is_positive() {
        assert!(PROTOCOL_VERSION > 0);
    }

    #[test]
    fn roundtrip_purge() {
        let rt = roundtrip(&DaemonRequest::Purge);
        assert!(matches!(rt, DaemonRequest::Purge));
    }

    #[test]
    fn purge_is_privileged() {
        assert!(DaemonRequest::Purge.is_privileged());
    }
}
