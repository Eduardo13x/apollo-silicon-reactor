use serde::{Deserialize, Serialize};

use crate::engine::context_agent::ContextSummary;
use crate::engine::policy_store::LearnedPolicy;
use crate::engine::webflow_types::WebFlowEvent;

/// Wire protocol version.  Bump when adding variants that older clients/daemons
/// cannot understand.  Both apollo-optimizerd and apollo-optimizerctl expose
/// this at runtime so a version mismatch can be reported cleanly.
///
/// Cross-crate visibility: read by apollo-optimizerctl to detect daemon version mismatches.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
pub const PROTOCOL_VERSION: u32 = 4;
use crate::engine::types::{
    BlockerScore, CapabilityReport, DaemonStatus, HealthReport, LatencyTarget, OptimizationProfile,
    ProfileTransition, RuntimeMetrics, UsageResponse,
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
    GetLearnedPolicy,
    UsageTop {
        limit: Option<usize>,
    },
    UsageExplain {
        name: String,
    },
    /// Turn the micro-canary on or off. Mutating: enabling it makes the daemon
    /// start withholding real pre-warms, so it needs the same authority as a
    /// profile change and is never available to a read-only client.
    SetCanaryEnabled {
        enabled: bool,
    },
    Feedback {
        rating: String,
        note: Option<String>,
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
    /// Submit a bounded, numeric-only user-session context summary.
    /// Authentication is intentionally handled separately by the daemon's
    /// peer-UID integration; this request is not generally privileged.
    SubmitContext {
        summary: ContextSummary,
    },
    /// Submit one validated, content-free browser navigation observation.
    /// The authenticated context agent is the only accepted production peer.
    SubmitWebFlow {
        event: WebFlowEvent,
    },
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
            | Self::UsageTop { .. }
            | Self::UsageExplain { .. }
            | Self::GetLearnedPolicy
            | Self::GetSysctlGovernor
            | Self::Subscribe
            | Self::GetVersion
            | Self::GetHealth
            | Self::SubmitContext { .. }
            | Self::SubmitWebFlow { .. } => false,

            Self::SetProfile { .. }
            | Self::SetLatencyTarget { .. }
            | Self::SetAutoProfile { .. }
            | Self::ClearProfileOverride
            | Self::Restore
            | Self::PanicRestore
            | Self::Feedback { .. }
            | Self::SetCanaryEnabled { .. }
            | Self::RevertSysctls
            | Self::Purge => true,
        }
    }

    pub fn sanitize(&mut self) {
        match self {
            Self::UsageExplain { name } if name.len() > 256 => {
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
    LearnedPolicy(LearnedPolicy),
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
    #[test]
    fn enabling_the_canary_requires_the_same_authority_as_a_profile_change() {
        // Enabling makes the daemon withhold real pre-warms. A read-only client
        // must never be able to start an intervention.
        assert!(DaemonRequest::SetCanaryEnabled { enabled: true }.is_privileged());
        assert!(DaemonRequest::SetCanaryEnabled { enabled: false }.is_privileged());
        assert!(!DaemonRequest::GetMetrics.is_privileged());
    }

    use super::*;
    use crate::engine::webflow_types::{
        OpaqueId, WebFlowEvent, WebFlowMetrics, WebFlowPhase, WebFlowSource, WEBFLOW_SCHEMA_VERSION,
    };

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
    fn submit_context_is_not_privileged() {
        let request = DaemonRequest::SubmitContext {
            summary: ContextSummary::default(),
        };
        assert!(!request.is_privileged());
        let rt = roundtrip(&request);
        assert!(matches!(rt, DaemonRequest::SubmitContext { .. }));
    }

    #[test]
    fn submit_webflow_is_not_privileged_and_roundtrips() {
        let request = DaemonRequest::SubmitWebFlow {
            event: WebFlowEvent {
                schema_version: WEBFLOW_SCHEMA_VERSION,
                browser_session_id: OpaqueId::new([1; 16]).unwrap(),
                tab_session_id: OpaqueId::new([2; 16]).unwrap(),
                navigation_id: OpaqueId::new([3; 16]).unwrap(),
                sequence: 1,
                phase: WebFlowPhase::Started,
                source: WebFlowSource::ExtensionLifecycle,
                site_bucket: None,
                metrics: WebFlowMetrics::default(),
                transport: Default::default(),
                producer_kind: Default::default(),
                extension_version: None,
                bridge_version: None,
                feature_capabilities: Default::default(),
            },
        };
        assert!(!request.is_privileged());
        assert!(matches!(
            roundtrip(&request),
            DaemonRequest::SubmitWebFlow { .. }
        ));
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

    // ── PROTOCOL_VERSION test ─────────────────────────────────────────────────

    #[test]
    fn protocol_version_is_positive() {
        assert!(PROTOCOL_VERSION > 0);
    }

    #[test]
    fn retired_teacher_requests_are_rejected_by_protocol_v2() {
        for request in [
            r#"{"type":"GetLlmStatus"}"#,
            r#"{"type":"LlmDisable"}"#,
            r#"{"type":"LlmTest"}"#,
            r#"{"type":"LlmSetKey","payload":{"api_key":"secret","ttl_days":7}}"#,
            r#"{"type":"SetLearnedPolicy","payload":{"policy":{}}}"#,
        ] {
            assert!(serde_json::from_str::<DaemonRequest>(request).is_err());
        }
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
