//! Action surface — `RootAction` + `SetSysctlAction`.
//!
//! Extracted from `types.rs` (graphify C0 trace, 2026-06-11): the action
//! surface was 90 of the 206 nodes gluing the otherwise-separate executor,
//! journal, io_tiering, and chromium-tick modules into one low-cohesion
//! (0.01) community. Moving it here gives the action vocabulary its own
//! module; `types.rs` re-exports everything, so call sites are unchanged.

use serde::{Deserialize, Serialize};

/// Sealed payload for `RootAction::SetSysctl`.
///
/// Sprint 4 Phase 4 (2026-05-07) — fields are private so the only path to
/// construction is `SetSysctlAction::new_clamped`, which routes every
/// proposed value through `sysctl_limits::clamp_to_allowed_range`. This
/// closes the Bug 6 regression class: external emit sites (e.g.
/// `network-optimizer` at `main.rs:3577`) can no longer struct-literal-bypass
/// the safety clamp — type-system enforcement, not convention.
///
/// Cross-crate visibility: type appears in the `RootAction::SetSysctl(SetSysctlAction)`
/// variant which is `pub` and used across the workspace. Bins match on `RootAction::SetSysctl`
/// and call the pub accessor methods. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
///
/// JSON serialization shape is preserved by serde's externally-tagged enum
/// default + a newtype-variant: a previous `RootAction::SetSysctl { key,
/// value, reason, decision_reason }` and the new `RootAction::SetSysctl(
/// SetSysctlAction { key, value, reason, decision_reason })` both serialize
/// to `{"SetSysctl": {"key": ..., "value": ..., "reason": ...,
/// "decision_reason": ...}}`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SetSysctlAction {
    key: String,
    value: String,
    reason: String,
    #[serde(default = "default_decision_reason")]
    decision_reason: crate::engine::audit_types::DecisionReason,
}

impl SetSysctlAction {
    /// Only public constructor for sysctl actions. Parses `value` as `i64`
    /// and clamps to the
    /// `safety::allowlisted_sysctls_with_ranges()` entry for `key`.
    /// Non-numeric values pass through unchanged. Non-allowlist keys pass
    /// through unchanged (execute_actions rejects them with
    /// `BlockReason::InvalidSysctl` — defense in depth).
    ///
    /// Emits a `tracing::debug!` only when the clamp actually changed the
    /// value (no log spam on the common no-op path). When `clamped ==
    /// proposed` this method is silent, and `execute_actions` separately
    /// skips no-op writes when the kernel already reports the same value
    /// (see `BlockReason` early-return at the `RootAction::SetSysctl` arm).
    pub fn new_clamped(
        key: impl Into<String>,
        value: impl Into<String>,
        reason: impl Into<String>,
        decision_reason: crate::engine::audit_types::DecisionReason,
    ) -> Self {
        let key = key.into();
        let value = value.into();
        let clamped = match value.parse::<i64>() {
            Ok(n) => {
                let limited = crate::engine::sysctl_limits::clamp_to_allowed_range(&key, n);
                if limited != n {
                    tracing::debug!(
                        target: "apollo.sysctl",
                        key = %key,
                        proposed = n,
                        clamped = limited,
                        "SetSysctlAction clamped to allowed range"
                    );
                    limited.to_string()
                } else {
                    value
                }
            }
            Err(_) => value,
        };
        Self {
            key,
            value: clamped,
            reason: reason.into(),
            decision_reason,
        }
    }

    pub fn key(&self) -> &str {
        &self.key
    }
    pub fn value(&self) -> &str {
        &self.value
    }
    pub fn reason(&self) -> &str {
        &self.reason
    }
    pub fn decision_reason(&self) -> &crate::engine::audit_types::DecisionReason {
        &self.decision_reason
    }
}

/// Cross-crate visibility: the primary action type dispatched by apollo-optimizerd across
/// daemon_dispatch_tick, daemon_agent_actions, learning_tick, and daemon_skill_tick.
/// Central to the workspace — removing pub would break the entire action pipeline.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RootAction {
    BoostProcess {
        pid: u32,
        name: String,
        reason: String,
        #[serde(default = "default_decision_reason")]
        decision_reason: crate::engine::audit_types::DecisionReason,
        /// Kernel start-time for PID identity validation (prevents A-B-A recycling).
        /// Sprint Inv#11 (2026-06-06): closes the no-op tautology at
        /// `execute_actions.rs` Boost arm (was verifying with `0,0` legacy fallback).
        #[serde(default)]
        start_sec: u64,
        #[serde(default)]
        start_usec: u64,
    },
    ThrottleProcess {
        pid: u32,
        name: String,
        aggressive: bool,
        reason: String,
        #[serde(default = "default_decision_reason")]
        decision_reason: crate::engine::audit_types::DecisionReason,
        /// Kernel start-time for PID identity validation (prevents A-B-A recycling).
        #[serde(default)]
        start_sec: u64,
        #[serde(default)]
        start_usec: u64,
    },
    FreezeProcess {
        pid: u32,
        name: String,
        reason: String,
        #[serde(default = "default_decision_reason")]
        decision_reason: crate::engine::audit_types::DecisionReason,
        /// Kernel start-time for PID identity validation (prevents A-B-A recycling).
        #[serde(default)]
        start_sec: u64,
        #[serde(default)]
        start_usec: u64,
    },
    UnfreezeProcess {
        pid: u32,
        name: String,
        #[serde(default)]
        reason: String,
        #[serde(default = "default_decision_reason")]
        decision_reason: crate::engine::audit_types::DecisionReason,
    },
    /// Sealed sysctl action — the only construction path is
    /// `SetSysctlAction::new_clamped(...)` (or its public delegate
    /// `RootAction::set_sysctl`), which routes the proposed value through
    /// `sysctl_limits::clamp_to_allowed_range` automatically.
    ///
    /// Sprint 4 Phase 4 (2026-05-07): wrapped in a private-fielded struct to
    /// prevent regressions of Bug 6 (`network-optimizer` emitting raw
    /// 4 MB buffers without clamping). Future emit sites that try to
    /// struct-literal-construct a `SetSysctlAction` will fail to compile
    /// outside `engine::types`, forcing them through the clamping factory.
    SetSysctl(SetSysctlAction),
    SetMemorystatus {
        pid: u32,
        priority: i32,
        reason: String,
        #[serde(default = "default_decision_reason")]
        decision_reason: crate::engine::audit_types::DecisionReason,
    },
    ToggleSpotlight {
        enabled: bool,
        reason: String,
        #[serde(default = "default_decision_reason")]
        decision_reason: crate::engine::audit_types::DecisionReason,
    },
    QuarantineDaemon {
        daemon: String,
        active: bool,
        reason: String,
        #[serde(default = "default_decision_reason")]
        decision_reason: crate::engine::audit_types::DecisionReason,
    },
    /// Per-thread QoS: route a specific thread to P-core or E-core.
    SetThreadQoS {
        pid: u32,
        name: String,
        thread_index: u32,
        /// "interactive", "background", or "utility"
        tier: String,
        reason: String,
        #[serde(default = "default_decision_reason")]
        decision_reason: crate::engine::audit_types::DecisionReason,
        /// Optional cluster-affinity hint (Phase B 2026-05-06).
        /// Some(1) → P-cluster preference (Firestorm/Avalanche)
        /// Some(2) → E-cluster preference (Icestorm/Blizzard)
        /// Some(0) or None → no hint, kernel default scheduling
        ///
        /// Heterogeneous-hardware-only: only emitted when CapabilityReport
        /// reports both p_core_count AND e_core_count Some(>0).
        /// [ARM big.LITTLE 2013 §3] thread-level affinity reduces migration
        /// cost when threads cooperate on shared data within a cluster.
        #[serde(default)]
        affinity_tag: Option<u32>,
        /// Kernel start-time for PID identity validation (prevents A-B-A recycling).
        /// Sprint Inv#11 (2026-06-06): closes the no-op tautology at the
        /// `execute_actions.rs` SetThreadQoS arm.
        #[serde(default)]
        start_sec: u64,
        #[serde(default)]
        start_usec: u64,
    },
}

fn default_decision_reason() -> crate::engine::audit_types::DecisionReason {
    crate::engine::audit_types::DecisionReason::PressureContext
}

impl RootAction {
    /// One-word action-class label used by the audit `Rationale` and any
    /// dashboard that aggregates by action kind. Static-string return so
    /// the rationale builder pays zero allocation per journal line.
    ///
    /// Phase 5.3 wiring (2026-05-16): consumed at the cycle-wide journal
    /// chokepoint in `execute_actions.rs` to label each `JournalEntry`.
    pub fn action_class(&self) -> &'static str {
        match self {
            RootAction::BoostProcess { .. } => "boost",
            RootAction::ThrottleProcess { .. } => "throttle",
            RootAction::FreezeProcess { .. } => "freeze",
            RootAction::UnfreezeProcess { .. } => "unfreeze",
            RootAction::SetSysctl(_) => "set_sysctl",
            RootAction::SetMemorystatus { .. } => "set_memorystatus",
            RootAction::ToggleSpotlight { .. } => "toggle_spotlight",
            RootAction::QuarantineDaemon { .. } => "quarantine_daemon",
            RootAction::SetThreadQoS { .. } => "set_thread_qos",
        }
    }

    /// Borrow the `DecisionReason` carried by any variant. Single-source
    /// accessor consumed by the Phase 5.3 rationale builder so the trigger
    /// label always reflects the actual decision context (Specialist,
    /// Rule, Skill, etc.) rather than a re-derived guess.
    pub fn decision_reason(&self) -> &crate::engine::audit_types::DecisionReason {
        use RootAction::*;
        match self {
            BoostProcess {
                decision_reason, ..
            }
            | ThrottleProcess {
                decision_reason, ..
            }
            | FreezeProcess {
                decision_reason, ..
            }
            | UnfreezeProcess {
                decision_reason, ..
            }
            | SetMemorystatus {
                decision_reason, ..
            }
            | ToggleSpotlight {
                decision_reason, ..
            }
            | QuarantineDaemon {
                decision_reason, ..
            }
            | SetThreadQoS {
                decision_reason, ..
            } => decision_reason,
            SetSysctl(action) => action.decision_reason(),
        }
    }

    /// The variant's free-text `reason` field. For Phase 5.3 this becomes
    /// the rationale's `evidence` payload — the concrete signal values
    /// that justified the action.
    pub fn reason(&self) -> &str {
        use RootAction::*;
        match self {
            BoostProcess { reason, .. }
            | ThrottleProcess { reason, .. }
            | FreezeProcess { reason, .. }
            | UnfreezeProcess { reason, .. }
            | SetMemorystatus { reason, .. }
            | ToggleSpotlight { reason, .. }
            | QuarantineDaemon { reason, .. }
            | SetThreadQoS { reason, .. } => reason,
            SetSysctl(action) => action.reason(),
        }
    }

    /// Extract the identity tuple `(pid, name, start_sec, start_usec)` from
    /// any PID-bearing variant. Returns `None` for actions that do not act on
    /// a process (`SetSysctl`, `ToggleSpotlight`, `QuarantineDaemon`).
    ///
    /// Variants without process birth timestamps (`BoostProcess`,
    /// `UnfreezeProcess`, `SetMemorystatus`, `SetThreadQoS`) report
    /// `start_sec = start_usec = 0` — see `ProcessIdentity::matches` for the
    /// legacy-fallback semantics. `SetMemorystatus` carries no name field in
    /// the action variant and reports `name = None`.
    ///
    /// Single source of truth for the field-extraction logic that previously
    /// duplicated between `daemon main.rs::pid_identity_still_valid` and the
    /// callers of `execute_actions::verify_pid_identity` (Sprint 4 merge).
    pub fn identity_fields(&self) -> Option<(u32, Option<&str>, u64, u64)> {
        match self {
            RootAction::ThrottleProcess {
                pid,
                name,
                start_sec,
                start_usec,
                ..
            }
            | RootAction::FreezeProcess {
                pid,
                name,
                start_sec,
                start_usec,
                ..
            } => Some((*pid, Some(name.as_str()), *start_sec, *start_usec)),
            RootAction::BoostProcess {
                pid,
                name,
                start_sec,
                start_usec,
                ..
            }
            | RootAction::SetThreadQoS {
                pid,
                name,
                start_sec,
                start_usec,
                ..
            } => Some((*pid, Some(name.as_str()), *start_sec, *start_usec)),
            RootAction::UnfreezeProcess { pid, name, .. } => {
                Some((*pid, Some(name.as_str()), 0, 0))
            }
            RootAction::SetMemorystatus { pid, .. } => Some((*pid, None, 0, 0)),
            RootAction::SetSysctl(_)
            | RootAction::ToggleSpotlight { .. }
            | RootAction::QuarantineDaemon { .. } => None,
        }
    }

    /// Construct a `ThrottleProcess` with zero start times.
    ///
    /// Use this when the action is queued before PID identity validation
    /// (start times are filled in by `execute_actions` at dispatch time).
    pub fn throttle(
        pid: u32,
        name: impl Into<String>,
        aggressive: bool,
        reason: impl Into<String>,
        decision_reason: crate::engine::audit_types::DecisionReason,
    ) -> Self {
        Self::throttle_full(pid, name, aggressive, reason, 0, 0, decision_reason)
    }

    pub fn throttle_full(
        pid: u32,
        name: impl Into<String>,
        aggressive: bool,
        reason: impl Into<String>,
        start_sec: u64,
        start_usec: u64,
        decision_reason: crate::engine::audit_types::DecisionReason,
    ) -> Self {
        RootAction::ThrottleProcess {
            pid,
            name: name.into(),
            aggressive,
            reason: reason.into(),
            decision_reason,
            start_sec,
            start_usec,
        }
    }

    pub fn freeze(
        pid: u32,
        name: impl Into<String>,
        reason: impl Into<String>,
        decision_reason: crate::engine::audit_types::DecisionReason,
    ) -> Self {
        Self::freeze_full(pid, name, reason, 0, 0, decision_reason)
    }

    pub fn freeze_full(
        pid: u32,
        name: impl Into<String>,
        reason: impl Into<String>,
        start_sec: u64,
        start_usec: u64,
        decision_reason: crate::engine::audit_types::DecisionReason,
    ) -> Self {
        RootAction::FreezeProcess {
            pid,
            name: name.into(),
            reason: reason.into(),
            decision_reason,
            start_sec,
            start_usec,
        }
    }

    /// Build a `RootAction::SetSysctl` with automatic value clamping.
    ///
    /// This is the only public construction path for sysctl actions
    /// (Sprint 4 Phase 4 seal). The proposed `value` is parsed as `i64` and
    /// clamped to the safety allowlist range for `key`; non-numeric or
    /// non-allowlist values pass through unchanged so `execute_actions`
    /// can reject them with `BlockReason::InvalidSysctl` (defense in
    /// depth).
    pub fn set_sysctl(
        key: impl Into<String>,
        value: impl Into<String>,
        reason: impl Into<String>,
        decision_reason: crate::engine::audit_types::DecisionReason,
    ) -> Self {
        RootAction::SetSysctl(SetSysctlAction::new_clamped(
            key,
            value,
            reason,
            decision_reason,
        ))
    }

    pub fn set_memorystatus(
        pid: u32,
        priority: i32,
        reason: impl Into<String>,
        decision_reason: crate::engine::audit_types::DecisionReason,
    ) -> Self {
        RootAction::SetMemorystatus {
            pid,
            priority,
            reason: reason.into(),
            decision_reason,
        }
    }

    pub fn toggle_spotlight(
        enabled: bool,
        reason: impl Into<String>,
        decision_reason: crate::engine::audit_types::DecisionReason,
    ) -> Self {
        RootAction::ToggleSpotlight {
            enabled,
            reason: reason.into(),
            decision_reason,
        }
    }

    pub fn unfreeze(
        pid: u32,
        name: impl Into<String>,
        reason: impl Into<String>,
        decision_reason: crate::engine::audit_types::DecisionReason,
    ) -> Self {
        RootAction::UnfreezeProcess {
            pid,
            name: name.into(),
            reason: reason.into(),
            decision_reason,
        }
    }
}

#[cfg(test)]
mod extraction_tests {
    /// Pin the extraction contract (2026-06-11): the old `types::RootAction`
    /// path and the new `action_types::RootAction` are the SAME type (the
    /// re-export compiles this assignment), and the serde wire shape is
    /// byte-identical to pre-extraction (journal.jsonl compatibility).
    #[test]
    fn reexport_is_same_type_and_serde_shape_unchanged() {
        let a: crate::engine::types::RootAction =
            crate::engine::action_types::RootAction::UnfreezeProcess {
                pid: 42,
                name: "x".into(),
                reason: "r".into(),
                decision_reason: crate::engine::audit_types::DecisionReason::PressureContext,
            };
        let json = serde_json::to_string(&a).expect("serialize");
        assert!(
            json.starts_with("{\"UnfreezeProcess\":{"),
            "wire shape changed: {json}"
        );
        let back: crate::engine::types::RootAction =
            serde_json::from_str(&json).expect("roundtrip");
        assert_eq!(format!("{back:?}"), format!("{a:?}"));
    }
}
