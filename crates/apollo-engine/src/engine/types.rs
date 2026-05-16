use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use crate::engine::usage_model::{UsageEntrySummary, UsageTopReport};

/// Centralized utility for hardened file system operations to prevent TOCTOU and symlink attacks.
///
/// Cross-crate visibility: required by apollo-optimizerd (main.rs, llm_daemon.rs) and
/// apollo-optimizerctl for secure path validation. Audited 2026-05-09 during Sprint 5
/// Mes 0 workspace split.
pub struct HardPath;

impl HardPath {
    /// Verifies that a path exists and is NOT a symlink.
    /// Returns Ok(()) if it's a real file/dir, or an Err if it's a symlink or missing.
    pub fn verify_no_symlink(path: &Path) -> anyhow::Result<()> {
        if !path.exists() {
            return Ok(());
        }
        let meta = fs::symlink_metadata(path)?;
        if meta.file_type().is_symlink() {
            // macOS standard system symlinks are allowed
            #[cfg(target_os = "macos")]
            {
                if path == Path::new("/var")
                    || path == Path::new("/etc")
                    || path == Path::new("/tmp")
                {
                    return Ok(());
                }
            }
            anyhow::bail!("security violation: path {} is a symlink", path.display());
        }
        Ok(())
    }

    /// Recursively creates directories while ensuring no component is a symlink.
    pub fn secure_create_dir_all(path: &Path) -> anyhow::Result<()> {
        let mut current = PathBuf::new();
        for component in path.components() {
            current.push(component);
            if current.exists() {
                let meta = fs::symlink_metadata(&current)?;
                if meta.file_type().is_symlink() {
                    // macOS standard system symlinks are allowed
                    #[cfg(target_os = "macos")]
                    {
                        if current == Path::new("/var")
                            || current == Path::new("/etc")
                            || current == Path::new("/tmp")
                        {
                            continue;
                        }
                    }
                    anyhow::bail!(
                        "security violation: path component {} is a symlink",
                        current.display()
                    );
                }
            } else {
                fs::create_dir(&current)?;
            }
        }
        Ok(())
    }

    /// Reads a file to string with a maximum byte limit.
    pub fn read_to_string_limited(path: &Path, max_bytes: u64) -> anyhow::Result<String> {
        use std::io::Read;
        let file = fs::File::open(path)?;
        let mut reader = file.take(max_bytes);
        let mut string = String::new();
        reader.read_to_string(&mut string)?;
        Ok(string)
    }
}

/// Cross-crate visibility: used by apollo-optimizerctl (profile set commands),
/// apollo-menubar (profile display and switching), and apollo-optimizerd main loop.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizationProfile {
    BalancedRoot,
    AggressiveRoot,
    SafeRoot,
}

impl Default for OptimizationProfile {
    fn default() -> Self {
        Self::BalancedRoot
    }
}

impl OptimizationProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BalancedRoot => "balanced-root",
            Self::AggressiveRoot => "aggressive-root",
            Self::SafeRoot => "safe-root",
        }
    }
}

/// Cross-crate visibility: used by apollo-optimizerctl (latency target commands) and
/// apollo-optimizerd main loop for per-cycle latency tuning decisions.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum LatencyTarget {
    Low,
    Normal,
    Max,
}

impl Default for LatencyTarget {
    fn default() -> Self {
        Self::Normal
    }
}

/// Cross-crate visibility: used by apollo-menubar (policy display), apollo-optimizerd
/// (process_enrichment.rs, main.rs safety validation). Audited 2026-05-09 during Sprint 5
/// Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyPolicy {
    pub max_boosts_per_cycle: usize,
    pub max_throttles_per_cycle: usize,
    pub max_paging_hints_per_cycle: usize,
    pub max_freezes_per_cycle: usize,
    pub max_sysctl_writes_per_cycle: usize,
    pub cooldown_seconds: u64,
    /// Maximum per-thread QoS changes per cycle (Phase 1: thread-level scheduling).
    #[serde(default = "default_thread_qos")]
    pub max_thread_qos_per_cycle: usize,
}

fn default_thread_qos() -> usize {
    10
}

impl SafetyPolicy {
    /// Returns a `SafetyPolicy` scaled to the available hardware.
    ///
    /// On M1 8 GB (cores=8, ram=8): `core_scale=1.0`, `ram_scale=1.0` — identical
    /// to the per-profile defaults.  On a 16-core 32 GB machine: `core_scale=2.0`,
    /// `ram_scale=4.0`, so budgets scale up proportionally.
    ///
    /// Call after `for_profile()` to apply hardware-aware scaling:
    /// ```ignore
    /// let base = SafetyPolicy::for_profile(profile);
    /// let scaled = SafetyPolicy::for_capabilities(base, cores, ram_gb);
    /// ```
    pub fn for_capabilities(base: Self, cores: u32, ram_gb: f64) -> Self {
        let core_scale = (cores as f64 / 8.0).clamp(0.5, 2.0);
        let ram_scale = (ram_gb / 8.0).clamp(0.5, 4.0);
        Self {
            max_boosts_per_cycle: ((base.max_boosts_per_cycle as f64 * core_scale).round()
                as usize)
                .max(1),
            max_throttles_per_cycle: ((base.max_throttles_per_cycle as f64 * core_scale).round()
                as usize)
                .max(1),
            max_paging_hints_per_cycle: ((base.max_paging_hints_per_cycle as f64 * ram_scale)
                .round() as usize)
                .max(1),
            max_freezes_per_cycle: ((base.max_freezes_per_cycle as f64 * core_scale).round()
                as usize)
                .max(1),
            max_sysctl_writes_per_cycle: ((base.max_sysctl_writes_per_cycle as f64 * core_scale)
                .round() as usize)
                .max(1),
            cooldown_seconds: base.cooldown_seconds,
            max_thread_qos_per_cycle: ((base.max_thread_qos_per_cycle as f64 * core_scale).round()
                as usize)
                .max(1),
        }
    }

    pub fn for_profile(profile: OptimizationProfile) -> Self {
        match profile {
            OptimizationProfile::AggressiveRoot => Self {
                max_boosts_per_cycle: 10,
                max_throttles_per_cycle: 20,
                max_paging_hints_per_cycle: 12,
                max_freezes_per_cycle: 8,
                max_sysctl_writes_per_cycle: 8,
                cooldown_seconds: 10,
                max_thread_qos_per_cycle: 20,
            },
            OptimizationProfile::SafeRoot => Self {
                max_boosts_per_cycle: 3,
                max_throttles_per_cycle: 6,
                max_paging_hints_per_cycle: 3,
                max_freezes_per_cycle: 2,
                max_sysctl_writes_per_cycle: 2,
                cooldown_seconds: 45,
                max_thread_qos_per_cycle: 4,
            },
            OptimizationProfile::BalancedRoot => Self {
                max_boosts_per_cycle: 6,
                max_throttles_per_cycle: 12,
                max_paging_hints_per_cycle: 6,
                max_freezes_per_cycle: 4,
                max_sysctl_writes_per_cycle: 4,
                cooldown_seconds: 20,
                max_thread_qos_per_cycle: 10,
            },
        }
    }
}

/// Cross-crate visibility: used by apollo-optimizerd process_enrichment.rs to classify
/// per-process interaction context for decision routing. Audited 2026-05-09 during Sprint 5
/// Mes 0 workspace split.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum InteractiveContext {
    InteractiveFocus,
    BackgroundPressure,
    ThermalConstrained,
}

/// Cross-crate visibility: used by apollo-optimizerd metrics_reporter.rs to build
/// per-process blocker score reports. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockerScore {
    pub name: String,
    pub pid: u32,
    pub score: f64,
    pub blocker_cpu_spike: f32,
    pub interactive_wait_ratio: f64,
    pub blocker_seen_recently: bool,
    pub reactor_event_weight: f64,
}

/// Cross-crate visibility: required because `safety::enforce_limits_with_budget` is `pub`
/// and takes `budget: &mut ActionBudgetState`; apollo-optimizerd main.rs calls it directly.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActionBudgetState {
    pub cycle_boosts: usize,
    pub cycle_throttles: usize,
    pub cycle_hints: usize,
    pub cycle_freezes: usize,
    pub cycle_sysctl_writes: usize,
    pub minute_actions: usize,
    pub boost_denied_cooldown: usize,
    #[serde(default)]
    pub cycle_thread_qos: usize,
}

/// Cross-crate visibility: used by apollo-optimizerd daemon_dispatch_tick.rs tests and
/// main.rs for capability-based decision gating. Audited 2026-05-09 during Sprint 5
/// Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityReport {
    pub can_taskpolicy: bool,
    pub can_sysctl: bool,
    pub can_memorystatus: bool,
    pub can_mdutil: bool,
    pub can_tmutil: bool,
    pub is_root: bool,
    pub p_core_count: Option<u32>,
    pub e_core_count: Option<u32>,
    pub unavailable: Vec<String>,
}

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
    },
}

fn default_decision_reason() -> crate::engine::audit_types::DecisionReason {
    crate::engine::audit_types::DecisionReason::PressureContext
}

impl RootAction {
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
            RootAction::BoostProcess { pid, name, .. }
            | RootAction::UnfreezeProcess { pid, name, .. }
            | RootAction::SetThreadQoS { pid, name, .. } => Some((*pid, Some(name.as_str()), 0, 0)),
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

/// Cross-crate visibility: used by apollo-optimizerd (daemon_chromium_tick, daemon_thermal_freeze,
/// daemon_dispatch_tick, daemon_turbo_manager) to tag the origin of freeze decisions.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum FreezeSource {
    MainLoop,
    Sentinel,
    Manual,
    /// Frozen by thermal pre-throttle (Phase3Aggressive ≥90°C).
    ThermalPreThrottle,
    /// Frozen by ChromiumManager (idle tab renderer — safe to SIGCONT on fg change).
    ChromiumManager,
    /// Unknown variant from future binary — treat as MainLoop for safe recovery.
    #[serde(other)]
    Unknown,
}

/// Cross-crate visibility: used by apollo-optimizerd (daemon_chromium_tick, daemon_thermal_freeze,
/// daemon_dispatch_tick, daemon_turbo_manager) as the value type in the frozen-state map.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenEntry {
    pub frozen_at: DateTime<Utc>,
    pub source: FreezeSource,
    /// Memory pressure at the time this process was frozen.
    /// Used by adaptive unfreeze: if pressure drops enough, unfreeze early.
    /// Defaults to 1.0 (max) for entries loaded from disk without this field,
    /// ensuring only the TTL path triggers for pre-existing freezes.
    #[serde(default = "frozen_entry_pressure_default")]
    pub pressure_at_freeze: f64,
    /// Process name at the time of freeze. Used on startup to detect PID reuse:
    /// if the current process at this PID has a different name, the PID was
    /// recycled and we skip SIGCONT (the original process is already gone).
    #[serde(default)]
    pub process_name: Option<String>,
    /// Kernel start-time (`pbi_start_tvsec`) captured at freeze time.
    ///
    /// A3 fix (round-3): prevents A-B-A PID recycling in the unfreeze pre-pass.
    /// On SIGCONT, the caller re-reads the current PID's start-time; if it no
    /// longer matches, the original process is gone and SIGCONT is skipped
    /// (the new process at that PID is not ours to resume).
    ///
    /// 0 = not captured (legacy entry or capture failed) → fall back to
    /// name-only check for backward compatibility.
    #[serde(default)]
    pub start_sec: u64,
    /// Jetsam priority captured at freeze time (snapshot of the value before
    /// we demoted to BACKGROUND).  A5/D1 fix (round-3): on unfreeze we
    /// restore this exact value instead of unconditionally setting
    /// Interactive, which previously lost AUDIO / AUDIO_AND_ACCESSORY /
    /// VITAL priorities.  `None` = unknown → leave jetsam untouched.
    #[serde(default)]
    pub original_jetsam_priority: Option<i32>,
}

fn frozen_entry_pressure_default() -> f64 {
    1.0
}

/// Summary of a currently frozen process, included in `DaemonStatus` for observability.
///
/// Cross-crate visibility: used by apollo-optimizerd socket_handler.rs to build the frozen
/// process list in `DaemonResponse::Status`. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenProcessInfo {
    pub pid: u32,
    pub name: String,
    pub frozen_seconds: u64,
    /// Which subsystem triggered the freeze.
    pub source: FreezeSource,
    /// Memory pressure when the freeze was applied.
    pub pressure_at_freeze: f64,
}

/// Cross-crate visibility: used by apollo-optimizerd learning_tick.rs to persist and
/// recover frozen-PID entries across daemon restarts. Audited 2026-05-09 during Sprint 5
/// Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenPidEntry {
    pub pid: u32,
    pub since: DateTime<Utc>,
    /// Process name captured at freeze time. Used on restart to detect PID reuse.
    #[serde(default)]
    pub name: Option<String>,
}

/// Cross-crate visibility: used by apollo-optimizerd learning_tick.rs to persist the full
/// frozen-state snapshot to disk for crash recovery. Audited 2026-05-09 during Sprint 5
/// Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenStatePersisted {
    pub frozen: Vec<FrozenPidEntry>,
}

// AUDIT-PENDING: JournalEntry is used only internally (journal.rs, execute_actions.rs).
// No bin imports it directly. However pub fn append_journal/read_journal in journal.rs
// reference it — those fns must be demoted to pub(crate) first before this can be
// tightened. Deferred to a follow-up sprint (cannot touch journal.rs in this commit).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    pub timestamp: DateTime<Utc>,
    pub action: RootAction,
    pub before: Option<String>,
    pub after: Option<String>,
    pub success: bool,
    pub reason: String,
}

/// Cross-crate visibility: embedded in `DaemonResponse::ProfileTimeline(Vec<ProfileTransition>)`.
/// apollo-optimizerctl deserializes and pretty-prints the timeline. Cannot be `pub(crate)` while
/// `DaemonResponse` remains `pub`. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileTransition {
    pub from: OptimizationProfile,
    pub to: OptimizationProfile,
    pub at: DateTime<Utc>,
    pub reason: String,
    pub pressure_score: f64,
}

/// Cross-crate visibility: exposed as `pub governor_state: GovernorState` on the `pub struct
/// ProfileGovernor` in profile_governor.rs, which is used cross-crate by daemon_memory_budget
/// and daemon_dispatch_tick. Cannot be `pub(crate)` while ProfileGovernor fields remain `pub`.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernorState {
    pub effective_profile: OptimizationProfile,
    pub cooldown_until: Option<DateTime<Utc>>,
    pub consecutive_high: u32,
    pub consecutive_low: u32,
}

/// Cross-crate visibility: exposed as `pub manual_override: Option<ManualOverride>` on the
/// `pub struct ProfileGovernor` in profile_governor.rs, which is used cross-crate.
/// Cannot be `pub(crate)` while ProfileGovernor fields remain `pub`.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualOverride {
    pub profile: OptimizationProfile,
    pub expires_at: DateTime<Utc>,
    pub reason: String,
}

/// Cross-crate visibility: used by apollo-optimizerd socket_handler.rs, metrics_reporter.rs,
/// and daemon_dispatch_tick.rs tests; also by apollo-menubar (indirectly via DaemonStatus).
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeMetrics {
    pub cycles: u64,
    pub boosts_applied: u64,
    pub throttles_applied: u64,
    pub freezes_applied: u64,
    pub unfreezes_applied: u64,
    pub paging_hints_applied: u64,
    pub sysctl_applied: u64,
    pub failures: u64,
    pub last_error: Option<String>,
    pub last_cycle_at: Option<DateTime<Utc>>,
    pub p95_cycle_ms: f64,
    pub refresh_duration_ms: f64,
    pub memory_budget_duration_ms: f64,
    pub reactor_duration_ms: f64,
    pub lock_wait_duration_ms: f64,
    pub throttle_reverted: u64,
    pub reactor_pulses: u64,
    pub effective_profile: OptimizationProfile,
    pub throttle_level: String,
    pub thermal_state: String,
    pub last_blockers: Vec<BlockerScore>,
    pub cycle_durations_ms: VecDeque<u64>,
    pub budgets: ActionBudgetState,
    pub profile_switches: u64,
    pub last_pressure_score: f64,
    pub override_activations: u64,
    pub override_expirations: u64,
    pub wake_events: u64,
    pub post_wake_grace_entries: u64,
    pub post_wake_defensive_unfreezes: u64,
    /// Teacher consolidation events (Gemma outcomes compiled into S1).
    #[serde(default)]
    pub teacher_consolidations: u64,
    /// Subset that IMPROVED pressure (positive valence).
    #[serde(default)]
    pub teacher_improvements: u64,
    pub post_wake_throttle_suppressed: u64,
    pub post_wake_freeze_suppressed: u64,
    pub swap_used_bytes: u64,
    /// Total swap capacity in bytes (dynamic on macOS).
    /// Zero only when sysctl read fails. Consumers computing swap ratios must
    /// guard against the zero case — divide-by-zero bugs here silently disable
    /// relative thresholds (see `safety::survival_mode_active_total`).
    #[serde(default)]
    pub swap_total_bytes: u64,
    pub swap_delta_bps: f64,
    pub memory_pressure: f64,
    /// Composite VM thrashing score from `VmRate::thrashing_score()`.
    /// 0 ≈ quiet, 5_000+ ≈ actively thrashing the compressor.
    #[serde(default)]
    pub thrashing_score: f64,
    /// System-wide CPU stall fraction from
    /// `ContentionTracker::stall_fraction(0.85)` — fraction of tracked pids
    /// whose PSI "some" contention ratio crossed 85% in the last cycle.
    /// Threshold chosen because Darwin's `ri_runnable_time` accumulates
    /// run-queue wait on every quantum and the baseline ratio is already
    /// ~0.7 under normal multitasking load.
    #[serde(default)]
    pub stall_fraction: f64,
    /// Mean per-core busy ratio across all online processors.
    #[serde(default)]
    pub cpu_mean_busy: f64,
    /// Hottest per-core busy ratio — "single saturated core" signal.
    #[serde(default)]
    pub cpu_max_busy: f64,
    /// Fraction of cores with busy ≥ 0.80 — pegged-core count / total.
    #[serde(default)]
    pub cpu_pegged_fraction: f64,
    pub thermal_level: String,
    pub invalid_sysctl_denied: u64,
    pub critical_background_skips: u64,
    pub critical_background_tree_protected: u64,
    pub reverts_applied: u64,
    pub reverts_failed: u64,
    pub reactor_events_total: u64,
    pub reactor_events_mem: u64,
    pub reactor_events_thermal: u64,
    pub reactor_events_spawn: u64,
    pub reactor_events_power: u64,
    pub reactor_last_event_at: Option<DateTime<Utc>>,
    pub reactor_last_error: Option<String>,
    pub reactor_mode: String,
    pub reactor_health: String,
    pub last_actions_summary: String,
    pub top_skipped_processes: Vec<String>,
    pub dev_session_active: bool,
    pub interactive_heavy: bool,
    pub profile_floor_hits: u64,
    /// Phase 3.1 — Skill-Aware Prediction tilt counter (Sprint 6).
    /// Cumulative number of non-Observe specialist votes whose confidence
    /// was multiplied by a non-neutral `skill_aware_factor`.
    #[serde(default)]
    pub skill_aware_modulations_total: u64,
    // Heuristic module metrics
    pub heuristic_decisions: u64,
    pub heuristic_throttles: u64,
    pub heuristic_freezes: u64,
    pub heuristic_kills_downgraded: u64,
    pub zombies_detected: u64,
    #[serde(default)]
    pub kills_applied: u64,
    /// Cumulative number of times survival mode was *entered* this session.
    /// Sticky counter — never decrements, even after recovery. Persists in
    /// JSON under the legacy key `survival_mode_activations` for backward
    /// compatibility (runtime_metrics.json, AIS rm_u lookups).
    ///
    /// **Do NOT use as a live state flag.** Today's value `> 0` only proves
    /// the daemon entered survival mode at least once since boot. For the
    /// current state, call
    /// [`crate::engine::safety::survival_mode_active_total`] which evaluates
    /// pressure + swap thresholds live.
    ///
    /// See `RuntimeMetrics::ever_entered_survival_mode` for the explicit
    /// "did we ever?" predicate, and `CLAUDE.md` for the bug history.
    #[serde(default, rename = "survival_mode_activations")]
    pub survival_mode_entry_count: u64,
    pub qos_foreground_count: u64,
    pub qos_background_count: u64,
    pub qos_errors: u64,
    pub iokit_snapshots: u64,
    pub iokit_errors: u64,
    pub iokit_p_cluster_temp: Option<f32>,
    pub iokit_e_cluster_temp: Option<f32>,
    pub iokit_package_watts: Option<f32>,
    pub current_workload: String,
    #[serde(default)]
    pub ml_confidence: f32, // 0.0–1.0; 0.0 until first classification
    #[serde(default)]
    pub ml_sources: Vec<String>, // evidence sources from last ML classification
    // Foreground app detection
    #[serde(default)]
    pub foreground_app: Option<ForegroundAppInfo>,
    #[serde(default)]
    pub foreground_idle: bool,
    /// Cambios de app en los últimos 5 minutos (detector TDA).
    #[serde(default)]
    pub context_switches_5min: u32,
    /// true cuando context_switches_5min >= 3 → perfil burst activo.
    #[serde(default)]
    pub context_switch_burst: bool,
    // Energy tracking
    #[serde(default)]
    pub energy_cpu_watts: Option<f64>,
    #[serde(default)]
    pub energy_gpu_watts: Option<f64>,
    #[serde(default)]
    pub energy_ane_watts: Option<f64>,
    #[serde(default)]
    pub energy_ane_util_pct: Option<f64>,
    #[serde(default)]
    pub energy_package_watts: Option<f64>,
    #[serde(default)]
    pub energy_session_wh: Option<f64>,
    #[serde(default)]
    pub energy_co2_avoided_g: Option<f64>,
    #[serde(default)]
    pub energy_savings_wh: Option<f64>,
    #[serde(default)]
    pub energy_top_consumers: Vec<EnergyConsumerInfo>,
    #[serde(default)]
    pub energy_package_wh: Option<f64>,
    // Process tree metrics
    #[serde(default)]
    pub process_tree_groups: usize,
    #[serde(default)]
    pub process_tree_total: usize,
    // SysctlGovernor metrics
    #[serde(default)]
    pub sysctl_reactive_writes: u64,
    #[serde(default)]
    pub sysctl_governor_active_tunings: usize,
    #[serde(default)]
    pub sysctl_governor_total_writes: u64,
    // Network monitor metrics
    #[serde(default)]
    pub network_retransmit_ratio: f64,
    #[serde(default)]
    pub network_listen_drop_rate: f64,
    // Resource interrupt (sentinel) metrics
    #[serde(default)]
    pub resource_interrupts_total: u64,
    #[serde(default)]
    pub resource_interrupt_last_phase: u8,
    #[serde(default)]
    pub resource_interrupt_active: bool,
    #[serde(default)]
    pub resource_interrupt_latency_us: u64,
    #[serde(default)]
    pub resource_interrupt_processes_frozen: u64,
    #[serde(default)]
    pub resource_interrupt_processes_migrated: u64,
    #[serde(default)]
    pub resource_interrupt_recovery_count: u64,
    // Hardening & collector health metrics
    #[serde(default)]
    pub collector_pressure_alive: bool,
    #[serde(default)]
    pub collector_smc_alive: bool,
    #[serde(default)]
    pub invalid_sysctl_value_denied: u64,
    #[serde(default)]
    pub journal_rotations: u64,
    // 2026-05-14: removed dead security counters (symlink_attacks_blocked,
    // policy_patterns_rejected, request_size_exceeded) — declared but no
    // writer in production code. `serde(default)` makes runtime_metrics.json
    // parsing from older daemon versions still work (extras ignored).
    // Overflow guard metrics
    #[serde(default)]
    pub overflow_events_total: u64,
    #[serde(default)]
    pub overflow_events_7d: usize,
    /// Ajuste actual al threshold de presión (negativo = más conservador).
    #[serde(default)]
    pub overflow_threshold_offset_pp: i32,
    /// Workload mode detected by the feature-based classifier (Phase 3).
    #[serde(default)]
    pub overflow_workload_mode: String,
    // Predictive agent metrics
    #[serde(default)]
    pub predictive_agent_active: bool,
    #[serde(default)]
    pub predictive_agent_cycles: u64,
    #[serde(default)]
    pub predictive_agent_arm_pulls: [u64; 5],
    #[serde(default)]
    pub predictive_agent_last_intervention: String,
    // Signal intelligence metrics
    #[serde(default)]
    pub si_pressure_smooth: f64,
    #[serde(default)]
    pub si_pressure_velocity: f64,
    #[serde(default)]
    pub si_p_oom_30s: f64,
    #[serde(default)]
    pub si_urgency: f64,
    #[serde(default)]
    pub si_regime_shifts: u64,
    #[serde(default)]
    pub si_monopoly_risk: f64,
    #[serde(default)]
    pub si_entropy_anomaly: f64,
    #[serde(default)]
    pub recently_applied_restore_status: Option<crate::engine::recently_applied::RestoreStatus>,
    /// Restore status telemetry (Sprint 3 Phase B — flushed from lf_metrics).
    /// Mutually-exclusive: at most one is non-zero per startup.
    /// Replaces/parallels the legacy `recently_applied_restore_status` Option.
    #[serde(default)]
    pub restore_status_missing: u64,
    #[serde(default)]
    pub restore_status_restored_n: u64,
    #[serde(default)]
    pub restore_status_discarded_corrupt: u64,
    #[serde(default)]
    pub restore_status_discarded_clock_delta: u64,
    #[serde(default)]
    pub restore_status_discarded_boot_crossed: u64,
    /// IdentityCache telemetry (Sprint 3 Phase A4 — flushed from lf_metrics).
    /// Hit ratio derivable as hits / (hits + misses).
    /// proc_pidpath_calls validates that p95 recovery is genuine amortization.
    #[serde(default)]
    pub identity_cache_hits: u64,
    #[serde(default)]
    pub identity_cache_misses: u64,
    #[serde(default)]
    pub identity_cache_evictions: u64,
    #[serde(default)]
    pub identity_cache_ttl_expired: u64,
    #[serde(default)]
    pub identity_cache_exit_invalidations: u64,
    #[serde(default)]
    pub identity_proc_pidpath_calls: u64,
    /// ActionAccumulator telemetry (Sprint 4 Fase 5 — flushed from lf_metrics).
    /// Per-variant push counters published from `ActionAccumulator::telemetry()`
    /// at finalize time. Counters are cumulative across all daemon cycles.
    /// Invariant: Σ(typed per-variant) + actions_pushed_raw_total ==
    /// total emitted into the dispatcher (push_raw is the escape-hatch path
    /// and does NOT increment any per-variant total).
    #[serde(default)]
    pub actions_pushed_throttle_total: u64,
    #[serde(default)]
    pub actions_pushed_freeze_total: u64,
    #[serde(default)]
    pub actions_pushed_unfreeze_total: u64,
    #[serde(default)]
    pub actions_pushed_boost_total: u64,
    #[serde(default)]
    pub actions_pushed_set_memorystatus_total: u64,
    #[serde(default)]
    pub actions_pushed_set_thread_qos_total: u64,
    #[serde(default)]
    pub actions_pushed_set_sysctl_total: u64,
    #[serde(default)]
    pub actions_pushed_toggle_spotlight_total: u64,
    #[serde(default)]
    pub actions_pushed_quarantine_daemon_total: u64,
    #[serde(default)]
    pub actions_pushed_raw_total: u64,
    #[serde(default)]
    pub actions_rejected_shape_total: u64,
    /// Lotka-Volterra Jacobian stability class: 0=Degenerate 1=StableNode 2=StableSpiral 3=UnstableSaddle 4=Unstable
    #[serde(default)]
    pub si_stability_regime: u8,
    // Thread-level QoS metrics (Phase 1)
    #[serde(default)]
    pub thread_qos_applied: u64,
    #[serde(default)]
    pub thread_qos_hot_routes: u64,
    #[serde(default)]
    pub thread_qos_cold_routes: u64,
    // RL threshold agent metrics (Phase 4)
    #[serde(default)]
    pub rl_adjustment_pp: i32,
    #[serde(default)]
    pub rl_total_ticks: u64,
    #[serde(default)]
    pub rl_total_overflows: u64,
    // IOReport hardware telemetry (P/E cluster utilization, power)
    #[serde(default)]
    pub ioreport_p_cluster_pct: f64,
    #[serde(default)]
    pub ioreport_e_cluster_pct: f64,
    #[serde(default)]
    pub ioreport_gpu_pct: f64,
    #[serde(default)]
    pub ioreport_ane_busy: bool,
    #[serde(default)]
    pub ioreport_cpu_mw: f64,
    #[serde(default)]
    pub ioreport_total_watts: f64,
    // SMC direct telemetry
    #[serde(default)]
    pub smc_system_power_watts: Option<f64>,
    #[serde(default)]
    pub smc_lid_closed: bool,
    #[serde(default)]
    pub smc_charger_watts: Option<f64>,
    #[serde(default)]
    pub smc_battery_tte_min: Option<u16>,
    /// 2026-05-14: SMC reader bit-rotted on macOS 26 Tahoe (AppleSMC userclient
    /// changed; smc_bridge.c keys unreadable). Field surfaces the failure mode
    /// so the dashboard reflects reality instead of silent nulls. One of:
    /// "ok" (live data), "unavailable_macos26" (SMC bridge broken),
    /// "uninitialized" (daemon just started).
    #[serde(default)]
    pub smc_diagnostic: String,
    // KPC hardware performance counters
    #[serde(default)]
    pub kpc_ipc: f64,
    /// Fraction of CPU cycles stalled on memory (0.0=compute-bound, 1.0=memory-stalled).
    /// Derived from KPC IPC vs Apple M1 P-core peak IPC (~5.0).
    /// >0.7 = system >70% memory-bound → freeze decisions are lower risk.
    #[serde(default)]
    pub kpc_memory_bound_score: f64,
    /// Top 3 wakeup vampire processes: "name(rate/s)" strings.
    /// Processes with >50 idle+interrupt wakeups/sec drain battery even when idle.
    #[serde(default)]
    pub wakeup_vampires: Vec<String>,
    /// Whether Apple AMX coprocessor is available on this chip.
    /// Detected via raw ASM probe: `.word 0x00201220` (AMX_SET instruction) in forked child.
    /// AMX is the undocumented matrix coprocessor used by Accelerate.framework for BLAS/ML.
    #[serde(default)]
    pub amx_available: bool,
    /// AMX context-switch overhead estimate (nanoseconds).
    /// When AMX state is dirty, context switches incur ~50ns overhead (5120B save/restore).
    #[serde(default)]
    pub amx_cs_overhead_ns: u64,
    /// Number of processes with anomaly_score ≥ 3.0 this cycle.
    /// Processes deviating ≥ 3 MADs from their learned hardware counter baseline.
    #[serde(default)]
    pub anomaly_process_count: usize,
    /// Top anomalous processes: "name(score×)" strings, sorted by score descending.
    /// e.g. "backupd(8.2×)" = backupd is 8.2 MADs above its baseline behavior.
    #[serde(default)]
    pub anomaly_processes: Vec<String>,
    /// Number of process baselines with ≥ 5 observations (warm, actively detecting anomalies).
    /// 0 = cold start (no anomaly detection yet); grows as processes are observed.
    #[serde(default)]
    pub process_baseline_warm: usize,
    // Cache contention detection (ContentionDetector)
    #[serde(default)]
    pub contention_score: f64,
    #[serde(default)]
    pub contention_heavy_count: usize,
    #[serde(default)]
    pub contention_pairs_active: u32,
    // Window/app lifecycle sensor (WindowSensor)
    #[serde(default)]
    pub window_tab_delta: i32,
    #[serde(default)]
    pub window_renderer_count: u32,
    #[serde(default)]
    pub window_freed_heavy_app: bool,
    #[serde(default)]
    pub window_tab_velocity_ema: f64,
    #[serde(default)]
    pub window_pressure_floor: f64,
    #[serde(default)]
    pub window_session_phase: String,
    #[serde(default)]
    pub window_workload_intent: String,
    // Build progress tracker
    #[serde(default)]
    pub build_phase: String,
    #[serde(default)]
    pub build_progress: f32,
    // Rosetta AOT compilation
    #[serde(default)]
    pub rosetta_aot_active: bool,
    // SMC thermal (direct, <100µs)
    #[serde(default)]
    pub smc_cpu_temp_celsius: Option<f64>,
    #[serde(default)]
    pub smc_gpu_temp_celsius: Option<f64>,
    #[serde(default)]
    pub smc_battery_temp_celsius: Option<f64>,
    #[serde(default)]
    pub smc_cpu_voltage: Option<f64>,
    #[serde(default)]
    pub smc_p_cluster_watts: Option<f64>,
    // IOReport memory bandwidth
    #[serde(default)]
    pub ioreport_amc_bandwidth_pct: f64,
    // IOPMrootDomain direct thermal
    #[serde(default)]
    pub iopm_thermal_warning: String,
    #[serde(default)]
    pub iopm_power_source: String,
    // Per-process energy (ri_billed_energy) — top consumer
    #[serde(default)]
    pub energy_top_pid_name: String,
    #[serde(default)]
    pub energy_top_pid_mw: f64,

    // Daemon self-IPC (thread_selfcounts syscall 186)
    #[serde(default)]
    pub daemon_cycle_ipc: f64,

    // ── v0.7.0 Deep Scan metrics ────────────────────────────────────────
    /// Number of vm_region scans performed this session.
    #[serde(default)]
    pub deep_scan_count: u64,
    /// Number of page temperature probes performed this session.
    #[serde(default)]
    pub deep_scan_temp_probes: u64,
    /// decide_enhanced outcomes: freeze / skip / hint.
    #[serde(default)]
    pub deep_scan_freeze: u64,
    #[serde(default)]
    pub deep_scan_skip: u64,
    #[serde(default)]
    pub deep_scan_hint: u64,

    // ── Behavioral protection metrics ───────────────────────────────────
    /// Dev runtimes evaluated by behavioral_protection_score this session.
    #[serde(default)]
    pub bps_evaluated: u64,
    /// Dev runtimes that kept protection (score >= pressure).
    #[serde(default)]
    pub bps_protected: u64,
    /// Dev runtimes that lost protection (score < pressure).
    #[serde(default)]
    pub bps_demoted: u64,
    /// Lowest behavioral score seen this cycle (0.0 = fully dormant hog).
    #[serde(default)]
    pub bps_min_score: f64,
    /// Name of the process with the lowest behavioral score.
    #[serde(default)]
    pub bps_min_score_name: String,
    /// Top causal process pairs (co-occur during pressure spikes).
    /// Format: "A + B (count)" for up to 5 pairs.
    #[serde(default)]
    pub causal_pairs: Vec<String>,
    /// Causal effect of throttling: observed drop minus natural drift.
    /// Positive = throttling actually helped beyond natural fluctuation.
    #[serde(default)]
    pub causal_effect_avg: f64,
    /// Natural pressure drift EMA (what happens without action).
    #[serde(default)]
    pub natural_drift: f64,
    /// Short-window (3-cycle) pressure velocity: mean delta over last 3 no-action cycles.
    /// Positive = pressure dropping naturally. Fast causal attribution signal.
    #[serde(default)]
    pub short_drift_velocity: f64,
    /// Experience memory size (resolved outcome records).
    #[serde(default)]
    pub experience_memory_size: usize,
    /// Causal edges with slow-horizon (15-cycle) data. Captures delayed effects.
    #[serde(default)]
    pub causal_slow_horizon_count: usize,
    /// Causal edges with mechanism attribution (RSS/CPU/swap channel identified).
    #[serde(default)]
    pub causal_mechanism_count: usize,
    /// Top causal mechanism summaries: "throttle:X via rss (−42MB)" format.
    #[serde(default)]
    pub causal_mechanisms: Vec<String>,
    // ── Action queue backpressure ───��───────────────────────────────���─────
    /// Current backpressure ratio of the action queue [0.0, 1.0].
    /// 0.0 = queue empty. 1.0 = queue at capacity.
    /// High values mean actions are accumulating faster than they are executed.
    #[serde(default)]
    pub action_queue_backpressure: f64,
    /// Pending unresolved outcome observations in OutcomeTracker [0, 300].
    /// High depth = throttles are being applied faster than outcomes resolve (30s window).
    #[serde(default)]
    pub outcome_pending_depth: usize,

    /// Habituation skips: processes unchanged for ≥5 CPU/RSS cycles, skipped in decide_actions.
    /// Incremented once per cycle by habituated_pids.len().
    #[serde(default)]
    pub habituation_skips: u64,

    /// Dr. Zero self-challenge score: average prediction error across hop groups.
    /// Low = solver is well-calibrated. High = needs more training.
    #[serde(default)]
    pub dr_zero_self_challenge: f64,
    /// Dr. Zero HRPO group summaries: "Browser(eff=70% n=50 err=0.15)"
    #[serde(default)]
    pub dr_zero_groups: Vec<String>,
    /// Dr. Zero exploration signal: groups needing more data.
    #[serde(default)]
    pub dr_zero_exploration: Vec<String>,
    /// NARS concept drift score [0.0, 1.0].
    /// EMA of per-process belief frequency shifts after Revision rule.
    /// > 0.05: notable drift. > 0.08: recalibration triggered.
    /// [Pei Wang 2013] Non-Axiomatic Reasoning System, §3.3.3
    #[serde(default)]
    pub nars_drift_score: f64,
    /// Number of process beliefs currently in drifted state (freq shift >= 20pp).
    #[serde(default)]
    pub nars_drifted_beliefs: usize,
    /// Global arousal EMA level ∈ [0,1]. Computed from memory pressure + swap.
    /// [Yerkes & Dodson 1908] arousal modulates learning rate.
    #[serde(default)]
    pub arousal_level: f32,
    /// Yerkes-Dodson zone label: Idle/Calm/Optimal/Stressed/Crisis.
    #[serde(default)]
    pub arousal_zone: String,

    // ── Telemetría enriched (2026-04-04) ────────────────────────────────────
    /// SwapTrend from swap_predictor: "Decreasing" | "Stable" | "Increasing" | "Critical"
    /// Previously computed but never exposed. Key leading indicator for display jank.
    #[serde(default)]
    pub swap_trend: String,
    /// WindowServer CPU usage % — display compositor load. Rising = rendering pressure.
    #[serde(default)]
    pub windowserver_cpu_pct: f32,
    /// Compression ratio: compressed_bytes / total_memory ∈ [0.0, 1.0].
    /// Approaches 1.0 = compressor fully loaded, swap I/O imminent.
    #[serde(default)]
    pub compressed_memory_ratio: f64,
    /// Sum of RSS (MB) of all currently frozen processes — RAM Apollo has reclaimed.
    #[serde(default)]
    pub frozen_ram_mb: f64,
    /// Times the display pipeline boost fired this session (swap_delta >= 0.5 MB/s).
    #[serde(default)]
    pub display_boost_count: u64,
    /// Consecutive cycles where memory_pressure > bg_pressure threshold.
    /// 0 = currently below threshold. High = sustained pressure.
    #[serde(default)]
    pub cycles_high_pressure: u32,
    /// Number of PIDs behaviorally classified as interactive (cpu_wall_ratio EMA < 0.05).
    /// Shows how many processes were learned dynamically vs hardcoded list.
    #[serde(default)]
    pub behavior_interactive_pid_count: usize,
    /// Current RL threshold as absolute value (bg_pressure + rl_adjustment).
    /// More actionable than rl_adjustment_pp alone.
    #[serde(default)]
    pub rl_threshold_current: f64,
    /// Which gate triggered the last freeze decision: "delta" | "committed" | "none".
    #[serde(default)]
    pub freeze_gate_last: String,
    /// What triggered the last ML daemon throttle: "swap-early" | "pressure" | "none".
    #[serde(default)]
    pub ml_throttle_source: String,

    // ── Fluidity Intelligence ────────────────────────────────────────────────
    /// Composite fluidity score 0–1 (1 = perfectly fluid, Kalman-smoothed).
    /// Combines WindowServer CPU, GPU load, and app launch pressure.
    #[serde(default)]
    pub fluidity_score: f32,
    /// True when WindowServer CPU spike detected (window resize/move active).
    #[serde(default)]
    pub window_op_active: bool,
    /// True when a new app launch is in progress (protection window active).
    #[serde(default)]
    pub app_launching: bool,
    /// Name of the app currently being launched (empty if none).
    #[serde(default)]
    pub app_launch_name: String,
    /// True when sustained fluidity degradation detected (EMA < 0.65).
    #[serde(default)]
    pub fluidity_degraded: bool,
    /// Kalman-predicted fluidity in 3 cycles (~6s ahead).
    /// [Welch & Bishop 2006] 1D Kalman for noise-rejected prediction.
    #[serde(default)]
    pub fluidity_predicted_3s: f32,
    /// Rate of fluidity change per second (positive = improving, negative = degrading).
    #[serde(default)]
    pub fluidity_velocity: f32,

    // ── User context telemetry ────────────────────────────────────────────────
    /// Seconds since last keyboard/mouse event (from IOHIDSystem HIDIdleTime).
    /// 0 = recently active or unknown.
    #[serde(default)]
    pub user_idle_secs: f64,
    /// True when any non-Apollo sleep-prevention assertion is active.
    /// Indicates active media playback, presentation, or call.
    #[serde(default)]
    pub user_has_sleep_assertion: bool,
    /// True when a video/audio call is likely in progress.
    #[serde(default)]
    pub user_call_in_progress: bool,
    /// True when audio is actively being output (coreaudiod assertion).
    #[serde(default)]
    pub user_audio_active: bool,

    // ── Chromium Renderer Manager ────────────────────────────────────────────
    /// Total renderer processes tracked (across all Chromium/Electron apps).
    #[serde(default)]
    pub chromium_renderers_total: u32,
    /// Renderer processes currently frozen (SIGSTOP) by ChromiumManager.
    #[serde(default)]
    pub chromium_renderers_frozen: u32,
    /// Renderer processes demoted to E-cores this cycle.
    #[serde(default)]
    pub chromium_renderers_ecore: u32,
    /// Estimated RAM freed (MB) by frozen renderers.
    #[serde(default)]
    pub chromium_freed_mb: f64,
    /// Names of browsers/apps with managed renderers.
    #[serde(default)]
    pub chromium_browsers_managed: Vec<String>,

    // ── Neurocognitive v2.0 (UCHS) ─────────────────────────────────────────
    /// Unified Cognitive Health Score [0,1]. 6 dimensions.
    #[serde(default)]
    pub uchs_composite: f32,
    /// UCHS grade label (S+, S, A, B, C, D, F).
    #[serde(default)]
    pub uchs_grade: String,
    /// Whether cognitive recovery mode is active.
    #[serde(default)]
    pub uchs_recovery_mode: bool,
    /// Epistemic uncertainty composite [0,1].
    #[serde(default)]
    pub epistemic_uncertainty: f32,
    /// Epistemic uncertainty level label.
    #[serde(default)]
    pub epistemic_level: String,
    /// Guard-tower over-protection signal (6th composite component, 2026-05-10).
    /// OutcomeTracker.mean_blocked_overprotection() — Bayesian-Laplace aggregate
    /// over mature blocked patterns. High = blocks "would have helped" per Rubin
    /// 1974 counterfactual → guard policy is over-protecting.
    #[serde(default)]
    pub guard_overprotection: f32,
    /// Active-coalition envelope size (recently-foreground coalitions, 5 min grace).
    /// 0 = no recent fg, 1-3 = healthy app-switching window.
    #[serde(default)]
    pub active_coalitions_count: u32,
    /// Phase 0 lock-decomp baseline (2026-05-10).
    /// Average + max wait time (µs) for the metrics god-lock. If avg << held
    /// → contention is NOT the bottleneck → lock-decomp won't help.
    #[serde(default)]
    pub metrics_lock_wait_avg_us: f64,
    #[serde(default)]
    pub metrics_lock_wait_max_us: u64,
    #[serde(default)]
    pub metrics_lock_held_avg_us: f64,
    #[serde(default)]
    pub metrics_lock_held_max_us: u64,
    /// Phase 0b cycle-stage split (NotebookLM priority #1, 2026-05-10).
    /// Mean + max latency per stage in ms. Total of all 5 means ≈ avg
    /// cycle latency. Identifies which stage dominates p95.
    #[serde(default)]
    pub stage_sense_avg_ms: f64,
    #[serde(default)]
    pub stage_sense_max_ms: f64,
    #[serde(default)]
    pub stage_reason_avg_ms: f64,
    #[serde(default)]
    pub stage_reason_max_ms: f64,
    #[serde(default)]
    pub stage_execute_avg_ms: f64,
    #[serde(default)]
    pub stage_execute_max_ms: f64,
    #[serde(default)]
    pub stage_learn_avg_ms: f64,
    #[serde(default)]
    pub stage_learn_max_ms: f64,
    #[serde(default)]
    pub stage_persist_avg_ms: f64,
    #[serde(default)]
    pub stage_persist_max_ms: f64,
    /// Phase 0c REASON sub-stage split (2026-05-10).
    #[serde(default)]
    pub stage_reason_signal_avg_ms: f64,
    #[serde(default)]
    pub stage_reason_signal_max_ms: f64,
    #[serde(default)]
    pub stage_reason_usercontext_avg_ms: f64,
    #[serde(default)]
    pub stage_reason_usercontext_max_ms: f64,
    #[serde(default)]
    pub stage_reason_holtwinters_avg_ms: f64,
    #[serde(default)]
    pub stage_reason_holtwinters_max_ms: f64,
    #[serde(default)]
    pub stage_reason_pagereclaim_avg_ms: f64,
    #[serde(default)]
    pub stage_reason_pagereclaim_max_ms: f64,
    #[serde(default)]
    pub stage_reason_chromium_avg_ms: f64,
    #[serde(default)]
    pub stage_reason_chromium_max_ms: f64,
    #[serde(default)]
    pub stage_reason_enrich_avg_ms: f64,
    #[serde(default)]
    pub stage_reason_enrich_max_ms: f64,
    /// MetaCognition meta_confidence [0,1].
    #[serde(default)]
    pub meta_confidence: f32,
    /// MetaCognition humble mode active.
    #[serde(default)]
    pub humble_mode: bool,
    /// AdversarialProbe pass rate [0,1].
    #[serde(default)]
    pub adversarial_pass_rate: f32,
    /// AdversarialProbe safety alert active.
    #[serde(default)]
    pub adversarial_safety_alert: bool,
    /// CognitiveRewardBus signal-to-noise.
    #[serde(default)]
    pub cognitive_snr: f64,
    /// SelfRewardingEvaluator mean quality [0,1].
    #[serde(default)]
    pub self_eval_quality: f32,
    /// ReptileMeta cached workload count.
    #[serde(default)]
    pub reptile_cached_workloads: usize,
    /// DriftDetector early warning score [0,1].
    #[serde(default)]
    pub drift_early_warning: f64,

    /// Predicted thermal throttle level from ThermalManager (0–100).
    /// Non-zero = ThermalManager forecasts throttling based on temperature trend.
    #[serde(default)]
    pub thermal_predicted_throttle: u8,
    /// Seconds until thermal throttling predicted by ThermalManager.
    /// null = no forecast (cooling or insufficient history). 0 = already throttling. >0 = seconds of headroom.
    #[serde(default)]
    pub thermal_seconds_to_throttle: Option<i32>,
    /// ThermalManager trend label: "Cooling" | "Stable" | "Warming" | "Critical".
    #[serde(default)]
    pub thermal_trend_predicted: String,

    /// FreezeProcess actions upgraded to ThrottleProcess (QoS Background) this cycle.
    /// Non-zero = causal attribution identified CPU-dominant processes.
    /// [Pearl 2009] causal mediation — QoS achieves CPU reduction without SIGSTOP.
    #[serde(default)]
    pub causal_qos_upgrades_cycle: u32,
    /// Sum of all active pressure boost factors (hardware + thermal + battery + …).
    /// effective_pressure = memory_pressure + pressure_total_boost (clamped 0..1).
    /// Shows WHY effective pressure differs from raw memory_pressure.
    #[serde(default)]
    pub pressure_total_boost: f64,
    /// Largest active pressure boost factor ("thermal", "hardware",
    /// "memory_bandwidth", "llm_workload", etc.). "none" = no active boosts.
    #[serde(default)]
    pub pressure_dominant_factor: String,

    // ── Apollo Intelligence Score (AIS) — composite quality metric ─────
    // Computed every AIS_COMPUTE_EVERY_N_CYCLES (~60s) in merge_cycle_metrics.
    // See engine::intelligence_score for dimension formulas.
    #[serde(default)]
    pub ais_score: f64,
    #[serde(default)]
    pub ais_grade: String,
    #[serde(default)]
    pub ais_decision: f64,
    #[serde(default)]
    pub ais_signal: f64,
    #[serde(default)]
    pub ais_learning: f64,
    #[serde(default)]
    pub ais_resource: f64,
    #[serde(default)]
    pub ais_safety: f64,
    #[serde(default)]
    pub ais_adaptability: f64,
    #[serde(default)]
    pub ais_wisdom: f64,
    /// 2026-05-12 — Active regime selected by the chromium Step 2 gate's
    /// priority chain in daemon_chromium_tick.rs. One of: "default",
    /// "media", "build", "call", "llm". Surfaces silently-failing regime
    /// transitions so a regression to "default" under crisis is visible.
    #[serde(default)]
    pub chromium_gate_regime: String,
    #[serde(default)]
    pub ais_pareto_balanced: bool,

    /// Maintenance Purge Gate telemetry (Sprint 5 Mes 0 — 2026-05-10).
    /// Flushed from lf_metrics each cycle via sync_from_lockfree.
    #[serde(default)]
    pub maintenance_purge_total: u64,
    #[serde(default)]
    pub maintenance_purge_skipped_pressure_total: u64,
    #[serde(default)]
    pub maintenance_purge_skipped_swap_floor_total: u64,
    #[serde(default)]
    pub maintenance_purge_skipped_growing_total: u64,
    #[serde(default)]
    pub maintenance_purge_skipped_idle_total: u64,
    #[serde(default)]
    pub maintenance_purge_skipped_build_mode_total: u64,
    #[serde(default)]
    pub maintenance_purge_skipped_rate_limit_total: u64,

    /// Phase 4.3 — Policy Rollback Guard observability (Sprint 7, 2026-05-16).
    /// Total `PolicyRollbackGuard::evaluate` calls (per cycle). A long
    /// stretch with `evaluations_total` increasing but `executions_total`
    /// flat is the expected steady state: the daemon is checking each
    /// cycle but no rollback was warranted. Both counters surface to
    /// `runtime_metrics.json`.
    #[serde(default)]
    pub policy_rollback_evaluations_total: u64,
    #[serde(default)]
    pub policy_rollback_executions_total: u64,
}

impl RuntimeMetrics {
    /// True iff survival mode was entered at least once during this session.
    ///
    /// **Cumulative semantic** — once true, stays true until daemon restart,
    /// even if the system long-since recovered to a healthy state. Use this
    /// for "did we ever?" questions (AI scoring, post-mortem audits). For
    /// "are we in survival now?" call
    /// [`crate::engine::safety::survival_mode_active_total`] with live
    /// pressure and swap inputs.
    pub fn ever_entered_survival_mode(&self) -> bool {
        self.survival_mode_entry_count > 0
    }
}

/// Serializable foreground app info for the protocol/dashboard.
///
/// Cross-crate visibility: used by apollo-optimizerd main.rs to build foreground-app reports.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForegroundAppInfo {
    pub pid: u32,
    pub name: String,
    pub bundle_id: Option<String>,
}

/// Serializable per-app energy info for the protocol/dashboard.
///
/// Cross-crate visibility: used by apollo-optimizerd main.rs to populate energy consumer
/// reports in DaemonStatus. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyConsumerInfo {
    pub name: String,
    pub current_watts: f64,
    pub percentage: f64,
}

/// Cross-crate visibility: the primary status response type. Used by apollo-menubar,
/// apollo-optimizerctl, and apollo-optimizerd socket_handler to transmit daemon state over IPC.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub running: bool,
    pub profile: OptimizationProfile,
    pub latency_target: LatencyTarget,
    pub effective_profile: OptimizationProfile,
    pub kill_switch: bool,
    pub throttle_level: String,
    pub thermal_state: String,
    pub last_blockers: Vec<BlockerScore>,
    pub auto_profile_enabled: bool,
    pub base_profile: OptimizationProfile,
    pub override_active: bool,
    pub override_expires_at: Option<DateTime<Utc>>,
    pub transition_reason: String,
    pub post_wake_grace_active: bool,
    pub post_wake_grace_remaining_secs: u64,
    pub last_wake_at: Option<DateTime<Utc>>,
    pub post_wake_policy: String,
    pub reactor_mode: String,
    pub reactor_health: String,
    pub metrics: RuntimeMetrics,
    #[serde(default)]
    pub llm: Option<LlmStatus>,
    /// Currently frozen processes. Empty if none.
    #[serde(default)]
    pub frozen_processes: Vec<FrozenProcessInfo>,
}

/// Cross-crate visibility: embedded in DaemonStatus.llm; read by apollo-optimizerctl and
/// apollo-menubar via the DaemonResponse IPC path. Audited 2026-05-09 during Sprint 5
/// Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmStatus {
    pub enabled: bool,
    pub training_active: bool,
    pub training_expires_at: Option<DateTime<Utc>>,
    pub has_api_key: bool,
    pub mode: LlmRunMode,
    pub last_call_at: Option<DateTime<Utc>>,
    pub last_attempt_at: Option<DateTime<Utc>>,
    pub last_http_status: Option<u16>,
    pub last_error: Option<String>,
    pub last_trigger_reason: Option<String>,
    pub calls_in_current_window: u32,
    pub min_confidence: f64,
    pub calls_today: u32,
    pub daily_budget: u32,
    pub daily_budget_remaining: u32,
    pub last_suggestion_confidence: Option<f64>,
    pub last_suggestion_rationale: Option<String>,
    pub learned_policy: LearnedPolicyStatus,
}

/// Demoted to pub(crate): no bin imports or uses this type; no pub function takes it as
/// parameter. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct UsageStatus {
    pub(crate) entries: usize,
    pub(crate) last_updated_at: Option<DateTime<Utc>>,
}

/// Cross-crate visibility: used by apollo-optimizerctl to handle usage-top and explain
/// commands over IPC. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UsageResponse {
    Top(UsageTopReport),
    Explain(UsageEntrySummary),
}

/// Cross-crate visibility: embedded in LlmStatus and returned via DaemonResponse; accessed by
/// apollo-optimizerctl and apollo-menubar. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LearnedPolicyStatus {
    pub interactive_patterns: usize,
    pub noise_patterns: usize,
    pub protected_patterns: usize,
    pub learned_at: Option<DateTime<Utc>>,
}

/// Cross-crate visibility: used by apollo-optimizerd llm_daemon.rs and returned in LlmStatus
/// over IPC to monitoring clients. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum LlmRunMode {
    Sensitive,
    Strict,
    Off,
}

impl Default for LlmRunMode {
    fn default() -> Self {
        Self::Sensitive
    }
}

/// Summary of circuit breaker and degradation state, returned by `GetHealth`.
///
/// Cross-crate visibility: used by apollo-optimizerd socket_handler.rs to build the
/// GetHealth response; accessed by apollo-optimizerctl. Audited 2026-05-09 during Sprint 5
/// Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// "healthy" | "degraded" | "emergency"
    pub status: String,
    /// Circuit breaker state: "closed" | "open" | "half_open"
    pub circuit_breaker: String,
    /// Operation mode: "full" | "conservative" | "observe" | "emergency"
    pub operation_mode: String,
    /// Failures in the last 60 seconds as a fraction of threshold (0.0–∞).
    pub failure_rate_60s: f32,
    /// Total optimization cycles completed.
    pub uptime_cycles: u64,
    /// Total execute_actions failures recorded (lifetime).
    pub total_failures: u64,
    /// Total circuit breaker trips (lifetime).
    pub cb_trips_total: u64,
    /// Total degradation mode transitions (lifetime).
    pub degradation_transitions: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── OptimizationProfile roundtrip ─────────────────────────────────────────

    #[test]
    fn optimization_profile_roundtrip() {
        for profile in [
            OptimizationProfile::BalancedRoot,
            OptimizationProfile::AggressiveRoot,
            OptimizationProfile::SafeRoot,
        ] {
            let json = serde_json::to_string(&profile).expect("serialize OptimizationProfile");
            let rt: OptimizationProfile =
                serde_json::from_str(&json).expect("deserialize OptimizationProfile");
            assert_eq!(rt, profile);
        }
    }

    #[test]
    fn optimization_profile_kebab_case_in_json() {
        let json = serde_json::to_string(&OptimizationProfile::BalancedRoot)
            .expect("serialize OptimizationProfile");
        assert!(
            json.contains('-'),
            "expected kebab-case dash in JSON, got: {json}"
        );
    }

    #[test]
    fn optimization_profile_as_str_matches_json() {
        assert_eq!(OptimizationProfile::BalancedRoot.as_str(), "balanced-root");
        assert_eq!(
            OptimizationProfile::AggressiveRoot.as_str(),
            "aggressive-root"
        );
        assert_eq!(OptimizationProfile::SafeRoot.as_str(), "safe-root");
    }

    // ── LatencyTarget roundtrip ───────────────────────────────────────────────

    #[test]
    fn latency_target_roundtrip() {
        for target in [
            LatencyTarget::Low,
            LatencyTarget::Normal,
            LatencyTarget::Max,
        ] {
            let json = serde_json::to_string(&target).expect("serialize LatencyTarget");
            let rt: LatencyTarget = serde_json::from_str(&json).expect("deserialize LatencyTarget");
            assert_eq!(rt, target);
        }
    }

    #[test]
    fn latency_target_default_is_normal() {
        assert_eq!(LatencyTarget::default(), LatencyTarget::Normal);
    }

    // ── FrozenEntry roundtrip ─────────────────────────────────────────────────

    #[test]
    fn frozen_entry_roundtrip() {
        let entry = FrozenEntry {
            frozen_at: chrono::Utc::now(),
            source: FreezeSource::MainLoop,
            pressure_at_freeze: 0.75,
            process_name: Some("TestProcess".to_string()),
            start_sec: 0,
            original_jetsam_priority: None,
        };
        let json = serde_json::to_string(&entry).expect("serialize FrozenEntry");
        let rt: FrozenEntry = serde_json::from_str(&json).expect("deserialize FrozenEntry");
        assert_eq!(rt.pressure_at_freeze, entry.pressure_at_freeze);
        assert_eq!(rt.process_name, entry.process_name);
    }

    // ── BlockerScore roundtrip ────────────────────────────────────────────────

    #[test]
    fn blocker_score_roundtrip() {
        let score = BlockerScore {
            name: "com.example.app".to_string(),
            pid: 1234,
            score: 0.85,
            blocker_cpu_spike: 0.5,
            interactive_wait_ratio: 0.3,
            blocker_seen_recently: true,
            reactor_event_weight: 1.0,
        };
        let json = serde_json::to_string(&score).expect("serialize BlockerScore");
        let rt: BlockerScore = serde_json::from_str(&json).expect("deserialize BlockerScore");
        assert_eq!(rt.name, score.name);
        assert_eq!(rt.pid, score.pid);
        assert!((rt.score - score.score).abs() < f64::EPSILON);
    }

    // ── RuntimeMetrics no NaN ─────────────────────────────────────────────────

    #[test]
    fn runtime_metrics_default_no_nan_f64() {
        let m = RuntimeMetrics::default();
        assert!(!m.p95_cycle_ms.is_nan(), "p95_cycle_ms should not be NaN");
        assert!(
            !m.last_pressure_score.is_nan(),
            "last_pressure_score should not be NaN"
        );
        assert!(
            !m.swap_delta_bps.is_nan(),
            "swap_delta_bps should not be NaN"
        );
        assert!(
            !m.memory_pressure.is_nan(),
            "memory_pressure should not be NaN"
        );
    }

    #[test]
    fn runtime_metrics_default_cycles_zero() {
        let m = RuntimeMetrics::default();
        assert_eq!(m.cycles, 0);
        assert_eq!(m.boosts_applied, 0);
        assert_eq!(m.failures, 0);
    }

    // ── RootAction::identity_fields — Sprint 4 IdentityVerifier merge ─────────

    use crate::engine::audit_types::DecisionReason;

    #[test]
    fn identity_fields_throttle_carries_full_tuple() {
        let a = RootAction::ThrottleProcess {
            pid: 100,
            name: "Brave".into(),
            aggressive: false,
            reason: "test".into(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 1_000_000,
            start_usec: 500_000,
        };
        assert_eq!(
            a.identity_fields(),
            Some((100u32, Some("Brave"), 1_000_000u64, 500_000u64))
        );
    }

    #[test]
    fn identity_fields_freeze_carries_full_tuple() {
        let a = RootAction::FreezeProcess {
            pid: 200,
            name: "Slack".into(),
            reason: "stale".into(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 9_999,
            start_usec: 1,
        };
        assert_eq!(
            a.identity_fields(),
            Some((200u32, Some("Slack"), 9_999u64, 1u64))
        );
    }

    #[test]
    fn identity_fields_boost_has_no_start_sec() {
        let a = RootAction::BoostProcess {
            pid: 300,
            name: "Alacritty".into(),
            reason: "fg".into(),
            decision_reason: DecisionReason::InteractiveFocus,
        };
        assert_eq!(
            a.identity_fields(),
            Some((300u32, Some("Alacritty"), 0u64, 0u64))
        );
    }

    #[test]
    fn identity_fields_unfreeze_has_no_start_sec() {
        let a = RootAction::UnfreezeProcess {
            pid: 400,
            name: "Code Helper".into(),
            reason: "thaw".into(),
            decision_reason: DecisionReason::PressureContext,
        };
        assert_eq!(
            a.identity_fields(),
            Some((400u32, Some("Code Helper"), 0u64, 0u64))
        );
    }

    #[test]
    fn identity_fields_set_memorystatus_has_pid_no_name() {
        let a = RootAction::SetMemorystatus {
            pid: 500,
            priority: -1,
            reason: "purge hint".into(),
            decision_reason: DecisionReason::PressureContext,
        };
        assert_eq!(a.identity_fields(), Some((500u32, None, 0u64, 0u64)));
    }

    #[test]
    fn identity_fields_set_thread_qos_carries_name() {
        let a = RootAction::SetThreadQoS {
            pid: 600,
            name: "Brave Helper".into(),
            thread_index: 3,
            tier: "background".into(),
            reason: "demote".into(),
            decision_reason: DecisionReason::ThreadQoSRouting,
            affinity_tag: None,
        };
        assert_eq!(
            a.identity_fields(),
            Some((600u32, Some("Brave Helper"), 0u64, 0u64))
        );
    }

    #[test]
    fn identity_fields_set_sysctl_returns_none() {
        let a = RootAction::set_sysctl(
            "net.inet.tcp.delayed_ack",
            "3",
            "tune",
            DecisionReason::PressureContext,
        );
        assert_eq!(a.identity_fields(), None);
    }

    #[test]
    fn set_sysctl_action_clamps_bug6_regression() {
        // Regression for Bug 6 (Sprint 4 Phase 4 motivator): network-optimizer
        // emitted `SetSysctl { value: "4194304", .. }` for
        // `net.inet.tcp.sendspace`, which exceeds the safety allowlist max.
        // The factory must clamp to the allowlist max — not pass through.
        let action = crate::engine::types::SetSysctlAction::new_clamped(
            "net.inet.tcp.sendspace",
            "4194304",
            "bug6 regression",
            DecisionReason::PressureContext,
        );
        let ranges = crate::engine::safety::allowlisted_sysctls_with_ranges();
        let max_for_key = ranges
            .iter()
            .find(|r| r.key == "net.inet.tcp.sendspace")
            .map(|r| r.max)
            .expect("net.inet.tcp.sendspace must be in allowlist");
        assert_eq!(
            action.value(),
            max_for_key.to_string(),
            "value must clamp to allowlist max, not the raw 4194304"
        );
        assert_ne!(
            action.value(),
            "4194304",
            "raw 4194304 must not survive — that was Bug 6"
        );
    }

    #[test]
    fn set_sysctl_action_non_allowlist_passthrough() {
        let action = crate::engine::types::SetSysctlAction::new_clamped(
            "kern.totally.fake",
            "999",
            "passthrough",
            DecisionReason::PressureContext,
        );
        // Non-allowlist keys pass through unchanged so execute_actions can
        // reject them with BlockReason::InvalidSysctl.
        assert_eq!(action.value(), "999");
        assert_eq!(action.key(), "kern.totally.fake");
    }

    #[test]
    fn set_sysctl_action_unparseable_preserved() {
        // Non-numeric values must not panic and must round-trip unchanged.
        let action = crate::engine::types::SetSysctlAction::new_clamped(
            "kern.maxfiles",
            "auto",
            "unparseable",
            DecisionReason::PressureContext,
        );
        assert_eq!(action.value(), "auto");
    }

    #[test]
    fn set_sysctl_json_shape_unchanged_after_seal() {
        // Sprint 4 Phase 4: the variant changed from a struct variant to a
        // newtype variant wrapping `SetSysctlAction`. With Serde's default
        // externally-tagged enum + a newtype-variant whose inner struct has
        // the same field names, the JSON shape must be identical to the
        // pre-seal struct-variant form. Dashboards / journals / ops tools
        // that grep `"SetSysctl":{"key":...}` continue to work.
        let action = RootAction::set_sysctl(
            "net.inet.tcp.delayed_ack",
            "0",
            "tune",
            DecisionReason::PressureContext,
        );
        let json = serde_json::to_string(&action).expect("serialize");
        // Externally tagged variant: top-level object {"SetSysctl": ...}.
        assert!(
            json.starts_with("{\"SetSysctl\":{\""),
            "expected externally-tagged shape, got: {}",
            json
        );
        // Inner fields are visible at the inner object level.
        assert!(json.contains("\"key\":\"net.inet.tcp.delayed_ack\""));
        assert!(json.contains("\"value\":\"0\""));
        assert!(json.contains("\"reason\":\"tune\""));

        // Round-trip back through deserialize to confirm bidirectional
        // shape stability.
        let back: RootAction = serde_json::from_str(&json).expect("deserialize");
        match back {
            RootAction::SetSysctl(s) => {
                assert_eq!(s.key(), "net.inet.tcp.delayed_ack");
                assert_eq!(s.value(), "0");
                assert_eq!(s.reason(), "tune");
            }
            _ => panic!("round-trip yielded wrong variant"),
        }
    }

    #[test]
    fn identity_fields_toggle_spotlight_returns_none() {
        let a = RootAction::ToggleSpotlight {
            enabled: false,
            reason: "off".into(),
            decision_reason: DecisionReason::PressureContext,
        };
        assert_eq!(a.identity_fields(), None);
    }

    #[test]
    fn identity_fields_quarantine_returns_none() {
        let a = RootAction::QuarantineDaemon {
            daemon: "mds".into(),
            active: true,
            reason: "noisy".into(),
            decision_reason: DecisionReason::PressureContext,
        };
        assert_eq!(a.identity_fields(), None);
    }

    // ── Sticky counter rename (Sprint 4 Fase 3) ─────────────────────────────

    #[test]
    fn survival_mode_entry_count_default_zero_and_helper_false() {
        let m = RuntimeMetrics::default();
        assert_eq!(m.survival_mode_entry_count, 0);
        assert!(!m.ever_entered_survival_mode());
    }

    #[test]
    fn ever_entered_survival_mode_true_after_one_entry() {
        let mut m = RuntimeMetrics::default();
        m.survival_mode_entry_count = 1;
        assert!(m.ever_entered_survival_mode());
        m.survival_mode_entry_count = 42;
        assert!(m.ever_entered_survival_mode());
    }

    #[test]
    fn round_trip_keeps_count_under_legacy_json_key() {
        // End-to-end: serialize a RuntimeMetrics with the new field name,
        // deserialize the resulting JSON back, count survives. This proves
        // serde rename is bidirectional — runtime_metrics.json round trips
        // without losing the survival counter.
        let mut m = RuntimeMetrics::default();
        m.survival_mode_entry_count = 7;
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("\"survival_mode_activations\":7"));
        let parsed: RuntimeMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.survival_mode_entry_count, 7);
        assert!(parsed.ever_entered_survival_mode());
    }

    #[test]
    fn serializes_under_legacy_json_key_for_consumers() {
        // Producer side: writers (runtime_metrics.json, ctl status JSON)
        // emit the legacy key so existing tools (rm_u lookups in
        // intelligence_score.rs, dashboards, external scripts) keep working.
        let mut m = RuntimeMetrics::default();
        m.survival_mode_entry_count = 5;
        let json = serde_json::to_string(&m).unwrap();
        assert!(
            json.contains("\"survival_mode_activations\":5"),
            "expected legacy key in serialization, got: {}",
            json
        );
        assert!(
            !json.contains("\"survival_mode_entry_count\":"),
            "new field name leaked into JSON; will break rm_u callers"
        );
    }
}
