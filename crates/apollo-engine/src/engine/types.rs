use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use crate::engine::usage_model::{UsageEntrySummary, UsageTopReport};

/// Centralized utility for hardened file system operations to prevent TOCTOU and symlink attacks.
///
/// Cross-crate visibility: required by apollo-optimizerd and
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
#[derive(Default)]
pub enum OptimizationProfile {
    #[default]
    BalancedRoot,
    AggressiveRoot,
    SafeRoot,
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
#[derive(Default)]
pub enum LatencyTarget {
    Low,
    #[default]
    Normal,
    Max,
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

    /// Returns a SafetyPolicy whose budgets are modulated by current memory pressure.
    ///
    /// When pressure > 0.70 (high): budgets increase to handle more aggressive load.
    /// When pressure < 0.40 (low): budgets decrease to avoid unnecessary overhead.
    /// When pressure is moderate, budgets are unchanged.
    ///
    /// This lets Apollo self-regulate: more headroom, more actions possible.
    pub fn with_pressure_modulation(base: &Self, pressure: f64) -> Self {
        let scale = if pressure > 0.70 {
            1.25
        } else if pressure < 0.40 {
            0.7
        } else {
            1.0
        };
        Self {
            max_boosts_per_cycle: ((base.max_boosts_per_cycle as f64 * scale).round() as usize)
                .max(1),
            max_throttles_per_cycle: ((base.max_throttles_per_cycle as f64 * scale).round()
                as usize)
                .max(1),
            max_paging_hints_per_cycle: ((base.max_paging_hints_per_cycle as f64 * scale).round()
                as usize)
                .max(1),
            max_freezes_per_cycle: ((base.max_freezes_per_cycle as f64 * scale).round() as usize)
                .max(1),
            max_sysctl_writes_per_cycle: ((base.max_sysctl_writes_per_cycle as f64 * scale).round()
                as usize)
                .max(1),
            cooldown_seconds: base.cooldown_seconds,
            max_thread_qos_per_cycle: ((base.max_thread_qos_per_cycle as f64 * scale).round()
                as usize)
                .max(1),
        }
    }

    pub fn for_profile(profile: OptimizationProfile) -> Self {
        match profile {
            OptimizationProfile::AggressiveRoot => Self {
                max_boosts_per_cycle: 10,
                max_throttles_per_cycle: 30,
                max_paging_hints_per_cycle: 20,
                max_freezes_per_cycle: 12,
                max_sysctl_writes_per_cycle: 8,
                cooldown_seconds: 10,
                max_thread_qos_per_cycle: 20,
            },
            OptimizationProfile::SafeRoot => Self {
                max_boosts_per_cycle: 3,
                max_throttles_per_cycle: 8,
                max_paging_hints_per_cycle: 5,
                max_freezes_per_cycle: 3,
                max_sysctl_writes_per_cycle: 2,
                cooldown_seconds: 45,
                max_thread_qos_per_cycle: 4,
            },
            OptimizationProfile::BalancedRoot => Self {
                max_boosts_per_cycle: 6,
                max_throttles_per_cycle: 18,
                max_paging_hints_per_cycle: 15,
                max_freezes_per_cycle: 8,
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
    /// Whether the legacy per-PID memory-pressure sysctl exists on this kernel.
    #[serde(default)]
    pub can_memory_pressure_send: bool,
    pub can_mdutil: bool,
    pub can_tmutil: bool,
    pub is_root: bool,
    pub p_core_count: Option<u32>,
    pub e_core_count: Option<u32>,
    pub unavailable: Vec<String>,
    /// Result of a live memorystatus_control write probe: `Some("ok")`,
    /// `Some("fail: ...")`, or `None` when skipped (not root).
    #[serde(default)]
    pub memorystatus_probe: Option<String>,
    /// Result of a live task_for_pid probe on the daemon's own PID:
    /// `Some("ok")`, `Some("fail: ...")`, or `None` when skipped.
    #[serde(default)]
    pub task_for_pid_probe: Option<String>,
}

// Action surface (`RootAction`, `SetSysctlAction`) lives in
// `action_types.rs` since 2026-06-11 (graphify C0 god-file trace).
// Re-exported here so every existing `types::RootAction` path keeps working.
pub use crate::engine::action_types::{RootAction, SetSysctlAction};

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
    /// Phase 5.3 — structured, machine-readable explanation of *why* the
    /// action was taken. Optional and `#[serde(default)]` so journal lines
    /// written by older daemons (no `rationale` key) still deserialize
    /// cleanly. See [`crate::engine::audit_types::Rationale`].
    ///
    /// `skip_serializing_if = "Option::is_none"` keeps unrationale'd entries
    /// byte-compatible with the prior schema — important because journals
    /// are tail-friendly artifacts shared with dashboards.
    ///
    /// Phase 5.3 wiring CLOSED (2026-05-16, verified 2026-06-11): the
    /// cross-cutting wiring promised by the original TODO is done. Every
    /// `JournalEntry` is constructed at exactly ONE site —
    /// `execute_actions.rs` cycle-wide chokepoint (the `pending_journal.push`
    /// at ~line 1345). That chokepoint builds a `Rationale` from the action's
    /// own `(action_class, decision_reason, reason)` tuple for every
    /// successful, non-skip action and bumps
    /// `LSE_COUNTERS.inc_journal_rationale_attached()`. The original
    /// follow-up items #2 (`chromium_manager` long-idle freeze) and #3
    /// (`decide_actions` swarm/graduated-idle rules) are subsumed: those
    /// paths do NOT write journals directly — they emit `RootAction`s that
    /// flow through `execute_actions`, so they inherit the chokepoint's
    /// rationale automatically. A repo-wide audit confirms the chokepoint is
    /// the only non-test `JournalEntry { .. }` construction in the tree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<crate::engine::audit_types::Rationale>,
}

impl JournalEntry {
    /// Attach a rationale to this entry, consuming and returning it for
    /// fluent-chain use at journal write sites. The rationale is
    /// constructed once per action via `Rationale::new(...)` and threaded
    /// through `.with_rationale(...)` so callers stay readable.
    pub fn with_rationale(mut self, rationale: crate::engine::audit_types::Rationale) -> Self {
        self.rationale = Some(rationale);
        self
    }
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
#[serde(default)]
pub struct RuntimeMetrics {
    pub cycles: u64,
    pub boosts_applied: u64,
    pub throttles_applied: u64,
    pub freezes_applied: u64,
    pub unfreezes_applied: u64,
    pub paging_hints_applied: u64,
    pub sysctl_applied: u64,
    pub failures: u64,
    #[serde(default)]
    pub privilege_boundary_mode: String,
    #[serde(default)]
    pub privileged_action_requests_total: u64,
    #[serde(default)]
    pub privileged_action_rejections_total: u64,
    pub last_error: Option<String>,
    pub last_cycle_at: Option<DateTime<Utc>>,
    /// p95 wall time of Apollo's own control loop. Kept separate from
    /// machine/user responsiveness so the dashboard cannot conflate them.
    #[serde(default)]
    pub daemon_cycle_p95_ms: f64,
    pub p95_cycle_ms: f64,
    /// p95 of the bounded cntvct_el0 scheduler-jitter probe. This is a
    /// process-responsiveness proxy, not a browser or HID latency metric.
    #[serde(default)]
    pub scheduler_jitter_p95_ms: f64,
    #[serde(default)]
    pub scheduler_jitter_samples: u64,
    /// Composite local responsiveness signal in [0,1], where lower is better.
    #[serde(default)]
    pub perceptual_latency_score: f64,
    #[serde(default)]
    pub perceptual_latency_category: String,
    #[serde(default)]
    pub perceptual_latency_source: String,
    /// Reserved for direct local HID/browser bridges. None means unmeasured;
    /// Apollo must never synthesize these values from daemon timing.
    #[serde(default)]
    pub user_interaction_p95_ms: Option<f64>,
    #[serde(default)]
    pub browser_lcp_p95_ms: Option<f64>,
    #[serde(default)]
    /// Tail of individual `PerformanceEventTiming.duration` values above the
    /// collector's 40 ms threshold — **not** INP. INP is a high percentile over
    /// interactions grouped by `interactionId`; this is a max over single
    /// entries, blind to fast interactions and rounded to 8 ms by the browser.
    /// Renamed from `browser_inp_p95_ms` (schema discontinuity: the two series
    /// are not comparable and must not be joined).
    pub browser_event_duration_tail_ms: Option<f64>,
    /// True INP estimate. Populated only once the collector groups by
    /// `interactionId`; `None` means the corrected path has not reported yet.
    #[serde(default)]
    pub browser_inp_estimate_ms: Option<f64>,
    /// Interactions (not entries) behind `browser_inp_estimate_ms`.
    #[serde(default)]
    pub browser_interaction_samples: u64,
    /// Interactions the bounded collector refused to track.
    #[serde(default)]
    pub browser_interactions_dropped: u64,
    /// Where interaction time goes. Cumulative milliseconds across the observed
    /// interactions — the split, not the magnitude, is the diagnostic: OS-level
    /// actuators can only plausibly move input delay and presentation.
    #[serde(default)]
    pub browser_input_delay_total_ms: u64,
    #[serde(default)]
    pub browser_processing_total_ms: u64,
    #[serde(default)]
    pub browser_presentation_total_ms: u64,
    /// Transport cost, browser clock, content script → native post. Contains
    /// the MV3 service-worker cold start. This is the measurement that decides
    /// whether any per-interaction fast path could ever exist.
    #[serde(default)]
    pub webflow_transport_client_p95_ms: Option<f64>,
    #[serde(default)]
    pub webflow_transport_sw_wake_p95_ms: Option<f64>,
    #[serde(default)]
    pub webflow_transport_cold_starts: u64,
    #[serde(default)]
    pub webflow_transport_samples: u64,
    #[serde(default)]
    pub webflow_transport_max_queue_depth: u32,
    /// Producer negotiation. Exists so a zero interaction count always carries
    /// its cause: a stale v1 extension, an uninterpretable schema and a silent
    /// browser are otherwise indistinguishable.
    #[serde(default)]
    pub webflow_extension_status: String,
    #[serde(default)]
    pub webflow_extension_version: String,
    #[serde(default)]
    pub webflow_accepted_v1_total: u64,
    #[serde(default)]
    pub webflow_accepted_v2_total: u64,
    #[serde(default)]
    pub webflow_schema_rejected_total: u64,
    #[serde(default)]
    pub webflow_last_event_at_ms: u64,
    #[serde(default)]
    pub webflow_capabilities_bits: u32,
    /// Phase 0B observational state. None of it grants causal authority: these
    /// are counts of what was seen, and the classification names what the
    /// window supports saying rather than what caused it.
    #[serde(default)]
    pub perceptual_episodes_stored: u64,
    #[serde(default)]
    pub perceptual_episodes_rejected: u64,
    #[serde(default)]
    pub perceptual_correlation_unique: u64,
    #[serde(default)]
    pub perceptual_correlation_ambiguous: u64,
    #[serde(default)]
    pub perceptual_correlation_unmatched: u64,
    #[serde(default)]
    pub perceptual_regimes_total: u64,
    #[serde(default)]
    pub perceptual_regime_class: String,
    #[serde(default)]
    pub perceptual_regime_level: String,
    #[serde(default)]
    pub perceptual_regime_interactions: u32,
    #[serde(default)]
    pub perceptual_regime_actionable: u32,
    #[serde(default)]
    pub perceptual_regime_dominant_family: String,
    #[serde(default)]
    pub perceptual_regime_median_total_ms: u32,
    /// Source-agnostic perceptual layer. These describe observations from every
    /// producer; per-source figures like INP stay in their own adapter section
    /// because no other source can produce them.
    #[serde(default)]
    pub perceptual_sources_active: u32,
    /// Lifetime count, persisted across restarts. Distinct from
    /// `perceptual_store_len`, which is how many are resident right now: a
    /// bounded store means the two diverge, and reading the lifetime figure as
    /// the store size makes a restart look like it gained observations.
    #[serde(default)]
    pub perceptual_observations_total: u64,
    #[serde(default)]
    pub perceptual_store_len: u32,
    #[serde(default)]
    pub perceptual_store_capacity: u32,
    #[serde(default)]
    pub perceptual_instrumented_total: u64,
    #[serde(default)]
    pub perceptual_inferred_total: u64,
    #[serde(default)]
    pub perceptual_windows_total: u64,
    #[serde(default)]
    pub perceptual_valid_total: u64,
    #[serde(default)]
    pub perceptual_invalid_total: u64,
    /// Mean overall quality on a 0..=1000 scale, weakest-link per observation.
    ///
    /// Aggregate across every modality, so it moves with the evidence *mix*:
    /// a burst of low-precision windows pulls it down while instrumented
    /// episodes are unchanged. Prefer the per-modality figures below when
    /// judging whether measurement actually got worse.
    #[serde(default)]
    pub perceptual_quality_q: u16,
    /// Per-modality quality with the resident count behind each figure.
    /// `None` means that modality has no resident observations — absence, not
    /// a quality of zero. Counts are resident, so they pair with the means
    /// above and never with the lifetime `*_total` counters.
    #[serde(default)]
    pub perceptual_quality_instrumented_q: Option<u16>,
    #[serde(default)]
    pub perceptual_quality_instrumented_n: u32,
    #[serde(default)]
    pub perceptual_quality_inferred_q: Option<u16>,
    #[serde(default)]
    pub perceptual_quality_inferred_n: u32,
    #[serde(default)]
    pub perceptual_quality_window_q: Option<u16>,
    #[serde(default)]
    pub perceptual_quality_window_n: u32,
    /// Stratified observational association. Correlation, never causation: the
    /// dashboard prints CAUSAL UNTESTED beside it for exactly that reason.
    #[serde(default)]
    pub perceptual_assoc_family: String,
    #[serde(default)]
    pub perceptual_assoc_verdict: String,
    #[serde(default)]
    pub perceptual_assoc_delta_ms: i32,
    #[serde(default)]
    pub perceptual_assoc_component: String,
    #[serde(default)]
    pub perceptual_assoc_samples_with: u32,
    #[serde(default)]
    pub perceptual_assoc_samples_without: u32,
    #[serde(default)]
    pub perceptual_assoc_confidence_q: u16,
    #[serde(default)]
    pub perceptual_assoc_actionable: bool,
    /// One sanitized record per modality, as JSON. Bounded to two by
    /// construction, carries hashes and closed categories only, and exists so
    /// the shape reaching the store can be inspected without a new RPC.
    #[serde(default)]
    pub perceptual_sample_instrumented: String,
    #[serde(default)]
    pub perceptual_sample_window: String,
    /// Samples backing the LCP/INP p95 above. A p95 over three samples and a
    /// p95 over a full window are very different claims, and reading one as
    /// the other already produced a bogus 120000ms LCP once.
    #[serde(default)]
    pub browser_latency_samples: u64,
    /// Cycle-local decision events dropped by bounded producer/merge buffers.
    #[serde(default)]
    pub decision_event_drops_total: u64,
    /// Auditable synthetic close receipts emitted when a bounded buffer drops.
    #[serde(default)]
    pub decision_event_overflow_summaries_total: u64,
    /// Latest resolved actuator episode, exported as an explanation rather
    /// than as authority. The evidence tier and quality remain the gate.
    #[serde(default)]
    pub last_episode_id: u64,
    #[serde(default)]
    pub last_episode_resolved_cycle: u64,
    #[serde(default)]
    pub last_episode_action: String,
    #[serde(default)]
    pub last_episode_target: String,
    #[serde(default)]
    pub last_episode_tier: String,
    #[serde(default)]
    pub last_episode_quality: f64,
    #[serde(default)]
    pub last_episode_utility: f64,
    #[serde(default)]
    pub last_episode_latency_improvement: f64,
    #[serde(default)]
    pub last_episode_system_gain: f64,
    #[serde(default)]
    pub last_episode_human_gain: f64,
    #[serde(default)]
    pub last_episode_intervention_cost: f64,
    #[serde(default)]
    pub last_episode_apollo_utility: f64,
    #[serde(default)]
    pub last_episode_proposer: String,
    #[serde(default)]
    pub last_episode_supporters: Vec<String>,
    #[serde(default)]
    pub last_episode_vetoes: Vec<String>,
    #[serde(default)]
    pub last_episode_predicted_gain: f64,
    /// Immutable scalar projection of the cached unified-learning view.
    #[serde(default)]
    pub unified_learning_schema_version: u8,
    #[serde(default)]
    pub decision_ledger_unattributed_applied_total: u64,
    #[serde(default)]
    pub ledger_closure: crate::engine::unified_learning_health::LedgerClosureSnapshot,
    #[serde(default)]
    pub trust_inventory: crate::engine::unified_learning_health::TrustInventorySnapshot,
    #[serde(default)]
    pub horizon_calibration:
        Vec<crate::engine::unified_learning_health::HorizonCalibrationSnapshot>,
    #[serde(default)]
    pub hierarchy_learning: crate::engine::unified_learning_health::HierarchyLearningSnapshot,
    #[serde(default)]
    pub exploration_learning: crate::engine::unified_learning_health::ExplorationLearningSnapshot,
    #[serde(default)]
    pub latest_resolved_episode:
        crate::engine::unified_learning_health::LatestResolvedEpisodeSnapshot,
    #[serde(default)]
    pub unified_learning_advice_overflow_total: u64,
    #[serde(default)]
    pub ais_local_learning_maturity: f64,
    #[serde(default)]
    pub ais_unified_learning_evidence: f64,
    #[serde(default)]
    pub unified_learning_ais: crate::engine::unified_learning_health::AisLearningSnapshot,
    /// Adaptive self-overhead governor. Critical sensing, execution and
    /// learning remain active; only optional refresh/speculation is shed.
    #[serde(default)]
    pub apollo_overhead_level: String,
    #[serde(default)]
    pub apollo_overhead_full_refresh_cadence: u64,
    #[serde(default)]
    pub apollo_overhead_reason_budget_ms: u64,
    #[serde(default)]
    pub apollo_overhead_speculation_allowed: bool,
    #[serde(default)]
    pub apollo_overhead_constrained_cycles: u64,
    #[serde(default)]
    pub apollo_overhead_optional_skips_total: u64,
    #[serde(default)]
    pub apollo_overhead_sensor_interval_secs: u64,
    #[serde(default)]
    pub apollo_overhead_holt_cadence: u64,
    #[serde(default)]
    pub apollo_overhead_page_reclaim_cadence: u64,
    #[serde(default)]
    pub apollo_overhead_holt_skips_total: u64,
    #[serde(default)]
    pub apollo_overhead_page_reclaim_skips_total: u64,
    /// Option C global optional-work scheduler. Shadow is observational and
    /// cannot alter legacy execution or grant action authority.
    #[serde(default)]
    pub value_scheduler_phase: String,
    #[serde(default)]
    pub value_scheduler_blocker: String,
    #[serde(default)]
    pub value_scheduler_valid_cycles: u64,
    #[serde(default)]
    pub value_scheduler_shadow_cycles: u64,
    #[serde(default)]
    pub value_snapshot_epoch: u64,
    #[serde(default)]
    pub value_snapshot_revision: u64,
    #[serde(default)]
    pub value_scheduler_registered_jobs: u64,
    #[serde(default)]
    pub value_scheduler_eligible_jobs: u64,
    #[serde(default)]
    pub value_scheduler_selected_jobs: u64,
    #[serde(default)]
    pub value_scheduler_selected_total: u64,
    #[serde(default)]
    pub value_scheduler_budget_us: u64,
    #[serde(default)]
    pub value_scheduler_predicted_us: u64,
    #[serde(default)]
    pub value_scheduler_selection_latency_us: u64,
    #[serde(default)]
    pub value_scheduler_max_selection_latency_us: u64,
    #[serde(default)]
    pub value_scheduler_budget_skips_total: u64,
    #[serde(default)]
    pub value_scheduler_capacity_skips_total: u64,
    #[serde(default)]
    pub value_scheduler_invalid_samples_total: u64,
    #[serde(default)]
    pub value_scheduler_invalid_unhealthy_total: u64,
    #[serde(default)]
    pub value_scheduler_invalid_sequence_total: u64,
    #[serde(default)]
    pub value_scheduler_invalid_features_total: u64,
    #[serde(default)]
    pub value_scheduler_invalid_publication_total: u64,
    /// Which gate refused the control sample. Sums to
    /// `value_scheduler_invalid_unhealthy_total` with no unclassified residue.
    #[serde(default)]
    pub value_scheduler_unhealthy_pressure_total: u64,
    #[serde(default)]
    pub value_scheduler_unhealthy_thermal_total: u64,
    #[serde(default)]
    pub value_scheduler_unhealthy_profile_total: u64,
    #[serde(default)]
    pub value_scheduler_unhealthy_p95_total: u64,
    #[serde(default)]
    pub value_scheduler_unhealthy_latency_total: u64,
    #[serde(default)]
    pub value_scheduler_unhealthy_kill_switch_total: u64,
    #[serde(default)]
    pub value_scheduler_unhealthy_sleeping_total: u64,
    /// Operational heterogeneous-compute telemetry. These fields are not AIS
    /// evidence and cannot authorize actions.
    #[serde(default)]
    pub fabric_phase: String,
    #[serde(default)]
    pub fabric_blocker: String,
    #[serde(default)]
    pub fabric_workers_active: u64,
    #[serde(default)]
    pub fabric_qos_failures: u64,
    #[serde(default)]
    pub fabric_result_drops: u64,
    #[serde(default)]
    pub fabric_submitted_total: u64,
    #[serde(default)]
    pub fabric_completed_total: u64,
    #[serde(default)]
    pub fabric_cancelled_total: u64,
    #[serde(default)]
    pub fabric_stale_total: u64,
    #[serde(default)]
    pub fabric_deadline_misses_total: u64,
    #[serde(default)]
    pub fabric_dispatch_skips_total: u64,
    #[serde(default)]
    pub fabric_eligible_total: u64,
    #[serde(default)]
    pub fabric_evaluation_total: u64,
    #[serde(default)]
    pub fabric_last_latency_us: u64,
    #[serde(default)]
    pub fabric_control_p95_baseline_ms: f64,
    #[serde(default)]
    pub fabric_cpu_percent: f64,
    #[serde(default)]
    pub fabric_rss_delta_bytes: u64,
    #[serde(default)]
    pub coreml_model_available: bool,
    #[serde(default)]
    pub coreml_requested_backend: String,
    /// Compute units Core ML accepted at model load — a *configuration*.
    /// Named `effective` until it was found to read as "where inference ran",
    /// which no macOS API reports. See `coreml_ane_observation` for what is
    /// actually known about execution.
    #[serde(default)]
    pub coreml_configured_backend: String,
    /// Execution-evidence status for the Neural Engine, one of
    /// `unsupported` / `unavailable` / `measured-idle` / `measured-active`.
    ///
    /// Replaces a `bool` that could not tell "we never implemented the
    /// observation" apart from "we measured it and the ANE stayed idle". Core
    /// ML routes each inference itself without publishing the unit it chose,
    /// so `unsupported` is the honest steady state on this platform, and a
    /// consumer must never read it as evidence either way.
    #[serde(default)]
    pub coreml_ane_observation: String,
    #[serde(default)]
    pub coreml_circuit_state: String,
    #[serde(default)]
    pub temporal_prediction_backend: String,
    #[serde(default)]
    pub temporal_prediction_load: f64,
    #[serde(default)]
    pub temporal_prediction_transition: f64,
    #[serde(default)]
    pub temporal_prediction_pressure: f64,
    #[serde(default)]
    pub temporal_prediction_p95: f64,
    #[serde(default)]
    pub temporal_prediction_authoritative: bool,
    /// Paired microexperiment lab. Shadow diagnostics never mutate effects;
    /// Pair Gold counts only fully closed local control/treatment pairs.
    #[serde(default)]
    pub microexperiment_phase: String,
    #[serde(default)]
    pub microexperiment_rollout_progress: u64,
    #[serde(default)]
    pub microexperiment_rollout_required: u64,
    /// Boot-scoped provenance for `microexperiment_rollout_progress`. A gate
    /// counter that fell backwards across a restart cannot be diagnosed from
    /// the counter alone: these separate "the persisted value was low" from
    /// "a runtime reset discarded it", and name the reset reason.
    #[serde(default)]
    pub microexperiment_restored_progress_at_boot: u64,
    #[serde(default)]
    pub microexperiment_progress_resets_total: u64,
    #[serde(default)]
    pub microexperiment_last_progress_reset_reason: String,
    #[serde(default)]
    pub microexperiment_blocker: String,
    #[serde(default)]
    pub microexperiment_restore: String,
    #[serde(default)]
    pub microexperiment_proposed_total: u64,
    #[serde(default)]
    pub microexperiment_eligible_total: u64,
    #[serde(default)]
    pub microexperiment_randomized_total: u64,
    #[serde(default)]
    pub microexperiment_shadow_would_open_total: u64,
    #[serde(default)]
    pub microexperiment_open_pairs: u64,
    /// EVIDENCE. Shadow pairs closed on a valid observed control endpoint —
    /// the quantity that actually gates promotion out of Shadow. Distinct from
    /// `microexperiment_shadow_would_open_total`, which is exposure.
    #[serde(default)]
    pub microexperiment_shadow_measurements_proven_total: u64,
    #[serde(default)]
    pub microexperiment_shadow_measurements_refused_total: u64,
    /// Shadow pairs reaped for passing their deadline. Without it, whether the
    /// lifecycle fix is working is unobservable from outside the process.
    #[serde(default)]
    pub microexperiment_shadow_pairs_expired_total: u64,
    /// Endpoints that arrived after their pair was terminalised.
    #[serde(default)]
    pub microexperiment_shadow_endpoints_late_total: u64,
    /// Shadow pairs actually removed. Contrast with `..._expired_total`: an
    /// expiry that never becomes a removal is a slot held forever, which is
    /// how the route went silent in the first place.
    #[serde(default)]
    pub microexperiment_shadow_pairs_reaped_total: u64,
    /// Current size of the shadow collection — the quantity a baseline needs
    /// in order to assert the ceiling directly rather than infer it from arms
    /// still being registered.
    #[serde(default)]
    pub microexperiment_shadow_open_pairs: u64,
    /// Largest size seen this boot. A gauge sampled once a cycle can read zero
    /// straight through a burst that touched the ceiling.
    #[serde(default)]
    pub microexperiment_shadow_open_high_watermark: u64,

    /// Certified pairs offered to the lab, and what became of each. If
    /// `canary_pairs_completed` grows and the gate does not, these say whether
    /// the pair never arrived, was rejected, or was a replay.
    #[serde(default)]
    pub microexperiment_lab_pairs_seen: u64,
    #[serde(default)]
    pub microexperiment_lab_pairs_accepted: u64,
    #[serde(default)]
    pub microexperiment_lab_pairs_duplicate: u64,
    #[serde(default)]
    pub microexperiment_lab_pairs_rejected: u64,
    /// Distinct causal experiments counted — the quantity the gate reads.
    #[serde(default)]
    pub microexperiment_causal_pairs_consumed: u64,

    // ── Micro-canary (MarkovPrewarm only, disabled until explicitly enabled) ──
    /// Whether the producer is sampling at all. False means every counter
    /// below is frozen by design and the daemon behaves exactly as baseline.
    #[serde(default)]
    pub canary_enabled: bool,
    /// Opportunities offered, all of them already eligible.
    #[serde(default)]
    pub canary_eligible_seen: u64,
    /// Opportunities drawn into an experiment.
    #[serde(default)]
    pub canary_sampled: u64,
    /// Draws refused because a budget was full, a family is not enabled, or
    /// the kill switch is off. Kept apart so a quiet canary can be told from a
    /// blocked one.
    #[serde(default)]
    pub canary_refused_budget: u64,
    #[serde(default)]
    pub canary_refused_family: u64,
    #[serde(default)]
    pub canary_refused_disabled: u64,
    /// Arms handed to the caller. `control_issued` counts real actions
    /// withheld from the machine — the size of the intervention.
    #[serde(default)]
    pub canary_treatment_issued: u64,
    #[serde(default)]
    pub canary_control_issued: u64,
    /// Controls the caller confirmed it honoured. A gap against
    /// `canary_control_issued` means something was asked to be withheld and
    /// went ahead, which no other counter would reveal.
    #[serde(default)]
    pub canary_control_honoured: u64,
    /// Arms by terminal state.
    #[serde(default)]
    pub canary_arms_applied_reverted: u64,
    #[serde(default)]
    pub canary_arms_applied_no_revert: u64,
    #[serde(default)]
    pub canary_arms_withheld: u64,
    /// Experiments that produced a certified pair.
    #[serde(default)]
    pub canary_pairs_completed: u64,
    /// Experiments abandoned at their horizon with an arm missing.
    #[serde(default)]
    pub canary_pairs_expired: u64,
    #[serde(default)]
    pub canary_endpoints_late: u64,
    #[serde(default)]
    pub canary_endpoints_duplicate: u64,
    /// Pairs the contract refused. Non-zero is a defect in the producer.
    #[serde(default)]
    pub canary_assembly_refused: u64,
    /// Currently open experiments, and the largest number seen this boot.
    #[serde(default)]
    pub canary_open_experiments: u64,
    #[serde(default)]
    pub canary_open_high_watermark: u64,
    /// Observed sampling rate per thousand eligible. The rate that happened,
    /// not the one configured.
    #[serde(default)]
    pub canary_observed_per_mille: f64,
    /// Attrition **by arm**. Excluding an unmeasurable window is correct, but
    /// if one arm loses more than the other, excluding them quietly biases the
    /// effect — the survivors would be the ones that went well. Split so the
    /// loss can be shown non-differential instead of taken on trust.
    #[serde(default)]
    pub canary_treatment_confounded: u64,
    #[serde(default)]
    pub canary_control_confounded: u64,
    #[serde(default)]
    pub canary_treatment_incomplete: u64,
    #[serde(default)]
    pub canary_control_incomplete: u64,
    /// Experiments that left the estimator without contributing an effect.
    #[serde(default)]
    pub canary_pairs_censored: u64,
    /// Pairs no longer open, whatever ended them. Renamed from
    /// `microexperiment_completed_pairs`, which read as "finished" while every
    /// one of the 9 it reported had been interrupted.
    #[serde(default)]
    pub microexperiment_terminal_pairs: u64,
    /// Pairs that closed without interruption — the number to read when asking
    /// whether the lab finished anything.
    #[serde(default)]
    pub microexperiment_completed_pairs_valid: u64,
    #[serde(default)]
    pub microexperiment_interrupted_pairs: u64,
    #[serde(default)]
    pub microexperiment_control_endpoints_total: u64,
    #[serde(default)]
    pub microexperiment_treatment_endpoints_total: u64,
    #[serde(default)]
    pub microexperiment_complete_horizons_total: u64,
    #[serde(default)]
    pub microexperiment_rollback_closed_total: u64,
    #[serde(default)]
    pub microexperiment_pair_gold_total: u64,
    #[serde(default)]
    pub microexperiment_effective_total: u64,
    #[serde(default)]
    pub microexperiment_harmful_total: u64,
    #[serde(default)]
    pub microexperiment_confounded_total: u64,
    #[serde(default)]
    pub microexperiment_interrupted_total: u64,
    #[serde(default)]
    pub microexperiment_synthetic_quarantined_total: u64,
    #[serde(default)]
    pub microexperiment_mean_effect: f64,
    /// Endpoint-wire observability. Every rejection reason has its own counter
    /// so a stalled circuit names its own cause instead of collapsing into one
    /// generic "blocked" total.
    #[serde(default)]
    pub microexperiment_endpoint_contract_ready: bool,
    #[serde(default)]
    pub microexperiment_arms_registered_total: u64,
    #[serde(default)]
    pub microexperiment_episodes_observed_total: u64,
    #[serde(default)]
    pub microexperiment_episodes_skipped_idle_total: u64,
    #[serde(default)]
    pub microexperiment_uncatalogued_episodes_total: u64,
    #[serde(default)]
    pub microexperiment_decisions_bound_total: u64,
    #[serde(default)]
    pub microexperiment_endpoints_emitted_total: u64,
    #[serde(default)]
    pub microexperiment_endpoints_pending_utility: u64,
    #[serde(default)]
    pub microexperiment_endpoint_action_mismatch_total: u64,
    /// Ordinary traffic no experiment was waiting on. Large by nature; a big
    /// number here is not a defect. Was `..._unknown_arm_total`, which made
    /// 7,809 routine decisions read as failures.
    #[serde(default)]
    pub microexperiment_endpoint_routine_unclaimed_total: u64,
    /// An episode for a watched action that could not fill its arm role.
    #[serde(default)]
    pub microexperiment_endpoint_invalid_experimental_total: u64,
    #[serde(default)]
    pub microexperiment_endpoint_duplicate_total: u64,
    #[serde(default)]
    pub microexperiment_endpoint_expired_total: u64,
    #[serde(default)]
    pub microexperiment_endpoint_epoch_rejected_total: u64,
    #[serde(default)]
    pub microexperiment_endpoint_authority_rejected_total: u64,
    #[serde(default)]
    pub microexperiment_endpoint_incomplete_metadata_total: u64,
    #[serde(default)]
    pub microexperiment_endpoint_capacity_drops_total: u64,
    #[serde(default)]
    pub microexperiment_endpoint_arms_expired_total: u64,
    #[serde(default)]
    pub microexperiment_endpoint_rollback_observed_total: u64,
    #[serde(default)]
    pub microexperiment_endpoint_rollback_failed_total: u64,
    #[serde(default)]
    pub microexperiment_control_withholds_total: u64,
    #[serde(default)]
    pub microexperiment_invalidated_total: u64,
    #[serde(default)]
    pub microexperiment_deadline_expired_total: u64,
    /// Arms that expired without ever reporting. Counted apart from canary
    /// failures: absence of data is not adverse data.
    #[serde(default)]
    pub microexperiment_unbound_expiries_total: u64,
    #[serde(default)]
    pub microexperiment_rollback_failed_total: u64,
    #[serde(default)]
    pub metrics_disk_writes_total: u64,
    #[serde(default)]
    pub metrics_snapshot_clone_skips_total: u64,
    pub refresh_duration_ms: f64,
    pub memory_budget_duration_ms: f64,
    pub reactor_duration_ms: f64,
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
    /// Universal Gold outcomes locally consolidated from System 2 into S1.
    #[serde(default)]
    pub local_consolidations: u64,
    #[serde(default)]
    pub local_consolidation_improvements: u64,
    #[serde(default)]
    pub local_consolidation_regressions: u64,
    #[serde(default)]
    pub local_consolidation_neutral: u64,
    #[serde(default)]
    pub local_consolidation_system1_updates: u64,
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
    /// Fault-in rate (pageins+swapins+decompressions pages/sec) — the STALL
    /// side of VM flow, where the microstutter lives. [Phase 0 telemetry]
    #[serde(default)]
    pub refault_delta_per_sec: f64,
    /// Session peak of `refault_delta_per_sec`. A transient storm between
    /// metric flushes would be invisible in the instantaneous field; this
    /// max-since-start captures the worst storm for baseline. [Phase 0]
    #[serde(default)]
    pub refault_peak_per_sec: f64,
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
    /// Apple platform PIDs short-circuited before AdaptiveGovernor action
    /// scoring. These would be rejected by SIP at dispatch time.
    #[serde(default)]
    pub heuristic_apple_platform_protected: u64,
    pub zombies_detected: u64,
    #[serde(default)]
    pub kills_applied: u64,
    /// Metrics history append failures.
    #[serde(default)]
    pub failed_history_writes: u64,
    /// WarmBand pre-stage fires: how many cycles since-daemon-start had a
    /// `warm_pressure_boost > 0.0`. Lets the audit-cron / dashboard verify
    /// that F-06 (sensor-noise / trend-detection reliability) holds in
    /// production: a long stretch of 1000+ cycles with this == 0 means
    /// WarmBand is dead in practice and the commit claim was wrong.
    #[serde(default)]
    pub warm_band_fires: u64,
    /// Cumulative `warm_pressure_boost` × 1000 (avoids f32 atomics). Divide
    /// by 1000 for a per-cycle average; together with `warm_band_fires`
    /// gives the operator a live view of "how hard is the band firing".
    #[serde(default)]
    pub warm_boost_sum_x1000: u64,
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
    /// 24h-windowed count of cycles observed while survival mode active.
    /// THIS is the field AIS D5 `safety_compliance()` reads — NOT the sticky
    /// lifetime counter above. See `survival_window.rs` and CLAUDE.md
    /// Sprint 3 doctrine entry #5 ("sticky > 0 as live state flag"
    /// anti-pattern). Producer: `daemon_survival_tick.rs`.
    #[serde(default)]
    pub survival_activations_recent_24h: u64,
    pub qos_foreground_count: u64,
    pub qos_background_count: u64,
    pub qos_errors: u64,
    /// Audited Mach task-policy routing requests. Foreground/background above
    /// count confirmed kernel mutations only; these fields explain the rest.
    #[serde(default)]
    pub qos_requests_count: u64,
    #[serde(default)]
    pub qos_noop_count: u64,
    #[serde(default)]
    pub qos_blocked_count: u64,
    /// Direct user-interaction QoS leases. These counters make it possible to
    /// verify that boosts both activate and return to their prior tier.
    #[serde(default)]
    pub interaction_qos_activations: u64,
    #[serde(default)]
    pub interaction_qos_reverts: u64,
    #[serde(default)]
    pub interaction_qos_active: bool,
    #[serde(default)]
    pub interaction_qos_reason: String,
    /// Parameter arm used by the current/last interaction lease. The
    /// medallion records this separately so it can learn whether shorter,
    /// standard, or longer leases actually caused the observed utility.
    #[serde(default)]
    pub interaction_qos_ttl_band: String,
    #[serde(default)]
    pub interaction_qos_ttl_ms: u64,
    #[serde(default)]
    pub interaction_qos_ttl_exploratory: bool,
    #[serde(default)]
    pub interaction_qos_parameter_explorations_total: u64,
    /// Exact browser and universal inferred network-flow observability.
    #[serde(default)]
    pub webflow_mode: String,
    #[serde(default)]
    pub webflow_phase: String,
    #[serde(default)]
    pub webflow_blocker: String,
    #[serde(default)]
    pub webflow_valid_health_cycles: u64,
    #[serde(default)]
    pub webflow_active_navigations: u16,
    #[serde(default)]
    pub webflow_proposed_total: u64,
    #[serde(default)]
    pub webflow_admitted_total: u64,
    #[serde(default)]
    pub webflow_skipped_total: u64,
    #[serde(default)]
    pub network_flow_active: bool,
    #[serde(default)]
    pub network_flow_traffic_bps: u64,
    #[serde(default)]
    pub network_flow_confidence_q: u16,
    #[serde(default)]
    pub network_flow_proposed_total: u64,
    #[serde(default)]
    pub network_flow_started_total: u64,
    #[serde(default)]
    pub network_flow_renewed_total: u64,
    #[serde(default)]
    pub network_flow_skipped_total: u64,
    #[serde(default)]
    pub network_flow_suppressed_exact_total: u64,
    #[serde(default)]
    pub network_flow_hard_cap_total: u64,
    /// Member-level observability for the bounded acceleration lease broker.
    /// The legacy interaction counters above remain the actuator-learning
    /// contract consumed by AIS and the telemetry medallion.
    #[serde(default)]
    pub acceleration_lease_members_active: u32,
    #[serde(default)]
    pub acceleration_lease_members_applied_total: u64,
    /// Lease members that received a confirmed TASK_POLICY_QOS mutation.
    /// Members accelerated only through the `nice` fallback are excluded.
    #[serde(default)]
    pub acceleration_lease_task_qos_applied_total: u64,
    #[serde(default)]
    pub acceleration_lease_member_reverts_total: u64,
    #[serde(default)]
    pub acceleration_lease_renewals_total: u64,
    #[serde(default)]
    pub acceleration_lease_chromium_total: u64,
    #[serde(default)]
    pub acceleration_lease_general_total: u64,
    #[serde(default)]
    pub acceleration_lease_identity_skips_total: u64,
    #[serde(default)]
    pub acceleration_lease_capability_skips_total: u64,
    #[serde(default)]
    pub acceleration_lease_task_port_denied_total: u64,
    #[serde(default)]
    pub acceleration_lease_qos_write_rejected_total: u64,
    /// Last kern_return text from a rejected task QoS write, so the cause is
    /// diagnosable without journal access.
    #[serde(default)]
    pub acceleration_lease_qos_write_error: String,
    #[serde(default)]
    pub acceleration_lease_nice_fallbacks_total: u64,
    #[serde(default)]
    pub acceleration_lease_nice_failures_total: u64,
    #[serde(default)]
    pub acceleration_lease_conflict_skips_total: u64,
    #[serde(default)]
    pub acceleration_lease_io_promotions_total: u64,
    #[serde(default)]
    pub acceleration_lease_family: String,
    #[serde(default)]
    pub acceleration_lease_last_family: String,
    /// Typed, reversible fast-lane observability. These counters describe the
    /// reflex route itself; legacy acceleration counters above remain stable.
    #[serde(default)]
    pub reflex_enabled: bool,
    #[serde(default)]
    pub reflex_phase: String,
    #[serde(default)]
    pub reflex_blocker: String,
    #[serde(default)]
    pub reflex_shadow_cycles: u64,
    #[serde(default)]
    pub reflex_valid_cycles: u64,
    #[serde(default)]
    pub reflex_invalid_samples_total: u64,
    #[serde(default)]
    pub reflex_baseline_p95_ms: f64,
    #[serde(default)]
    pub reflex_candidate_p95_ms: f64,
    #[serde(default)]
    pub reflex_baseline_churn: f64,
    #[serde(default)]
    pub reflex_candidate_churn: f64,
    #[serde(default)]
    pub reflex_proposed_total: u64,
    #[serde(default)]
    pub reflex_admitted_total: u64,
    #[serde(default)]
    pub reflex_applied_total: u64,
    #[serde(default)]
    pub reflex_shadowed_total: u64,
    #[serde(default)]
    pub reflex_omitted_total: u64,
    #[serde(default)]
    pub reflex_noop_total: u64,
    #[serde(default)]
    pub reflex_vetoed_total: u64,
    #[serde(default)]
    pub reflex_reverted_total: u64,
    #[serde(default)]
    pub reflex_failed_total: u64,
    #[serde(default)]
    pub reflex_protected_blocked_total: u64,
    #[serde(default)]
    pub reflex_last_decision_latency_us: u64,
    #[serde(default)]
    pub reflex_reasoning_submitted_total: u64,
    #[serde(default)]
    pub reflex_reasoning_dropped_total: u64,
    #[serde(default)]
    pub reflex_reasoning_completed_total: u64,
    #[serde(default)]
    pub reflex_reasoning_failed_total: u64,
    #[serde(default)]
    pub reflex_reasoning_deadline_misses_total: u64,
    #[serde(default)]
    pub reflex_reasoning_identity_mismatches_total: u64,
    #[serde(default)]
    pub reflex_reasoning_last_result_age_cycles: u64,
    #[serde(default)]
    pub reflex_reasoning_last_latency_us: u64,
    pub iokit_snapshots: u64,
    pub iokit_errors: u64,
    pub iokit_p_cluster_temp: Option<f32>,
    pub iokit_e_cluster_temp: Option<f32>,
    pub iokit_package_watts: Option<f32>,
    pub current_workload: String,
    /// Hardware-local phase timing. This starts fresh on each daemon launch so
    /// imported M1 state cannot bias M4 duration estimates.
    #[serde(default)]
    pub workload_phase: String,
    #[serde(default)]
    pub workload_phase_expected_duration_secs: f64,
    #[serde(default)]
    pub workload_phase_duration_observations: u64,
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
    pub energy_co2_emitted_g: Option<f64>,
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
    // Network monitor metrics. The compatibility field name says "ratio",
    // but its unit is retransmissions per 1,000 TCP segments.
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
    pub parallel_runtime_enabled: bool,
    #[serde(default)]
    pub parallel_worker_threads: usize,
    #[serde(default)]
    pub parallel_expected_profile: String,
    #[serde(default)]
    pub parallel_compiled_profile: String,
    #[serde(default)]
    pub parallel_effective_profile: String,
    #[serde(default)]
    pub parallel_feature_compiled: bool,
    #[serde(default)]
    pub parallel_max_worker_threads: usize,
    #[serde(default)]
    pub parallel_disabled_reason: String,
    #[serde(default)]
    pub parallel_worker_qos_intent: String,
    #[serde(default)]
    pub parallel_worker_qos_status: String,
    #[serde(default)]
    pub parallel_worker_qos_failures: u64,
    #[serde(default)]
    pub journal_rotations: u64,
    #[serde(default)]
    pub journal_rotation_failures: u64,
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
    pub predictive_agent_arm_avg_rewards: [f64; 5],
    #[serde(default)]
    pub predictive_agent_last_intervention: String,
    #[serde(default)]
    pub predictive_agent_pending_outcomes: usize,
    #[serde(default)]
    pub predictive_agent_resolved_outcomes: u64,
    // Signal intelligence metrics
    #[serde(default)]
    pub si_pressure_smooth: f64,
    #[serde(default)]
    pub si_pressure_velocity: f64,
    #[serde(default)]
    pub si_p_oom_30s: f64,
    #[serde(default)]
    pub si_p_jank_60s: f64,
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
    ///
    /// Invariant (post-ffa0b29): Σ(typed per-variant) == total_pushed.
    /// `actions_pushed_raw_total` is an INDEPENDENT diagnostic of escape-hatch
    /// volume — it is a SUBSET of the typed totals (every raw push also bumps
    /// the matching per-variant counter so a raw BoostProcess still moves the
    /// boost counter). Dashboards must NOT add raw to the typed sum.
    ///
    /// DO NOT compute Σ(typed) + raw — this double-counts every escape-hatch
    /// emission and inflates dispatcher volume by the raw fraction.
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
    /// Final dispatch prefilter telemetry. These counters sit immediately
    /// before action budgeting so rejected Apple/protected/stale candidates
    /// cannot consume the minute budget and starve executable work.
    #[serde(default)]
    pub dispatch_prefilter_input_total: u64,
    #[serde(default)]
    pub dispatch_prefilter_recent_drops: u64,
    #[serde(default)]
    pub dispatch_prefilter_apple_drops: u64,
    #[serde(default)]
    pub dispatch_prefilter_protected_drops: u64,
    #[serde(default)]
    pub dispatch_prefilter_identity_drops: u64,
    #[serde(default)]
    pub dispatch_prefilter_unsupported_drops: u64,
    #[serde(default)]
    pub dispatch_prefilter_survivors_total: u64,
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
    /// Whether IOReport produced a snapshot at all this cycle.
    ///
    /// IOReport is deprecated on macOS 26+, where the subscription requires
    /// Apple-private entitlements a third-party binary cannot hold. Without a
    /// snapshot the cluster and GPU percentages below are written as zeros —
    /// this is what separates those placeholders from a genuinely idle
    /// machine, and no consumer should read them without checking it.
    #[serde(default)]
    pub ioreport_available: bool,
    /// IOReport's view of the Neural Engine as an explicit status:
    /// `unavailable` (no ANE channel in this subscription), `measured-idle`,
    /// or `measured-active`.
    ///
    /// `ioreport_ane_busy` alone keeps its `false` default when no channel is
    /// published, so it reads identically to a genuinely idle accelerator.
    #[serde(default)]
    pub ioreport_ane_observation: String,
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
    pub kpc_available: bool,
    #[serde(default)]
    pub kpc_ipc: f64,
    /// Heuristic memory-bound proxy derived from achieved IPC.
    /// This is not a hardware stall counter and remains telemetry-only.
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
    /// Non-urgent actions rejected because the bounded action queue was full.
    #[serde(default)]
    pub action_queue_capacity_drops: u64,
    /// Background actions displaced to admit higher-priority normal work.
    #[serde(default)]
    pub action_queue_background_evictions: u64,
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
    /// > [Pei Wang 2013] Non-Axiomatic Reasoning System, §3.3.3
    #[serde(default)]
    pub nars_drift_score: f64,
    /// Number of process beliefs currently in drifted state (freq shift >= 20pp).
    #[serde(default)]
    pub nars_drifted_beliefs: usize,
    /// Total NARS process beliefs held (capacity-capped). The dashboard shows
    /// drifted/total so "0" reads as "0 of N in drift", not "0 beliefs".
    #[serde(default)]
    pub nars_beliefs_total: usize,
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
    /// Direct CoreAudio source health. Root LaunchDaemons normally report
    /// `session-fallback` and rely on pmset/process/capture context instead.
    #[serde(default)]
    pub coreaudio_probe_state: String,
    #[serde(default)]
    pub coreaudio_probe_samples_total: u64,
    #[serde(default)]
    pub coreaudio_probe_cache_hits_total: u64,
    #[serde(default)]
    pub coreaudio_probe_failures_total: u64,

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
    #[serde(default)]
    pub chromium_ecore_demotions_total: u64,
    #[serde(default)]
    pub chromium_purge_hints_total: u64,
    #[serde(default)]
    pub chromium_purge_noops_total: u64,
    /// Estimated RAM freed (MB) by frozen renderers.
    #[serde(default)]
    pub chromium_freed_mb: f64,
    /// Successful temporary Jetsam demotions of visible background renderers.
    #[serde(default)]
    pub chromium_jetsam_demotions_total: u64,
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
    pub stage_reason_neuro_avg_ms: f64,
    #[serde(default)]
    pub stage_reason_neuro_max_ms: f64,
    #[serde(default)]
    pub stage_reason_decide_avg_ms: f64,
    #[serde(default)]
    pub stage_reason_decide_max_ms: f64,
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
    /// Additive instrumentation (2026-06-23): untimed enrich→decide ops.
    #[serde(default)]
    pub stage_reason_procscan_avg_ms: f64,
    #[serde(default)]
    pub stage_reason_procscan_max_ms: f64,
    #[serde(default)]
    pub stage_reason_rusage_avg_ms: f64,
    #[serde(default)]
    pub stage_reason_rusage_max_ms: f64,
    #[serde(default)]
    pub stage_reason_signalintel_avg_ms: f64,
    #[serde(default)]
    pub stage_reason_signalintel_max_ms: f64,
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
    pub ais_capability: f64,
    #[serde(default)]
    pub ais_optimization_opportunity: f64,
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
    #[serde(default)]
    pub ais_evidence_coverage: f64,
    #[serde(default)]
    pub ais_operational_health: f64,
    /// Medallion learning telemetry. Bronze counts raw resolved outcomes;
    /// Silver passed structural validation; Gold is the only tier admitted to
    /// long-lived learners.
    #[serde(default)]
    pub learning_bronze_total: u64,
    #[serde(default)]
    pub learning_silver_total: u64,
    #[serde(default)]
    pub learning_gold_total: u64,
    #[serde(default)]
    pub learning_rejected_total: u64,
    #[serde(default)]
    pub learning_invalid_total: u64,
    #[serde(default)]
    pub learning_duplicate_total: u64,
    #[serde(default)]
    pub learning_data_quality: f64,
    #[serde(default)]
    pub learning_gold_rate: f64,
    /// Context-only medallion. Unlike `learning_*`, these samples describe
    /// live system state and do not claim an action caused an outcome.
    #[serde(default)]
    pub world_model_context_bronze_total: u64,
    #[serde(default)]
    pub world_model_context_silver_total: u64,
    #[serde(default)]
    pub world_model_context_gold_total: u64,
    #[serde(default)]
    pub world_model_context_rejected_total: u64,
    #[serde(default)]
    pub world_model_context_non_finite_total: u64,
    #[serde(default)]
    pub world_model_context_range_total: u64,
    #[serde(default)]
    pub world_model_context_stale_total: u64,
    #[serde(default)]
    pub world_model_context_temporal_total: u64,
    #[serde(default)]
    pub world_model_context_foreign_total: u64,
    #[serde(default)]
    pub world_model_context_coherence_total: u64,
    /// Most frequently rejected numeric field since this daemon start.
    #[serde(default)]
    pub world_model_context_top_rejected_field: String,
    #[serde(default)]
    pub world_model_context_top_rejected_field_total: u64,
    /// Most recent field-level validation failure. Non-finite values are
    /// represented by `None` so runtime_metrics.json remains valid JSON.
    #[serde(default)]
    pub world_model_context_last_rejected_field: String,
    #[serde(default)]
    pub world_model_context_last_rejected_reason: String,
    #[serde(default)]
    pub world_model_context_last_rejected_value: Option<f64>,
    #[serde(default)]
    pub world_model_context_local_gold_total: u64,
    #[serde(default)]
    pub world_model_context_current_tier: String,
    #[serde(default)]
    pub world_model_context_authority_phase: String,
    #[serde(default)]
    pub world_model_context_quality: f64,
    /// Universal actuator-evidence lane. Bronze is a resolved applied action,
    /// Silver is structurally valid, and Gold is low-confounding evidence.
    #[serde(default)]
    pub world_model_actuator_issued_total: u64,
    #[serde(default)]
    pub world_model_actuator_pending_total: u64,
    #[serde(default)]
    pub world_model_actuator_bronze_total: u64,
    #[serde(default)]
    pub world_model_actuator_silver_total: u64,
    #[serde(default)]
    pub world_model_actuator_gold_total: u64,
    #[serde(default)]
    pub world_model_actuator_effective_total: u64,
    #[serde(default)]
    pub world_model_actuator_rejected_total: u64,
    #[serde(default)]
    pub world_model_actuator_expired_total: u64,
    #[serde(default)]
    pub world_model_actuator_quality: f64,
    #[serde(default)]
    pub world_model_actuator_mean_utility: f64,
    /// Objective-independent recent utility trend: human gain + system gain
    /// minus intervention cost, all counterfactual-adjusted.
    #[serde(default)]
    pub world_model_apollo_utility: f64,
    #[serde(default)]
    pub world_model_decision_credit_sources: u64,
    #[serde(default)]
    pub world_model_decision_credit_leader: String,
    #[serde(default)]
    pub world_model_decision_credit_leader_score: f64,
    #[serde(default)]
    pub world_model_decision_credit_leader_accuracy: f64,
    #[serde(default)]
    pub world_model_decision_credit_leader_observations: u32,
    /// GPU imagination lineage. Bronze is a completed forecast, Silver means
    /// an existing specialist action consumed it, and Gold means the measured
    /// actuator outcome calibrated that forecast on this hardware regime.
    #[serde(default)]
    pub world_model_gpu_bronze_total: u64,
    #[serde(default)]
    pub world_model_gpu_silver_total: u64,
    #[serde(default)]
    pub world_model_gpu_gold_total: u64,
    #[serde(default)]
    pub world_model_gpu_rejected_total: u64,
    #[serde(default)]
    pub world_model_gpu_evicted_total: u64,
    #[serde(default)]
    pub world_model_gpu_unused_total: u64,
    #[serde(default)]
    pub world_model_gpu_bronze_rejected_total: u64,
    /// Rejections older than the per-reason breakdown. Non-zero means the
    /// three buckets above describe only part of `world_model_gpu_rejected_total`;
    /// they and this field partition it exactly.
    ///
    /// Every `world_model_gpu_*` counter is lifetime-cumulative (persisted in
    /// `telemetry_medallion_state`), unlike the per-boot `gpu_imagination_*`
    /// family. Ratios across the two families are meaningless.
    #[serde(default)]
    pub world_model_gpu_unclassified_rejections: u64,
    #[serde(default)]
    pub world_model_gpu_pending_total: u64,
    #[serde(default)]
    pub world_model_gpu_calibrated_models: u64,
    #[serde(default)]
    pub world_model_gpu_calibration_mae: f64,
    #[serde(default)]
    pub world_model_gpu_calibration_brier: f64,
    #[serde(default)]
    pub world_model_gpu_calibration_quality: f64,
    /// Action/workload utility models retained for observation, excluding
    /// family-only aggregates. They may be stale or belong to another machine.
    #[serde(default)]
    pub world_model_actuator_known_models: u64,
    /// Known utility models currently fresh, sufficiently sampled, and
    /// compatible with this machine's hardware regime.
    #[serde(default)]
    pub world_model_actuator_ready_models: u64,
    #[serde(default)]
    pub world_model_readiness_no_gold: u64,
    #[serde(default)]
    pub world_model_readiness_immature: u64,
    /// Immature *and* no longer accumulating. Split out of `immature` so an
    /// inactive model is never presented as learning in progress.
    #[serde(default)]
    pub world_model_readiness_dormant: u64,
    #[serde(default)]
    pub world_model_readiness_low_quality: u64,
    #[serde(default)]
    pub world_model_readiness_stale: u64,
    #[serde(default)]
    pub world_model_readiness_foreign: u64,
    #[serde(default)]
    pub world_model_readiness_hardware: u64,
    #[serde(default)]
    pub world_model_readiness_uncertain: u64,
    /// Action-model capacity. When `len` sits on `capacity`, every newly
    /// observed key destroys a learned one, so a key can be reborn instead of
    /// maturing. Evictions and births make that pressure countable.
    #[serde(default)]
    pub world_model_action_model_len: u64,
    #[serde(default)]
    pub world_model_action_model_capacity: u64,
    #[serde(default)]
    pub world_model_action_model_evictions_total: u64,
    #[serde(default)]
    pub world_model_action_model_births_total: u64,
    /// Evidence actually reaching action models, and the cycle of the newest
    /// one. Distinguishes "nothing is arriving" from "arriving but blocked".
    #[serde(default)]
    pub world_model_evidence_updates_total: u64,
    #[serde(default)]
    pub world_model_last_evidence_cycle: u64,
    #[serde(default)]
    pub world_model_family_known_models: u64,
    #[serde(default)]
    pub world_model_family_ready_models: u64,
    #[serde(default)]
    pub world_model_family_prior_uses_total: u64,
    #[serde(default)]
    pub world_model_actuator_families: Vec<ActuatorEvidenceStatus>,
    #[serde(default)]
    pub world_model_utility_vetoes_total: u64,
    /// Mature positive utility predictions promoted ahead of exploratory
    /// responsiveness actions in the bounded dispatch queue.
    #[serde(default)]
    pub world_model_utility_promotions_total: u64,
    /// Positive candidates that actually changed position in the queue.
    #[serde(default)]
    pub world_model_utility_reorders_total: u64,
    /// Central intent planner telemetry. Proposed counts all candidates that
    /// reached planning; admitted excludes contradictory same-target intents.
    #[serde(default)]
    pub action_planner_proposed_total: u64,
    #[serde(default)]
    pub action_planner_admitted_total: u64,
    #[serde(default)]
    pub action_planner_conflict_drops_total: u64,
    #[serde(default)]
    pub action_planner_reorders_total: u64,
    #[serde(default)]
    pub action_planner_last_resolution: String,
    #[serde(default)]
    pub world_model_temporal_memory_samples: u64,
    #[serde(default)]
    pub world_model_sequence_candidates_total: u64,
    #[serde(default)]
    pub world_model_sequence_rollouts_total: u64,
    #[serde(default)]
    pub world_model_sequence_promotions_total: u64,
    #[serde(default)]
    pub world_model_sequence_authoritative_total: u64,
    #[serde(default)]
    pub world_model_sequence_authoritative_rollouts_total: u64,
    #[serde(default)]
    pub world_model_sequence_authoritative: bool,
    #[serde(default)]
    pub world_model_sequence_expected_gain: f64,
    #[serde(default)]
    pub world_model_sequence_uncertainty: f64,
    #[serde(default)]
    pub world_model_sequence_pressure_delta: f64,
    #[serde(default)]
    pub world_model_sequence_fluidity_delta: f64,
    #[serde(default)]
    pub world_model_sequence_energy_delta: f64,
    #[serde(default)]
    pub world_model_sequence_authoritative_expected_gain: f64,
    #[serde(default)]
    pub world_model_sequence_authoritative_uncertainty: f64,
    #[serde(default)]
    pub world_model_sequence_authoritative_pressure_delta: f64,
    #[serde(default)]
    pub world_model_sequence_authoritative_fluidity_delta: f64,
    #[serde(default)]
    pub world_model_sequence_authoritative_energy_delta: f64,
    /// Compact causal latent-dynamics ensemble and MPC observability.
    #[serde(default)]
    pub world_model_dynamics_phase: String,
    #[serde(default)]
    pub world_model_dynamics_action_models: u64,
    #[serde(default)]
    pub world_model_dynamics_ready_models: u64,
    #[serde(default)]
    pub world_model_dynamics_ranking_models: u64,
    #[serde(default)]
    pub world_model_dynamics_authoritative_models: u64,
    #[serde(default)]
    pub world_model_dynamics_baseline_models: u64,
    #[serde(default)]
    pub world_model_dynamics_baseline_ready_models: u64,
    #[serde(default)]
    pub world_model_dynamics_gold_updates: u64,
    #[serde(default)]
    pub world_model_dynamics_no_action_updates: u64,
    #[serde(default)]
    pub world_model_dynamics_validation_samples: u64,
    #[serde(default)]
    pub world_model_dynamics_validation_mae: f64,
    #[serde(default)]
    pub world_model_dynamics_validation_coverage: f64,
    #[serde(default)]
    pub world_model_dynamics_publication_revision: u64,
    #[serde(default)]
    pub world_model_dynamics_predictions_total: u64,
    #[serde(default)]
    pub world_model_dynamics_ranking_predictions_total: u64,
    #[serde(default)]
    pub world_model_dynamics_authoritative_predictions_total: u64,
    #[serde(default)]
    pub world_model_dynamics_baseline_uses_total: u64,
    #[serde(default)]
    pub world_model_dynamics_mean_uncertainty: f64,
    /// Asynchronous Metal Monte Carlo advisor. It contributes bounded ranking
    /// evidence only and never creates, vetoes, or executes an action.
    #[serde(default)]
    pub gpu_imagination_backend: String,
    #[serde(default)]
    pub gpu_imagination_device: String,
    #[serde(default)]
    pub gpu_imagination_circuit_state: String,
    #[serde(default)]
    pub gpu_imagination_circuit_reason: String,
    #[serde(default)]
    pub gpu_imagination_jobs_submitted_total: u64,
    #[serde(default)]
    pub gpu_imagination_jobs_completed_total: u64,
    #[serde(default)]
    pub gpu_imagination_jobs_failed_total: u64,
    #[serde(default)]
    pub gpu_imagination_skips_total: u64,
    #[serde(default)]
    pub gpu_imagination_samples_total: u64,
    #[serde(default)]
    pub gpu_imagination_gpu_time_ns_total: u64,
    #[serde(default)]
    pub gpu_imagination_wall_time_ns_total: u64,
    #[serde(default)]
    pub gpu_imagination_support_uses_total: u64,
    /// Root-planner rankings that consumed a fresh GPU forecast.
    #[serde(default)]
    pub gpu_imagination_root_rank_uses_total: u64,
    /// Specialist decisions whose contextual World Model bias included a fresh
    /// GPU forecast. This is advisory use, not an independent GPU action.
    #[serde(default)]
    pub gpu_imagination_contextual_uses_total: u64,
    #[serde(default)]
    pub gpu_imagination_last_influence_scope: String,
    #[serde(default)]
    pub gpu_imagination_last_influence_action: String,
    #[serde(default)]
    pub gpu_imagination_last_influence_support: f64,
    #[serde(default)]
    pub gpu_imagination_last_submit_outcome: String,
    #[serde(default)]
    pub gpu_imagination_last_error: String,
    #[serde(default)]
    pub gpu_imagination_last_best_action: String,
    #[serde(default)]
    pub gpu_imagination_last_positive_probability: f64,
    #[serde(default)]
    pub gpu_imagination_last_p10_gain: f64,
    /// Local, non-LLM synthesis of System 1, Dr Zero, World Model, medallion
    /// and GPU evidence. It only scales existing advisory paths.
    #[serde(default)]
    pub system_deliberation_mode: String,
    #[serde(default)]
    pub system_deliberation_confidence: f64,
    #[serde(default)]
    pub system_deliberation_advisory_support_scale: f64,
    #[serde(default)]
    pub system_deliberation_gpu_support_scale: f64,
    #[serde(default)]
    pub system_deliberation_dr_zero_challenge: f64,
    #[serde(default)]
    pub system_deliberation_system1_struggling: bool,
    #[serde(default)]
    pub system_deliberation_local_gold: u64,
    #[serde(default)]
    pub system_deliberation_gpu_forecasts: u32,
    #[serde(default)]
    pub system_deliberation_local_confidence: f64,
    #[serde(default)]
    pub system_deliberation_local_families: u32,
    #[serde(default)]
    pub local_consolidation_last_family: String,
    #[serde(default)]
    pub local_consolidation_last_action: String,
    #[serde(default)]
    pub local_consolidation_last_verdict: String,
    #[serde(default)]
    pub local_consolidation_last_utility: f64,
    #[serde(default)]
    pub world_model_sequence_best_first: String,
    #[serde(default)]
    pub world_model_sequence_best_second: String,
    #[serde(default)]
    pub world_model_sequence_authoritative_best_first: String,
    #[serde(default)]
    pub world_model_sequence_authoritative_best_second: String,
    #[serde(default)]
    pub world_model_sequence_abstention_reason: String,
    #[serde(default)]
    pub world_model_last_influence_kind: String,
    #[serde(default)]
    pub world_model_last_influence_action: String,
    #[serde(default)]
    pub world_model_last_influence_workload: String,
    #[serde(default)]
    pub world_model_last_influence_scope: String,
    #[serde(default)]
    pub world_model_last_influence_utility: f64,
    #[serde(default)]
    pub world_model_last_influence_lower_bound: f64,
    #[serde(default)]
    pub world_model_last_influence_upper_bound: f64,
    #[serde(default)]
    pub world_model_last_influence_evidence: f64,
    #[serde(default)]
    pub world_model_last_influence_quality: f64,
    #[serde(default)]
    pub world_model_last_influence_margin: f64,
    #[serde(default)]
    pub world_model_abstentions_total: u64,
    #[serde(default)]
    pub world_model_last_abstention_reason: String,
    #[serde(default)]
    pub world_model_last_abstention_action: String,
    #[serde(default)]
    pub world_model_last_abstention_workload: String,
    #[serde(default)]
    pub world_model_abstention_no_gold_total: u64,
    #[serde(default)]
    pub world_model_abstention_unknown_total: u64,
    #[serde(default)]
    pub world_model_abstention_immature_total: u64,
    #[serde(default)]
    pub world_model_abstention_dormant_total: u64,
    #[serde(default)]
    pub world_model_abstention_quality_total: u64,
    #[serde(default)]
    pub world_model_abstention_stale_total: u64,
    #[serde(default)]
    pub world_model_abstention_origin_total: u64,
    #[serde(default)]
    pub world_model_abstention_hardware_total: u64,
    #[serde(default)]
    pub world_model_abstention_uncertain_total: u64,
    #[serde(default)]
    pub world_model_counterfactual_eligible_total: u64,
    #[serde(default)]
    pub world_model_counterfactual_issued_total: u64,
    #[serde(default)]
    pub world_model_counterfactual_pending_total: u64,
    #[serde(default)]
    pub world_model_counterfactual_resolved_total: u64,
    #[serde(default)]
    pub world_model_counterfactual_rejected_total: u64,
    #[serde(default)]
    pub world_model_counterfactual_would_help_total: u64,
    #[serde(default)]
    pub world_model_counterfactual_control_utility: f64,
    /// Mature exact-workload control arms used only as a bounded ranking hint.
    #[serde(default)]
    pub world_model_counterfactual_rank_uses_total: u64,
    /// Context-nearest universal actuator episodes used for ranking only.
    #[serde(default)]
    pub world_model_episodic_rank_uses_total: u64,
    #[serde(default)]
    pub world_model_episodic_memory_samples: u64,
    #[serde(default)]
    pub world_model_episodic_memory_families: u64,
    /// Advisory World Model modulation of already-admitted acceleration
    /// lanes. These counters never represent newly authorized actions.
    #[serde(default)]
    pub world_model_contextual_markov_total: u64,
    #[serde(default)]
    pub world_model_contextual_interaction_total: u64,
    #[serde(default)]
    pub world_model_contextual_io_total: u64,
    #[serde(default)]
    pub world_model_contextual_predictive_total: u64,
    #[serde(default)]
    pub world_model_contextual_chromium_total: u64,
    #[serde(default)]
    pub world_model_contextual_chromium_suppressed_total: u64,
    #[serde(default)]
    pub world_model_contextual_last_action: String,
    #[serde(default)]
    pub world_model_contextual_last_bias: f64,
    /// Gold-only world-model telemetry. Unlike process-local medallion totals,
    /// these counters describe the persisted evidence currently available for
    /// action imagination and vetoes.
    #[serde(default)]
    pub world_model_curated_actions: u64,
    #[serde(default)]
    pub world_model_ready_actions: u64,
    #[serde(default)]
    pub world_model_gold_evidence: u64,
    /// Gold outcomes from the universal actuator medallion admitted into the
    /// pressure-causal graph (legacy pressure actions excluded to avoid dupes).
    #[serde(default)]
    pub world_model_causal_actuator_gold_total: u64,
    #[serde(default)]
    pub world_model_contextual_actions: u64,
    #[serde(default)]
    pub world_model_data_quality: f64,
    #[serde(default)]
    pub world_model_causal_refreshes: u64,
    #[serde(default)]
    pub world_model_causal_cache_hits: u64,
    #[serde(default)]
    pub world_model_utility_refreshes: u64,
    #[serde(default)]
    pub world_model_utility_cache_hits: u64,
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
    /// Sprint 12 Convergence #5 (2026-05-17). Cumulative count of
    /// maintenance purge attempts skipped because the unified-memory
    /// bus was saturated (entropy_anomaly fallback proxy on M1).
    /// Stays at 0 on idle systems; ramps under sustained LLM inference
    /// or heavy media transcode. See
    /// [`crate::engine::lse_counters::LockFreeMetrics::maintenance_purge_skipped_bus_saturated_total`].
    #[serde(default)]
    pub maintenance_purge_skipped_bus_saturated_total: u64,

    /// Phase 5.2 — Battery-aware cost penalty emissions (Sprint 8,
    /// 2026-05-16). Cumulative count of action-cost computations where
    /// `battery_aware_cost_penalty` returned a strictly positive value
    /// (battery + noise raised the action's cost). Stays at 0 on AC power
    /// and on idle battery — non-zero only when the penalty actually
    /// influenced a scoring decision.
    ///
    /// Wiring is deferred to a follow-up commit (see `OPENS: 1` on the
    /// introducing commit). Producers will call
    /// [`crate::engine::lse_counters::LockFreeMetrics::inc_battery_aware_penalty_emission`]
    /// from the `decide_actions` cost-composition site.
    #[serde(default)]
    pub battery_aware_penalty_emissions_total: u64,

    /// Phase 4.2 — External-event causal attribution counters (Sprint 7,
    /// 2026-05-16). Cumulative count of causal edges whose `pressure_drop`
    /// credit is confounded by a recent external event firing inside
    /// `EXTERNAL_BLAME_WINDOW` (10s). Operators read these to detect
    /// "Apollo claiming credit for thermal/IO-driven pressure drops".
    /// [Pearl 2009 §4] / [Rubin 1974].
    #[serde(default)]
    pub causal_external_thermal_blames_total: u64,
    #[serde(default)]
    pub causal_external_disk_blames_total: u64,
    #[serde(default)]
    pub causal_external_net_blames_total: u64,

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

    /// Phase 3.2 — Arousal-Modulated NARS Decay counter (Sprint 6,
    /// 2026-05-16). Cumulative number of persist cycles whose
    /// `DriftDetector::arousal_modulated_decay_factor(...)` produced a
    /// factor strictly less than `base_factor` — i.e. the daemon was in
    /// the Stressed/Crisis arousal zone and accelerated NARS forgetting.
    /// Flushed from lf_metrics each cycle via sync_from_lockfree.
    /// [McGaugh 2004]; [Yerkes & Dodson 1908].
    #[serde(default)]
    pub arousal_decay_accelerations_total: u64,

    /// Phase 3.3 — Cross-Group Companion Attention inferences (Sprint 6,
    /// 2026-05-16). Cumulative count of (A, B, score) triples returned by
    /// [`crate::engine::companion_graph::CompanionGraph::propagate_attention_across_groups`].
    /// Stays at 0 until the daemon main-loop wires the call site (see
    /// `OPENS: 1` on the introducing commit). Dashboards verify the
    /// feature is actually inferring cross-coalition companions instead
    /// of no-op'ing.
    #[serde(default)]
    pub companion_cross_group_inferences_total: u64,

    /// Phase 4.1 — Adaptive Drift Threshold raises counter (Sprint 7,
    /// 2026-05-16). Cumulative number of
    /// `AdaptiveDriftThreshold::recommended_threshold` calls that
    /// returned strictly greater than the base threshold — i.e. the
    /// adaptive layer raised the bar based on observed noise variance.
    /// Surfaces to `runtime_metrics.json` so operators can see whether
    /// the noise floor is actually being measured and acted on in prod.
    /// Flushed each cycle via `sync_from_lockfree`.
    /// [Brown 1959]; [Welford 1962]; [Kuncheva 2004].
    #[serde(default)]
    pub adaptive_drift_threshold_raises_total: u64,

    /// Phase 5.1 — User-presence suppression emissions (Sprint 8,
    /// 2026-05-16). Cumulative count of action-aggressiveness multipliers
    /// emitted by [`crate::engine::user_presence::user_presence_modulator`]
    /// that were strictly less than 1.0 (active or semi-active tier, no
    /// crisis override). Stays at 0 while the user is fully idle and
    /// while the daemon is in Crisis arousal (where survival overrides UX).
    ///
    /// Wiring is deferred to a follow-up commit (see `OPENS: 1` on the
    /// introducing commit). Producers will call
    /// [`crate::engine::lse_counters::LockFreeMetrics::add_user_presence_suppressions`]
    /// from the `decide_actions` cost-composition site or the cognitive
    /// tick's specialist voting step.
    ///
    /// [Iqbal & Bailey 2008] "Effects of Interruptions on Task Performance".
    #[serde(default)]
    pub user_presence_suppressions_total: u64,

    /// Phase 5.3 — Structured-rationale attachment emissions (Sprint 8,
    /// 2026-05-16). Cumulative count of `JournalEntry` writes where the
    /// `rationale` field was populated (i.e. the action carried a
    /// machine-parseable explanation). Stays at 0 until the cross-cutting
    /// wiring follow-up lands and journal write sites start emitting
    /// [`crate::engine::audit_types::Rationale`] alongside their journal
    /// pushes.
    ///
    /// Wiring is deferred to a follow-up commit (see `OPENS: 1` on the
    /// introducing commit). Producers will call
    /// [`crate::engine::lse_counters::LockFreeMetrics::inc_journal_rationale_attached`]
    /// from each `JournalEntry::with_rationale(..)` site.
    ///
    /// Dashboards compute the "rationale coverage ratio" as
    /// `journal_rationales_attached_total / total journal entries`. The
    /// counter is the only way to verify (without grepping logs) that the
    /// explainability feature actually attaches rationales in prod rather
    /// than no-op'ing on every action — same tautology-trap mitigation
    /// pattern as Phase 3.1 (skill_aware_modulations_total) and 5.2
    /// (battery_aware_penalty_emissions_total).
    ///
    /// [Doshi-Velez & Kim 2017] "Towards a Rigorous Science of Interpretable
    /// Machine Learning"; [Ribeiro et al. 2016] LIME — per-decision
    /// explanations.
    #[serde(default)]
    pub journal_rationales_attached_total: u64,

    /// Phase 4.3.1 — Specialist accuracy purge inhibitions (Sprint 8,
    /// 2026-05-16). Cumulative count of cycles where
    /// `daemon_cognitive_tick::apply_specialist_voting` skipped the EMA
    /// accuracy update block because a maintenance purge happened in the
    /// previous 30 s. Mirrors the Phase 2 inhibition pattern for
    /// `outcome_tracker` / `causal_graph` post-purge.
    ///
    /// A purge causes pressure to drop sharply, which would otherwise be
    /// graded as "hazard wrong" / "kalman wrong" — depressing specialist
    /// EMA weights and weakening reaction to the NEXT real crisis.
    ///
    /// [Rubin 1974] "Estimating Causal Effects of Treatments in Randomized
    /// and Nonrandomized Studies" — intervention vs confounder distinction.
    #[serde(default)]
    pub specialist_accuracy_purge_inhibitions_total: u64,

    /// Phase C SCORER-OVERRIDE (Sprint 11 finale, 2026-05-16).
    /// Cumulative count of actions where the gate tower ACCEPTED a
    /// candidate but the [`crate::engine::action_policy::PolicyScorer`]
    /// returned a composite score strictly less than −0.30 (strong
    /// reject) — the asymmetric partial cutover deferred to the scorer
    /// and the action was REJECTED. A `BlockedActionEvent` is also
    /// emitted with `BlockerKind::Other("scorer-override-accept-to-reject")`
    /// so offline tooling can correlate the override with t+30s
    /// outcomes (did the system regress? did pressure spike?).
    ///
    /// Flushed each cycle from
    /// [`crate::engine::lse_counters::LockFreeMetrics::scorer_override_rejects_total`]
    /// via `sync_from_lockfree`. Mirrors the Phase 3.1 / 5.2 tilt-counter
    /// pattern so dashboards can verify the partial cutover actually
    /// engages instead of silently no-op'ing — the "tautology trap"
    /// mitigation CLAUDE.md flags.
    ///
    /// [Nygard 2018 §8.5] — adaptive capacity limits via shadowing.
    #[serde(default)]
    pub scorer_override_rejects_total: u64,

    /// Phase C SCORER-OVERRIDE (Sprint 11 finale, 2026-05-16).
    /// Cumulative count of actions where the gate tower REJECTED a
    /// candidate and the [`crate::engine::action_policy::PolicyScorer`]
    /// returned a composite score strictly greater than +0.30 (strong
    /// accept). Per NotebookLM 2026-05-16 Candidate-C verdict, the
    /// asymmetric mode does NOT let the scorer beat the gate in the
    /// unsafe direction — the action stays REJECTED and we ONLY
    /// journal the disagreement (`BlockerKind::Other("scorer-disagreement-strong-accept")`)
    /// so Sprint 12 can decide whether to promote to symmetric cutover
    /// after N≥500 events of evidence.
    ///
    /// Flushed each cycle from
    /// [`crate::engine::lse_counters::LockFreeMetrics::scorer_disagreement_strong_accepts_total`]
    /// via `sync_from_lockfree`.
    ///
    /// [Nygard 2018 §8.5]; [Bengio 2013] — counterfactual reasoning
    /// requires observing the rejected path.
    #[serde(default)]
    pub scorer_disagreement_strong_accepts_total: u64,

    /// Phase D PURGE-INHIBITION (Sprint 12 candidate #1, 2026-05-17).
    ///
    /// Cycles where a predictor swap-update was suppressed because a
    /// `vm_purge` fired in the prior 5 s and would have been mis-learned
    /// as a load improvement. See
    /// [`crate::engine::lse_counters::LockFreeMetrics::purge_inhibition_skips_total`]
    /// for the producer. Flushed each cycle via `sync_from_lockfree`.
    ///
    /// [Hellerstein 2004 §9] disturbance rejection in closed-loop systems.
    #[serde(default)]
    pub purge_inhibition_skips_total: u64,

    /// RAM Phase B (2026-06-03) mediator chokepoint counters. See
    /// [`crate::engine::lse_counters::LockFreeMetrics::mediator_blocks_total`]
    /// (+ noop_writes_total + postcondition_violation_total) for producer
    /// docs. Flushed each cycle via `sync_from_lockfree`. The trio of
    /// counters lets the dashboard distinguish three classes of mediated
    /// failure: refused-before-syscall vs no-op-write vs lying-syscall.
    #[serde(default)]
    pub mediator_blocks_total: u64,
    #[serde(default)]
    pub mediator_noop_writes_total: u64,
    #[serde(default)]
    pub mediator_postcondition_violation_total: u64,
    /// Per-action-kind dedup drops (2026-06-18 audit: was silent telemetry
    /// death — producer `record_dedup_drops` fired live but the counters
    /// never reached runtime_metrics.json). How many duplicate actions the
    /// recently-applied cache suppressed before emit.
    #[serde(default)]
    pub dedup_drops_setmemorystatus: u64,
    #[serde(default)]
    pub dedup_drops_throttle: u64,
    #[serde(default)]
    pub dedup_drops_freeze: u64,
    #[serde(default)]
    pub dedup_drops_unfreeze: u64,
    #[serde(default)]
    pub dedup_drops_boost: u64,
    #[serde(default)]
    pub dedup_drops_thread_qos: u64,

    /// Sprint 12 Convergence #4 (2026-05-17). Cumulative coincidence
    /// count: cycles in which the scorer override fired AND the causal
    /// graph reports a recent thermal-throttle external event. Strong
    /// signal the policy is mis-learning under thermal pressure;
    /// dashboards should plot this against `policy_rollback_evaluations_total`
    /// — if alignments accumulate without rollback evaluations, the
    /// thermal sensor and the rollback guard are talking past each
    /// other. See LSE producer for details.
    #[serde(default)]
    pub causal_thermal_scorer_override_alignments_total: u64,

    /// Sprint 12 Convergence #1 (2026-05-17). Cold-thread routing
    /// decisions that flipped from default E-cluster to P-cluster
    /// because the owning process is a foreground companion AND DRAM
    /// bandwidth is below the safety floor. Stays at 0 until the user
    /// is running a multi-process foreground workflow (e.g. Brave +
    /// renderers, IDE + LSP). See
    /// [`crate::engine::lse_counters::LockFreeMetrics::companion_affinity_alignments_total`].
    #[serde(default)]
    pub companion_affinity_alignments_total: u64,

    /// Sprint 13 Pressure-Router Gate (2026-05-30). Cycles where the
    /// daemon main loop skipped the per-cycle
    /// `companion_graph.observe_cycle` + Phase 3.3 propagation block
    /// because `memory_pressure < mid_entry` AND the modulo-4
    /// forced-exploration fallback did not fire. Ratio against `cycles`
    /// should approach ~0.75 on an idle laptop and drop toward 0 under
    /// sustained pressure. See
    /// [`crate::engine::lse_counters::LockFreeMetrics::companion_observe_router_skips_total`].
    #[serde(default)]
    pub companion_observe_router_skips_total: u64,
    /// Sprint 12 perf-fix (2026-05-30). Cumulative per-cycle hits on
    /// the `companion_of_fg_pids` memoization cache. Producer rebuilds
    /// only when the (foreground_app, top_processes_fingerprint,
    /// companion_graph_witness) tuple changes; every other cycle is
    /// served from the cached HashSet and this counter bumps. Steady
    /// state ratio (hits / cycles) approaches 1.0 because the
    /// foreground app rarely flips and `top_processes` is stable
    /// across consecutive 5-s ticks. Drop ratio indicates either
    /// frequent fg switching or rapid CompanionGraph mutation under
    /// `self_improve` decay. See
    /// [`crate::engine::lse_counters::LockFreeMetrics::companion_fg_cache_hits_total`].
    #[serde(default)]
    pub companion_fg_cache_hits_total: u64,

    // ── Sprint follow-up (2026-06-05) — Silent-telemetry-death fix
    // (HIGH #1 / HIGH #2). The five LSE counters below were added to
    // `LockFreeMetrics` + `MetricsSnapshot` + `inc_*()` helpers in the
    // parent sprint but never mirrored into `RuntimeMetrics`, which is
    // exactly the Sprint 9 `4b13a39` regression class. Without the
    // following fields, `sync_from_lockfree` cannot fan them out into
    // `runtime_metrics.json`, so dashboards never see the values.
    //
    // Each counter's producer + dormant-state notes live on the LSE
    // field doc-comments; see `lse_counters.rs`.
    #[serde(default)]
    pub ac_cache_evictions_total: u64,
    #[serde(default)]
    pub mediator_thread_policy_total: u64,
    #[serde(default)]
    pub pid_recycle_blocks_total: u64,
    #[serde(default)]
    pub policy_scorer_uncertainty_saturated_total: u64,
    #[serde(default)]
    pub effect_decay_detected_total: u64,

    /// Round-4 (2026-06-07). FIX-3-v2 HP MachPolicy attempts (no re-read).
    /// Separated from `effect_decay_detected_total` so the Jetsam/Sysctl
    /// re-read-disagreement baseline 27 stays comparable post-deploy.
    #[serde(default)]
    pub effect_decay_hp_mach_attempts_total: u64,

    /// WebRTC guard (2026-06-09 prod incident). Bumped by
    /// `SysctlGovernor::tick_tcp` for each TCP scale-down or delayed_ack
    /// branch suppressed because `realtime_call_active == true`. Producer:
    /// `coreaudio_active::is_realtime_call_active` (default-output AND
    /// default-input both running). Mirrors the LSE counter so
    /// `runtime_metrics.json` surfaces the gate-fires count for operator
    /// review. Non-zero rate during prod calls = guard is intercepting
    /// the path that froze the Meet audio on 2026-06-09T17:12 PT.
    #[serde(default)]
    pub sysctl_governor_realtime_call_inhibit_total: u64,

    /// B.6 gap fix (2026-06-10). Jetsam hints emitted while in macOS
    /// cooperation step-back mode (previously-dead should_emit_jetsam_hints
    /// surface, now wired). Mirrors the LSE counter.
    #[serde(default)]
    pub cooperation_jetsam_hints_total: u64,

    /// B.6 gap fix (2026-06-10). ZombieHunter classifications + actions.
    #[serde(default)]
    pub zombie_dead_weight_detected_total: u64,
    #[serde(default)]
    pub zombie_actions_emitted_total: u64,

    /// Anti-ratchet (2026-06-10). Boost decay reverts (nice/tier restored).
    #[serde(default)]
    pub boost_reverts_total: u64,

    /// Evolve iter-3 (2026-06-10). EffectLedger unified anti-ratchet reverts.
    #[serde(default)]
    pub effect_ledger_reverts_total: u64,

    /// Calibration loop-closure (2026-06-11) [Guo 2017; Platt 1999]. Cycles
    /// where the meta-cognition debias multiplier rescaled raw predictions.
    #[serde(default)]
    pub prediction_debias_applied_total: u64,

    /// World-model Mode-2 gate (2026-06-11) [LeCun 2022]. Freeze candidates
    /// skipped because the learned model predicts do-nothing dominates.
    #[serde(default)]
    pub world_model_dominance_skips_total: u64,

    /// Planner Phase-1 pre-arm (2026-06-11). Rising edges of the cadence
    /// pinning 300ms on a fresh forecast spike hint.
    #[serde(default)]
    pub planner_prearm_total: u64,

    /// Focus prediction accelerator. Attempts count one lease acquisition,
    /// not every daemon cycle; applied counts leases that changed at least
    /// one kernel/cache state. Hits and misses are resolved on an actual
    /// foreground transition or lease expiry.
    #[serde(default)]
    pub markov_prewarm_attempts: u64,
    #[serde(default)]
    pub markov_prewarm_applied: u64,
    #[serde(default)]
    pub markov_prewarm_hits: u64,
    #[serde(default)]
    pub markov_prewarm_misses: u64,
    /// Passive next-app predictions scored without changing kernel or cache
    /// state. These samples calibrate transition reliability while the real
    /// accelerator is quarantined, unavailable, or intentionally idle.
    #[serde(default)]
    pub markov_shadow_predictions_total: u64,
    #[serde(default)]
    pub markov_shadow_resolved_total: u64,
    #[serde(default)]
    pub markov_shadow_hits: u64,
    #[serde(default)]
    pub markov_shadow_misses: u64,
    /// The two opposite failures `markov_shadow_misses` hides. Wrong-app is the
    /// model being wrong; no-switch is the model being right about an event
    /// that had not happened when the window closed. They call for different
    /// fixes, and 1 hit against 10 misses says nothing until they are split.
    #[serde(default)]
    pub markov_shadow_miss_wrong_app: u64,
    #[serde(default)]
    pub markov_shadow_miss_no_switch: u64,
    /// Hits, by whether a real pre-warm could have been issued at the moment
    /// the prediction was made. A correct prediction blocked by resources is a
    /// different problem from a correct prediction nobody could act on in time.
    #[serde(default)]
    pub markov_shadow_hits_actionable: u64,
    #[serde(default)]
    pub markov_shadow_hits_blocked_resources: u64,
    #[serde(default)]
    pub markov_shadow_hits_blocked_timing: u64,
    /// Summed lead time over hits, for a mean without keeping a histogram.
    #[serde(default)]
    pub markov_shadow_hit_lead_ms_total: u64,
    #[serde(default)]
    pub markov_shadow_active: bool,
    #[serde(default)]
    pub markov_shadow_superseded_total: u64,
    #[serde(default)]
    pub markov_prewarm_reverts: u64,
    #[serde(default)]
    pub markov_prewarm_active: bool,
    /// Number of same-coalition members held by the active predictive lease.
    #[serde(default)]
    pub markov_prewarm_members: u32,
    /// Cumulative members whose cache or reversible kernel state was changed.
    #[serde(default)]
    pub markov_prewarm_members_applied: u64,
    /// Cumulative executable bytes submitted to F_RDADVISE by predictive leases.
    #[serde(default)]
    pub markov_prewarm_cache_bytes: u64,
    /// Milliseconds of advance notice from lease acquisition to a prediction hit.
    #[serde(default)]
    pub markov_prewarm_last_lead_ms: u64,
    /// Milliseconds from prediction hit until launch protection is clear and
    /// fluidity is no longer degraded. Zero is a valid immediate settle.
    #[serde(default)]
    pub markov_prewarm_last_settle_ms: u64,
    #[serde(default)]
    pub markov_prewarm_settle_observations: u64,
    #[serde(default)]
    pub markov_prediction_app: String,
    #[serde(default)]
    pub markov_prediction_confidence: f64,
    #[serde(default)]
    pub markov_prediction_eta_secs: f64,
    /// Age of a prediction whose expected switch point has already passed.
    /// Mutually exclusive with the remaining ETA above.
    #[serde(default)]
    pub markov_prediction_overdue_secs: f64,
    /// Probability that the predicted destination becomes foreground within
    /// each intent horizon. These are context signals, not prewarm authority.
    #[serde(default)]
    pub markov_prediction_5s: f64,
    #[serde(default)]
    pub markov_prediction_30s: f64,
    #[serde(default)]
    pub markov_prediction_2m: f64,
    #[serde(default)]
    pub markov_prediction_10m: f64,
    #[serde(default)]
    pub markov_prediction_dwell_observations: u32,
    #[serde(default)]
    pub markov_prediction_dwell_deviation_secs: f64,
    #[serde(default)]
    pub markov_prewarm_eligible: bool,
    /// Current admission and first blocking gate for the predicted pair.
    #[serde(default)]
    pub markov_prewarm_admission: String,
    #[serde(default)]
    pub markov_prewarm_blocker: String,
    /// Transition-local prewarm calibration. Quarantine suppresses only the
    /// speculative accelerator; prediction learning continues normally.
    #[serde(default)]
    pub markov_prewarm_quarantined: bool,
    #[serde(default)]
    pub markov_prewarm_reliability: f64,
    /// Samples in the current bounded workload/time/pressure/activity bucket.
    #[serde(default)]
    pub markov_prewarm_context_trials: u32,
    #[serde(default)]
    pub markov_prewarm_probe_transitions_remaining: u64,
    #[serde(default)]
    pub markov_prewarm_quarantines_total: u64,
    #[serde(default)]
    pub markov_prewarm_quarantine_skips_total: u64,
    #[serde(default)]
    pub markov_prewarm_probes_total: u64,
    /// Probe leases change cache only; mature leases may use reversible QoS.
    #[serde(default)]
    pub markov_prewarm_cache_only_total: u64,
    /// Kernel mutations skipped because another Apollo actuator already owned
    /// that PID/effect kind in the unified ledger.
    #[serde(default)]
    pub markov_prewarm_conflict_skips_total: u64,

    /// Temporal-only cache prefetches are bounded to one candidate per focus
    /// transition and never mutate QoS/jetsam state.
    #[serde(default)]
    pub temporal_prewarm_attempts: u64,
    #[serde(default)]
    pub temporal_prewarm_applied: u64,
    #[serde(default)]
    pub temporal_prewarm_cache_bytes: u64,
    #[serde(default)]
    pub temporal_prewarm_last_app: String,

    /// B.4 purge band split (2026-06-10). Legacy aggregate keeps the sum.
    #[serde(default)]
    pub maintenance_purge_skipped_pressure_low_total: u64,
    #[serde(default)]
    pub maintenance_purge_skipped_pressure_survival_total: u64,
    #[serde(default)]
    pub maintenance_purge_skipped_rising_edge_total: u64,

    /// B.2 replayd gate (2026-06-09 incident follow-up). Bumped at the
    /// daemon composition point each cycle where the screen-capture probe
    /// (`realtime_signals::ScreenCaptureCache` — replayd /
    /// screencaptureui / ScreenSharingAgent in the proc table) is the
    /// DECIDING signal for `realtime_call_active` (audio full-duplex gate
    /// false, screen-capture true). Mirrors the LSE counter so
    /// `runtime_metrics.json` separates screen-share-only inhibitions from
    /// full-duplex-call inhibitions. `#[serde(default)]` keeps legacy
    /// snapshots deserializing as 0.
    #[serde(default)]
    pub sysctl_governor_screen_capture_inhibit_total: u64,

    // ── Approach 2 (2026-06-07) — OutcomeTracker class-reclassification HP exclusion.
    // Producer: `PatternWeight::effectiveness_for_classification(name)` returns
    // `None` when `safety::hard_protected_contains(name)` is true. Mirrors the
    // LSE counter so `runtime_metrics.json` exposes the per-cycle suppression
    // count for operator review. Sustained non-zero values mean the prod
    // Brave/Chromium Boost-loop is being correctly suppressed at the signal
    // layer (CLAUDE.md "Chromium SIGSTOP never" + soft-throttle structurally-
    // degraded effectiveness). `#[serde(default)]` preserves backwards
    // compatibility with pre-fix `runtime_metrics.json` snapshots — legacy
    // files deserialize this field as 0.
    #[serde(default)]
    pub hard_protected_reclassify_excluded_total: u64,

    // ── Group C (2026-06-06) — Invariant #13 port-hub gate + Dempster-Shafer.
    // Producers:
    //   * `mediator_port_hub_blocks_total` — `MachPolicyEffector::apply`
    //     refused a tier demote because `MachQoSManager::get_mach_port_count`
    //     reported > `PORT_HUB_THRESHOLD` rights for the target PID.
    //   * `mediator_port_hub_probe_unavailable_total` — same call site, but
    //     `get_mach_port_count` returned `None` (entitlement-denied or PID
    //     exited). Honest deferral marker: a sustained non-zero ratio means
    //     the gate is observationally dark and must not be trusted.
    //   * `policy_scorer_ds_high_conflict_fallback_total` — Dempster-Shafer
    //     aggregation produced K > 0.7 (Yager 1987 §3 — incompatible
    //     evidence) and the scorer fell back to RSS for that single call.
    //     Only meaningful when `LearnedState::policy_aggregator_mode` is
    //     "ds"; under the shipped default "rss" the counter stays at 0.
    #[serde(default)]
    pub mediator_port_hub_blocks_total: u64,
    #[serde(default)]
    pub mediator_port_hub_probe_unavailable_total: u64,
    #[serde(default)]
    pub policy_scorer_ds_high_conflict_fallback_total: u64,

    // ── Brave-Boost feedback loop fix (2026-06-07, APPROACH 1). ──────────────
    // Producer: `decide_actions::decide_actions` at the wait-graph blocker,
    // interactive-focus, and ML/AMX BOOST emission arms. Increments when the
    // target name hits `safety::hard_protected_contains` and the would-be
    // BoostProcess is suppressed. Stays at 0 until a hard-protected name
    // (e.g. "Brave Browser Helper", "WindowServer" outside the display-
    // pipeline carve-out) tries to take a Boost path. See LSE field doc.
    #[serde(default)]
    pub hard_protected_boost_skipped_total: u64,
    /// Throttle proposals dropped for targeting a process too small to matter.
    /// Rising with `throttles_applied` flat is the intended steady state.
    #[serde(default)]
    pub throttle_below_resource_floor_total: u64,
    /// Throttle proposals skipped because execution already refused this PID.
    #[serde(default)]
    pub throttle_refusal_suppressed_total: u64,
    /// PIDs currently held in the throttle-refusal cooldown.
    #[serde(default)]
    pub throttle_refusal_cooldown_active: u32,

    // ── Approach-3 wire (2026-06-07). ────────────────────────────────────────
    // Producer: `learned_state::poke_rollback_guard_via_decay`. Increments
    // exactly once per successful `PolicyRollbackGuard::evaluate_from_decay`
    // application — the decay-driven complement of
    // `policy_rollback_executions_total`. Stays at 0 unless ≥5 hard-protected
    // disagreements (per the effect-decay sliding window) land inside the
    // 5-min rollback-recency window AND the guard's cooldown has elapsed.
    // Compare against `effect_decay_detected_total` (raw HP+free disagreements)
    // to verify the wire fires only when hard-protected targets dominate.
    #[serde(default)]
    pub policy_rollback_triggered_by_decay_total: u64,

    // ── FIX-4-v2 (2026-06-07). Phantom-enrollment guard. ─────────────────────
    // Producer: `execute_actions::execute_actions` Boost + SetThreadQoS arms.
    // Increments every time the pre-syscall capability chain (caps.can_taskpolicy
    // && qos_mgr.is_some() && syscall_success) failed so the
    // `effect_decay::record_global` call was correctly suppressed. Under the
    // Round-2 unconditional design those calls would have enrolled
    // PendingObservations against a tier the kernel never actually applied,
    // skewing `effect_decay_detected_total` with false-positive HP
    // disagreements. Stays at 0 on a fully-capable daemon with healthy
    // syscall returns; a sustained non-zero ratio against
    // `boosts_applied + thread_qos_applied` quantifies the capability gap.
    #[serde(default)]
    pub effect_decay_phantom_enroll_skipped_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ActuatorEvidenceStatus {
    pub family: String,
    pub issued: u64,
    pub resolved: u64,
    pub gold: u64,
    pub effective: u64,
    pub rejected: u64,
    pub expired: u64,
    pub mean_quality: f64,
    pub mean_utility: f64,
}

impl RuntimeMetrics {
    /// Human-readable summary for the most recent dispatch batch.
    ///
    /// Keep the `last_cycle_*` prefix explicit: this field is overwritten every
    /// cycle and must not be mistaken for cumulative action totals.
    pub fn format_last_actions_summary(&self, actions: &[RootAction]) -> String {
        format!(
            "last_cycle_actions={} last_cycle_boosts={} last_cycle_throttles={} last_cycle_freezes={} last_cycle_sysctl={} invalid_sysctl_denied={} total_boosts_applied={} total_actions_pushed_raw={} total_actions_pushed_boost_typed={}",
            actions.len(),
            actions
                .iter()
                .filter(|a| matches!(a, RootAction::BoostProcess { .. }))
                .count(),
            actions
                .iter()
                .filter(|a| matches!(a, RootAction::ThrottleProcess { .. }))
                .count(),
            actions
                .iter()
                .filter(|a| matches!(a, RootAction::FreezeProcess { .. }))
                .count(),
            actions
                .iter()
                .filter(|a| matches!(a, RootAction::SetSysctl(_)))
                .count(),
            self.invalid_sysctl_denied,
            self.boosts_applied,
            self.actions_pushed_raw_total,
            self.actions_pushed_boost_total,
        )
    }

    /// Record a bounded GPU forecast that changed only a root-planner rank.
    /// This is intentionally separate from a successful root action: the
    /// usual safety and capability gates still decide whether it can execute.
    pub fn record_gpu_root_rank_influence(&mut self, action_key: &str, support: f64) {
        self.gpu_imagination_support_uses_total =
            self.gpu_imagination_support_uses_total.saturating_add(1);
        self.gpu_imagination_root_rank_uses_total =
            self.gpu_imagination_root_rank_uses_total.saturating_add(1);
        self.gpu_imagination_last_influence_scope = "root-ranking".to_string();
        self.gpu_imagination_last_influence_action = action_key.chars().take(96).collect();
        self.gpu_imagination_last_influence_support = support.clamp(-0.005, 0.005);
    }

    /// Record an existing specialist decision that consumed GPU-derived World
    /// Model context. It does not imply that the GPU initiated or executed it.
    pub fn record_gpu_contextual_influence(&mut self, scope: &str, action_key: &str, support: f64) {
        self.gpu_imagination_support_uses_total =
            self.gpu_imagination_support_uses_total.saturating_add(1);
        self.gpu_imagination_contextual_uses_total =
            self.gpu_imagination_contextual_uses_total.saturating_add(1);
        self.gpu_imagination_last_influence_scope = scope.chars().take(48).collect();
        self.gpu_imagination_last_influence_action = action_key.chars().take(96).collect();
        self.gpu_imagination_last_influence_support = support.clamp(-0.08, 0.08);
    }

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
    /// Currently frozen processes. Empty if none.
    #[serde(default)]
    pub frozen_processes: Vec<FrozenProcessInfo>,
}

/// Demoted to pub(crate): no bin imports or uses this type; no pub function takes it as
/// parameter. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
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

    /// Telemetry-death guard (2026-06-18 audit): the dedup_drops_* counters had
    /// a live producer but no RuntimeMetrics field, so they never reached
    /// runtime_metrics.json. Pin that the fields exist and survive serde — if a
    /// future edit drops them, this fails instead of silently losing telemetry.
    #[test]
    fn dispatch_diagnostics_survive_serde_roundtrip() {
        let m = RuntimeMetrics {
            dedup_drops_setmemorystatus: 11,
            dedup_drops_throttle: 22,
            dedup_drops_freeze: 33,
            dedup_drops_unfreeze: 44,
            dedup_drops_boost: 55,
            dedup_drops_thread_qos: 66,
            dispatch_prefilter_input_total: 101,
            dispatch_prefilter_recent_drops: 12,
            dispatch_prefilter_apple_drops: 23,
            dispatch_prefilter_protected_drops: 34,
            dispatch_prefilter_identity_drops: 4,
            dispatch_prefilter_unsupported_drops: 5,
            dispatch_prefilter_survivors_total: 23,
            heuristic_apple_platform_protected: 707,
            ..RuntimeMetrics::default()
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: RuntimeMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dedup_drops_setmemorystatus, 11);
        assert_eq!(back.dedup_drops_throttle, 22);
        assert_eq!(back.dedup_drops_freeze, 33);
        assert_eq!(back.dedup_drops_unfreeze, 44);
        assert_eq!(back.dedup_drops_boost, 55);
        assert_eq!(back.dedup_drops_thread_qos, 66);
        assert_eq!(back.dispatch_prefilter_input_total, 101);
        assert_eq!(back.dispatch_prefilter_recent_drops, 12);
        assert_eq!(back.dispatch_prefilter_apple_drops, 23);
        assert_eq!(back.dispatch_prefilter_protected_drops, 34);
        assert_eq!(back.dispatch_prefilter_identity_drops, 4);
        assert_eq!(back.dispatch_prefilter_unsupported_drops, 5);
        assert_eq!(back.dispatch_prefilter_survivors_total, 23);
        assert_eq!(back.heuristic_apple_platform_protected, 707);
    }

    // ── OptimizationProfile roundtrip ─────────────────────────────────────────

    // ── Phase 5.3 wiring helpers ─────────────────────────────────────────────

    #[test]
    fn root_action_action_class_covers_every_variant() {
        // Spot-check one of each — if a new variant lands without an
        // action_class arm the match expression will fail compile, but
        // this test guards the existing labels against silent rename.
        let throttle = RootAction::ThrottleProcess {
            pid: 1,
            name: "x".into(),
            aggressive: false,
            reason: String::new(),
            decision_reason: crate::engine::audit_types::DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        };
        assert_eq!(throttle.action_class(), "throttle");
        let boost = RootAction::BoostProcess {
            pid: 1,
            name: "x".into(),
            reason: String::new(),
            decision_reason: crate::engine::audit_types::DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        };
        assert_eq!(boost.action_class(), "boost");
        let freeze = RootAction::FreezeProcess {
            pid: 1,
            name: "x".into(),
            reason: String::new(),
            decision_reason: crate::engine::audit_types::DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        };
        assert_eq!(freeze.action_class(), "freeze");
    }

    #[test]
    fn root_action_reason_and_decision_reason_accessors_match_variant() {
        let action = RootAction::ThrottleProcess {
            pid: 7,
            name: "foo".into(),
            aggressive: false,
            reason: "p_oom=0.62".into(),
            decision_reason: crate::engine::audit_types::DecisionReason::CausalInference,
            start_sec: 0,
            start_usec: 0,
        };
        assert_eq!(action.reason(), "p_oom=0.62");
        assert!(matches!(
            action.decision_reason(),
            crate::engine::audit_types::DecisionReason::CausalInference
        ));
    }

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

    #[test]
    fn last_actions_summary_distinguishes_cycle_from_totals() {
        let mut m = RuntimeMetrics {
            boosts_applied: 61,
            actions_pushed_raw_total: 14_345,
            actions_pushed_boost_total: 0,
            invalid_sysctl_denied: 2,
            ..RuntimeMetrics::default()
        };
        let summary = m.format_last_actions_summary(&[]);

        assert!(summary.contains("last_cycle_actions=0"), "{summary}");
        assert!(summary.contains("last_cycle_boosts=0"), "{summary}");
        assert!(summary.contains("total_boosts_applied=61"), "{summary}");
        assert!(
            summary.contains("total_actions_pushed_raw=14345"),
            "{summary}"
        );
        assert!(
            summary.contains("total_actions_pushed_boost_typed=0"),
            "{summary}"
        );
        assert!(summary.contains("invalid_sysctl_denied=2"), "{summary}");

        m.last_actions_summary = summary;
        assert!(!m.last_actions_summary.starts_with("actions=0 "));
    }

    #[test]
    fn pressure_modulation_caps_high_pressure_growth_to_25_percent() {
        let base = SafetyPolicy::for_profile(OptimizationProfile::AggressiveRoot);
        let high = SafetyPolicy::with_pressure_modulation(&base, 0.80);

        assert_eq!(high.max_throttles_per_cycle, 38);
        assert_eq!(high.max_freezes_per_cycle, 15);
        assert_eq!(high.max_paging_hints_per_cycle, 25);
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
    fn identity_fields_boost_surfaces_start_sec() {
        // Inv#11 (2026-06-06): Boost now carries start_sec; identity_fields
        // returns the real value rather than the legacy `0,0` fallback.
        let a = RootAction::BoostProcess {
            pid: 300,
            name: "Alacritty".into(),
            reason: "fg".into(),
            decision_reason: DecisionReason::InteractiveFocus,
            start_sec: 12345,
            start_usec: 6789,
        };
        assert_eq!(
            a.identity_fields(),
            Some((300u32, Some("Alacritty"), 12345u64, 6789u64))
        );
    }

    #[test]
    fn identity_fields_unfreeze_surfaces_start_time() {
        let a = RootAction::UnfreezeProcess {
            pid: 400,
            name: "Code Helper".into(),
            reason: "thaw".into(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 12_345,
            start_usec: 678,
        };
        assert_eq!(
            a.identity_fields(),
            Some((400u32, Some("Code Helper"), 12_345u64, 678u64))
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
            start_sec: 0,
            start_usec: 0,
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

    #[test]
    fn reflex_and_build_capability_metrics_default_for_legacy_json() {
        let legacy = r#"{"cycles":7}"#;
        let metrics: RuntimeMetrics = serde_json::from_str(legacy).expect("legacy metrics");
        assert_eq!(metrics.reflex_phase, "");
        assert_eq!(metrics.reflex_proposed_total, 0);
        assert_eq!(metrics.reflex_applied_total, 0);
        assert_eq!(metrics.reflex_omitted_total, 0);
        assert_eq!(metrics.reflex_last_decision_latency_us, 0);
        assert_eq!(metrics.reflex_reasoning_dropped_total, 0);
        assert_eq!(metrics.parallel_expected_profile, "");
        assert_eq!(metrics.parallel_compiled_profile, "");
        assert_eq!(metrics.parallel_effective_profile, "");
    }
}
