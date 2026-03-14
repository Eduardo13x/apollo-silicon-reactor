use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use crate::engine::usage_model::{UsageEntrySummary, UsageTopReport};

/// Centralized utility for hardened file system operations to prevent TOCTOU and symlink attacks.
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum InteractiveContext {
    InteractiveFocus,
    BackgroundPressure,
    ThermalConstrained,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityReport {
    pub can_taskpolicy: bool,
    pub can_sysctl: bool,
    pub can_memorystatus: bool,
    pub can_mdutil: bool,
    pub can_tmutil: bool,
    pub is_root: bool,
    pub unavailable: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RootAction {
    BoostProcess {
        pid: u32,
        name: String,
        reason: String,
    },
    ThrottleProcess {
        pid: u32,
        name: String,
        aggressive: bool,
        reason: String,
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
        /// Kernel start-time for PID identity validation (prevents A-B-A recycling).
        #[serde(default)]
        start_sec: u64,
        #[serde(default)]
        start_usec: u64,
    },
    UnfreezeProcess {
        pid: u32,
        name: String,
    },
    SetSysctl {
        key: String,
        value: String,
        reason: String,
    },
    SetMemorystatus {
        pid: u32,
        priority: i32,
        reason: String,
    },
    ToggleSpotlight {
        enabled: bool,
        reason: String,
    },
    QuarantineDaemon {
        daemon: String,
        active: bool,
        reason: String,
    },
    /// Per-thread QoS: route a specific thread to P-core or E-core.
    SetThreadQoS {
        pid: u32,
        name: String,
        thread_index: u32,
        /// "interactive", "background", or "utility"
        tier: String,
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FreezeSource {
    MainLoop,
    Sentinel,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenEntry {
    pub frozen_at: DateTime<Utc>,
    pub source: FreezeSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenPidEntry {
    pub pid: u32,
    pub since: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenStatePersisted {
    pub frozen: Vec<FrozenPidEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    pub timestamp: DateTime<Utc>,
    pub action: RootAction,
    pub before: Option<String>,
    pub after: Option<String>,
    pub success: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileTransition {
    pub from: OptimizationProfile,
    pub to: OptimizationProfile,
    pub at: DateTime<Utc>,
    pub reason: String,
    pub pressure_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernorState {
    pub effective_profile: OptimizationProfile,
    pub cooldown_until: Option<DateTime<Utc>>,
    pub consecutive_high: u32,
    pub consecutive_low: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualOverride {
    pub profile: OptimizationProfile,
    pub expires_at: DateTime<Utc>,
    pub reason: String,
}

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
    pub post_wake_throttle_suppressed: u64,
    pub post_wake_freeze_suppressed: u64,
    pub swap_used_bytes: u64,
    pub swap_delta_bps: f64,
    pub memory_pressure: f64,
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
    // Heuristic module metrics
    pub heuristic_decisions: u64,
    pub heuristic_throttles: u64,
    pub heuristic_freezes: u64,
    pub heuristic_kills_downgraded: u64,
    pub zombies_detected: u64,
    #[serde(default)]
    pub kills_applied: u64,
    #[serde(default)]
    pub survival_mode_activations: u64,
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
    #[serde(default)]
    pub symlink_attacks_blocked: u64,
    #[serde(default)]
    pub policy_patterns_rejected: u64,
    #[serde(default)]
    pub request_size_exceeded: u64,
    // Overflow guard metrics
    #[serde(default)]
    pub overflow_events_total: u64,
    #[serde(default)]
    pub overflow_events_7d: usize,
    /// Ajuste actual al threshold de presión (negativo = más conservador).
    #[serde(default)]
    pub overflow_threshold_offset_pp: i32,
    /// true cuando hay compilación activa (build_mode).
    #[serde(default)]
    pub overflow_build_mode: bool,
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
    // Thread-level QoS metrics (Phase 1)
    #[serde(default)]
    pub thread_qos_applied: u64,
    #[serde(default)]
    pub thread_qos_hot_routes: u64,
    #[serde(default)]
    pub thread_qos_cold_routes: u64,
}

/// Serializable foreground app info for the protocol/dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForegroundAppInfo {
    pub pid: u32,
    pub name: String,
    pub bundle_id: Option<String>,
}

/// Serializable per-app energy info for the protocol/dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyConsumerInfo {
    pub name: String,
    pub current_watts: f64,
    pub percentage: f64,
}

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
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStatus {
    pub entries: usize,
    pub last_updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UsageResponse {
    Top(UsageTopReport),
    Explain(UsageEntrySummary),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LearnedPolicyStatus {
    pub interactive_patterns: usize,
    pub noise_patterns: usize,
    pub protected_patterns: usize,
    pub learned_at: Option<DateTime<Utc>>,
}

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
