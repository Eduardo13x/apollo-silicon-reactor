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
            max_boosts_per_cycle: ((base.max_boosts_per_cycle as f64 * core_scale).round() as usize).max(1),
            max_throttles_per_cycle: ((base.max_throttles_per_cycle as f64 * core_scale).round() as usize).max(1),
            max_paging_hints_per_cycle: ((base.max_paging_hints_per_cycle as f64 * ram_scale).round() as usize).max(1),
            max_freezes_per_cycle: ((base.max_freezes_per_cycle as f64 * core_scale).round() as usize).max(1),
            max_sysctl_writes_per_cycle: ((base.max_sysctl_writes_per_cycle as f64 * core_scale).round() as usize).max(1),
            cooldown_seconds: base.cooldown_seconds,
            max_thread_qos_per_cycle: ((base.max_thread_qos_per_cycle as f64 * core_scale).round() as usize).max(1),
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

impl RootAction {
    /// Construct a `ThrottleProcess` with zero start times.
    ///
    /// Use this when the action is queued before PID identity validation
    /// (start times are filled in by `execute_actions` at dispatch time).
    pub fn throttle(
        pid: u32,
        name: impl Into<String>,
        aggressive: bool,
        reason: impl Into<String>,
    ) -> Self {
        Self::throttle_full(pid, name, aggressive, reason, 0, 0)
    }

    pub fn throttle_full(
        pid: u32,
        name: impl Into<String>,
        aggressive: bool,
        reason: impl Into<String>,
        start_sec: u64,
        start_usec: u64,
    ) -> Self {
        RootAction::ThrottleProcess {
            pid,
            name: name.into(),
            aggressive,
            reason: reason.into(),
            start_sec,
            start_usec,
        }
    }

    pub fn freeze(
        pid: u32,
        name: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::freeze_full(pid, name, reason, 0, 0)
    }

    pub fn freeze_full(
        pid: u32,
        name: impl Into<String>,
        reason: impl Into<String>,
        start_sec: u64,
        start_usec: u64,
    ) -> Self {
        RootAction::FreezeProcess {
            pid,
            name: name.into(),
            reason: reason.into(),
            start_sec,
            start_usec,
        }
    }

    pub fn set_sysctl(
        key: impl Into<String>,
        value: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        RootAction::SetSysctl {
            key: key.into(),
            value: value.into(),
            reason: reason.into(),
        }
    }

    pub fn set_memorystatus(
        pid: u32,
        priority: i32,
        reason: impl Into<String>,
    ) -> Self {
        RootAction::SetMemorystatus {
            pid,
            priority,
            reason: reason.into(),
        }
    }

    pub fn toggle_spotlight(enabled: bool, reason: impl Into<String>) -> Self {
        RootAction::ToggleSpotlight {
            enabled,
            reason: reason.into(),
        }
    }

    pub fn unfreeze(pid: u32, name: impl Into<String>) -> Self {
        RootAction::UnfreezeProcess {
            pid,
            name: name.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FreezeSource {
    MainLoop,
    Sentinel,
    Manual,
    /// Frozen by thermal pre-throttle (Phase3Aggressive ≥90°C).
    ThermalPreThrottle,
}

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
}

fn frozen_entry_pressure_default() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenPidEntry {
    pub pid: u32,
    pub since: DateTime<Utc>,
    /// Process name captured at freeze time. Used on restart to detect PID reuse.
    #[serde(default)]
    pub name: Option<String>,
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
    // KPC hardware performance counters
    #[serde(default)]
    pub kpc_ipc: f64,
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
    /// Experience memory size (resolved outcome records).
    #[serde(default)]
    pub experience_memory_size: usize,
    // ── Action queue backpressure ─────────────────────────────────────────
    /// Current backpressure ratio of the action queue [0.0, 1.0].
    /// 0.0 = queue empty. 1.0 = queue at capacity.
    /// High values mean actions are accumulating faster than they are executed.
    #[serde(default)]
    pub action_queue_backpressure: f64,
    /// Pending unresolved outcome observations in OutcomeTracker [0, 300].
    /// High depth = throttles are being applied faster than outcomes resolve (30s window).
    #[serde(default)]
    pub outcome_pending_depth: usize,

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

/// Summary of circuit breaker and degradation state, returned by `GetHealth`.
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
        for target in [LatencyTarget::Low, LatencyTarget::Normal, LatencyTarget::Max] {
            let json = serde_json::to_string(&target).expect("serialize LatencyTarget");
            let rt: LatencyTarget =
                serde_json::from_str(&json).expect("deserialize LatencyTarget");
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
        assert!(
            !m.p95_cycle_ms.is_nan(),
            "p95_cycle_ms should not be NaN"
        );
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
}
