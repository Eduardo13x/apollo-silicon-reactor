use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::CString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Global stop flag for signal handlers (SIGTERM/SIGINT).
/// Signal handlers cannot capture Arc/closures, so we use a static AtomicBool
/// that the main loop checks alongside `state.stop`.
static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

/// SIGTERM handler — async-signal-safe: only sets an atomic flag.
extern "C" fn handle_sigterm(_sig: libc::c_int) {
    STOP_REQUESTED.store(true, Ordering::Release);
}

use anyhow::Context;
use apollo_optimizer::collector::SystemCollector;
use apollo_optimizer::engine::adaptive_governor::{
    AdaptiveGovernor, GovernorDecision, ProcessDecision,
};
use apollo_optimizer::engine::amx_detector;
use apollo_optimizer::engine::analytics::AnalyticsEngine;
use apollo_optimizer::engine::background_collectors::PressureCollector;
use apollo_optimizer::engine::capabilities::detect_capabilities;
use apollo_optimizer::engine::compressor_aware::{
    decide_enhanced, purge_purgeable_regions, query_memory_profile, scan_regions,
    sample_process_temperature, MemoryAction,
};
use apollo_optimizer::engine::workload_classifier::classify_by_memory;
use apollo_optimizer::engine::decide_actions::decide_actions;
use apollo_optimizer::engine::energy::EnergyTracker;
use apollo_optimizer::engine::execute_actions::execute_actions;
use apollo_optimizer::engine::focus_markov::FocusMarkov;
use apollo_optimizer::engine::foreground::{ForegroundDetector, ForegroundState};
use apollo_optimizer::engine::evolved_anomaly::EvolvedAnomalyDetector;
use apollo_optimizer::engine::gpu_manager::{GPUManager, GPUMetrics, GPUPowerState};
use apollo_optimizer::engine::holt_winters::HoltWinters;
use apollo_optimizer::engine::hw_bayes::HwFeatures;
use apollo_optimizer::engine::hw_predictor::{sample_hw_pressure, HwPressure};
use apollo_optimizer::engine::iokit_sensors::HardwareSnapshot;
use apollo_optimizer::engine::coalition::CoalitionTracker;
use apollo_optimizer::engine::ioreport::IOReportReader;
use apollo_optimizer::engine::jetsam_control;
use apollo_optimizer::engine::kqueue_pressure;
use apollo_optimizer::engine::latency_monitor::{self, LatencySignals};
use apollo_optimizer::engine::llm::{
    append_jsonl, delete_file_best_effort, feedback_path_root, load_repo_config, policy_path_root,
    read_json, state_paths_root, suggestions_path_root, write_json, write_secret, FeedbackEntry,
    LearnedPolicy, LlmAdvisor, LlmConfig, LlmState,
};
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::lse_counters::LockFreeMetrics;
use apollo_optimizer::engine::mach_qos::{MachQoSManager, SchedulingTier};
use apollo_optimizer::engine::memory_analyzer::MemoryAnalyzer;
use apollo_optimizer::engine::memory_budget::{self, ProcessBudgetInput};
use apollo_optimizer::engine::network_monitor::NetworkMonitor;
use apollo_optimizer::engine::network_optimizer::{NetworkOptimizer, NetworkProfile};
use apollo_optimizer::engine::causal_graph::CausalGraph;
use apollo_optimizer::engine::neuromodulator::{ApolloNeuromodulator, NeuroSignals};
use apollo_optimizer::engine::outcome_tracker::OutcomeTracker;
use apollo_optimizer::engine::overflow_guard::{OverflowGuard, BUILD_TOOLS};
use apollo_optimizer::engine::power_management::{detect_battery_status, PowerManager};
use apollo_optimizer::engine::predictive_agent::{
    specialist, AgentContext, Intervention, PredictiveAgent, SpecialistAccuracyTracker,
};
use apollo_optimizer::engine::proc_taskinfo;
use apollo_optimizer::engine::process_classifier::{ProcessSnapshot, ProcessTier};
use apollo_optimizer::engine::process_identity::ProcessIdentity;
use apollo_optimizer::engine::process_recovery::ProcessRecoveryManager;
use apollo_optimizer::engine::process_tree::{ProcessEntry, ProcessTree};
use apollo_optimizer::engine::profile_governor::{
    GovernorInput, GovernorPersisted, ProfileGovernor,
};
use apollo_optimizer::engine::protocol::{DaemonRequest, DaemonResponse};
use apollo_optimizer::engine::safety::{
    behavioral_protection_score, critical_background_processes, enforce_limits_with_budget,
    infrastructure_processes, matches_dev_runtime, pattern_conflicts_with_protected,
    protected_processes,
};
use apollo_optimizer::engine::signal_intelligence::SignalIntelligence;
use apollo_optimizer::engine::smc_reader::SmcReader;
use apollo_optimizer::engine::swap_predictor::SwapPredictor;
use apollo_optimizer::engine::sysctl_governor::{
    SysctlGovernor, SysctlGovernorInput, SysctlGovernorStatus,
};
use apollo_optimizer::engine::thermal_bailout::ThermalBailout;
use apollo_optimizer::engine::thermal_interrupt::{
    spawn_resource_sentinel, ResourceInterruptState, SentinelConfig,
};
use apollo_optimizer::engine::thermal_manager::ThermalManager;
use apollo_optimizer::engine::types::{
    BlockerScore, DaemonStatus, EnergyConsumerInfo, ForegroundAppInfo, FreezeSource, FrozenEntry,
    FrozenPidEntry, FrozenStatePersisted, HardPath, InteractiveContext, LatencyTarget,
    LearnedPolicyStatus, LlmRunMode, LlmStatus, OptimizationProfile, ProfileTransition, RootAction,
    RuntimeMetrics, SafetyPolicy, UsageResponse,
};
use apollo_optimizer::engine::usage_model::{usage_model_path_root, UsageModel};
use apollo_optimizer::engine::user_profile::{UserProfile, UserProfilePersisted};
use apollo_optimizer::engine::wait_graph;
use apollo_optimizer::engine::wake_storm_detector::WakeStormDetector;
use apollo_optimizer::engine::workload_classifier::{
    classify_workload_mode, WorkloadFeatures, WorkloadMode,
};
use apollo_optimizer::engine::zombie_hunter::HuntSnapshot;
use chrono::{DateTime, Duration as ChronoDuration, Local, Timelike, Utc};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sysinfo::ProcessStatus;

const FREEZE_TTL_SECS: i64 = 3 * 60;
const REACTOR_FAST_TICK_SECS: u64 = 30;

/// Battery-aware pressure boost: on battery, effective pressure is raised so
/// all decision gates trigger sooner.  This proactively freezes backgrounds
/// before hardware thermal throttling kicks in (critical on fanless M1 Air).
///
/// Returns a value to ADD to the raw memory_pressure reading:
///   AC power  → 0.00  (no change)
///   Battery >50% → +0.04  (slightly more aggressive)
///   Battery 20-50% → +0.10  (noticeably more aggressive)
///   Battery <20%  → +0.18  (maximum aggressiveness)
fn battery_pressure_boost(power_mgr: &PowerManager) -> f64 {
    use apollo_optimizer::engine::power_management::BatteryMode;
    if !power_mgr.is_on_battery() {
        return 0.0;
    }
    match power_mgr.battery_mode_current() {
        BatteryMode::Normal => 0.04,
        BatteryMode::LowPower => 0.10,
        BatteryMode::Critical => 0.18,
    }
}

/// Seed policy embedded at compile time — guarantees Brave, Claude, Warp, etc.
/// are always in interactive_patterns even on fresh installs or corrupt disk policy.
static SEED_POLICY: &str = include_str!("../../policy_clean.json");

/// Merge seed policy patterns into `policy` as a floor.
/// Seed patterns are always present — they can be added to but never removed.
/// Deduplicates across lists: a pattern in protected won't be added to interactive/noise.
fn merge_seed_into(policy: &mut LearnedPolicy) {
    let seed: LearnedPolicy =
        serde_json::from_str(SEED_POLICY).expect("BUG: embedded policy_clean.json is invalid");
    // Protected first (highest priority).
    for pat in &seed.protected_patterns {
        if !policy.protected_patterns.contains(pat) {
            policy.protected_patterns.push(pat.clone());
        }
    }
    // Interactive: skip if already in protected.
    for pat in &seed.interactive_patterns {
        if !policy.interactive_patterns.contains(pat)
            && !policy.protected_patterns.contains(pat)
        {
            policy.interactive_patterns.push(pat.clone());
        }
    }
    // Noise: skip if already in protected or interactive.
    for pat in &seed.noise_patterns {
        if !policy.noise_patterns.contains(pat)
            && !policy.protected_patterns.contains(pat)
            && !policy.interactive_patterns.contains(pat)
        {
            policy.noise_patterns.push(pat.clone());
        }
    }
    // Clean up cross-list conflicts: protected wins over interactive/noise.
    policy
        .interactive_patterns
        .retain(|p| !policy.protected_patterns.contains(p));
    policy
        .noise_patterns
        .retain(|p| !policy.protected_patterns.contains(p) && !policy.interactive_patterns.contains(p));
}

/// Query kernel start-time for a PID. Returns `(start_sec, start_usec)`.
/// Falls back to `(0, 0)` if the process is already dead — the action will
/// be safely skipped by `execute_actions` identity check.
fn pid_start_time(pid: u32) -> (u64, u64) {
    ProcessIdentity::from_pid(pid)
        .map(|id| (id.start_sec, id.start_usec))
        .unwrap_or((0, 0))
}

fn socket_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/run/apollo-optimizer.sock"
    } else {
        "/tmp/apollo-optimizer.sock"
    }
}

fn kill_switch_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/run/apollo.disable"
    } else {
        "/tmp/apollo.disable"
    }
}

fn journal_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/journal.jsonl"
    } else {
        "/tmp/apollo-journal.jsonl"
    }
}

fn audit_log_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/deep_scan_audit.jsonl"
    } else {
        "/tmp/apollo-deep_scan_audit.jsonl"
    }
}

/// Append a JSON line to the audit log (best-effort, never fails the caller).
fn audit_log(entry: &serde_json::Value) {
    use std::fs::OpenOptions;
    use std::io::Write;
    let path = audit_log_path();
    // Rotate at 5MB to avoid unbounded growth.
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() > 5 * 1024 * 1024 {
            let rotated = format!("{}.1", path);
            let _ = std::fs::remove_file(&rotated);
            let _ = std::fs::rename(path, &rotated);
        }
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{}", entry);
    }
}

fn metrics_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/runtime_metrics.json"
    } else {
        "/tmp/apollo-runtime_metrics.json"
    }
}

fn governor_state_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/governor_state.json"
    } else {
        "/tmp/apollo-governor_state.json"
    }
}

fn overflow_history_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/overflow_history.json"
    } else {
        "/tmp/apollo-overflow_history.json"
    }
}

fn rl_threshold_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/rl_threshold.json"
    } else {
        "/tmp/apollo-rl_threshold.json"
    }
}

fn predictive_agent_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/predictive_agent.json"
    } else {
        "/tmp/apollo-predictive_agent.json"
    }
}

fn markov_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/markov_transitions.json"
    } else {
        "/tmp/apollo-markov_transitions.json"
    }
}

fn signal_intelligence_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/signal_intelligence.json"
    } else {
        "/tmp/apollo-signal_intelligence.json"
    }
}

fn holt_winters_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/holt_winters.json"
    } else {
        "/tmp/apollo-holt_winters.json"
    }
}

fn timeline_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/profile_timeline.jsonl"
    } else {
        "/tmp/apollo-profile_timeline.jsonl"
    }
}

fn wake_state_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/wake_state.json"
    } else {
        "/tmp/apollo-wake_state.json"
    }
}

fn frozen_state_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/frozen_state.json"
    } else {
        "/tmp/apollo-frozen_state.json"
    }
}

fn hop_groups_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/hrpo_groups.json"
    } else {
        "/tmp/apollo-hrpo_groups.json"
    }
}

#[derive(Parser)]
#[command(name = "apollo-optimizerd")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Daemon {
        #[arg(long, default_value = "balanced-root")]
        profile: String,
    },
}
#[derive(Clone)]
struct SharedState {
    profile: Arc<Mutex<OptimizationProfile>>,
    latency_target: Arc<Mutex<LatencyTarget>>,
    metrics: Arc<Mutex<RuntimeMetrics>>,
    frozen_state: Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    last_blockers: Arc<Mutex<Vec<BlockerScore>>>,
    thermal_state: Arc<Mutex<String>>,
    throttle_level: Arc<Mutex<String>>,
    reactor_event_weight: Arc<Mutex<f64>>,
    fast_tick_until: Arc<Mutex<Option<Instant>>>,
    thermal_level_real: Arc<Mutex<String>>,
    reactor_status: Arc<Mutex<ReactorStatus>>,
    governor: Arc<Mutex<ProfileGovernor>>,
    timeline: Arc<Mutex<VecDeque<ProfileTransition>>>,
    wake_state: Arc<Mutex<WakeRuntimeState>>,
    stop: Arc<AtomicBool>,

    llm_cfg: Arc<LlmConfig>,
    llm_state: Arc<Mutex<LlmState>>,
    learned_policy: Arc<Mutex<LearnedPolicy>>,
    llm_state_path: PathBuf,
    llm_key_path: PathBuf,
    learned_policy_path: PathBuf,
    feedback_path: PathBuf,
    suggestions_path: PathBuf,

    config_path: PathBuf,

    usage_model: Arc<Mutex<UsageModel>>,
    usage_model_path: PathBuf,
    usage_events_path: PathBuf,
    usage_tracker: Arc<Mutex<UsageTrackerState>>,

    // Heuristic modules
    adaptive_governor: Arc<Mutex<AdaptiveGovernor>>,
    mach_qos: Arc<Mutex<MachQoSManager>>,
    last_hw_snapshot: Arc<Mutex<Option<HardwareSnapshot>>>,

    // ML Ligero
    discrepancy_log_path: PathBuf,
    user_profile_path: PathBuf,

    // Sysctl Governor status (shared with socket handler threads)
    sysctl_governor_status: Arc<Mutex<SysctlGovernorStatus>>,

    // Reactive daemon: condvar to wake the main loop on reactor events
    cycle_condvar: Arc<(Mutex<bool>, Condvar)>,
    // Resource interrupt handler state (lock-free atomics)
    resource_interrupt: Arc<ResourceInterruptState>,

    /// Clientes suscritos a push de estado (menubar, etc.)
    subscribers: Arc<Mutex<Vec<UnixStream>>>,
}

/// Reactor thread counters and status — 9 fields merged into 1 Mutex.
struct ReactorStatus {
    events_total: u64,
    events_mem: u64,
    events_thermal: u64,
    events_spawn: u64,
    events_power: u64,
    last_event_at: Option<DateTime<Utc>>,
    last_error: Option<String>,
    /// "normal" | "degraded"
    mode: String,
    /// "ok" | "stalled" | "collector-stalled"
    health: String,
}

impl Default for ReactorStatus {
    fn default() -> Self {
        Self {
            events_total: 0,
            events_mem: 0,
            events_thermal: 0,
            events_spawn: 0,
            events_power: 0,
            last_event_at: None,
            last_error: None,
            mode: "normal".to_string(),
            health: "ok".to_string(),
        }
    }
}

/// Usage model lifecycle counters — 3 fields merged into 1 Mutex.
#[derive(Default)]
struct UsageTrackerState {
    last_persist_at: Option<DateTime<Utc>>,
    promotions_day: Option<String>,
    promotions_today: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WakeStatePersisted {
    last_wake_at: Option<DateTime<Utc>>,
    post_wake_grace_until: Option<DateTime<Utc>>,
    post_wake_policy: String,
}

#[derive(Debug, Clone)]
struct WakeRuntimeState {
    last_cycle_wallclock: DateTime<Utc>,
    last_wake_at: Option<DateTime<Utc>>,
    post_wake_grace_until: Option<DateTime<Utc>>,
    post_wake_policy: String,
}

#[derive(Default)]
struct ThrashState {
    minute_started: Option<Instant>,
    cooldowns: HashMap<u32, Instant>,
}

#[derive(Default)]
struct LlmReactiveCounters {
    ws_high: u32,
    mem_high: u32,
    swap_high: u32,
    prev_trigger_active: bool,
}

fn parse_profile(input: &str) -> OptimizationProfile {
    match input {
        "aggressive-root" => OptimizationProfile::AggressiveRoot,
        "safe-root" => OptimizationProfile::SafeRoot,
        _ => OptimizationProfile::BalancedRoot,
    }
}

fn write_metrics(path: &Path, metrics: &RuntimeMetrics) {
    write_json(path, metrics, Some(0o600));
}

fn write_governor_state(path: &Path, governor: &ProfileGovernor) {
    write_json(path, &governor.to_persisted(), Some(0o600));
}

fn load_governor_state(path: &Path, fallback_profile: OptimizationProfile) -> ProfileGovernor {
    if let Ok(data) = HardPath::read_to_string_limited(path, 1024 * 1024) {
        if let Ok(state) = serde_json::from_str::<GovernorPersisted>(&data) {
            return ProfileGovernor::from_persisted(state);
        }
    }
    ProfileGovernor::new(fallback_profile)
}

fn append_timeline(path: &Path, transition: &ProfileTransition) {
    append_jsonl(path, transition);
    rotate_timeline(path);
}

fn write_wake_state(path: &Path, state: &WakeRuntimeState) {
    let persisted = WakeStatePersisted {
        last_wake_at: state.last_wake_at,
        post_wake_grace_until: state.post_wake_grace_until,
        post_wake_policy: state.post_wake_policy.clone(),
    };
    write_json(path, &persisted, Some(0o600));
}

fn load_wake_state(path: &Path) -> WakeRuntimeState {
    let now = Utc::now();
    if let Ok(data) = HardPath::read_to_string_limited(path, 1024 * 1024) {
        if let Ok(state) = serde_json::from_str::<WakeStatePersisted>(&data) {
            return WakeRuntimeState {
                last_cycle_wallclock: now,
                last_wake_at: state.last_wake_at,
                post_wake_grace_until: state.post_wake_grace_until,
                post_wake_policy: state.post_wake_policy,
            };
        }
    }
    WakeRuntimeState {
        last_cycle_wallclock: now,
        last_wake_at: None,
        post_wake_grace_until: None,
        post_wake_policy: "grace-60s".to_string(),
    }
}

fn write_frozen_state(path: &Path, frozen_state: &HashMap<u32, FrozenEntry>) {
    let persisted = FrozenStatePersisted {
        frozen: frozen_state
            .iter()
            .map(|(pid, entry)| FrozenPidEntry {
                pid: *pid,
                since: entry.frozen_at,
            })
            .collect(),
    };
    write_json(path, &persisted, Some(0o600));
}

fn load_frozen_state(path: &Path) -> HashMap<u32, FrozenEntry> {
    if let Ok(data) = HardPath::read_to_string_limited(path, 10 * 1024 * 1024) {
        if let Ok(state) = serde_json::from_str::<FrozenStatePersisted>(&data) {
            return state
                .frozen
                .into_iter()
                .map(|e| {
                    (
                        e.pid,
                        FrozenEntry {
                            frozen_at: e.since,
                            source: FreezeSource::MainLoop,
                            // Unknown pressure when loaded from disk; use 1.0 so
                            // only the TTL path can trigger unfreeze for these entries.
                            pressure_at_freeze: 1.0,
                        },
                    )
                })
                .collect();
        }
    }
    HashMap::new()
}

fn unfreeze_pids(pids: impl Iterator<Item = u32>) -> u64 {
    let mut count = 0_u64;
    for pid in pids {
        unsafe {
            libc::kill(pid as i32, libc::SIGCONT);
        }
        count += 1;
    }
    count
}

/// Returns true when a frozen process should be thawed.
///
/// Two independent conditions can trigger an early unfreeze:
/// - **TTL**: the process has been frozen longer than `FREEZE_TTL_SECS` (safety net).
/// - **Pressure recovery**: current pressure has dropped to ≤60% of the pressure at
///   freeze time AND at least 30 s have elapsed (prevents immediate re-thaw on transient
///   pressure spikes and avoids thrash when frozen just moments ago).
///
/// This is a pure function so it can be tested without daemon state.
fn should_unfreeze(
    elapsed_secs: i64,
    pressure_at_freeze: f64,
    current_pressure: f64,
) -> bool {
    let ttl_expired = elapsed_secs > FREEZE_TTL_SECS;
    // Pressure recovery: two paths — relative drop OR absolute improvement.
    // On 8GB M1, pressure rarely drops much, so use 5pp threshold (was 10pp).
    let pressure_recovered = elapsed_secs >= 30
        && pressure_at_freeze > 0.0
        && (current_pressure <= pressure_at_freeze * 0.6
            || (pressure_at_freeze - current_pressure) >= 0.05);
    // Stale safety: after 2 min, unfreeze if pressure is at least mildly lower.
    let stale_with_improvement = elapsed_secs >= 120
        && current_pressure < pressure_at_freeze;
    ttl_expired || pressure_recovered || stale_with_improvement
}

/// On memory-constrained hardware (8GB), rotate frozen processes: if ≥3 are
/// frozen and the oldest has been frozen ≥90s, unfreeze it to prevent resource
/// hoarding even when pressure stays high.
fn should_rotate_oldest(elapsed_secs: i64, total_frozen: usize) -> bool {
    total_frozen >= 2 && elapsed_secs >= 60
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_unfreeze_ttl_path() {
        // TTL expired → always unfreeze regardless of pressure.
        assert!(should_unfreeze(FREEZE_TTL_SECS + 1, 0.80, 0.80));
        assert!(should_unfreeze(FREEZE_TTL_SECS + 1, 0.80, 0.90));
    }

    #[test]
    fn test_should_unfreeze_pressure_recovery() {
        // Pressure at freeze was 0.80, now 0.45 (< 0.80 * 0.6 = 0.48) → unfreeze.
        assert!(should_unfreeze(60, 0.80, 0.45));
        // 5pp drop: 0.80 → 0.75 → unfreeze (relaxed for 8GB).
        assert!(should_unfreeze(60, 0.80, 0.75));
        // Only 3pp drop: not enough.
        assert!(!should_unfreeze(60, 0.80, 0.77));
    }

    #[test]
    fn test_should_unfreeze_min_30s_guard() {
        // Even if pressure recovered, don't unfreeze before 30 s to avoid thrash.
        assert!(!should_unfreeze(29, 0.80, 0.10));
        assert!(should_unfreeze(30, 0.80, 0.10));
    }

    #[test]
    fn test_should_unfreeze_high_pressure_at_freeze() {
        // pressure_at_freeze == 1.0 (max): pressure_recovered threshold = 0.60.
        // current = 0.10 < 0.60 → unfreeze after 30 s (pressure clearly recovered).
        assert!(should_unfreeze(60, 1.0, 0.10));
        // current = 0.65 > 0.60 → still above threshold, do NOT unfreeze early.
        assert!(!should_unfreeze(60, 1.0, 0.65));
        // TTL always triggers regardless.
        assert!(should_unfreeze(FREEZE_TTL_SECS + 1, 1.0, 0.90));
    }

    #[test]
    fn test_should_unfreeze_zero_pressure_at_freeze() {
        // pressure_at_freeze == 0.0: guard against always-true result.
        assert!(!should_unfreeze(60, 0.0, 0.0));
        assert!(!should_unfreeze(60, 0.0, 0.10));
    }

    #[test]
    fn test_should_unfreeze_stale_at_2min() {
        // After 2 min (was 5 min), any pressure improvement → unfreeze.
        assert!(should_unfreeze(120, 0.75, 0.74));
        assert!(!should_unfreeze(119, 0.75, 0.74));
        // No improvement → no unfreeze.
        assert!(!should_unfreeze(120, 0.75, 0.75));
    }

    #[test]
    fn test_should_rotate_oldest() {
        // ≥2 frozen and oldest ≥60s → rotate.
        assert!(should_rotate_oldest(60, 2));
        assert!(should_rotate_oldest(200, 5));
        // Not enough frozen.
        assert!(!should_rotate_oldest(60, 1));
        // Too fresh.
        assert!(!should_rotate_oldest(59, 2));
    }
}

fn run_reactor(state: SharedState) -> anyhow::Result<()> {
    unsafe {
        let kq = libc::kqueue();
        if kq == -1 {
            state.reactor_status.lock_recover().last_error = Some("kqueue failed".to_string());
            return Ok(());
        }

        // Memory pressure via sysctl polling (all push APIs are broken on macOS 15).
        // Polls kern.memorystatus_vm_pressure_level (~1µs) on each loop iteration.
        let mut pressure_monitor = apollo_optimizer::engine::dispatch_pressure::KernelPressureMonitor::new();

        // notify -> thermal
        let mut thermal_fd: libc::c_int = 0;
        let mut thermal_token: libc::c_int = 0;
        let thermal_name = CString::new("com.apple.system.thermalpressurelevel")
            .expect("static string should not contain NUL");
        let thermal_reg = notify_register_file_descriptor(
            thermal_name.as_ptr(),
            &mut thermal_fd,
            0,
            &mut thermal_token,
        );
        if thermal_reg != 0 {
            state.reactor_status.lock_recover().last_error = Some(format!(
                "thermal notify_register_file_descriptor failed: {}",
                thermal_reg
            ));
        }
        if thermal_fd > 0 {
            let kev = libc::kevent {
                ident: thermal_fd as usize,
                filter: libc::EVFILT_READ,
                flags: libc::EV_ADD | libc::EV_ENABLE,
                fflags: 0,
                data: 0,
                udata: 2 as *mut libc::c_void, // ID 2 = Thermal
            };
            let _ = libc::kevent(kq, &kev, 1, std::ptr::null_mut(), 0, std::ptr::null());
        }

        // notify -> lifecycle spawn
        // NOTE: com.apple.launchd.spawn is a private notification and never
        // delivers to external processes (reactor_events_spawn stays 0).
        // Replaced with EVFILT_PROC NOTE_FORK on launchd PID 1 — fires on
        // every process fork from launchd, which is the actual mechanism we
        // wanted to observe.
        let launch_fd: libc::c_int = -1;
        let launchd_kev = libc::kevent {
            ident: 1, // launchd PID
            filter: libc::EVFILT_PROC,
            flags: libc::EV_ADD | libc::EV_ENABLE | libc::EV_CLEAR,
            fflags: libc::NOTE_FORK as u32,
            data: 0,
            udata: 3 as *mut libc::c_void, // ID 3 = Lifecycle
        };
        let fork_rc = libc::kevent(
            kq,
            &launchd_kev,
            1,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
        );
        if fork_rc < 0 {
            let errno = *libc::__error();
            state.reactor_status.lock_recover().last_error = Some(format!(
                "EVFILT_PROC NOTE_FORK on launchd failed errno={}",
                errno
            ));
        }

        // notify -> power
        let mut power_fd: libc::c_int = 0;
        let mut power_token: libc::c_int = 0;
        let power_name = CString::new("com.apple.system.powersources.source")
            .expect("static string should not contain NUL");
        let power_reg = notify_register_file_descriptor(
            power_name.as_ptr(),
            &mut power_fd,
            0,
            &mut power_token,
        );
        if power_reg != 0 {
            state.reactor_status.lock_recover().last_error = Some(format!(
                "power notify_register_file_descriptor failed: {}",
                power_reg
            ));
        }
        if power_fd > 0 {
            let kev = libc::kevent {
                ident: power_fd as usize,
                filter: libc::EVFILT_READ,
                flags: libc::EV_ADD | libc::EV_ENABLE,
                fflags: 0,
                data: 0,
                udata: 4 as *mut libc::c_void, // ID 4 = Power
            };
            let _ = libc::kevent(kq, &kev, 1, std::ptr::null_mut(), 0, std::ptr::null());
        }

        let mut out_ev = std::mem::zeroed::<libc::kevent>();
        let timeout = libc::timespec {
            tv_sec: 1,
            tv_nsec: 0,
        };
        while !state.stop.load(Ordering::Acquire) && !STOP_REQUESTED.load(Ordering::Acquire) {
            let n = libc::kevent(kq, std::ptr::null(), 0, &mut out_ev, 1, &timeout);
            // Pulse on every iteration (event or timeout) so main loop can
            // distinguish a live-but-quiet reactor from a dead one.
            {
                let mut m = state.metrics.lock_recover();
                m.reactor_pulses += 1;
            }
            // Poll kernel pressure level on every iteration (~1µs sysctl read).
            // Fires memory signal on level transitions (Normal↔Warning↔Critical).
            if let Some(level) = pressure_monitor.poll() {
                use apollo_optimizer::engine::dispatch_pressure::KernelPressureLevel;
                if level >= KernelPressureLevel::Warning {
                    state
                        .resource_interrupt
                        .memory_signal
                        .store(true, Ordering::Release);
                }
                state.reactor_status.lock_recover().events_mem += 1;
                // Wake main loop for pressure transition.
                {
                    let (lock, cvar) = &*state.cycle_condvar;
                    let mut triggered = lock.lock_recover();
                    *triggered = true;
                    cvar.notify_one();
                }
            }
            if n == 0 {
                // Timeout — no events within 1 second.  Continue the loop so the
                // condvar pulse above keeps the main loop aware the reactor is alive.
                continue;
            }
            if n < 0 {
                // kevent error (e.g. EINTR).  Record the error and retry.
                let errno = *libc::__error();
                if errno != libc::EINTR {
                    state.reactor_status.lock_recover().last_error =
                        Some(format!("kevent error errno={}", errno));
                }
                continue;
            }

            let id = out_ev.udata as usize;
            // Update shared counters + status in one lock acquisition.
            let reactor_mode = {
                let mut rs = state.reactor_status.lock_recover();
                rs.events_total += 1;
                rs.last_event_at = Some(Utc::now());
                rs.health = "ok".to_string();
                rs.mode.clone()
            };
            if id == 2 {
                // Drain thermal pipe
                let mut dummy: i32 = 0;
                let _ = libc::read(thermal_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                state.reactor_status.lock_recover().events_thermal += 1;
                let level = match dummy {
                    0 => "nominal",
                    1 => "moderate",
                    2 => "serious",
                    _ => "critical",
                };
                *state.thermal_level_real.lock_recover() = level.to_string();
                // Signal resource sentinel for thermal ≥ serious.
                if dummy >= 2 {
                    state
                        .resource_interrupt
                        .thermal_signal
                        .store(true, Ordering::Release);
                }
            } else if id == 3 {
                let mut dummy: i32 = 0;
                let _ = libc::read(launch_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                state.reactor_status.lock_recover().events_spawn += 1;
            } else if id == 4 {
                let mut dummy: i32 = 0;
                let _ = libc::read(power_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                state.reactor_status.lock_recover().events_power += 1;
                // Signal resource sentinel for power source changes.
                state
                    .resource_interrupt
                    .power_signal
                    .store(true, Ordering::Release);
            } else if id == 1 {
                state.reactor_status.lock_recover().events_mem += 1;
                state
                    .resource_interrupt
                    .memory_signal
                    .store(true, Ordering::Release);
            }

            *state.reactor_event_weight.lock_recover() = 1.0;
            if reactor_mode.as_str() == "normal" {
                *state.fast_tick_until.lock_recover() =
                    Some(Instant::now() + Duration::from_secs(REACTOR_FAST_TICK_SECS));
            }

            // NOTE: reactor_pulses is already incremented once per loop
            // iteration at the top of the loop (including timeouts). Do not
            // increment again here — that would double-count real events.

            // Wake the main loop immediately via condvar.
            {
                let (lock, cvar) = &*state.cycle_condvar;
                let mut triggered = lock.lock_recover();
                *triggered = true;
                cvar.notify_one();
            }
        }

        if thermal_fd > 0 {
            libc::close(thermal_fd);
        }
        // launch_fd removed: now using EVFILT_PROC on launchd PID 1 (no fd to close)
        let _ = launch_fd; // suppress unused variable warning
        if power_fd > 0 {
            libc::close(power_fd);
        }
        libc::close(kq);
    }

    Ok(())
}

fn rotate_timeline(path: &Path) {
    const MAX_BYTES: u64 = 10 * 1024 * 1024;
    // Use symlink_metadata to avoid following symlinks
    if fs::symlink_metadata(path)
        .map(|m| !m.file_type().is_symlink() && m.len() > MAX_BYTES)
        .unwrap_or(false)
    {
        let p1 = format!("{}.1", path.display());
        let p2 = format!("{}.2", path.display());
        let p3 = format!("{}.3", path.display());
        let _ = fs::remove_file(&p3);
        let _ = fs::rename(&p2, &p3);
        let _ = fs::rename(&p1, &p2);
        let _ = fs::rename(path, &p1);
    }
}

#[link(name = "System")]
extern "C" {
    fn notify_register_file_descriptor(
        name: *const libc::c_char,
        out_fd: *mut libc::c_int,
        flags: libc::c_int,
        out_token: *mut libc::c_int,
    ) -> u32;
}

fn compute_p95(samples: &[u64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let idx = (((sorted.len() - 1) as f64) * 0.95).round() as usize;
    sorted[idx] as f64
}

fn filter_boost_cooldown(
    actions: Vec<RootAction>,
    policy: &SafetyPolicy,
    thrash: &mut ThrashState,
) -> Vec<RootAction> {
    let now = Instant::now();
    let cooldown = Duration::from_secs(policy.cooldown_seconds);
    let mut out = Vec::new();

    thrash
        .cooldowns
        .retain(|_, ts| now.duration_since(*ts) <= Duration::from_secs(300));

    for action in actions {
        match &action {
            RootAction::BoostProcess { pid, .. } => {
                if let Some(last) = thrash.cooldowns.get(pid) {
                    if now.duration_since(*last) < cooldown {
                        continue;
                    }
                }
                thrash.cooldowns.insert(*pid, now);
                out.push(action);
            }
            _ => out.push(action),
        }
    }

    out
}

fn apply_post_wake_grace_policy(
    actions: Vec<RootAction>,
    grace_active: bool,
) -> (Vec<RootAction>, u64, u64) {
    if !grace_active {
        return (actions, 0, 0);
    }

    let mut out = Vec::with_capacity(actions.len());
    let mut freeze_suppressed = 0_u64;
    let mut throttle_suppressed = 0_u64;

    for action in actions {
        match action {
            RootAction::FreezeProcess { .. } | RootAction::QuarantineDaemon { .. } => {
                freeze_suppressed += 1;
            }
            RootAction::ThrottleProcess {
                pid,
                name,
                aggressive: true,
                reason,
                start_sec,
                start_usec,
            } => {
                throttle_suppressed += 1;
                out.push(RootAction::ThrottleProcess {
                    pid,
                    name,
                    aggressive: false,
                    reason,
                    start_sec,
                    start_usec,
                });
            }
            _ => out.push(action),
        }
    }

    (out, throttle_suppressed, freeze_suppressed)
}

fn is_peer_root(stream: &UnixStream) -> bool {
    // If we're not running as root, anyone who can connect is allowed (usually protected by dir perms)
    if unsafe { libc::geteuid() } != 0 {
        return true;
    }

    #[cfg(target_os = "macos")]
    {
        let mut euid: libc::uid_t = 0;
        let mut egid: libc::gid_t = 0;
        let res = unsafe { libc::getpeereid(stream.as_raw_fd(), &mut euid, &mut egid) };
        if res == 0 {
            return euid == 0;
        }
    }
    // Default to false for security if we can't verify
    false
}

fn handle_client(mut stream: UnixStream, state: &SharedState) {
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(5)));
    let is_root = is_peer_root(&stream);

    // Lee y parsea la peticion (reader se libera al salir del bloque)
    let req_result = {
        let mut reader = BufReader::new(&stream);
        const MAX_REQUEST_BYTES: u64 = 65_536;
        let mut line = String::new();
        match reader.by_ref().take(MAX_REQUEST_BYTES).read_line(&mut line) {
            Ok(_) => serde_json::from_str::<DaemonRequest>(&line)
                .map_err(|e| format!("invalid request: {e}")),
            Err(e) => Err(format!("read error: {e}")),
        }
    };

    let mut req = match req_result {
        Ok(r) => r,
        Err(msg) => {
            if let Ok(text) = serde_json::to_string(&DaemonResponse::Error { message: msg }) {
                let _ = writeln!(stream, "{}", text);
            }
            return;
        }
    };
    req.sanitize();

    // Suscripcion push: conexion persistente, el daemon enviara StatusPush cada ciclo
    if let DaemonRequest::Subscribe = req {
        if let Ok(text) = serde_json::to_string(&DaemonResponse::Ok) {
            let _ = writeln!(stream, "{}", text);
        }
        if let Ok(write_clone) = stream.try_clone() {
            state.subscribers.lock_recover().push(write_clone);
        }
        // Bloquear hasta que el cliente desconecte; la limpieza es lazy (fallo de escritura)
        let _ = stream.set_read_timeout(None);
        let mut buf = [0u8; 1];
        loop {
            match Read::read(&mut stream, &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
        return;
    }

    if req.is_privileged() && !is_root {
        if let Ok(text) = serde_json::to_string(&DaemonResponse::Error {
            message: "privileged command requires root/sudo".to_string(),
        }) {
            let _ = writeln!(stream, "{}", text);
        }
        return;
    }

    let response = process_request(req, state);
    if let Ok(text) = serde_json::to_string(&response) {
        let _ = writeln!(stream, "{}", text);
    }
}

/// Broadcast del estado actual a todos los suscriptores.
/// Los streams que fallen (cliente desconectado) se eliminan automaticamente.
fn broadcast_current_status(state: &SharedState) {
    let mut subs = state.subscribers.lock_recover();
    if subs.is_empty() {
        return;
    }
    let DaemonResponse::Status(status) = process_request(DaemonRequest::GetStatus, state) else {
        return;
    };
    let Ok(text) = serde_json::to_string(&DaemonResponse::StatusPush(status)) else {
        return;
    };
    subs.retain_mut(|stream| writeln!(stream, "{}", text).is_ok());
}

fn process_request(req: DaemonRequest, state: &SharedState) -> DaemonResponse {
    match req {
        DaemonRequest::GetStatus => {
            let now = Utc::now();
            let profile = *state.profile.lock_recover();
            let latency_target = *state.latency_target.lock_recover();
            // Non-blocking metrics: try_lock avoids stalling when the main loop
            // holds the metrics lock during its end-of-cycle update (~100 lines).
            // Fall back to default metrics if busy — dashboard shows stale data
            // briefly, but never hangs.
            let metrics = match state.metrics.try_lock() {
                Ok(m) => m.clone(),
                Err(_) => {
                    // Lock held by main loop — read last-written snapshot from disk.
                    // This is always ≤1 cycle old (written at end of each cycle).
                    match std::fs::read_to_string(metrics_path()) {
                        Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
                        Err(_) => RuntimeMetrics::default(),
                    }
                }
            };
            let blockers = state.last_blockers.lock_recover().clone();
            let thermal_state = state.thermal_state.lock_recover().clone();
            let throttle_level = state.throttle_level.lock_recover().clone();
            // Snapshot governor + wake_state, then DROP locks before build_llm_status.
            let (auto_profile_enabled, base_profile, override_active, override_expires_at,
                 transition_reason) = {
                let gov = state.governor.lock_recover();
                (gov.auto_profile_enabled, gov.base_profile,
                 gov.manual_override.is_some(),
                 gov.manual_override.as_ref().map(|o| o.expires_at),
                 gov.transition_reason.clone())
            };
            let (grace_active, grace_remaining, last_wake_at, post_wake_policy) = {
                let ws = state.wake_state.lock_recover();
                let ga = ws.post_wake_grace_until.map(|t| t > now).unwrap_or(false);
                let gr = ws.post_wake_grace_until
                    .and_then(|t| (t - now).to_std().ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                (ga, gr, ws.last_wake_at, ws.post_wake_policy.clone())
            };
            let (reactor_mode, reactor_health) = {
                let rs = state.reactor_status.lock_recover();
                (rs.mode.clone(), rs.health.clone())
            };
            let llm = build_llm_status(state);
            let status = DaemonStatus {
                running: !state.stop.load(Ordering::Acquire),
                profile,
                latency_target,
                effective_profile: metrics.effective_profile,
                kill_switch: Path::new(kill_switch_path()).exists(),
                throttle_level,
                thermal_state,
                last_blockers: blockers,
                auto_profile_enabled,
                base_profile,
                override_active,
                override_expires_at,
                transition_reason,
                post_wake_grace_active: grace_active,
                post_wake_grace_remaining_secs: grace_remaining,
                last_wake_at,
                post_wake_policy,
                reactor_mode,
                reactor_health,
                metrics,
                llm: Some(llm),
            };
            DaemonResponse::Status(status)
        }
        DaemonRequest::GetMetrics => DaemonResponse::Metrics(state.metrics.lock_recover().clone()),
        DaemonRequest::GetTopBlockers => {
            DaemonResponse::TopBlockers(state.last_blockers.lock_recover().clone())
        }
        DaemonRequest::GetProfileTimeline => {
            DaemonResponse::ProfileTimeline(state.timeline.lock_recover().iter().cloned().collect())
        }
        DaemonRequest::GetCapabilities => DaemonResponse::Capabilities(detect_capabilities()),
        DaemonRequest::SetProfile {
            profile,
            ttl_minutes,
        } => {
            let ttl = ttl_minutes.unwrap_or(20).clamp(1, 1440);
            state.governor.lock_recover().set_manual_override(
                profile,
                ttl,
                "cli-set-profile".to_string(),
            );
            DaemonResponse::Ok
        }
        DaemonRequest::SetLatencyTarget { target } => {
            *state.latency_target.lock_recover() = target;
            DaemonResponse::Ok
        }
        DaemonRequest::SetAutoProfile { enabled } => {
            state.governor.lock_recover().set_auto_profile(enabled);
            DaemonResponse::Ok
        }
        DaemonRequest::ClearProfileOverride => {
            state.governor.lock_recover().clear_manual_override();
            DaemonResponse::Ok
        }
        DaemonRequest::Restore => {
            let mut frozen_state = state.frozen_state.lock_recover();
            for pid in frozen_state.keys() {
                unsafe {
                    libc::kill(*pid as i32, libc::SIGCONT);
                }
            }
            frozen_state.clear();
            let _ = fs::remove_file(kill_switch_path());
            DaemonResponse::Ok
        }
        DaemonRequest::PanicRestore => {
            // Symlink protection: refuse to create kill switch through a symlink
            let ks = kill_switch_path();
            let ks_path = Path::new(ks);
            if ks_path.exists() {
                if let Ok(meta) = fs::symlink_metadata(ks_path) {
                    if meta.file_type().is_symlink() {
                        return DaemonResponse::Error {
                            message: "kill switch path is a symlink — refusing".to_string(),
                        };
                    }
                }
            }
            let _ = File::create(ks);
            state.governor.lock_recover().set_auto_profile(false);
            let mut frozen_state = state.frozen_state.lock_recover();
            for pid in frozen_state.keys() {
                unsafe {
                    libc::kill(*pid as i32, libc::SIGCONT);
                }
            }
            frozen_state.clear();
            DaemonResponse::Ok
        }
        DaemonRequest::Doctor => {
            let caps = detect_capabilities();
            let checks = vec![
                format!("is_root: {}", caps.is_root),
                format!("taskpolicy: {}", caps.can_taskpolicy),
                format!("sysctl: {}", caps.can_sysctl),
                format!("mdutil: {}", caps.can_mdutil),
                format!("tmutil: {}", caps.can_tmutil),
                format!("socket_exists: {}", Path::new(socket_path()).exists()),
                format!("kill_switch: {}", Path::new(kill_switch_path()).exists()),
                {
                    let rs = state.reactor_status.lock_recover();
                    format!("reactor_mode: {}", rs.mode)
                },
                {
                    let rs = state.reactor_status.lock_recover();
                    format!("reactor_health: {}", rs.health)
                },
                format!(
                    "swapusage_readable: {}",
                    apollo_optimizer::engine::sysctl_direct::read_swap_usage().is_some()
                ),
                format!(
                    "memory_pressure_readable: {}",
                    apollo_optimizer::engine::host_vm_info::read_vm_stats().is_some()
                ),
            ];
            DaemonResponse::Doctor { checks }
        }
        DaemonRequest::GetLlmStatus => DaemonResponse::LlmStatus(build_llm_status(state)),
        DaemonRequest::UsageTop { limit } => {
            let limit = limit.unwrap_or(10).clamp(3, 30);
            let model = state.usage_model.lock_recover();
            let report = model.top_report(limit);
            DaemonResponse::Usage(UsageResponse::Top(report))
        }
        DaemonRequest::UsageExplain { name } => {
            let model = state.usage_model.lock_recover();
            match model.entry_summary(&name) {
                Some(s) => DaemonResponse::Usage(UsageResponse::Explain(s)),
                None => DaemonResponse::Error {
                    message: "usage entry not found".to_string(),
                },
            }
        }
        DaemonRequest::LlmSetKey { api_key, ttl_days } => {
            let now = Utc::now();
            let ttl_clamped = ttl_days.clamp(1, 365);
            let expires = now + ChronoDuration::days(ttl_clamped as i64);
            if write_secret(&state.llm_key_path, api_key.trim()).is_err() {
                return DaemonResponse::Error {
                    message: "failed to write llm key".to_string(),
                };
            }
            {
                let mut llm_state = state.llm_state.lock_recover();
                llm_state.enabled = true;
                llm_state.training_started_at = Some(now);
                llm_state.training_expires_at = Some(expires);
                llm_state.last_call_at = None;
                llm_state.last_attempt_at = None;
                llm_state.last_http_status = None;
                llm_state.last_error = None;
                llm_state.last_trigger_reason = None;
                llm_state.consecutive_failures = 0;
                llm_state.calls_in_window = 0;
                llm_state.hour_window_started_at = Some(now);
                llm_state.calls_today_day = None;
                llm_state.calls_today = 0;
                llm_state.mode = LlmRunMode::Sensitive;
                llm_state.last_trigger_at = None;
                llm_state.trigger_events.clear();
                llm_state.no_trigger_since = Some(now);
                llm_state.last_suggestion = None;
                llm_state.policy_updates_day = None;
                llm_state.policy_updates_today = 0;
                write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            }
            DaemonResponse::Ok
        }
        DaemonRequest::LlmDisable => {
            delete_file_best_effort(&state.llm_key_path);
            {
                let mut llm_state = state.llm_state.lock_recover();
                llm_state.enabled = false;
                llm_state.training_expires_at = None;
                llm_state.last_suggestion = None;
                write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            }
            DaemonResponse::Ok
        }
        DaemonRequest::LlmTest => {
            let now = Utc::now();
            let llm_cfg = load_repo_config(&state.config_path)
                .llm
                .unwrap_or_else(|| state.llm_cfg.as_ref().clone());
            if !llm_cfg.enabled() {
                return DaemonResponse::LlmTestResult {
                    ok: false,
                    http_status: None,
                    error: Some("llm disabled in config".to_string()),
                    suggestion: None,
                };
            }
            if !state.llm_key_path.exists() {
                return DaemonResponse::LlmTestResult {
                    ok: false,
                    http_status: None,
                    error: Some("missing llm api key".to_string()),
                    suggestion: None,
                };
            }
            {
                let llm_state = state.llm_state.lock_recover();
                if !llm_state.training_active() {
                    return DaemonResponse::LlmTestResult {
                        ok: false,
                        http_status: None,
                        error: Some("training not active (enable + ttl)".to_string()),
                        suggestion: None,
                    };
                }
            }

            let api_key = match HardPath::read_to_string_limited(&state.llm_key_path, 4096) {
                Ok(v) => v,
                Err(_) => {
                    return DaemonResponse::LlmTestResult {
                        ok: false,
                        http_status: None,
                        error: Some("cannot read llm key".to_string()),
                        suggestion: None,
                    }
                }
            };

            // Collect a one-off snapshot for this test.
            let mut collector = SystemCollector::new();
            let mut snapshot = collector.collect_snapshot();
            snapshot.pressure.thermal_level = state.thermal_level_real.lock_recover().clone();

            // Record attempt immediately.
            {
                let mut llm_state = state.llm_state.lock_recover();
                if llm_state.training_started_at.is_none() {
                    llm_state.training_started_at = Some(now);
                }
                llm_state.last_attempt_at = Some(now);
                llm_state.last_trigger_reason = Some("manual-test".to_string());
                llm_state.last_error = None;
                llm_state.last_http_status = None;

                // Count this as a call attempt for observability/budget.
                let today = Local::now().date_naive().to_string();
                if llm_state.calls_today_day.as_deref() != Some(&today) {
                    llm_state.calls_today_day = Some(today);
                    llm_state.calls_today = 0;
                }
                llm_state.calls_today += 1;
                if llm_state
                    .hour_window_started_at
                    .map(|t| now - t > ChronoDuration::hours(1))
                    .unwrap_or(true)
                {
                    llm_state.hour_window_started_at = Some(now);
                    llm_state.calls_in_window = 0;
                }
                llm_state.calls_in_window += 1;

                write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            }

            let mut advisor = LlmAdvisor::new(llm_cfg.clone());
            let current_policy = state.learned_policy.lock_recover().clone();
            match advisor.call_raw(&snapshot, &api_key, Some(&current_policy)) {
                Ok(suggestion) => {
                    {
                        let mut llm_state = state.llm_state.lock_recover();
                        llm_state.last_call_at = Some(now);
                        llm_state.last_http_status = Some(200);
                        llm_state.last_suggestion = Some(suggestion.clone());
                        llm_state.last_error = None;
                        write_json(&state.llm_state_path, &*llm_state, Some(0o600));
                    }
                    DaemonResponse::LlmTestResult {
                        ok: true,
                        http_status: Some(200),
                        error: None,
                        suggestion: Some(suggestion),
                    }
                }
                Err(err) => {
                    let (http_status, msg) = match err {
                        apollo_optimizer::engine::llm::LlmCallError::Cooldown => {
                            (None, "cooldown".to_string())
                        }
                        apollo_optimizer::engine::llm::LlmCallError::HttpStatus {
                            code,
                            body_excerpt,
                        } => (
                            Some(code),
                            format!("http {} {}", code, body_excerpt.unwrap_or_default()),
                        ),
                        apollo_optimizer::engine::llm::LlmCallError::Transport(e) => {
                            (None, format!("transport {}", e))
                        }
                        apollo_optimizer::engine::llm::LlmCallError::Parse(e) => {
                            (None, format!("parse {}", e))
                        }
                        apollo_optimizer::engine::llm::LlmCallError::Rejected(e) => {
                            (None, format!("rejected {}", e))
                        }
                    };
                    {
                        let mut llm_state = state.llm_state.lock_recover();
                        llm_state.last_http_status = http_status;
                        llm_state.last_error = Some(msg.clone());
                        write_json(&state.llm_state_path, &*llm_state, Some(0o600));
                    }
                    DaemonResponse::LlmTestResult {
                        ok: false,
                        http_status,
                        error: Some(msg),
                        suggestion: None,
                    }
                }
            }
        }
        DaemonRequest::GetLearnedPolicy => {
            let policy = state.learned_policy.lock_recover().clone();
            DaemonResponse::LearnedPolicy(policy)
        }
        DaemonRequest::SetLearnedPolicy { policy: new_policy } => {
            // Validate size limits to prevent OOM attacks
            const MAX_PATTERNS: usize = 500;
            if new_policy.interactive_patterns.len() > MAX_PATTERNS
                || new_policy.noise_patterns.len() > MAX_PATTERNS
                || new_policy.protected_patterns.len() > MAX_PATTERNS
            {
                DaemonResponse::Error {
                    message: format!(
                        "Policy too large: max {} patterns per category",
                        MAX_PATTERNS
                    ),
                }
            } else {
                // Validate individual pattern lengths.
                const MAX_PATTERN_LEN: usize = 256;
                const MIN_PATTERN_LEN: usize = 4;
                let has_invalid_pattern = new_policy
                    .interactive_patterns
                    .iter()
                    .chain(new_policy.noise_patterns.iter())
                    .chain(new_policy.protected_patterns.iter())
                    .any(|p| {
                        p.len() > MAX_PATTERN_LEN
                            || p.len() < MIN_PATTERN_LEN
                            || p.trim().is_empty()
                            || p.chars().any(|c| {
                                // Reject control chars and glob/regex metacharacters.
                                // Parentheses are intentionally allowed: macOS process
                                // names use them legitimately, e.g. "Helper (GPU)".
                                // Patterns are matched with str::contains(), not regex.
                                c.is_control()
                                    || c == '*'
                                    || c == '['
                                    || c == ']'
                                    || c == '|'
                                    || c == '\\'
                                    || c == '{'
                                    || c == '}'
                            })
                    });
                if has_invalid_pattern {
                    return DaemonResponse::Error {
                        message: format!(
                            "pattern length must be {}-{} chars, non-empty",
                            MIN_PATTERN_LEN, MAX_PATTERN_LEN
                        ),
                    };
                }

                // Sanitize: strip any patterns that could match a
                // hardcoded protected or critical-background process.
                // Uses bidirectional prefix/suffix overlap (75% threshold)
                // to block evasion attempts like "kernel_tas" for "kernel_task".
                let mut sanitized = new_policy;
                sanitized
                    .noise_patterns
                    .retain(|pat| !pattern_conflicts_with_protected(pat));
                sanitized
                    .interactive_patterns
                    .retain(|pat| !pattern_conflicts_with_protected(pat));
                sanitized
                    .protected_patterns
                    .retain(|pat| !pattern_conflicts_with_protected(pat));
                let mut policy = state.learned_policy.lock_recover();
                *policy = sanitized;
                // Re-merge seed as floor — seed patterns can never be removed.
                merge_seed_into(&mut policy);
                policy.learned_at = Some(Utc::now());
                write_json(&state.learned_policy_path, &*policy, Some(0o600));
                // Propagate to ML classifier.
                {
                    let mut gov = state.adaptive_governor.lock_recover();
                    gov.update_learned_policy(&policy);
                }
                DaemonResponse::Ok
            }
        }
        DaemonRequest::Feedback { rating, note } => {
            if rating.len() > 256 {
                return DaemonResponse::Error {
                    message: "rating too long (max 256)".to_string(),
                };
            }
            if let Some(ref n) = note {
                if n.len() > 2048 {
                    return DaemonResponse::Error {
                        message: "note too long (max 2048)".to_string(),
                    };
                }
            }
            let entry = FeedbackEntry {
                at: Utc::now(),
                rating,
                note,
            };
            append_jsonl(&state.feedback_path, &entry);
            DaemonResponse::Ok
        }
        DaemonRequest::GetSysctlGovernor => {
            let status = state.sysctl_governor_status.lock_recover().clone();
            DaemonResponse::SysctlGovernor(status)
        }
        // Subscribe es manejado antes de llegar aqui (en handle_client)
        DaemonRequest::Subscribe => DaemonResponse::Ok,
    }
}

fn build_llm_status(state: &SharedState) -> LlmStatus {
    let llm_cfg = load_repo_config(&state.config_path)
        .llm
        .unwrap_or_else(|| state.llm_cfg.as_ref().clone());
    let enabled_from_disk = llm_cfg.enabled();
    let llm_state = state.llm_state.lock_recover().clone();
    let policy = state.learned_policy.lock_recover().clone();

    let has_key = state.llm_key_path.exists();
    let enabled = enabled_from_disk && llm_state.enabled;
    let training_active = enabled && llm_state.training_active() && has_key;

    let now_local = Local::now();
    let today = now_local.date_naive().to_string();

    // Backward compatible: older persisted state may not have `training_started_at`.
    // Use the first observed call/attempt as a proxy.
    let training_started = llm_state
        .training_started_at
        .or(llm_state.last_call_at)
        .or(llm_state.last_attempt_at);
    let bootcamp = training_started
        .map(|t| Utc::now() - t < ChronoDuration::days(5))
        .unwrap_or(false);
    let daily_budget: u32 = if bootcamp { 24 } else { 8 };
    let calls_today = if llm_state.calls_today_day.as_deref() == Some(&today) {
        llm_state.calls_today
    } else {
        0
    };
    let daily_budget_remaining = daily_budget.saturating_sub(calls_today);

    LlmStatus {
        enabled,
        training_active,
        training_expires_at: llm_state.training_expires_at,
        has_api_key: has_key,
        mode: llm_state.mode,
        last_call_at: llm_state.last_call_at,
        last_attempt_at: llm_state.last_attempt_at,
        last_http_status: llm_state.last_http_status,
        last_error: llm_state.last_error.clone(),
        last_trigger_reason: llm_state.last_trigger_reason.clone(),
        calls_in_current_window: llm_state.calls_in_window,
        min_confidence: llm_cfg.min_confidence(),
        calls_today,
        daily_budget,
        daily_budget_remaining,
        last_suggestion_confidence: llm_state.last_suggestion.as_ref().map(|s| s.confidence),
        last_suggestion_rationale: llm_state
            .last_suggestion
            .as_ref()
            .map(|s| s.rationale.clone()),
        learned_policy: LearnedPolicyStatus {
            interactive_patterns: policy.interactive_patterns.len(),
            noise_patterns: policy.noise_patterns.len(),
            protected_patterns: policy.protected_patterns.len(),
            learned_at: policy.learned_at,
        },
    }
}

fn llm_reactive_tick(
    state: &SharedState,
    advisor: &mut LlmAdvisor,
    snapshot: &apollo_optimizer::collector::SystemSnapshot,
    counters: &mut LlmReactiveCounters,
    heuristic_struggling: bool,
) {
    let now = Utc::now();
    let has_key = state.llm_key_path.exists();

    // TTL housekeeping: if training expired, disable and delete key.
    {
        let mut llm_state = state.llm_state.lock_recover();
        if llm_state.enabled
            && llm_state
                .training_expires_at
                .map(|t| t <= now)
                .unwrap_or(true)
        {
            llm_state.enabled = false;
            llm_state.training_expires_at = None;
            llm_state.last_suggestion = None;
            llm_state.mode = LlmRunMode::Off;
            llm_state.last_error = Some("training-expired".to_string());
            write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            delete_file_best_effort(&state.llm_key_path);
            return;
        }
    }

    let llm_cfg = load_repo_config(&state.config_path)
        .llm
        .unwrap_or_else(|| state.llm_cfg.as_ref().clone());
    if !llm_cfg.enabled() {
        return;
    }

    // Keep advisor in sync with config edits.
    advisor.update_cfg(llm_cfg.clone());
    if !has_key {
        return;
    }

    let api_key = match HardPath::read_to_string_limited(&state.llm_key_path, 4096) {
        Ok(v) => v,
        Err(_) => return,
    };

    // Determine reactive trigger.
    let ws_cpu = windowserver_cpu(snapshot);
    let mem_pressure = snapshot.pressure.memory_pressure;
    let swap_delta_bps = snapshot.pressure.swap_delta_bytes_per_sec;
    let thermal = snapshot.pressure.thermal_level.as_str();

    // Decide desired mode (sensitive vs strict) using cost governor.
    let now_local = Local::now();
    let today = now_local.date_naive().to_string();
    let quiet_hours = {
        let h = now_local.hour();
        (1..8).contains(&h)
    };

    let (mode, daily_budget, min_interval_secs, max_calls_per_hour, pattern_budget_per_day) = {
        let mut llm_state = state.llm_state.lock_recover();
        if !llm_state.training_active() {
            write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            return;
        }

        // Reset daily budget window.
        if llm_state.calls_today_day.as_deref() != Some(&today) {
            llm_state.calls_today_day = Some(today.clone());
            llm_state.calls_today = 0;
        }

        // Keep trigger events only for a short horizon.
        llm_state
            .trigger_events
            .retain(|t| now - *t < ChronoDuration::minutes(30));
        let trigger_len = llm_state.trigger_events.len();
        if trigger_len > 100 {
            llm_state.trigger_events.drain(..trigger_len - 100);
        }
        let triggers_recent = llm_state.trigger_events.len() as u32;

        let bootcamp = llm_state
            .training_started_at
            .map(|t| now - t < ChronoDuration::days(5))
            .unwrap_or(false);
        let daily_budget = if bootcamp { 24 } else { 8 };

        // If we've been stable for a while, bias to strict.
        let stable_for = llm_state
            .no_trigger_since
            .map(|t| now - t)
            .unwrap_or_else(|| ChronoDuration::seconds(0));
        let stable_long = stable_for > ChronoDuration::hours(3);

        let consumed = llm_state.calls_today;
        let consumed_ratio = if daily_budget == 0 {
            1.0
        } else {
            (consumed as f64) / (daily_budget as f64)
        };

        let mut mode = llm_state.mode;
        if quiet_hours {
            mode = LlmRunMode::Strict;
        } else if consumed >= daily_budget {
            mode = LlmRunMode::Off;
        } else if triggers_recent >= 2 {
            mode = LlmRunMode::Sensitive;
        } else if consumed_ratio >= 0.60 {
            // Once we've consumed most of the daily budget, tighten up.
            mode = LlmRunMode::Strict;
        } else if stable_long && !bootcamp {
            // During bootcamp we keep mode sensitive for faster learning.
            mode = LlmRunMode::Strict;
        } else if mode == LlmRunMode::Off {
            // Recover from off when the budget permits.
            mode = LlmRunMode::Strict;
        }
        llm_state.mode = mode;

        let (base_min_interval, base_max_calls, pattern_budget) = match mode {
            LlmRunMode::Sensitive => (600_u64, 4_u32, if bootcamp { 5_u32 } else { 3_u32 }),
            LlmRunMode::Strict => (1800_u64, 2_u32, 2_u32),
            LlmRunMode::Off => (u64::MAX, 0_u32, 0_u32),
        };

        // Respect config as a hard limiter for cadence.
        let effective_min_interval = base_min_interval.max(llm_cfg.min_interval_secs());
        let effective_max_calls = base_max_calls.min(llm_cfg.max_calls_per_hour().max(1));

        write_json(&state.llm_state_path, &*llm_state, Some(0o600));
        (
            mode,
            daily_budget,
            effective_min_interval,
            effective_max_calls,
            pattern_budget,
        )
    };

    if mode == LlmRunMode::Off {
        return;
    }

    // Thresholds by mode.
    // WindowServer >35% es normal durante uso intensivo de UI (especialmente con TDA).
    // Subimos el umbral para no desperdiciar budget del LLM en síntomas, no causas.
    let (ws_thresh, mem_thresh, swap_thresh_bps, cycles_needed) = match mode {
        LlmRunMode::Sensitive => (65.0_f32, 0.78_f64, 20.0 * 1024.0 * 1024.0, 3_u32),
        LlmRunMode::Strict => (75.0_f32, 0.88_f64, 50.0 * 1024.0 * 1024.0, 5_u32),
        LlmRunMode::Off => (f32::MAX, 1.0, f64::MAX, u32::MAX),
    };

    if ws_cpu >= ws_thresh {
        counters.ws_high += 1;
    } else {
        counters.ws_high = 0;
    }
    if mem_pressure >= mem_thresh {
        counters.mem_high += 1;
    } else {
        counters.mem_high = 0;
    }
    if swap_delta_bps >= swap_thresh_bps {
        counters.swap_high += 1;
    } else {
        counters.swap_high = 0;
    }

    let thermal_critical = matches!(thermal, "serious" | "critical");
    let mut trigger_active = thermal_critical
        || counters.ws_high >= cycles_needed
        || counters.mem_high >= cycles_needed
        || counters.swap_high >= cycles_needed;
    let mut rising_edge = trigger_active && !counters.prev_trigger_active;
    counters.prev_trigger_active = trigger_active;

    // One-time baseline call after enabling training so it doesn't look "stuck"
    // when the system is stable and no triggers fire.
    let baseline_call = {
        let llm_state = state.llm_state.lock_recover();
        llm_state.last_attempt_at.is_none()
            && llm_state
                .training_started_at
                .map(|t| now - t > ChronoDuration::minutes(2))
                .unwrap_or(false)
    };
    if baseline_call {
        trigger_active = true;
        rising_edge = true;
    }

    // Heurístico fallando: el outcome tracker detectó que throttlear ciertos procesos
    // no baja la presión de memoria. El LLM puede sugerir qué patrones proteger/ruido.
    if heuristic_struggling && !trigger_active {
        trigger_active = true;
        rising_edge = !counters.prev_trigger_active;
    }

    if !trigger_active {
        // Bootcamp sampling: even when the system is "fine", take an occasional sample call
        // so the teacher can learn normal workload patterns.
        let sampling_due = {
            let llm_state = state.llm_state.lock_recover();
            let since_last = llm_state
                .last_attempt_at
                .map(|t| now - t)
                .unwrap_or_else(|| ChronoDuration::hours(24));
            let user_active_proxy = ws_cpu >= 10.0 || snapshot.cpu.global_usage >= 15.0;
            mode == LlmRunMode::Sensitive
                && llm_state
                    .training_started_at
                    .map(|t| now - t < ChronoDuration::days(5))
                    .unwrap_or(false)
                && user_active_proxy
                && since_last > ChronoDuration::minutes(45)
        };

        let mut llm_state = state.llm_state.lock_recover();
        if llm_state.no_trigger_since.is_none() {
            llm_state.no_trigger_since = Some(now);
        }

        if sampling_due {
            llm_state.last_trigger_at = Some(now);
            llm_state.last_trigger_reason = Some("sampling".to_string());
            llm_state.trigger_events.push(now);
            llm_state.no_trigger_since = None;
            write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            drop(llm_state);
            // Turn sampling into a synthetic rising-edge trigger.
            rising_edge = true;
        } else {
            write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            return;
        }
    }

    // Set/refresh trigger state.
    let trigger_reason = if baseline_call {
        "baseline".to_string()
    } else if thermal_critical {
        format!("thermal:{}", thermal)
    } else if counters.ws_high >= cycles_needed {
        format!("ui-lag windowserver cpu {:.1}%", ws_cpu)
    } else if counters.swap_high >= cycles_needed {
        format!("swap-thrash delta {:.0} B/s", swap_delta_bps)
    } else {
        format!("memory-pressure {:.2}", mem_pressure)
    };

    if rising_edge {
        let mut llm_state = state.llm_state.lock_recover();
        llm_state.last_trigger_at = Some(now);
        llm_state.last_trigger_reason = Some(trigger_reason.clone());
        llm_state.trigger_events.push(now);
        llm_state.no_trigger_since = None;
        write_json(&state.llm_state_path, &*llm_state, Some(0o600));
    }

    // Call gating: only call on rising edge.
    if !rising_edge {
        return;
    }

    // Budget + cadence.
    {
        let mut llm_state = state.llm_state.lock_recover();

        if llm_state.calls_today >= daily_budget {
            llm_state.mode = LlmRunMode::Off;
            llm_state.last_error = Some("daily-budget-exhausted".to_string());
            write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            return;
        }

        if let Some(last) = llm_state.last_attempt_at {
            if now - last < ChronoDuration::seconds(min_interval_secs as i64) {
                return;
            }
        }

        // Per-hour window.
        if llm_state
            .hour_window_started_at
            .map(|t| now - t > ChronoDuration::hours(1))
            .unwrap_or(true)
        {
            llm_state.hour_window_started_at = Some(now);
            llm_state.calls_in_window = 0;
        }
        if llm_state.calls_in_window >= max_calls_per_hour {
            return;
        }

        // Record attempt before the network call so status updates immediately.
        llm_state.last_attempt_at = Some(now);
        llm_state.last_http_status = None;
        llm_state.last_error = None;
        llm_state.calls_in_window += 1;
        llm_state.calls_today += 1;
        write_json(&state.llm_state_path, &*llm_state, Some(0o600));
    }

    // Network call (no locks held).
    let current_policy = state.learned_policy.lock_recover().clone();
    let suggestion_res = advisor.call_raw(snapshot, &api_key, Some(&current_policy));

    // Apply suggestion and persist state.
    match suggestion_res {
        Ok(suggestion) => {
            let accepted = suggestion.confidence >= llm_cfg.min_confidence();
            {
                let mut llm_state = state.llm_state.lock_recover();
                llm_state.last_http_status = Some(200);
                llm_state.last_call_at = Some(now);
                llm_state.last_suggestion = Some(suggestion.clone());
                llm_state.consecutive_failures = 0;
                if !accepted {
                    llm_state.last_error = Some("below-min-confidence".to_string());
                }
                write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            }

            append_jsonl(
                &state.suggestions_path,
                &serde_json::json!({
                    "at": now,
                    "trigger": trigger_reason,
                    "mode": format!("{:?}", mode),
                    "accepted": accepted,
                    "suggestion": suggestion,
                }),
            );

            if !accepted {
                return;
            }

            // 1) Profile: apply as a short-lived override.
            if let Some(p) = suggestion.suggested_profile {
                let mut gov = state.governor.lock_recover();
                if gov.manual_override.is_none() {
                    gov.set_manual_override(p, 20, "llm-reactive".to_string());
                }
            }
            // 2) Latency target.
            if let Some(t) = suggestion.suggested_latency_target {
                *state.latency_target.lock_recover() = t;
            }

            // 3) Learned patterns: merge with daily cap.
            {
                let mut llm_state = state.llm_state.lock_recover();
                let day = now.date_naive();
                let reset_day = llm_state
                    .policy_updates_day
                    .map(|d| d.date_naive() != day)
                    .unwrap_or(true);
                if reset_day {
                    llm_state.policy_updates_day = Some(now);
                    llm_state.policy_updates_today = 0;
                }
                let remaining =
                    pattern_budget_per_day.saturating_sub(llm_state.policy_updates_today);
                if remaining == 0 {
                    write_json(&state.llm_state_path, &*llm_state, Some(0o600));
                    return;
                }

                let mut policy = state.learned_policy.lock_recover();

                let mut added = 0u32;
                for p in suggestion
                    .add_interactive_patterns
                    .iter()
                    .take(remaining as usize)
                {
                    if !policy.interactive_patterns.contains(p)
                        && !pattern_conflicts_with_protected(p)
                    {
                        // Remove from noise if promoted to interactive.
                        policy.noise_patterns.retain(|n| n != p);
                        policy.interactive_patterns.push(p.clone());
                        added += 1;
                    }
                }
                for p in suggestion
                    .add_noise_patterns
                    .iter()
                    .take(remaining.saturating_sub(added) as usize)
                {
                    // Skip if already protected or interactive — cannot downgrade.
                    if !policy.noise_patterns.contains(p)
                        && !pattern_conflicts_with_protected(p)
                        && !policy.protected_patterns.contains(p)
                        && !policy.interactive_patterns.contains(p)
                    {
                        policy.noise_patterns.push(p.clone());
                        added += 1;
                    }
                }
                for p in suggestion
                    .add_protected_patterns
                    .iter()
                    .take(remaining.saturating_sub(added) as usize)
                {
                    if !policy.protected_patterns.contains(p)
                        && !pattern_conflicts_with_protected(p)
                    {
                        // Remove from noise when promoted to protected.
                        policy.noise_patterns.retain(|n| n != p);
                        policy.protected_patterns.push(p.clone());
                        added += 1;
                    }
                }

                if added > 0 {
                    policy.interactive_patterns.sort();
                    policy.noise_patterns.sort();
                    policy.protected_patterns.sort();
                    policy.learned_at = Some(now);
                    write_json(&state.learned_policy_path, &*policy, Some(0o600));
                    llm_state.policy_updates_today += added;
                    // Propagate updated patterns to the ML Ligero classifier.
                    {
                        let mut gov = state.adaptive_governor.lock_recover();
                        gov.update_learned_policy(&policy);
                    }
                }
                write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            }
        }
        Err(err) => {
            let mut llm_state = state.llm_state.lock_recover();
            llm_state.consecutive_failures += 1;
            match err {
                apollo_optimizer::engine::llm::LlmCallError::Cooldown => {
                    llm_state.last_error = Some("cooldown".to_string());
                }
                apollo_optimizer::engine::llm::LlmCallError::HttpStatus { code, body_excerpt } => {
                    llm_state.last_http_status = Some(code);
                    llm_state.last_error = Some(format!(
                        "http-status {} {}",
                        code,
                        body_excerpt.unwrap_or_default()
                    ));
                }
                apollo_optimizer::engine::llm::LlmCallError::Transport(e) => {
                    llm_state.last_error = Some(format!("transport {}", e));
                }
                apollo_optimizer::engine::llm::LlmCallError::Parse(e) => {
                    llm_state.last_error = Some(format!("parse {}", e));
                }
                apollo_optimizer::engine::llm::LlmCallError::Rejected(e) => {
                    llm_state.last_error = Some(format!("rejected {}", e));
                }
            }

            // Fail-safe: if it's repeatedly failing, go strict to save cost.
            if llm_state.consecutive_failures >= 3 {
                llm_state.mode = LlmRunMode::Strict;
            }
            write_json(&state.llm_state_path, &*llm_state, Some(0o600));
        }
    }
}

fn windowserver_cpu(snapshot: &apollo_optimizer::collector::SystemSnapshot) -> f32 {
    snapshot
        .top_processes
        .iter()
        .find(|p| p.name.contains("WindowServer"))
        .map(|p| p.cpu_usage)
        .unwrap_or(0.0)
}

fn usage_learning_tick(
    state: &SharedState,
    snapshot: &apollo_optimizer::collector::SystemSnapshot,
    has_foreground: bool,
    cpu_wall_ratios: &HashMap<String, f32>,
) {
    let now = Utc::now();
    let ws_cpu = windowserver_cpu(snapshot);
    // Refine interactive_proxy: require both CPU activity signals AND an actual
    // foreground app (not idle/screensaver). This prevents background CPU spikes
    // from triggering interactive mode when the user isn't at the keyboard.
    let cpu_proxy = ws_cpu >= 10.0 || snapshot.cpu.global_usage >= 15.0;
    let interactive_proxy = cpu_proxy && has_foreground;
    let mem_pressure = snapshot.pressure.memory_pressure;
    let swap_delta = snapshot.pressure.swap_delta_bytes_per_sec;

    let jank_proxy = ws_cpu >= 35.0
        && (mem_pressure >= 0.75 || swap_delta >= 20.0 * 1024.0 * 1024.0)
        || matches!(
            snapshot.pressure.thermal_level.as_str(),
            "serious" | "critical"
        );

    {
        let mut model = state.usage_model.lock_recover();
        model.update_from_snapshot(
            snapshot,
            now,
            interactive_proxy,
            jank_proxy,
            10,
            cpu_wall_ratios,
        );
    }

    // Persist usage model periodically (every ~2 minutes).
    {
        let tracker = state.usage_tracker.lock_recover();
        let due = tracker
            .last_persist_at
            .map(|t| now - t > ChronoDuration::minutes(2))
            .unwrap_or(true);
        if due {
            drop(tracker); // release before locking usage_model
            {
                let model = state.usage_model.lock_recover();
                model.persist(&state.usage_model_path);
            }
            state.usage_tracker.lock_recover().last_persist_at = Some(now);
        }
    }

    // Daily promotion counters (conservative).
    let today = Local::now().date_naive().to_string();
    let promotions_used = {
        let mut tracker = state.usage_tracker.lock_recover();
        if tracker.promotions_day.as_deref() != Some(&today) {
            tracker.promotions_day = Some(today.clone());
            tracker.promotions_today = 0;
        }
        tracker.promotions_today
    };
    // Propose promotions without holding locks across scoring.
    let (started_at, existing_interactive, existing_noise, existing_protected) = {
        let model = state.usage_model.lock_recover();
        let started_at = model.top_report(1).model_started_at;
        drop(model);
        let policy = state.learned_policy.lock_recover().clone();
        (
            started_at,
            policy.interactive_patterns,
            policy.noise_patterns,
            policy.protected_patterns,
        )
    };
    let promotions = {
        let model = state.usage_model.lock_recover();
        model.maybe_promote_patterns(
            now,
            &existing_interactive,
            &existing_noise,
            &existing_protected,
            promotions_used,
            started_at,
        )
    };

    if promotions.is_empty() {
        return;
    }

    // Apply promotions to learned policy.
    let mut applied = 0u32;
    {
        let mut policy = state.learned_policy.lock_recover();
        for (kind, pattern) in &promotions {
            match kind.as_str() {
                "interactive" => {
                    if !policy.interactive_patterns.contains(pattern)
                        && !pattern_conflicts_with_protected(pattern)
                    {
                        policy.interactive_patterns.push(pattern.clone());
                        applied += 1;
                    }
                }
                "noise" => {
                    if !policy.noise_patterns.contains(pattern)
                        && !pattern_conflicts_with_protected(pattern)
                    {
                        policy.noise_patterns.push(pattern.clone());
                        applied += 1;
                    }
                }
                "protected" => {
                    // Protected patterns are safety labels — they bypass the daily
                    // cap and only require that the pattern isn't already present.
                    if !policy.protected_patterns.contains(pattern)
                        && !pattern_conflicts_with_protected(pattern)
                    {
                        policy.protected_patterns.push(pattern.clone());
                        applied += 1;
                    }
                }
                _ => {}
            }
        }
        if applied > 0 {
            policy.interactive_patterns.sort();
            policy.noise_patterns.sort();
            policy.protected_patterns.sort();
            policy.learned_at = Some(now);
            write_json(&state.learned_policy_path, &*policy, Some(0o600));
            // Propagate updated patterns to the ML Ligero classifier.
            {
                let mut gov = state.adaptive_governor.lock_recover();
                gov.update_learned_policy(&policy);
            }
        }
    }

    if applied > 0 {
        state.usage_tracker.lock_recover().promotions_today += applied;
        append_jsonl(
            &state.usage_events_path,
            &serde_json::json!({"at": now, "promotions": promotions}),
        );
    }
}

fn apply_learned_policy_actions(
    snapshot: &apollo_optimizer::collector::SystemSnapshot,
    policy: &LearnedPolicy,
    mut actions: Vec<RootAction>,
) -> Vec<RootAction> {
    // Filter: never act on protected patterns (case-insensitive).
    if !policy.protected_patterns.is_empty() {
        actions.retain(|a| {
            let name = match a {
                RootAction::BoostProcess { name, .. }
                | RootAction::ThrottleProcess { name, .. }
                | RootAction::FreezeProcess { name, .. }
                | RootAction::UnfreezeProcess { name, .. } => name,
                _ => return true,
            };
            let name_lc = name.to_lowercase();
            !policy
                .protected_patterns
                .iter()
                .any(|p| name_lc.contains(&p.to_lowercase()))
        });
    }

    // Add targeted boost/throttle for top processes if policy matches.
    if policy.interactive_patterns.is_empty() && policy.noise_patterns.is_empty() {
        return actions;
    }
    let mut seen: HashSet<(u32, &'static str)> = HashSet::new();
    for a in &actions {
        match a {
            RootAction::BoostProcess { pid, .. } => {
                seen.insert((*pid, "boost"));
            }
            RootAction::ThrottleProcess { pid, .. } => {
                seen.insert((*pid, "throttle"));
            }
            _ => {}
        }
    }

    for p in &snapshot.top_processes {
        if policy
            .interactive_patterns
            .iter()
            .any(|pat| p.name.contains(pat))
            && !seen.contains(&(p.pid, "boost"))
        {
            actions.push(RootAction::BoostProcess {
                pid: p.pid,
                name: p.name.clone(),
                reason: "learned-policy interactive".to_string(),
            });
            seen.insert((p.pid, "boost"));
        }
        if policy.noise_patterns.iter().any(|pat| p.name.contains(pat))
            && !seen.contains(&(p.pid, "throttle"))
        {
            let (ss, su) = pid_start_time(p.pid);
            actions.push(RootAction::ThrottleProcess {
                pid: p.pid,
                name: p.name.clone(),
                aggressive: false,
                reason: "learned-policy noise".to_string(),
                start_sec: ss,
                start_usec: su,
            });
            seen.insert((p.pid, "throttle"));
        }
    }

    actions
}

fn run_socket_server(state: SharedState) -> anyhow::Result<()> {
    let socket_path = Path::new(socket_path());
    println!("Socket server starting for path: {:?}", socket_path);
    if let Some(parent) = socket_path.parent() {
        HardPath::secure_create_dir_all(parent)?;
    }
    HardPath::verify_no_symlink(socket_path)?;
    if socket_path.exists() {
        println!("Stale socket found, removing: {:?}", socket_path);
        fs::remove_file(socket_path)?;
    }

    let listener = UnixListener::bind(socket_path).context("bind socket")?;
    println!("Socket server listening on: {:?}", socket_path);
    // Socket permissions: 0o660 root:staff — all human users (staff group, GID 20)
    // can connect for read-only queries (status, metrics, subscribe).
    // Mutating commands (SetProfile, SetLearnedPolicy, etc.) require root via getpeereid.
    if unsafe { libc::getuid() } == 0 {
        let _ = fs::set_permissions(socket_path, fs::Permissions::from_mode(0o660));
        if let Ok(c_path) = CString::new(socket_path.as_os_str().as_encoded_bytes()) {
            unsafe {
                const STAFF_GID: libc::gid_t = 20;
                libc::chown(c_path.as_ptr(), 0, STAFF_GID); // root:staff
            }
        }
    } else {
        // Non-root: restrict to owner only.
        let _ = fs::set_permissions(socket_path, fs::Permissions::from_mode(0o600));
    }

    // BUG 6 fix: spawn a thread per client so one slow/malicious client doesn't
    // block all others. The old synchronous loop also blocked indefinitely on
    // accept(), preventing clean shutdown when stop=true was set.
    let active_clients = Arc::new(std::sync::atomic::AtomicU32::new(0));
    const MAX_CONCURRENT_CLIENTS: u32 = 32;

    for conn in listener.incoming() {
        if state.stop.load(Ordering::Acquire) || STOP_REQUESTED.load(Ordering::Acquire) {
            break;
        }
        if let Ok(stream) = conn {
            let clients = active_clients.clone();
            // Atomically increment first, then check — prevents race where
            // multiple threads pass the limit check simultaneously.
            let prev = clients.fetch_add(1, Ordering::AcqRel);
            if prev >= MAX_CONCURRENT_CLIENTS {
                clients.fetch_sub(1, Ordering::Relaxed);
                drop(stream);
                continue;
            }
            let state_clone = state.clone();
            thread::spawn(move || {
                handle_client(stream, &state_clone);
                clients.fetch_sub(1, Ordering::Release);
            });
        }
    }

    Ok(())
}

fn context_to_thermal(context: InteractiveContext) -> String {
    match context {
        InteractiveContext::ThermalConstrained => "constrained".to_string(),
        InteractiveContext::BackgroundPressure => "elevated".to_string(),
        InteractiveContext::InteractiveFocus => "nominal".to_string(),
    }
}

// Foreground detection is now handled by `ForegroundDetector` from
// `engine::foreground`. The old `get_foreground_app()` /
// `get_foreground_app_inner()` functions have been removed.

fn append_discrepancy_log(
    path: &std::path::Path,
    protected_app: &str,
    actions_removed: usize,
    workload: &str,
    confidence: f32,
    reason: &str,
) {
    let entry = serde_json::json!({
        "at": chrono::Utc::now().to_rfc3339(),
        "event": "safety_precedence_override",
        "protected_app": protected_app,
        "actions_removed": actions_removed,
        "ml_workload": workload,
        "ml_confidence": confidence,
        "reason": reason,
    });
    append_jsonl(path, &entry);
    rotate_timeline(path);
}

/// Tree-aware enriched process data builder.
///
/// Uses the foreground PID and process tree to determine foreground status for
/// Build the set of PIDs belonging to the foreground app group (parent + children).
fn build_foreground_family(foreground_pid: Option<u32>, tree: &ProcessTree) -> HashSet<u32> {
    foreground_pid
        .map(|pid| tree.cascade_pids(pid).into_iter().collect())
        .unwrap_or_default()
}

/// each process. A process is "foreground" if:
///   1. It IS the foreground PID, or
///   2. It belongs to the same process tree app group as the foreground PID
///      (i.e., it is a child/grandchild of the foreground app).
///
/// This gives accurate foreground detection for multi-process apps like Chrome,
/// Electron, VS Code, etc. where the heuristic classifier previously missed
/// helper/renderer processes because they have different names.
fn build_enriched_process_data_with_tree(
    sys: &sysinfo::System,
    foreground_pid: Option<u32>,
    tree: &ProcessTree,
) -> (Vec<ProcessSnapshot>, Vec<HuntSnapshot>) {
    // Pre-compute the set of PIDs in the foreground family for O(1) lookups.
    let fg_family: HashSet<u32> = build_foreground_family(foreground_pid, tree);

    // Bulk-read idle_wakeups + Mach messages via proc_taskinfo (~1.3ms for ~400 pids).
    // This replaces the hardcoded wakeups_per_sec: 0.0 with REAL kernel data.
    // pid → (idle_wakeups, mach_msgs, faults, pageins)
    let mut rusage_map: HashMap<u32, (u64, u32, u32, u32)> = HashMap::new();
    for &pid in &fg_family {
        // Only enrich non-foreground in the loop below
        let _ = pid;
    }
    // Build rusage map for all PIDs — O(n) syscalls, ~3µs each
    for (pid, _process) in sys.processes() {
        let pid_u32 = pid.as_u32();
        if let Some(ri) = proc_taskinfo::get_rusage_info(pid_u32) {
            let idle_wk = ri.idle_wakeups;
            if let Some(ti) = proc_taskinfo::get_task_info(pid_u32) {
                rusage_map.insert(
                    pid_u32,
                    (
                        idle_wk,
                        ti.messages_sent + ti.messages_received,
                        ti.faults,
                        ti.pageins,
                    ),
                );
            } else {
                rusage_map.insert(pid_u32, (idle_wk, 0, 0, 0));
            }
        }
    }

    let mut proc_snaps = Vec::new();
    let mut hunt_snaps = Vec::new();

    let now_unix_secs: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    for (pid, process) in sys.processes() {
        let pid_u32 = pid.as_u32();
        let name = process.name().to_string();
        let is_foreground = fg_family.contains(&pid_u32);
        let ppid = process.parent().map(|p| p.as_u32()).unwrap_or(0);
        let parent_alive = ppid > 0;
        let is_zombie = process.status() == ProcessStatus::Zombie;
        let rss = process.memory();
        let cpu = process.cpu_usage();
        // process.start_time() → seconds since Unix epoch; 0 if unknown.
        let process_uptime_secs = {
            let start = process.start_time();
            if start > 0 {
                now_unix_secs.saturating_sub(start)
            } else {
                u64::MAX // unknown start → treat as long-lived
            }
        };

        // Real idle wakeups from proc_pid_rusage — the #1 signal for wasteful daemons.
        // Estimate wakeups/sec: idle_wakeups is cumulative, divide by uptime estimate.
        // Mach messages > 0 implies the process has active IPC (network, XPC, etc.)
        let (wakeups_per_sec, has_network_signal, faults_total, pageins_total) =
            match rusage_map.get(&pid_u32) {
                Some(&(idle_wk, mach_msgs, faults, pageins)) => {
                    // Rough estimate: if idle_wakeups > 1000, it's a chatty daemon
                    let wps = if idle_wk > 10_000 {
                        (idle_wk as f32 / 3600.0).min(100.0)
                    } else if idle_wk > 100 {
                        (idle_wk as f32 / 7200.0).min(50.0)
                    } else {
                        0.0
                    };
                    // Rate-based network detection: cumulative mach_msgs / uptime.
                    // Avoids false positives on long-lived daemons with high cumulative
                    // counts but near-zero actual IPC rate.
                    let msg_rate = if process_uptime_secs > 0 {
                        mach_msgs as f64 / process_uptime_secs as f64
                    } else {
                        0.0
                    };
                    let has_net = msg_rate > 0.1; // >0.1 msg/sec = active IPC
                    (wps, has_net, faults, pageins)
                }
                None => (0.0, false, 0, 0),
            };

        proc_snaps.push(ProcessSnapshot {
            pid: pid_u32,
            name: name.clone(),
            cpu_percent: cpu,
            rss_bytes: rss,
            is_zombie,
            secs_since_foreground: if is_foreground { 0 } else { 3600 },
            secs_since_user_interaction: if is_foreground { 0 } else { 3600 },
            has_network: has_network_signal,
            has_gui_window: is_foreground,
            wakeups_per_sec,
            parent_alive,
            process_uptime_secs,
            faults_total,
            pageins_total,
            is_translated: apollo_optimizer::engine::process_identity::is_translated(pid_u32),
            mach_port_count: 0, // populated lazily for hoarder candidates only
        });

        hunt_snaps.push(HuntSnapshot {
            pid: pid_u32,
            ppid,
            name,
            is_kernel_zombie: is_zombie,
            parent_alive,
            has_gui_window: is_foreground,
            rss_bytes: rss,
            cpu_percent: cpu,
            wakeups_per_sec,
            secs_since_user_interaction: if is_foreground { 0 } else { 3600 },
            host_app_pid: process.parent().map(|p| p.as_u32()),
            host_app_running: parent_alive,
            host_app_absent_secs: if parent_alive { 0 } else { 3600 },
        });
    }

    (proc_snaps, hunt_snaps)
}

struct HeuristicStats {
    decisions_total: u64,
    throttles: u64,
    freezes: u64,
    kills_downgraded: u64,
    zombies_detected: u64,
}

fn convert_and_merge_heuristic_decisions(
    decisions: &[ProcessDecision],
    existing_actions: &[RootAction],
    critical_pids: &HashSet<u32>,
) -> (Vec<RootAction>, HeuristicStats) {
    let mut stats = HeuristicStats {
        decisions_total: decisions.len() as u64,
        throttles: 0,
        freezes: 0,
        kills_downgraded: 0,
        zombies_detected: 0,
    };

    // Build set of PIDs already acted on by decide_actions + learned_policy
    let existing_pids: HashSet<u32> = existing_actions
        .iter()
        .filter_map(|a| match a {
            RootAction::BoostProcess { pid, .. }
            | RootAction::ThrottleProcess { pid, .. }
            | RootAction::FreezeProcess { pid, .. } => Some(*pid),
            _ => None,
        })
        .collect();

    let mut new_actions = Vec::new();

    for decision in decisions {
        // Count zombies
        if decision.tier == ProcessTier::ZombieOrphan {
            stats.zombies_detected += 1;
        }

        // Skip if Allow
        if decision.decision == GovernorDecision::Allow {
            continue;
        }

        // Skip if already has an action from decide_actions/learned_policy
        if existing_pids.contains(&decision.pid) {
            continue;
        }

        // Skip critical processes
        if critical_pids.contains(&decision.pid) {
            continue;
        }

        match decision.decision {
            GovernorDecision::Throttle => {
                let (ss, su) = pid_start_time(decision.pid);
                new_actions.push(RootAction::ThrottleProcess {
                    pid: decision.pid,
                    name: decision.name.clone(),
                    aggressive: false,
                    reason: format!("heuristic: {}", decision.reason),
                    start_sec: ss,
                    start_usec: su,
                });
                stats.throttles += 1;
            }
            GovernorDecision::Freeze => {
                let (ss, su) = pid_start_time(decision.pid);
                new_actions.push(RootAction::FreezeProcess {
                    pid: decision.pid,
                    name: decision.name.clone(),
                    reason: format!("heuristic: {}", decision.reason),
                    start_sec: ss,
                    start_usec: su,
                });
                stats.freezes += 1;
            }
            GovernorDecision::Kill => {
                let (ss, su) = pid_start_time(decision.pid);
                // Safety: downgrade Kill to Freeze — never auto-kill from heuristics
                new_actions.push(RootAction::FreezeProcess {
                    pid: decision.pid,
                    name: decision.name.clone(),
                    reason: format!("heuristic (kill→freeze): {}", decision.reason),
                    start_sec: ss,
                    start_usec: su,
                });
                stats.kills_downgraded += 1;
                stats.freezes += 1;
            }
            GovernorDecision::Allow => unreachable!(),
        }
    }

    (new_actions, stats)
}

/// Toggle Spotlight indexing via `mdutil -a -i on/off`.
///
/// mdutil communicates with the Spotlight server via XPC (com.apple.spotlightserver).
/// There is no public or private framework function equivalent — MDSetIndexingEnabled
/// does not exist in the dyld shared cache on Apple Silicon macOS 15.
fn spotlight_set_indexing(enabled: bool) {
    let flag = if enabled { "on" } else { "off" };
    let _ = std::process::Command::new("/usr/bin/mdutil")
        .args(["-a", "-i", flag])
        .status();
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Daemon { profile } => {
            let profile = parse_profile(&profile);
            let is_root = unsafe { libc::geteuid() } == 0;

            let config_path = PathBuf::from("/etc/apollo-optimizer/config.toml");
            let repo_cfg = load_repo_config(&config_path);
            let llm_cfg = repo_cfg.llm.unwrap_or(LlmConfig {
                enabled: Some(false),
                endpoint: None,
                model: None,
                min_confidence: None,
                max_calls_per_hour: None,
                min_interval_secs: None,
                timeout_ms: None,
                force_json: None,
            });

            let (llm_state_path, llm_key_path) = state_paths_root(is_root);
            let learned_policy_path = policy_path_root(is_root);
            let feedback_path = feedback_path_root(is_root);
            let suggestions_path = suggestions_path_root(is_root);

            let usage_model_path = usage_model_path_root(is_root);
            let usage_model = UsageModel::load(&usage_model_path);
            let usage_events_path = if is_root {
                PathBuf::from("/var/lib/apollo/learn/usage_events.jsonl")
            } else {
                PathBuf::from("/tmp/apollo-usage_events.jsonl")
            };

            let llm_state = read_json::<LlmState>(&llm_state_path).unwrap_or_default();

            let learned_policy = {
                let disk_policy = read_json::<LearnedPolicy>(&learned_policy_path);
                if disk_policy.is_none() && learned_policy_path.exists() {
                    eprintln!(
                        "WARNING: learned policy at '{}' is missing or corrupt — \
                         falling back to seed policy only",
                        learned_policy_path.display()
                    );
                }
                let mut p = disk_policy.unwrap_or_default();
                merge_seed_into(&mut p);
                p
            };

            let governor_state_path = PathBuf::from(governor_state_path());
            let timeline_path = PathBuf::from(timeline_path());
            let wake_state_path = PathBuf::from(wake_state_path());
            let frozen_state_path = PathBuf::from(frozen_state_path());
            let governor = load_governor_state(&governor_state_path, profile);
            let wake_state = load_wake_state(&wake_state_path);
            let frozen_since_boot = load_frozen_state(&frozen_state_path);
            let state = SharedState {
                profile: Arc::new(Mutex::new(profile)),
                latency_target: Arc::new(Mutex::new(LatencyTarget::Normal)),
                metrics: Arc::new(Mutex::new(RuntimeMetrics {
                    effective_profile: profile,
                    throttle_level: "balanced".to_string(),
                    thermal_state: "nominal".to_string(),
                    thermal_level: "unknown".to_string(),
                    current_workload: "idle".to_string(),
                    collector_pressure_alive: true,
                    collector_smc_alive: true,
                    ..RuntimeMetrics::default()
                })),
                frozen_state: Arc::new(Mutex::new(frozen_since_boot.clone())),
                last_blockers: Arc::new(Mutex::new(Vec::new())),
                thermal_state: Arc::new(Mutex::new("nominal".to_string())),
                throttle_level: Arc::new(Mutex::new("balanced".to_string())),
                reactor_event_weight: Arc::new(Mutex::new(0.0)),
                fast_tick_until: Arc::new(Mutex::new(None)),
                thermal_level_real: Arc::new(Mutex::new("unknown".to_string())),
                reactor_status: Arc::new(Mutex::new(ReactorStatus::default())),
                governor: Arc::new(Mutex::new(governor)),
                timeline: Arc::new(Mutex::new(VecDeque::new())),
                wake_state: Arc::new(Mutex::new(wake_state)),
                stop: Arc::new(AtomicBool::new(false)),

                llm_cfg: Arc::new(llm_cfg),
                llm_state: Arc::new(Mutex::new(llm_state)),
                learned_policy: Arc::new(Mutex::new(learned_policy)),
                llm_state_path,
                llm_key_path,
                learned_policy_path,
                feedback_path,
                suggestions_path,

                config_path,

                usage_model: Arc::new(Mutex::new(usage_model)),
                usage_model_path,
                usage_events_path,
                usage_tracker: Arc::new(Mutex::new(UsageTrackerState::default())),

                adaptive_governor: Arc::new(Mutex::new(AdaptiveGovernor::new())),
                mach_qos: Arc::new(Mutex::new(MachQoSManager::new())),
                last_hw_snapshot: Arc::new(Mutex::new(None)),

                discrepancy_log_path: if is_root {
                    PathBuf::from("/var/lib/apollo/discrepancy.jsonl")
                } else {
                    PathBuf::from("/tmp/apollo-discrepancy.jsonl")
                },
                user_profile_path: if is_root {
                    PathBuf::from("/var/lib/apollo/user_profile.json")
                } else {
                    PathBuf::from("/tmp/apollo-user_profile.json")
                },

                sysctl_governor_status: Arc::new(Mutex::new(SysctlGovernorStatus {
                    active: false,
                    current_values: HashMap::new(),
                    defaults: HashMap::new(),
                    total_writes: 0,
                    active_tunings: 0,
                    retransmission_rate: 0.0,
                    listen_drop_rate: 0.0,
                    last_tune_secs_ago: HashMap::new(),
                    tcp_consecutive_high: 0,
                    tcp_consecutive_low: 0,
                    ipc_consecutive_drops: 0,
                    ipc_consecutive_clean: 0,
                    vm_consecutive_high: 0,
                    vm_consecutive_low: 0,
                    fs_consecutive_high: 0,
                    fs_consecutive_low: 0,
                })),

                cycle_condvar: Arc::new((Mutex::new(false), Condvar::new())),
                resource_interrupt: Arc::new(ResourceInterruptState::new()),

                subscribers: Arc::new(Mutex::new(Vec::new())),
            };

            // Load persisted UserProfile (learning survives daemon restarts).
            if let Some(persisted) = read_json::<UserProfilePersisted>(&state.user_profile_path) {
                let mut gov = state.adaptive_governor.lock_recover();
                gov.user_profile = UserProfile::from_persisted(persisted);
            }

            // Scrub learned policy: remove patterns that should never be interactive.
            // This list is curated by LLM Teacher analysis of usage_model data.
            {
                let mut policy = state.learned_policy.lock_recover();
                let bad_interactive: Vec<&str> = vec![
                    // Self-reference
                    "apollo-optimizerd",
                    // Telemetry / analytics
                    "UsageTrackingAgent",
                    "amsengagementd",
                    "ecosystemanalyticsd",
                    "PerfPowerServices",
                    "triald",
                    // Background asset / sync
                    "assetsubscriptiond",
                    "mobileassetd",
                    "searchpartyd",
                    "cloudd",
                    "fileproviderd",
                    "photolibraryd",
                    "softwareupdated",
                    "accessoryupdaterd",
                    // Background ML
                    "photoanalysisd",
                    "mediaanalysisd",
                    "ModelCatalogAgent",
                    "duetexpertd",
                    // Spotlight / diagnostics
                    "corespotlightd",
                    "spotlightknowledged",
                    "spindump",
                    // System daemons
                    "dasd",
                    "deleted",
                    "ecosystemd",
                    "fseventsd",
                    "logd",
                    "runningboardd",
                    "airportd",
                    "corebrightnessd",
                    // Siri / assistant
                    "assistantd",
                    "contextstored",
                    "corespeechd",
                    "com.apple.siri.embeddedspeech",
                    "suggestd",
                    // Preferences / contacts
                    "cfprefsd",
                    "contactsd",
                    // Updaters
                    "logioptionsplus_updater",
                    // Security
                    "XprotectService",
                    // Decorative
                    "WallpaperAerialsExtension",
                    // Transient
                    "xpcproxy",
                    "iconservicesagent",
                    "linkd",
                    "siriactionsd",
                    "com.apple.Safari.SafeBrowsing.Service",
                ];
                let before = policy.interactive_patterns.len();
                policy
                    .interactive_patterns
                    .retain(|p| !bad_interactive.iter().any(|bad| p.contains(bad)));
                // Add noise patterns from LLM Teacher analysis.
                if !policy.noise_patterns.contains(&"apsd".to_string()) {
                    policy.noise_patterns.push("apsd".to_string());
                }
                let removed = before - policy.interactive_patterns.len();
                if removed > 0 || policy.noise_patterns.len() == 1 {
                    write_json(&state.learned_policy_path, &*policy, Some(0o600));
                }
            }

            // Initialize ML Ligero classifier with the already-loaded LearnedPolicy.
            {
                let policy = state.learned_policy.lock_recover().clone();
                let mut gov = state.adaptive_governor.lock_recover();
                gov.update_learned_policy(&policy);
            }

            let reactor_state = state.clone();
            thread::spawn(move || {
                let _ = run_reactor(reactor_state);
            });

            // Defensive: if a previous run froze processes and crashed/restarted, unfreeze them on startup.
            {
                let mut frozen_state = state.frozen_state.lock_recover();
                if !frozen_state.is_empty() {
                    let count = unfreeze_pids(frozen_state.keys().copied());
                    frozen_state.clear();
                    frozen_state.clear();
                    write_frozen_state(&frozen_state_path, &frozen_state);
                    {
                        let mut metrics = state.metrics.lock_recover();
                        metrics.post_wake_defensive_unfreezes += count;
                        metrics.unfreezes_applied += count;
                        metrics.throttle_reverted += count;
                    }
                }
            }

            let socket_state = state.clone();
            thread::spawn(move || {
                if let Err(e) = run_socket_server(socket_state) {
                    eprintln!("CRITICAL: Socket server failed: {:?}", e);
                }
            });

            let stop = state.stop.clone();
            ctrlc::set_handler(move || {
                STOP_REQUESTED.store(true, Ordering::Release);
                stop.store(true, Ordering::Release);
            })?;

            // Register SIGTERM handler so launchd graceful shutdown triggers cleanup.
            // SIGKILL cannot be caught — the defensive unfreeze at startup covers that case.
            unsafe {
                let mut sa: libc::sigaction = std::mem::zeroed();
                sa.sa_sigaction = handle_sigterm as *const () as usize;
                sa.sa_flags = libc::SA_RESTART;
                libc::sigemptyset(&mut sa.sa_mask);
                libc::sigaction(libc::SIGTERM, &sa, std::ptr::null_mut());
            }

            let mut collector = SystemCollector::new();
            let mut thrash = ThrashState::default();
            let mut llm_counters = LlmReactiveCounters::default();
            let journal_path = PathBuf::from(journal_path());
            let metrics_path = PathBuf::from(metrics_path());
            let mut critical_failure_timestamps: Vec<Instant> = Vec::new();
            let mut override_was_active = false;
            let daemon_start = Instant::now();
            let mut llm_advisor = LlmAdvisor::new(state.llm_cfg.as_ref().clone());

            // Secondary optimization modules — all run each cycle without locks.
            let mut analytics = AnalyticsEngine::new();
            let mut mem_analyzer = MemoryAnalyzer::new();
            let mut focus_markov = FocusMarkov::new(PathBuf::from(markov_path()));
            let hw_path = PathBuf::from(holt_winters_path());
            let mut holt_winters = HoltWinters::load(&hw_path).unwrap_or_default();
            let mut hw_last_hour: Option<u8> = None;
            let mut hw_pressure_accum: f64 = 0.0;
            let mut hw_pressure_count: u32 = 0;
            let mut power_mgr = PowerManager::new();
            let mut proc_recovery = ProcessRecoveryManager::new();
            let mut swap_predictor = SwapPredictor::new();
            let mut network_monitor = NetworkMonitor::new();
            let mut sysctl_governor = SysctlGovernor::new(is_root);
            let mut thermal_mgr = ThermalManager::new();
            let mut wake_storm = WakeStormDetector::new();
            // GPU thermal monitoring: integrates with thermal_manager for GPU-aware decisions.
            let gpu_mgr = GPUManager::new();
            // Darwin-Boltzmann Anomaly Detector: replaces disabled TransformerPredictor
            // with online Hopfield memory + evolving SAE population + free energy fusion.
            let mut darwin_anomaly = EvolvedAnomalyDetector::new();
            // Network profile optimizer: complements sysctl_governor with profile-driven tuning.
            let net_optimizer = NetworkOptimizer::new();
            // Foreground detection: replaces get_foreground_app() with cached, richer detection.
            // Wrapped in Arc so it can be shared with the resource sentinel thread.
            let fg_detector = Arc::new(ForegroundDetector::new());
            // Per-app energy estimation: accumulates energy attribution each cycle.
            let mut energy_tracker = EnergyTracker::new();
            let mut outcome_tracker = OutcomeTracker::new();
            outcome_tracker.load_hop_groups(std::path::Path::new(hop_groups_path()));

            // Habituation filter (Thompson & Spencer 1966, inspired by memoria-core).
            // Tracks per-process (cpu_bucket, rss_bucket, cycles_unchanged).
            // Processes unchanged for ≥5 cycles are skipped in decide_actions.
            const HABITUATION_THRESHOLD: u32 = 5;
            let mut habituation_map: HashMap<u32, (u8, u8, u32)> = HashMap::new();

            // Causal graph (Pearl 2009, adapted from memoria-core/causal_inference.rs).
            // Tracks action → outcome relationships with Bayesian confidence.
            // "throttle:X → pressure_drop" edges inform future prioritization.
            let mut causal_graph = CausalGraph::new();

            // Neuromodulator (memoria-core/neuromodulator.rs):
            // Bio-inspired parameter modulation — DA/NA/SE/ACh drive RL alpha,
            // Dyna-Q steps, router zones, and exploration rate.
            let mut neuromod = ApolloNeuromodulator::new();
            // Track cycle-to-cycle wall time for energy dt calculation.
            let mut last_cycle_instant = Instant::now();
            // Audit fix #5: Background powermetrics polling (replaces 5-cycle IOKit tick).
            let mut smc_reader = SmcReader::spawn(Duration::from_secs(3));
            // Background pressure collector: moves memory_pressure + sysctl out of main loop.
            let mut pressure_collector = PressureCollector::spawn(Duration::from_secs(3));
            // Resource sentinel: sub-100ms interrupt handler for thermal/memory/power emergencies.
            // Shares the fg_detector so the sentinel never freezes the active foreground app.
            spawn_resource_sentinel(
                smc_reader.cache_arc(),
                pressure_collector.cache_arc(),
                state.resource_interrupt.clone(),
                state.frozen_state.clone(),
                state.stop.clone(),
                SentinelConfig::default(),
                fg_detector.clone(),
                Some(state.mach_qos.clone()),
            );
            // Overflow guard: aprende de eventos OOM y ajusta thresholds adaptativamente.
            let mut overflow_guard = OverflowGuard::load_or_default(
                std::path::Path::new(overflow_history_path()),
                Some(std::path::Path::new(rl_threshold_path())),
            );
            // Predictive agent: LinUCB contextual bandit for proactive interventions.
            let mut predictive_agent =
                PredictiveAgent::load_or_default(std::path::Path::new(predictive_agent_path()));
            // Specialist accuracy tracker: EMA per-specialist confidence weights.
            // Starts at 0.70 (matching legacy hardcoded multipliers) and adapts.
            let mut specialist_accuracy = SpecialistAccuracyTracker::new();
            // Track previous cycle pressure to detect spikes (for accuracy feedback).
            let mut prev_pressure_smooth: f64 = 0.0;

            // ZeroTune: seed with hardware meta-features on cold start.
            // Reduces warmup from 200→50 cycles by injecting domain knowledge priors.
            if !predictive_agent.is_active() && predictive_agent.total_cycles() == 0 {
                let ram_gb = apollo_optimizer::engine::sysctl_direct::read_u64("hw.memsize")
                    .unwrap_or(8 * 1024 * 1024 * 1024) as f64
                    / (1024.0 * 1024.0 * 1024.0);
                let cores = apollo_optimizer::engine::sysctl_direct::read_u64("hw.ncpu")
                    .unwrap_or(4) as usize;
                predictive_agent.meta_seed(ram_gb, cores);
            }
            // Signal intelligence: Kalman + CUSUM + Entropy + Hazard + LV + MPC.
            // Restore persisted hazard model + MPC effects so the system doesn't cold-start
            // after a reboot (Cox hazard base_rate calibrates over days of observation).
            let mut signal_intel = SignalIntelligence::new();
            if let Some(si_persisted) =
                SignalIntelligence::load(std::path::Path::new(signal_intelligence_path()))
            {
                signal_intel.restore(si_persisted);
            }
            // Telemetry logger: ring-buffer data collector for Transformer training.
            // Telemetry logger + Transformer disabled: classical stack is sufficient.
            // Code preserved for future evaluation.
            // let telemetry_dir = if unsafe { libc::geteuid() } == 0 {
            //     std::path::PathBuf::from("/var/lib/apollo/telemetry")
            // } else {
            //     std::path::PathBuf::from("/tmp/apollo-telemetry")
            // };
            // let mut telemetry_logger =
            //     apollo_optimizer::engine::telemetry_logger::TelemetryLogger::new(
            //         telemetry_dir.clone(),
            //     );
            // File cache warmer: pre-read predicted app executables into buffer cache.
            // Cao et al. 1994 — app-controlled prefetch eliminates cold page faults.
            let mut cache_warmer = apollo_optimizer::engine::cache_warmer::CacheWarmer::new();
            // Transformer predictor: ONNX inference for anomaly detection.
            // Without `transformer` feature flag, this is a zero-cost no-op.
            // With the flag + trained model, it runs real inference each cycle.
            // let transformer_model_path = telemetry_dir
            //     .parent()
            //     .unwrap_or(std::path::Path::new("/var/lib/apollo"))
            //     .join("apollo_transformer.onnx");
            // let transformer_stats_path = telemetry_dir
            //     .parent()
            //     .unwrap_or(std::path::Path::new("/var/lib/apollo"))
            //     .join("feature_stats.json");
            // Transformer disabled: the classical stack (Kalman + CUSUM + Hazard +
            // Holt-Winters + Markov + Temporal) already covers anomaly detection and
            // prediction.  Code preserved for future evaluation.
            // let mut transformer_predictor =
            //     apollo_optimizer::engine::transformer_predictor::TransformerPredictor::new(
            //         &transformer_model_path,
            //         &transformer_stats_path,
            //     );
            // Display-Off Turbo: Android Doze-like mode for macOS.
            // Project Volta (Google 2014) — freeze non-essential when display off,
            // instant restore on wake. Saves 15-25% battery (Chuang et al. 2013).
            let mut display_turbo = apollo_optimizer::engine::display_turbo::DisplayTurbo::new();
            // Temporal app predictor: time-of-day aware app launch prediction.
            // Shin et al. 2012 — temporal patterns predict app launches with ~80% accuracy.
            // Combined with Markov chain for 85% top-3 accuracy (Baeza-Yates et al. 2015).
            let temporal_path = if unsafe { libc::geteuid() } == 0 {
                std::path::PathBuf::from("/var/lib/apollo/temporal_histograms.json")
            } else {
                std::path::PathBuf::from("/tmp/apollo-temporal_histograms.json")
            };
            let mut temporal_predictor =
                apollo_optimizer::engine::temporal_predictor::TemporalPredictor::new(temporal_path);
            // I/O Traffic Shaper: foreground-aware disk bandwidth allocation.
            // Iyer & Druschel 2001 — anticipatory scheduling reduces foreground I/O
            // latency by 50-70% under concurrent background load.
            let mut io_shaper = apollo_optimizer::engine::io_tiering::IoShaper::new();
            // Adaptive Page Reclaim: pressure-driven file cache purging.
            // Jiang & Zhang 2005 — proactive reclaim of low-IRR pages outperforms
            // reactive LRU eviction by 20-40% in cache hit ratio.
            let mut page_reclaim =
                apollo_optimizer::engine::page_reclaim::PageReclaim::new(is_root);
            // Audit fix #6: Multi-phase thermal bail-out with hysteresis.
            let mut thermal_bailout = ThermalBailout::new();

            // ── Coalition tracker (kernel-authoritative process families) ─────
            // Groups app + all XPC/GPU/framework helpers by kernel coalition ID.
            // Used to augment foreground family detection beyond heuristic name-matching.
            let coalition_tracker = CoalitionTracker::new();

            // ── IOReport reader (hardware telemetry without subprocess overhead) ─
            // Provides P/E cluster utilization, GPU%, ANE activity, per-component mW.
            // Samples the first baseline here; delta is computed each cycle.
            let mut ioreport = IOReportReader::new();
            if ioreport.available {
                #[cfg(target_os = "macos")]
                ioreport.begin_sample();
                println!("[ioreport] IOReport subscription active");
            } else {
                println!("[ioreport] IOReport unavailable, using SMC fallback");
            }
            // Last IOReport snapshot (updated each cycle).
            let mut last_ioreport: Option<apollo_optimizer::engine::ioreport::IOReportSnapshot> = None;
            // Throttle IOReport to every ~2 cycles (≥1s between samples).
            let mut last_ioreport_sample = Instant::now();

            // ── Warn-limit tracking (non-fatal targeted memory pressure) ──────
            // PIDs that have an active warn_limit set; cleared after 3 cycles.
            let mut warn_limit_pids: HashMap<u32, u8> = HashMap::new();

            // ── Feature 1: LLM Inference Mode ────────────────────────────────
            // Auto-detect ollama / llama.cpp / MLX / LM Studio inference.
            // When active: +20pp pressure boost, Spotlight off, App Nap non-essential.
            let mut llm_detector =
                apollo_optimizer::engine::llm_inference_mode::LlmInferenceDetector::new();
            let mut llm_spotlight_disabled = false;

            // ── Feature 3: RT Boost for Foreground ───────────────────────────
            // THREAD_TIME_CONSTRAINT_POLICY: guarantee 2ms/10ms to foreground UI thread.
            // Eliminates UI hitches during heavy CPU load (e.g., LLM inference + browser).
            let mut rt_boosted_pid: Option<u32> = None;

            // ── Feature 4: Post-Wake Suppression ─────────────────────────────
            // Detect sleep/wake by comparing elapsed time vs cycle interval.
            // 60s App-Nap window after wake suppresses background update storms.
            // (reuses last_cycle_instant declared above for energy dt tracking)
            let mut wake_suppression_until: Option<Instant> = None;

            // ── SMC Direct Read ──────────────────────────────────────────────
            // Sub-100µs power, lid, sleep/wake, battery telemetry (replaces powermetrics).
            let smc_direct = apollo_optimizer::engine::smc_direct::SmcDirectReader::new();
            let mut last_smc: Option<apollo_optimizer::engine::smc_direct::SmcSnapshot> = None;
            if smc_direct.available {
                println!("[smc-direct] SMC direct reader active");
            } else {
                println!("[smc-direct] SMC direct reader unavailable");
            }

            // ── KPC Hardware Performance Counters ────────────────────────────
            // Per-core IPC via libkpc.dylib (fixed counters: cycles + instructions).
            let mut kpc_reader = apollo_optimizer::engine::kpc_counters::KpcReader::new();
            if kpc_reader.available {
                println!("[kpc] Hardware performance counters active");
            } else {
                println!("[kpc] KPC counters unavailable (SIP or not root)");
            }

            // ── Rosetta AOT Monitor ─────────────────────────────────────────
            // Watches /var/db/oah/ for write events → suppress freezing oahd.
            let mut rosetta_monitor = apollo_optimizer::engine::rosetta_monitor::RosettaMonitor::new();
            if rosetta_monitor.available {
                println!("[rosetta] AOT compilation monitor active");
            } else {
                println!("[rosetta] Rosetta not installed or /var/db/oah inaccessible");
            }

            // ── Per-Process Energy Ranking (ri_billed_energy) ────────────────
            let mut energy_pid_tracker = apollo_optimizer::engine::energy_pid::EnergyPidTracker::new();

            // ── Daemon self-IPC monitoring (thread_selfcounts syscall 186) ───
            let mut cycle_ipc_tracker = apollo_optimizer::engine::thread_selfcounts::CycleIpcTracker::new();

            // Freeze confirmation cache: pid → consecutive cycles flagged.
            // Only freeze processes that have been candidates for 2+ cycles,
            // filtering out short-lived transients that die before execute_actions.
            let mut freeze_candidates: HashMap<u32, u8> = HashMap::new();
            let mut cycle_count: u64 = 0;
            // Minimum cycle floor: prevent CPU burn from rapid condvar wakeups.
            let mut last_cycle_end = Instant::now() - Duration::from_secs(1);
            // Gate network_monitor.tick() to every ~10s since netstat is blocking.
            let mut last_netstat_tick = Instant::now() - Duration::from_secs(10);
            // Context-switch burst detector (TDA-aware).
            let mut ctx_switch_times: VecDeque<Instant> = VecDeque::new();
            let mut last_fg_name: Option<String> = None;
            // Track last hw_pressure level to decide light vs full snapshot.
            let mut last_hw_pressure = HwPressure::Nominal;
            // EMA interactivity classifier: track per-PID rusage CPU deltas
            // to compute cpu_wall_ratio. Key = PID, value = (prev_user_ns,
            // prev_system_ns, proc_start_abstime) for delta computation.
            let mut rusage_cpu_prev: HashMap<u32, (u64, u64, u64)> = HashMap::new();
            let mut last_rusage_at = Instant::now();
            // Lock-free metrics for hot-path counters (no mutex overhead).
            let lf_metrics = std::sync::Arc::new(LockFreeMetrics::new());
            // vm_surgeon: pin the lock-free metrics buffer in physical RAM.
            // Guarantees zero page-fault latency on the hot path under memory pressure.
            {
                use apollo_optimizer::engine::vm_surgeon;
                let ptr = &*lf_metrics as *const LockFreeMetrics as *const u8;
                let len = std::mem::size_of::<LockFreeMetrics>();
                if let Err(e) = vm_surgeon::pin_memory(ptr, len) {
                    eprintln!(
                        "warn: mlock on LockFreeMetrics failed ({}), continuing unpinned",
                        e
                    );
                }
            }
            // kqueue reactor for frozen-PID death detection (push, not poll).
            // When a frozen process dies (OOM, jetsam), the kernel pushes
            // EVFILT_PROC/NOTE_EXIT instantly — no polling latency.
            let mut kq_frozen: Option<kqueue_pressure::KqueuePressure> =
                match kqueue_pressure::KqueuePressure::new() {
                    Ok(kq) => Some(kq),
                    Err(e) => {
                        eprintln!("warn: kqueue_pressure init failed ({}), frozen-death detection degraded", e);
                        None
                    }
                };

            loop {
                // Check both: Arc flag (set by ctrlc) and static flag (set by SIGTERM handler).
                if state.stop.load(Ordering::Acquire) || STOP_REQUESTED.load(Ordering::Acquire) {
                    state.stop.store(true, Ordering::Release);
                    println!("Daemon stopping: stop signal received");
                    break;
                }

                cycle_count += 1;
                lf_metrics.inc_cycles();
                println!(">>> Daemon cycle: {}", cycle_count);

                // ── Feature 4: Post-Wake Suppression ─────────────────────────
                // If more than 30s passed since the last cycle, the system was
                // sleeping. Apply 60s App-Nap window to all non-essential
                // backgrounds so the foreground app restores its state first.
                let elapsed_since_last_cycle = last_cycle_instant.elapsed();
                if elapsed_since_last_cycle > Duration::from_secs(30) {
                    wake_suppression_until = Some(Instant::now() + Duration::from_secs(60));
                    println!(
                        "[wake] System woke from sleep ({}s gap) — 60s background suppression active",
                        elapsed_since_last_cycle.as_secs()
                    );
                    // Release any App Nap set before sleep; re-evaluate fresh.
                    let mut qos = state.mach_qos.lock_recover();
                    qos.release_all_app_nap();
                }
                last_cycle_instant = Instant::now();
                let in_wake_suppression = wake_suppression_until
                    .map(|t| Instant::now() < t)
                    .unwrap_or(false);

                // Enforce minimum 300ms between cycles to prevent event-storm CPU burn.
                let since_last = last_cycle_end.elapsed();
                if since_last < Duration::from_millis(300) {
                    thread::sleep(Duration::from_millis(300) - since_last);
                }

                if Path::new(kill_switch_path()).exists() {
                    // Even when paused, populate basic observability metrics
                    // so the dashboard shows real system state.
                    {
                        let cached = pressure_collector.latest();
                        let mut metrics = state.metrics.lock_recover();
                        if pressure_collector.data_age() < Duration::from_secs(10) {
                            metrics.memory_pressure = cached.memory_pressure;
                            metrics.swap_used_bytes = cached.swap_used_bytes;
                            metrics.swap_delta_bps = cached.swap_delta_bps;
                        }
                        if let Some(hw) = smc_reader.latest() {
                            metrics.iokit_p_cluster_temp = hw.temps.p_cluster_celsius;
                            metrics.iokit_e_cluster_temp = hw.temps.e_cluster_celsius;
                            metrics.iokit_package_watts = hw.power.package_watts;
                        }
                        metrics.thermal_state = state.thermal_state.lock_recover().clone();
                    }
                    last_cycle_end = Instant::now();
                    thread::sleep(Duration::from_secs(5));
                    continue;
                }

                let cycle_start = Instant::now();
                // Mark reactor as stalled only if the reactor thread has sent
                // zero pulses after 60 s — that means the thread itself died,
                // not just that the system has been quiet.
                if daemon_start.elapsed() > Duration::from_secs(60) {
                    let pulses = state.metrics.lock_recover().reactor_pulses;
                    if pulses == 0 {
                        {
                            let mut rs = state.reactor_status.lock_recover();
                            rs.mode = "degraded".to_string();
                            rs.health = "stalled".to_string();
                        }
                        *state.fast_tick_until.lock_recover() = None;
                    } else {
                        // Reactor thread is alive; health tracks actual events.
                        let mut rs = state.reactor_status.lock_recover();
                        if rs.mode == "degraded" {
                            rs.mode = "normal".to_string();
                            rs.health = "ok".to_string();
                        }
                    }

                    // Watchdog: check background collector health every 60 cycles (also cycle 1).
                    if cycle_count % 60 == 0 || cycle_count == 1 {
                        let pressure_alive = pressure_collector.is_alive(120);
                        let smc_alive = smc_reader.is_alive(120);
                        {
                            let mut m = state.metrics.lock_recover();
                            m.collector_pressure_alive = pressure_alive;
                            m.collector_smc_alive = smc_alive;
                        }
                        if !pressure_alive || !smc_alive {
                            state.reactor_status.lock_recover().health =
                                "collector-stalled".to_string();
                            // Respawn stalled collectors so the main loop gets fresh data.
                            if !smc_alive {
                                eprintln!("watchdog: SmcReader stalled — respawning");
                                smc_reader = SmcReader::spawn(Duration::from_secs(3));
                            }
                            if !pressure_alive {
                                eprintln!("watchdog: PressureCollector stalled — respawning");
                                pressure_collector =
                                    PressureCollector::spawn(Duration::from_secs(3));
                            }
                        }
                    }
                }
                let now_wall = Utc::now();
                let mut wake_state_guard = state.wake_state.lock_recover();
                let wake_jump = now_wall - wake_state_guard.last_cycle_wallclock;
                let mut grace_active = wake_state_guard
                    .post_wake_grace_until
                    .map(|t| t > now_wall)
                    .unwrap_or(false);
                if wake_jump > ChronoDuration::seconds(90) {
                    // Treat as wake: engage grace window and unfreeze anything Apollo froze.
                    wake_state_guard.last_wake_at = Some(now_wall);
                    wake_state_guard.post_wake_grace_until =
                        Some(now_wall + ChronoDuration::seconds(60));
                    grace_active = true;

                    let mut frozen_state = state.frozen_state.lock_recover();
                    let unfreeze_count = unfreeze_pids(frozen_state.keys().copied());
                    frozen_state.clear();
                    write_frozen_state(&frozen_state_path, &frozen_state);

                    {
                        let mut metrics = state.metrics.lock_recover();
                        metrics.wake_events += 1;
                        metrics.post_wake_grace_entries += 1;
                        metrics.post_wake_defensive_unfreezes += unfreeze_count;
                        metrics.unfreezes_applied += unfreeze_count;
                        metrics.throttle_reverted += unfreeze_count;
                    }
                }
                wake_state_guard.last_cycle_wallclock = now_wall;
                write_wake_state(&wake_state_path, &wake_state_guard);
                drop(wake_state_guard);

                // Display-Off Turbo: Android Doze-like power management.
                // Battery-aware dwell: on battery shorten dwell to 2s so turbo activates
                // faster → more aggressive power savings when user steps away.
                display_turbo.set_dwell_secs(if power_mgr.is_on_battery() { 2 } else { 5 });

                // When display is off for >5s (or 2s on battery), freeze all non-essential processes.
                // On display-on, instantly unfreeze everything we froze.
                {
                    use apollo_optimizer::engine::display_turbo::TurboAction;
                    match display_turbo.tick() {
                        TurboAction::ActivateTurbo => {
                            // Freeze non-essential background processes.
                            let critical_pats = critical_background_processes();
                            let protected_pats = protected_processes();
                            let policy_protected = state
                                .learned_policy
                                .lock_recover()
                                .protected_patterns
                                .clone();
                            let fg_pid = fg_detector.detect().pid();
                            let mut frozen_guard = state.frozen_state.lock_recover();
                            let mut turbo_frozen = 0u32;
                            let max_freeze = display_turbo.max_freeze_count();

                            for (pid, process) in collector.system().processes() {
                                let pid_u32 = pid.as_u32();
                                let name = process.name().to_string();
                                // Never freeze: foreground, essential, protected, Apollo itself.
                                if Some(pid_u32) == fg_pid
                                    || critical_pats.iter().any(|p| name.contains(p))
                                    || protected_pats.iter().any(|p| name.contains(p))
                                    || policy_protected.iter().any(|p| name.contains(p.as_str()))
                                    || name == "apollo-optimizerd"
                                    || frozen_guard.contains_key(&pid_u32)
                                {
                                    continue;
                                }
                                if turbo_frozen as usize >= max_freeze {
                                    break;
                                }
                                // SIGSTOP the process.
                                if unsafe { libc::kill(pid_u32 as i32, libc::SIGSTOP) } == 0 {
                                    display_turbo.record_turbo_freeze(pid_u32);
                                    frozen_guard.insert(
                                        pid_u32,
                                        FrozenEntry {
                                            frozen_at: Utc::now(),
                                            source: FreezeSource::MainLoop,
                                            pressure_at_freeze: pressure_collector
                                                .latest()
                                                .memory_pressure,
                                        },
                                    );
                                    turbo_frozen += 1;
                                }
                            }
                            write_frozen_state(&frozen_state_path, &frozen_guard);
                            drop(frozen_guard);
                            state.metrics.lock_recover().freezes_applied += turbo_frozen as u64;
                        }
                        TurboAction::DeactivateTurbo {
                            unfreeze_pids: pids,
                        } => {
                            // Unfreeze all PIDs we froze during turbo.
                            let unfreeze_count = unfreeze_pids(pids.iter().copied());
                            let mut frozen_guard = state.frozen_state.lock_recover();
                            for pid in &pids {
                                frozen_guard.remove(pid);
                            }
                            write_frozen_state(&frozen_state_path, &frozen_guard);
                            drop(frozen_guard);
                            state.metrics.lock_recover().unfreezes_applied += unfreeze_count;
                        }
                        TurboAction::None => {}
                    }
                }

                // Adaptive snapshot: use lightweight path (no disk/net refresh) every cycle
                // except a full-refresh heartbeat every 30 cycles (~15s).
                // Disk/network data from sysinfo is not consumed on the hot path — the
                // network monitor and sysctl governor read directly from sysctl/netstat.
                // Dropping the pressure gate removes ~15-25ms of disk/net I/O at 0.70+ pressure
                // where the old 0.40 threshold never fired anyway.
                let cached_mem_pressure = pressure_collector.latest().memory_pressure;
                let use_light = cycle_count % 30 != 0;
                let mut snapshot = if use_light {
                    collector.collect_snapshot_light()
                } else {
                    collector.collect_snapshot()
                };
                // Overlay pressure data from background PressureCollector cache
                // when it's fresh (< 10s old), avoiding blocking subprocesses on hot path.
                {
                    let cached_pressure = pressure_collector.latest();
                    if pressure_collector.data_age() < Duration::from_secs(10) {
                        snapshot.pressure.memory_pressure = cached_pressure.memory_pressure;
                        snapshot.pressure.swap_used_bytes = cached_pressure.swap_used_bytes;
                        snapshot.pressure.swap_total_bytes = cached_pressure.swap_total_bytes;
                        snapshot.pressure.swap_delta_bytes_per_sec = cached_pressure.swap_delta_bps;
                    }
                }
                snapshot.pressure.thermal_level = state.thermal_level_real.lock_recover().clone();
                let latency_target = *state.latency_target.lock_recover();

                // Foreground detection: use ForegroundDetector instead of get_foreground_app().
                let fg_state = fg_detector.detect();
                let foreground_app = fg_state.name().map(|s| s.to_string());
                let foreground_pid = fg_state.pid();
                let foreground_idle = fg_state.is_idle();

                // Markov chain: observe foreground transition, predict next app.
                // Pre-warm the predicted app by unfreezing + boosting QoS before
                // the user switches to it — eliminates perceived switch latency.
                let markov_prediction = focus_markov.observe(foreground_app.as_deref());
                if let Some(ref pred) = markov_prediction {
                    // Find the PID of the predicted app in the process table.
                    let pred_name_lc = pred.app_name.to_ascii_lowercase();
                    let predicted_pid: Option<u32> = collector
                        .system()
                        .processes()
                        .iter()
                        .find(|(_, p)| p.name().to_ascii_lowercase() == pred_name_lc)
                        .map(|(pid, _)| pid.as_u32());

                    if let Some(pid) = predicted_pid {
                        // Pre-warm: if predicted app is frozen, unfreeze it now.
                        let mut frozen_guard = state.frozen_state.lock_recover();
                        if frozen_guard.remove(&pid).is_some() {
                            unfreeze_pids(std::iter::once(pid));
                            write_frozen_state(&frozen_state_path, &frozen_guard);
                            state.metrics.lock_recover().unfreezes_applied += 1;
                        }
                        drop(frozen_guard);

                        // Boost jetsam priority so kernel protects this app's pages
                        // before the user switches to it (pages stay resident).
                        if pred.probability >= 0.50 {
                            let _ = jetsam_control::set_priority(
                                pid,
                                jetsam_control::priority::FOREGROUND,
                            );
                            // Cable C: Proactive QoS — route predicted app to P-cores
                            // BEFORE the user switches to it (predictive DVFS pattern).
                            // Eliminates the ~50ms QoS transition lag on app switch.
                            {
                                let mut qos = state.mach_qos.lock_recover();
                                qos.set_tier(pid, SchedulingTier::Foreground);
                            }
                            // File cache warming: pre-read the app's executable into
                            // the buffer cache so code pages don't fault from SSD.
                            // Cao et al. 1994 — app-controlled prefetch cuts I/O wait 50%.
                            cache_warmer.warm_pid(pid);
                        }
                    }
                }

                // Temporal app predictor: observe foreground app + hour for time-of-day patterns.
                // Shin et al. 2012 — temporal patterns predict app launches with ~80% accuracy.
                // On foreground change, record observation + get temporal prediction for
                // proactive pre-warming of apps the user habitually opens at this hour.
                // Observe only on app transition (not every cycle) to avoid count inflation
                // and excess disk writes. last_fg_name is updated at end of ctx-switch block.
                if let Some(ref fg_name) = foreground_app {
                    let now_chrono = Utc::now();
                    let hour = now_chrono.hour() as u8;
                    let weekday =
                        chrono::Datelike::weekday(&now_chrono).num_days_from_monday() as u8;
                    let fg_changed = last_fg_name.as_deref() != Some(fg_name.as_str());
                    if fg_changed {
                        temporal_predictor.observe(fg_name, hour, weekday);
                    }

                    // Build Markov probability map for blending with temporal model.
                    let markov_probs: std::collections::HashMap<String, f64> = focus_markov
                        .predict_top_n(fg_name, 5)
                        .into_iter()
                        .map(|p| (p.app_name, p.probability))
                        .collect();

                    // Get temporal-blended predictions: apps likely needed at this time.
                    let temporal_preds = temporal_predictor.predict(hour, weekday, &markov_probs);

                    // Pre-warm temporal predictions that Markov alone wouldn't catch.
                    // Only warm if temporal_score > 0.3 (strong time signal) and
                    // probability > 0.15 (avoid warming everything).
                    for tpred in &temporal_preds {
                        if tpred.temporal_score > 0.3
                            && tpred.probability > 0.15
                            && tpred.markov_score < 0.30
                        {
                            // This is a purely temporal prediction — Markov wouldn't
                            // have caught it.  Pre-warm via cache warmer.
                            let pred_lc = tpred.app_name.to_ascii_lowercase();
                            if let Some(pid) = collector
                                .system()
                                .processes()
                                .iter()
                                .find(|(_, p)| p.name().to_ascii_lowercase() == pred_lc)
                                .map(|(pid, _)| pid.as_u32())
                            {
                                cache_warmer.warm_pid(pid);
                            }
                        }
                    }
                }

                // Context-switch burst detector + reactive unfreeze.
                // Si el foreground cambió y el nuevo app estaba congelado, lo descongelamos
                // de inmediato — sin esperar al siguiente ciclo de optimización.
                {
                    let fg_now = foreground_app.clone();
                    let fg_changed =
                        fg_now.is_some() && last_fg_name.is_some() && fg_now != last_fg_name;

                    if fg_changed {
                        ctx_switch_times.push_back(Instant::now());
                    }

                    // Reactive unfreeze: si el pid activo está en frozen_state, SIGCONT inmediato.
                    // Usamos solo el foreground_pid aquí (process_tree aún no está construido);
                    // el resto de la familia se descongela en el siguiente ciclo normal.
                    if let Some(fg_pid) = foreground_pid {
                        let mut frozen_guard = state.frozen_state.lock_recover();
                        if frozen_guard.remove(&fg_pid).is_some() {
                            unfreeze_pids(std::iter::once(fg_pid));
                            write_frozen_state(&frozen_state_path, &frozen_guard);
                            drop(frozen_guard);
                            state.metrics.lock_recover().unfreezes_applied += 1;
                        }
                    }

                    last_fg_name = fg_now;
                    let cutoff = Instant::now() - Duration::from_secs(300);
                    ctx_switch_times.retain(|t| *t > cutoff);
                }

                // Process tree: build from the full process table for child grouping.
                let process_tree = {
                    let sys = collector.system();
                    let entries: Vec<ProcessEntry> = sys
                        .processes()
                        .iter()
                        .map(|(pid, process)| ProcessEntry {
                            pid: pid.as_u32(),
                            ppid: process.parent().map(|p| p.as_u32()).unwrap_or(0),
                            name: process.name().to_string(),
                            cpu_usage: process.cpu_usage(),
                            memory_bytes: process.memory(),
                        })
                        .collect();
                    ProcessTree::build(&entries)
                };

                // Build enriched process data using foreground detector + process tree.
                // A process is considered foreground if it IS the foreground app or a
                // descendant of it (via process tree), giving accurate foreground family
                // detection for multi-process apps like Chrome, Electron, etc.
                let (proc_snaps, hunt_snaps) = build_enriched_process_data_with_tree(
                    collector.system(),
                    foreground_pid,
                    &process_tree,
                );
                let all_proc_names: Vec<&str> =
                    proc_snaps.iter().map(|p| p.name.as_str()).collect();
                let hour_of_day = Utc::now().hour() as u8;

                // MemoryAnalyzer: profile top-50 processes for memory leaks each cycle.
                // For the top-10 by RSS, refine WSS with real TASK_VM_INFO data.
                for (i, snap) in proc_snaps.iter().take(50).enumerate() {
                    let mut profile = mem_analyzer.analyze_process(
                        snap.pid,
                        &snap.name,
                        snap.rss_bytes,
                        snap.rss_bytes, // vms not tracked at this level; use rss as proxy
                        snap.pageins_total as u64, // major faults (page-ins from disk/swap/compressor)
                    );
                    // Top-10 by RSS: refine WSS with Mach TASK_VM_INFO (~50µs per call).
                    if i < 10 {
                        if let Some(mem_profile) = query_memory_profile(snap.pid) {
                            MemoryAnalyzer::refine_wss(&mut profile, mem_profile.working_set_bytes);
                        }
                    }
                    if profile.memory_leak_probability >= 0.75 {
                        proc_recovery.register_leak(
                            snap.pid,
                            snap.name.clone(),
                            profile.memory_leak_probability,
                            snap.rss_bytes,
                        );
                    }
                }
                proc_recovery.cleanup_resolved();

                // WakeStormDetector: record wakeups for any process reporting elevated rates.
                for snap in proc_snaps.iter().take(50) {
                    if snap.wakeups_per_sec > 10.0 {
                        wake_storm.record_wakeup(snap.pid, snap.name.clone());
                    }
                }
                wake_storm.cleanup_stale(Duration::from_secs(300));

                // Memory budgets: compute and enforce jetsam limits when pressure ≥ 0.60.
                // Only recompute under pressure to avoid unnecessary syscalls in idle.
                if snapshot.pressure.memory_pressure >= 0.60 {
                    let usage_model = state.usage_model.lock_recover();
                    let budget_inputs: Vec<ProcessBudgetInput> = proc_snaps
                        .iter()
                        .take(30) // Top 30 processes
                        .filter(|s| s.rss_bytes > 50 * 1024 * 1024) // Only >50MB
                        .map(|s| {
                            let (presence, interactive) = usage_model
                                .entries()
                                .get(&s.name.to_ascii_lowercase())
                                .map(|e| (e.presence_ema, e.interactive_ema))
                                .unwrap_or((0.1, 0.0));
                            // Use real WSS from TASK_VM_INFO when available,
                            // fall back to fault-rate heuristic.
                            let wss_bytes = query_memory_profile(s.pid)
                                .map(|p| p.working_set_bytes)
                                .unwrap_or_else(|| {
                                    let fault_rate = mem_analyzer.major_fault_rate(s.pid);
                                    if fault_rate > 50.0 {
                                        (s.rss_bytes as f64 * 1.3) as u64
                                    } else {
                                        s.rss_bytes
                                    }
                                });
                            ProcessBudgetInput {
                                pid: s.pid,
                                name: s.name.clone(),
                                rss_bytes: s.rss_bytes,
                                working_set_bytes: wss_bytes,
                                is_foreground: s.has_gui_window && s.secs_since_foreground == 0,
                                is_build_tool: BUILD_TOOLS.iter().any(|t| s.name.contains(t)),
                                presence_ema: presence,
                                interactive_ema: interactive,
                            }
                        })
                        .collect();
                    drop(usage_model);

                    if !budget_inputs.is_empty() {
                        let budgets = memory_budget::compute_budgets(
                            snapshot.memory.total_ram,
                            &budget_inputs,
                        );

                        // Apply jetsam inactive limits for over-budget processes.
                        for budget in budgets.iter().filter(|b| b.over_budget) {
                            let _ = jetsam_control::set_memlimit(
                                budget.pid,
                                0, // active: unlimited (don't kill foreground)
                                budget.inactive_limit_mb,
                            );
                        }
                    }
                }

                // Audit fix #5: Read cached hardware data from background SmcReader thread.
                // No more blocking 500 ms powermetrics calls on the hot path.
                {
                    if let Some(hw) = smc_reader.latest() {
                        {
                            let mut m = state.metrics.lock_recover();
                            m.iokit_snapshots = smc_reader.success_count();
                            m.iokit_errors = smc_reader.error_count();
                            m.iokit_p_cluster_temp = hw.temps.p_cluster_celsius;
                            m.iokit_e_cluster_temp = hw.temps.e_cluster_celsius;
                            m.iokit_package_watts = hw.power.package_watts;
                        }
                        *state.last_hw_snapshot.lock_recover() = Some(hw);
                    } else {
                        state.metrics.lock_recover().iokit_errors = smc_reader.error_count();
                    }
                }

                // Battery status: detect real battery state every 10 cycles (~30s)
                // to avoid spawning pmset too frequently.
                if cycle_count % 10 == 1 {
                    if let Some(batt) = detect_battery_status() {
                        power_mgr.update_battery_status(batt);
                    }
                }

                // Snapshot hardware data once per cycle (avoids 6 redundant mutex+clone operations).
                let cycle_hw_snap: Option<HardwareSnapshot> =
                    state.last_hw_snapshot.lock_recover().clone();

                // EnergyTracker: update per-app energy estimates with this cycle's data.
                let cycle_dt_secs = last_cycle_instant.elapsed().as_secs_f64();
                last_cycle_instant = Instant::now();
                {
                    if let Some(ref hw) = cycle_hw_snap {
                        energy_tracker.update(&snapshot.top_processes, hw, cycle_dt_secs);
                    }
                }

                // Audit fix #6: Multi-phase thermal bail-out with hysteresis.
                let thermal_action = {
                    if let Some(hw) = &cycle_hw_snap {
                        thermal_bailout.evaluate(hw)
                    } else {
                        thermal_bailout.evaluate(&HardwareSnapshot {
                            thermal_state:
                                apollo_optimizer::engine::iokit_sensors::ThermalState::Normal,
                            temps: apollo_optimizer::engine::iokit_sensors::ClusterTemps {
                                p_cluster_celsius: None,
                                e_cluster_celsius: None,
                                gpu_celsius: None,
                                nand_celsius: None,
                            },
                            power: apollo_optimizer::engine::iokit_sensors::PowerReading {
                                package_watts: None,
                                cpu_watts: None,
                                gpu_watts: None,
                                dram_watts: None,
                            },
                            p_cluster_util: None,
                            e_cluster_util: None,
                            battery_percent: None,
                            battery_watts: None,
                        })
                    }
                };
                let thermal_emergency = thermal_action.force_ecores;
                // Thermal pre-throttle boost: raise effective pressure early (Phase1=80°C)
                // so page_reclaim + io_shaper + governor act before hardware throttles.
                // M1 Air has no fan — acting 5-10°C before the hardware ceiling prevents
                // visible stutter caused by hardware-level frequency reduction.
                let thermal_pressure_boost = match thermal_action.phase {
                    apollo_optimizer::engine::thermal_bailout::CoolingPhase::Normal => 0.0,
                    apollo_optimizer::engine::thermal_bailout::CoolingPhase::Phase1Gentle => 0.07,
                    apollo_optimizer::engine::thermal_bailout::CoolingPhase::Phase2Moderate => 0.15,
                    apollo_optimizer::engine::thermal_bailout::CoolingPhase::Phase3Aggressive => 0.25,
                    apollo_optimizer::engine::thermal_bailout::CoolingPhase::Phase4Emergency => 0.40,
                };

                // Thermal Pre-Throttle: proactively freeze SilentDaemon/Stale backgrounds at
                // Phase3Aggressive (≥90°C) before hardware throttling causes visible stutter.
                // M1 Air has no fan — acting here is 5-10°C ahead of the hardware ceiling.
                // Unfreeze when temperature drops back to Phase2 or below (hysteresis built into
                // ThermalBailout keeps us from thrashing).
                if thermal_action.freeze_background || thermal_action.freeze_all_non_critical {
                    let critical_pats = critical_background_processes();
                    let protected_pats = protected_processes();
                    let policy_protected = state
                        .learned_policy
                        .lock_recover()
                        .protected_patterns
                        .clone();
                    let fg_pid = foreground_pid;
                    let mut frozen_guard = state.frozen_state.lock_recover();
                    let mut thermal_frozen = 0u32;
                    // Phase3: only freeze idle backgrounds (<2% CPU).
                    // Phase4: freeze everything non-critical regardless of CPU.
                    let cpu_threshold: f32 = if thermal_action.freeze_all_non_critical {
                        100.0 // Phase4: no CPU filter
                    } else {
                        2.0 // Phase3: only idle processes
                    };

                    for (pid, process) in collector.system().processes() {
                        let pid_u32 = pid.as_u32();
                        let name = process.name().to_string();
                        let cpu = process.cpu_usage();
                        if cpu > cpu_threshold
                            || Some(pid_u32) == fg_pid
                            || critical_pats.iter().any(|p| name.contains(p))
                            || protected_pats.iter().any(|p| name.contains(p))
                            || policy_protected.iter().any(|p| name.contains(p.as_str()))
                            || name == "apollo-optimizerd"
                            || frozen_guard.contains_key(&pid_u32)
                        {
                            continue;
                        }
                        if thermal_frozen >= 80 {
                            break;
                        }
                        if unsafe { libc::kill(pid_u32 as i32, libc::SIGSTOP) } == 0 {
                            frozen_guard.insert(
                                pid_u32,
                                FrozenEntry {
                                    frozen_at: Utc::now(),
                                    source: FreezeSource::ThermalPreThrottle,
                                    pressure_at_freeze: snapshot.pressure.memory_pressure,
                                },
                            );
                            thermal_frozen += 1;
                        }
                    }
                    if thermal_frozen > 0 {
                        write_frozen_state(&frozen_state_path, &frozen_guard);
                        state.metrics.lock_recover().freezes_applied += thermal_frozen as u64;
                        println!(
                            "[thermal] Phase {:?}: froze {} background processes (pre-throttle)",
                            thermal_action.phase, thermal_frozen
                        );
                    }
                    drop(frozen_guard);
                } else {
                    // Temperature dropped back to Phase2 or below — unfreeze any PIDs we froze
                    // thermally so the system returns to normal when it's cool enough.
                    let thermal_frozen_pids: Vec<u32> = {
                        let frozen_guard = state.frozen_state.lock_recover();
                        frozen_guard
                            .iter()
                            .filter(|(_, e)| e.source == FreezeSource::ThermalPreThrottle)
                            .map(|(&pid, _)| pid)
                            .collect()
                    };
                    if !thermal_frozen_pids.is_empty() {
                        let n = unfreeze_pids(thermal_frozen_pids.iter().copied());
                        let mut frozen_guard = state.frozen_state.lock_recover();
                        for pid in &thermal_frozen_pids {
                            frozen_guard.remove(pid);
                        }
                        write_frozen_state(&frozen_state_path, &frozen_guard);
                        drop(frozen_guard);
                        state.metrics.lock_recover().unfreezes_applied += n;
                        println!("[thermal] Cooled: unfroze {} pre-throttled processes", n);
                    }
                }

                // HwPredictor: sample hardware signals every 10 cycles (~5s at normal rate).
                // Runs in <50ms (16MB cache probe + 32MB BW probe) and gives advance warning
                // before metrics APIs catch up. 5s is sufficient — thermal buildup takes ≥10s.
                let (hw_pressure, jitter_us, hw_features) = if cycle_count % 10 == 0 {
                    let snap = sample_hw_pressure();
                    if snap.is_critical() {
                        *state.fast_tick_until.lock_recover() =
                            Some(Instant::now() + Duration::from_secs(15));
                        println!(
                            "hw_predictor: CRITICAL — jitter={}µs throughput={}Mips cache={}µs → fast-tick engaged",
                            snap.jitter_us, snap.throughput_mips, snap.cache_latency_us
                        );
                    } else if snap.needs_attention() {
                        println!(
                            "hw_predictor: WARNING — jitter={}µs throughput={}Mips cache={}µs",
                            snap.jitter_us, snap.throughput_mips, snap.cache_latency_us
                        );
                    }
                    let feat = HwFeatures {
                        throughput_mips: snap.throughput_mips as f64,
                        jitter_us: snap.jitter_us as f64,
                        cache_latency_us: snap.cache_latency_us as f64,
                    };
                    (snap.overall, snap.jitter_us, Some(feat))
                } else {
                    (HwPressure::Nominal, 0u64, None)
                };
                last_hw_pressure = hw_pressure;

                // ThermalManager + GPUManager: tick every cycle with latest IOKit temperatures.
                // gpu_thermal_throttled escapes this block to feed into governor input.
                let mut gpu_thermal_throttled = false;
                {
                    if let Some(hw) = &cycle_hw_snap {
                        let cpu_t = hw.temps.p_cluster_celsius.unwrap_or(0.0);
                        let gpu_t = hw.temps.gpu_celsius.unwrap_or(cpu_t);
                        let _thermal_state = thermal_mgr.update(cpu_t, gpu_t, 0.0, 0, jitter_us);

                        // GPU-aware thermal management: build GPU metrics from IOKit data
                        // and feed into gpu_manager for workload-specific recommendations.
                        let gpu_watts = hw.power.gpu_watts.unwrap_or(0.0);
                        let gpu_util = (gpu_watts / 15.0 * 100.0).clamp(0.0, 100.0);
                        let gpu_metrics = GPUMetrics {
                            gpu_temp: gpu_t,
                            gpu_utilization: gpu_util,
                            gpu_frequency: 0, // Not available from IOKit
                            gpu_memory_used: 0,
                            gpu_memory_total: 0,
                            throttle_active: gpu_mgr.needs_cooling(&GPUMetrics {
                                gpu_temp: gpu_t,
                                gpu_utilization: gpu_util,
                                gpu_frequency: 0,
                                gpu_memory_used: 0,
                                gpu_memory_total: 0,
                                throttle_active: false,
                                power_state: GPUPowerState::Dynamic,
                            }),
                            power_state: gpu_mgr.recommend_power_state(gpu_util, gpu_t),
                        };
                        // If GPU is thermally throttled, engage fast-tick for quicker response.
                        if gpu_metrics.power_state == GPUPowerState::Throttled {
                            gpu_thermal_throttled = true;
                            *state.fast_tick_until.lock_recover() =
                                Some(Instant::now() + Duration::from_secs(15));
                        }
                        // Cable: GPU thermal audit — log thermal_recommendations on throttle
                        // transitions and workload-specific hints for observability.
                        if gpu_metrics.throttle_active || gpu_metrics.power_state == GPUPowerState::Throttled {
                            let recs = gpu_mgr.thermal_recommendations(&gpu_metrics);
                            if !recs.is_empty() {
                                audit_log(&serde_json::json!({
                                    "event": "gpu_thermal",
                                    "gpu_temp": gpu_t,
                                    "gpu_util": gpu_util,
                                    "power_state": format!("{:?}", gpu_metrics.power_state),
                                    "recommendations": recs,
                                }));
                            }
                        }
                        // Store GPU power state in metrics for status reporting.
                        state.metrics.lock_recover().energy_gpu_watts =
                            Some(hw.power.gpu_watts.unwrap_or(0.0) as f64);
                    }
                }

                // SwapPredictor: update trend forecast every cycle.
                let swap_forecast = swap_predictor.update(
                    snapshot.pressure.swap_used_bytes,
                    snapshot.pressure.swap_total_bytes,
                );

                // PowerManager: advisory tick (no real sensor data yet).
                let _power_rec = power_mgr.get_recommendation();

                // EMA interactivity classifier: compute cpu_wall_ratio per PID
                // from proc_pid_rusage deltas. This measures how CPU-bound each
                // process is (low ratio = I/O-bound/interactive, high = CPU-bound).
                let elapsed_rusage = last_rusage_at.elapsed();
                last_rusage_at = Instant::now();
                let mut cpu_wall_ratios: HashMap<String, f32> = HashMap::new();
                let mut new_rusage_prev: HashMap<u32, (u64, u64, u64)> = HashMap::new();
                for p in &snapshot.top_processes {
                    if let Some(ri) = proc_taskinfo::get_rusage_info(p.pid) {
                        let total_cpu = ri.user_time_ns + ri.system_time_ns;
                        if let Some(&(prev_user, prev_system, prev_start)) =
                            rusage_cpu_prev.get(&p.pid)
                        {
                            // PID recycling guard: if proc_start_abstime changed,
                            // this is a different process reusing the PID.
                            if ri.proc_start_abstime == prev_start {
                                let prev_total = prev_user + prev_system;
                                if total_cpu >= prev_total {
                                    let delta_cpu = total_cpu - prev_total;
                                    let delta_wall = elapsed_rusage.as_nanos() as u64;
                                    if delta_wall > 0 {
                                        let ratio =
                                            (delta_cpu as f64 / delta_wall as f64).min(1.0) as f32;
                                        cpu_wall_ratios.insert(p.name.clone(), ratio);
                                    }
                                }
                            }
                        }
                        new_rusage_prev.insert(
                            p.pid,
                            (ri.user_time_ns, ri.system_time_ns, ri.proc_start_abstime),
                        );
                    }
                }
                rusage_cpu_prev = new_rusage_prev;

                // Online usage learning (root-only, no UI sensors): infer frequently-used apps
                // and processes correlated with jank, then promote patterns conservatively.
                usage_learning_tick(
                    &state,
                    &snapshot,
                    !foreground_idle && foreground_app.is_some(),
                    &cpu_wall_ratios,
                );

                // LLM teacher mode (cloud) - optional, rate-limited, and guarded.
                // This runs before governor evaluation so a high-confidence suggestion can set a
                // short-lived manual override during the training window.
                llm_reactive_tick(
                    &state,
                    &mut llm_advisor,
                    &snapshot,
                    &mut llm_counters,
                    outcome_tracker.heuristic_is_struggling(),
                );

                let mut reactor_weight = state.reactor_event_weight.lock_recover();
                *reactor_weight = (*reactor_weight * 0.75).clamp(0.0, 1.0);

                // kqueue: consume VM pressure events from kernel push notifications.
                // Critical/SuddenTerminate → boost reactor_weight + engage fast-tick.
                // This is the fastest possible pressure detection — zero polling latency.
                if let Some(ref mut kq) = kq_frozen {
                    for event in kq.poll_events() {
                        match event {
                            kqueue_pressure::PressureEvent::VmPressure(level) => {
                                use kqueue_pressure::VmPressureLevel;
                                match level {
                                    VmPressureLevel::Critical
                                    | VmPressureLevel::SuddenTerminate => {
                                        *reactor_weight = 1.0;
                                        *state.fast_tick_until.lock_recover() =
                                            Some(Instant::now() + Duration::from_secs(30));
                                        println!(
                                            "kqueue: VM pressure {:?} — fast-tick engaged",
                                            level
                                        );
                                        // Registrar overflow: ajustar thresholds para prevenir próxima vez.
                                        // Excluir el propio daemon — aparece en top_processes durante
                                        // survival-mode por el trabajo intensivo que hace, contaminando
                                        // el diagnóstico de causa del overflow.
                                        let heavy: Vec<String> = snapshot
                                            .top_processes
                                            .iter()
                                            .filter(|p| p.name != "apollo-optimizerd")
                                            .take(8)
                                            .map(|p| p.name.clone())
                                            .collect();
                                        overflow_guard.record_event(
                                            snapshot.pressure.memory_pressure,
                                            snapshot.pressure.swap_delta_bytes_per_sec,
                                            &heavy,
                                            &format!("kqueue-{:?}", level),
                                            snapshot.pressure.compressor_pressure,
                                        );
                                        // Teach hazard model about this overflow.
                                        let sr = if snapshot.pressure.swap_total_bytes > 0 {
                                            snapshot.pressure.swap_used_bytes as f64
                                                / snapshot.pressure.swap_total_bytes as f64
                                        } else {
                                            0.0
                                        };
                                        signal_intel.record_overflow(
                                            snapshot.pressure.memory_pressure,
                                            sr,
                                            snapshot.pressure.memory_pressure,
                                            1.0,
                                        );
                                    }
                                    VmPressureLevel::Warning => {
                                        *reactor_weight = (*reactor_weight + 0.5).min(1.0);
                                    }
                                    VmPressureLevel::Normal => {}
                                }
                            }
                            kqueue_pressure::PressureEvent::ProcessExited(pid) => {
                                // Frozen process died (jetsam/OOM) — clean up immediately.
                                let mut frozen_state = state.frozen_state.lock_recover();
                                if frozen_state.remove(&pid).is_some() {
                                    write_frozen_state(&frozen_state_path, &frozen_state);
                                    state.metrics.lock_recover().unfreezes_applied += 1;
                                }
                                // Also clean up display turbo's set — prevents unbounded
                                // growth if many processes die while frozen during turbo.
                                display_turbo.remove_pid(pid);
                            }
                            kqueue_pressure::PressureEvent::TimerTick => {}
                        }
                    }
                }

                // hw_predictor can elevate pressure before standard metrics catch up.
                let hw_boost = match hw_pressure {
                    HwPressure::Critical => 0.3,
                    HwPressure::Warning => 0.15,
                    HwPressure::Nominal => 0.0,
                };
                let pressure_cpu =
                    ((snapshot.cpu.global_usage as f64 / 100.0) + hw_boost).clamp(0.0, 1.0);
                let batt_boost = battery_pressure_boost(&power_mgr);

                // ── IOReport: P/E cluster utilization + real power telemetry ──
                // Sample delta every cycle (≥500ms interval typical).
                // end_sample() + begin_sample() gives rolling inter-cycle window.
                if ioreport.available && last_ioreport_sample.elapsed() >= Duration::from_millis(900) {
                    #[cfg(target_os = "macos")]
                    {
                        last_ioreport = ioreport.end_sample();
                        ioreport.begin_sample();
                    }
                    last_ioreport_sample = Instant::now();
                }

                // ── SMC Direct: power, lid, sleep/wake, battery ─────────────
                if smc_direct.available {
                    last_smc = smc_direct.read_snapshot();
                }

                // ── KPC: hardware performance counters (IPC) ────────────────
                let kpc_snap = if kpc_reader.available {
                    kpc_reader.sample()
                } else {
                    None
                };

                // ── Rosetta AOT: poll for oahd-helper activity ──────────────
                rosetta_monitor.poll();

                // ── Per-process energy ranking (ri_billed_energy) ────────────
                let energy_pid_results = {
                    let procs: Vec<(u32, &str)> = snapshot
                        .top_processes
                        .iter()
                        .map(|p| (p.pid, p.name.as_str()))
                        .collect();
                    energy_pid_tracker.sample(&procs, cycle_dt_secs)
                };

                // Build IPC hint map for decide_actions (pid → IPC from rusage).
                let ipc_hints: HashMap<u32, f64> = energy_pid_results
                    .iter()
                    .filter(|e| e.ipc > 0.0)
                    .map(|e| (e.pid, e.ipc))
                    .collect();

                // ── IOPMrootDomain direct thermal (every 10 cycles, aligned with HwPredictor) ──
                let iopm_snap = if cycle_count % 10 == 0 {
                    apollo_optimizer::engine::thermal_iokit::read_iopm_state()
                } else {
                    None
                };

                // ── Memory bandwidth pressure boost ─────────────────────────
                // AMC bandwidth > 80% = memory-bound → freeze more aggressively.
                let mem_bw_boost = last_ioreport
                    .as_ref()
                    .filter(|ir| ir.memory_bandwidth_saturated())
                    .map(|_| 0.10)
                    .unwrap_or(0.0);

                // ── SMC thermal direct boost ────────────────────────────────
                // CPU temp from SMC is real-time (<100µs). Use it to augment
                // thermal_bailout when powermetrics is stale.
                let smc_thermal_boost = last_smc
                    .as_ref()
                    .and_then(|s| s.cpu_temp_celsius)
                    .map(|t| {
                        if t >= 100.0 { 0.30 }      // critical
                        else if t >= 90.0 { 0.15 }   // severe
                        else if t >= 80.0 { 0.05 }    // moderate
                        else { 0.0 }
                    })
                    .unwrap_or(0.0);

                // ── Battery overheat protection ─────────────────────────────
                let battery_overheat_boost = last_smc
                    .as_ref()
                    .filter(|s| s.battery_overheating())
                    .map(|_| 0.12)
                    .unwrap_or(0.0);

                // ── Feature 1: LLM Inference Mode ─────────────────────────────
                // Detect ollama/llama.cpp/MLX and boost pressure gates aggressively.
                let llm_boost = {
                    let proc_iter = snapshot
                        .top_processes
                        .iter()
                        .map(|p| (p.pid, p.name.as_str(), p.cpu_usage));
                    llm_detector.observe(proc_iter);
                    llm_detector.pressure_boost()
                };
                let llm_active = llm_detector.is_active();

                // Spotlight management: disable during LLM inference, re-enable when done.
                if is_root {
                    if llm_active && !llm_spotlight_disabled {
                        spotlight_set_indexing(false);
                        llm_spotlight_disabled = true;
                        println!("[llm-mode] Spotlight indexing disabled for inference");
                    } else if !llm_active && llm_spotlight_disabled {
                        spotlight_set_indexing(true);
                        llm_spotlight_disabled = false;
                        println!("[llm-mode] Spotlight indexing re-enabled");
                    }
                }

                // ── Feature 3: RT Boost for Foreground ────────────────────────
                // Apply THREAD_TIME_CONSTRAINT_POLICY to foreground UI thread.
                // Only when thermal is not critical (Phase3+ would negate the benefit).
                if thermal_action.phase < apollo_optimizer::engine::thermal_bailout::CoolingPhase::Phase3Aggressive {
                    if let Some(fg_pid) = foreground_pid {
                        if rt_boosted_pid != Some(fg_pid) {
                            // Clear RT boost from previous foreground.
                            if let Some(old_pid) = rt_boosted_pid {
                                let mut qos = state.mach_qos.lock_recover();
                                qos.clear_realtime_boost(old_pid);
                            }
                            // Apply RT boost to new foreground.
                            let mut qos = state.mach_qos.lock_recover();
                            if qos.set_realtime_boost(fg_pid) {
                                rt_boosted_pid = Some(fg_pid);
                            } else {
                                rt_boosted_pid = None;
                            }
                        }
                    } else if let Some(old_pid) = rt_boosted_pid {
                        // No foreground — clear boost.
                        let mut qos = state.mach_qos.lock_recover();
                        qos.clear_realtime_boost(old_pid);
                        rt_boosted_pid = None;
                    }
                }

                // ── Charging thermal stress ──────────────────────────────────
                // On fanless M1 Air, charging + heavy compute simultaneously
                // causes SoC thermal throttling.  IOReport total_watts > 8W
                // while charging is a strong indicator.
                // Boost pressure by 0.06 to proactively freeze backgrounds
                // before hardware throttles.
                // Prefer SMC PSTR (real-time, <100µs) over IOReport total_watts.
                let system_watts = last_smc
                    .as_ref()
                    .and_then(|s| s.system_power_watts)
                    .or_else(|| last_ioreport.as_ref().map(|ir| ir.total_watts()));

                let charging_stress_boost = if let Some(watts) = system_watts {
                    let is_charging = last_smc
                        .as_ref()
                        .and_then(|s| s.charger_watts)
                        .map(|cw| cw > 0.0)
                        .unwrap_or_else(|| {
                            cycle_hw_snap
                                .as_ref()
                                .and_then(|h| h.battery_watts)
                                .map(|w| w < 0.0) // negative = charging
                                .unwrap_or(false)
                        });
                    if is_charging && watts > 8.0 {
                        0.06
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                // ── Battery aggressiveness: B0TE < 20 min → extra pressure ──
                let battery_low_boost = last_smc
                    .as_ref()
                    .and_then(|s| s.battery_time_to_empty_min)
                    .filter(|&tte| tte < 20)
                    .map(|_| 0.08)
                    .unwrap_or(0.0);

                let pressure_ram = (snapshot.pressure.memory_pressure + hw_boost + batt_boost + thermal_pressure_boost + llm_boost + charging_stress_boost + battery_low_boost + mem_bw_boost + smc_thermal_boost + battery_overheat_boost).clamp(0.0, 1.0);
                let pressure_wait = snapshot
                    .top_processes
                    .iter()
                    .take(8)
                    .filter(|p| p.cpu_usage < 8.0 && p.memory_usage > 100 * 1024 * 1024)
                    .count() as f64
                    / 8.0_f64;
                let pressure_wait = pressure_wait.clamp(0.0, 1.0);

                let critical_patterns = critical_background_processes();
                let context_switch_burst = ctx_switch_times.len() >= 3;
                let context_switches_5min = ctx_switch_times.len() as u32;

                let dev_session_active = snapshot
                    .top_processes
                    .iter()
                    .any(|p| critical_patterns.iter().any(|pat| p.name.contains(pat)));
                let interactive_cpu_sum: f32 = snapshot
                    .top_processes
                    .iter()
                    .filter(|p| {
                        p.name.contains("WindowServer")
                            || p.name.contains("Google Chrome")
                            || p.name.contains("Antigravity")
                    })
                    .map(|p| p.cpu_usage)
                    .sum();
                let interactive_heavy = interactive_cpu_sum >= 60.0;

                // Populate real pressure metrics for observability.
                {
                    let mut metrics = state.metrics.lock_recover();
                    metrics.swap_used_bytes = snapshot.pressure.swap_used_bytes;
                    metrics.swap_delta_bps = snapshot.pressure.swap_delta_bytes_per_sec;
                    metrics.memory_pressure = snapshot.pressure.memory_pressure;
                    metrics.thermal_level = snapshot.pressure.thermal_level.clone();
                    {
                        let rs = state.reactor_status.lock_recover();
                        metrics.reactor_events_total = rs.events_total;
                        metrics.reactor_events_mem = rs.events_mem;
                        metrics.reactor_events_thermal = rs.events_thermal;
                        metrics.reactor_events_spawn = rs.events_spawn;
                        metrics.reactor_events_power = rs.events_power;
                        metrics.reactor_last_event_at = rs.last_event_at;
                        metrics.reactor_last_error = rs.last_error.clone();
                        metrics.reactor_mode = rs.mode.clone();
                        metrics.reactor_health = rs.health.clone();
                    }
                    metrics.dev_session_active = dev_session_active;
                    metrics.interactive_heavy = interactive_heavy;
                    metrics.context_switches_5min = context_switches_5min;
                    metrics.context_switch_burst = context_switch_burst;

                    // Resource interrupt (sentinel) metrics.
                    metrics.resource_interrupts_total =
                        state.resource_interrupt.total_fires.load(Ordering::Relaxed);
                    metrics.resource_interrupt_last_phase =
                        state.resource_interrupt.phase.load(Ordering::Relaxed);
                    metrics.resource_interrupt_active =
                        state.resource_interrupt.active.load(Ordering::Relaxed);
                    metrics.resource_interrupt_latency_us = state
                        .resource_interrupt
                        .last_latency_us
                        .load(Ordering::Relaxed);
                    metrics.resource_interrupt_processes_frozen = state
                        .resource_interrupt
                        .total_frozen
                        .load(Ordering::Relaxed);
                    metrics.resource_interrupt_processes_migrated = state
                        .resource_interrupt
                        .total_migrated
                        .load(Ordering::Relaxed);
                    metrics.resource_interrupt_recovery_count = state
                        .resource_interrupt
                        .total_recoveries
                        .load(Ordering::Relaxed);

                    // Foreground detection metrics.
                    metrics.foreground_app = match &fg_state {
                        ForegroundState::App(app) => Some(ForegroundAppInfo {
                            pid: app.pid,
                            name: app.name.clone(),
                            bundle_id: app.bundle_id.clone(),
                        }),
                        _ => None,
                    };
                    metrics.foreground_idle = foreground_idle;

                    // Energy tracking metrics.
                    let energy_summary = energy_tracker.session_summary();
                    metrics.energy_savings_wh = Some(energy_summary.estimated_savings_wh);
                    metrics.energy_co2_avoided_g = Some(energy_summary.estimated_co2_kg * 1000.0);
                    metrics.energy_package_wh = Some(energy_summary.total_package_wh);
                    metrics.energy_session_wh =
                        Some(energy_summary.total_cpu_wh + energy_summary.total_gpu_wh);
                    // Use cycle-level hardware snapshot for per-process power,
                    // falling back to smc_direct when IOKit returns None.
                    metrics.energy_cpu_watts = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.cpu_watts)
                        .map(|w| w as f64)
                        .or(last_smc.as_ref().and_then(|s| s.p_cluster_watts));
                    metrics.energy_gpu_watts = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.gpu_watts)
                        .map(|w| w as f64);
                    metrics.energy_package_watts = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.package_watts)
                        .map(|w| w as f64)
                        .or(last_smc.as_ref().and_then(|s| s.system_power_watts));
                    metrics.energy_top_consumers = energy_tracker
                        .top_consumers(5)
                        .into_iter()
                        .map(|e| EnergyConsumerInfo {
                            name: e.name,
                            current_watts: e.current_watts,
                            percentage: e.percentage_of_total,
                        })
                        .collect();

                    // Process tree metrics (informational).
                    metrics.process_tree_groups = process_tree.group_count();
                    metrics.process_tree_total = process_tree.len();

                    // IOReport hardware telemetry.
                    if let Some(ref ir) = last_ioreport {
                        metrics.ioreport_p_cluster_pct = ir.p_cluster_pct;
                        metrics.ioreport_e_cluster_pct = ir.e_cluster_pct;
                        metrics.ioreport_gpu_pct = ir.gpu_pct;
                        metrics.ioreport_ane_busy = ir.ane_busy;
                        metrics.ioreport_cpu_mw = ir.cpu_mw;
                        metrics.ioreport_total_watts = ir.total_watts();
                    }

                    // SMC direct metrics
                    if let Some(ref smc) = last_smc {
                        metrics.smc_system_power_watts = smc.system_power_watts;
                        metrics.smc_lid_closed = smc.lid_closed;
                        metrics.smc_charger_watts = smc.charger_watts;
                        metrics.smc_battery_tte_min = smc.battery_time_to_empty_min;
                        metrics.smc_cpu_temp_celsius = smc.cpu_temp_celsius;
                        metrics.smc_gpu_temp_celsius = smc.gpu_temp_celsius;
                        metrics.smc_battery_temp_celsius = smc.battery_temp_celsius;
                        metrics.smc_cpu_voltage = smc.cpu_voltage;
                        metrics.smc_p_cluster_watts = smc.p_cluster_watts;
                    }

                    // KPC IPC metric
                    if let Some(ref kpc) = kpc_snap {
                        metrics.kpc_ipc = kpc.ipc;
                    }

                    // Rosetta AOT state
                    metrics.rosetta_aot_active = rosetta_monitor.is_compiling();

                    // IOReport AMC bandwidth
                    if let Some(ref ir) = last_ioreport {
                        metrics.ioreport_amc_bandwidth_pct = ir.amc_bandwidth_pct;
                    }

                    // IOPMrootDomain thermal
                    if let Some(ref iopm) = iopm_snap {
                        metrics.iopm_thermal_warning = format!("{:?}", iopm.thermal_warning);
                        metrics.iopm_power_source = format!("{:?}", iopm.power_source);
                    }

                    // Per-process energy top consumer
                    if let Some(top) = energy_pid_results.first() {
                        metrics.energy_top_pid_name = top.name.clone();
                        metrics.energy_top_pid_mw = top.power_mw;
                    }

                    // Daemon self-IPC (thread_selfcounts syscall 186)
                    let _cycle_ipc = cycle_ipc_tracker.tick();
                    metrics.daemon_cycle_ipc = cycle_ipc_tracker.ema_ipc();
                }

                let (decide_interactive, decide_noise, decide_weights, outcome_baseline) = {
                    let policy = state.learned_policy.lock_recover();
                    (
                        policy.interactive_patterns.clone(),
                        policy.noise_patterns.clone(),
                        policy.pattern_weights.clone(),
                        outcome_tracker.calibrated_threshold(),
                    )
                };

                // Phase 3: Feature-based workload classification.
                let workload_mode: WorkloadMode = {
                    let ratios: Vec<f64> = snapshot
                        .top_processes
                        .iter()
                        .filter_map(|p| p.cpu_wall_ratio.map(|r| r as f64))
                        .collect();
                    let avg_cpu_wall_ratio = if ratios.is_empty() {
                        0.0
                    } else {
                        ratios.iter().sum::<f64>() / ratios.len() as f64
                    };
                    let build_tool_count = all_proc_names
                        .iter()
                        .filter(|n| BUILD_TOOLS.iter().any(|t| n.to_lowercase().contains(t)))
                        .count() as f64;
                    let gpu_watts = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.gpu_watts)
                        .unwrap_or(0.0);
                    let gpu_active = if gpu_watts > 2.0 { 1.0 } else { 0.0 };
                    let total_rss_gb = snapshot
                        .top_processes
                        .iter()
                        .map(|p| p.memory_usage as f64)
                        .sum::<f64>()
                        / (1024.0 * 1024.0 * 1024.0);
                    let interactive_count = snapshot
                        .top_processes
                        .iter()
                        .filter(|p| {
                            decide_interactive.iter().any(|pat| {
                                p.name
                                    .to_ascii_lowercase()
                                    .contains(&pat.to_ascii_lowercase())
                            })
                        })
                        .count() as f64;
                    let features = WorkloadFeatures {
                        avg_cpu_wall_ratio,
                        build_tool_count,
                        gpu_active,
                        total_rss_gb,
                        interactive_count,
                    };
                    let (mode, _confidence) = classify_workload_mode(&features);
                    mode
                };

                let mut governor = state.governor.lock_recover();
                let governor_decision = governor.evaluate(GovernorInput {
                    cpu_pressure: pressure_cpu,
                    ram_pressure: pressure_ram,
                    interactive_wait_ratio: pressure_wait,
                    reactor_event_weight: *reactor_weight,
                    thermal_constrained: matches!(
                        snapshot.pressure.thermal_level.as_str(),
                        "serious" | "critical"
                    ) || gpu_thermal_throttled,
                    dev_session_active,
                    interactive_heavy,
                    context_switch_burst,
                    workload_mode: Some(workload_mode),
                });
                if governor_decision.transition_reason.contains("floor") {
                    state.metrics.lock_recover().profile_floor_hits += 1;
                }
                let current_profile = governor_decision.effective_profile;
                write_governor_state(&governor_state_path, &governor);
                drop(governor);

                // Thresholds adaptativos: workload-aware via Phase 3 classifier.
                let mut overflow_thresholds = overflow_guard.thresholds(workload_mode);

                // Signal intelligence: Kalman + CUSUM + Entropy + Hazard + LV + MPC.
                let signal_digest = {
                    let cpu_vals: Vec<f64> = snapshot
                        .top_processes
                        .iter()
                        .map(|p| p.cpu_usage as f64)
                        .collect();
                    let mem_vals: Vec<f64> = snapshot
                        .top_processes
                        .iter()
                        .map(|p| p.memory_usage as f64)
                        .collect();
                    let (dom_name, dom_bytes) = snapshot
                        .top_processes
                        .iter()
                        .max_by_key(|p| p.memory_usage)
                        .map(|p| (p.name.as_str(), p.memory_usage))
                        .unwrap_or(("", 0));
                    let total_used: u64 =
                        snapshot.top_processes.iter().map(|p| p.memory_usage).sum();
                    let swap_ratio = if snapshot.pressure.swap_total_bytes > 0 {
                        snapshot.pressure.swap_used_bytes as f64
                            / snapshot.pressure.swap_total_bytes as f64
                    } else {
                        0.0
                    };
                    // Energy-aware routing: shift subsystem thresholds by battery/thermal.
                    signal_intel.set_energy_bias(
                        power_mgr.battery_status.percentage,
                        power_mgr.battery_status.is_charging,
                        thermal_emergency,
                    );
                    let _si_result = signal_intel.tick(
                        snapshot.pressure.memory_pressure,
                        snapshot.pressure.swap_delta_bytes_per_sec,
                        swap_ratio,
                        snapshot.pressure.memory_pressure, // compressor proxy
                        &cpu_vals,
                        &mem_vals,
                        dom_name,
                        dom_bytes,
                        total_used,
                        snapshot.memory.total_ram,
                        cycle_dt_secs,
                    );
                    _si_result
                };

                // v0.7.0: Mark memory scan available when pressure is in mid/high zone.
                // The actual scan runs lazily during freeze decision (cost-gated).
                // DBAD: build telemetry vector from signal digest and score.
                let signal_digest = {
                    let mut d = signal_digest;
                    if d.pressure_smooth >= 0.30 {
                        d.memory_scan_available = true;
                    }
                    // Darwin-Boltzmann anomaly scoring: feed signal digest into
                    // Hopfield memory + evolving SAE population for learned anomaly detection.
                    // Replaces hardcoded transformer_anomaly: 0.0.
                    use apollo_optimizer::engine::telemetry_logger::TelemetryVector;
                    let dom_share = {
                        let max_mem = snapshot.top_processes.iter()
                            .map(|p| p.memory_usage)
                            .max()
                            .unwrap_or(0) as f64;
                        let total = snapshot.memory.total_ram as f64;
                        if total > 0.0 { (max_mem / total) as f32 } else { 0.0 }
                    };
                    let thermal_score = match snapshot.pressure.thermal_level.as_str() {
                        "nominal" => 0.0f32,
                        "light" => 0.33,
                        "serious" => 0.66,
                        "critical" => 1.0,
                        _ => 0.0,
                    };
                    let cpu_total = snapshot.top_processes.iter()
                        .map(|p| p.cpu_usage)
                        .sum::<f32>() / 100.0;
                    let active_count = (snapshot.top_processes.len() as f32 / 200.0).min(1.0);
                    let tv = TelemetryVector {
                        pressure_smooth: d.pressure_smooth as f32,
                        pressure_velocity: d.pressure_velocity as f32,
                        pressure_predicted_5s: d.pressure_predicted_5s as f32,
                        swap_velocity_smooth: (d.swap_velocity_smooth as f32).clamp(-5.0, 5.0),
                        pressure_integral: d.pressure_integral as f32,
                        cusum_score: d.cusum_score as f32,
                        entropy_anomaly: d.entropy_anomaly as f32,
                        p_oom_30s: d.p_oom_30s as f32,
                        monopoly_risk: d.monopoly_risk as f32,
                        urgency: d.urgency as f32,
                        cpu_total: cpu_total.min(1.0),
                        compressor_ratio: snapshot.pressure.memory_pressure as f32,
                        dominant_share: dom_share,
                        latency_score: 0.0, // no perceptual latency sensor yet
                        active_proc_count: active_count,
                        thermal_score,
                    };
                    d.transformer_anomaly = darwin_anomaly.score(
                        tv.as_f32_slice(),
                        d.pressure_smooth as f32,
                    );
                    // Audit DBAD score every ~60 cycles or when anomaly detected.
                    if d.transformer_anomaly > 0.3 || cycle_count % 60 == 0 {
                        audit_log(&serde_json::json!({
                            "event": "dbad_score",
                            "score": (d.transformer_anomaly * 1000.0).round() / 1000.0,
                            "alpha": (darwin_anomaly.alpha() * 100.0).round() / 100.0,
                            "samples": darwin_anomaly.sample_count(),
                            "ready": darwin_anomaly.is_ready(),
                            "pressure": (d.pressure_smooth * 1000.0).round() / 1000.0,
                        }));
                    }
                    d
                };

                // Signal intelligence → reactor_weight boosting.
                // CUSUM regime shift: pressure drifting up significantly.
                if signal_digest.regime_shift_up {
                    *reactor_weight = (*reactor_weight + 0.3).min(1.0);
                }
                // High composite urgency: multiple signals converging on danger.
                if signal_digest.urgency > 0.7 {
                    *reactor_weight = (*reactor_weight + 0.2).min(1.0);
                }
                // Entropy anomaly: chaotic process distribution change.
                if signal_digest.entropy_anomaly > 2.0 {
                    *reactor_weight = (*reactor_weight + 0.15).min(1.0);
                }
                // Darwin-Boltzmann anomaly: learned pattern deviation.
                // Score > 0.5 means the system state deviates significantly from
                // the Hopfield memory + SAE ensemble's learned "normal" manifold.
                if signal_digest.transformer_anomaly > 0.5 {
                    *reactor_weight = (*reactor_weight + 0.2).min(1.0);
                }

                // Predictive agent: build context from existing signals and select intervention.
                // Feed Kalman-smoothed pressure instead of raw — cleaner signal for LinUCB.
                let agent_intervention = {
                    let prev_workload = state
                        .adaptive_governor
                        .lock_recover()
                        .last_ml_classification()
                        .workload;
                    let (hw_tp, hw_jt, hw_cl) = match &hw_features {
                        Some(f) => (f.throughput_mips, f.jitter_us, f.cache_latency_us),
                        None => (800.0, 50.0, 5000.0),
                    };
                    // Cable B: OutcomeTracker → PredictiveAgent context.
                    // low_value_ratio tells LinUCB how much of its effort is wasted.
                    let lv_ratio = {
                        let total = outcome_tracker.weights.len() as f64;
                        if total > 0.0 {
                            let threshold = outcome_tracker.calibrated_threshold();
                            let low = outcome_tracker.weights.values()
                                .filter(|w| w.is_low_value_vs_baseline(threshold))
                                .count() as f64;
                            low / total
                        } else {
                            0.0
                        }
                    };
                    let agent_ctx = AgentContext::build(
                        signal_digest.pressure_smooth, // Kalman-filtered instead of raw
                        swap_forecast.swap_trend,
                        swap_forecast.time_to_swap_critical,
                        hw_tp,
                        hw_jt,
                        hw_cl,
                        prev_workload,
                        hour_of_day,
                        *reactor_weight,
                        overflow_guard.history.threshold_offset,
                        outcome_tracker.overall_effectiveness(),
                        lv_ratio,
                    );
                    let linucb_choice = predictive_agent.select_action(&agent_ctx);

                    // ── Specialist accuracy feedback (Super Learner) ─────────────────
                    // Compare prev cycle's specialist predictions against observed outcome.
                    // A spike is a pressure rise of ≥0.08 over the previous cycle.
                    {
                        let pressure_spiked = signal_digest.pressure_smooth
                            >= prev_pressure_smooth + 0.08;
                        // Hazard: predicted high risk when p_oom_30s > 0.30 last cycle.
                        // We can't replay last cycle's p_oom value, so we approximate:
                        // hazard fired (voted) iff prev_pressure was already elevated.
                        let hazard_predicted_high = prev_pressure_smooth > 0.40;
                        let hazard_correct = (hazard_predicted_high && pressure_spiked)
                            || (!hazard_predicted_high && !pressure_spiked);
                        specialist_accuracy.update(specialist::HAZARD, hazard_correct);

                        // Monopoly: predicted high when monopoly_risk > 0.5.
                        // Proxy: prev pressure was in monopoly range (>0.55).
                        let monopoly_predicted_high = prev_pressure_smooth > 0.55;
                        let monopoly_correct = (monopoly_predicted_high && pressure_spiked)
                            || (!monopoly_predicted_high && !pressure_spiked);
                        specialist_accuracy.update(specialist::MONOPOLY, monopoly_correct);

                        // Kalman: predicted spike when pressure_predicted_5s > 0.85.
                        // Proxy: prev pressure was high enough to trigger the specialist.
                        let kalman_predicted_high = prev_pressure_smooth > 0.70;
                        let kalman_correct = (kalman_predicted_high && pressure_spiked)
                            || (!kalman_predicted_high && !pressure_spiked);
                        specialist_accuracy.update(specialist::KALMAN, kalman_correct);

                        // LinUCB: voted for action. Correct if pressure improved or stayed calm.
                        let linucb_predicted_intervention = linucb_choice != Intervention::Observe;
                        let linucb_correct = (linucb_predicted_intervention && pressure_spiked)
                            || (!linucb_predicted_intervention && !pressure_spiked);
                        specialist_accuracy.update(specialist::LINUCB, linucb_correct);
                    }
                    // Save current pressure for next cycle's accuracy feedback.
                    prev_pressure_smooth = signal_digest.pressure_smooth;

                    // ── Specialist voting: weighted ensemble replaces override chain ──
                    use apollo_optimizer::engine::predictive_agent::{SpecialistVote, tally_votes};
                    let mut votes = vec![
                        // LinUCB: primary agent, moderate confidence.
                        SpecialistVote {
                            name: "linucb",
                            intervention: linucb_choice,
                            confidence: 0.5,
                        },
                    ];

                    // Hazard specialist: high P(OOM) → use MPC recommendation.
                    if signal_digest.p_oom_30s > 0.30 {
                        votes.push(SpecialistVote {
                            name: "hazard",
                            intervention: Intervention::from_index(signal_digest.mpc_recommendation),
                            confidence: signal_digest.p_oom_30s.min(1.0),
                        });
                    }

                    // Monopoly specialist: one process hogging RAM → throttle noise.
                    if signal_digest.monopoly_risk > 0.5 {
                        votes.push(SpecialistVote {
                            name: "monopoly",
                            intervention: Intervention::PreThrottleNoise,
                            confidence: signal_digest.monopoly_risk.min(1.0) * 0.7,
                        });
                    }

                    // Kalman specialist: predicted pressure spike → tighten.
                    if signal_digest.pressure_predicted_5s > 0.85 {
                        votes.push(SpecialistVote {
                            name: "kalman",
                            intervention: Intervention::TightenThresholds,
                            confidence: (signal_digest.pressure_predicted_5s - 0.85).min(0.15) / 0.15 * 0.8,
                        });
                    }

                    let vote_result = tally_votes(&votes);
                    let intervention = vote_result.intervention;

                    // Cable: had_disagreement → conservative safety route.
                    // When specialists disagree AND the winning score is weak (<0.4),
                    // the signal is ambiguous. Fall back to Observe instead of risking
                    // a wrong aggressive action. Only override if not in survival mode.
                    let intervention = if vote_result.had_disagreement {
                        audit_log(&serde_json::json!({
                            "t": Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                            "event": "specialist_disagreement",
                            "winner": format!("{:?}", intervention),
                            "score": (vote_result.winning_score * 100.0).round() / 100.0,
                            "n_votes": votes.len(),
                            "pressure": (signal_digest.pressure_smooth * 1000.0).round() / 1000.0,
                        }));
                        if vote_result.winning_score < 0.4 && signal_digest.pressure_smooth < 0.80 {
                            // Low confidence + not critical pressure → play it safe.
                            Intervention::Observe
                        } else {
                            intervention
                        }
                    } else {
                        intervention
                    };

                    // Apply threshold tightening if selected.
                    overflow_thresholds = predictive_agent.adjust_thresholds(overflow_thresholds);

                    // SuggestAggressive: set a 5-minute manual override to aggressive profile.
                    if intervention == Intervention::SuggestAggressive {
                        let mut gov = state.governor.lock_recover();
                        if gov.manual_override.is_none() {
                            gov.set_manual_override(
                                OptimizationProfile::AggressiveRoot,
                                5,
                                "predictive-agent: proactive pressure mitigation".to_string(),
                            );
                        }
                    }

                    intervention
                };

                // Build behavior-interactive PID set from usage model EMA data.
                // Processes with sustained low cpu_wall_ratio are I/O-bound (interactive).
                let behavior_interactive_pids: HashSet<u32> = {
                    let model = state.usage_model.lock_recover();
                    let interactive_names: HashSet<&str> = model
                        .entries()
                        .iter()
                        .filter(|(_, entry)| {
                            apollo_optimizer::engine::usage_model::is_behavior_interactive(entry)
                        })
                        .map(|(name, _)| name.as_str())
                        .collect();
                    // Map interactive names back to running PIDs.
                    snapshot
                        .top_processes
                        .iter()
                        .filter(|p| interactive_names.contains(p.name.as_str()))
                        .map(|p| p.pid)
                        .collect()
                };

                // PID integral adjustment: if pressure has been chronically above target,
                // lower thresholds proportionally. Ki = 0.02 → 1.0 pressure-second of
                // integral error lowers thresholds by 0.02 (2 percentage points).
                // Clamped to max 5pp reduction from integral alone.
                if signal_digest.pressure_integral > 0.5 {
                    let ki = 0.02;
                    let integral_adjustment = (signal_digest.pressure_integral * ki).min(0.05);
                    overflow_thresholds.bg_pressure -= integral_adjustment;
                    overflow_thresholds.critical_pressure -= integral_adjustment;
                    overflow_thresholds.extreme_pressure -= integral_adjustment;
                }

                // Holt-Winters seasonal forecasting: accumulate pressure samples,
                // observe once per hour, and use forecast to proactively lower thresholds
                // before predicted high-pressure periods.
                {
                    hw_pressure_accum += snapshot.pressure.memory_pressure;
                    hw_pressure_count += 1;

                    // When the hour changes, feed the average pressure to Holt-Winters.
                    if hw_last_hour != Some(hour_of_day) {
                        if let Some(prev_hour) = hw_last_hour {
                            if hw_pressure_count > 0 {
                                let avg = hw_pressure_accum / hw_pressure_count as f64;
                                holt_winters.observe(prev_hour, avg);
                            }
                        }
                        hw_last_hour = Some(hour_of_day);
                        hw_pressure_accum = 0.0;
                        hw_pressure_count = 0;
                    }

                    // Forecast: if next hour's predicted pressure is high, tighten now.
                    let (forecast_1h, confidence) = holt_winters.forecast(hour_of_day, 1);
                    if confidence > 0.3 && forecast_1h > 0.75 {
                        // Scale adjustment by confidence and how high the forecast is.
                        let hw_adjustment = (forecast_1h - 0.75) * confidence * 0.10;
                        let hw_adjustment = hw_adjustment.min(0.04); // Max 4pp from forecast

                        // Cross-reference with UserProfile: if the next hour is typically
                        // a build session, apply extra tightening (builds spike fast).
                        let next_hour = (hour_of_day + 1) % 24;
                        let next_workload = {
                            let gov = state.adaptive_governor.lock_recover();
                            gov.user_profile.likely_workload_at_hour(next_hour)
                        };
                        let workload_multiplier = match next_workload {
                            apollo_optimizer::engine::user_profile::WorkloadType::Coding => 1.5,
                            apollo_optimizer::engine::user_profile::WorkloadType::VideoEdit => 1.3,
                            _ => 1.0,
                        };

                        let final_adjustment = (hw_adjustment * workload_multiplier).min(0.06);
                        overflow_thresholds.bg_pressure -= final_adjustment;
                        overflow_thresholds.critical_pressure -= final_adjustment;
                        overflow_thresholds.extreme_pressure -= final_adjustment;
                    }
                }

                // Perceptual latency monitor: composite score from existing signals.
                // If UI responsiveness is degraded, boost reactor_weight to trigger
                // faster/more aggressive scheduling decisions.
                let _latency_score_val = {
                    let fg_cpu = foreground_pid
                        .and_then(|pid| {
                            proc_snaps
                                .iter()
                                .find(|s| s.pid == pid)
                                .map(|s| s.cpu_percent as f64)
                        })
                        .unwrap_or(0.0);
                    let fg_csw = foreground_pid
                        .and_then(|pid| proc_taskinfo::get_task_info(pid))
                        .map(|ti| {
                            // Rough csw/s: divide cumulative by uptime (capped).
                            let uptime = proc_snaps
                                .iter()
                                .find(|s| s.pid == foreground_pid.unwrap_or(0))
                                .map(|s| s.process_uptime_secs.max(1))
                                .unwrap_or(1);
                            ti.context_switches as f64 / uptime as f64
                        })
                        .unwrap_or(0.0);
                    let latency = latency_monitor::compute_latency(&LatencySignals {
                        jitter_us: jitter_us as f64,
                        windowserver_cpu: windowserver_cpu(&snapshot) as f64,
                        foreground_cpu: fg_cpu,
                        foreground_csw_per_sec: fg_csw,
                        has_foreground: foreground_pid.is_some(),
                    });
                    if latency.needs_boost {
                        // Elevate reactor weight → faster tick + more aggressive decisions.
                        *reactor_weight = (*reactor_weight + 0.25).min(1.0);
                    }
                    latency.score
                };

                // Adaptive Page Reclaim: purge file cache when pressure is building
                // but before the kernel is forced to evict reactively (which causes stalls).
                // Jiang & Zhang 2005 — proactive beats reactive by 20-40%.
                // Runs every 10 cycles (~5s) to avoid vm_stat overhead every cycle.
                if cycle_count % 10 == 0 {
                    let freed = page_reclaim.tick(
                        (snapshot.pressure.memory_pressure + battery_pressure_boost(&power_mgr) + thermal_pressure_boost).clamp(0.0, 1.0),
                        display_turbo.is_turbo_active() || thermal_action.phase >= apollo_optimizer::engine::thermal_bailout::CoolingPhase::Phase2Moderate,
                        foreground_idle,
                    );
                    if freed > 0 {
                        state.metrics.lock_recover().paging_hints_applied += 1;
                    }
                }

                // Telemetry logger + Transformer scoring disabled.
                // Classical signal stack (Kalman/CUSUM/Hazard/HW/Markov/Temporal)
                // already covers anomaly detection and prediction.
                // Code preserved in telemetry_logger.rs / transformer_predictor.rs.

                // ── Habituation: update per-process state tracking ─────────
                // Inspired by Thompson & Spencer 1966 / memoria-core habituation.rs.
                // Processes whose CPU and RSS bucket are unchanged for ≥5 cycles
                // are skipped in decide_actions (their last action is maintained).
                // Dishabituation: any bucket change resets the counter.
                let habituated_pids: HashSet<u32> = {
                    let mut hab_set = HashSet::new();
                    for (pid, process) in collector.system().processes() {
                        let pid_u32 = pid.as_u32();
                        let cpu_bucket = (process.cpu_usage() / 5.0) as u8;
                        let rss_bucket = (process.memory() / (50 * 1024 * 1024)) as u8;
                        match habituation_map.get_mut(&pid_u32) {
                            Some(entry) => {
                                if entry.0 == cpu_bucket && entry.1 == rss_bucket {
                                    entry.2 += 1; // unchanged
                                    if entry.2 >= HABITUATION_THRESHOLD {
                                        hab_set.insert(pid_u32);
                                    }
                                } else {
                                    // Dishabituation: state changed.
                                    *entry = (cpu_bucket, rss_bucket, 0);
                                }
                            }
                            None => {
                                habituation_map.insert(pid_u32, (cpu_bucket, rss_bucket, 0));
                            }
                        }
                    }
                    // GC dead PIDs every 100 cycles.
                    if cycle_count % 100 == 0 {
                        let live: HashSet<u32> = collector.system().processes()
                            .keys().map(|p| p.as_u32()).collect();
                        habituation_map.retain(|pid, _| live.contains(pid));
                    }
                    hab_set
                };

                let decision = {
                    let mut qos = state.mach_qos.lock_recover();
                    decide_actions(
                        &snapshot,
                        collector.system(),
                        current_profile,
                        latency_target,
                        *reactor_weight,
                        &decide_interactive,
                        &decide_noise,
                        overflow_thresholds,
                        Some(&mut qos),
                        &decide_weights,
                        outcome_baseline,
                        &behavior_interactive_pids,
                        &ipc_hints,
                        &outcome_tracker.hop_groups,
                        &habituated_pids,
                        &causal_graph.confidence_map(),
                    )
                };
                *state.last_blockers.lock_recover() = decision.blockers.clone();
                *state.thermal_state.lock_recover() = context_to_thermal(decision.context);

                // Propagar skips de OutcomeTracker a top_skipped_processes para observabilidad.
                {
                    let mut metrics = state.metrics.lock_recover();
                    for name in &decision.low_value_skipped {
                        if metrics.top_skipped_processes.len() < 12
                            && !metrics.top_skipped_processes.contains(name)
                        {
                            metrics.top_skipped_processes.push(name.clone());
                        }
                    }
                }

                // Apply any locally learned policy patterns (and keep them even after LLM is disabled).
                let mut actions = decision.actions;
                {
                    let policy = state.learned_policy.lock_recover().clone();
                    actions = apply_learned_policy_actions(&snapshot, &policy, actions);
                }

                // Predictive agent: inject soft actions for PreThrottleNoise / ProactivePurge.
                match agent_intervention {
                    Intervention::PreThrottleNoise => {
                        // Renice top 3 noise processes (soft throttle, no SIGSTOP).
                        let noise_pats = state.learned_policy.lock_recover().noise_patterns.clone();
                        let mut noise_procs: Vec<_> = snapshot
                            .top_processes
                            .iter()
                            .filter(|p| noise_pats.iter().any(|pat| p.name.contains(pat.as_str())))
                            .collect();
                        noise_procs.sort_by(|a, b| {
                            b.cpu_usage
                                .partial_cmp(&a.cpu_usage)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        for proc in noise_procs.iter().take(3) {
                            actions.push(RootAction::ThrottleProcess {
                                pid: proc.pid as u32,
                                name: proc.name.clone(),
                                aggressive: false,
                                reason: "predictive-agent: pre-throttle noise".to_string(),
                                start_sec: 0,
                                start_usec: 0,
                            });
                        }
                    }
                    Intervention::ProactivePurge => {
                        // Send paging hints to top 3 background processes by RSS.
                        // SetMemorystatus with priority -1 asks the process to release caches
                        // voluntarily — no freeze, no kill. Passes through safety in execute_actions.
                        let interactive_pats = decide_interactive.clone();
                        let protected_pats = state
                            .learned_policy
                            .lock_recover()
                            .protected_patterns
                            .clone();
                        let mut bg_procs: Vec<_> = snapshot
                            .top_processes
                            .iter()
                            .filter(|p| {
                                !interactive_pats
                                    .iter()
                                    .any(|pat| p.name.contains(pat.as_str()))
                                    && !protected_pats
                                        .iter()
                                        .any(|pat| p.name.contains(pat.as_str()))
                                    && p.memory_usage > 50 * 1024 * 1024 // >50 MB RSS
                            })
                            .collect();
                        bg_procs.sort_by(|a, b| b.memory_usage.cmp(&a.memory_usage));
                        for proc in bg_procs.iter().take(3) {
                            actions.push(RootAction::SetMemorystatus {
                                pid: proc.pid,
                                priority: -1,
                                reason: "predictive-agent: proactive purge hint".to_string(),
                            });
                        }
                    }
                    _ => {} // Observe, TightenThresholds, SuggestAggressive handled above
                }

                // Direct paging hints: when pressure > 0.60, hint top 3 background
                // memory consumers. Safe (voluntary cache release, no freeze/kill).
                // Rate-limited by safety module's max_paging_hints_per_cycle.
                let already_has_hints = actions.iter().any(|a| matches!(a, RootAction::SetMemorystatus { .. }));
                if signal_digest.pressure_smooth >= 0.60
                    && !already_has_hints
                {
                    let interactive_pats = decide_interactive.clone();
                    let protected_pats = state
                        .learned_policy
                        .lock_recover()
                        .protected_patterns
                        .clone();
                    // Use proc_snaps (full process list) not top_processes (top 10 by CPU).
                    // Only skip core interactive apps — paging hints are gentle (voluntary
                    // cache release), so we use a tighter filter than freeze/throttle.
                    let hard_protected = apollo_optimizer::engine::safety::protected_processes();
                    let mut bg_procs: Vec<_> = proc_snaps
                        .iter()
                        .filter(|p| {
                            // Skip system-critical processes and self.
                            !hard_protected.iter().any(|hp| p.name.contains(hp))
                                && p.rss_bytes > 80 * 1024 * 1024 // >80 MB RSS
                                && p.pid != std::process::id()
                                && !p.has_gui_window
                                // Skip foreground app
                                && foreground_app.as_ref().map(|fg| p.name != *fg).unwrap_or(true)
                                // Skip processes with recent interaction (<60s)
                                && p.secs_since_user_interaction > 60
                        })
                        .collect();
                    bg_procs.sort_by(|a, b| b.rss_bytes.cmp(&a.rss_bytes));
                    for proc in bg_procs.iter().take(3) {
                        actions.push(RootAction::SetMemorystatus {
                            pid: proc.pid,
                            priority: -1,
                            reason: format!(
                                "pressure-driven hint (p={:.0}%): {} ({}MB)",
                                signal_digest.pressure_smooth * 100.0,
                                proc.name,
                                proc.rss_bytes / 1024 / 1024,
                            ),
                        });
                    }
                }

                // Heuristic pass: AdaptiveGovernor
                // Pass hw_features (sampled every 5 cycles) for Bayesian fusion + online learning.
                let heuristic_decisions = {
                    let mut gov = state.adaptive_governor.lock_recover();
                    gov.decide_all_with_hw(
                        &proc_snaps,
                        &hunt_snaps,
                        foreground_app.as_deref(),
                        &all_proc_names,
                        hour_of_day,
                        hw_features,
                    )
                };

                // Build critical_pids set for heuristic merge.
                //
                // Infrastructure (docker, postgres, redis) → always protected.
                // Dev runtimes (python, node, java, go, nginx) → protected only
                // when behaviorally active (Android LMK + TMO ASPLOS'22 model).
                // Score compared against system pressure: as memory stress rises,
                // only truly active dev runtimes keep their exemption.
                let heuristic_critical_pids: HashSet<u32> = {
                    let sys = collector.system();
                    let infra_pats = infrastructure_processes();
                    let protected_pats = protected_processes();
                    let policy_protected = state
                        .learned_policy
                        .lock_recover()
                        .protected_patterns
                        .clone();
                    let pressure = signal_digest.pressure_smooth;
                    let total_ram = apollo_optimizer::engine::sysctl_direct::read_u64("hw.memsize")
                        .unwrap_or(8 * 1024 * 1024 * 1024);
                    let mut cpids: HashSet<u32> = HashSet::new();
                    let mut bps_eval = 0u64;
                    let mut bps_prot = 0u64;
                    let mut bps_dem = 0u64;
                    let mut bps_min = f64::MAX;
                    let mut bps_min_name = String::new();
                    for (pid, process) in sys.processes() {
                        let name = process.name().to_string();
                        // Always-protected: system essentials + infrastructure + learned
                        if protected_pats.iter().any(|p| name.contains(p))
                            || infra_pats.iter().any(|p| name.contains(p))
                            || policy_protected.iter().any(|p| name.contains(p.as_str()))
                        {
                            cpids.insert(pid.as_u32());
                            continue;
                        }
                        // Dev runtimes: behavioral gate — protection earned, not given.
                        if matches_dev_runtime(&name) {
                            let pid_u32 = pid.as_u32();
                            // Prefer enriched ProcessSnapshot if available;
                            // fall back to raw sysinfo data for processes not in top-N.
                            let (cpu, wakeups, net, gui, idle_s, rss) =
                                if let Some(snap) = proc_snaps.iter().find(|s| s.pid == pid_u32) {
                                    (snap.cpu_percent, snap.wakeups_per_sec, snap.has_network,
                                     snap.has_gui_window, snap.secs_since_user_interaction, snap.rss_bytes)
                                } else {
                                    // Fallback: sysinfo process — limited signals but real RSS.
                                    (process.cpu_usage(), 0.0, false, false, 3600, process.memory())
                                };
                            let raw_score = behavioral_protection_score(
                                cpu, wakeups, net, gui, idle_s, rss, total_ram,
                            );
                            // Cable 5: process_relevance() → modulate BPS with user profile.
                            // If the user actively uses this process (relevance > 0), boost
                            // its behavioral score. If irrelevant (0.0), no change.
                            // This means a dev runtime the user has interacted with recently
                            // gets a relevance bonus, while one that's been stale loses it.
                            let relevance = {
                                let gov = state.adaptive_governor.lock_recover();
                                gov.user_profile.process_relevance(&name)
                            };
                            // Boost: relevance 1.0 adds up to +0.15 to score, relevance 0.0 adds 0.
                            let score = raw_score + (relevance as f64 * 0.15);
                            bps_eval += 1;
                            let protected = score >= pressure as f64;
                            if score < bps_min {
                                bps_min = score;
                                bps_min_name = format!("{}({})", name, pid_u32);
                            }
                            audit_log(&serde_json::json!({
                                "t": Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                                "event": "bps_eval",
                                "pid": pid_u32,
                                "name": name,
                                "score": (score * 10000.0).round() / 10000.0,
                                "raw_score": (raw_score * 10000.0).round() / 10000.0,
                                "relevance": (relevance * 100.0).round() / 100.0,
                                "pressure": (pressure * 1000.0).round() / 1000.0,
                                "protected": protected,
                                "cpu": cpu,
                                "wakeups": wakeups,
                                "net": net,
                                "gui": gui,
                                "idle_s": idle_s,
                                "rss_mb": rss / 1024 / 1024,
                            }));
                            if protected {
                                bps_prot += 1;
                                cpids.insert(pid_u32);
                            } else {
                                bps_dem += 1;
                            }
                        }
                    }
                    {
                        let mut m = state.metrics.lock_recover();
                        m.bps_evaluated += bps_eval;
                        m.bps_protected += bps_prot;
                        m.bps_demoted += bps_dem;
                        if bps_min < f64::MAX {
                            m.bps_min_score = bps_min;
                            m.bps_min_score_name = bps_min_name;
                        }
                    }
                    // AMX/ML workloads: never throttle/freeze ML inference processes.
                    cpids.extend(amx_detector::ml_protected_pids());
                    cpids
                };

                // Convert heuristic decisions to RootActions and merge
                let (heuristic_actions, heuristic_stats) = convert_and_merge_heuristic_decisions(
                    &heuristic_decisions,
                    &actions,
                    &heuristic_critical_pids,
                );
                // Cable 2: query_similar() → skip throttles that experience says won't work.
                // If we have ≥3 records of throttling process X at similar pressure and it
                // never helped (avg_drop ≤ 0), skip wasting the action budget on it.
                let current_pressure = snapshot.pressure.memory_pressure;
                let heuristic_actions: Vec<RootAction> = heuristic_actions.into_iter().filter(|a| {
                    if let RootAction::ThrottleProcess { ref name, .. } = a {
                        if let Some((avg_drop, confidence)) = outcome_tracker.experience.query_similar(name, current_pressure) {
                            if confidence >= 0.5 && avg_drop <= 0.0 {
                                // Experience says throttling this process at this pressure
                                // has never reduced pressure. Skip it.
                                return false;
                            }
                        }
                    }
                    true
                }).collect();
                actions.extend(heuristic_actions);

                // Cable: stale_apps() → nominate stale background apps as freeze candidates.
                // When pressure is elevated, apps the user hasn't interacted with for >30min
                // are prime freeze targets — they're consuming RAM without doing useful work.
                // Only nominate non-foreground, non-critical, non-already-acting processes.
                if signal_digest.pressure_smooth >= 0.50 {
                    let existing_pids: HashSet<u32> = actions.iter().filter_map(|a| match a {
                        RootAction::FreezeProcess { pid, .. }
                        | RootAction::ThrottleProcess { pid, .. }
                        | RootAction::BoostProcess { pid, .. } => Some(*pid),
                        _ => None,
                    }).collect();
                    let stale_names = {
                        let gov = state.adaptive_governor.lock_recover();
                        let running: Vec<&str> = all_proc_names.iter().copied().collect();
                        gov.user_profile.stale_apps(&running, 1800) // 30 min threshold
                    };
                    let sys = collector.system();
                    for (pid, process) in sys.processes() {
                        let pid_u32 = pid.as_u32();
                        let name = process.name().to_string();
                        if !stale_names.contains(&name) { continue; }
                        if Some(pid_u32) == foreground_pid { continue; }
                        if heuristic_critical_pids.contains(&pid_u32) { continue; }
                        if existing_pids.contains(&pid_u32) { continue; }
                        // Only freeze if using meaningful memory (>50MB RSS).
                        if process.memory() < 50 * 1024 * 1024 { continue; }
                        let (ss, su) = pid_start_time(pid_u32);
                        actions.push(RootAction::FreezeProcess {
                            pid: pid_u32,
                            name: name.clone(),
                            reason: format!("stale-app: no user interaction for >30min, rss={}MB",
                                process.memory() / 1024 / 1024),
                            start_sec: ss,
                            start_usec: su,
                        });
                    }
                }

                // Survival Mode: active when memory pressure is critical or swap is thrashing.
                // swap_delta_bps > 1MB/s means we're actively writing to swap (thrashing).
                // Survival Mode: critical memory pressure or swap thrashing.
                // p_oom_30s amplifies: if SI predicts OOM with high confidence AND
                // pressure is already elevated (≥0.70), escalate to survival.
                // Requires warmup (≥5 cycles) to avoid stale persisted p_oom values.
                let p_oom_escalation = cycle_count > 5
                    && signal_digest.p_oom_30s > 0.80
                    && snapshot.pressure.memory_pressure >= 0.70;
                let survival_mode = snapshot.pressure.memory_pressure > 0.85
                    || snapshot.pressure.swap_delta_bytes_per_sec > 1_000_000.0
                    || p_oom_escalation;

                // Overflow guard: only record as overflow when there is real memory
                // pressure (≥ 0.60).  Swap storms at low pressure (36-42%) were
                // poisoning the guard with false positives, keeping thresholds
                // permanently at the floor and making Apollo overly aggressive.
                //
                // survival_mode still gates aggressive actions (jetsam kill,
                // freeze recovery) regardless of this gate — we just don't let
                // low-pressure swap storms train the adaptive thresholds.
                let real_overflow = survival_mode && snapshot.pressure.memory_pressure >= 0.60;
                if real_overflow {
                    let heavy: Vec<String> = snapshot
                        .top_processes
                        .iter()
                        .filter(|p| p.name != "apollo-optimizerd")
                        .take(8)
                        .map(|p| p.name.clone())
                        .collect();
                    overflow_guard.record_event(
                        snapshot.pressure.memory_pressure,
                        snapshot.pressure.swap_delta_bytes_per_sec,
                        &heavy,
                        "survival-mode",
                        snapshot.pressure.compressor_pressure,
                    );
                    let sr = if snapshot.pressure.swap_total_bytes > 0 {
                        snapshot.pressure.swap_used_bytes as f64
                            / snapshot.pressure.swap_total_bytes as f64
                    } else {
                        0.0
                    };
                    signal_intel.record_overflow(
                        snapshot.pressure.memory_pressure,
                        sr,
                        snapshot.pressure.memory_pressure,
                        1.0,
                    );
                }
                // Decaimiento gradual: si el sistema está en calma, relajar thresholds.
                overflow_guard.tick_decay(
                    snapshot.pressure.memory_pressure,
                    snapshot.pressure.compressor_pressure,
                );

                // ── Neuromodulator: bio-inspired parameter modulation ────────
                {
                    let overflow_occurred = overflow_guard.history.total_overflows > 0;
                    let neuro_signals = NeuroSignals {
                        pressure_drop: signal_digest.pressure_smooth as f64 * -1.0
                            * signal_digest.pressure_velocity,
                        outcome_penalty: outcome_tracker.rl_penalty(),
                        overflow_occurred,
                        urgency: signal_digest.urgency,
                        regime_shift_up: signal_digest.regime_shift_up,
                        pressure_velocity: signal_digest.pressure_velocity,
                        thermal_emergency: thermal_action.phase
                            >= apollo_optimizer::engine::thermal_bailout::CoolingPhase::Phase2Moderate,
                        pressure_smooth: signal_digest.pressure_smooth as f64,
                        regime_shift_down: signal_digest.regime_shift_down,
                        process_count: collector.system().processes().len(),
                        entropy_anomaly: signal_digest.entropy_anomaly as f64,
                        rl_exploring: overflow_guard.rl_agent.as_ref()
                            .map_or(false, |rl| rl.total_ticks() < 200),
                    };
                    neuromod.tick(&neuro_signals);

                    // Push derived params to subsystems.
                    if let Some(rl) = &mut overflow_guard.rl_agent {
                        rl.neuro_alpha_mult = neuromod.alpha_multiplier;
                        rl.neuro_epsilon_bonus = neuromod.epsilon_bonus;
                        rl.dyna_steps = neuromod.dyna_steps;
                    }
                    signal_intel.neuro_serotonin_shift = neuromod.serotonin_shift;
                }

                // ProcessRecoveryManager: freeze (or kill in survival mode) confirmed leakers.
                let recovery_targets = proc_recovery.get_recovery_targets();
                for target in &recovery_targets {
                    if heuristic_critical_pids.contains(&target.pid) {
                        continue;
                    }
                    // Jetsam Kill: under critical pressure, kill confirmed long-running leakers
                    // instead of freezing. Requires: survival_mode + rss > 200MB + 2+ attempts.
                    if survival_mode
                        && target.rss_bytes > 200 * 1024 * 1024
                        && target.recovery_attempts >= 2
                        && target.pid > 1 // never signal init/kernel/self
                    {
                        if unsafe { libc::kill(target.pid as i32, 0) } == 0 {
                            unsafe {
                                libc::kill(target.pid as i32, libc::SIGKILL);
                            }
                            proc_recovery.record_kill_attempt(target.pid);
                            {
                                let mut m = state.metrics.lock_recover();
                                m.kills_applied += 1;
                                m.survival_mode_activations += 1;
                            }
                        }
                    } else {
                        let (ss, su) = pid_start_time(target.pid);
                        actions.push(RootAction::FreezeProcess {
                            pid: target.pid,
                            name: target.name.clone(),
                            reason: format!(
                                "memory-leak recovery: prob={:.2} rss={}MB attempts={}",
                                target.leak_probability,
                                target.rss_bytes / 1024 / 1024,
                                target.recovery_attempts,
                            ),
                            start_sec: ss,
                            start_usec: su,
                        });
                        proc_recovery.record_kill_attempt(target.pid);
                    }
                }

                // ── Feature 5: Wakeup Budget Enforcer ───────────────────────
                // Upgrade from ThrottleProcess to App Nap for wakeup offenders.
                // App Nap suppresses CPU + timers + I/O without SIGSTOP artifacts.
                // Processes that calm down (storm cleared) are released automatically.
                let storms = wake_storm.detect_storms();
                {
                    let storm_pids: std::collections::HashSet<u32> =
                        storms.iter().map(|s| s.pid).collect();
                    let mut qos = state.mach_qos.lock_recover();

                    // App-Nap new offenders.
                    for storm in &storms {
                        if !heuristic_critical_pids.contains(&storm.pid)
                            && Some(storm.pid) != foreground_pid
                        {
                            qos.set_app_nap(storm.pid, true);
                        }
                    }

                    // Release App Nap for pids that are no longer in a storm.
                    // (gc_dead_pids handles dead pids; this handles calmed pids)
                    let app_napped_snapshot: Vec<u32> = qos
                        .current_tier_keys()
                        .iter()
                        .filter(|(pid, _)| qos.is_app_napped(*pid))
                        .map(|(pid, _)| *pid)
                        .collect();
                    for pid in app_napped_snapshot {
                        if !storm_pids.contains(&pid) {
                            qos.set_app_nap(pid, false);
                        }
                    }
                }

                // ── Feature 2 + 4: App Nap for LLM mode and post-wake window ──
                // During LLM inference: App-Nap all non-foreground non-essential.
                // During wake suppression: same, to give foreground first crack.
                if llm_active || in_wake_suppression {
                    let protected_pats = protected_processes();
                    let policy_protected = state
                        .learned_policy
                        .lock_recover()
                        .protected_patterns
                        .clone();
                    let mut qos = state.mach_qos.lock_recover();
                    for (pid, process) in collector.system().processes() {
                        let pid_u32 = pid.as_u32();
                        let name = process.name();
                        if Some(pid_u32) == foreground_pid
                            || heuristic_critical_pids.contains(&pid_u32)
                            || protected_pats.iter().any(|p| name.contains(p))
                            || policy_protected.iter().any(|p| name.contains(p.as_str()))
                            || name == "apollo-optimizerd"
                        {
                            // Foreground and protected: ensure NOT app-napped.
                            if qos.is_app_napped(pid_u32) {
                                qos.set_app_nap(pid_u32, false);
                            }
                            continue;
                        }
                        // Skip if already app-napped (dedup).
                        if !qos.is_app_napped(pid_u32) {
                            qos.set_app_nap(pid_u32, true);
                        }
                    }
                } else if !in_wake_suppression && !llm_active {
                    // Neither LLM nor wake: release any LLM/wake App Naps that
                    // aren't also wake-storm offenders.
                    let storm_pids: std::collections::HashSet<u32> =
                        storms.iter().map(|s| s.pid).collect();
                    let mut qos = state.mach_qos.lock_recover();
                    let app_napped: Vec<u32> = qos
                        .current_tier_keys()
                        .iter()
                        .filter(|(pid, _)| qos.is_app_napped(*pid) && !storm_pids.contains(pid))
                        .map(|(pid, _)| *pid)
                        .collect();
                    for pid in app_napped {
                        qos.set_app_nap(pid, false);
                    }
                }

                // Paging hints: targeted non-fatal memory pressure to idle hoarders.
                // Uses memorystatus_control warn limit (non-fatal memlimit_inactive)
                // to send DISPATCH_SOURCE_TYPE_MEMORYPRESSURE to specific processes —
                // much more surgical than system-wide vm_pressure_notify().
                // Coalition API augments the foreground family beyond heuristic name-matching:
                // browser XPC helpers and GPU processes share the foreground coalition.
                let mem_pressure = snapshot.pressure.memory_pressure;
                let swap_active = snapshot.pressure.swap_used_bytes > 256 * 1024 * 1024;
                if mem_pressure > 0.45 && swap_active && is_root {
                    // Build foreground family via process tree (heuristic).
                    let mut fg_pids = build_foreground_family(foreground_pid, &process_tree);
                    // Augment with kernel-authoritative coalition membership.
                    // Any PID sharing a coalition with the foreground PID is excluded.
                    if let Some(fg_pid) = foreground_pid {
                        let all_pids: Vec<u32> = proc_snaps.iter().map(|s| s.pid).collect();
                        for coalition_pid in coalition_tracker.family_of(fg_pid, &all_pids) {
                            fg_pids.insert(coalition_pid);
                        }
                    }
                    let interactive_pats: Vec<String> = state
                        .learned_policy
                        .lock_recover()
                        .interactive_patterns
                        .clone();
                    for snap in proc_snaps.iter().take(100) {
                        if heuristic_critical_pids.contains(&snap.pid)
                            || fg_pids.contains(&snap.pid)
                        {
                            continue;
                        }
                        if interactive_pats
                            .iter()
                            .any(|p| snap.name.contains(p.as_str()))
                        {
                            continue;
                        }
                        // Rosetta 2 processes incur ~10-30% JIT overhead.
                        // Under memory pressure, they get a lower RSS threshold
                        // because freezing them recovers more real throughput.
                        let rss_threshold = if snap.is_translated {
                            80 * 1024 * 1024 // 80MB for Rosetta (vs 120MB for native)
                        } else {
                            120 * 1024 * 1024
                        };
                        let is_hoarder = snap.rss_bytes > rss_threshold
                            && snap.secs_since_user_interaction > 120
                            && !snap.has_gui_window;
                        let is_bg_renderer = snap.rss_bytes > 60 * 1024 * 1024
                            && snap.secs_since_user_interaction > 120
                            && (snap.name.contains("Helper (Renderer)")
                                || snap.name.contains("Helper (Plugin)")
                                || snap.name.contains(" Renderer"));
                        // Mach port leak: >5000 ports is suspicious IPC flooding.
                        // Only check lazily for processes already meeting RSS threshold.
                        let is_port_leaker = if snap.rss_bytes > 50 * 1024 * 1024
                            && snap.secs_since_user_interaction > 60
                        {
                            let qos = state.mach_qos.lock_recover();
                            qos.get_mach_port_count(snap.pid)
                                .map(|c| c > 5000)
                                .unwrap_or(false)
                        } else {
                            false
                        };
                        if is_hoarder || is_bg_renderer || is_port_leaker {
                            // Targeted non-fatal warn limit: set to 75% of current RSS.
                            // Rosetta processes get a tighter squeeze (60% of RSS).
                            let ratio = if snap.is_translated { 3u64 } else { 4u64 };
                            let warn_mb = (snap.rss_bytes * ratio / 5 / 1024 / 1024) as i32;
                            let warn_mb = warn_mb.max(32); // floor: 32 MB
                            if let Err(e) = jetsam_control::set_warn_limit(snap.pid, warn_mb) {
                                // Non-fatal: log at debug level and continue.
                                if cfg!(debug_assertions) {
                                    eprintln!("[warn-limit] {}", e);
                                }
                            } else {
                                warn_limit_pids.insert(snap.pid, 3); // clear after 3 cycles
                            }
                        }
                    }
                }

                // Clear expired warn limits (process has had time to respond).
                warn_limit_pids.retain(|&pid, countdown| {
                    *countdown -= 1;
                    if *countdown == 0 {
                        let _ = jetsam_control::set_warn_limit(pid, 0);
                        false
                    } else {
                        true
                    }
                });

                // Update heuristic metrics
                {
                    let mut m = state.metrics.lock_recover();
                    m.heuristic_decisions += heuristic_stats.decisions_total;
                    m.heuristic_throttles += heuristic_stats.throttles;
                    m.heuristic_freezes += heuristic_stats.freezes;
                    m.heuristic_kills_downgraded += heuristic_stats.kills_downgraded;
                    m.zombies_detected += heuristic_stats.zombies_detected;
                    // Update current workload from adaptive governor's user profile
                    let gov = state.adaptive_governor.lock_recover();
                    m.current_workload = format!("{:?}", gov.user_profile.current_workload());
                }

                // F2 — ML Ligero: read classification result (computed inside decide_all this cycle).
                // GovernorConfig aggressiveness was already updated inside decide_all().
                let ml_class = {
                    let gov = state.adaptive_governor.lock_recover();
                    gov.last_ml_classification().clone()
                };
                {
                    let mut m = state.metrics.lock_recover();
                    m.ml_confidence = ml_class.confidence;
                    m.current_workload = format!("{:?}", ml_class.workload).to_lowercase();
                    m.ml_sources = ml_class.sources_summary();
                }
                // Cable: GPU optimize_for_workload → log GPU-specific hints when
                // workload changes AND GPU is drawing power (gpu_active in features).
                // This feeds observability: what GPU strategy Apollo recommends per workload.
                if cycle_hw_snap.as_ref().and_then(|h| h.power.gpu_watts).unwrap_or(0.0) > 2.0 {
                    let wl_str = format!("{:?}", ml_class.workload).to_lowercase();
                    let gpu_hints = gpu_mgr.optimize_for_workload(&wl_str);
                    if !gpu_hints.is_empty() && cycle_count % 30 == 0 {
                        audit_log(&serde_json::json!({
                            "event": "gpu_workload_hint",
                            "workload": wl_str,
                            "hints": gpu_hints,
                        }));
                    }
                }

                // Sysctl Governor: reactive tuning based on TCP health, memory pressure, and workload.
                // Gate netstat to every ~10s since the main loop now cycles every 500ms-2s.
                if last_netstat_tick.elapsed() >= Duration::from_secs(10) {
                    let _ = network_monitor.tick();
                    last_netstat_tick = Instant::now();
                }
                let sysctl_actions = sysctl_governor.tick(&SysctlGovernorInput {
                    net_monitor: &network_monitor,
                    swap_trend: swap_forecast.swap_trend,
                    memory_pressure: snapshot.pressure.memory_pressure,
                    workload: &format!("{:?}", ml_class.workload).to_lowercase(),
                    on_battery: power_mgr.is_on_battery(),
                    is_root,
                });
                actions.extend(sysctl_actions);

                // NetworkOptimizer: profile-driven TCP tuning complements sysctl_governor.
                // Select network profile based on optimization profile + battery state.
                // Emits SetSysctl actions for TCP buffers, delayed_ack, window scale.
                if is_root && cycle_count % 30 == 1 {
                    let net_profile = if power_mgr.is_on_battery() {
                        NetworkProfile::Battery
                    } else {
                        match current_profile {
                            OptimizationProfile::AggressiveRoot => NetworkProfile::HighThroughput,
                            OptimizationProfile::BalancedRoot => NetworkProfile::Balanced,
                            OptimizationProfile::SafeRoot => NetworkProfile::LowLatency,
                        }
                    };
                    for (key, value) in net_optimizer.get_sysctl_recommendations(net_profile) {
                        actions.push(RootAction::SetSysctl {
                            key,
                            value,
                            reason: format!("network-optimizer: {:?} profile", net_profile),
                        });
                    }
                }

                // Update SharedState with latest sysctl governor status for ctl queries.
                {
                    let status = sysctl_governor.status(&network_monitor);
                    *state.sysctl_governor_status.lock_recover() = status;
                }

                // F3 — Safety Precedence: foreground app is NEVER throttled or frozen.
                // Also protects recently active apps (minimized but used in the last 5 min).
                // Only logs to discrepancy when the reason is ambiguous (not covered by
                // foreground detection or activity sensor) — those are the cases where
                // the LLM teacher actually adds value.
                {
                    let fg_family_pids = build_foreground_family(foreground_pid, &process_tree);
                    let recently_active_window = std::time::Duration::from_secs(300);

                    let mut ambiguous_removed = 0usize;
                    actions.retain(|a| match a {
                        RootAction::ThrottleProcess { pid, name, .. }
                        | RootAction::FreezeProcess { pid, name, .. } => {
                            // Foreground by PID family — deterministic, don't log.
                            if fg_family_pids.contains(pid) {
                                return false;
                            }
                            // Foreground by name — deterministic, don't log.
                            if let Some(fg) = &foreground_app {
                                if name.contains(fg.as_str()) {
                                    return false;
                                }
                            }
                            // Recently active — deterministic, don't log.
                            if fg_detector.is_recently_active(name, recently_active_window) {
                                return false;
                            }
                            // From here: protected by learned_policy or critical_bg —
                            // these ARE ambiguous cases worth logging for LLM learning.
                            ambiguous_removed += 1;
                            true
                        }
                        _ => true,
                    });
                    // Only log truly ambiguous cases — where signals didn't explain the
                    // protection. This is the useful signal for the LLM teacher.
                    if ambiguous_removed > 0 {
                        if let Some(fg) = &foreground_app {
                            append_discrepancy_log(
                                &state.discrepancy_log_path,
                                fg,
                                ambiguous_removed,
                                &format!("{:?}", ml_class.workload),
                                ml_class.confidence,
                                "ambiguous",
                            );
                        }
                    }
                }

                // F4 — Thermal Master Switch: >95°C P-cluster — suppress all Boost actions.
                // Also suppress during resource interrupt Emergency/SuperEmergency.
                let interrupt_phase = state.resource_interrupt.phase.load(Ordering::Acquire);
                if thermal_emergency || interrupt_phase >= 2 {
                    actions.retain(|a| !matches!(a, RootAction::BoostProcess { .. }));
                }

                let policy = SafetyPolicy::for_profile(current_profile);

                let now = Instant::now();
                if thrash
                    .minute_started
                    .map(|s| now.duration_since(s) >= Duration::from_secs(60))
                    .unwrap_or(true)
                {
                    thrash.minute_started = Some(now);
                    state.metrics.lock_recover().budgets.minute_actions = 0;
                }

                let caps = detect_capabilities();

                // Phase 1: Compute budget-filtered actions (metrics lock held briefly).
                // BUG 5 fix: split into three phases so the metrics mutex is never held
                // across the blocking I/O inside execute_actions.
                let final_actions = {
                    let mut metrics = state.metrics.lock_recover();
                    // TTL: don't leave freezes hanging forever.
                    // Skip PIDs currently frozen by the resource interrupt handler.
                    {
                        let now = Utc::now();
                        let interrupt_pids = state
                            .resource_interrupt
                            .interrupt_frozen_pids
                            .try_lock()
                            .ok()
                            .map(|g| g.clone())
                            .unwrap_or_default();
                        let current_pressure = snapshot.pressure.memory_pressure;
                        let mut frozen_state = state.frozen_state.lock_recover();
                        let total_frozen = frozen_state.len();
                        let mut expired: Vec<u32> = frozen_state
                            .iter()
                            .filter(|(pid, entry)| {
                                let elapsed =
                                    now.signed_duration_since(entry.frozen_at).num_seconds();
                                should_unfreeze(
                                    elapsed,
                                    entry.pressure_at_freeze,
                                    current_pressure,
                                ) && !interrupt_pids.contains(pid)
                            })
                            .map(|(pid, _)| *pid)
                            .collect();
                        // FIFO rotation: on 8GB hardware, rotate oldest frozen
                        // process to prevent resource hoarding under sustained pressure.
                        if let Some((&oldest_pid, oldest_entry)) = frozen_state
                            .iter()
                            .filter(|(pid, _)| !interrupt_pids.contains(pid) && !expired.contains(pid))
                            .min_by_key(|(_, e)| e.frozen_at)
                        {
                            let elapsed = now
                                .signed_duration_since(oldest_entry.frozen_at)
                                .num_seconds();
                            if should_rotate_oldest(elapsed, total_frozen) {
                                expired.push(oldest_pid);
                            }
                        }
                        if !expired.is_empty() {
                            let count = unfreeze_pids(expired.iter().copied());
                            for pid in &expired {
                                frozen_state.remove(pid);
                            }
                            write_frozen_state(&frozen_state_path, &frozen_state);
                            metrics.post_wake_defensive_unfreezes += count;
                            metrics.unfreezes_applied += count;
                            metrics.throttle_reverted += count;
                        }
                    }
                    metrics.budgets.cycle_boosts = 0;
                    metrics.budgets.cycle_throttles = 0;
                    metrics.budgets.cycle_hints = 0;
                    metrics.budgets.cycle_freezes = 0;
                    metrics.budgets.cycle_sysctl_writes = 0;
                    metrics.budgets.boost_denied_cooldown = 0;

                    let (graced_actions, throttle_suppressed, freeze_suppressed) =
                        apply_post_wake_grace_policy(actions, grace_active);
                    metrics.post_wake_throttle_suppressed += throttle_suppressed;
                    metrics.post_wake_freeze_suppressed += freeze_suppressed;

                    // Freeze confirmation: only freeze PIDs flagged for 2+ consecutive cycles.
                    // This filters out short-lived transients that die before execute_actions.
                    // First, collect all PIDs proposed for freeze this cycle (before filtering).
                    let proposed_freeze_pids: HashSet<u32> = graced_actions
                        .iter()
                        .filter_map(|a| {
                            if let RootAction::FreezeProcess { pid, .. } = a {
                                Some(*pid)
                            } else {
                                None
                            }
                        })
                        .collect();
                    let confirmed_actions: Vec<RootAction> = graced_actions
                        .into_iter()
                        .filter(|a| {
                            if let RootAction::FreezeProcess { pid, .. } = a {
                                let count = freeze_candidates.entry(*pid).or_insert(0);
                                *count += 1;
                                *count >= 2
                            } else {
                                true
                            }
                        })
                        .collect();
                    // Decay: remove PIDs no longer proposed for freeze this cycle.
                    // Use proposed_freeze_pids (all proposals) not just confirmed ones,
                    // so first-cycle candidates survive to reach count >= 2.
                    freeze_candidates.retain(|pid, _| proposed_freeze_pids.contains(pid));

                    // Compressor-aware + deep-scan freeze decisions (v0.7.0).
                    // For top 3 freeze candidates, run vm_region scan + temperature probe.
                    // Uses decide_enhanced when deep data available, else falls through to legacy.
                    // Approximate active process count for SLC budget.
                    // Precise count not critical — SLC share is a rough heuristic.
                    let active_count = confirmed_actions.len().max(5);
                    let mut ds_scans = 0u64;
                    let mut ds_probes = 0u64;
                    let mut ds_freeze = 0u64;
                    let mut ds_skip = 0u64;
                    let mut ds_hint = 0u64;
                    let confirmed_actions: Vec<RootAction> = confirmed_actions.into_iter().filter_map(|a| {
                        if let RootAction::FreezeProcess { pid, name: ref freeze_name, ref reason, .. } = a {
                            // query_memory_profile falls back to proc_pid_rusage (~3µs)
                            // when task_for_pid fails (ad-hoc signing). No timeout needed.
                            if let Some(profile) = query_memory_profile(pid) {
                                ds_scans += 1;
                                let fault_rate = mem_analyzer.major_fault_rate(pid);
                                // Deep scan: vm_region + temperature (only in mid/high zone).
                                let temp = if signal_digest.pressure_smooth >= 0.30 {
                                    ds_probes += 1;
                                    sample_process_temperature(pid)
                                } else {
                                    None
                                };
                                // Cable: classify_by_memory() → skip freezing LLM/Database processes.
                                // If vm_region scan reveals an LLM inference or database layout,
                                // freezing would be destructive (model eviction, buffer pool loss).
                                let memory_hint = scan_regions(pid)
                                    .and_then(|regions| classify_by_memory(&regions));
                                if let Some((hint, conf)) = &memory_hint {
                                    use apollo_optimizer::engine::workload_classifier::MemoryLayoutHint;
                                    if conf > &0.7 && matches!(hint, MemoryLayoutHint::LlmInference | MemoryLayoutHint::DatabaseEngine) {
                                        ds_skip += 1;
                                        audit_log(&serde_json::json!({
                                            "t": Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                                            "event": "deep_scan_layout_skip",
                                            "pid": pid,
                                            "name": freeze_name,
                                            "hint": format!("{:?}", hint),
                                            "confidence": conf,
                                        }));
                                        return None;
                                    }
                                }
                                let action = decide_enhanced(
                                    &profile,
                                    temp.as_ref(),
                                    None, // DAMON WSS integrated later per-process
                                    active_count,
                                    metrics.memory_pressure,
                                    fault_rate,
                                );
                                let decision_str = match action {
                                    MemoryAction::Freeze => "freeze",
                                    MemoryAction::Skip => "skip",
                                    MemoryAction::PressureHint => "hint",
                                };
                                audit_log(&serde_json::json!({
                                    "t": Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                                    "event": "deep_scan",
                                    "pid": pid,
                                    "decision": decision_str,
                                    "ratio": (profile.compression_ratio * 100.0).round() / 100.0,
                                    "phys_mb": profile.phys_footprint / 1024 / 1024,
                                    "compressed_mb": profile.compressed_bytes / 1024 / 1024,
                                    "purgeable_mb": profile.purgeable_bytes / 1024 / 1024,
                                    "temp": temp.as_ref().map(|t| serde_json::json!({
                                        "hot": (t.pct_hot * 100.0).round(),
                                        "dram": (t.pct_dram * 100.0).round(),
                                        "compressed": (t.pct_compressed * 100.0).round(),
                                        "samples": t.sample_count,
                                    })),
                                    "fault_rate": (fault_rate * 10.0).round() / 10.0,
                                    "pressure": (metrics.memory_pressure * 1000.0).round() / 1000.0,
                                    "memory_layout": memory_hint.as_ref().map(|(h, c)| format!("{:?}({:.0}%)", h, c * 100.0)),
                                }));
                                match action {
                                    MemoryAction::PressureHint => {
                                        ds_hint += 1;
                                        // Cable: purge_purgeable_regions() → reclaim RAM without freeze.
                                        // When we'd only send a hint, also actively purge purgeable
                                        // regions. This is the "secret weapon": free RAM from a live
                                        // process without SIGSTOP, by marking purgeable pages volatile.
                                        if profile.purgeable_bytes > 10 * 1024 * 1024 {
                                            let purged = purge_purgeable_regions(pid).unwrap_or(0);
                                            if purged > 0 {
                                                audit_log(&serde_json::json!({
                                                    "t": Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                                                    "event": "purge_purgeable",
                                                    "pid": pid,
                                                    "name": freeze_name,
                                                    "regions_purged": purged,
                                                    "purgeable_mb": profile.purgeable_bytes / 1024 / 1024,
                                                }));
                                            }
                                        }
                                        Some(RootAction::SetMemorystatus {
                                            pid,
                                            priority: -1,
                                            reason: format!(
                                                "{} (deep-scan: ratio={:.1} purgeable={}MB temp={} → hint+purge)",
                                                reason,
                                                profile.compression_ratio,
                                                profile.purgeable_bytes / 1024 / 1024,
                                                temp.as_ref().map(|t| format!("hot={:.0}%", t.pct_hot * 100.0))
                                                    .unwrap_or_else(|| "n/a".to_string()),
                                            ),
                                        })
                                    }
                                    MemoryAction::Skip => {
                                        ds_skip += 1;
                                        // Cable: check_resident equivalent — if we're skipping because
                                        // pct_compressed > 0.60, the process is already mostly swapped.
                                        // No action needed: the process isn't consuming physical RAM.
                                        // Log this so we can verify the skip was correct.
                                        if let Some(ref t) = temp {
                                            if t.pct_compressed > 0.60 {
                                                audit_log(&serde_json::json!({
                                                    "t": Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                                                    "event": "skip_already_cold",
                                                    "pid": pid,
                                                    "name": freeze_name,
                                                    "pct_compressed": (t.pct_compressed * 100.0).round(),
                                                    "reason": "process already swapped/compressed, freeze pointless",
                                                }));
                                            }
                                        }
                                        None
                                    }
                                    MemoryAction::Freeze => { ds_freeze += 1; Some(a) }
                                }
                            } else {
                                Some(a)
                            }
                        } else {
                            Some(a)
                        }
                    }).collect();
                    metrics.deep_scan_count += ds_scans;
                    metrics.deep_scan_temp_probes += ds_probes;
                    metrics.deep_scan_freeze += ds_freeze;
                    metrics.deep_scan_skip += ds_skip;
                    metrics.deep_scan_hint += ds_hint;

                    // Rosetta AOT: skip freezing oahd/oahd-helper during AOT compilation.
                    let confirmed_actions: Vec<RootAction> = if rosetta_monitor.is_compiling() {
                        let rosetta_immune = apollo_optimizer::engine::rosetta_monitor::RosettaMonitor::immune_processes();
                        confirmed_actions
                            .into_iter()
                            .filter(|a| {
                                if let RootAction::FreezeProcess { name, .. } = a {
                                    !rosetta_immune.iter().any(|ri| name.contains(ri))
                                } else {
                                    true
                                }
                            })
                            .collect()
                    } else {
                        confirmed_actions
                    };

                    // Subatomic: skip freeze for processes with tiny RSS (< 5MB).
                    // These processes are already idle/paged-out — SIGSTOP adds no value
                    // and risks stalling IPC for zero memory savings.
                    let confirmed_actions: Vec<RootAction> = confirmed_actions
                        .into_iter()
                        .filter(|a| {
                            if let RootAction::FreezeProcess { pid, .. } = a {
                                match proc_taskinfo::get_task_info(*pid) {
                                    Some(ti) if ti.resident_size < 5 * 1024 * 1024 => false,
                                    _ => true,
                                }
                            } else {
                                true
                            }
                        })
                        .collect();

                    // Audit fix #4: Wait-graph deadlock prevention.
                    // Check each freeze candidate against the wait-graph before execution.
                    let frozen_pids: HashSet<u32> =
                        state.frozen_state.lock_recover().keys().copied().collect();
                    let confirmed_actions: Vec<RootAction> = confirmed_actions
                        .into_iter()
                        .filter(|a| {
                            if let RootAction::FreezeProcess { pid, .. } = a {
                                wait_graph::is_freeze_safe(*pid, &frozen_pids)
                            } else {
                                true
                            }
                        })
                        .collect();

                    // Audit fix #4b: Unfreeze stuck frozen processes (IPC deadlock recovery).
                    let stuck_pids = wait_graph::find_stuck_frozen(&frozen_pids);
                    for stuck_pid in &stuck_pids {
                        if unsafe { libc::kill(*stuck_pid as i32, 0) } == 0 {
                            unsafe {
                                libc::kill(*stuck_pid as i32, libc::SIGCONT);
                            }
                        }
                    }
                    if !stuck_pids.is_empty() {
                        let mut frozen_map = state.frozen_state.lock_recover();
                        for pid in &stuck_pids {
                            frozen_map.remove(pid);
                        }
                        metrics.unfreezes_applied += stuck_pids.len() as u64;
                    }

                    let filtered = filter_boost_cooldown(confirmed_actions, &policy, &mut thrash);
                    let minute_cap = match latency_target {
                        LatencyTarget::Max => 120,
                        LatencyTarget::Low => 50,
                        LatencyTarget::Normal => 80,
                    };
                    let fa = enforce_limits_with_budget(
                        filtered,
                        &policy,
                        &mut metrics.budgets,
                        minute_cap,
                    );
                    metrics.last_actions_summary = format!(
                        "actions={} boosts={} throttles={} freezes={} sysctl={} invalid_sysctl_denied={}",
                        fa.len(),
                        fa.iter().filter(|a| matches!(a, RootAction::BoostProcess { .. })).count(),
                        fa.iter().filter(|a| matches!(a, RootAction::ThrottleProcess { .. })).count(),
                        fa.iter().filter(|a| matches!(a, RootAction::FreezeProcess { .. })).count(),
                        fa.iter().filter(|a| matches!(a, RootAction::SetSysctl { .. })).count(),
                        metrics.invalid_sysctl_denied
                    );
                    fa
                    // metrics lock released here
                };

                // Phase 2: Execute actions WITHOUT holding the metrics lock.
                // Captura los nombres de throttles antes de mover final_actions.
                let throttle_names_for_outcome: Vec<String> = final_actions
                    .iter()
                    .filter_map(|a| {
                        if let RootAction::ThrottleProcess { name, .. } = a {
                            Some(name.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                let exec_outcomes = {
                    // Extract a temporary HashSet for execute_actions (which requires &mut HashSet<u32>).
                    let mut frozen_set: HashSet<u32> =
                        state.frozen_state.lock_recover().keys().copied().collect();
                    // Snapshot before execution — used to detect changes and skip redundant disk writes.
                    let frozen_before: HashSet<u32> = frozen_set.clone();
                    let (learned_protected, learned_interactive) = {
                        let policy = state.learned_policy.lock_recover();
                        (
                            policy.protected_patterns.clone(),
                            policy.interactive_patterns.clone(),
                        )
                    };
                    let mut qos = state.mach_qos.lock_recover();
                    let outcomes = execute_actions(
                        final_actions,
                        &caps,
                        &journal_path,
                        &mut frozen_set,
                        &learned_protected,
                        &learned_interactive,
                        Some(&mut qos),
                    );
                    // Sync the temporary set back into the unified frozen_state map.
                    let now = Utc::now();
                    let mut frozen_state = state.frozen_state.lock_recover();
                    // Add newly frozen PIDs.
                    for pid in &frozen_set {
                        frozen_state.entry(*pid).or_insert(FrozenEntry {
                            frozen_at: now,
                            source: FreezeSource::MainLoop,
                            pressure_at_freeze: snapshot.pressure.memory_pressure,
                        });
                    }
                    // Remove PIDs that are no longer frozen.
                    frozen_state.retain(|pid, _| frozen_set.contains(pid));
                    // Only persist to disk when the frozen set actually changed.
                    if frozen_set != frozen_before {
                        write_frozen_state(&frozen_state_path, &frozen_state);
                    }
                    outcomes
                    // frozen_state lock released here
                };

                // kqueue: watch newly frozen PIDs for death (OOM/jetsam push notification).
                if exec_outcomes.freezes_applied > 0 {
                    if let Some(ref mut kq) = kq_frozen {
                        let frozen_state = state.frozen_state.lock_recover();
                        for &pid in frozen_state.keys() {
                            let _ = kq.watch_pid(pid); // best-effort, ENOENT is fine
                        }
                    }
                }

                // EnergyTracker: record savings for newly frozen processes.
                // Estimate watts saved using the process tree's aggregate CPU data
                // combined with the current CPU power reading from the hw snapshot.
                if exec_outcomes.freezes_applied > 0 {
                    let cpu_watts = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.cpu_watts)
                        .unwrap_or(0.0) as f64;
                    let total_cpu_pct: f64 = snapshot
                        .top_processes
                        .iter()
                        .map(|p| p.cpu_usage as f64)
                        .sum::<f64>()
                        .max(0.01);
                    let frozen_state = state.frozen_state.lock_recover();
                    for p in &snapshot.top_processes {
                        if frozen_state.contains_key(&p.pid) && p.cpu_usage > 0.0 {
                            let fraction = (p.cpu_usage as f64) / total_cpu_pct;
                            let saved_watts = fraction * cpu_watts;
                            // Record savings for 1 cycle duration (will accumulate over time).
                            energy_tracker.record_savings(saved_watts, 30.0);
                        }
                    }
                }

                // Outcome tracking: registra los throttles ejecutados esta ronda.
                // Necesitamos el proceso + watts actuales + presión antes.
                if exec_outcomes.throttles_applied > 0 {
                    let cpu_watts = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.cpu_watts)
                        .unwrap_or(0.0) as f64;
                    let total_cpu_pct: f64 = snapshot
                        .top_processes
                        .iter()
                        .map(|p| p.cpu_usage as f64)
                        .sum::<f64>()
                        .max(0.01);
                    let mem_pressure_now = snapshot.pressure.memory_pressure;
                    for name in &throttle_names_for_outcome {
                        let proc_watts = snapshot
                            .top_processes
                            .iter()
                            .find(|p| &p.name == name)
                            .map(|p| (p.cpu_usage as f64 / total_cpu_pct) * cpu_watts)
                            .unwrap_or(0.0);
                        outcome_tracker.record_throttle(name, mem_pressure_now, proc_watts);
                    }
                }

                // ── Causal graph (Pearl 2009, memoria-core/causal_inference.rs) ──
                // Record throttle/freeze actions for causal evaluation.
                // Each action becomes a pending observation; eval_delay cycles later
                // we check if pressure actually dropped (cause → effect).
                {
                    let pressure_now = snapshot.pressure.memory_pressure as f32;
                    for name in &throttle_names_for_outcome {
                        causal_graph.record_action(
                            &format!("throttle:{}", name),
                            pressure_now,
                            cycle_count,
                        );
                    }
                    // Also record freeze actions.
                    if exec_outcomes.freezes_applied > 0 {
                        let frozen_state = state.frozen_state.lock_recover();
                        for &pid in frozen_state.keys() {
                            if let Some(process) = collector.system().process(sysinfo::Pid::from_u32(pid)) {
                                causal_graph.record_action(
                                    &format!("freeze:{}", process.name()),
                                    pressure_now,
                                    cycle_count,
                                );
                            }
                        }
                    }
                    // Evaluate pending actions: did pressure actually drop?
                    causal_graph.evaluate(pressure_now, cycle_count);
                }

                // Causal graph: record process co-occurrence during high-pressure events.
                if snapshot.pressure.memory_pressure >= 0.60 {
                    let active: Vec<String> = snapshot.top_processes
                        .iter()
                        .take(10)
                        .map(|p| p.name.clone())
                        .collect();
                    outcome_tracker.record_co_occurrence(&active);
                }

                // Counterfactual: observe pressure drift. If no throttles this cycle,
                // the tracker learns the natural drift rate (what happens without action).
                outcome_tracker.observe_cycle(
                    snapshot.pressure.memory_pressure,
                    !throttle_names_for_outcome.is_empty(),
                );

                // Outcome tracker tick: resuelve outcomes de hace 30s, actualiza pesos y energy savings.
                {
                    let batch = outcome_tracker.tick(snapshot.pressure.memory_pressure);
                    if batch.savings_watts > 0.0 {
                        energy_tracker.record_savings(batch.savings_watts, 30.0);
                    }
                    // Cable 1: causal_effect() → correct PatternWeight using real causal signal.
                    // For each effective throttle, check if the drop was truly caused by the
                    // action (causal_effect > 0) or just natural drift. Demote weights that
                    // only appear effective due to natural pressure fluctuation.
                    if !batch.effective_names.is_empty() {
                        let drift = outcome_tracker.natural_drift();
                        if drift > 0.01 {
                            // Pre-compute causal effects per process before mutating weights.
                            let demotions: Vec<String> = batch.effective_names.iter().filter_map(|name| {
                                let avg_drop = outcome_tracker.experience
                                    .query_similar(name, snapshot.pressure.memory_pressure)
                                    .map(|(drop, _)| drop)
                                    .unwrap_or(0.05);
                                let causal = outcome_tracker.causal_effect(avg_drop);
                                if causal < 0.01 { Some(name.clone()) } else { None }
                            }).collect();
                            // Now mutate: roll back effective_count for drift-only "successes".
                            for name in &demotions {
                                if let Some(w) = outcome_tracker.weights.get_mut(name) {
                                    if w.effective_count > 0 {
                                        w.effective_count -= 1;
                                    }
                                }
                            }
                        }
                    }
                    // Sincroniza pesos Bayesianos a la LearnedPolicy persistida.
                    if !batch.effective_names.is_empty() || !batch.low_value_names.is_empty() {
                        let mut policy = state.learned_policy.lock_recover();
                        for (name, weight) in &outcome_tracker.weights {
                            policy.pattern_weights.insert(name.clone(), weight.clone());
                        }
                    }
                }

                // Lifelong zone learning: feed outcome effectiveness to router zones.
                // Effective actions → lower zone thresholds (engage earlier).
                // Ineffective actions → raise thresholds (be more conservative).
                {
                    let effectiveness = outcome_tracker.overall_effectiveness();
                    let pressure = signal_digest.pressure_smooth;
                    if outcome_tracker.total_resolved > 10 {
                        signal_intel.zone_feedback(pressure, effectiveness > 0.50);
                    }
                }

                // Cable A: OutcomeTracker → RL reward signal.
                // When throttling is wasteful (low-value patterns detected),
                // penalize the RL agent so it learns to adjust thresholds.
                {
                    let penalty = outcome_tracker.rl_penalty();
                    if penalty < 0.0 {
                        if let Some(rl) = &mut overflow_guard.rl_agent {
                            rl.inject_external_reward(penalty);
                        }
                    }
                }

                // Dr. Zero feedback loop: read external score from watcher's
                // autoresearch and use it to reinforce/penalize the RL agent.
                // File written by watch-deploy.sh after each autoresearch run.
                if cycle_count % 60 == 30 {
                    if let Ok(data) = std::fs::read_to_string("/tmp/apollo-dr-zero-feedback.json") {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                            if let Some(score) = v.get("score").and_then(|s| s.as_f64()) {
                                // Normalize: score 90+ is good (reward), <70 is bad (penalty).
                                // Range maps to [-0.3, +0.3] RL reward.
                                let reward = ((score - 80.0) / 33.3).clamp(-0.3, 0.3);
                                if let Some(rl) = &mut overflow_guard.rl_agent {
                                    rl.inject_external_reward(reward);
                                }
                            }
                        }
                    }
                }

                // Predictive agent: observe outcome and update model.
                predictive_agent.observe_outcome(snapshot.pressure.memory_pressure);
                predictive_agent.maybe_persist();
                // MPC feedback: tell MPC what happened after its recommendation.
                signal_intel.mpc_feedback(
                    signal_digest.mpc_recommendation,
                    signal_digest.pressure_smooth,
                    snapshot.pressure.memory_pressure,
                );
                // Persist signal intelligence state every 100 cycles so hazard model + MPC
                // effects survive crashes (not just clean shutdowns).
                if cycle_count % 100 == 0 {
                    signal_intel.persist(std::path::Path::new(signal_intelligence_path()));
                    outcome_tracker.persist_hop_groups(std::path::Path::new(hop_groups_path()));
                    // Causal graph observability: log solid/weak links discovered.
                    let solid = causal_graph.solid_count();
                    let total = causal_graph.edge_count();
                    if total > 0 {
                        println!("causal_graph: {}/{} edges solid, {} pending",
                            solid, total, causal_graph.solid_edges().len());
                    }
                }
                // Hourly housekeeping (7200 cycles × 500ms ≈ 1 hour).
                if cycle_count % 7200 == 1 {
                    // GC stale entries from cache warmer + I/O shaper.
                    cache_warmer.gc();
                    io_shaper.gc();
                    // Persist temporal predictor state.
                    temporal_predictor.persist();
                }
                // Update predictive agent + signal intelligence metrics for status reporting.
                {
                    let mut m = state.metrics.lock_recover();
                    m.predictive_agent_active = predictive_agent.is_active();
                    m.predictive_agent_cycles = predictive_agent.total_cycles();
                    m.predictive_agent_arm_pulls = predictive_agent.arm_pulls();
                    m.predictive_agent_last_intervention = format!("{:?}", agent_intervention);
                    m.si_pressure_smooth = signal_digest.pressure_smooth;
                    m.si_pressure_velocity = signal_digest.pressure_velocity;
                    m.si_p_oom_30s = signal_digest.p_oom_30s;
                    m.si_urgency = signal_digest.urgency;
                    if signal_digest.regime_shift_up {
                        m.si_regime_shifts += 1;
                    }
                    m.si_monopoly_risk = signal_digest.monopoly_risk;
                    m.si_entropy_anomaly = signal_digest.entropy_anomaly;
                    // Cable 4: top_causal_pairs() → expose in metrics for observability.
                    m.causal_pairs = outcome_tracker.top_causal_pairs(5)
                        .iter()
                        .map(|(a, b, c)| format!("{} + {} ({})", a, b, c))
                        .collect();
                    m.natural_drift = outcome_tracker.natural_drift();
                    m.experience_memory_size = outcome_tracker.experience.len();
                    // Causal effect average: mean effect across last resolved outcomes.
                    m.causal_effect_avg = {
                        let effectiveness = outcome_tracker.overall_effectiveness();
                        let avg_drop = if outcome_tracker.total_resolved > 0 {
                            effectiveness * 0.05
                        } else {
                            0.0
                        };
                        outcome_tracker.causal_effect(avg_drop)
                    };
                    // HRPO / Dr. Zero metrics
                    m.dr_zero_self_challenge = outcome_tracker.self_challenge_score();
                    m.dr_zero_groups = outcome_tracker.hop_group_summary()
                        .iter()
                        .map(|(hop, eff, count, pred_err)| {
                            format!("{:?}(eff={:.0}% n={} err={:.2})", hop, eff * 100.0, count, pred_err)
                        })
                        .collect();
                    m.dr_zero_exploration = outcome_tracker.exploration_needed()
                        .iter()
                        .map(|(hop, err)| format!("{:?}(err={:.2})", hop, err))
                        .collect();
                }

                // I/O Traffic Shaping: foreground-aware disk bandwidth allocation.
                // Iyer & Druschel 2001 — anticipatory scheduling + I/O priority classes
                // reduce foreground I/O latency by 50-70% under concurrent background load.
                // Runs every 20 cycles (~10s): MIN_REAPPLY_SECS=60 so nothing reapplies within 60s anyway.
                if cycle_count % 20 == 0 && is_root {
                    let fg_family_io = build_foreground_family(foreground_pid, &process_tree);
                    let fg_pids: Vec<u32> = fg_family_io.iter().copied().collect();
                    let process_tiers: Vec<(
                        u32,
                        apollo_optimizer::engine::process_classifier::ProcessTier,
                    )> = heuristic_decisions
                        .iter()
                        .map(|d| (d.pid, d.tier))
                        .collect();
                    let under_pressure = snapshot.pressure.memory_pressure + battery_pressure_boost(&power_mgr) + thermal_pressure_boost > 0.60;
                    let mut qos = state.mach_qos.lock_recover();
                    let io_changes =
                        io_shaper.shape(&fg_pids, &process_tiers, under_pressure, Some(&mut qos));
                    drop(qos);
                    if io_changes > 0 {
                        state.metrics.lock_recover().sysctl_reactive_writes += io_changes as u64;
                    }
                }

                // F5 — MachQoS: route processes to P-Cores / E-Cores based on heuristic decisions.
                // Skip SIGSTOP'd processes; force E-Cores for all during thermal emergency.
                // Uses process tree to cascade Foreground tier to all children of the
                // foreground app (e.g., Chrome Helper processes get P-core routing too).
                {
                    let frozen_pids: HashSet<u32> =
                        state.frozen_state.lock_recover().keys().copied().collect();

                    // Build the foreground family set from the process tree.
                    let fg_family = build_foreground_family(foreground_pid, &process_tree);

                    let interrupt_frozen = state
                        .resource_interrupt
                        .interrupt_frozen_pids
                        .try_lock()
                        .ok()
                        .map(|g| g.clone())
                        .unwrap_or_default();
                    let mut qos_changes: Vec<(u32, SchedulingTier)> = heuristic_decisions
                        .iter()
                        .filter(|d| {
                            !frozen_pids.contains(&d.pid)
                                && !heuristic_critical_pids.contains(&d.pid)
                                && !interrupt_frozen.contains(&d.pid)
                        })
                        .filter_map(|decision| {
                            let tier = if thermal_action.force_ecores && !fg_family.contains(&decision.pid) {
                                // Thermal pre-throttle: route backgrounds to E-Cores at Phase2+ (85°C).
                                // Foreground app stays on P-Cores for responsiveness.
                                SchedulingTier::Background
                            } else if fg_family.contains(&decision.pid) {
                                // Process tree cascade: children of the foreground app
                                // get Foreground tier even if the heuristic didn't
                                // classify them as ActiveForeground by name alone.
                                SchedulingTier::Foreground
                            } else {
                                match decision.decision {
                                    GovernorDecision::Allow => {
                                        if decision.tier == ProcessTier::ActiveForeground {
                                            SchedulingTier::Foreground
                                        } else {
                                            // Normal/TASK_UNSPECIFIED is a no-op — skip the
                                            // syscall to avoid wasting task_for_pid on ~400
                                            // processes that either don't need changes or are
                                            // SIP-protected and always fail.
                                            return None;
                                        }
                                    }
                                    GovernorDecision::Throttle => return None,
                                    GovernorDecision::Freeze | GovernorDecision::Kill => {
                                        SchedulingTier::Background
                                    }
                                }
                            };
                            Some((decision.pid, tier))
                        })
                        .collect();

                    // Deduplicate: if a PID appeared in both heuristic decisions and
                    // fg_family cascade, the last entry wins (which is fine since both
                    // would map to Foreground). The MachQoSManager handles dupes internally.
                    let _ = &mut qos_changes; // suppress unused_mut if no further manipulation

                    let mut qos = state.mach_qos.lock_recover();
                    // GC dead PIDs every 30 cycles to prevent unbounded growth
                    // and handle PID recycling (recycled PID must be re-evaluated).
                    if cycle_count % 30 == 0 {
                        qos.gc_dead_pids();
                    }
                    let outcomes = qos.apply_batch(&qos_changes);
                    {
                        let mut m = state.metrics.lock_recover();
                        m.qos_foreground_count += outcomes
                            .iter()
                            .filter(|o| o.tier == SchedulingTier::Foreground && o.success)
                            .count() as u64;
                        m.qos_background_count += outcomes
                            .iter()
                            .filter(|o| o.tier == SchedulingTier::Background && o.success)
                            .count() as u64;
                        m.qos_errors += outcomes.iter().filter(|o| !o.success).count() as u64;
                    }
                }

                // Phase 3: Merge outcomes into metrics (reacquire lock after I/O).
                {
                    let mut metrics = state.metrics.lock_recover();
                    metrics.boosts_applied += exec_outcomes.boosts_applied;
                    metrics.throttles_applied += exec_outcomes.throttles_applied;
                    metrics.freezes_applied += exec_outcomes.freezes_applied;
                    metrics.unfreezes_applied += exec_outcomes.unfreezes_applied;
                    metrics.paging_hints_applied += exec_outcomes.paging_hints_applied;
                    metrics.sysctl_applied += exec_outcomes.sysctl_applied;
                    metrics.failures += exec_outcomes.failures;
                    if let Some(e) = exec_outcomes.last_error {
                        metrics.last_error = Some(e);
                    }
                    metrics.critical_background_skips += exec_outcomes.critical_background_skips;
                    metrics.invalid_sysctl_denied += exec_outcomes.invalid_sysctl_denied;
                    for skip in exec_outcomes.top_skipped {
                        if metrics.top_skipped_processes.len() < 12
                            && !metrics.top_skipped_processes.contains(&skip)
                        {
                            metrics.top_skipped_processes.push(skip);
                        }
                    }
                    metrics.top_skipped_processes.truncate(12);
                    metrics.throttle_reverted += exec_outcomes.throttle_reverted;
                    metrics.thread_qos_applied += exec_outcomes.thread_qos_applied;

                    // SysctlGovernor + NetworkMonitor metrics.
                    metrics.sysctl_reactive_writes += exec_outcomes.sysctl_applied;
                    {
                        let gov_status = state.sysctl_governor_status.lock_recover();
                        metrics.sysctl_governor_active_tunings = gov_status.active_tunings;
                        metrics.sysctl_governor_total_writes = gov_status.total_writes;
                    }
                    metrics.network_retransmit_ratio = network_monitor.retransmission_rate();
                    metrics.network_listen_drop_rate = network_monitor.listen_drop_rate();

                    let had_new_failures = exec_outcomes.failures > 0;

                    metrics.cycles += 1;
                    metrics.reactor_pulses += if decision.reactor_event_weight > 0.2 {
                        1
                    } else {
                        0
                    };
                    metrics.last_cycle_at = Some(Utc::now());
                    metrics.last_blockers = decision.blockers;
                    metrics.effective_profile = current_profile;
                    metrics.throttle_level = governor_decision.throttle_level.clone();
                    metrics.thermal_state = state.thermal_state.lock_recover().clone();
                    metrics.last_pressure_score = governor_decision.pressure_score;
                    if governor_decision.override_expired {
                        metrics.override_expirations += 1;
                    }
                    if governor_decision.override_active && !override_was_active {
                        metrics.override_activations += 1;
                    }
                    if let Some(transition) = governor_decision.transition.clone() {
                        metrics.profile_switches += 1;
                        let mut timeline = state.timeline.lock_recover();
                        timeline.push_back(transition.clone());
                        if timeline.len() > 200 {
                            timeline.pop_front();
                        }
                        append_timeline(&timeline_path, &transition);
                    }
                    override_was_active = governor_decision.override_active;

                    let elapsed = cycle_start.elapsed().as_millis() as u64;
                    metrics.cycle_durations_ms.push_back(elapsed);
                    if metrics.cycle_durations_ms.len() > 120 {
                        metrics.cycle_durations_ms.pop_front();
                    }
                    metrics.p95_cycle_ms =
                        compute_p95(metrics.cycle_durations_ms.make_contiguous());

                    *state.throttle_level.lock_recover() = metrics.throttle_level.clone();

                    let nowi = Instant::now();
                    critical_failure_timestamps
                        .retain(|t| nowi.duration_since(*t) <= Duration::from_secs(180));
                    if had_new_failures {
                        critical_failure_timestamps.push(nowi);
                    }
                    if critical_failure_timestamps.len() > 5 {
                        state.governor.lock_recover().force_safe_on_errors();
                        critical_failure_timestamps.clear();
                    }

                    // Actualizar métricas del overflow guard antes de escribir.
                    metrics.overflow_events_total = overflow_guard.history.total_overflows;
                    metrics.overflow_events_7d = overflow_guard.recent_overflow_count(7);
                    metrics.overflow_threshold_offset_pp =
                        (overflow_guard.compute_dynamic_offset() * 100.0).round() as i32;
                    metrics.overflow_workload_mode =
                        overflow_thresholds.workload_mode.as_str().to_string();

                    // RL threshold agent metrics (Phase 4).
                    if let Some(rl) = &overflow_guard.rl_agent {
                        metrics.rl_adjustment_pp = (rl.current_adjustment * 100.0).round() as i32;
                        metrics.rl_total_ticks = rl.total_ticks();
                        metrics.rl_total_overflows = rl.total_overflows();
                    }

                    // Clone before releasing lock — write_metrics does file I/O
                    // and holding the lock during I/O blocks GetStatus requests.
                    let metrics_snapshot = metrics.clone();
                    drop(metrics);
                    write_metrics(&metrics_path, &metrics_snapshot);
                }

                // Push estado a suscriptores activos (menubar, etc.)
                broadcast_current_status(&state);

                // Analytics: record this cycle's metrics for trend tracking.
                {
                    let thermal_now = state
                        .last_hw_snapshot
                        .lock_recover()
                        .as_ref()
                        .and_then(|h| h.temps.p_cluster_celsius)
                        .unwrap_or(0.0);
                    analytics.record_optimization(
                        snapshot.cpu.global_usage,
                        snapshot.cpu.global_usage,
                        snapshot.memory.used_ram,
                        snapshot.memory.used_ram,
                        thermal_now,
                        thermal_now,
                        (exec_outcomes.boosts_applied
                            + exec_outcomes.throttles_applied
                            + exec_outcomes.freezes_applied) as u32,
                    );
                }

                // Persist UserProfile every 10 cycles (~5 min at 30 s/cycle) so learning
                // survives daemon restarts.
                {
                    let cycles = state.metrics.lock_recover().cycles;
                    if cycles % 10 == 0 {
                        let persisted = {
                            let gov = state.adaptive_governor.lock_recover();
                            gov.user_profile.to_persisted()
                        };
                        write_json(&state.user_profile_path, &persisted, Some(0o600));
                    }
                }

                let fast = state
                    .fast_tick_until
                    .lock_recover()
                    .map(|t| Instant::now() < t)
                    .unwrap_or(false);
                last_cycle_end = Instant::now();
                lf_metrics.set_cycle_time_us(cycle_start.elapsed().as_micros() as u64);
                lf_metrics.commit();
                // Reactive: condvar.wait_timeout instead of thread::sleep.
                // Wakes immediately on reactor events; otherwise max 500ms (fast) or 2s (idle).
                let wait_duration = if fast {
                    Duration::from_millis(500)
                } else {
                    Duration::from_secs(2)
                };
                {
                    let (lock, cvar) = &*state.cycle_condvar;
                    let mut triggered = lock.lock_recover();
                    if !*triggered {
                        let (mut guard, _) = cvar
                            .wait_timeout(triggered, wait_duration)
                            .unwrap_or_else(|e| e.into_inner());
                        *guard = false;
                    } else {
                        *triggered = false;
                    }
                }
            }

            // Persist Markov chain + Holt-Winters + SignalIntelligence state on shutdown.
            focus_markov.persist();
            holt_winters.persist(&hw_path);
            signal_intel.persist(std::path::Path::new(signal_intelligence_path()));

            // Revert sysctls to defaults on shutdown.
            {
                let revert_actions = sysctl_governor.revert_to_defaults();
                if !revert_actions.is_empty() {
                    let caps = detect_capabilities();
                    let mut frozen_dummy = HashSet::new();
                    let outcomes = execute_actions(
                        revert_actions,
                        &caps,
                        &journal_path,
                        &mut frozen_dummy,
                        &[],
                        &[],
                        None,
                    );
                    if outcomes.failures == 0 {
                        sysctl_governor.mark_reverted();
                    } else {
                        eprintln!(
                            "sysctl-governor: WARNING: {} revert failures; \
                             persisted defaults retained for next startup",
                            outcomes.failures
                        );
                    }
                } else {
                    // No actions to revert — clean up persisted defaults anyway.
                    sysctl_governor.mark_reverted();
                }
            }

            // BUG 19 fix: unfreeze all frozen processes on daemon shutdown so
            // processes don't remain stopped if the daemon exits or crashes.
            {
                let frozen_state = state.frozen_state.lock_recover();
                let pids: Vec<u32> = frozen_state.keys().copied().collect();
                if !pids.is_empty() {
                    unfreeze_pids(pids.into_iter());
                }
            }

            // Unfreeze any PIDs held by the resource interrupt handler.
            {
                let interrupt_pids: Vec<u32> = state
                    .resource_interrupt
                    .interrupt_frozen_pids
                    .lock_recover()
                    .drain()
                    .collect();
                if !interrupt_pids.is_empty() {
                    unfreeze_pids(interrupt_pids.into_iter());
                }
            }

            let _ = fs::remove_file(socket_path());
        }
    }

    Ok(())
}
