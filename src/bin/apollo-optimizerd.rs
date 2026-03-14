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
    decide_memory_action, query_memory_profile, MemoryAction,
};
use apollo_optimizer::engine::decide_actions::decide_actions;
use apollo_optimizer::engine::energy::EnergyTracker;
use apollo_optimizer::engine::execute_actions::execute_actions;
use apollo_optimizer::engine::foreground::{ForegroundDetector, ForegroundState};
use apollo_optimizer::engine::gpu_manager::{GPUManager, GPUMetrics, GPUPowerState};
use apollo_optimizer::engine::hw_bayes::HwFeatures;
use apollo_optimizer::engine::hw_predictor::{sample_hw_pressure, HwPressure};
use apollo_optimizer::engine::iokit_sensors::HardwareSnapshot;
use apollo_optimizer::engine::kqueue_pressure;
use apollo_optimizer::engine::llm::{
    append_jsonl, delete_file_best_effort, feedback_path_root, load_repo_config, policy_path_root,
    read_json, state_paths_root, suggestions_path_root, write_json, write_secret, FeedbackEntry,
    LearnedPolicy, LlmAdvisor, LlmConfig, LlmState,
};
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::lse_counters::LockFreeMetrics;
use apollo_optimizer::engine::mach_qos::{MachQoSManager, SchedulingTier};
use apollo_optimizer::engine::memory_analyzer::MemoryAnalyzer;
use apollo_optimizer::engine::network_monitor::NetworkMonitor;
use apollo_optimizer::engine::network_optimizer::{NetworkOptimizer, NetworkProfile};
use apollo_optimizer::engine::outcome_tracker::OutcomeTracker;
use apollo_optimizer::engine::overflow_guard::OverflowGuard;
use apollo_optimizer::engine::power_management::{detect_battery_status, PowerManager};
use apollo_optimizer::engine::predictive_agent::{AgentContext, Intervention, PredictiveAgent};
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
    critical_background_processes, enforce_limits_with_budget, pattern_conflicts_with_protected,
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
use apollo_optimizer::engine::zombie_hunter::HuntSnapshot;
use chrono::{DateTime, Duration as ChronoDuration, Local, Timelike, Utc};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sysinfo::ProcessStatus;

const FREEZE_TTL_SECS: i64 = 10 * 60;
const REACTOR_FAST_TICK_SECS: u64 = 30;

/// Seed policy embedded at compile time — guarantees Brave, Claude, Warp, etc.
/// are always in interactive_patterns even on fresh installs or corrupt disk policy.
static SEED_POLICY: &str = include_str!("../../policy_clean.json");

/// Merge seed policy patterns into `policy` as a floor.
/// Seed patterns are always present — they can be added to but never removed.
fn merge_seed_into(policy: &mut LearnedPolicy) {
    let seed: LearnedPolicy =
        serde_json::from_str(SEED_POLICY).expect("BUG: embedded policy_clean.json is invalid");
    for pat in &seed.interactive_patterns {
        if !policy.interactive_patterns.contains(pat) {
            policy.interactive_patterns.push(pat.clone());
        }
    }
    for pat in &seed.noise_patterns {
        if !policy.noise_patterns.contains(pat) {
            policy.noise_patterns.push(pat.clone());
        }
    }
    for pat in &seed.protected_patterns {
        if !policy.protected_patterns.contains(pat) {
            policy.protected_patterns.push(pat.clone());
        }
    }
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

fn predictive_agent_path() -> &'static str {
    if unsafe { libc::geteuid() } == 0 {
        "/var/lib/apollo/predictive_agent.json"
    } else {
        "/tmp/apollo-predictive_agent.json"
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
    reactor_events_total: Arc<Mutex<u64>>,
    reactor_events_mem: Arc<Mutex<u64>>,
    reactor_events_thermal: Arc<Mutex<u64>>,
    reactor_events_spawn: Arc<Mutex<u64>>,
    reactor_events_power: Arc<Mutex<u64>>,
    reactor_last_event_at: Arc<Mutex<Option<DateTime<Utc>>>>,
    reactor_last_error: Arc<Mutex<Option<String>>>,
    reactor_mode: Arc<Mutex<String>>,
    reactor_health: Arc<Mutex<String>>,
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
    usage_last_persist_at: Arc<Mutex<Option<DateTime<Utc>>>>,
    usage_promotions_day: Arc<Mutex<Option<String>>>,
    usage_promotions_today: Arc<Mutex<u32>>,

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

fn run_reactor(state: SharedState) -> anyhow::Result<()> {
    unsafe {
        let kq = libc::kqueue();
        if kq == -1 {
            *state.reactor_last_error.lock_recover() = Some("kqueue failed".to_string());
            return Ok(());
        }

        // EVFILT_VM / NOTE_VM_PRESSURE
        let mem_kev = libc::kevent {
            ident: 0,
            filter: -12, // EVFILT_VM (Darwin)
            flags: libc::EV_ADD | libc::EV_ENABLE | libc::EV_CLEAR,
            fflags: 0x80000000, // NOTE_VM_PRESSURE
            data: 0,
            udata: 1 as *mut libc::c_void, // ID 1 = Memory
        };
        let _ = libc::kevent(kq, &mem_kev, 1, std::ptr::null_mut(), 0, std::ptr::null());

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
            *state.reactor_last_error.lock_recover() = Some(format!(
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
        let mut launch_fd: libc::c_int = 0;
        let mut launch_token: libc::c_int = 0;
        let launch_name =
            CString::new("com.apple.launchd.spawn").expect("static string should not contain NUL");
        let launch_reg = notify_register_file_descriptor(
            launch_name.as_ptr(),
            &mut launch_fd,
            0,
            &mut launch_token,
        );
        if launch_reg != 0 {
            *state.reactor_last_error.lock_recover() = Some(format!(
                "launch notify_register_file_descriptor failed: {}",
                launch_reg
            ));
        }
        if launch_fd > 0 {
            let kev = libc::kevent {
                ident: launch_fd as usize,
                filter: libc::EVFILT_READ,
                flags: libc::EV_ADD | libc::EV_ENABLE,
                fflags: 0,
                data: 0,
                udata: 3 as *mut libc::c_void, // ID 3 = Lifecycle
            };
            let _ = libc::kevent(kq, &kev, 1, std::ptr::null_mut(), 0, std::ptr::null());
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
            *state.reactor_last_error.lock_recover() = Some(format!(
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
            if n == 0 {
                // Timeout — no events within 1 second.  Continue the loop so the
                // condvar pulse above keeps the main loop aware the reactor is alive.
                continue;
            }
            if n < 0 {
                // kevent error (e.g. EINTR).  Record the error and retry.
                let errno = *libc::__error();
                if errno != libc::EINTR {
                    *state.reactor_last_error.lock_recover() =
                        Some(format!("kevent error errno={}", errno));
                }
                continue;
            }

            let id = out_ev.udata as usize;
            *state.reactor_events_total.lock_recover() += 1;
            *state.reactor_last_event_at.lock_recover() = Some(Utc::now());
            *state.reactor_health.lock_recover() = "ok".to_string();
            if id == 2 {
                // Drain thermal pipe
                let mut dummy: i32 = 0;
                let _ = libc::read(thermal_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                *state.reactor_events_thermal.lock_recover() += 1;
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
                *state.reactor_events_spawn.lock_recover() += 1;
            } else if id == 4 {
                let mut dummy: i32 = 0;
                let _ = libc::read(power_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                *state.reactor_events_power.lock_recover() += 1;
                // Signal resource sentinel for power source changes.
                state
                    .resource_interrupt
                    .power_signal
                    .store(true, Ordering::Release);
            } else if id == 1 {
                *state.reactor_events_mem.lock_recover() += 1;
                // Signal resource sentinel for memory pressure events.
                state
                    .resource_interrupt
                    .memory_signal
                    .store(true, Ordering::Release);
            }

            *state.reactor_event_weight.lock_recover() = 1.0;
            if state.reactor_mode.lock_recover().as_str() == "normal" {
                *state.fast_tick_until.lock_recover() =
                    Some(Instant::now() + Duration::from_secs(REACTOR_FAST_TICK_SECS));
            }

            {
                let mut metrics = state.metrics.lock_recover();
                metrics.reactor_pulses += 1;
            }

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
        if launch_fd > 0 {
            libc::close(launch_fd);
        }
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
            let metrics = state.metrics.lock_recover().clone();
            let blockers = state.last_blockers.lock_recover().clone();
            let thermal_state = state.thermal_state.lock_recover().clone();
            let throttle_level = state.throttle_level.lock_recover().clone();
            let governor = state.governor.lock_recover();
            let wake_state = state.wake_state.lock_recover();
            let grace_active = wake_state
                .post_wake_grace_until
                .map(|t| t > now)
                .unwrap_or(false);
            let grace_remaining = wake_state
                .post_wake_grace_until
                .and_then(|t| (t - now).to_std().ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let reactor_mode = state.reactor_mode.lock_recover().clone();
            let reactor_health = state.reactor_health.lock_recover().clone();
            let status = DaemonStatus {
                running: !state.stop.load(Ordering::Acquire),
                profile,
                latency_target,
                effective_profile: metrics.effective_profile,
                kill_switch: Path::new(kill_switch_path()).exists(),
                throttle_level,
                thermal_state,
                last_blockers: blockers,
                auto_profile_enabled: governor.auto_profile_enabled,
                base_profile: governor.base_profile,
                override_active: governor.manual_override.is_some(),
                override_expires_at: governor.manual_override.as_ref().map(|o| o.expires_at),
                transition_reason: governor.transition_reason.clone(),
                post_wake_grace_active: grace_active,
                post_wake_grace_remaining_secs: grace_remaining,
                last_wake_at: wake_state.last_wake_at,
                post_wake_policy: wake_state.post_wake_policy.clone(),
                reactor_mode,
                reactor_health,
                metrics,
                llm: Some(build_llm_status(state)),
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
                format!(
                    "reactor_mode: {}",
                    state.reactor_mode.lock_recover().clone()
                ),
                format!(
                    "reactor_health: {}",
                    state.reactor_health.lock_recover().clone()
                ),
                format!(
                    "swapusage_readable: {}",
                    std::process::Command::new("/usr/sbin/sysctl")
                        .args(["vm.swapusage"])
                        .output()
                        .map(|o| o.status.success())
                        .unwrap_or(false)
                ),
                format!(
                    "memory_pressure_readable: {}",
                    std::process::Command::new("/usr/bin/memory_pressure")
                        .args(["-Q"])
                        .output()
                        .map(|o| o.status.success())
                        .unwrap_or(false)
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
        model.update_from_snapshot(snapshot, now, interactive_proxy, jank_proxy, 10);
    }

    // Persist usage model periodically (every ~2 minutes).
    {
        let mut last = state.usage_last_persist_at.lock_recover();
        let due = last
            .map(|t| now - t > ChronoDuration::minutes(2))
            .unwrap_or(true);
        if due {
            {
                let model = state.usage_model.lock_recover();
                model.persist(&state.usage_model_path);
            }
            *last = Some(now);
        }
    }

    // Daily promotion counters (conservative).
    let today = Local::now().date_naive().to_string();
    {
        let mut day = state.usage_promotions_day.lock_recover();
        if day.as_deref() != Some(&today) {
            *day = Some(today.clone());
            *state.usage_promotions_today.lock_recover() = 0;
        }
    }

    let promotions_used = *state.usage_promotions_today.lock_recover();
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
        let mut used = state.usage_promotions_today.lock_recover();
        *used += applied;
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
    let mut rusage_map: HashMap<u32, (u64, u32)> = HashMap::new(); // pid → (idle_wakeups, mach_msgs)
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
                rusage_map.insert(pid_u32, (idle_wk, ti.messages_sent + ti.messages_received));
            } else {
                rusage_map.insert(pid_u32, (idle_wk, 0));
            }
        }
    }

    let mut proc_snaps = Vec::new();
    let mut hunt_snaps = Vec::new();

    for (pid, process) in sys.processes() {
        let pid_u32 = pid.as_u32();
        let name = process.name().to_string();
        let is_foreground = fg_family.contains(&pid_u32);
        let ppid = process.parent().map(|p| p.as_u32()).unwrap_or(0);
        let parent_alive = ppid > 0;
        let is_zombie = process.status() == ProcessStatus::Zombie;
        let rss = process.memory();
        let cpu = process.cpu_usage();

        // Real idle wakeups from proc_pid_rusage — the #1 signal for wasteful daemons.
        // Estimate wakeups/sec: idle_wakeups is cumulative, divide by uptime estimate.
        // Mach messages > 0 implies the process has active IPC (network, XPC, etc.)
        let (wakeups_per_sec, has_network_signal) = match rusage_map.get(&pid_u32) {
            Some(&(idle_wk, mach_msgs)) => {
                // Rough estimate: if idle_wakeups > 1000, it's a chatty daemon
                let wps = if idle_wk > 10_000 {
                    (idle_wk as f32 / 3600.0).min(100.0)
                } else if idle_wk > 100 {
                    (idle_wk as f32 / 7200.0).min(50.0)
                } else {
                    0.0
                };
                // Mach messages indicate IPC activity (XPC, network, etc.)
                let has_net = mach_msgs > 100;
                (wps, has_net)
            }
            None => (0.0, false),
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
                reactor_events_total: Arc::new(Mutex::new(0)),
                reactor_events_mem: Arc::new(Mutex::new(0)),
                reactor_events_thermal: Arc::new(Mutex::new(0)),
                reactor_events_spawn: Arc::new(Mutex::new(0)),
                reactor_events_power: Arc::new(Mutex::new(0)),
                reactor_last_event_at: Arc::new(Mutex::new(None)),
                reactor_last_error: Arc::new(Mutex::new(None)),
                reactor_mode: Arc::new(Mutex::new("normal".to_string())),
                reactor_health: Arc::new(Mutex::new("ok".to_string())),
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
                usage_last_persist_at: Arc::new(Mutex::new(None)),
                usage_promotions_day: Arc::new(Mutex::new(None)),
                usage_promotions_today: Arc::new(Mutex::new(0)),

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
            let mut power_mgr = PowerManager::new();
            let mut proc_recovery = ProcessRecoveryManager::new();
            let mut swap_predictor = SwapPredictor::new();
            let mut network_monitor = NetworkMonitor::new();
            let mut sysctl_governor = SysctlGovernor::new(is_root);
            let mut thermal_mgr = ThermalManager::new();
            let mut wake_storm = WakeStormDetector::new();
            // GPU thermal monitoring: integrates with thermal_manager for GPU-aware decisions.
            let gpu_mgr = GPUManager::new();
            // Network profile optimizer: complements sysctl_governor with profile-driven tuning.
            let net_optimizer = NetworkOptimizer::new();
            // Foreground detection: replaces get_foreground_app() with cached, richer detection.
            // Wrapped in Arc so it can be shared with the resource sentinel thread.
            let fg_detector = Arc::new(ForegroundDetector::new());
            // Per-app energy estimation: accumulates energy attribution each cycle.
            let mut energy_tracker = EnergyTracker::new();
            let mut outcome_tracker = OutcomeTracker::new();
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
            let mut overflow_guard =
                OverflowGuard::load_or_default(std::path::Path::new(overflow_history_path()));
            // Predictive agent: LinUCB contextual bandit for proactive interventions.
            let mut predictive_agent =
                PredictiveAgent::load_or_default(std::path::Path::new(predictive_agent_path()));
            // Signal intelligence: Kalman + CUSUM + Entropy + Hazard + LV + MPC.
            let mut signal_intel = SignalIntelligence::new();
            // Audit fix #6: Multi-phase thermal bail-out with hysteresis.
            let mut thermal_bailout = ThermalBailout::new();
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
                        *state.reactor_mode.lock_recover() = "degraded".to_string();
                        *state.reactor_health.lock_recover() = "stalled".to_string();
                        *state.fast_tick_until.lock_recover() = None;
                    } else {
                        // Reactor thread is alive; health tracks actual events.
                        let current_mode = state.reactor_mode.lock_recover().clone();
                        if current_mode == "degraded" {
                            *state.reactor_mode.lock_recover() = "normal".to_string();
                            *state.reactor_health.lock_recover() = "ok".to_string();
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
                            *state.reactor_health.lock_recover() = "collector-stalled".to_string();
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

                // Adaptive snapshot: use lightweight path (no disk/net, direct sysctl)
                // when hw pressure was Nominal last cycle AND memory is not stressed.
                // Every 30 cycles force a full refresh to pick up disk/net changes.
                let cached_mem_pressure = pressure_collector.latest().memory_pressure;
                let use_light = last_hw_pressure == HwPressure::Nominal
                    && cached_mem_pressure < 0.40
                    && cycle_count % 30 != 0;
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
                for snap in proc_snaps.iter().take(50) {
                    let profile = mem_analyzer.analyze_process(
                        snap.pid,
                        &snap.name,
                        snap.rss_bytes,
                        snap.rss_bytes, // vms not tracked at this level; use rss as proxy
                        0,              // page_faults not available from sysinfo
                    );
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

                // HwPredictor: sample hardware signals every 5 cycles (~2.5s at normal rate).
                // Runs in <50ms and gives advance warning before metrics APIs catch up.
                let (hw_pressure, jitter_us, hw_features) = if cycle_count % 5 == 0 {
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
                            *state.fast_tick_until.lock_recover() =
                                Some(Instant::now() + Duration::from_secs(15));
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

                // Online usage learning (root-only, no UI sensors): infer frequently-used apps
                // and processes correlated with jank, then promote patterns conservatively.
                usage_learning_tick(
                    &state,
                    &snapshot,
                    !foreground_idle && foreground_app.is_some(),
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
                                        let heavy: Vec<String> = snapshot
                                            .top_processes
                                            .iter()
                                            .take(8)
                                            .map(|p| p.name.clone())
                                            .collect();
                                        overflow_guard.record_event(
                                            snapshot.pressure.memory_pressure,
                                            snapshot.pressure.swap_delta_bytes_per_sec,
                                            &heavy,
                                            &format!("kqueue-{:?}", level),
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
                let pressure_ram = (snapshot.pressure.memory_pressure + hw_boost).clamp(0.0, 1.0);
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
                    metrics.reactor_events_total = *state.reactor_events_total.lock_recover();
                    metrics.reactor_events_mem = *state.reactor_events_mem.lock_recover();
                    metrics.reactor_events_thermal = *state.reactor_events_thermal.lock_recover();
                    metrics.reactor_events_spawn = *state.reactor_events_spawn.lock_recover();
                    metrics.reactor_events_power = *state.reactor_events_power.lock_recover();
                    metrics.reactor_last_event_at = *state.reactor_last_event_at.lock_recover();
                    metrics.reactor_last_error = state.reactor_last_error.lock_recover().clone();
                    metrics.reactor_mode = state.reactor_mode.lock_recover().clone();
                    metrics.reactor_health = state.reactor_health.lock_recover().clone();
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
                    // Use cycle-level hardware snapshot for per-process power.
                    metrics.energy_cpu_watts = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.cpu_watts)
                        .map(|w| w as f64);
                    metrics.energy_gpu_watts = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.gpu_watts)
                        .map(|w| w as f64);
                    metrics.energy_package_watts = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.package_watts)
                        .map(|w| w as f64);
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
                }

                let mut governor = state.governor.lock_recover();
                let governor_decision = governor.evaluate(GovernorInput {
                    cpu_pressure: pressure_cpu,
                    ram_pressure: pressure_ram,
                    interactive_wait_ratio: pressure_wait,
                    reactor_event_weight: *reactor_weight,
                    thermal_constrained: matches!(
                        snapshot.pressure.thermal_level.as_str(),
                        "serious" | "critical"
                    ),
                    dev_session_active,
                    interactive_heavy,
                    context_switch_burst,
                });
                if governor_decision.transition_reason.contains("floor") {
                    state.metrics.lock_recover().profile_floor_hits += 1;
                }
                let current_profile = governor_decision.effective_profile;
                write_governor_state(&governor_state_path, &governor);
                drop(governor);

                let (decide_interactive, decide_noise) = {
                    let policy = state.learned_policy.lock_recover();
                    (
                        policy.interactive_patterns.clone(),
                        policy.noise_patterns.clone(),
                    )
                };
                // Thresholds adaptativos: más conservadores si hubo overflows recientes,
                // y aún más si hay una compilación activa.
                let mut overflow_thresholds = overflow_guard.thresholds(&all_proc_names);

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
                    signal_intel.tick(
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
                    )
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
                    );
                    let mut intervention = predictive_agent.select_action(&agent_ctx);

                    // ── Hazard override: if P(OOM) > 30%, don't just Observe. ────
                    if intervention == Intervention::Observe && signal_digest.p_oom_30s > 0.30 {
                        // Use MPC recommendation as the override action.
                        intervention = Intervention::from_index(signal_digest.mpc_recommendation);
                    }

                    // ── Monopoly risk: if a single process is hogging RAM, prefer throttle. ──
                    if intervention == Intervention::Observe && signal_digest.monopoly_risk > 0.5 {
                        intervention = Intervention::PreThrottleNoise;
                    }

                    // ── Kalman predicted pressure: if 5s prediction > 0.85, tighten. ────
                    if intervention == Intervention::Observe
                        && signal_digest.pressure_predicted_5s > 0.85
                    {
                        intervention = Intervention::TightenThresholds;
                    }

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
                    )
                };
                *state.last_blockers.lock_recover() = decision.blockers.clone();
                *state.thermal_state.lock_recover() = context_to_thermal(decision.context);

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

                // Build critical_pids set for heuristic merge (same logic as decide_actions)
                let heuristic_critical_pids: HashSet<u32> = {
                    let sys = collector.system();
                    let critical_pats = critical_background_processes();
                    let protected_pats = protected_processes();
                    // Incluir los protected_patterns de la policy aprendida —
                    // sin esto, procesos como com.apple.DriverKit-AppleBCMWLAN
                    // son throttleados por WakeStormDetector aunque estén protegidos.
                    let policy_protected = state
                        .learned_policy
                        .lock_recover()
                        .protected_patterns
                        .clone();
                    let mut cpids: HashSet<u32> = HashSet::new();
                    for (pid, process) in sys.processes() {
                        let name = process.name().to_string();
                        if critical_pats.iter().any(|p| name.contains(p))
                            || protected_pats.iter().any(|p| name.contains(p))
                            || policy_protected.iter().any(|p| name.contains(p.as_str()))
                        {
                            cpids.insert(pid.as_u32());
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
                actions.extend(heuristic_actions);

                // Survival Mode: active when memory pressure is critical or swap is thrashing.
                // swap_delta_bps > 1MB/s means we're actively writing to swap (thrashing).
                let survival_mode = snapshot.pressure.memory_pressure > 0.85
                    || snapshot.pressure.swap_delta_bytes_per_sec > 1_000_000.0;

                // Overflow guard: registrar si estamos en survival mode para ajustar thresholds.
                if survival_mode {
                    let heavy: Vec<String> = snapshot
                        .top_processes
                        .iter()
                        .take(8)
                        .map(|p| p.name.clone())
                        .collect();
                    overflow_guard.record_event(
                        snapshot.pressure.memory_pressure,
                        snapshot.pressure.swap_delta_bytes_per_sec,
                        &heavy,
                        "survival-mode",
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
                overflow_guard.tick_decay();

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

                // WakeStormDetector: throttle processes generating excessive wakeups.
                let storms = wake_storm.detect_storms();
                for storm in &storms {
                    if !heuristic_critical_pids.contains(&storm.pid) {
                        let (ss, su) = pid_start_time(storm.pid);
                        actions.push(RootAction::ThrottleProcess {
                            pid: storm.pid,
                            name: storm.name.clone(),
                            aggressive: false,
                            reason: format!(
                                "wake-storm: {:.0} wakeups/sec",
                                storm.wakeups_per_second
                            ),
                            start_sec: ss,
                            start_usec: su,
                        });
                    }
                }

                // Paging hints: send memory pressure signal to idle memory hoarders.
                // Triggers the process's pressure handler to release caches.
                let mem_pressure = snapshot.pressure.memory_pressure;
                let swap_active = snapshot.pressure.swap_used_bytes > 256 * 1024 * 1024;
                if mem_pressure > 0.45 && swap_active {
                    // Build foreground process family to avoid hinting active renderers.
                    let fg_pids = build_foreground_family(foreground_pid, &process_tree);
                    // Collect interactive pattern names — never hint these even in background.
                    let interactive_pats: Vec<String> = state
                        .learned_policy
                        .lock_recover()
                        .interactive_patterns
                        .clone();
                    for snap in proc_snaps.iter().take(100) {
                        if heuristic_critical_pids.contains(&snap.pid) {
                            continue;
                        }
                        // Skip interactive apps (e.g. Claude Helper, Spotify Helper, Code Helper).
                        if interactive_pats
                            .iter()
                            .any(|p| snap.name.contains(p.as_str()))
                        {
                            continue;
                        }
                        // Classic hoarder: large, idle, no GUI window.
                        let is_hoarder = snap.rss_bytes > 120 * 1024 * 1024
                            && snap.secs_since_user_interaction > 120
                            && !snap.has_gui_window;
                        // Background renderer: browser/Electron helper not in
                        // the foreground app's process family and not interactive.
                        let is_bg_renderer = snap.rss_bytes > 60 * 1024 * 1024
                            && snap.secs_since_user_interaction > 120
                            && (snap.name.contains("Helper (Renderer)")
                                || snap.name.contains("Helper (Plugin)")
                                || snap.name.contains(" Renderer"))
                            && !fg_pids.contains(&snap.pid);
                        if is_hoarder || is_bg_renderer {
                            actions.push(RootAction::SetMemorystatus {
                                pid: snap.pid,
                                priority: -1,
                                reason: format!(
                                    "memory-pressure hint: {}MB RSS bg={}",
                                    snap.rss_bytes / 1024 / 1024,
                                    is_bg_renderer
                                ),
                            });
                        }
                    }
                }

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
                        let mut frozen_state = state.frozen_state.lock_recover();
                        let expired: Vec<u32> = frozen_state
                            .iter()
                            .filter(|(pid, entry)| {
                                now.signed_duration_since(entry.frozen_at).num_seconds()
                                    > FREEZE_TTL_SECS
                                    && !interrupt_pids.contains(pid)
                            })
                            .map(|(pid, _)| *pid)
                            .collect();
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

                    // Audit fix #3: Compressor-aware freeze decisions.
                    // Query memory profile for freeze candidates and convert to
                    // PressureHint if the process has low compression ratio.
                    let confirmed_actions: Vec<RootAction> = confirmed_actions.into_iter().filter_map(|a| {
                        if let RootAction::FreezeProcess { pid, name: _, ref reason, .. } = a {
                            if let Some(profile) = query_memory_profile(pid) {
                                match decide_memory_action(&profile, metrics.memory_pressure) {
                                    MemoryAction::PressureHint => {
                                        Some(RootAction::SetMemorystatus {
                                            pid,
                                            priority: -1,
                                            reason: format!(
                                                "{} (compressor: ratio={:.1} purgeable={}MB → hint)",
                                                reason,
                                                profile.compression_ratio,
                                                profile.purgeable_bytes / 1024 / 1024,
                                            ),
                                        })
                                    }
                                    MemoryAction::Skip => None,
                                    MemoryAction::Freeze => Some(a),
                                }
                            } else {
                                Some(a)
                            }
                        } else {
                            Some(a)
                        }
                    }).collect();

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

                // Outcome tracker tick: resuelve outcomes de hace 30s, actualiza pesos y energy savings.
                {
                    let batch = outcome_tracker.tick(snapshot.pressure.memory_pressure);
                    if batch.savings_watts > 0.0 {
                        energy_tracker.record_savings(batch.savings_watts, 30.0);
                    }
                    // Sincroniza pesos Bayesianos a la LearnedPolicy persistida.
                    if !batch.effective_names.is_empty() || !batch.low_value_names.is_empty() {
                        let mut policy = state.learned_policy.lock_recover();
                        for (name, weight) in &outcome_tracker.weights {
                            policy.pattern_weights.insert(name.clone(), weight.clone());
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
                            let tier = if thermal_emergency {
                                // Force all to E-Cores during thermal emergency
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
                        (overflow_guard.history.threshold_offset * 100.0).round() as i32;
                    metrics.overflow_build_mode = overflow_thresholds.build_mode;

                    write_metrics(&metrics_path, &metrics);
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
