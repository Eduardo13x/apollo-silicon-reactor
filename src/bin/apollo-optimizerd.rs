use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;
use apollo_optimizer::collector::SystemCollector;
use apollo_optimizer::engine::capabilities::detect_capabilities;
use apollo_optimizer::engine::decide_actions::decide_actions;
use apollo_optimizer::engine::execute_actions::execute_actions;
use apollo_optimizer::engine::llm::{
    append_jsonl, delete_file_best_effort, feedback_path_root, load_repo_config, policy_path_root,
    read_json, state_paths_root, suggestions_path_root, write_json, write_secret, FeedbackEntry,
    LearnedPolicy, LlmAdvisor, LlmConfig, LlmState,
};
use apollo_optimizer::engine::profile_governor::{
    GovernorInput, GovernorPersisted, ProfileGovernor,
};
use apollo_optimizer::engine::protocol::{DaemonRequest, DaemonResponse};
use apollo_optimizer::engine::safety::{critical_background_processes, enforce_limits_with_budget};
use apollo_optimizer::engine::types::{
    BlockerScore, DaemonStatus, InteractiveContext, LatencyTarget, LearnedPolicyStatus, LlmRunMode,
    LlmStatus, OptimizationProfile, ProfileTransition, RootAction, RuntimeMetrics, SafetyPolicy,
    UsageResponse,
};
use apollo_optimizer::engine::adaptive_governor::{AdaptiveGovernor, GovernerDecision, ProcessDecision};
use apollo_optimizer::engine::user_profile::{UserProfile, UserProfilePersisted};
use apollo_optimizer::engine::iokit_sensors::{HardwareSnapshot, IOKitSensorReader};
use apollo_optimizer::engine::mach_qos::{MachQoSManager, SchedulingTier};
use apollo_optimizer::engine::process_classifier::{ProcessSnapshot, ProcessTier};
use apollo_optimizer::engine::usage_model::{usage_model_path_root, UsageModel};
use apollo_optimizer::engine::zombie_hunter::HuntSnapshot;
use chrono::{DateTime, Duration as ChronoDuration, Local, Timelike, Utc};
use sysinfo::ProcessStatus;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use apollo_optimizer::engine::analytics::AnalyticsEngine;
use apollo_optimizer::engine::gpu_manager::GPUManager;
use apollo_optimizer::engine::memory_analyzer::MemoryAnalyzer;
use apollo_optimizer::engine::network_optimizer::NetworkOptimizer;
use apollo_optimizer::engine::power_management::PowerManager;
use apollo_optimizer::engine::process_recovery::ProcessRecoveryManager;
use apollo_optimizer::engine::swap_predictor::SwapPredictor;
use apollo_optimizer::engine::thermal_manager::ThermalManager;
use apollo_optimizer::engine::wake_storm_detector::WakeStormDetector;

const FREEZE_TTL_SECS: i64 = 10 * 60;
const REACTOR_FAST_TICK_SECS: u64 = 30;

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
    frozen: Arc<Mutex<HashSet<u32>>>,
    frozen_since: Arc<Mutex<HashMap<u32, DateTime<Utc>>>>,
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
    timeline: Arc<Mutex<Vec<ProfileTransition>>>,
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
    iokit_reader: Arc<IOKitSensorReader>,
    last_hw_snapshot: Arc<Mutex<Option<HardwareSnapshot>>>,
    iokit_cycle_counter: Arc<Mutex<u32>>,

    // ML Ligero
    discrepancy_log_path: PathBuf,
    user_profile_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FrozenPidEntry {
    pid: u32,
    since: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FrozenStatePersisted {
    frozen: Vec<FrozenPidEntry>,
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
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(metrics) {
        let _ = fs::write(path, json);
    }
}

fn write_governor_state(path: &Path, governor: &ProfileGovernor) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&governor.to_persisted()) {
        let _ = fs::write(path, json);
    }
}

fn load_governor_state(path: &Path, fallback_profile: OptimizationProfile) -> ProfileGovernor {
    if let Ok(data) = fs::read_to_string(path) {
        if let Ok(state) = serde_json::from_str::<GovernorPersisted>(&data) {
            return ProfileGovernor::from_persisted(state);
        }
    }
    ProfileGovernor::new(fallback_profile)
}

fn append_timeline(path: &Path, transition: &ProfileTransition) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(path) {
        if let Ok(line) = serde_json::to_string(transition) {
            let _ = writeln!(f, "{}", line);
        }
    }
    rotate_timeline(path);
}

fn write_wake_state(path: &Path, state: &WakeRuntimeState) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let persisted = WakeStatePersisted {
        last_wake_at: state.last_wake_at,
        post_wake_grace_until: state.post_wake_grace_until,
        post_wake_policy: state.post_wake_policy.clone(),
    };
    if let Ok(json) = serde_json::to_string_pretty(&persisted) {
        let _ = fs::write(path, json);
    }
}

fn load_wake_state(path: &Path) -> WakeRuntimeState {
    let now = Utc::now();
    if let Ok(data) = fs::read_to_string(path) {
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

fn write_frozen_state(path: &Path, frozen_since: &HashMap<u32, DateTime<Utc>>) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let persisted = FrozenStatePersisted {
        frozen: frozen_since
            .iter()
            .map(|(pid, since)| FrozenPidEntry {
                pid: *pid,
                since: *since,
            })
            .collect(),
    };
    if let Ok(json) = serde_json::to_string_pretty(&persisted) {
        let _ = fs::write(path, json);
    }
}

fn load_frozen_state(path: &Path) -> HashMap<u32, DateTime<Utc>> {
    if let Ok(data) = fs::read_to_string(path) {
        if let Ok(state) = serde_json::from_str::<FrozenStatePersisted>(&data) {
            return state.frozen.into_iter().map(|e| (e.pid, e.since)).collect();
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
            *state
                .reactor_last_error
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some("kqueue failed".to_string());
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
            *state
                .reactor_last_error
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(format!(
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
            *state
                .reactor_last_error
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(format!(
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
            *state
                .reactor_last_error
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(format!(
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
        let mut timeout = libc::timespec {
            tv_sec: 1,
            tv_nsec: 0,
        };
        while !state.stop.load(Ordering::SeqCst) {
            let n = libc::kevent(
                kq,
                std::ptr::null(),
                0,
                &mut out_ev,
                1,
                &mut timeout as *mut libc::timespec,
            );
            if n <= 0 {
                continue;
            }

            let id = out_ev.udata as usize;
            *state
                .reactor_events_total
                .lock()
                .unwrap_or_else(|e| e.into_inner()) += 1;
            *state
                .reactor_last_event_at
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(Utc::now());
            *state
                .reactor_health
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = "ok".to_string();
            if id == 2 {
                // Drain thermal pipe
                let mut dummy: i32 = 0;
                let _ = libc::read(thermal_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                *state
                    .reactor_events_thermal
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) += 1;
                let level = match dummy {
                    0 => "nominal",
                    1 => "moderate",
                    2 => "serious",
                    _ => "critical",
                };
                *state
                    .thermal_level_real
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = level.to_string();
            } else if id == 3 {
                let mut dummy: i32 = 0;
                let _ = libc::read(launch_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                *state
                    .reactor_events_spawn
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) += 1;
            } else if id == 4 {
                let mut dummy: i32 = 0;
                let _ = libc::read(power_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                *state
                    .reactor_events_power
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) += 1;
            } else if id == 1 {
                *state
                    .reactor_events_mem
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) += 1;
            }

            *state
                .reactor_event_weight
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = 1.0;
            if state
                .reactor_mode
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_str()
                == "normal"
            {
                *state
                    .fast_tick_until
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) =
                    Some(Instant::now() + Duration::from_secs(REACTOR_FAST_TICK_SECS));
            }

            if let Ok(mut metrics) = state.metrics.lock() {
                metrics.reactor_pulses += 1;
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
    if fs::metadata(path)
        .map(|m| m.len() > MAX_BYTES)
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
            } => {
                throttle_suppressed += 1;
                out.push(RootAction::ThrottleProcess {
                    pid,
                    name,
                    aggressive: false,
                    reason,
                });
            }
            _ => out.push(action),
        }
    }

    (out, throttle_suppressed, freeze_suppressed)
}

fn handle_client(mut stream: UnixStream, state: &SharedState) {
    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    let response = match reader.read_line(&mut line) {
        Ok(_) => match serde_json::from_str::<DaemonRequest>(&line) {
            Ok(req) => process_request(req, state),
            Err(e) => DaemonResponse::Error {
                message: format!("invalid request: {}", e),
            },
        },
        Err(e) => DaemonResponse::Error {
            message: format!("read error: {}", e),
        },
    };

    if let Ok(text) = serde_json::to_string(&response) {
        let _ = writeln!(stream, "{}", text);
    }
}

fn process_request(req: DaemonRequest, state: &SharedState) -> DaemonResponse {
    match req {
        DaemonRequest::GetStatus => {
            let now = Utc::now();
            let profile = *state.profile.lock().unwrap_or_else(|e| e.into_inner());
            let latency_target = *state
                .latency_target
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let metrics = state
                .metrics
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let blockers = state
                .last_blockers
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let thermal_state = state
                .thermal_state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let throttle_level = state
                .throttle_level
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let governor = state.governor.lock().unwrap_or_else(|e| e.into_inner());
            let wake_state = state.wake_state.lock().unwrap_or_else(|e| e.into_inner());
            let grace_active = wake_state
                .post_wake_grace_until
                .map(|t| t > now)
                .unwrap_or(false);
            let grace_remaining = wake_state
                .post_wake_grace_until
                .and_then(|t| (t - now).to_std().ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let reactor_mode = state
                .reactor_mode
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let reactor_health = state
                .reactor_health
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let status = DaemonStatus {
                running: !state.stop.load(Ordering::SeqCst),
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
        DaemonRequest::GetMetrics => DaemonResponse::Metrics(
            state
                .metrics
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
        ),
        DaemonRequest::GetTopBlockers => DaemonResponse::TopBlockers(
            state
                .last_blockers
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
        ),
        DaemonRequest::GetProfileTimeline => DaemonResponse::ProfileTimeline(
            state
                .timeline
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
        ),
        DaemonRequest::GetCapabilities => DaemonResponse::Capabilities(detect_capabilities()),
        DaemonRequest::SetProfile {
            profile,
            ttl_minutes,
        } => {
            let ttl = ttl_minutes.unwrap_or(20);
            state
                .governor
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .set_manual_override(profile, ttl, "cli-set-profile".to_string());
            DaemonResponse::Ok
        }
        DaemonRequest::SetLatencyTarget { target } => {
            *state
                .latency_target
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = target;
            DaemonResponse::Ok
        }
        DaemonRequest::SetAutoProfile { enabled } => {
            state
                .governor
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .set_auto_profile(enabled);
            DaemonResponse::Ok
        }
        DaemonRequest::ClearProfileOverride => {
            state
                .governor
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clear_manual_override();
            DaemonResponse::Ok
        }
        DaemonRequest::Restore => {
            let mut frozen = state.frozen.lock().unwrap_or_else(|e| e.into_inner());
            for pid in frozen.iter() {
                unsafe {
                    libc::kill(*pid as i32, libc::SIGCONT);
                }
            }
            frozen.clear();
            let _ = fs::remove_file(kill_switch_path());
            DaemonResponse::Ok
        }
        DaemonRequest::PanicRestore => {
            let _ = File::create(kill_switch_path());
            state
                .governor
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .set_auto_profile(false);
            let mut frozen = state.frozen.lock().unwrap_or_else(|e| e.into_inner());
            for pid in frozen.iter() {
                unsafe {
                    libc::kill(*pid as i32, libc::SIGCONT);
                }
            }
            frozen.clear();
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
                    state
                        .reactor_mode
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone()
                ),
                format!(
                    "reactor_health: {}",
                    state
                        .reactor_health
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone()
                ),
                format!(
                    "swapusage_readable: {}",
                    std::process::Command::new("sysctl")
                        .args(["vm.swapusage"])
                        .output()
                        .map(|o| o.status.success())
                        .unwrap_or(false)
                ),
                format!(
                    "memory_pressure_readable: {}",
                    std::process::Command::new("memory_pressure")
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
            let model = state.usage_model.lock().unwrap_or_else(|e| e.into_inner());
            let report = model.top_report(limit);
            DaemonResponse::Usage(UsageResponse::Top(report))
        }
        DaemonRequest::UsageExplain { name } => {
            let model = state.usage_model.lock().unwrap_or_else(|e| e.into_inner());
            match model.entry_summary(&name) {
                Some(s) => DaemonResponse::Usage(UsageResponse::Explain(s)),
                None => DaemonResponse::Error {
                    message: "usage entry not found".to_string(),
                },
            }
        }
        DaemonRequest::LlmSetKey { api_key, ttl_days } => {
            let now = Utc::now();
            let expires = now + ChronoDuration::days(ttl_days as i64);
            if write_secret(&state.llm_key_path, api_key.trim()).is_err() {
                return DaemonResponse::Error {
                    message: "failed to write llm key".to_string(),
                };
            }
            {
                let mut llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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
                let mut llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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
                let llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
                if !llm_state.training_active() {
                    return DaemonResponse::LlmTestResult {
                        ok: false,
                        http_status: None,
                        error: Some("training not active (enable + ttl)".to_string()),
                        suggestion: None,
                    };
                }
            }

            let api_key = match fs::read_to_string(&state.llm_key_path) {
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
            snapshot.pressure.thermal_level = state
                .thermal_level_real
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();

            // Record attempt immediately.
            {
                let mut llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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
            match advisor.call_raw(&snapshot, &api_key) {
                Ok(suggestion) => {
                    {
                        let mut llm_state =
                            state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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
                        let mut llm_state =
                            state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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
            let policy = state
                .learned_policy
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            DaemonResponse::LearnedPolicy(policy)
        }
        DaemonRequest::Feedback { rating, note } => {
            let entry = FeedbackEntry {
                at: Utc::now(),
                rating,
                note,
            };
            append_jsonl(&state.feedback_path, &entry);
            DaemonResponse::Ok
        }
    }
}

fn build_llm_status(state: &SharedState) -> LlmStatus {
    let llm_cfg = load_repo_config(&state.config_path)
        .llm
        .unwrap_or_else(|| state.llm_cfg.as_ref().clone());
    let enabled_from_disk = llm_cfg.enabled();
    let llm_state = state
        .llm_state
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let policy = state
        .learned_policy
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();

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
) {
    let now = Utc::now();
    let has_key = state.llm_key_path.exists();

    // TTL housekeeping: if training expired, disable and delete key.
    {
        let mut llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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

    let api_key = match fs::read_to_string(&state.llm_key_path) {
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
        let mut llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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
    let (ws_thresh, mem_thresh, swap_thresh_bps, cycles_needed) = match mode {
        LlmRunMode::Sensitive => (35.0_f32, 0.75_f64, 20.0 * 1024.0 * 1024.0, 3_u32),
        LlmRunMode::Strict => (50.0_f32, 0.85_f64, 50.0 * 1024.0 * 1024.0, 5_u32),
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
        let llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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

    if !trigger_active {
        // Bootcamp sampling: even when the system is "fine", take an occasional sample call
        // so the teacher can learn normal workload patterns.
        let sampling_due = {
            let llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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

        let mut llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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
        let mut llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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
        let mut llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());

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
    let suggestion_res = advisor.call_raw(snapshot, &api_key);

    // Apply suggestion and persist state.
    match suggestion_res {
        Ok(suggestion) => {
            let accepted = suggestion.confidence >= llm_cfg.min_confidence();
            {
                let mut llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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
                let mut gov = state.governor.lock().unwrap_or_else(|e| e.into_inner());
                if gov.manual_override.is_none() {
                    gov.set_manual_override(p, 20, "llm-reactive".to_string());
                }
            }
            // 2) Latency target.
            if let Some(t) = suggestion.suggested_latency_target {
                *state
                    .latency_target
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = t;
            }

            // 3) Learned patterns: merge with daily cap.
            {
                let mut llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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

                let mut policy = state
                    .learned_policy
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());

                let mut added = 0u32;
                for p in suggestion
                    .add_interactive_patterns
                    .iter()
                    .take(remaining as usize)
                {
                    if !policy.interactive_patterns.contains(p) {
                        policy.interactive_patterns.push(p.clone());
                        added += 1;
                    }
                }
                for p in suggestion
                    .add_noise_patterns
                    .iter()
                    .take(remaining.saturating_sub(added) as usize)
                {
                    if !policy.noise_patterns.contains(p) {
                        policy.noise_patterns.push(p.clone());
                        added += 1;
                    }
                }
                for p in suggestion
                    .add_protected_patterns
                    .iter()
                    .take(remaining.saturating_sub(added) as usize)
                {
                    if !policy.protected_patterns.contains(p) {
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
                        let mut gov = state.adaptive_governor.lock().unwrap_or_else(|e| e.into_inner());
                        gov.update_learned_policy(&policy);
                    }
                }
                write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            }
        }
        Err(err) => {
            let mut llm_state = state.llm_state.lock().unwrap_or_else(|e| e.into_inner());
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
) {
    let now = Utc::now();
    let ws_cpu = windowserver_cpu(snapshot);
    let interactive_proxy = ws_cpu >= 10.0 || snapshot.cpu.global_usage >= 15.0;
    let mem_pressure = snapshot.pressure.memory_pressure;
    let swap_delta = snapshot.pressure.swap_delta_bytes_per_sec;

    let jank_proxy = ws_cpu >= 35.0
        && (mem_pressure >= 0.75 || swap_delta >= 20.0 * 1024.0 * 1024.0)
        || matches!(
            snapshot.pressure.thermal_level.as_str(),
            "serious" | "critical"
        );

    {
        let mut model = state.usage_model.lock().unwrap_or_else(|e| e.into_inner());
        model.update_from_snapshot(snapshot, now, interactive_proxy, jank_proxy, 10);
    }

    // Persist usage model periodically (every ~2 minutes).
    {
        let mut last = state
            .usage_last_persist_at
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let due = last
            .map(|t| now - t > ChronoDuration::minutes(2))
            .unwrap_or(true);
        if due {
            if let Ok(model) = state.usage_model.lock() {
                model.persist(&state.usage_model_path);
            }
            *last = Some(now);
        }
    }

    // Daily promotion counters (conservative).
    let today = Local::now().date_naive().to_string();
    {
        let mut day = state
            .usage_promotions_day
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if day.as_deref() != Some(&today) {
            *day = Some(today.clone());
            *state
                .usage_promotions_today
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = 0;
        }
    }

    let promotions_used = *state
        .usage_promotions_today
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // Propose promotions without holding locks across scoring.
    let (started_at, existing_interactive, existing_noise) = {
        let model = state.usage_model.lock().unwrap_or_else(|e| e.into_inner());
        let started_at = model.top_report(1).model_started_at;
        drop(model);
        let policy = state
            .learned_policy
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        (
            started_at,
            policy.interactive_patterns,
            policy.noise_patterns,
        )
    };
    let promotions = {
        let model = state.usage_model.lock().unwrap_or_else(|e| e.into_inner());
        model.maybe_promote_patterns(
            now,
            &existing_interactive,
            &existing_noise,
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
        let mut policy = state
            .learned_policy
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for (kind, pattern) in &promotions {
            match kind.as_str() {
                "interactive" => {
                    if !policy.interactive_patterns.contains(pattern) {
                        policy.interactive_patterns.push(pattern.clone());
                        applied += 1;
                    }
                }
                "noise" => {
                    if !policy.noise_patterns.contains(pattern) {
                        policy.noise_patterns.push(pattern.clone());
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
                let mut gov = state.adaptive_governor.lock().unwrap_or_else(|e| e.into_inner());
                gov.update_learned_policy(&policy);
            }
        }
    }

    if applied > 0 {
        let mut used = state
            .usage_promotions_today
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
    // Filter: never act on protected patterns.
    if !policy.protected_patterns.is_empty() {
        actions.retain(|a| match a {
            RootAction::BoostProcess { name, .. }
            | RootAction::ThrottleProcess { name, .. }
            | RootAction::FreezeProcess { name, .. }
            | RootAction::UnfreezeProcess { name, .. } => {
                !policy.protected_patterns.iter().any(|p| name.contains(p))
            }
            _ => true,
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
            actions.push(RootAction::ThrottleProcess {
                pid: p.pid,
                name: p.name.clone(),
                aggressive: false,
                reason: "learned-policy noise".to_string(),
            });
            seen.insert((p.pid, "throttle"));
        }
    }

    actions
}

fn run_socket_server(state: SharedState) -> anyhow::Result<()> {
    let socket_path = Path::new(socket_path());
    if let Some(parent) = socket_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if socket_path.exists() {
        fs::remove_file(socket_path)?;
    }

    let listener = UnixListener::bind(socket_path).context("bind socket")?;
    fs::set_permissions(socket_path, fs::Permissions::from_mode(0o660))?;
    if let Ok(c_path) = CString::new(socket_path.as_os_str().as_encoded_bytes()) {
        unsafe {
            libc::chown(c_path.as_ptr(), 0, 20);
        }
    }

    // BUG 6 fix: spawn a thread per client so one slow/malicious client doesn't
    // block all others. The old synchronous loop also blocked indefinitely on
    // accept(), preventing clean shutdown when stop=true was set.
    for conn in listener.incoming() {
        if state.stop.load(Ordering::SeqCst) {
            break;
        }
        if let Ok(stream) = conn {
            let state_clone = state.clone();
            thread::spawn(move || {
                handle_client(stream, &state_clone);
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

fn get_foreground_app_inner() -> Option<String> {
    // lsappinfo works as root and doesn't require a GUI session token.
    // Output format varies by session context:
    //   user session:  "AppName" ASN:0x0-0x2c02c: ...
    //   root session:  ASN:0x0-0x46046: "AppName" ...
    // Extract the first double-quoted string to handle both formats.
    let output = std::process::Command::new("lsappinfo")
        .arg("front")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let name = text.split('"').nth(1)?.trim().to_string();
    if name.is_empty() || name.starts_with("ASN:") {
        None
    } else {
        Some(name)
    }
}

fn get_foreground_app() -> Option<String> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(get_foreground_app_inner());
    });
    rx.recv_timeout(std::time::Duration::from_millis(100))
        .ok()
        .flatten()
}

fn append_discrepancy_log(
    path: &std::path::Path,
    protected_app: &str,
    actions_removed: usize,
    workload: &str,
    confidence: f32,
) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = std::fs::OpenOptions::new().append(true).create(true).open(path)?;
    let entry = serde_json::json!({
        "at": chrono::Utc::now().to_rfc3339(),
        "event": "safety_precedence_override",
        "protected_app": protected_app,
        "actions_removed": actions_removed,
        "ml_workload": workload,
        "ml_confidence": confidence,
    });
    writeln!(f, "{}", entry)
}

fn build_enriched_process_data(
    sys: &sysinfo::System,
    foreground_app: Option<&str>,
) -> (Vec<ProcessSnapshot>, Vec<HuntSnapshot>) {
    let mut proc_snaps = Vec::new();
    let mut hunt_snaps = Vec::new();

    for (pid, process) in sys.processes() {
        let pid_u32 = pid.as_u32();
        let name = process.name().to_string();
        let is_foreground = foreground_app
            .map(|fg| name.contains(fg) || fg.contains(&name))
            .unwrap_or(false);
        // On UNIX, the kernel automatically re-parents orphaned processes to PID 1 (launchd).
        // So any process with ppid > 0 has a living parent — we must NOT rely on sysinfo
        // to enumerate the parent, because sysinfo misses many privileged system processes.
        let ppid = process.parent().map(|p| p.as_u32()).unwrap_or(0);
        let parent_alive = ppid > 0;
        let is_zombie = process.status() == ProcessStatus::Zombie;
        let rss = process.memory();
        let cpu = process.cpu_usage();

        proc_snaps.push(ProcessSnapshot {
            pid: pid_u32,
            name: name.clone(),
            cpu_percent: cpu,
            rss_bytes: rss,
            is_zombie,
            secs_since_foreground: if is_foreground { 0 } else { 3600 },
            secs_since_user_interaction: if is_foreground { 0 } else { 3600 },
            has_network: false,
            has_gui_window: is_foreground,
            wakeups_per_sec: 0.0,
            parent_alive,
        });

        hunt_snaps.push(HuntSnapshot {
            pid: pid_u32,
            ppid: process.parent().map(|p| p.as_u32()).unwrap_or(0),
            name,
            is_kernel_zombie: is_zombie,
            parent_alive,
            has_gui_window: is_foreground,
            rss_bytes: rss,
            cpu_percent: cpu,
            wakeups_per_sec: 0.0,
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
        if decision.decision == GovernerDecision::Allow {
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
            GovernerDecision::Throttle => {
                new_actions.push(RootAction::ThrottleProcess {
                    pid: decision.pid,
                    name: decision.name.clone(),
                    aggressive: false,
                    reason: format!("heuristic: {}", decision.reason),
                });
                stats.throttles += 1;
            }
            GovernerDecision::Freeze => {
                new_actions.push(RootAction::FreezeProcess {
                    pid: decision.pid,
                    name: decision.name.clone(),
                    reason: format!("heuristic: {}", decision.reason),
                });
                stats.freezes += 1;
            }
            GovernerDecision::Kill => {
                // Safety: downgrade Kill to Freeze — never auto-kill from heuristics
                new_actions.push(RootAction::FreezeProcess {
                    pid: decision.pid,
                    name: decision.name.clone(),
                    reason: format!("heuristic (kill→freeze): {}", decision.reason),
                });
                stats.kills_downgraded += 1;
                stats.freezes += 1;
            }
            GovernerDecision::Allow => unreachable!(),
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
            let learned_policy =
                read_json::<LearnedPolicy>(&learned_policy_path).unwrap_or_default();

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
                    ..RuntimeMetrics::default()
                })),
                frozen: Arc::new(Mutex::new(HashSet::new())),
                frozen_since: Arc::new(Mutex::new(frozen_since_boot)),
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
                timeline: Arc::new(Mutex::new(Vec::new())),
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
                iokit_reader: Arc::new(IOKitSensorReader::new()),
                last_hw_snapshot: Arc::new(Mutex::new(None)),
                iokit_cycle_counter: Arc::new(Mutex::new(0)),

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
            };

            // Load persisted UserProfile (learning survives daemon restarts).
            if let Some(persisted) = read_json::<UserProfilePersisted>(&state.user_profile_path) {
                let mut gov = state.adaptive_governor.lock().unwrap_or_else(|e| e.into_inner());
                gov.user_profile = UserProfile::from_persisted(persisted);
            }

            // Initialize ML Ligero classifier with the already-loaded LearnedPolicy.
            {
                let policy = state.learned_policy.lock().unwrap_or_else(|e| e.into_inner()).clone();
                let mut gov = state.adaptive_governor.lock().unwrap_or_else(|e| e.into_inner());
                gov.update_learned_policy(&policy);
            }

            let reactor_state = state.clone();
            thread::spawn(move || {
                let _ = run_reactor(reactor_state);
            });

            // Defensive: if a previous run froze processes and crashed/restarted, unfreeze them on startup.
            {
                let mut frozen_since = state.frozen_since.lock().unwrap_or_else(|e| e.into_inner());
                if !frozen_since.is_empty() {
                    let count = unfreeze_pids(frozen_since.keys().copied());
                    frozen_since.clear();
                    write_frozen_state(&frozen_state_path, &frozen_since);
                    if let Ok(mut metrics) = state.metrics.lock() {
                        metrics.post_wake_defensive_unfreezes += count;
                        metrics.unfreezes_applied += count;
                        metrics.throttle_reverted += count;
                    }
                }
            }

            let socket_state = state.clone();
            thread::spawn(move || {
                let _ = run_socket_server(socket_state);
            });

            let stop = state.stop.clone();
            ctrlc::set_handler(move || {
                stop.store(true, Ordering::SeqCst);
            })?;

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
            let gpu_mgr = GPUManager::new();
            let mut mem_analyzer = MemoryAnalyzer::new();
            let net_optimizer = NetworkOptimizer::new();
            let power_mgr = PowerManager::new();
            let mut proc_recovery = ProcessRecoveryManager::new();
            let mut swap_predictor = SwapPredictor::new();
            let mut thermal_mgr = ThermalManager::new();
            let mut wake_storm = WakeStormDetector::new();
            // Freeze confirmation cache: pid → consecutive cycles flagged.
            // Only freeze processes that have been candidates for 2+ cycles,
            // filtering out short-lived transients that die before execute_actions.
            let mut freeze_candidates: HashMap<u32, u8> = HashMap::new();

            loop {
                if state.stop.load(Ordering::SeqCst) {
                    break;
                }

                if Path::new(kill_switch_path()).exists() {
                    thread::sleep(Duration::from_secs(5));
                    continue;
                }

                let cycle_start = Instant::now();
                if daemon_start.elapsed() > Duration::from_secs(600) {
                    let events = *state
                        .reactor_events_total
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    if events == 0 {
                        *state.reactor_mode.lock().unwrap_or_else(|e| e.into_inner()) =
                            "degraded".to_string();
                        *state
                            .reactor_health
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()) = "stalled".to_string();
                        *state
                            .fast_tick_until
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()) = None;
                    }
                }
                let now_wall = Utc::now();
                let mut wake_state_guard =
                    state.wake_state.lock().unwrap_or_else(|e| e.into_inner());
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

                    let mut frozen = state.frozen.lock().unwrap_or_else(|e| e.into_inner());
                    let mut frozen_since =
                        state.frozen_since.lock().unwrap_or_else(|e| e.into_inner());
                    let unfreeze_count =
                        unfreeze_pids(frozen.iter().copied().chain(frozen_since.keys().copied()));
                    frozen.clear();
                    frozen_since.clear();
                    write_frozen_state(&frozen_state_path, &frozen_since);

                    if let Ok(mut metrics) = state.metrics.lock() {
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

                let mut snapshot = collector.collect_snapshot();
                snapshot.pressure.thermal_level = state
                    .thermal_level_real
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                let latency_target = *state
                    .latency_target
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());

                // Heuristic: collect foreground app and build enriched process data
                let foreground_app = get_foreground_app();
                let (proc_snaps, hunt_snaps) = build_enriched_process_data(
                    collector.system(),
                    foreground_app.as_deref(),
                );
                let all_proc_names: Vec<&str> = proc_snaps.iter().map(|p| p.name.as_str()).collect();
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

                // Heuristic: IOKit tick every 5th cycle
                {
                    let mut iokit_counter = state.iokit_cycle_counter.lock().unwrap_or_else(|e| e.into_inner());
                    *iokit_counter += 1;
                    if *iokit_counter >= 5 {
                        *iokit_counter = 0;
                        drop(iokit_counter);
                        match state.iokit_reader.snapshot() {
                            Ok(hw) => {
                                if let Ok(mut m) = state.metrics.lock() {
                                    m.iokit_snapshots += 1;
                                    m.iokit_p_cluster_temp = hw.temps.p_cluster_celsius;
                                    m.iokit_e_cluster_temp = hw.temps.e_cluster_celsius;
                                    m.iokit_package_watts = hw.power.package_watts;
                                }
                                *state.last_hw_snapshot.lock().unwrap_or_else(|e| e.into_inner()) = Some(hw);
                            }
                            Err(_) => {
                                if let Ok(mut m) = state.metrics.lock() {
                                    m.iokit_errors += 1;
                                }
                            }
                        }
                    }
                }

                // F4: Thermal master switch — computed once after IOKit data is fresh.
                let thermal_emergency = state
                    .last_hw_snapshot
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .and_then(|h| h.temps.p_cluster_celsius)
                    .map(|t| t > 95.0)
                    .unwrap_or(false);

                // ThermalManager + GPUManager: tick every cycle with latest IOKit temperatures.
                {
                    let hw_snap = state
                        .last_hw_snapshot
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone();
                    if let Some(hw) = &hw_snap {
                        let cpu_t = hw.temps.p_cluster_celsius.unwrap_or(0.0);
                        let gpu_t = hw.temps.e_cluster_celsius.unwrap_or(cpu_t);
                        let _thermal_state = thermal_mgr.update(cpu_t, gpu_t, 0.0, 0);
                        let _gpu_power = gpu_mgr.recommend_power_state(0.0, cpu_t);
                    }
                }

                // SwapPredictor: update trend forecast every cycle.
                let _swap_forecast = swap_predictor.update(
                    snapshot.pressure.swap_used_bytes,
                    snapshot.pressure.swap_total_bytes,
                );

                // NetworkOptimizer + PowerManager: advisory tick (no real sensor data yet).
                let _net_profile = net_optimizer.recommend_profile();
                let _power_rec = power_mgr.get_recommendation();

                // Online usage learning (root-only, no UI sensors): infer frequently-used apps
                // and processes correlated with jank, then promote patterns conservatively.
                usage_learning_tick(&state, &snapshot);

                // LLM teacher mode (cloud) - optional, rate-limited, and guarded.
                // This runs before governor evaluation so a high-confidence suggestion can set a
                // short-lived manual override during the training window.
                llm_reactive_tick(&state, &mut llm_advisor, &snapshot, &mut llm_counters);

                let mut reactor_weight = state
                    .reactor_event_weight
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                *reactor_weight = (*reactor_weight * 0.75).clamp(0.0, 1.0);

                let pressure_cpu = (snapshot.cpu.global_usage as f64 / 100.0).clamp(0.0, 1.0);
                let pressure_ram = snapshot.pressure.memory_pressure.clamp(0.0, 1.0);
                let pressure_wait = snapshot
                    .top_processes
                    .iter()
                    .take(8)
                    .filter(|p| p.cpu_usage < 8.0 && p.memory_usage > 100 * 1024 * 1024)
                    .count() as f64
                    / 8.0_f64;
                let pressure_wait = pressure_wait.clamp(0.0, 1.0);

                let critical_patterns = critical_background_processes();
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
                    if let Ok(mut metrics) = state.metrics.lock() {
                        metrics.swap_used_bytes = snapshot.pressure.swap_used_bytes;
                        metrics.swap_delta_bps = snapshot.pressure.swap_delta_bytes_per_sec;
                        metrics.memory_pressure = snapshot.pressure.memory_pressure;
                        metrics.thermal_level = snapshot.pressure.thermal_level.clone();
                        metrics.reactor_events_total = *state
                            .reactor_events_total
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        metrics.reactor_events_mem = *state
                            .reactor_events_mem
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        metrics.reactor_events_thermal = *state
                            .reactor_events_thermal
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        metrics.reactor_events_spawn = *state
                            .reactor_events_spawn
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        metrics.reactor_events_power = *state
                            .reactor_events_power
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        metrics.reactor_last_event_at = *state
                            .reactor_last_event_at
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        metrics.reactor_last_error = state
                            .reactor_last_error
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .clone();
                        metrics.reactor_mode = state
                            .reactor_mode
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .clone();
                        metrics.reactor_health = state
                            .reactor_health
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .clone();
                        metrics.dev_session_active = dev_session_active;
                        metrics.interactive_heavy = interactive_heavy;
                    }
                }

                let mut governor = state.governor.lock().unwrap_or_else(|e| e.into_inner());
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
                });
                if governor_decision.transition_reason.contains("floor") {
                    if let Ok(mut metrics) = state.metrics.lock() {
                        metrics.profile_floor_hits += 1;
                    }
                }
                let current_profile = governor_decision.effective_profile;
                write_governor_state(&governor_state_path, &governor);
                drop(governor);

                let decision =
                    decide_actions(&snapshot, collector.system(), current_profile, latency_target, *reactor_weight);
                *state
                    .last_blockers
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = decision.blockers.clone();
                *state
                    .thermal_state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = context_to_thermal(decision.context);

                // Apply any locally learned policy patterns (and keep them even after LLM is disabled).
                let mut actions = decision.actions;
                {
                    let policy = state
                        .learned_policy
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone();
                    actions = apply_learned_policy_actions(&snapshot, &policy, actions);
                }

                // Heuristic pass: AdaptiveGovernor
                let heuristic_decisions = {
                    let mut gov = state.adaptive_governor.lock().unwrap_or_else(|e| e.into_inner());
                    gov.decide_all(
                        &proc_snaps,
                        &hunt_snaps,
                        foreground_app.as_deref(),
                        &all_proc_names,
                        hour_of_day,
                    )
                };

                // Build critical_pids set for heuristic merge (same logic as decide_actions)
                let heuristic_critical_pids: HashSet<u32> = {
                    let sys = collector.system();
                    let critical_pats = critical_background_processes();
                    let mut cpids: HashSet<u32> = HashSet::new();
                    for (pid, process) in sys.processes() {
                        let name = process.name().to_string();
                        if critical_pats.iter().any(|p| name.contains(p)) {
                            cpids.insert(pid.as_u32());
                        }
                    }
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
                            unsafe { libc::kill(target.pid as i32, libc::SIGKILL); }
                            proc_recovery.record_kill_attempt(target.pid);
                            if let Ok(mut m) = state.metrics.lock() {
                            m.kills_applied += 1;
                            m.survival_mode_activations += 1;
                        }
                        }
                    } else {
                        actions.push(RootAction::FreezeProcess {
                            pid: target.pid,
                            name: target.name.clone(),
                            reason: format!(
                                "memory-leak recovery: prob={:.2} rss={}MB attempts={}",
                                target.leak_probability,
                                target.rss_bytes / 1024 / 1024,
                                target.recovery_attempts,
                            ),
                        });
                        proc_recovery.record_kill_attempt(target.pid);
                    }
                }

                // WakeStormDetector: throttle processes generating excessive wakeups.
                let storms = wake_storm.detect_storms();
                for storm in &storms {
                    if !heuristic_critical_pids.contains(&storm.pid) {
                        actions.push(RootAction::ThrottleProcess {
                            pid: storm.pid,
                            name: storm.name.clone(),
                            aggressive: false,
                            reason: format!(
                                "wake-storm: {:.0} wakeups/sec",
                                storm.wakeups_per_second
                            ),
                        });
                    }
                }

                // Paging hints: send memory pressure signal to idle memory hoarders.
                // Triggers the process's pressure handler to release caches.
                // Only when memory pressure is elevated (>0.5) and swap is in use.
                let mem_pressure = snapshot.pressure.memory_pressure;
                let swap_active = snapshot.pressure.swap_used_bytes > 512 * 1024 * 1024;
                if mem_pressure > 0.5 && swap_active {
                    for snap in proc_snaps.iter().take(100) {
                        let is_hoarder = snap.rss_bytes > 200 * 1024 * 1024
                            && snap.secs_since_user_interaction > 600
                            && !snap.has_gui_window;
                        if is_hoarder && !heuristic_critical_pids.contains(&snap.pid) {
                            actions.push(RootAction::SetMemorystatus {
                                pid: snap.pid,
                                priority: -1,
                                reason: format!(
                                    "memory-pressure hint: {}MB RSS idle",
                                    snap.rss_bytes / 1024 / 1024
                                ),
                            });
                        }
                    }
                }

                // Update heuristic metrics
                {
                    if let Ok(mut m) = state.metrics.lock() {
                        m.heuristic_decisions += heuristic_stats.decisions_total;
                        m.heuristic_throttles += heuristic_stats.throttles;
                        m.heuristic_freezes += heuristic_stats.freezes;
                        m.heuristic_kills_downgraded += heuristic_stats.kills_downgraded;
                        m.zombies_detected += heuristic_stats.zombies_detected;
                        // Update current workload from adaptive governor's user profile
                        let gov = state.adaptive_governor.lock().unwrap_or_else(|e| e.into_inner());
                        m.current_workload = format!("{:?}", gov.user_profile.current_workload());
                    }
                }

                // F2 — ML Ligero: read classification result (computed inside decide_all this cycle).
                // GovernorConfig aggressiveness was already updated inside decide_all().
                let ml_class = {
                    let gov = state.adaptive_governor.lock().unwrap_or_else(|e| e.into_inner());
                    gov.last_ml_classification().clone()
                };
                if let Ok(mut m) = state.metrics.lock() {
                    m.ml_confidence = ml_class.confidence;
                    m.current_workload = format!("{:?}", ml_class.workload).to_lowercase();
                    m.ml_sources = ml_class.sources_summary();
                }

                // F3 — Safety Precedence: foreground app is NEVER throttled or frozen.
                if let Some(fg) = &foreground_app {
                    let before = actions.len();
                    actions.retain(|a| match a {
                        RootAction::ThrottleProcess { name, .. } => !name.contains(fg.as_str()),
                        RootAction::FreezeProcess { name, .. } => !name.contains(fg.as_str()),
                        _ => true,
                    });
                    let removed = before - actions.len();
                    if removed > 0 {
                        // Discrepancy log — gold data for LLM retraining
                        let _ = append_discrepancy_log(
                            &state.discrepancy_log_path,
                            fg,
                            removed,
                            &format!("{:?}", ml_class.workload),
                            ml_class.confidence,
                        );
                    }
                }

                // F4 — Thermal Master Switch: >95°C P-cluster — suppress all Boost actions.
                if thermal_emergency {
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
                    if let Ok(mut metrics) = state.metrics.lock() {
                        metrics.budgets.minute_actions = 0;
                    }
                }

                let caps = detect_capabilities();

                // Phase 1: Compute budget-filtered actions (metrics lock held briefly).
                // BUG 5 fix: split into three phases so the metrics mutex is never held
                // across the blocking I/O inside execute_actions.
                let final_actions = {
                    let mut metrics = state.metrics.lock().unwrap_or_else(|e| e.into_inner());
                    // TTL: don't leave freezes hanging forever.
                    {
                        let now = Utc::now();
                        let mut frozen_since =
                            state.frozen_since.lock().unwrap_or_else(|e| e.into_inner());
                        let expired: Vec<u32> = frozen_since
                            .iter()
                            .filter(|(_, since)| (now - **since).num_seconds() > FREEZE_TTL_SECS)
                            .map(|(pid, _)| *pid)
                            .collect();
                        if !expired.is_empty() {
                            let count = unfreeze_pids(expired.iter().copied());
                            for pid in &expired {
                                frozen_since.remove(pid);
                            }
                            write_frozen_state(&frozen_state_path, &frozen_since);
                            metrics.post_wake_defensive_unfreezes += count;
                            metrics.unfreezes_applied += count;
                            metrics.throttle_reverted += count;
                        }
                    }
                    metrics.budgets.cycle_boosts = 0;
                    metrics.budgets.cycle_throttles = 0;
                    metrics.budgets.cycle_hints = 0;
                    metrics.budgets.cycle_freezes = 0;

                    let (graced_actions, throttle_suppressed, freeze_suppressed) =
                        apply_post_wake_grace_policy(actions, grace_active);
                    metrics.post_wake_throttle_suppressed += throttle_suppressed;
                    metrics.post_wake_freeze_suppressed += freeze_suppressed;

                    // Freeze confirmation: only freeze PIDs flagged for 2+ consecutive cycles.
                    // This filters out short-lived transients that die before execute_actions.
                    let confirmed_actions: Vec<RootAction> = graced_actions.into_iter().filter(|a| {
                        if let RootAction::FreezeProcess { pid, .. } = a {
                            let count = freeze_candidates.entry(*pid).or_insert(0);
                            *count += 1;
                            *count >= 2
                        } else {
                            true
                        }
                    }).collect();
                    // Decay: remove PIDs no longer proposed for freeze this cycle.
                    let freeze_pids_this_cycle: HashSet<u32> = confirmed_actions.iter()
                        .filter_map(|a| if let RootAction::FreezeProcess { pid, .. } = a { Some(*pid) } else { None })
                        .collect();
                    freeze_candidates.retain(|pid, _| freeze_pids_this_cycle.contains(pid));

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
                let exec_outcomes = {
                    let mut frozen = state.frozen.lock().unwrap_or_else(|e| e.into_inner());
                    let outcomes = execute_actions(final_actions, &caps, &journal_path, &mut frozen);
                    // Persist any currently-frozen PIDs so we can unfreeze on restart if needed.
                    let now = Utc::now();
                    let mut frozen_since =
                        state.frozen_since.lock().unwrap_or_else(|e| e.into_inner());
                    for pid in frozen.iter() {
                        frozen_since.entry(*pid).or_insert(now);
                    }
                    // Remove PIDs that are no longer frozen.
                    frozen_since.retain(|pid, _| frozen.contains(pid));
                    write_frozen_state(&frozen_state_path, &frozen_since);
                    outcomes
                    // frozen and frozen_since locks released here
                };

                // F5 — MachQoS: route processes to P-Cores / E-Cores based on heuristic decisions.
                // Skip SIGSTOP'd processes; force E-Cores for all during thermal emergency.
                {
                    let frozen_pids = state.frozen.lock().unwrap_or_else(|e| e.into_inner()).clone();
                    let qos_changes: Vec<(u32, SchedulingTier)> = heuristic_decisions
                        .iter()
                        .filter(|d| {
                            !frozen_pids.contains(&d.pid)
                                && !heuristic_critical_pids.contains(&d.pid)
                        })
                        .map(|decision| {
                            let tier = if thermal_emergency {
                                // Force all to E-Cores during thermal emergency
                                SchedulingTier::Background
                            } else {
                                match decision.decision {
                                    GovernerDecision::Allow => {
                                        if decision.tier == ProcessTier::ActiveForeground {
                                            SchedulingTier::Foreground
                                        } else {
                                            SchedulingTier::Normal
                                        }
                                    }
                                    GovernerDecision::Throttle => SchedulingTier::Normal,
                                    GovernerDecision::Freeze | GovernerDecision::Kill => {
                                        SchedulingTier::Background
                                    }
                                }
                            };
                            (decision.pid, tier)
                        })
                        .collect();

                    let mut qos = state.mach_qos.lock().unwrap_or_else(|e| e.into_inner());
                    qos.tick_backoff();
                    let outcomes = qos.apply_batch(&qos_changes);
                    if let Ok(mut m) = state.metrics.lock() {
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
                    let mut metrics = state.metrics.lock().unwrap_or_else(|e| e.into_inner());
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
                    metrics.throttle_reverted += exec_outcomes.throttle_reverted;
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
                    metrics.thermal_state = state
                        .thermal_state
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone();
                    metrics.last_pressure_score = governor_decision.pressure_score;
                    if governor_decision.override_expired {
                        metrics.override_expirations += 1;
                    }
                    if governor_decision.override_active && !override_was_active {
                        metrics.override_activations += 1;
                    }
                    if let Some(transition) = governor_decision.transition.clone() {
                        metrics.profile_switches += 1;
                        let mut timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
                        timeline.push(transition.clone());
                        if timeline.len() > 200 {
                            let _ = timeline.remove(0);
                        }
                        append_timeline(&timeline_path, &transition);
                    }
                    override_was_active = governor_decision.override_active;

                    let elapsed = cycle_start.elapsed().as_millis() as u64;
                    metrics.cycle_durations_ms.push(elapsed);
                    if metrics.cycle_durations_ms.len() > 120 {
                        let _ = metrics.cycle_durations_ms.remove(0);
                    }
                    metrics.p95_cycle_ms = compute_p95(&metrics.cycle_durations_ms);

                    *state
                        .throttle_level
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = metrics.throttle_level.clone();

                    let nowi = Instant::now();
                    critical_failure_timestamps
                        .retain(|t| nowi.duration_since(*t) <= Duration::from_secs(180));
                    if had_new_failures {
                        critical_failure_timestamps.push(nowi);
                    }
                    if critical_failure_timestamps.len() > 5 {
                        state
                            .governor
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .force_safe_on_errors();
                        critical_failure_timestamps.clear();
                    }

                    write_metrics(&metrics_path, &metrics);
                }

                // Analytics: record this cycle's metrics for trend tracking.
                {
                    let thermal_now = state
                        .last_hw_snapshot
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
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
                    let cycles = state
                        .metrics
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .cycles;
                    if cycles % 10 == 0 {
                        let persisted = {
                            let gov =
                                state.adaptive_governor.lock().unwrap_or_else(|e| e.into_inner());
                            gov.user_profile.to_persisted()
                        };
                        write_json(&state.user_profile_path, &persisted, None);
                    }
                }

                let fast = state
                    .fast_tick_until
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .map(|t| Instant::now() < t)
                    .unwrap_or(false);
                let sleep_secs = if fast { 5 } else { 30 };
                thread::sleep(Duration::from_secs(sleep_secs));
            }

            // BUG 19 fix: unfreeze all frozen processes on daemon shutdown so
            // processes don't remain stopped if the daemon exits or crashes.
            {
                let frozen = state.frozen.lock().unwrap_or_else(|e| e.into_inner());
                let pids: Vec<u32> = frozen.iter().copied().collect();
                if !pids.is_empty() {
                    unfreeze_pids(pids.into_iter());
                }
            }

            let _ = fs::remove_file(socket_path());
        }
    }

    Ok(())
}
