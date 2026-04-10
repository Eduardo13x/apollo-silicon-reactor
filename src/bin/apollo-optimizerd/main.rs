use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Global stop flag for signal handlers (SIGTERM/SIGINT).
/// Signal handlers cannot capture Arc/closures, so we use a static AtomicBool
/// that the main loop checks alongside `state.stop`.
mod cognitive_tick;
mod daemon_init;
mod learning_tick;
mod llm_daemon;
mod metrics_reporter;
mod process_enrichment;
mod socket_handler;

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

/// SIGTERM handler — async-signal-safe: only sets an atomic flag.
extern "C" fn handle_sigterm(_sig: libc::c_int) {
    STOP_REQUESTED.store(true, Ordering::Release);
}

use apollo_optimizer::collector::SystemCollector;
use apollo_optimizer::engine::adaptive_governor::AdaptiveGovernor;
use apollo_optimizer::engine::amx_detector;
use apollo_optimizer::engine::background_collectors::PressureCollector;
use apollo_optimizer::engine::capabilities::detect_capabilities;
use apollo_optimizer::engine::causal_graph::CausalGraph;
use apollo_optimizer::engine::compressor_aware::{
    decide_enhanced, purge_purgeable_regions, query_memory_profile, sample_process_temperature,
    scan_regions, MemoryAction,
};
use apollo_optimizer::engine::daemon_helpers::{
    audit_log, battery_pressure_boost, frozen_state_path, governor_state_path, holt_winters_path,
    hop_groups_path, journal_path, kill_switch_path, learned_state_path, load_frozen_state,
    load_governor_state, load_wake_state, markov_path, merge_seed_into, metrics_path,
    overflow_history_path, parse_profile, pid_start_time, predictive_agent_path, rl_threshold_path,
    detect_prior_crash, remove_crash_sentinel, should_rotate_oldest, should_unfreeze,
    signal_intelligence_path, skills_path, socket_path, spotlight_set_indexing,
    telemetry_output_dir, temporal_histograms_path, timeline_path, unfreeze_pids,
    wake_state_path, write_frozen_state, write_governor_state, write_wake_state,
};
use apollo_optimizer::engine::effective_pressure;
use apollo_optimizer::engine::execute_actions::execute_actions;
use apollo_optimizer::engine::focus_markov::FocusMarkov;
use apollo_optimizer::engine::foreground::{ForegroundDetector, ForegroundState};
use apollo_optimizer::engine::gpu_manager::{GPUManager, GPUMetrics, GPUPowerState};
use apollo_optimizer::engine::holt_winters::HoltWinters;
use apollo_optimizer::engine::hw_bayes::HwFeatures;
use apollo_optimizer::engine::hw_predictor::{sample_hw_pressure, HwPressure};
use apollo_optimizer::engine::iokit_sensors::{HardwareSnapshot, ThermalState};
use apollo_optimizer::engine::jetsam_control;
use apollo_optimizer::engine::kqueue_pressure;
use apollo_optimizer::engine::latency_monitor::{self, LatencySignals};
use apollo_optimizer::engine::learned_state::{LearnableParams, LearnedState, RestoreQualityMonitor};
use apollo_optimizer::engine::llm::{
    feedback_path_root, load_repo_config, policy_path_root, read_json, state_paths_root,
    suggestions_path_root, write_json, LearnedPolicy, LlmAdvisor, LlmConfig, LlmState,
};
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::lse_counters::LockFreeMetrics;
use apollo_optimizer::engine::mach_qos::{MachQoSManager, SchedulingTier};
use apollo_optimizer::engine::memory_analyzer::MemoryAnalyzer;
use apollo_optimizer::engine::memory_budget::{self, ProcessBudgetInput};
use apollo_optimizer::engine::network_optimizer::NetworkProfile;
use apollo_optimizer::engine::neuromodulator::NeuroSignals;
use apollo_optimizer::engine::overflow_guard::{OverflowGuard, BUILD_TOOLS};
use apollo_optimizer::engine::pipeline::decision_stage::{DecisionStage, PolicyContext};
use apollo_optimizer::engine::pipeline::learning_context::LearningContext;
use apollo_optimizer::engine::pipeline::periodic_stage::{run_periodic, PeriodicContext};
use apollo_optimizer::engine::power_management::detect_battery_status;
use apollo_optimizer::engine::predictive_agent::{
    specialist, tally_votes, AgentContext, Intervention, PredictiveAgent, SpecialistVote,
};
use apollo_optimizer::engine::proc_taskinfo;
use apollo_optimizer::engine::process_tree::{ProcessEntry, ProcessTree};
use apollo_optimizer::engine::profile_governor::GovernorInput;
use apollo_optimizer::engine::safety::{
    behavioral_protection_score, classify_protection, critical_background_processes,
    enforce_limits_with_budget, infrastructure_processes, is_user_interactive_app,
    matches_dev_runtime, protected_processes, ProtectionLevel,
};
use apollo_optimizer::engine::signal_intelligence::SignalIntelligence;
use apollo_optimizer::engine::smc_reader::SmcReader;
use apollo_optimizer::engine::sysctl_governor::{
    SysctlGovernor, SysctlGovernorInput, SysctlGovernorStatus,
};
use apollo_optimizer::engine::thermal_interrupt::{
    spawn_resource_sentinel, ResourceInterruptState, SentinelConfig,
};
use apollo_optimizer::engine::types::{
    EnergyConsumerInfo, ForegroundAppInfo, FreezeSource, FrozenEntry, FrozenPidEntry,
    FrozenStatePersisted, LatencyTarget, OptimizationProfile, RootAction, RuntimeMetrics,
    SafetyPolicy,
};
use apollo_optimizer::engine::usage_model::{usage_model_path_root, UsageModel};
use apollo_optimizer::engine::user_profile::{UserProfile, UserProfilePersisted};
use apollo_optimizer::engine::wait_graph;
use apollo_optimizer::engine::workload_classifier::classify_by_memory;
use apollo_optimizer::engine::workload_classifier::{
    classify_workload_mode, WorkloadFeatures, WorkloadMode,
};
use chrono::{Duration as ChronoDuration, Timelike, Utc};
use clap::{Parser, Subcommand};

// v0.9.0: canonical SharedState — all domain groups live in daemon_state.rs
use apollo_optimizer::engine::daemon_state::{
    HardwareState, LlmDomainState, MetricsState, PolicyState, ProcessState,
    ReactorStatus as DomainReactorStatus, SharedState, UsageDomainState, UsageTrackerState,
};

// FREEZE_TTL_SECS → daemon_helpers
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
// battery_pressure_boost, merge_seed_into, pid_start_time → daemon_helpers

// Path functions (socket_path, kill_switch_path, journal_path, etc.) → daemon_helpers

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
        /// Skip all OS-mutating calls (SIGSTOP/SIGCONT/taskpolicy/sysctl/mdutil).
        /// The full pipeline runs but no real processes are frozen or throttled.
        /// Intended for testing and benchmarking.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
}
// SharedState → daemon_state (PR#15: canonical definition in daemon_state.rs)
// ReactorStatus → daemon_state (PR#10: unified with MetricsState)
// UsageTrackerState → daemon_state (PR#13: unified single definition)

// WakeStatePersisted, WakeRuntimeState → daemon_helpers

// ThrashState → process_enrichment
// LlmReactiveCounters → llm_daemon

// parse_profile, write_metrics → daemon_helpers

// write_governor_state, load_governor_state, append_timeline → daemon_helpers

// wake_state, frozen_state, unfreeze, should_unfreeze, should_rotate_oldest + tests → daemon_helpers

fn run_reactor(state: SharedState) -> anyhow::Result<()> {
    unsafe {
        let kq = libc::kqueue();
        if kq == -1 {
            state.metrics.lock_recover().reactor_status.last_error =
                Some("kqueue failed".to_string());
            return Ok(());
        }

        // Memory pressure via sysctl polling (all push APIs are broken on macOS 15).
        // Polls kern.memorystatus_vm_pressure_level (~1µs) on each loop iteration.
        let mut pressure_monitor =
            apollo_optimizer::engine::dispatch_pressure::KernelPressureMonitor::new();

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
            state.metrics.lock_recover().reactor_status.last_error = Some(format!(
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
            state.metrics.lock_recover().reactor_status.last_error = Some(format!(
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
            state.metrics.lock_recover().reactor_status.last_error = Some(format!(
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
                m.metrics.reactor_pulses += 1;
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
                state.metrics.lock_recover().reactor_status.events_mem += 1;
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
                    state.metrics.lock_recover().reactor_status.last_error =
                        Some(format!("kevent error errno={}", errno));
                }
                continue;
            }

            let id = out_ev.udata as usize;
            // Update shared counters + status in one lock acquisition.
            let reactor_mode = {
                let mut m = state.metrics.lock_recover();
                m.reactor_status.events_total += 1;
                m.reactor_status.last_event_at = Some(Utc::now());
                m.reactor_status.health = "ok".to_string();
                m.reactor_status.mode.clone()
            };
            if id == 2 {
                // Drain thermal pipe
                let mut dummy: i32 = 0;
                let _ = libc::read(thermal_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                state.metrics.lock_recover().reactor_status.events_thermal += 1;
                let level = match dummy {
                    0 => "nominal",
                    1 => "moderate",
                    2 => "serious",
                    _ => "critical",
                };
                state.metrics.lock_recover().thermal_level_real = level.to_string();
                // Signal resource sentinel for thermal ≥ serious.
                if dummy >= 2 {
                    state
                        .resource_interrupt
                        .thermal_signal
                        .store(true, Ordering::Release);
                }
            } else if id == 3 {
                // EVFILT_PROC NOTE_FORK on launchd (pid 1) — no pipe to drain.
                // launch_fd == -1 (sentinel); reading from it is always EBADF.
                // Just increment the counter; the kernel event itself is the notification.
                state.metrics.lock_recover().reactor_status.events_spawn += 1;
            } else if id == 4 {
                let mut dummy: i32 = 0;
                let _ = libc::read(power_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                state.metrics.lock_recover().reactor_status.events_power += 1;
                // Signal resource sentinel for power source changes.
                state
                    .resource_interrupt
                    .power_signal
                    .store(true, Ordering::Release);
            } else if id == 1 {
                state.metrics.lock_recover().reactor_status.events_mem += 1;
                state
                    .resource_interrupt
                    .memory_signal
                    .store(true, Ordering::Release);
            }

            state.metrics.lock_recover().reactor_event_weight = 1.0;
            if reactor_mode.as_str() == "normal" {
                state.metrics.lock_recover().fast_tick_until =
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

// rotate_timeline → daemon_helpers

#[link(name = "System")]
extern "C" {
    fn notify_register_file_descriptor(
        name: *const libc::c_char,
        out_fd: *mut libc::c_int,
        flags: libc::c_int,
        out_token: *mut libc::c_int,
    ) -> u32;
}

// compute_p95 → daemon_helpers

// filter_boost_cooldown, apply_post_wake_grace_policy, context_to_thermal,
// append_discrepancy_log, build_foreground_family, build_enriched_process_data_with_tree,
// convert_and_merge_heuristic_decisions, HeuristicStats → process_enrichment

/// Toggle Spotlight indexing via `mdutil -a -i on/off`.
///
fn main() -> anyhow::Result<()> {
    // Structured JSON logging to stderr (captured by launchd → apollo-optimizer.err.log).
    // Override level at runtime: APOLLO_LOG=debug apollo-optimizerd
    {
        use tracing_subscriber::{fmt, EnvFilter};
        let filter =
            EnvFilter::try_from_env("APOLLO_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
        fmt()
            .json()
            .with_env_filter(filter)
            .with_current_span(false)
            .init();
    }

    let cli = Cli::parse();

    match cli.command {
        Commands::Daemon { profile, dry_run } => {
            let profile = parse_profile(&profile);
            let is_root = unsafe { libc::geteuid() } == 0;

            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                profile = profile.as_str(),
                root = is_root,
                "apollo-optimizerd starting"
            );
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
                    tracing::warn!(
                        path = %learned_policy_path.display(),
                        "learned policy missing or corrupt — falling back to seed policy"
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
                policy: Arc::new(Mutex::new(PolicyState {
                    profile,
                    latency_target: LatencyTarget::Normal,
                    governor,
                    learned_policy,
                    adaptive_governor: AdaptiveGovernor::new(),
                    timeline: VecDeque::new(),
                    circuit_breaker:
                        apollo_optimizer::engine::circuit_breaker::CircuitBreaker::default(),
                    degradation:
                        apollo_optimizer::engine::degradation::DegradationController::default(),
                })),
                metrics: Arc::new(Mutex::new(MetricsState {
                    metrics: RuntimeMetrics {
                        effective_profile: profile,
                        throttle_level: "balanced".to_string(),
                        thermal_state: "nominal".to_string(),
                        thermal_level: "unknown".to_string(),
                        current_workload: "idle".to_string(),
                        collector_pressure_alive: true,
                        collector_smc_alive: true,
                        ..RuntimeMetrics::default()
                    },
                    throttle_level: "balanced".to_string(),
                    thermal_state: "nominal".to_string(),
                    thermal_level_real: "unknown".to_string(),
                    fast_tick_until: None,
                    reactor_event_weight: 0.0,
                    reactor_status: DomainReactorStatus::default(),
                })),
                frozen_state: Arc::new(Mutex::new(frozen_since_boot.clone())),
                process: Arc::new(Mutex::new(ProcessState {
                    last_blockers: Vec::new(),
                    wake_state,
                })),
                stop: Arc::new(AtomicBool::new(false)),

                llm: Arc::new(Mutex::new(LlmDomainState {
                    llm_cfg,
                    llm_state,
                    llm_state_path,
                    llm_key_path,
                    learned_policy_path,
                    feedback_path,
                    suggestions_path,
                })),

                config_path,

                usage: Arc::new(Mutex::new(UsageDomainState {
                    usage_model,
                    usage_model_path,
                    usage_events_path,
                    usage_tracker: UsageTrackerState::default(),
                })),

                mach_qos: Arc::new(Mutex::new(MachQoSManager::new())),
                hardware: Arc::new(Mutex::new(HardwareState {
                    last_hw_snapshot: None,
                    sysctl_governor_status: SysctlGovernorStatus {
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
                    },
                })),

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

                cycle_condvar: Arc::new((Mutex::new(false), Condvar::new())),
                resource_interrupt: Arc::new(ResourceInterruptState::new()),
                revert_sysctls_requested: Arc::new(AtomicBool::new(false)),

                subscribers: Arc::new(Mutex::new(Vec::new())),
            };

            // Load persisted UserProfile (learning survives daemon restarts).
            if let Some(persisted) = read_json::<UserProfilePersisted>(&state.user_profile_path) {
                state.policy.lock_recover().adaptive_governor.user_profile =
                    UserProfile::from_persisted(persisted);
            }

            // Scrub learned policy: remove patterns that should never be interactive.
            // This list is curated by LLM Teacher analysis of usage_model data.
            let learned_policy_path = state.llm.lock_recover().learned_policy_path.clone();
            {
                let mut policy = state.policy.lock_recover().learned_policy.clone();
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
                    // Write back to shared state, then persist.
                    state.policy.lock_recover().learned_policy = policy.clone();
                    write_json(&learned_policy_path, &policy, Some(0o600));
                }
            }

            // Initialize ML Ligero classifier with the already-loaded LearnedPolicy.
            {
                let policy = state.policy.lock_recover().learned_policy.clone();
                state
                    .policy
                    .lock_recover()
                    .adaptive_governor
                    .update_learned_policy(&policy);
            }

            let reactor_state = state.clone();
            thread::spawn(move || {
                let _ = run_reactor(reactor_state);
            });

            // Defensive: if a previous run froze processes and crashed/restarted, unfreeze them on startup.
            // PID reuse guard: if a process_name was recorded at freeze time, verify the current
            // process at that PID still has the same name before sending SIGCONT. If names differ,
            // the PID was recycled — skip SIGCONT (the original process is gone; the new one is
            // running normally and doesn't need SIGCONT).
            {
                let mut frozen_state = state.frozen_state.lock_recover();
                if !frozen_state.is_empty() {
                    // Build a lightweight set of live process names for PID-reuse detection.
                    // We spin up sysinfo only if there are frozen entries to check.
                    use sysinfo::{ProcessRefreshKind, RefreshKind, System};
                    let mut sys = System::new_with_specifics(
                        RefreshKind::new().with_processes(ProcessRefreshKind::new()),
                    );
                    sys.refresh_processes_specifics(ProcessRefreshKind::new());

                    let safe_pids: Vec<u32> = frozen_state
                        .iter()
                        .filter(|(pid, entry)| {
                            if let Some(ref expected_name) = entry.process_name {
                                // A name was recorded: verify the current process still matches.
                                let pid_sysinfo = sysinfo::Pid::from_u32(**pid);
                                match sys.process(pid_sysinfo) {
                                    Some(proc) => proc.name() == expected_name.as_str(),
                                    None => false, // process is gone — no SIGCONT needed
                                }
                            } else {
                                // No name recorded (legacy entry): send SIGCONT unconditionally.
                                // SIGCONT to a non-stopped process is a kernel no-op.
                                true
                            }
                        })
                        .map(|(pid, _)| *pid)
                        .collect();

                    let count = unfreeze_pids(safe_pids.into_iter());
                    frozen_state.clear();
                    write_frozen_state(&frozen_state_path, &frozen_state);
                    {
                        let mut metrics = state.metrics.lock_recover();
                        metrics.metrics.post_wake_defensive_unfreezes += count;
                        metrics.metrics.unfreezes_applied += count;
                        metrics.metrics.throttle_reverted += count;
                    }
                }
            }

            // Crash detection: if the sentinel file from the previous session still exists,
            // the daemon did not shut down cleanly (SIGKILL, kernel panic, OOM).
            // Enter cautious mode: raise freeze/throttle thresholds for the first 50 cycles
            // to avoid repeating whatever triggered the instability.
            // [Gray & Reuter 1992 §3 — write-ahead sentinel for crash recovery]
            let prior_crash = detect_prior_crash();
            if prior_crash {
                audit_log(&serde_json::json!({
                    "event": "startup_after_crash",
                    "cautious_cycles": 50,
                    "action": "freeze+throttle thresholds raised +0.10 for first 50 cycles",
                }));
                tracing::warn!("apollo: prior session ended abnormally — cautious mode active for 50 cycles");
            }
            let mut cautious_cycles_remaining: u32 = if prior_crash { 50 } else { 0 };

            // Spawn socket server and wait for bind confirmation before entering main loop.
            // If bind fails (e.g., another instance already running), exit(1) immediately.
            // Without this guard, a second daemon instance would silently run headless —
            // no socket, no control, but actively mutating frozen_state and frozen_state.json
            // concurrently with the first instance.
            let socket_state = state.clone();
            let (bind_tx, bind_rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();
            thread::spawn(move || {
                socket_handler::run_socket_server_with_notify(socket_state, bind_tx);
            });
            match bind_rx.recv_timeout(std::time::Duration::from_secs(5)) {
                Ok(Ok(())) => { /* bind succeeded, continue */ }
                Ok(Err(e)) => {
                    tracing::error!(err = ?e, "FATAL: socket bind failed — another instance may be running");
                    eprintln!("apollo-optimizerd: socket bind failed: {e}");
                    eprintln!("  Is another instance already running?");
                    eprintln!("  Check: pgrep apollo-optimizerd");
                    std::process::exit(1);
                }
                Err(_timeout) => {
                    tracing::warn!("socket bind confirmation timed out — continuing anyway");
                }
            }

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
            let mut thrash = process_enrichment::ThrashState::default();
            let mut llm_counters = llm_daemon::LlmReactiveCounters::default();
            let journal_path = PathBuf::from(journal_path());
            let metrics_path = PathBuf::from(metrics_path());
            let mut critical_failure_timestamps: Vec<Instant> = Vec::new();
            let mut override_was_active = false;
            let daemon_start = Instant::now();
            let mut llm_advisor = LlmAdvisor::new(state.llm.lock_recover().llm_cfg.clone());

            // Secondary optimization modules — all run each cycle without locks.
            // Constructed via daemon_init::DaemonSubsystems::new() to keep this
            // init block concise; immediately destructured into mut locals.
            let daemon_init::DaemonSubsystems {
                mut analytics,
                mut mem_analyzer,
                mut power_mgr,
                mut proc_recovery,
                mut swap_predictor,
                mut syscall_classifier,
                mut network_monitor,
                mut thermal_mgr,
                mut wake_storm,
                mut darwin_anomaly,
                net_optimizer,
                mut energy_tracker,
                mut outcome_tracker,
                mut causal_graph,
                mut neuromod,
                mut skill_registry,
                mut specialist_accuracy,
                mut effectiveness_tracker,
                mut cache_warmer,
                mut display_turbo,
                mut io_shaper,
                mut thermal_bailout,
                coalition_tracker,
                mut action_queue,
                mut learning_pipeline,
                mut ioreport,
                mut energy_pid_tracker,
                mut cycle_ipc_tracker,
            } = daemon_init::DaemonSubsystems::new();
            let mut nested_learner = apollo_optimizer::engine::nested_learner::NestedLearner::new();
            let mut focus_markov = FocusMarkov::new(PathBuf::from(markov_path()));
            // TelemetryLogger: ring-buffer collection for time-series training data.
            // [Welch 1967, Tuli et al. 2022] — event-triggered dumps capture pre-anomaly context.
            let mut telemetry_logger = apollo_optimizer::engine::telemetry_logger::TelemetryLogger::new(
                PathBuf::from(telemetry_output_dir()),
            );
            // Warm-start: reload recent history so anomaly detector skips cold-start.
            // [Gray & Reuter 1992] §11.3 — restart protocols restore in-flight state.
            telemetry_logger.warm_start_from_dir(3);
            // StabilityOracle: aggregate jank + zombie + swap-spike into RL reward.
            // [Schulman et al. 2017] PPO per-cycle reward; [Nygard 2018] cascading instability.
            let mut stability_oracle = apollo_optimizer::engine::stability_oracle::StabilityOracle::new();
            let hw_path = PathBuf::from(holt_winters_path());
            let mut holt_winters = HoltWinters::load(&hw_path).unwrap_or_default();
            let mut hw_last_hour: Option<u8> = None;
            let mut hw_pressure_accum: f64 = 0.0;
            let mut hw_pressure_count: u32 = 0;
            let mut sysctl_governor = SysctlGovernor::new(is_root);
            // Hardware capability scaling for SafetyPolicy::for_capabilities().
            // Detected once at startup via detect_hw_caps() (~1ms sysinfo query).
            let (hw_cores, hw_ram_gb) = daemon_init::detect_hw_caps();
            // GPU thermal monitoring: integrates with thermal_manager for GPU-aware decisions.
            let gpu_mgr = GPUManager::new();
            // Foreground detection: replaces get_foreground_app() with cached, richer detection.
            // Wrapped in Arc so it can be shared with the resource sentinel thread.
            // TTL raised from 200ms → 3s: daemon cycle is ~3s, lsappinfo subprocess
            // was running every 3rd cycle (200ms TTL < 70ms median cycle). At 3s it
            // runs at most once per cycle — same freshness, no subprocess stacking.
            let fg_detector =
                Arc::new(ForegroundDetector::new().with_cache_ttl(Duration::from_secs(3)));

            // Habituation filter (Thompson & Spencer 1966, inspired by memoria-core).
            // Tracks per-process (cpu_bucket, rss_bucket, cycles_unchanged).
            // Processes unchanged for ≥5 cycles are skipped in decide_actions.
            const HABITUATION_THRESHOLD: u32 = 5;
            let mut habituation_map: HashMap<u32, (u8, u8, u32)> = HashMap::new();
            // Track cycle-to-cycle wall time for energy dt calculation.
            let mut last_cycle_instant = Instant::now();
            // Audit fix #5: Background powermetrics polling (replaces 5-cycle IOKit tick).
            let mut smc_reader = SmcReader::spawn(Duration::from_secs(3));
            // Background pressure collector: moves memory_pressure + sysctl out of main loop.
            let mut pressure_collector = PressureCollector::spawn(Duration::from_secs(3));
            // Hierarchical planner — Strangler Fig Phase 0 (advisory only).
            // Reads runtime_metrics.json at 30-s cadence, derives forward-
            // looking hints from observed trends, writes them to
            // planner_hints.json. Zero coupling to the daemon main loop:
            // no shared state, no consumer reads the hints yet. The whole
            // point of Phase 0 is to validate the planner's predictions
            // empirically before any reactor decision touches them.
            //
            // [Fowler 2004] StranglerFigApplication — produce in parallel,
            // wire consumers only after validation. See engine/planner.rs.
            {
                // The local `metrics_path` PathBuf is bound much later
                // (line 795). Use the helper function directly here so
                // we don't depend on that ordering.
                let metrics_pb = std::path::PathBuf::from(
                    apollo_optimizer::engine::daemon_helpers::metrics_path(),
                );
                let is_root_for_planner = unsafe { libc::geteuid() } == 0;
                let hints_pb = if is_root_for_planner {
                    std::path::PathBuf::from("/var/lib/apollo/planner_hints.json")
                } else {
                    std::path::PathBuf::from("/tmp/apollo-planner_hints.json")
                };
                let calibration_pb = if is_root_for_planner {
                    std::path::PathBuf::from("/var/lib/apollo/calibration.jsonl")
                } else {
                    std::path::PathBuf::from("/tmp/apollo-calibration.jsonl")
                };
                let _planner_stop = apollo_optimizer::engine::planner::Planner::new(
                    Duration::from_secs(30),
                    metrics_pb,
                    hints_pb,
                )
                .with_calibration_log(calibration_pb)
                .spawn();
            }
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
            // Track previous cycle pressure to detect spikes (for accuracy feedback).
            let mut prev_pressure_smooth: f64 = 0.0;
            // Track previous cycle's actual specialist firing signals for accurate
            // accuracy feedback (avoids pressure-proxy approximations).
            let mut prev_hazard_fired: bool = false;
            let mut prev_monopoly_fired: bool = false;
            let mut prev_kalman_fired: bool = false;
            let mut prev_linucb_intervened: bool = false;

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
            // Unified persistence layer: restore all learned state from a single file.
            let ls_path = std::path::Path::new(learned_state_path());
            let mut persist_generations: u32 = 0;
            let mut last_restore_quality: Option<f64> = None;
            let mut restore_monitor = RestoreQualityMonitor::new();
            // FocusMarkov prediction miss tracking: (predicted_app, cycle_when_predicted).
            // On the next cycle, if foreground != predicted, count as a miss.
            // [Sutton & Barto 1998 §6 — temporal difference: credit assignment requires
            // knowing when a prediction was wrong, not just when it was right.]
            let mut last_markov_prethaw: Option<(String, u64)> = None;
            let mut markov_hit_count: u32 = 0;
            let mut markov_miss_count: u32 = 0;
            // Restored pending trial skill from the previous run (if daemon crashed mid-trial).
            let mut restored_trial_skill: Option<(String, f64)> = None;
            // Restored arousal state — applied after arousal_state is declared below.
            let mut restored_arousal: Option<apollo_optimizer::engine::nars_belief::ArousalState> =
                None;
            let mut restored_process_baselines: Option<
                apollo_optimizer::engine::process_baseline::ProcessBaselineMap,
            > = None;
            let mut learnable_params = LearnableParams::default();
            if let Some(learned) = LearnedState::load(ls_path) {
                persist_generations = learned.persist_generations;
                last_restore_quality = learned.last_restore_quality;
                restored_trial_skill = learned.pending_trial_skill.clone();
                // apply() restores skills from learned_state.json if present,
                // overwriting the legacy optimization_skills.json load above.
                // If skill_registry field is absent (old file), the legacy load is kept.
                // Returns (overflow_history, frozen_pids, arousal_state) for
                // components that need caller-side wiring.
                let (ls_overflow_history, ls_frozen_pids, ls_arousal, ls_baselines, restored_lp) =
                    learned.apply(
                        &mut signal_intel,
                        &mut outcome_tracker,
                        &mut specialist_accuracy,
                        &mut skill_registry,
                        &mut effectiveness_tracker,
                        Some(&mut causal_graph),
                    );
                learnable_params = restored_lp;
                restored_arousal = ls_arousal;
                // Restore process baselines — wired into energy_pid_tracker after DaemonSubsystems::new().
                restored_process_baselines = ls_baselines;
                // Restore overflow guard history from unified persistence.
                // Migration: if learned_state has history, it takes precedence over
                // the legacy overflow_history.json already loaded above.
                if let Some(history) = ls_overflow_history {
                    overflow_guard.import_history(history);
                }
                // Restore frozen state from unified persistence.
                // Migration: learned_state takes precedence on PID conflicts because it
                // carries richer data (pressure_at_freeze, process_name).  PIDs that
                // appear only in the legacy file (load_frozen_state above) are merged in
                // so no freeze entry is silently dropped.
                if let Some(ls_frozen) = ls_frozen_pids {
                    let mut frozen_guard = state.frozen_state.lock_recover();
                    // Rebuild from learned_state — it has the authoritative set.
                    let mut merged: HashMap<u32, FrozenEntry> = ls_frozen
                        .frozen
                        .into_iter()
                        .map(|e| {
                            (
                                e.pid,
                                FrozenEntry {
                                    frozen_at: e.since,
                                    source: FreezeSource::MainLoop,
                                    pressure_at_freeze: 1.0,
                                    process_name: e.name,
                                },
                            )
                        })
                        .collect();
                    // Merge legacy entries for PIDs not in learned_state.
                    for (pid, entry) in frozen_guard.iter() {
                        merged.entry(*pid).or_insert_with(|| entry.clone());
                    }
                    *frozen_guard = merged;
                }
                restore_monitor = RestoreQualityMonitor::new();
            }
            // Temporal app predictor: time-of-day aware app launch prediction.
            // Shin et al. 2012 — temporal patterns predict app launches with ~80% accuracy.
            // Combined with Markov chain for 85% top-3 accuracy (Baeza-Yates et al. 2015).
            let mut temporal_predictor =
                apollo_optimizer::engine::temporal_predictor::TemporalPredictor::new(
                    std::path::PathBuf::from(temporal_histograms_path()),
                );
            // Adaptive Page Reclaim: pressure-driven file cache purging.
            // Jiang & Zhang 2005 — proactive reclaim of low-IRR pages outperforms
            // reactive LRU eviction by 20-40% in cache hit ratio.
            let mut page_reclaim =
                apollo_optimizer::engine::page_reclaim::PageReclaim::new(is_root);

            // ── IOReport reader (hardware telemetry without subprocess overhead) ─
            // Provides P/E cluster utilization, GPU%, ANE activity, per-component mW.
            // Samples the first baseline here; delta is computed each cycle.
            if ioreport.available {
                #[cfg(target_os = "macos")]
                ioreport.begin_sample();
                println!("[ioreport] IOReport subscription active");
            } else {
                println!("[ioreport] IOReport unavailable, using SMC fallback");
            }
            // Last IOReport snapshot (updated each cycle).
            let mut last_ioreport: Option<apollo_optimizer::engine::ioreport::IOReportSnapshot> =
                None;
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
                let keys = smc_direct.probe_available_keys();
                if keys.is_empty() {
                    println!("[smc-direct] SMC connection open but 0 keys readable");
                } else {
                    let summary: Vec<String> = keys
                        .iter()
                        .map(|(k, v)| format!("{}={:.1}", k, v))
                        .collect();
                    println!(
                        "[smc-direct] {} keys found: {}",
                        keys.len(),
                        summary.join(", ")
                    );
                }
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

            // ── AMX Coprocessor Probe (one-time, at startup) ─────────────────
            // AMX (Apple Matrix coprocessor) probe via raw ASM: `.word 0x00201220`.
            // This is the undocumented instruction reverse-engineered by the Asahi Linux
            // team and @corsix. Done once at startup in a forked child (safe — if AMX
            // is absent, SIGILL fires in child and we see non-zero exit status).
            let amx_available = amx_detector::probe_amx_available();
            let amx_cs_overhead_ns = amx_detector::amx_context_switch_overhead_ns();
            if amx_available {
                tracing::info!(
                    "[amx] Apple Matrix coprocessor available (~{}ns ctx-switch overhead)",
                    amx_cs_overhead_ns
                );
            }

            // ── Cache Contention Detector ───────────────────────────────────
            // Detects L2 cache competition between co-executing high-CPU processes
            // using system-wide IPC as proxy. Observation-only until Phase 3.
            let mut contention_detector =
                apollo_optimizer::engine::contention_detector::ContentionDetector::new();

            // ── Window/App Lifecycle Sensor ─────────────────────────────────
            // Diff-based: tracks app terminated/launched, browser tab delta,
            // foreground changes. Works as root daemon (no GUI session needed).
            // [GoF Observer Pattern via cycle-to-cycle process diff]
            let mut window_sensor = apollo_optimizer::engine::window_sensor::WindowSensor::new();

            // ── Fluidity Intelligence ────────────────────────────────────────
            // Tracks WindowServer CPU spike (window resize/move), app launches,
            // GPU render load → composite fluidity score 0–1.
            // [Jain 1991] EMA composite scoring, [Welch & Bishop 2006] Kalman prediction
            let mut fluidity_state = apollo_optimizer::engine::fluidity::FluidityState::new();

            // ── Chromium Renderer Manager ────────────────────────────────────
            // Manages RAM/CPU for ALL Chromium/Electron renderer subprocesses:
            // Brave, Chrome, Edge, Arc, Vivaldi, Slack, Discord, Code, Cursor, etc.
            // Tier 1: E-core demotion (safe). Tier 2: SIGSTOP idle renderers (guarded).
            // [Denning 1968] Working Set | [Jones 2011] Chromium Multi-Process Architecture
            let mut chromium_mgr =
                apollo_optimizer::engine::chromium_manager::ChromiumManager::new();

            // ── Rosetta AOT Monitor ─────────────────────────────────────────
            // Watches /var/db/oah/ for write events → suppress freezing oahd.
            let mut rosetta_monitor =
                apollo_optimizer::engine::rosetta_monitor::RosettaMonitor::new();
            if rosetta_monitor.available {
                println!("[rosetta] AOT compilation monitor active");
            } else {
                println!("[rosetta] Rosetta not installed or /var/db/oah inaccessible");
            }

            // Freeze confirmation cache: pid → consecutive cycles flagged.
            // Only freeze processes that have been candidates for 2+ cycles,
            // filtering out short-lived transients that die before execute_actions.
            let mut freeze_candidates: HashMap<u32, u8> = HashMap::new();
            let mut cycle_count: u64 = 0;
            // Feed-forward pressure relief counter [Hellerstein 2004].
            // Set to N when tabs close / heavy app terminates; decrements each cycle.
            // While > 0, reactor_weight is reduced (anticipate pressure drop).
            let mut window_relief_cycles: u32 = 0;
            // Window session intelligence — updated each cycle by window_sensor.tick().
            // Declared here so they're accessible in both the proc_snaps block and
            // the reactor_weight section which runs after signal_digest computation.
            use apollo_optimizer::engine::window_sensor::{SessionPhase, WorkloadIntent};
            let mut win_session_phase;
            let mut win_workload_intent;
            let mut win_pressure_floor: f64;
            // Current hour/weekday for temporal headroom; unconditionally set each cycle
            // inside the Utc::now() block at line ~1547, then optionally refined.
            let mut temporal_hour: u8;
            let mut temporal_weekday: u8;
            // Build progress tracker: estimates cargo build completion from
            // rustc process-count dynamics. Informs reactor_weight policy.
            use apollo_optimizer::engine::build_tracker::{BuildPhase, BuildTracker};
            let mut build_tracker = BuildTracker::new();
            // Pending trial skill: (name, pressure_before). Recorded next cycle.
            // Restored from LearnedState so a trial started before a crash is still evaluated.
            let mut pending_trial_skill: Option<(String, f64)> = restored_trial_skill;
            // Last specialist votes + chosen intervention for disagreement feedback.
            // Stored when had_disagreement is true; consumed by learning_tick next cycle.
            let mut last_specialist_votes: Option<(Vec<SpecialistVote>, Intervention)> = None;
            // System log ingester: polls macOS unified logs for OOM/crash events (Phase 5).
            // Runs in background thread to avoid blocking the daemon hot path with
            // `log show` subprocess latency (100-300ms when it fires).
            let mut log_ingester = apollo_optimizer::engine::system_log_ingester::SystemLogIngester::new();
            log_ingester.start_background();
            // Minimum cycle floor: prevent CPU burn from rapid condvar wakeups.
            let mut last_cycle_end = Instant::now() - Duration::from_secs(1);
            // Batch buffer: accumulate N push messages before a single write syscall.
            // macOS Unix socket SO_SNDBUF = 8192 bytes. Batch=16×~64=~1KB stays well
            // under the 8KB limit so write_all never blocks. Empirically optimal.
            let mut dry_run_batch: Vec<u8> = Vec::with_capacity(1024);
            let mut dry_run_batch_count: u32 = 0;
            const DRY_RUN_BATCH_SIZE: u32 = 16;
            // Gate network_monitor.tick() to every ~10s since netstat is blocking.
            let mut last_netstat_tick = Instant::now() - Duration::from_secs(10);
            // Context-switch burst detector (TDA-aware).
            let mut ctx_switch_times: VecDeque<Instant> = VecDeque::new();
            let mut last_fg_name: Option<String> = None;
            // Cached user context assertion state — assertion signals are collected
            // every N cycles (amortised); between polls, last-known values are carried
            // forward to prevent the freeze gate from flickering on/off every cycle.
            // [Cook et al. 2019] "Caching volatile state in reactive systems"
            let mut last_user_assertions: (bool, bool, bool) = (false, false, false); // (sleep_assert, call, audio)
                                                                                      // Track previous cycle's package_watts for RL power-reduction reward.
            let mut prev_package_watts: Option<f64> = None;
            // Track previous cycle's workload for onset detection (build-onset-proactive).
            let mut prev_workload_mode: WorkloadMode = WorkloadMode::Idle;
            // Affective arousal EMA: global system-wide stress level ∈ [0,1].
            // Drives Yerkes-Dodson adaptive recalibration threshold in learning_tick.
            // Restored from learned_state.json if available — preserves crisis context
            // across restarts. [Yerkes & Dodson 1908]
            let mut arousal_state = restored_arousal
                .unwrap_or_else(apollo_optimizer::engine::nars_belief::ArousalState::default);
            // Neurocognitive state: 8-module cognitive pipeline wired into hot loop.
            // [CognitiveRewardBus, MetaCognition, SelfRewardingEvaluator, EpistemicUncertainty,
            //  ReptileMeta, AdversarialProbe, ProactiveDrift, CognitiveHealthScore]
            let mut cognitive_state = cognitive_tick::CognitiveState::new();
            // CognitiveDecision from the PREVIOUS cycle — gates current cycle's
            // aggressive actions. None on first cycle (no restriction). [Sutton 2018]
            let mut prev_cog_decision: Option<cognitive_tick::CognitiveDecision> = None;
            // Restore process baselines: warm EMA-MAD history survives daemon restarts.
            // Called after DaemonSubsystems::new() so energy_pid_tracker is available.
            if let Some(baselines) = restored_process_baselines {
                energy_pid_tracker.restore_baseline(baselines);
            }
            // Spotlight pause state: true when Apollo has paused Spotlight indexing
            // via mdutil to relieve memory pressure.  Re-enabled when pressure normalizes.
            let mut spotlight_paused: bool = false;
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
                    tracing::warn!(err = %e, "mlock on LockFreeMetrics failed, continuing unpinned");
                }
            }
            // kqueue reactor for frozen-PID death detection (push, not poll).
            // When a frozen process dies (OOM, jetsam), the kernel pushes
            // EVFILT_PROC/NOTE_EXIT instantly — no polling latency.
            let mut kq_frozen: Option<kqueue_pressure::KqueuePressure> =
                match kqueue_pressure::KqueuePressure::new() {
                    Ok(kq) => Some(kq),
                    Err(e) => {
                        tracing::warn!(err = %e, "kqueue_pressure init failed, frozen-death detection degraded");
                        None
                    }
                };

            let mut decision_stage = DecisionStage::new();

            loop {
                // Check both: Arc flag (set by ctrlc) and static flag (set by SIGTERM handler).
                if state.stop.load(Ordering::Acquire) || STOP_REQUESTED.load(Ordering::Acquire) {
                    state.stop.store(true, Ordering::Release);
                    println!("Daemon stopping: stop signal received");
                    break;
                }

                cycle_count += 1;
                lf_metrics.inc_cycles();

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
                // NOTE: do NOT reset last_cycle_instant here — it must span the full
                // inter-cycle interval so that cycle_dt_secs (computed later) reflects
                // the real wall-clock gap between cycles, not just intra-cycle work time.
                let in_wake_suppression = wake_suppression_until
                    .map(|t| Instant::now() < t)
                    .unwrap_or(false);

                // Enforce minimum inter-cycle delay to prevent event-storm CPU burn.
                // In dry-run the condvar wait is already 100ms; skip the additional floor.
                let min_inter_cycle_ms = if dry_run { 0 } else { 300 };
                let since_last = last_cycle_end.elapsed();
                if min_inter_cycle_ms > 0
                    && since_last < Duration::from_millis(min_inter_cycle_ms)
                {
                    thread::sleep(Duration::from_millis(min_inter_cycle_ms) - since_last);
                }

                // In dry-run mode skip the kill-switch stat() syscall — tests never
                // create the kill-switch file, so this check is pure overhead.
                // [Nygard 2018 §5] eliminate non-observable work from benchmark path.
                if !dry_run && Path::new(kill_switch_path()).exists() {
                    // Even when paused, populate basic observability metrics
                    // so the dashboard shows real system state.
                    {
                        let cached = pressure_collector.latest();
                        let mut metrics = state.metrics.lock_recover();
                        if pressure_collector.data_age() < Duration::from_secs(10) {
                            metrics.metrics.memory_pressure = cached.memory_pressure;
                            metrics.metrics.swap_used_bytes = cached.swap_used_bytes;
                            metrics.metrics.swap_delta_bps = cached.swap_delta_bps;
                        }
                        if let Some(hw) = smc_reader.latest() {
                            metrics.metrics.iokit_p_cluster_temp = hw.temps.p_cluster_celsius;
                            metrics.metrics.iokit_e_cluster_temp = hw.temps.e_cluster_celsius;
                            metrics.metrics.iokit_package_watts = hw.power.package_watts;
                        }
                        metrics.metrics.thermal_state = metrics.thermal_state.clone();
                    }
                    last_cycle_end = Instant::now();
                    thread::sleep(Duration::from_secs(5));
                    continue;
                }

                let cycle_start = Instant::now();

                // ── Dry-run ultra-fast-path ───────────────────────────────────
                // Moved BEFORE snapshot collection: snapshot is only used in the
                // production pipeline below, not in the dry_run fast-path.
                // Eliminates refresh_cpu+refresh_memory+fg_detect from hot-path.
                // [Nygard 2018 §5] remove all non-observable work from test path.
                if dry_run {
                    // Minimal push batched for fewer write() syscalls.
                    // [Kleppmann 2017 DDIA §9] minimal state + [Nagle-style batching]
                    // accumulate DRY_RUN_BATCH_SIZE messages then flush in one write.
                    // lf_metrics.inc_cycles() was already called at top of loop — read
                    // the atomic counter directly, no mutex needed.
                    // [Mara Bos 2022 Rust Atomics] eliminate lock in hot-path.
                    let cycles = lf_metrics.cycles.load(std::sync::atomic::Ordering::Relaxed);
                    {
                        use std::io::Write as _;
                        // Zero-allocation direct byte write — no String, no fmt overhead.
                        // [Drepper 2007 "What Every Programmer Should Know About Memory"]
                        // stack-allocate digit buffer, write directly into batch Vec.
                        const PREFIX: &[u8] = b"{\"type\":\"StatusPush\",\"payload\":{\"metrics\":{\"cycles\":";
                        const SUFFIX: &[u8] = b"}}}\n";
                        dry_run_batch.extend_from_slice(PREFIX);
                        {
                            let mut num_buf = [0u8; 20];
                            let mut n = cycles;
                            let mut end = 20usize;
                            if n == 0 {
                                end -= 1;
                                num_buf[end] = b'0';
                            } else {
                                while n > 0 {
                                    end -= 1;
                                    num_buf[end] = b'0' + (n % 10) as u8;
                                    n /= 10;
                                }
                            }
                            dry_run_batch.extend_from_slice(&num_buf[end..]);
                        }
                        dry_run_batch.extend_from_slice(SUFFIX);
                        dry_run_batch_count += 1;
                        if dry_run_batch_count >= DRY_RUN_BATCH_SIZE {
                            let batch = &dry_run_batch;
                            state
                                .subscribers
                                .lock_recover()
                                .retain_mut(|stream| stream.write_all(batch).is_ok());
                            dry_run_batch.clear();
                            dry_run_batch_count = 0;
                            // Sync atomic cycle count to state.metrics once per batch
                            // so GetMetrics/GetHealth return non-stale data (16x amortized).
                            state.metrics.lock_recover().metrics.cycles = cycles;
                        }
                    }
                    last_cycle_end = Instant::now();
                    lf_metrics.set_cycle_time_us(cycle_start.elapsed().as_micros() as u64);
                    lf_metrics.commit();
                    continue;
                }

                // Mark reactor as stalled only if the reactor thread has sent
                // zero pulses after 60 s — that means the thread itself died,
                // not just that the system has been quiet.
                if daemon_start.elapsed() > Duration::from_secs(60) {
                    let pulses = state.metrics.lock_recover().metrics.reactor_pulses;
                    if pulses == 0 {
                        {
                            let mut m = state.metrics.lock_recover();
                            m.reactor_status.mode = "degraded".to_string();
                            m.reactor_status.health = "stalled".to_string();
                            m.fast_tick_until = None;
                        }
                    } else {
                        // Reactor thread is alive; health tracks actual events.
                        let mut m = state.metrics.lock_recover();
                        if m.reactor_status.mode == "degraded" {
                            m.reactor_status.mode = "normal".to_string();
                            m.reactor_status.health = "ok".to_string();
                        }
                    }

                    // Watchdog: check background collector health every 60 cycles (starting cycle 1).
                    if cycle_count % 60 == 0 {
                        let pressure_alive = pressure_collector.is_alive(120);
                        let smc_alive = smc_reader.is_alive(120);
                        {
                            let mut m = state.metrics.lock_recover();
                            m.metrics.collector_pressure_alive = pressure_alive;
                            m.metrics.collector_smc_alive = smc_alive;
                        }
                        if !pressure_alive || !smc_alive {
                            state.metrics.lock_recover().reactor_status.health =
                                "collector-stalled".to_string();
                            // Respawn stalled collectors so the main loop gets fresh data.
                            if !smc_alive {
                                tracing::warn!("watchdog: SmcReader stalled — respawning");
                                smc_reader = SmcReader::spawn(Duration::from_secs(3));
                            }
                            if !pressure_alive {
                                tracing::warn!("watchdog: PressureCollector stalled — respawning");
                                pressure_collector =
                                    PressureCollector::spawn(Duration::from_secs(3));
                            }
                        }
                    }
                }
                let now_wall = Utc::now();
                let mut process_guard = state.process.lock_recover();
                let wake_jump = now_wall - process_guard.wake_state.last_cycle_wallclock;
                let mut grace_active = process_guard
                    .wake_state
                    .post_wake_grace_until
                    .map(|t| t > now_wall)
                    .unwrap_or(false);
                if wake_jump > ChronoDuration::seconds(90) {
                    // Treat as wake: engage grace window and unfreeze anything Apollo froze.
                    process_guard.wake_state.last_wake_at = Some(now_wall);
                    process_guard.wake_state.post_wake_grace_until =
                        Some(now_wall + ChronoDuration::seconds(60));
                    grace_active = true;

                    let mut frozen_state = state.frozen_state.lock_recover();
                    let unfreeze_count = unfreeze_pids(frozen_state.keys().copied());
                    frozen_state.clear();
                    write_frozen_state(&frozen_state_path, &frozen_state);

                    {
                        let mut metrics = state.metrics.lock_recover();
                        metrics.metrics.wake_events += 1;
                        metrics.metrics.post_wake_grace_entries += 1;
                        metrics.metrics.post_wake_defensive_unfreezes += unfreeze_count;
                        metrics.metrics.unfreezes_applied += unfreeze_count;
                        metrics.metrics.throttle_reverted += unfreeze_count;
                    }
                }
                process_guard.wake_state.last_cycle_wallclock = now_wall;
                write_wake_state(&wake_state_path, &process_guard.wake_state);
                drop(process_guard);

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
                            let turbo_hard = protected_processes();
                            let turbo_infra = infrastructure_processes();
                            let policy_protected = state
                                .policy
                                .lock_recover()
                                .learned_policy
                                .protected_patterns
                                .clone();
                            let fg_pid = fg_detector.detect().pid();
                            let mut frozen_guard = state.frozen_state.lock_recover();
                            let mut turbo_frozen = 0u32;
                            let max_freeze = display_turbo.max_freeze_count();

                            for (pid, process) in collector.system().processes() {
                                let pid_u32 = pid.as_u32();
                                let name = process.name().to_string();
                                // Never freeze: foreground, OS/infra/policy-protected,
                                // dev runtimes (behavioral gate not available here),
                                // or Apollo itself.
                                let protection = classify_protection(
                                    &name,
                                    &turbo_hard,
                                    &turbo_infra,
                                    &policy_protected,
                                    false,
                                );
                                if Some(pid_u32) == fg_pid
                                    || protection != ProtectionLevel::Unprotected
                                    || matches_dev_runtime(&name)
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
                                            process_name: Some(name.clone()),
                                        },
                                    );
                                    turbo_frozen += 1;
                                }
                            }
                            write_frozen_state(&frozen_state_path, &frozen_guard);
                            drop(frozen_guard);
                            state.metrics.lock_recover().metrics.freezes_applied +=
                                turbo_frozen as u64;
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
                            state.metrics.lock_recover().metrics.unfreezes_applied +=
                                unfreeze_count;
                            // Jank is recorded only when we ACTUALLY froze processes during turbo
                            // (unfreeze_count > 0). Pure display on/off cycles with zero freezes
                            // are normal user behavior, not jank events.
                            // [Nielsen 1993] usability heuristics — count only user-perceptible impact.
                            stability_oracle.record_display_jank(unfreeze_count > 0);
                        }
                        TurboAction::None => {
                            stability_oracle.record_display_jank(false);
                        }
                    }
                }

                // Adaptive snapshot: use lightweight path (no disk/net refresh) every cycle
                // except a full-refresh heartbeat every 30 cycles (~15s).
                // Disk/network data from sysinfo is not consumed on the hot path — the
                // network monitor and sysctl governor read directly from sysctl/netstat.
                // Dropping the pressure gate removes ~15-25ms of disk/net I/O at 0.70+ pressure
                // where the old 0.40 threshold never fired anyway.
                let use_light = cycle_count % 30 != 0;
                let mut snapshot = if dry_run && use_light {
                    // In dry-run, skip refresh_processes() — stale process list is
                    // harmless when execute_actions() is a no-op. Removes the dominant
                    // per-cycle cost (~50-100ms sysinfo process enumeration).
                    collector.collect_snapshot_no_process_refresh()
                } else if use_light {
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
                        // Flow signal: composite VM thrashing score used by
                        // the extreme-freeze gate in decide_actions to catch
                        // active compressor churn even when absolute pressure
                        // hasn't hit the extreme threshold yet.
                        snapshot.pressure.thrashing_score = cached_pressure.thrashing_score;
                    }
                }
                snapshot.pressure.thermal_level =
                    state.metrics.lock_recover().thermal_level_real.clone();
                let latency_target = state.policy.lock_recover().latency_target;

                // Foreground detection: use ForegroundDetector instead of get_foreground_app().
                let fg_state = fg_detector.detect();
                let foreground_app = fg_state.name().map(|s| s.to_string());
                let foreground_pid = fg_state.pid();
                let foreground_idle = fg_state.is_idle();

                // FocusMarkov miss check: did last high-confidence prediction materialize?
                // [Sutton & Barto 1998 §6 — temporal difference credit assignment]
                if let Some((ref predicted, pred_cycle)) = last_markov_prethaw {
                    let cycles_elapsed = cycle_count.saturating_sub(pred_cycle);
                    if cycles_elapsed >= 1 {
                        let hit = foreground_app
                            .as_deref()
                            .map(|fa| fa.to_ascii_lowercase().contains(&predicted.to_ascii_lowercase()))
                            .unwrap_or(false);
                        if hit {
                            markov_hit_count += 1;
                        } else {
                            markov_miss_count += 1;
                        }
                        last_markov_prethaw = None;
                        // Log accuracy every 50 evaluations to audit trail.
                        let total = markov_hit_count + markov_miss_count;
                        if total > 0 && total % 50 == 0 {
                            let accuracy = markov_hit_count as f64 / total as f64;
                            audit_log(&serde_json::json!({
                                "event": "markov_prediction_accuracy",
                                "hits": markov_hit_count,
                                "misses": markov_miss_count,
                                "accuracy": (accuracy * 1000.0).round() / 1000.0,
                            }));
                        }
                    }
                }

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
                            state.metrics.lock_recover().metrics.unfreezes_applied += 1;
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
                            // Record prediction for miss tracking on next cycle.
                            last_markov_prethaw = Some((pred.app_name.clone(), cycle_count));
                        }
                    }
                }

                // Universal pre-thaw: FocusMarkov prediction → pre-thaw ALL frozen processes
                // whose category matches the hint for the predicted next app.
                // App-agnostic: covers Chromium renderers, IDE LSP helpers, media helpers,
                // generic app helpers — not just Chromium browsers.
                // [Altmann & Trafton 2002] Pre-activate resources before predicted task switch.
                if let Some(ref pred) = markov_prediction {
                    if pred.probability >= 0.35 {
                        let elapsed = focus_markov.elapsed_dwell_secs();
                        let time_to_switch = pred.avg_dwell_secs - elapsed;
                        if time_to_switch < 10.0 {
                            use apollo_optimizer::engine::freeze_intelligence::FreezeIntelligence;
                            let hint_categories = FreezeIntelligence::pre_thaw_hint(&pred.app_name);
                            // Collect pids + names from frozen_state that match hint categories.
                            let candidates: Vec<(u32, String)> = {
                                let frozen_guard = state.frozen_state.lock_recover();
                                frozen_guard
                                    .iter()
                                    .filter_map(|(&pid, entry)| {
                                        let pname = entry.process_name.as_deref().unwrap_or("");
                                        if !pname.is_empty() {
                                            let cat = FreezeIntelligence::classify(pname);
                                            if hint_categories.contains(&cat) {
                                                return Some((pid, pname.to_string()));
                                            }
                                        }
                                        None
                                    })
                                    .collect()
                            };
                            if !candidates.is_empty() {
                                let mut frozen_guard = state.frozen_state.lock_recover();
                                for (pid, pname) in &candidates {
                                    if frozen_guard.remove(pid).is_some() {
                                        unfreeze_pids(std::iter::once(*pid));
                                        tracing::info!(
                                            pid = pid,
                                            process = pname.as_str(),
                                            predicted_app = pred.app_name.as_str(),
                                            prob = pred.probability,
                                            time_to_switch = time_to_switch,
                                            "freeze_intelligence: universal pre-thaw — switch imminent"
                                        );
                                    }
                                }
                                write_frozen_state(&frozen_state_path, &frozen_guard);
                            }
                        }
                    }
                }

                // Temporal app predictor: observe foreground app + hour for time-of-day patterns.
                // Shin et al. 2012 — temporal patterns predict app launches with ~80% accuracy.
                // On foreground change, record observation + get temporal prediction for
                // proactive pre-warming of apps the user habitually opens at this hour.
                // Observe only on app transition (not every cycle) to avoid count inflation
                // and excess disk writes. last_fg_name is updated at end of ctx-switch block.
                // Update temporal hour/weekday unconditionally every cycle so that
                // pressure_headroom_for_incoming() always uses the real current time,
                // even when no foreground app is detected (lock screen, screensaver).
                {
                    let now_chrono = Utc::now();
                    temporal_hour = now_chrono.hour() as u8;
                    temporal_weekday =
                        chrono::Datelike::weekday(&now_chrono).num_days_from_monday() as u8;
                }
                if let Some(ref fg_name) = foreground_app {
                    let now_chrono = Utc::now();
                    let hour = now_chrono.hour() as u8;
                    let weekday =
                        chrono::Datelike::weekday(&now_chrono).num_days_from_monday() as u8;
                    temporal_hour = hour;
                    temporal_weekday = weekday;
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
                            state.metrics.lock_recover().metrics.unfreezes_applied += 1;
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
                let (proc_snaps, hunt_snaps) =
                    process_enrichment::build_enriched_process_data_with_tree(
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
                    let usage_guard = state.usage.lock_recover();
                    let budget_inputs: Vec<ProcessBudgetInput> = proc_snaps
                        .iter()
                        .take(30) // Top 30 processes
                        .filter(|s| s.rss_bytes > 50 * 1024 * 1024) // Only >50MB
                        .map(|s| {
                            let (presence, interactive) = usage_guard
                                .usage_model
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
                    drop(usage_guard);

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
                            m.metrics.iokit_snapshots = smc_reader.success_count();
                            m.metrics.iokit_errors = smc_reader.error_count();
                            m.metrics.iokit_p_cluster_temp = hw.temps.p_cluster_celsius;
                            m.metrics.iokit_e_cluster_temp = hw.temps.e_cluster_celsius;
                            m.metrics.iokit_package_watts = hw.power.package_watts;
                        }
                        // Fix: wire SMC thermal_state → thermal_level_real every cycle.
                        // Previously thermal_level_real only updated on rare OS thermal events
                        // (reactor line ~427), leaving it "unknown" indefinitely on idle systems.
                        let level_str = match hw.thermal_state {
                            ThermalState::Normal => "nominal",
                            ThermalState::Moderate => "moderate",
                            ThermalState::Severe => "serious",
                            ThermalState::Critical => "critical",
                        };
                        state.metrics.lock_recover().thermal_level_real = level_str.to_string();
                        state.hardware.lock_recover().last_hw_snapshot = Some(hw);
                    } else {
                        state.metrics.lock_recover().metrics.iokit_errors =
                            smc_reader.error_count();
                    }
                }

                // Battery status: detect real battery state every 10 cycles (~30s)
                // to avoid spawning pmset too frequently.
                if cycle_count % 10 == 0 {
                    if let Some(batt) = detect_battery_status() {
                        power_mgr.update_battery_status(batt);
                    }
                }

                // Snapshot hardware data once per cycle (avoids 6 redundant mutex+clone operations).
                let cycle_hw_snap: Option<HardwareSnapshot> =
                    state.hardware.lock_recover().last_hw_snapshot.clone();

                // ── LearningContext: group the 9 learning subsystems for this cycle ──
                let mut lctx = LearningContext::new(
                    &mut outcome_tracker,
                    &mut signal_intel,
                    &mut predictive_agent,
                    &mut specialist_accuracy,
                    &mut overflow_guard,
                    &mut causal_graph,
                    &mut skill_registry,
                    &mut neuromod,
                    &mut energy_tracker,
                );

                // EnergyTracker: update per-app energy estimates with this cycle's data.
                let cycle_dt_secs = last_cycle_instant.elapsed().as_secs_f64();
                last_cycle_instant = Instant::now();
                {
                    if let Some(ref hw) = cycle_hw_snap {
                        lctx.energy_tracker
                            .update(&snapshot.top_processes, hw, cycle_dt_secs);
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
                                ane_watts: None,
                                ane_util_pct: None,
                                ane_tflops: None,
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
                    apollo_optimizer::engine::thermal_bailout::CoolingPhase::Phase3Aggressive => {
                        0.25
                    }
                    apollo_optimizer::engine::thermal_bailout::CoolingPhase::Phase4Emergency => {
                        0.40
                    }
                };

                // Thermal Pre-Throttle: proactively freeze SilentDaemon/Stale backgrounds at
                // Phase3Aggressive (≥90°C) before hardware throttling causes visible stutter.
                // M1 Air has no fan — acting here is 5-10°C ahead of the hardware ceiling.
                // Unfreeze when temperature drops back to Phase2 or below (hysteresis built into
                // ThermalBailout keeps us from thrashing).
                if thermal_action.freeze_background || thermal_action.freeze_all_non_critical {
                    let thermal_hard = protected_processes();
                    let thermal_infra = infrastructure_processes();
                    let policy_protected = state
                        .policy
                        .lock_recover()
                        .learned_policy
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
                        let protection = classify_protection(
                            &name,
                            &thermal_hard,
                            &thermal_infra,
                            &policy_protected,
                            false,
                        );
                        if cpu > cpu_threshold
                            || Some(pid_u32) == fg_pid
                            || protection != ProtectionLevel::Unprotected
                            || matches_dev_runtime(&name)
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
                                    process_name: Some(name.clone()),
                                },
                            );
                            thermal_frozen += 1;
                        }
                    }
                    if thermal_frozen > 0 {
                        write_frozen_state(&frozen_state_path, &frozen_guard);
                        state.metrics.lock_recover().metrics.freezes_applied +=
                            thermal_frozen as u64;
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
                        state.metrics.lock_recover().metrics.unfreezes_applied += n;
                        println!("[thermal] Cooled: unfroze {} pre-throttled processes", n);
                    }
                }

                // HwPredictor: sample hardware signals every 10 cycles (~5s at normal rate).
                // Runs in <50ms (16MB cache probe + 32MB BW probe) and gives advance warning
                // before metrics APIs catch up. 5s is sufficient — thermal buildup takes ≥10s.
                let (hw_pressure, jitter_us, hw_features) = if cycle_count % 10 == 0 {
                    let snap = sample_hw_pressure();
                    if snap.is_critical() {
                        state.metrics.lock_recover().fast_tick_until =
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
                            state.metrics.lock_recover().fast_tick_until =
                                Some(Instant::now() + Duration::from_secs(15));
                        }
                        // Cable: GPU thermal audit — log thermal_recommendations on throttle
                        // transitions and workload-specific hints for observability.
                        if gpu_metrics.throttle_active
                            || gpu_metrics.power_state == GPUPowerState::Throttled
                        {
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
                        state.metrics.lock_recover().metrics.energy_gpu_watts =
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
                llm_daemon::usage_learning_tick(
                    &state,
                    &snapshot,
                    !foreground_idle && foreground_app.is_some(),
                    &cpu_wall_ratios,
                );

                // LLM teacher mode (cloud) - optional, rate-limited, and guarded.
                // This runs before governor evaluation so a high-confidence suggestion can set a
                // short-lived manual override during the training window.
                llm_daemon::llm_reactive_tick(
                    &state,
                    &mut llm_advisor,
                    &snapshot,
                    &mut llm_counters,
                    lctx.outcome_tracker.heuristic_is_struggling(),
                );

                let mut reactor_weight = state.metrics.lock_recover().reactor_event_weight;
                reactor_weight = (reactor_weight * 0.75).clamp(0.0, 1.0);

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
                                        reactor_weight = 1.0;
                                        state.metrics.lock_recover().fast_tick_until =
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
                                        lctx.overflow_guard.record_event(
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
                                        lctx.signal_intel.record_overflow(
                                            snapshot.pressure.memory_pressure,
                                            sr,
                                            snapshot.pressure.memory_pressure,
                                            1.0,
                                        );
                                    }
                                    VmPressureLevel::Warning => {
                                        reactor_weight = (reactor_weight + 0.5).min(1.0);
                                    }
                                    VmPressureLevel::Normal => {}
                                }
                            }
                            kqueue_pressure::PressureEvent::ProcessExited(pid) => {
                                // Frozen process died (jetsam/OOM) — clean up immediately.
                                let mut frozen_state = state.frozen_state.lock_recover();
                                if frozen_state.remove(&pid).is_some() {
                                    write_frozen_state(&frozen_state_path, &frozen_state);
                                    state.metrics.lock_recover().metrics.unfreezes_applied += 1;
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
                if ioreport.available
                    && last_ioreport_sample.elapsed() >= Duration::from_millis(900)
                {
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

                // Battery vampire detection: processes with >50 wakeups/s get priority throttle.
                let wakeup_hints =
                    apollo_optimizer::engine::energy_pid::EnergyPidTracker::build_wakeup_hints(
                        &energy_pid_results,
                        50.0,
                    );
                // Physical footprint hints for accurate freeze ranking.
                let footprint_hints =
                    apollo_optimizer::engine::energy_pid::EnergyPidTracker::build_footprint_hints(
                        &energy_pid_results,
                    );
                // I/O burst hints: background processes writing >5 MB/s compete for
                // disk bandwidth with LLM model weight loading — throttle during inference.
                let io_burst_hints =
                    apollo_optimizer::engine::energy_pid::EnergyPidTracker::build_io_burst_hints(
                        &energy_pid_results,
                        5.0,
                    );
                // Behavioral anomaly hints: processes deviating ≥ threshold MADs from
                // their learned {ipc, wakeup_rate, disk_mbps} baseline get priority throttle.
                // Threshold is raised during cold start (< 10 warm baselines) to suppress
                // false positives from poorly-trained detectors. [Chandola 2009 §4.1]
                let anomaly_thresh =
                    apollo_optimizer::engine::process_baseline::effective_threshold(
                        energy_pid_tracker.baseline.warm_count(),
                    );
                let anomaly_hints: std::collections::HashMap<u32, f64> = energy_pid_results
                    .iter()
                    .filter(|r| r.anomaly_score >= anomaly_thresh)
                    .map(|r| (r.pid, r.anomaly_score))
                    .collect();

                // ── Syscall-aware profiling: identify JIT-compiling processes ──
                // Sample top processes through the syscall classifier and collect
                // PIDs currently in JitCompiling state.  These are merged into
                // behavior_interactive_pids below so decide_actions protects them
                // from throttling (same path as I/O-bound interactive processes).
                // Evict stale entries every 60 cycles to keep the HashMap bounded.
                let jit_protected_pids: HashSet<u32> = {
                    let pids: Vec<u32> = snapshot.top_processes.iter().map(|p| p.pid).collect();
                    if cycle_count % 60 == 0 {
                        syscall_classifier.evict_stale(&pids);
                    }
                    pids.iter()
                        .filter_map(|&pid| {
                            syscall_classifier
                                .sample(pid)
                                .filter(|p| {
                                    *p == apollo_optimizer::engine::syscall_classifier::SyscallProfile::JitCompiling
                                })
                                .map(|_| pid)
                        })
                        .collect()
                };

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
                        if t >= 100.0 {
                            0.30
                        }
                        // critical
                        else if t >= 90.0 {
                            0.15
                        }
                        // severe
                        else if t >= 80.0 {
                            0.05
                        }
                        // moderate
                        else {
                            0.0
                        }
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
                if thermal_action.phase
                    < apollo_optimizer::engine::thermal_bailout::CoolingPhase::Phase3Aggressive
                {
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

                // ── Effective pressure: aggregate all boost factors ──────────
                // Raw memory_pressure misses hardware stress (thermal, battery,
                // bandwidth saturation). effective_pressure::compute() is the
                // authoritative value. We write it back into snapshot so all
                // downstream consumers (decide_actions, page_reclaim, io_shaper,
                // skill_registry, signal_intel) see the fully-boosted value
                // without requiring individual call-site changes.
                let (pressure_ram, _pressure_components) = effective_pressure::compute(
                    snapshot.pressure.memory_pressure,
                    hw_boost,
                    batt_boost,
                    thermal_pressure_boost,
                    llm_boost,
                    charging_stress_boost,
                    battery_low_boost,
                    mem_bw_boost,
                    smc_thermal_boost,
                    battery_overheat_boost,
                );
                snapshot.pressure.memory_pressure = pressure_ram;

                // Cautious mode: if previous session crashed, raise effective pressure
                // by +0.10 for the first 50 cycles so freeze/throttle gates trigger
                // at a higher threshold — reducing risk of repeating the instability.
                // [Gray & Reuter 1992 §3 — conservative restart after abnormal termination]
                if cautious_cycles_remaining > 0 {
                    snapshot.pressure.memory_pressure =
                        (snapshot.pressure.memory_pressure - 0.10).max(0.0);
                    cautious_cycles_remaining -= 1;
                    if cautious_cycles_remaining == 0 {
                        audit_log(&serde_json::json!({
                            "event": "cautious_mode_ended",
                            "message": "returning to normal thresholds after post-crash caution period",
                        }));
                    }
                }

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
                    metrics.metrics.swap_used_bytes = snapshot.pressure.swap_used_bytes;
                    metrics.metrics.swap_delta_bps = snapshot.pressure.swap_delta_bytes_per_sec;
                    metrics.metrics.memory_pressure = snapshot.pressure.memory_pressure;
                    metrics.metrics.thermal_level = snapshot.pressure.thermal_level.clone();
                    // Expose the new sensor surface (thrashing flow + per-core
                    // saturation + PSI stall fraction) so the dashboard,
                    // socket status and runtime_metrics.json all see the same
                    // signals the RL reward path already reacts to.
                    let pd = pressure_collector.latest();
                    metrics.metrics.thrashing_score = pd.thrashing_score;
                    metrics.metrics.cpu_mean_busy = pd.cpu_saturation.mean_busy;
                    metrics.metrics.cpu_max_busy = pd.cpu_saturation.max_busy;
                    metrics.metrics.cpu_pegged_fraction = pd.cpu_saturation.pegged_fraction;
                    if let Ok(tracker) =
                        apollo_optimizer::engine::contention_tracker::global().lock()
                    {
                        // Threshold 0.85: Darwin's ri_runnable_time accumulates
                        // run-queue wait time on EVERY scheduling quantum, so the
                        // baseline ratio for any moderately active process is
                        // already ~0.7 under cpu_mean_busy ~0.4. 0.85 means
                        // "actually starved", not "compitiendo bajo carga normal".
                        metrics.metrics.stall_fraction = tracker.stall_fraction(0.85);
                    }
                    {
                        let rs_total = metrics.reactor_status.events_total;
                        let rs_mem = metrics.reactor_status.events_mem;
                        let rs_thermal = metrics.reactor_status.events_thermal;
                        let rs_spawn = metrics.reactor_status.events_spawn;
                        let rs_power = metrics.reactor_status.events_power;
                        let rs_last_at = metrics.reactor_status.last_event_at;
                        let rs_last_err = metrics.reactor_status.last_error.clone();
                        let rs_mode = metrics.reactor_status.mode.clone();
                        let rs_health = metrics.reactor_status.health.clone();
                        metrics.metrics.reactor_events_total = rs_total;
                        metrics.metrics.reactor_events_mem = rs_mem;
                        metrics.metrics.reactor_events_thermal = rs_thermal;
                        metrics.metrics.reactor_events_spawn = rs_spawn;
                        metrics.metrics.reactor_events_power = rs_power;
                        metrics.metrics.reactor_last_event_at = rs_last_at;
                        metrics.metrics.reactor_last_error = rs_last_err;
                        metrics.metrics.reactor_mode = rs_mode;
                        metrics.metrics.reactor_health = rs_health;
                    }
                    metrics.metrics.dev_session_active = dev_session_active;
                    metrics.metrics.interactive_heavy = interactive_heavy;
                    metrics.metrics.context_switches_5min = context_switches_5min;
                    metrics.metrics.context_switch_burst = context_switch_burst;

                    // Resource interrupt (sentinel) metrics.
                    metrics.metrics.resource_interrupts_total =
                        state.resource_interrupt.total_fires.load(Ordering::Relaxed);
                    metrics.metrics.resource_interrupt_last_phase =
                        state.resource_interrupt.phase.load(Ordering::Relaxed);
                    metrics.metrics.resource_interrupt_active =
                        state.resource_interrupt.active.load(Ordering::Relaxed);
                    metrics.metrics.resource_interrupt_latency_us = state
                        .resource_interrupt
                        .last_latency_us
                        .load(Ordering::Relaxed);
                    metrics.metrics.resource_interrupt_processes_frozen = state
                        .resource_interrupt
                        .total_frozen
                        .load(Ordering::Relaxed);
                    metrics.metrics.resource_interrupt_processes_migrated = state
                        .resource_interrupt
                        .total_migrated
                        .load(Ordering::Relaxed);
                    metrics.metrics.resource_interrupt_recovery_count = state
                        .resource_interrupt
                        .total_recoveries
                        .load(Ordering::Relaxed);

                    // Foreground detection metrics.
                    metrics.metrics.foreground_app = match &fg_state {
                        ForegroundState::App(app) => Some(ForegroundAppInfo {
                            pid: app.pid,
                            name: app.name.clone(),
                            bundle_id: app.bundle_id.clone(),
                        }),
                        _ => None,
                    };
                    metrics.metrics.foreground_idle = foreground_idle;

                    // Energy tracking metrics.
                    let energy_summary = lctx.energy_tracker.session_summary();
                    metrics.metrics.energy_savings_wh = Some(energy_summary.estimated_savings_wh);
                    metrics.metrics.energy_co2_avoided_g =
                        Some(energy_summary.estimated_co2_kg * 1000.0);
                    metrics.metrics.energy_package_wh = Some(energy_summary.total_package_wh);
                    metrics.metrics.energy_session_wh =
                        Some(energy_summary.total_cpu_wh + energy_summary.total_gpu_wh);
                    // Use cycle-level hardware snapshot for per-process power,
                    // falling back to smc_direct when IOKit returns None.
                    metrics.metrics.energy_cpu_watts = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.cpu_watts)
                        .map(|w| w as f64)
                        .or(last_smc.as_ref().and_then(|s| s.p_cluster_watts));
                    metrics.metrics.energy_gpu_watts = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.gpu_watts)
                        .map(|w| w as f64)
                        .or(last_smc.as_ref().and_then(|s| s.gpu_watts));
                    metrics.metrics.energy_ane_watts = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.ane_watts)
                        .map(|w| w as f64);
                    metrics.metrics.energy_ane_util_pct = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.ane_util_pct)
                        .map(|v| v as f64);
                    metrics.metrics.energy_package_watts = cycle_hw_snap
                        .as_ref()
                        .and_then(|h| h.power.package_watts)
                        .map(|w| w as f64)
                        .or(last_smc.as_ref().and_then(|s| s.system_power_watts));
                    metrics.metrics.energy_top_consumers = lctx
                        .energy_tracker
                        .top_consumers(5)
                        .into_iter()
                        .map(|e| EnergyConsumerInfo {
                            name: e.name,
                            current_watts: e.current_watts,
                            percentage: e.percentage_of_total,
                        })
                        .collect();

                    // Process tree metrics (informational).
                    metrics.metrics.process_tree_groups = process_tree.group_count();
                    metrics.metrics.process_tree_total = process_tree.len();

                    // IOReport hardware telemetry.
                    if let Some(ref ir) = last_ioreport {
                        metrics.metrics.ioreport_p_cluster_pct = ir.p_cluster_pct;
                        metrics.metrics.ioreport_e_cluster_pct = ir.e_cluster_pct;
                        metrics.metrics.ioreport_gpu_pct = ir.gpu_pct;
                        metrics.metrics.ioreport_ane_busy = ir.ane_busy;
                        metrics.metrics.ioreport_cpu_mw = ir.cpu_mw;
                        metrics.metrics.ioreport_total_watts = ir.total_watts();
                    }

                    // SMC direct metrics
                    if let Some(ref smc) = last_smc {
                        metrics.metrics.smc_system_power_watts = smc.system_power_watts;
                        metrics.metrics.smc_lid_closed = smc.lid_closed;
                        metrics.metrics.smc_charger_watts = smc.charger_watts;
                        metrics.metrics.smc_battery_tte_min = smc.battery_time_to_empty_min;
                        metrics.metrics.smc_cpu_temp_celsius = smc.cpu_temp_celsius;
                        metrics.metrics.smc_gpu_temp_celsius = smc.gpu_temp_celsius;
                        metrics.metrics.smc_battery_temp_celsius = smc.battery_temp_celsius;
                        metrics.metrics.smc_cpu_voltage = smc.cpu_voltage;
                        metrics.metrics.smc_p_cluster_watts = smc.p_cluster_watts;
                    }

                    // KPC IPC metric + signal intelligence modulation
                    if let Some(ref kpc) = kpc_snap {
                        metrics.metrics.kpc_ipc = kpc.ipc;
                        lctx.signal_intel.set_kpc_ipc(kpc.ipc);
                        lctx.signal_intel.set_kpc_trend(kpc.ipc_trend);
                        // memory_bound_score: fraction of CPU cycles stalled on memory.
                        // >0.7 = system >70% memory-bound → aggressive freeze safe.
                        metrics.metrics.kpc_memory_bound_score = kpc.memory_bound_score;
                    }

                    // Cache contention detection + cluster separation (Phase 3)
                    //
                    // On Apple Silicon P-cluster and E-cluster have separate L2 caches.
                    // Routing competing processes to different clusters eliminates L2
                    // thrashing. We use QoS tiers as cluster hints (not strict binding):
                    //   heavy_pid → Foreground (P-cores, low latency)
                    //   light_pid → Background (E-cores, energy efficient)
                    //
                    // Safety gates:
                    //   1. Only if confidence ≥ 0.25 (≥3 consecutive co-exec cycles)
                    //   2. Only if contention score > 0.45
                    //   3. Neither pid in heuristic_critical_pids
                    //   4. Pressure > 0.30 (only under real load)
                    //
                    // [Apple WWDC21 "Optimize for Apple Silicon" — P/E cluster semantics]
                    {
                        let system_ipc = kpc_snap.as_ref().map(|k| k.ipc).unwrap_or(0.0);
                        let proc_list: Vec<(u32, String, f32)> = proc_snaps
                            .iter()
                            .map(|p| (p.pid, p.name.clone(), p.cpu_percent))
                            .collect();
                        let pressure = metrics.metrics.memory_pressure;
                        let contention =
                            contention_detector.tick(system_ipc, &proc_list, pressure, 15.0);
                        metrics.metrics.contention_score = contention.score;
                        metrics.metrics.contention_heavy_count = contention.heavy_count;
                        metrics.metrics.contention_pairs_active = contention.pairs.len() as u32;

                        // Apply cluster separation for confirmed pairs.
                        if contention.score > 0.45
                            && pressure > 0.30
                            && !contention.pairs.is_empty()
                        {
                            let sep_hard = apollo_optimizer::engine::safety::protected_processes();
                            let sep_infra = infrastructure_processes();
                            let policy_prot = state
                                .policy
                                .lock_recover()
                                .learned_policy
                                .protected_patterns
                                .clone();
                            let mut qos = state.mach_qos.lock_recover();
                            for pair in &contention.pairs {
                                if pair.confidence() < 0.25 {
                                    continue;
                                }
                                // Skip if either process matches any protection list.
                                let protected = |name: &str| {
                                    sep_hard.iter().any(|p| name.contains(p))
                                        || sep_infra.iter().any(|p| name.contains(p))
                                        || policy_prot.iter().any(|p| {
                                            name.to_ascii_lowercase()
                                                .contains(&p.to_ascii_lowercase())
                                        })
                                };
                                if protected(&pair.heavy_name) || protected(&pair.light_name) {
                                    continue;
                                }
                                // heavy → P-cores (Foreground tier)
                                qos.set_tier(pair.heavy_pid, SchedulingTier::Foreground);
                                // light → E-cores (Background tier)
                                qos.set_tier(pair.light_pid, SchedulingTier::Background);
                            }
                        }

                        // GC dead PIDs every 100 cycles
                        if cycle_count % 100 == 0 {
                            let alive: std::collections::HashSet<u32> =
                                proc_snaps.iter().map(|p| p.pid).collect();
                            contention_detector.gc(&alive);
                        }
                    }

                    // Window/app lifecycle sensor — process diff for tab/app events.
                    // [Hellerstein 2004] Feed-forward: act on disturbance (tab close),
                    // don't wait for pressure to fall (feedback lag).
                    // Window sensor: full delta with session intelligence.
                    // [Pirolli & Card 1999] session phase | [Denning 1968] pressure floor
                    // Window sensor: full delta with session intelligence.
                    // [Pirolli & Card 1999] session phase | [Denning 1968] pressure floor
                    let (win_tab_delta, win_freed_heavy) = {
                        let fg_name = match fg_detector.detect() {
                            ForegroundState::App(ref a) => a.name.clone(),
                            _ => String::new(),
                        };
                        let proc_pairs: Vec<(u32, &str)> = proc_snaps
                            .iter()
                            .map(|p| (p.pid, p.name.as_str()))
                            .collect();
                        let win = window_sensor.tick(
                            &proc_pairs,
                            if fg_name.is_empty() {
                                None
                            } else {
                                Some(fg_name.as_str())
                            },
                        );
                        metrics.metrics.window_tab_delta = win.tab_delta;
                        metrics.metrics.window_renderer_count = win.renderer_count;
                        metrics.metrics.window_freed_heavy_app = win.freed_heavy_app;
                        metrics.metrics.window_tab_velocity_ema = win.tab_velocity_ema;
                        metrics.metrics.window_pressure_floor = win.pressure_floor_correction;
                        metrics.metrics.window_session_phase = format!("{:?}", win.session_phase);
                        metrics.metrics.window_workload_intent =
                            format!("{:?}", win.workload_intent);
                        // Lift to outer scope for reactor_weight section.
                        win_session_phase = win.session_phase;
                        win_workload_intent = win.workload_intent;
                        win_pressure_floor = win.pressure_floor_correction;
                        (win.tab_delta, win.freed_heavy_app)
                    };
                    // Feed-forward relief: tabs closed or heavy app quit → RAM freed soon.
                    if win_tab_delta < -1 || win_freed_heavy {
                        window_relief_cycles = window_relief_cycles.max(3);
                    } else if win_tab_delta < 0 {
                        window_relief_cycles = window_relief_cycles.max(2);
                    }

                    // ── Fluidity Intelligence ────────────────────────────────
                    // Update FluidityState from process snapshot + GPU load.
                    // WindowServer CPU → spike detection (window resize/move).
                    // New PIDs → app launch detection + protection window.
                    // [Jain 1991] composite score, [Welch & Bishop 2006] Kalman prediction
                    {
                        let fl_procs: Vec<(u32, &str, f32)> = proc_snaps
                            .iter()
                            .map(|p| (p.pid, p.name.as_str(), p.cpu_percent))
                            .collect();
                        // GPU load from IOKit hw_snap (normalized 0–1).
                        let fl_gpu_load = cycle_hw_snap
                            .as_ref()
                            .and_then(|hw| hw.power.gpu_watts)
                            .map(|w| (w / 15.0).clamp(0.0, 1.0) as f32)
                            .unwrap_or(0.0);
                        fluidity_state.update(&fl_procs, fl_gpu_load, cycle_dt_secs as f32);

                        // Snapshot signal for use later in the cycle
                        let fl_sig = apollo_optimizer::engine::fluidity::FluiditySignal::from(
                            &fluidity_state,
                        );

                        // Wire into RuntimeMetrics for status/dashboard reporting
                        metrics.metrics.fluidity_score = fl_sig.fluidity_score;
                        metrics.metrics.window_op_active = fl_sig.window_op_active;
                        metrics.metrics.app_launching = fl_sig.app_launching;
                        metrics.metrics.app_launch_name = fl_sig.launch_name.clone();
                        metrics.metrics.fluidity_degraded = fl_sig.fluidity_degraded;
                        // Kalman prediction for pre-emptive response
                        metrics.metrics.fluidity_predicted_3s = fl_sig.fluidity_predicted_3s;
                        metrics.metrics.fluidity_velocity = fl_sig.fluidity_velocity;
                        // Also update windowserver_cpu_pct (existing field)
                        metrics.metrics.windowserver_cpu_pct = fluidity_state.windowserver_cpu_ema;
                    }

                    // Build progress tick — updates build_tracker.phase and
                    // build_progress each cycle from rustc/cargo process counts.
                    {
                        let proc_pairs: Vec<(u32, &str)> = proc_snaps
                            .iter()
                            .map(|p| (p.pid, p.name.as_str()))
                            .collect();
                        build_tracker.tick(&proc_pairs);
                        metrics.metrics.build_phase = format!("{:?}", build_tracker.phase);
                        metrics.metrics.build_progress = build_tracker.build_progress;
                    }

                    // Rosetta AOT state
                    metrics.metrics.rosetta_aot_active = rosetta_monitor.is_compiling();

                    // IOReport AMC bandwidth
                    if let Some(ref ir) = last_ioreport {
                        metrics.metrics.ioreport_amc_bandwidth_pct = ir.amc_bandwidth_pct;
                    }

                    // IOPMrootDomain thermal
                    if let Some(ref iopm) = iopm_snap {
                        metrics.metrics.iopm_thermal_warning =
                            format!("{:?}", iopm.thermal_warning);
                        metrics.metrics.iopm_power_source = format!("{:?}", iopm.power_source);
                    }

                    // Per-process energy top consumer
                    if let Some(top) = energy_pid_results.first() {
                        metrics.metrics.energy_top_pid_name = top.name.clone();
                        metrics.metrics.energy_top_pid_mw = top.power_mw;
                    }

                    // AMX availability (static — probed once at startup).
                    metrics.metrics.amx_available = amx_available;
                    metrics.metrics.amx_cs_overhead_ns = amx_cs_overhead_ns;

                    // Wakeup vampire report: top 3 processes by wakeup rate.
                    // [Apple Activity Monitor] wakeup rate = primary "Energy Impact" signal.
                    let mut wakeup_sorted: Vec<_> = energy_pid_results
                        .iter()
                        .filter(|e| e.wakeup_rate >= 50.0)
                        .collect();
                    wakeup_sorted.sort_by(|a, b| {
                        b.wakeup_rate
                            .partial_cmp(&a.wakeup_rate)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    metrics.metrics.wakeup_vampires = wakeup_sorted
                        .iter()
                        .take(3)
                        .map(|e| format!("{}({:.0}/s)", e.name, e.wakeup_rate))
                        .collect();
                    metrics.metrics.kpc_memory_bound_score = kpc_snap
                        .as_ref()
                        .map(|k| k.memory_bound_score)
                        .unwrap_or(metrics.metrics.kpc_memory_bound_score);

                    // Behavioral anomaly telemetry: top 3 anomalous processes.
                    // "name(score×)" e.g. "backupd(8.2×)" = 8.2 MADs above baseline.
                    let mut anomaly_sorted: Vec<_> = energy_pid_results
                        .iter()
                        .filter(|e| e.anomaly_score >= anomaly_thresh)
                        .collect();
                    anomaly_sorted.sort_by(|a, b| {
                        b.anomaly_score
                            .partial_cmp(&a.anomaly_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    metrics.metrics.anomaly_process_count = anomaly_sorted.len();
                    metrics.metrics.anomaly_processes = anomaly_sorted
                        .iter()
                        .take(3)
                        .map(|e| format!("{}({:.1}×)", e.name, e.anomaly_score))
                        .collect();
                    metrics.metrics.process_baseline_warm =
                        energy_pid_tracker.baseline.warm_count();

                    // Daemon self-IPC (thread_selfcounts syscall 186)
                    let _cycle_ipc = cycle_ipc_tracker.tick();
                    metrics.metrics.daemon_cycle_ipc = cycle_ipc_tracker.ema_ipc();
                }

                let (decide_interactive, decide_noise, decide_weights, outcome_baseline) = {
                    let pg = state.policy.lock_recover();
                    (
                        pg.learned_policy.interactive_patterns.clone(),
                        pg.learned_policy.noise_patterns.clone(),
                        pg.learned_policy.pattern_weights.clone(),
                        lctx.outcome_tracker.calibrated_threshold(),
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

                // Workload-onset: fired once when transitioning INTO Build mode.
                // Lets the governor proactively switch to AggressiveRoot before
                // pressure builds, rather than waiting for the reactive threshold.
                let workload_onset = workload_mode == WorkloadMode::Build
                    && prev_workload_mode != WorkloadMode::Build;

                let governor_decision = {
                    let mut pg = state.policy.lock_recover();
                    pg.governor.evaluate(GovernorInput {
                        cpu_pressure: pressure_cpu,
                        ram_pressure: pressure_ram,
                        interactive_wait_ratio: pressure_wait,
                        reactor_event_weight: reactor_weight,
                        thermal_constrained: matches!(
                            snapshot.pressure.thermal_level.as_str(),
                            "serious" | "critical"
                        ) || gpu_thermal_throttled,
                        dev_session_active,
                        interactive_heavy,
                        context_switch_burst,
                        workload_mode: Some(workload_mode),
                        workload_onset,
                        swap_used_bytes: snapshot.pressure.swap_used_bytes,
                    })
                };
                if governor_decision.transition_reason.contains("floor") {
                    state.metrics.lock_recover().metrics.profile_floor_hits += 1;
                }
                let current_profile = governor_decision.effective_profile;
                {
                    let pg = state.policy.lock_recover();
                    write_governor_state(&governor_state_path, &pg.governor);
                }

                // Thresholds adaptativos: workload-aware via Phase 3 classifier.
                let mut overflow_thresholds = lctx.overflow_guard.thresholds(workload_mode);

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
                    lctx.signal_intel.set_energy_bias(
                        power_mgr.battery_status.percentage,
                        power_mgr.battery_status.is_charging,
                        thermal_emergency,
                    );
                    // Power-aware bias: when real watts are high, engage optimizer earlier.
                    // M1 Air TDP ~15W; >8W = active load, >12W = stressed.
                    if let Some(pkg_w) = cycle_hw_snap.as_ref().and_then(|h| h.power.package_watts)
                    {
                        lctx.signal_intel.adjust_bias_for_power(pkg_w);
                    }
                    // Workload-aware bias: heavy workloads (Coding/VideoEdit) spike pressure
                    // fast — engage optimizer 2pp earlier during those hours.
                    {
                        let wl = state
                            .policy
                            .lock_recover()
                            .adaptive_governor
                            .user_profile
                            .likely_workload_at_hour(hour_of_day);
                        lctx.signal_intel.adjust_bias_for_workload(wl);
                    }
                    let _si_result = lctx.signal_intel.tick(
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
                    use apollo_optimizer::engine::telemetry_logger::TelemetryVector;
                    let dom_share = {
                        let max_mem = snapshot
                            .top_processes
                            .iter()
                            .map(|p| p.memory_usage)
                            .max()
                            .unwrap_or(0) as f64;
                        let total = snapshot.memory.total_ram as f64;
                        if total > 0.0 {
                            (max_mem / total) as f32
                        } else {
                            0.0
                        }
                    };
                    let thermal_score = match snapshot.pressure.thermal_level.as_str() {
                        "nominal" => 0.0f32,
                        "light" => 0.33,
                        "serious" => 0.66,
                        "critical" => 1.0,
                        _ => 0.0,
                    };
                    let cpu_total = snapshot
                        .top_processes
                        .iter()
                        .map(|p| p.cpu_usage)
                        .sum::<f32>()
                        / 100.0;
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
                    d.transformer_anomaly =
                        darwin_anomaly.score(tv.as_f32_slice(), d.pressure_smooth as f32);
                    // Record to TelemetryLogger ring buffer.  record() self-triggers
                    // disk dumps (event-triggered at OOM/urgency/latency thresholds,
                    // periodic every ~10 min). [Welch 1967, Tuli et al. 2022]
                    telemetry_logger.record(tv);
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

                    // Fluidity signals → SignalDigest.
                    // [Jain 1991] Composite urgency includes fluidity degradation:
                    // sustained low fluidity (< 0.65) adds up to 0.30 to urgency,
                    // allowing the predictive agent to account for rendering pressure.
                    d.fluidity_score = fluidity_state.fluidity_ema;
                    d.window_op_active = fluidity_state.window_op_active();
                    d.app_launching = fluidity_state.launch_active;
                    if fluidity_state.fluidity_degraded {
                        let fluidity_urgency =
                            ((0.65 - fluidity_state.fluidity_ema as f64) * 0.4).max(0.0);
                        d.urgency = (d.urgency + fluidity_urgency).min(1.0);
                    }

                    d
                };

                // Signal intelligence → reactor_weight boosting.
                // CUSUM regime shift: pressure drifting up significantly.
                if signal_digest.regime_shift_up {
                    reactor_weight = (reactor_weight + 0.3).min(1.0);
                }
                // High composite urgency: multiple signals converging on danger.
                if signal_digest.urgency > 0.7 {
                    reactor_weight = (reactor_weight + 0.2).min(1.0);
                }
                // Entropy anomaly: chaotic process distribution change.
                if signal_digest.entropy_anomaly > 2.0 {
                    reactor_weight = (reactor_weight + 0.15).min(1.0);
                }
                // Darwin-Boltzmann anomaly: learned pattern deviation.
                // Score > 0.5 means the system state deviates significantly from
                // the Hopfield memory + SAE ensemble's learned "normal" manifold.
                if signal_digest.transformer_anomaly > 0.5 {
                    reactor_weight = (reactor_weight + 0.2).min(1.0);
                }
                // Feed-forward pressure relief [Hellerstein 2004]: tabs closed or heavy
                // app terminated → RAM will be freed. Back off reactor_weight for N cycles
                // instead of waiting for Kalman filter to catch the pressure drop.
                if window_relief_cycles > 0 {
                    reactor_weight = (reactor_weight - 0.25).max(0.0);
                    window_relief_cycles -= 1;
                }

                // Session phase feed-forward [Pirolli & Card 1999].
                // Ramping = user is expanding session → expect pressure rise → pre-position.
                // WindingDown = already handled by window_relief_cycles above.
                if win_session_phase == SessionPhase::Ramping {
                    reactor_weight = (reactor_weight + 0.15).min(1.0);
                }

                // Pressure floor correction [Denning 1968].
                // If current pressure is largely "explained" by the browser's working set,
                // dial back reactor aggressiveness. 13 tabs → floor=0.156: pressure of
                // 0.65 is not an emergency — it's the expected baseline for this session.
                let raw_pressure = snapshot.pressure.memory_pressure;
                if win_pressure_floor > 0.08 && raw_pressure < win_pressure_floor + 0.15 {
                    reactor_weight = (reactor_weight - win_pressure_floor * 0.5).max(0.0);
                }

                // Workload intent adjustments [Yang et al. 2013 PowerAPI].
                // Apollo applies workload-specific resource policy based on what the
                // user is actually doing — inferred from process signatures.
                match win_workload_intent {
                    WorkloadIntent::AISession => {
                        // Ollama/Python inference: high memory IS expected and intentional.
                        // Don't be aggressive while AI inference is running.
                        // Conservative: only back off if pressure is not critical.
                        if raw_pressure < 0.85 {
                            reactor_weight = (reactor_weight - 0.20).max(0.0);
                        }
                    }
                    WorkloadIntent::ResearchSession => {
                        // Many tabs open: renderer memory is load-bearing for the user's
                        // research context. Back off moderately — don't freeze their tabs.
                        if raw_pressure < 0.80 {
                            reactor_weight = (reactor_weight - 0.10).max(0.0);
                        }
                    }
                    WorkloadIntent::BuildSession => {
                        // Build session: cargo/rustc need RAM and CPU priority.
                        // Boost slightly so Apollo acts faster to clear non-build memory.
                        reactor_weight = (reactor_weight + 0.10).min(1.0);
                    }
                    WorkloadIntent::MediaSession => {
                        // Media playing: avoid heavy I/O actions (sysctl writes, spotlight)
                        // that could cause audio glitches. Slight back-off.
                        if raw_pressure < 0.75 {
                            reactor_weight = (reactor_weight - 0.08).max(0.0);
                        }
                    }
                    WorkloadIntent::General => {}
                }

                // Temporal pre-positioning [Denning 1968 Working Set Model].
                // Pre-carve headroom before predicted heavy app arrives.
                // Skip when build is already active: BuildTracker handles the boost
                // and adding temporal headroom on top would double-count the signal.
                let temporal_headroom = temporal_predictor
                    .pressure_headroom_for_incoming(temporal_hour, temporal_weekday);
                if temporal_headroom > 0.02 && !build_tracker.build_active {
                    reactor_weight = (reactor_weight + temporal_headroom).min(1.0);
                }

                // Build progress [McKenney 2004]: rustc-count dynamics proxy.
                // Starting phase: cargo just spawned — pre-clear non-build memory so
                // compilation gets all available RAM.
                // Finishing phase: rustc count declining — build about to complete,
                // relax to avoid disruptive actions during linker phase.
                match build_tracker.phase {
                    BuildPhase::Starting => {
                        // Boost aggressiveness: help cargo get RAM now.
                        reactor_weight = (reactor_weight + 0.15).min(1.0);
                    }
                    BuildPhase::Finishing => {
                        // Back off: linker/metadata writes are latency-sensitive.
                        let raw_pressure = snapshot.pressure.memory_pressure;
                        if raw_pressure < 0.80 {
                            reactor_weight = (reactor_weight - 0.12).max(0.0);
                        }
                    }
                    _ => {}
                }

                // Predictive agent: build context from existing signals and select intervention.
                // Feed Kalman-smoothed pressure instead of raw — cleaner signal for LinUCB.
                let agent_intervention = {
                    let prev_workload = state
                        .policy
                        .lock_recover()
                        .adaptive_governor
                        .last_ml_classification()
                        .workload;
                    let (hw_tp, hw_jt, hw_cl) = match &hw_features {
                        Some(f) => (f.throughput_mips, f.jitter_us, f.cache_latency_us),
                        None => (800.0, 50.0, 5000.0),
                    };
                    // Cable B: OutcomeTracker → PredictiveAgent context.
                    // low_value_ratio tells LinUCB how much of its effort is wasted.
                    let lv_ratio = {
                        let total = lctx.outcome_tracker.weights.len() as f64;
                        if total > 0.0 {
                            let threshold = lctx.outcome_tracker.calibrated_threshold();
                            let low = lctx
                                .outcome_tracker
                                .weights
                                .values()
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
                        reactor_weight,
                        lctx.overflow_guard.history.threshold_offset,
                        lctx.outcome_tracker.overall_effectiveness(),
                        lv_ratio,
                    );
                    let (linucb_choice, linucb_confidence) = lctx
                        .predictive_agent
                        .select_action_with_confidence(&agent_ctx);

                    // ── Specialist accuracy feedback (Super Learner) ─────────────────
                    // Compare prev cycle's ACTUAL specialist signals against observed outcome.
                    // Using real firing conditions (not pressure proxies) ensures the tracker
                    // measures what the specialist actually predicted, not a heuristic stand-in.
                    // A spike is a pressure rise of ≥0.08 over the previous cycle.
                    {
                        let pressure_spiked =
                            signal_digest.pressure_smooth >= prev_pressure_smooth + 0.08;
                        // Hazard: did prev cycle's hazard specialist fire (p_oom_30s > 0.30)?
                        let hazard_correct = (prev_hazard_fired && pressure_spiked)
                            || (!prev_hazard_fired && !pressure_spiked);
                        lctx.specialist_accuracy
                            .update(specialist::HAZARD, hazard_correct);

                        // Monopoly: did prev cycle's monopoly specialist fire (monopoly_risk > 0.5)?
                        let monopoly_correct = (prev_monopoly_fired && pressure_spiked)
                            || (!prev_monopoly_fired && !pressure_spiked);
                        lctx.specialist_accuracy
                            .update(specialist::MONOPOLY, monopoly_correct);

                        // Kalman: did prev cycle's Kalman predict spike (pressure_predicted_5s > 0.85)?
                        let kalman_correct = (prev_kalman_fired && pressure_spiked)
                            || (!prev_kalman_fired && !pressure_spiked);
                        lctx.specialist_accuracy
                            .update(specialist::KALMAN, kalman_correct);

                        // LinUCB: voted for non-Observe intervention. Correct if pressure spiked.
                        let linucb_correct = (prev_linucb_intervened && pressure_spiked)
                            || (!prev_linucb_intervened && !pressure_spiked);
                        lctx.specialist_accuracy
                            .update(specialist::LINUCB, linucb_correct);
                    }
                    // Save current cycle's actual specialist firing signals for next cycle's feedback.
                    prev_pressure_smooth = signal_digest.pressure_smooth;
                    prev_hazard_fired = signal_digest.p_oom_30s > 0.30;
                    prev_monopoly_fired = signal_digest.monopoly_risk > 0.5;
                    prev_kalman_fired = signal_digest.pressure_predicted_5s > 0.85;
                    prev_linucb_intervened = linucb_choice != Intervention::Observe;

                    // ── Specialist voting: weighted ensemble replaces override chain ──
                    // Confidences are modulated by learned accuracy weights (Super Learner).
                    // SpecialistAccuracyTracker EMA-tracks per-specialist correctness;
                    // a specialist consistently right gets weight→1.0, wrong gets→0.0.
                    let mut votes = vec![
                        // LinUCB: primary agent — UCB confidence × learned accuracy weight.
                        // linucb_confidence is the normalized margin of the winning arm [0.5, 1.0]:
                        // dominant winner → near 1.0, all arms tied → 0.5.
                        SpecialistVote {
                            name: "linucb",
                            intervention: linucb_choice,
                            confidence: linucb_confidence
                                * lctx.specialist_accuracy.weight(specialist::LINUCB),
                        },
                    ];

                    // Hazard specialist: high P(OOM) → use MPC recommendation.
                    if signal_digest.p_oom_30s > 0.30 {
                        votes.push(SpecialistVote {
                            name: "hazard",
                            intervention: Intervention::from_index(
                                signal_digest.mpc_recommendation,
                            ),
                            confidence: signal_digest.p_oom_30s.min(1.0)
                                * lctx.specialist_accuracy.weight(specialist::HAZARD),
                        });
                    }

                    // Monopoly specialist: one process hogging RAM → throttle noise.
                    if signal_digest.monopoly_risk > 0.5 {
                        votes.push(SpecialistVote {
                            name: "monopoly",
                            intervention: Intervention::PreThrottleNoise,
                            confidence: signal_digest.monopoly_risk.min(1.0)
                                * lctx.specialist_accuracy.weight(specialist::MONOPOLY),
                        });
                    }

                    // Kalman specialist: predicted pressure spike → tighten.
                    if signal_digest.pressure_predicted_5s > 0.85 {
                        votes.push(SpecialistVote {
                            name: "kalman",
                            intervention: Intervention::TightenThresholds,
                            confidence: (signal_digest.pressure_predicted_5s - 0.85).min(0.15)
                                / 0.15
                                * lctx.specialist_accuracy.weight(specialist::KALMAN),
                        });
                    }

                    // Proactive-30s specialist: Kalman projects overflow in ~30s but we're
                    // still below the action threshold — act NOW before RAM fills up.
                    // This is the key advantage over purely reactive systems:
                    // the OS can only react; Apollo can predict and pre-empt.
                    let p30_trigger = overflow_thresholds.bg_pressure as f64 - 0.05;
                    let p30_clear = overflow_thresholds.bg_pressure as f64 - 0.08;
                    if signal_digest.pressure_predicted_30s > p30_trigger
                        && signal_digest.pressure_smooth < p30_clear
                    {
                        let strength = ((signal_digest.pressure_predicted_30s - p30_trigger)
                            / 0.10)
                            .clamp(0.0, 1.0);
                        votes.push(SpecialistVote {
                            name: "proactive-30s",
                            intervention: Intervention::TightenThresholds,
                            confidence: strength
                                * lctx.specialist_accuracy.weight(specialist::KALMAN),
                        });
                    }

                    let vote_result = tally_votes(&votes);
                    let intervention = vote_result.intervention;

                    // Loop 3: store votes for disagreement outcome feedback next cycle.
                    if vote_result.had_disagreement {
                        last_specialist_votes = Some((votes.clone(), intervention));
                    } else {
                        last_specialist_votes = None;
                    }

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
                    overflow_thresholds =
                        lctx.predictive_agent.adjust_thresholds(overflow_thresholds);

                    // SuggestAggressive: set a 5-minute manual override to aggressive profile.
                    if intervention == Intervention::SuggestAggressive {
                        let mut pg = state.policy.lock_recover();
                        if pg.governor.manual_override.is_none() {
                            pg.governor.set_manual_override(
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
                // JIT-compiling PIDs (from syscall_classifier) are merged in so that
                // decide_actions treats them as interactive and skips throttling.
                let behavior_interactive_pids: HashSet<u32> = {
                    // Apple background daemons that are I/O-bound (low cpu_wall_ratio)
                    // but must NEVER be classified as interactive or boosted.
                    // The bad_interactive scrubber only covers learned_interactive;
                    // this denylist covers the behavioral (cpu_wall_ratio) path.
                    // [Android LMK] Protection earned by user interaction, not I/O.
                    // Production data: searchpartyd got 17 incorrect boosts via this path.
                    // Journal audit 2026-04-09: these daemons have low cpu_wall_ratio
                    // (I/O-bound) which the behavioral detector mis-classifies as
                    // "interactive". Each name was observed receiving 30-100+ incorrect
                    // boosts per session. Without the denylist they'd get Foreground QoS
                    // every cycle, wasting scheduler priority on system background work.
                    const BEHAVIOR_DENYLIST: &[&str] = &[
                        "searchpartyd",         // Find My / Handoff BLE scanning
                        "corespeechd",          // Siri speech (background)
                        "suggestd",             // Spotlight/Siri suggestions ML
                        "duetexpertd",          // Siri predictions / Proactive
                        "photoanalysisd",       // Photos ML tagging
                        "mediaanalysisd",       // Media content analysis
                        "intelligencecontextd", // Apple Intelligence
                        "mlhostd",              // Core ML inference host
                        "modelmanagerd",        // On-device model cache
                        "rtcreportingd",        // RealTimeComm diagnostics
                        // Added 2026-04-09 from journal boost audit:
                        "cfprefsd",             // Preference caching daemon — I/O-bound, 33 false boosts
                        "xpcproxy",             // XPC service launcher — ephemeral, 40 false boosts
                        "log",                  // Unified log CLI — spawned by log_ingester, 30 false boosts
                        "apollo-optimizerd",    // Self — execute_actions blocks but 45 wasted journal entries
                        "apollo-optimizerctl",  // Our own CLI client
                        "diagnostics_agent",    // System diagnostics — throttled 749x but also boosted
                        "socketfilterfw",       // Application firewall — I/O-bound, not interactive
                        "stable",              // /usr/libexec/stable — system process, 67 false boosts
                    ];
                    let model = state.usage.lock_recover();
                    let interactive_names: HashSet<&str> = model
                        .usage_model
                        .entries()
                        .iter()
                        .filter(|(name, entry)| {
                            apollo_optimizer::engine::usage_model::is_behavior_interactive(entry)
                                && !BEHAVIOR_DENYLIST.iter().any(|d| name.contains(d))
                        })
                        .map(|(name, _)| name.as_str())
                        .collect();
                    // Map interactive names back to running PIDs, then union JIT PIDs.
                    let mut pids: HashSet<u32> = snapshot
                        .top_processes
                        .iter()
                        .filter(|p| interactive_names.contains(p.name.as_str()))
                        .map(|p| p.pid)
                        .collect();
                    pids.extend(jit_protected_pids.iter().copied());
                    pids
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
                        let next_workload = state
                            .policy
                            .lock_recover()
                            .adaptive_governor
                            .user_profile
                            .likely_workload_at_hour(next_hour);
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
                        windowserver_cpu: llm_daemon::windowserver_cpu(&snapshot) as f64,
                        foreground_cpu: fg_cpu,
                        foreground_csw_per_sec: fg_csw,
                        has_foreground: foreground_pid.is_some(),
                    });
                    if latency.needs_boost {
                        // Elevate reactor weight → faster tick + more aggressive decisions.
                        reactor_weight = (reactor_weight + 0.25).min(1.0);
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
                        state.metrics.lock_recover().metrics.paging_hints_applied += 1;
                    }
                }

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
                        let live: HashSet<u32> = collector
                            .system()
                            .processes()
                            .keys()
                            .map(|p| p.as_u32())
                            .collect();
                        habituation_map.retain(|pid, _| live.contains(pid));
                    }
                    hab_set
                };
                // Emit habituation count so AIS runtime benchmark can read it.
                {
                    let mut m = state.metrics.lock_recover();
                    m.metrics.habituation_skips += habituated_pids.len() as u64;
                }

                // [Pearl 2009 + Kahneman 1973] Blend causal graph with experience priors.
                // Cold processes (< 5 observations) get warm-start priors from
                // similar past episodes, preventing the causal skip filter from
                // requiring 5 wasteful throttles before it can make a judgment.
                let mut causal_confidence = lctx.causal_graph.confidence_map_with_experience(
                    &lctx.outcome_tracker.experience,
                    snapshot.pressure.memory_pressure,
                );
                // [Pearl 2009 Ch.2] Co-occurrence cluster boost: if B always appears
                // with solid A during pressure events, B gets a confidence boost.
                let co_pairs: Vec<(String, String, u32)> = lctx
                    .outcome_tracker
                    .top_causal_pairs(20)
                    .into_iter()
                    .map(|(a, b, c)| (a.to_string(), b.to_string(), c))
                    .collect();
                lctx.causal_graph
                    .apply_cluster_boost(&mut causal_confidence, &co_pairs);
                // [Pei Wang 2013] NARS × Causal: discount confidence for drifted beliefs.
                // Unstable NARS beliefs (low confidence) → causal relationship may be stale.
                CausalGraph::apply_nars_discount(
                    &mut causal_confidence,
                    &lctx.outcome_tracker.drift_detector,
                );

                // ── User context: "telepathy" — what is the user doing right now? ──
                // idle_secs from IOHIDSystem HIDIdleTime — fast ioreg call, safe every cycle.
                // Sleep assertions + call detection from pmset — amortised: every 5 cycles.
                // cpu_temp comes from hw_snapshot (already collected by SMC reader above).
                // [Riva & Mantovani 2014] idle time + media state = highest-signal context cues.
                let user_context = {
                    use apollo_optimizer::engine::user_context::UserContext;
                    // Poll pmset every 3 cycles (~9s) — balances subprocess cost vs
                    // responsiveness (call starts → detected within 9s, not 15s).
                    let collect_assertions = cycle_count % 3 == 0;
                    let mut ctx = UserContext::collect(collect_assertions);
                    // Merge: on non-assertion cycles, carry forward last known state.
                    // Prevents freeze_gate from flickering between "user-protected" and
                    // "delta/committed" every cycle. [Cook et al. 2019]
                    if collect_assertions {
                        last_user_assertions = (
                            ctx.has_sleep_assertion,
                            ctx.call_in_progress,
                            ctx.audio_active,
                        );
                    } else {
                        ctx.has_sleep_assertion = last_user_assertions.0;
                        ctx.call_in_progress = last_user_assertions.1;
                        ctx.audio_active = last_user_assertions.2;
                    }
                    // Merge cpu_temp from hw_snapshot (already in RuntimeMetrics).
                    // If P-cluster temp > 75°C, treat as if more active (raise pressure gate)
                    // so Apollo conserves thermal headroom for the user's workload.
                    if let Some(ref hw) = cycle_hw_snap {
                        if let Some(p_temp) = hw.temps.p_cluster_celsius {
                            if p_temp > 75.0 && !ctx.is_idle_long() {
                                // Simulate "recently active" to raise freeze gates and
                                // protect thermal headroom — overrides any idle signal.
                                ctx.idle_secs = ctx.idle_secs.min(10.0);
                            }
                        }
                    }
                    ctx
                };

                // Bypass habituation under critical conditions.
                // Habituation assumes "stable RSS = no problem", but under heavy swap
                // processes have stable RSS (swapped out) and stable CPU (not running) —
                // they LOOK calm but are the source of thrashing.  Force a fresh look
                // when p_oom ≥ 0.95 or swap ≥ 8 GB.
                let swap_critical = snapshot.pressure.swap_used_bytes >= 8 * 1_073_741_824;
                let oom_critical = signal_digest.p_oom_30s >= 0.95;
                let empty_hab: HashSet<u32> = HashSet::new();
                let effective_habituated: &HashSet<u32> = if swap_critical || oom_critical {
                    &empty_hab // bypass: re-evaluate all processes
                } else {
                    &habituated_pids
                };

                let decision = {
                    let mut qos = state.mach_qos.lock_recover();
                    let dram_bandwidth_pct = last_ioreport
                        .as_ref()
                        .map(|ir| ir.amc_bandwidth_pct)
                        .unwrap_or(0.0);
                    let policy = PolicyContext {
                        decide_interactive: &decide_interactive,
                        decide_noise: &decide_noise,
                        decide_weights: &decide_weights,
                        outcome_baseline,
                        behavior_interactive_pids: &behavior_interactive_pids,
                        ipc_hints: &ipc_hints,
                        hop_groups: &lctx.outcome_tracker.hop_groups,
                        habituated_pids: effective_habituated,
                        causal_confidence: &causal_confidence,
                        user_ctx: &user_context,
                        wakeup_hints: &wakeup_hints,
                        footprint_hints: &footprint_hints,
                        dram_bandwidth_pct,
                        io_burst_hints: &io_burst_hints,
                        anomaly_hints: &anomaly_hints,
                    };
                    decision_stage
                        .run(
                            &snapshot,
                            collector.system(),
                            current_profile,
                            latency_target,
                            reactor_weight,
                            overflow_thresholds,
                            Some(&mut qos),
                            &policy,
                        )
                        .decision
                };
                state.process.lock_recover().last_blockers = decision.blockers.clone();
                state.metrics.lock_recover().thermal_state =
                    process_enrichment::context_to_thermal(decision.context);

                // Propagar skips de OutcomeTracker a top_skipped_processes para observabilidad.
                // También propagar los nuevos campos de observabilidad de DecisionOutput.
                {
                    let mut metrics = state.metrics.lock_recover();
                    for name in &decision.low_value_skipped {
                        if metrics.metrics.top_skipped_processes.len() < 12
                            && !metrics.metrics.top_skipped_processes.contains(name)
                        {
                            metrics.metrics.top_skipped_processes.push(name.clone());
                        }
                    }
                    // Enrich telemetría: display boost count + freeze gate + ml source.
                    if decision.display_boosts_emitted > 0 {
                        metrics.metrics.display_boost_count = metrics
                            .metrics
                            .display_boost_count
                            .saturating_add(decision.display_boosts_emitted as u64);
                    }
                    if decision.freeze_gate != "none" {
                        metrics.metrics.freeze_gate_last = decision.freeze_gate.clone();
                    }
                    if decision.ml_throttle_source != "none" {
                        metrics.metrics.ml_throttle_source = decision.ml_throttle_source.clone();
                    }
                    // User context telemetry: wire idle/call/audio/assertion signals.
                    metrics.metrics.user_idle_secs = user_context.idle_secs;
                    metrics.metrics.user_has_sleep_assertion = user_context.has_sleep_assertion;
                    metrics.metrics.user_call_in_progress = user_context.call_in_progress;
                    metrics.metrics.user_audio_active = user_context.audio_active;
                }

                // Apply any locally learned policy patterns (and keep them even after LLM is disabled).
                let mut actions = decision.actions;
                {
                    let policy = state.policy.lock_recover().learned_policy.clone();
                    actions = llm_daemon::apply_learned_policy_actions(&snapshot, &policy, actions);
                }

                // Apply learned skills: throttle processes with solid causal links to
                // pressure reduction. Skills are earned from causal graph solid edges
                // (confidence × avg_delta). matching_skills() already gates on
                // pressure ≥ skill.min_pressure AND is_reliable() (≥5 obs, ≥60% success).
                {
                    let skill_matches = lctx.skill_registry.matching_skills(
                        snapshot.pressure.memory_pressure as f32,
                        workload_mode.as_str(),
                    );
                    if !skill_matches.is_empty() {
                        let already_actioned: std::collections::HashSet<String> = actions
                            .iter()
                            .filter_map(|a| match a {
                                RootAction::ThrottleProcess { name, .. }
                                | RootAction::FreezeProcess { name, .. } => Some(name.clone()),
                                _ => None,
                            })
                            .collect();
                        let skill_targets: std::collections::HashSet<String> = skill_matches
                            .iter()
                            .flat_map(|s| s.throttle_targets.iter().cloned())
                            .collect();
                        for (pid, process) in collector.system().processes() {
                            let name = process.name().to_string();
                            if skill_targets.contains(&name) && !already_actioned.contains(&name) {
                                let skill_name = skill_matches
                                    .iter()
                                    .find(|s| s.throttle_targets.contains(&name))
                                    .map(|s| s.name.as_str())
                                    .unwrap_or("skill");
                                actions.push(RootAction::throttle(
                                    pid.as_u32(),
                                    name,
                                    false,
                                    format!("skill:{}", skill_name),
                                ));
                            }
                        }
                    }
                }

                // Trial induced skills: group:/batch: skills start at apply_count=0
                // and can never reach is_reliable() without real observations.
                // Each cycle at elevated pressure we try one unproven skill and record
                // the result on the NEXT cycle by comparing pressure before vs after.

                {
                    // Record result from previous cycle's trial if pending.
                    if let Some((ref pending_name, pressure_before)) = pending_trial_skill {
                        let effective = snapshot.pressure.memory_pressure < pressure_before - 0.01;
                        lctx.skill_registry.record_result_with_pressure(
                            pending_name,
                            effective,
                            pressure_before as f32,
                        );
                        pending_trial_skill = None;
                    }

                    let trial = lctx.skill_registry.next_trial_skill(
                        snapshot.pressure.memory_pressure as f32,
                        workload_mode.as_str(),
                    );
                    if let Some(skill) = trial {
                        let skill_name = skill.name.clone();
                        let pressure_before = snapshot.pressure.memory_pressure;
                        let hard_protected =
                            apollo_optimizer::engine::safety::protected_processes();
                        let infra_protected = infrastructure_processes();
                        let policy_prot = state
                            .policy
                            .lock_recover()
                            .learned_policy
                            .protected_patterns
                            .clone();
                        let already_actioned: std::collections::HashSet<String> = actions
                            .iter()
                            .filter_map(|a| match a {
                                RootAction::ThrottleProcess { name, .. } => Some(name.clone()),
                                _ => None,
                            })
                            .collect();
                        let mut trialed = false;
                        // Tracks whether at least one target exists in the process list
                        // but was blocked solely because it is the current foreground app.
                        // "Foreground-blocked" ≠ "ineffective" — we must not penalise the
                        // skill for respecting the foreground gate.
                        let mut targets_found_but_skipped = false;
                        for target in &skill.throttle_targets.clone() {
                            // Skip targets that are hard-protected, infra-protected, or
                            // policy-protected daemons.
                            // ConditionalForeground (user apps) are NOT skipped here —
                            // the foreground check happens per-pid in the inner loop below.
                            // is_interactive=false: no behavioral data at target-name level.
                            if classify_protection(
                                target,
                                &hard_protected,
                                &infra_protected,
                                &policy_prot,
                                false,
                            ) == ProtectionLevel::Unconditional
                            {
                                continue;
                            }
                            for (pid, process) in collector.system().processes() {
                                if process.name() == target {
                                    if Some(pid.as_u32()) == foreground_pid {
                                        // Process exists but is the active foreground app —
                                        // we intentionally skip it this cycle.
                                        targets_found_but_skipped = true;
                                    } else {
                                        // Add throttle only if not already actioned by individual skills.
                                        // But mark trialed=true regardless — the pressure measurement
                                        // captures the combined effect of all throttles in this cycle,
                                        // including targets already covered by throttle:X skills.
                                        if !already_actioned.contains(target) {
                                            actions.push(RootAction::throttle(
                                                pid.as_u32(),
                                                target.clone(),
                                                false,
                                                format!("trial:{}", skill_name),
                                            ));
                                        }
                                        trialed = true;
                                    }
                                    break;
                                }
                            }
                        }
                        if trialed {
                            pending_trial_skill = Some((skill_name, pressure_before));
                        } else if targets_found_but_skipped {
                            // At least one target exists but is foreground-protected this cycle.
                            // This is NOT an ineffective outcome — the skill simply couldn't run.
                            // Leave pending_trial_skill as None and wait for the next cycle when
                            // the process may be in the background.
                            // (apply_count is NOT incremented, so the skill is not GC'd.)
                        } else {
                            // No targets found in the process list at all — the skill's targets
                            // are genuinely absent (crashed, jetsam'd, or never launched).
                            // Mark as ineffective so the skill gets GC'd after enough failures.
                            lctx.skill_registry.record_result(&skill_name, false);
                        }
                    }
                }

                // Coordinated multi-process freezing (Pearl 2009 causal clusters).
                // If process A is already being actioned AND B co-occurs with A during
                // pressure spikes (≥8 observed co-occurrences), include B in this cycle.
                // This exploits the causal graph: "Safari + cloudd together cause 20%
                // pressure drop; individually each is only 10%."
                // Only triggers near the overflow threshold to avoid false over-throttling.
                if snapshot.pressure.memory_pressure
                    >= overflow_thresholds.bg_pressure as f64 - 0.05
                {
                    let causal_pairs = lctx.outcome_tracker.top_causal_pairs(5);
                    let actioned: std::collections::HashSet<String> = actions
                        .iter()
                        .filter_map(|a| match a {
                            RootAction::ThrottleProcess { name, .. }
                            | RootAction::FreezeProcess { name, .. } => Some(name.clone()),
                            _ => None,
                        })
                        .collect();
                    for (pa, pb, count) in &causal_pairs {
                        if *count < 8 {
                            continue;
                        }
                        let a_acted = actioned.iter().any(|n| n.contains(pa));
                        let b_acted = actioned.iter().any(|n| n.contains(pb));
                        if a_acted == b_acted {
                            continue; // both already actioned or neither
                        }
                        let missing = if a_acted { pb } else { pa };
                        let partner = if a_acted { pa } else { pb };
                        // Find the missing co-cluster partner and throttle it.
                        for (pid, proc) in collector.system().processes() {
                            let proc_name = proc.name().to_string();
                            if proc_name.contains(missing)
                                && !actioned.iter().any(|n| n.contains(missing))
                            {
                                actions.push(RootAction::throttle(
                                    pid.as_u32(),
                                    proc_name,
                                    false,
                                    format!(
                                        "coordinated-cluster: co-occurs with {} (n={})",
                                        partner, count
                                    ),
                                ));
                                break;
                            }
                        }
                    }
                }

                // Spotlight pressure gate: pause indexing when swap is heavy and
                // re-enable when pressure normalizes.  Uses mdutil (clean handshake
                // with Spotlight server) rather than SIGSTOP — no index corruption risk.
                // Gate: memory_pressure ≥ 0.75 AND swap ≥ 1.5 GB → pause.
                // Re-enable: memory_pressure < 0.55 AND spotlight was paused by us.
                {
                    let mem_p = snapshot.pressure.memory_pressure;
                    let swap_gb =
                        snapshot.pressure.swap_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    let can_mdutil = std::path::Path::new("/usr/bin/mdutil").exists();
                    if can_mdutil {
                        if !spotlight_paused && mem_p >= 0.75 && swap_gb >= 1.5 {
                            actions.push(
                                apollo_optimizer::engine::types::RootAction::ToggleSpotlight {
                                    enabled: false,
                                    reason: format!(
                                        "swap-pressure: mem={:.2} swap={:.1}GB",
                                        mem_p, swap_gb
                                    ),
                                },
                            );
                            spotlight_paused = true;
                        } else if spotlight_paused && mem_p < 0.55 {
                            actions.push(
                                apollo_optimizer::engine::types::RootAction::ToggleSpotlight {
                                    enabled: true,
                                    reason: "pressure-normalized: re-enabling spotlight"
                                        .to_string(),
                                },
                            );
                            spotlight_paused = false;
                        }
                    }
                }

                // Predictive agent: inject soft actions for PreThrottleNoise / ProactivePurge.
                match agent_intervention {
                    Intervention::PreThrottleNoise => {
                        // Renice top 3 noise processes (soft throttle, no SIGSTOP).
                        let noise_pats = state
                            .policy
                            .lock_recover()
                            .learned_policy
                            .noise_patterns
                            .clone();
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
                            actions.push(RootAction::throttle(
                                proc.pid as u32,
                                proc.name.clone(),
                                false,
                                "predictive-agent: pre-throttle noise",
                            ));
                        }
                    }
                    Intervention::ProactivePurge => {
                        // Send paging hints to top 3 background processes by RSS.
                        // SetMemorystatus with priority -1 asks the process to release caches
                        // voluntarily — no freeze, no kill. Passes through safety in execute_actions.
                        let interactive_pats = decide_interactive.clone();
                        let protected_pats = state
                            .policy
                            .lock_recover()
                            .learned_policy
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
                let already_has_hints = actions
                    .iter()
                    .any(|a| matches!(a, RootAction::SetMemorystatus { .. }));
                if signal_digest.pressure_smooth >= 0.60 && !already_has_hints {
                    let protected_pats = state
                        .policy
                        .lock_recover()
                        .learned_policy
                        .protected_patterns
                        .clone();
                    // Use proc_snaps (full process list) not top_processes (top 10 by CPU).
                    // Only skip core interactive apps — paging hints are gentle (voluntary
                    // cache release), so we use a tighter filter than freeze/throttle.
                    let hard_protected = protected_processes();
                    let infra_protected = infrastructure_processes();
                    let mut bg_procs: Vec<_> = proc_snaps
                        .iter()
                        .filter(|p| {
                            // Skip system-critical, infra, and policy-protected processes.
                            let is_interactive =
                                is_user_interactive_app(p.has_gui_window, p.secs_since_user_interaction, p.rss_bytes, &p.name);
                            classify_protection(&p.name, &hard_protected, &infra_protected, &protected_pats, is_interactive)
                                == ProtectionLevel::Unprotected
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
                        actions.push(RootAction::set_memorystatus(
                            proc.pid,
                            -1,
                            format!(
                                "pressure-driven hint (p={:.0}%): {} ({}MB)",
                                signal_digest.pressure_smooth * 100.0,
                                proc.name,
                                proc.rss_bytes / 1024 / 1024,
                            ),
                        ));
                    }
                }

                // Heuristic pass: AdaptiveGovernor
                // Pass hw_features (sampled every 5 cycles) for Bayesian fusion + online learning.
                let heuristic_decisions = {
                    let mut pg = state.policy.lock_recover();
                    pg.adaptive_governor.decide_all_with_hw(
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
                        .policy
                        .lock_recover()
                        .learned_policy
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
                        let pid_u32 = pid.as_u32();
                        let name = process.name().to_string();
                        // Evaluate interactive-app behavioral signals before calling
                        // classify_protection so the result is available for Tier 4.
                        let snap = proc_snaps.iter().find(|s| s.pid == pid_u32);
                        let has_gui = snap.map_or(false, |s| s.has_gui_window);
                        let idle_s = snap.map_or(3600, |s| s.secs_since_user_interaction);
                        let rss = snap.map_or(process.memory(), |s| s.rss_bytes);
                        let is_interactive = is_user_interactive_app(has_gui, idle_s, rss, &name);
                        match classify_protection(
                            &name,
                            &protected_pats,
                            &infra_pats,
                            &policy_protected,
                            is_interactive,
                        ) {
                            ProtectionLevel::Unconditional => {
                                // OS/system essentials, infrastructure, policy-learned
                                // daemons → always skip.
                                cpids.insert(pid_u32);
                                continue;
                            }
                            ProtectionLevel::ConditionalForeground => {
                                // User-interactive apps: protect when foreground, eligible
                                // for QoS hint / throttle when in background.
                                if Some(pid_u32) == foreground_pid {
                                    cpids.insert(pid_u32);
                                }
                                // Background user app: not inserted → eligible for
                                // throttle/QoS.
                                continue;
                            }
                            ProtectionLevel::Unprotected => {
                                // Fall through to dev-runtime behavioral gate below.
                            }
                        }
                        // Dev runtimes: behavioral gate — protection earned, not given.
                        if matches_dev_runtime(&name) {
                            let pid_u32 = pid.as_u32();
                            // Re-use the enriched ProcessSnapshot already looked up above
                            // (snap/has_gui/idle_s/rss are in scope from classify_protection
                            // evaluation), adding wakeups and network from the same snapshot.
                            let (cpu, wakeups, net, gui) = if let Some(s) = snap {
                                (
                                    s.cpu_percent,
                                    s.wakeups_per_sec,
                                    s.has_network,
                                    s.has_gui_window,
                                )
                            } else {
                                // Fallback: sysinfo process — limited signals but real RSS.
                                (process.cpu_usage(), 0.0, false, false)
                            };
                            let raw_score = behavioral_protection_score(
                                cpu, wakeups, net, gui, idle_s, rss, total_ram,
                            );
                            // Cable 5: process_relevance() → modulate BPS with user profile.
                            // If the user actively uses this process (relevance > 0), boost
                            // its behavioral score. If irrelevant (0.0), no change.
                            // This means a dev runtime the user has interacted with recently
                            // gets a relevance bonus, while one that's been stale loses it.
                            let relevance = state
                                .policy
                                .lock_recover()
                                .adaptive_governor
                                .user_profile
                                .process_relevance(&name);
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
                        m.metrics.bps_evaluated += bps_eval;
                        m.metrics.bps_protected += bps_prot;
                        m.metrics.bps_demoted += bps_dem;
                        if bps_min < f64::MAX {
                            m.metrics.bps_min_score = bps_min;
                            m.metrics.bps_min_score_name = bps_min_name;
                        }
                    }
                    // AMX/ML workloads: never throttle/freeze ML inference processes.
                    cpids.extend(amx_detector::ml_protected_pids());
                    cpids
                };

                // Convert heuristic decisions to RootActions and merge
                let (heuristic_actions, heuristic_stats) =
                    process_enrichment::convert_and_merge_heuristic_decisions(
                        &heuristic_decisions,
                        &actions,
                        &heuristic_critical_pids,
                    );
                // Cable 2: query_similar() → skip throttles that experience says won't work.
                // If we have ≥3 records of throttling process X at similar pressure and it
                // never helped (avg_drop ≤ 0), skip wasting the action budget on it.
                let current_pressure = snapshot.pressure.memory_pressure;
                let exp_band = learnable_params.experience_pressure_band;
                let heuristic_actions: Vec<RootAction> = heuristic_actions
                    .into_iter()
                    .filter(|a| {
                        if let RootAction::ThrottleProcess { ref name, .. } = a {
                            if let Some((avg_drop, confidence)) = lctx
                                .outcome_tracker
                                .experience
                                .query_similar_with_band(name, current_pressure, exp_band)
                            {
                                if confidence >= 0.5 && avg_drop <= 0.0 {
                                    // Experience says throttling this process at this pressure
                                    // has never reduced pressure. Skip it.
                                    return false;
                                }
                            }
                        }
                        true
                    })
                    .collect();
                actions.extend(heuristic_actions);

                // Cable: stale_apps() → nominate stale background apps as freeze candidates.
                // When pressure is elevated, apps the user hasn't interacted with for >30min
                // are prime freeze targets — they're consuming RAM without doing useful work.
                // Only nominate non-foreground, non-critical, non-already-acting processes.
                if signal_digest.pressure_smooth >= 0.50 {
                    let existing_pids: HashSet<u32> = actions
                        .iter()
                        .filter_map(|a| match a {
                            RootAction::FreezeProcess { pid, .. }
                            | RootAction::ThrottleProcess { pid, .. }
                            | RootAction::BoostProcess { pid, .. } => Some(*pid),
                            _ => None,
                        })
                        .collect();
                    let stale_names = {
                        let running: Vec<&str> = all_proc_names.iter().copied().collect();
                        let pg = state.policy.lock_recover();
                        pg.adaptive_governor.user_profile.stale_apps(&running, 1800)
                        // 30 min threshold
                    };
                    let sys = collector.system();
                    for (pid, process) in sys.processes() {
                        let pid_u32 = pid.as_u32();
                        let name = process.name().to_string();
                        if !stale_names.contains(&name) {
                            continue;
                        }
                        if Some(pid_u32) == foreground_pid {
                            continue;
                        }
                        if heuristic_critical_pids.contains(&pid_u32) {
                            continue;
                        }
                        if existing_pids.contains(&pid_u32) {
                            continue;
                        }
                        // Only freeze if using meaningful memory (>50MB RSS).
                        if process.memory() < 50 * 1024 * 1024 {
                            continue;
                        }
                        let (ss, su) = pid_start_time(pid_u32);
                        actions.push(RootAction::freeze_full(
                            pid_u32,
                            name.clone(),
                            format!(
                                "stale-app: no user interaction for >30min, rss={}MB",
                                process.memory() / 1024 / 1024
                            ),
                            ss,
                            su,
                        ));
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
                    lctx.overflow_guard.record_event(
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
                    lctx.signal_intel.record_overflow(
                        snapshot.pressure.memory_pressure,
                        sr,
                        snapshot.pressure.memory_pressure,
                        1.0,
                    );
                }
                // Decaimiento gradual: si el sistema está en calma, relajar thresholds.
                lctx.overflow_guard.tick_decay(
                    snapshot.pressure.memory_pressure,
                    snapshot.pressure.compressor_pressure,
                );

                // ── Neuromodulator: bio-inspired parameter modulation ────────
                {
                    let overflow_occurred = lctx.overflow_guard.history.total_overflows > 0;
                    let neuro_signals = NeuroSignals {
                        pressure_drop: signal_digest.pressure_smooth as f64 * -1.0
                            * signal_digest.pressure_velocity,
                        // Combine outcome-tracker RL penalty with stability oracle signal.
                        // rl_penalty ∈ [-3, 0]; instability_penalty ∈ [0, 1] scaled by 0.5
                        // → max additional penalty = -0.5, keeping the existing penalty
                        // dominant while letting stability shape policy at the margin.
                        // [Sutton & Barto 2018] §17.4 — reward shaping must preserve scale
                        // hierarchy or it inverts the optimal policy.
                        outcome_penalty: lctx.outcome_tracker.rl_penalty()
                            - 0.5
                                * stability_oracle.instability_penalty_attenuated(
                                    apollo_optimizer::engine::daemon_helpers::system_uptime_secs(),
                                ),
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
                        rl_exploring: lctx.overflow_guard.rl_agent.as_ref()
                            .map_or(false, |rl| rl.total_ticks() < 200),
                    };
                    lctx.neuromod.tick(&neuro_signals);

                    // Push derived params to subsystems + enforce constraints.
                    if let Some(rl) = &mut lctx.overflow_guard.rl_agent {
                        rl.neuro_alpha_mult = lctx.neuromod.alpha_multiplier;
                        rl.neuro_epsilon_bonus = lctx.neuromod.epsilon_bonus;
                        rl.dyna_steps = lctx.neuromod.dyna_steps;
                        rl.enforce_constraints(); // Infrastructure-locked (Hermes)
                    }
                    lctx.signal_intel.neuro_serotonin_shift = lctx.neuromod.serotonin_shift;
                }

                // ProcessRecoveryManager: freeze confirmed leakers. NEVER kill.
                //
                // Previous revisions of this block escalated to SIGKILL under
                // survival_mode + rss > 200 MB + attempts >= 2. That was
                // catastrophically wrong: the leak detector (see memory_analyzer
                // `detect_memory_leak`) fires on ANY process whose RSS grew in
                // 7/10 recent snapshots, which is literally normal behaviour
                // for every active user app (Chrome tab, Cursor buffer, Slack
                // message, Figma canvas). The kill branch had no foreground /
                // interactive / protected-pattern guard, so a Chrome tab
                // holding 500 MB would trivially satisfy rss > 200 MB and
                // attempts >= 2 and get SIGKILLed — the user observed exactly
                // this ("me cierra las apps").
                //
                // The recovery path is now freeze-only: SIGSTOP is reversible,
                // the user can SIGCONT anything apollo got wrong, and the
                // existing unfreeze fast-path recovers transparently when
                // pressure drops. The leak detector itself is also being
                // tightened in memory_analyzer.rs in the same commit so
                // normal interactive apps stop being flagged in the first
                // place; this block is the safety net.
                let recovery_targets = proc_recovery.get_recovery_targets();
                for target in &recovery_targets {
                    if heuristic_critical_pids.contains(&target.pid) {
                        continue;
                    }
                    let (ss, su) = pid_start_time(target.pid);
                    actions.push(RootAction::freeze_full(
                        target.pid,
                        target.name.clone(),
                        format!(
                            "memory-leak recovery: prob={:.2} rss={}MB attempts={}",
                            target.leak_probability,
                            target.rss_bytes / 1024 / 1024,
                            target.recovery_attempts,
                        ),
                        ss,
                        su,
                    ));
                    proc_recovery.record_kill_attempt(target.pid);
                }

                // ── Feature 5: Wakeup Budget Enforcer ───────────────────────
                // Graduated severity response: Critical/High → App Nap,
                // Medium → Background tier (E-cores), Low → skip.
                // [Nygard 2018 "Release It!" Ch.5 — graduated response avoids
                // over-reaction to transient wakeup bursts.]
                let storms = wake_storm.detect_storms();
                {
                    use apollo_optimizer::engine::mach_qos::SchedulingTier;
                    use apollo_optimizer::engine::wake_storm_detector::StormSeverity;
                    let storm_pids: std::collections::HashSet<u32> =
                        storms.iter().map(|s| s.pid).collect();
                    let mut qos = state.mach_qos.lock_recover();

                    // Apply severity-graduated mitigation.
                    for storm in &storms {
                        if heuristic_critical_pids.contains(&storm.pid)
                            || Some(storm.pid) == foreground_pid
                        {
                            continue;
                        }
                        let severity = wake_storm.get_severity(storm.wakeups_per_second);
                        match severity {
                            StormSeverity::Critical | StormSeverity::High => {
                                qos.set_app_nap(storm.pid, true);
                            }
                            StormSeverity::Medium => {
                                // E-core routing without full App Nap suppression.
                                qos.set_tier(storm.pid, SchedulingTier::Background);
                            }
                            StormSeverity::Low => {
                                // Below threshold: monitor only, no intervention.
                            }
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
                    let appnap_hard = protected_processes();
                    let appnap_infra = infrastructure_processes();
                    let appnap_policy = state
                        .policy
                        .lock_recover()
                        .learned_policy
                        .protected_patterns
                        .clone();
                    let mut qos = state.mach_qos.lock_recover();
                    for (pid, process) in collector.system().processes() {
                        let pid_u32 = pid.as_u32();
                        let name = process.name();
                        let is_foreground = Some(pid_u32) == foreground_pid;
                        // Evaluate behavioral signals for Tier-4 interactive detection.
                        let snap = proc_snaps.iter().find(|s| s.pid == pid_u32);
                        let has_gui = snap.map_or(false, |s| s.has_gui_window);
                        let idle_s = snap.map_or(3600, |s| s.secs_since_user_interaction);
                        let rss = snap.map_or(process.memory(), |s| s.rss_bytes);
                        let is_interactive = is_user_interactive_app(has_gui, idle_s, rss, name);
                        let protection = classify_protection(
                            name,
                            &appnap_hard,
                            &appnap_infra,
                            &appnap_policy,
                            is_interactive,
                        );
                        // Apollo itself is never app-napped (self-protection).
                        // Unconditional: OS/infra/policy — always skip.
                        // ConditionalForeground: user-interactive apps — skip only when foreground.
                        let should_protect = name == "apollo-optimizerd"
                            || protection == ProtectionLevel::Unconditional
                            || (protection == ProtectionLevel::ConditionalForeground
                                && is_foreground);
                        if should_protect {
                            // Protected: ensure NOT app-napped.
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
                    let mut fg_pids =
                        process_enrichment::build_foreground_family(foreground_pid, &process_tree);
                    // Augment with kernel-authoritative coalition membership.
                    // Any PID sharing a coalition with the foreground PID is excluded.
                    if let Some(fg_pid) = foreground_pid {
                        let all_pids: Vec<u32> = proc_snaps.iter().map(|s| s.pid).collect();
                        for coalition_pid in coalition_tracker.family_of(fg_pid, &all_pids) {
                            fg_pids.insert(coalition_pid);
                        }
                    }
                    let interactive_pats: Vec<String> = state
                        .policy
                        .lock_recover()
                        .learned_policy
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
                                    tracing::warn!(err = %e, "warn-limit");
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

                // Snapshot workload + ml_class from policy BEFORE acquiring metrics lock
                // (avoids holding two domain locks simultaneously).
                let current_workload_str = format!(
                    "{:?}",
                    state
                        .policy
                        .lock_recover()
                        .adaptive_governor
                        .user_profile
                        .current_workload()
                );
                // F2 — ML Ligero: read classification result (computed inside decide_all this cycle).
                // GovernorConfig aggressiveness was already updated inside decide_all().
                let ml_class = state
                    .policy
                    .lock_recover()
                    .adaptive_governor
                    .last_ml_classification()
                    .clone();
                // Update heuristic metrics
                {
                    let mut m = state.metrics.lock_recover();
                    m.metrics.heuristic_decisions += heuristic_stats.decisions_total;
                    m.metrics.heuristic_throttles += heuristic_stats.throttles;
                    m.metrics.heuristic_freezes += heuristic_stats.freezes;
                    m.metrics.heuristic_kills_downgraded += heuristic_stats.kills_downgraded;
                    m.metrics.zombies_detected += heuristic_stats.zombies_detected;
                    m.metrics.current_workload = current_workload_str;
                }
                // StabilityOracle: record zombie count + swap bytes + VM
                // thrashing score each cycle. The thrashing score comes from
                // the background pressure collector's VmRate derivation, which
                // captures per-second compression/decompression/swap churn —
                // the flow view of memory pressure that absolute percentages
                // can't see.
                stability_oracle.record_zombie_count(heuristic_stats.zombies_detected as usize);
                stability_oracle.record_swap_bytes(snapshot.pressure.swap_used_bytes);
                stability_oracle
                    .record_thrashing_score(pressure_collector.latest().thrashing_score);
                // System-wide CPU stall fraction from the global contention
                // tracker — fraction of tracked pids with PSI ratio ≥ 0.5.
                if let Ok(tracker) =
                    apollo_optimizer::engine::contention_tracker::global().lock()
                {
                    // 0.85: see metrics-population site for the rationale —
                    // Darwin's runnable counter saturates above 0.5 under any
                    // normal load, so a lower threshold misclassifies normal
                    // multitasking as a stability problem.
                    stability_oracle.record_stall_fraction(tracker.stall_fraction(0.85));
                }
                {
                    let mut m = state.metrics.lock_recover();
                    m.metrics.ml_confidence = ml_class.confidence;
                    m.metrics.current_workload = format!("{:?}", ml_class.workload).to_lowercase();
                    m.metrics.ml_sources = ml_class.sources_summary();
                }
                // Cable: GPU optimize_for_workload → log GPU-specific hints when
                // workload changes AND GPU is drawing power (gpu_active in features).
                // This feeds observability: what GPU strategy Apollo recommends per workload.
                if cycle_hw_snap
                    .as_ref()
                    .and_then(|h| h.power.gpu_watts)
                    .unwrap_or(0.0)
                    > 2.0
                {
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
                if is_root && cycle_count % 30 == 0 {
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
                        actions.push(RootAction::set_sysctl(
                            key,
                            value,
                            format!("network-optimizer: {:?} profile", net_profile),
                        ));
                    }
                }

                // Update SharedState with latest sysctl governor status for ctl queries.
                {
                    let status = sysctl_governor.status(&network_monitor);
                    state.hardware.lock_recover().sysctl_governor_status = status;
                }

                // RevertSysctls RPC: if requested via socket, revert all sysctl changes now.
                if state.revert_sysctls_requested.swap(false, Ordering::AcqRel) {
                    tracing::info!("RevertSysctls RPC: reverting sysctl changes to defaults");
                    let revert_actions = sysctl_governor.revert_to_defaults();
                    if !revert_actions.is_empty() {
                        let caps = detect_capabilities();
                        let mut frozen_dummy = std::collections::HashSet::new();
                        let outcomes = execute_actions(
                            revert_actions,
                            &caps,
                            &journal_path,
                            &mut frozen_dummy,
                            &[],
                            &[],
                            None,
                            dry_run,
                        );
                        if outcomes.failures == 0 {
                            sysctl_governor.mark_reverted();
                        } else {
                            tracing::warn!(
                                failures = outcomes.failures,
                                "RevertSysctls RPC: revert had failures"
                            );
                        }
                    } else {
                        sysctl_governor.mark_reverted();
                    }
                }

                // F3 — Safety Precedence: foreground app is NEVER throttled or frozen.
                // Also protects recently active apps (minimized but used in the last 5 min).
                // Only logs to discrepancy when the reason is ambiguous (not covered by
                // foreground detection or activity sensor) — those are the cases where
                // the LLM teacher actually adds value.
                {
                    let fg_family_pids =
                        process_enrichment::build_foreground_family(foreground_pid, &process_tree);
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
                            process_enrichment::append_discrepancy_log(
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

                // ── Chromium Renderer Manager ────────────────────────────────────
                // Update renderer inventory, apply E-core demotions and SIGSTOP/SIGCONT
                // for idle tab renderers across ALL Chromium/Electron apps.
                // Runs after all main-loop freeze decisions so we can exclude those PIDs.
                // [Denning 1968] Working Set | [Jones 2011] Chromium Multi-Process Architecture
                {
                    use apollo_optimizer::engine::chromium_manager::ChromiumAction;

                    // FocusMarkov context: top-5 predictions + elapsed dwell time.
                    // [Altmann & Trafton 2002] Pre-thaw browsers predicted as next switch.
                    {
                        let preds: Vec<(String, f64, f64)> = focus_markov
                            .predict_top_n(foreground_app.as_deref().unwrap_or(""), 5)
                            .into_iter()
                            .map(|p| (p.app_name, p.probability, p.avg_dwell_secs))
                            .collect();
                        let elapsed = focus_markov.elapsed_dwell_secs();
                        chromium_mgr.set_markov_context(&preds, elapsed);
                    }

                    // Pressure-adaptive aggressiveness
                    chromium_mgr.set_pressure_context(snapshot.pressure.memory_pressure as f32);
                    // Arousal-adaptive aggressiveness [Yerkes-Dodson 1908]
                    // Override pressure thresholds with arousal-based ones when
                    // arousal signal is available — crisis arousal freezes faster,
                    // idle arousal thaws everything.
                    chromium_mgr.set_arousal_context(arousal_state.level);
                    // Build preemption: when the user is in a build session,
                    // force maximum freeze aggressiveness on background
                    // renderers BEFORE rustc/cargo/clang demand RAM. This is
                    // what prevents OOM-driven reboots on 8GB systems: bulkhead
                    // renderer memory from build memory.
                    chromium_mgr.set_build_preemption(
                        win_workload_intent == WorkloadIntent::BuildSession,
                    );
                    // Pause freeze decisions during window ops / app launches
                    chromium_mgr.set_fluidity_context(
                        fluidity_state.window_op_active(),
                        fluidity_state.launch_active,
                    );

                    // Build assertion PID set from existing frozen/active tracking
                    let chromium_assertion_pids =
                        apollo_optimizer::engine::activity_sensor::pids_with_assertions();

                    // Current main-daemon frozen set (don't interfere)
                    let main_frozen_set: HashSet<u32> =
                        state.frozen_state.lock_recover().keys().copied().collect();

                    let proc_list: Vec<(u32, &str, f32, u64)> = proc_snaps
                        .iter()
                        .map(|p| (p.pid, p.name.as_str(), p.cpu_percent, p.rss_bytes))
                        .collect();

                    let chromium_actions = chromium_mgr.update(
                        &proc_list,
                        foreground_pid,
                        &chromium_assertion_pids,
                        &main_frozen_set,
                    );

                    // Execute chromium actions
                    for action in &chromium_actions {
                        match action {
                            ChromiumAction::FreezeRenderer {
                                pid,
                                name,
                                estimated_mb,
                            } => {
                                tracing::info!(
                                    pid = pid,
                                    name = name.as_str(),
                                    estimated_mb = estimated_mb,
                                    "chromium: freezing idle renderer"
                                );
                                let ok = apollo_optimizer::engine::chromium_manager::ChromiumManager::freeze_renderer(*pid);
                                if ok {
                                    // Register in the main frozen_state for crash-recovery
                                    // and unified observability.
                                    let mut fs = state.frozen_state.lock_recover();
                                    fs.entry(*pid).or_insert(
                                        apollo_optimizer::engine::types::FrozenEntry {
                                            frozen_at: chrono::Utc::now(),
                                            source: apollo_optimizer::engine::types::FreezeSource::ChromiumManager,
                                            pressure_at_freeze: snapshot.pressure.memory_pressure,
                                            process_name: Some(name.clone()),
                                        },
                                    );
                                }
                            }
                            ChromiumAction::ThawRenderer { pid, name } => {
                                tracing::info!(
                                    pid = pid,
                                    name = name.as_str(),
                                    "chromium: thawing renderer (became active)"
                                );
                                apollo_optimizer::engine::chromium_manager::ChromiumManager::thaw_renderer(*pid);
                                state.frozen_state.lock_recover().remove(pid);
                                // NARS belief update: observe whether renderer survived freeze.
                                // Pass the full process name — FreezeIntelligence.classify()
                                // maps it to the correct category ("chromium-renderer" etc.)
                                // so evidence accumulates across all renderers of the same type.
                                // Alive after thaw = success (0.3 salience). Dead = failure (0.8).
                                // [Pei Wang 2013] Revision rule accumulates evidence over time.
                                let alive = proc_snaps.iter().any(|p| p.pid == *pid);
                                chromium_mgr.observe_freeze_outcome(
                                    name.as_str(),
                                    alive,
                                    if alive { 0.3 } else { 0.8 },
                                );
                            }
                            ChromiumAction::DemoteToEcores { pid, name } => {
                                tracing::debug!(
                                    pid = pid,
                                    name = name.as_str(),
                                    "chromium: E-core demotion for background renderer"
                                );
                                let mut qos = state.mach_qos.lock_recover();
                                let _ = qos.set_tier(
                                    *pid,
                                    apollo_optimizer::engine::mach_qos::SchedulingTier::Background,
                                );
                            }
                        }
                    }

                    // Update chromium metrics in RuntimeMetrics
                    {
                        let cm = chromium_mgr.metrics();
                        let mut m = state.metrics.lock_recover();
                        m.metrics.chromium_renderers_total = cm.total_renderers;
                        m.metrics.chromium_renderers_frozen = cm.frozen_renderers;
                        m.metrics.chromium_renderers_ecore = cm.ecore_renderers;
                        m.metrics.chromium_freed_mb = cm.estimated_freed_mb;
                        m.metrics.chromium_browsers_managed = cm.browsers_managed;
                    }
                }

                let policy = SafetyPolicy::for_capabilities(
                    SafetyPolicy::for_profile(current_profile),
                    hw_cores,
                    hw_ram_gb,
                );

                let now = Instant::now();
                if thrash
                    .minute_started
                    .map(|s| now.duration_since(s) >= Duration::from_secs(60))
                    .unwrap_or(true)
                {
                    thrash.minute_started = Some(now);
                    state.metrics.lock_recover().metrics.budgets.minute_actions = 0;
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
                                should_unfreeze(elapsed, entry.pressure_at_freeze, current_pressure)
                                    && !interrupt_pids.contains(pid)
                            })
                            .map(|(pid, _)| *pid)
                            .collect();
                        // FIFO rotation: on 8GB hardware, rotate oldest frozen
                        // process to prevent resource hoarding under sustained pressure.
                        if let Some((&oldest_pid, oldest_entry)) = frozen_state
                            .iter()
                            .filter(|(pid, _)| {
                                !interrupt_pids.contains(pid) && !expired.contains(pid)
                            })
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
                            metrics.metrics.post_wake_defensive_unfreezes += count;
                            metrics.metrics.unfreezes_applied += count;
                            metrics.metrics.throttle_reverted += count;
                        }
                    }
                    metrics.metrics.budgets.cycle_boosts = 0;
                    metrics.metrics.budgets.cycle_throttles = 0;
                    metrics.metrics.budgets.cycle_hints = 0;
                    metrics.metrics.budgets.cycle_freezes = 0;
                    metrics.metrics.budgets.cycle_sysctl_writes = 0;
                    metrics.metrics.budgets.boost_denied_cooldown = 0;

                    let (graced_actions, throttle_suppressed, freeze_suppressed) =
                        process_enrichment::apply_post_wake_grace_policy(actions, grace_active);
                    metrics.metrics.post_wake_throttle_suppressed += throttle_suppressed;
                    metrics.metrics.post_wake_freeze_suppressed += freeze_suppressed;

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
                    // Fluidity launch guard: defer ALL new freezes while an app launch
                    // is in progress. App startup is a latency-sensitive critical path —
                    // background interference causes visible jank and slow launch times.
                    // [Selkowitz 1984] "Graphical Simulation" — launch is the user's
                    // primary interaction moment; background must yield completely.
                    let fluidity_launch_active = fluidity_state.launch_active;
                    if fluidity_launch_active {
                        tracing::debug!(
                            launch = %fluidity_state.launch_name,
                            cycles_remaining = fluidity_state.launch_cycles_remaining,
                            "fluidity: launch active — deferring new freezes"
                        );
                    }

                    // Pre-emptive fluidity response: when Kalman predicts fluidity will drop
                    // below 0.60 within 3 cycles AND velocity is negative, lower the freeze
                    // confirmation threshold from 2 to 1 cycle. This allows Apollo to act
                    // before the degradation is perceptible to the user.
                    // [Welch & Bishop 2006] Kalman prediction enables anticipatory control.
                    // [Shavit & Lotan 2000] pre-emptive action on predicted queue saturation.
                    let fluidity_preemptive = !fluidity_launch_active
                        && fluidity_state.fluidity_predicted_3s < 0.60
                        && fluidity_state.fluidity_velocity < -0.05;
                    if fluidity_preemptive {
                        tracing::info!(
                            predicted = fluidity_state.fluidity_predicted_3s,
                            velocity = fluidity_state.fluidity_velocity,
                            "fluidity: predicted drop to {:.2} — pre-emptive freeze threshold lowered",
                            fluidity_state.fluidity_predicted_3s
                        );
                    }

                    let confirmed_actions: Vec<RootAction> = graced_actions
                        .into_iter()
                        .filter(|a| {
                            if let RootAction::FreezeProcess { pid, .. } = a {
                                // Skip new freezes during app launch (launch acceleration)
                                if fluidity_launch_active {
                                    return false;
                                }
                                let count = freeze_candidates.entry(*pid).or_insert(0);
                                *count += 1;
                                // Pre-emptive mode: act after 1 cycle instead of 2
                                let required = if fluidity_preemptive { 1 } else { 2 };
                                *count >= required
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
                                    metrics.metrics.memory_pressure,
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
                                    "pressure": (metrics.metrics.memory_pressure * 1000.0).round() / 1000.0,
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
                    metrics.metrics.deep_scan_count += ds_scans;
                    metrics.metrics.deep_scan_temp_probes += ds_probes;
                    metrics.metrics.deep_scan_freeze += ds_freeze;
                    metrics.metrics.deep_scan_skip += ds_skip;
                    metrics.metrics.deep_scan_hint += ds_hint;

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
                        metrics.metrics.unfreezes_applied += stuck_pids.len() as u64;
                    }

                    let filtered = process_enrichment::filter_boost_cooldown(
                        confirmed_actions,
                        &policy,
                        &mut thrash,
                    );
                    let minute_cap = match latency_target {
                        LatencyTarget::Max => 120,
                        LatencyTarget::Low => 50,
                        LatencyTarget::Normal => 80,
                    };
                    let fa = enforce_limits_with_budget(
                        filtered,
                        &policy,
                        &mut metrics.metrics.budgets,
                        minute_cap,
                    );
                    metrics.metrics.last_actions_summary = format!(
                        "actions={} boosts={} throttles={} freezes={} sysctl={} invalid_sysctl_denied={}",
                        fa.len(),
                        fa.iter().filter(|a| matches!(a, RootAction::BoostProcess { .. })).count(),
                        fa.iter().filter(|a| matches!(a, RootAction::ThrottleProcess { .. })).count(),
                        fa.iter().filter(|a| matches!(a, RootAction::FreezeProcess { .. })).count(),
                        fa.iter().filter(|a| matches!(a, RootAction::SetSysctl { .. })).count(),
                        metrics.metrics.invalid_sysctl_denied
                    );
                    fa
                    // metrics lock released here
                };

                // Phase 2: Execute actions WITHOUT holding the metrics lock.
                //
                // Priority action queue: buffer this cycle's decided actions and
                // dispatch at most max_per_cycle per cycle. Urgent (Unfreeze) actions
                // bypass the cap. Any overflow stays in the queue for the next cycle.
                action_queue.push_all(final_actions);
                let final_actions = action_queue.drain_cycle();
                // Update backpressure metrics (observable in runtime_metrics.json).
                {
                    let bp = action_queue.backpressure_ratio();
                    let pending_depth = lctx.outcome_tracker.pending_depth();
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.action_queue_backpressure = bp;
                    metrics.metrics.outcome_pending_depth = pending_depth;
                }

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
                    use apollo_optimizer::engine::degradation::{DegradationInputs, OperationMode};

                    // ── Circuit breaker + degradation pre-check ───────────────
                    // Snapshot circuit breaker state before acquiring heavy locks.
                    let (cb_is_open, cb_open_duration) = {
                        let pg = state.policy.lock_recover();
                        let is_open = *pg.circuit_breaker.state()
                            == apollo_optimizer::engine::circuit_breaker::CircuitState::Open;
                        let dur = pg.circuit_breaker.open_duration();
                        (is_open, dur)
                    };

                    // Evaluate degradation tier; update last-cycle inputs.
                    let op_mode = {
                        // kernel_task CPU from top_processes (already captured this cycle).
                        let kernel_cpu = snapshot
                            .top_processes
                            .iter()
                            .find(|p| p.name == "kernel_task")
                            .map(|p| p.cpu_usage as f64)
                            .unwrap_or(0.0);
                        let mut pg = state.policy.lock_recover();
                        let inp = DegradationInputs {
                            new_failures: 0, // incremental failures added after execution
                            kernel_task_cpu_pct: kernel_cpu,
                            circuit_open: cb_is_open,
                            circuit_open_duration: cb_open_duration,
                        };
                        pg.degradation.update(&inp).clone()
                    };

                    // ── Cognitive gate: block_aggressive / observe_only ──────────────
                    // Epistemic uncertainty > 0.70 → Conservative (no SIGSTOP, no throttle).
                    // Epistemic uncertainty > 0.85 → Observe (no actions at all).
                    // [Sutton 2018 §13: reduce action scope under high policy uncertainty]
                    // [Lakshminarayanan 2017: predictive uncertainty → action inhibition]
                    let op_mode = if prev_cog_decision.as_ref().map_or(false, |d| d.observe_only)
                        && op_mode == OperationMode::Full
                    {
                        tracing::debug!("cognitive gate: observe_only → OperationMode::Observe");
                        OperationMode::Observe
                    } else if prev_cog_decision
                        .as_ref()
                        .map_or(false, |d| d.block_aggressive)
                        && op_mode == OperationMode::Full
                    {
                        tracing::debug!(
                            "cognitive gate: block_aggressive → OperationMode::Conservative"
                        );
                        OperationMode::Conservative
                    } else {
                        op_mode
                    };

                    // Filter actions based on degradation tier.
                    let filtered_actions: Vec<RootAction> = if op_mode == OperationMode::Emergency {
                        // Emergency: only unfreeze, no new actions.
                        final_actions
                            .into_iter()
                            .filter(|a| matches!(a, RootAction::UnfreezeProcess { .. }))
                            .collect()
                    } else if op_mode == OperationMode::Observe {
                        // Observe: no actions at all.
                        Vec::new()
                    } else if op_mode == OperationMode::Conservative {
                        // Conservative: only unfreeze + QoS hints (no SIGSTOP, no throttle).
                        final_actions
                            .into_iter()
                            .filter(|a| {
                                matches!(
                                    a,
                                    RootAction::UnfreezeProcess { .. }
                                        | RootAction::SetThreadQoS { .. }
                                        | RootAction::BoostProcess { .. }
                                )
                            })
                            .collect()
                    } else {
                        // Full: all actions pass through.
                        final_actions
                    };

                    // Dedup: skip ThrottleProcess for PIDs already throttled last cycle.
                    // Without this, decide_actions re-throttles 30+ PIDs every cycle,
                    // each producing a journal write → I/O saturation → system freeze.
                    let filtered_actions = {
                        use std::sync::Mutex;
                        static PREV_THROTTLED: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
                        let prev = PREV_THROTTLED.lock().unwrap_or_else(|e| e.into_inner());
                        let prev_set = prev.clone().unwrap_or_default();
                        drop(prev);
                        let mut this_cycle = HashSet::new();
                        let deduped: Vec<RootAction> = filtered_actions
                            .into_iter()
                            .filter(|a| {
                                if let RootAction::ThrottleProcess { pid, .. } = a {
                                    this_cycle.insert(*pid);
                                    !prev_set.contains(pid)
                                } else {
                                    true
                                }
                            })
                            .collect();
                        *PREV_THROTTLED.lock().unwrap_or_else(|e| e.into_inner()) =
                            Some(this_cycle);
                        deduped
                    };

                    // Extract a temporary HashSet for execute_actions (which requires &mut HashSet<u32>).
                    let mut frozen_set: HashSet<u32> =
                        state.frozen_state.lock_recover().keys().copied().collect();
                    // Snapshot before execution — used to detect changes and skip redundant disk writes.
                    let frozen_before: HashSet<u32> = frozen_set.clone();
                    let (learned_protected, learned_interactive) = {
                        let pg = state.policy.lock_recover();
                        (
                            pg.learned_policy.protected_patterns.clone(),
                            pg.learned_policy.interactive_patterns.clone(),
                        )
                    };
                    let mut qos = state.mach_qos.lock_recover();

                    // ── Circuit breaker + execute_actions ────────────────────
                    // We use the external record_success/record_failure API so the
                    // Mutex is never held across blocking I/O.
                    let outcomes = if cb_is_open {
                        // Circuit Open: only dispatch unfreeze (always safe).
                        tracing::warn!(
                            op_mode = op_mode.as_str(),
                            "circuit-breaker: open — skipping execute_actions, dispatching unfreeze only"
                        );
                        let safe_actions: Vec<RootAction> = filtered_actions
                            .into_iter()
                            .filter(|a| matches!(a, RootAction::UnfreezeProcess { .. }))
                            .collect();
                        execute_actions(
                            safe_actions,
                            &caps,
                            &journal_path,
                            &mut frozen_set,
                            &learned_protected,
                            &learned_interactive,
                            Some(&mut qos),
                            dry_run,
                        )
                    } else {
                        // Circuit Closed or HalfOpen: run normally, then report outcome.
                        let out = execute_actions(
                            filtered_actions,
                            &caps,
                            &journal_path,
                            &mut frozen_set,
                            &learned_protected,
                            &learned_interactive,
                            Some(&mut qos),
                            dry_run,
                        );
                        // Report outcome to circuit breaker (lock released before I/O above).
                        {
                            let mut pg = state.policy.lock_recover();
                            if out.failures == 0 {
                                pg.circuit_breaker.record_success();
                            } else {
                                for _ in 0..out.failures {
                                    pg.circuit_breaker.record_failure();
                                }
                            }
                        }
                        out
                    };

                    // Update degradation controller with new failure count from this cycle.
                    if outcomes.failures > 0 {
                        let mut pg = state.policy.lock_recover();
                        let inp = DegradationInputs {
                            new_failures: outcomes.failures,
                            kernel_task_cpu_pct: 0.0,
                            circuit_open: false,
                            circuit_open_duration: None,
                        };
                        pg.degradation.update(&inp);
                    }

                    // Sync the temporary set back into the unified frozen_state map.
                    let now = Utc::now();
                    let mut frozen_state = state.frozen_state.lock_recover();
                    // Add newly frozen PIDs.
                    for pid in &frozen_set {
                        frozen_state.entry(*pid).or_insert(FrozenEntry {
                            frozen_at: now,
                            source: FreezeSource::MainLoop,
                            pressure_at_freeze: snapshot.pressure.memory_pressure,
                            process_name: None, // name not available in this execute path
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
                            lctx.energy_tracker.record_savings(saved_watts, 30.0);
                        }
                    }
                }

                // ── Cognitive gate: pause_learning ───────────────────────────────────
                // UCHS < 0.40 (recovery mode) → skip weight updates this cycle.
                // Arousal EMA and causal graph still update (safe, no Bayesian corruption).
                // [Goodfellow 2016 §7: regularization via confidence-adaptive learning rate]
                let cognitive_pause = prev_cog_decision
                    .as_ref()
                    .map_or(false, |d| d.pause_learning);
                if cognitive_pause {
                    tracing::debug!(
                        uchs = prev_cog_decision.as_ref().map_or(0.0, |d| d.uchs_composite),
                        "cognitive gate: learning paused (UCHS recovery mode)"
                    );
                }

                // Learning tick: outcome tracking, causal graph, RL cables, predictive
                // agent, and periodic persist (every 100 cycles). Extracted to
                // learning_tick.rs for readability; behaviour is unchanged.
                // Skipped when UCHS recovery mode active (cognitive_pause).
                if !cognitive_pause {
                    learning_tick::run_learning_tick(
                        &snapshot,
                        &cycle_hw_snap,
                        &exec_outcomes,
                        &throttle_names_for_outcome,
                        &signal_digest,
                        workload_mode,
                        cycle_count,
                        &state,
                        &collector,
                        &mut lctx,
                        &mut learning_pipeline,
                        &mut effectiveness_tracker,
                        &mut restore_monitor,
                        &mut last_restore_quality,
                        &mut prev_package_watts,
                        &mut prev_workload_mode,
                        &mut arousal_state,
                        pending_trial_skill.clone(),
                        last_specialist_votes.as_ref().map(|(v, i)| (v.as_slice(), *i)),
                        &mut log_ingester,
                        &mut learnable_params,
                        ls_path.to_str().unwrap_or(""),
                        persist_generations,
                        skills_path(),
                        &mut nested_learner,
                    );
                    // Apply ws_spike_threshold / fluidity_degraded_threshold from LearnableParams.
                    // Keeps fluidity detection calibrated with learned values.
                    if persist_generations % 100 == 50 {
                        fluidity_state.apply_thresholds(
                            learnable_params.ws_spike_threshold,
                            learnable_params.fluidity_degraded_threshold,
                        );
                    }
                } // end if !cognitive_pause
                  // ── Neurocognitive tick ──────────────────────────────────────────────
                  // Runs after learning_tick so all signals (drift, causal, arousal) are
                  // fresh. Feeds 8 cognitive modules. Result stored in prev_cog_decision
                  // for gating next-cycle learning and current-cycle metrics.
                let cog_decision = {
                    // ── Derive real epistemic signals from subsystems ─────────────────
                    // [Lakshminarayanan 2017] predictive uncertainty from ensemble variance.
                    // RL Q-value variance: std-dev across arm avg-rewards → spread = uncertainty.
                    let rl_q_variance = {
                        let avg = lctx.predictive_agent.arm_avg_rewards();
                        let n = avg.len() as f64;
                        let mean = avg.iter().sum::<f64>() / n;
                        let var = avg.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n;
                        (var.sqrt() as f32).clamp(0.0, 1.0)
                    };
                    // LinUCB exploration: UCB for the most-pulled arm (lower = more exploited).
                    let linucb_exploration = {
                        let pulls = lctx.predictive_agent.arm_pulls();
                        let total = lctx.predictive_agent.total_cycles();
                        if total > 1 {
                            let best = pulls.iter().copied().max().unwrap_or(1).max(1);
                            ((2.0 * (total as f64).ln() / best as f64).sqrt().min(1.0) as f32)
                                .clamp(0.0, 1.0)
                        } else {
                            1.0 // maximum uncertainty on cold start
                        }
                    };
                    // Full causal confidence map — lets SelfRewardingEvaluator look up
                    // any past action's confidence, not just the current one.
                    // [Yuan 2024 §3 DR-ZERO]: CausalGraph as internal oracle for JuicyScore.
                    let causal_confidence_map: Vec<(String, f32)> =
                        lctx.causal_graph.confidence_map().into_iter().collect();
                    let top_causal = lctx
                        .causal_graph
                        .solid_edges_by_impact()
                        .first()
                        .map(|e| e.confidence)
                        .unwrap_or(0.0);
                    let cog_inputs = cognitive_tick::CognitiveTickInputs {
                        cycle: cycle_count as u64,
                        pressure: signal_digest.pressure_smooth,
                        drift_score: lctx.outcome_tracker.nars_drift_score(),
                        rl_q_variance,
                        linucb_exploration,
                        nars_min_confidence: (1.0 - lctx.outcome_tracker.nars_drift_score() as f32)
                            .clamp(0.0, 1.0),
                        outcome_effectiveness: lctx.outcome_tracker.overall_effectiveness(),
                        causal_confidence: top_causal,
                        causal_confidence_map,
                        latest_action: throttle_names_for_outcome
                            .first()
                            .map(|n| format!("throttle:{}", n)),
                        predicted_score: lctx
                            .predictive_agent
                            .arm_avg_rewards()
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max)
                            .clamp(0.0, 1.0) as f32,
                        workload_fingerprint: workload_mode
                            .as_str()
                            .bytes()
                            .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64)),
                        rl_state_idx: 0,
                        rl_q_delta: 0.0,
                        linucb_arm_idx: 0,
                        linucb_delta: 0.0,
                    };
                    cognitive_tick::run_cognitive_tick(
                        &mut cognitive_state,
                        &cog_inputs,
                        Some(&mut lctx.outcome_tracker.drift_detector),
                    )
                };
                prev_cog_decision = Some(cog_decision);
                // Autonomous rule induction: mine experience memory + co-occurrence
                // graph for new skills every 100 cycles (~50s).  No human needed —
                // Apollo crystallizes its own observations into reusable rules.
                if cycle_count % 100 == 0 {
                    let existing_names = lctx.skill_registry.name_set();
                    let top_pairs = lctx.outcome_tracker.top_causal_pairs(100);
                    let protected_set = apollo_optimizer::engine::safety::protected_processes();
                    // Also exclude policy-protected processes (learned by LLM/user).
                    // Without this, rule_inducer generates skills whose targets are
                    // unthrottleable — they accumulate zero observations forever.
                    let policy_prot = state
                        .policy
                        .lock_recover()
                        .learned_policy
                        .protected_patterns
                        .clone();
                    let policy_prot_refs: Vec<&str> =
                        policy_prot.iter().map(|s| s.as_str()).collect();
                    let mut all_protected: Vec<&str> = protected_set.iter().copied().collect();
                    all_protected.extend_from_slice(&policy_prot_refs);
                    let new_skills = apollo_optimizer::engine::rule_inducer::induce(
                        &lctx.outcome_tracker.experience,
                        &top_pairs,
                        &existing_names,
                        &all_protected,
                        workload_mode.as_str(),
                    );
                    let induced_count = new_skills.len();
                    for skill in new_skills {
                        lctx.skill_registry.register_induced(skill);
                    }
                    // Purge induced skills whose targets are all protected —
                    // they can never execute and would spin in the trial loop forever.
                    lctx.skill_registry.purge_unexecutable(&all_protected);
                    if induced_count > 0 {
                        println!(
                            "rule_inducer: {} new skills crystallized (total={})",
                            induced_count,
                            lctx.skill_registry.len()
                        );
                        lctx.skill_registry
                            .persist(std::path::Path::new(skills_path()));
                    }
                }
                // State compression (% 500) is handled by run_periodic() below.
                // Hourly housekeeping (7200 cycles × 500ms ≈ 1 hour).
                if cycle_count % 7200 == 0 {
                    // GC stale entries from cache warmer + I/O shaper.
                    cache_warmer.gc();
                    io_shaper.gc();
                    // Persist temporal predictor state.
                    temporal_predictor.persist();
                }
                // Metrics reporting: update learning metrics, apply I/O shaping,
                // route processes to P/E cores, and merge execution outcomes.
                // Extracted to metrics_reporter.rs; behaviour is unchanged.
                metrics_reporter::update_learning_metrics(
                    &state,
                    &lctx,
                    &signal_digest,
                    &agent_intervention,
                    &arousal_state,
                );
                metrics_reporter::apply_io_shaping(
                    cycle_count,
                    is_root,
                    &snapshot,
                    foreground_pid,
                    &process_tree,
                    &heuristic_decisions,
                    &power_mgr,
                    thermal_pressure_boost,
                    &mut io_shaper,
                    &state,
                );
                metrics_reporter::apply_qos_routing(
                    cycle_count,
                    &state,
                    foreground_pid,
                    &process_tree,
                    &heuristic_decisions,
                    &heuristic_critical_pids,
                    &thermal_action,
                );

                // ── Fluidity QoS elevation ───────────────────────────────────
                // When a window operation or app launch is active, elevate the
                // foreground app to Foreground (P-Core) tier immediately.
                // [Apple QoS Programming Guide 2014] user-interactive QoS =
                // render-frame priority on P-Cores (Firestorm).
                if (fluidity_state.window_op_active() || fluidity_state.app_launching())
                    && !thermal_action.force_ecores
                {
                    if let Some(fg_pid) = foreground_pid {
                        let mut qos = state.mach_qos.lock_recover();
                        let outcome = qos.set_tier(
                            fg_pid,
                            apollo_optimizer::engine::mach_qos::SchedulingTier::Foreground,
                        );
                        if outcome.success {
                            tracing::debug!(
                                pid = fg_pid,
                                window_op = fluidity_state.window_op_active(),
                                launching = fluidity_state.launch_active,
                                "fluidity: elevated foreground to P-Core (Foreground QoS)"
                            );
                        }
                    }
                }

                metrics_reporter::merge_cycle_metrics(
                    &state,
                    &exec_outcomes,
                    &network_monitor,
                    decision.reactor_event_weight,
                    &decision.blockers,
                    current_profile,
                    &governor_decision,
                    &lctx,
                    &overflow_thresholds,
                    &cycle_start,
                    reactor_weight,
                    &mut override_was_active,
                    &mut critical_failure_timestamps,
                    Path::new(&timeline_path),
                    Path::new(&metrics_path),
                );

                // ── Enriched telemetry wiring (2026-04-04) ──────────────────────────────
                // Fields added to RuntimeMetrics that can only be computed here in the
                // main loop where swap_forecast, sys, and cycle state are in scope.
                {
                    let mut m = state.metrics.lock_recover();
                    // SwapTrend — previously computed but never exposed.
                    m.metrics.swap_trend = format!("{:?}", swap_forecast.swap_trend);
                    // WindowServer CPU — use EMA from FluidityState (already computed
                    // each cycle in the proc_snaps block). More stable than raw sample.
                    m.metrics.windowserver_cpu_pct = fluidity_state.windowserver_cpu_ema;
                    // Compression signal from the EMA-smoothed compressor_pressure already
                    // computed by the collector (ratio of compressor pages to total physical
                    // pages × 0.85). The old formula used_ram - (total - free) was wrong:
                    // on macOS total ≠ used + free (inactive/wired/speculative pages exist),
                    // producing saturating_sub underflow → always 0 or nonsense.
                    m.metrics.compressed_memory_ratio =
                        snapshot.pressure.compressor_pressure.clamp(0.0, 1.0);
                    // Frozen RAM: sum of RSS of currently frozen PIDs.
                    let sys = collector.system();
                    let frozen_pids = state.frozen_state.lock_recover().clone();
                    m.metrics.frozen_ram_mb = frozen_pids
                        .iter()
                        .filter_map(|(pid, _)| sys.process(sysinfo::Pid::from_u32(*pid)))
                        .map(|p| p.memory() as f64 / (1024.0 * 1024.0))
                        .sum();
                    // cycles_high_pressure — consecutive cycles above bg_pressure.
                    let bg_threshold = overflow_thresholds.bg_pressure;
                    if snapshot.pressure.memory_pressure > bg_threshold {
                        m.metrics.cycles_high_pressure =
                            m.metrics.cycles_high_pressure.saturating_add(1);
                    } else {
                        m.metrics.cycles_high_pressure = 0;
                    }
                    // behavior_interactive_pid_count — how many PIDs learned dynamically.
                    m.metrics.behavior_interactive_pid_count = behavior_interactive_pids.len();
                    // rl_threshold_current — absolute threshold (bg_pressure + rl_adj).
                    m.metrics.rl_threshold_current =
                        bg_threshold + m.metrics.rl_adjustment_pp as f64 / 100.0;
                    // ── UCHS / Neurocognitive metrics (8 cognitive modules) ──────────
                    m.metrics.uchs_composite = cog_decision.uchs_composite;
                    m.metrics.uchs_grade = cognitive_state.health.grade.clone();
                    m.metrics.uchs_recovery_mode = cognitive_state.health.recovery_mode;
                    m.metrics.epistemic_uncertainty = cognitive_state.epistemic.composite;
                    m.metrics.epistemic_level = cognitive_state.epistemic.level_label().to_string();
                    m.metrics.meta_confidence = cognitive_state.meta_cognition.meta_confidence;
                    m.metrics.humble_mode = cog_decision.humble_mode;
                    m.metrics.adversarial_pass_rate =
                        cognitive_state.adversarial.lifetime_pass_rate() as f32;
                    m.metrics.adversarial_safety_alert = cog_decision.safety_alert;
                    m.metrics.cognitive_snr = cognitive_state.reward_bus.signal_to_noise();
                    m.metrics.self_eval_quality = cognitive_state.self_evaluator.evaluator_trust();
                    m.metrics.reptile_cached_workloads = cognitive_state.reptile.cached_workloads();
                    m.metrics.drift_early_warning =
                        lctx.outcome_tracker.drift_detector.early_warning();
                }

                // ── Periodic stage: GC and observability (% 100 / % 500 / % 7200 gates) ──
                // % 500 GC (experience compress, weight prune, skill GC) runs here.
                // % 100 persist and rule-induction remain inline above (need SharedState).
                // % 7200 hourly GC remains inline above (binary-local types).
                {
                    let mut pctx = PeriodicContext {
                        cycle_count,
                        current_pressure: snapshot.pressure.memory_pressure,
                        workload_mode: workload_mode.as_str(),
                        skills_path: std::path::Path::new(skills_path()),
                        hop_groups_path: std::path::Path::new(hop_groups_path()),
                        signal_intel_path: std::path::Path::new(signal_intelligence_path()),
                        learned_state_path: ls_path,
                        persist_generations,
                        last_restore_quality,
                        pending_trial_skill: pending_trial_skill.clone(),
                        lctx: &mut lctx,
                    };
                    let _periodic_result = run_periodic(&mut pctx);
                }

                // Push estado a suscriptores activos (menubar, etc.)
                socket_handler::broadcast_current_status(&state);

                // Analytics: record this cycle's metrics for trend tracking.
                {
                    let thermal_now = state
                        .hardware
                        .lock_recover()
                        .last_hw_snapshot
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
                    let cycles = state.metrics.lock_recover().metrics.cycles;
                    if cycles % 10 == 0 {
                        let persisted = state
                            .policy
                            .lock_recover()
                            .adaptive_governor
                            .user_profile
                            .to_persisted();
                        write_json(&state.user_profile_path, &persisted, Some(0o600));
                    }
                }

                let fast = state
                    .metrics
                    .lock_recover()
                    .fast_tick_until
                    .map(|t| Instant::now() < t)
                    .unwrap_or(false);
                last_cycle_end = Instant::now();
                lf_metrics.set_cycle_time_us(cycle_start.elapsed().as_micros() as u64);
                lf_metrics.commit();
                // Reactive: condvar.wait_timeout instead of thread::sleep.
                // Wakes immediately on reactor events; otherwise max 500ms (fast) or 2s (idle).
                // In dry-run mode, skip the condvar wait entirely — pure cycle throughput.
                // [Nygard 2018 §5] fast-path: remove production rate limiters from test hot-path.
                let wait_duration = if dry_run {
                    Duration::ZERO
                } else if fast {
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
            // On clean shutdown, clear the pending trial: the result can't be measured
            // reliably after a restart since system pressure state will differ.
            let frozen_snap_shutdown: FrozenStatePersisted = {
                let fg = state.frozen_state.lock_recover();
                FrozenStatePersisted {
                    frozen: fg
                        .iter()
                        .map(|(pid, e)| FrozenPidEntry {
                            pid: *pid,
                            since: e.frozen_at,
                            name: e.process_name.clone(),
                        })
                        .collect(),
                }
            };
            LearnedState::persist_improved(
                &signal_intel,
                &outcome_tracker,
                &specialist_accuracy,
                &skill_registry,
                &effectiveness_tracker,
                Some(overflow_guard.export_history()),
                Some(frozen_snap_shutdown),
                ls_path,
                persist_generations,
                last_restore_quality,
                None,
                Some(arousal_state.clone()),
                Some(&causal_graph),
                Some(energy_pid_tracker.baseline.clone()),
                Some(learnable_params.clone()),
            );

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
                        dry_run,
                    );
                    if outcomes.failures == 0 {
                        sysctl_governor.mark_reverted();
                    } else {
                        tracing::warn!(
                            failures = outcomes.failures,
                            "sysctl-governor: revert failures; persisted defaults retained for next startup"
                        );
                    }
                } else {
                    // No actions to revert — clean up persisted defaults anyway.
                    sysctl_governor.mark_reverted();
                }
            }

            // Chromium renderer cleanup: SIGCONT all renderers frozen by ChromiumManager.
            // These are separate from the main frozen_state (different source tracking).
            {
                let thawed = chromium_mgr.shutdown_cleanup();
                if !thawed.is_empty() {
                    tracing::info!(
                        count = thawed.len(),
                        "chromium: shutdown cleanup — thawed all frozen renderers"
                    );
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

            // Clean shutdown: remove crash sentinel so next startup knows this was graceful.
            remove_crash_sentinel();
            let _ = fs::remove_file(socket_path());
        }
    }

    Ok(())
}
