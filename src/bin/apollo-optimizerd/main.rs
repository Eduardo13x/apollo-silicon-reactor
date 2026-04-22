use std::collections::{HashMap, HashSet, VecDeque};
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
mod daemon_action_pipeline;
mod daemon_action_safety;
mod daemon_agent_actions;
mod daemon_behavior_pids;
mod daemon_cluster_actions;
mod daemon_stale_apps;
mod daemon_thermal_freeze;
mod daemon_paging_hints;
mod daemon_signal_tick;
mod daemon_skill_tick;
mod daemon_chromium_tick;
mod daemon_cognitive_tick;
mod daemon_cycle_tail;
mod daemon_feature_gates;
mod daemon_freeze_executor;
mod daemon_init;
mod daemon_neuro_tick;
mod daemon_pressure_aggregator;
mod daemon_process_collector;
mod daemon_reactor;
mod daemon_sensor_tick;
mod daemon_socket_handler;
mod daemon_turbo_manager;
mod daemon_wake_handler;
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
    audit_log, battery_pressure_boost, detect_prior_crash, frozen_state_path, governor_state_path,
    holt_winters_path, hop_groups_path, journal_path, kill_switch_path, learned_state_path,
    load_frozen_state, load_governor_state, load_wake_state, markov_path, merge_seed_into,
    metrics_path, overflow_history_path, parse_profile, pid_start_time, predictive_agent_path,
    remove_crash_sentinel, rl_threshold_path,
    signal_intelligence_path, skills_path, socket_path, spotlight_set_indexing,
    telemetry_output_dir, temporal_histograms_path, timeline_path, unfreeze_pids,
    unfreeze_pids_verified, wake_state_path, write_frozen_state, write_governor_state,
};
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
use apollo_optimizer::engine::learned_state::{
    LearnableParams, LearnedState, RestoreQualityMonitor,
};
use apollo_optimizer::engine::llm::{
    feedback_path_root, load_repo_config, pending_trial_path,
    policy_path_root, read_json, state_paths_root, suggestions_path_root, write_json,
    LearnedPolicy, LlmAdvisor, LlmConfig, LlmState,
};
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::lse_counters::LockFreeMetrics;
use apollo_optimizer::engine::mach_qos::{MachQoSManager, SchedulingTier};
use apollo_optimizer::engine::memory_analyzer::MemoryAnalyzer;
use apollo_optimizer::engine::memory_budget::{self, ProcessBudgetInput};
use apollo_optimizer::engine::network_optimizer::NetworkProfile;
use apollo_optimizer::engine::overflow_guard::{is_build_tool_name, OverflowGuard};
use apollo_optimizer::engine::pipeline::decision_stage::{DecisionStage, PolicyContext};
use apollo_optimizer::engine::pipeline::learning_context::LearningContext;
use apollo_optimizer::engine::power_management::detect_battery_status;
use apollo_optimizer::engine::predictive_agent::{
    AgentContext, Intervention, PredictiveAgent, SpecialistVote,
};
use apollo_optimizer::engine::proc_taskinfo;
use apollo_optimizer::engine::profile_governor::GovernorInput;
use apollo_optimizer::engine::safety::{critical_background_processes, enforce_limits_with_budget, is_protected_name};
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
use chrono::{Timelike, Utc};
use clap::{Parser, Subcommand};

// v0.9.0: canonical SharedState — all domain groups live in daemon_state.rs
use apollo_optimizer::engine::daemon_state::{
    HardwareState, LlmDomainState, MetricsState, PolicyState, ProcessState,
    ReactorStatus as DomainReactorStatus, SharedState, UsageDomainState, UsageTrackerState,
};

// FREEZE_TTL_SECS → daemon_helpers

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

// rotate_timeline → daemon_helpers

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
                always_on: None,
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

            // C5 fix (round-3): wrap the reactor thread so fast_tick_until is
            // forcibly cleared if it panics or returns.  Otherwise a dead
            // reactor would leave the main loop stuck in 500ms-tick mode for
            // up to REACTOR_FAST_TICK_SECS with stale signal data.
            let reactor_state = state.clone();
            thread::spawn(move || {
                // Drop-guard: runs whether the closure panics or returns.
                struct FastTickGuard {
                    state: SharedState,
                }
                impl Drop for FastTickGuard {
                    fn drop(&mut self) {
                        // Reactor exited unexpectedly (panic or return). Clear
                        // fast_tick so the main loop goes back to the normal
                        // 2s cadence instead of spinning on stale data.
                        self.state.metrics.lock_recover().fast_tick_until = None;
                    }
                }
                let _guard = FastTickGuard {
                    state: reactor_state.clone(),
                };
                let _ = daemon_reactor::run_reactor(reactor_state, &STOP_REQUESTED);
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
                tracing::warn!(
                    "apollo: prior session ended abnormally — cautious mode active for 50 cycles"
                );
            }
            let mut cautious_cycles_remaining: u32 = if prior_crash { 50 } else { 0 };

            // Startup glue: spawn control socket server + synchronously wait for
            // bind confirmation. On bind failure this exits(1) — prevents a second
            // headless instance racing on frozen_state.json. See
            // `daemon_socket_handler::spawn_control_socket` for the full rationale.
            daemon_socket_handler::spawn_control_socket(&state);

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
                mut unfreeze_decay,
                mut swap_reclaim,
            } = daemon_init::DaemonSubsystems::new();
            let mut nested_learner = apollo_optimizer::engine::nested_learner::NestedLearner::new();
            let mut focus_markov = FocusMarkov::new(PathBuf::from(markov_path()));
            // TelemetryLogger: ring-buffer collection for time-series training data.
            // [Welch 1967, Tuli et al. 2022] — event-triggered dumps capture pre-anomaly context.
            let mut telemetry_logger =
                apollo_optimizer::engine::telemetry_logger::TelemetryLogger::new(PathBuf::from(
                    telemetry_output_dir(),
                ));
            // Warm-start: reload recent history so anomaly detector skips cold-start.
            // [Gray & Reuter 1992] §11.3 — restart protocols restore in-flight state.
            telemetry_logger.warm_start_from_dir(3);
            // StabilityOracle: aggregate jank + zombie + swap-spike into RL reward.
            // [Schulman et al. 2017] PPO per-cycle reward; [Nygard 2018] cascading instability.
            let mut stability_oracle =
                apollo_optimizer::engine::stability_oracle::StabilityOracle::new();
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
            // Per-cycle update + GC lives in `daemon_cognitive_tick::update_habituation_state`;
            // only the (pid → bucket-state) map is owned here for cross-cycle carry.
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
            // SleepNotifier: IOKit pre-sleep callback — fires kIOMessageSystemWillSleep
            // ~30s before the kernel suspends, giving us time to release all SIGSTOP'd
            // processes so macOS can compress/Jetsam them normally during hibernation.
            // Without this, frozen processes block Jetsam's target set, leading to
            // more aggressive kills of other processes (widgets, extensions).
            let sleep_notifier = apollo_optimizer::engine::sleep_notifier::SleepNotifier::new();
            tracing::info!(
                available = sleep_notifier.available,
                "sleep_notifier: IOKit pre-sleep hook {}",
                if sleep_notifier.available {
                    "registered"
                } else {
                    "unavailable (running without IOKit access)"
                }
            );
            let mut overflow_guard = OverflowGuard::load_or_default(
                std::path::Path::new(overflow_history_path()),
                Some(std::path::Path::new(rl_threshold_path())),
            );
            // Predictive agent: LinUCB contextual bandit for proactive interventions.
            let mut predictive_agent =
                PredictiveAgent::load_or_default(std::path::Path::new(predictive_agent_path()));
            // Super Learner cross-cycle feedback state: prev pressure + actual
            // firing signals for each specialist.  Owned by the main loop so the
            // extracted `daemon_cognitive_tick::apply_specialist_voting` can grade
            // next cycle's accuracy against what *actually* fired.
            let mut specialist_feedback =
                daemon_cognitive_tick::SpecialistFeedbackState::default();

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
                // BUG-01: WAL fallback — if LearnedState didn't carry a pending trial
                // (e.g., daemon crashed before periodic persist), recover from WAL file.
                if restored_trial_skill.is_none() {
                    if let Ok(data) =
                        apollo_optimizer::engine::types::HardPath::read_to_string_limited(
                            &pending_trial_path(is_root),
                            512,
                        )
                    {
                        restored_trial_skill = serde_json::from_str::<Option<(String, f64)>>(&data)
                            .ok()
                            .flatten();
                    }
                }
                // Restore learned τ for unfreeze-decay BEFORE apply() consumes
                // `learned`.  The field is `Option<HashMap<…>>`; absent on old
                // files or cold start → decay model stays at defaults.
                if let Some(tau_map) = learned.unfreeze_decay_tau.clone() {
                    let count = tau_map.len();
                    unfreeze_decay.restore(tau_map);
                    tracing::info!(
                        target: "apollo.unfreeze_decay",
                        learned_apps = count,
                        "restored unfreeze-decay τ estimates from learned_state"
                    );
                }
                // Warm-start the neuromodulator from persisted signal levels.
                // Without this, DA/ACh/NA/5-HT cold-start at 0.5 neutral on every
                // restart, discarding accumulated reward-prediction history.
                // [Schultz 1997] — continuity of prediction-error signals is critical.
                if let Some(ns) = learned.neuro_state.clone() {
                    neuromod.restore(ns);
                    tracing::info!(
                        target: "apollo.neuromodulator",
                        "warm-started neuromodulator from learned_state"
                    );
                }
                // apply() restores skills from learned_state.json if present,
                // overwriting the legacy optimization_skills.json load above.
                // If skill_registry field is absent (old file), the legacy load is kept.
                // Returns (overflow_history, frozen_pids, arousal_state) for
                // components that need caller-side wiring.
                let (
                    ls_overflow_history,
                    ls_frozen_pids,
                    ls_arousal,
                    ls_baselines,
                    restored_lp,
                    restored_nl,
                ) = learned.apply(
                    &mut signal_intel,
                    &mut outcome_tracker,
                    &mut specialist_accuracy,
                    &mut skill_registry,
                    &mut effectiveness_tracker,
                    Some(&mut causal_graph),
                );
                learnable_params = restored_lp;
                if let Some(nl) = restored_nl {
                    nested_learner = nl;
                }
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
                                    start_sec: 0,
                                    original_jetsam_priority: None,
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

            // G12 fallback: previous cycle's entropy anomaly used as proxy when
            // ioreport_amc_bandwidth_pct is dead (M1 without IOReport entitlement).
            // 1-cycle lag is acceptable — DRAM BP is integrative, not cycle-precise.
            let mut prev_entropy_anomaly: f64 = 0.0;

            // ── Warn-limit tracking (non-fatal targeted memory pressure) ──────
            // PIDs that have an active warn_limit set; cleared after 3 cycles.
            let mut warn_limit_pids: HashMap<u32, u8> = HashMap::new();

            // ── Feature 1: LLM Inference Mode ────────────────────────────────
            // Auto-detect ollama / llama.cpp / MLX / LM Studio inference.
            // When active: +20pp pressure boost, Spotlight off, App Nap non-essential.
            let mut llm_detector =
                apollo_optimizer::engine::llm_inference_mode::LlmInferenceDetector::new();
            let mut llm_spotlight_disabled = false;
            // Read last saved metrics to decide startup Spotlight state.
            // Blind re-enable caused oscillation: mds spins up → pressure
            // rises back above 0.75 → disable → loop [Nygard 2018 §4].
            // Use the same calm-threshold as the re-enable gate (p<0.35 + swap<1GB).
            //
            // Fail-safe defaults: when any field is missing or the file is
            // corrupted/absent we fall back to values that make `startup_calm`
            // FALSE (pressure=1.0, swap=99.0). The previous default of
            // swap=0.0 was fail-open: a file with bad pressure parsed as calm
            // when swap was actually multi-GB. Defaulting swap high matches
            // the pressure default so any read failure pauses Spotlight.
            let (startup_pressure, startup_swap_gb): (f64, f64) =
                std::fs::read_to_string(
                    apollo_optimizer::engine::daemon_helpers::metrics_path(),
                )
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .map(|v| {
                    let p = v["memory_pressure"].as_f64().unwrap_or(1.0);
                    let swap = v["swap_used_bytes"].as_f64().unwrap_or(99.0 * 1024.0 * 1024.0 * 1024.0)
                        / (1024.0 * 1024.0 * 1024.0);
                    (p, swap)
                })
                .unwrap_or((1.0, 99.0));
            let startup_calm = startup_pressure < 0.35 && startup_swap_gb < 1.0;
            let mut spotlight_paused: bool = is_root && !startup_calm;
            let mut spotlight_paused_at: Option<Instant> =
                if spotlight_paused { Some(Instant::now()) } else { None };
            if is_root {
                spotlight_set_indexing(startup_calm);
            }

            // Consecutive cycles where swap_delta > 1MB/s. Fed to RL meta-gate
            // to veto Raise1pp during sustained swap growth (see rl_threshold.rs).
            let mut swap_growth_streak: u32 = 0;

            // Rate-limit `purge(8)` invocations to at most once per 10 minutes
            // under severe swap pressure. `purge` forces the kernel to drain
            // inactive pages — effective but expensive (blocks for ~2s on 8GB).
            let mut last_purge_at: Option<Instant> = None;
            // D-term for overflow threshold PID: pressure velocity from previous cycle.
            // One-cycle lag is acceptable; 0.0 default is conservative (no D-offset on cold start).
            let mut last_pressure_velocity: f64 = 0.0;

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
            // Staggered wake unfreeze queue: instead of SIGCONT-ing all frozen PIDs
            // at once on display wake (which decompresses 1-3GB in <500ms on 8GB M1),
            // spread unfreezes across cycles — 5 PIDs per cycle (~2s apart).
            const WAKE_UNFREEZE_BATCH: usize = 5;
            let mut wake_unfreeze_queue: VecDeque<u32> = VecDeque::new();
            // PIDs SIGCONT'd via the staggered wake path this cycle.
            // Drained by the unfreeze_decay ODE wiring below so τ learning
            // starts at the correct T0 (actual SIGCONT, not decision time).
            let mut wake_thaw_pids: Vec<u32> = Vec::new();
            // Pending trial skill: (name, pressure_before). Recorded next cycle.
            // Restored from LearnedState so a trial started before a crash is still evaluated.
            let mut pending_trial_skill: Option<(String, f64)> = restored_trial_skill;
            // Last specialist votes + chosen intervention for disagreement feedback.
            // Stored when had_disagreement is true; consumed by learning_tick next cycle.
            // The initial `None` is overwritten by every path in the cycle body before the
            // read at the learning_tick call site — mark the init assignment as allowed.
            #[allow(unused_assignments)]
            let mut last_specialist_votes: Option<(Vec<SpecialistVote>, Intervention)> = None;
            // System log ingester: polls macOS unified logs for OOM/crash events (Phase 5).
            // Runs in background thread to avoid blocking the daemon hot path with
            // `log show` subprocess latency (100-300ms when it fires).
            let mut log_ingester =
                apollo_optimizer::engine::system_log_ingester::SystemLogIngester::new();
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
            // Teacher consolidation: compiles Gemma 4 suggestions into S1
            // pattern_weights + NARS beliefs via dopamine/acetylcholine modulation.
            // [McGaugh 2004, Yerkes-Dodson 1908, Kahneman 2011]
            let mut teacher_consolidator =
                apollo_optimizer::engine::teacher_consolidation::TeacherConsolidator::new();
            // Tracks the last resolved outcome's applied_at so we only
            // consolidate each outcome exactly once.
            let mut last_consolidated_at: Option<chrono::DateTime<chrono::Utc>> = None;
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
            // Spotlight pause state moved earlier (see startup defensive restore).
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
            // LlmConfig live-reload: polls /etc/apollo-optimizer/config.toml every 100
            // cycles for mtime changes, applies whitelisted diffs only.
            // Guard: only reload when pending_trial_skill.is_none() to prevent
            // corrupting BUG-01 WAL outcome attribution during an active experiment.
            // [Gray & Reuter 1992 §11 — stable params during causal observation window]
            let mut llm_cfg_reloader =
                apollo_optimizer::engine::config_reloader::LlmConfigReloader::new(
                    PathBuf::from("/etc/apollo-optimizer/config.toml"),
                    pending_trial_path(is_root),
                );

            loop {
                // Check both: Arc flag (set by ctrlc) and static flag (set by SIGTERM handler).
                if state.stop.load(Ordering::Acquire) || STOP_REQUESTED.load(Ordering::Acquire) {
                    state.stop.store(true, Ordering::Release);
                    println!("Daemon stopping: stop signal received");
                    break;
                }

                wake_thaw_pids.clear();
                cycle_count += 1;
                lf_metrics.inc_cycles();

                // ── Feature 4: Post-Wake Suppression ─────────────────────────
                // If more than 30s passed since the last cycle, the system was
                // sleeping. Apply 60s App-Nap window to all non-essential
                // backgrounds so the foreground app restores its state first.
                // (last_cycle_instant is NOT reset here — must span the full
                // inter-cycle interval so cycle_dt_secs reflects real wall-clock.)
                let in_wake_suppression = daemon_feature_gates::apply_post_wake_suppression(
                    &state,
                    last_cycle_instant,
                    &mut wake_suppression_until,
                );

                // Enforce minimum inter-cycle delay to prevent event-storm CPU burn.
                // In dry-run the condvar wait is already 100ms; skip the additional floor.
                // BUG #3 fix: also bypass the 300ms floor during fast-tick mode so the
                // daemon can respond to kqueue Critical / hw_predictor events at full speed.
                let is_fast_tick_raw = state
                    .metrics
                    .lock_recover()
                    .fast_tick_until
                    .map(|t| Instant::now() < t)
                    .unwrap_or(false);
                // C1/C4 fix (round-3): disable fast-tick when running on battery
                // below 20%.  Continuous 500ms cycles drain ~5%/2min — unacceptable
                // near empty.  Also ensures min_inter_cycle_ms is at least 1000 in
                // that case so event-storm CPU burn can't chew battery either.
                let battery_low = !power_mgr.battery_status.is_charging
                    && power_mgr.battery_status.percentage < 20;
                let is_fast_tick = is_fast_tick_raw && !battery_low;
                let min_inter_cycle_ms = if dry_run || is_fast_tick {
                    0
                } else if battery_low {
                    1000
                } else {
                    300
                };
                let since_last = last_cycle_end.elapsed();
                if min_inter_cycle_ms > 0 && since_last < Duration::from_millis(min_inter_cycle_ms)
                {
                    thread::sleep(Duration::from_millis(min_inter_cycle_ms) - since_last);
                }

                // C8 fix (round-3): halve the normal action-queue budget when
                // battery < 30% and not charging — same work volume on battery
                // as on AC was wasting energy with no latency benefit. Urgent
                // unfreezes still bypass the cap by design.
                {
                    let battery_conservation = !power_mgr.battery_status.is_charging
                        && power_mgr.battery_status.percentage < 30;
                    let base: usize = 20;
                    let target = if battery_conservation { base / 2 } else { base };
                    if action_queue.max_per_cycle() != target {
                        action_queue.set_max_per_cycle(target);
                    }
                    // G12 — DRAM Bandwidth Backpressure: if memory bandwidth is saturated
                    // (amc_bandwidth_pct > 80), halve max_per_cycle to prevent additional
                    // process activations from amplifying DRAM congestion.
                    // [Hellerstein 2004 §9 — backpressure gates downstream work under saturation]
                    //
                    // M1 fallback: when amc_bandwidth_pct == 0.0 (no IOReport
                    // entitlement), use prev-cycle entropy_anomaly > 2.0 as proxy.
                    // Raw entropy_anomaly can exceed 1.0 on real contention events —
                    // ≥2.0 is the same threshold used by signal_intel for true anomalies.
                    // [Heil 2021 PACT §3] — workload entropy tracks LLC pressure.
                    let dram_bw_pct = last_ioreport
                        .as_ref()
                        .map(|ir| ir.amc_bandwidth_pct)
                        .unwrap_or(0.0);
                    let dram_bp_trigger = dram_bw_pct > 80.0
                        || (dram_bw_pct == 0.0 && prev_entropy_anomaly > 2.0);
                    if dram_bp_trigger {
                        let capped = (action_queue.max_per_cycle() / 2).max(1);
                        action_queue.set_max_per_cycle(capped);
                    }
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
                            metrics.metrics.swap_total_bytes = cached.swap_total_bytes;
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
                        const PREFIX: &[u8] =
                            b"{\"type\":\"StatusPush\",\"payload\":{\"metrics\":{\"cycles\":";
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

                // ── Staggered wake unfreeze: drain batch per cycle ──────────
                // Instead of SIGCONT-ing 50 PIDs at once, spread across cycles
                // to avoid decompressing 1-3GB in one shot on 8GB M1.
                // [Nygard 2018] Dynamic bulkhead: shrink batch when memory grows
                // fast (dM/dt high) — prevents RSS spike from multiple thaws.
                // pressure_velocity > 0 = rising; scale 5→1 over [0.0, 0.2].
                if !wake_unfreeze_queue.is_empty() {
                    let wake_batch = {
                        // G21 — Thermal Bulkhead: serious/critical thermal → single-process
                        // thaw prevents CPU surge from simultaneous reactivation.
                        // [Nygard 2018 §4.3 — bulkhead limits blast radius under resource stress]
                        let thermal_str = state.metrics.lock_recover().thermal_level_real.clone();
                        if thermal_str == "serious" || thermal_str == "critical" {
                            1_usize
                        } else {
                            // dM/dt proxy: swap_delta_bps > 0 = swap growing.
                            // 50 MB/s growth → rate_factor = 1.0 → batch = 1.
                            let rate_factor = (pressure_collector
                                .latest()
                                .swap_delta_bps
                                / (50.0 * 1024.0 * 1024.0))
                                .clamp(0.0, 1.0);
                            (WAKE_UNFREEZE_BATCH as f64 * (1.0 - rate_factor * 0.8))
                                .max(1.0)
                                .round() as usize
                        }
                    };
                    let batch: Vec<u32> = wake_unfreeze_queue
                        .drain(..wake_unfreeze_queue.len().min(wake_batch))
                        .collect();
                    // A-B-A defense: lock frozen_guard first to read identity
                    // (start_sec) before signalling. Crash before SIGCONT leaves
                    // PIDs in frozen_state for recovery on restart (WAL semantics).
                    // [Saltzer & Kaashoek 2009] §3.3 Complete Mediation.
                    {
                        let mut frozen_guard = state.frozen_state.lock_recover();
                        let entries: std::collections::HashMap<u32, FrozenEntry> = batch
                            .iter()
                            .filter_map(|&pid| frozen_guard.get(&pid).map(|e| (pid, e.clone())))
                            .collect();
                        unfreeze_pids_verified(&entries);
                        for pid in &batch {
                            frozen_guard.remove(pid);
                        }
                        write_frozen_state(&frozen_state_path, &frozen_guard);
                    }
                    // Restore Mach QoS from Background (E-cores) → Normal so
                    // processes resume on P-cores. Wake unfreeze is the highest-
                    // urgency thaw path (user just returned to desktop), so P-core
                    // routing is critical for perceived responsiveness.
                    {
                        let mut qos = state.mach_qos.lock_recover();
                        for pid in &batch {
                            let _ = qos.set_tier(*pid, apollo_optimizer::engine::mach_qos::SchedulingTier::Normal);
                        }
                    }
                    // Record actual-SIGCONT T0 for unfreeze_decay ODE τ learning.
                    wake_thaw_pids.extend_from_slice(&batch);
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
                // Sleep/wake detection + post-wake grace window.
                // Extracted to daemon_wake_handler::run_wake_tick().
                // [Nygard 2018] bulkhead: staggered SIGCONT avoids 1-3GB decompression spike.
                let grace_active = daemon_wake_handler::run_wake_tick(
                    &state,
                    &mut signal_intel,
                    &mut outcome_tracker,
                    &mut wake_unfreeze_queue,
                    &mut display_turbo,
                    &wake_state_path,
                );

                // Display-Off Turbo: Android Doze-like power management.
                // Extracted to daemon_turbo_manager::run_turbo_tick().
                // [Nygard 2018] bulkhead: bound blast radius of display state transitions.
                daemon_turbo_manager::run_turbo_tick(
                    &mut display_turbo,
                    &state,
                    &fg_detector,
                    &collector,
                    &pressure_collector,
                    &frozen_state_path,
                    &mut stability_oracle,
                    power_mgr.is_on_battery(),
                );

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
                            .map(|fa| {
                                fa.to_ascii_lowercase()
                                    .contains(&predicted.to_ascii_lowercase())
                            })
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
                        // Only pre-thaw within [-5s, +10s] window.  Deeply negative
                        // values = stale prediction → skip to avoid freeze/thaw thrashing.
                        if time_to_switch > -5.0 && time_to_switch < 10.0 {
                            use apollo_optimizer::engine::freeze_intelligence::FreezeIntelligence;
                            let hint_categories = FreezeIntelligence::pre_thaw_hint(&pred.app_name);
                            // Single lock scope: collect + act atomically to avoid
                            // TOCTOU where another thread removes PIDs between
                            // candidate collection and SIGCONT.
                            let mut frozen_guard = state.frozen_state.lock_recover();
                            let candidates: Vec<(u32, String)> = frozen_guard
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
                                .collect();
                            if !candidates.is_empty() {
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
                // Extracted to daemon_process_collector::build_process_tree().
                let process_tree = daemon_process_collector::build_process_tree(&collector);

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

                // Pre-sleep unfreeze: release every SIGSTOP'd PID before kernel suspends.
                // Extracted to daemon_process_collector::run_pre_sleep_unfreeze().
                // [Saltzer & Kaashoek 2009] §3.3 A-B-A defense via unfreeze_pids_verified.
                daemon_process_collector::run_pre_sleep_unfreeze(
                    &state,
                    &frozen_state_path,
                    &mut display_turbo,
                    &sleep_notifier,
                );

                // Ghost-PID reconciliation — evict dead PIDs from frozen_state +
                // turbo set; GC mach_qos HashMaps every 60 cycles.
                let live_pids: HashSet<u32> =
                    proc_snaps.iter().map(|p| p.pid).collect();
                daemon_process_collector::run_ghost_pid_reconciliation(
                    &state,
                    &live_pids,
                    &frozen_state_path,
                    &mut display_turbo,
                    cycle_count,
                );

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
                                is_build_tool: is_build_tool_name(&s.name),
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

                // Thermal pre-throttle freeze/unfreeze.
                // Extracted to daemon_thermal_freeze::run_thermal_freeze (Wave 20).
                // M1 Air has no fan — acting 5-10°C ahead of the hardware ceiling.
                daemon_thermal_freeze::run_thermal_freeze(
                    &thermal_action,
                    &state,
                    &collector,
                    foreground_pid,
                    snapshot.pressure.memory_pressure,
                    std::path::Path::new(&frozen_state_path),
                );

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
                // thermal_predicted_* escape to metrics (predictive thermal observability).
                let mut gpu_thermal_throttled = false;
                let mut thermal_predicted_throttle: u8 = 0;
                let mut thermal_seconds_to_throttle: Option<i32> = None;
                let mut thermal_trend_predicted = String::new();
                {
                    if let Some(hw) = &cycle_hw_snap {
                        let cpu_t = hw.temps.p_cluster_celsius.unwrap_or(0.0);
                        let gpu_t = hw.temps.gpu_celsius.unwrap_or(cpu_t);
                        let thermal_state = thermal_mgr.update(cpu_t, gpu_t, 0.0, 0, jitter_us);
                        thermal_predicted_throttle = thermal_state.predicted_throttle_level;
                        thermal_seconds_to_throttle = thermal_state.seconds_to_throttle;
                        thermal_trend_predicted = format!("{:?}", thermal_state.thermal_trend);

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
                //
                // Arousal gate [Kahneman 2011] S1/S2 dual-path: Gemma (System 2, ~124s latency)
                // must not be consulted during a memory crisis — by the time it responds, the
                // crisis is over or the OOM already fired. Suppress calls when arousal_state is
                // high (system under stress) so the fast reactive path (System 1) operates
                // uncontested. Gemma runs during calm periods and compiles insights into
                // SkillRegistry / pattern_weights for future fast-path use.
                let llm_calm_gate = arousal_state.level <= 0.70;
                if llm_calm_gate {
                    llm_daemon::llm_reactive_tick(
                        &state,
                        &mut llm_advisor,
                        &snapshot,
                        &mut llm_counters,
                        lctx.outcome_tracker.heuristic_is_struggling(),
                    );
                }

                // ── Teacher consolidation: S2 → S1 memory transfer ────────
                // When llm_reactive_tick resolves a pending outcome, compile
                // Gemma's suggestion into pattern_weights + NARS beliefs using
                // dopamine/acetylcholine modulation. Each outcome consolidated
                // exactly once (tracked by applied_at timestamp).
                {
                    let (new_outcome, matching_suggestion) = {
                        let guard = state.llm.lock_recover();
                        let outcome = guard.llm_state.last_suggestion_outcome.clone();
                        let suggestion = guard.llm_state.last_suggestion.clone();
                        (outcome, suggestion)
                    };
                    if let (Some(outcome), Some(suggestion)) = (new_outcome, matching_suggestion) {
                        if last_consolidated_at != Some(outcome.applied_at) {
                            let natural_drift = lctx.outcome_tracker.natural_drift();
                            let report = teacher_consolidator.consolidate(
                                &outcome,
                                &suggestion,
                                natural_drift,
                                &mut lctx.outcome_tracker.weights,
                                &mut lctx.outcome_tracker.drift_detector,
                                &mut arousal_state,
                            );
                            last_consolidated_at = Some(outcome.applied_at);
                            // Journal: record the consolidation event for observability.
                            if !matches!(report.verdict, "BELOW_DEADBAND") {
                                let mut mx = state.metrics.lock_recover();
                                mx.metrics.teacher_consolidations += 1;
                                if report.verdict == "IMPROVED" {
                                    mx.metrics.teacher_improvements += 1;
                                }
                            }
                        }
                    }
                }

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
                                            &learnable_params.rl_pressure_bands,
                                            &learnable_params.rl_compressor_bands,
                                        );
                                        // Teach hazard model only on real OOM indicators:
                                        // swap must be GROWING (delta > 512KB/s) and present
                                        // (> 10% used). kqueue Critical fires at ~80% pressure
                                        // which is normal; training on it saturates base_rate.
                                        let sr = if snapshot.pressure.swap_total_bytes > 0 {
                                            snapshot.pressure.swap_used_bytes as f64
                                                / snapshot.pressure.swap_total_bytes as f64
                                        } else {
                                            0.0
                                        };
                                        let swap_growing =
                                            snapshot.pressure.swap_delta_bytes_per_sec > 524_288.0;
                                        if sr > 0.10 && swap_growing {
                                            lctx.signal_intel.record_overflow(
                                                snapshot.pressure.memory_pressure,
                                                sr,
                                                snapshot.pressure.memory_pressure,
                                            );
                                        }
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
                // ── Per-cycle sensor telemetry pass (Strangler Fig extraction) ──
                // IOReport delta, SMC direct, KPC IPC, Rosetta poll, per-process
                // energy ranking + hint maps, syscall-aware JIT detection,
                // IOPMrootDomain thermal, and upstream boost factors (mem-bw,
                // SMC thermal, battery overheat). No lock acquisition, never
                // blocks. See `daemon_sensor_tick` module. [Fowler 2004]
                let daemon_sensor_tick::SensorTickOutput {
                    kpc_snap,
                    energy_pid_results,
                    ipc_hints,
                    wakeup_hints,
                    footprint_hints,
                    io_burst_hints,
                    anomaly_hints,
                    anomaly_thresh,
                    jit_protected_pids,
                    iopm_snap,
                    mem_bw_boost,
                    smc_thermal_boost,
                    battery_overheat_boost,
                } = daemon_sensor_tick::run_sensor_tick(
                    &snapshot,
                    cycle_count,
                    cycle_dt_secs,
                    &mut ioreport,
                    &mut last_ioreport,
                    &mut last_ioreport_sample,
                    &smc_direct,
                    &mut last_smc,
                    &mut kpc_reader,
                    &mut rosetta_monitor,
                    &mut energy_pid_tracker,
                    &mut syscall_classifier,
                );

                // ── Feature 1: LLM Inference Mode ─────────────────────────────
                // Detect ollama/llama.cpp/MLX and boost pressure gates aggressively.
                let daemon_feature_gates::LlmInferenceOutcome {
                    llm_boost,
                    llm_active,
                } = daemon_feature_gates::run_llm_inference_mode_tick(
                    &snapshot,
                    &mut llm_detector,
                    &mut llm_spotlight_disabled,
                    is_root,
                );

                // ── Feature 3: RT Boost for Foreground ────────────────────────
                // Apply THREAD_TIME_CONSTRAINT_POLICY to foreground UI thread.
                // Skipped during thermal Phase3+ (P-core pinning would defeat cooling).
                daemon_feature_gates::apply_rt_boost_foreground(
                    &state,
                    &thermal_action,
                    foreground_pid,
                    &mut rt_boosted_pid,
                );

                // ── Effective pressure aggregation (Strangler Fig) ───────────
                // Charging thermal stress + B0TE aggressiveness + 9-boost
                // additive sum (capped at +0.30) + cautious post-crash
                // subtract. Pure transform — see `daemon_pressure_aggregator`
                // for invariants. `audit_log` side effect stays here so the
                // helper remains stateless.
                let pressure_aggregation = daemon_pressure_aggregator::aggregate_cycle_pressure(
                    snapshot.pressure.memory_pressure,
                    hw_boost,
                    batt_boost,
                    thermal_pressure_boost,
                    llm_boost,
                    mem_bw_boost,
                    smc_thermal_boost,
                    battery_overheat_boost,
                    last_smc.as_ref(),
                    last_ioreport.as_ref(),
                    cycle_hw_snap.as_ref(),
                    cautious_cycles_remaining,
                );
                let pressure_ram = pressure_aggregation.effective_pressure;
                let pressure_components = pressure_aggregation.components;
                snapshot.pressure.memory_pressure = pressure_ram;
                cautious_cycles_remaining = pressure_aggregation.cautious.remaining;
                if pressure_aggregation.cautious.ended {
                    audit_log(&serde_json::json!({
                        "event": "cautious_mode_ended",
                        "message": "returning to normal thresholds after post-crash caution period",
                    }));
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
                    metrics.metrics.swap_total_bytes = snapshot.pressure.swap_total_bytes;
                    metrics.metrics.swap_delta_bps = snapshot.pressure.swap_delta_bytes_per_sec;
                    metrics.metrics.memory_pressure = snapshot.pressure.memory_pressure;
                    metrics.metrics.thermal_level = snapshot.pressure.thermal_level.clone();
                    // Expose pressure boost breakdown so the dashboard shows WHY effective
                    // pressure exceeds raw memory_pressure (hardware/thermal/battery factors).
                    metrics.metrics.pressure_total_boost = pressure_components.total_boost();
                    metrics.metrics.pressure_dominant_factor =
                        pressure_components.dominant_factor().to_string();
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
                                // [Saltzer & Kaashoek 2009] single truth point.
                                let protected = |name: &str| {
                                    is_protected_name(name)
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
                        // IOPMrootDomain CurrentPowerSource key fails on macOS 26+.
                        // Fall back to power_mgr (IOPSCopyPowerSourcesInfo) which works.
                        metrics.metrics.iopm_power_source =
                            if iopm.power_source == apollo_optimizer::engine::thermal_iokit::PowerSource::Unknown {
                                if power_mgr.battery_status.is_charging {
                                    "AC".to_string()
                                } else if power_mgr.battery_status.percentage > 0 {
                                    "Battery".to_string()
                                } else {
                                    format!("{:?}", iopm.power_source)
                                }
                            } else {
                                format!("{:?}", iopm.power_source)
                            };
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
                        .filter(|n| is_build_tool_name(n))
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

                // Thresholds adaptativos: workload-aware + D-term (pressure velocity).
                // [Hellerstein 2004] §9 PID operating-regime control.
                let mut overflow_thresholds = lctx
                    .overflow_guard
                    .thresholds_with_d_term(workload_mode, last_pressure_velocity);

                // ── Signal intelligence: Kalman + CUSUM + Entropy + Hazard + LV + MPC ──
                // Extracted to daemon_signal_tick::run_signal_tick (Wave 14).
                // [Fowler 2004] Strangler Fig — pure move, no semantic change.
                // NLM warning: cycle_dt_secs passed as parameter (never recalculate
                // here) — avoids mid-loop reset bug dac6de9 that corrupted ODE models.
                let daemon_signal_tick::SignalTickOutput {
                    signal_digest,
                    last_pressure_velocity: new_lpv,
                    entropy_anomaly: new_entropy,
                } = daemon_signal_tick::run_signal_tick(
                    lctx.signal_intel,
                    &snapshot,
                    cycle_dt_secs,
                    power_mgr.battery_status.percentage,
                    power_mgr.battery_status.is_charging,
                    thermal_emergency,
                    cycle_hw_snap.as_ref().and_then(|h| h.power.package_watts),
                    hour_of_day,
                    &state,
                    &mut darwin_anomaly,
                    &mut telemetry_logger,
                    &fluidity_state,
                    cycle_count as u64,
                );
                last_pressure_velocity = new_lpv;
                prev_entropy_anomaly = new_entropy;

                // Swap Reclaim ODE — feed vm_rate from background collector.
                // Produces SaturationForecast used by the freeze gate below.
                // [Denning 1968; Zhao et al. 2009 WKdm rate model]
                let reclaim_forecast = {
                    use apollo_optimizer::engine::swap_reclaim::VmFlowSample;
                    let pd = pressure_collector.latest();
                    let flow = VmFlowSample {
                        compressions_per_sec: pd.vm_rate.compressions_per_sec,
                        decompressions_per_sec: pd.vm_rate.decompressions_per_sec,
                        purges_per_sec: pd.vm_rate.purges_per_sec,
                        swapouts_per_sec: pd.vm_rate.swapouts_per_sec,
                        swap_used_bytes: pd.swap_used_bytes,
                        swap_total_bytes: pd.swap_total_bytes,
                    };
                    swap_reclaim.update(&flow)
                };

                // ODE swap urgency — hoisted for use in Neuromodulator AND LinUCB.
                // Normalization owned by TsatUrgency [CyberPhysicalSignal trait].
                let ode_t_sat_urgency = {
                    use apollo_optimizer::engine::swap_reclaim::{CyberPhysicalSignal, TsatUrgency};
                    TsatUrgency(reclaim_forecast.t_sat_sec).normalized()
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

                // G10 — AMX Proactive Steering: when AMX is active AND LLM inference is
                // running, boost reactor_weight to pre-position before ML memory pressure.
                // AMX matrix-multiply pipelines fill in 100-200ms bursts; without this boost
                // the daemon reacts after pressure already spikes.
                // [Hellerstein 2004 §9 — derivative control precedes integrator saturation]
                if amx_available && llm_active {
                    reactor_weight = (reactor_weight + 0.15).min(1.0);
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
                    // ODE physics features for LinUCB context (slots 12-13).
                    // [Hellerstein 2004] — derivative control closes the epistemic loop.
                    // Normalization owned by NetRateNorm [CyberPhysicalSignal trait].
                    let ode_net_rate_norm = {
                        use apollo_optimizer::engine::swap_reclaim::{CyberPhysicalSignal, NetRateNorm};
                        NetRateNorm(reclaim_forecast.net_rate_bps).normalized()
                    };

                    // KalmanMV8: fuse all 8 signals after ODE outputs are available.
                    // [Welch & Bishop 2006] H=I cross-covariance propagation.
                    {
                        use apollo_optimizer::engine::swap_reclaim::NET_RATE_CEILING_BPS;
                        let swap_raw_norm = (snapshot.pressure.swap_delta_bytes_per_sec
                            / NET_RATE_CEILING_BPS)
                            .clamp(0.0, 1.0);
                        let cpu_mean = {
                            let procs = &snapshot.top_processes;
                            if procs.is_empty() {
                                0.0
                            } else {
                                (procs.iter().map(|p| p.cpu_usage as f64).sum::<f64>()
                                    / procs.len() as f64
                                    / 100.0)
                                    .clamp(0.0, 1.0)
                            }
                        };
                        let thermal_f64 = match snapshot.pressure.thermal_level.as_str() {
                            "light" => 0.33,
                            "serious" => 0.66,
                            "critical" => 1.0,
                            _ => 0.0,
                        };
                        let z_mv: [f64; 8] = [
                            snapshot.pressure.memory_pressure, // [0] pressure
                            signal_digest.pressure_velocity,   // [1] velocity (1D KF)
                            swap_raw_norm,                     // [2] swap_norm
                            snapshot.pressure.memory_pressure, // [3] compressor proxy
                            ode_net_rate_norm,                 // [4] ode_net_rate
                            ode_t_sat_urgency,                 // [5] ode_t_sat
                            cpu_mean,                          // [6] cpu_saturation
                            thermal_f64,                       // [7] thermal_stress
                        ];
                        lctx.signal_intel.tick_mv(&z_mv, cycle_dt_secs);
                    }

                    // LinUCB slot 0: blend 1D→MV8 pressure over 200 cycles.
                    // Prevents "feature shock" from abrupt context vector shift.
                    // α=0 for first 200 cycles → pure 1D; α=1 after → pure MV8.
                    // [NotebookLM KalmanMV8 wiring spec; Welch & Bishop 2006]
                    let agent_pressure = {
                        let alpha = lctx.signal_intel.kf_mv_blend_alpha();
                        (1.0 - alpha) * signal_digest.pressure_smooth
                            + alpha * lctx.signal_intel.kf_mv_pressure()
                    };
                    let agent_ctx = AgentContext::build(
                        agent_pressure,
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
                        ode_t_sat_urgency,
                        ode_net_rate_norm,
                    );
                    let (linucb_choice, linucb_confidence) = lctx
                        .predictive_agent
                        .select_action_with_confidence(&agent_ctx);

                    // Super Learner specialist voting + accuracy feedback.
                    // Extracted to `daemon_cognitive_tick::apply_specialist_voting`
                    // during V1.1.0 Strangler Fig wave 7 — pure move, no semantic
                    // change.  Ordering: runs after LinUCB select and before the
                    // decision_stage, same as the original inline form.
                    let voting_out = daemon_cognitive_tick::apply_specialist_voting(
                        &state,
                        &mut lctx,
                        &signal_digest,
                        &mut specialist_feedback,
                        &mut overflow_thresholds,
                        linucb_choice,
                        linucb_confidence,
                        cycle_count,
                        ode_t_sat_urgency,
                    );
                    last_specialist_votes = voting_out.disagreement_record;
                    voting_out.intervention
                };

                // Build behavior-interactive PID set from usage model EMA data.
                // Extracted to daemon_behavior_pids::build_behavior_interactive_pids (Wave 15).
                // [Android LMK] Sustained low cpu_wall_ratio → I/O-bound → interactive.
                // JIT PIDs from syscall_classifier are merged in unconditionally.
                let behavior_interactive_pids: HashSet<u32> =
                    daemon_behavior_pids::build_behavior_interactive_pids(
                        &state,
                        &snapshot,
                        &jit_protected_pids,
                    );

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
                    // snapshot.pressure.memory_pressure already includes battery
                    // + thermal boosts via effective_pressure::compute(). Don't add again.
                    let freed = page_reclaim.tick(
                        snapshot.pressure.memory_pressure,
                        display_turbo.is_turbo_active() || thermal_action.phase >= apollo_optimizer::engine::thermal_bailout::CoolingPhase::Phase2Moderate,
                        foreground_idle,
                    );
                    if freed > 0 {
                        state.metrics.lock_recover().metrics.paging_hints_applied += 1;
                    }
                }

                // Habituation per-process state tracking — extracted to
                // `daemon_cognitive_tick::update_habituation_state`.
                // Inspired by Thompson & Spencer 1966 / memoria-core habituation.rs.
                // Processes whose CPU and RSS bucket are unchanged for ≥5 cycles
                // are skipped in decide_actions.  Dishabituation on any change.
                let habituated_pids: HashSet<u32> = daemon_cognitive_tick::update_habituation_state(
                    &state,
                    collector.system(),
                    &mut habituation_map,
                    cycle_count,
                );

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

                // User context "telepathy" — extracted to
                // `daemon_cognitive_tick::compute_user_context`.
                // [Riva & Mantovani 2014] idle time + media state = highest-signal
                // context cues.  Uses IOHIDSystem HIDIdleTime every cycle, pmset
                // (sleep/call/audio) every 3 cycles with carry-forward to avoid
                // freeze_gate flicker, and SMC P-cluster temp > 75 °C to clamp
                // idle_secs for thermal headroom protection.
                let user_context = daemon_cognitive_tick::compute_user_context(
                    cycle_count,
                    &mut last_user_assertions,
                    cycle_hw_snap.as_ref(),
                );

                // Bypass habituation under critical conditions.
                // Habituation assumes "stable RSS = no problem", but under heavy swap
                // processes have stable RSS (swapped out) and stable CPU (not running) —
                // they LOOK calm but are the source of thrashing.  Force a fresh look
                // when p_oom ≥ 0.95 or swap ≥ 8 GB.
                // Swap reclaim ODE gate: when the model predicts saturation
                // within CRITICAL_ETA_SEC, pre-emptively boost reactor_weight
                // so the freeze decision stage acts before the threshold is hit.
                // [Denning 1968] — working-set overflow must be caught early;
                // [Zhao 2009] — compression-rate signal leads level signal.
                {
                    use apollo_optimizer::engine::swap_reclaim::{SwapRisk, CRITICAL_ETA_SEC};
                    match reclaim_forecast.risk {
                        SwapRisk::Critical => {
                            // T_sat ≤ 60 s: moderate pre-emptive boost.
                            reactor_weight = (reactor_weight + 0.25).min(1.0);
                            if let Some(eta) = reclaim_forecast.t_sat_sec {
                                tracing::info!(
                                    target: "apollo.swap_reclaim",
                                    eta_sec = format!("{:.1}", eta),
                                    net_mbps = format!("{:.2}",
                                        reclaim_forecast.net_rate_bps / (1024.0 * 1024.0)),
                                    "swap reclaim ODE: Critical — reactor boosted +0.25"
                                );
                            }
                        }
                        SwapRisk::Overflow => {
                            // Already past threshold — maximum boost, bypass habituated list.
                            reactor_weight = 1.0;
                            tracing::warn!(
                                target: "apollo.swap_reclaim",
                                swap_ratio = format!("{:.2}", reclaim_forecast.swap_ratio),
                                "swap reclaim ODE: Overflow — reactor_weight=1.0"
                            );
                        }
                        SwapRisk::Building => {
                            // Net positive but far from threshold — small early nudge.
                            let _ = CRITICAL_ETA_SEC; // used in risk classifier
                            reactor_weight = (reactor_weight + 0.05).min(1.0);
                        }
                        SwapRisk::Safe => {}
                    }
                }

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

                // Apply learned skills + trial induced skills.
                // Extracted to daemon_skill_tick::run_skill_tick (Wave 16).
                // [Fowler 2004] Strangler Fig — pure move, no semantic change.
                {
                    let skill_new = daemon_skill_tick::run_skill_tick(
                        &mut lctx.skill_registry,
                        &snapshot,
                        &state,
                        &collector,
                        foreground_pid,
                        workload_mode.as_str(),
                        is_root,
                        &actions,
                        &mut pending_trial_skill,
                    );
                    actions.extend(skill_new);
                }

                // Coordinated cluster freezing + Spotlight pressure gate.
                // Extracted to daemon_cluster_actions::run_cluster_actions (Wave 18).
                {
                    let causal_pairs = lctx.outcome_tracker.top_causal_pairs(5);
                    let cluster_out = daemon_cluster_actions::run_cluster_actions(
                        &causal_pairs,
                        &actions,
                        &collector,
                        snapshot.pressure.memory_pressure,
                        snapshot.pressure.swap_used_bytes,
                        overflow_thresholds.bg_pressure,
                        spotlight_paused,
                        spotlight_paused_at,
                    );
                    actions.extend(cluster_out.new_actions);
                    spotlight_paused = cluster_out.spotlight_paused;
                    spotlight_paused_at = cluster_out.spotlight_paused_at;
                }

                // Predictive agent: inject soft actions for PreThrottleNoise / ProactivePurge.
                // Extracted to daemon_agent_actions::run_agent_actions (Wave 19).
                {
                    let agent_new = daemon_agent_actions::run_agent_actions(
                        &agent_intervention,
                        &snapshot.top_processes,
                        &state,
                        &decide_interactive,
                    );
                    actions.extend(agent_new);
                }

                // Direct pressure hints + G20 ODE velocity hints.
                // Extracted to daemon_paging_hints::run_paging_hints (Wave 17).
                // [Jiang & Zhang 2005] proactive beats reactive; [Hellerstein 2004 §9] ODE leads.
                {
                    let hint_new = daemon_paging_hints::run_paging_hints(
                        &proc_snaps,
                        &state,
                        signal_digest.pressure_smooth,
                        reclaim_forecast.net_rate_bps,
                        foreground_app.as_deref(),
                        &actions,
                    );
                    actions.extend(hint_new);
                }

                // ── Heuristic pass: AdaptiveGovernor + protection scoring ────────────
                // Extracted to daemon_action_safety::run_heuristic_pass (Wave 13).
                // [Saltzer & Kaashoek 2009] Complete Mediation — single callsite for
                // all protection decisions. [Denning 1968] high-τ PIDs wired to ODE model.
                let heuristic_pass = daemon_action_safety::run_heuristic_pass(
                    &proc_snaps,
                    &hunt_snaps,
                    foreground_app.as_deref(),
                    foreground_pid,
                    &all_proc_names,
                    hour_of_day,
                    hw_features,
                    &state,
                    signal_digest.pressure_smooth,
                    &unfreeze_decay,
                    &reclaim_forecast,
                    &collector,
                    &actions,
                    &lctx.outcome_tracker.experience,
                    learnable_params.experience_pressure_band,
                    snapshot.pressure.memory_pressure,
                );
                let heuristic_decisions = heuristic_pass.heuristic_decisions;
                let heuristic_critical_pids = heuristic_pass.heuristic_critical_pids;
                let heuristic_stats = heuristic_pass.heuristic_stats;
                actions.extend(heuristic_pass.additional_actions);

                // Cable: stale_apps() → nominate stale background apps as freeze candidates.
                // Extracted to daemon_stale_apps::run_stale_app_freeze (Wave 21).
                {
                    let stale_new = daemon_stale_apps::run_stale_app_freeze(
                        signal_digest.pressure_smooth,
                        &all_proc_names,
                        &state,
                        &collector,
                        foreground_pid,
                        &heuristic_critical_pids,
                        &actions,
                    );
                    actions.extend(stale_new);
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
                        &learnable_params.rl_pressure_bands,
                        &learnable_params.rl_compressor_bands,
                    );
                    let sr = if snapshot.pressure.swap_total_bytes > 0 {
                        snapshot.pressure.swap_used_bytes as f64
                            / snapshot.pressure.swap_total_bytes as f64
                    } else {
                        0.0
                    };
                    // Only train hazard model when swap is actively growing (real OOM risk).
                    let swap_growing = snapshot.pressure.swap_delta_bytes_per_sec > 524_288.0;
                    if sr > 0.10 && swap_growing {
                        lctx.signal_intel.record_overflow(
                            snapshot.pressure.memory_pressure,
                            sr,
                            snapshot.pressure.memory_pressure,
                        );
                    }
                }
                // Track swap growth streak → RL meta-gate.
                if snapshot.pressure.swap_delta_bytes_per_sec > 1_048_576.0 {
                    swap_growth_streak = swap_growth_streak.saturating_add(1);
                } else {
                    swap_growth_streak = 0;
                }
                if let Some(rl) = lctx.overflow_guard.rl_agent.as_mut() {
                    rl.set_swap_growth_streak(swap_growth_streak);
                }

                // Observability: count one activation per cycle survival is active.
                // Previously the counter was declared but never incremented, so
                // survival_mode_activations was always 0 in runtime_metrics.json.
                let survival_active = apollo_optimizer::engine::safety::survival_mode_active_total(
                    snapshot.pressure.memory_pressure,
                    snapshot.pressure.swap_used_bytes,
                    snapshot.pressure.swap_total_bytes,
                );
                if survival_active {
                    state
                        .metrics
                        .lock_recover()
                        .metrics
                        .survival_mode_activations += 1;

                    // Jetsam demotion: mark non-foreground Chromium renderers
                    // as BACKGROUND so the kernel kills them first under OOM
                    // pressure — softer than SIGSTOP, keeps them responsive
                    // until the kernel actually reclaims. Idempotent syscall.
                    let _ = chromium_mgr.demote_background_renderers();

                    // Last-resort page reclaim: when swap crosses 80% of the
                    // exhaustion threshold, spawn `purge` to force the kernel
                    // to drain inactive file-backed pages. Rate-limited to
                    // once per 10 min — purge is expensive (~2s latency) and
                    // moderating the cadence avoids cascading I/O storms.
                    let threshold = apollo_optimizer::engine::safety::swap_exhaustion_threshold_bytes(
                        snapshot.pressure.swap_total_bytes,
                    );
                    let swap_used = snapshot.pressure.swap_used_bytes;
                    if swap_used as f64 >= threshold as f64 * 0.80 {
                        let can_purge = last_purge_at
                            .map(|t| t.elapsed() >= Duration::from_secs(600))
                            .unwrap_or(true);
                        if can_purge {
                            // Spawn non-blocking — purge runs in its own process
                            // and we don't wait on it. If spawn fails the daemon
                            // continues; the RL gate still throttles allocators.
                            if std::process::Command::new("purge").spawn().is_ok() {
                                last_purge_at = Some(Instant::now());
                            }
                        }
                    }
                }

                // Decaimiento gradual: si el sistema está en calma, relajar thresholds.
                lctx.overflow_guard.tick_decay(
                    snapshot.pressure.memory_pressure,
                    snapshot.pressure.compressor_pressure,
                    &learnable_params.rl_pressure_bands,
                    &learnable_params.rl_compressor_bands,
                );

                // ── Neuromodulator: bio-inspired parameter modulation ────────
                // Extracted to daemon_neuro_tick::apply_neuromodulator (Wave 8).
                // Best-available CPU temperature: SMC direct first, then IOKit estimate.
                let cpu_temp_celsius: Option<f64> = last_smc
                    .as_ref()
                    .and_then(|s| s.cpu_temp_celsius)
                    .or_else(|| {
                        cycle_hw_snap
                            .as_ref()
                            .and_then(|h| h.temps.p_cluster_celsius)
                            .map(|t| t as f64)
                    });
                daemon_neuro_tick::apply_neuromodulator(
                    &mut lctx,
                    &signal_digest,
                    &stability_oracle,
                    &thermal_action,
                    collector.system().processes().len(),
                    cpu_temp_celsius,
                    ode_t_sat_urgency,
                    unfreeze_decay.tau_novelty(),
                );
                // G14 — ODE Surprise Arousal: inject leading ODE prediction error
                // into arousal EMA before kernel pressure reacts.
                // [Schultz 1997 RPE] — prediction error is the primary arousal driver.
                {
                    let ode_rss_surprise = (ode_t_sat_urgency
                        * (-signal_digest.pressure_velocity as f64).max(0.0))
                        .clamp(0.0, 1.0);
                    arousal_state.inject_ode_surprise(ode_rss_surprise);
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
                // [Nygard 2018 "Release It!" Ch.5]
                let storms = daemon_feature_gates::enforce_wakeup_budget(
                    &state,
                    &mut wake_storm,
                    &heuristic_critical_pids,
                    foreground_pid,
                );

                // ── Feature 2 + 4: App Nap for LLM mode and post-wake window ──
                // During LLM inference or wake suppression: App-Nap all
                // non-foreground non-essential. Otherwise: release any
                // LLM/wake App-Nap that isn't also a live wake-storm offender.
                daemon_feature_gates::apply_app_nap_scheduling(
                    &state,
                    &collector,
                    &proc_snaps,
                    foreground_pid,
                    llm_active,
                    in_wake_suppression,
                    &storms,
                );

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
                if let Ok(tracker) = apollo_optimizer::engine::contention_tracker::global().lock() {
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
                            // No freeze actions in a sysctl-revert batch — pressure
                            // is irrelevant here; pass 0.0 so the assertion gate stays armed.
                            0.0,
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
                {
                    let fg_family_pids =
                        process_enrichment::build_foreground_family(foreground_pid, &process_tree);
                    let recently_active_window = std::time::Duration::from_secs(300);

                    actions.retain(|a| match a {
                        RootAction::ThrottleProcess { pid, name, .. }
                        | RootAction::FreezeProcess { pid, name, .. } => {
                            if fg_family_pids.contains(pid) {
                                return false;
                            }
                            if let Some(fg) = &foreground_app {
                                if name.contains(fg.as_str()) {
                                    return false;
                                }
                            }
                            if fg_detector.is_recently_active(name, recently_active_window) {
                                return false;
                            }
                            true
                        }
                        _ => true,
                    });
                }

                // F4 — Thermal Master Switch: >95°C P-cluster — suppress all Boost actions.
                // Also suppress during resource interrupt Emergency/SuperEmergency.
                let interrupt_phase = state.resource_interrupt.phase.load(Ordering::Acquire);
                if thermal_emergency || interrupt_phase >= 2 {
                    actions.retain(|a| !matches!(a, RootAction::BoostProcess { .. }));
                }

                // ── Chromium Renderer Manager ────────────────────────────────────
                // Extracted to daemon_chromium_tick::run_chromium_tick (Wave 11).
                // [Denning 1968] Working Set | [Jones 2011] Chromium Multi-Process Architecture
                daemon_chromium_tick::run_chromium_tick(
                    &mut chromium_mgr,
                    &focus_markov,
                    foreground_app.as_deref(),
                    foreground_pid,
                    &proc_snaps,
                    &state,
                    win_workload_intent,
                    &arousal_state,
                    &fluidity_state,
                    signal_digest.pressure_smooth as f32,
                    snapshot.pressure.memory_pressure,
                    cycle_count as u64,
                );

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
                    // TTL unfreeze + FIFO rotation.
                    // Extracted to daemon_freeze_executor::run_ttl_unfreeze_sweep().
                    // [Belady 1966] FIFO replacement under sustained memory pressure.
                    daemon_freeze_executor::run_ttl_unfreeze_sweep(
                        &state,
                        &frozen_state_path,
                        snapshot.pressure.memory_pressure,
                        &mut metrics,
                    );
                    metrics.metrics.budgets.cycle_boosts = 0;
                    metrics.metrics.budgets.cycle_throttles = 0;
                    metrics.metrics.budgets.cycle_hints = 0;
                    metrics.metrics.budgets.cycle_freezes = 0;
                    metrics.metrics.budgets.cycle_sysctl_writes = 0;
                    metrics.metrics.budgets.boost_denied_cooldown = 0;

                    let (mut graced_actions, throttle_suppressed, freeze_suppressed) =
                        process_enrichment::apply_post_wake_grace_policy(actions, grace_active);
                    metrics.metrics.post_wake_throttle_suppressed += throttle_suppressed;
                    metrics.metrics.post_wake_freeze_suppressed += freeze_suppressed;

                    // G19 — τ-Informed Ranking: sort freeze actions by learned τ ascending.
                    // Short τ = fast reload = freeze first (maximum RSS reclaim, minimum
                    // disruption when thawed). Non-freeze actions are left in original order.
                    // [Denning 1968] τ-based WSS — lower τ processes have smaller working-set
                    // reload cost and are safer to freeze/thaw first.
                    graced_actions.sort_by(|a, b| {
                        let tau_of = |action: &RootAction| -> f64 {
                            if let RootAction::FreezeProcess { name, .. } = action {
                                unfreeze_decay.tau_for_app(name)
                            } else {
                                f64::MAX // non-freeze actions sort to end
                            }
                        };
                        tau_of(a).partial_cmp(&tau_of(b)).unwrap_or(std::cmp::Ordering::Equal)
                    });

                    // Freeze confirmation gate: 2 cycles normal, 1 pre-emptive,
                    // 0 during launch. Per-cycle dedup + decay of stale candidates.
                    // Extracted to daemon_freeze_executor::apply_freeze_confirmation().
                    let confirmed_actions = daemon_freeze_executor::apply_freeze_confirmation(
                        graced_actions,
                        &fluidity_state,
                        &mut freeze_candidates,
                    );

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

                // Snapshot causal QoS preferences before exec_outcomes consumes final_actions.
                // FreezeProcess actions for CPU-dominant processes will be upgraded to
                // ThrottleProcess(aggressive=true) — QoS Background tier is less invasive
                // than SIGSTOP when CPU reduction is the only causal mechanism needed.
                // [Pearl 2009 §3] mediation analysis  [Nygard 2018] bulkhead: least-invasive first
                let causal_qos_names: std::collections::HashSet<String> =
                    lctx.causal_graph.qos_preferred_names();
                let mut causal_qos_upgrades_cycle = 0u32;

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
                    use apollo_optimizer::engine::degradation::DegradationInputs;

                    // ── Filter pipeline (Wave 9 Pass 1) ──────────────────────
                    // Circuit-breaker snapshot + degradation tier + cognitive
                    // gates (observe_only / block_aggressive) + mode filter +
                    // throttle dedup + causal QoS upgrade.  All mutation
                    // stays inside daemon_action_pipeline; this helper only
                    // touches `state.policy` — no metrics/frozen_state/mach_qos.
                    let filter_outcome = daemon_action_pipeline::run_filter_pipeline(
                        final_actions,
                        &state,
                        &snapshot,
                        prev_cog_decision.as_ref(),
                        &causal_qos_names,
                        reclaim_forecast.risk,
                    );
                    let cb_is_open = filter_outcome.cb_is_open;
                    let op_mode = filter_outcome.op_mode;
                    let mut filtered_actions = filter_outcome.filtered_actions;
                    causal_qos_upgrades_cycle += filter_outcome.causal_qos_upgrades;

                    // ── Predictive thaw gate ─────────────────────────────
                    // When pressure is already high, refuse to thaw any
                    // process whose ODE model predicts > MAX_PRED_GROWTH_BYTES
                    // of RSS re-accumulation within 5 seconds.  Prevents the
                    // classic failure mode: thaw browser tab under 0.82 →
                    // 0.95 pressure spike 3 s later → swap storm.
                    // [Strogatz 2015 §2.3] model-informed control;
                    // [Nygard 2018 §5] backpressure by action refusal.
                    {
                        const PRED_GATE_PRESSURE: f64 = 0.80;
                        const MAX_PRED_GROWTH_BYTES: u64 = 200 * 1024 * 1024; // 200 MB
                        let pressure = snapshot.pressure.memory_pressure as f64;
                        if pressure > PRED_GATE_PRESSURE {
                            let mut deferred = 0u32;
                            filtered_actions.retain(|a| {
                                if let RootAction::UnfreezeProcess { pid, name, .. } = a {
                                    let m_0 = collector
                                        .system()
                                        .process(sysinfo::Pid::from_u32(*pid))
                                        .map(|p| p.memory())
                                        .unwrap_or(0);
                                    let predicted =
                                        unfreeze_decay.predict_rss(name, m_0, 5.0);
                                    let growth = predicted.saturating_sub(m_0);
                                    if growth > MAX_PRED_GROWTH_BYTES {
                                        tracing::info!(
                                            target: "apollo.unfreeze_decay",
                                            pid = *pid,
                                            name = %name,
                                            pressure = %format!("{:.2}", pressure),
                                            growth_mb = growth / (1024 * 1024),
                                            "deferring thaw: predicted RSS growth exceeds headroom"
                                        );
                                        deferred += 1;
                                        return false;
                                    }
                                }
                                true
                            });
                            if deferred > 0 {
                                tracing::warn!(
                                    target: "apollo.unfreeze_decay",
                                    deferred,
                                    active_thaws = unfreeze_decay.active_thaw_count(),
                                    learned_apps = unfreeze_decay.learned_app_count(),
                                    "predictive thaw gate dropped {} candidate(s)",
                                    deferred
                                );
                            }
                        }
                    }

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
                            snapshot.pressure.memory_pressure,
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
                            snapshot.pressure.memory_pressure,
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
                        frozen_state.entry(*pid).or_insert_with(|| {
                            let name = apollo_optimizer::engine::process_identity::proc_name_for_pid(*pid);
                            FrozenEntry {
                                frozen_at: now,
                                source: FreezeSource::MainLoop,
                                pressure_at_freeze: snapshot.pressure.memory_pressure,
                                process_name: name,
                                start_sec: apollo_optimizer::engine::process_identity::ProcessIdentity::from_pid(*pid)
                                    .map(|pi| pi.start_sec)
                                    .unwrap_or(0),
                                original_jetsam_priority: None,
                            }
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

                // Unfreeze decay ODE: record fresh thaws, observe active ones, GC.
                // Bounded per-cycle: O(active_thaws) sysinfo lookups, no I/O.
                // [Strogatz 2015 §2.3 linear ODE; Denning 1968 working set]
                {
                    let now_instant = std::time::Instant::now();
                    let now_epoch_sec = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    // Track PIDs from both the regular action path and the
                    // staggered wake-unfreeze queue (5 PIDs/cycle from display wake).
                    let all_thaw_pids = exec_outcomes.newly_unfrozen_pids.iter().copied()
                        .chain(wake_thaw_pids.iter().copied());
                    for pid in all_thaw_pids {
                        if let Some(proc) =
                            collector.system().process(sysinfo::Pid::from_u32(pid))
                        {
                            unfreeze_decay.record_thaw(
                                pid,
                                proc.name().to_string(),
                                proc.memory(),
                                now_instant,
                            );
                        }
                    }
                    for pid in unfreeze_decay.active_thaw_pids() {
                        if let Some(proc) =
                            collector.system().process(sysinfo::Pid::from_u32(pid))
                        {
                            // Provide WSS from TASK_VM_INFO as M∞ ground-truth anchor.
                            // [Denning 1968] — WSS is the reliable predictor of steady-state
                            // RAM demand; eliminates running-max convergence noise.
                            let wss_hint = query_memory_profile(pid)
                                .map(|mp| mp.working_set_bytes);
                            unfreeze_decay.observe_sample_with_wss(
                                pid,
                                proc.memory(),
                                now_instant,
                                now_epoch_sec,
                                wss_hint,
                            );
                        }
                    }
                    unfreeze_decay.gc(now_instant, now_epoch_sec);
                }

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
                        last_specialist_votes
                            .as_ref()
                            .map(|(v, i)| (v.as_slice(), *i)),
                        &mut log_ingester,
                        &mut learnable_params,
                        ls_path.to_str().unwrap_or(""),
                        persist_generations,
                        skills_path(),
                        &mut nested_learner,
                        sleep_notifier.is_sleeping(),
                        ode_t_sat_urgency,
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
                // Extracted to daemon_neuro_tick::run_neurocognitive_tick (Wave 8).
                let cog_decision = daemon_neuro_tick::run_neurocognitive_tick(
                    &mut lctx,
                    &mut cognitive_state,
                    cycle_count as u64,
                    &signal_digest,
                    &throttle_names_for_outcome,
                    workload_mode.as_str(),
                );
                prev_cog_decision = Some(cog_decision);
                // LlmConfig live-reload: whitelisted fields only; skip if trial active
                // to avoid corrupting GemmaTrust outcome attribution. [Gray & Reuter 1992]
                if cycle_count % 100 == 0 && pending_trial_skill.is_none() {
                    use apollo_optimizer::engine::pipeline::periodic_stage::maybe_reload_llm_config;
                    let current_cfg = state.llm.lock_recover().llm_cfg.clone();
                    if let Some(outcome) =
                        maybe_reload_llm_config(cycle_count, &mut llm_cfg_reloader, &current_cfg)
                    {
                        if let Some(new_cfg) = outcome.new_cfg {
                            state.llm.lock_recover().llm_cfg = new_cfg;
                            tracing::info!("llm config live-reloaded from disk");
                        }
                        for rejected in &outcome.rejected {
                            tracing::warn!(field = %rejected.field, "llm config reload: field rejected (not in whitelist)");
                        }
                    }
                }
                // Autonomous rule induction every 100 cycles.
                // Extracted to daemon_skill_tick::run_rule_induction (Wave 22).
                if cycle_count % 100 == 0 {
                    daemon_skill_tick::run_rule_induction(
                        &mut lctx.skill_registry,
                        &lctx.outcome_tracker,
                        &state,
                        workload_mode.as_str(),
                        std::path::Path::new(skills_path()),
                    );
                }
                // State compression (% 500) is handled by run_periodic() below.
                // Hourly housekeeping (7200 cycles × 500ms ≈ 1 hour).
                if cycle_count % 7200 == 0 {
                    // GC stale entries from cache warmer + I/O shaper.
                    cache_warmer.gc();
                    io_shaper.gc();
                    // EffectivenessTracker: drop entries stale >7200 cycles
                    // (~1h) with <3 observations. Persists across restarts, so
                    // without this, process-name variants (build tools, CI
                    // scripts, short-lived workers) accumulate forever.
                    effectiveness_tracker.gc(3, 7200, cycle_count);
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
                // Extracted to daemon_cycle_tail::apply_fluidity_qos (Wave 10).
                daemon_cycle_tail::apply_fluidity_qos(
                    &state,
                    &fluidity_state,
                    &thermal_action,
                    foreground_pid,
                );

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
                    cycle_count,
                    sleep_notifier.is_sleeping(),
                );

                // ── Enriched telemetry + UCHS neurocognitive metrics ────────────────────
                // Extracted to daemon_cycle_tail::wire_enriched_telemetry (Wave 10).
                // Combines the two original blocks under a single state.metrics lock guard.
                daemon_cycle_tail::wire_enriched_telemetry(
                    &state,
                    &collector,
                    &daemon_cycle_tail::EnrichedTelemetryInputs {
                        snapshot: &snapshot,
                        swap_forecast: &swap_forecast,
                        fluidity_state: &fluidity_state,
                        overflow_thresholds: &overflow_thresholds,
                        behavior_interactive_pids: &behavior_interactive_pids,
                        cog_decision: &cog_decision,
                        cognitive_state: &cognitive_state,
                        lctx: &lctx,
                        causal_qos_upgrades_cycle,
                        thermal_predicted_throttle,
                        thermal_seconds_to_throttle,
                        thermal_trend_predicted: &thermal_trend_predicted,
                    },
                );

                // ── Periodic stage: GC and observability (% 100 / % 500 / % 7200 gates) ──
                // Extracted to daemon_cycle_tail::run_periodic_stage (Wave 10).
                // % 500 GC (experience compress, weight prune, skill GC) runs inside.
                // % 100 persist and rule-induction remain inline above (need SharedState).
                // % 7200 hourly GC remains inline above (binary-local types).
                let _periodic_result = daemon_cycle_tail::run_periodic_stage(
                    daemon_cycle_tail::PeriodicStageInputs {
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
                    },
                    &mut lctx,
                );

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
                // C2 fix (round-3): use `wait_timeout_while` so the predicate
                // and the wait release are a single atomic operation. The
                // previous `if !*triggered { wait_timeout } else { reset }`
                // pattern was technically correct for the "event before wait"
                // case, but spurious wakeups and a possible timeout-coinciding-
                // with-notify race could swallow a signal and delay the next
                // cycle by up to 2 seconds. `wait_timeout_while` explicitly
                // loops on the predicate under the lock, eliminating the race.
                {
                    let (lock, cvar) = &*state.cycle_condvar;
                    let guard = lock.lock_recover();
                    let (mut triggered, _) = cvar
                        .wait_timeout_while(guard, wait_duration, |t| !*t)
                        .unwrap_or_else(|e| e.into_inner());
                    *triggered = false;
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
                Some(nested_learner.clone()),
            );
            // Patch unfreeze-decay τ snapshot after the main persist so a crash
            // mid-persist leaves the previous learned-τ file intact.
            LearnedState::patch_unfreeze_decay(ls_path, unfreeze_decay.tau_snapshot());
            // Persist neuromodulator signal levels so DA/ACh/NA/5-HT survive restart.
            // [Schultz 1997] — reward prediction error signals require continuity.
            LearnedState::patch_neuro_state(ls_path, neuromod.snapshot());

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
                        // Shutdown sysctl-revert batch — no freezes in here.
                        0.0,
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
