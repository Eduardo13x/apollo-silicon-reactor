use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use apollo_engine::engine::platform::{MacOsPlatformAdapter, PlatformAdapter};

mod adaptive_overhead;
/// Global stop flag for signal handlers (SIGTERM/SIGINT).
/// Signal handlers cannot capture Arc/closures, so we use a static AtomicBool
/// that the main loop checks alongside `state.stop`.
mod cognitive_tick;
mod daemon_action_pipeline;
mod daemon_action_safety;
mod daemon_agent_actions;
mod daemon_behavior_pids;
mod daemon_chromium_tick;
mod daemon_cluster_actions;
mod daemon_cognitive_tick;
mod daemon_ctx_switch_tick;
mod daemon_cycle_tail;
mod daemon_dispatch_tick;
mod daemon_feature_gates;
mod daemon_fluidity_tick;
mod daemon_freeze_executor;
mod daemon_holt_winters_tick;
mod daemon_init;
mod daemon_kqueue_tick;
mod daemon_maintenance_tick;
mod daemon_markov_tick;
mod daemon_memory_budget;
mod daemon_neuro_tick;
mod daemon_paging_hints;
mod daemon_pressure_aggregator;
mod daemon_proc_scan_tick;
mod daemon_process_collector;
mod daemon_reactor;
mod daemon_reactor_tick;
mod daemon_resource_interrupt_tick;
mod daemon_rusage_tick;
mod daemon_sensor_tick;
mod daemon_signal_tick;
mod daemon_skill_tick;
mod daemon_socket_handler;
mod daemon_stale_apps;
mod daemon_survival_tick;
mod daemon_swap_reclaim_tick;
mod daemon_thermal_freeze;
mod daemon_thermal_tick;
mod daemon_turbo_manager;
mod daemon_wake_handler;
mod daemon_wake_unfreeze;
mod daemon_warn_limits;
mod heterogeneous_tick;
mod learning_tick;
mod local_policy_learning;
mod main_loop_msg;
mod metrics_reporter;
mod microexperiment_tick;
mod networkflow_tick;
mod process_enrichment;
mod socket_handler;
mod value_scheduler_tick;
mod webflow_tick;

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

/// SIGTERM handler — async-signal-safe: only sets an atomic flag.
extern "C" fn handle_sigterm(_sig: libc::c_int) {
    STOP_REQUESTED.store(true, Ordering::Release);
}

fn wait_for_cycle_signal(cycle_condvar: &Arc<(Mutex<bool>, Condvar)>, timeout: Duration) -> bool {
    let (lock, cvar) = &**cycle_condvar;
    let guard = lock.lock_recover();
    let (mut triggered, _) = cvar
        .wait_timeout_while(guard, timeout, |pending| !*pending)
        .unwrap_or_else(|error| error.into_inner());
    let was_triggered = *triggered;
    *triggered = false;
    was_triggered
}

fn current_exploration_time() -> apollo_engine::engine::exploration_scheduler::TimePoint {
    let (boot_id, monotonic_secs) =
        apollo_engine::engine::daemon_helpers::system_boot_session().unwrap_or((0, 0));
    apollo_engine::engine::exploration_scheduler::TimePoint {
        wall_unix_secs: Utc::now().timestamp(),
        monotonic_secs,
        boot_id,
    }
}

fn capture_exploration_environment(
    state: &SharedState,
    memory_pressure: f64,
    build_phase: apollo_engine::engine::build_tracker::BuildPhase,
    cognitive_paused: bool,
    fluidity_degraded_threshold: f32,
) -> (
    apollo_engine::engine::exploration_scheduler::ExplorationGates,
    apollo_engine::engine::exploration_scheduler::TimePoint,
) {
    let audio = apollo_engine::engine::coreaudio_active::audio_activity_snapshot();
    let hazard = apollo_engine::engine::shadow_signals::get_p_oom_30s();
    let (
        metric_audio,
        metric_call,
        metric_sleep,
        media_probe_state,
        app_launching,
        window_operation,
        fluidity_degraded,
        predicted,
        thermal,
        speculation_allowed,
    ) = {
        let metrics = state.metrics.lock_recover();
        (
            metrics.metrics.user_audio_active,
            metrics.metrics.user_call_in_progress,
            metrics.metrics.user_has_sleep_assertion,
            metrics.metrics.coreaudio_probe_state.clone(),
            metrics.metrics.app_launching,
            metrics.metrics.window_op_active,
            metrics.metrics.fluidity_degraded,
            metrics.metrics.fluidity_predicted_3s,
            metrics.thermal_level_real.clone(),
            metrics.metrics.apollo_overhead_speculation_allowed,
        )
    };
    let circuit_closed = {
        let policy = state.policy.lock_recover();
        matches!(
            *policy.circuit_breaker.state(),
            apollo_engine::engine::circuit_breaker::CircuitState::Closed
        )
    };
    let media_available = if audio.session_supported {
        audio.output_probe_available && audio.input_probe_available
    } else {
        media_probe_state == "session-fallback"
    };
    let build_active = build_phase != apollo_engine::engine::build_tracker::BuildPhase::Idle;

    (
        apollo_engine::engine::exploration_scheduler::ExplorationGates {
            daemon_shutdown: state.stop.load(Ordering::Relaxed),
            kill_switch: Path::new(kill_switch_path()).exists(),
            cognitive_paused,
            audio_output_active: audio.output_active || metric_audio,
            audio_input_active: audio.input_active,
            call_active: audio.realtime_call_active() || metric_call,
            sleep_assertion: metric_sleep,
            media_available,
            app_launching,
            window_operation,
            fluidity_degraded,
            predicted_fluidity_degraded: !predicted.is_finite()
                || predicted <= 0.0
                || predicted < fluidity_degraded_threshold,
            memory_pressure,
            thermal_available: !thermal.is_empty() && thermal != "unknown",
            thermal_nominal: thermal == "nominal",
            hazard_available: hazard.is_some(),
            p_oom_30s: hazard.unwrap_or(f64::NAN),
            circuit_closed,
            speculation_allowed,
            build_workload: build_active,
            build_phase_idle: !build_active,
            compiler_protection_active: build_active,
            ..apollo_engine::engine::exploration_scheduler::ExplorationGates::default()
        },
        current_exploration_time(),
    )
}

use apollo_engine::collector::SystemCollector;
use apollo_engine::engine::action_accumulator::{ActionAccumulator, ActionPhase, EmitContext};
use apollo_engine::engine::actuation_broker::{ActuationBroker, ActuationRequest};
use apollo_engine::engine::adaptive_governor::AdaptiveGovernor;
use apollo_engine::engine::amx_detector;
use apollo_engine::engine::audit_types::{DecisionReason, PolicyDecisionTrace};
use apollo_engine::engine::background_collectors::PressureCollector;
use apollo_engine::engine::capabilities::detect_capabilities;
use apollo_engine::engine::causal_graph::CausalGraph;
use apollo_engine::engine::compressor_aware::{
    decide_enhanced, query_memory_profile, sample_process_temperature, scan_regions, MemoryAction,
};
use apollo_engine::engine::daemon_helpers::{
    audit_log, battery_pressure_boost, detect_prior_crash, frozen_state_path, governor_state_path,
    holt_winters_path, hop_groups_path, installation_id_path, journal_path, kill_switch_path,
    learned_state_path, load_frozen_state, load_governor_state, load_wake_state, markov_path,
    merge_seed_into, metrics_path, overflow_history_path, parse_profile, pid_start_time,
    predictive_agent_path, reflex_state_path, remove_crash_sentinel, rl_threshold_path,
    signal_intelligence_path, skills_path, socket_path, telemetry_output_dir,
    temporal_histograms_path, timeline_path, unfreeze_outcome_events, unfreeze_pids_outcome,
    unfreeze_pids_verified_outcome, wake_state_path, write_frozen_state, write_governor_state,
    AsyncCommandMetric, AsyncCommandQueue, AsyncCommandSubmitError,
};
use apollo_engine::engine::focus_markov::FocusMarkov;
use apollo_engine::engine::foreground::{ForegroundDetector, ForegroundState};
use apollo_engine::engine::gpu_manager::GPUManager;
use apollo_engine::engine::holt_winters::HoltWinters;
use apollo_engine::engine::hw_bayes::HwFeatures;
use apollo_engine::engine::hw_predictor::{HwPredictorWorker, HwPressure};
use apollo_engine::engine::iokit_sensors::{HardwareSnapshot, ThermalState};
use apollo_engine::engine::kqueue_pressure;
use apollo_engine::engine::latency_monitor::{self, LatencySignals};
use apollo_engine::engine::learned_state::{LearnableParams, LearnedState, RestoreQualityMonitor};
use apollo_engine::engine::local_consolidation::{LocalConsolidationVerdict, LocalConsolidator};
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::lse_counters::LockFreeMetrics;
use apollo_engine::engine::mach_qos::{
    MachQoSManager, QoSAuditDisposition, QoSOutcome, SchedulingTier,
};
use apollo_engine::engine::overflow_guard::{is_build_tool_name, OverflowGuard};
use apollo_engine::engine::pipeline::decision_stage::{DecisionStage, PolicyContext};
use apollo_engine::engine::pipeline::learning_context::LearningContext;
use apollo_engine::engine::policy_store::{
    delete_file_best_effort, feedback_path_root, legacy_teacher_paths, load_repo_config,
    pending_trial_path, policy_path_root, read_json, write_json, LearnedPolicy,
};
use apollo_engine::engine::power_management::detect_battery_status;
use apollo_engine::engine::predictive_agent::{
    AgentContext, Intervention, PredictiveAgent, SpecialistVote,
};
use apollo_engine::engine::proc_taskinfo;
use apollo_engine::engine::profile_governor::GovernorInput;
use apollo_engine::engine::safety::{
    critical_background_processes, enforce_limits_with_budget, is_protected_name,
};
use apollo_engine::engine::signal_intelligence::SignalIntelligence;
use apollo_engine::engine::smc_reader::SmcReader;
use apollo_engine::engine::sysctl_governor::{
    SysctlGovernor, SysctlGovernorInput, SysctlGovernorStatus,
};
use apollo_engine::engine::thermal_interrupt::{
    resource_interrupt_proposal_channel, spawn_resource_sentinel_with_proposals,
    ResourceInterruptActuationWindow, ResourceInterruptState, SentinelConfig,
};
use apollo_engine::engine::types::{
    EnergyConsumerInfo, ForegroundAppInfo, FreezeSource, FrozenEntry, FrozenPidEntry,
    FrozenStatePersisted, HardPath, LatencyTarget, RootAction, RuntimeMetrics, SafetyPolicy,
};
use apollo_engine::engine::usage_model::{usage_model_path_root, UsageModel};
use apollo_engine::engine::user_profile::{UserProfile, UserProfilePersisted};
use apollo_engine::engine::wait_graph;
use apollo_engine::engine::work_session::WorkSession;
use apollo_engine::engine::workload_classifier::classify_by_memory;
use apollo_engine::engine::workload_classifier::{
    classify_workload_mode, WorkloadFeatures, WorkloadMode,
};
use chrono::{Timelike, Utc};
use clap::{Parser, Subcommand};

// v0.9.0: canonical SharedState — all domain groups live in daemon_state.rs
use apollo_engine::engine::daemon_state::{
    HardwareState, MetricsState, PolicyState, ProcessState, ReactorStatus as DomainReactorStatus,
    SharedState, UsageDomainState, UsageTrackerState,
};

// FREEZE_TTL_SECS → daemon_helpers

/// Battery-aware pressure boost: on battery, effective pressure is raised so
/// all decision gates trigger sooner.  This proactively freezes backgrounds
/// before hardware thermal throttling kicks in.
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
// Local policy outcome counters

// parse_profile, write_metrics → daemon_helpers

// write_governor_state, load_governor_state, append_timeline → daemon_helpers

// wake_state, frozen_state, unfreeze, should_unfreeze, should_rotate_oldest + tests → daemon_helpers

// rotate_timeline → daemon_helpers

// compute_p95 → daemon_helpers

// filter_boost_cooldown, apply_post_wake_grace_policy, context_to_thermal,
// append_discrepancy_log, build_foreground_family, build_enriched_process_data_with_tree,
// convert_and_merge_heuristic_decisions, HeuristicStats → process_enrichment

/// Verify a RootAction's target PID still has the same identity at filter time.
///
/// Returns `true` if the action is safe to emit/dispatch, `false` if the PID is
/// dead or has been recycled (different process at same numeric PID).
///
/// Mirrors `execute_actions::verify_pid_identity` exactly:
/// - For per-PID actions: start_sec match + start_usec match (when both >0) +
///   name match (always evaluated as defense-in-depth).
/// - For non-PID actions (SetSysctl/ToggleSpotlight/QuarantineDaemon): always
///   returns true (no PID to verify).
///
/// Used by Phase A1 (universal pre-emit filter) and Phase A2 (post-drain
/// re-verify) to drop actions whose target PID became invalid before they
/// reach the safety layer (where they would log as `block_reason: PidRecycled`).
///
/// [Idempotency Pattern — 1001 patterns slide 7]
/// [Anti-pattern: Ignoring Idempotency — 1001 patterns slide 59]
/// Verify a RootAction's target PID still has the same identity at filter time.
///
/// Returns `true` if the action is safe to emit/dispatch, `false` if the PID is
/// dead or has been recycled (different process at same numeric PID).
///
/// Mirrors `execute_actions::verify_pid_identity` exactly:
/// - For per-PID actions: start_sec match + start_usec match (when both >0) +
///   name match (always evaluated as defense-in-depth).
/// - For non-PID actions (SetSysctl/ToggleSpotlight/QuarantineDaemon): always
///   returns true (no PID to verify).
///
/// Sprint 3 cost recovery: results memoized in `IdentityCache` for 30s.
/// Cache hit skips proc_pidpath/csops syscalls. Cache miss does the full
/// verify_pid_identity-equivalent check then inserts.
///
/// [Idempotency Pattern — 1001 patterns slide 7]
/// [Cache-Aside Pattern — 1001 patterns slide 11]
/// Sprint 12 perf-fix (2026-05-30): cheap, stable fingerprint of the
/// `(pid, name)` projection of `top_processes`. Used as the cache key for
/// the per-cycle `companion_of_fg_pids` memoization (see
/// [`CompanionFgCache`]). Order-independent — XOR is commutative — and
/// allocation-free.
///
/// The fingerprint must change when ANY element of the relevant projection
/// changes; collisions only cause spurious cache hits (i.e. a stale set is
/// kept one extra cycle), which is bounded by CompanionGraph mutation
/// witness on the next cycle. A 64-bit XOR over `pid as u64` and a per-name
/// hash is sufficient — the alternative `epoch`-keyed witness loses
/// independence from snapshot identity and would invalidate even when the
/// top-process projection is unchanged.
fn fingerprint_top_processes(top: &[apollo_engine::collector::ProcessStats]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    top.iter().fold(0u64, |acc, p| {
        let mut h = DefaultHasher::new();
        p.name.hash(&mut h);
        let name_hash = h.finish();
        acc ^ (p.pid as u64).rotate_left(1) ^ name_hash
    })
}

fn swap_reclaim_bypass_active(swap_used_bytes: u64, swap_total_bytes: u64, p_oom_30s: f64) -> bool {
    apollo_engine::engine::safety::survival_mode_active_total(
        0.0,
        swap_used_bytes,
        swap_total_bytes,
    ) || p_oom_30s >= 0.95
}

fn contention_mach_tier_event(
    cycle: u64,
    pid: u32,
    tier: SchedulingTier,
    outcome: &QoSOutcome,
    audit_disposition: QoSAuditDisposition,
) -> apollo_engine::engine::decision_ledger::ActuatorDecisionEvent {
    use apollo_engine::engine::decision_ledger::{ActuatorDecisionEvent, ActuatorDecisionOutcome};

    let (disposition, detail) = match audit_disposition {
        QoSAuditDisposition::Applied => (
            ActuatorDecisionOutcome::Applied,
            "Mach tier applied".to_string(),
        ),
        QoSAuditDisposition::NoOp => (
            ActuatorDecisionOutcome::NoOp,
            "requested Mach tier already active".to_string(),
        ),
        QoSAuditDisposition::Blocked => (
            ActuatorDecisionOutcome::Blocked,
            "Mach tier unavailable or protected".to_string(),
        ),
        QoSAuditDisposition::Failed => (
            ActuatorDecisionOutcome::Failed,
            outcome
                .error
                .clone()
                .unwrap_or_else(|| "Mach tier syscall failed".to_string()),
        ),
    };
    let tier_name = match tier {
        SchedulingTier::Foreground => "foreground",
        SchedulingTier::Normal => "normal",
        SchedulingTier::Background => "background",
    };
    ActuatorDecisionEvent::local(
        format!("contention:mach_tier:{tier_name}"),
        format!("pid:{pid}"),
        cycle,
        disposition,
        "contention-separation",
        detail,
    )
}

/// Sprint 12 perf-fix (2026-05-30). Single-slot cache for the
/// `companion_of_fg_pids` HashSet that decide_actions reads on every
/// cycle. The set is derived from
/// `top_processes × CompanionGraph::is_companion_of(fg_app, name)` which
/// is O(top_processes.len() × 2 HashMap lookups) per cycle; in steady
/// state `top_processes` is stable across consecutive 5-s ticks and the
/// foreground app rarely flips, so cache hit ratio approaches 1.0.
///
/// Invalidation witnesses (see [`CompanionFgCache::is_valid`]):
///   - foreground app identity changes,
///   - `top_processes` (pid, name) projection changes,
///   - `CompanionGraph::total_cycles` or `anchor_count` advances
///     (witnesses the only mutators of `is_companion_of`'s output —
///     `observe_cycle` past `ATTENTION_FLOOR` and `self_improve` decay).
///
/// [Saltzer & Schroeder 1975] Economy of Mechanism — single chokepoint
/// memoize keyed on the observable mutation witness, not on wall clock.
struct CompanionFgCache {
    fg_app: Option<String>,
    top_proc_fingerprint: u64,
    graph_total_cycles: u64,
    graph_anchor_count: usize,
    pids: HashSet<u32>,
}

impl CompanionFgCache {
    fn is_valid(
        &self,
        fg_app: Option<&str>,
        fingerprint: u64,
        graph_total_cycles: u64,
        graph_anchor_count: usize,
    ) -> bool {
        self.fg_app.as_deref() == fg_app
            && self.top_proc_fingerprint == fingerprint
            && self.graph_total_cycles == graph_total_cycles
            && self.graph_anchor_count == graph_anchor_count
    }
}

fn pid_identity_still_valid(
    action: &apollo_engine::engine::types::RootAction,
    manager: &apollo_engine::engine::identity_cache_manager::IdentityCacheManager,
    lf_metrics: &apollo_engine::engine::lse_counters::LockFreeMetrics,
) -> bool {
    // Action-aware shim over the cache lifecycle manager. Identity-bearing
    // actions delegate to manager.verify; non-PID actions (Spotlight,
    // sysctl, daemon quarantine) trivially pass.
    match action.identity_fields() {
        Some((pid, name, start_sec, start_usec)) => {
            manager.verify(pid, name, start_sec, start_usec, lf_metrics)
        }
        None => true,
    }
}

/// Case-insensitive substring match of a foreground app name against a small
/// allowlist of known dev tools. Used only to feed the work-session hysteresis
/// latch ("is this a dev session?"). Reuses the already-detected foreground app
/// name — no new process scan. Substring match handles macOS 15-char name
/// truncation (e.g. `language_server`) and bundle suffixes.
fn is_dev_tool_name(name: &str) -> bool {
    const DEV_TOOLS: &[&str] = &[
        "cargo",
        "rustc",
        "node",
        "zed",
        "claude",
        "alacritty",
        "esbuild",
        "language_server",
        "code",
        "iterm2",
        "warp",
    ];
    let lc = name.to_ascii_lowercase();
    DEV_TOOLS.iter().any(|tool| lc.contains(tool))
}

/// Toggle Spotlight indexing via `mdutil -a -i on/off`.
///
fn main() -> anyhow::Result<()> {
    // Compact text logging to stderr (captured by launchd → apollo-optimizer.err.log).
    // Override level at runtime: APOLLO_LOG=debug apollo-optimizerd
    //
    // 2026-05-31 (post-flamegraph round 2): switched from .json() to .compact().
    // Samply (PID 93434, 6304 samples, 60 s) confirmed tracing_subscriber JSON
    // formatter accounted for 30.7% top self-time + 12.4% drop_in_place (~47%
    // combined) AFTER the round-1 INFO→DEBUG demotes — the residual cost was
    // structural: JSON field escaping, RFC3339 timestamp formatting, and
    // String allocation per event. Compact format produces 3-5× cheaper events
    // with similar diagnostic value (no field-quoting overhead, fixed
    // timestamp format). Audit journal (journal.jsonl) remains structured —
    // this only changes the diagnostic stderr stream.
    // [Pirolli & Card 1999] information foraging cost: cheap-enough logs > rich-but-expensive.
    {
        use tracing_subscriber::{fmt, EnvFilter};
        let filter =
            EnvFilter::try_from_env("APOLLO_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
        fmt()
            .compact()
            .with_env_filter(filter)
            .with_target(false)
            .init();
    }

    let cli = Cli::parse();

    match cli.command {
        Commands::Daemon { profile, dry_run } => {
            // sysinfo initializes Rayon's global pool on its first process census.
            // Configure it before startup recovery performs any such census.
            let parallel_runtime = daemon_init::configure_parallel_runtime();
            let mut platform_adapter = MacOsPlatformAdapter::detect();
            let mut runtime_capability_graph = platform_adapter.probe();
            let profile = parse_profile(&profile);
            let is_root = unsafe { libc::geteuid() } == 0;
            let caps = detect_capabilities();

            // One-way migration: the local S2 -> Dr Zero -> S1 circuit owns
            // learning now. Retire persisted prompt state and API credentials
            // so an upgrade cannot accidentally revive the external Teacher.
            let (legacy_teacher_state, legacy_teacher_key) = legacy_teacher_paths(is_root);
            for path in [&legacy_teacher_state, &legacy_teacher_key] {
                let existed = path.exists();
                delete_file_best_effort(path);
                if existed {
                    if path.exists() {
                        tracing::warn!(
                            path = %path.display(),
                            "retired Teacher state could not be removed safely"
                        );
                    } else {
                        tracing::info!(path = %path.display(), "removed retired Teacher state");
                    }
                }
            }

            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                profile = profile.as_str(),
                root = is_root,
                "apollo-optimizerd starting"
            );
            let config_path = PathBuf::from("/etc/apollo-optimizer/config.toml");
            let repo_cfg = load_repo_config(&config_path);
            let reflex_cfg = repo_cfg.reflex.clone();
            let learned_policy_path = policy_path_root(is_root);
            let feedback_path = feedback_path_root(is_root);

            let usage_model_path = usage_model_path_root(is_root);
            let mut usage_model = UsageModel::load(&usage_model_path);
            let stale_usage_entries = usage_model.prune_stale(Utc::now());
            if stale_usage_entries > 0 {
                usage_model.persist(&usage_model_path);
                tracing::info!(
                    stale_usage_entries,
                    "usage-model calibration: discarded stale process identities"
                );
            }
            let usage_events_path = if is_root {
                PathBuf::from("/var/lib/apollo/learn/usage_events.jsonl")
            } else {
                PathBuf::from("/tmp/apollo-usage_events.jsonl")
            };

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
                let weight_report =
                    apollo_engine::engine::outcome_tracker::sanitize_pattern_weights(
                        &mut p.pattern_weights,
                    );
                let mut policy_changed = weight_report.changed();
                if policy_changed {
                    tracing::info!(
                        removed_ephemeral = weight_report.removed_ephemeral,
                        bounded = weight_report.bounded,
                        "learned-policy sanitize: bounded portable evidence"
                    );
                }
                // Sanitize (2026-06-20): drop any NOISE pattern that shadows an
                // interactive or safety-protected process. A legacy policy
                // noise-classified `language_server` (the LSP) — a noise+
                // interactive conflict that could throttle a protected process.
                // Truncation-aware. Self-heals on every daemon start.
                {
                    let interactive: Vec<String> = p.interactive_patterns.to_vec();
                    let lpm = apollo_engine::engine::decide_actions::learned_pattern_matches;
                    let conflicts = |n: &String| -> bool {
                        if apollo_engine::engine::safety::is_protected_name(n) {
                            return true;
                        }
                        if !apollo_engine::engine::usage_model::is_noise_candidate_name(n) {
                            return true;
                        }
                        let nl = n.to_lowercase();
                        interactive.iter().any(|ip| {
                            let il = ip.to_lowercase();
                            lpm(&nl, &il) || lpm(&il, &nl)
                        })
                    };
                    let removed: Vec<String> = p
                        .noise_patterns
                        .iter()
                        .filter(|n| conflicts(n))
                        .cloned()
                        .collect();
                    if !removed.is_empty() {
                        std::sync::Arc::make_mut(&mut p.noise_patterns)
                            .retain(|n| !removed.contains(n));
                        policy_changed = true;
                        tracing::warn!(
                            ?removed,
                            "learned-policy sanitize: dropped noise patterns shadowing interactive/protected"
                        );
                    }
                }
                if policy_changed {
                    write_json(&learned_policy_path, &p, Some(0o600));
                }
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
                    learned_policy_path: learned_policy_path.clone(),
                    feedback_path,
                    adaptive_governor: AdaptiveGovernor::new(),
                    timeline: VecDeque::new(),
                    circuit_breaker:
                        apollo_engine::engine::circuit_breaker::CircuitBreaker::default(),
                    degradation: apollo_engine::engine::degradation::DegradationController::default(
                    ),
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
                    survival_window:
                        apollo_engine::engine::survival_window::SurvivalActivationWindow::new(),
                })),
                frozen_state: Arc::new(Mutex::new(frozen_since_boot.clone())),
                process: Arc::new(Mutex::new(ProcessState {
                    last_blockers: Vec::new(),
                    wake_state,
                })),
                stop: Arc::new(AtomicBool::new(false)),

                usage: Arc::new(Mutex::new(UsageDomainState {
                    usage_model,
                    usage_model_path,
                    usage_events_path,
                    usage_tracker: UsageTrackerState::default(),
                })),

                mach_qos: Arc::new(Mutex::new(MachQoSManager::new())),
                freeze_cooldown: Arc::new(Mutex::new(
                    apollo_engine::engine::freeze_cooldown::FreezeCooldown::new(),
                )),
                effect_decay: Arc::new(Mutex::new(
                    apollo_engine::engine::effect_decay::DecayWatchdog::new(),
                )),
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
                        tcp_last_scale_up_secs_ago: None,
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

            // S10 (2026-06-06): install the shared effect_decay handle so
            // producer call sites in `execute_actions.rs` can enroll
            // observations without threading a new parameter. Idempotent.
            apollo_engine::engine::effect_decay::install_global(state.effect_decay.clone());

            // Load persisted UserProfile (learning survives daemon restarts).
            if let Some(persisted) = read_json::<UserProfilePersisted>(&state.user_profile_path) {
                state.policy.lock_recover().adaptive_governor.user_profile =
                    UserProfile::from_persisted(persisted);
            }

            // Scrub learned policy: remove identities that can never be interactive.
            // The local learner may refine weights but cannot override this safety seed.
            let learned_policy_path = state.policy.lock_recover().learned_policy_path.clone();
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
                std::sync::Arc::make_mut(&mut policy.interactive_patterns)
                    .retain(|p| !bad_interactive.iter().any(|bad| p.contains(bad)));
                // Preserve a conservative background-only seed.
                if !policy.noise_patterns.contains(&"apsd".to_string()) {
                    std::sync::Arc::make_mut(&mut policy.noise_patterns).push("apsd".to_string());
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
                    let outcome = unfreeze_pids_verified_outcome(&frozen_state);
                    for pid in outcome.forgettable_pids() {
                        frozen_state.remove(&pid);
                    }
                    write_frozen_state(&frozen_state_path, &frozen_state);
                    let count = outcome.applied_count();
                    {
                        let mut metrics = state.metrics.lock_recover();
                        metrics.metrics.post_wake_defensive_unfreezes += count;
                        metrics.metrics.unfreezes_applied += count;
                        metrics.metrics.throttle_reverted += count;
                    }
                }
            }

            // Evolve iter-5 (2026-06-10): adopt orphan boosts. The EffectLedger
            // is in-memory and resets on restart, so nice=-10 boosts applied
            // by a PREVIOUS daemon instance are orphaned — the fresh ledger
            // never reverts them, and they propagate to children via fork
            // inheritance (observed: 10 processes at -10 across a restart,
            // including a shell and all its children). nice EXACTLY -10 is
            // Apollo's boost signature (the kernel uses -20 for kernel_task,
            // other negative values for its own daemons); reset those on the
            // non-hard-protected set. The current foreground app, if any, is
            // re-boosted within one cycle by the normal loop.
            {
                use sysinfo::{ProcessRefreshKind, RefreshKind, System};
                let sys = System::new_with_specifics(
                    RefreshKind::new().with_processes(ProcessRefreshKind::new()),
                );
                let mut adopted = 0u64;
                for (pid, proc) in sys.processes() {
                    let pid_u32 = pid.as_u32();
                    let nice = unsafe { libc::getpriority(libc::PRIO_PROCESS, pid_u32) };
                    let hp = apollo_engine::engine::safety::hard_protected_contains(proc.name());
                    if apollo_engine::engine::effect_ledger::is_orphan_boost_signature(nice, hp) {
                        unsafe {
                            libc::setpriority(libc::PRIO_PROCESS, pid_u32, 0);
                        }
                        adopted += 1;
                    }
                }
                if adopted > 0 {
                    tracing::info!(
                        adopted,
                        "startup: reverted orphan nice=-10 boosts from prior daemon"
                    );
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

            // IPC mpsc channel: socket threads forward CLI Purge requests here.
            // Initialized before spawn_control_socket so the OnceLock is set
            // before any socket thread can attempt MAIN_LOOP_TX.get().
            let (main_loop_tx, main_loop_rx) =
                std::sync::mpsc::channel::<main_loop_msg::MainLoopMsg>();
            main_loop_msg::MAIN_LOOP_TX
                .set(std::sync::Mutex::new(main_loop_tx))
                .map_err(|_| anyhow::anyhow!("MAIN_LOOP_TX already set"))?;

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

            {
                let mut metrics = state.metrics.lock_recover();
                metrics.metrics.parallel_runtime_enabled = parallel_runtime.enabled;
                metrics.metrics.parallel_worker_threads = parallel_runtime.worker_threads;
                metrics.metrics.parallel_expected_profile =
                    parallel_runtime.build.expected_profile.clone();
                metrics.metrics.parallel_compiled_profile =
                    parallel_runtime.build.compiled_profile.clone();
                metrics.metrics.parallel_effective_profile =
                    parallel_runtime.build.effective_profile.clone();
                metrics.metrics.parallel_feature_compiled =
                    parallel_runtime.build.adaptive_feature_compiled;
                metrics.metrics.parallel_max_worker_threads =
                    parallel_runtime.build.max_worker_threads;
                metrics.metrics.parallel_disabled_reason =
                    parallel_runtime.build.disabled_reason.clone();
                metrics.metrics.parallel_worker_qos_intent =
                    parallel_runtime.build.worker_qos_intent.clone();
                metrics.metrics.parallel_worker_qos_status =
                    parallel_runtime.build.worker_qos_status.clone();
                metrics.metrics.parallel_worker_qos_failures =
                    parallel_runtime.build.worker_qos_failures;
            }
            tracing::info!(
                enabled = parallel_runtime.enabled,
                workers = parallel_runtime.worker_threads,
                "adaptive parallel sensing runtime configured"
            );
            let mut collector = SystemCollector::new();
            let mut thrash = process_enrichment::ThrashState::default();
            let journal_path = PathBuf::from(journal_path());
            let metrics_path = PathBuf::from(metrics_path());
            // Phase 1.5a — per-cycle telemetry archive (MLP router unblock).
            let metrics_history_path =
                PathBuf::from(apollo_engine::engine::daemon_helpers::metrics_history_path());
            let history_cfg = repo_cfg.history.unwrap_or_default();
            let mut metrics_history_writer =
                apollo_engine::engine::daemon_metrics_history::MetricsHistoryWriter::new(
                    metrics_history_path,
                    history_cfg,
                );
            let mut critical_failure_timestamps: Vec<Instant> = Vec::new();
            let mut override_was_active = false;
            let daemon_start = Instant::now();

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
                mut memory_budget,
                mut self_diagnosis,
                mut recently_applied,
                recently_applied_restore_status,
                identity_cache,
                mut maintenance_state,
                mut companion_graph,
                mut active_coalitions,
            } = daemon_init::DaemonSubsystems::new();
            let installation_id = apollo_engine::engine::installation_identity::load_or_create(
                Path::new(installation_id_path()),
            )
            .unwrap_or_else(|error| {
                tracing::error!(
                    %error,
                    "installation identity unavailable; World Model authority is protected"
                );
                apollo_engine::engine::installation_identity::InstallationId::UNKNOWN
            });
            let mut telemetry_medallion =
                apollo_engine::engine::telemetry_medallion::TelemetryMedallion::new(
                    installation_id,
                );
            let mut decision_ledger = apollo_engine::engine::decision_ledger::DecisionLedger::new();
            let mut unified_learning_health =
                apollo_engine::engine::unified_learning_health::UnifiedLearningHealthCache::default(
                );
            let async_commands = AsyncCommandQueue::new();
            {
                let mut m_guard = state.metrics.lock_recover();
                m_guard.metrics.recently_applied_restore_status =
                    Some(recently_applied_restore_status);
            }
            // Cumulative dedup_drops counters from prior cycle — used to
            // compute per-cycle delta for self_diagnosis recording.
            let mut last_dedup_setmem: u64 = 0;
            let mut last_dedup_throttle: u64 = 0;
            let mut last_dedup_freeze: u64 = 0;
            let mut last_dedup_unfreeze: u64 = 0;
            let mut last_dedup_boost: u64 = 0;
            let mut last_dedup_thread_qos: u64 = 0;
            let mut nested_learner = apollo_engine::engine::nested_learner::NestedLearner::new();
            let mut focus_markov = FocusMarkov::new(PathBuf::from(markov_path()));
            // TelemetryLogger: ring-buffer collection for time-series training data.
            // [Welch 1967, Tuli et al. 2022] — event-triggered dumps capture pre-anomaly context.
            let mut telemetry_logger =
                apollo_engine::engine::telemetry_logger::TelemetryLogger::new(PathBuf::from(
                    telemetry_output_dir(),
                ));
            // Warm-start: reload recent history so anomaly detector skips cold-start.
            // [Gray & Reuter 1992] §11.3 — restart protocols restore in-flight state.
            telemetry_logger.warm_start_from_dir(3);
            // StabilityOracle: aggregate jank + zombie + swap-spike into RL reward.
            // [Schulman et al. 2017] PPO per-cycle reward; [Nygard 2018] cascading instability.
            let mut stability_oracle =
                apollo_engine::engine::stability_oracle::StabilityOracle::new();
            let hw_path = PathBuf::from(holt_winters_path());
            let mut holt_winters = HoltWinters::load(&hw_path).unwrap_or_default();
            let mut hw_last_hour: Option<u8> = None;
            let mut hw_pressure_accum: f64 = 0.0;
            let mut hw_pressure_count: u32 = 0;
            let mut hw_predictor_worker = HwPredictorWorker::spawn();
            let mut sysctl_governor = SysctlGovernor::new(is_root);
            // Hardware capability scaling for SafetyPolicy::for_capabilities().
            // Detected once at startup via detect_hw_caps() (~1ms sysinfo query).
            let (hw_cores, hw_ram_gb) = daemon_init::detect_hw_caps();
            let hierarchy_hardware = apollo_engine::engine::telemetry_medallion::HardwareRegime {
                p_core_count: caps.p_core_count.unwrap_or(0),
                e_core_count: caps.e_core_count.unwrap_or(0),
                ram_gib: hw_ram_gb.round().clamp(0.0, u32::MAX as f64) as u32,
            };
            telemetry_medallion.bind_live_hardware(hierarchy_hardware);
            let exploration_origin =
                apollo_engine::engine::exploration_scheduler::ExplorationOrigin {
                    installation_id: installation_id.0,
                    hardware: apollo_engine::engine::exploration_scheduler::HardwareIdentity {
                        p_core_count: hierarchy_hardware.p_core_count,
                        e_core_count: hierarchy_hardware.e_core_count,
                        ram_gib: hierarchy_hardware.ram_gib,
                    },
                };
            let mut exploration_scheduler =
                apollo_engine::engine::exploration_scheduler::ExplorationScheduler::cold_start(
                    exploration_origin,
                );
            // GPU thermal monitoring: integrates with thermal_manager for GPU-aware decisions.
            let mut gpu_mgr = GPUManager::new();
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
            // Phase 1 consumer (2026-06-11): the cadence pre-arm reads the
            // planner's hint file every ~15 cycles. Path mirrors the
            // producer's root/non-root selection above.
            let mut planner_hints_path_for_cadence = std::path::PathBuf::new();
            {
                // The local `metrics_path` PathBuf is bound much later
                // (line 795). Use the helper function directly here so
                // we don't depend on that ordering.
                let metrics_pb =
                    std::path::PathBuf::from(apollo_engine::engine::daemon_helpers::metrics_path());
                let is_root_for_planner = unsafe { libc::geteuid() } == 0;
                let hints_pb = if is_root_for_planner {
                    std::path::PathBuf::from("/var/lib/apollo/planner_hints.json")
                } else {
                    std::path::PathBuf::from("/tmp/apollo-planner_hints.json")
                };
                planner_hints_path_for_cadence = hints_pb.clone();
                let calibration_pb = if is_root_for_planner {
                    std::path::PathBuf::from("/var/lib/apollo/calibration.jsonl")
                } else {
                    std::path::PathBuf::from("/tmp/apollo-calibration.jsonl")
                };
                let _planner_stop = apollo_engine::engine::planner::Planner::new(
                    Duration::from_secs(30),
                    metrics_pb,
                    hints_pb,
                )
                .with_calibration_log(calibration_pb)
                .spawn();
            }
            // Resource sentinel: low-latency observer for thermal/memory/power emergencies.
            // It shares the foreground detector and wakes the loop with bounded proposals.
            let (resource_interrupt_proposal_tx, resource_interrupt_proposal_rx) =
                resource_interrupt_proposal_channel(Some(state.cycle_condvar.clone()));
            let mut resource_interrupt_window = ResourceInterruptActuationWindow::open();
            let mut resource_sentinel_shutdown = spawn_resource_sentinel_with_proposals(
                smc_reader.cache_arc(),
                pressure_collector.cache_arc(),
                state.resource_interrupt.clone(),
                state.frozen_state.clone(),
                state.stop.clone(),
                SentinelConfig::default(),
                fg_detector.clone(),
                Some(state.mach_qos.clone()),
                frozen_state_path.clone(),
                resource_interrupt_proposal_tx,
            );
            // Overflow guard: aprende de eventos OOM y ajusta thresholds adaptativamente.
            // SleepNotifier: IOKit pre-sleep callback — fires kIOMessageSystemWillSleep
            // ~30s before the kernel suspends, giving us time to release all SIGSTOP'd
            // processes so macOS can compress/Jetsam them normally during hibernation.
            // Without this, frozen processes block Jetsam's target set, leading to
            // more aggressive kills of other processes (widgets, extensions).
            let sleep_notifier = apollo_engine::engine::sleep_notifier::SleepNotifier::new();
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
            let mut specialist_feedback = daemon_cognitive_tick::SpecialistFeedbackState::default();

            // ZeroTune: seed with hardware meta-features on cold start.
            // Reduces warmup from 200→50 cycles by injecting domain knowledge priors.
            if !predictive_agent.is_active() && predictive_agent.total_cycles() == 0 {
                let ram_gb = apollo_engine::engine::sysctl_direct::read_u64("hw.memsize")
                    .unwrap_or(8 * 1024 * 1024 * 1024) as f64
                    / (1024.0 * 1024.0 * 1024.0);
                let cores =
                    apollo_engine::engine::sysctl_direct::read_u64("hw.ncpu").unwrap_or(4) as usize;
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
            // Restore companion graph from disk if a prior run persisted one.
            // load_companion_graph drops the field on absurd / corrupt data
            // (counter cap check + JSON sanity), so a poisoned file falls back
            // to a fresh empty graph rather than carrying garbage forward.
            if let Some(restored) = LearnedState::load_companion_graph(ls_path) {
                companion_graph = restored;
            }
            let mut persist_generations: u32 = 0;
            let mut last_restore_quality: Option<f64> = None;
            let mut restore_monitor = RestoreQualityMonitor::new();
            // FocusMarkov acceleration lease. Resolution follows an actual
            // foreground transition or a wall-clock deadline, never the next
            // daemon cycle alone.
            let mut last_markov_prethaw: Option<daemon_markov_tick::MarkovPrewarmLease> = None;
            let mut markov_shadow = daemon_markov_tick::MarkovShadowTracker::default();
            let mut markov_hit_count: u32 = 0;
            let mut markov_miss_count: u32 = 0;
            let reflex_state_json = HardPath::read_to_string_limited(
                std::path::Path::new(reflex_state_path()),
                64 * 1024,
            )
            .ok();
            let mut acceleration_lease = daemon_cycle_tail::ReflexBroker::restore(
                reflex_cfg.enabled,
                reflex_cfg.effective_shadow_cycles(),
                &parallel_runtime.build.compiled_profile,
                reflex_state_json.as_deref(),
            );
            // Restored pending trial skill from the previous run (if daemon crashed mid-trial).
            let mut restored_trial_skill: Option<(String, f64)> = None;
            // Restored arousal state — applied after arousal_state is declared below.
            let mut restored_arousal: Option<apollo_engine::engine::nars_belief::ArousalState> =
                None;
            let mut restored_process_baselines: Option<
                apollo_engine::engine::process_baseline::ProcessBaselineMap,
            > = None;
            let mut learnable_params = LearnableParams::default();
            let mut restored_meta_cognition: Option<
                apollo_engine::engine::meta_cognition::MetaCognition,
            > = None;
            let mut restored_local_consolidator: Option<LocalConsolidator> = None;
            let mut restored_microexperiment_lab = None;
            if let Some(learned) = LearnedState::load(ls_path) {
                persist_generations = learned.persist_generations;
                last_restore_quality = learned.last_restore_quality;
                restored_trial_skill = learned.pending_trial_skill.clone();
                restored_local_consolidator = learned.local_consolidator.clone();
                restored_microexperiment_lab = learned.microexperiment_lab.clone();
                if let Some(persisted) = learned.exploration_scheduler.clone() {
                    let (restored, disposition) =
                        apollo_engine::engine::exploration_scheduler::ExplorationScheduler::restore(
                            persisted,
                            apollo_engine::engine::exploration_scheduler::RestoreContext::local(
                                exploration_origin,
                                current_exploration_time(),
                            ),
                        );
                    exploration_scheduler = restored;
                    tracing::info!(?disposition, "restored bounded exploration scheduler");
                }
                if let Some(medallion_state) = learned.medallion_state.clone() {
                    let bronze_total = medallion_state.bronze_total;
                    let gold_total = medallion_state.gold_total;
                    learning_pipeline.restore_medallion(medallion_state);
                    tracing::info!(
                        bronze_total,
                        gold_total,
                        "restored medallion curation telemetry"
                    );
                }
                if let Some(telemetry_state) = learned.telemetry_medallion_state.clone() {
                    let bronze_total = telemetry_state.bronze_total;
                    let gold_total = telemetry_state.gold_total;
                    let actuator_bronze_total = telemetry_state.actuator_resolved_total;
                    let actuator_gold_total = telemetry_state.actuator_gold_total;
                    telemetry_medallion.restore(telemetry_state);
                    tracing::info!(
                        bronze_total,
                        gold_total,
                        actuator_bronze_total,
                        actuator_gold_total,
                        "restored World Model context telemetry"
                    );
                }
                // BUG-01: WAL fallback — if LearnedState didn't carry a pending trial
                // (e.g., daemon crashed before periodic persist), recover from WAL file.
                if restored_trial_skill.is_none() {
                    if let Ok(data) = apollo_engine::engine::types::HardPath::read_to_string_limited(
                        &pending_trial_path(is_root),
                        512,
                    ) {
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
                // Stash MetaCognition snapshot for restore after CognitiveState::new()
                // is constructed below. Cloned here because `learned` is consumed by
                // the upcoming apply() call.
                restored_meta_cognition = learned.meta_cognition.clone();
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
                    &mut maintenance_state,
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
            decision_ledger.seed_high_water(telemetry_medallion.decision_id_high_water());
            // Temporal app predictor: time-of-day aware app launch prediction.
            // Shin et al. 2012 — temporal patterns predict app launches with ~80% accuracy.
            // Combined with Markov chain for 85% top-3 accuracy (Baeza-Yates et al. 2015).
            let mut temporal_predictor =
                apollo_engine::engine::temporal_predictor::TemporalPredictor::new(
                    std::path::PathBuf::from(temporal_histograms_path()),
                );
            // Adaptive Page Reclaim: pressure-driven file cache purging.
            // Jiang & Zhang 2005 — proactive reclaim of low-IRR pages outperforms
            // reactive LRU eviction by 20-40% in cache hit ratio.
            let mut page_reclaim = apollo_engine::engine::page_reclaim::PageReclaim::new(is_root);

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
            let mut last_ioreport: Option<apollo_engine::engine::ioreport::IOReportSnapshot> = None;
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
                apollo_engine::engine::llm_inference_mode::LlmInferenceDetector::new();
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
                std::fs::read_to_string(apollo_engine::engine::daemon_helpers::metrics_path())
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .map(|v| {
                        let p = v["memory_pressure"].as_f64().unwrap_or(1.0);
                        let swap = v["swap_used_bytes"]
                            .as_f64()
                            .unwrap_or(99.0 * 1024.0 * 1024.0 * 1024.0)
                            / (1024.0 * 1024.0 * 1024.0);
                        (p, swap)
                    })
                    .unwrap_or((1.0, 99.0));
            // Spotlight management removed entirely 2026-04-30. Apollo no
            // longer toggles `mdutil` because `-i off` aborts indexing rather
            // than pausing it, causing repeated restart-from-zero cycles that
            // prevent the index from ever completing. macOS manages Spotlight
            // natively; Apollo handles pressure via other mechanisms (freezes,
            // throttles, paging hints).
            let _ = (startup_pressure, startup_swap_gb);

            // Consecutive cycles where swap_delta > 1MB/s. Fed to RL meta-gate
            // to veto Raise1pp during sustained swap growth (see rl_threshold.rs).
            let mut swap_growth_streak: u32 = 0;

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
            let smc_direct = apollo_engine::engine::smc_direct::SmcDirectReader::new();
            let mut last_smc: Option<apollo_engine::engine::smc_direct::SmcSnapshot> = None;
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
            let mut kpc_reader = apollo_engine::engine::kpc_counters::KpcReader::new();
            state.metrics.lock_recover().metrics.kpc_available = kpc_reader.available;
            if kpc_reader.available {
                println!("[kpc] Hardware performance counters active");
            } else {
                println!(
                    "[kpc] KPC counters unavailable (kernel policy, entitlement, or counter owner)"
                );
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
                apollo_engine::engine::contention_detector::ContentionDetector::new();

            // ── Window/App Lifecycle Sensor ─────────────────────────────────
            // Diff-based: tracks app terminated/launched, browser tab delta,
            // foreground changes. Works as root daemon (no GUI session needed).
            // [GoF Observer Pattern via cycle-to-cycle process diff]
            let mut window_sensor = apollo_engine::engine::window_sensor::WindowSensor::new();

            // ── Fluidity Intelligence ────────────────────────────────────────
            // Tracks WindowServer CPU spike (window resize/move), app launches,
            // GPU render load → composite fluidity score 0–1.
            // [Jain 1991] EMA composite scoring, [Welch & Bishop 2006] Kalman prediction
            let mut fluidity_state = apollo_engine::engine::fluidity::FluidityState::new();

            // ── Chromium Renderer Manager ────────────────────────────────────
            // Manages RAM/CPU for ALL Chromium/Electron renderer subprocesses:
            // Brave, Chrome, Edge, Arc, Vivaldi, Slack, Discord, Code, Cursor, etc.
            // Tier 1: E-core demotion (safe). Tier 2: SIGSTOP idle renderers (guarded).
            // [Denning 1968] Working Set | [Jones 2011] Chromium Multi-Process Architecture
            let mut chromium_mgr = apollo_engine::engine::chromium_manager::ChromiumManager::new();

            // ── Rosetta AOT Monitor ─────────────────────────────────────────
            // Watches /var/db/oah/ for write events → suppress freezing oahd.
            let mut rosetta_monitor = apollo_engine::engine::rosetta_monitor::RosettaMonitor::new();
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
            // Reused across cycles. Its internal revisions avoid rebuilding
            // unchanged causal and actuator maps on the daemon hot path.
            let mut world_model = apollo_engine::engine::world_model::WorldModel::default();
            let mut local_consolidator = restored_local_consolidator.unwrap_or_default();
            if local_consolidator.restore_for_origin(installation_id, hierarchy_hardware) {
                outcome_tracker.drift_detector.clear_hierarchy_beliefs();
            }
            world_model.attach_local_consolidation(
                local_consolidator.view_for_installation(installation_id),
            );
            let mut gpu_imagination =
                apollo_engine::engine::gpu_imagination::GpuImaginationWorker::spawn_metal_only();
            let mut gpu_context_epoch = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(1)
                ^ u64::from(std::process::id());
            let mut overhead_governor = adaptive_overhead::AdaptiveOverheadGovernor::default();
            let mut value_scheduler_runtime =
                value_scheduler_tick::ValueSchedulerRuntime::new(runtime_capability_graph.revision);
            value_scheduler_runtime.connect_handoff_jobs(&[
                apollo_engine::engine::value_scheduler::JobId::HardwarePrediction,
            ]);
            let mut webflow_runtime = webflow_tick::WebFlowRuntime::new(
                apollo_engine::engine::webflow_controller::WebFlowRolloutPhase::Shadow,
            );
            let mut networkflow_runtime = networkflow_tick::NetworkFlowRuntime::new();
            let mut heterogeneous_runtime: Option<heterogeneous_tick::HeterogeneousRuntime> = None;
            let mut heterogeneous_workload_id = 0_u64;
            let mut microexperiment_runtime = microexperiment_tick::MicroexperimentRuntime::new(
                exploration_origin,
                restored_microexperiment_lab,
            );
            // Feed-forward pressure relief counter [Hellerstein 2004].
            // Set to N when tabs close / heavy app terminates; decrements each cycle.
            // While > 0, reactor_weight is reduced (anticipate pressure drop).
            let mut window_relief_cycles: u32 = 0;
            // Window session intelligence — updated each cycle by window_sensor.tick().
            // Declared here so they're accessible in both the proc_snaps block and
            // the reactor_weight section which runs after signal_digest computation.

            let mut win_session_phase;
            let mut win_workload_intent;
            let mut win_pressure_floor: f64;
            // Current hour/weekday for temporal headroom; unconditionally set each cycle
            // inside the Utc::now() block at line ~1547, then optionally refined.
            let mut temporal_hour: u8;
            let mut temporal_weekday: u8;
            // Build progress tracker: estimates cargo build completion from
            // rustc process-count dynamics. Informs reactor_weight policy.
            use apollo_engine::engine::build_tracker::BuildTracker;
            let mut build_tracker = BuildTracker::new();
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
                apollo_engine::engine::system_log_ingester::SystemLogIngester::new();
            log_ingester.start_background();
            // B.6 gap fix (2026-06-10): zombie/dead-weight hunter. Classifies
            // GhostHelper / MemoryHoarder / WakeupBurner / TrueZombie / Orphan
            // from the per-cycle hunt_snaps and emits CONSERVATIVE actions
            // (jetsam idle-band hints + non-aggressive throttles — never kills;
            // the kernel keeps kill authority). Evaluated every 30 cycles —
            // dead weight accumulates over minutes, not milliseconds.
            // Persistent across cycles for its 3-cycle confirmation history.
            let mut zombie_hunter = apollo_engine::engine::zombie_hunter::ZombieHunter::new();
            // Minimum cycle floor: prevent CPU burn from rapid condvar wakeups.
            let mut last_cycle_end = Instant::now() - Duration::from_secs(1);
            // Prev-cycle smoothed pressure, used by the high-pressure cycle-rate
            // gate (Change C) and the enrichment cache gate (Changes A+B) to
            // throttle Apollo's own hot path under stress. 0.0 on first
            // cycle = treated as low pressure (300ms floor, fresh enrich).
            let mut prev_pressure_smooth: f64 = 0.0;
            // Planner Phase-1 pre-arm cache: re-read planner_hints.json only
            // every 15 cycles (~5-45s depending on cadence) — never file I/O
            // on every loop iteration. Stale/missing file reads as false.
            let mut planner_spike_imminent_cached = false;
            // Phase 4.2 WIRED (Sprint 10, 2026-05-16) — track thermal-state
            // transitions across cycles so we can emit exactly one
            // `record_external_event(ThermalThrottle, ...)` per upward
            // crossing into Moderate+ regime. Without this debounce the
            // causal graph would see N "external events" per N cycles in
            // a sustained throttle, swamping the attribution model.
            let mut prev_thermal_throttling: bool = false;
            // Sprint 12 Convergence #4 (2026-05-17). Per-cycle delta of
            // `scorer_override_rejects_total`. When the delta is positive AND
            // the causal graph has a recent ThermalThrottle blame inside
            // EXTERNAL_BLAME_WINDOW, we bump
            // `causal_thermal_scorer_override_alignments_total` — the
            // unique signal that learned policy is misbehaving under
            // thermal stress (vs benign disagreement).
            let mut prev_scorer_override_rejects: u64 = 0;
            // Batch buffer: accumulate N push messages before a single write syscall.
            // macOS Unix socket SO_SNDBUF = 8192 bytes. Batch=16×~64=~1KB stays well
            // under the 8KB limit so write_all never blocks. Empirically optimal.
            let mut dry_run_batch: Vec<u8> = Vec::with_capacity(1024);
            let mut dry_run_batch_count: u32 = 0;
            const DRY_RUN_BATCH_SIZE: u32 = 16;
            // Direct kernel TCP deltas are sampled at 1 Hz and shared by the
            // universal flow controller, governor, and metrics.
            let mut last_netstat_tick = Instant::now() - Duration::from_secs(10);
            let mut last_network_flow_stats =
                apollo_engine::engine::network_monitor::TcpStats::default();
            // B.2 replayd gate (2026-06-09): TTL-cached (6s) proc-table scan
            // for active screen capture (replayd / screencaptureui /
            // ScreenSharingAgent). Feeds the SysctlGovernor realtime gate so
            // screen shares without full-duplex audio still inhibit TCP
            // buffer scale-downs (post-screen-share whipsaw fix).
            let mut screen_capture_cache =
                apollo_engine::engine::realtime_signals::ScreenCaptureCache::new();
            // Context-switch burst detector (TDA-aware).
            let mut ctx_switch_times: VecDeque<Instant> = VecDeque::new();
            // Direct scheduler probe samples are produced every ~10 cycles.
            // Keep the last real sample for the responsiveness estimator and a
            // bounded window for an explicitly-labelled scheduler p95.
            let mut last_schedule_jitter_us: u64 = 0;
            let mut schedule_jitter_samples_us: VecDeque<u64> = VecDeque::with_capacity(120);
            let mut last_fg_name: Option<String> = None;
            // Cached user context assertion state — assertion signals are collected
            // every N cycles (amortised); between polls, last-known values are carried
            // forward to prevent the freeze gate from flickering on/off every cycle.
            // [Cook et al. 2019] "Caching volatile state in reactive systems"
            let mut last_user_assertions: (bool, bool, bool) = (false, false, false); // (sleep_assert, call, audio)
                                                                                      // Phase 0c cache: last ioreg HIDIdleTime sample + when. Lets
                                                                                      // compute_user_context interpolate idle_secs between every-10-cycle
                                                                                      // ioreg subprocess spawns. ~25 ms saved per intermediate cycle.
            let mut last_idle_sample: Option<(f64, Instant)> = None;
            // Phase 5.1 wiring (2026-05-16) — last cycle's UserContext, carried
            // forward so `apply_specialist_voting` can compute the
            // user-presence multiplier BEFORE `compute_user_context` runs
            // this cycle (the daemon's ordering invariant places
            // specialist voting strictly *before* user_context). The
            // one-cycle lag is acceptable: at the 80 ms cycle period a user
            // who was typing 80 ms ago is overwhelmingly still typing now.
            // First cycle uses the `UserContext::default()` "safe/unknown"
            // values → no suppression, identical to pre-wiring behaviour.
            let mut last_user_context_for_voting: apollo_engine::engine::user_context::UserContext =
                apollo_engine::engine::user_context::UserContext::default();
            // Track previous cycle's package_watts for RL power-reduction reward.
            let mut prev_package_watts: Option<f64> = None;
            // Track previous cycle's workload for onset detection (build-onset-proactive).
            let mut prev_workload_mode: WorkloadMode = WorkloadMode::Idle;
            // Work-session hysteresis: latch "work mode" through the user's dev
            // session so the profile governor stays AggressiveRoot for a grace
            // window after the last dev activity, decaying gracefully instead of
            // snapping back to Idle the instant a build finishes / Claude Code
            // closes. Per-cycle in-memory state (not persisted), like
            // prev_workload_mode above. See engine::work_session.
            let mut work_session = WorkSession::new();
            let mut workload_phase_tracker =
                apollo_engine::engine::workload_phase::WorkloadPhaseTracker::new(Instant::now());
            // Affective arousal EMA: global system-wide stress level ∈ [0,1].
            // Drives Yerkes-Dodson adaptive recalibration threshold in learning_tick.
            // Restored from learned_state.json if available — preserves crisis context
            // across restarts. [Yerkes & Dodson 1908]
            let mut arousal_state = restored_arousal.unwrap_or_default();
            // Neurocognitive state: 8-module cognitive pipeline wired into hot loop.
            // [CognitiveRewardBus, MetaCognition, SelfRewardingEvaluator, EpistemicUncertainty,
            //  ReptileMeta, AdversarialProbe, ProactiveDrift, CognitiveHealthScore]
            let mut cognitive_state = cognitive_tick::CognitiveState::new();
            // Restore MetaCognition from learned_state if present. Preserves
            // per-subsystem calibration history across daemon restarts so the
            // first ~50 cycles after a restart aren't blindly optimistic.
            if let Some(mc) = restored_meta_cognition {
                cognitive_state.meta_cognition = mc;
                tracing::info!(
                    target: "apollo.meta_cognition",
                    calibration_error = cognitive_state.meta_cognition.calibration_error,
                    humble_mode = cognitive_state.meta_cognition.humble_mode,
                    "restored MetaCognition from learned_state"
                );
            }
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
            //
            // CRITICAL fix (2026-05-16): unify with the global `LSE_COUNTERS`
            // static. Previously this was `Arc::new(LockFreeMetrics::new())`
            // — a separate instance — which silently broke every `inc_*` call
            // routed through `LSE_COUNTERS` (Phase 2 god-lock decomp, Phase 3.1
            // skill_aware, Phase 5.1 user_presence, Phase 5.3 rationale, GAP 2
            // reactor sticky, GAP 6 specialist purge, etc). Those wrote to the
            // global, but `sync_from_lockfree` snapshot-read the local Arc, so
            // the counters never reached runtime_metrics.json. Diagnostic on
            // 2026-05-16 confirmed two distinct addresses (`0x100ee2a58` static
            // vs `0x95385c7d8` Arc). Aliasing here closes the duplicate-state
            // gap with a single line.
            //
            // mlock pinning removed: the static lives in BSS (zero-init at
            // process start, kernel keeps it resident under normal pressure).
            let lf_metrics: &'static LockFreeMetrics =
                &apollo_engine::engine::lse_counters::LSE_COUNTERS;
            // Phase B1.5 — wire RecentlyApplied restore_status to lf_metrics telemetry.
            // Exactly one of the 5 counters is set to a non-zero value per startup,
            // letting NotebookLM debrief distinguish "persistence helps" vs "always
            // starts empty". [Inbox Pattern — 1001 patterns slide 42]
            {
                use apollo_engine::engine::recently_applied::RestoreStatus;
                use std::sync::atomic::Ordering;
                match recently_applied_restore_status {
                    RestoreStatus::Missing => {
                        lf_metrics
                            .restore_status_missing
                            .store(1, Ordering::Relaxed);
                    }
                    RestoreStatus::RestoredN(n) => {
                        lf_metrics
                            .restore_status_restored_n
                            .store(n as u64, Ordering::Relaxed);
                    }
                    RestoreStatus::DiscardedCorrupt => {
                        lf_metrics
                            .restore_status_discarded_corrupt
                            .store(1, Ordering::Relaxed);
                    }
                    RestoreStatus::DiscardedClockDelta => {
                        lf_metrics
                            .restore_status_discarded_clock_delta
                            .store(1, Ordering::Relaxed);
                    }
                    RestoreStatus::DiscardedBootCrossed => {
                        lf_metrics
                            .restore_status_discarded_boot_crossed
                            .store(1, Ordering::Relaxed);
                    }
                }
                tracing::info!(
                    target: "apollo.recently_applied",
                    status = ?recently_applied_restore_status,
                    "restore status telemetry wired"
                );
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
            let mut ais_worker = metrics_reporter::AisRuntimeWorker::spawn();
            // Sprint 12 perf-fix (2026-05-30). Cross-cycle memoization
            // slot for `companion_of_fg_pids`. See [`CompanionFgCache`]
            // for the invalidation contract. Stays None until the first
            // cycle populates it; per-cycle policy build then borrows
            // `&cache.pids` directly, skipping rebuild + allocation.
            let mut companion_fg_cache: Option<CompanionFgCache> = None;

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
                let mut cycle_decision_events =
                    apollo_engine::engine::decision_ledger::CycleDecisionEvents::default();
                {
                    let pressure = pressure_collector.latest().memory_pressure;
                    let mut executor = daemon_resource_interrupt_tick::ResourceInterruptExecutor {
                        state: &state,
                        caps: &caps,
                        journal_path: &journal_path,
                        frozen_state_path: &frozen_state_path,
                        dry_run,
                        cycle: cycle_count,
                        memory_pressure: pressure,
                    };
                    let events = executor.execute_batch(
                        &resource_interrupt_window,
                        resource_interrupt_proposal_rx.drain(),
                    );
                    cycle_decision_events.extend_buffer(&events);
                }
                let async_completion_stats =
                    async_commands.drain_decision_events(cycle_count, &mut cycle_decision_events);
                if async_completion_stats.maintenance_purges_applied > 0
                    || async_completion_stats.cli_purges_applied > 0
                {
                    lf_metrics.maintenance_purge_total.fetch_add(
                        async_completion_stats
                            .maintenance_purges_applied
                            .saturating_add(async_completion_stats.cli_purges_applied),
                        Ordering::Relaxed,
                    );
                }

                // Decrement freeze cooldown counters once per cycle.
                // [Nygard 2018] §8.5 circuit-breaker hold-down decay.
                state.freeze_cooldown.lock_recover().tick();

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
                // Change C (2026-05-16): under sustained high pressure (>0.80
                // smoothed) the daemon's own hot path becomes a contributor
                // to thrashing (proc_taskinfo per-PID storm × 2.5 Hz). Drop
                // cycle rate to 1 Hz when stressed so Apollo's CPU footprint
                // doesn't worsen the very pressure it's trying to mitigate.
                // Fast-tick bypasses (kqueue Critical / hw_predictor events).
                let high_pressure_throttle = prev_pressure_smooth > 0.80;
                // Evolve 2026-06-10 (battery footprint): idle-aware cadence.
                // When the user has been idle for minutes on battery, drop
                // the daemon from 3.3 Hz to 0.2-0.33 Hz — the per-PID
                // enrichment storm + ioreg poll were pure energy drain with
                // no responsiveness benefit while away. Crises still preempt:
                // is_fast_tick forces 0 here, and the reactor condvar wakes
                // the loop immediately on kqueue Critical regardless of floor.
                if cycle_count % 15 == 0 {
                    let was = planner_spike_imminent_cached;
                    planner_spike_imminent_cached =
                        apollo_engine::engine::planner::read_spike_imminent(
                            &planner_hints_path_for_cadence,
                            90,
                            chrono::Utc::now(),
                        );
                    if planner_spike_imminent_cached && !was {
                        apollo_engine::engine::lse_counters::LSE_COUNTERS.inc_planner_prearm();
                        tracing::info!(
                            "planner-prearm: forecast spike <=120s — pinning 300ms cadence"
                        );
                    }
                }
                let min_inter_cycle_ms = if dry_run || is_fast_tick {
                    0
                } else {
                    apollo_engine::engine::power_management::adaptive_cycle_floor_ms(
                        apollo_engine::engine::power_management::CadenceInputs {
                            on_battery: power_mgr.is_on_battery(),
                            battery_low,
                            pressure_smooth: prev_pressure_smooth,
                            idle_secs: last_user_context_for_voting.idle_secs.max(0.0) as u64,
                            high_pressure: high_pressure_throttle,
                            planner_spike_imminent: planner_spike_imminent_cached,
                        },
                    )
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
                    let dram_bp_trigger =
                        dram_bw_pct > 80.0 || (dram_bw_pct == 0.0 && prev_entropy_anomaly > 2.0);
                    if dram_bp_trigger {
                        let capped = (action_queue.max_per_cycle() / 2).max(1);
                        action_queue.set_max_per_cycle(capped);
                    }
                }

                // In dry-run mode skip the kill-switch stat() syscall — tests never
                // create the kill-switch file, so this check is pure overhead.
                // [Nygard 2018 §5] eliminate non-observable work from benchmark path.
                if !dry_run && Path::new(kill_switch_path()).exists() {
                    if let Some(lease) = last_markov_prethaw.take() {
                        let report = daemon_markov_tick::release_markov_prewarm_recorded(
                            lease,
                            &state,
                            cycle_count,
                            &mut acceleration_lease,
                        );
                        cycle_decision_events.extend_buffer(&report.decision_events);
                    }
                    cycle_decision_events.extend_buffer(
                        &daemon_cycle_tail::release_acceleration_lease_recorded(
                            &mut acceleration_lease,
                            &state,
                            cycle_count,
                        ),
                    );
                    exploration_scheduler.cancel_active(
                        apollo_engine::engine::exploration_scheduler::TerminalDiagnostic::KillSwitch,
                    );
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
                            metrics.metrics.iokit_p_cluster_temp = (!hw.temps_estimated)
                                .then_some(hw.temps.p_cluster_celsius)
                                .flatten();
                            metrics.metrics.iokit_e_cluster_temp = (!hw.temps_estimated)
                                .then_some(hw.temps.e_cluster_celsius)
                                .flatten();
                            metrics.metrics.iokit_package_watts = hw.power.package_watts;
                        }
                        metrics.metrics.thermal_state = metrics.thermal_state.clone();
                    }
                    last_cycle_end = Instant::now();
                    // Phase-1 stall-candidate F1 (audit 2026-06-24): emit a warn
                    // BEFORE the 5s sleep so we can correlate with the watchdog's
                    // EARLY-WARNING / STALL events. Zero behavior change — only
                    // log emission. [F1 HIGH] per /Users/eduardocortez/hardening-audit-2026-06-24/main-loop-stall-candidates.md
                    tracing::warn!(
                        target: "apollo.stall_candidate",
                        cycle_count,
                        "stall_candidate_F1: interruptible 5s pause in kill-switch branch"
                    );

                    // The resource sentinel remains active while ordinary
                    // optimization is paused. Resolve proposals emitted after
                    // the entry drain, then settle this cycle's exact receipts
                    // before waiting. A proposal arriving after this drain sets
                    // the condvar predicate and starts the next cycle
                    // immediately instead of waiting five seconds.
                    {
                        let pressure = pressure_collector.latest().memory_pressure;
                        let mut executor =
                            daemon_resource_interrupt_tick::ResourceInterruptExecutor {
                                state: &state,
                                caps: &caps,
                                journal_path: &journal_path,
                                frozen_state_path: &frozen_state_path,
                                dry_run,
                                cycle: cycle_count,
                                memory_pressure: pressure,
                            };
                        let events = executor.execute_batch(
                            &resource_interrupt_window,
                            resource_interrupt_proposal_rx.drain(),
                        );
                        cycle_decision_events.extend_buffer(&events);
                    }
                    if let Some(coordinated) =
                        apollo_engine::engine::decision_ledger::coordinated_action_event(
                            &cycle_decision_events,
                            cycle_count,
                        )
                    {
                        cycle_decision_events.push(coordinated);
                    }
                    let overflow_summary_emitted =
                        cycle_decision_events.seal_overflow_summary(cycle_count);
                    let dropped_decision_events = cycle_decision_events.dropped_total();
                    if dropped_decision_events > 0 {
                        tracing::warn!(
                            dropped = dropped_decision_events,
                            "decision event pause buffer reached its bounded capacity"
                        );
                        let mut metrics = state.metrics.lock_recover();
                        metrics.metrics.decision_event_drops_total = metrics
                            .metrics
                            .decision_event_drops_total
                            .saturating_add(dropped_decision_events);
                        metrics.metrics.decision_event_overflow_summaries_total = metrics
                            .metrics
                            .decision_event_overflow_summaries_total
                            .saturating_add(u64::from(overflow_summary_emitted));
                    }
                    let resolved_decisions =
                        decision_ledger.ingest_cycle_events(&mut cycle_decision_events);
                    telemetry_medallion.stage_decision_episodes(&resolved_decisions);

                    wait_for_cycle_signal(&state.cycle_condvar, Duration::from_secs(5));
                    continue;
                }

                let cycle_start = Instant::now();
                let _t_sense_start = cycle_start;

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
                        }
                    }
                    last_cycle_end = Instant::now();
                    lf_metrics.set_cycle_time_us(cycle_start.elapsed().as_micros() as u64);
                    lf_metrics.commit();

                    // Periodic sync for observability in dry-run mode.
                    if cycle_count.is_multiple_of(5) {
                        let snap = lf_metrics.snapshot();
                        state.metrics.lock_recover().sync_from_lockfree(&snap);
                    }
                    continue;
                }

                // Staggered wake unfreeze: drain one batch per cycle.
                // Extracted to daemon_wake_unfreeze::run_wake_unfreeze (Wave 25).
                // [Nygard 2018] bulkhead: spread SIGCONT across cycles; shrink under
                // thermal/swap-velocity stress to avoid 1-3GB decompression spike.
                let wake_unfreeze_output = daemon_wake_unfreeze::run_wake_unfreeze(
                    &mut wake_unfreeze_queue,
                    &mut wake_thaw_pids,
                    &state,
                    &pressure_collector,
                    &frozen_state_path,
                    cycle_count,
                );
                cycle_decision_events.extend_buffer(&wake_unfreeze_output.decision_events);

                // Mark reactor as stalled only if the reactor thread has sent
                // zero pulses after 60 s — that means the thread itself died,
                // not just that the system has been quiet.
                if daemon_start.elapsed() > Duration::from_secs(60) {
                    let pulses = apollo_engine::engine::lse_counters::LSE_COUNTERS
                        .snapshot()
                        .reactor_pulses;
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
                    if cycle_count.is_multiple_of(60) {
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
                let (grace_active, wake_just_detected) = daemon_wake_handler::run_wake_tick(
                    &state,
                    &mut signal_intel,
                    &mut outcome_tracker,
                    &mut wake_unfreeze_queue,
                    &mut display_turbo,
                    &wake_state_path,
                );
                if wake_just_detected {
                    gpu_imagination.invalidate_for_wake();
                    gpu_context_epoch = gpu_context_epoch.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    world_model.invalidate_gpu_advice(gpu_context_epoch);
                    if let Some(lease) = last_markov_prethaw.take() {
                        let report = daemon_markov_tick::release_markov_prewarm_recorded(
                            lease,
                            &state,
                            cycle_count,
                            &mut acceleration_lease,
                        );
                        cycle_decision_events.extend_buffer(&report.decision_events);
                    }
                    cycle_decision_events.extend_buffer(
                        &daemon_cycle_tail::release_acceleration_lease_recorded(
                            &mut acceleration_lease,
                            &state,
                            cycle_count,
                        ),
                    );
                    exploration_scheduler.cancel_active(
                        apollo_engine::engine::exploration_scheduler::TerminalDiagnostic::Wake,
                    );
                    maintenance_state.observe_wake();
                }
                if sleep_notifier.is_sleeping() {
                    if let Some(lease) = last_markov_prethaw.take() {
                        let report = daemon_markov_tick::release_markov_prewarm_recorded(
                            lease,
                            &state,
                            cycle_count,
                            &mut acceleration_lease,
                        );
                        cycle_decision_events.extend_buffer(&report.decision_events);
                    }
                    cycle_decision_events.extend_buffer(
                        &daemon_cycle_tail::release_acceleration_lease_recorded(
                            &mut acceleration_lease,
                            &state,
                            cycle_count,
                        ),
                    );
                    exploration_scheduler.cancel_active(
                        apollo_engine::engine::exploration_scheduler::TerminalDiagnostic::Cancelled,
                    );
                }

                if wake_just_detected || cycle_count.is_multiple_of(256) {
                    let reprobed = platform_adapter.probe();
                    if reprobed.revision != runtime_capability_graph.revision {
                        runtime_capability_graph = reprobed;
                    }
                }

                // Display-Off Turbo: Android Doze-like power management.
                // Extracted to daemon_turbo_manager::run_turbo_tick().
                // [Nygard 2018] bulkhead: bound blast radius of display state transitions.
                let turbo_output = daemon_turbo_manager::run_turbo_tick(
                    &mut display_turbo,
                    &state,
                    &fg_detector,
                    &collector,
                    &pressure_collector,
                    &frozen_state_path,
                    &mut stability_oracle,
                    power_mgr.is_on_battery(),
                    cycle_count,
                );
                cycle_decision_events.extend_buffer(&turbo_output.decision_events);

                let overhead_input = {
                    let metrics = state.metrics.lock_recover();
                    adaptive_overhead::OverheadInput {
                        p95_cycle_ms: metrics.metrics.p95_cycle_ms,
                        reason_avg_ms: metrics.metrics.stage_reason_avg_ms,
                        memory_pressure: metrics.metrics.memory_pressure,
                        fluidity_degraded: metrics.metrics.fluidity_degraded,
                        holt_winters_avg_ms: metrics.metrics.stage_reason_holtwinters_avg_ms,
                        page_reclaim_avg_ms: metrics.metrics.stage_reason_pagereclaim_avg_ms,
                    }
                };
                let overhead_budget = overhead_governor.observe(overhead_input);
                {
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.apollo_overhead_level =
                        overhead_budget.level.as_str().to_string();
                    metrics.metrics.apollo_overhead_full_refresh_cadence =
                        overhead_budget.full_refresh_cadence;
                    metrics.metrics.apollo_overhead_reason_budget_ms =
                        overhead_budget.optional_reason_budget_ms;
                    metrics.metrics.apollo_overhead_speculation_allowed =
                        overhead_budget.allow_speculation;
                    metrics.metrics.apollo_overhead_sensor_interval_secs =
                        overhead_budget.sensor_interval_secs;
                    metrics.metrics.apollo_overhead_holt_cadence =
                        overhead_budget.holt_winters.cadence;
                    metrics.metrics.apollo_overhead_page_reclaim_cadence =
                        overhead_budget.page_reclaim.cadence;
                    metrics.metrics.apollo_overhead_constrained_cycles = metrics
                        .metrics
                        .apollo_overhead_constrained_cycles
                        .saturating_add(u64::from(
                            overhead_budget.level == adaptive_overhead::OverheadLevel::Constrained,
                        ));
                }
                smc_reader.set_interval(Duration::from_secs(overhead_budget.sensor_interval_secs));

                // Adaptive snapshot: use lightweight path (no disk/net refresh) every cycle
                // except for a governor-selected full-refresh heartbeat.
                // Disk/network data from sysinfo is not consumed on the hot path — the
                // network monitor and sysctl governor read directly from sysctl/netstat.
                // Dropping the pressure gate removes ~15-25ms of disk/net I/O at 0.70+ pressure
                // where the old 0.40 threshold never fired anyway.
                let use_light = !cycle_count.is_multiple_of(overhead_budget.full_refresh_cadence);
                let (mut snapshot, refresh_duration) = if dry_run && use_light {
                    collector.collect_snapshot_no_process_refresh()
                } else if use_light {
                    collector.collect_snapshot_light(pressure_collector.latest().memory_pressure)
                } else {
                    collector.collect_snapshot()
                };
                lf_metrics.set_refresh_duration_us(refresh_duration.as_micros() as u64);
                // Phase 0b stage timing: sense complete, reason starts.
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::Sense,
                    _t_sense_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );
                let _t_reason_start = Instant::now();
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
                // Build once early enough for both universal network sensing
                // and the existing Markov/acceleration paths.
                let _t_tree_start = Instant::now();
                let process_tree = daemon_process_collector::build_process_tree(&collector);
                let process_tree_nanos = _t_tree_start.elapsed().as_nanos();
                let context_summary = state.latest_context_summary();
                let context_audio_active = context_summary.and_then(|summary| {
                    use apollo_engine::engine::context_agent::TriState;
                    match summary.audio_output {
                        TriState::Yes => Some(true),
                        TriState::No => Some(false),
                        TriState::Unknown => None,
                    }
                });
                let context_permissions_bits = context_summary.map_or(0, |summary| {
                    (summary.permissions.screen_capture as u16)
                        | ((summary.permissions.microphone as u16) << 2)
                        | ((summary.permissions.accessibility as u16) << 4)
                        | ((summary.permissions.input_monitoring as u16) << 6)
                });
                let interaction_active = !foreground_idle
                    || context_summary.is_some_and(|summary| summary.interaction_q >= 0.05);
                if last_netstat_tick.elapsed() >= Duration::from_secs(1) {
                    last_network_flow_stats = network_monitor.tick();
                    last_netstat_tick = Instant::now();
                }
                let network_traffic = last_network_flow_stats.flow_sample();
                let tcp_sample_age_ms = last_netstat_tick
                    .elapsed()
                    .as_millis()
                    .min(u128::from(u64::MAX)) as u64;
                let webflow_now_ms = apollo_engine::engine::webflow_types::webflow_monotonic_ms();
                let webflow_events = apollo_engine::engine::webflow_types::drain_process_webflow(
                    apollo_engine::engine::webflow_types::MAX_WEBFLOW_EVENTS_PER_CYCLE,
                );
                let foreground_browser = foreground_app
                    .as_deref()
                    .is_some_and(webflow_tick::is_supported_browser);
                let traffic_active = network_traffic
                    .send_bps
                    .saturating_add(network_traffic.recv_bps)
                    >= apollo_engine::engine::network_flow::GENERIC_FLOW_MIN_TRAFFIC_BPS
                    || network_traffic.new_connections > 0;
                let should_probe_foreground_sockets = webflow_events.is_empty()
                    && interaction_active
                    && traffic_active
                    && foreground_pid.is_some();
                let foreground_socket_active = if should_probe_foreground_sockets {
                    let candidates = networkflow_tick::bounded_family_candidates(
                        &process_tree,
                        foreground_pid.expect("probe requires foreground PID"),
                    );
                    !apollo_engine::engine::activity_sensor::pids_with_open_sockets(&candidates, 1)
                        .is_empty()
                } else {
                    false
                };
                let browser_socket_active = foreground_browser
                    && foreground_socket_active
                    && webflow_runtime.needs_daemon_inference_probe(webflow_now_ms);
                let low_power = state
                    .hardware
                    .lock_recover()
                    .last_hw_snapshot
                    .as_ref()
                    .is_some_and(
                    apollo_engine::engine::iokit_sensors::IOKitSensorReader::is_battery_critical,
                );
                let sleeping = sleep_notifier.is_sleeping();
                let kill_switch = Path::new(kill_switch_path()).exists();
                let thermal_constrained = matches!(
                    snapshot
                        .pressure
                        .thermal_level
                        .to_ascii_lowercase()
                        .as_str(),
                    "serious" | "high" | "critical"
                );
                let (webflow_control_p95_ms, webflow_failures, webflow_rollback_failures) = {
                    let metrics = state.metrics.lock_recover();
                    (
                        metrics.metrics.p95_cycle_ms,
                        metrics.metrics.reflex_failed_total,
                        metrics.metrics.reverts_failed,
                    )
                };
                let webflow_output = webflow_runtime.tick(webflow_tick::WebFlowCycleInput {
                    now_ms: webflow_now_ms,
                    events: &webflow_events,
                    foreground_browser,
                    foreground_identity_available: foreground_pid.is_some(),
                    interaction_active,
                    browser_socket_active,
                    pressure_constrained: snapshot.pressure.memory_pressure >= 0.85,
                    thermal_constrained,
                    low_power,
                    sleeping,
                    kill_switch,
                    session_revision: context_summary.map_or(1, |summary| summary.daemon_epoch),
                    control_p95_ms: webflow_control_p95_ms,
                    profile_matches: parallel_runtime.build.expected_profile
                        == parallel_runtime.build.effective_profile,
                    failures_total: webflow_failures,
                    rollback_failures_total: webflow_rollback_failures,
                    protected_actions_total: 0,
                });
                let networkflow_output =
                    networkflow_runtime.tick(networkflow_tick::NetworkFlowCycleInput {
                        now_ms: webflow_now_ms,
                        session_revision: context_summary.map_or(1, |summary| summary.daemon_epoch),
                        foreground_pid,
                        interaction_active,
                        foreground_socket_active,
                        traffic: network_traffic,
                        tcp_sample_age_ms,
                        exact_web_active: webflow_output.observation.active_navigations > 0,
                        pressure_constrained: snapshot.pressure.memory_pressure >= 0.85,
                        thermal_constrained,
                        low_power,
                        sleeping,
                        kill_switch,
                    });
                {
                    use apollo_engine::engine::webflow_types::WebFlowSource;
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.webflow_mode = match webflow_output.observation.source {
                        Some(WebFlowSource::ExtensionVitals) => "vitals",
                        Some(WebFlowSource::ExtensionLifecycle) => "lifecycle",
                        Some(WebFlowSource::DaemonInference) => "inferred",
                        None => "unavailable",
                    }
                    .to_string();
                    metrics.metrics.webflow_phase =
                        format!("{:?}", webflow_output.rollout).to_ascii_lowercase();
                    metrics.metrics.webflow_blocker = webflow_output.rollout_blocker.to_string();
                    metrics.metrics.webflow_valid_health_cycles =
                        webflow_output.valid_health_cycles;
                    metrics.metrics.webflow_active_navigations =
                        webflow_output.observation.active_navigations;
                    metrics.metrics.webflow_proposed_total = webflow_output.counters.proposed;
                    metrics.metrics.webflow_admitted_total = webflow_output.counters.admitted;
                    metrics.metrics.webflow_skipped_total = webflow_output.counters.skipped;
                    // The extension already reports LCP/INP inside
                    // WebFlowMetrics; before this these fields stayed None
                    // forever because nothing copied them out.
                    metrics.metrics.browser_lcp_p95_ms = webflow_output.lcp_p95_ms;
                    metrics.metrics.browser_inp_p95_ms = webflow_output.inp_p95_ms;
                    metrics.metrics.network_flow_active = networkflow_output.observation.active;
                    metrics.metrics.network_flow_traffic_bps =
                        networkflow_output.observation.traffic_bps;
                    metrics.metrics.network_flow_confidence_q =
                        networkflow_output.observation.confidence_q;
                    metrics.metrics.network_flow_proposed_total =
                        networkflow_output.counters.proposals;
                    metrics.metrics.network_flow_started_total = networkflow_output.counters.starts;
                    metrics.metrics.network_flow_renewed_total =
                        networkflow_output.counters.renewals;
                    metrics.metrics.network_flow_skipped_total =
                        networkflow_output.counters.skipped;
                    metrics.metrics.network_flow_suppressed_exact_total =
                        networkflow_output.counters.suppressed_exact;
                    metrics.metrics.network_flow_hard_cap_total =
                        networkflow_output.counters.hard_cap_expirations;
                }

                // Option C starts observationally: one immutable cycle identity
                // and one global value plan, with no change to legacy execution.
                let value_metrics =
                    value_scheduler_runtime.tick(value_scheduler_tick::ValueSchedulerTickInput {
                        cycle: cycle_count,
                        overhead_level: overhead_budget.level,
                        allow_speculation: overhead_budget.allow_speculation,
                        holt_cadence: overhead_budget.holt_winters.cadence,
                        page_reclaim_cadence: overhead_budget.page_reclaim.cadence,
                        pressure: snapshot.pressure.memory_pressure,
                        pressure_age: pressure_collector.data_age(),
                        thermal_level: &snapshot.pressure.thermal_level,
                        interaction_active,
                        context_epoch: context_summary.map(|summary| summary.daemon_epoch),
                        context_visual_q: context_summary.map(|summary| summary.visual_change_q),
                        context_interaction_q: context_summary.map(|summary| summary.interaction_q),
                        context_audio_active,
                        context_permissions_bits,
                        workload: foreground_app.as_deref().unwrap_or("idle"),
                        capability_revision: runtime_capability_graph.revision,
                        kill_switch,
                        sleeping,
                        profile_matches: parallel_runtime.build.expected_profile
                            == parallel_runtime.build.effective_profile,
                        p95_cycle_ms: overhead_input.p95_cycle_ms,
                        holt_cost_ms: overhead_input.holt_winters_avg_ms,
                        page_reclaim_cost_ms: overhead_input.page_reclaim_avg_ms,
                        webflow_events: &webflow_output.mesh_events,
                        webflow_observation: Some(webflow_output.observation),
                        network_observation: Some(networkflow_output.observation),
                    });
                let value_scheduler_active = value_metrics.phase == "active";
                let value_handoff = value_scheduler_runtime.take_handoff();
                let hardware_prediction_permit = value_handoff.as_ref().and_then(|handoff| {
                    handoff
                        .permits()
                        .iter()
                        .find(|permit| {
                            permit.job_id()
                                == apollo_engine::engine::value_scheduler::JobId::HardwarePrediction
                        })
                        .cloned()
                });
                let fabric_tick_started = Instant::now();
                let fabric_submitted_before = heterogeneous_runtime
                    .as_ref()
                    .map_or(0, heterogeneous_tick::HeterogeneousRuntime::submitted_total);
                let fabric_metrics = value_metrics.world_snapshot.as_ref().map(|world| {
                    if heterogeneous_runtime.is_none() {
                        heterogeneous_runtime =
                            Some(heterogeneous_tick::HeterogeneousRuntime::new(world));
                    }
                    let transition = if heterogeneous_workload_id != 0
                        && heterogeneous_workload_id != world.identity.workload_id
                    {
                        1.0
                    } else {
                        context_summary.map_or(0.0, |summary| summary.visual_change_q)
                    };
                    heterogeneous_workload_id = world.identity.workload_id;
                    heterogeneous_runtime
                        .as_mut()
                        .expect("heterogeneous runtime initialized")
                        .tick(heterogeneous_tick::HeterogeneousTickInput {
                            world: Arc::clone(world),
                            cpu_utilization: snapshot.cpu.global_usage as f64 / 100.0,
                            pressure: snapshot.pressure.memory_pressure,
                            p95_cycle_ms: overhead_input.p95_cycle_ms,
                            transition,
                            interaction: context_summary
                                .map_or(if interaction_active { 1.0 } else { 0.0 }, |summary| {
                                    summary.interaction_q
                                }),
                            // Same normalization the published world feature
                            // vector already uses, so the model and the
                            // snapshot agree on what "io pressure" means.
                            io_pressure: world
                                .network
                                .filter(|flow| flow.sample_fresh)
                                .map_or(0.0, |flow| {
                                    flow.traffic_bps as f64 / (8.0 * 1024.0 * 1024.0)
                                }),
                            optional_allowed: value_scheduler_tick::optional_job_allowed(
                                value_scheduler_active,
                                overhead_budget.allow_speculation,
                                hardware_prediction_permit.is_some(),
                            ),
                            budget_allows_speculation: overhead_budget.allow_speculation,
                            optional_recovery_healthy: !value_scheduler_active
                                && overhead_input.p95_cycle_ms < 60.0
                                && overhead_input.reason_avg_ms < 50.0
                                && overhead_input.memory_pressure < 0.55
                                && !overhead_input.fluidity_degraded,
                            thermal_nominal: snapshot.pressure.thermal_level == "nominal",
                        })
                });
                if let Some(permit) = hardware_prediction_permit.as_ref() {
                    let elapsed_us = fabric_tick_started
                        .elapsed()
                        .as_micros()
                        .min(u128::from(u32::MAX)) as u32;
                    let status = match fabric_metrics.as_ref() {
                        Some(metrics) if metrics.submitted_total > fabric_submitted_before => {
                            apollo_engine::engine::value_scheduler::JobCompletionStatus::Succeeded
                        }
                        Some(_) => {
                            apollo_engine::engine::value_scheduler::JobCompletionStatus::Cancelled
                        }
                        None => apollo_engine::engine::value_scheduler::JobCompletionStatus::Failed,
                    };
                    let _ = value_scheduler_runtime.acknowledge(permit, elapsed_us, status);
                }
                world_model.set_runtime_cycle(cycle_count);
                {
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.value_scheduler_phase = value_metrics.phase;
                    metrics.metrics.value_scheduler_blocker = value_metrics.blocker;
                    metrics.metrics.value_scheduler_valid_cycles = value_metrics.valid_cycles;
                    metrics.metrics.value_scheduler_shadow_cycles =
                        value_metrics.shadow_cycles_required;
                    metrics.metrics.value_snapshot_epoch = value_metrics.snapshot_epoch;
                    metrics.metrics.value_snapshot_revision = value_metrics.snapshot_revision;
                    metrics.metrics.value_scheduler_registered_jobs = value_metrics.registered_jobs;
                    metrics.metrics.value_scheduler_eligible_jobs = value_metrics.eligible_jobs;
                    metrics.metrics.value_scheduler_selected_jobs = value_metrics.selected_jobs;
                    metrics.metrics.value_scheduler_selected_total = value_metrics.selected_total;
                    metrics.metrics.value_scheduler_budget_us = value_metrics.budget_us;
                    metrics.metrics.value_scheduler_predicted_us = value_metrics.predicted_us;
                    metrics.metrics.value_scheduler_selection_latency_us =
                        value_metrics.selection_latency_us;
                    metrics.metrics.value_scheduler_max_selection_latency_us =
                        value_metrics.max_selection_latency_us;
                    metrics.metrics.value_scheduler_budget_skips_total =
                        value_metrics.budget_skips_total;
                    metrics.metrics.value_scheduler_capacity_skips_total =
                        value_metrics.capacity_skips_total;
                    metrics.metrics.value_scheduler_invalid_samples_total =
                        value_metrics.invalid_samples_total;
                    metrics.metrics.value_scheduler_invalid_unhealthy_total =
                        value_metrics.invalid_unhealthy_total;
                    metrics.metrics.value_scheduler_invalid_sequence_total =
                        value_metrics.invalid_sequence_total;
                    metrics.metrics.value_scheduler_invalid_features_total =
                        value_metrics.invalid_features_total;
                    metrics.metrics.value_scheduler_invalid_publication_total =
                        value_metrics.invalid_publication_total;
                    if let Some(fabric) = fabric_metrics.as_ref() {
                        metrics.metrics.fabric_phase = fabric.phase.clone();
                        metrics.metrics.fabric_blocker = fabric.blocker.clone();
                        metrics.metrics.fabric_workers_active = fabric.workers_active;
                        metrics.metrics.fabric_qos_failures = fabric.qos_failures;
                        metrics.metrics.fabric_result_drops = fabric.result_drops;
                        metrics.metrics.fabric_submitted_total = fabric.submitted_total;
                        metrics.metrics.fabric_completed_total = fabric.completed_total;
                        metrics.metrics.fabric_cancelled_total = fabric.cancelled_total;
                        metrics.metrics.fabric_stale_total = fabric.stale_total;
                        metrics.metrics.fabric_deadline_misses_total = fabric.deadline_misses_total;
                        metrics.metrics.fabric_dispatch_skips_total = fabric.dispatch_skips_total;
                        metrics.metrics.fabric_eligible_total = fabric.eligible_total;
                        metrics.metrics.fabric_evaluation_total = fabric.evaluation_total;
                        metrics.metrics.fabric_last_latency_us = fabric.last_latency_us;
                        metrics.metrics.fabric_control_p95_baseline_ms =
                            fabric.control_p95_baseline_ms;
                        metrics.metrics.fabric_cpu_percent = fabric.fabric_cpu_percent;
                        metrics.metrics.fabric_rss_delta_bytes = fabric.rss_delta_bytes;
                        metrics.metrics.coreml_model_available = fabric.coreml_model_available;
                        metrics.metrics.coreml_requested_backend = fabric.coreml_requested.clone();
                        metrics.metrics.coreml_effective_backend = fabric.coreml_effective.clone();
                        metrics.metrics.coreml_ane_execution_measured =
                            fabric.ane_execution_measured;
                        metrics.metrics.coreml_circuit_state = fabric.coreml_circuit.clone();
                        metrics.metrics.temporal_prediction_backend =
                            fabric.prediction_backend.clone();
                        metrics.metrics.temporal_prediction_load = fabric.prediction_load;
                        metrics.metrics.temporal_prediction_transition =
                            fabric.prediction_transition;
                        metrics.metrics.temporal_prediction_pressure = fabric.prediction_pressure;
                        metrics.metrics.temporal_prediction_p95 = fabric.prediction_p95;
                        metrics.metrics.temporal_prediction_authoritative =
                            fabric.prediction_authoritative;
                    }
                }

                let (interaction_activations, markov_applied, user_call, user_audio, degraded) = {
                    let metrics = state.metrics.lock_recover();
                    (
                        metrics.metrics.interaction_qos_activations,
                        metrics.metrics.markov_prewarm_applied,
                        metrics.metrics.user_call_in_progress,
                        metrics.metrics.user_audio_active,
                        metrics.metrics.fluidity_degraded,
                    )
                };
                // Presence of the opt-in file is the only way a pair ever
                // mutates anything. Absent, the lab stays a Shadow diagnostic.
                let experiments_opted_in =
                    Path::new(apollo_engine::engine::daemon_helpers::experiments_opt_in_path())
                        .exists();
                let capture_active = screen_capture_cache.check();
                let micro_output =
                    microexperiment_runtime.tick(microexperiment_tick::MicroexperimentTickInput {
                        cycle: cycle_count,
                        interaction_activations,
                        markov_applied,
                        workload: foreground_app.as_deref().unwrap_or("idle"),
                        inherited_safe: !Path::new(kill_switch_path()).exists()
                            && !sleep_notifier.is_sleeping()
                            && snapshot.pressure.memory_pressure.is_finite()
                            && snapshot.pressure.memory_pressure < 0.55
                            && matches!(
                                snapshot
                                    .pressure
                                    .thermal_level
                                    .to_ascii_lowercase()
                                    .as_str(),
                                "nominal" | "normal"
                            )
                            && !user_call
                            && !user_audio
                            && !degraded
                            && parallel_runtime.build.expected_profile
                                == parallel_runtime.build.effective_profile,
                        experiments_enabled: experiments_opted_in,
                        // Privacy posture counts as known only once the daemon
                        // can actually read the sensitive-context signals it
                        // gates on.
                        privacy_known: experiments_opted_in,
                        secure_input: false,
                        screen_capture: capture_active,
                        camera_active: user_audio,
                        sensitive_context: user_call,
                    });
                let micro_metrics = micro_output.metrics;
                for gold in &micro_output.pair_gold {
                    tracing::info!(
                        action_key = %gold.action_key,
                        control_decision_id = gold.control_decision_id,
                        treatment_decision_id = gold.treatment_decision_id,
                        effect_micros = gold.effect_micros,
                        effective = gold.effective,
                        harmful = gold.harmful,
                        "microexperiment pair gold closed"
                    );
                }
                {
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.microexperiment_phase = micro_metrics.phase;
                    metrics.metrics.microexperiment_rollout_progress =
                        micro_metrics.rollout_progress;
                    metrics.metrics.microexperiment_rollout_required =
                        micro_metrics.rollout_required;
                    metrics.metrics.microexperiment_blocker = micro_metrics.blocker;
                    metrics.metrics.microexperiment_restore = micro_metrics.restore;
                    metrics.metrics.microexperiment_proposed_total = micro_metrics.proposed_total;
                    metrics.metrics.microexperiment_eligible_total = micro_metrics.eligible_total;
                    metrics.metrics.microexperiment_randomized_total =
                        micro_metrics.randomized_total;
                    metrics.metrics.microexperiment_shadow_would_open_total =
                        micro_metrics.shadow_would_open_total;
                    metrics.metrics.microexperiment_open_pairs = micro_metrics.open_pairs;
                    metrics.metrics.microexperiment_completed_pairs = micro_metrics.completed_pairs;
                    metrics.metrics.microexperiment_control_endpoints_total =
                        micro_metrics.control_endpoints_total;
                    metrics.metrics.microexperiment_treatment_endpoints_total =
                        micro_metrics.treatment_endpoints_total;
                    metrics.metrics.microexperiment_complete_horizons_total =
                        micro_metrics.complete_horizons_total;
                    metrics.metrics.microexperiment_rollback_closed_total =
                        micro_metrics.rollback_closed_total;
                    metrics.metrics.microexperiment_pair_gold_total = micro_metrics.pair_gold_total;
                    metrics.metrics.microexperiment_effective_total = micro_metrics.effective_total;
                    metrics.metrics.microexperiment_harmful_total = micro_metrics.harmful_total;
                    metrics.metrics.microexperiment_confounded_total =
                        micro_metrics.confounded_total;
                    metrics.metrics.microexperiment_interrupted_total =
                        micro_metrics.interrupted_total;
                    metrics.metrics.microexperiment_synthetic_quarantined_total =
                        micro_metrics.synthetic_quarantined_total;
                    metrics.metrics.microexperiment_mean_effect = micro_metrics.mean_effect;
                    metrics.metrics.microexperiment_invalidated_total =
                        micro_metrics.invalidated_total;
                    metrics.metrics.microexperiment_deadline_expired_total =
                        micro_metrics.deadline_expired_total;
                    metrics.metrics.microexperiment_rollback_failed_total =
                        micro_metrics.rollback_failed_total;

                    let endpoint = microexperiment_runtime.adapter_counters();
                    metrics.metrics.microexperiment_endpoint_contract_ready =
                        microexperiment_runtime.endpoint_contract_ready(cycle_count);
                    metrics.metrics.microexperiment_arms_registered_total =
                        endpoint.registered_arms;
                    metrics.metrics.microexperiment_episodes_observed_total =
                        endpoint.observed_episodes;
                    metrics.metrics.microexperiment_episodes_skipped_idle_total =
                        endpoint.episodes_skipped_idle;
                    metrics.metrics.microexperiment_uncatalogued_episodes_total =
                        endpoint.uncatalogued_episodes;
                    metrics.metrics.microexperiment_decisions_bound_total =
                        endpoint.bound_decisions;
                    metrics.metrics.microexperiment_endpoints_emitted_total =
                        endpoint.emitted_endpoints;
                    metrics.metrics.microexperiment_endpoints_pending_utility =
                        endpoint.pending_utility;
                    metrics
                        .metrics
                        .microexperiment_endpoint_action_mismatch_total =
                        endpoint.rejected_action_mismatch;
                    metrics.metrics.microexperiment_endpoint_unknown_arm_total =
                        endpoint.rejected_unknown_arm;
                    metrics.metrics.microexperiment_endpoint_duplicate_total =
                        endpoint.rejected_duplicate;
                    metrics.metrics.microexperiment_endpoint_expired_total =
                        endpoint.rejected_expired;
                    metrics
                        .metrics
                        .microexperiment_endpoint_epoch_rejected_total = endpoint.rejected_epoch;
                    metrics
                        .metrics
                        .microexperiment_endpoint_authority_rejected_total =
                        endpoint.rejected_authority;
                    metrics
                        .metrics
                        .microexperiment_endpoint_incomplete_metadata_total =
                        endpoint.rejected_incomplete_metadata;
                    metrics
                        .metrics
                        .microexperiment_endpoint_capacity_drops_total = endpoint.dropped_capacity;
                    metrics.metrics.microexperiment_endpoint_arms_expired_total =
                        endpoint.expired_arms;
                    metrics
                        .metrics
                        .microexperiment_endpoint_rollback_observed_total =
                        endpoint.rollback_observed;
                    metrics
                        .metrics
                        .microexperiment_endpoint_rollback_failed_total = endpoint.rollback_failed;
                    metrics.metrics.microexperiment_control_withholds_total =
                        apollo_engine::engine::microexperiment_endpoints::control_withholds_total();
                }
                // The lab's control requests are advisory only: an actuator may
                // decline to take a lease, never take one it was not going to.
                apollo_engine::engine::microexperiment_endpoints::publish_control_withholds(
                    &microexperiment_runtime.control_withhold_requests(),
                );

                // FocusMarkov miss check, Markov observe+pre-warm, universal pre-thaw, temporal predictor.
                // Extracted to daemon_markov_tick::run_markov_tick (Wave 29).
                // [Fowler 2004] Strangler Fig — pure move, no semantic change.
                exploration_scheduler.begin_exploration_cycle(cycle_count);
                let markov_cognitive_paused = prev_cog_decision
                    .as_ref()
                    .is_some_and(|decision| decision.pause_learning);
                let mut markov_exploration_environment = || {
                    capture_exploration_environment(
                        &state,
                        pressure_collector.latest().memory_pressure,
                        build_tracker.phase,
                        markov_cognitive_paused,
                        learnable_params.fluidity_degraded_threshold,
                    )
                };
                let daemon_markov_tick::MarkovTickOutput {
                    temporal_hour: markov_temporal_hour,
                    temporal_weekday: markov_temporal_weekday,
                    decision_events: markov_decision_events,
                } = daemon_markov_tick::run_markov_tick(
                    foreground_app.as_deref(),
                    foreground_pid,
                    last_fg_name.as_deref(),
                    cycle_count,
                    &mut focus_markov,
                    &mut temporal_predictor,
                    &mut last_markov_prethaw,
                    &mut markov_shadow,
                    &mut markov_hit_count,
                    &mut markov_miss_count,
                    &state,
                    &collector,
                    &process_tree,
                    &coalition_tracker,
                    &mut cache_warmer,
                    &frozen_state_path,
                    &world_model,
                    &mut acceleration_lease,
                    &mut exploration_scheduler,
                    &mut markov_exploration_environment,
                );
                cycle_decision_events.extend_buffer(&markov_decision_events);
                temporal_hour = markov_temporal_hour;
                temporal_weekday = markov_temporal_weekday;

                // Context-switch burst detector + reactive unfreeze.
                // Extracted to daemon_ctx_switch_tick::run_ctx_switch_tick (Wave 31).
                let ctx_switch_decision_events = daemon_ctx_switch_tick::run_ctx_switch_tick(
                    foreground_app.clone(),
                    foreground_pid,
                    &mut last_fg_name,
                    &mut ctx_switch_times,
                    &state,
                    &frozen_state_path,
                    cycle_count,
                );
                cycle_decision_events.extend_buffer(&ctx_switch_decision_events);

                // Build enriched process data using foreground detector + process tree.
                // A process is considered foreground if it IS the foreground app or a
                // descendant of it (via process tree), giving accurate foreground family
                // detection for multi-process apps like Chrome, Electron, etc.
                let _t_enrich_start = Instant::now();
                let (proc_snaps, hunt_snaps, apple_platform_pids) =
                    process_enrichment::build_enriched_process_data_with_tree(
                        collector.system(),
                        foreground_pid,
                        &process_tree,
                        cycle_count,
                        prev_pressure_smooth,
                        hw_cores,
                        lf_metrics,
                    );
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::ReasonEnrich,
                    process_tree_nanos
                        .saturating_add(_t_enrich_start.elapsed().as_nanos())
                        .min(u64::MAX as u128) as u64,
                );
                let all_proc_names: Vec<&str> =
                    proc_snaps.iter().map(|p| p.name.as_str()).collect();
                let hour_of_day = Utc::now().hour() as u8;

                // Pre-sleep unfreeze: release every SIGSTOP'd PID before kernel suspends.
                // Extracted to daemon_process_collector::run_pre_sleep_unfreeze().
                // [Saltzer & Kaashoek 2009] §3.3 A-B-A defense via unfreeze_pids_verified.
                let pre_sleep_decisions = daemon_process_collector::run_pre_sleep_unfreeze(
                    &state,
                    &frozen_state_path,
                    &mut display_turbo,
                    &sleep_notifier,
                    cycle_count,
                );
                cycle_decision_events.extend_buffer(&pre_sleep_decisions);

                // Ghost-PID reconciliation — evict dead PIDs from frozen_state +
                // turbo set; GC mach_qos HashMaps every 60 cycles.
                // Reuse the immutable identity classification produced by
                // process_enrichment. Building both sets in one in-memory pass
                // avoids repeating proc_pidpath for every process in the
                // decision stage while preserving the same protection policy.
                let mut live_pids = HashSet::with_capacity(proc_snaps.len());
                let mut app_bundle_pids = HashSet::with_capacity(proc_snaps.len() / 4);
                for process in &proc_snaps {
                    live_pids.insert(process.pid);
                    if process.is_app_bundle {
                        app_bundle_pids.insert(process.pid);
                    }
                }
                let recently_dead_dedup = recently_applied.invalidate_dead_pids(&live_pids);
                if recently_dead_dedup > 0 {
                    tracing::debug!(
                        target: "apollo.recently_applied",
                        removed = recently_dead_dedup,
                        remaining = recently_applied.len(),
                        "evicted recently-applied entries for dead PIDs"
                    );
                }
                daemon_process_collector::run_ghost_pid_reconciliation(
                    &state,
                    &live_pids,
                    &frozen_state_path,
                    &mut display_turbo,
                    cycle_count,
                    &identity_cache,
                );

                // MemoryAnalyzer profiling + WakeStormDetector per-cycle scan.
                // Extracted to daemon_proc_scan_tick::run_proc_scan_tick (Wave 32).
                let _t_procscan_start = Instant::now();
                daemon_proc_scan_tick::run_proc_scan_tick(
                    &proc_snaps,
                    &mut mem_analyzer,
                    &mut proc_recovery,
                    &mut wake_storm,
                    cycle_count,
                    snapshot.pressure.memory_pressure,
                );
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::ReasonProcScan,
                    _t_procscan_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );

                // Memory budgets: jetsam inactive limits for over-budget processes.
                // Extracted to daemon_memory_budget::run_memory_budget (Wave 28).
                // [Fowler 2004] Strangler Fig — pure move.
                let mem_budget_start = Instant::now();
                let memory_budget_output = daemon_memory_budget::run_memory_budget_with_decisions(
                    snapshot.pressure.memory_pressure,
                    snapshot.memory.total_ram,
                    &state,
                    &proc_snaps,
                    &mem_analyzer,
                    &mut memory_budget,
                    cycle_count,
                );
                cycle_decision_events.extend_buffer(&memory_budget_output.decision_events);
                lf_metrics
                    .set_memory_budget_duration_us(mem_budget_start.elapsed().as_micros() as u64);

                // Audit fix #5: Read cached hardware data from background SmcReader thread.
                // No more blocking 500 ms powermetrics calls on the hot path.
                {
                    if let Some(hw) = smc_reader.latest() {
                        {
                            let mut m = state.metrics.lock_recover();
                            m.metrics.iokit_snapshots = smc_reader.success_count();
                            m.metrics.iokit_errors = smc_reader.error_count();
                            m.metrics.iokit_p_cluster_temp = (!hw.temps_estimated)
                                .then_some(hw.temps.p_cluster_celsius)
                                .flatten();
                            m.metrics.iokit_e_cluster_temp = (!hw.temps_estimated)
                                .then_some(hw.temps.e_cluster_celsius)
                                .flatten();
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
                        apollo_engine::engine::lse_counters::LSE_COUNTERS
                            .set_iokit_errors(smc_reader.error_count());
                    }
                }

                // Battery status: detect real battery state every 10 cycles (~30s)
                // to avoid spawning pmset too frequently.
                if cycle_count.is_multiple_of(10) {
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
                                apollo_engine::engine::iokit_sensors::ThermalState::Normal,
                            temps: apollo_engine::engine::iokit_sensors::ClusterTemps {
                                p_cluster_celsius: None,
                                e_cluster_celsius: None,
                                gpu_celsius: None,
                                nand_celsius: None,
                            },
                            temps_estimated: false,
                            sampled_at: Instant::now(),
                            power: apollo_engine::engine::iokit_sensors::PowerReading {
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
                // Thermal pre-throttle boost: normalized OS state first, with a
                // conservative measured-temperature fallback when available.
                // so page_reclaim + io_shaper + governor act before hardware throttles.
                //
                // WarmBand (2026-06-28 heat-aware throttle scheduling): pre-stage
                // pressure boost triggered on temperature trend OR absolute temp in
                // 75°C-to-Phase1 band, before Phase1Gentle fires. This is read-only: only
                // adds to effective_pressure via thermal_pressure_boost. The trend
                // analysis lives in ThermalBailout::compute_warm_boost and is
                // already in the `thermal_action.warm_pressure_boost` field. We
                // add it on top of the phase-derived base.
                let mut thermal_pressure_boost = match thermal_action.phase {
                    apollo_engine::engine::thermal_bailout::CoolingPhase::Normal => 0.0,
                    apollo_engine::engine::thermal_bailout::CoolingPhase::WarmBand => 0.0,
                    apollo_engine::engine::thermal_bailout::CoolingPhase::Phase1Gentle => 0.07,
                    apollo_engine::engine::thermal_bailout::CoolingPhase::Phase2Moderate => 0.15,
                    apollo_engine::engine::thermal_bailout::CoolingPhase::Phase3Aggressive => 0.25,
                    apollo_engine::engine::thermal_bailout::CoolingPhase::Phase4Emergency => 0.40,
                };
                // Heat-aware pre-stage boost (2026-06-28). Capped at 0.05 in the
                // thermal_bailout module; ONLY added, never overrides. Independent
                // of the main phase ladder — can coexist with any phase.
                thermal_pressure_boost +=
                    (thermal_action.warm_pressure_boost as f64).clamp(0.0, 0.05);

                // Thermal pre-throttle freeze/unfreeze.
                // Extracted to daemon_thermal_freeze::run_thermal_freeze (Wave 20).
                // Passes &mut outcome_tracker (via LearningContext) so blocked
                // freezes feed the survival-bias closure (shadow-mode-only).
                let thermal_freeze_output = daemon_thermal_freeze::run_thermal_freeze(
                    &thermal_action,
                    &state,
                    &collector,
                    foreground_pid,
                    snapshot.pressure.memory_pressure,
                    std::path::Path::new(&frozen_state_path),
                    lctx.outcome_tracker,
                    cycle_count,
                );
                cycle_decision_events.extend_buffer(&thermal_freeze_output.decision_events);

                // HwPredictor: schedule hardware signals every 10 cycles (~5s at normal rate).
                // Cache and bandwidth probes run on a dedicated worker; completed samples
                // provide advance warning before metrics APIs catch up.
                let (hw_pressure, jitter_us, hw_features) = if let Some(snap) =
                    hw_predictor_worker.poll_and_schedule(cycle_count)
                {
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

                // ThermalManager + GPUManager tick.
                // Extracted to daemon_thermal_tick::run_thermal_tick (Wave 24).
                let daemon_thermal_tick::ThermalTickOutput {
                    gpu_thermal_throttled,
                    thermal_predicted_throttle,
                    thermal_seconds_to_throttle,
                    thermal_trend_predicted,
                } = daemon_thermal_tick::run_thermal_tick(
                    cycle_hw_snap.as_ref(),
                    &mut thermal_mgr,
                    &mut gpu_mgr,
                    &state,
                    jitter_us,
                    hw_cores,
                );

                // SwapPredictor: update trend forecast every cycle.
                let swap_forecast = swap_predictor.update(
                    snapshot.pressure.swap_used_bytes,
                    snapshot.pressure.swap_total_bytes,
                );

                // PowerManager: advisory tick (no real sensor data yet).
                let _power_rec = power_mgr.get_recommendation();

                // EMA interactivity classifier: cpu_wall_ratio per PID from rusage deltas.
                // Extracted to daemon_rusage_tick::compute_cpu_wall_ratios (Wave 33).
                let _t_rusage_start = Instant::now();
                let cpu_wall_ratios = daemon_rusage_tick::compute_cpu_wall_ratios(
                    &snapshot,
                    &mut last_rusage_at,
                    &mut rusage_cpu_prev,
                );
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::ReasonRusage,
                    _t_rusage_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );

                // Online usage learning (root-only, no UI sensors): infer frequently-used apps
                // and processes correlated with jank, then promote patterns conservatively.
                local_policy_learning::usage_learning_tick(
                    &state,
                    &snapshot,
                    !foreground_idle && foreground_app.is_some(),
                    &cpu_wall_ratios,
                );

                // Local deliberation replaces the former prompt/JSON teacher.
                // It fuses System 1, Dr Zero, universal medallion, World Model
                // and GPU evidence, then scales only advisory paths owned by
                // specialists that already passed their own safety gates.
                world_model.attach_local_consolidation(
                    local_consolidator.view_for_installation(installation_id),
                );
                let calibration_summary = telemetry_medallion.model_calibration().summary();
                world_model.attach_unified_trust(
                    telemetry_medallion.authoritative_gold_decision_count() as u64,
                    calibration_summary.validated_models as u64,
                    calibration_summary.trusted_models as u64,
                    calibration_summary.degraded_models as u64,
                );
                let deliberation_workload = world_model
                    .latest_context()
                    .map(|context| context.workload.clone())
                    .unwrap_or_else(|| "unknown".to_string());
                let deliberation = world_model.synthesize_deliberation(
                    &deliberation_workload,
                    lctx.outcome_tracker.heuristic_is_struggling(),
                    lctx.outcome_tracker.self_challenge_score(),
                );
                {
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.system_deliberation_mode = deliberation.mode;
                    metrics.metrics.system_deliberation_confidence = deliberation.confidence;
                    metrics.metrics.system_deliberation_advisory_support_scale =
                        deliberation.advisory_support_scale;
                    metrics.metrics.system_deliberation_gpu_support_scale =
                        deliberation.gpu_support_scale;
                    metrics.metrics.system_deliberation_dr_zero_challenge =
                        deliberation.dr_zero_self_challenge;
                    metrics.metrics.system_deliberation_system1_struggling =
                        deliberation.system1_struggling;
                    metrics.metrics.system_deliberation_local_gold =
                        deliberation.evidence.local_gold;
                    metrics.metrics.system_deliberation_gpu_forecasts =
                        deliberation.evidence.gpu_fresh_predictions;
                    metrics.metrics.system_deliberation_local_confidence =
                        deliberation.evidence.local_consolidation_confidence;
                    metrics.metrics.system_deliberation_local_families =
                        deliberation.evidence.local_consolidation_families;
                    metrics.metrics.local_consolidations = local_consolidator.total_consolidations;
                    metrics.metrics.local_consolidation_improvements =
                        local_consolidator.total_improvements;
                    metrics.metrics.local_consolidation_regressions =
                        local_consolidator.total_regressions;
                    metrics.metrics.local_consolidation_neutral = local_consolidator.total_neutral;
                    metrics.metrics.local_consolidation_system1_updates =
                        local_consolidator.total_system1_updates;
                }

                let mut reactor_weight = apollo_engine::engine::lse_counters::LSE_COUNTERS
                    .snapshot()
                    .reactor_event_weight;
                reactor_weight = (reactor_weight * 0.75).clamp(0.0, 1.0);
                // Persist the decayed value back so the next snapshot reads the
                // post-decay state, not the pre-decay sticky 1.0. Without this,
                // reactor_weight stays pinned at 1.0 after any reactor pulse until
                // metrics_reporter overwrites it.
                apollo_engine::engine::lse_counters::LSE_COUNTERS
                    .set_reactor_event_weight(reactor_weight);

                // kqueue: consume VM pressure events (kernel push, zero latency).
                // Extracted to daemon_kqueue_tick::run_kqueue_tick (Wave 26).
                // Critical/SuddenTerminate → reactor_weight=1.0 + fast-tick 30s.
                daemon_kqueue_tick::run_kqueue_tick(
                    &mut kq_frozen,
                    &mut reactor_weight,
                    &state,
                    &snapshot,
                    lctx.overflow_guard,
                    lctx.signal_intel,
                    &mut display_turbo,
                    &frozen_state_path,
                    &learnable_params,
                    &identity_cache,
                    Some(lf_metrics),
                );

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
                    decision_events: llm_decision_events,
                } = daemon_feature_gates::run_llm_inference_mode_tick(
                    &snapshot,
                    &state,
                    &mut llm_detector,
                    &mut llm_spotlight_disabled,
                    is_root,
                    foreground_pid,
                    &thermal_action,
                    cycle_count,
                );
                cycle_decision_events.extend_buffer(&llm_decision_events);

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
                // Fight-hunt fix (2026-06-10): preserve the PHYSICAL pressure
                // before overwriting with the effective (boosted) value.
                // signal_intel learning + the maintenance purge gate read the
                // raw field — purge can't fix thermal/battery boosts, and
                // models trained on boosted values learn battery-skewed
                // baselines.
                snapshot.pressure.memory_pressure_raw = snapshot.pressure.memory_pressure;
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
                // Phase 0 lock-decomp instrumentation: time wait + hold of
                // the longest known metrics-lock hold (~410 LoC).
                {
                    let t_wait = std::time::Instant::now();
                    let mut metrics = state.metrics.lock_recover();
                    let wait_ns = t_wait.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                    let t_held = std::time::Instant::now();
                    metrics.metrics.swap_used_bytes = snapshot.pressure.swap_used_bytes;
                    metrics.metrics.swap_total_bytes = snapshot.pressure.swap_total_bytes;
                    metrics.metrics.swap_delta_bps = snapshot.pressure.swap_delta_bytes_per_sec;
                    // Fault-in rate (the stall side) + session peak. [Phase 0
                    // telemetry — no decision consumes this yet; baseline first.]
                    let refault = snapshot.pressure.refault_delta_per_sec;
                    metrics.metrics.refault_delta_per_sec = refault;
                    metrics.metrics.refault_peak_per_sec =
                        metrics.metrics.refault_peak_per_sec.max(refault);
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
                    if let Ok(tracker) = apollo_engine::engine::contention_tracker::global().lock()
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
                        Some(energy_summary.estimated_co2_avoided_kg * 1000.0);
                    metrics.metrics.energy_co2_emitted_g =
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

                    // IOReport hardware telemetry — DEPRECATED on macOS 26+
                    // (requires Apple-private entitlements unavailable to
                    // third-party binaries). When subscription is None, fall
                    // expose cluster fields as unavailable zeros and preserve
                    // package power from the public powermetrics path. Do not
                    // relabel aggregate CPU as P-cluster activity.
                    if let Some(ref ir) = last_ioreport {
                        metrics.metrics.ioreport_p_cluster_pct = ir.p_cluster_pct;
                        metrics.metrics.ioreport_e_cluster_pct = ir.e_cluster_pct;
                        metrics.metrics.ioreport_gpu_pct = ir.gpu_pct;
                        metrics.metrics.ioreport_ane_busy = ir.ane_busy;
                        metrics.metrics.ioreport_cpu_mw = ir.cpu_mw;
                        metrics.metrics.ioreport_total_watts = ir.total_watts();
                    } else {
                        metrics.metrics.ioreport_p_cluster_pct = 0.0;
                        metrics.metrics.ioreport_e_cluster_pct = 0.0;
                        metrics.metrics.ioreport_gpu_pct = 0.0;
                        metrics.metrics.ioreport_total_watts = cycle_hw_snap
                            .as_ref()
                            .and_then(|h| h.power.package_watts)
                            .map(f64::from)
                            .or_else(|| last_smc.as_ref().and_then(|s| s.system_power_watts))
                            .unwrap_or(0.0);
                    }

                    // SMC direct metrics. 2026-05-14: bit-rotted on macOS 26
                    // Tahoe — AppleSMC userclient struct layout changed.
                    // smc_diagnostic surfaces the failure mode so dashboards
                    // can distinguish "sensor returned 0" from "sensor broken".
                    if let Some(ref smc) = last_smc {
                        let any_read = smc.cpu_temp_celsius.is_some()
                            || smc.gpu_temp_celsius.is_some()
                            || smc.system_power_watts.is_some();
                        metrics.metrics.smc_diagnostic = if any_read {
                            "ok".to_string()
                        } else {
                            "unavailable_macos26".to_string()
                        };
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
                                    for (pid, tier) in [
                                        (pair.heavy_pid, SchedulingTier::Foreground),
                                        (pair.light_pid, SchedulingTier::Background),
                                    ] {
                                        let tier_name = match tier {
                                            SchedulingTier::Foreground => "foreground",
                                            SchedulingTier::Normal => "normal",
                                            SchedulingTier::Background => "background",
                                        };
                                        cycle_decision_events.push(
                                            apollo_engine::engine::decision_ledger::ActuatorDecisionEvent::local(
                                                format!("contention:mach_tier:{tier_name}"),
                                                format!("pid:{pid}"),
                                                cycle_count,
                                                apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome::Blocked,
                                                "contention-separation",
                                                "protected by contention safety gate",
                                            ),
                                        );
                                    }
                                    continue;
                                }
                                // heavy → P-cores (Foreground tier)
                                let (heavy_outcome, heavy_disposition) = qos
                                    .set_tier_audited(pair.heavy_pid, SchedulingTier::Foreground);
                                cycle_decision_events.push(contention_mach_tier_event(
                                    cycle_count,
                                    pair.heavy_pid,
                                    SchedulingTier::Foreground,
                                    &heavy_outcome,
                                    heavy_disposition,
                                ));
                                // light → E-cores (Background tier)
                                let (light_outcome, light_disposition) = qos
                                    .set_tier_audited(pair.light_pid, SchedulingTier::Background);
                                cycle_decision_events.push(contention_mach_tier_event(
                                    cycle_count,
                                    pair.light_pid,
                                    SchedulingTier::Background,
                                    &light_outcome,
                                    light_disposition,
                                ));
                            }
                        }

                        // GC dead PIDs every 100 cycles
                        if cycle_count.is_multiple_of(100) {
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
                    // Extracted to daemon_fluidity_tick (Wave 37).
                    // Update FluidityState from process snapshot + GPU load.
                    {
                        let fl_sig = daemon_fluidity_tick::run_fluidity_tick(
                            daemon_fluidity_tick::FluidityTickInput {
                                proc_snaps: &proc_snaps,
                                cycle_hw_snap: cycle_hw_snap.as_ref(),
                                cycle_dt_secs: cycle_dt_secs as f32,
                                fluidity_state: &mut fluidity_state,
                            },
                        )
                        .fl_signal;

                        // Wire into RuntimeMetrics for status/dashboard reporting
                        metrics.metrics.fluidity_score = fl_sig.fluidity_score;
                        metrics.metrics.window_op_active = fl_sig.window_op_active;
                        metrics.metrics.app_launching = fl_sig.app_launching;
                        metrics.metrics.app_launch_name = fl_sig.launch_name;
                        metrics.metrics.fluidity_degraded = fl_sig.fluidity_degraded;
                        // Kalman prediction for pre-emptive response
                        metrics.metrics.fluidity_predicted_3s = fl_sig.fluidity_predicted_3s;
                        metrics.metrics.fluidity_velocity = fl_sig.fluidity_velocity;
                        let p_jank_60s = fluidity_state.predicted_jank_probability_60s();
                        apollo_engine::engine::shadow_signals::set_p_jank_60s(p_jank_60s);
                        metrics.metrics.si_p_jank_60s = p_jank_60s;
                        // Also update windowserver_cpu_pct (existing field)
                        metrics.metrics.windowserver_cpu_pct = fl_sig.windowserver_cpu_ema;
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
                        metrics.metrics.iopm_power_source = if iopm.power_source
                            == apollo_engine::engine::thermal_iokit::PowerSource::Unknown
                        {
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
                    if wakeup_sorted.len() > 3 {
                        wakeup_sorted.select_nth_unstable_by(3, |a, b| {
                            b.wakeup_rate
                                .partial_cmp(&a.wakeup_rate)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        wakeup_sorted.truncate(3);
                    }
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
                    let anomaly_process_count = anomaly_sorted.len();
                    if anomaly_sorted.len() > 3 {
                        anomaly_sorted.select_nth_unstable_by(3, |a, b| {
                            b.anomaly_score
                                .partial_cmp(&a.anomaly_score)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        anomaly_sorted.truncate(3);
                    }
                    anomaly_sorted.sort_by(|a, b| {
                        b.anomaly_score
                            .partial_cmp(&a.anomaly_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    metrics.metrics.anomaly_process_count = anomaly_process_count;
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
                    drop(metrics);
                    let held_ns = t_held.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                    lf_metrics.record_metrics_lock(wait_ns, held_ns);
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
                        total_ram_gb: snapshot.memory.total_ram as f64 / (1024.0 * 1024.0 * 1024.0),
                        interactive_count,
                    };
                    let (mode, _confidence) = classify_workload_mode(&features);
                    mode
                };
                if let Some(fabric) = fabric_metrics
                    .as_ref()
                    .filter(|fabric| fabric.prediction_authoritative)
                {
                    let _ = world_model.attach_fabric_prediction(
                        cycle_count,
                        workload_mode.as_str(),
                        fabric.prediction_load,
                        fabric.prediction_transition,
                        fabric.prediction_pressure,
                        fabric.prediction_p95,
                    );
                }

                // Workload-onset: fired once when transitioning INTO Build mode.
                // Lets the governor proactively switch to AggressiveRoot before
                // pressure builds, rather than waiting for the reactive threshold.
                let raw_workload_onset = workload_mode == WorkloadMode::Build
                    && prev_workload_mode != WorkloadMode::Build;

                // Work-session hysteresis keeps a BalancedRoot floor after recent
                // development activity, but only the true Build transition edge may
                // trigger AggressiveRoot. Reusing the full grace window as an onset
                // pulse kept re-escalating an otherwise calm machine for five minutes.
                let fg_is_dev_tool = foreground_app
                    .as_deref()
                    .map(is_dev_tool_name)
                    .unwrap_or(false);
                let is_dev_active =
                    workload_mode == WorkloadMode::Build || dev_session_active || fg_is_dev_tool;
                let now_instant = std::time::Instant::now();
                work_session.note_activity(is_dev_active, now_instant);
                let work_session_active =
                    work_session.is_active(power_mgr.is_on_battery(), battery_low, now_instant);
                let workload_onset = raw_workload_onset;

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
                        dev_session_active: dev_session_active || work_session_active,
                        interactive_heavy,
                        context_switch_burst,
                        arousal_level: arousal_state.level as f64,
                        workload_mode: Some(workload_mode),
                        workload_onset,
                        swap_used_bytes: snapshot.pressure.swap_used_bytes,
                    })
                };
                if governor_decision.transition_reason.contains("floor") {
                    apollo_engine::engine::lse_counters::LSE_COUNTERS
                        .increment_profile_floor_hits();
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
                // Phase 0c sub-stage: time run_signal_tick (Kalman MV8 + CUSUM +
                // Entropy + Hazard + LV + MPC — the 666-LoC inline block).
                // Phase D PURGE-INHIBITION (Sprint 12 candidate #1, 2026-05-17):
                // tell the swap predictor to skip this cycle if a vm_purge fired
                // in the last 5s. Without this gate the post-purge artificial
                // dip is learned as a load improvement and Kalman/Hazard/MPC
                // cool down OOM risk artificially.
                // [Hellerstein 2004 §9] disturbance rejection.
                //
                // 2026-05-30 latch-clear: feed the current swap delta into the
                // maintenance state so that, once the compressor settles (≥2
                // consecutive non-negative deltas), `compressor_still_flushing`
                // auto-clears and the inhibition window collapses from 12s
                // back to the 5s base. Without this tick the latch was
                // monotonic-set (re-asserted on every purge) and never fell.
                // Additive instrumentation (2026-06-23): wrap the whole
                // signal-intelligence block (Kalman/CUSUM/Entropy/Hazard/LV/MPC
                // + ODE + downstream forecasting). Nests the existing tiny
                // ReasonSignalTick span — independent counter, intentional.
                let _t_signalintel_start = Instant::now();
                maintenance_state
                    .tick_compressor_status(snapshot.pressure.swap_delta_bytes_per_sec);
                lctx.signal_intel.purge_inhibited =
                    maintenance_state.is_in_purge_inhibition_window();
                let _t_signal_start = Instant::now();
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
                    hw_cores,
                    hour_of_day,
                    &state,
                    &mut darwin_anomaly,
                    &mut telemetry_logger,
                    &fluidity_state,
                    cycle_count,
                );
                last_pressure_velocity = new_lpv;
                prev_entropy_anomaly = new_entropy;
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::ReasonSignalTick,
                    _t_signal_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );

                // Swap Reclaim ODE — feed vm_rate from background collector.
                // Produces SaturationForecast used by the freeze gate below.
                // [Denning 1968; Zhao et al. 2009 WKdm rate model]
                let reclaim_forecast = {
                    use apollo_engine::engine::swap_reclaim::VmFlowSample;
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

                // Wire ODE σ into SignalDigest. [Øksendal 2003 §3] diffusion term.
                let mut signal_digest = signal_digest;
                signal_digest.swap_net_rate_volatility = reclaim_forecast.net_rate_volatility;

                // Publish shadow signals for decide_actions' shadow-mode ActionContext.
                // Consumed by ShadowEvaluator via shadow_signals::get_* — keeps
                // decide_actions' signature stable while wiring predictive + context.
                apollo_engine::engine::shadow_signals::set_p_oom_30s(signal_digest.p_oom_30s);
                apollo_engine::engine::shadow_signals::set_thermal_emergency(thermal_emergency);
                apollo_engine::engine::shadow_signals::set_interrupt_phase(
                    state
                        .resource_interrupt
                        .phase
                        .load(std::sync::atomic::Ordering::Relaxed),
                );
                apollo_engine::engine::shadow_signals::set_foreground_pid(foreground_pid);
                // Epistemic proxy — urgency is "need to act", not "how uncertain".
                // Better signal: entropy_anomaly (distribution shift) + transformer_anomaly
                // (learned-deviation from Hopfield memory). Both measure IGNORANCE of
                // current state, not urgency. [Lakshminarayanan 2017] epistemic from
                // distribution shift. NotebookLM audit 2026-04-22 flagged the urgency
                // mis-wire; this is the corrected proxy until full EpistemicState is
                // instantiated in the daemon (separate effort).
                let epistemic_proxy = {
                    let entropy_term = (signal_digest.entropy_anomaly.abs() / 3.0).clamp(0.0, 1.0);
                    let anomaly_term = signal_digest.transformer_anomaly.clamp(0.0, 1.0);
                    // Max — either source indicates ignorance on its own.
                    entropy_term.max(anomaly_term)
                };
                apollo_engine::engine::shadow_signals::set_epistemic_uncertainty(epistemic_proxy);

                // Phase 5.2 WIRED (Sprint 10 finisher, 2026-05-16) — publish
                // battery + wake-up signals so `BatteryAwareCostFeature`
                // (registered in shadow_evaluator) can emit cost
                // contributions. `is_on_battery` is taken from
                // `power_mgr.battery_status.is_charging` (inverted), updated
                // every 10 cycles via `detect_battery_status`. Wakeups +
                // ctx-switches are placeholders (0.0) until a per-cycle
                // delta accumulator lands — the setters drop non-finite /
                // negative values and the consumer returns None on 0.0
                // wakeups (no false penalty). Single hot-path cost: 3
                // atomic stores.
                apollo_engine::engine::shadow_signals::set_is_on_battery(
                    !power_mgr.battery_status.is_charging,
                );
                // Phase 5.2 REAL producer (2026-05-16): use the already-
                // computed 5-minute context-switch window divided by 300s
                // for ctxsw/s, and the wake_storm aggregate rate for
                // wakeups/s. Both quantities are bounded, non-negative,
                // and already produced for other dashboard fields — zero
                // extra hot-path cost.
                let ctxsw_rate = (context_switches_5min as f64) / 300.0;
                apollo_engine::engine::shadow_signals::set_ctx_switches_per_sec(ctxsw_rate);
                // Sum per-process wakeup rates from detected storms (best
                // available aggregate without adding new state). Range
                // [0.0, ~thousands]; the cost function clamps internally.
                let wakes_rate: f64 = wake_storm
                    .detect_storms()
                    .iter()
                    .map(|p| p.wakeups_per_second as f64)
                    .sum();
                apollo_engine::engine::shadow_signals::set_wakeups_per_sec(wakes_rate);

                // ODE swap urgency — hoisted for use in Neuromodulator AND LinUCB.
                // Normalization owned by TsatUrgency [CyberPhysicalSignal trait].
                let ode_t_sat_urgency = {
                    use apollo_engine::engine::swap_reclaim::{CyberPhysicalSignal, TsatUrgency};
                    TsatUrgency(reclaim_forecast.t_sat_sec).normalized()
                };

                // ── Reactor weight: adaptive modulation ──────────────────────────────
                // Extracted to daemon_reactor_tick::run_reactor_tick (Wave 39).
                // [Denning 1968; Pirolli & Card 1999; Hellerstein 2004]
                let reactor_start = Instant::now();
                reactor_weight =
                    daemon_reactor_tick::run_reactor_tick(daemon_reactor_tick::ReactorTickInput {
                        signal_digest: &signal_digest,
                        holt_winters: &holt_winters,
                        window_relief_cycles: &mut window_relief_cycles,
                        win_session_phase: &win_session_phase,
                        win_pressure_floor,
                        win_workload_intent: &win_workload_intent,
                        raw_pressure: snapshot.pressure.memory_pressure,
                        temporal_predictor: &temporal_predictor,
                        temporal_hour,
                        temporal_weekday,
                        build_tracker: &build_tracker,
                        amx_available,
                        llm_active,
                        base_reactor_weight: reactor_weight,
                    });
                lf_metrics.set_reactor_duration_us(reactor_start.elapsed().as_micros() as u64);

                // Predictive agent: build context from existing signals and select intervention.
                // Feed Kalman-smoothed pressure instead of raw — cleaner signal for LinUCB.
                let (agent_intervention, applied_predictive_intervention) = {
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
                        use apollo_engine::engine::swap_reclaim::{
                            CyberPhysicalSignal, NetRateNorm,
                        };
                        NetRateNorm(reclaim_forecast.net_rate_bps).normalized()
                    };

                    // KalmanMV8: fuse all 8 signals after ODE outputs are available.
                    // [Welch & Bishop 2006] H=I cross-covariance propagation.
                    {
                        use apollo_engine::engine::swap_reclaim::NET_RATE_CEILING_BPS;
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
                            // [3] lyapunov_norm replaces duplicate pressure proxy.
                            // [Wolf et al. 1985] orthogonal chaos signal improves KF covariance.
                            (signal_digest.lyapunov_exponent / 2.0).clamp(0.0, 1.0),
                            ode_net_rate_norm, // [4] ode_net_rate
                            ode_t_sat_urgency, // [5] ode_t_sat
                            cpu_mean,          // [6] cpu_saturation
                            thermal_f64,       // [7] thermal_stress
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
                        signal_digest.cumulative_stress,
                        (signal_digest.lyapunov_exponent / 2.0).clamp(0.0, 1.0),
                    );
                    let (linucb_choice, linucb_confidence) = lctx
                        .predictive_agent
                        .select_action_with_confidence(&agent_ctx);

                    // Super Learner specialist voting + accuracy feedback.
                    // Extracted to `daemon_cognitive_tick::apply_specialist_voting`
                    // during V1.1.0 Strangler Fig wave 7 — pure move, no semantic
                    // change.  Ordering: runs after LinUCB select and before the
                    // decision_stage, same as the original inline form.
                    // Phase 5.1 wiring — build PresenceInputs from last cycle's
                    // UserContext + current arousal. Phase 5.1-D (2026-05-16)
                    // wires `hid_events_per_minute` to a real producer that
                    // tracks resets of HIDIdleTime across the last 30 real
                    // samples (≈ 10 min wall-clock at the daemon's 10-cycle
                    // sampling cadence). Returns 0.0 — the modulator's
                    // idle-only fallback — until enough samples accumulate.
                    let presence_inputs = daemon_cognitive_tick::PresenceInputs {
                        idle_seconds: last_user_context_for_voting.idle_secs,
                        hid_events_per_minute:
                            apollo_engine::engine::user_context::hid_events_per_minute(),
                        current_arousal: arousal_state.level as f64,
                        audio_active: last_user_context_for_voting.audio_active,
                        has_sleep_assertion: last_user_context_for_voting.has_sleep_assertion,
                        // Phase 5.1.1 production fix (2026-05-16) — raw
                        // pressure feeds the critical-pressure bypass that
                        // defuses the cascade-paralysis cycle observed in
                        // prod (Score 0.85 + 0 actions). `signal_digest`
                        // exposes the smoothed value used by every other
                        // hot-path consumer, so reuse it here for consistency.
                        memory_pressure: signal_digest.pressure_smooth,
                    };
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
                        workload_mode,
                        presence_inputs,
                        &world_model,
                        &maintenance_state,
                    );
                    last_specialist_votes = voting_out.disagreement_record;
                    (voting_out.intervention, voting_out.applied_intervention)
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

                // Holt-Winters seasonal forecasting: accumulate samples, observe hourly,
                // tighten overflow_thresholds proactively.
                // Extracted to daemon_holt_winters_tick::run_holt_winters_tick (Wave 30).
                //
                // Optional forecasting and page-reclaim have independent
                // cadence/deadline budgets. Expensive lanes slow themselves
                // without suppressing unrelated sensing or observability. The
                // critical Sense -> Signal -> Decide -> Execute -> Learn ->
                // Persist path always runs. [Hellerstein 2004 §9].
                let holt_allowed = overhead_budget.admits(
                    adaptive_overhead::OptionalSubsystem::HoltWinters,
                    cycle_count,
                    _t_reason_start.elapsed().as_millis(),
                );
                if !holt_allowed {
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.apollo_overhead_optional_skips_total = metrics
                        .metrics
                        .apollo_overhead_optional_skips_total
                        .saturating_add(1);
                    metrics.metrics.apollo_overhead_holt_skips_total = metrics
                        .metrics
                        .apollo_overhead_holt_skips_total
                        .saturating_add(1);
                    // Sprint 13 perf: demoted INFO→DEBUG. ~4 k events in
                    // 6 h — fired whenever a reason stage cycle exceeded
                    // 150 ms. Stage_reason_max_ms in runtime_metrics
                    // already surfaces the same signal without per-cycle
                    // JSON formatting cost.
                    tracing::debug!(
                        elapsed_ms = _t_reason_start.elapsed().as_millis() as u64,
                        "cycle: skipping HoltWinters (subsystem budget)"
                    );
                }
                let _t_hw_start = Instant::now();
                if holt_allowed {
                    daemon_holt_winters_tick::run_holt_winters_tick(
                        snapshot.pressure.memory_pressure,
                        hour_of_day,
                        &mut holt_winters,
                        &mut hw_pressure_accum,
                        &mut hw_pressure_count,
                        &mut hw_last_hour,
                        &state,
                        &mut overflow_thresholds,
                    );
                }
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::ReasonHoltWinters,
                    _t_hw_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );

                // HW seasonal anomaly: ratio of actual pressure to seasonal expectation.
                // [Holt 1957, Winters 1960] >1.0 = above norm; >1.5 at quiet hour = structural.
                {
                    let level = holt_winters.level();
                    let sf = holt_winters.seasonal_factor(hour_of_day);
                    let expected = (level * sf).max(1e-6);
                    signal_digest.hw_seasonal_anomaly =
                        (snapshot.pressure.memory_pressure / expected).clamp(0.0, 3.0);
                }

                // Perceptual latency monitor: composite score from existing signals.
                // If UI responsiveness is degraded, boost reactor_weight to trigger
                // faster/more aggressive scheduling decisions.
                let _latency_score_val = {
                    if jitter_us > 0 {
                        last_schedule_jitter_us = jitter_us;
                        schedule_jitter_samples_us.push_back(jitter_us);
                        if schedule_jitter_samples_us.len() > 120 {
                            schedule_jitter_samples_us.pop_front();
                        }
                    }
                    let fg_cpu = foreground_pid
                        .and_then(|pid| {
                            proc_snaps
                                .iter()
                                .find(|s| s.pid == pid)
                                .map(|s| s.cpu_percent as f64)
                        })
                        .unwrap_or(0.0);
                    let fg_csw = foreground_pid
                        .and_then(proc_taskinfo::get_task_info)
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
                        jitter_us: last_schedule_jitter_us as f64,
                        windowserver_cpu: local_policy_learning::windowserver_cpu(&snapshot) as f64,
                        foreground_cpu: fg_cpu,
                        foreground_csw_per_sec: fg_csw,
                        has_foreground: foreground_pid.is_some(),
                    });
                    if latency.needs_boost {
                        // Elevate reactor weight → faster tick + more aggressive decisions.
                        reactor_weight = (reactor_weight + 0.25).min(1.0);
                    }
                    {
                        let mut metrics = state.metrics.lock_recover();
                        metrics.metrics.perceptual_latency_score = latency.score;
                        metrics.metrics.perceptual_latency_category =
                            latency.category.as_str().to_string();
                        metrics.metrics.perceptual_latency_source =
                            "scheduler-windowserver-foreground-proxy".to_string();
                        metrics.metrics.scheduler_jitter_p95_ms =
                            apollo_engine::engine::daemon_helpers::compute_p95(
                                schedule_jitter_samples_us.make_contiguous(),
                            ) / 1_000.0;
                        metrics.metrics.scheduler_jitter_samples =
                            schedule_jitter_samples_us.len() as u64;
                    }
                    latency.score
                };

                // Adaptive Page Reclaim: purge file cache when pressure is building
                // but before the kernel is forced to evict reactively (which causes stalls).
                // Jiang & Zhang 2005 — proactive beats reactive by 20-40%.
                // Runs every 10 cycles (~5s) to avoid vm_stat overhead every cycle.
                let _t_pr_start = Instant::now();
                let page_reclaim_allowed = overhead_budget.admits(
                    adaptive_overhead::OptionalSubsystem::PageReclaim,
                    cycle_count,
                    _t_reason_start.elapsed().as_millis(),
                );
                if page_reclaim_allowed {
                    // snapshot.pressure.memory_pressure already includes battery
                    // + thermal boosts via effective_pressure::compute(). Don't add again.
                    // 2026-05-12: post-wake aggressive purge — when the wake
                    // handler set `post_wake_reclaim_until` within the last
                    // 90s, bypass the pressure/foreground gates so stale
                    // file-backed pages + cold daemon residency get reclaimed
                    // even though the Kalman pressure reading is cold-start
                    // noisy. Window auto-expires; this is not a permanent
                    // override.
                    let post_wake_reclaim_active = {
                        let now_wall = chrono::Utc::now();
                        state
                            .process
                            .lock_recover()
                            .wake_state
                            .post_wake_reclaim_until
                            .map(|t| t > now_wall)
                            .unwrap_or(false)
                    };
                    let freed = page_reclaim.tick_with_post_wake(
                        snapshot.pressure.memory_pressure,
                        display_turbo.is_turbo_active() || thermal_action.phase >= apollo_engine::engine::thermal_bailout::CoolingPhase::Phase2Moderate,
                        foreground_idle,
                        post_wake_reclaim_active,
                    );
                    if freed > 0 {
                        apollo_engine::engine::lse_counters::LSE_COUNTERS
                            .increment_paging_hints_applied();
                    }
                } else if cycle_count.is_multiple_of(10) {
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.apollo_overhead_optional_skips_total = metrics
                        .metrics
                        .apollo_overhead_optional_skips_total
                        .saturating_add(1);
                    metrics.metrics.apollo_overhead_page_reclaim_skips_total = metrics
                        .metrics
                        .apollo_overhead_page_reclaim_skips_total
                        .saturating_add(1);
                }
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::ReasonPageReclaim,
                    _t_pr_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );

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
                let mut causal_impact = lctx.causal_graph.impact_map();
                // World-model Mode-2 snapshot (2026-06-11): Gold-only action
                // predictions + Rubin do-nothing drift, queried before actions.
                // [LeCun 2022 §4.2; Sutton Dyna 1991]
                // Calibrated imagination: the same MetaCognition debias that
                // corrects the CausalGraph's meta-observe (87c342f) scales the
                // world model's predicted deltas — calibration and imagination
                // share one belief about causal over-promising.
                world_model.refresh_from_parts_for_workload(
                    lctx.causal_graph,
                    lctx.outcome_tracker,
                    cognitive_state.meta_cognition.subsystem_debias_multiplier(
                        apollo_engine::engine::meta_cognition::SubsystemId::CausalGraph,
                    ),
                    workload_mode.as_str(),
                );
                world_model.attach_context(telemetry_medallion.trusted_view());
                CausalGraph::apply_nars_discount(
                    &mut causal_impact,
                    &lctx.outcome_tracker.drift_detector,
                );

                // User context "telepathy" — extracted to
                // `daemon_cognitive_tick::compute_user_context`.
                // Phase 0c sub-stage: pmset+ioreg can be latent.
                let _t_uc_start = Instant::now();
                let user_context = daemon_cognitive_tick::compute_user_context(
                    cycle_count,
                    &mut last_user_assertions,
                    &mut last_idle_sample,
                    cycle_hw_snap.as_ref(),
                );
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::ReasonUserContext,
                    _t_uc_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );
                // Phase 5.1 wiring — carry this cycle's UserContext forward
                // for next cycle's specialist-voting PresenceInputs. The
                // copy is cheap (4 f64/bool fields, Clone).
                last_user_context_for_voting = user_context.clone();

                let phase_observation = workload_phase_tracker.observe(
                    apollo_engine::engine::workload_phase::classify_phase(
                        workload_mode,
                        user_context.call_in_progress || user_context.audio_active,
                        context_switch_burst,
                    ),
                    Instant::now(),
                );
                {
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.workload_phase = phase_observation.phase.as_str().to_string();
                    metrics.metrics.workload_phase_expected_duration_secs =
                        phase_observation.expected_duration_secs;
                    metrics.metrics.workload_phase_duration_observations =
                        phase_observation.duration_observations;
                }

                // Swap reclaim ODE: pre-emptive reactor_weight boost on saturation risk.
                // [Denning 1968; Zhao 2009] Extracted to daemon_swap_reclaim_tick (Wave 35).
                daemon_swap_reclaim_tick::apply_swap_reclaim_boost(
                    &reclaim_forecast,
                    &mut reactor_weight,
                );

                let reclaim_bypass = swap_reclaim_bypass_active(
                    snapshot.pressure.swap_used_bytes,
                    snapshot.pressure.swap_total_bytes,
                    signal_digest.p_oom_30s,
                );
                let empty_hab: HashSet<u32> = HashSet::new();
                let effective_habituated: &HashSet<u32> = if reclaim_bypass {
                    &empty_hab // bypass: re-evaluate all processes
                } else {
                    &habituated_pids
                };

                // Sprint 12 Convergence #1 (2026-05-17). Build the set
                // of PIDs the CompanionGraph classifies as companions
                // of the current foreground app. decide_actions reads
                // this set in the cold-thread loop to keep companions
                // on the same P-cluster as foreground hot threads
                // (preserving L2 working-set locality across user
                // focus switches) when DRAM bandwidth is below the
                // 0.50 safety floor. Empty set = disabled.
                // [ARM big.LITTLE 2013 §3] cluster-local scheduling.
                //
                // Sprint 12 perf-fix (2026-05-30): memoized across
                // cycles via `companion_fg_cache`. The naive build is
                // O(top_processes.len() × 2 HashMap lookups) ≈ 25–35
                // μs per cycle on M1 8GB; in steady state (single fg
                // app + stable top_processes) the cache hit ratio
                // approaches 1.0, dropping the cost to a fingerprint
                // XOR over ≈50 entries + 4 equality checks. Invalidated
                // by foreground app flip, top_processes (pid,name)
                // mutation, or CompanionGraph mutation witness
                // (`total_cycles`, `anchor_count`). [Saltzer & Schroeder
                // 1975] Economy of Mechanism. See `CompanionFgCache`.
                let fg_app_opt: Option<&str> = foreground_app.as_deref().filter(|s| !s.is_empty());
                let top_proc_fingerprint = fingerprint_top_processes(&snapshot.top_processes);
                let graph_total_cycles = companion_graph.total_cycles();
                let graph_anchor_count = companion_graph.anchor_count();
                if companion_fg_cache.as_ref().is_some_and(|c| {
                    c.is_valid(
                        fg_app_opt,
                        top_proc_fingerprint,
                        graph_total_cycles,
                        graph_anchor_count,
                    )
                }) {
                    lf_metrics.inc_companion_fg_cache_hit();
                } else {
                    let pids: HashSet<u32> = match fg_app_opt {
                        Some(fg_name) => snapshot
                            .top_processes
                            .iter()
                            .filter(|p| companion_graph.is_companion_of(fg_name, &p.name))
                            .map(|p| p.pid)
                            .collect(),
                        None => HashSet::new(),
                    };
                    companion_fg_cache = Some(CompanionFgCache {
                        fg_app: fg_app_opt.map(|s| s.to_string()),
                        top_proc_fingerprint,
                        graph_total_cycles,
                        graph_anchor_count,
                        pids,
                    });
                }
                // SAFETY: just populated above on miss; on hit the slot
                // was non-None when we entered the branch. unwrap is
                // infallible at this point.
                let companion_of_fg_pids: &HashSet<u32> = &companion_fg_cache
                    .as_ref()
                    .expect("companion_fg_cache populated above")
                    .pids;
                // Additive instrumentation (2026-06-23): close the
                // ReasonSignalIntel span — wraps the entire untimed
                // enrich→decide signal-intelligence region.
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::ReasonSignalIntel,
                    _t_signalintel_start
                        .elapsed()
                        .as_nanos()
                        .min(u64::MAX as u128) as u64,
                );

                // 2026-05-30: ReasonDecide stage instrumentation. Closes
                // the 12-of-13 staging gap exposed in cycle-tail avg/max
                // table — every declared CycleStage MUST have ≥1 producer
                // [Hellerstein 2004 §3 observability invariant + Sprint 9
                // silent-telemetry-death `4b13a39`].
                let _t_decide_start = Instant::now();
                let decision = {
                    // S4 cutover (2026-06-06): pass shared Arc to decision_stage.
                    let qos_arc = state.mach_qos.clone();
                    let dram_bandwidth_pct = last_ioreport
                        .as_ref()
                        .map(|ir| ir.amc_bandwidth_pct)
                        .unwrap_or(0.0);
                    // Snapshot the cooldown set under lock, then drop the
                    // guard so the decision pass holds no mutex on
                    // freeze_cooldown. Cloning is O(n) over active PIDs
                    // (typically <30) and avoids holding the lock across
                    // the full decision computation.
                    let freeze_cooldown_snapshot = state.freeze_cooldown.lock_recover().clone();

                    let policy = PolicyContext {
                        decide_interactive: &decide_interactive,
                        decide_noise: &decide_noise,
                        decide_weights: &decide_weights,
                        outcome_baseline,
                        behavior_interactive_pids: &behavior_interactive_pids,
                        app_bundle_pids: &app_bundle_pids,
                        ipc_hints: &ipc_hints,
                        hop_groups: &lctx.outcome_tracker.hop_groups,
                        habituated_pids: effective_habituated,
                        causal_confidence: &causal_confidence,
                        causal_impact: &causal_impact,
                        user_ctx: &user_context,
                        wakeup_hints: &wakeup_hints,
                        footprint_hints: &footprint_hints,
                        dram_bandwidth_pct,
                        io_burst_hints: &io_burst_hints,
                        anomaly_hints: &anomaly_hints,
                        freeze_cooldown: &freeze_cooldown_snapshot,
                        companion_of_foreground_pids: companion_of_fg_pids,
                        world_model: &world_model,
                        effectiveness: &effectiveness_tracker,
                    };
                    decision_stage
                        .run(
                            &snapshot,
                            collector.system(),
                            current_profile,
                            latency_target,
                            reactor_weight,
                            overflow_thresholds,
                            Some(&qos_arc),
                            &policy,
                        )
                        .decision
                };
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::ReasonDecide,
                    _t_decide_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );
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
                    let audio_probe =
                        apollo_engine::engine::coreaudio_active::audio_activity_snapshot();
                    metrics.metrics.coreaudio_probe_state = audio_probe.probe_state().to_string();
                    metrics.metrics.coreaudio_probe_samples_total = audio_probe.direct_samples;
                    metrics.metrics.coreaudio_probe_cache_hits_total = audio_probe.cache_hits;
                    metrics.metrics.coreaudio_probe_failures_total = audio_probe.failures;
                }

                // Apply locally consolidated policy patterns across restart cycles.
                // Read-only cross-cycle state-memory filter. The final dispatch
                // chokepoint is the sole writer; recording here made every
                // admitted decision look duplicate when it reached that second
                // filter later in the same cycle.
                // decide_actions has ~14 emission sites (multi-thread QoS, freeze
                // gates, swarm throttle, predictive policy etc.) without per-site
                // cross-cycle dedup. Single chokepoint here filters ALL of them
                // against the recently_applied cache. [Saltzer & Schroeder 1975]
                // Economy of Mechanism — single filter beats per-site fixes.
                //
                // Sprint 4 Fase 5 (2026-05-07): the raw `Vec<RootAction>` accumulator
                // is now wrapped in `ActionAccumulator`, a typed builder that:
                //  - validates shape on `push_*` methods,
                //  - emits per-variant counters + tracing on every push,
                //  - exposes `view()` for read-only peeks (heuristic_pass et al),
                //  - has a single terminal exit `finalize()` (`#[must_use]`).
                // The dedup chokepoint and learned-policy filter still run BEFORE
                // the accumulator (semantics-preserving — no reorder).
                let initial_filtered: Vec<RootAction> = {
                    let raw = decision.actions;
                    let mut filtered = Vec::with_capacity(raw.len());
                    for action in raw {
                        if let Some((pid, kind, discriminator)) =
                            apollo_engine::engine::recently_applied::CachedActionKind::from_root_action(&action)
                        {
                            if recently_applied.is_recent_scoped(pid, kind, discriminator) {
                                continue;
                            }
                        }
                        filtered.push(action);
                    }
                    filtered
                };
                let initial_filtered = {
                    let policy = state.policy.lock_recover().learned_policy.clone();
                    local_policy_learning::apply_learned_policy_actions(
                        &snapshot,
                        &policy,
                        initial_filtered,
                        foreground_app.as_deref(),
                    )
                };
                let mut acc = ActionAccumulator::with_capacity(initial_filtered.len() + 16);
                acc.extend_raw(
                    initial_filtered,
                    EmitContext::new(
                        ActionPhase::Decide,
                        "main.rs::3102 decide+learned_policy",
                        "decide_actions+learned_policy",
                    ),
                    lf_metrics,
                );

                // Apply learned skills + trial induced skills.
                // Extracted to daemon_skill_tick::run_skill_tick (Wave 16).
                // [Fowler 2004] Strangler Fig — pure move, no semantic change.
                {
                    let skill_new = daemon_skill_tick::run_skill_tick(
                        lctx.skill_registry,
                        &snapshot,
                        &state,
                        &collector,
                        foreground_pid,
                        workload_mode.as_str(),
                        is_root,
                        acc.view(),
                        &mut pending_trial_skill,
                        &world_model,
                    );
                    acc.extend_raw(
                        skill_new,
                        EmitContext::new(
                            ActionPhase::SkillTick,
                            "main.rs::3138 skill_tick",
                            "learned_skills+trial",
                        ),
                        lf_metrics,
                    );
                }

                // Coordinated cluster freezing + Spotlight pressure gate.
                // Extracted to daemon_cluster_actions::run_cluster_actions (Wave 18).
                {
                    let causal_pairs = lctx.outcome_tracker.top_causal_pairs(5);
                    let cluster_out = daemon_cluster_actions::run_cluster_actions(
                        &causal_pairs,
                        acc.view(),
                        &collector,
                        snapshot.pressure.memory_pressure,
                        overflow_thresholds.bg_pressure,
                    );
                    acc.extend_raw(
                        cluster_out.new_actions,
                        EmitContext::new(
                            ActionPhase::ClusterActions,
                            "main.rs::3152 cluster_actions",
                            "coordinated_freeze+spotlight",
                        ),
                        lf_metrics,
                    );
                }

                // CompanionGraph observation — once per cycle, record what is
                // alive while `foreground_app` is fg. Cheap (HashMap inserts on
                // top_processes ~50 entries). Membership query consumed below.
                // Decay+GC every 500 cycles (~40 min @ 5s/cycle) keeps the
                // graph adapting and bounded.
                //
                // Sprint 13 Pressure-Router Gate (2026-05-30). Mirrors the
                // existing 4-subsystem MoR-style gate at
                // signal_intelligence.rs:404-424: skip the per-cycle
                // observation + Phase 3.3 propagation when pressure is
                // below the workload-adjusted mid_entry. Saves the
                // `alive: Vec<String>` clones + HashMap inserts + O(V²)
                // propagation when the system has no symptoms to learn
                // from. The `cycle_count % 4 == 0` fallback is
                // [Sutton & Barto §2.7] forced exploration: keeps the
                // Lift denominator updating ~every 20 s @ 5 s/cycle so
                // graph statistics don't go stale under sustained low
                // pressure. Telemetry shape matches Sprint 12 G12
                // `bus_saturated` skip-with-counter precedent (commit
                // `5f1c984`).
                let mid_entry_threshold = lctx.signal_intel.effective_zones(0).0;
                let pressure_router_open = snapshot.pressure.memory_pressure >= mid_entry_threshold
                    || cycle_count.is_multiple_of(4);
                if pressure_router_open {
                    let alive: Vec<String> = snapshot
                        .top_processes
                        .iter()
                        .map(|p| p.name.clone())
                        .collect();
                    companion_graph.observe_cycle(foreground_app.as_deref(), &alive, cycle_count);
                    // Phase 3.3 WIRED (Sprint 9, 2026-05-16): cross-group
                    // attention propagation. After per-cycle co-occurrence
                    // accumulation, ask the graph to infer (A, B, score)
                    // triples across the live coalition/app-group keys.
                    // The decider returns triples but does NOT mutate the
                    // graph — v1 wire only counts emissions for
                    // observability. Future consumers can read them via
                    // a separate call site.
                    //
                    // Cardinality bulkhead (NotebookLM 2026-05-16): cap at
                    // 64 group keys per cycle. The propagation is O(V²);
                    // 64² = 4096 ops fits well inside the 10 ms per-cycle
                    // budget even on M1 8GB worst-case scheduling.
                    {
                        let mut group_keys: Vec<String> = process_tree
                            .app_groups()
                            .map(|g| g.root_name.clone())
                            .collect();
                        if group_keys.len() > 64 {
                            group_keys.truncate(64);
                        }
                        if !group_keys.is_empty() {
                            let triples =
                                companion_graph.propagate_attention_across_groups(&group_keys);
                            if !triples.is_empty() {
                                apollo_engine::engine::lse_counters::LSE_COUNTERS
                                    .add_companion_cross_group_inferences(triples.len() as u64);
                            }
                        }
                    }
                    if cycle_count.is_multiple_of(500) {
                        let evicted = companion_graph.self_improve(cycle_count);
                        if evicted > 0 {
                            apollo_engine::engine::daemon_helpers::audit_log(&serde_json::json!({
                                "event": "companion_graph_gc",
                                "evicted": evicted,
                                "anchors": companion_graph.anchor_count(),
                                "edges": companion_graph.edge_count(),
                            }));
                        }
                    }
                } else {
                    // Pressure-router skip: bump skip counter and move on.
                    // Sprint 13 (2026-05-30) — saves ~40-55 μs alloc +
                    // O(V²) propagation when there are no symptoms to
                    // learn from.
                    apollo_engine::engine::lse_counters::LSE_COUNTERS
                        .inc_companion_observe_router_skip();
                }
                // ActiveCoalitionEnvelope — record current fg coalition so
                // recent fg apps keep coalition protection during the 5-min
                // grace window. Closes rapid-app-switch gap (tabbing to
                // Terminal for `git status` doesn't strip Antigravity helpers).
                if let Some(fpid) = foreground_pid {
                    let cid = coalition_tracker.get_coalition_id(fpid);
                    active_coalitions.record_foreground(cid);
                    if cycle_count.is_multiple_of(60) {
                        active_coalitions.evict_stale();
                    }
                }

                // Predictive agent: inject soft actions for PreThrottleNoise / ProactivePurge.
                // Extracted to daemon_agent_actions::run_agent_actions (Wave 19).
                let pressure_send_supported = caps.can_memory_pressure_send;
                {
                    let agent_new = daemon_agent_actions::run_agent_actions(
                        &agent_intervention,
                        &snapshot.top_processes,
                        &state,
                        &decide_interactive,
                        &user_context,
                        foreground_app.as_deref(),
                        foreground_pid,
                        &companion_graph,
                        &coalition_tracker,
                        &active_coalitions,
                        pressure_send_supported,
                    );
                    acc.extend_raw(
                        agent_new,
                        EmitContext::new(
                            ActionPhase::AgentActions,
                            "main.rs::3164 agent_actions",
                            "predictive_agent",
                        ),
                        lf_metrics,
                    );
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
                        acc.view(),
                        memory_budget.recovering_from_critical(),
                        &recently_applied,
                        user_context.call_in_progress,
                        user_context.audio_active,
                        pressure_send_supported,
                    );
                    acc.extend_raw(
                        hint_new,
                        EmitContext::new(
                            ActionPhase::PagingHints,
                            "main.rs::3181 paging_hints",
                            "pressure+ode_velocity",
                        ),
                        lf_metrics,
                    );
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
                    acc.view(),
                    &lctx.outcome_tracker.experience,
                    learnable_params.experience_pressure_band,
                    snapshot.pressure.memory_pressure,
                    workload_mode.code(),
                    (hw_ram_gb * 1024.0 * 1024.0 * 1024.0) as u64,
                    &recently_applied,
                    &apple_platform_pids,
                );
                let heuristic_decisions = heuristic_pass.heuristic_decisions;
                let heuristic_critical_pids = heuristic_pass.heuristic_critical_pids;
                let heuristic_stats = heuristic_pass.heuristic_stats;
                acc.extend_raw(
                    heuristic_pass.additional_actions,
                    EmitContext::new(
                        ActionPhase::Heuristic,
                        "main.rs::3210 heuristic_pass",
                        "adaptive_governor+protection",
                    ),
                    lf_metrics,
                );

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
                        acc.view(),
                        snapshot.pressure.refault_delta_per_sec,
                    );
                    acc.extend_raw(
                        stale_new,
                        EmitContext::new(
                            ActionPhase::StaleApps,
                            "main.rs::3224 stale_apps",
                            "background_freeze",
                        ),
                        lf_metrics,
                    );
                }

                // B.6 gap fix (2026-06-10): zombie/dead-weight consumer. The
                // hunter classifies — this block ACTS. Conservative mapping:
                //   GhostHelper / MemoryHoarder → jetsam idle-band hint (-1):
                //     the kernel keeps kill authority, Apollo only marks
                //     sacrificable dead weight (host app gone >24h / >256MB
                //     RSS with no UI and no interaction >30min).
                //   WakeupBurner → non-aggressive throttle (PRIO_DARWIN_BG).
                //   TrueZombie / Orphan → telemetry only: SIGKILL on a Z-state
                //     process is a no-op (parent must wait()) and orphan kills
                //     are too risky without lineage proof.
                // Every 30 cycles — dead weight accumulates over minutes.
                // Hunter has its own 3-cycle confirmation before classifying.
                if cycle_count % 30 == 0 && !hunt_snaps.is_empty() {
                    let dead_weight = zombie_hunter.evaluate_all(&hunt_snaps);
                    // Phase 2: during a high-volume workload (call, media, or a
                    // fault-in storm), do not jetsam-demote apps. A demote makes
                    // the target more compressible/kill-prone — so demoting right
                    // before the user switches to it deepens the refault on
                    // switch (the microstutter). Throttles (gentler) still run.
                    let high_bw_physical_pressure = if snapshot.pressure.memory_pressure_raw > 0.0 {
                        snapshot.pressure.memory_pressure_raw
                    } else {
                        snapshot.pressure.memory_pressure
                    };
                    let high_bw_workload =
                        apollo_engine::engine::coreaudio_active::is_high_bw_workload_active(
                            snapshot.pressure.refault_delta_per_sec,
                            high_bw_physical_pressure,
                        );
                    if !dead_weight.is_empty() {
                        let reclaimable_mb =
                            apollo_engine::engine::zombie_hunter::ZombieHunter::total_reclaimable_bytes(
                                &dead_weight,
                            ) / 1024
                                / 1024;
                        let mut zombie_actions: Vec<RootAction> = Vec::new();
                        for dw in &dead_weight {
                            // 2026-06-20 (regression probe caught live: node
                            // nominated 3x): hard_protected_contains only covers
                            // the hard list — a dev-runtime (node/rustc) or infra
                            // (docker/postgres) zombie slipped through and got
                            // jetsam-demoted. is_protected_name covers all three
                            // tiers. Additive — only adds protection.
                            // 2026-06-21 (P1, playback-easing Wave 1): also never
                            // nominate Chromium/Brave helpers — a live 4K renderer
                            // helper has no window of its own so it looks like an
                            // "idle MemoryHoarder", but demoting it under jetsam
                            // during playback = frame drop. is_chromium_family
                            // matches "Brave Browser Helper*". Was firing 974x
                            // (always failing) on a live renderer — additive.
                            // Reuse the full per-cycle envelope. It includes
                            // Apple platform ownership and dynamic behavioral
                            // protection, which name-only checks cannot see.
                            if daemon_action_safety::is_protected_action_candidate(
                                dw.pid,
                                &dw.name,
                                &heuristic_critical_pids,
                            ) {
                                continue;
                            }
                            lf_metrics.inc_zombie_dead_weight_detected();
                            use apollo_engine::engine::zombie_hunter::ZombieClass;
                            match dw.zombie_class {
                                ZombieClass::GhostHelper | ZombieClass::MemoryHoarder
                                    if high_bw_workload =>
                                {
                                    // Hold the demote off during the workload.
                                    tracing::debug!(
                                        pid = dw.pid,
                                        name = %dw.name,
                                        "phase2: jetsam-demote suppressed (high-bw workload)"
                                    );
                                }
                                ZombieClass::GhostHelper | ZombieClass::MemoryHoarder => {
                                    zombie_actions.push(RootAction::set_memorystatus(
                                        dw.pid,
                                        -1,
                                        format!(
                                            "zombie-hunter {:?}: {} ({}MB) — {}",
                                            dw.zombie_class,
                                            dw.name,
                                            dw.wasted_rss_bytes / 1024 / 1024,
                                            dw.reason,
                                        ),
                                        apollo_engine::engine::audit_types::DecisionReason::PressureContext,
                                    ));
                                }
                                ZombieClass::WakeupBurner => {
                                    let (ss, su) =
                                        apollo_engine::engine::daemon_helpers::pid_start_time(
                                            dw.pid,
                                        );
                                    zombie_actions.push(RootAction::ThrottleProcess {
                                        pid: dw.pid,
                                        name: dw.name.clone(),
                                        aggressive: false,
                                        reason: format!(
                                            "zombie-hunter WakeupBurner ({:.0} wakeups/s): {}",
                                            dw.wakeups_per_sec, dw.reason,
                                        ),
                                        start_sec: ss,
                                        start_usec: su,
                                        decision_reason:
                                            apollo_engine::engine::audit_types::DecisionReason::PressureContext,
                                    });
                                }
                                ZombieClass::TrueZombie | ZombieClass::Orphan => {
                                    // Telemetry-only v1: no safe signal-based remedy.
                                }
                            }
                        }
                        if !zombie_actions.is_empty() {
                            tracing::info!(
                                detected = dead_weight.len(),
                                actions = zombie_actions.len(),
                                reclaimable_mb,
                                "zombie-hunter: dead weight marked for kernel reclaim"
                            );
                            for _ in 0..zombie_actions.len() {
                                lf_metrics.inc_zombie_action_emitted();
                            }
                            acc.extend_raw(
                                zombie_actions,
                                EmitContext::new(
                                    ActionPhase::StaleApps,
                                    "main.rs zombie_hunter consumer",
                                    "dead_weight_reclaim",
                                ),
                                lf_metrics,
                            );
                        }
                    }
                    // Drop confirmation history for PIDs that exited.
                    let live: Vec<u32> = hunt_snaps.iter().map(|h| h.pid).collect();
                    zombie_hunter.cleanup(&live);
                }

                // Evolve iter-4 (2026-06-10): unified EffectLedger reconcile
                // replaces the ad-hoc boost-decay sweep. ALL recorded kernel
                // mutations (nice, tier, jetsam band, App Nap, memlimit,
                // Darwin-BG) revert here once their justification expires —
                // identity-guarded, foreground-exempt. One chokepoint instead
                // of five bespoke trackers. [Saltzer & Schroeder 1975]
                if cycle_count % 30 == 0 {
                    let reconcile = apollo_engine::engine::effect_ledger::reconcile_global(
                        foreground_pid,
                        &state.mach_qos,
                    );
                    for receipt in &reconcile.receipts {
                        let outcome = match receipt.disposition {
                            apollo_engine::engine::effect_ledger::ReconcileDisposition::Reverted => {
                                apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome::Reverted
                            }
                            apollo_engine::engine::effect_ledger::ReconcileDisposition::Failed => {
                                apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome::Failed
                            }
                            apollo_engine::engine::effect_ledger::ReconcileDisposition::NoOp => {
                                apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome::NoOp
                            }
                        };
                        cycle_decision_events.push(
                            apollo_engine::engine::decision_ledger::ActuatorDecisionEvent::local(
                                receipt.action_key(),
                                receipt.target(),
                                cycle_count,
                                outcome,
                                "effect-ledger-reconcile",
                                format!("expired effect reconciliation: {:?}", receipt.effect),
                            ),
                        );
                    }
                    if reconcile.reverted > 0 || reconcile.failed > 0 {
                        tracing::info!(
                            reverted = reconcile.reverted,
                            failed = reconcile.failed,
                            "effect-ledger: reconcile pass"
                        );
                        let mut metrics = state.metrics.lock_recover();
                        metrics.metrics.reverts_applied += reconcile.reverted;
                        metrics.metrics.reverts_failed += reconcile.failed;
                    }
                    let live: Vec<u32> = hunt_snaps.iter().map(|h| h.pid).collect();
                    apollo_engine::engine::effect_ledger::cleanup_global(&live);
                }

                // Survival mode: overflow recording, swap streak, purge, threshold decay.
                // Extracted to daemon_survival_tick::run_survival_tick (Wave 27).
                // [Fowler 2004] Strangler Fig — pure move.
                let survival_output = daemon_survival_tick::run_survival_tick(
                    &snapshot,
                    &signal_digest,
                    cycle_count,
                    lctx.overflow_guard,
                    lctx.signal_intel,
                    &learnable_params,
                    &mut swap_growth_streak,
                    &state,
                    &mut chromium_mgr,
                    &mut maintenance_state,
                    &async_commands,
                );
                cycle_decision_events.extend_buffer(&survival_output.decision_events);

                // Maintenance Purge Gate (2026-05-10) — opportunistic non-crisis purge
                // between survival_tick and dispatch_tick. Asymmetric cooldown: survival
                // is sovereign and bypasses last_any_purge_at; maintenance reads+writes.
                // Phase 4.2 WIRED (Sprint 10, 2026-05-16) — thermal external
                // event producer. On upward crossing from Normal → any throttle
                // tier (Moderate/Severe/Critical), record one external event.
                // The causal graph tags any subsequent action edge with
                // `external_blame: Some(ThermalThrottle)` for EXTERNAL_BLAME_WINDOW,
                // so credit for the next pressure drop is attributed to the
                // thermal event, not to Apollo's coincident intervention.
                {
                    let thermal_throttling_now = state
                        .hardware
                        .lock_recover()
                        .last_hw_snapshot
                        .as_ref()
                        .map(|hw| !matches!(hw.thermal_state, ThermalState::Normal))
                        .unwrap_or(false);
                    if thermal_throttling_now && !prev_thermal_throttling {
                        lctx.causal_graph.record_external_event(
                            apollo_engine::engine::causal_graph::ExternalEventKind::ThermalThrottle,
                            snapshot.pressure.memory_pressure,
                            std::time::SystemTime::now(),
                        );
                    }
                    prev_thermal_throttling = thermal_throttling_now;
                }

                // Sprint 12 Convergence #4 (2026-05-17). Real-time
                // architect probe: did the scorer override fire this
                // cycle AND was a thermal-throttle event recorded inside
                // EXTERNAL_BLAME_WINDOW? When both are true, the
                // learned policy disagreed with the gate during the
                // exact window the SoC was thermally throttled — the
                // strongest evidence available that the policy itself
                // is misbehaving (vs the gate, vs the thermal sensor).
                // [Pearl 2009 §3] confounder adjustment; [Sutton & Barto
                // 2018 §11.7] model-free policy correction.
                {
                    let cur_override_rejects = apollo_engine::engine::lse_counters::LSE_COUNTERS
                        .scorer_override_rejects_total
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let delta = cur_override_rejects.saturating_sub(prev_scorer_override_rejects);
                    if delta > 0
                        && lctx.causal_graph.has_recent_external_event(
                            apollo_engine::engine::causal_graph::ExternalEventKind::ThermalThrottle,
                            std::time::SystemTime::now(),
                        )
                    {
                        for _ in 0..delta {
                            apollo_engine::engine::lse_counters::LSE_COUNTERS
                                .inc_causal_thermal_scorer_override_alignment();
                        }
                    }
                    prev_scorer_override_rejects = cur_override_rejects;
                }

                // Sprint 12 Convergence #5 (2026-05-17). Same formula G12 uses
                // for action_queue DRAM backpressure (main.rs:1476): trust the
                // entitled IOReport reading when it's alive, fall back to
                // entropy_anomaly > 2.0 when amc_bandwidth_pct is dead on M1.
                // Keeps both consumers in lockstep — there is no second
                // "bus saturated" definition floating around.
                let dram_bw_pct = last_ioreport
                    .as_ref()
                    .map(|ir| ir.amc_bandwidth_pct)
                    .unwrap_or(0.0);
                let bus_saturated =
                    dram_bw_pct > 80.0 || (dram_bw_pct == 0.0 && prev_entropy_anomaly > 2.0);
                let maintenance_output =
                    daemon_maintenance_tick::run_maintenance_tick_with_decisions(
                        &snapshot,
                        &user_context,
                        &mut maintenance_state,
                        lf_metrics,
                        build_tracker.build_active,
                        bus_saturated,
                        cycle_count,
                        &async_commands,
                    );
                cycle_decision_events.extend_buffer(&maintenance_output.decision_events);
                for _ in 0..async_completion_stats.maintenance_purges_applied {
                    // Record cause for observational outcome tracking via CausalGraph.
                    // Only a confirmed zero exit status reaches this path.
                    lctx.causal_graph.record_action_with_resources(
                        "system_maintenance_purge",
                        snapshot.pressure.memory_pressure as f32,
                        cycle_count,
                        Default::default(),
                    );
                }

                // ── Neuromodulator: bio-inspired parameter modulation ────────
                // Extracted to daemon_neuro_tick::apply_neuromodulator (Wave 8).
                // Best-available measured CPU temperature. Normalized IOPM
                // thermal states are handled separately and are not degrees.
                let cpu_temp_celsius: Option<f64> = last_smc
                    .as_ref()
                    .and_then(|s| s.cpu_temp_celsius)
                    .or_else(|| {
                        cycle_hw_snap
                            .as_ref()
                            .filter(|h| !h.temps_estimated)
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
                        * (-signal_digest.pressure_velocity).max(0.0))
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
                    acc.push_freeze(
                        target.pid,
                        target.name.clone(),
                        format!(
                            "memory-leak recovery: prob={:.2} rss={}MB attempts={}",
                            target.leak_probability,
                            target.rss_bytes / 1024 / 1024,
                            target.recovery_attempts,
                        ),
                        DecisionReason::PressureContext,
                        ss,
                        su,
                        EmitContext::new(
                            ActionPhase::Survival,
                            "main.rs::3302 proc_recovery_freeze",
                            "memory_leak_recovery",
                        ),
                        lf_metrics,
                    );
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

                // Targeted warn-limit paging + expiry.
                // Extracted to daemon_warn_limits::run_warn_limits (Wave 23).
                let applied_warn_limits = daemon_warn_limits::run_warn_limits(
                    snapshot.pressure.memory_pressure,
                    snapshot.pressure.swap_used_bytes,
                    is_root,
                    foreground_pid,
                    &process_tree,
                    &proc_snaps,
                    &coalition_tracker,
                    &state,
                    &heuristic_critical_pids,
                    &mut warn_limit_pids,
                );

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
                if let Ok(tracker) = apollo_engine::engine::contention_tracker::global().lock() {
                    // 0.85: see metrics-population site for the rationale —
                    // Darwin's runnable counter saturates above 0.5 under any
                    // normal load, so a lower threshold misclassifies normal
                    // multitasking as a stability problem.
                    stability_oracle.record_stall_fraction(tracker.stall_fraction(0.85));
                }
                {
                    let mut m = state.metrics.lock_recover();
                    m.metrics.ml_confidence = ml_class.confidence;
                    // 2026-05-12: Media-playback override — the static workload
                    // signatures list IINA/VLC/QuickTime but miss the common
                    // browser-streaming case (YouTube/Twitch/Netflix). The
                    // CoreAudio `kAudioDevicePropertyDeviceIsRunningSomewhere`
                    // signal is the canonical "media is flowing somewhere"
                    // probe and does NOT depend on foreground. Triggering on
                    // audio_active alone correctly covers:
                    //   - YouTube in Brave while user tabbed to Alacritty
                    //   - Spotify in background while coding
                    //   - VLC playing offscreen
                    // The downstream chromium gate still scopes its action to
                    // INVISIBLE renderers (CGWindowList), so a foreground
                    // browser tab playing audio never gets demoted/frozen by
                    // the media-aware threshold.
                    let media_override = user_context.audio_active;
                    m.metrics.current_workload = if media_override {
                        "mediaplayback".to_string()
                    } else {
                        // Sprint patch (2026-06-05): canonical kebab (round-trips
                        // with workload_type_from_str). Legacy `Debug.to_lowercase()`
                        // collapsed "VideoCall" → "videocall" which the parser
                        // could not recover.
                        ml_class.workload.as_kebab().to_string()
                    };
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
                    // Sprint patch (2026-06-05): canonical kebab name.
                    let wl_str = ml_class.workload.as_kebab();
                    let gpu_hints = gpu_mgr.optimize_for_workload(wl_str);
                    if !gpu_hints.is_empty() && cycle_count.is_multiple_of(30) {
                        audit_log(&serde_json::json!({
                            "event": "gpu_workload_hint",
                            "workload": wl_str,
                            "hints": gpu_hints,
                        }));
                    }
                }

                // Sysctl Governor reuses the direct-kernel TCP sample refreshed
                // above; no second probe or subprocess occurs here.
                // WebRTC guard (2026-06-09 prod incident): both default
                // audio devices running = full-duplex realtime call.
                // Inhibits TCP buffer scale-down and delayed_ack=3.
                // See `coreaudio_active::is_realtime_call_active` doc.
                let audio_call_active =
                    apollo_engine::engine::coreaudio_active::is_realtime_call_active();
                // B.2 replayd gate (2026-06-09): screen capture without
                // full-duplex audio (screen share leg of a call, recording)
                // must ALSO inhibit the governor — the post-screen-share
                // whipsaw ("high retransmissions scaling UP" then "-25%
                // scale-down") fired with the audio gate dark.
                let screen_capture_active = screen_capture_cache.check();
                if !audio_call_active && screen_capture_active {
                    // Count only when the screen probe is the DECIDING
                    // signal — keeps the two inhibit counters disjoint.
                    lf_metrics.inc_sysctl_governor_screen_capture_inhibit();
                }
                let sysctl_actions = sysctl_governor.tick(&SysctlGovernorInput {
                    net_monitor: &network_monitor,
                    swap_trend: swap_forecast.swap_trend,
                    memory_pressure: snapshot.pressure.memory_pressure,
                    // Sprint patch (2026-06-05): canonical kebab name.
                    workload: ml_class.workload.as_kebab(),
                    on_battery: power_mgr.is_on_battery(),
                    is_root,
                    realtime_call_active: audio_call_active || screen_capture_active,
                });
                // sysctl_governor.tick() returns sealed `SetSysctlAction` values
                // already constructed via the clamping factory (Fase 4 seal). They
                // are validated; we go through `push_raw` to record the emit
                // context and wire the per-variant counter without re-clamping.
                acc.extend_raw(
                    sysctl_actions,
                    EmitContext::new(
                        ActionPhase::SysctlGovernor,
                        "main.rs::3446 sysctl_governor.tick",
                        "reactive_tcp+memory",
                    ),
                    lf_metrics,
                );

                // TOMBSTONE (2026-06-09): NetworkOptimizer write path DELETED.
                // Two writers on the same TCP sysctls caused a tug-of-war flap
                // (12x "Battery profile" writes vs 14x "reverting to default"
                // in 300 actions) and an ungated mid-call write during the
                // 2026-06-09T17:12 Meet incident — 257baea only gated the
                // SysctlGovernor path, not this one. SysctlGovernor is the
                // SOLE owner of TCP sysctl writes. network_optimizer.rs stays
                // as an advisory-only library; do NOT re-wire it to a write
                // path.

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
                        let mut frozen_dummy = std::collections::HashSet::new();
                        let outcomes = ActuationBroker::from_runtime(&caps, dry_run)
                            .execute(ActuationRequest {
                                actions: revert_actions,
                                caps: &caps,
                                journal_path: &journal_path,
                                frozen: &mut frozen_dummy,
                                learned_protected: &[],
                                learned_interactive: &[],
                                qos_mgr: None,
                                memory_pressure: 0.0,
                                thrashing_score: 0.0,
                                coalition_guard: None,
                                cpu_pegged_fraction: 0.0,
                                async_commands: Some(&async_commands),
                            })
                            .outcomes;
                        cycle_decision_events
                            .extend_buffer_at_cycle(&outcomes.decision_events, cycle_count);
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

                // Sprint 4 Fase 5 — terminal exit of the typed accumulator.
                // Per-variant + raw + rejected_shape counters were already
                // published into `lf_metrics` per push; we read the local
                // telemetry here for cycle-level audit logging when verbose.
                let acc_telemetry = acc.telemetry();
                let mut actions: Vec<RootAction> = acc.finalize();
                if acc_telemetry.rejected_shape > 0 {
                    tracing::warn!(
                        target: "apollo.accumulator",
                        rejected = acc_telemetry.rejected_shape,
                        total_pushed = acc_telemetry.total_pushed,
                        "ActionAccumulator finalize: shape-rejected pushes this cycle"
                    );
                }

                // F3 + F4 — Safety Precedence + Thermal Master Switch.
                // Extracted to daemon_action_safety::apply_pre_exec_safety_filters (Wave 36).
                // [Fowler 2004] Strangler Fig — pure move, no semantic change.
                daemon_action_safety::apply_pre_exec_safety_filters(
                    &mut actions,
                    foreground_pid,
                    &process_tree,
                    foreground_app.as_deref(),
                    &fg_detector,
                    thermal_emergency,
                    &state,
                );

                // ── Chromium Renderer Manager ────────────────────────────────────
                // Extracted to daemon_chromium_tick::run_chromium_tick (Wave 11).
                // [Denning 1968] Working Set | [Jones 2011] Chromium Multi-Process Architecture
                let _t_chrom_start = Instant::now();
                // 2026-05-13: removed watchdog skip on chromium tick — was
                // suppressing chromium_gate_regime + chromium_renderers_frozen
                // + chromium_freed_mb metric writes under stress, leaving them
                // null/stale and masking the daemon's actual behavior. Empirical:
                // stage_reason_chromium_avg ~0.78ms — far cheaper than the
                // observability cost of skipping it. Watchdog still applies to
                // HoltWinters + PageReclaim where the cost-benefit favors skip.
                {
                    // Workload-aware chromium gates. Three independent signals
                    // feed the priority-strict-max chain inside run_chromium_tick:
                    //   media : any audio flowing (CoreAudio).
                    //   build : BuildPhase != Idle (cargo+rustc detected).
                    //   call  : pmset PreventUserIdleSleep + call_app_name match.
                    // The chain in daemon_chromium_tick.rs picks ONE regime —
                    // never additive — so freeze_protected invariants hold.
                    let media_active_chromium = user_context.audio_active;
                    let build_active_chromium = build_tracker.phase
                        != apollo_engine::engine::build_tracker::BuildPhase::Idle;
                    let call_active_chromium = user_context.call_in_progress;
                    // 2026-05-12: LLM inference also relaxes the chromium gate.
                    // llama-server / ollama hold 4GB resident; their detection
                    // already adds +0.20 pressure boost via aggregator, but that
                    // path is dampened by smoothing/clamps. A direct chromium gate
                    // entry at parity with media (0.60, 5000) is more immediate.
                    // `softly_protected_processes()` keeps llama-server itself
                    // safe from being frozen at sub-survival pressure.
                    let llm_active_chromium = llm_active;
                    // 2026-05-12: cached external-display snapshot. cg_display
                    // FFI costs ~50µs but topology changes rarely — sample every
                    // 50 cycles (~5 s @ 10 Hz). The cached value lives in a
                    // function-level static since DisplayState is Copy and the
                    // daemon loop is single-threaded for chromium tick.
                    static mut DISPLAY_STATE_CACHE:
                        apollo_engine::engine::cg_display::DisplayState =
                        apollo_engine::engine::cg_display::DisplayState {
                            display_count: 0,
                            external_4k_attached: false,
                        };
                    let display_state_now = unsafe {
                        if cycle_count.is_multiple_of(50) {
                            DISPLAY_STATE_CACHE = apollo_engine::engine::cg_display::snapshot();
                        }
                        DISPLAY_STATE_CACHE
                    };
                    let external_4k_attached = display_state_now.external_4k_attached;
                    let chromium_output = daemon_chromium_tick::run_chromium_tick(
                        &mut chromium_mgr,
                        &focus_markov,
                        foreground_app.as_deref(),
                        foreground_pid,
                        &proc_snaps,
                        &state,
                        &frozen_state_path,
                        win_workload_intent,
                        &arousal_state,
                        &fluidity_state,
                        signal_digest.pressure_smooth as f32,
                        snapshot.pressure.memory_pressure,
                        cycle_count,
                        signal_digest.swap_velocity_smooth as f32,
                        snapshot.pressure.thrashing_score,
                        media_active_chromium,
                        build_active_chromium,
                        call_active_chromium,
                        llm_active_chromium,
                        external_4k_attached,
                        &world_model,
                        workload_mode.as_str(),
                    );
                    cycle_decision_events.extend_buffer(&chromium_output.decision_events);
                    lf_metrics.record_stage(
                        apollo_engine::engine::lse_counters::CycleStage::ReasonChromium,
                        _t_chrom_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                    );
                }

                // F1: Causal attribution for velocity-anticipatory purges
                // [Pearl 2009] interventional reasoning. Without this the
                // LearningPipeline has a blind spot for whether the purge hint
                // actually drops pressure for this user's workload.
                {
                    let purged = chromium_mgr.drain_purged_this_cycle();
                    if !purged.is_empty() {
                        use apollo_engine::engine::causal_graph::ResourceSnapshot;
                        let pressure_now = snapshot.pressure.memory_pressure as f32;
                        let swap_mb_now = snapshot.pressure.swap_used_bytes as f32 / 1_048_576.0;
                        for (pid, name) in &purged {
                            let res = proc_snaps
                                .iter()
                                .find(|p| p.pid == *pid)
                                .map(|p| ResourceSnapshot {
                                    rss_mb: p.rss_bytes as f32 / 1_048_576.0,
                                    cpu_pct: p.cpu_percent,
                                    swap_mb: swap_mb_now,
                                })
                                .unwrap_or_default();
                            lctx.causal_graph.record_action_with_resources(
                                &format!("purge_purgeable:{}", name),
                                pressure_now,
                                cycle_count,
                                res,
                            );
                        }
                    }
                }

                let base_policy = SafetyPolicy::for_capabilities(
                    SafetyPolicy::for_profile(current_profile),
                    hw_cores,
                    hw_ram_gb,
                );
                let policy = SafetyPolicy::with_pressure_modulation(
                    &base_policy,
                    signal_digest.pressure_smooth,
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

                // Phase 1: Compute budget-filtered actions (metrics lock held briefly).
                // BUG 5 fix: split into three phases so the metrics mutex is never held
                // across the blocking I/O inside execute_actions.
                let prebudget_actions = {
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
                    // Schwartzian transform: compute tau once per action, sort by key.
                    // Halves tau_for_app() calls (O(N log N) lookups → O(N)).
                    // [Knuth TAOCP Vol 3 §5.2] decorate-sort-undecorate.
                    let total_actions = graced_actions.len();
                    let mut keyed_freezes: Vec<(f64, RootAction)> = Vec::new();
                    let mut non_freezes: Vec<RootAction> = Vec::new();
                    for action in graced_actions {
                        if let RootAction::FreezeProcess { ref name, .. } = action {
                            keyed_freezes.push((unfreeze_decay.tau_for_app(name), action));
                        } else {
                            non_freezes.push(action);
                        }
                    }
                    keyed_freezes
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    graced_actions = Vec::with_capacity(total_actions);
                    graced_actions.extend(keyed_freezes.into_iter().map(|(_, action)| action));
                    graced_actions.extend(non_freezes);

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
                    // Shadow aggregate: max hot-page fraction / WSS across candidates this cycle.
                    // Published to shadow_signals after the loop for next cycle's class-level probe.
                    let mut max_hot_frac: f64 = 0.0;
                    let mut max_wss_mb: f64 = 0.0;
                    let confirmed_actions: Vec<RootAction> = confirmed_actions.into_iter().filter_map(|a| {
                        if let RootAction::FreezeProcess { pid, name: ref freeze_name, ref reason, .. } = a {
                            // query_memory_profile falls back to proc_pid_rusage (~3µs)
                            // when task_for_pid fails (ad-hoc signing). No timeout needed.
                            if let Some(profile) = query_memory_profile(pid) {
                                ds_scans += 1;
                                let fault_rate = mem_analyzer.major_fault_rate(pid);
                                // Shadow aggregate: track max WSS across candidates for next-cycle probe.
                                let wss_mb_i = profile.working_set_bytes as f64 / (1024.0 * 1024.0);
                                if wss_mb_i > max_wss_mb { max_wss_mb = wss_mb_i; }
                                // Deep scan: vm_region + temperature (only in mid/high zone).
                                let temp = if signal_digest.pressure_smooth >= 0.30 {
                                    ds_probes += 1;
                                    sample_process_temperature(pid)
                                } else {
                                    None
                                };
                                // Shadow aggregate: track max hot-page fraction when temp probe ran.
                                if let Some(tp) = &temp {
                                    if tp.pct_hot > max_hot_frac { max_hot_frac = tp.pct_hot; }
                                }
                                // Cable: classify_by_memory() → skip freezing LLM/Database processes.
                                // If vm_region scan reveals an LLM inference or database layout,
                                // freezing would be destructive (model eviction, buffer pool loss).
                                let memory_hint = scan_regions(pid)
                                    .and_then(|regions| classify_by_memory(&regions));
                                if let Some((hint, conf)) = &memory_hint {
                                    use apollo_engine::engine::workload_classifier::MemoryLayoutHint;
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
                                        // Cross-cycle dedup (SuperPlan Iter 7):
                        // deep-scan path emits SetMemorystatus -1 every cycle for
                        // stale apps. Check prior-cycle state here; the final
                        // dispatch chokepoint records survivors.
                                        if recently_applied.is_recent(
                                            pid,
                                            apollo_engine::engine::recently_applied::CachedActionKind::SetMemorystatus
                                        ) {
                                            None::<RootAction>
                                        } else {
                                        // Cable: purge_purgeable_regions() → reclaim RAM without freeze.
                                        // Switch-5b (2026-06-03): route through PurgeableEffector
                                        // via mediator chokepoint. Receipt no_op flag accumulates
                                        // into mediator_noop_writes_total when zero regions purged.
                                        if profile.purgeable_bytes > 10 * 1024 * 1024 {
                                            let purge_effector =
                                                apollo_engine::engine::mediator::PurgeableEffector;
                                            let purge_eff =
                                                apollo_engine::engine::mediator::Effect::PurgeHint {
                                                    pid,
                                                    start_sec: 0,
                                                    target_bytes: profile.purgeable_bytes,
                                                };
                                            let purge_receipt =
                                                apollo_engine::engine::mediator::mediate(
                                                    &purge_eff,
                                                    &apollo_engine::engine::mediator::PreCondition::default(),
                                                    &purge_effector,
                                                );
                                            // Receipt doesn't carry region count; use no_op as proxy.
                                            let purged: u32 = match &purge_receipt {
                                                Ok(r) if !r.no_op => 1,
                                                _ => 0,
                                            };
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
                                            decision_reason: DecisionReason::MemoryBudget,
                                        })
                                        }
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
                    // Publish shadow aggregates for next cycle's class-level scorer probe.
                    if max_hot_frac > 0.0 {
                        apollo_engine::engine::shadow_signals::set_max_hot_page_fraction(
                            max_hot_frac,
                        );
                    }
                    if max_wss_mb > 0.0 {
                        apollo_engine::engine::shadow_signals::set_max_wss_mb(max_wss_mb);
                    }

                    // Rosetta AOT: skip freezing oahd/oahd-helper during AOT compilation.
                    let confirmed_actions: Vec<RootAction> = if rosetta_monitor.is_compiling() {
                        let rosetta_immune = apollo_engine::engine::rosetta_monitor::RosettaMonitor::immune_processes();
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
                    if !stuck_pids.is_empty() {
                        let outcome = apollo_engine::engine::daemon_helpers::unfreeze_pids_outcome(
                            stuck_pids.iter().copied(),
                        );
                        let recovery_events = daemon_wake_unfreeze::recovery_unfreeze_events(
                            &outcome,
                            cycle_count,
                            "unfreeze:deadlock_recovery",
                            "deadlock-recovery",
                        );
                        cycle_decision_events.extend_buffer(&recovery_events);
                        let mut frozen_map = state.frozen_state.lock_recover();
                        for pid in outcome.forgettable_pids() {
                            frozen_map.remove(&pid);
                        }
                        write_frozen_state(&frozen_state_path, &frozen_map);
                        metrics.metrics.unfreezes_applied += outcome.applied_count();
                    }

                    let filtered = process_enrichment::filter_boost_cooldown(
                        confirmed_actions,
                        &policy,
                        &mut thrash,
                    );
                    filtered
                    // metrics lock released here
                };

                // Universal safety filter runs BEFORE budget accounting. Previously
                // Apple/protected candidates consumed the complete minute budget and
                // were then discarded here, starving valid third-party work behind
                // them. Preserve the exact safety rules while charging only actions
                // that can actually reach dispatch.
                let policy_protected: Vec<String> = {
                    let pg = state.policy.lock_recover();
                    pg.learned_policy
                        .protected_patterns
                        .iter()
                        .chain(pg.learned_policy.interactive_patterns.iter())
                        .cloned()
                        .collect()
                };
                let (mut budget_candidates, prefilter_stats) =
                    daemon_action_safety::filter_dispatch_candidates(
                        prebudget_actions,
                        &policy_protected,
                        &recently_applied,
                        apollo_engine::engine::process_identity::is_apple_platform_process,
                        |action| pid_identity_still_valid(action, &identity_cache, lf_metrics),
                    );

                // Modern macOS kernels may omit the legacy per-PID pressure
                // sysctl. Drop those hints before budgeting so an unsupported
                // actuator cannot consume budget, create backpressure, or fake work.
                let before_capability_filter = budget_candidates.len();
                if !caps.can_memory_pressure_send {
                    budget_candidates
                        .retain(|action| !matches!(action, RootAction::SetMemorystatus { .. }));
                }
                let unsupported_drops =
                    before_capability_filter.saturating_sub(budget_candidates.len()) as u64;

                let minute_cap = match latency_target {
                    LatencyTarget::Max => 120,
                    LatencyTarget::Low => 50,
                    LatencyTarget::Normal => 80,
                };
                let final_actions = {
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.dispatch_prefilter_input_total += prefilter_stats.input;
                    metrics.metrics.dispatch_prefilter_recent_drops += prefilter_stats.recent_drops;
                    metrics.metrics.dispatch_prefilter_apple_drops += prefilter_stats.apple_drops;
                    metrics.metrics.dispatch_prefilter_protected_drops +=
                        prefilter_stats.protected_drops;
                    metrics.metrics.dispatch_prefilter_identity_drops +=
                        prefilter_stats.identity_drops;
                    metrics.metrics.dispatch_prefilter_unsupported_drops += unsupported_drops;
                    metrics.metrics.dispatch_prefilter_survivors_total +=
                        prefilter_stats.survivors.saturating_sub(unsupported_drops);
                    let admitted = enforce_limits_with_budget(
                        budget_candidates,
                        &policy,
                        &mut metrics.metrics.budgets,
                        minute_cap,
                    );
                    metrics.metrics.last_actions_summary =
                        metrics.metrics.format_last_actions_summary(&admitted);
                    admitted
                };

                // Central intent planning: resolve contradictory same-process
                // requests and rank equivalent accelerators using conservative
                // World Model utility. Specialist safety gates still run again
                // in dispatch immediately before root execution.
                let gpu_gate = {
                    let metrics = state.metrics.lock_recover();
                    apollo_engine::engine::gpu_imagination::GpuImaginationGate {
                        speculation_allowed: metrics.metrics.apollo_overhead_speculation_allowed,
                        memory_pressure: snapshot.pressure.memory_pressure,
                        gpu_load: metrics
                            .metrics
                            .ioreport_gpu_pct
                            .max(fluidity_state.gpu_render_load as f64),
                        gpu_watts: metrics.metrics.energy_gpu_watts,
                        thermal_nominal: snapshot.pressure.thermal_level == "nominal",
                        app_launching: metrics.metrics.app_launching,
                        fluidity_degraded: metrics.metrics.fluidity_degraded,
                    }
                };
                let gpu_external_candidates = {
                    let metrics = state.metrics.lock_recover();
                    daemon_dispatch_tick::build_gpu_candidate_portfolio(
                        &world_model,
                        workload_mode.as_str(),
                        &metrics.metrics,
                    )
                };
                let gpu_context_revision = gpu_context_epoch
                    ^ ((parallel_runtime.worker_threads as u64) << 32)
                    ^ (u64::from(gpu_gate.thermal_nominal) << 8)
                    ^ (u64::from(gpu_gate.gpu_load >= 0.20) << 7)
                    ^ (u64::from(gpu_gate.memory_pressure >= 0.55) << 6)
                    ^ (u64::from(gpu_gate.app_launching) << 5)
                    ^ (u64::from(gpu_gate.fluidity_degraded) << 4)
                    ^ (u64::from(!gpu_gate.speculation_allowed) << 3)
                    ^ (u64::from(gpu_gate.gpu_watts.is_some()) << 2)
                    ^ u64::from(
                        gpu_gate
                            .gpu_watts
                            .is_some_and(|watts| !watts.is_finite() || watts >= 2.5),
                    );
                world_model.set_gpu_context_revision(gpu_context_revision);
                let (final_actions, plan_report) =
                    daemon_dispatch_tick::plan_action_intents_with_gpu(
                        final_actions,
                        &world_model,
                        workload_mode.as_str(),
                        snapshot.pressure.memory_pressure,
                        gpu_gate.app_launching,
                        gpu_gate.fluidity_degraded,
                        daemon_dispatch_tick::GpuPlannerRuntime {
                            cycle: cycle_count,
                            context_revision: gpu_context_revision,
                            worker: &mut gpu_imagination,
                            gate: gpu_gate,
                            external_candidates: gpu_external_candidates,
                        },
                    );
                if let Some(result) = plan_report.gpu_imagination_result.as_ref() {
                    world_model.attach_gpu_imagination(result);
                    telemetry_medallion.observe_gpu_imagination(result, cycle_count);
                }
                // A root rank commonly consumes advice cached from an earlier
                // async batch, so consumption cannot be nested under "a job
                // completed this cycle". Record it independently for the
                // Bronze → Silver → Gold calibration lifecycle.
                for influence in &plan_report.gpu_imagination_supported_actions {
                    telemetry_medallion.mark_gpu_prediction_consumed(
                        &influence.action_key,
                        workload_mode.as_str(),
                        cycle_count,
                    );
                }
                let gpu_circuit_state = gpu_imagination.circuit_state().as_str().to_string();
                let gpu_circuit_reason = gpu_imagination
                    .quarantine_reason()
                    .unwrap_or("")
                    .to_string();
                {
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.action_planner_proposed_total = metrics
                        .metrics
                        .action_planner_proposed_total
                        .saturating_add(plan_report.proposed);
                    metrics.metrics.action_planner_admitted_total = metrics
                        .metrics
                        .action_planner_admitted_total
                        .saturating_add(plan_report.admitted);
                    metrics.metrics.action_planner_conflict_drops_total = metrics
                        .metrics
                        .action_planner_conflict_drops_total
                        .saturating_add(plan_report.conflict_drops);
                    metrics.metrics.action_planner_reorders_total = metrics
                        .metrics
                        .action_planner_reorders_total
                        .saturating_add(plan_report.reordered);
                    metrics.metrics.world_model_utility_promotions_total = metrics
                        .metrics
                        .world_model_utility_promotions_total
                        .saturating_add(plan_report.evidence_ranked);
                    metrics.metrics.world_model_utility_reorders_total = metrics
                        .metrics
                        .world_model_utility_reorders_total
                        .saturating_add(plan_report.reordered);
                    metrics.metrics.world_model_family_prior_uses_total = metrics
                        .metrics
                        .world_model_family_prior_uses_total
                        .saturating_add(plan_report.family_priors_used);
                    metrics.metrics.world_model_counterfactual_rank_uses_total = metrics
                        .metrics
                        .world_model_counterfactual_rank_uses_total
                        .saturating_add(plan_report.counterfactual_ranked);
                    metrics.metrics.world_model_episodic_rank_uses_total = metrics
                        .metrics
                        .world_model_episodic_rank_uses_total
                        .saturating_add(plan_report.episodic_ranked);
                    metrics.metrics.world_model_temporal_memory_samples =
                        plan_report.temporal_memory_samples;
                    metrics.metrics.world_model_sequence_candidates_total = metrics
                        .metrics
                        .world_model_sequence_candidates_total
                        .saturating_add(plan_report.temporal_candidates);
                    metrics.metrics.world_model_sequence_rollouts_total = metrics
                        .metrics
                        .world_model_sequence_rollouts_total
                        .saturating_add(plan_report.temporal_rollouts);
                    metrics.metrics.world_model_sequence_promotions_total = metrics
                        .metrics
                        .world_model_sequence_promotions_total
                        .saturating_add(plan_report.temporal_promotions);
                    metrics.metrics.world_model_sequence_authoritative_total = metrics
                        .metrics
                        .world_model_sequence_authoritative_total
                        .saturating_add(u64::from(plan_report.temporal_authoritative));
                    metrics
                        .metrics
                        .world_model_sequence_authoritative_rollouts_total = metrics
                        .metrics
                        .world_model_sequence_authoritative_rollouts_total
                        .saturating_add(plan_report.temporal_authoritative_rollouts);
                    metrics.metrics.world_model_sequence_authoritative =
                        plan_report.temporal_authoritative;
                    metrics.metrics.world_model_dynamics_predictions_total = metrics
                        .metrics
                        .world_model_dynamics_predictions_total
                        .saturating_add(plan_report.dynamics_predictions);
                    metrics
                        .metrics
                        .world_model_dynamics_ranking_predictions_total = metrics
                        .metrics
                        .world_model_dynamics_ranking_predictions_total
                        .saturating_add(plan_report.dynamics_ranking_predictions);
                    metrics
                        .metrics
                        .world_model_dynamics_authoritative_predictions_total = metrics
                        .metrics
                        .world_model_dynamics_authoritative_predictions_total
                        .saturating_add(plan_report.dynamics_authoritative_predictions);
                    metrics.metrics.world_model_dynamics_baseline_uses_total = metrics
                        .metrics
                        .world_model_dynamics_baseline_uses_total
                        .saturating_add(u64::from(plan_report.dynamics_baseline_used));
                    metrics.metrics.world_model_dynamics_mean_uncertainty =
                        plan_report.dynamics_mean_uncertainty;
                    metrics.metrics.gpu_imagination_backend =
                        plan_report.gpu_imagination_backend.clone();
                    metrics.metrics.gpu_imagination_device =
                        plan_report.gpu_imagination_device.clone();
                    metrics.metrics.gpu_imagination_circuit_state = gpu_circuit_state;
                    metrics.metrics.gpu_imagination_circuit_reason = gpu_circuit_reason;
                    metrics.metrics.gpu_imagination_last_submit_outcome =
                        plan_report.gpu_imagination_submit_outcome.clone();
                    if let Some(error) = plan_report.gpu_imagination_error.as_ref() {
                        metrics.metrics.gpu_imagination_last_error = error.clone();
                    }
                    match plan_report.gpu_imagination_submit_outcome.as_str() {
                        "submitted" => {
                            metrics.metrics.gpu_imagination_jobs_submitted_total = metrics
                                .metrics
                                .gpu_imagination_jobs_submitted_total
                                .saturating_add(1);
                        }
                        "busy" | "overhead-budget" | "memory-pressure" | "gpu-busy"
                        | "gpu-power" | "thermal" | "app-launch" | "fluidity" => {
                            metrics.metrics.gpu_imagination_skips_total = metrics
                                .metrics
                                .gpu_imagination_skips_total
                                .saturating_add(1);
                        }
                        _ => {}
                    }
                    for influence in &plan_report.gpu_imagination_supported_actions {
                        metrics.metrics.record_gpu_root_rank_influence(
                            &influence.action_key,
                            influence.support,
                        );
                    }
                    if plan_report.gpu_imagination_completed {
                        if let Some(error) = plan_report.gpu_imagination_error.as_ref() {
                            metrics.metrics.gpu_imagination_jobs_failed_total = metrics
                                .metrics
                                .gpu_imagination_jobs_failed_total
                                .saturating_add(1);
                            metrics.metrics.gpu_imagination_last_error = error.clone();
                        } else {
                            metrics.metrics.gpu_imagination_jobs_completed_total = metrics
                                .metrics
                                .gpu_imagination_jobs_completed_total
                                .saturating_add(1);
                            metrics.metrics.gpu_imagination_last_error.clear();
                        }
                        metrics.metrics.gpu_imagination_samples_total = metrics
                            .metrics
                            .gpu_imagination_samples_total
                            .saturating_add(plan_report.gpu_imagination_samples);
                        metrics.metrics.gpu_imagination_gpu_time_ns_total = metrics
                            .metrics
                            .gpu_imagination_gpu_time_ns_total
                            .saturating_add(plan_report.gpu_imagination_gpu_time_ns);
                        metrics.metrics.gpu_imagination_wall_time_ns_total = metrics
                            .metrics
                            .gpu_imagination_wall_time_ns_total
                            .saturating_add(plan_report.gpu_imagination_wall_time_ns);
                        if let Some(best) = plan_report.gpu_imagination_best_action.as_ref() {
                            metrics.metrics.gpu_imagination_last_best_action = best.clone();
                            metrics.metrics.gpu_imagination_last_positive_probability =
                                plan_report.gpu_imagination_best_positive_probability;
                            metrics.metrics.gpu_imagination_last_p10_gain =
                                plan_report.gpu_imagination_best_p10_gain;
                        }
                    }
                    metrics.metrics.world_model_sequence_expected_gain =
                        plan_report.temporal_expected_gain;
                    metrics.metrics.world_model_sequence_uncertainty =
                        plan_report.temporal_uncertainty;
                    metrics.metrics.world_model_sequence_pressure_delta =
                        plan_report.temporal_pressure_delta;
                    metrics.metrics.world_model_sequence_fluidity_delta =
                        plan_report.temporal_fluidity_delta;
                    metrics.metrics.world_model_sequence_energy_delta =
                        plan_report.temporal_energy_delta;
                    metrics
                        .metrics
                        .world_model_sequence_authoritative_expected_gain =
                        plan_report.temporal_authoritative_expected_gain;
                    metrics
                        .metrics
                        .world_model_sequence_authoritative_uncertainty =
                        plan_report.temporal_authoritative_uncertainty;
                    metrics
                        .metrics
                        .world_model_sequence_authoritative_pressure_delta =
                        plan_report.temporal_authoritative_pressure_delta;
                    metrics
                        .metrics
                        .world_model_sequence_authoritative_fluidity_delta =
                        plan_report.temporal_authoritative_fluidity_delta;
                    metrics
                        .metrics
                        .world_model_sequence_authoritative_energy_delta =
                        plan_report.temporal_authoritative_energy_delta;
                    metrics.metrics.world_model_sequence_best_first =
                        plan_report.temporal_best_first.unwrap_or_default();
                    metrics.metrics.world_model_sequence_best_second =
                        plan_report.temporal_best_second.unwrap_or_default();
                    metrics
                        .metrics
                        .world_model_sequence_authoritative_best_first = plan_report
                        .temporal_authoritative_best_first
                        .unwrap_or_default();
                    metrics
                        .metrics
                        .world_model_sequence_authoritative_best_second = plan_report
                        .temporal_authoritative_best_second
                        .unwrap_or_default();
                    metrics.metrics.world_model_sequence_abstention_reason =
                        plan_report.temporal_abstention_reason.unwrap_or_default();
                    if let Some(resolution) = plan_report.last_resolution {
                        metrics.metrics.action_planner_last_resolution = resolution;
                    }
                }

                // Priority action queue: buffer this cycle's decided actions and
                // dispatch at most max_per_cycle per cycle. Urgent (Unfreeze) actions
                // bypass the cap. Any overflow stays in the queue for the next cycle;
                // the queue owns pending dedup until an action is drained.
                action_queue.push_all(final_actions);
                // Phase A2 (Sprint 2 2026-05-07) — post-drain identity re-verify.
                // Actions queued cycle N may dispatch cycle N+1 due to priority
                // budget; PID can die between push and drain. Re-verify here
                // closes the multi-cycle race window. See pid_identity_still_valid
                // helper for full semantics (mirrors verify_pid_identity).
                // [Idempotency Pattern — 1001 patterns slide 7]
                let final_actions: Vec<RootAction> = action_queue
                    .drain_cycle()
                    .into_iter()
                    .filter(|a| pid_identity_still_valid(a, &identity_cache, lf_metrics))
                    .collect();
                // Update backpressure metrics (observable in runtime_metrics.json).
                {
                    let bp = action_queue.backpressure_ratio();
                    let pending_depth = lctx.outcome_tracker.pending_depth();
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.action_queue_backpressure = bp;
                    metrics.metrics.action_queue_capacity_drops = action_queue.capacity_drops();
                    metrics.metrics.action_queue_background_evictions =
                        action_queue.background_evictions();
                    metrics.metrics.outcome_pending_depth = pending_depth;
                }

                // Give the higher-priority InteractionQos candidate its same-cycle
                // opportunity before dispatch can reserve a Boost omission.
                let qos_cognitive_paused = prev_cog_decision
                    .as_ref()
                    .is_some_and(|decision| decision.pause_learning);
                let mut qos_exploration_environment = || {
                    capture_exploration_environment(
                        &state,
                        pressure_collector.latest().memory_pressure,
                        build_tracker.phase,
                        qos_cognitive_paused,
                        learnable_params.fluidity_degraded_threshold,
                    )
                };
                let markov_confidence = state
                    .metrics
                    .lock_recover()
                    .metrics
                    .markov_prediction_confidence;
                let reflex_model_signals = daemon_cycle_tail::ReflexModelSignals {
                    nars_reliability: (1.0 - lctx.outcome_tracker.nars_drift_score())
                        .clamp(0.0, 1.0),
                    mpc_urgency: signal_digest.urgency.clamp(0.0, 1.0),
                    mpc_recommendation: signal_digest.mpc_recommendation,
                    causal_confidence: causal_confidence
                        .values()
                        .copied()
                        .map(f64::from)
                        .filter(|value| value.is_finite())
                        .fold(0.0, f64::max)
                        .clamp(0.0, 1.0),
                    markov_confidence: if markov_confidence.is_finite() {
                        markov_confidence.clamp(0.0, 1.0)
                    } else {
                        0.0
                    },
                };
                let flow_acceleration_hint = webflow_output
                    .intent
                    .filter(|_| webflow_output.admitted)
                    .and_then(|intent| {
                        foreground_pid.map(|target_pid| daemon_cycle_tail::AccelerationHint {
                            kind: daemon_cycle_tail::AccelerationHintKind::WebNavigation,
                            target_pid,
                            ttl_ms: intent.ttl_ms,
                        })
                    })
                    .or_else(|| {
                        networkflow_output.intent.map(|intent| {
                            daemon_cycle_tail::AccelerationHint {
                                kind: daemon_cycle_tail::AccelerationHintKind::NetworkActivity,
                                target_pid: intent.target_pid,
                                ttl_ms: intent.ttl_ms,
                            }
                        })
                    });
                let acceleration_output = daemon_cycle_tail::update_acceleration_lease(
                    &state,
                    &mut acceleration_lease,
                    &fluidity_state,
                    &thermal_action,
                    foreground_pid,
                    build_tracker.phase,
                    user_context.idle_secs,
                    flow_acceleration_hint,
                    &process_tree,
                    &proc_snaps,
                    &mut io_shaper,
                    &world_model,
                    workload_mode.as_str(),
                    reflex_model_signals,
                    cycle_count,
                    &mut exploration_scheduler,
                    &mut qos_exploration_environment,
                );
                cycle_decision_events.extend_buffer(&acceleration_output.decision_events);
                // Snapshot causal QoS preferences before exec_outcomes consumes final_actions.
                // FreezeProcess actions for CPU-dominant processes will be upgraded to
                // ThrottleProcess(aggressive=true) — QoS Background tier is less invasive
                // than SIGSTOP when CPU reduction is the only causal mechanism needed.
                // [Pearl 2009 §3] mediation analysis  [Nygard 2018] bulkhead: least-invasive first
                let causal_qos_names: std::collections::HashSet<String> =
                    lctx.causal_graph.qos_preferred_names();
                let mut causal_qos_upgrades_cycle = 0u32;

                // Phase 0b: hoist execute-start outside the block scope so it
                // remains visible after the inner block closes (where the
                // record_stage call lives).
                let _t_execute_start_outer;
                let (mut exec_outcomes, causal_qos_upgrades, counterfactual_holdouts) = {
                    use daemon_dispatch_tick::{run_dispatch_tick, DispatchTickInput};
                    // Build the coalition guard: tracker + envelope are owned
                    // long-lived in this scope; the guard is a thin borrow
                    // bundle, cheap to construct each cycle.
                    let cg = apollo_engine::engine::active_coalition_envelope::CoalitionGuard::new(
                        &coalition_tracker,
                        &active_coalitions,
                    );
                    // Phase 0b stage timing: reason complete, execute starts.
                    lf_metrics.record_stage(
                        apollo_engine::engine::lse_counters::CycleStage::Reason,
                        _t_reason_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                    );
                    _t_execute_start_outer = Instant::now();
                    let dispatch_cognitive_paused = prev_cog_decision
                        .as_ref()
                        .is_some_and(|decision| decision.pause_learning);
                    let mut dispatch_exploration_environment = || {
                        capture_exploration_environment(
                            &state,
                            pressure_collector.latest().memory_pressure,
                            build_tracker.phase,
                            dispatch_cognitive_paused,
                            learnable_params.fluidity_degraded_threshold,
                        )
                    };
                    let output = run_dispatch_tick(DispatchTickInput {
                        state: &state,
                        caps: &caps,
                        journal_path: &journal_path,
                        frozen_state_path: &frozen_state_path,
                        final_actions,
                        snapshot: &snapshot,
                        prev_cog_decision: prev_cog_decision.as_ref(),
                        causal_qos_names: &causal_qos_names,
                        reclaim_risk: reclaim_forecast.risk,
                        unfreeze_decay: &mut unfreeze_decay,
                        collector: &collector,
                        dry_run,
                        async_commands: Some(&async_commands),
                        lf_metrics: Some(lf_metrics),
                        coalition_guard: Some(&cg),
                        cpu_pegged_fraction: pressure_collector
                            .latest()
                            .cpu_saturation
                            .pegged_fraction,
                        world_model: &world_model,
                        workload: workload_mode.as_str(),
                        cycle_count,
                        foreground_pid,
                        exploration_scheduler: &mut exploration_scheduler,
                        telemetry_medallion: &mut telemetry_medallion,
                        reflex_broker: Some(&mut acceleration_lease),
                        exploration_environment: &mut dispatch_exploration_environment,
                    });
                    (
                        output.outcomes,
                        output.causal_qos_upgrades,
                        output.counterfactual_holdouts,
                    )
                };
                causal_qos_upgrades_cycle += causal_qos_upgrades;
                cycle_decision_events
                    .extend_buffer_at_cycle(&exec_outcomes.decision_events, cycle_count);
                acceleration_lease.observe_health(
                    &state,
                    cycle_count,
                    &parallel_runtime.build.expected_profile,
                    &parallel_runtime.build.compiled_profile,
                    &parallel_runtime.build.effective_profile,
                    dry_run,
                );
                if cycle_count % 100 == 0 {
                    write_json(
                        std::path::Path::new(reflex_state_path()),
                        acceleration_lease.rollout_state(),
                        Some(0o600),
                    );
                }

                for holdout in &counterfactual_holdouts {
                    tracing::debug!(
                        action_key = %holdout.action_key,
                        correlation = holdout.exploration.correlation.0,
                        workload = workload_mode.as_str(),
                        "world model opened a controlled no-action arm"
                    );
                }

                // Warn limits are already a real, non-fatal kernel intervention.
                // Until now they bypassed the execution audit, so OutcomeTracker
                // never learned whether an accepted warn limit reduced pressure.
                // Record only successful calls; proposals and failed kernel calls
                // remain excluded from the learning signal.
                for applied in applied_warn_limits {
                    exec_outcomes.paging_hints_applied += 1;
                    exec_outcomes.audit_traces.push(PolicyDecisionTrace {
                        t: Utc::now(),
                        cycle: cycle_count,
                        intended_action: RootAction::set_memorystatus(
                            applied.pid,
                            -1,
                            format!(
                                "warn-limit paging: {} ({}MB inactive limit)",
                                applied.name, applied.warn_mb
                            ),
                            DecisionReason::MemoryBudget,
                        ),
                        decision_reason: DecisionReason::MemoryBudget,
                        applied: true,
                        block_reason: None,
                        pressure: snapshot.pressure.memory_pressure as f32,
                        swap_gb: (snapshot.pressure.swap_used_bytes as f32)
                            / (1024.0 * 1024.0 * 1024.0),
                        thrashing: snapshot.pressure.thrashing_score as f32,
                    });
                }
                // Commit cross-cycle suppression only after the actuator confirms
                // a real mutation. Admission, dry-run simulation, cognitive drops
                // and failed kernel calls remain retryable.
                daemon_dispatch_tick::record_applied_actions(
                    &exec_outcomes.audit_traces,
                    &mut recently_applied,
                );
                // Capture only applied pressure-reduction actions for the
                // self-evaluator. Intents blocked during dispatch must not
                // become neurocognitive "latest_action" evidence.
                let action_names_for_outcome =
                    learning_tick::outcome_action_names_from_applied_traces(&exec_outcomes);

                // ActiveCoalition blocks → OutcomeTracker survival-bias channel.
                // The new coalition guard skips actions silently inside
                // execute_actions; without this hook RL/Bayes go blind to the
                // opportunity cost of over-protection. record_blocked sets up
                // the Rubin 1974 counterfactual: if pressure rose >2pp above
                // natural drift in the next 30 s, the block was probably wrong.
                // SHADOW-MODE-ONLY (per OutcomeTracker::record_blocked doc) —
                // never auto-unblocks. NotebookLM peer-review 2026-05-10.
                {
                    use apollo_engine::engine::audit_types::BlockReason;
                    use apollo_engine::engine::types::RootAction;
                    let pressure = snapshot.pressure.memory_pressure;
                    for trace in &exec_outcomes.audit_traces {
                        // Only count blocks that may indicate over-protection.
                        // Hard invariants (ProtectedProcess, MlProtected,
                        // PidRecycled, ApplePlatform, Zombie) are by design
                        // permanent — recording them adds noise. The two
                        // *policy* gates (ActiveCoalition, AssertionActive)
                        // can over-protect: ActiveCoalition is brand-new and
                        // AssertionActive blocks freezes when a per-PID power
                        // assertion is held, which on M1 8GB has historically
                        // caused stuck-pressure (commit e9a5603).
                        let gate = match trace.block_reason {
                            Some(BlockReason::ActiveCoalition) => "active-coalition",
                            Some(BlockReason::AssertionActive) => "assertion-active",
                            _ => continue,
                        };
                        let class = match &trace.intended_action {
                            RootAction::ThrottleProcess { .. } => "throttle",
                            RootAction::FreezeProcess { .. } => "freeze",
                            RootAction::SetMemorystatus { .. } => "memorystatus",
                            RootAction::SetThreadQoS { .. } => "thread-qos",
                            _ => continue,
                        };
                        // Use the lctx-held mutable borrow; the LearningContext
                        // owns &mut outcome_tracker for the rest of the cycle
                        // (see line ~1800 where lctx is constructed).
                        lctx.outcome_tracker.record_blocked(class, gate, pressure);
                    }
                }

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
                    let all_thaw_pids = exec_outcomes
                        .newly_unfrozen_pids
                        .iter()
                        .copied()
                        .chain(wake_thaw_pids.iter().copied());
                    for pid in all_thaw_pids {
                        if let Some(proc) = collector.system().process(sysinfo::Pid::from_u32(pid))
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
                        if let Some(proc) = collector.system().process(sysinfo::Pid::from_u32(pid))
                        {
                            // Provide WSS from TASK_VM_INFO as M∞ ground-truth anchor.
                            // [Denning 1968] — WSS is the reliable predictor of steady-state
                            // RAM demand; eliminates running-max convergence noise.
                            let wss_hint = query_memory_profile(pid).map(|mp| mp.working_set_bytes);
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
                let cognitive_pause = prev_cog_decision.as_ref().is_some_and(|d| d.pause_learning);
                if cognitive_pause {
                    if let Some(lease) = last_markov_prethaw.take() {
                        let report = daemon_markov_tick::release_markov_prewarm_recorded(
                            lease,
                            &state,
                            cycle_count,
                            &mut acceleration_lease,
                        );
                        cycle_decision_events.extend_buffer(&report.decision_events);
                    }
                    cycle_decision_events.extend_buffer(
                        &daemon_cycle_tail::release_acceleration_lease_recorded(
                            &mut acceleration_lease,
                            &state,
                            cycle_count,
                        ),
                    );
                    exploration_scheduler.cancel_active(
                        apollo_engine::engine::exploration_scheduler::TerminalDiagnostic::Cancelled,
                    );
                    tracing::debug!(
                        uchs = prev_cog_decision.as_ref().map_or(0.0, |d| d.uchs_composite),
                        "cognitive gate: learning paused (UCHS recovery mode)"
                    );
                }

                // Record the complete live context and every confirmed
                // actuator independently of the cognitive learning gate.
                // A recovery-mode pause must not erase what Apollo or macOS
                // did; it only pauses downstream parameter updates.
                let mut attribution_by_key: HashMap<_, VecDeque<_>> = HashMap::new();
                for attribution in plan_report.decision_attributions.iter().cloned() {
                    attribution_by_key
                        .entry(attribution.action_key.clone())
                        .or_default()
                        .push_back(attribution);
                }
                for trace in exec_outcomes
                    .audit_traces
                    .iter()
                    .filter(|trace| trace.applied)
                {
                    let Some(key) = apollo_engine::engine::telemetry_medallion::actuator_action_key(
                        &trace.intended_action,
                    ) else {
                        continue;
                    };
                    if let Some(attribution) = attribution_by_key
                        .get_mut(&key)
                        .and_then(VecDeque::pop_front)
                    {
                        telemetry_medallion.stage_decision_attribution(attribution);
                    }
                }
                let causal_runtime_context = state.metrics.lock_recover().metrics.clone();
                telemetry_medallion.observe(
                    apollo_engine::engine::telemetry_medallion::TelemetryObservation {
                        snapshot: &snapshot,
                        hardware: cycle_hw_snap.as_ref(),
                        runtime: &causal_runtime_context,
                        capabilities: Some(&caps),
                        signal: &signal_digest,
                        workload: workload_mode.as_str(),
                        cycle: cycle_count,
                        outcomes: &exec_outcomes,
                        intervention: agent_intervention,
                        applied_intervention: applied_predictive_intervention,
                        purge_recent: maintenance_state.is_purge_recent(30),
                        nars_drift_score: lctx.outcome_tracker.nars_drift_score(),
                        nars_beliefs_total: lctx.outcome_tracker.drift_detector.len() as u64,
                        natural_drift: lctx.outcome_tracker.natural_drift(),
                        arousal_level: arousal_state.level as f64,
                    },
                );
                let mut causal_actuator_gold = 0_u64;
                let new_gold_evidence = telemetry_medallion.drain_new_gold_evidence();
                let system1_struggling = lctx.outcome_tracker.heuristic_is_struggling();
                let dr_zero_self_challenge = lctx.outcome_tracker.self_challenge_score();
                for evidence in &new_gold_evidence {
                    causal_actuator_gold = causal_actuator_gold.saturating_add(u64::from(
                        lctx.causal_graph
                            .observe_actuator_outcome(evidence, installation_id),
                    ));
                    if !cognitive_pause {
                        let report = local_consolidator.consolidate(
                            evidence,
                            &mut lctx.outcome_tracker.drift_detector,
                            &mut arousal_state,
                            system1_struggling,
                            dr_zero_self_challenge,
                        );
                        if !matches!(
                            report.verdict,
                            LocalConsolidationVerdict::Rejected
                                | LocalConsolidationVerdict::Duplicate
                        ) {
                            let mut metrics = state.metrics.lock_recover();
                            metrics.metrics.local_consolidations =
                                metrics.metrics.local_consolidations.saturating_add(1);
                            match report.verdict {
                                LocalConsolidationVerdict::Improved => {
                                    metrics.metrics.local_consolidation_improvements = metrics
                                        .metrics
                                        .local_consolidation_improvements
                                        .saturating_add(1);
                                }
                                LocalConsolidationVerdict::Worsened => {
                                    metrics.metrics.local_consolidation_regressions = metrics
                                        .metrics
                                        .local_consolidation_regressions
                                        .saturating_add(1);
                                }
                                LocalConsolidationVerdict::Neutral => {
                                    metrics.metrics.local_consolidation_neutral = metrics
                                        .metrics
                                        .local_consolidation_neutral
                                        .saturating_add(1);
                                }
                                LocalConsolidationVerdict::Duplicate
                                | LocalConsolidationVerdict::Rejected => {}
                            }
                            metrics.metrics.local_consolidation_system1_updates = metrics
                                .metrics
                                .local_consolidation_system1_updates
                                .saturating_add(u64::from(report.system1_updates));
                            metrics.metrics.local_consolidation_last_family = report.family;
                            metrics.metrics.local_consolidation_last_action = report.action_key;
                            metrics.metrics.local_consolidation_last_verdict =
                                report.verdict.as_str().to_string();
                            metrics.metrics.local_consolidation_last_utility = report.utility;
                        }
                    }
                }
                if causal_actuator_gold > 0 {
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.world_model_causal_actuator_gold_total = metrics
                        .metrics
                        .world_model_causal_actuator_gold_total
                        .saturating_add(causal_actuator_gold);
                }
                // Refresh the facade after observation so metrics history and
                // dashboard consume this cycle's context and any outcomes that
                // just matured. Dispatch already used the prior-cycle model.
                world_model.attach_context(telemetry_medallion.trusted_view());
                world_model.attach_local_consolidation(
                    local_consolidator.view_for_installation(installation_id),
                );

                // NARS belief routing for newly-frozen non-chromium PIDs
                // (Sprint A 2026-05-10). chromium_mgr already routes its own
                // renderer freezes via observe_freeze_outcome → classify(name)
                // (legacy path). For everything else, classify_full now routes
                // apple-owned and companion-of-fg processes to their own NARS
                // categories instead of corrupting `generic` / `app-helper`
                // truth-values. Closes the NotebookLM round-3 NARS visibility
                // gap (regime-shift drift).
                {
                    let intel = chromium_mgr.intelligence_mut();
                    for &pid in &exec_outcomes.newly_frozen_pids {
                        // Look up name from snapshot — bounded O(top_processes).
                        let name = snapshot
                            .top_processes
                            .iter()
                            .find(|p| p.pid == pid)
                            .map(|p| p.name.clone())
                            .unwrap_or_default();
                        if name.is_empty() {
                            continue;
                        }
                        // Live freeze observation: success=alive (re-thaw worked
                        // / process still healthy). For routing-only purposes
                        // we record success=true with low salience so legacy
                        // beliefs aren't perturbed; the goal is preserving
                        // `generic` calibration by diverting the
                        // structurally-protected category traffic.
                        intel.observe_full(
                            &name,
                            Some(pid),
                            foreground_app.as_deref(),
                            Some(&companion_graph),
                            true,
                            0.10,
                        );
                    }
                }

                // Phase 0b stage timing: execute complete, learn starts.
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::Execute,
                    _t_execute_start_outer
                        .elapsed()
                        .as_nanos()
                        .min(u64::MAX as u128) as u64,
                );
                let _t_learn_start = Instant::now();

                // Learning tick: outcome tracking, causal graph, RL cables, predictive
                // agent, and periodic persist (every 100 cycles). Extracted to
                // learning_tick.rs for readability; behaviour is unchanged.
                // Skipped when UCHS recovery mode active (cognitive_pause).
                if !cognitive_pause {
                    let microexperiment_checkpoint = cycle_count
                        .is_multiple_of(100)
                        .then(|| microexperiment_runtime.persisted());
                    learning_tick::run_learning_tick(
                        &snapshot,
                        &cycle_hw_snap,
                        &exec_outcomes,
                        &signal_digest,
                        workload_mode,
                        cycle_count,
                        &state,
                        &collector,
                        &mut lctx,
                        &mut learning_pipeline,
                        &mut telemetry_medallion,
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
                        &maintenance_state,
                        &cognitive_state.meta_cognition,
                        &local_consolidator,
                        &exploration_scheduler,
                        microexperiment_checkpoint.as_ref(),
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
                  // 2026-05-12: ReasonNeuro stage instrumentation. Was the
                  // largest unmeasured chunk of REASON — blind spot when
                  // p95 spikes under stress.
                let _t_neuro_start = Instant::now();
                let cog_decision = daemon_neuro_tick::run_neurocognitive_tick(
                    &mut lctx,
                    &mut cognitive_state,
                    cycle_count,
                    &signal_digest,
                    &action_names_for_outcome,
                    workload_mode.as_str(),
                );
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::ReasonNeuro,
                    _t_neuro_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );
                prev_cog_decision = Some(cog_decision);
                // Autonomous rule induction every 100 cycles.
                // Extracted to daemon_skill_tick::run_rule_induction (Wave 22).
                if cycle_count.is_multiple_of(100) {
                    daemon_skill_tick::run_rule_induction(
                        lctx.skill_registry,
                        lctx.outcome_tracker,
                        &state,
                        workload_mode.as_str(),
                        std::path::Path::new(skills_path()),
                    );
                }
                // State compression (% 500) is handled by run_periodic() below.
                // Hourly housekeeping (7200 cycles × 500ms ≈ 1 hour).
                if cycle_count.is_multiple_of(7200) {
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
                metrics_reporter::refresh_unified_learning_health(
                    &mut unified_learning_health,
                    &decision_ledger,
                    &telemetry_medallion,
                    &local_consolidator,
                    &exploration_scheduler,
                );
                metrics_reporter::update_learning_metrics(
                    &state,
                    &lctx,
                    &signal_digest,
                    &agent_intervention,
                    &arousal_state,
                    &learning_pipeline,
                    &telemetry_medallion,
                    &world_model,
                    unified_learning_health.health(),
                );
                metrics_reporter::apply_io_shaping(
                    cycle_count,
                    is_root,
                    &snapshot,
                    foreground_pid,
                    &process_tree,
                    &heuristic_decisions,
                    &heuristic_critical_pids,
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

                metrics_reporter::merge_cycle_metrics(
                    &state,
                    &exec_outcomes,
                    &network_monitor,
                    decision.reactor_event_weight,
                    &decision.blockers,
                    current_profile,
                    &governor_decision,
                    &lctx,
                    &mut ais_worker,
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

                // Phase 0b stage timing: learn complete, persist starts.
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::Learn,
                    _t_learn_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );
                let _t_persist_start = Instant::now();

                // ── Enriched telemetry + UCHS neurocognitive metrics ────────────────────
                // Extracted to daemon_cycle_tail::wire_enriched_telemetry (Wave 10).
                // Combines the two original blocks under a single state.metrics lock guard.
                //
                // Phase-2a (2026-06-27): compute_frozen_ram_mb runs OUTSIDE the
                // metrics god-lock. The sysinfo walk can spike to 30ms under load
                // (Phase-1 stall_candidate_F2 trace) — keeping it outside the lock
                // shrinks `metrics_lock_held_max_us` back to steady-state.
                let frozen_ram_mb = daemon_cycle_tail::compute_frozen_ram_mb(&state, &collector);
                daemon_cycle_tail::wire_enriched_telemetry(
                    &state,
                    frozen_ram_mb,
                    &mut daemon_cycle_tail::EnrichedTelemetryInputs {
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
                        active_coalitions_count: active_coalitions.len() as u32,
                        lf_metrics,
                        // Phase 1.5a — wire the per-cycle telemetry archive.
                        // Reads from the same `world_model` constructed at
                        // line 3586 (post-NARS-discount). Causal debias is
                        // precomputed here (caller scope has cognitive_state
                        // + MetaCognition in scope; pass through the f32 to
                        // keep the lib-side writer free of MetaCognition dep).
                        metrics_history: Some(daemon_cycle_tail::MetricsHistoryInputs {
                            writer: &mut metrics_history_writer,
                            cycle: cycle_count,
                            world_model: &world_model,
                            drift_detector: &lctx.outcome_tracker.drift_detector,
                            learnable_params: &learnable_params,
                            causal_subsystem_debias: cognitive_state
                                .meta_cognition
                                .subsystem_debias_multiplier(
                                    apollo_engine::engine::meta_cognition::SubsystemId::CausalGraph,
                                ),
                        }),
                    },
                );

                // ── S10 effect-decay drain (Hellerstein 2004 §9.3) ──
                // Drain expired post-Receipt observations and re-read each
                // observable; bump effect_decay_detected_total on mismatch.
                // Wake-grace: skip drain on the first 6 cycles after daemon
                // startup (~30 s) since immediately after wake the kernel
                // may not have reapplied tier hints — false-positive
                // disagreements would inflate the counter.
                if cycle_count > 6 {
                    daemon_cycle_tail::drain_effect_decay(&state, &mut learnable_params);
                }

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

                // Drain CLI purge requests from socket threads.
                // The mpsc receiver is in scope as `main_loop_rx`. Each iteration of the
                // outer loop (one daemon cycle) drains pending requests and replies inline.
                while let Ok(msg) = main_loop_rx.try_recv() {
                    match msg {
                        main_loop_msg::MainLoopMsg::CliPurge { response_tx } => {
                            let cli_event = |outcome, detail: String| {
                                apollo_engine::engine::decision_ledger::ActuatorDecisionEvent::local(
                                    "predictive_purge:cli",
                                    "host",
                                    cycle_count,
                                    outcome,
                                    "cli-maintenance-purge",
                                    detail,
                                )
                            };
                            let (resp, decision_event) = if maintenance_state.secs_since_cli_purge()
                                < 300
                            {
                                let wait = 300 - maintenance_state.secs_since_cli_purge();
                                (
                                    apollo_engine::engine::protocol::DaemonResponse::PurgeResult {
                                        fired: false,
                                        reason: format!("rate_limited — wait {}s", wait),
                                    },
                                    cli_event(
                                        apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome::Blocked,
                                        format!("rate-limited:{wait}s"),
                                    ),
                                )
                            } else if maintenance_state.secs_since_any_purge() < 60 {
                                (
                                    apollo_engine::engine::protocol::DaemonResponse::PurgeResult {
                                        fired: false,
                                        reason: "rate_limited — auto-purge fired recently".into(),
                                    },
                                    cli_event(
                                        apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome::Blocked,
                                        "recent-auto-purge".to_string(),
                                    ),
                                )
                            } else if user_context.audio_active
                                || user_context.call_in_progress
                                || user_context.has_sleep_assertion
                            {
                                // Audio/video/conferencing active → page-cache invalidation
                                // would cause stutter. User can `sudo purge` directly to bypass.
                                (
                                    apollo_engine::engine::protocol::DaemonResponse::PurgeResult {
                                        fired: false,
                                        reason: "media_active — audio/video/call running; pause media or use `sudo purge` to bypass".into(),
                                    },
                                    cli_event(
                                        apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome::Blocked,
                                        "media-active".to_string(),
                                    ),
                                )
                            } else {
                                match async_commands.submit_purge(
                                    "predictive_purge:cli",
                                    "host",
                                    "cli-maintenance-purge",
                                    AsyncCommandMetric::CliPurge,
                                ) {
                                    Ok(submission) => {
                                        maintenance_state.mark_cli_purged();
                                        (
                                            apollo_engine::engine::protocol::DaemonResponse::PurgeResult {
                                                fired: true,
                                                reason: "queued".into(),
                                            },
                                            submission.pending_event(cycle_count),
                                        )
                                    }
                                    Err(error) => (
                                        apollo_engine::engine::protocol::DaemonResponse::PurgeResult {
                                            fired: false,
                                            reason: error.to_string(),
                                        },
                                        cli_event(
                                            match error {
                                                AsyncCommandSubmitError::Full => apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome::Blocked,
                                                AsyncCommandSubmitError::WorkerUnavailable => apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome::Failed,
                                            },
                                            format!("purge queue rejected: {error}"),
                                        ),
                                    ),
                                }
                            };
                            cycle_decision_events.push(decision_event);
                            let _ = response_tx.send(resp);
                        }
                    }
                }

                // Catch proposals emitted while this cycle was running. The
                // sentinel wakes the existing condvar, so an observation that
                // arrives after this point starts the next cycle immediately.
                {
                    let pressure = pressure_collector.latest().memory_pressure;
                    let mut executor = daemon_resource_interrupt_tick::ResourceInterruptExecutor {
                        state: &state,
                        caps: &caps,
                        journal_path: &journal_path,
                        frozen_state_path: &frozen_state_path,
                        dry_run,
                        cycle: cycle_count,
                        memory_pressure: pressure,
                    };
                    let events = executor.execute_batch(
                        &resource_interrupt_window,
                        resource_interrupt_proposal_rx.drain(),
                    );
                    cycle_decision_events.extend_buffer(&events);
                }

                if let Some(coordinated) =
                    apollo_engine::engine::decision_ledger::coordinated_action_event(
                        &cycle_decision_events,
                        cycle_count,
                    )
                {
                    cycle_decision_events.push(coordinated);
                }
                let overflow_summary_emitted =
                    cycle_decision_events.seal_overflow_summary(cycle_count);
                let dropped_decision_events = cycle_decision_events.dropped_total();
                if dropped_decision_events > 0 {
                    tracing::warn!(
                        dropped = dropped_decision_events,
                        "decision event cycle buffer reached its bounded capacity"
                    );
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.decision_event_drops_total = metrics
                        .metrics
                        .decision_event_drops_total
                        .saturating_add(dropped_decision_events);
                    metrics.metrics.decision_event_overflow_summaries_total = metrics
                        .metrics
                        .decision_event_overflow_summaries_total
                        .saturating_add(u64::from(overflow_summary_emitted));
                }
                let resolved_decisions =
                    decision_ledger.ingest_cycle_events(&mut cycle_decision_events);
                telemetry_medallion.stage_decision_episodes(&resolved_decisions);

                // Endpoint wire, cycle tail. Decisions resolved now reach the
                // lab on the next cycle; their measured utility arrives later
                // still, when the medallion resolves that decision to Gold.
                microexperiment_runtime.observe_decisions(&resolved_decisions, cycle_count);
                // Each arm that just acquired a real decision identity opens a
                // measurement window. The medallion measures that window without
                // admitting it as evidence, so no learning aggregate moves and
                // the Gold queue keeps belonging to the causal graph above.
                for binding in microexperiment_runtime.drain_new_bindings() {
                    telemetry_medallion.open_lab_utility_window(
                        binding.decision_id,
                        binding.family,
                        u64::from(binding.horizon_cycles),
                        cycle_count,
                    );
                }
                let microexperiment_utilities: Vec<_> = telemetry_medallion
                    .drain_lab_utility()
                    .into_iter()
                    .map(|sample| {
                        apollo_engine::engine::microexperiment_endpoints::EndpointUtilitySample {
                            decision_id: sample.decision_id,
                            utility_micros: sample.utility_micros,
                            resolved_cycle: sample.resolved_cycle,
                            confounded: sample.confounded,
                        }
                    })
                    .collect();
                microexperiment_runtime.observe_utilities(&microexperiment_utilities, cycle_count);

                // Push estado a suscriptores activos (menubar, etc.)
                socket_handler::broadcast_current_status(&state);

                // Analytics: record this cycle's metrics for trend tracking.
                {
                    let thermal_now = state
                        .hardware
                        .lock_recover()
                        .last_hw_snapshot
                        .as_ref()
                        .filter(|h| !h.temps_estimated)
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
                // Cache prev-cycle pressure for the high-pressure cycle-rate
                // gate (Change C) on the next iteration. Smoothed value to
                // avoid one-shot spikes flipping the daemon to 1 Hz.
                prev_pressure_smooth = signal_digest.pressure_smooth;
                lf_metrics.set_cycle_time_us(cycle_start.elapsed().as_micros() as u64);
                // Phase 0b stage timing: persist complete, cycle done.
                lf_metrics.record_stage(
                    apollo_engine::engine::lse_counters::CycleStage::Persist,
                    _t_persist_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );
                lf_metrics.finish_stage_cycle();
                lf_metrics.commit();

                // ── recently_applied cache cleanup (SuperPlan iter 3) ──────────
                // O(n) sweep every 60 cycles to amortize. Drops expired entries
                // (TTL 30s default) so cache size stays bounded under sustained load.
                if cycle_count.is_multiple_of(60) {
                    let drained = recently_applied.cleanup_expired();
                    if drained > 0 {
                        tracing::debug!(
                            target: "apollo.recently_applied",
                            drained,
                            remaining = recently_applied.len(),
                            "cache cleanup expired entries"
                        );
                    }
                    recently_applied.save_to_disk(std::path::Path::new(
                        apollo_engine::engine::daemon_helpers::recently_applied_path(),
                    ));
                }

                // Phase A3 (Sprint 3 2026-05-07) — periodic IdentityCache cleanup.
                // Lazy expiry on lookup is sufficient for correctness, but a
                // periodic sweep keeps memory bounded under sustained load.
                if cycle_count.is_multiple_of(60) {
                    let drained = identity_cache.tick_cleanup();
                    if drained > 0 {
                        tracing::debug!(
                            target: "apollo.identity_cache",
                            drained,
                            remaining = identity_cache.len(),
                            "cache cleanup expired entries"
                        );
                    }
                }

                // ── Self-diagnosis (Phase 6 self-healing layer) ────────────────
                // Record this cycle's signals + check thresholds. Detection-only;
                // alerts surface via tracing::warn! and append to
                // /var/lib/apollo/self_diagnosis.jsonl for next-session pickup.
                //
                // dedup_drops counters are CUMULATIVE (atomic add) — convert to
                // per-cycle delta by tracking last-seen totals.
                {
                    let snap = lf_metrics.snapshot();
                    let cur_setmem = snap.dedup_drops_setmemorystatus;
                    let cur_throttle = snap.dedup_drops_throttle;
                    let cur_freeze = snap.dedup_drops_freeze;
                    let cur_unfreeze = snap.dedup_drops_unfreeze;
                    let cur_boost = snap.dedup_drops_boost;
                    let cur_thread_qos = snap.dedup_drops_thread_qos;
                    let delta_setmem = cur_setmem.saturating_sub(last_dedup_setmem);
                    let delta_throttle = cur_throttle.saturating_sub(last_dedup_throttle);
                    let delta_freeze = cur_freeze.saturating_sub(last_dedup_freeze);
                    let delta_unfreeze = cur_unfreeze.saturating_sub(last_dedup_unfreeze);
                    let delta_boost = cur_boost.saturating_sub(last_dedup_boost);
                    let delta_thread_qos = cur_thread_qos.saturating_sub(last_dedup_thread_qos);
                    last_dedup_setmem = cur_setmem;
                    last_dedup_throttle = cur_throttle;
                    last_dedup_freeze = cur_freeze;
                    last_dedup_unfreeze = cur_unfreeze;
                    last_dedup_boost = cur_boost;
                    last_dedup_thread_qos = cur_thread_qos;
                    self_diagnosis.record_cycle(
                        delta_setmem,
                        delta_throttle,
                        delta_freeze,
                        delta_unfreeze,
                        delta_boost,
                        delta_thread_qos,
                        snap.refresh_duration_us,
                        collector.last_process_refresh_performed(),
                        snapshot.pressure.memory_pressure,
                    );
                    // Run threshold check every 60 cycles to amortize cost.
                    if cycle_count.is_multiple_of(60) {
                        let alerts = self_diagnosis.check();
                        for alert in &alerts {
                            tracing::warn!(
                                target: "apollo.self_diagnosis",
                                kind = %alert.kind,
                                severity = ?alert.severity,
                                summary = %alert.summary,
                                action = %alert.recommended_action,
                                "self-diagnosis: regression detected"
                            );
                        }
                        self_diagnosis.persist(&alerts);
                    }
                }

                // Periodic sync of lock-free hot path metrics to the Mutex-protected state
                // once every 5 cycles (~1.5s - 10s depending on load). Reduces lock contention.
                if cycle_count.is_multiple_of(5) {
                    let snap = lf_metrics.snapshot();
                    state.metrics.lock_recover().sync_from_lockfree(&snap);
                }
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
                wait_for_cycle_signal(&state.cycle_condvar, wait_duration);
            }

            // Structurally close sentinel actuation before shutdown. The
            // observer can no longer enqueue accepted proposals, and the
            // loop-owned window can never invoke an executor after this point.
            resource_interrupt_proposal_rx.close();
            resource_interrupt_window.close();
            resource_sentinel_shutdown.request_stop();

            // Release speculative kernel state before persisting on shutdown.
            let mut shutdown_decision_events =
                apollo_engine::engine::decision_ledger::CycleDecisionEvents::default();
            let resource_sentinel_quiesced =
                resource_sentinel_shutdown.quiesce(Duration::from_millis(750));
            let expired_proposals = resource_interrupt_window.resolve(
                resource_interrupt_proposal_rx.drain(),
                cycle_count,
                |_| unreachable!("closed resource-interrupt window invoked executor"),
            );
            shutdown_decision_events.extend_buffer(&expired_proposals);
            if !resource_sentinel_quiesced {
                tracing::warn!(
                    worker_finished = resource_sentinel_shutdown.worker_finished(),
                    "resource-sentinel: bounded observer quiescence wait expired; no background effector exists"
                );
            }
            if let Some(lease) = last_markov_prethaw.take() {
                let report = daemon_markov_tick::release_markov_prewarm_recorded(
                    lease,
                    &state,
                    cycle_count,
                    &mut acceleration_lease,
                );
                shutdown_decision_events.extend_buffer(&report.decision_events);
            }
            let acceleration_events = daemon_cycle_tail::release_acceleration_lease_recorded(
                &mut acceleration_lease,
                &state,
                cycle_count,
            );
            shutdown_decision_events.extend_buffer(&acceleration_events);
            exploration_scheduler.cancel_active(
                apollo_engine::engine::exploration_scheduler::TerminalDiagnostic::Shutdown,
            );

            // Revert sysctls to defaults and retain the exact broker receipts.
            {
                let revert_actions = sysctl_governor.revert_to_defaults();
                if !revert_actions.is_empty() {
                    let mut frozen_dummy = HashSet::new();
                    let outcomes = ActuationBroker::from_runtime(&caps, dry_run)
                        .execute(ActuationRequest {
                            actions: revert_actions,
                            caps: &caps,
                            journal_path: &journal_path,
                            frozen: &mut frozen_dummy,
                            learned_protected: &[],
                            learned_interactive: &[],
                            qos_mgr: None,
                            memory_pressure: 0.0,
                            thrashing_score: 0.0,
                            coalition_guard: None,
                            cpu_pegged_fraction: 0.0,
                            async_commands: Some(&async_commands),
                        })
                        .outcomes;
                    shutdown_decision_events
                        .extend_buffer_at_cycle(&outcomes.decision_events, cycle_count);
                    if outcomes.failures == 0 {
                        sysctl_governor.mark_reverted();
                    } else {
                        tracing::warn!(
                            failures = outcomes.failures,
                            "sysctl-governor: revert failures; persisted defaults retained for next startup"
                        );
                    }
                } else {
                    sysctl_governor.mark_reverted();
                }
            }

            // Chromium renderer cleanup: exact SIGCONT disposition per PID.
            {
                let cleanup = chromium_mgr.shutdown_cleanup_outcome();
                for pid in &cleanup.applied_pids {
                    shutdown_decision_events.push(
                        apollo_engine::engine::decision_ledger::ActuatorDecisionEvent::local(
                            "unfreeze:chromium_shutdown",
                            format!("pid:{pid}"),
                            cycle_count,
                            apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome::Reverted,
                            "shutdown-recovery",
                            "renderer SIGCONT applied",
                        ),
                    );
                }
                if !cleanup.applied_pids.is_empty() {
                    let mut frozen_state = state.frozen_state.lock_recover();
                    for pid in &cleanup.applied_pids {
                        frozen_state.remove(pid);
                    }
                }
                for pid in &cleanup.failed_pids {
                    shutdown_decision_events.push(
                        apollo_engine::engine::decision_ledger::ActuatorDecisionEvent::local(
                            "unfreeze:chromium_shutdown",
                            format!("pid:{pid}"),
                            cycle_count,
                            apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome::Failed,
                            "shutdown-recovery",
                            "renderer SIGCONT failed",
                        ),
                    );
                }
                if !cleanup.applied_pids.is_empty() || !cleanup.failed_pids.is_empty() {
                    tracing::info!(
                        applied = cleanup.applied_pids.len(),
                        failed = cleanup.failed_pids.len(),
                        "chromium: shutdown cleanup completed"
                    );
                }
            }

            // Final verified release of main-loop frozen processes.
            {
                let mut frozen_state = state.frozen_state.lock_recover();
                let outcome = unfreeze_pids_verified_outcome(&frozen_state);
                for pid in outcome.forgettable_pids() {
                    frozen_state.remove(&pid);
                }
                write_frozen_state(&frozen_state_path, &frozen_state);
                shutdown_decision_events.extend_buffer(&unfreeze_outcome_events(
                    "unfreeze:shutdown",
                    "shutdown-recovery",
                    cycle_count,
                    &outcome,
                ));
            }

            // Resource-interrupt PIDs do not retain a FrozenEntry, but every
            // existing direct SIGCONT still receives an exact bounded receipt.
            {
                let interrupt_pids: Vec<u32> = state
                    .resource_interrupt
                    .interrupt_frozen_pids
                    .lock_recover()
                    .drain()
                    .collect();
                let outcome = unfreeze_pids_outcome(interrupt_pids.into_iter());
                shutdown_decision_events.extend_buffer(&unfreeze_outcome_events(
                    "unfreeze:shutdown_interrupt",
                    "shutdown-recovery",
                    cycle_count,
                    &outcome,
                ));
            }

            let shutdown_completion_stats =
                async_commands.finalize_shutdown(cycle_count, &mut shutdown_decision_events);
            if shutdown_completion_stats.maintenance_purges_applied > 0
                || shutdown_completion_stats.cli_purges_applied > 0
            {
                lf_metrics.maintenance_purge_total.fetch_add(
                    shutdown_completion_stats
                        .maintenance_purges_applied
                        .saturating_add(shutdown_completion_stats.cli_purges_applied),
                    Ordering::Relaxed,
                );
            }

            let shutdown_overflow_summary =
                shutdown_decision_events.seal_overflow_summary(cycle_count);
            let shutdown_dropped = shutdown_decision_events.dropped_total();
            if shutdown_dropped > 0 {
                let mut metrics = state.metrics.lock_recover();
                metrics.metrics.decision_event_drops_total = metrics
                    .metrics
                    .decision_event_drops_total
                    .saturating_add(shutdown_dropped);
                metrics.metrics.decision_event_overflow_summaries_total = metrics
                    .metrics
                    .decision_event_overflow_summaries_total
                    .saturating_add(u64::from(shutdown_overflow_summary));
            }
            let shutdown_resolved =
                decision_ledger.ingest_cycle_events(&mut shutdown_decision_events);
            telemetry_medallion.stage_decision_episodes(&shutdown_resolved);

            // Persist Markov chain + Holt-Winters + SignalIntelligence state on shutdown.
            focus_markov.persist();
            holt_winters.persist(&hw_path);
            signal_intel.persist(std::path::Path::new(signal_intelligence_path()));
            write_json(
                std::path::Path::new(reflex_state_path()),
                acceleration_lease.rollout_state(),
                Some(0o600),
            );
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
                &maintenance_state,
                apollo_engine::engine::learned_state::LearnedStateSupplement {
                    local_consolidator: Some(
                        local_consolidator.checkpoint_snapshot(Utc::now().timestamp()),
                    ),
                    unfreeze_decay_tau: Some(unfreeze_decay.tau_snapshot()),
                    neuro_state: Some(neuromod.snapshot()),
                    meta_cognition: Some(cognitive_state.meta_cognition.clone()),
                    companion_graph: Some(companion_graph.clone()),
                    medallion_state: Some(learning_pipeline.medallion_snapshot()),
                    telemetry_medallion_state: Some(telemetry_medallion.snapshot()),
                    exploration_scheduler: Some(exploration_scheduler.persisted()),
                    microexperiment_lab: Some(microexperiment_runtime.persisted()),
                },
            );

            // Phase B1.4 (Sprint 2 2026-05-07) — persist RecentlyApplied for next boot.
            // Best-effort: errors are logged but do NOT block shutdown.
            // [Inbox Pattern — 1001 patterns slide 42]
            {
                let path = std::path::PathBuf::from(
                    apollo_engine::engine::daemon_helpers::recently_applied_path(),
                );
                recently_applied.save_to_disk(&path);
                tracing::info!(
                    target: "apollo.recently_applied",
                    path = %path.display(),
                    n = recently_applied.len(),
                    "persist on shutdown"
                );
            }

            // Clean shutdown: remove crash sentinel so next startup knows this was graceful.
            remove_crash_sentinel();
            let _ = fs::remove_file(socket_path());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    //! Sprint 12 perf-fix (2026-05-30): cross-cycle memoization for
    //! `companion_of_fg_pids`. The brief mandates: drive 1000 cycles
    //! with constant fg + identical top_processes, assert
    //! `companion_fg_cache_hits_total >= 990` and `is_companion_of`
    //! call count `< 30` (one per fg-burst or graph mutation).

    use super::{
        contention_mach_tier_event, fingerprint_top_processes, is_dev_tool_name,
        swap_reclaim_bypass_active, wait_for_cycle_signal, CompanionFgCache,
    };
    use apollo_engine::collector::ProcessStats;
    use apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome;
    use apollo_engine::engine::mach_qos::{QoSAuditDisposition, QoSOutcome, SchedulingTier};
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    #[test]
    fn exploration_rechecks_use_fresh_real_gate_sources() {
        let production = include_str!("main.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production source");

        assert!(production.contains("capture_exploration_environment"));
        assert!(!production.contains("last_exploration_gates"));
        assert!(!production.contains("kill_switch: false"));
        assert!(!production.contains("circuit_closed: true"));
        assert!(
            production
                .matches("capture_exploration_environment(")
                .count()
                >= 4
        );
        for seam in [
            include_str!("daemon_markov_tick.rs"),
            include_str!("daemon_cycle_tail.rs"),
            include_str!("daemon_dispatch_tick.rs"),
        ] {
            assert!(seam.matches("exploration_environment()").count() >= 3);
        }
    }

    #[test]
    fn exploratory_member_mutations_reverify_captured_identity() {
        let markov = include_str!("daemon_markov_tick.rs");
        let qos = include_str!("daemon_cycle_tail.rs");

        assert!(markov.contains("reverify_member_identity"));
        assert!(markov.contains("release_markov_prewarm"));
        assert!(qos.contains("reverify_member_identity"));
        assert!(qos.contains("release_acceleration_lease"));
        assert!(markov.matches("reverify_member_identity(").count() >= 2);
        assert!(qos.matches("reverify_member_identity(").count() >= 3);
    }

    #[test]
    fn exploration_seams_offer_cycle_global_priority_and_natural_baselines() {
        let production = include_str!("main.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production source");
        let markov = production.find("run_markov_tick(").expect("Markov seam");
        let qos = production
            .find("update_acceleration_lease(")
            .expect("QoS seam");
        let dispatch = production.find("run_dispatch_tick(").expect("Boost seam");

        assert!(markov < qos && qos < dispatch);
        assert!(production.contains("begin_exploration_cycle(cycle_count)"));
        assert!(include_str!("daemon_markov_tick.rs").contains(".natural_observation()"));
        assert!(include_str!("daemon_cycle_tail.rs").contains(".natural_observation()"));
        assert!(include_str!("daemon_dispatch_tick.rs").contains(".natural_observation()"));
    }

    #[test]
    fn kill_switch_pause_returns_immediately_for_a_pending_resource_wake() {
        let cycle_condvar = Arc::new((Mutex::new(true), Condvar::new()));
        let started = Instant::now();

        assert!(wait_for_cycle_signal(
            &cycle_condvar,
            Duration::from_secs(5)
        ));
        assert!(
            started.elapsed() < Duration::from_millis(100),
            "a proposal queued between the pause drain and wait must start the next cycle"
        );
        assert!(!*cycle_condvar.0.lock().unwrap());
    }

    #[test]
    fn contention_mach_receipt_classifies_exact_effector_disposition() {
        let applied = contention_mach_tier_event(
            5,
            101,
            SchedulingTier::Foreground,
            &QoSOutcome {
                pid: 101,
                tier: SchedulingTier::Foreground,
                success: true,
                mutated: true,
                error: None,
            },
            QoSAuditDisposition::Applied,
        );
        let no_op = contention_mach_tier_event(
            5,
            102,
            SchedulingTier::Background,
            &QoSOutcome {
                pid: 102,
                tier: SchedulingTier::Background,
                success: true,
                mutated: false,
                error: None,
            },
            QoSAuditDisposition::NoOp,
        );
        let blocked = contention_mach_tier_event(
            5,
            103,
            SchedulingTier::Background,
            &QoSOutcome {
                pid: 103,
                tier: SchedulingTier::Background,
                success: true,
                mutated: false,
                error: None,
            },
            QoSAuditDisposition::Blocked,
        );
        let failed = contention_mach_tier_event(
            5,
            104,
            SchedulingTier::Foreground,
            &QoSOutcome {
                pid: 104,
                tier: SchedulingTier::Foreground,
                success: false,
                mutated: false,
                error: Some("task_policy_set failed".to_string()),
            },
            QoSAuditDisposition::Failed,
        );

        assert_eq!(applied.outcome, ActuatorDecisionOutcome::Applied);
        assert_eq!(no_op.outcome, ActuatorDecisionOutcome::NoOp);
        assert_eq!(blocked.outcome, ActuatorDecisionOutcome::Blocked);
        assert_eq!(failed.outcome, ActuatorDecisionOutcome::Failed);
        assert_eq!(failed.proposal.target, "pid:104");
        assert_eq!(
            failed.proposal.action_key,
            "contention:mach_tier:foreground"
        );
    }

    #[test]
    fn dev_tool_name_matches_known_tools_case_insensitively() {
        // Known dev tools (incl. macOS 15-char-truncated language_server and
        // bundle/suffix variants) classify as dev-active.
        assert!(is_dev_tool_name("cargo"));
        assert!(is_dev_tool_name("Claude"));
        assert!(is_dev_tool_name("zed"));
        assert!(is_dev_tool_name("language_server")); // truncated
        assert!(is_dev_tool_name("language_server_macos_arm")); // full
        assert!(is_dev_tool_name("Code Helper"));
        assert!(is_dev_tool_name("iTerm2"));
        assert!(is_dev_tool_name("Warp"));
        // Non-dev foreground apps must NOT latch work mode.
        assert!(!is_dev_tool_name("Spotify"));
        assert!(!is_dev_tool_name("WindowServer"));
        assert!(!is_dev_tool_name(""));
    }

    fn mk_proc(pid: u32, name: &str) -> ProcessStats {
        ProcessStats {
            pid,
            name: name.to_string(),
            cpu_usage: 0.0,
            memory_usage: 0,
            cpu_wall_ratio: None,
        }
    }

    fn mk_top_50() -> Vec<ProcessStats> {
        (0..50u32)
            .map(|i| mk_proc(1000 + i, &format!("proc_{}", i)))
            .collect()
    }

    /// Companion-graph test double: counts every call to
    /// `is_companion_of` so the test can assert the cache actually
    /// suppresses redundant rebuilds. Returns true for a fixed set
    /// of "companion" PIDs so the resulting HashSet has a stable
    /// fingerprint across cycles.
    struct CountingGraph {
        calls: AtomicU64,
        companion_names: HashSet<String>,
        total_cycles: u64,
        anchor_count: usize,
    }

    impl CountingGraph {
        fn new() -> Self {
            // Mark every 5th proc as a companion (10 of 50). Stable across
            // cycles when top_processes is unchanged.
            let companion_names = (0..50u32)
                .filter(|i| i % 5 == 0)
                .map(|i| format!("proc_{}", i))
                .collect();
            Self {
                calls: AtomicU64::new(0),
                companion_names,
                total_cycles: 1,
                anchor_count: 1,
            }
        }

        fn is_companion_of(&self, _fg: &str, proc: &str) -> bool {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.companion_names.contains(proc)
        }

        fn call_count(&self) -> u64 {
            self.calls.load(Ordering::Relaxed)
        }
    }

    /// Drive the cache through 1000 cycles with constant fg +
    /// identical top_processes; assert that the cache short-circuits
    /// every cycle after the first miss.
    ///
    /// The brief mandates ≥990 cache hits and <30 `is_companion_of`
    /// calls (one rebuild per fg-burst or graph mutation). With a
    /// single stable burst we expect exactly 1 miss and 999 hits, with
    /// exactly 50 `is_companion_of` calls (one per process in
    /// top_processes, only on the miss cycle).
    #[test]
    fn companion_of_fg_cache_skips_rebuild_when_fg_and_topset_stable() {
        let top = mk_top_50();
        let graph = CountingGraph::new();
        let fingerprint = fingerprint_top_processes(&top);
        let fg_app: Option<&str> = Some("Brave");

        let mut cache: Option<CompanionFgCache> = None;
        let mut hits: u64 = 0;

        for _cycle in 0..1000 {
            if cache.as_ref().is_some_and(|c| {
                c.is_valid(fg_app, fingerprint, graph.total_cycles, graph.anchor_count)
            }) {
                hits += 1;
            } else {
                let pids: HashSet<u32> = match fg_app {
                    Some(fg_name) => top
                        .iter()
                        .filter(|p| graph.is_companion_of(fg_name, &p.name))
                        .map(|p| p.pid)
                        .collect(),
                    None => HashSet::new(),
                };
                cache = Some(CompanionFgCache {
                    fg_app: fg_app.map(|s| s.to_string()),
                    top_proc_fingerprint: fingerprint,
                    graph_total_cycles: graph.total_cycles,
                    graph_anchor_count: graph.anchor_count,
                    pids,
                });
            }
        }

        // Steady-state expectation: only the very first cycle missed.
        assert!(
            hits >= 990,
            "expected ≥990 cache hits across 1000 stable cycles, got {}",
            hits
        );
        // is_companion_of must NOT be called on hit cycles. One miss
        // ⇒ exactly 50 calls (top_processes.len()).
        let calls = graph.call_count();
        assert!(
            calls < 30 || calls == 50,
            "expected <30 OR exactly 50 (one-shot top_processes scan), got {}",
            calls
        );
        // Strong assertion: exactly one rebuild on the cold-start cycle.
        assert_eq!(
            calls, 50,
            "exactly 50 is_companion_of calls expected (one per top_process on miss)"
        );
        // Strong assertion: 999 hits / 1000 cycles in the stable case.
        assert_eq!(hits, 999, "exactly one cold-start miss expected");
    }

    /// Foreground app flip invalidates the cache (rebuild fires).
    #[test]
    fn companion_of_fg_cache_invalidates_on_fg_flip() {
        let top = mk_top_50();
        let graph = CountingGraph::new();
        let fingerprint = fingerprint_top_processes(&top);

        let mut cache: Option<CompanionFgCache> = None;
        let mut rebuilds: u64 = 0;

        for cycle in 0..10 {
            let fg_app: Option<&str> = if cycle < 5 {
                Some("Brave")
            } else {
                Some("Code")
            };
            if !cache.as_ref().is_some_and(|c| {
                c.is_valid(fg_app, fingerprint, graph.total_cycles, graph.anchor_count)
            }) {
                rebuilds += 1;
                let pids: HashSet<u32> = match fg_app {
                    Some(name) => top
                        .iter()
                        .filter(|p| graph.is_companion_of(name, &p.name))
                        .map(|p| p.pid)
                        .collect(),
                    None => HashSet::new(),
                };
                cache = Some(CompanionFgCache {
                    fg_app: fg_app.map(|s| s.to_string()),
                    top_proc_fingerprint: fingerprint,
                    graph_total_cycles: graph.total_cycles,
                    graph_anchor_count: graph.anchor_count,
                    pids,
                });
            }
        }
        assert_eq!(
            rebuilds, 2,
            "fg flip at cycle 5 must trigger one extra rebuild"
        );
    }

    /// CompanionGraph `total_cycles` advance invalidates the cache.
    #[test]
    fn companion_of_fg_cache_invalidates_on_graph_mutation_witness() {
        let top = mk_top_50();
        let mut graph = CountingGraph::new();
        let fingerprint = fingerprint_top_processes(&top);
        let fg_app: Option<&str> = Some("Brave");

        let mut cache: Option<CompanionFgCache> = None;
        let mut rebuilds: u64 = 0;

        for cycle in 0..10 {
            if cycle == 7 {
                // Simulate observe_cycle() crediting the anchor past
                // ATTENTION_FLOOR, bumping total_cycles.
                graph.total_cycles += 1;
            }
            if !cache.as_ref().is_some_and(|c| {
                c.is_valid(fg_app, fingerprint, graph.total_cycles, graph.anchor_count)
            }) {
                rebuilds += 1;
                let pids: HashSet<u32> = match fg_app {
                    Some(name) => top
                        .iter()
                        .filter(|p| graph.is_companion_of(name, &p.name))
                        .map(|p| p.pid)
                        .collect(),
                    None => HashSet::new(),
                };
                cache = Some(CompanionFgCache {
                    fg_app: fg_app.map(|s| s.to_string()),
                    top_proc_fingerprint: fingerprint,
                    graph_total_cycles: graph.total_cycles,
                    graph_anchor_count: graph.anchor_count,
                    pids,
                });
            }
        }
        assert_eq!(
            rebuilds, 2,
            "graph mutation witness must invalidate cache exactly once"
        );
    }

    /// Fingerprint is sensitive to (pid, name) projection changes,
    /// insensitive to internal field ordering (XOR is commutative).
    #[test]
    fn fingerprint_top_processes_detects_projection_change() {
        let a = vec![mk_proc(1, "alpha"), mk_proc(2, "beta")];
        let b = vec![mk_proc(2, "beta"), mk_proc(1, "alpha")];
        assert_eq!(
            fingerprint_top_processes(&a),
            fingerprint_top_processes(&b),
            "XOR-fold is order-independent"
        );
        let c = vec![mk_proc(1, "alpha"), mk_proc(3, "beta")];
        assert_ne!(
            fingerprint_top_processes(&a),
            fingerprint_top_processes(&c),
            "pid mutation must alter fingerprint"
        );
        let d = vec![mk_proc(1, "alpha"), mk_proc(2, "gamma")];
        assert_ne!(
            fingerprint_top_processes(&a),
            fingerprint_top_processes(&d),
            "name mutation must alter fingerprint"
        );
    }

    #[test]
    fn swap_reclaim_bypass_scales_with_swap_total() {
        let gib = 1024u64 * 1024 * 1024;
        assert!(!swap_reclaim_bypass_active(
            (4.5 * gib as f64) as u64,
            16 * gib,
            0.0,
        ));
        assert!(swap_reclaim_bypass_active(6 * gib, 16 * gib, 0.0));
        assert!(swap_reclaim_bypass_active(
            (4.5 * gib as f64) as u64,
            8 * gib,
            0.0,
        ));
    }

    #[test]
    fn swap_reclaim_bypass_still_honors_oom_probability() {
        assert!(swap_reclaim_bypass_active(0, 0, 0.95));
        assert!(!swap_reclaim_bypass_active(0, 0, 0.94));
    }
}
