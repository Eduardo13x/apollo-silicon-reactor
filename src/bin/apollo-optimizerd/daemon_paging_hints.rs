//! # Daemon Paging Hints
//!
//! Per-cycle SetMemorystatus paging hint injection extracted from main.rs (Wave 17).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Direct pressure hints: when pressure_smooth ≥ 0.60, hint top 3 background procs
//! - ODE velocity hints: when ODE net_rate > 0.5 (leading indicator before threshold fires)
//!   [Hellerstein 2004 §9 — derivative control acts before integrator saturates]
//!
//! ## Ordering invariant
//! Must run AFTER signal_digest and reclaim_forecast are computed.
//! Must run BEFORE heuristic_pass (so hinted PIDs are visible for dedup).

use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::decide_actions::is_interactive_app_name;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::process_classifier::ProcessSnapshot;
use apollo_engine::engine::recently_applied::{CachedActionKind, RecentlyApplied};
use apollo_engine::engine::safety::{is_protected_name, is_user_interactive_app};
use apollo_engine::engine::swap_reclaim::{CyberPhysicalSignal, NetRateNorm};
use apollo_engine::engine::types::RootAction;

fn rejects_memorystatus_hint_by_name(name: &str) -> bool {
    name.contains("VirtualMachine")
}

/// Emit direct pressure-driven and ODE-velocity paging hints for this cycle.
///
/// Returns new SetMemorystatus actions to append to the main actions vec.
/// Deduplicates against any pids that already have hints in `current_actions`.
///
/// # Parameters
/// - `proc_snaps` — full process snapshot list (not just top 10 by CPU)
/// - `state` — SharedState (policy lock for learned protection patterns)
/// - `pressure_smooth` — EMA pressure from signal_digest
/// - `ode_net_rate_bps` — ODE net rate from reclaim_forecast (bytes/sec)
/// - `foreground_app` — current foreground app name (hint filter)
/// - `current_actions` — actions accumulated so far (for per-PID dedup)
pub fn run_paging_hints(
    proc_snaps: &[ProcessSnapshot],
    state: &SharedState,
    pressure_smooth: f64,
    ode_net_rate_bps: f64,
    foreground_app: Option<&str>,
    current_actions: &[RootAction],
    recovering_from_critical: bool,
    recently_applied: &mut RecentlyApplied,
    call_in_progress: bool,
    audio_active: bool,
) -> Vec<RootAction> {
    let mut new_actions: Vec<RootAction> = Vec::new();

    // ── Media safety guard ──────────────────────────────────────────────────
    // Never send paging hints during calls or audio playback — purging
    // audio buffers causes glitches. [Jiang & Zhang 2005] proactive beats
    // reactive, but not at the cost of audio dropouts.
    if call_in_progress || audio_active {
        return new_actions;
    }

    // ── Direct pressure hints ────────────────────────────────────────────────
    // When pressure > 0.60, hint top 3 background memory consumers to release
    // caches voluntarily. SetMemorystatus priority -1 = voluntary cache release.
    // [Jiang & Zhang 2005] proactive beats reactive by 20-40%.
    // BUG #2 fix: per-PID dedup instead of "any SetMemorystatus → skip all".
    if pressure_smooth >= 0.60 {
        let hinted_pids: std::collections::HashSet<u32> = current_actions
            .iter()
            .filter_map(|a| {
                if let RootAction::SetMemorystatus { pid, .. } = a {
                    Some(*pid)
                } else {
                    None
                }
            })
            .collect();
        let protected_pats = state
            .policy
            .lock_recover()
            .learned_policy
            .protected_patterns
            .clone();
        let mut bg_procs: Vec<_> = proc_snaps
            .iter()
            .filter(|p| {
                if is_interactive_app_name(&p.name) {
                    return false;
                }
                let is_interactive = is_user_interactive_app(
                    p.has_gui_window,
                    p.secs_since_user_interaction,
                    p.rss_bytes,
                    &p.name,
                );
                !is_protected_name(&p.name)
                    && !is_interactive
                    && !protected_pats.iter().any(|pat| p.name.contains(pat.as_str()))
                    && p.rss_bytes > 80 * 1024 * 1024 // >80 MB RSS
                    && p.pid != std::process::id()
                    && !p.has_gui_window
                    && foreground_app.map(|fg| p.name != fg).unwrap_or(true)
                    && p.secs_since_user_interaction > 60
                    // Virtualization.framework VMs reject memorystatus_control even
                    // as root (sandbox + hardened runtime blocks the sysctl write).
                    // Skip early to avoid repeated journal spam for known-failed hints.
                    && !rejects_memorystatus_hint_by_name(&p.name)
            })
            .collect();
        bg_procs.sort_by(|a, b| b.rss_bytes.cmp(&a.rss_bytes));
        let mut added = 0usize;
        for proc in bg_procs.iter() {
            if added >= 3 {
                break;
            }
            if hinted_pids.contains(&proc.pid) {
                continue;
            }
            // Cross-cycle state memory (SuperPlan Iter 6 2026-05-06):
            // SetMemorystatus -1 is a hint to the kernel — repeat hints for the
            // same PID across cycles get no-op'd at the syscall layer. Skip if
            // we already hinted this PID within the TTL window.
            if recently_applied.is_recent(proc.pid, CachedActionKind::SetMemorystatus) {
                continue;
            }
            new_actions.push(RootAction::set_memorystatus(
                proc.pid,
                -1,
                format!(
                    "pressure-driven hint (p={:.0}%): {} ({}MB)",
                    pressure_smooth * 100.0,
                    proc.name,
                    proc.rss_bytes / 1024 / 1024,
                ),
                if pressure_smooth >= 0.80 {
                    DecisionReason::CriticalBypass
                } else if recovering_from_critical {
                    DecisionReason::HysteresisRecovery
                } else {
                    DecisionReason::MemoryBudget
                },
            ));
            recently_applied.record(proc.pid, CachedActionKind::SetMemorystatus);
            added += 1;
        }
    }

    // ── G20: ODE velocity hints ──────────────────────────────────────────────
    // When ODE net_rate > 0.5 AND pressure < 0.60: proactively hint top 2 procs.
    // ODE is a leading indicator — rising compression rate predicts pressure before
    // the kernel threshold fires. [Hellerstein 2004 §9]
    let ode_rate_norm = NetRateNorm(ode_net_rate_bps).normalized();
    if ode_rate_norm > 0.5 && pressure_smooth < 0.60 {
        let hinted_pids_ode: std::collections::HashSet<u32> = current_actions
            .iter()
            .chain(new_actions.iter())
            .filter_map(|a| {
                if let RootAction::SetMemorystatus { pid, .. } = a {
                    Some(*pid)
                } else {
                    None
                }
            })
            .collect();
        let protected_pats = state
            .policy
            .lock_recover()
            .learned_policy
            .protected_patterns
            .clone();
        let mut bg_procs: Vec<_> = proc_snaps
            .iter()
            .filter(|p| {
                if is_interactive_app_name(&p.name) {
                    return false;
                }
                let is_interactive = is_user_interactive_app(
                    p.has_gui_window,
                    p.secs_since_user_interaction,
                    p.rss_bytes,
                    &p.name,
                );
                !is_protected_name(&p.name)
                    && !is_interactive
                    && !protected_pats
                        .iter()
                        .any(|pat| p.name.contains(pat.as_str()))
                    && p.rss_bytes > 80 * 1024 * 1024
                    && p.pid != std::process::id()
                    && !p.has_gui_window
                    && foreground_app.map(|fg| p.name != fg).unwrap_or(true)
                    && p.secs_since_user_interaction > 60
                    && !rejects_memorystatus_hint_by_name(&p.name)
            })
            .collect();
        bg_procs.sort_by(|a, b| b.rss_bytes.cmp(&a.rss_bytes));
        let mut added = 0usize;
        for proc in bg_procs.iter() {
            if added >= 2 {
                break;
            }
            if hinted_pids_ode.contains(&proc.pid) {
                continue;
            }
            // Cross-cycle dedup (SuperPlan Iter 6).
            if recently_applied.is_recent(proc.pid, CachedActionKind::SetMemorystatus) {
                continue;
            }
            new_actions.push(RootAction::set_memorystatus(
                proc.pid,
                -1,
                format!(
                    "ode-velocity hint (net_rate={:.0}%): {} ({}MB)",
                    ode_rate_norm * 100.0,
                    proc.name,
                    proc.rss_bytes / 1024 / 1024,
                ),
                DecisionReason::MemoryBudget,
            ));
            recently_applied.record(proc.pid, CachedActionKind::SetMemorystatus);
            added += 1;
        }
    }

    new_actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::engine::adaptive_governor::AdaptiveGovernor;
    use apollo_engine::engine::circuit_breaker::CircuitBreaker;
    use apollo_engine::engine::daemon_helpers::WakeRuntimeState;
    use apollo_engine::engine::daemon_state::{
        HardwareState, LlmDomainState, MetricsState, PolicyState, ProcessState, UsageDomainState,
    };
    use apollo_engine::engine::degradation::DegradationController;
    use apollo_engine::engine::llm::{LearnedPolicy, LlmConfig, LlmState};
    use apollo_engine::engine::mach_qos::MachQoSManager;
    use apollo_engine::engine::profile_governor::ProfileGovernor;
    use apollo_engine::engine::survival_window::SurvivalActivationWindow;
    use apollo_engine::engine::sysctl_governor::SysctlGovernorStatus;
    use apollo_engine::engine::thermal_interrupt::ResourceInterruptState;
    use apollo_engine::engine::types::{LatencyTarget, OptimizationProfile, RuntimeMetrics};
    use apollo_engine::engine::usage_model::UsageModel;
    use std::collections::{HashMap, VecDeque};
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Condvar, Mutex};

    fn test_state() -> SharedState {
        SharedState {
            metrics: Arc::new(Mutex::new(MetricsState {
                metrics: RuntimeMetrics::default(),
                throttle_level: "balanced".to_string(),
                thermal_state: "nominal".to_string(),
                thermal_level_real: "unknown".to_string(),
                fast_tick_until: None,
                reactor_event_weight: 0.0,
                reactor_status: Default::default(),
                survival_window: SurvivalActivationWindow::new(),
            })),
            policy: Arc::new(Mutex::new(PolicyState {
                profile: OptimizationProfile::BalancedRoot,
                governor: ProfileGovernor::new(OptimizationProfile::BalancedRoot),
                learned_policy: LearnedPolicy::default(),
                adaptive_governor: AdaptiveGovernor::new(),
                latency_target: LatencyTarget::Normal,
                timeline: VecDeque::new(),
                circuit_breaker: CircuitBreaker::default(),
                degradation: DegradationController::default(),
            })),
            process: Arc::new(Mutex::new(ProcessState {
                last_blockers: Vec::new(),
                wake_state: WakeRuntimeState {
                    last_cycle_wallclock: chrono::Utc::now(),
                    last_wake_at: None,
                    post_wake_grace_until: None,
                    post_wake_policy: "normal".to_string(),
                    post_wake_reclaim_until: None,
                },
            })),
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
            llm: Arc::new(Mutex::new(LlmDomainState {
                llm_cfg: LlmConfig {
                    enabled: None,
                    endpoint: None,
                    model: None,
                    min_confidence: None,
                    max_calls_per_hour: None,
                    min_interval_secs: None,
                    timeout_ms: None,
                    force_json: None,
                    always_on: None,
                    max_tokens: None,
                    disable_thinking: None,
                },
                llm_state: LlmState::default(),
                llm_state_path: PathBuf::from("/tmp/apollo_test_llm_state"),
                llm_key_path: PathBuf::from("/tmp/apollo_test_llm_key"),
                learned_policy_path: PathBuf::from("/tmp/apollo_test_lp"),
                feedback_path: PathBuf::from("/tmp/apollo_test_feedback"),
                suggestions_path: PathBuf::from("/tmp/apollo_test_suggestions"),
            })),
            usage: Arc::new(Mutex::new(UsageDomainState {
                usage_model: UsageModel::default(),
                usage_tracker: Default::default(),
                usage_model_path: PathBuf::from("/tmp/apollo_test_um"),
                usage_events_path: PathBuf::from("/tmp/apollo_test_ue"),
            })),
            frozen_state: Arc::new(Mutex::new(HashMap::new())),
            mach_qos: Arc::new(Mutex::new(MachQoSManager::new())),
            freeze_cooldown: Arc::new(Mutex::new(Default::default())),
            effect_decay: Arc::new(Mutex::new(Default::default())),
            stop: Arc::new(AtomicBool::new(false)),
            revert_sysctls_requested: Arc::new(AtomicBool::new(false)),
            cycle_condvar: Arc::new((Mutex::new(false), Condvar::new())),
            resource_interrupt: Arc::new(ResourceInterruptState::new()),
            subscribers: Arc::new(Mutex::new(Vec::new())),
            config_path: PathBuf::from("/tmp/apollo_test_config"),
            user_profile_path: PathBuf::from("/tmp/apollo_test_user_profile"),
        }
    }

    fn proc_snapshot(pid: u32, name: &str, rss_mb: u64) -> ProcessSnapshot {
        ProcessSnapshot {
            pid,
            name: name.to_string(),
            cpu_percent: 0.0,
            rss_bytes: rss_mb * 1024 * 1024,
            is_zombie: false,
            secs_since_foreground: 3600,
            secs_since_user_interaction: 3600,
            has_network: false,
            has_gui_window: false,
            wakeups_per_sec: 0.0,
            parent_alive: true,
            process_uptime_secs: 3600,
            faults_total: 0,
            pageins_total: 0,
            is_translated: false,
            mach_port_count: 0,
            cpu_contention: None,
            is_app_bundle: false,
        }
    }

    #[test]
    fn ode_velocity_hints_skip_virtualization_vm() {
        let state = test_state();
        let mut recently_applied = RecentlyApplied::new();
        let procs = vec![proc_snapshot(
            1050,
            "com.apple.Virtualization.VirtualMachine",
            512,
        )];

        let actions = run_paging_hints(
            &procs,
            &state,
            0.55,
            500_000_000.0,
            Some("Brave Browser"),
            &[],
            false,
            &mut recently_applied,
            false,
            false,
        );

        assert!(
            actions.is_empty(),
            "Virtualization.framework VMs reject memorystatus; ODE hints must skip them"
        );
    }
}
