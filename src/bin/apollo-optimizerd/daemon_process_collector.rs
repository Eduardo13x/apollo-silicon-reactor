//! # Daemon Process Collector
//!
//! Per-cycle process-table operations extracted from the daemon main loop:
//!
//! - `build_process_tree`       — build the parent/child tree from sysinfo.
//! - `run_pre_sleep_unfreeze`   — release all SIGSTOP'd PIDs before system sleep.
//! - `run_ghost_pid_reconciliation` — evict dead PIDs from `frozen_state` / turbo.
//!
//! All three are small, self-contained, and free of cross-cycle state.

use std::collections::HashSet;
use std::path::Path;

use apollo_engine::collector::SystemCollector;
use apollo_engine::engine::daemon_helpers::{unfreeze_pids_verified, write_frozen_state};
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::display_turbo::DisplayTurbo;
use apollo_engine::engine::identity_cache_manager::IdentityCacheManager;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::process_tree::{ProcessEntry, ProcessTree};
use apollo_engine::engine::sleep_notifier::SleepNotifier;

/// Build the parent/child process tree from the latest sysinfo snapshot.
///
/// Used by foreground-family detection, enrichment, and chromium visibility
/// checks. Cost is dominated by sysinfo's internal iteration (~1–2ms for
/// ~500 processes).
pub fn build_process_tree(collector: &SystemCollector) -> ProcessTree {
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
}

/// Pre-sleep unfreeze — release every SIGSTOP'd PID before the kernel suspends.
///
/// `kIOMessageSystemWillSleep` fires ~30s before kernel suspension. Without
/// releasing our frozen PIDs here, they remain ineligible for jetsam / compressor
/// eviction during sleep, which forces macOS to kill more interactive helpers
/// (widgets, extensions) to reclaim memory.
///
/// A-B-A defense: `unfreeze_pids_verified` re-checks (pid, start_sec, name)
/// identity before SIGCONT so PIDs recycled during the race window are skipped.
/// [Saltzer & Kaashoek 2009] §3.3 Complete Mediation.
pub fn run_pre_sleep_unfreeze(
    state: &SharedState,
    frozen_state_path: &Path,
    display_turbo: &mut DisplayTurbo,
    sleep_notifier: &SleepNotifier,
) {
    if !sleep_notifier.will_sleep_pending() {
        return;
    }
    let mut frozen_guard = state.frozen_state.lock_recover();
    // Turbo PIDs live in frozen_guard too, so this covers both regular + turbo.
    let count = unfreeze_pids_verified(&frozen_guard);
    if count > 0 {
        // Snapshot thawed PIDs before clearing for cooldown bookkeeping.
        let thawed_pids: Vec<u32> = frozen_guard.keys().copied().collect();
        tracing::info!(
            count,
            "pre-sleep: released {} frozen PID(s) — \
             handing back to macOS memory manager",
            count
        );
        frozen_guard.clear();
        write_frozen_state(frozen_state_path, &frozen_guard);
        drop(frozen_guard);
        state.metrics.lock_recover().metrics.unfreezes_applied += count;
        // Mark thawed PIDs in cooldown to prevent gate_e re-freeze oscillation.
        // [Nygard 2018] §8.5 — circuit breaker hold-down after recovery.
        {
            let mut cooldown = state.freeze_cooldown.lock_recover();
            for pid in &thawed_pids {
                cooldown.mark_thawed(*pid);
            }
        }
    }
    display_turbo.clear_frozen();
    sleep_notifier.acknowledge();
}

/// Ghost-PID reconciliation — evict frozen_state entries whose PID is dead.
///
/// A frozen process can die via manual kill, Force Quit, or jetsam while kqueue
/// `NOTE_EXIT` isn't registered (e.g., after a daemon restart). Without this,
/// `frozen_state` retains ghost entries whose RSS is counted as `frozen_ram_mb`
/// even though the OS already reclaimed that memory.
///
/// `live_pids` must come from the authoritative sysinfo snapshot used this cycle.
/// Also triggers:
/// - `display_turbo.gc_dead_pids()` (in-memory, no disk write)
/// - `mach_qos.gc_dead_pids()` every 60 cycles (~30s) — libc::kill(pid,0) is
///   cheap but the internal HashMaps can grow large under Chrome.
pub fn run_ghost_pid_reconciliation(
    state: &SharedState,
    live_pids: &HashSet<u32>,
    frozen_state_path: &Path,
    display_turbo: &mut DisplayTurbo,
    cycle_count: u64,
    identity_cache: &IdentityCacheManager,
) {
    let mut frozen_guard = state.frozen_state.lock_recover();
    let before = frozen_guard.len();
    let dead_pids: Vec<u32> = frozen_guard
        .keys()
        .copied()
        .filter(|pid| !live_pids.contains(pid))
        .collect();
    frozen_guard.retain(|pid, _| live_pids.contains(pid));
    let removed = before - frozen_guard.len();
    if removed > 0 {
        tracing::info!(
            removed,
            "frozen_state: evicted {} ghost PID(s) \
             (died without kqueue notification)",
            removed
        );
        write_frozen_state(frozen_state_path, &frozen_guard);
    }
    for pid in dead_pids {
        identity_cache.notify_exited(pid);
        crate::process_enrichment::invalidate_cached_enrich(pid);
    }
    display_turbo.gc_dead_pids(live_pids);
    drop(frozen_guard);

    if cycle_count.is_multiple_of(60) {
        state.mach_qos.lock_recover().gc_dead_pids();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Condvar, Mutex};

    use apollo_engine::engine::adaptive_governor::AdaptiveGovernor;
    use apollo_engine::engine::circuit_breaker::CircuitBreaker;
    use apollo_engine::engine::daemon_helpers::WakeRuntimeState;
    use apollo_engine::engine::daemon_state::{
        HardwareState, LlmDomainState, MetricsState, PolicyState, ProcessState, UsageDomainState,
    };
    use apollo_engine::engine::degradation::DegradationController;
    use apollo_engine::engine::display_turbo::DisplayTurbo;
    use apollo_engine::engine::freeze_cooldown::FreezeCooldown;
    use apollo_engine::engine::identity_cache_manager::IdentityCacheManager;
    use apollo_engine::engine::llm::{LearnedPolicy, LlmConfig, LlmState};
    use apollo_engine::engine::lse_counters::LockFreeMetrics;
    use apollo_engine::engine::mach_qos::MachQoSManager;
    use apollo_engine::engine::process_identity::ProcessIdentity;
    use apollo_engine::engine::profile_governor::ProfileGovernor;
    use apollo_engine::engine::sysctl_governor::SysctlGovernorStatus;
    use apollo_engine::engine::thermal_interrupt::ResourceInterruptState;
    use apollo_engine::engine::types::{
        FreezeSource, FrozenEntry, LatencyTarget, OptimizationProfile, RuntimeMetrics,
    };
    use apollo_engine::engine::usage_model::UsageModel;

    fn test_state() -> SharedState {
        SharedState {
            metrics: Arc::new(Mutex::new(MetricsState {
                metrics: RuntimeMetrics::default(),
                throttle_level: "balanced".to_string(),
                thermal_state: "nominal".to_string(),
                thermal_level_real: "unknown".to_string(),
                fast_tick_until: None,
                reactor_event_weight: 0.0,
                reactor_status: apollo_engine::engine::daemon_state::ReactorStatus::default(),
                survival_window:
                    apollo_engine::engine::survival_window::SurvivalActivationWindow::new(),
            })),
            policy: Arc::new(Mutex::new(PolicyState {
                profile: OptimizationProfile::BalancedRoot,
                latency_target: LatencyTarget::Normal,
                governor: ProfileGovernor::new(OptimizationProfile::BalancedRoot),
                learned_policy: LearnedPolicy::default(),
                adaptive_governor: AdaptiveGovernor::new(),
                timeline: std::collections::VecDeque::new(),
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
                usage_model_path: PathBuf::from("/tmp/apollo_test_um"),
                usage_events_path: PathBuf::from("/tmp/apollo_test_ue"),
                usage_tracker: apollo_engine::engine::daemon_state::UsageTrackerState::default(),
            })),
            frozen_state: Arc::new(Mutex::new(HashMap::new())),
            mach_qos: Arc::new(Mutex::new(MachQoSManager::new())),
            freeze_cooldown: Arc::new(Mutex::new(FreezeCooldown::new())),
            stop: Arc::new(AtomicBool::new(false)),
            revert_sysctls_requested: Arc::new(AtomicBool::new(false)),
            cycle_condvar: Arc::new((Mutex::new(false), Condvar::new())),
            resource_interrupt: Arc::new(ResourceInterruptState::new()),
            subscribers: Arc::new(Mutex::new(Vec::new())),
            config_path: PathBuf::from("/tmp/apollo_test_config"),
            user_profile_path: PathBuf::from("/tmp/apollo_test_user_profile"),
        }
    }

    #[test]
    fn ghost_reconciliation_invalidates_identity_cache_for_removed_pid() {
        let state = test_state();
        let identity_cache = IdentityCacheManager::new();
        let lf = LockFreeMetrics::new();
        let me = std::process::id();
        let id = ProcessIdentity::from_pid(me).unwrap();

        assert!(identity_cache.verify(me, Some(&id.name), id.start_sec, id.start_usec, &lf));
        assert_eq!(identity_cache.len(), 1);
        state.frozen_state.lock_recover().insert(
            me,
            FrozenEntry {
                frozen_at: chrono::Utc::now(),
                source: FreezeSource::MainLoop,
                pressure_at_freeze: 0.8,
                process_name: Some(id.name.clone()),
                start_sec: id.start_sec,
                original_jetsam_priority: None,
            },
        );

        let live_pids = HashSet::new();
        let mut display_turbo = DisplayTurbo::new();
        run_ghost_pid_reconciliation(
            &state,
            &live_pids,
            Path::new("/tmp/apollo_test_frozen_state.json"),
            &mut display_turbo,
            1,
            &identity_cache,
        );

        assert_eq!(identity_cache.len(), 0);
    }
}
