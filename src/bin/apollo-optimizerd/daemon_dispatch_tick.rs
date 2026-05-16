//! Dispatch Tick — final decision and execution phase of the daemon loop.
//!
//! Handles:
//! 1. Filter pipeline execution (circuit breaker, degradation, cognitive gates).
//! 2. Predictive thaw gate (model-informed control to prevent spikes).
//! 3. Action dispatch via `execute_actions`.
//! 4. Circuit breaker and degradation state updates.
//! 5. Frozen state persistence.

use chrono::Utc;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use apollo_engine::collector::{SystemCollector, SystemSnapshot};
use apollo_engine::engine::daemon_helpers::write_frozen_state;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::degradation::DegradationInputs;
use apollo_engine::engine::execute_actions::{execute_actions, ExecuteOutcomes};
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::lse_counters::LockFreeMetrics;
use apollo_engine::engine::swap_reclaim::SwapRisk;
use apollo_engine::engine::types::{FreezeSource, FrozenEntry, RootAction};
use apollo_engine::engine::unfreeze_decay::UnfreezeDecayModel;

/// Action kinds tracked for per-PID dedup.
/// Variants without a target PID (SetSysctl, ToggleSpotlight, QuarantineDaemon)
/// bypass the consolidator and are kept verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DedupKind {
    SetMemorystatus,
    Throttle,
    Freeze,
    Unfreeze,
    Boost,
    SetThreadQoS,
}

/// Counts of duplicate actions dropped per kind in a single dispatch cycle.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DedupStats {
    pub set_memorystatus: u64,
    pub throttle: u64,
    pub freeze: u64,
    pub unfreeze: u64,
    pub boost: u64,
    pub set_thread_qos: u64,
}

impl DedupStats {
    pub fn total_dropped(&self) -> u64 {
        self.set_memorystatus
            + self.throttle
            + self.freeze
            + self.unfreeze
            + self.boost
            + self.set_thread_qos
    }
}

/// Extract `(pid, kind)` for actions targeting a specific process. Returns
/// `None` for actions without a PID target (sysctl, spotlight, quarantine).
///
/// SetThreadQoS uses `(pid, thread_index)`-aware key by encoding thread_index
/// into the kind via secondary discriminator — but because all SetThreadQoS
/// for the same pid+thread are equivalent for dedup purposes, we treat
/// `(pid, SetThreadQoS, thread_index)` as the key.
fn dedup_key(action: &RootAction) -> Option<(u32, DedupKind, u32)> {
    match action {
        RootAction::SetMemorystatus { pid, .. } => Some((*pid, DedupKind::SetMemorystatus, 0)),
        RootAction::ThrottleProcess { pid, .. } => Some((*pid, DedupKind::Throttle, 0)),
        RootAction::FreezeProcess { pid, .. } => Some((*pid, DedupKind::Freeze, 0)),
        RootAction::UnfreezeProcess { pid, .. } => Some((*pid, DedupKind::Unfreeze, 0)),
        RootAction::BoostProcess { pid, .. } => Some((*pid, DedupKind::Boost, 0)),
        RootAction::SetThreadQoS {
            pid, thread_index, ..
        } => Some((*pid, DedupKind::SetThreadQoS, *thread_index)),
        RootAction::SetSysctl(_)
        | RootAction::ToggleSpotlight { .. }
        | RootAction::QuarantineDaemon { .. } => None,
    }
}

/// Consolidate per-PID actions: keep at most one action per `(pid, kind)`,
/// drop subsequent duplicates. Conflict resolution between different kinds
/// for the same PID is intentionally NOT performed here — those represent
/// distinct intents (e.g., Throttle and SetMemorystatus on the same PID
/// can coexist legitimately).
///
/// Closes the Critical gap from NotebookLM peer review (2026-05-06):
/// 14 emission paths constructed RootActions without per-PID dedup,
/// causing pid 65808 to receive SetMemorystatus 8× in same second.
///
/// [Saltzer & Schroeder 1975] Economy of Mechanism — single chokepoint
/// before execute_actions eliminates the bug class without touching
/// every emission site.
pub fn consolidate_actions_per_pid(actions: Vec<RootAction>) -> (Vec<RootAction>, DedupStats) {
    let mut seen: HashSet<(u32, DedupKind, u32)> = HashSet::with_capacity(actions.len());
    let mut stats = DedupStats::default();
    let mut out: Vec<RootAction> = Vec::with_capacity(actions.len());

    for action in actions {
        match dedup_key(&action) {
            Some(key) => {
                if seen.insert(key) {
                    out.push(action);
                } else {
                    match key.1 {
                        DedupKind::SetMemorystatus => stats.set_memorystatus += 1,
                        DedupKind::Throttle => stats.throttle += 1,
                        DedupKind::Freeze => stats.freeze += 1,
                        DedupKind::Unfreeze => stats.unfreeze += 1,
                        DedupKind::Boost => stats.boost += 1,
                        DedupKind::SetThreadQoS => stats.set_thread_qos += 1,
                    }
                }
            }
            None => out.push(action),
        }
    }
    (out, stats)
}

/// Increment lock-free dedup_drops counters from DedupStats.
/// Called by run_dispatch_tick after consolidate_actions_per_pid.
pub fn record_dedup_drops(lf: &LockFreeMetrics, stats: &DedupStats) {
    if stats.set_memorystatus > 0 {
        lf.add_dedup_drops_setmemorystatus(stats.set_memorystatus);
    }
    if stats.throttle > 0 {
        lf.add_dedup_drops_throttle(stats.throttle);
    }
    if stats.freeze > 0 {
        lf.add_dedup_drops_freeze(stats.freeze);
    }
    if stats.unfreeze > 0 {
        lf.add_dedup_drops_unfreeze(stats.unfreeze);
    }
}

use crate::{cognitive_tick, daemon_action_pipeline};

/// Input dependencies for the dispatch tick.
pub struct DispatchTickInput<'a> {
    pub state: &'a SharedState,
    pub caps: &'a apollo_engine::engine::types::CapabilityReport,
    pub journal_path: &'a Path,
    pub frozen_state_path: &'a Path,
    pub final_actions: Vec<RootAction>,
    pub snapshot: &'a SystemSnapshot,
    pub prev_cog_decision: Option<&'a cognitive_tick::CognitiveDecision>,
    pub causal_qos_names: &'a HashSet<String>,
    pub reclaim_risk: SwapRisk,
    pub unfreeze_decay: &'a mut UnfreezeDecayModel,
    pub collector: &'a SystemCollector,
    pub dry_run: bool,
    /// Lock-free metrics for per-cycle dedup_drops accounting.
    /// Optional so legacy callers and unit tests can pass `None`.
    pub lf_metrics: Option<&'a LockFreeMetrics>,
    /// Coalition guard: tracker + recent-fg envelope. None opts out of
    /// coalition-aware skipping (legacy callers / tests).
    pub coalition_guard:
        Option<&'a apollo_engine::engine::active_coalition_envelope::CoalitionGuard<'a>>,
    /// Per-cycle fraction of CPU cores pegged ≥0.80 busy (from
    /// background_collectors.cpu_saturation.pegged_fraction). When this
    /// rises above 0.80 with memory pressure <0.75, freeze/throttle are
    /// gated as `BlockReason::CpuSaturated`.
    pub cpu_pegged_fraction: f64,
}

/// Output results from the dispatch tick.
pub struct DispatchTickOutput {
    pub outcomes: ExecuteOutcomes,
    pub causal_qos_upgrades: u32,
    /// Dedup statistics from this cycle's consolidation pass.
    /// Currently consumed only by lf_metrics counters; may be read by
    /// downstream observers (Phase 6 self-healing layer) in future.
    #[allow(dead_code)]
    pub dedup_stats: DedupStats,
}

/// Runs the dispatch and execution orchestration logic.
pub fn run_dispatch_tick(input: DispatchTickInput) -> DispatchTickOutput {
    let DispatchTickInput {
        state,
        caps,
        journal_path,
        frozen_state_path,
        final_actions,
        snapshot,
        prev_cog_decision,
        causal_qos_names,
        reclaim_risk,
        unfreeze_decay,
        collector,
        dry_run,
        lf_metrics,
        coalition_guard,
        cpu_pegged_fraction,
    } = input;

    // ── Filter pipeline ──────────────────────────────────────────────────────
    let filter_outcome = daemon_action_pipeline::run_filter_pipeline(
        final_actions,
        state,
        snapshot,
        prev_cog_decision,
        causal_qos_names,
        reclaim_risk,
    );
    let cb_is_open = filter_outcome.cb_is_open;
    let op_mode = filter_outcome.op_mode;
    let mut filtered_actions = filter_outcome.filtered_actions;
    let causal_qos_upgrades = filter_outcome.causal_qos_upgrades;

    // ── Per-PID dedup chokepoint ─────────────────────────────────────────────
    // Single consolidation pass before execute_actions. 14 upstream emission
    // paths (decide_actions, daemon_paging_hints, daemon_agent_actions,
    // process_enrichment, llm_daemon, freeze-confirmation, etc.) push freely;
    // here we collapse duplicate (pid, kind) pairs. Without this, pid 65808
    // received SetMemorystatus 8× in the same second (prod observation).
    // [Saltzer & Schroeder 1975] Economy of Mechanism.
    let (deduped, dedup_stats) = consolidate_actions_per_pid(filtered_actions);
    filtered_actions = deduped;
    if let Some(lf) = lf_metrics {
        record_dedup_drops(lf, &dedup_stats);
    }
    if dedup_stats.total_dropped() > 0 {
        tracing::debug!(
            target: "apollo.dispatch.dedup",
            dropped_total = dedup_stats.total_dropped(),
            sm_status = dedup_stats.set_memorystatus,
            throttle = dedup_stats.throttle,
            freeze = dedup_stats.freeze,
            unfreeze = dedup_stats.unfreeze,
            "consolidate_actions_per_pid: collapsed duplicates"
        );
    }

    // ── Predictive thaw gate ─────────────────────────────────────────────
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
                    let predicted = unfreeze_decay.predict_rss(name, m_0, 5.0);
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

    // ── Circuit breaker + execute_actions ────────────────────
    let mut frozen_set: HashSet<u32> = state.frozen_state.lock_recover().keys().copied().collect();
    let frozen_before: HashSet<u32> = frozen_set.clone();

    let (learned_protected, learned_interactive) = {
        let pg = state.policy.lock_recover();
        (
            pg.learned_policy.protected_patterns.clone(),
            pg.learned_policy.interactive_patterns.clone(),
        )
    };
    let mut qos = state.mach_qos.lock_recover();

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
            caps,
            journal_path,
            &mut frozen_set,
            &learned_protected,
            &learned_interactive,
            Some(&mut qos),
            dry_run,
            snapshot.pressure.memory_pressure,
            snapshot.pressure.thrashing_score,
            coalition_guard,
            cpu_pegged_fraction,
        )
    } else {
        // Circuit Closed or HalfOpen: run normally, then report outcome.
        let out = execute_actions(
            filtered_actions,
            caps,
            journal_path,
            &mut frozen_set,
            &learned_protected,
            &learned_interactive,
            Some(&mut qos),
            dry_run,
            snapshot.pressure.memory_pressure,
            snapshot.pressure.thrashing_score,
            coalition_guard,
            cpu_pegged_fraction,
        );
        // Report outcome to circuit breaker.
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

    // Update degradation controller with new failure count.
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

    // Sync frozen state back and persist if changed.
    {
        let now = Utc::now();
        // Build identity map from this cycle's freeze results so FrozenEntry carries
        // the correct start_sec and original_jetsam_priority captured at SIGSTOP time.
        // [A5/D1 fix] Without this, the normal loop path always stored None for
        // original_jetsam_priority, preventing proper priority restoration on thaw.
        let identity_map: HashMap<u32, (u64, Option<i32>)> = outcomes
            .newly_frozen_identity
            .iter()
            .map(|(pid, start_sec, pri)| (*pid, (*start_sec, *pri)))
            .collect();
        let mut frozen_state = state.frozen_state.lock_recover();
        for pid in &frozen_set {
            frozen_state.entry(*pid).or_insert_with(|| {
                let name = apollo_engine::engine::process_identity::proc_name_for_pid(*pid);
                let (start_sec, original_jetsam_priority) =
                    identity_map.get(pid).copied().unwrap_or_else(|| {
                        let s = apollo_engine::engine::process_identity::ProcessIdentity::from_pid(
                            *pid,
                        )
                        .map(|pi| pi.start_sec)
                        .unwrap_or(0);
                        (s, None)
                    });
                FrozenEntry {
                    frozen_at: now,
                    source: FreezeSource::MainLoop,
                    pressure_at_freeze: snapshot.pressure.memory_pressure,
                    process_name: name,
                    start_sec,
                    original_jetsam_priority,
                }
            });
        }
        frozen_state.retain(|pid, _| frozen_set.contains(pid));
        if frozen_set != frozen_before {
            write_frozen_state(frozen_state_path, &frozen_state);
        }
    }

    DispatchTickOutput {
        outcomes,
        causal_qos_upgrades,
        dedup_stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::collector::{CpuStats, MemoryStats, PressureStats};
    use apollo_engine::engine::adaptive_governor::AdaptiveGovernor;
    use apollo_engine::engine::audit_types::DecisionReason;
    use apollo_engine::engine::circuit_breaker::{CircuitBreaker, CircuitState};
    use apollo_engine::engine::daemon_helpers::WakeRuntimeState;
    use apollo_engine::engine::daemon_state::{
        HardwareState, LlmDomainState, MetricsState, PolicyState, ProcessState, UsageDomainState,
    };
    use apollo_engine::engine::degradation::DegradationController;
    use apollo_engine::engine::llm::{LearnedPolicy, LlmConfig, LlmState};
    use apollo_engine::engine::mach_qos::MachQoSManager;
    use apollo_engine::engine::sysctl_governor::SysctlGovernorStatus;
    use apollo_engine::engine::types::{
        CapabilityReport, LatencyTarget, OptimizationProfile, RuntimeMetrics,
    };
    use apollo_engine::engine::usage_model::UsageModel;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};

    fn create_test_state() -> SharedState {
        SharedState {
            policy: Arc::new(Mutex::new(PolicyState {
                profile: OptimizationProfile::BalancedRoot,
                latency_target: LatencyTarget::Normal,
                governor: apollo_engine::engine::profile_governor::ProfileGovernor::new(
                    OptimizationProfile::BalancedRoot,
                ),
                learned_policy: LearnedPolicy::default(),
                adaptive_governor: AdaptiveGovernor::new(),
                timeline: std::collections::VecDeque::new(),
                circuit_breaker: CircuitBreaker::default(),
                degradation: DegradationController::default(),
            })),
            metrics: Arc::new(Mutex::new(MetricsState {
                metrics: RuntimeMetrics::default(),
                throttle_level: "balanced".to_string(),
                thermal_state: "nominal".to_string(),
                thermal_level_real: "unknown".to_string(),
                fast_tick_until: None,
                reactor_event_weight: 0.0,
                reactor_status: apollo_engine::engine::daemon_state::ReactorStatus::default(),
            })),
            frozen_state: Arc::new(Mutex::new(HashMap::new())),
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
            stop: Arc::new(AtomicBool::new(false)),
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
            config_path: PathBuf::from("/tmp/apollo_test_config"),
            user_profile_path: PathBuf::from("/tmp/apollo_test_user_profile"),
            usage: Arc::new(Mutex::new(UsageDomainState {
                usage_model: UsageModel::default(),
                usage_model_path: PathBuf::from("/tmp/apollo_test_um"),
                usage_events_path: PathBuf::from("/tmp/apollo_test_ue"),
                usage_tracker: apollo_engine::engine::daemon_state::UsageTrackerState::default(),
            })),
            mach_qos: Arc::new(Mutex::new(MachQoSManager::new())),
            freeze_cooldown: Arc::new(Mutex::new(
                apollo_engine::engine::freeze_cooldown::FreezeCooldown::new(),
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
                    ipc_consecutive_drops: 0,
                    ipc_consecutive_clean: 0,
                    vm_consecutive_high: 0,
                    vm_consecutive_low: 0,
                    fs_consecutive_high: 0,
                    fs_consecutive_low: 0,
                },
            })),
            revert_sysctls_requested: Arc::new(AtomicBool::new(false)),
            cycle_condvar: Arc::new((Mutex::new(false), std::sync::Condvar::new())),
            resource_interrupt: Arc::new(
                apollo_engine::engine::thermal_interrupt::ResourceInterruptState::new(),
            ),
            subscribers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    #[test]
    fn test_dispatch_tick_circuit_open() {
        let state = create_test_state();
        {
            let mut pg = state.policy.lock().unwrap();
            // Record enough failures to trip the circuit breaker.
            for _ in 0..10 {
                pg.circuit_breaker.record_failure();
            }
            assert_eq!(*pg.circuit_breaker.state(), CircuitState::Open);
        }

        let caps = CapabilityReport {
            can_taskpolicy: true,
            can_sysctl: true,
            can_memorystatus: true,
            can_mdutil: true,
            can_tmutil: true,
            is_root: true,
            p_core_count: Some(8),
            e_core_count: Some(4),
            unavailable: Vec::new(),
        };

        let mut unfreeze_decay = UnfreezeDecayModel::new();
        let collector = SystemCollector::new();
        let snapshot = SystemSnapshot {
            timestamp: Utc::now(),
            cpu: CpuStats {
                global_usage: 0.0,
                core_count: 1,
            },
            memory: MemoryStats {
                total_ram: 0,
                used_ram: 0,
                free_ram: 0,
                total_swap: 0,
                used_swap: 0,
            },
            pressure: PressureStats {
                memory_pressure: 0.0,
                swap_used_bytes: 0,
                swap_total_bytes: 0,
                swap_delta_bytes_per_sec: 0.0,
                thermal_level: "nominal".to_string(),
                compressor_pressure: 0.0,
                thrashing_score: 0.0,
            },
            disks: Vec::new(),
            networks: Vec::new(),
            top_processes: Vec::new(),
        };
        let causal_qos = HashSet::new();

        let input = DispatchTickInput {
            state: &state,
            caps: &caps,
            journal_path: Path::new("/tmp/apollo_test_journal"),
            frozen_state_path: Path::new("/tmp/apollo_test_frozen"),
            final_actions: vec![
                RootAction::throttle(1234, "test", true, "test", DecisionReason::PressureContext),
                RootAction::unfreeze(
                    5678,
                    "test_unfreeze",
                    "test",
                    DecisionReason::PressureContext,
                ),
            ],
            snapshot: &snapshot,
            prev_cog_decision: None,
            causal_qos_names: &causal_qos,
            reclaim_risk: SwapRisk::Safe,
            unfreeze_decay: &mut unfreeze_decay,
            collector: &collector,
            dry_run: true,
            lf_metrics: None,
            coalition_guard: None,
            cpu_pegged_fraction: 0.0,
        };

        let output = run_dispatch_tick(input);

        // When circuit is open, only unfreeze actions should be dispatched.
        assert_eq!(output.outcomes.unfreezes_applied, 1);
        assert_eq!(output.outcomes.throttles_applied, 0);
    }

    // ── Per-PID dedup unit tests ─────────────────────────────────────────────

    fn sm_status(pid: u32) -> RootAction {
        RootAction::SetMemorystatus {
            pid,
            priority: -1,
            reason: format!("test pid {}", pid),
            decision_reason: DecisionReason::MemoryBudget,
        }
    }

    fn throttle(pid: u32) -> RootAction {
        RootAction::ThrottleProcess {
            pid,
            name: format!("p{}", pid),
            aggressive: false,
            reason: "test".to_string(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }
    }

    fn freeze(pid: u32) -> RootAction {
        RootAction::FreezeProcess {
            pid,
            name: format!("p{}", pid),
            reason: "test".to_string(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }
    }

    #[test]
    fn consolidate_drops_4x_setmemorystatus_same_pid() {
        // Reproduces prod observation: pid 65808 SetMemorystatus 4× per cycle.
        let actions = vec![
            sm_status(65808),
            sm_status(65808),
            sm_status(65808),
            sm_status(65808),
        ];
        let (out, stats) = consolidate_actions_per_pid(actions);
        assert_eq!(out.len(), 1, "should keep first occurrence only");
        assert_eq!(stats.set_memorystatus, 3, "should drop 3 duplicates");
        assert_eq!(stats.total_dropped(), 3);
    }

    #[test]
    fn consolidate_keeps_distinct_pid_setmemorystatus() {
        let actions = vec![sm_status(100), sm_status(200), sm_status(300)];
        let (out, stats) = consolidate_actions_per_pid(actions);
        assert_eq!(out.len(), 3, "distinct PIDs not deduped");
        assert_eq!(stats.total_dropped(), 0);
    }

    #[test]
    fn consolidate_keeps_throttle_and_setmemorystatus_same_pid() {
        // Different kinds for same PID coexist legitimately.
        let actions = vec![throttle(100), sm_status(100)];
        let (out, stats) = consolidate_actions_per_pid(actions);
        assert_eq!(out.len(), 2, "different kinds for same PID coexist");
        assert_eq!(stats.total_dropped(), 0);
    }

    #[test]
    fn consolidate_drops_mixed_duplicates_per_kind() {
        // 3× SetMemorystatus + 2× Throttle + 1 Freeze for pid 100; 1 Freeze for pid 200.
        let actions = vec![
            sm_status(100),
            sm_status(100),
            sm_status(100),
            throttle(100),
            throttle(100),
            freeze(100),
            freeze(200),
        ];
        let (out, stats) = consolidate_actions_per_pid(actions);
        // Survivors: 1 SM(100), 1 Throttle(100), 1 Freeze(100), 1 Freeze(200) = 4
        assert_eq!(out.len(), 4);
        assert_eq!(stats.set_memorystatus, 2);
        assert_eq!(stats.throttle, 1);
        assert_eq!(stats.freeze, 0);
        assert_eq!(stats.total_dropped(), 3);
    }

    #[test]
    fn consolidate_preserves_action_order() {
        // First occurrence wins — order must be deterministic.
        let actions = vec![sm_status(1), sm_status(2), sm_status(1), sm_status(3)];
        let (out, stats) = consolidate_actions_per_pid(actions);
        assert_eq!(out.len(), 3);
        assert_eq!(stats.set_memorystatus, 1);
        // Verify pid order is 1, 2, 3 (not re-sorted).
        if let RootAction::SetMemorystatus { pid, .. } = &out[0] {
            assert_eq!(*pid, 1);
        } else {
            panic!("expected SetMemorystatus first");
        }
        if let RootAction::SetMemorystatus { pid, .. } = &out[1] {
            assert_eq!(*pid, 2);
        } else {
            panic!("expected SetMemorystatus second");
        }
        if let RootAction::SetMemorystatus { pid, .. } = &out[2] {
            assert_eq!(*pid, 3);
        } else {
            panic!("expected SetMemorystatus third");
        }
    }

    #[test]
    fn consolidate_passes_through_non_pid_actions() {
        // SetSysctl / ToggleSpotlight have no PID — never deduped, always pass through.
        let actions = vec![
            RootAction::ToggleSpotlight {
                enabled: false,
                reason: "test".to_string(),
                decision_reason: DecisionReason::PressureContext,
            },
            RootAction::ToggleSpotlight {
                enabled: false,
                reason: "test2".to_string(),
                decision_reason: DecisionReason::PressureContext,
            },
        ];
        let (out, stats) = consolidate_actions_per_pid(actions);
        assert_eq!(out.len(), 2, "non-PID actions pass through");
        assert_eq!(stats.total_dropped(), 0);
    }
}
