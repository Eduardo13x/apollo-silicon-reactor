//! # Daemon Memory Budget
//!
//! Jetsam inactive-limit enforcement from memory budget computation extracted
//! from main.rs (Wave 28). [Fowler 2004] Strangler Fig — pure move.
//!
//! ## Responsibilities
//! - When pressure ≥ 0.60: compute per-process jetsam inactive limits
//! - Use TASK_VM_INFO WSS when available, fault-rate heuristic otherwise
//! - Apply set_memlimit() to over-budget processes (active=0 = never kill)
//!
//! ## Ordering invariant
//! Must run AFTER proc_snaps is populated (process_enrichment) and BEFORE
//! the main decision pass.

use std::collections::HashMap;
use std::time::{Instant, Duration};

use apollo_optimizer::engine::compressor_aware::query_memory_profile;
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::jetsam_control;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::memory_analyzer::MemoryAnalyzer;
use apollo_optimizer::engine::memory_budget::{self, ProcessBudgetInput};
use apollo_optimizer::engine::overflow_guard::is_build_tool_name;
use apollo_optimizer::engine::process_classifier::ProcessSnapshot;

/// Pressure zones with hysteresis thresholds.
/// [Hellerstein 2004] Operating-regime control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureZone {
    Normal,
    Elevated,
    Critical,
}

/// Persistent state for memory budget enforcement.
pub struct MemoryBudgetState {
    /// PID -> last applied inactive limit in MiB.
    pub last_applied_limits: HashMap<u32, u64>,
    /// Last time any limit was applied.
    pub last_applied_at: Option<Instant>,
    /// Current operating regime for hysteresis.
    pub current_zone: PressureZone,
    /// Last time a Critical zone bypass triggered immediate evaluation.
    pub last_critical_bypass_at: Option<Instant>,
}

impl Default for MemoryBudgetState {
    fn default() -> Self {
        Self {
            last_applied_limits: HashMap::new(),
            last_applied_at: None,
            current_zone: PressureZone::Normal,
            last_critical_bypass_at: None,
        }
    }
}

/// Explicit thresholds for pressure zones.
/// [Hellerstein 2004] Operating-regime control.
const ELEVATED_ENTER_PRESSURE: f64 = 0.65;
const ELEVATED_EXIT_PRESSURE: f64 = 0.55;
const CRITICAL_ENTER_PRESSURE: f64 = 0.80;
const CRITICAL_EXIT_PRESSURE: f64 = 0.74; // Widened to prevent bouncing
const CRITICAL_BYPASS_COOLDOWN: Duration = Duration::from_secs(20);

/// Get the enforcement interval based on the current pressure zone.
/// [Hellerstein 2004] Adaptive sampling frequency.
pub fn memory_budget_enforcement_interval(zone: PressureZone) -> Duration {
    match zone {
        PressureZone::Normal => Duration::from_secs(60),
        PressureZone::Elevated => Duration::from_secs(10), // Closes the OOM window gap
        PressureZone::Critical => Duration::from_secs(5),
    }
}

/// Enforce jetsam inactive limits for over-budget processes under memory pressure.
///
/// Includes hysteresis and rate-limiting to prevent "thrashing" syscall spam
/// when pressure oscillates around thresholds.
pub fn run_memory_budget(
    memory_pressure: f64,
    total_ram: u64,
    state: &SharedState,
    proc_snaps: &[ProcessSnapshot],
    mem_analyzer: &MemoryAnalyzer,
    budget_state: &mut MemoryBudgetState,
) {
    // 1. Update Pressure Zone with explicit hysteresis.
    let next_zone = match budget_state.current_zone {
        PressureZone::Normal => {
            if memory_pressure >= ELEVATED_ENTER_PRESSURE {
                PressureZone::Elevated
            } else {
                PressureZone::Normal
            }
        }
        PressureZone::Elevated => {
            if memory_pressure >= CRITICAL_ENTER_PRESSURE {
                PressureZone::Critical
            } else if memory_pressure <= ELEVATED_EXIT_PRESSURE {
                PressureZone::Normal
            } else {
                PressureZone::Elevated
            }
        }
        PressureZone::Critical => {
            if memory_pressure <= CRITICAL_EXIT_PRESSURE {
                PressureZone::Elevated
            } else {
                PressureZone::Critical
            }
        }
    };
    let zone_changed = next_zone != budget_state.current_zone;
    if zone_changed {
        tracing::info!(
            target: "apollo.memory_budget",
            old_zone = ?budget_state.current_zone,
            new_zone = ?next_zone,
            pressure = memory_pressure,
            "pressure_zone_changed"
        );
    }
    budget_state.current_zone = next_zone;

    // Normal zone: no enforcement.
    if next_zone == PressureZone::Normal {
        return;
    }

    // 2. Decide if we should evaluate budgets this cycle.
    let now = Instant::now();
    let time_since_last = budget_state
        .last_applied_at
        .map(|t| now.duration_since(t))
        .unwrap_or(Duration::from_secs(u64::MAX));

    let rate_limit = memory_budget_enforcement_interval(next_zone);

    // Evaluate if zone changed, adaptive time passed, or if we just entered Critical (bypass).
    // Entering Critical triggers immediate action, but with a cooldown to prevent
    // "bang-bang" oscillation (e.g. 0.79-0.81 bouncing).
    let time_since_last_bypass = budget_state
        .last_critical_bypass_at
        .map(|t| now.duration_since(t))
        .unwrap_or(Duration::from_secs(u64::MAX));

    let entering_critical = zone_changed && next_zone == PressureZone::Critical;
    let bypass_cooldown_ok = time_since_last_bypass >= CRITICAL_BYPASS_COOLDOWN;
    
    let force_eval = zone_changed || time_since_last >= rate_limit || (entering_critical && bypass_cooldown_ok);

    if entering_critical && bypass_cooldown_ok {
        budget_state.last_critical_bypass_at = Some(now);
    }

    let usage_guard = state.usage.lock_recover();
    let budget_inputs: Vec<ProcessBudgetInput> = proc_snaps
        .iter()
        .take(30)
        .filter(|s| s.rss_bytes > 50 * 1024 * 1024)
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

    if budget_inputs.is_empty() {
        return;
    }

    // GC dead PIDs from history.
    let live_pids: std::collections::HashSet<u32> = proc_snaps.iter().map(|s| s.pid).collect();
    budget_state
        .last_applied_limits
        .retain(|pid, _| live_pids.contains(pid));

    let budgets = memory_budget::compute_budgets(total_ram, &budget_inputs);
    for budget in budgets.iter().filter(|b| b.over_budget) {
        let last_limit = budget_state.last_applied_limits.get(&budget.pid).copied();
        let limit_delta = last_limit
            .map(|l| (l as i64 - budget.inactive_limit_mb as i64).abs() as u64)
            .unwrap_or(u64::MAX);

        // Significant change: >15% change in limit AND at least 50MiB.
        // Prevents jitter in budget computation from triggering syscalls.
        let significant_change = limit_delta > (last_limit.unwrap_or(0) / 7).max(50);

        if force_eval || significant_change {
            let _ = jetsam_control::set_memlimit(
                budget.pid,
                0, // active: unlimited (don't kill foreground)
                budget.inactive_limit_mb,
            );
            budget_state
                .last_applied_limits
                .insert(budget.pid, budget.inactive_limit_mb as u64);
            budget_state.last_applied_at = Some(now);

            tracing::info!(
                target: "apollo.memory_budget",
                pid = budget.pid,
                name = %budget.name,
                limit_mb = budget.inactive_limit_mb,
                zone = ?next_zone,
                "memlimit_applied"
            );
        } else if time_since_last < rate_limit {
            tracing::debug!(
                target: "apollo.memory_budget",
                pid = budget.pid,
                name = %budget.name,
                ?time_since_last,
                ?rate_limit,
                "memlimit_skipped_due_to_cooldown"
            );
        } else {
            tracing::debug!(
                target: "apollo.memory_budget",
                pid = budget.pid,
                name = %budget.name,
                limit_delta,
                "memlimit_skipped_no_significant_delta"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::sync::atomic::AtomicBool;
    use std::path::PathBuf;
    use std::collections::{HashMap, VecDeque};
    use apollo_optimizer::engine::daemon_state::{
        MetricsState, PolicyState, ProcessState, HardwareState, LlmDomainState, UsageDomainState,
        ReactorStatus, UsageTrackerState,
    };
    use apollo_optimizer::engine::types::{OptimizationProfile, LatencyTarget, RuntimeMetrics};
    use apollo_optimizer::engine::adaptive_governor::AdaptiveGovernor;
    use apollo_optimizer::engine::profile_governor::ProfileGovernor;
    use apollo_optimizer::engine::llm::{LlmConfig, LlmState, LearnedPolicy};
    use apollo_optimizer::engine::circuit_breaker::CircuitBreaker;
    use apollo_optimizer::engine::degradation::DegradationController;
    use apollo_optimizer::engine::usage_model::UsageModel;
    use apollo_optimizer::engine::mach_qos::MachQoSManager;
    use apollo_optimizer::engine::daemon_helpers::WakeRuntimeState;
    use apollo_optimizer::engine::sysctl_governor::SysctlGovernorStatus;
    use apollo_optimizer::engine::thermal_interrupt::ResourceInterruptState;
    use std::sync::Condvar;

    fn mock_state() -> SharedState {
        SharedState {
            metrics: Arc::new(Mutex::new(MetricsState {
                metrics: RuntimeMetrics::default(),
                throttle_level: "balanced".to_string(),
                thermal_state: "nominal".to_string(),
                thermal_level_real: "nominal".to_string(),
                fast_tick_until: None,
                reactor_event_weight: 0.0,
                reactor_status: ReactorStatus::default(),
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
                    ipc_consecutive_drops: 0,
                    ipc_consecutive_clean: 0,
                    vm_consecutive_high: 0,
                    vm_consecutive_low: 0,
                    fs_consecutive_high: 0,
                    fs_consecutive_low: 0,
                },
            })),
            llm: Arc::new(Mutex::new(LlmDomainState {
                llm_cfg: LlmConfig::default(),
                llm_state: LlmState::default(),
                llm_state_path: PathBuf::from("/tmp/apollo_mock_llm_state"),
                llm_key_path: PathBuf::from("/tmp/apollo_mock_llm_key"),
                learned_policy_path: PathBuf::from("/tmp/apollo_mock_lp"),
                feedback_path: PathBuf::from("/tmp/apollo_mock_feedback"),
                suggestions_path: PathBuf::from("/tmp/apollo_mock_suggestions"),
            })),
            usage: Arc::new(Mutex::new(UsageDomainState {
                usage_model: UsageModel::default(),
                usage_tracker: UsageTrackerState::default(),
                usage_model_path: PathBuf::from("/tmp/apollo_mock_um"),
                usage_events_path: PathBuf::from("/tmp/apollo_mock_ue"),
            })),
            frozen_state: Arc::new(Mutex::new(HashMap::new())),
            mach_qos: Arc::new(Mutex::new(MachQoSManager::new())),
            freeze_cooldown: Arc::new(Mutex::new(apollo_optimizer::engine::freeze_cooldown::FreezeCooldown::new())),
            stop: Arc::new(AtomicBool::new(false)),
            revert_sysctls_requested: Arc::new(AtomicBool::new(false)),
            cycle_condvar: Arc::new((Mutex::new(false), Condvar::new())),
            resource_interrupt: Arc::new(ResourceInterruptState::new()),
            subscribers: Arc::new(Mutex::new(Vec::new())),
            config_path: PathBuf::from("/tmp/apollo_mock_config"),
            user_profile_path: PathBuf::from("/tmp/apollo_mock_user_profile"),
        }
    }

    #[test]
    fn test_pressure_zone_hysteresis() {
        let mut state = MemoryBudgetState::default();
        let shared = mock_state();
        let analyzer = MemoryAnalyzer::new();
        assert_eq!(state.current_zone, PressureZone::Normal);

        // Entry to Elevated >= 0.65
        run_memory_budget(0.64, 8589934592, &shared, &[], &analyzer, &mut state);
        assert_eq!(state.current_zone, PressureZone::Normal);

        run_memory_budget(0.65, 8589934592, &shared, &[], &analyzer, &mut state);
        assert_eq!(state.current_zone, PressureZone::Elevated);

        // Exit from Elevated <= 0.55
        run_memory_budget(0.56, 8589934592, &shared, &[], &analyzer, &mut state);
        assert_eq!(state.current_zone, PressureZone::Elevated);

        run_memory_budget(0.55, 8589934592, &shared, &[], &analyzer, &mut state);
        assert_eq!(state.current_zone, PressureZone::Normal);
    }

    #[test]
    fn test_memory_budget_enforcement_intervals() {
        assert_eq!(memory_budget_enforcement_interval(PressureZone::Normal), Duration::from_secs(60));
        assert_eq!(memory_budget_enforcement_interval(PressureZone::Elevated), Duration::from_secs(10));
        assert_eq!(memory_budget_enforcement_interval(PressureZone::Critical), Duration::from_secs(5));
    }

    #[test]
    fn test_pressure_zone_hysteresis_hardened() {
        let mut state = MemoryBudgetState::default();
        let shared = mock_state();
        let analyzer = MemoryAnalyzer::new();
        
        // Entry to Elevated >= 0.65
        run_memory_budget(0.65, 8589934592, &shared, &[], &analyzer, &mut state);
        assert_eq!(state.current_zone, PressureZone::Elevated);

        // Entry to Critical >= 0.80
        run_memory_budget(0.80, 8589934592, &shared, &[], &analyzer, &mut state);
        assert_eq!(state.current_zone, PressureZone::Critical);

        // Exit Critical <= 0.74 (Hysteresis)
        run_memory_budget(0.75, 8589934592, &shared, &[], &analyzer, &mut state);
        assert_eq!(state.current_zone, PressureZone::Critical);

        run_memory_budget(0.74, 8589934592, &shared, &[], &analyzer, &mut state);
        assert_eq!(state.current_zone, PressureZone::Elevated);
    }

    #[test]
    fn test_critical_bypass_cooldown() {
        let mut state = MemoryBudgetState::default();
        let shared = mock_state();
        let analyzer = MemoryAnalyzer::new();
        let now = Instant::now();

        // 1. Initial bypass allowed
        // Step 1: Normal -> Elevated
        run_memory_budget(0.70, 8589934592, &shared, &[], &analyzer, &mut state);
        assert_eq!(state.current_zone, PressureZone::Elevated);
        
        // Step 2: Elevated -> Critical (triggers bypass)
        run_memory_budget(0.81, 8589934592, &shared, &[], &analyzer, &mut state);
        assert_eq!(state.current_zone, PressureZone::Critical);
        assert!(state.last_critical_bypass_at.is_some());
        let first_bypass = state.last_critical_bypass_at.unwrap();

        // 2. Drop to Elevated then back to Critical immediately
        run_memory_budget(0.70, 8589934592, &shared, &[], &analyzer, &mut state);
        assert_eq!(state.current_zone, PressureZone::Elevated);
        
        run_memory_budget(0.81, 8589934592, &shared, &[], &analyzer, &mut state);
        // Bypass should NOT re-trigger because of cooldown
        assert_eq!(state.last_critical_bypass_at.unwrap(), first_bypass);

        // 3. Mock time passing for cooldown (20s)
        state.last_critical_bypass_at = Some(now - Duration::from_secs(21));
        run_memory_budget(0.70, 8589934592, &shared, &[], &analyzer, &mut state);
        run_memory_budget(0.81, 8589934592, &shared, &[], &analyzer, &mut state);
        // Bypass should re-trigger
        assert!(state.last_critical_bypass_at.unwrap() > first_bypass);
    }
}
