//! # Metrics Reporter
//!
//! Per-cycle observability and dispatch work extracted from the daemon main loop.
//!
//! ## What this module does
//!
//! Every cycle, after the learning tick completes, the daemon performs three
//! distinct observability/dispatch operations:
//!
//! 1. **`update_learning_metrics`** — Write predictive agent + signal intelligence
//!    fields into the shared `MetricsState` for `GetStatus` consumption.
//!
//! 2. **`apply_io_shaping`** — Foreground-aware I/O bandwidth allocation
//!    (Iyer & Druschel 2001, every 20 cycles). Updates `sysctl_reactive_writes`.
//!
//! 3. **`apply_qos_routing`** — MachQoS P-Core / E-Core routing for heuristic
//!    decisions, with foreground-family cascade and thermal override.
//!
//! 4. **`merge_cycle_metrics`** — Phase 3: merge `ExecuteOutcomes` into
//!    `MetricsState`, record cycle duration, write to disk.

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use chrono::Utc;

use apollo_optimizer::collector::SystemSnapshot;
use apollo_optimizer::engine::adaptive_governor::ProcessDecision;
use apollo_optimizer::engine::daemon_helpers::{
    append_timeline, battery_pressure_boost, compute_p95, write_metrics,
};
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::execute_actions::ExecuteOutcomes;
use apollo_optimizer::engine::io_tiering::IoShaper;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::mach_qos::SchedulingTier;
use apollo_optimizer::engine::nars_belief::ArousalState;
use apollo_optimizer::engine::network_monitor::NetworkMonitor;
use apollo_optimizer::engine::overflow_guard::OverflowThresholds;
use apollo_optimizer::engine::pipeline::learning_context::LearningContext;
use apollo_optimizer::engine::power_management::PowerManager;
use apollo_optimizer::engine::predictive_agent::Intervention;
use apollo_optimizer::engine::process_classifier::ProcessTier;
use apollo_optimizer::engine::process_tree::ProcessTree;
use apollo_optimizer::engine::profile_governor::GovernorDecision;
use apollo_optimizer::engine::signal_intelligence::SignalDigest;
use apollo_optimizer::engine::thermal_bailout::ThermalAction;
use apollo_optimizer::engine::types::BlockerScore;
use apollo_optimizer::engine::types::OptimizationProfile;

use crate::process_enrichment;

/// Update predictive agent + signal intelligence metrics for status reporting.
///
/// Locks `state.metrics` once, updates all learning-observable fields, then
/// releases the lock. The caller should not hold the lock when calling.
pub fn update_learning_metrics<'a>(
    state: &SharedState,
    lctx: &LearningContext<'a>,
    signal_digest: &SignalDigest,
    agent_intervention: &Intervention,
    arousal_state: &ArousalState,
) {
    let mut m = state.metrics.lock_recover();
    m.metrics.predictive_agent_active = lctx.predictive_agent.is_active();
    m.metrics.predictive_agent_cycles = lctx.predictive_agent.total_cycles();
    m.metrics.predictive_agent_arm_pulls = lctx.predictive_agent.arm_pulls();
    m.metrics.predictive_agent_last_intervention = format!("{:?}", agent_intervention);
    m.metrics.si_pressure_smooth = signal_digest.pressure_smooth;
    m.metrics.si_pressure_velocity = signal_digest.pressure_velocity;
    m.metrics.si_p_oom_30s = signal_digest.p_oom_30s;
    m.metrics.si_urgency = signal_digest.urgency;
    if signal_digest.regime_shift_up {
        m.metrics.si_regime_shifts += 1;
    }
    m.metrics.si_monopoly_risk = signal_digest.monopoly_risk;
    m.metrics.si_entropy_anomaly = signal_digest.entropy_anomaly;
    // Cable 4: top_causal_pairs() → expose in metrics for observability.
    m.metrics.causal_pairs = lctx
        .outcome_tracker
        .top_causal_pairs(5)
        .iter()
        .map(|(a, b, c)| format!("{} + {} ({})", a, b, c))
        .collect();
    m.metrics.natural_drift = lctx.outcome_tracker.natural_drift();
    m.metrics.short_drift_velocity = lctx.outcome_tracker.pressure_velocity_short();
    m.metrics.nars_drift_score = lctx.outcome_tracker.nars_drift_score();
    m.metrics.nars_drifted_beliefs = lctx.outcome_tracker.drift_detector.drifted_count;
    m.metrics.arousal_level = arousal_state.level;
    m.metrics.arousal_zone = arousal_state.zone().to_string();
    m.metrics.experience_memory_size = lctx.outcome_tracker.experience.len();
    m.metrics.causal_slow_horizon_count = lctx.causal_graph.slow_horizon_count();
    m.metrics.causal_mechanism_count = lctx.causal_graph.mechanism_count();
    // Top mechanism summaries for observability.
    m.metrics.causal_mechanisms = lctx
        .causal_graph
        .solid_edges_by_impact()
        .iter()
        .take(5)
        .filter_map(|e| {
            if e.mechanism.observations >= 3 {
                let m_type = e.mechanism.primary();
                let detail = match m_type {
                    "rss" => format!("−{:.0}MB", e.mechanism.rss_delta_mb),
                    "cpu" => format!("−{:.0}%", e.mechanism.cpu_delta_pct),
                    "swap" => format!("−{:.0}MB", e.mechanism.swap_delta_mb),
                    _ => "?".to_string(),
                };
                Some(format!("{} via {} ({})", e.cause, m_type, detail))
            } else {
                None
            }
        })
        .collect();
    // Causal effect average: mean impact_score across solid causal edges.
    // This is a real signal (confidence × avg_delta from observed pressure drops)
    // instead of the previous synthetic (effectiveness × 0.05) heuristic.
    m.metrics.causal_effect_avg = {
        let solid = lctx.causal_graph.solid_edges();
        if solid.is_empty() {
            0.0
        } else {
            let sum: f64 = solid.iter().map(|e| e.impact_score() as f64).sum();
            sum / solid.len() as f64
        }
    };
    // HRPO / Dr. Zero metrics
    m.metrics.dr_zero_self_challenge = lctx.outcome_tracker.self_challenge_score();
    m.metrics.dr_zero_groups = lctx
        .outcome_tracker
        .hop_group_summary()
        .iter()
        .map(|(hop, eff, count, pred_err)| {
            format!(
                "{:?}(eff={:.0}% n={} err={:.2})",
                hop,
                eff * 100.0,
                count,
                pred_err
            )
        })
        .collect();
    m.metrics.dr_zero_exploration = lctx
        .outcome_tracker
        .exploration_needed()
        .iter()
        .map(|(hop, err)| format!("{:?}(err={:.2})", hop, err))
        .collect();
}

/// I/O Traffic Shaping: foreground-aware disk bandwidth allocation.
///
/// Runs every 20 cycles (~10s). Only runs when root (`is_root` is true).
/// Updates `state.metrics.sysctl_reactive_writes` if changes were made.
///
/// Based on Iyer & Druschel 2001 — anticipatory scheduling + I/O priority classes
/// reduce foreground I/O latency by 50-70% under concurrent background load.
/// MIN_REAPPLY_SECS=60 means nothing actually reapplies within 60s anyway.
#[allow(clippy::too_many_arguments)]
pub fn apply_io_shaping(
    cycle_count: u64,
    is_root: bool,
    snapshot: &SystemSnapshot,
    foreground_pid: Option<u32>,
    process_tree: &ProcessTree,
    heuristic_decisions: &[ProcessDecision],
    power_mgr: &PowerManager,
    thermal_pressure_boost: f64,
    io_shaper: &mut IoShaper,
    state: &SharedState,
) {
    if cycle_count % 20 != 0 || !is_root {
        return;
    }
    let fg_family_io = process_enrichment::build_foreground_family(foreground_pid, process_tree);
    let fg_pids: Vec<u32> = fg_family_io.iter().copied().collect();
    let process_tiers: Vec<(u32, ProcessTier)> = heuristic_decisions
        .iter()
        .map(|d| (d.pid, d.tier))
        .collect();
    let under_pressure = snapshot.pressure.memory_pressure
        + battery_pressure_boost(power_mgr)
        + thermal_pressure_boost
        > 0.60;
    let mut qos = state.mach_qos.lock_recover();
    let io_changes = io_shaper.shape(&fg_pids, &process_tiers, under_pressure, Some(&mut qos));
    drop(qos);
    if io_changes > 0 {
        state.metrics.lock_recover().metrics.sysctl_reactive_writes += io_changes as u64;
    }
}

/// MachQoS routing: assign P-Cores / E-Cores based on heuristic decisions.
///
/// Skips SIGSTOP'd processes and forces E-Cores for all non-foreground processes
/// during thermal emergency (`thermal_action.force_ecores`). Cascades Foreground
/// tier to all children of the foreground app via the process tree.
///
/// GCs dead PIDs every 30 cycles. Updates qos_foreground_count, qos_background_count,
/// and qos_errors in `state.metrics`.
#[allow(clippy::too_many_arguments)]
pub fn apply_qos_routing(
    cycle_count: u64,
    state: &SharedState,
    foreground_pid: Option<u32>,
    process_tree: &ProcessTree,
    heuristic_decisions: &[ProcessDecision],
    heuristic_critical_pids: &HashSet<u32>,
    thermal_action: &ThermalAction,
) {
    // F5 — MachQoS: route processes to P-Cores / E-Cores based on heuristic decisions.
    // Skip SIGSTOP'd processes; force E-Cores for all during thermal emergency.
    let frozen_pids: HashSet<u32> = state.frozen_state.lock_recover().keys().copied().collect();

    // Build the foreground family set from the process tree.
    let fg_family = process_enrichment::build_foreground_family(foreground_pid, process_tree);

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
                use apollo_optimizer::engine::adaptive_governor::GovernorDecision as GovDecision;
                match decision.decision {
                    GovDecision::Allow => {
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
                    GovDecision::Throttle => return None,
                    GovDecision::Freeze | GovDecision::Kill => SchedulingTier::Background,
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
        m.metrics.qos_foreground_count += outcomes
            .iter()
            .filter(|o| o.tier == SchedulingTier::Foreground && o.success)
            .count() as u64;
        m.metrics.qos_background_count += outcomes
            .iter()
            .filter(|o| o.tier == SchedulingTier::Background && o.success)
            .count() as u64;
        m.metrics.qos_errors += outcomes.iter().filter(|o| !o.success).count() as u64;
    }
}

/// How often to flush runtime_metrics.json to disk.
/// At 300ms/cycle the file would otherwise be written ~3/s (73KB/s with
/// atomic-write overhead), hitting macOS's daily 2GB write budget in ~8h.
/// Writing every 25 cycles (~7.5s) reduces disk I/O by 25x.
const METRICS_DISK_WRITE_EVERY_N_CYCLES: u64 = 25;

/// Phase 3: Merge execution outcomes into metrics, update cycle timing, write to disk.
///
/// Acquires `state.metrics` once, merges all `exec_outcomes` counters, records
/// timing, profile transitions, and RL metrics, then writes the snapshot to disk.
///
/// # Mutated locals (via `&mut`)
/// - `override_was_active` — updated to reflect current override state
/// - `critical_failure_timestamps` — rolling 3-minute failure window
#[allow(clippy::too_many_arguments)]
pub fn merge_cycle_metrics<'a>(
    state: &SharedState,
    exec_outcomes: &ExecuteOutcomes,
    network_monitor: &NetworkMonitor,
    decision_reactor_weight: f64,
    decision_blockers: &[BlockerScore],
    current_profile: OptimizationProfile,
    governor_decision: &GovernorDecision,
    lctx: &LearningContext<'a>,
    overflow_thresholds: &OverflowThresholds,
    cycle_start: &Instant,
    reactor_weight: f64,
    override_was_active: &mut bool,
    critical_failure_timestamps: &mut Vec<Instant>,
    timeline_path: &Path,
    metrics_path: &Path,
    cycle_count: u64,
    in_sleep: bool,
) {
    let mut metrics = state.metrics.lock_recover();
    metrics.metrics.boosts_applied += exec_outcomes.boosts_applied;
    metrics.metrics.throttles_applied += exec_outcomes.throttles_applied;
    metrics.metrics.freezes_applied += exec_outcomes.freezes_applied;
    metrics.metrics.unfreezes_applied += exec_outcomes.unfreezes_applied;
    metrics.metrics.paging_hints_applied += exec_outcomes.paging_hints_applied;
    metrics.metrics.sysctl_applied += exec_outcomes.sysctl_applied;
    metrics.metrics.failures += exec_outcomes.failures;
    if let Some(e) = exec_outcomes.last_error.clone() {
        metrics.metrics.last_error = Some(e);
    }
    metrics.metrics.critical_background_skips += exec_outcomes.critical_background_skips;
    metrics.metrics.invalid_sysctl_denied += exec_outcomes.invalid_sysctl_denied;
    for skip in exec_outcomes.top_skipped.iter() {
        let skip = skip.clone();
        if metrics.metrics.top_skipped_processes.len() < 12
            && !metrics.metrics.top_skipped_processes.contains(&skip)
        {
            metrics.metrics.top_skipped_processes.push(skip);
        }
    }
    metrics.metrics.top_skipped_processes.truncate(12);
    metrics.metrics.throttle_reverted += exec_outcomes.throttle_reverted;
    metrics.metrics.thread_qos_applied += exec_outcomes.thread_qos_applied;

    // SysctlGovernor + NetworkMonitor metrics.
    metrics.metrics.sysctl_reactive_writes += exec_outcomes.sysctl_applied;
    {
        let hw = state.hardware.lock_recover();
        metrics.metrics.sysctl_governor_active_tunings = hw.sysctl_governor_status.active_tunings;
        metrics.metrics.sysctl_governor_total_writes = hw.sysctl_governor_status.total_writes;
    }
    metrics.metrics.network_retransmit_ratio = network_monitor.retransmission_rate();
    metrics.metrics.network_listen_drop_rate = network_monitor.listen_drop_rate();

    let had_new_failures = exec_outcomes.failures > 0;

    metrics.metrics.cycles += 1;
    metrics.metrics.reactor_pulses += if decision_reactor_weight > 0.2 { 1 } else { 0 };
    metrics.metrics.last_cycle_at = Some(Utc::now());
    metrics.metrics.last_blockers = decision_blockers.to_vec();
    metrics.metrics.effective_profile = current_profile;
    metrics.throttle_level = governor_decision.throttle_level.clone();
    metrics.metrics.throttle_level = governor_decision.throttle_level.clone();
    // Use MetricsState.thermal_state (set by reactor) — no re-lock needed
    metrics.metrics.thermal_state = metrics.thermal_state.clone();
    metrics.metrics.last_pressure_score = governor_decision.pressure_score;
    if governor_decision.override_expired {
        metrics.metrics.override_expirations += 1;
    }
    if governor_decision.override_active && !*override_was_active {
        metrics.metrics.override_activations += 1;
    }
    if let Some(transition) = governor_decision.transition.clone() {
        metrics.metrics.profile_switches += 1;
        {
            let mut pg = state.policy.lock_recover();
            pg.timeline.push_back(transition.clone());
            if pg.timeline.len() > 200 {
                pg.timeline.pop_front();
            }
        }
        append_timeline(timeline_path, &transition);
    }
    *override_was_active = governor_decision.override_active;

    let elapsed = cycle_start.elapsed().as_millis() as u64;
    metrics.metrics.cycle_durations_ms.push_back(elapsed);
    if metrics.metrics.cycle_durations_ms.len() > 120 {
        metrics.metrics.cycle_durations_ms.pop_front();
    }
    metrics.metrics.p95_cycle_ms =
        compute_p95(metrics.metrics.cycle_durations_ms.make_contiguous());

    // reactor_weight: write back local accumulated value to MetricsState
    metrics.reactor_event_weight = reactor_weight;

    let nowi = Instant::now();
    critical_failure_timestamps
        .retain(|t| nowi.duration_since(*t) <= std::time::Duration::from_secs(180));
    if had_new_failures {
        critical_failure_timestamps.push(nowi);
    }
    if critical_failure_timestamps.len() > 5 {
        state.policy.lock_recover().governor.force_safe_on_errors();
        critical_failure_timestamps.clear();
    }

    // Actualizar métricas del overflow guard antes de escribir.
    metrics.metrics.overflow_events_total = lctx.overflow_guard.history.total_overflows;
    metrics.metrics.overflow_events_7d = lctx.overflow_guard.recent_overflow_count(7);
    // B6 fix (round-3): report the *applied* compound offset (dynamic + RL +
    // workload + device, capped at -0.15) rather than the dynamic component
    // alone — otherwise the dashboard could show "recovered" while the live
    // threshold was still pinned at the floor.
    metrics.metrics.overflow_threshold_offset_pp = (lctx
        .overflow_guard
        .applied_offset(overflow_thresholds.workload_mode)
        * 100.0)
        .round() as i32;
    metrics.metrics.overflow_workload_mode = overflow_thresholds.workload_mode.as_str().to_string();

    // RL threshold agent metrics (Phase 4).
    if let Some(rl) = &lctx.overflow_guard.rl_agent {
        metrics.metrics.rl_adjustment_pp = (rl.current_adjustment * 100.0).round() as i32;
        metrics.metrics.rl_total_ticks = rl.total_ticks();
        metrics.metrics.rl_total_overflows = rl.total_overflows();
    }

    // Clone before releasing lock — write_metrics does file I/O
    // and holding the lock during I/O blocks GetStatus requests.
    let metrics_snapshot = metrics.metrics.clone();
    drop(metrics);
    // Rate-limit disk writes: atomic write (temp+rename) at 300ms/cycle was
    // writing 11KB × 2 = 22KB every 300ms = 73KB/s. Write every 25 cycles
    // (~7.5s) to stay within macOS's 24.86KB/s daily disk write budget.
    // Also skip writes while the system is sleeping — macOS accounts disk
    // writes against the daemon even during pre-sleep, burning the daily
    // budget while the machine is idle.
    if !in_sleep && cycle_count % METRICS_DISK_WRITE_EVERY_N_CYCLES == 0 {
        write_metrics(metrics_path, &metrics_snapshot);
    }
}
