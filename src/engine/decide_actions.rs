use std::collections::{HashMap, HashSet};

use sysinfo::System;

use crate::collector::SystemSnapshot;
use crate::engine::amx_detector;
use crate::engine::outcome_tracker::{PatternWeight, WorkloadHop, HopGroupWeight};
use crate::engine::overflow_guard::OverflowThresholds;
use crate::engine::safety::critical_background_processes;
use crate::engine::thread_selfcounts::IpcClass;
use crate::engine::types::{
    BlockerScore, InteractiveContext, LatencyTarget, OptimizationProfile, RootAction,
};

const INTERACTIVE_APPS: [&str; 9] = [
    "Code",
    "Arc",
    "Google Chrome",
    "Terminal",
    "iTerm",
    "Warp",
    "Antigravity",
    "Cursor",
    "LM Studio",
];

const NOISE_APPS: [&str; 6] = [
    "Dropbox",
    "Google Drive",
    "OneDrive",
    "corespeechd",
    "logioptionsplus",
    "suggestd",
];

const BLOCKER_APPS: [&str; 7] = [
    "WindowServer",
    "accountsd",
    "cfprefsd",
    "distnoted",
    "cloudd",
    "coreaudiod",
    "runningboardd",
];

#[derive(Debug, Clone)]
pub struct DecisionOutput {
    pub context: InteractiveContext,
    pub reactor_event_weight: f64,
    pub blockers: Vec<BlockerScore>,
    pub actions: Vec<RootAction>,
    /// Procesos skipeados en esta decisión por ser low_value según OutcomeTracker.
    /// Aparecen en `metrics.top_skipped_processes` para observabilidad.
    pub low_value_skipped: Vec<String>,
}

fn is_interactive_base(name: &str) -> bool {
    INTERACTIVE_APPS.iter().any(|n| name.contains(n))
}

fn is_background_noise_base(name: &str) -> bool {
    NOISE_APPS.iter().any(|n| name.contains(n))
}

fn is_known_blocker(name: &str) -> bool {
    BLOCKER_APPS.iter().any(|n| name.contains(n))
}

/// Classify system pressure into an interactive context.
/// Used by `decide_actions` and exposed for benchmarking.
pub fn context_from_pressure(
    snapshot: &SystemSnapshot,
    thresholds: &OverflowThresholds,
) -> InteractiveContext {
    let ram_pressure = snapshot.pressure.memory_pressure;
    let cpu_pressure = snapshot.cpu.global_usage as f64;

    if cpu_pressure > 88.0 || ram_pressure > thresholds.critical_pressure {
        InteractiveContext::ThermalConstrained
    } else if cpu_pressure > 72.0 || ram_pressure > thresholds.bg_pressure {
        InteractiveContext::BackgroundPressure
    } else {
        InteractiveContext::InteractiveFocus
    }
}

/// Pure blocker score formula — exposed for benchmarking.
///
/// Combines interactive wait ratio, CPU spike, recent sighting, and reactor weight
/// into a single blocker importance score.
pub fn blocker_score_formula(
    interactive_wait_ratio: f64,
    blocker_cpu_spike: f64,
    blocker_seen_recently: bool,
    reactor_event_weight: f64,
) -> f64 {
    (interactive_wait_ratio * 0.45)
        + (blocker_cpu_spike * 0.35)
        + (if blocker_seen_recently { 0.1 } else { 0.0 })
        + (reactor_event_weight * 0.1)
}

fn top_blockers(
    sys: &System,
    snapshot: &SystemSnapshot,
    reactor_event_weight: f64,
    learned_interactive: &[String],
) -> Vec<BlockerScore> {
    let mut blockers = Vec::new();

    // Pre-lowercase learned patterns once for this function.
    let interactive_lc: Vec<String> = learned_interactive
        .iter()
        .map(|s| s.to_ascii_lowercase())
        .collect();
    let interactive_waiters = snapshot
        .top_processes
        .iter()
        .filter(|p| {
            let lc = p.name.to_ascii_lowercase();
            (is_interactive_base(&p.name)
                || interactive_lc.iter().any(|pat| lc.contains(pat.as_str())))
                && p.cpu_usage < 8.0
                && p.memory_usage > 100 * 1024 * 1024
        })
        .count() as f64;

    let interactive_wait_ratio = (interactive_waiters / 5.0).clamp(0.0, 1.0);

    for (pid, process) in sys.processes() {
        let name = process.name().to_string();
        if !is_known_blocker(&name) {
            continue;
        }

        let cpu = process.cpu_usage();
        let blocker_cpu_spike = (cpu / 100.0).clamp(0.0, 1.0);
        let blocker_seen_recently = cpu > 8.0;
        let score = blocker_score_formula(
            interactive_wait_ratio,
            blocker_cpu_spike as f64,
            blocker_seen_recently,
            reactor_event_weight,
        );

        if score > 0.30 {
            blockers.push(BlockerScore {
                name,
                pid: pid.as_u32(),
                score,
                blocker_cpu_spike: cpu,
                interactive_wait_ratio,
                blocker_seen_recently,
                reactor_event_weight,
            });
        }
    }

    blockers.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    blockers.truncate(8);
    blockers
}

#[allow(clippy::too_many_arguments)]
pub fn decide_actions(
    snapshot: &SystemSnapshot,
    sys: &System,
    profile: OptimizationProfile,
    latency_target: LatencyTarget,
    reactor_event_weight: f64,
    learned_interactive: &[String],
    learned_noise: &[String],
    thresholds: OverflowThresholds,
    mut qos_mgr: Option<&mut crate::engine::mach_qos::MachQoSManager>,
    // Bayesian weights from OutcomeTracker.
    pattern_weights: &HashMap<String, PatternWeight>,
    // Tasa base de caídas de presión naturales (sin acción). Calibrada por
    // OutcomeTracker. Un proceso se skipea solo si su efectividad es <90%
    // de este baseline — i.e., no aporta más que la fluctuación de fondo.
    outcome_baseline: f64,
    // PIDs detected as behavior-interactive via cpu_wall_ratio EMA.
    // These are I/O-bound processes (low CPU/wall ratio) that behave like
    // interactive apps regardless of their name.
    behavior_interactive_pids: &HashSet<u32>,
    // Per-process IPC hints from energy_pid tracker (ri_instructions/ri_cycles).
    // Used for IPC-aware throttling: low IPC = memory-bound (safe to throttle),
    // high IPC = compute-bound (throttling hurts throughput).
    ipc_hints: &HashMap<u32, f64>,
    // HRPO group effectiveness from Dr. Zero — skip throttling groups with
    // consistently low effectiveness (< 15%) after sufficient observations.
    hop_groups: &HashMap<WorkloadHop, HopGroupWeight>,
    // Habituation filter (Thompson & Spencer 1966, inspired by memoria-core):
    // PIDs whose (cpu_bucket, rss_bucket) haven't changed in N cycles.
    // Skipped in the main loop — their last decision is maintained.
    habituated_pids: &HashSet<u32>,
) -> DecisionOutput {
    // Pre-lowercase learned patterns once (avoids per-process allocations).
    let interactive_lc: Vec<String> = learned_interactive
        .iter()
        .map(|s| s.to_ascii_lowercase())
        .collect();
    let noise_lc: Vec<String> = learned_noise
        .iter()
        .map(|s| s.to_ascii_lowercase())
        .collect();

    // Build closures that merge hardcoded lists with the learned policy.
    // Also checks behavior-interactive PIDs (cpu_wall_ratio EMA < 0.05).
    let is_interactive = |name: &str, pid: u32| -> bool {
        let name_lc = name.to_ascii_lowercase();
        is_interactive_base(name)
            || interactive_lc.iter().any(|p| name_lc.contains(p.as_str()))
            || behavior_interactive_pids.contains(&pid)
    };
    let is_background_noise = |name: &str| -> bool {
        let name_lc = name.to_ascii_lowercase();
        is_background_noise_base(name) || noise_lc.iter().any(|p| name_lc.contains(p.as_str()))
    };

    let mut actions = Vec::new();
    let mut low_value_skipped: Vec<String> = Vec::new();

    // Dev-first: protect critical background workloads and their children.
    let critical_patterns = critical_background_processes();
    let mut critical_pids: HashSet<u32> = HashSet::new();
    for (pid, process) in sys.processes() {
        let name = process.name().to_string();
        if critical_patterns.iter().any(|p| name.contains(p)) {
            critical_pids.insert(pid.as_u32());
        }
    }
    // AMX/ML protection: ML inference workloads must NEVER be throttled or frozen.
    // They run on P-cores, are expensive to restart, and are user-initiated.
    let ml_protected = amx_detector::ml_protected_pids();
    for &ml_pid in &ml_protected {
        critical_pids.insert(ml_pid);
    }

    // Add children-of-critical by walking parent chain once.
    // Depth-limited to prevent infinite loops from PID recycling (BUG 10 fix).
    const MAX_PARENT_DEPTH: usize = 20;
    for pid in sys.processes().keys() {
        let mut cur = Some(*pid);
        let mut is_child = false;
        let mut depth = 0usize;
        while let Some(p) = cur {
            if depth >= MAX_PARENT_DEPTH {
                break;
            }
            if critical_pids.contains(&p.as_u32()) {
                is_child = true;
                break;
            }
            cur = sys.process(p).and_then(|pp| pp.parent());
            depth += 1;
        }
        if is_child {
            critical_pids.insert(pid.as_u32());
        }
    }

    let context = context_from_pressure(snapshot, &thresholds);
    let blockers = top_blockers(sys, snapshot, reactor_event_weight, learned_interactive);

    // 1) Wait-graph practical: temporary boost for top blockers.
    let blocker_boost_count = match latency_target {
        LatencyTarget::Max => 3,
        LatencyTarget::Low => 1,
        LatencyTarget::Normal => 2,
    };
    for blocker in blockers.iter().take(blocker_boost_count) {
        actions.push(RootAction::BoostProcess {
            pid: blocker.pid,
            name: blocker.name.clone(),
            reason: format!("wait-graph blocker score {:.2}", blocker.score),
        });
    }

    // 2) Context-aware scheduling.
    for (pid, process) in sys.processes() {
        let name = process.name().to_string();
        let pid = pid.as_u32();

        if critical_pids.contains(&pid) {
            continue;
        }

        // Habituation: skip processes whose state hasn't changed in N cycles.
        // Their previous throttle/boost action is maintained by the executor.
        // Dishabituation occurs when CPU or RSS bucket changes (daemon side).
        if habituated_pids.contains(&pid) {
            continue;
        }

        if is_interactive(&name, pid) {
            actions.push(RootAction::BoostProcess {
                pid,
                name,
                reason: "interactive focus boost".to_string(),
            });
            continue;
        }

        if is_background_noise(&name) {
            // Outcome-tracker feedback: skip processes whose effectiveness is
            // not meaningfully above the natural baseline fluctuation rate.
            // Uses calibrated threshold (90% of baseline) to avoid false positives
            // from correlated pressure drops. Requires ≥20 throttle observations.
            if pattern_weights
                .get(&name)
                .map(|w| w.is_low_value_vs_baseline(outcome_baseline))
                .unwrap_or(false)
            {
                low_value_skipped.push(name);
                continue;
            }

            // HRPO group-level intelligence (Dr. Zero):
            // 1. Skip: groups throttled ≥20x with effectiveness <15%
            // 2. Explore: groups with high prediction error get tested regardless
            {
                let hop = WorkloadHop::from_process_name(&name);
                if let Some(group) = hop_groups.get(&hop) {
                    // Groups needing exploration bypass the skip — the solver
                    // is uncertain about their effectiveness and needs more data.
                    if !group.needs_exploration() {
                        if group.throttle_count >= 20 && group.effectiveness() < 0.15
                            && !matches!(context, InteractiveContext::ThermalConstrained)
                        {
                            low_value_skipped.push(format!("hrpo-skip:{}", name));
                            continue;
                        }
                    }
                }
            }

            // IPC-aware throttling: use per-process IPC to modulate aggressiveness.
            // Low IPC = memory-bound (stalled on cache misses) → throttle won't slow it down.
            // High IPC = compute-efficient → throttling directly hurts throughput.
            let ipc_class = ipc_hints
                .get(&pid)
                .map(|&ipc| IpcClass::from_ipc(ipc))
                .unwrap_or(IpcClass::Mixed);

            // Skip throttling for highly optimized compute-bound processes
            // unless we're in thermal emergency.
            if !ipc_class.safe_to_throttle()
                && !matches!(context, InteractiveContext::ThermalConstrained)
            {
                low_value_skipped.push(format!("ipc-protected:{}", name));
                continue;
            }

            let aggressive = match context {
                InteractiveContext::ThermalConstrained => true,
                InteractiveContext::BackgroundPressure => {
                    // Memory-bound processes: always throttle aggressively
                    // (throttling won't make them slower).
                    // Mixed/other: aggressive only if profile says so.
                    ipc_class.safe_to_throttle_aggressive()
                        || matches!(profile, OptimizationProfile::AggressiveRoot)
                }
                InteractiveContext::InteractiveFocus => {
                    ipc_class.safe_to_throttle_aggressive()
                        && matches!(profile, OptimizationProfile::AggressiveRoot)
                }
            };
            actions.push(RootAction::ThrottleProcess {
                pid,
                name,
                aggressive,
                reason: format!(
                    "ipc-aware throttle ({:?}, ipc={:.2})",
                    context,
                    ipc_hints.get(&pid).copied().unwrap_or(0.0)
                ),
                start_sec: process.start_time(),
                start_usec: 0,
            });
        }
    }

    // 3) ML workload boost: route AMX/ML inference processes to P-cores.
    // These are user-initiated, expensive to restart, and need maximum bandwidth.
    for &ml_pid in &ml_protected {
        if let Some(process) = sys.process(sysinfo::Pid::from_u32(ml_pid)) {
            actions.push(RootAction::BoostProcess {
                pid: ml_pid,
                name: process.name().to_string(),
                reason: "ML/AMX workload — P-core routing".to_string(),
            });
        }
    }

    // 4) Phase 1: Thread-level scheduling for multi-threaded processes.
    // For non-interactive, non-critical processes using >15% CPU with multiple threads,
    // route hot threads to P-cores and cold threads to E-cores.
    if let Some(ref mut mgr) = qos_mgr {
        let thread_cpu_threshold = 15.0;
        let mut thread_actions_emitted = 0usize;
        let max_thread_actions = match profile {
            OptimizationProfile::AggressiveRoot => 20,
            OptimizationProfile::SafeRoot => 4,
            OptimizationProfile::BalancedRoot => 10,
        };

        for (pid, process) in sys.processes() {
            if thread_actions_emitted >= max_thread_actions {
                break;
            }
            let pid_u32 = pid.as_u32();
            let name = process.name().to_string();
            let cpu = process.cpu_usage();

            // Skip critical, interactive, and low-CPU processes.
            if critical_pids.contains(&pid_u32)
                || is_interactive(&name, pid_u32)
                || cpu < thread_cpu_threshold
            {
                continue;
            }

            // Enumerate threads and analyze patterns.
            if let Some(threads) = mgr.enumerate_threads(pid_u32) {
                if threads.len() < 2 {
                    continue; // Single-threaded: task-level scheduling is sufficient.
                }

                let analysis = mgr.analyze_threads(pid_u32, &threads);

                // Pattern-aware scheduling:
                match analysis.pattern {
                    crate::engine::mach_qos::ThreadPattern::Runaway => {
                        // Runaway thread: throttle only the hot thread(s),
                        // don't penalize the whole process.
                        for &idx in &analysis.hot {
                            if thread_actions_emitted >= max_thread_actions {
                                break;
                            }
                            actions.push(RootAction::SetThreadQoS {
                                pid: pid_u32,
                                name: name.clone(),
                                thread_index: idx,
                                tier: "utility".to_string(),
                                reason: format!(
                                    "runaway thread #{} in {} (1 hot / {} threads)",
                                    idx, name, analysis.thread_count
                                ),
                            });
                            thread_actions_emitted += 1;
                        }
                    }
                    crate::engine::mach_qos::ThreadPattern::Saturated => {
                        // CPU-bound saturation: legitimate workload (build, render).
                        // Don't interfere — the process needs all its threads on P-cores.
                        // Only route truly idle threads to E-cores.
                        for &idx in &analysis.cold {
                            if thread_actions_emitted >= max_thread_actions {
                                break;
                            }
                            actions.push(RootAction::SetThreadQoS {
                                pid: pid_u32,
                                name: name.clone(),
                                thread_index: idx,
                                tier: "background".to_string(),
                                reason: format!(
                                    "cold thread #{} in saturated {} ({}/{} active)",
                                    idx, name, analysis.active_count, analysis.thread_count
                                ),
                            });
                            thread_actions_emitted += 1;
                        }
                    }
                    crate::engine::mach_qos::ThreadPattern::IoBound => {
                        // I/O-bound: most threads waiting. Move entire process
                        // to E-cores aggressively — it's not using CPU anyway.
                        for &idx in &analysis.cold {
                            if thread_actions_emitted >= max_thread_actions {
                                break;
                            }
                            actions.push(RootAction::SetThreadQoS {
                                pid: pid_u32,
                                name: name.clone(),
                                thread_index: idx,
                                tier: "background".to_string(),
                                reason: format!(
                                    "I/O-bound thread #{} in {} ({}/{} waiting)",
                                    idx,
                                    name,
                                    analysis.cold.len(),
                                    analysis.thread_count,
                                ),
                            });
                            thread_actions_emitted += 1;
                        }
                    }
                    crate::engine::mach_qos::ThreadPattern::Normal => {
                        // Normal mixed pattern: original hot→P-core, cold→E-core logic.
                        for &idx in &analysis.hot {
                            if thread_actions_emitted >= max_thread_actions {
                                break;
                            }
                            actions.push(RootAction::SetThreadQoS {
                                pid: pid_u32,
                                name: name.clone(),
                                thread_index: idx,
                                tier: "interactive".to_string(),
                                reason: format!(
                                    "hot thread #{} in {} (cpu={:.1}%)",
                                    idx, name, cpu
                                ),
                            });
                            thread_actions_emitted += 1;
                        }
                        for &idx in &analysis.cold {
                            if thread_actions_emitted >= max_thread_actions {
                                break;
                            }
                            actions.push(RootAction::SetThreadQoS {
                                pid: pid_u32,
                                name: name.clone(),
                                thread_index: idx,
                                tier: "background".to_string(),
                                reason: format!("cold thread #{} in {} (waiting)", idx, name),
                            });
                            thread_actions_emitted += 1;
                        }
                    }
                }
            }
        }
    }

    // 5) Pressure actions with hysteresis-ish behavior by context.
    match context {
        InteractiveContext::BackgroundPressure | InteractiveContext::ThermalConstrained => {
            // Dev-first: no-freeze by default. Only consider freeze under extreme memory pressure
            // AND swap growth, and never for protected/critical workloads.
            let extreme_freeze_ok = snapshot.pressure.memory_pressure
                >= thresholds.extreme_pressure
                && snapshot.pressure.swap_delta_bytes_per_sec > (5.0 * 1024.0 * 1024.0);
            if extreme_freeze_ok {
                for (pid, process) in sys.processes() {
                    let pid = pid.as_u32();
                    if critical_pids.contains(&pid) {
                        continue;
                    }
                    let name = process.name().to_string();
                    // Never freeze interactive apps even under extreme pressure.
                    if is_interactive(&name, pid) {
                        continue;
                    }
                    if ["Slack", "Discord", "Spotify", "Teams"]
                        .iter()
                        .any(|n| name.contains(n))
                    {
                        actions.push(RootAction::FreezeProcess {
                            pid,
                            name,
                            reason: format!("extreme pressure quarantine under {:?}", context),
                            start_sec: process.start_time(),
                            start_usec: 0,
                        });
                    }
                }
            }

            // SysctlGovernor now owns vm.compressor_poll_interval tuning
        }
        InteractiveContext::InteractiveFocus => {
            // SysctlGovernor now owns debug.lowpri_throttle_enabled tuning
        }
    }

    DecisionOutput {
        context,
        reactor_event_weight,
        blockers,
        actions,
        low_value_skipped,
    }
}
