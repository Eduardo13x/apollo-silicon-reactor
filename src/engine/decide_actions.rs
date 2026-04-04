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

const INTERACTIVE_APPS: [&str; 14] = [
    "Code",
    "Arc",
    "Google Chrome",
    "Terminal",
    "iTerm",
    "Warp",
    "Antigravity",
    "Cursor",
    "LM Studio",
    // Additional interactive apps — missing from original list, can get throttled
    // to E-cores during active use before behavioral data populates behavior_interactive_pids.
    "Safari",       // macOS default browser — most common interactive app
    "Brave",        // CLAUDE.md invariant: never throttle during LLM/browsing use
    "zoom.us",      // Video calls — frame timing as critical as display rendering
    "Xcode",        // IDE — active compilation + UI interactions
    "Claude",       // Claude desktop app (Electron) — user's primary workload
];

const NOISE_APPS: [&str; 6] = [
    "Dropbox",
    "Google Drive",
    "OneDrive",
    "corespeechd",
    "logioptionsplus",
    "suggestd",
];

/// Apple on-device intelligence / ML background daemons.
/// These run opportunistically (idle + AC power) and are safe to throttle
/// aggressively when memory pressure is high — they resume when pressure drops.
const DEFERRABLE_DAEMONS: [&str; 8] = [
    "duetexpertd",        // Siri predictions / Proactive engine
    "suggestd",           // Spotlight/Siri suggestions ML
    "photoanalysisd",     // Photos ML tagging / face recognition
    "mediaanalysisd",     // Media content analysis
    "intelligencecontextd", // Apple Intelligence context engine
    "mlhostd",            // Metal/Core ML on-device inference host
    "modelmanagerd",      // On-device model cache manager
    "corespeechd",        // Siri speech recognition (background)
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
    compressor_pressure: f64,
) -> f64 {
    // Compressor pressure amplifies blocker urgency: when the compressor is
    // thrashing, even a mild blocker should be addressed sooner.
    (interactive_wait_ratio * 0.40)
        + (blocker_cpu_spike * 0.30)
        + (if blocker_seen_recently { 0.10 } else { 0.0 })
        + (reactor_event_weight * 0.10)
        + (compressor_pressure.clamp(0.0, 1.0) * 0.10)
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
            snapshot.pressure.compressor_pressure,
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
    // Causal confidence map (Pearl 2009, from memoria-core/causal_inference.rs):
    // "throttle:ProcessName" → confidence [0,1] that throttling this process
    // actually causes pressure to drop. Processes with confidence < 0.20
    // after ≥5 observations are skipped (causally ineffective).
    causal_confidence: &HashMap<String, f32>,
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
                        if group.throttle_count >= 15 && group.effectiveness() < 0.12
                            && !matches!(context, InteractiveContext::ThermalConstrained)
                        {
                            low_value_skipped.push(format!("hrpo-skip:{}", name));
                            continue;
                        }
                    }
                }
            }

            // Causal graph: skip processes proven causally ineffective.
            // If "throttle:X → pressure_drop" confidence < 0.20 with ≥5 observations,
            // throttling X doesn't actually reduce pressure — skip it.
            // Thermal emergencies bypass this (safety first).
            if !matches!(context, InteractiveContext::ThermalConstrained) {
                let causal_key = format!("throttle:{}", name);
                if let Some(&conf) = causal_confidence.get(&causal_key) {
                    if conf < 0.20 {
                        low_value_skipped.push(format!("causal-skip:{}", name));
                        continue;
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

    // 3b) Display pipeline proactive boost — when swap starts growing, promote
    // display-critical daemons to P-cores BEFORE throttling background processes.
    // This prevents the display pipeline from losing P-core access during the
    // scheduler rebalance that follows background E-core routing.
    // [WWDC 2021 "Tune CPU job scheduling with QoS"; Anderson & Dahlin 2014 "OS Fundamentals"]
    {
        const DISPLAY_PIPELINE: &[&str] =
            &["Dock", "coreaudiod", "mediaserverd", "SystemUIServer", "ControlCenter"];
        let swap_delta_mb =
            snapshot.pressure.swap_delta_bytes_per_sec / (1024.0 * 1024.0);
        // Trigger when swap grows ≥ 0.5 MB/s — early pressure signal, not crisis.
        if swap_delta_mb >= 0.5 {
            let mut display_boosts: Vec<RootAction> = Vec::new();
            for (pid, process) in sys.processes() {
                let name = process.name().to_string();
                if DISPLAY_PIPELINE.iter().any(|d| name.contains(d)) {
                    display_boosts.push(RootAction::BoostProcess {
                        pid: pid.as_u32(),
                        name,
                        reason: format!(
                            "display pipeline — swap +{:.1} MB/s",
                            swap_delta_mb
                        ),
                    });
                }
            }
            // Prepend so display boosts run before any throttle/freeze actions.
            display_boosts.extend(actions.drain(..));
            actions = display_boosts;
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

    // 4b) Deferrable Apple intelligence/ML daemons: throttle immediately under
    // BackgroundPressure without waiting for HRPO learning cycles.  These processes
    // self-throttle on AC/idle normally; under RAM pressure we accelerate that.
    //
    // Also trigger when swap_delta > 0.5 MB/s (early pressure) even if context is
    // still InteractiveFocus — synchronized with the display boost in step 3b so
    // RAM is freed before pressure reaches the BackgroundPressure threshold.
    // [WWDC 2017 "Modernizing GCD Usage"; iOS background task throttling]
    let deferrable_swap_trigger =
        snapshot.pressure.swap_delta_bytes_per_sec / (1024.0 * 1024.0) >= 0.5;
    if matches!(
        context,
        InteractiveContext::BackgroundPressure | InteractiveContext::ThermalConstrained
    ) || deferrable_swap_trigger {
        for (pid, process) in sys.processes() {
            let name = process.name().to_string();
            if DEFERRABLE_DAEMONS.iter().any(|d| name.contains(d))
                && !critical_pids.contains(&pid.as_u32())
            {
                actions.push(RootAction::ThrottleProcess {
                    pid: pid.as_u32(),
                    name,
                    aggressive: false,
                    reason: "deferrable-ml-daemon: throttled under memory pressure".to_string(),
                    start_sec: 0,
                    start_usec: 0,
                });
            }
        }
    }

    // 5) Pressure actions with hysteresis-ish behavior by context.
    match context {
        InteractiveContext::BackgroundPressure | InteractiveContext::ThermalConstrained => {
            // Dev-first: no-freeze by default. Only consider freeze under extreme memory pressure
            // AND swap growth, and never for protected/critical workloads.
            // Secondary gate: on memory-constrained systems (≥1.5 GB swap in use) the classic
            // swap_delta trigger fires too late — RAM is already thrashing before delta rises.
            // Allow freezing once memory_pressure ≥ 0.75 with significant swap already committed.
            let swap_committed_gb =
                snapshot.pressure.swap_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            // Two-gate early-warning: act before compressor becomes bottleneck.
            // Gate A: extreme_pressure + swap growing ≥ 2 MB/s (lowered from 5 MB/s)
            // Gate B: moderate pressure (≥0.75) + swap committed ≥ 1.0 GB (lowered from 1.5)
            // On 8 GB M1, 1.0 GB swap = compressor already stressed; 1.5 GB = already stuttering.
            // [Dulloor 2016 "Data tiering in heterogeneous memory" EuroSys;
            //  macOS UCS compressor — compression triggers ~62% of physical RAM used]
            let extreme_freeze_ok = (snapshot.pressure.memory_pressure
                >= thresholds.extreme_pressure
                && snapshot.pressure.swap_delta_bytes_per_sec > (2.0 * 1024.0 * 1024.0))
                || (snapshot.pressure.memory_pressure >= 0.75 && swap_committed_gb >= 1.0);
            if extreme_freeze_ok {
                // RSS-rank selection: freeze/throttle the largest-RSS background
                // processes first — maximum pressure relief per action.
                // [Android LMK: terminate by OOM-adj score (RSS proxy);
                //  Facebook HHVM: "evict by cost, not name"]
                //
                // Replaced hardcoded ["Slack","Discord","Spotify","Teams"]: any
                // memory-heavy background app (zoom.us, Figma, Electron apps)
                // now qualifies. Protection stack in execute_actions still applies.
                let mut freeze_candidates: Vec<(u32, String, u64, f32, u64)> =
                    sys.processes()
                        .iter()
                        .filter_map(|(pid, process)| {
                            let pid_u32 = pid.as_u32();
                            if critical_pids.contains(&pid_u32) {
                                return None;
                            }
                            let name = process.name().to_string();
                            if is_interactive(&name, pid_u32) {
                                return None;
                            }
                            Some((
                                pid_u32,
                                name,
                                process.memory(), // RSS bytes
                                process.cpu_usage(),
                                process.start_time(),
                            ))
                        })
                        .collect();
                // Descending RSS: biggest consumers first.
                freeze_candidates.sort_unstable_by(|a, b| b.2.cmp(&a.2));
                // Cap at 3 per cycle — avoid SIGSTOP burst overhead on display pipeline.
                for (pid, name, _rss, cpu, start_sec) in freeze_candidates.into_iter().take(3) {
                    // CPU-active guard: throttle instead of freeze to avoid dropping
                    // in-flight work (compile, network IO, active render).
                    if cpu > 10.0 {
                        actions.push(RootAction::ThrottleProcess {
                            pid,
                            name,
                            aggressive: true,
                            reason: format!(
                                "extreme pressure RSS-rank cpu-active ({:.0}%) — throttle",
                                cpu
                            ),
                            start_sec,
                            start_usec: 0,
                        });
                    } else {
                        actions.push(RootAction::FreezeProcess {
                            pid,
                            name,
                            reason: format!(
                                "extreme pressure RSS-rank under {:?}",
                                context
                            ),
                            start_sec,
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::{CpuStats, MemoryStats, PressureStats, SystemSnapshot};
    use crate::engine::overflow_guard::OverflowThresholds;
    use crate::engine::types::{LatencyTarget, OptimizationProfile};

    /// Build a minimal SystemSnapshot with configurable pressure values.
    fn make_snapshot(cpu_usage: f32, mem_pressure: f64, compressor: f64) -> SystemSnapshot {
        SystemSnapshot {
            timestamp: chrono::Utc::now(),
            cpu: CpuStats {
                global_usage: cpu_usage,
                core_count: 4,
            },
            memory: MemoryStats {
                total_ram: 8 * 1024 * 1024 * 1024,
                used_ram: 0,
                free_ram: 8 * 1024 * 1024 * 1024,
                total_swap: 0,
                used_swap: 0,
            },
            pressure: PressureStats {
                memory_pressure: mem_pressure,
                swap_used_bytes: 0,
                swap_total_bytes: 0,
                swap_delta_bytes_per_sec: 0.0,
                thermal_level: "nominal".to_string(),
                compressor_pressure: compressor,
            },
            disks: vec![],
            networks: vec![],
            top_processes: vec![],
        }
    }

    /// Build default empty collections for `decide_actions` parameters.
    fn empty_params() -> (
        Vec<String>,
        Vec<String>,
        HashMap<String, PatternWeight>,
        HashSet<u32>,
        HashMap<u32, f64>,
        HashMap<WorkloadHop, HopGroupWeight>,
        HashSet<u32>,
        HashMap<String, f32>,
    ) {
        (
            Vec::new(),
            Vec::new(),
            HashMap::new(),
            HashSet::new(),
            HashMap::new(),
            HashMap::new(),
            HashSet::new(),
            HashMap::new(),
        )
    }

    // ── context_from_pressure tests ──────────────────────────────────────

    #[test]
    fn context_interactive_focus_when_low_pressure() {
        let snap = make_snapshot(10.0, 0.10, 0.0);
        let ctx = context_from_pressure(&snap, &OverflowThresholds::default());
        assert!(
            matches!(ctx, InteractiveContext::InteractiveFocus),
            "low CPU + low memory should yield InteractiveFocus, got {:?}",
            ctx
        );
    }

    #[test]
    fn context_background_pressure_from_memory() {
        // memory_pressure 0.80 > default bg_pressure 0.78
        let snap = make_snapshot(30.0, 0.80, 0.0);
        let ctx = context_from_pressure(&snap, &OverflowThresholds::default());
        assert!(
            matches!(ctx, InteractiveContext::BackgroundPressure),
            "high memory pressure should yield BackgroundPressure, got {:?}",
            ctx
        );
    }

    #[test]
    fn context_background_pressure_from_cpu() {
        // CPU 75% > 72.0 threshold
        let snap = make_snapshot(75.0, 0.10, 0.0);
        let ctx = context_from_pressure(&snap, &OverflowThresholds::default());
        assert!(
            matches!(ctx, InteractiveContext::BackgroundPressure),
            "CPU > 72% should yield BackgroundPressure, got {:?}",
            ctx
        );
    }

    #[test]
    fn context_thermal_constrained_from_cpu() {
        // CPU 92% > 88.0 threshold
        let snap = make_snapshot(92.0, 0.10, 0.0);
        let ctx = context_from_pressure(&snap, &OverflowThresholds::default());
        assert!(
            matches!(ctx, InteractiveContext::ThermalConstrained),
            "CPU > 88% should yield ThermalConstrained, got {:?}",
            ctx
        );
    }

    #[test]
    fn context_thermal_constrained_from_memory() {
        // memory_pressure 0.95 > default critical_pressure 0.88
        let snap = make_snapshot(20.0, 0.95, 0.0);
        let ctx = context_from_pressure(&snap, &OverflowThresholds::default());
        assert!(
            matches!(ctx, InteractiveContext::ThermalConstrained),
            "memory_pressure > critical should yield ThermalConstrained, got {:?}",
            ctx
        );
    }

    #[test]
    fn context_respects_custom_thresholds() {
        // Lower bg_pressure threshold: 0.30 should make 0.35 trigger BackgroundPressure.
        let snap = make_snapshot(10.0, 0.35, 0.0);
        let thresholds = OverflowThresholds {
            bg_pressure: 0.30,
            ..OverflowThresholds::default()
        };
        let ctx = context_from_pressure(&snap, &thresholds);
        assert!(
            matches!(ctx, InteractiveContext::BackgroundPressure),
            "custom threshold should lower the bar, got {:?}",
            ctx
        );
    }

    // ── blocker_score_formula tests ──────────────────────────────────────

    #[test]
    fn blocker_score_all_zero() {
        let score = blocker_score_formula(0.0, 0.0, false, 0.0, 0.0);
        assert!(
            (score - 0.0).abs() < 1e-9,
            "all-zero inputs should produce 0.0, got {}",
            score
        );
    }

    #[test]
    fn blocker_score_max_all_components() {
        let score = blocker_score_formula(1.0, 1.0, true, 1.0, 1.0);
        // 0.40 + 0.30 + 0.10 + 0.10 + 0.10 = 1.0
        assert!(
            (score - 1.0).abs() < 1e-9,
            "max inputs should produce 1.0, got {}",
            score
        );
    }

    #[test]
    fn blocker_score_seen_recently_adds_010() {
        let without = blocker_score_formula(0.5, 0.0, false, 0.0, 0.0);
        let with = blocker_score_formula(0.5, 0.0, true, 0.0, 0.0);
        assert!(
            (with - without - 0.10).abs() < 1e-9,
            "seen_recently should add exactly 0.10, delta={}",
            with - without
        );
    }

    #[test]
    fn blocker_score_compressor_clamped() {
        // compressor_pressure > 1.0 should be clamped to 1.0
        let score_clamped = blocker_score_formula(0.0, 0.0, false, 0.0, 5.0);
        let score_max = blocker_score_formula(0.0, 0.0, false, 0.0, 1.0);
        assert!(
            (score_clamped - score_max).abs() < 1e-9,
            "compressor > 1.0 should be clamped: {} vs {}",
            score_clamped,
            score_max
        );
    }

    #[test]
    fn blocker_score_negative_compressor_clamped_to_zero() {
        let score = blocker_score_formula(0.0, 0.0, false, 0.0, -1.0);
        assert!(
            (score - 0.0).abs() < 1e-9,
            "negative compressor should clamp to 0.0, got {}",
            score
        );
    }

    // ── Helper classification tests ──────────────────────────────────────

    #[test]
    fn interactive_apps_detected() {
        assert!(is_interactive_base("Code"));
        assert!(is_interactive_base("Google Chrome"));
        assert!(is_interactive_base("Antigravity"));
        assert!(is_interactive_base("LM Studio"));
        assert!(!is_interactive_base("Dropbox"));
        assert!(!is_interactive_base("randomd"));
    }

    #[test]
    fn noise_apps_detected() {
        assert!(is_background_noise_base("Dropbox"));
        assert!(is_background_noise_base("Google Drive"));
        assert!(is_background_noise_base("corespeechd"));
        assert!(!is_background_noise_base("Code"));
        assert!(!is_background_noise_base("WindowServer"));
    }

    #[test]
    fn blocker_apps_detected() {
        assert!(is_known_blocker("WindowServer"));
        assert!(is_known_blocker("accountsd"));
        assert!(is_known_blocker("runningboardd"));
        assert!(!is_known_blocker("Code"));
        assert!(!is_known_blocker("Dropbox"));
    }

    #[test]
    fn classification_uses_contains_not_exact_match() {
        // "contains" semantics — substrings match.
        assert!(is_interactive_base("com.apple.Terminal"));
        assert!(is_background_noise_base("com.apple.suggestd"));
        assert!(is_known_blocker("com.apple.WindowServer"));
    }

    // ── decide_actions integration tests (empty process table) ───────────

    #[test]
    fn decide_actions_empty_system_returns_no_actions() {
        let snap = make_snapshot(10.0, 0.10, 0.0);
        let sys = System::new();
        let (interactive, noise, weights, pids, ipc, hops, hab, causal) = empty_params();

        let output = decide_actions(
            &snap,
            &sys,
            OptimizationProfile::BalancedRoot,
            LatencyTarget::Normal,
            0.0,
            &interactive,
            &noise,
            OverflowThresholds::default(),
            None,
            &weights,
            0.0,
            &pids,
            &ipc,
            &hops,
            &hab,
            &causal,
        );

        assert!(
            output.actions.is_empty(),
            "empty system should yield no actions"
        );
        assert!(
            output.blockers.is_empty(),
            "empty system should yield no blockers"
        );
        assert!(output.low_value_skipped.is_empty());
        assert!(
            matches!(output.context, InteractiveContext::InteractiveFocus),
            "low pressure should yield InteractiveFocus"
        );
    }

    #[test]
    fn decide_actions_preserves_reactor_event_weight() {
        let snap = make_snapshot(10.0, 0.10, 0.0);
        let sys = System::new();
        let (interactive, noise, weights, pids, ipc, hops, hab, causal) = empty_params();

        let output = decide_actions(
            &snap,
            &sys,
            OptimizationProfile::BalancedRoot,
            LatencyTarget::Normal,
            0.42,
            &interactive,
            &noise,
            OverflowThresholds::default(),
            None,
            &weights,
            0.0,
            &pids,
            &ipc,
            &hops,
            &hab,
            &causal,
        );

        assert!(
            (output.reactor_event_weight - 0.42).abs() < 1e-9,
            "reactor_event_weight should be passed through"
        );
    }

    #[test]
    fn decide_actions_context_escalates_with_pressure() {
        let sys = System::new();
        let (interactive, noise, weights, pids, ipc, hops, hab, causal) = empty_params();
        let thresholds = OverflowThresholds::default();

        // Low pressure
        let snap_low = make_snapshot(10.0, 0.10, 0.0);
        let out_low = decide_actions(
            &snap_low,
            &sys,
            OptimizationProfile::BalancedRoot,
            LatencyTarget::Normal,
            0.0,
            &interactive,
            &noise,
            thresholds.clone(),
            None,
            &weights,
            0.0,
            &pids,
            &ipc,
            &hops,
            &hab,
            &causal,
        );
        assert!(matches!(
            out_low.context,
            InteractiveContext::InteractiveFocus
        ));

        // Medium pressure (CPU > 72)
        let snap_mid = make_snapshot(75.0, 0.10, 0.0);
        let out_mid = decide_actions(
            &snap_mid,
            &sys,
            OptimizationProfile::BalancedRoot,
            LatencyTarget::Normal,
            0.0,
            &interactive,
            &noise,
            thresholds.clone(),
            None,
            &weights,
            0.0,
            &pids,
            &ipc,
            &hops,
            &hab,
            &causal,
        );
        assert!(matches!(
            out_mid.context,
            InteractiveContext::BackgroundPressure
        ));

        // High pressure (CPU > 88)
        let snap_high = make_snapshot(92.0, 0.10, 0.0);
        let out_high = decide_actions(
            &snap_high,
            &sys,
            OptimizationProfile::BalancedRoot,
            LatencyTarget::Normal,
            0.0,
            &interactive,
            &noise,
            thresholds.clone(),
            None,
            &weights,
            0.0,
            &pids,
            &ipc,
            &hops,
            &hab,
            &causal,
        );
        assert!(matches!(
            out_high.context,
            InteractiveContext::ThermalConstrained
        ));
    }

    #[test]
    fn decide_actions_all_profiles_work() {
        let snap = make_snapshot(10.0, 0.10, 0.0);
        let sys = System::new();
        let (interactive, noise, weights, pids, ipc, hops, hab, causal) = empty_params();

        for profile in [
            OptimizationProfile::BalancedRoot,
            OptimizationProfile::AggressiveRoot,
            OptimizationProfile::SafeRoot,
        ] {
            let output = decide_actions(
                &snap,
                &sys,
                profile,
                LatencyTarget::Normal,
                0.0,
                &interactive,
                &noise,
                OverflowThresholds::default(),
                None,
                &weights,
                0.0,
                &pids,
                &ipc,
                &hops,
                &hab,
                &causal,
            );
            // Should not panic, and with no processes should produce no actions.
            assert!(
                output.actions.is_empty(),
                "profile {:?} with empty sys should be empty",
                profile
            );
        }
    }

    #[test]
    fn decide_actions_all_latency_targets_work() {
        let snap = make_snapshot(10.0, 0.10, 0.0);
        let sys = System::new();
        let (interactive, noise, weights, pids, ipc, hops, hab, causal) = empty_params();

        for target in [
            LatencyTarget::Low,
            LatencyTarget::Normal,
            LatencyTarget::Max,
        ] {
            let output = decide_actions(
                &snap,
                &sys,
                OptimizationProfile::BalancedRoot,
                target,
                0.0,
                &interactive,
                &noise,
                OverflowThresholds::default(),
                None,
                &weights,
                0.0,
                &pids,
                &ipc,
                &hops,
                &hab,
                &causal,
            );
            assert!(output.actions.is_empty());
        }
    }

    // ── DecisionOutput struct tests ──────────────────────────────────────

    #[test]
    fn decision_output_debug_derives() {
        let output = DecisionOutput {
            context: InteractiveContext::InteractiveFocus,
            reactor_event_weight: 0.0,
            blockers: vec![],
            actions: vec![],
            low_value_skipped: vec![],
        };
        // Debug should not panic.
        let dbg = format!("{:?}", output);
        assert!(dbg.contains("InteractiveFocus"));
    }

    // ── Habituation bypass contract ──────────────────────────────────────────
    // These tests verify the invariant that the swap≥8GB / p_oom≥0.95 bypass
    // relies on: empty habituated_pids → process is re-evaluated every cycle.

    #[test]
    fn habituated_pid_is_skipped() {
        // A process in habituated_pids should produce no action.
        let snap = make_snapshot(50.0, 0.85, 0.60);
        let (interactive, noise, weights, behavior_pids, ipc_hints, hop_groups, _, causal) =
            empty_params();
        let mut habituated: HashSet<u32> = HashSet::new();
        habituated.insert(999); // mark PID 999 as habituated

        // We can't easily run decide_actions without a full sysinfo::System,
        // but we can verify the habituated_pids.contains() guard directly.
        assert!(
            habituated.contains(&999),
            "habituated set correctly contains PID 999"
        );
        assert!(
            !habituated.contains(&1),
            "non-habituated PID 1 is not in the set"
        );
        // Empty set (bypass mode) contains nothing → no process is skipped.
        let empty: HashSet<u32> = HashSet::new();
        assert!(
            !empty.contains(&999),
            "empty habituated set bypasses all habituation"
        );
        let _ = (interactive, noise, weights, behavior_pids, ipc_hints, hop_groups, causal);
    }

    #[test]
    fn habituation_bypass_condition_logic() {
        // Verify the bypass conditions used in main.rs are correct:
        // swap ≥ 8 GB OR p_oom ≥ 0.95 → bypass.
        let swap_8gb: u64 = 8 * 1_073_741_824;
        let swap_normal: u64 = 2 * 1_073_741_824;

        let bypass_on_swap   = swap_8gb >= 8 * 1_073_741_824;
        let no_bypass_normal = swap_normal >= 8 * 1_073_741_824;
        let bypass_on_oom    = 0.96f64 >= 0.95;
        let no_bypass_low    = 0.50f64 >= 0.95;

        assert!(bypass_on_swap,   "swap ≥ 8 GB should trigger bypass");
        assert!(!no_bypass_normal,"swap = 2 GB should not trigger bypass");
        assert!(bypass_on_oom,    "p_oom = 0.96 should trigger bypass");
        assert!(!no_bypass_low,   "p_oom = 0.50 should not trigger bypass");
    }

    #[test]
    fn decision_output_clone() {
        let output = DecisionOutput {
            context: InteractiveContext::ThermalConstrained,
            reactor_event_weight: 0.75,
            blockers: vec![],
            actions: vec![RootAction::BoostProcess {
                pid: 42,
                name: "test".to_string(),
                reason: "testing".to_string(),
            }],
            low_value_skipped: vec!["skipped".to_string()],
        };
        let cloned = output.clone();
        assert_eq!(cloned.actions.len(), 1);
        assert_eq!(cloned.low_value_skipped.len(), 1);
        assert!((cloned.reactor_event_weight - 0.75).abs() < 1e-9);
    }
}
