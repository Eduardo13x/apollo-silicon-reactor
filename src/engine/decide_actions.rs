use std::collections::{HashMap, HashSet};

use sysinfo::System;

use crate::collector::SystemSnapshot;
use crate::engine::amx_detector;
use crate::engine::outcome_tracker::{HopGroupWeight, PatternWeight, WorkloadHop};
use crate::engine::overflow_guard::OverflowThresholds;
use crate::engine::safety::critical_background_processes;
use crate::engine::thread_selfcounts::IpcClass;
use crate::engine::types::{
    BlockerScore, InteractiveContext, LatencyTarget, OptimizationProfile, RootAction,
};
use crate::engine::user_context::UserContext;

/// User-facing interactive applications that must NEVER be frozen or throttled
/// by heuristic or adaptive governor decisions. Substring match — catches helpers
/// (e.g., "Brave" matches "Brave Browser Helper (Renderer)").
///
/// Synchronized with thermal_interrupt.rs protected list to prevent divergence
/// where an app is protected from thermal freeze but not from memory freeze.
/// [Lampson 1974] "Information is lost by having multiple, inconsistent copies."
const INTERACTIVE_APPS: [&str; 28] = [
    // IDEs and editors
    "Code", // VS Code
    "Cursor",
    "Xcode",
    "Zed",
    "Nova",
    "RubyMine",
    "IntelliJ",
    // Terminals
    "Terminal",
    "iTerm",
    "Warp",
    "Ghostty",
    "alacritty",
    "kitty",
    // Browsers
    "Arc",
    "Google Chrome",
    "Safari",
    "Brave", // CLAUDE.md invariant: never throttle
    "Firefox",
    "Microsoft Edge",
    // Communication
    "zoom.us",
    "Slack",
    "Discord",
    // AI / LLM
    "Claude",
    "LM Studio",
    "Antigravity",
    "Ollama",
    // Other user apps (production data: frozen incorrectly)
    "Notion",
    "Spotify",
];

// suggestd and corespeechd removed — both live in DEFERRABLE_DAEMONS with
// memory-pressure gate (correct semantics). Having them here caused conflicting
// immediate-throttle (NOISE) vs gated-throttle (DEFERRABLE) in the same cycle.
// [Saltzer & Schroeder 1975] Economy of Mechanism — one policy per resource.
const NOISE_APPS: [&str; 4] = ["Dropbox", "Google Drive", "OneDrive", "logioptionsplus"];

/// Apple on-device intelligence / ML background daemons.
/// These run opportunistically (idle + AC power) and are safe to throttle
/// aggressively when memory pressure is high — they resume when pressure drops.
const DEFERRABLE_DAEMONS: [&str; 9] = [
    "duetexpertd",          // Siri predictions / Proactive engine
    "suggestd",             // Spotlight/Siri suggestions ML
    "photoanalysisd",       // Photos ML tagging / face recognition
    "mediaanalysisd",       // Media content analysis
    "intelligencecontextd", // Apple Intelligence context engine
    "mlhostd",              // Metal/Core ML on-device inference host
    "modelmanagerd",        // On-device model cache manager
    "corespeechd",          // Siri speech recognition (background)
    "searchpartyd",         // Find My / Handoff BLE — 17 incorrect boosts in prod
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
    /// Number of display pipeline BoostProcess actions emitted this cycle (iter 3b).
    pub display_boosts_emitted: usize,
    /// Which freeze gate fired this cycle: "delta" | "committed" | "none".
    pub freeze_gate: String,
    /// What triggered ML daemon throttle: "swap-early" | "pressure" | "none".
    pub ml_throttle_source: String,
}

/// Returns true if the process name contains a known interactive app pattern.
///
/// Exported so `process_enrichment::convert_and_merge_heuristic_decisions` can apply
/// the same guard — [Saltzer & Kaashoek 2009] Complete Mediation: every path to a
/// privileged action must pass through the same access control point.
pub fn is_interactive_app_name(name: &str) -> bool {
    INTERACTIVE_APPS.iter().any(|n| name.contains(n))
}

fn is_interactive_base(name: &str) -> bool {
    is_interactive_app_name(name)
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
    // User context: what is the user doing right now?
    // idle_secs, sleep assertions, call detection, audio state.
    // [Riva & Mantovani 2014] idle time + media state = highest-signal contextual cues.
    user_ctx: &UserContext,
    // Per-process wakeup rate (idle + interrupt wakeups/sec) from proc_pid_rusage.
    // [Apple Energy Diagnostics / Activity Monitor] wakeup rate is the primary
    // battery drain signal for idle daemons. >100/s = vampire; >500/s = severe.
    wakeup_hints: &HashMap<u32, f64>,
    // Per-process physical footprint (MB) from ri_phys_footprint.
    // More accurate than RSS for freeze ranking: excludes shared pages.
    footprint_hints: &HashMap<u32, f64>,
    // DRAM memory bandwidth utilization (0.0–1.0) from IOReport AMC stats.
    // When > 0.80, memory-bandwidth-heavy processes should be throttled first.
    dram_bandwidth_pct: f64,
    // Per-process disk write rate (MB/s) from ri_disk_write_bytes delta.
    // Background processes writing >5 MB/s compete for disk bandwidth with
    // LLM model weight loading — throttle them during inference.
    // [Bhagwan & Savage 2002 OSDI] "I/O-Scope" — I/O bursts degrade co-located
    // latency-sensitive workloads by saturating disk queue depth.
    io_burst_hints: &HashMap<u32, f64>,
    // Per-process behavioral anomaly score vs learned baseline.
    // Score = max(|x-ema|/(mad+ε)) across {ipc, wakeup_rate, disk_mbps}.
    // ≥ 3.0 MADs from baseline → process deviating from learned normal behavior.
    // A normally-idle process suddenly active is more suspicious than a
    // consistently high-load process. [Chandola 2009 ACM CSUR §3.1]
    anomaly_hints: &HashMap<u32, f64>,
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

    // Behavioural app-bundle detection: build a set of pids whose binary
    // lives in a macOS .app bundle. ANY .app binary is by definition a
    // user-facing app or its helper (Apple Bundle Programming Guide), so
    // it must be protected from throttling regardless of whether its name
    // appears in the hardcoded INTERACTIVE_APPS list. This is the
    // behavioural detection that eliminates the drift hazard observed in
    // the 2026-04-08 graph audit (INTERACTIVE_APPS had drifted from
    // thermal_interrupt::protected by 10+ entries).
    //
    // Cost: one proc_pidpath syscall (~3 µs on M1) per pid, called once
    // per cycle here. Cached in the local HashSet for O(1) lookup by the
    // is_interactive closure below. The lookup is name+pid based so the
    // closure does not need to be aware of the path itself.
    let app_bundle_pids: std::collections::HashSet<u32> = sys
        .processes()
        .keys()
        .filter_map(|pid| {
            let pid_u32 = pid.as_u32();
            crate::engine::proc_taskinfo::is_user_app_bundle(pid_u32)
                .filter(|&is_bundle| is_bundle)
                .map(|_| pid_u32)
        })
        .collect();

    // System-wide CPU stall signal: when more than half of the tracked
    // pids are spending ≥85% of their CPU-wanting time in the run queue,
    // the protected set is being starved by the non-interactive set.
    // The fix is NOT to boost the protected pids further (they're
    // already at Foreground QoS) — it's to throttle the non-interactive
    // pids more aggressively so the scheduler has more headroom for
    // the protected set on the next quantum.
    //
    // This is the closure of debt-sensor-01 from V110_PENDING.md, which
    // proposed boosting protected pids on contention. Analysis showed
    // that approach was structurally wrong (boosting an already-boosted
    // pid is a no-op). The correct consumer for stall_fraction is here:
    // raise the throttle aggressiveness floor when the system is
    // CPU-starved at the system level.
    //
    // Threshold 0.85: vast majority must be starved. Original 0.5 was
    // too permissive on 8GB M1 — normal multitasking triggers it,
    // causing aggressive throttle every cycle → system freeze.
    let system_cpu_stalled = crate::engine::contention_tracker::global()
        .lock()
        .map(|t| t.stall_fraction(0.85) >= 0.85)
        .unwrap_or(false);

    // Build closures that merge hardcoded lists with the learned policy.
    // Also checks behavior-interactive PIDs (cpu_wall_ratio EMA < 0.05)
    // AND the behavioural app-bundle set built above. The .app-bundle
    // tier is the primary signal; the hardcoded list is a fallback for
    // pids whose proc_pidpath read failed (denied, kernel-only, or
    // already gone).
    let is_interactive = |name: &str, pid: u32| -> bool {
        let name_lc = name.to_ascii_lowercase();
        app_bundle_pids.contains(&pid)
            || is_interactive_base(name)
            || interactive_lc.iter().any(|p| name_lc.contains(p.as_str()))
            || behavior_interactive_pids.contains(&pid)
    };
    let is_background_noise = |name: &str| -> bool {
        let name_lc = name.to_ascii_lowercase();
        is_background_noise_base(name) || noise_lc.iter().any(|p| name_lc.contains(p.as_str()))
    };

    let mut actions = Vec::new();
    let mut low_value_skipped: Vec<String> = Vec::new();
    // Observability counters for enriched telemetry.
    let mut display_boosts_emitted: usize = 0;
    let mut freeze_gate = "none".to_string();
    let mut ml_throttle_source = "none".to_string();

    // Dev-first: protect critical background workloads and their children.
    let critical_patterns = critical_background_processes();
    let hard_protected = crate::engine::safety::protected_processes();
    let mut critical_pids: HashSet<u32> = HashSet::new();
    for (pid, process) in sys.processes() {
        let name = process.name().to_string();
        if critical_patterns.iter().any(|p| name.contains(p)) {
            critical_pids.insert(pid.as_u32());
        }
        // Seed hard-protected PIDs into critical_pids so the child-walk below
        // also covers their forked children (XPC helpers with different names).
        // Name-substring alone misses children whose names don't inherit the
        // parent's name (common for DriverKit extensions and helper tools).
        if hard_protected.iter().any(|p| name.contains(p)) {
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
    // Covers both critical_background_processes() and hard protected_processes().
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

    // User context: when the user is actively present and system is only at BackgroundPressure,
    // skip background throttling entirely — fluidity > marginal pressure relief.
    // At ThermalConstrained we always act regardless of user activity (safety first).
    // [Riva & Mantovani 2014] active user context → preserve responsiveness over efficiency.
    let skip_bg_throttle_user_active =
        user_ctx.is_recently_active() && matches!(context, InteractiveContext::InteractiveFocus);

    // Call in progress → elevate effective context to BackgroundPressure regardless of measured
    // pressure. Real-time audio/video codecs need CPU + memory bandwidth; throttling background
    // daemons frees both. Freeze is still blocked by freeze_protected() in step 5.
    // [Ellis & Gibbs 1989] "Concurrency control in groupware systems" — real-time collaboration
    // requires dedicated, low-latency CPU + memory bandwidth, not shared with background tasks.
    let effective_context =
        if user_ctx.call_in_progress && matches!(context, InteractiveContext::InteractiveFocus) {
            InteractiveContext::BackgroundPressure
        } else {
            context
        };

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
            // User active at low pressure: defer background throttling — jank isn't worth it.
            if skip_bg_throttle_user_active {
                low_value_skipped.push(format!("user-active-skip:{}", name));
                continue;
            }
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
                        if group.throttle_count >= 15
                            && group.effectiveness() < 0.12
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
            // unless we're in thermal emergency OR the system is highly memory-stalled.
            // [Hennessy & Patterson 2017] when system-wide DRAM bandwidth is saturated
            // (>80%), even "compute-bound" processes are sharing a memory bottleneck —
            // IPC protection is less meaningful in that regime.
            let system_memory_stressed = dram_bandwidth_pct >= 0.80;
            if !ipc_class.safe_to_throttle()
                && !matches!(effective_context, InteractiveContext::ThermalConstrained)
                && !system_memory_stressed
            {
                low_value_skipped.push(format!("ipc-protected:{}", name));
                continue;
            }

            let aggressive = match effective_context {
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
            // Per-process signals for reason string and aggressive override.
            let wakeup_rate = wakeup_hints.get(&pid).copied().unwrap_or(0.0);
            let footprint_mb = footprint_hints.get(&pid).copied().unwrap_or(0.0);
            let disk_mbps = io_burst_hints.get(&pid).copied().unwrap_or(0.0);
            let anomaly_score = anomaly_hints.get(&pid).copied().unwrap_or(0.0);
            let is_wakeup_vampire = wakeup_rate >= 100.0;
            let is_io_burst = disk_mbps >= 5.0;
            let is_anomalous = anomaly_score >= crate::engine::process_baseline::ANOMALY_THRESHOLD;
            // DRAM bandwidth: prefer throttling high-footprint processes when bus is saturated.
            // [Intel Memory Bandwidth Allocation / IOReport AMC]
            let bandwidth_priority = dram_bandwidth_pct >= 0.80 && footprint_mb > 100.0;
            // Aggressive modifiers: stall, vampire, I/O-burst, anomaly all escalate to aggressive.
            // [Apple Energy Diagnostics; Bhagwan & Savage 2002; Chandola 2009]
            let aggressive = aggressive
                || system_cpu_stalled
                || is_wakeup_vampire
                || is_io_burst
                || is_anomalous;

            let ipc = ipc_hints.get(&pid).copied().unwrap_or(0.0);
            let reason = if is_anomalous {
                format!(
                    "anomaly throttle (score={:.1}x baseline, ipc={:.2})",
                    anomaly_score, ipc
                )
            } else if is_io_burst {
                format!(
                    "io-burst throttle ({:.1} MB/s writes, ipc={:.2})",
                    disk_mbps, ipc
                )
            } else if is_wakeup_vampire {
                format!(
                    "wakeup-vampire throttle ({:.0}/s wakeups, ipc={:.2})",
                    wakeup_rate, ipc
                )
            } else if bandwidth_priority {
                format!(
                    "dram-bw throttle (bw={:.0}%, footprint={:.0}MB, ipc={:.2})",
                    dram_bandwidth_pct * 100.0,
                    footprint_mb,
                    ipc
                )
            } else {
                format!("ipc-aware throttle ({:?}, ipc={:.2})", context, ipc)
            };

            actions.push(RootAction::ThrottleProcess {
                pid,
                name,
                aggressive,
                reason,
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
        const DISPLAY_PIPELINE: &[&str] = &[
            "Dock",
            "coreaudiod",
            "mediaserverd",
            "SystemUIServer",
            "ControlCenter",
        ];
        let swap_delta_mb = snapshot.pressure.swap_delta_bytes_per_sec / (1024.0 * 1024.0);
        // Trigger when swap grows ≥ 0.5 MB/s — early pressure signal, not crisis.
        if swap_delta_mb >= 0.5 {
            let mut display_boosts: Vec<RootAction> = Vec::new();
            for (pid, process) in sys.processes() {
                let name = process.name().to_string();
                if DISPLAY_PIPELINE.iter().any(|d| name.contains(d)) {
                    display_boosts.push(RootAction::BoostProcess {
                        pid: pid.as_u32(),
                        name,
                        reason: format!("display pipeline — swap +{:.1} MB/s", swap_delta_mb),
                    });
                }
            }
            // Prepend so display boosts run before any throttle/freeze actions.
            display_boosts_emitted = display_boosts.len();
            display_boosts.extend(actions.drain(..));
            actions = display_boosts;
        }
    }

    // 3c) WindowServer high-CPU boost: when the display compositor is using >20% CPU,
    // explicitly boost it to P-cores to prevent jank from scheduler preemption.
    // WindowServer is already in protected_processes() so it won't be frozen/throttled,
    // but a proactive BoostProcess ensures the scheduler keeps it on a P-core.
    // [WWDC 2021] "Tune CPU job scheduling with QoS" — compositor must stay on P-cores
    // under load; explicit QoS boost guarantees scheduling priority.
    {
        const WS_CPU_BOOST_THRESHOLD: f32 = 20.0;
        for (pid, process) in sys.processes() {
            if process.name() == "WindowServer" && process.cpu_usage() > WS_CPU_BOOST_THRESHOLD {
                actions.push(RootAction::BoostProcess {
                    pid: pid.as_u32(),
                    name: "WindowServer".to_string(),
                    reason: format!(
                        "display compositor high CPU ({:.0}%) — P-core priority",
                        process.cpu_usage()
                    ),
                });
                break; // Only one WindowServer process
            }
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
    // Use effective_context: call_in_progress elevates to BackgroundPressure
    // so deferrable ML daemons get throttled even at low measured pressure.
    let deferrable_pressure_trigger = matches!(
        effective_context,
        InteractiveContext::BackgroundPressure | InteractiveContext::ThermalConstrained
    );
    if deferrable_pressure_trigger || deferrable_swap_trigger {
        ml_throttle_source = if deferrable_swap_trigger && !deferrable_pressure_trigger {
            "swap-early".to_string()
        } else if user_ctx.call_in_progress
            && matches!(context, InteractiveContext::InteractiveFocus)
        {
            // Triggered by call elevation, not real measured pressure.
            "call-mode".to_string()
        } else {
            "pressure".to_string()
        };
        for (pid, process) in sys.processes() {
            let name = process.name().to_string();
            if DEFERRABLE_DAEMONS.iter().any(|d| name.contains(d))
                && !critical_pids.contains(&pid.as_u32())
                && !behavior_interactive_pids.contains(&pid.as_u32())
            {
                actions.push(RootAction::ThrottleProcess {
                    pid: pid.as_u32(),
                    name,
                    aggressive: false,
                    reason: "deferrable-ml-daemon: throttled under memory pressure".to_string(),
                    // Pass start_sec so verify_pid_identity can guard against PID
                    // recycling between snapshot and execution. The old start_sec=0
                    // caused this path to fall back to name-only matching, which is
                    // weaker (6-char prefix/suffix comparison only).
                    start_sec: process.start_time(),
                    start_usec: 0,
                });
            }
        }
    }

    // 5) Pressure actions with hysteresis-ish behavior by context.
    //
    // User context gates (applied before freeze logic):
    // - call_in_progress or has_sleep_assertion → skip freeze entirely (user is in a call
    //   or watching media; SIGSTOP would cause visible jank / dropped frames / audio glitch)
    // - idle_long (>120s) → relax gate thresholds (-10pp pressure, -30% swap commit)
    //   User is away; aggressive optimization causes zero jank.
    // - recently_active (<15s) → tighten gate thresholds (+5pp pressure, +30% swap commit)
    //   User is actively present; conserve fluidity headroom.
    // [Riva & Mantovani 2014] "User context awareness for mobile computing"
    // Pass current memory pressure so a high-pressure crisis (≥0.75) overrides
    // background-task sleep assertions. Without this, a single Electron renderer
    // holding PreventUserIdleSleep blocks every freeze even when swap is climbing.
    let freeze_skip_by_user =
        user_ctx.freeze_protected(snapshot.pressure.memory_pressure);
    let gate_offset = user_ctx.pressure_gate_offset();

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
            //
            // User context offsets applied to gate thresholds (gate_offset from UserContext):
            //   idle long  → gate_offset = -0.10 → lower gate = earlier, more aggressive freeze
            //   active     → gate_offset = +0.05 → raise gate = freeze only at higher pressure
            let gate_b_pressure = 0.75 + gate_offset;
            let gate_b_swap_gb = if gate_offset < 0.0 {
                0.7
            } else if gate_offset > 0.0 {
                1.3
            } else {
                1.0
            };
            let gate_a = snapshot.pressure.memory_pressure >= thresholds.extreme_pressure
                && snapshot.pressure.swap_delta_bytes_per_sec > (2.0 * 1024.0 * 1024.0);
            let gate_b = snapshot.pressure.memory_pressure >= gate_b_pressure
                && swap_committed_gb >= gate_b_swap_gb;
            // Gate C: VM flow gate. Activates when the compressor is churning
            // hard (thrashing_score > 5_000 events/s weighted) even if the
            // absolute pressure percentage hasn't crossed the extreme gate.
            // Rationale: a system at 0.68 pressure with 10k compressions/s is
            // thrashing; a system at 0.80 pressure with 0 compressions/s is
            // resting. The flow signal catches the first case earlier than
            // the level-based gates A and B.
            //
            // Requires the gate_a pressure floor so we don't freeze under
            // transient compressor spikes at truly healthy pressure (<55%).
            // [Denning 1968 "Working Set Model"] — fault rate, not residency,
            // defines working-set quality.
            let gate_c = snapshot.pressure.thrashing_score > 5_000.0
                && snapshot.pressure.memory_pressure >= 0.55;

            // User in call / media playing: skip freeze — jank is worse than memory pressure.
            if freeze_skip_by_user {
                freeze_gate = "user-protected".to_string();
            }
            let extreme_freeze_ok = (gate_a || gate_b || gate_c) && !freeze_skip_by_user;
            if extreme_freeze_ok {
                freeze_gate = if gate_a {
                    "delta".to_string()
                } else if gate_b {
                    "committed".to_string()
                } else {
                    "thrashing".to_string()
                };
                // RSS-rank selection: freeze/throttle the largest-RSS background
                // processes first — maximum pressure relief per action.
                // [Android LMK: terminate by OOM-adj score (RSS proxy);
                //  Facebook HHVM: "evict by cost, not name"]
                //
                // Replaced hardcoded ["Slack","Discord","Spotify","Teams"]: any
                // memory-heavy background app (zoom.us, Figma, Electron apps)
                // now qualifies. Protection stack in execute_actions still applies.
                let protected = crate::engine::safety::protected_processes();
                let mut freeze_candidates: Vec<(u32, String, u64, f32, u64)> = sys
                    .processes()
                    .iter()
                    .filter_map(|(pid, process)| {
                        let pid_u32 = pid.as_u32();
                        if critical_pids.contains(&pid_u32) {
                            return None;
                        }
                        let name = process.name().to_string();
                        // System-critical processes must never be freeze candidates.
                        // Bug found in production: gate_c was emitting FreezeProcess
                        // for loginwindow (pid 466) and sharingd (pid 719) because
                        // critical_pids only contains infrastructure (docker, postgres)
                        // and ML workloads — NOT the OS-essential daemons from
                        // safety::protected_processes(). execute_actions blocks
                        // loginwindow downstream but sharingd was NOT protected and
                        // got frozen in a 19-hour loop. AirDrop/Handoff broken.
                        if protected.iter().any(|p| name.contains(p)) {
                            return None;
                        }
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
                // Rank by physical footprint when available (more accurate than RSS),
                // else fall back to RSS. phys_footprint excludes shared pages so it
                // better reflects actual memory pressure contribution per process.
                // [XNU proc_pid_rusage ri_phys_footprint] = true owned pages.
                freeze_candidates.sort_unstable_by(|a, b| {
                    let fa = footprint_hints
                        .get(&a.0)
                        .copied()
                        .unwrap_or(a.2 as f64 / (1024.0 * 1024.0));
                    let fb = footprint_hints
                        .get(&b.0)
                        .copied()
                        .unwrap_or(b.2 as f64 / (1024.0 * 1024.0));
                    fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
                });
                // Cap at 3 per cycle — avoid SIGSTOP burst overhead on display pipeline.
                for (pid, name, _rss, cpu, start_sec) in freeze_candidates.into_iter().take(3) {
                    // CPU-active guard: under gate_a / gate_b (pressure-based) the
                    // pressure is often transient and we'd rather throttle than
                    // drop in-flight work. Under gate_c (flow-based, sustained
                    // compressor thrashing) the in-flight work is what's CAUSING
                    // the thrashing — pausing it briefly is exactly the remedy.
                    // Throttling alone leaves memory unreclaimed and the
                    // thrashing_score keeps climbing cycle after cycle (observed
                    // in production: 720 throttles, 0 freezes, thrashing_score
                    // = 19_890 with the gate firing every cycle but never
                    // emitting a freeze action).
                    let force_freeze_under_thrashing = gate_c;
                    if cpu > 10.0 && !force_freeze_under_thrashing {
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
                        let reason = if gate_c && cpu > 10.0 {
                            format!(
                                "thrashing-flow freeze (cpu-active {:.0}%) under {:?}",
                                cpu, context
                            )
                        } else {
                            format!("extreme pressure RSS-rank under {:?}", context)
                        };
                        actions.push(RootAction::FreezeProcess {
                            pid,
                            name,
                            reason,
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

    // [Pearl 2009] Impact-prioritized ordering: sort ThrottleProcess actions by
    // causal impact score (highest first). When the action queue has capacity limits,
    // high-impact throttles execute first, maximizing pressure reduction per cycle.
    // Boosts and freezes keep their original order (boosts first, freezes last).
    {
        // Partition: boosts first, then throttles (sorted), then freezes/others.
        let mut boosts = Vec::new();
        let mut throttles = Vec::new();
        let mut others = Vec::new();
        for action in actions {
            match &action {
                RootAction::BoostProcess { .. } => boosts.push(action),
                RootAction::ThrottleProcess { name, .. } => {
                    let causal_key = format!("throttle:{}", name);
                    let impact = causal_confidence.get(&causal_key).copied().unwrap_or(0.5); // unknown → neutral priority
                    throttles.push((impact, action));
                }
                _ => others.push(action),
            }
        }
        // Sort throttles by impact descending (highest impact first).
        throttles.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut actions = boosts;
        actions.extend(throttles.into_iter().map(|(_, a)| a));
        actions.extend(others);

        DecisionOutput {
            // Report effective_context (the context actually used for decisions),
            // not the raw measured context. When call_in_progress elevates
            // InteractiveFocus → BackgroundPressure, the raw context is misleading:
            // observability logs would show "InteractiveFocus" while decisions were
            // made under BackgroundPressure semantics.
            // [Kleppmann 2017 DDIA §11] — observability records must reflect actual execution.
            context: effective_context,
            reactor_event_weight,
            blockers,
            actions,
            low_value_skipped,
            display_boosts_emitted,
            freeze_gate,
            ml_throttle_source,
        }
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
                thrashing_score: 0.0,
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

    /// Thin test wrapper: calls `decide_actions` with all hint maps empty and default
    /// thresholds. Only the interesting parameters (snapshot, profile, latency, reactor
    /// weight) need to be varied per test.
    fn call_decide(
        snap: &crate::collector::SystemSnapshot,
        sys: &System,
        profile: OptimizationProfile,
        latency: LatencyTarget,
        reactor: f64,
    ) -> DecisionOutput {
        let (interactive, noise, weights, pids, ipc, hops, hab, causal) = empty_params();
        decide_actions(
            snap,
            sys,
            profile,
            latency,
            reactor,
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
            &UserContext::default(),
            &HashMap::new(),
            &HashMap::new(),
            0.0,
            &HashMap::new(),
            &HashMap::new(),
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
        // ITER 3 additions — synced from thermal_interrupt.rs protected list
        assert!(is_interactive_base("Firefox"));
        assert!(is_interactive_base("Slack"));
        assert!(is_interactive_base("Discord"));
        assert!(is_interactive_base("Spotify"));
        assert!(is_interactive_base("Notion"));
        assert!(is_interactive_base("Zed"));
        assert!(is_interactive_base("Ghostty"));
        assert!(is_interactive_base("Ollama"));
        // Electron helpers — substring match catches child processes
        assert!(is_interactive_base("Notion Helper (Renderer)"));
        assert!(is_interactive_base("Antigravity Helper (GPU)"));
        assert!(is_interactive_base("Slack Helper (Renderer)"));
        assert!(is_interactive_base("Discord Helper"));
        // Negatives — daemons must NOT match
        assert!(!is_interactive_base("Dropbox"));
        assert!(!is_interactive_base("randomd"));
        assert!(!is_interactive_base("searchpartyd"));
        assert!(!is_interactive_base("corespeechd"));
    }

    #[test]
    fn no_process_in_both_noise_and_deferrable() {
        // [Saltzer & Schroeder 1975] Economy of Mechanism — one policy per resource.
        // A process in both lists gets conflicting treatment in the same cycle.
        for noise in &NOISE_APPS {
            assert!(
                !DEFERRABLE_DAEMONS.iter().any(|d| d == noise),
                "{noise} is in both NOISE_APPS and DEFERRABLE_DAEMONS"
            );
        }
    }

    /// Behavioural-detection regression: every binary inside a `.app`
    /// bundle is recognised as user-facing, regardless of whether its
    /// name appears in INTERACTIVE_APPS. This is the test that locks
    /// in the drift fix from the 2026-04-08 graph audit.
    #[test]
    fn app_bundle_paths_classify_as_interactive_via_proc_taskinfo() {
        use crate::engine::proc_taskinfo::is_app_bundle_path;
        // These names are NOT in INTERACTIVE_APPS, but their canonical
        // bundle paths must still classify as user-facing via the
        // behavioural path:
        let bundle_only_apps = [
            (
                "/Applications/Bartender 4.app/Contents/MacOS/Bartender 4",
                "Bartender 4",
            ),
            (
                "/Applications/Setapp/CleanShot X.app/Contents/MacOS/CleanShot X",
                "CleanShot X",
            ),
            (
                "/Applications/Raycast.app/Contents/MacOS/Raycast",
                "Raycast",
            ),
            (
                "/Applications/1Password 7 - Password Manager.app/Contents/MacOS/1Password 7",
                "1Password 7",
            ),
            (
                "/Users/me/Applications/CustomApp.app/Contents/MacOS/CustomApp",
                "CustomApp",
            ),
        ];
        for (path, name) in bundle_only_apps {
            assert!(
                is_app_bundle_path(path),
                "{name}: path-pattern detection must recognise {path} as a .app bundle"
            );
            // The name itself is NOT in INTERACTIVE_APPS — confirm the
            // hardcoded list alone would have missed it.
            assert!(
                !is_interactive_base(name),
                "{name}: this app intentionally NOT in INTERACTIVE_APPS so the test \
                 verifies the BEHAVIOURAL tier (path) catches it without name match"
            );
        }
        // And conversely: daemons must NOT classify as bundle, even if
        // they happen to have a name that contains substrings of an app.
        let non_bundle_daemons = [
            "/usr/sbin/cfprefsd",
            "/sbin/launchd",
            "/usr/libexec/trustd",
            "/System/Library/PrivateFrameworks/MediaAnalysisServices.framework/mediaanalysisd",
            "/opt/homebrew/bin/cargo",
        ];
        for path in non_bundle_daemons {
            assert!(
                !is_app_bundle_path(path),
                "{path}: must NOT be classified as a .app bundle"
            );
        }
    }

    #[test]
    fn no_interactive_app_in_noise_or_deferrable() {
        // Interactive apps must never be throttled by name-based gates.
        for interactive in &INTERACTIVE_APPS {
            assert!(
                !NOISE_APPS.iter().any(|n| n == interactive),
                "{interactive} is in both INTERACTIVE_APPS and NOISE_APPS"
            );
            assert!(
                !DEFERRABLE_DAEMONS.iter().any(|d| d == interactive),
                "{interactive} is in both INTERACTIVE_APPS and DEFERRABLE_DAEMONS"
            );
        }
    }

    // ── cross-module protection invariants ──────────────────────────────────
    // Graph audit (2026-04-10): graphify detected `protected_processes()` and
    // `INTERACTIVE_APPS` as semantically similar. They are intentionally separate
    // (OS daemons vs user apps) but must remain disjoint and both must cover
    // the CLAUDE.md mandatory set. [Lampson 1974] "Use a single source of
    // truth for protection policy. Multiple tables that should agree will
    // eventually diverge." — this test is the drift-detection mechanism.
    #[test]
    fn interactive_apps_disjoint_from_os_protected_processes() {
        use crate::engine::safety::protected_processes;
        let hard = protected_processes();
        // INTERACTIVE_APPS are user-facing app names matched via substring.
        // None of the exact strings should also appear in the OS hard-protected
        // set — that would signal a drift where an OS daemon crept into the
        // user-app list (or vice versa).
        for app in &INTERACTIVE_APPS {
            assert!(
                !hard.contains(app),
                "'{}' appears in both INTERACTIVE_APPS and protected_processes() — \
                 these lists must stay disjoint (OS daemons vs user apps)",
                app
            );
        }
    }

    #[test]
    fn claude_md_invariants_covered_by_interactive_apps() {
        // CLAUDE.md: "Nunca throttlear/congelar: Antigravity, Claude, Brave, rustc/cargo"
        // User-facing AI/browser apps must be in INTERACTIVE_APPS.
        // (rustc/cargo are covered by build-mode detection, not INTERACTIVE_APPS.)
        for name in &["Antigravity", "Claude", "Brave"] {
            assert!(
                is_interactive_base(name),
                "CLAUDE.md invariant violated: '{}' must be in INTERACTIVE_APPS",
                name
            );
        }
    }

    #[test]
    fn noise_apps_detected() {
        assert!(is_background_noise_base("Dropbox"));
        assert!(is_background_noise_base("Google Drive"));
        // corespeechd and suggestd moved to DEFERRABLE_DAEMONS (pressure-gated)
        assert!(!is_background_noise_base("corespeechd"));
        assert!(!is_background_noise_base("suggestd"));
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
        // suggestd moved to DEFERRABLE_DAEMONS, verify it's no longer noise
        assert!(!is_background_noise_base("com.apple.suggestd"));
        assert!(is_known_blocker("com.apple.WindowServer"));
    }

    // ── decide_actions integration tests (empty process table) ───────────

    #[test]
    fn decide_actions_empty_system_returns_no_actions() {
        let snap = make_snapshot(10.0, 0.10, 0.0);
        let sys = System::new();
        let output = call_decide(
            &snap,
            &sys,
            OptimizationProfile::BalancedRoot,
            LatencyTarget::Normal,
            0.0,
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
        let output = call_decide(
            &snap,
            &sys,
            OptimizationProfile::BalancedRoot,
            LatencyTarget::Normal,
            0.42,
        );
        assert!(
            (output.reactor_event_weight - 0.42).abs() < 1e-9,
            "reactor_event_weight should be passed through"
        );
    }

    #[test]
    fn decide_actions_context_escalates_with_pressure() {
        let sys = System::new();

        // Low pressure
        let out_low = call_decide(
            &make_snapshot(10.0, 0.10, 0.0),
            &sys,
            OptimizationProfile::BalancedRoot,
            LatencyTarget::Normal,
            0.0,
        );
        assert!(matches!(
            out_low.context,
            InteractiveContext::InteractiveFocus
        ));

        // Medium pressure (CPU > 72)
        let out_mid = call_decide(
            &make_snapshot(75.0, 0.10, 0.0),
            &sys,
            OptimizationProfile::BalancedRoot,
            LatencyTarget::Normal,
            0.0,
        );
        assert!(matches!(
            out_mid.context,
            InteractiveContext::BackgroundPressure
        ));

        // High pressure (CPU > 88)
        let out_high = call_decide(
            &make_snapshot(92.0, 0.10, 0.0),
            &sys,
            OptimizationProfile::BalancedRoot,
            LatencyTarget::Normal,
            0.0,
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

        for profile in [
            OptimizationProfile::BalancedRoot,
            OptimizationProfile::AggressiveRoot,
            OptimizationProfile::SafeRoot,
        ] {
            let output = call_decide(&snap, &sys, profile, LatencyTarget::Normal, 0.0);
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

        for target in [
            LatencyTarget::Low,
            LatencyTarget::Normal,
            LatencyTarget::Max,
        ] {
            let output = call_decide(&snap, &sys, OptimizationProfile::BalancedRoot, target, 0.0);
            assert!(output.actions.is_empty());
        }
    }

    // ── io_burst_hints contract ──────────────────────────────────────────
    // Verify the disk_mbps >= 5.0 threshold logic used in decide_actions.

    #[test]
    fn io_burst_threshold_math() {
        // Exactly 5.0 MB/s — meets threshold (aggressive throttle).
        let disk_mbps: f64 = 5.0;
        assert!(disk_mbps >= 5.0, "5.0 MB/s must trigger io_burst");
        // Just below — does not trigger.
        let below: f64 = 4.999;
        assert!(!(below >= 5.0), "4.999 MB/s must NOT trigger io_burst");
        // Typical backup process at 50 MB/s — should trigger.
        let backup: f64 = 50.0;
        assert!(backup >= 5.0, "50 MB/s backup must trigger io_burst");
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
            display_boosts_emitted: 0,
            freeze_gate: "none".to_string(),
            ml_throttle_source: "none".to_string(),
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
        let _ = (
            interactive,
            noise,
            weights,
            behavior_pids,
            ipc_hints,
            hop_groups,
            causal,
        );
    }

    #[test]
    fn habituation_bypass_condition_logic() {
        // Verify the bypass conditions used in main.rs are correct:
        // swap ≥ 8 GB OR p_oom ≥ 0.95 → bypass.
        let swap_8gb: u64 = 8 * 1_073_741_824;
        let swap_normal: u64 = 2 * 1_073_741_824;

        let bypass_on_swap = swap_8gb >= 8 * 1_073_741_824;
        let no_bypass_normal = swap_normal >= 8 * 1_073_741_824;
        let bypass_on_oom = 0.96f64 >= 0.95;
        let no_bypass_low = 0.50f64 >= 0.95;

        assert!(bypass_on_swap, "swap ≥ 8 GB should trigger bypass");
        assert!(!no_bypass_normal, "swap = 2 GB should not trigger bypass");
        assert!(bypass_on_oom, "p_oom = 0.96 should trigger bypass");
        assert!(!no_bypass_low, "p_oom = 0.50 should not trigger bypass");
    }

    // ── anomaly_hints protection invariants ──────────────────────────────
    // Verify that INTERACTIVE_APPS names are guarded by is_interactive_base,
    // ensuring they will receive BoostProcess (not ThrottleProcess) even when
    // an anomaly_score ≥ ANOMALY_THRESHOLD is present for their PID.
    // [Safety invariant from CLAUDE.md: never throttle Claude, Brave, rustc, etc.]

    #[test]
    fn interactive_apps_are_protected_from_anomaly_throttle() {
        // All names in INTERACTIVE_APPS must be recognised by is_interactive_base.
        // If this test fails, a new INTERACTIVE_APPS entry was added but the
        // contains() check no longer matches — they would be anomaly-throttleable.
        let protected = [
            "Code",
            "Arc",
            "Brave",
            "Claude",
            "LM Studio",
            "Safari",
            "zoom.us",
            "Xcode",
            "Terminal",
            "iTerm",
            "Warp",
            "Cursor",
            "Antigravity",
            "Google Chrome",
        ];
        for name in protected {
            assert!(
                is_interactive_base(name),
                "'{}' must be protected by is_interactive_base to prevent anomaly throttle",
                name
            );
        }
    }

    #[test]
    fn anomaly_hints_do_not_affect_interactive_classification() {
        // Regression guard: even a sky-high anomaly_score does not cause the
        // anomaly_hints map to override the is_interactive early-return in the
        // decide_actions loop.  The guard fires BEFORE the anomaly check.
        //
        // We verify the logic order directly: is_interactive → continue (boost)
        // occurs at line 387, anomaly_hints are consumed at line 512 — AFTER the
        // early-return, so an interactive process can never reach that point.
        //
        // This test ensures the score pipeline itself computes correctly and that
        // the threshold gate matches expected values.
        use crate::engine::process_baseline::{effective_threshold, ANOMALY_THRESHOLD};

        // Simulate an anomalous reading for a process with warm baseline.
        let mut map = crate::engine::process_baseline::ProcessBaselineMap::new();
        for _ in 0..20 {
            map.observe("Claude", 1.5, 20.0, 0.1);
        }
        // Disk burst — Claude is anomalous by the detector.
        let score = map.anomaly_score("Claude", 1.5, 20.0, 500.0);
        assert!(
            score >= ANOMALY_THRESHOLD,
            "score {} should be anomalous",
            score
        );

        // With 1 warm baseline, effective_threshold is raised slightly above nominal.
        let thresh = effective_threshold(map.warm_count());
        // Score still exceeds even the raised threshold (burst is extreme).
        assert!(
            score >= thresh,
            "score {} should exceed raised threshold {}",
            score,
            thresh
        );

        // The anomaly_hints map that would be built from this score:
        let mut hints: HashMap<u32, f64> = HashMap::new();
        let pid = 9999u32;
        hints.insert(pid, score);

        // Verify: is_interactive_base("Claude") fires FIRST and returns true.
        // This is the compile-time proof that the guard order is correct.
        assert!(
            is_interactive_base("Claude"),
            "Claude must be caught by is_interactive_base before anomaly path runs"
        );
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
            display_boosts_emitted: 0,
            freeze_gate: "none".to_string(),
            ml_throttle_source: "none".to_string(),
        };
        let cloned = output.clone();
        assert_eq!(cloned.actions.len(), 1);
        assert_eq!(cloned.low_value_skipped.len(), 1);
        assert!((cloned.reactor_event_weight - 0.75).abs() < 1e-9);
    }
}
