//! Process Enrichment — pure helper functions extracted from daemon monolith.
//!
//! Contains:
//! - `filter_boost_cooldown()` — dedup boost actions with per-PID cooldowns
//! - `apply_post_wake_grace_policy()` — suppress freeze/throttle during post-wake grace
//! - `context_to_thermal()` — interactive context → thermal string
//! - `append_discrepancy_log()` — log safety precedence overrides
//! - `build_foreground_family()` — compute foreground PID set from process tree
//! - `build_enriched_process_data_with_tree()` — build ProcessSnapshot + HuntSnapshot
//! - `convert_and_merge_heuristic_decisions()` — merge heuristic decisions into actions
//! - `HeuristicStats` — counters for heuristic action conversions
//! - `ThrashState` — per-PID cooldown tracking

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use apollo_optimizer::engine::adaptive_governor::{GovernorDecision, ProcessDecision};
use apollo_optimizer::engine::daemon_helpers::pid_start_time;
use apollo_optimizer::engine::daemon_helpers::rotate_timeline;
use apollo_optimizer::engine::decide_actions::is_interactive_app_name;
use apollo_optimizer::engine::llm::append_jsonl;
use apollo_optimizer::engine::proc_taskinfo;
use apollo_optimizer::engine::process_classifier::{ProcessSnapshot, ProcessTier};
use apollo_optimizer::engine::process_tree::ProcessTree;
use apollo_optimizer::engine::types::{InteractiveContext, RootAction, SafetyPolicy};
use apollo_optimizer::engine::zombie_hunter::HuntSnapshot;
use sysinfo::ProcessStatus;

// ── Types ──────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct ThrashState {
    pub minute_started: Option<Instant>,
    pub cooldowns: HashMap<u32, Instant>,
}

pub struct HeuristicStats {
    pub decisions_total: u64,
    pub throttles: u64,
    pub freezes: u64,
    pub kills_downgraded: u64,
    pub zombies_detected: u64,
}

// ── Action Filters ─────────────────────────────────────────────────────────

pub fn filter_boost_cooldown(
    actions: Vec<RootAction>,
    policy: &SafetyPolicy,
    thrash: &mut ThrashState,
) -> Vec<RootAction> {
    let now = Instant::now();
    let cooldown = Duration::from_secs(policy.cooldown_seconds);
    let mut out = Vec::new();

    thrash
        .cooldowns
        .retain(|_, ts| now.duration_since(*ts) <= Duration::from_secs(300));

    for action in actions {
        match &action {
            RootAction::BoostProcess { pid, .. } => {
                if let Some(last) = thrash.cooldowns.get(pid) {
                    if now.duration_since(*last) < cooldown {
                        continue;
                    }
                }
                thrash.cooldowns.insert(*pid, now);
                out.push(action);
            }
            _ => out.push(action),
        }
    }

    out
}

pub fn apply_post_wake_grace_policy(
    actions: Vec<RootAction>,
    grace_active: bool,
) -> (Vec<RootAction>, u64, u64) {
    if !grace_active {
        return (actions, 0, 0);
    }

    let mut out = Vec::with_capacity(actions.len());
    let mut freeze_suppressed = 0_u64;
    let mut throttle_suppressed = 0_u64;

    for action in actions {
        match action {
            RootAction::FreezeProcess { .. } | RootAction::QuarantineDaemon { .. } => {
                freeze_suppressed += 1;
            }
            RootAction::ThrottleProcess {
                pid,
                name,
                aggressive: true,
                reason,
                start_sec,
                start_usec,
            } => {
                throttle_suppressed += 1;
                out.push(RootAction::ThrottleProcess {
                    pid,
                    name,
                    aggressive: false,
                    reason,
                    start_sec,
                    start_usec,
                });
            }
            _ => out.push(action),
        }
    }

    (out, throttle_suppressed, freeze_suppressed)
}

// ── Helpers ────────────────────────────────────────────────────────────────

pub fn context_to_thermal(context: InteractiveContext) -> String {
    match context {
        InteractiveContext::ThermalConstrained => "constrained".to_string(),
        InteractiveContext::BackgroundPressure => "elevated".to_string(),
        InteractiveContext::InteractiveFocus => "nominal".to_string(),
    }
}

pub fn append_discrepancy_log(
    path: &std::path::Path,
    protected_app: &str,
    actions_removed: usize,
    workload: &str,
    confidence: f32,
    reason: &str,
) {
    let entry = serde_json::json!({
        "at": chrono::Utc::now().to_rfc3339(),
        "event": "safety_precedence_override",
        "protected_app": protected_app,
        "actions_removed": actions_removed,
        "ml_workload": workload,
        "ml_confidence": confidence,
        "reason": reason,
    });
    append_jsonl(path, &entry);
    rotate_timeline(path);
}

// ── Foreground Family ──────────────────────────────────────────────────────

/// Build the set of PIDs belonging to the foreground app group (parent + children).
pub fn build_foreground_family(foreground_pid: Option<u32>, tree: &ProcessTree) -> HashSet<u32> {
    foreground_pid
        .map(|pid| tree.cascade_pids(pid).into_iter().collect())
        .unwrap_or_default()
}

// ── Enriched Process Data ──────────────────────────────────────────────────

/// Tree-aware enriched process data builder.
///
/// Uses the foreground PID and process tree to determine foreground status for
/// each process. A process is "foreground" if:
///   1. It IS the foreground PID, or
///   2. It belongs to the same process tree app group as the foreground PID
///      (i.e., it is a child/grandchild of the foreground app).
///
/// This gives accurate foreground detection for multi-process apps like Chrome,
/// Electron, VS Code, etc. where the heuristic classifier previously missed
/// helper/renderer processes because they have different names.
pub fn build_enriched_process_data_with_tree(
    sys: &sysinfo::System,
    foreground_pid: Option<u32>,
    tree: &ProcessTree,
) -> (Vec<ProcessSnapshot>, Vec<HuntSnapshot>) {
    // Pre-compute the set of PIDs in the foreground family for O(1) lookups.
    let fg_family: HashSet<u32> = build_foreground_family(foreground_pid, tree);

    // Bulk-read idle_wakeups + Mach messages via proc_taskinfo (~1.3ms for ~400 pids).
    // This replaces the hardcoded wakeups_per_sec: 0.0 with REAL kernel data.
    // pid → (idle_wakeups, mach_msgs, faults, pageins)
    let mut rusage_map: HashMap<u32, (u64, u32, u32, u32)> = HashMap::new();
    for &pid in &fg_family {
        // Only enrich non-foreground in the loop below
        let _ = pid;
    }
    // Build rusage map for all PIDs — O(n) syscalls, ~3µs each
    for (pid, _process) in sys.processes() {
        let pid_u32 = pid.as_u32();
        if let Some(ri) = proc_taskinfo::get_rusage_info(pid_u32) {
            let idle_wk = ri.idle_wakeups;
            if let Some(ti) = proc_taskinfo::get_task_info(pid_u32) {
                rusage_map.insert(
                    pid_u32,
                    (
                        idle_wk,
                        ti.messages_sent + ti.messages_received,
                        ti.faults,
                        ti.pageins,
                    ),
                );
            } else {
                rusage_map.insert(pid_u32, (idle_wk, 0, 0, 0));
            }
        }
    }

    let mut proc_snaps = Vec::new();
    let mut hunt_snaps = Vec::new();

    let now_unix_secs: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    for (pid, process) in sys.processes() {
        let pid_u32 = pid.as_u32();
        let name = process.name().to_string();
        let is_foreground = fg_family.contains(&pid_u32);
        let ppid = process.parent().map(|p| p.as_u32()).unwrap_or(0);
        let parent_alive = ppid > 0;
        let is_zombie = process.status() == ProcessStatus::Zombie;
        let rss = process.memory();
        let cpu = process.cpu_usage();
        // process.start_time() → seconds since Unix epoch; 0 if unknown.
        let process_uptime_secs = {
            let start = process.start_time();
            if start > 0 {
                now_unix_secs.saturating_sub(start)
            } else {
                u64::MAX // unknown start → treat as long-lived
            }
        };

        // Real idle wakeups from proc_pid_rusage — the #1 signal for wasteful daemons.
        // Estimate wakeups/sec: idle_wakeups is cumulative, divide by uptime estimate.
        // Mach messages > 0 implies the process has active IPC (network, XPC, etc.)
        let (wakeups_per_sec, has_network_signal, faults_total, pageins_total) =
            match rusage_map.get(&pid_u32) {
                Some(&(idle_wk, mach_msgs, faults, pageins)) => {
                    // Rough estimate: if idle_wakeups > 1000, it's a chatty daemon
                    let wps = if idle_wk > 10_000 {
                        (idle_wk as f32 / 3600.0).min(100.0)
                    } else if idle_wk > 100 {
                        (idle_wk as f32 / 7200.0).min(50.0)
                    } else {
                        0.0
                    };
                    // Rate-based network detection: cumulative mach_msgs / uptime.
                    // Avoids false positives on long-lived daemons with high cumulative
                    // counts but near-zero actual IPC rate.
                    let msg_rate = if process_uptime_secs > 0 {
                        mach_msgs as f64 / process_uptime_secs as f64
                    } else {
                        0.0
                    };
                    let has_net = msg_rate > 0.1; // >0.1 msg/sec = active IPC
                    (wps, has_net, faults, pageins)
                }
                None => (0.0, false, 0, 0),
            };

        proc_snaps.push(ProcessSnapshot {
            pid: pid_u32,
            name: name.clone(),
            cpu_percent: cpu,
            rss_bytes: rss,
            is_zombie,
            secs_since_foreground: if is_foreground { 0 } else { 3600 },
            secs_since_user_interaction: if is_foreground { 0 } else { 3600 },
            has_network: has_network_signal,
            has_gui_window: is_foreground,
            wakeups_per_sec,
            parent_alive,
            process_uptime_secs,
            faults_total,
            pageins_total,
            is_translated: apollo_optimizer::engine::process_identity::is_translated(pid_u32),
            mach_port_count: 0, // populated lazily for hoarder candidates only
        });

        hunt_snaps.push(HuntSnapshot {
            pid: pid_u32,
            ppid,
            name,
            is_kernel_zombie: is_zombie,
            parent_alive,
            has_gui_window: is_foreground,
            rss_bytes: rss,
            cpu_percent: cpu,
            wakeups_per_sec,
            secs_since_user_interaction: if is_foreground { 0 } else { 3600 },
            host_app_pid: process.parent().map(|p| p.as_u32()),
            host_app_running: parent_alive,
            host_app_absent_secs: if parent_alive { 0 } else { 3600 },
        });
    }

    (proc_snaps, hunt_snaps)
}

// ── Heuristic Decision Merger ──────────────────────────────────────────────

pub fn convert_and_merge_heuristic_decisions(
    decisions: &[ProcessDecision],
    existing_actions: &[RootAction],
    critical_pids: &HashSet<u32>,
) -> (Vec<RootAction>, HeuristicStats) {
    let mut stats = HeuristicStats {
        decisions_total: decisions.len() as u64,
        throttles: 0,
        freezes: 0,
        kills_downgraded: 0,
        zombies_detected: 0,
    };

    // Build set of PIDs already acted on by decide_actions + learned_policy
    let existing_pids: HashSet<u32> = existing_actions
        .iter()
        .filter_map(|a| match a {
            RootAction::BoostProcess { pid, .. }
            | RootAction::ThrottleProcess { pid, .. }
            | RootAction::FreezeProcess { pid, .. } => Some(*pid),
            _ => None,
        })
        .collect();

    let mut new_actions = Vec::new();

    for decision in decisions {
        // Count zombies
        if decision.tier == ProcessTier::ZombieOrphan {
            stats.zombies_detected += 1;
        }

        // Skip if Allow
        if decision.decision == GovernorDecision::Allow {
            continue;
        }

        // Skip if already has an action from decide_actions/learned_policy
        if existing_pids.contains(&decision.pid) {
            continue;
        }

        // Skip critical processes
        if critical_pids.contains(&decision.pid) {
            continue;
        }

        // Complete Mediation guard: apply the same interactive-app name check that
        // decide_actions uses, so Freeze/Kill decisions from the AdaptiveGovernor
        // never bypass it. Production data (2026-04-06): "Antigravity Helper (Renderer)"
        // frozen 1x, "Notion"/"Notion Helper (Renderer)" frozen 7x — both contain a
        // known interactive app name but their PIDs were absent from critical_pids
        // because classify_protection yields ConditionalForeground for helper subprocesses
        // (only the exact foreground PID is inserted, not the parent's helpers).
        // [Saltzer & Kaashoek 2009] "Principles of Computer System Design" §3.3
        // Complete Mediation — every access path must go through the same gate.
        // Extend to ALL action types (Freeze, Kill, AND Throttle) for interactive
        // app names. Production data: "Brave Helper (Renderer)" with AppHelper tier
        // can receive GovernorDecision::Throttle — throttling a renderer subprocess
        // degrades the parent browser's frame rate just as much as freezing it.
        // [Saltzer & Kaashoek 2009] Complete Mediation — same gate for all paths.
        if is_interactive_app_name(&decision.name) {
            continue;
        }

        match decision.decision {
            GovernorDecision::Throttle => {
                let (ss, su) = pid_start_time(decision.pid);
                new_actions.push(RootAction::ThrottleProcess {
                    pid: decision.pid,
                    name: decision.name.clone(),
                    aggressive: false,
                    reason: format!("heuristic: {}", decision.reason),
                    start_sec: ss,
                    start_usec: su,
                });
                stats.throttles += 1;
            }
            GovernorDecision::Freeze => {
                let (ss, su) = pid_start_time(decision.pid);
                new_actions.push(RootAction::FreezeProcess {
                    pid: decision.pid,
                    name: decision.name.clone(),
                    reason: format!("heuristic: {}", decision.reason),
                    start_sec: ss,
                    start_usec: su,
                });
                stats.freezes += 1;
            }
            GovernorDecision::Kill => {
                let (ss, su) = pid_start_time(decision.pid);
                // Safety: downgrade Kill to Freeze — never auto-kill from heuristics
                new_actions.push(RootAction::FreezeProcess {
                    pid: decision.pid,
                    name: decision.name.clone(),
                    reason: format!("heuristic (kill→freeze): {}", decision.reason),
                    start_sec: ss,
                    start_usec: su,
                });
                stats.kills_downgraded += 1;
                stats.freezes += 1;
            }
            GovernorDecision::Allow => unreachable!(),
        }
    }

    (new_actions, stats)
}
