//! Process Enrichment — pure helper functions extracted from daemon monolith.
//!
//! Contains:
//! - `filter_boost_cooldown()` — dedup boost actions with per-PID cooldowns
//! - `apply_post_wake_grace_policy()` — suppress freeze/throttle during post-wake grace
//! - `context_to_thermal()` — interactive context → thermal string
//! - `build_foreground_family()` — compute foreground PID set from process tree
//! - `build_enriched_process_data_with_tree()` — build ProcessSnapshot + HuntSnapshot
//! - `convert_and_merge_heuristic_decisions()` — merge heuristic decisions into actions
//! - `HeuristicStats` — counters for heuristic action conversions
//! - `ThrashState` — per-PID cooldown tracking

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use apollo_engine::engine::adaptive_governor::{GovernorDecision, ProcessDecision};
use apollo_engine::engine::recently_applied::{CachedActionKind, RecentlyApplied};
use apollo_engine::engine::daemon_helpers::pid_start_time;
use apollo_engine::engine::decide_actions::is_interactive_app_name;
use apollo_engine::engine::proc_taskinfo;
use apollo_engine::engine::process_classifier::{ProcessSnapshot, ProcessTier};
use apollo_engine::engine::process_tree::ProcessTree;
use apollo_engine::engine::safety::is_protected_name;
use apollo_engine::engine::types::{InteractiveContext, RootAction, SafetyPolicy};
use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::zombie_hunter::HuntSnapshot;
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
                decision_reason,
            } => {
                throttle_suppressed += 1;
                out.push(RootAction::ThrottleProcess {
                    pid,
                    name,
                    aggressive: false,
                    reason,
                    start_sec,
                    start_usec,
                    decision_reason,
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
    // CPU contention map: pid → ratio ∈ [0, 1] between the prev rusage
    // sample cached in the global ContentionTracker and the one we read
    // this cycle. None on the first cycle for a pid, or when the process
    // was fully idle. Feeds ProcessSnapshot.cpu_contention below.
    let mut contention_map: HashMap<u32, f64> = HashMap::new();
    for &pid in &fg_family {
        // Only enrich non-foreground in the loop below
        let _ = pid;
    }
    // Build rusage map for all PIDs — O(n) syscalls, ~3µs each
    let mut live_pids: HashSet<u32> = HashSet::new();
    for (pid, _process) in sys.processes() {
        let pid_u32 = pid.as_u32();
        live_pids.insert(pid_u32);
        if let Some(ri) = proc_taskinfo::get_rusage_info(pid_u32) {
            let idle_wk = ri.idle_wakeups;
            // Observe into the global contention tracker. This returns the
            // ratio vs the previous cached sample (None on the first cycle
            // or when the process was idle) and stores the new sample as
            // the next baseline. The mutex is held only for the observe
            // call itself; no other I/O happens under it.
            if let Ok(mut tracker) = apollo_engine::engine::contention_tracker::global().lock() {
                if let Some(ratio) = tracker.observe(pid_u32, ri.clone()) {
                    contention_map.insert(pid_u32, ratio);
                }
            }
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
    // GC any tracker entries for pids that disappeared this cycle so the
    // map can't grow beyond the live pid set over a long-running session.
    if let Ok(mut tracker) = apollo_engine::engine::contention_tracker::global().lock() {
        tracker.gc(&live_pids);
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

        // Behavioural app-bundle detection: one extra proc_pidpath syscall
        // (~3 µs on M1) per pid, only here in enrichment. The result is
        // cached on ProcessSnapshot so downstream consumers don't repeat
        // the syscall.
        let is_app_bundle =
            apollo_engine::engine::proc_taskinfo::is_user_app_bundle(pid_u32).unwrap_or(false);

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
            is_translated: apollo_engine::engine::process_identity::is_translated(pid_u32),
            mach_port_count: 0, // populated lazily for hoarder candidates only
            cpu_contention: contention_map.get(&pid_u32).copied(),
            is_app_bundle,
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
    recently_applied: &mut RecentlyApplied,
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

        // Cross-cycle state memory (SuperPlan 2026-05-06): if this PID had the
        // SAME decision applied within the last 30s, suppress emission. The
        // kernel would just say no-op ("PID already in target state") wasting
        // a syscall + journal entry. [Hellerstein 2004 §9] state-aware control.
        // CachedActionKind::from_governor maps Kill→Freeze automatically.
        if let Some(kind) = CachedActionKind::from_governor(decision.decision) {
            if recently_applied.is_recent(decision.pid, kind) {
                continue;
            }
        }

        // Complete Mediation guard — [Saltzer & Kaashoek 2009] §3.3: every path to a
        // privileged action must pass through the same access control point.
        //
        // Two-layer check (both must pass before an action is emitted):
        //
        // Layer 1 — is_protected_name(): single truth point for name-based protection.
        //   Covers OS essentials (protected_processes), infrastructure (docker/postgres),
        //   and dev runtimes (rustc/clippy-driver). Hot-path safe via OnceLock caches.
        //   Closes bypass class 1 (sharingd/logd loop): OS daemons not in INTERACTIVE_APPS
        //   were previously missed by the interactive-only check below.
        //
        // Layer 2 — is_interactive_app_name(): user-facing apps (Brave, Claude, Arc…).
        //   Covers Electron/WebKit helpers via substring match, closing bypass class 2
        //   (Notion Helper/Antigravity frozen 7x — not in OS list but in INTERACTIVE_APPS).
        //
        // Applies to ALL action types (Freeze, Kill, Throttle) — bypass class 3 was
        // that the original guard covered Freeze/Kill but not Throttle for renderer helpers.
        if is_protected_name(&decision.name) || is_interactive_app_name(&decision.name) {
            continue;
        }

        // Map governor reason string → specific DecisionReason variant.
        // Closes NotebookLM Low-priority gap: PressureContext was 62.5%
        // catch-all; SwarmThrottling/GraduatedIdle differentiate two
        // well-known governor rule classes that account for ~20% of throttles.
        let dr = classify_governor_reason(&decision.reason);

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
                    decision_reason: dr.clone(),
                });
                stats.throttles += 1;
                recently_applied.record(decision.pid, CachedActionKind::Throttle);
            }
            GovernorDecision::Freeze => {
                let (ss, su) = pid_start_time(decision.pid);
                new_actions.push(RootAction::FreezeProcess {
                    pid: decision.pid,
                    name: decision.name.clone(),
                    reason: format!("heuristic: {}", decision.reason),
                    start_sec: ss,
                    start_usec: su,
                    decision_reason: dr.clone(),
                });
                stats.freezes += 1;
                recently_applied.record(decision.pid, CachedActionKind::Freeze);
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
                    decision_reason: dr,
                });
                stats.kills_downgraded += 1;
                stats.freezes += 1;
                recently_applied.record(decision.pid, CachedActionKind::Freeze);
            }
            GovernorDecision::Allow => unreachable!(),
        }
    }

    (new_actions, stats)
}

/// Map an adaptive_governor reason string → specific DecisionReason variant.
///
/// Closes NotebookLM Low-priority gap (2026-05-06): PressureContext was a
/// 62.5% catch-all in the audit log. Two well-known governor rule classes
/// account for ~20% of throttles and deserve their own labels:
///
/// - `Swarm throttle (...)` (adaptive_governor.rs:616) → `SwarmThrottling`
/// - `graduated idle` / `GUI app abandoned >24h` → `GraduatedIdle`
///
/// All other governor rules continue to fall back to PressureContext.
/// Future iteration: wire ThreadQoSRouting at SetThreadQoS sites once the
/// downstream mach_qos affinity consumer lands (see Phase 3 commit bef1f0b).
pub fn classify_governor_reason(reason: &str) -> DecisionReason {
    if reason.starts_with("Swarm throttle") {
        DecisionReason::SwarmThrottling
    } else if reason.contains("graduated idle")
        || reason.contains("GUI app abandoned")
        || reason.contains("idle >6h")
        || reason.contains("idle >12h")
    {
        DecisionReason::GraduatedIdle
    } else {
        DecisionReason::PressureContext
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::engine::process_tree::ProcessEntry;

    // ── context_to_thermal ────────────────────────────────────────────────────

    // ── classify_governor_reason ──────────────────────────────────────────────

    #[test]
    fn classify_swarm_throttle_string() {
        let r = "Swarm throttle (52 procs, waste=0.65, util=0.40)";
        assert_eq!(classify_governor_reason(r), DecisionReason::SwarmThrottling);
    }

    #[test]
    fn classify_graduated_idle_strings() {
        // Multiple phrasings produced by adaptive_governor.rs.
        assert_eq!(
            classify_governor_reason("graduated idle 6h+ throttle"),
            DecisionReason::GraduatedIdle
        );
        assert_eq!(
            classify_governor_reason("GUI app abandoned >24h (idle=26h)"),
            DecisionReason::GraduatedIdle
        );
    }

    fn make_decision(pid: u32, name: &str, kind: GovernorDecision) -> ProcessDecision {
        ProcessDecision {
            pid,
            name: name.to_string(),
            decision: kind,
            tier: ProcessTier::SilentDaemon,
            utility_score: 0.1,
            waste_score: 0.5,
            reason: format!("test {:?}", kind),
        }
    }

    #[test]
    fn convert_and_merge_emits_first_throttle_normally() {
        let mut cache = RecentlyApplied::new();
        let critical = HashSet::new();
        let decisions = vec![make_decision(1234, "testproc", GovernorDecision::Throttle)];
        let (actions, stats) = convert_and_merge_heuristic_decisions(
            &decisions,
            &[],
            &critical,
            &mut cache,
        );
        assert_eq!(actions.len(), 1);
        assert_eq!(stats.throttles, 1);
        assert!(cache.is_recent(1234, CachedActionKind::Throttle));
    }

    #[test]
    fn convert_and_merge_suppresses_duplicate_within_ttl() {
        // Same decision for same PID across two calls — second call must drop.
        let mut cache = RecentlyApplied::new();
        let critical = HashSet::new();
        let decisions = vec![make_decision(1234, "testproc", GovernorDecision::Throttle)];

        // Cycle 1: first emission
        let (actions1, _) = convert_and_merge_heuristic_decisions(
            &decisions,
            &[],
            &critical,
            &mut cache,
        );
        assert_eq!(actions1.len(), 1);

        // Cycle 2: same decision must be SUPPRESSED (within 30s TTL).
        let (actions2, stats2) = convert_and_merge_heuristic_decisions(
            &decisions,
            &[],
            &critical,
            &mut cache,
        );
        assert_eq!(actions2.len(), 0, "duplicate within TTL must be suppressed");
        assert_eq!(stats2.throttles, 0);
    }

    #[test]
    fn convert_and_merge_allows_freeze_after_throttle() {
        // Per-kind cache: a PID can be throttled, then later upgraded to freeze.
        let mut cache = RecentlyApplied::new();
        let critical = HashSet::new();

        let throttle = vec![make_decision(1234, "testproc", GovernorDecision::Throttle)];
        let freeze = vec![make_decision(1234, "testproc", GovernorDecision::Freeze)];

        let (a1, _) = convert_and_merge_heuristic_decisions(&throttle, &[], &critical, &mut cache);
        assert_eq!(a1.len(), 1);

        // Freeze for SAME pid is a different cache key — should pass through.
        let (a2, _) = convert_and_merge_heuristic_decisions(&freeze, &[], &critical, &mut cache);
        assert_eq!(a2.len(), 1, "Freeze with prior Throttle must emit");
    }

    #[test]
    fn convert_and_merge_kill_caches_as_freeze() {
        // Apollo downgrades Kill→Freeze; cache key must reflect the EFFECTIVE
        // decision so a follow-up Freeze for the same PID is suppressed (no
        // double-freezing the same PID).
        let mut cache = RecentlyApplied::new();
        let critical = HashSet::new();

        let kill = vec![make_decision(1234, "testproc", GovernorDecision::Kill)];
        let (a1, stats1) = convert_and_merge_heuristic_decisions(&kill, &[], &critical, &mut cache);
        assert_eq!(a1.len(), 1);
        assert_eq!(stats1.kills_downgraded, 1);
        assert!(cache.is_recent(1234, CachedActionKind::Freeze));

        // Subsequent Freeze for same PID must be suppressed.
        let freeze = vec![make_decision(1234, "testproc", GovernorDecision::Freeze)];
        let (a2, _) = convert_and_merge_heuristic_decisions(&freeze, &[], &critical, &mut cache);
        assert_eq!(a2.len(), 0, "Freeze after Kill→Freeze must be suppressed");
    }

    #[test]
    fn classify_unknown_reason_falls_back_to_pressurecontext() {
        // Default safety: any unrecognized string maps to PressureContext.
        let r = "extreme pressure RSS-rank cpu-active 25%";
        assert_eq!(classify_governor_reason(r), DecisionReason::PressureContext);
    }

    // ── context_to_thermal ────────────────────────────────────────────────────

    #[test]
    fn context_to_thermal_constrained() {
        assert_eq!(context_to_thermal(InteractiveContext::ThermalConstrained), "constrained");
    }

    #[test]
    fn context_to_thermal_background_pressure() {
        assert_eq!(context_to_thermal(InteractiveContext::BackgroundPressure), "elevated");
    }

    #[test]
    fn context_to_thermal_interactive_focus() {
        assert_eq!(context_to_thermal(InteractiveContext::InteractiveFocus), "nominal");
    }

    // ── build_foreground_family ───────────────────────────────────────────────

    #[test]
    fn foreground_family_none_pid_returns_empty() {
        let tree = ProcessTree::build(&[]);
        assert!(build_foreground_family(None, &tree).is_empty());
    }

    #[test]
    fn foreground_family_root_only_no_children() {
        let entries = vec![ProcessEntry {
            pid: 100, ppid: 1, name: "app".into(), cpu_usage: 0.0, memory_bytes: 0,
        }];
        let tree = ProcessTree::build(&entries);
        let result = build_foreground_family(Some(100), &tree);
        assert!(result.contains(&100), "root pid must be in family");
    }

    #[test]
    fn foreground_family_includes_children_excludes_unrelated() {
        let entries = vec![
            ProcessEntry { pid: 100, ppid: 1,   name: "app".into(),    cpu_usage: 0.0, memory_bytes: 0 },
            ProcessEntry { pid: 200, ppid: 100, name: "helper".into(), cpu_usage: 0.0, memory_bytes: 0 },
            ProcessEntry { pid: 300, ppid: 1,   name: "other".into(),  cpu_usage: 0.0, memory_bytes: 0 },
        ];
        let tree = ProcessTree::build(&entries);
        let result = build_foreground_family(Some(100), &tree);
        assert!(result.contains(&100));
        assert!(result.contains(&200), "child of foreground must be in family");
        assert!(!result.contains(&300), "unrelated PID must not be in family");
    }

    // ── apply_post_wake_grace_policy ──────────────────────────────────────────
    // [Aniche 2022 §2] Category-partition: each RootAction variant is a distinct
    // category; grace_active is the toggle.

    fn freeze(pid: u32) -> RootAction {
        RootAction::FreezeProcess { pid, name: "p".into(), reason: "r".into(), start_sec: 0, start_usec: 0, decision_reason: DecisionReason::PressureContext }
    }
    fn throttle(pid: u32, aggressive: bool) -> RootAction {
        RootAction::ThrottleProcess { pid, name: "p".into(), aggressive, reason: "r".into(), start_sec: 0, start_usec: 0, decision_reason: DecisionReason::PressureContext }
    }
    fn quarantine() -> RootAction {
        RootAction::QuarantineDaemon { daemon: "d".into(), active: true, reason: "r".into(), decision_reason: DecisionReason::PressureContext }
    }
    fn boost(pid: u32) -> RootAction {
        RootAction::BoostProcess { pid, name: "p".into(), reason: "r".into(), decision_reason: DecisionReason::PressureContext }
    }

    #[test]
    fn grace_inactive_passes_all_actions_unchanged() {
        let actions = vec![freeze(1), throttle(2, true), boost(3)];
        let (out, ts, fs) = apply_post_wake_grace_policy(actions, false);
        assert_eq!(out.len(), 3);
        assert_eq!(ts, 0);
        assert_eq!(fs, 0);
    }

    #[test]
    fn grace_active_suppresses_freeze_and_quarantine() {
        let actions = vec![freeze(1), quarantine()];
        let (out, _ts, fs) = apply_post_wake_grace_policy(actions, true);
        assert!(out.is_empty());
        assert_eq!(fs, 2);
    }

    #[test]
    fn grace_active_downgrades_aggressive_throttle_to_gentle() {
        let actions = vec![throttle(1, true)];
        let (out, ts, fs) = apply_post_wake_grace_policy(actions, true);
        assert_eq!(out.len(), 1);
        assert_eq!(ts, 1);
        assert_eq!(fs, 0);
        match &out[0] {
            RootAction::ThrottleProcess { aggressive, .. } => assert!(!aggressive, "must be downgraded"),
            _ => panic!("expected ThrottleProcess"),
        }
    }

    #[test]
    fn grace_active_passes_non_aggressive_throttle_unchanged() {
        let actions = vec![throttle(1, false)];
        let (out, ts, _fs) = apply_post_wake_grace_policy(actions, true);
        assert_eq!(out.len(), 1);
        assert_eq!(ts, 0);
    }

    #[test]
    fn grace_active_passes_boost_unchanged() {
        let actions = vec![boost(42)];
        let (out, ts, fs) = apply_post_wake_grace_policy(actions, true);
        assert_eq!(out.len(), 1);
        assert_eq!(ts, 0);
        assert_eq!(fs, 0);
    }
}
