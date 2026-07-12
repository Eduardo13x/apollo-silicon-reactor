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
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use apollo_engine::engine::adaptive_governor::{GovernorDecision, ProcessDecision};
use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::daemon_helpers::pid_start_time;
use apollo_engine::engine::decide_actions::is_interactive_app_name;
use apollo_engine::engine::proc_taskinfo;
use apollo_engine::engine::process_classifier::{ProcessSnapshot, ProcessTier};
use apollo_engine::engine::process_tree::ProcessTree;
use apollo_engine::engine::recently_applied::{CachedActionKind, RecentlyApplied};
use apollo_engine::engine::safety::is_protected_name;
use apollo_engine::engine::types::{InteractiveContext, RootAction, SafetyPolicy};
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

// ── proc_taskinfo cache (Changes A+B, 2026-05-16) ──────────────────────────
//
// Under sustained high pressure (>0.80), Apollo's own hot path becomes a
// noticeable contributor to thrashing: proc_taskinfo + rusage_info are two
// kernel syscalls per enriched PID per cycle (~150 PIDs × 2 × 2.5 Hz =
// ~750 syscalls/sec just for the enrichment stage). The data those calls
// produce — idle_wakeups, mach msg counts, faults, pageins, CPU contention —
// changes slowly relative to the cycle period; refreshing every 4 cycles
// (~1.6 s) costs little signal under stress and recovers ~2-3% of Apollo's
// own CPU footprint, which is precisely the work that was making the
// pressure worse.
//
// Live RSS / cpu_usage / status still come from the cheap sysinfo refresh
// every cycle, so the rest of the enrichment stays fresh. Only the
// per-PID syscall payload is reused.

#[derive(Default, Clone)]
struct CachedEnrichSyscalls {
    rusage_map: HashMap<u32, (u64, u32, u32, u32)>,
    contention_map: HashMap<u32, f64>,
    identity_map: HashMap<u32, CachedProcessIdentity>,
    cycle_filled: u64,
    initialized: bool,
}

#[derive(Clone, Copy)]
struct CachedProcessIdentity {
    start_time: u64,
    is_app_bundle: bool,
    is_translated: bool,
}

/// Scale task-counter cache capacity with the live process table instead of a
/// chip-specific seat assumption, while retaining hard memory bounds.
const ENRICH_CACHE_MIN_CAP: usize = 512;
const ENRICH_CACHE_MAX_CAP: usize = 2048;

/// Identity fields are stable for a process lifetime, so cache them by
/// PID plus start time. This cap covers large Apple Silicon sessions while
/// still bounding a PID-spawn storm.
const IDENTITY_CACHE_HARD_CAP: usize = 2048;

fn enrich_syscall_cache() -> &'static Mutex<CachedEnrichSyscalls> {
    static CACHE: OnceLock<Mutex<CachedEnrichSyscalls>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(CachedEnrichSyscalls::default()))
}

/// Public invalidation hook — called from `daemon_kqueue_tick` on
/// `NOTE_EXIT` so we can purge the dead PID from the cache immediately
/// instead of waiting for the next cache-miss cycle. Without this, the
/// 4-cycle reuse window can serve stale rusage / contention values for
/// a recycled PID (the ABA bug pattern that Sprint 3 closed in the
/// `IdentityCache` — same hazard, same fix).
pub fn invalidate_cached_enrich(pid: u32) {
    if let Ok(mut cache) = enrich_syscall_cache().lock() {
        cache.rusage_map.remove(&pid);
        cache.contention_map.remove(&pid);
        cache.identity_map.remove(&pid);
    }
}

/// Test-only reset of the global cache. Safe to call between tests
/// because the cache is keyed by PID; tests that don't share PIDs
/// would not collide, but explicit reset keeps regression surface
/// predictable.
#[cfg(test)]
pub fn reset_enrich_cache_for_test() {
    if let Ok(mut cache) = enrich_syscall_cache().lock() {
        cache.rusage_map.clear();
        cache.contention_map.clear();
        cache.identity_map.clear();
        cache.cycle_filled = 0;
        cache.initialized = false;
    }
}

/// Returns true when the proc_taskinfo bulk read should be skipped this
/// cycle in favour of the cached values. Hold on the cache: only skip
/// when the cache has been filled and this cycle is off the adaptive refresh
/// cadence. Stable and severe-pressure regimes refresh every fourth cycle;
/// the transition band refreshes every second cycle for faster response.
fn should_reuse_enrich_cache(
    cycle_count: u64,
    pressure_smooth: f64,
    cache_initialized: bool,
    last_refresh_cycle: u64,
) -> bool {
    if !cache_initialized {
        return false;
    }

    let refresh_every = if pressure_smooth <= 0.50 || pressure_smooth > 0.80 {
        4
    } else {
        2
    };
    cycle_count.saturating_sub(last_refresh_cycle) < refresh_every
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
    cycle_count: u64,
    pressure_smooth: f64,
    lf_metrics: &apollo_engine::engine::lse_counters::LockFreeMetrics,
) -> (Vec<ProcessSnapshot>, Vec<HuntSnapshot>) {
    // Pre-compute the set of PIDs in the foreground family for O(1) lookups.
    let fg_family: HashSet<u32> = build_foreground_family(foreground_pid, tree);

    // Bulk-read idle_wakeups + Mach messages via proc_taskinfo (~1.3ms for ~400 pids).
    // This replaces the hardcoded wakeups_per_sec: 0.0 with REAL kernel data.
    // pid → (idle_wakeups, mach_msgs, faults, pageins)
    //
    // 2026-05-12: pre-allocated capacity to avoid the repeated grow-and-rehash
    // pattern on each cycle. Typical M1 8GB seat has ~400 PIDs of which ~150
    // pass the ENRICH_MIN_RSS_BYTES gate; sizing for 256 covers the steady
    // state plus headroom without over-committing. Same for contention_map.
    // Saves ~0.1-0.3ms/cycle vs the default HashMap::new() which starts at
    // capacity 0 and grows through 4 → 8 → 16 → 32 → 64 → 128 → 256.
    // Phase 0d performance gate (2026-05-10): skip proc_taskinfo syscalls
    // for PIDs we'll never act on (RSS < ENRICH_MIN_RSS_BYTES). On a
    // typical Mac with 400 PIDs, ~250 are <2 MB tiny daemons we never
    // throttle/freeze. Skipping their 2× syscalls cuts ~500 syscalls per
    // cycle, ≈ 1.5 ms saved. Foreground family bypasses the gate so we
    // never miss their state. [Hellerstein 2004 §9 sampling under load]
    const ENRICH_MIN_RSS_BYTES: u64 = 2 * 1024 * 1024;

    // Kernel task counters move much more slowly than sysinfo's live RSS/CPU.
    // Reuse them on an adaptive cadence in both stable and stressed regimes;
    // the transition band refreshes more often so pressure changes remain
    // responsive. Identity fields are lifetime-stable and are cached by
    // (PID, start_time), independently of the task-counter cadence.
    let process_count = sys.processes().len();
    let enrich_cache_capacity = process_count.clamp(ENRICH_CACHE_MIN_CAP, ENRICH_CACHE_MAX_CAP);
    let (reuse_cache, mut rusage_map, mut contention_map, mut identity_map) = {
        let cache = enrich_syscall_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let reuse = should_reuse_enrich_cache(
            cycle_count,
            pressure_smooth,
            cache.initialized,
            cache.cycle_filled,
        );
        let rusage = if reuse {
            cache.rusage_map.clone()
        } else {
            HashMap::with_capacity(enrich_cache_capacity)
        };
        let contention = if reuse {
            cache.contention_map.clone()
        } else {
            HashMap::with_capacity(enrich_cache_capacity)
        };
        (reuse, rusage, contention, cache.identity_map.clone())
    };
    if reuse_cache {
        lf_metrics.inc_taskinfo_cache_hit();
    } else {
        lf_metrics.inc_taskinfo_cache_miss();
    }

    let mut live_pids: HashSet<u32> = HashSet::with_capacity(process_count);

    if !reuse_cache {
        let mut rusage_samples = Vec::with_capacity(enrich_cache_capacity);
        for (pid, process) in sys.processes() {
            let pid_u32 = pid.as_u32();
            live_pids.insert(pid_u32);
            // Gate: enrich only PIDs with meaningful RSS or in the fg family.
            if process.memory() < ENRICH_MIN_RSS_BYTES && !fg_family.contains(&pid_u32) {
                continue;
            }
            if let Some(ri) = proc_taskinfo::get_rusage_info(pid_u32) {
                rusage_samples.push((pid_u32, ri));
            }
        }

        // Finish all kernel reads before taking the contention lock, then
        // observe the batch under one acquisition instead of one per PID.
        for (pid_u32, ri) in &rusage_samples {
            let idle_wk = ri.idle_wakeups;
            if let Some(ti) = proc_taskinfo::get_task_info(*pid_u32) {
                rusage_map.insert(
                    *pid_u32,
                    (
                        idle_wk,
                        ti.messages_sent + ti.messages_received,
                        ti.faults,
                        ti.pageins,
                    ),
                );
            } else {
                rusage_map.insert(*pid_u32, (idle_wk, 0, 0, 0));
            }
        }
        if let Ok(mut tracker) = apollo_engine::engine::contention_tracker::global().lock() {
            for (pid_u32, ri) in rusage_samples {
                if let Some(ratio) = tracker.observe(pid_u32, ri) {
                    contention_map.insert(pid_u32, ratio);
                }
            }
            tracker.gc(&live_pids);
        }
    } else {
        // Cache reuse path: still populate live_pids from the cheap
        // sysinfo iter so downstream uses (process classification, hunt)
        // see today's PID set, even though the syscall data is stale.
        for pid in sys.processes().keys() {
            live_pids.insert(pid.as_u32());
        }
    }

    // Both vectors receive one entry per live process, so size them from the
    // actual table instead of repeatedly growing from historical M1 guesses.
    let mut proc_snaps = Vec::with_capacity(process_count);
    let mut hunt_snaps = Vec::with_capacity(process_count);

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
        let process_start_time = process.start_time();
        let process_uptime_secs = {
            if process_start_time > 0 {
                now_unix_secs.saturating_sub(process_start_time)
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

        // Bundle membership and Rosetta translation cannot change during a
        // process lifetime. Re-probe only for a new PID/start-time identity.
        let identity = identity_map
            .get(&pid_u32)
            .copied()
            .filter(|cached| process_start_time > 0 && cached.start_time == process_start_time)
            .unwrap_or_else(|| CachedProcessIdentity {
                start_time: process_start_time,
                is_app_bundle: apollo_engine::engine::proc_taskinfo::is_user_app_bundle(pid_u32)
                    .unwrap_or(false),
                is_translated: apollo_engine::engine::process_identity::is_translated(pid_u32),
            });
        identity_map.insert(pid_u32, identity);

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
            is_translated: identity.is_translated,
            mach_port_count: 0, // populated lazily for hoarder candidates only
            cpu_contention: contention_map.get(&pid_u32).copied(),
            is_app_bundle: identity.is_app_bundle,
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

    // Keep all enrichment caches tied to the current process table. The
    // start-time identity check also prevents PID reuse from inheriting stale
    // bundle or translation state if an exit notification races this cycle.
    rusage_map.retain(|pid, _| live_pids.contains(pid));
    contention_map.retain(|pid, _| live_pids.contains(pid));
    identity_map.retain(|pid, _| live_pids.contains(pid));

    if rusage_map.len() > enrich_cache_capacity {
        let drop: Vec<u32> = rusage_map
            .keys()
            .copied()
            .skip(enrich_cache_capacity)
            .collect();
        let evicted = drop.len() as u64;
        for pid in &drop {
            rusage_map.remove(pid);
            contention_map.remove(pid);
        }
        lf_metrics.add_taskinfo_cache_cap_evictions(evicted);
    }
    if identity_map.len() > IDENTITY_CACHE_HARD_CAP {
        let drop: Vec<u32> = identity_map
            .keys()
            .copied()
            .skip(IDENTITY_CACHE_HARD_CAP)
            .collect();
        for pid in drop {
            identity_map.remove(&pid);
        }
    }

    {
        let mut cache = enrich_syscall_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        cache.rusage_map = rusage_map;
        cache.contention_map = contention_map;
        cache.identity_map = identity_map;
        if !reuse_cache {
            cache.cycle_filled = cycle_count;
            cache.initialized = true;
        }
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

    fn cache_test_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

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
        let (actions, stats) =
            convert_and_merge_heuristic_decisions(&decisions, &[], &critical, &mut cache);
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
        let (actions1, _) =
            convert_and_merge_heuristic_decisions(&decisions, &[], &critical, &mut cache);
        assert_eq!(actions1.len(), 1);

        // Cycle 2: same decision must be SUPPRESSED (within 30s TTL).
        let (actions2, stats2) =
            convert_and_merge_heuristic_decisions(&decisions, &[], &critical, &mut cache);
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
        assert_eq!(
            context_to_thermal(InteractiveContext::ThermalConstrained),
            "constrained"
        );
    }

    #[test]
    fn context_to_thermal_background_pressure() {
        assert_eq!(
            context_to_thermal(InteractiveContext::BackgroundPressure),
            "elevated"
        );
    }

    #[test]
    fn context_to_thermal_interactive_focus() {
        assert_eq!(
            context_to_thermal(InteractiveContext::InteractiveFocus),
            "nominal"
        );
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
            pid: 100,
            ppid: 1,
            name: "app".into(),
            cpu_usage: 0.0,
            memory_bytes: 0,
        }];
        let tree = ProcessTree::build(&entries);
        let result = build_foreground_family(Some(100), &tree);
        assert!(result.contains(&100), "root pid must be in family");
    }

    #[test]
    fn foreground_family_includes_children_excludes_unrelated() {
        let entries = vec![
            ProcessEntry {
                pid: 100,
                ppid: 1,
                name: "app".into(),
                cpu_usage: 0.0,
                memory_bytes: 0,
            },
            ProcessEntry {
                pid: 200,
                ppid: 100,
                name: "helper".into(),
                cpu_usage: 0.0,
                memory_bytes: 0,
            },
            ProcessEntry {
                pid: 300,
                ppid: 1,
                name: "other".into(),
                cpu_usage: 0.0,
                memory_bytes: 0,
            },
        ];
        let tree = ProcessTree::build(&entries);
        let result = build_foreground_family(Some(100), &tree);
        assert!(result.contains(&100));
        assert!(
            result.contains(&200),
            "child of foreground must be in family"
        );
        assert!(
            !result.contains(&300),
            "unrelated PID must not be in family"
        );
    }

    // ── apply_post_wake_grace_policy ──────────────────────────────────────────
    // [Aniche 2022 §2] Category-partition: each RootAction variant is a distinct
    // category; grace_active is the toggle.

    fn freeze(pid: u32) -> RootAction {
        RootAction::FreezeProcess {
            pid,
            name: "p".into(),
            reason: "r".into(),
            start_sec: 0,
            start_usec: 0,
            decision_reason: DecisionReason::PressureContext,
        }
    }
    fn throttle(pid: u32, aggressive: bool) -> RootAction {
        RootAction::ThrottleProcess {
            pid,
            name: "p".into(),
            aggressive,
            reason: "r".into(),
            start_sec: 0,
            start_usec: 0,
            decision_reason: DecisionReason::PressureContext,
        }
    }
    fn quarantine() -> RootAction {
        RootAction::QuarantineDaemon {
            daemon: "d".into(),
            active: true,
            reason: "r".into(),
            decision_reason: DecisionReason::PressureContext,
        }
    }
    fn boost(pid: u32) -> RootAction {
        RootAction::BoostProcess {
            pid,
            name: "p".into(),
            reason: "r".into(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }
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
            RootAction::ThrottleProcess { aggressive, .. } => {
                assert!(!aggressive, "must be downgraded")
            }
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

    // ── Phase 1 prod-grade (2026-05-16): enrichment cache invariants ──────
    //
    // The taskinfo cache must:
    //   (a) purge a PID when invalidate_cached_enrich is called (NOTE_EXIT)
    //   (b) not let dead PIDs survive a cache-miss fill
    //   (c) cap at 512 entries and bump the eviction counter when triggered
    //
    // Tests (a) directly. (b) and (c) require the full enrichment
    // path which is hard to exercise in a pure unit test — those are
    // covered by post-deploy metric checks (taskinfo_cache_hits,
    // exit_invalidations, cap_evictions) per the Disobedience Rule
    // mechanical verification step (CLAUDE.md 2026-05-07).

    #[test]
    fn invalidate_cached_enrich_purges_pid_from_all_maps() {
        let _guard = cache_test_guard();
        reset_enrich_cache_for_test();
        // Seed all maps with PID 4242.
        {
            let mut cache = enrich_syscall_cache().lock().unwrap();
            cache.rusage_map.insert(4242, (1, 2, 3, 4));
            cache.contention_map.insert(4242, 0.75);
            cache.identity_map.insert(
                4242,
                CachedProcessIdentity {
                    start_time: 100,
                    is_app_bundle: true,
                    is_translated: false,
                },
            );
            cache.rusage_map.insert(4243, (5, 6, 7, 8));
        }
        invalidate_cached_enrich(4242);
        let cache = enrich_syscall_cache().lock().unwrap();
        assert!(
            !cache.rusage_map.contains_key(&4242),
            "rusage entry for 4242 must be purged"
        );
        assert!(
            !cache.contention_map.contains_key(&4242),
            "contention entry for 4242 must be purged"
        );
        assert!(
            !cache.identity_map.contains_key(&4242),
            "identity entry for 4242 must be purged"
        );
        assert!(
            cache.rusage_map.contains_key(&4243),
            "neighbouring PID 4243 must NOT be touched"
        );
    }

    #[test]
    fn invalidate_cached_enrich_on_missing_pid_is_noop() {
        let _guard = cache_test_guard();
        reset_enrich_cache_for_test();
        // No panic, no error, cache stays empty.
        invalidate_cached_enrich(9999);
        let cache = enrich_syscall_cache().lock().unwrap();
        assert!(cache.rusage_map.is_empty());
        assert!(cache.contention_map.is_empty());
        assert!(cache.identity_map.is_empty());
    }

    #[test]
    fn warm_enrichment_cache_preserves_live_snapshot_shape() {
        let _guard = cache_test_guard();
        reset_enrich_cache_for_test();

        let sys = sysinfo::System::new_all();
        let tree = ProcessTree::build(&[]);
        let metrics = apollo_engine::engine::lse_counters::LockFreeMetrics::default();

        let cold_started = Instant::now();
        let (cold_processes, cold_hunts) =
            build_enriched_process_data_with_tree(&sys, None, &tree, 0, 0.30, &metrics);
        let cold_elapsed = cold_started.elapsed();
        let identities_after_cold = enrich_syscall_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .identity_map
            .len();

        let warm_started = Instant::now();
        let (warm_processes, warm_hunts) =
            build_enriched_process_data_with_tree(&sys, None, &tree, 1, 0.30, &metrics);
        let warm_elapsed = warm_started.elapsed();
        let identities_after_warm = enrich_syscall_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .identity_map
            .len();

        assert_eq!(cold_processes.len(), sys.processes().len());
        assert_eq!(cold_hunts.len(), cold_processes.len());
        assert_eq!(warm_processes.len(), cold_processes.len());
        assert_eq!(warm_hunts.len(), cold_hunts.len());
        assert_eq!(identities_after_warm, identities_after_cold);
        assert!(identities_after_cold > 0);
        assert_eq!(
            metrics
                .taskinfo_cache_misses
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            metrics
                .taskinfo_cache_hits
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        eprintln!(
            "enrichment cold={:?} warm={:?} processes={}",
            cold_elapsed,
            warm_elapsed,
            cold_processes.len()
        );
    }

    #[test]
    fn should_reuse_enrich_cache_uses_adaptive_cadence() {
        // An empty cache always refreshes, regardless of cadence.
        assert!(!should_reuse_enrich_cache(1, 0.30, false, 0));
        assert!(!should_reuse_enrich_cache(3, 0.90, false, 0));

        // Stable pressure: one refresh followed by three reuse cycles.
        assert!(should_reuse_enrich_cache(1, 0.30, true, 0));
        assert!(should_reuse_enrich_cache(3, 0.50, true, 0));
        assert!(!should_reuse_enrich_cache(4, 0.30, true, 0));

        // Transition band: refresh every second cycle for responsiveness.
        assert!(should_reuse_enrich_cache(11, 0.65, true, 10));
        assert!(!should_reuse_enrich_cache(12, 0.65, true, 10));
        assert!(!should_reuse_enrich_cache(14, 0.80, true, 12));

        // Severe pressure returns to a four-cycle cadence to reduce self-load.
        assert!(should_reuse_enrich_cache(22, 0.85, true, 20));
        assert!(!should_reuse_enrich_cache(24, 0.85, true, 20));
    }
}
