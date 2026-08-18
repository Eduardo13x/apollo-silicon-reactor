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

#[inline]
fn parent_is_alive_or_process_root(ppid: u32, parent_present: bool) -> bool {
    // PID 0 is the kernel process root, not a dead parent. PID 1 is launchd,
    // which legitimately adopts children; treat that relationship as live
    // even if a staggered sysinfo refresh omitted launchd from this snapshot.
    ppid <= 1 || parent_present
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
    visible_pids: HashSet<u32>,
    mach_port_map: HashMap<u32, u32>,
    cycle_filled: u64,
    visible_cycle_filled: u64,
    mach_ports_cycle_filled: u64,
    initialized: bool,
}

#[derive(Clone, Copy)]
struct CachedProcessIdentity {
    start_time: u64,
    is_app_bundle: bool,
    is_translated: bool,
    is_apple_platform: bool,
    last_foreground_at: Option<Instant>,
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
        cache.visible_pids.clear();
        cache.mach_port_map.clear();
        cache.cycle_filled = 0;
        cache.visible_cycle_filled = 0;
        cache.mach_ports_cycle_filled = 0;
        cache.initialized = false;
    }
}

/// Number of rolling PID groups used for kernel task-counter refreshes.
/// Cumulative counters tolerate a four-cycle (~1.2s) age in stable/severe
/// regimes; the transition band halves that bound for faster response.
fn taskinfo_refresh_period(pressure_smooth: f64) -> u32 {
    if pressure_smooth <= 0.50 || pressure_smooth > 0.80 {
        4
    } else {
        2
    }
}

/// Stable PID sharding spreads Mach reads across cycles. Cold and newly seen
/// processes bypass the shard so every snapshot gets an initial value.
fn should_refresh_taskinfo(
    pid: u32,
    cycle_count: u64,
    refresh_period: u32,
    cold_start: bool,
    has_cached_sample: bool,
) -> bool {
    cold_start
        || !has_cached_sample
        || pid % refresh_period == (cycle_count % u64::from(refresh_period)) as u32
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
    hardware_cores: u32,
    lf_metrics: &apollo_engine::engine::lse_counters::LockFreeMetrics,
) -> (Vec<ProcessSnapshot>, Vec<HuntSnapshot>, HashSet<u32>) {
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
    let high_capacity = hardware_cores >= 10;
    let taskinfo_refresh_period = taskinfo_refresh_period(pressure_smooth);
    let visible_refresh_every = if high_capacity { 10 } else { 20 };
    let mach_refresh_every = if high_capacity { 60 } else { 120 };
    let (
        cold_start,
        mut rusage_map,
        mut contention_map,
        mut identity_map,
        mut visible_pids,
        mut mach_port_map,
        refresh_visible,
        refresh_mach_ports,
    ) = {
        let mut cache = enrich_syscall_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let cold = !cache.initialized;
        let mut rusage = std::mem::take(&mut cache.rusage_map);
        let mut contention = std::mem::take(&mut cache.contention_map);
        if cold {
            rusage.reserve(enrich_cache_capacity.saturating_sub(rusage.capacity()));
            contention.reserve(enrich_cache_capacity.saturating_sub(contention.capacity()));
        }
        let identity = std::mem::take(&mut cache.identity_map);
        let visible = std::mem::take(&mut cache.visible_pids);
        let mach_ports = std::mem::take(&mut cache.mach_port_map);
        (
            cold,
            rusage,
            contention,
            identity,
            visible,
            mach_ports,
            !cache.initialized
                || cycle_count.saturating_sub(cache.visible_cycle_filled) >= visible_refresh_every,
            !cache.initialized
                || cycle_count.saturating_sub(cache.mach_ports_cycle_filled) >= mach_refresh_every,
        )
    };
    if refresh_visible {
        visible_pids = apollo_engine::engine::cg_window::visible_pids();
    }
    if refresh_mach_ports {
        mach_port_map = apollo_engine::engine::mach_qos::batch_mach_port_counts()
            .into_iter()
            .filter_map(|(pid, count)| (pid > 0).then_some((pid as u32, count)))
            .collect();
    }
    if cold_start {
        lf_metrics.inc_taskinfo_cache_miss();
    } else {
        lf_metrics.inc_taskinfo_cache_hit();
    }

    let mut live_pids: HashSet<u32> = HashSet::with_capacity(process_count);

    let expected_shard = if cold_start {
        enrich_cache_capacity
    } else {
        enrich_cache_capacity.div_ceil(taskinfo_refresh_period as usize)
    };
    let mut rusage_samples = Vec::with_capacity(expected_shard);
    for (pid, process) in sys.processes() {
        let pid_u32 = pid.as_u32();
        live_pids.insert(pid_u32);
        // Gate: enrich only PIDs with meaningful RSS or in the fg family.
        if process.memory() < ENRICH_MIN_RSS_BYTES && !fg_family.contains(&pid_u32) {
            continue;
        }
        if !should_refresh_taskinfo(
            pid_u32,
            cycle_count,
            taskinfo_refresh_period,
            cold_start,
            rusage_map.contains_key(&pid_u32),
        ) {
            continue;
        }

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
            rusage_samples.push((pid_u32, ri));
        } else {
            rusage_map.remove(&pid_u32);
            contention_map.remove(&pid_u32);
        }
    }

    // All kernel reads finish before the contention lock; the lock is acquired
    // once for the refreshed shard rather than once per PID.
    if let Ok(mut tracker) = apollo_engine::engine::contention_tracker::global().lock() {
        for (pid_u32, ri) in rusage_samples {
            if let Some(ratio) = tracker.observe(pid_u32, ri) {
                contention_map.insert(pid_u32, ratio);
            } else {
                contention_map.remove(&pid_u32);
            }
        }
        tracker.gc(&live_pids);
    }

    // Both vectors receive one entry per live process, so size them from the
    // actual table instead of repeatedly growing from historical M1 guesses.
    let mut proc_snaps = Vec::with_capacity(process_count);
    let mut hunt_snaps = Vec::with_capacity(process_count);
    let mut apple_platform_pids = HashSet::with_capacity(process_count);

    let now_instant = Instant::now();

    for (pid, process) in sys.processes() {
        let pid_u32 = pid.as_u32();
        let name = process.name().to_string();
        let is_foreground = fg_family.contains(&pid_u32);
        let ppid = process.parent().map(|p| p.as_u32()).unwrap_or(0);
        let parent_alive = parent_is_alive_or_process_root(
            ppid,
            sys.process(sysinfo::Pid::from_u32(ppid)).is_some(),
        );
        let is_zombie = process.status() == ProcessStatus::Zombie;
        let rss = process.memory();
        let cpu = process.cpu_usage();
        // process.start_time() → seconds since Unix epoch; 0 if unknown.
        let process_start_time = process.start_time();
        let process_uptime_secs = process.run_time();

        // Real idle wakeups from proc_pid_rusage — the #1 signal for wasteful daemons.
        // Estimate wakeups/sec: idle_wakeups is cumulative, divide by uptime estimate.
        // Mach messages > 0 implies the process has active IPC (network, XPC, etc.)
        let (wakeups_per_sec, has_network_signal, faults_total, pageins_total) =
            match rusage_map.get(&pid_u32) {
                Some(&(idle_wk, mach_msgs, faults, pageins)) => {
                    // Rusage counters are cumulative. Divide by the process's
                    // real uptime so the value remains a rate instead of
                    // growing forever with daemon age.
                    let wps = if process_uptime_secs > 0 && process_uptime_secs != u64::MAX {
                        (idle_wk as f64 / process_uptime_secs as f64).min(100.0) as f32
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
        let mut identity = identity_map
            .get(&pid_u32)
            .copied()
            .filter(|cached| process_start_time > 0 && cached.start_time == process_start_time)
            .unwrap_or_else(|| CachedProcessIdentity {
                start_time: process_start_time,
                is_app_bundle: apollo_engine::engine::proc_taskinfo::is_user_app_bundle(pid_u32)
                    .unwrap_or(false),
                is_translated: apollo_engine::engine::process_identity::is_translated(pid_u32),
                is_apple_platform:
                    apollo_engine::engine::process_identity::is_apple_platform_process(pid_u32),
                last_foreground_at: None,
            });
        if identity.is_apple_platform {
            apple_platform_pids.insert(pid_u32);
        }
        if is_foreground {
            identity.last_foreground_at = Some(now_instant);
        }
        let has_gui_window =
            is_foreground || visible_pids.contains(&pid_u32) || identity.is_app_bundle;
        let secs_since_foreground = if is_foreground {
            0
        } else if let Some(last) = identity.last_foreground_at {
            now_instant.duration_since(last).as_secs()
        } else if has_gui_window {
            // Unknown is not evidence of abandonment. A GUI process must be
            // observed leaving foreground before idle rules may act on it.
            0
        } else {
            3600
        };
        identity_map.insert(pid_u32, identity);

        proc_snaps.push(ProcessSnapshot {
            pid: pid_u32,
            name: name.clone(),
            cpu_percent: cpu,
            rss_bytes: rss,
            is_zombie,
            secs_since_foreground,
            secs_since_user_interaction: secs_since_foreground,
            has_network: has_network_signal,
            has_gui_window,
            wakeups_per_sec,
            parent_alive,
            process_uptime_secs,
            faults_total,
            pageins_total,
            is_translated: identity.is_translated,
            mach_port_count: mach_port_map.get(&pid_u32).copied().unwrap_or(0),
            cpu_contention: contention_map.get(&pid_u32).copied(),
            is_app_bundle: identity.is_app_bundle,
        });

        hunt_snaps.push(HuntSnapshot {
            pid: pid_u32,
            ppid,
            name,
            is_kernel_zombie: is_zombie,
            parent_alive,
            has_gui_window,
            rss_bytes: rss,
            cpu_percent: cpu,
            wakeups_per_sec,
            secs_since_user_interaction: secs_since_foreground,
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
    mach_port_map.retain(|pid, _| live_pids.contains(pid));
    visible_pids.retain(|pid| live_pids.contains(pid));

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
            mach_port_map.remove(pid);
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
        cache.visible_pids = visible_pids;
        cache.mach_port_map = mach_port_map;
        cache.cycle_filled = cycle_count;
        cache.initialized = true;
        if refresh_visible {
            cache.visible_cycle_filled = cycle_count;
        }
        if refresh_mach_ports {
            cache.mach_ports_cycle_filled = cycle_count;
        }
    }

    (proc_snaps, hunt_snaps, apple_platform_pids)
}

// ── Heuristic Decision Merger ──────────────────────────────────────────────

pub fn convert_and_merge_heuristic_decisions(
    decisions: &[ProcessDecision],
    existing_actions: &[RootAction],
    critical_pids: &HashSet<u32>,
    recently_applied: &RecentlyApplied,
    refusal_cooldown: &apollo_engine::engine::throttle_refusal::ThrottleRefusalCooldown,
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
        // Skip if Allow
        if decision.decision == GovernorDecision::Allow {
            continue;
        }

        // Skip critical processes
        if critical_pids.contains(&decision.pid) {
            continue;
        }

        // Complete Mediation guard — [Saltzer & Kaashoek 2009] §3.3: every path to a
        // privileged action must pass through the same access control point.
        // Count ZombieOrphan only after this guard: protected process-table
        // anomalies are not actionable instability and must not penalize RL.
        if is_protected_name(&decision.name) || is_interactive_app_name(&decision.name) {
            continue;
        }

        if decision.tier == ProcessTier::ZombieOrphan {
            stats.zombies_detected += 1;
        }

        // A real actionable suspect remains part of the live stability signal
        // even when another path already queued its action this cycle.
        if existing_pids.contains(&decision.pid) {
            continue;
        }

        // Cross-cycle state memory (SuperPlan 2026-05-06): if this PID had the
        // SAME decision admitted in a prior cycle within the last 30s,
        // suppress emission. This proposal stage is deliberately read-only;
        // the final dispatch chokepoint records surviving actions.
        // kernel would just say no-op ("PID already in target state") wasting
        // a syscall + journal entry. [Hellerstein 2004 §9] state-aware control.
        // CachedActionKind::from_governor maps Kill→Freeze automatically.
        if let Some(kind) = CachedActionKind::from_governor(decision.decision) {
            if recently_applied.is_recent(decision.pid, kind) {
                continue;
            }
        }

        // Map governor reason string → specific DecisionReason variant.
        // Closes NotebookLM Low-priority gap: PressureContext was 62.5%
        // catch-all; SwarmThrottling/GraduatedIdle differentiate two
        // well-known governor rule classes that account for ~20% of throttles.
        let dr = classify_governor_reason(&decision.reason);

        // Execution already refused a throttle for this PID and the refusal is
        // still warm. `recently_applied` above only remembers proposals that
        // were *admitted*, so without this a permanently-refused target is
        // re-proposed every cycle — 795 times for one process over 18 days,
        // every one refused, every one journalled.
        if decision.decision == GovernorDecision::Throttle
            && refusal_cooldown.is_in_cooldown(decision.pid)
        {
            apollo_engine::engine::lse_counters::LSE_COUNTERS.inc_throttle_refusal_suppressed();
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
                    decision_reason: dr.clone(),
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
                    decision_reason: dr.clone(),
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
                    decision_reason: dr,
                });
                stats.kills_downgraded += 1;
                stats.freezes += 1;
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

    #[test]
    fn process_roots_and_launchd_adoptions_are_not_dead_parents() {
        assert!(parent_is_alive_or_process_root(0, false));
        assert!(parent_is_alive_or_process_root(1, false));
        assert!(parent_is_alive_or_process_root(42, true));
        assert!(!parent_is_alive_or_process_root(42, false));
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

    fn make_zombie_decision(pid: u32, name: &str) -> ProcessDecision {
        let mut decision = make_decision(pid, name, GovernorDecision::Kill);
        decision.tier = ProcessTier::ZombieOrphan;
        decision
    }

    #[test]
    fn protected_zombie_suspects_do_not_penalize_stability() {
        let cache = RecentlyApplied::new();
        let decisions = vec![
            make_zombie_decision(404, "WindowServer"),
            make_zombie_decision(405, "unknown-critical-process"),
        ];
        let critical = HashSet::from([405]);

        let (actions, stats) = convert_and_merge_heuristic_decisions(
            &decisions,
            &[],
            &critical,
            &cache,
            &Default::default(),
        );
        assert!(actions.is_empty());
        assert_eq!(stats.zombies_detected, 0);
    }

    #[test]
    fn global_input_monitor_never_emits_heuristic_throttle() {
        let cache = RecentlyApplied::new();
        let decisions = vec![make_decision(
            972,
            "bare-modifier-monitor",
            GovernorDecision::Throttle,
        )];

        let (actions, stats) = convert_and_merge_heuristic_decisions(
            &decisions,
            &[],
            &HashSet::new(),
            &cache,
            &Default::default(),
        );

        assert!(actions.is_empty());
        assert_eq!(stats.throttles, 0);
    }

    #[test]
    fn actionable_zombie_suspect_remains_in_stability_signal() {
        let cache = RecentlyApplied::new();
        let decisions = vec![make_zombie_decision(12_345, "unprotected-orphan")];

        let (actions, stats) = convert_and_merge_heuristic_decisions(
            &decisions,
            &[],
            &HashSet::new(),
            &cache,
            &Default::default(),
        );
        assert_eq!(actions.len(), 1);
        assert_eq!(stats.zombies_detected, 1);
    }

    #[test]
    fn convert_and_merge_emits_first_throttle_normally() {
        let cache = RecentlyApplied::new();
        let critical = HashSet::new();
        let decisions = vec![make_decision(1234, "testproc", GovernorDecision::Throttle)];
        let (actions, stats) = convert_and_merge_heuristic_decisions(
            &decisions,
            &[],
            &critical,
            &cache,
            &Default::default(),
        );
        assert_eq!(actions.len(), 1);
        assert_eq!(stats.throttles, 1);
        assert!(
            !cache.is_recent(1234, CachedActionKind::Throttle),
            "proposal stages must not pre-record before final dispatch admission"
        );
    }

    #[test]
    fn convert_and_merge_suppresses_duplicate_within_ttl() {
        // Same decision for same PID across two calls — second call must drop.
        let mut cache = RecentlyApplied::new();
        let critical = HashSet::new();
        let decisions = vec![make_decision(1234, "testproc", GovernorDecision::Throttle)];

        // Simulate the final dispatch chokepoint recording cycle 1.
        cache.record(1234, CachedActionKind::Throttle);
        let (actions2, stats2) = convert_and_merge_heuristic_decisions(
            &decisions,
            &[],
            &critical,
            &cache,
            &Default::default(),
        );
        assert_eq!(actions2.len(), 0, "duplicate within TTL must be suppressed");
        assert_eq!(stats2.throttles, 0);
    }

    #[test]
    fn convert_and_merge_allows_freeze_after_throttle() {
        // Per-kind cache: a PID can be throttled, then later upgraded to freeze.
        let mut cache = RecentlyApplied::new();
        let critical = HashSet::new();

        let freeze = vec![make_decision(1234, "testproc", GovernorDecision::Freeze)];

        cache.record(1234, CachedActionKind::Throttle);

        // Freeze for SAME pid is a different cache key — should pass through.
        let (a2, _) = convert_and_merge_heuristic_decisions(
            &freeze,
            &[],
            &critical,
            &cache,
            &Default::default(),
        );
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
        let (a1, stats1) = convert_and_merge_heuristic_decisions(
            &kill,
            &[],
            &critical,
            &cache,
            &Default::default(),
        );
        assert_eq!(a1.len(), 1);
        assert_eq!(stats1.kills_downgraded, 1);
        assert!(!cache.is_recent(1234, CachedActionKind::Freeze));

        // The final chokepoint records the effective Kill→Freeze action.
        cache.record(1234, CachedActionKind::Freeze);
        let freeze = vec![make_decision(1234, "testproc", GovernorDecision::Freeze)];
        let (a2, _) = convert_and_merge_heuristic_decisions(
            &freeze,
            &[],
            &critical,
            &cache,
            &Default::default(),
        );
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
                    is_apple_platform: false,
                    last_foreground_at: None,
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
        let (cold_processes, cold_hunts, cold_apple_pids) =
            build_enriched_process_data_with_tree(&sys, None, &tree, 0, 0.30, 8, &metrics);
        let cold_elapsed = cold_started.elapsed();
        let identities_after_cold = enrich_syscall_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .identity_map
            .len();

        let warm_started = Instant::now();
        let (warm_processes, warm_hunts, warm_apple_pids) =
            build_enriched_process_data_with_tree(&sys, None, &tree, 1, 0.30, 8, &metrics);
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
        assert_eq!(warm_apple_pids, cold_apple_pids);
        let cold_app_bundle_pids: HashSet<u32> = cold_processes
            .iter()
            .filter(|process| process.is_app_bundle)
            .map(|process| process.pid)
            .collect();
        let warm_app_bundle_pids: HashSet<u32> = warm_processes
            .iter()
            .filter(|process| process.is_app_bundle)
            .map(|process| process.pid)
            .collect();
        assert_eq!(warm_app_bundle_pids, cold_app_bundle_pids);
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
    fn taskinfo_refresh_period_adapts_to_pressure_regime() {
        assert_eq!(taskinfo_refresh_period(0.30), 4);
        assert_eq!(taskinfo_refresh_period(0.50), 4);
        assert_eq!(taskinfo_refresh_period(0.65), 2);
        assert_eq!(taskinfo_refresh_period(0.80), 2);
        assert_eq!(taskinfo_refresh_period(0.81), 4);
    }

    #[test]
    fn rolling_taskinfo_shards_cover_every_pid_once_per_period() {
        const PERIOD: u32 = 4;
        for pid in 100..180 {
            let refreshes = (0..u64::from(PERIOD))
                .filter(|cycle| should_refresh_taskinfo(pid, *cycle, PERIOD, false, true))
                .count();
            assert_eq!(refreshes, 1, "pid {pid} must refresh exactly once");
        }

        assert!(should_refresh_taskinfo(101, 0, PERIOD, true, true));
        assert!(should_refresh_taskinfo(101, 0, PERIOD, false, false));
    }
}
