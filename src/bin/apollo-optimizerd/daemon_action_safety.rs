//! # Daemon Action Safety
//!
//! Heuristic protection pass extracted from main.rs (Wave 13).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Run AdaptiveGovernor heuristic pass (ODE swap risk + high-τ PIDs wired)
//! - Compute `heuristic_critical_pids` via behavioral protection scoring
//!   [Saltzer & Kaashoek 2009] Complete Mediation — single callsite for protection
//! - Merge heuristic actions and filter via Cable 2 experience gate
//!
//! ## Ordering invariant
//! Must run AFTER `signal_digest`, `reclaim_forecast`, and `behavior_interactive_pids`
//! are computed for this cycle, and AFTER `decide_actions` has produced `actions`.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

use apollo_engine::collector::SystemCollector;
use apollo_engine::engine::adaptive_governor::ProcessDecision;
use apollo_engine::engine::daemon_helpers::audit_log_batch;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::foreground::ForegroundDetector;
use apollo_engine::engine::hw_bayes::HwFeatures;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::outcome_tracker::ExperienceMemory;
use apollo_engine::engine::process_classifier::ProcessSnapshot;
use apollo_engine::engine::swap_reclaim::SaturationForecast;
use apollo_engine::engine::types::RootAction;
use apollo_engine::engine::unfreeze_decay::UnfreezeDecayModel;
use apollo_engine::engine::zombie_hunter::HuntSnapshot;
use apollo_engine::engine::{
    amx_detector,
    safety::{
        behavioral_protection_score, classify_protection, infrastructure_processes,
        is_user_interactive_app, matches_dev_runtime, protected_processes, ProtectionLevel,
    },
};
use chrono::Utc;

use apollo_engine::engine::process_tree::ProcessTree;

use crate::process_enrichment::{
    build_foreground_family, convert_and_merge_heuristic_decisions, HeuristicStats,
};
use apollo_engine::engine::recently_applied::RecentlyApplied;

pub struct HeuristicPassOutput {
    pub heuristic_decisions: Vec<ProcessDecision>,
    pub heuristic_critical_pids: HashSet<u32>,
    pub heuristic_stats: HeuristicStats,
    pub additional_actions: Vec<RootAction>,
}

struct BehavioralCandidate {
    pid: u32,
    name: String,
    cpu: f32,
    wakeups: f32,
    net: bool,
    gui: bool,
    idle_s: u64,
    rss: u64,
    raw_score: f64,
}

/// Heuristic protection pass — runs AdaptiveGovernor, scores behavioral protection,
/// merges heuristic actions, and applies the Cable 2 experience gate.
///
/// # Parameters
/// - `proc_snaps` / `hunt_snaps` — per-process snapshot slices
/// - `foreground_app` / `foreground_pid` — current foreground context
/// - `all_proc_names` — flat name list for AdaptiveGovernor
/// - `hour_of_day` — for nocturnal scheduling rules in AdaptiveGovernor
/// - `hw_features` — hardware Bayesian features (sampled every 5 cycles)
/// - `state` — SharedState (policy, metrics locks)
/// - `pressure_smooth` — EMA pressure from signal_digest (BPS threshold)
/// - `unfreeze_decay` — ODE model for high-τ PID identification
/// - `reclaim_forecast` — swap saturation forecast (risk + t_sat_sec)
/// - `collector` — SystemCollector (sysinfo process iterator)
/// - `current_actions` — actions accumulated so far (for Cable 2 dedup)
/// - `experience` — ExperienceMemory for Cable 2 throttle outcome gate
/// - `experience_pressure_band` — learnable_params band for query_similar_with_band
/// - `current_pressure` — raw memory pressure (snapshot.pressure.memory_pressure)
#[allow(clippy::too_many_arguments)]
pub fn run_heuristic_pass(
    proc_snaps: &[ProcessSnapshot],
    hunt_snaps: &[HuntSnapshot],
    foreground_app: Option<&str>,
    foreground_pid: Option<u32>,
    all_proc_names: &[&str],
    hour_of_day: u8,
    hw_features: Option<HwFeatures>,
    state: &SharedState,
    pressure_smooth: f64,
    unfreeze_decay: &UnfreezeDecayModel,
    reclaim_forecast: &SaturationForecast,
    collector: &SystemCollector,
    current_actions: &[RootAction],
    experience: &ExperienceMemory,
    experience_pressure_band: f64,
    current_pressure: f64,
    recently_applied: &mut RecentlyApplied,
) -> HeuristicPassOutput {
    const HIGH_TAU_SEC: f64 = 300.0;

    // ── heuristic_critical_pids: behavioral protection scoring ───────────────
    // [Saltzer & Kaashoek 2009] Complete Mediation — single callsite for all
    // protection decisions. Infrastructure always protected; dev runtimes earn
    // protection by behavioral activity score ≥ current pressure.
    let sys = collector.system();
    let snap_by_pid: HashMap<u32, &ProcessSnapshot> =
        proc_snaps.iter().map(|snap| (snap.pid, snap)).collect();
    let infra_pats = infrastructure_processes();
    let protected_pats = protected_processes();
    let policy_protected = state
        .policy
        .lock_recover()
        .learned_policy
        .protected_patterns
        .clone();
    let policy_protected_ac =
        apollo_engine::engine::safety::build_policy_protected_ac(&policy_protected);
    let total_ram = apollo_engine::engine::sysctl_direct::read_u64("hw.memsize")
        .unwrap_or(8 * 1024 * 1024 * 1024);
    let mut heuristic_critical_pids: HashSet<u32> = HashSet::new();
    let mut behavioral_candidates = Vec::new();

    // One linear pass over sysinfo. The previous `iter().find()` made this
    // O(processes^2) and allocated a String for every process.
    for (pid, process) in sys.processes() {
        let pid_u32 = pid.as_u32();
        let name = process.name();
        let snap = snap_by_pid.get(&pid_u32).copied();
        let has_gui = snap.is_some_and(|s| s.has_gui_window);
        let idle_s = snap.map_or(3600, |s| s.secs_since_user_interaction);
        let rss = snap.map_or(process.memory(), |s| s.rss_bytes);
        let is_interactive = is_user_interactive_app(has_gui, idle_s, rss, name);

        match classify_protection(
            name,
            &protected_pats,
            &infra_pats,
            &policy_protected,
            policy_protected_ac.as_ref(),
            is_interactive,
        ) {
            ProtectionLevel::Unconditional => {
                heuristic_critical_pids.insert(pid_u32);
                continue;
            }
            ProtectionLevel::ConditionalForeground => {
                if Some(pid_u32) == foreground_pid {
                    heuristic_critical_pids.insert(pid_u32);
                }
                continue;
            }
            ProtectionLevel::Unprotected => {}
        }

        if matches_dev_runtime(name) {
            let (cpu, wakeups, net, gui) = snap.map_or_else(
                || (process.cpu_usage(), 0.0, false, false),
                |s| {
                    (
                        s.cpu_percent,
                        s.wakeups_per_sec,
                        s.has_network,
                        s.has_gui_window,
                    )
                },
            );
            behavioral_candidates.push(BehavioralCandidate {
                pid: pid_u32,
                name: name.to_string(),
                cpu,
                wakeups,
                net,
                gui,
                idle_s,
                rss,
                raw_score: behavioral_protection_score(
                    cpu, wakeups, net, gui, idle_s, rss, total_ram,
                ),
            });
        }
    }
    heuristic_critical_pids.extend(amx_detector::ml_protected_pids());

    // ── AdaptiveGovernor heuristic pass ─────────────────────────────────────
    // Name/policy-protected PIDs are mediated before action scoring. Learned
    // behavioral candidates are finalized immediately afterwards, once this
    // cycle's user-profile observation has been applied by the governor.
    let heuristic_decisions: Vec<ProcessDecision> = {
        let mut pg = state.policy.lock_recover();
        pg.adaptive_governor.swap_risk = reclaim_forecast.risk;
        pg.adaptive_governor.high_tau_pids = proc_snaps
            .iter()
            .filter(|s| unfreeze_decay.tau_for_app(&s.name) > HIGH_TAU_SEC)
            .map(|s| s.pid)
            .collect();
        pg.adaptive_governor.decide_all_with_hw_and_protected(
            proc_snaps,
            hunt_snaps,
            foreground_app,
            all_proc_names,
            hour_of_day,
            hw_features,
            &heuristic_critical_pids,
        )
    };

    let relevances: Vec<f32> = {
        let pg = state.policy.lock_recover();
        behavioral_candidates
            .iter()
            .map(|candidate| {
                pg.adaptive_governor
                    .user_profile
                    .process_relevance(&candidate.name)
            })
            .collect()
    };
    let mut bps_prot = 0u64;
    let mut bps_dem = 0u64;
    let mut bps_min = f64::MAX;
    let mut bps_min_name = String::new();
    let audit_timestamp = Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    let mut audit_entries = Vec::with_capacity(behavioral_candidates.len());

    for (candidate, relevance) in behavioral_candidates.iter().zip(relevances) {
        let score = candidate.raw_score + (relevance as f64 * 0.15);
        let protected = score >= pressure_smooth;
        if score < bps_min {
            bps_min = score;
            bps_min_name = format!("{}({})", candidate.name, candidate.pid);
        }
        audit_entries.push(serde_json::json!({
            "t": audit_timestamp.as_str(),
            "event": "bps_eval",
            "pid": candidate.pid,
            "name": candidate.name.as_str(),
            "score": (score * 10000.0).round() / 10000.0,
            "raw_score": (candidate.raw_score * 10000.0).round() / 10000.0,
            "relevance": (relevance * 100.0).round() / 100.0,
            "pressure": (pressure_smooth * 1000.0).round() / 1000.0,
            "protected": protected,
            "cpu": candidate.cpu,
            "wakeups": candidate.wakeups,
            "net": candidate.net,
            "gui": candidate.gui,
            "idle_s": candidate.idle_s,
            "rss_mb": candidate.rss / 1024 / 1024,
        }));
        if protected {
            bps_prot += 1;
            heuristic_critical_pids.insert(candidate.pid);
        } else {
            bps_dem += 1;
        }
    }
    audit_log_batch(&audit_entries);
    {
        let mut m = state.metrics.lock_recover();
        m.metrics.bps_evaluated += behavioral_candidates.len() as u64;
        m.metrics.bps_protected += bps_prot;
        m.metrics.bps_demoted += bps_dem;
        if bps_min < f64::MAX {
            m.metrics.bps_min_score = bps_min;
            m.metrics.bps_min_score_name = bps_min_name;
        }
    }

    // ── Merge + Cable 2 experience gate ─────────────────────────────────────
    // Cable 2: skip throttles that experience shows never reduce pressure.
    // [Sutton & Barto 2018] experience replay informs action selection.
    let (heuristic_actions, heuristic_stats) = convert_and_merge_heuristic_decisions(
        &heuristic_decisions,
        current_actions,
        &heuristic_critical_pids,
        recently_applied,
    );
    let additional_actions: Vec<RootAction> = heuristic_actions
        .into_iter()
        .filter(|a| {
            if let RootAction::ThrottleProcess { ref name, .. } = a {
                if let Some((avg_drop, confidence)) = experience.query_similar_with_band(
                    name,
                    current_pressure,
                    experience_pressure_band,
                ) {
                    if confidence >= 0.5 && avg_drop <= 0.0 {
                        return false;
                    }
                }
            }
            true
        })
        .collect();

    HeuristicPassOutput {
        heuristic_decisions,
        heuristic_critical_pids,
        heuristic_stats,
        additional_actions,
    }
}

/// F3 + F4 safety filters — extracted from main.rs (Wave 36).
/// [Fowler 2004] Strangler Fig — pure move, no semantic change.
///
/// F3 — Safety Precedence: foreground family and recently-active apps are
/// never throttled or frozen. Protects the user's active context.
///
/// F4 — Thermal Master Switch: suppress Boost actions during thermal
/// emergency (>95°C P-cluster) or resource interrupt Emergency/SuperEmergency.
///
/// Both filters mutate `actions` in place; ordering (F3 before F4) is stable.
pub fn apply_pre_exec_safety_filters(
    actions: &mut Vec<RootAction>,
    foreground_pid: Option<u32>,
    process_tree: &ProcessTree,
    foreground_app: Option<&str>,
    fg_detector: &ForegroundDetector,
    thermal_emergency: bool,
    state: &SharedState,
) {
    // F3 — foreground family + recently-active protection.
    {
        let fg_family_pids = build_foreground_family(foreground_pid, process_tree);
        let recently_active_window = std::time::Duration::from_secs(300);
        actions.retain(|a| match a {
            RootAction::ThrottleProcess { pid, name, .. }
            | RootAction::FreezeProcess { pid, name, .. } => {
                if fg_family_pids.contains(pid) {
                    return false;
                }
                if let Some(fg) = foreground_app {
                    if name.contains(fg) {
                        return false;
                    }
                }
                if fg_detector.is_recently_active(name, recently_active_window) {
                    return false;
                }
                true
            }
            _ => true,
        });
    }

    // F4 — thermal / resource-interrupt Boost suppression.
    let interrupt_phase = state.resource_interrupt.phase.load(Ordering::Acquire);
    if thermal_emergency || interrupt_phase >= 2 {
        actions.retain(|a| !matches!(a, RootAction::BoostProcess { .. }));
    }
}
