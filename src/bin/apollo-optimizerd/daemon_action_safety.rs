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
use apollo_engine::engine::decide_actions::is_interactive_app_name;
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
        behavioral_protection_score, cached_policy_protected_ac, classify_protection_canonical,
        is_chromium_family, is_protected_name, is_user_interactive_app, matches_dev_runtime,
        ProtectionLevel,
    },
};
use chrono::Utc;

use apollo_engine::engine::process_tree::ProcessTree;

use crate::process_enrichment::{
    build_foreground_family, convert_and_merge_heuristic_decisions, HeuristicStats,
};
use apollo_engine::engine::recently_applied::{CachedActionKind, RecentlyApplied};

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

/// Reuse the complete heuristic protection envelope before a secondary
/// producer emits a per-process action. The final dispatcher remains the
/// authority, but rejecting here avoids doomed proposals and polluted learning.
pub fn is_protected_action_candidate(
    pid: u32,
    name: &str,
    heuristic_critical_pids: &HashSet<u32>,
) -> bool {
    heuristic_critical_pids.contains(&pid)
        || is_protected_name(name)
        || is_interactive_app_name(name)
        || is_chromium_family(name)
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
    total_ram_bytes: u64,
    recently_applied: &RecentlyApplied,
    apple_platform_pids: &HashSet<u32>,
) -> HeuristicPassOutput {
    const HIGH_TAU_SEC: f64 = 300.0;

    // ── heuristic_critical_pids: behavioral protection scoring ───────────────
    // [Saltzer & Kaashoek 2009] Complete Mediation — single callsite for all
    // protection decisions. Infrastructure always protected; dev runtimes earn
    // protection by behavioral activity score ≥ current pressure.
    let sys = collector.system();
    let snap_by_pid: HashMap<u32, &ProcessSnapshot> =
        proc_snaps.iter().map(|snap| (snap.pid, snap)).collect();
    let policy_protected = state
        .policy
        .lock_recover()
        .learned_policy
        .protected_patterns
        .clone();
    let policy_protected_ac = cached_policy_protected_ac(&policy_protected);
    let mut heuristic_critical_pids = apple_platform_pids.clone();
    let mut behavioral_candidates = Vec::new();
    let apple_platform_protected = apple_platform_pids.len() as u64;

    // One linear pass over sysinfo. The previous `iter().find()` made this
    // O(processes^2) and allocated a String for every process.
    for (pid, process) in sys.processes() {
        let pid_u32 = pid.as_u32();
        let name = process.name();
        // The kernel will reject Apollo's throttle/freeze actuators for Apple
        // platform binaries under SIP. Mark them protected before governor
        // scoring so we do not manufacture hundreds of doomed actions and
        // discard them again at the final dispatch filter.
        if apple_platform_pids.contains(&pid_u32) {
            continue;
        }
        let snap = snap_by_pid.get(&pid_u32).copied();
        let has_gui = snap.is_some_and(|s| s.has_gui_window);
        let idle_s = snap.map_or(3600, |s| s.secs_since_user_interaction);
        let rss = snap.map_or(process.memory(), |s| s.rss_bytes);
        let is_interactive = is_user_interactive_app(has_gui, idle_s, rss, name);

        match classify_protection_canonical(
            name,
            &policy_protected,
            policy_protected_ac.as_deref(),
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
                    cpu,
                    wakeups,
                    net,
                    gui,
                    idle_s,
                    rss,
                    total_ram_bytes,
                ),
            });
        }
    }
    heuristic_critical_pids.extend(amx_detector::ml_protected_pids_cached().iter().copied());

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
        pg.adaptive_governor
            .decide_actionable_with_hw_and_protected(
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
        m.metrics.heuristic_apple_platform_protected += apple_platform_protected;
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

/// Breakdown of the universal pre-budget dispatch filter.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DispatchPrefilterStats {
    pub input: u64,
    pub recent_drops: u64,
    pub apple_drops: u64,
    pub protected_drops: u64,
    pub identity_drops: u64,
    pub survivors: u64,
}

/// Remove actions that cannot safely reach the kernel before they consume the
/// per-cycle or per-minute action budget. The two closures keep platform and
/// identity probes injectable so the ordering invariant is unit-testable.
pub fn filter_dispatch_candidates<FApple, FIdentity>(
    actions: Vec<RootAction>,
    policy_protected: &[String],
    recently_applied: &RecentlyApplied,
    mut is_apple_platform: FApple,
    mut identity_valid: FIdentity,
) -> (Vec<RootAction>, DispatchPrefilterStats)
where
    FApple: FnMut(u32) -> bool,
    FIdentity: FnMut(&RootAction) -> bool,
{
    let mut stats = DispatchPrefilterStats {
        input: actions.len() as u64,
        ..DispatchPrefilterStats::default()
    };
    let mut filtered = Vec::with_capacity(actions.len());
    let policy_protected_ac = cached_policy_protected_ac(policy_protected);

    for action in actions {
        if let Some((pid, kind, discriminator)) = CachedActionKind::from_root_action(&action) {
            if recently_applied.is_recent_scoped(pid, kind, discriminator) {
                stats.recent_drops += 1;
                continue;
            }

            let blocks_under_sip = matches!(
                kind,
                CachedActionKind::Throttle
                    | CachedActionKind::Freeze
                    | CachedActionKind::Unfreeze
                    | CachedActionKind::SetThreadQoS
            );
            if blocks_under_sip && is_apple_platform(pid) {
                stats.apple_drops += 1;
                continue;
            }

            let blocks_for_protected =
                !matches!(kind, CachedActionKind::Boost | CachedActionKind::Unfreeze);
            if blocks_for_protected {
                let action_name = match &action {
                    RootAction::ThrottleProcess { name, .. }
                    | RootAction::FreezeProcess { name, .. }
                    | RootAction::SetThreadQoS { name, .. }
                    | RootAction::BoostProcess { name, .. }
                    | RootAction::UnfreezeProcess { name, .. } => Some(name.as_str()),
                    _ => None,
                };
                if action_name.is_some_and(|name| {
                    classify_protection_canonical(
                        name,
                        policy_protected,
                        policy_protected_ac.as_deref(),
                        false,
                    ) == ProtectionLevel::Unconditional
                }) {
                    stats.protected_drops += 1;
                    continue;
                }
            }

            if !identity_valid(&action) {
                stats.identity_drops += 1;
                continue;
            }
        }

        stats.survivors += 1;
        filtered.push(action);
    }

    (filtered, stats)
}

#[cfg(test)]
mod dispatch_prefilter_tests {
    use super::*;
    use apollo_engine::engine::audit_types::DecisionReason;
    use apollo_engine::engine::safety::enforce_limits_with_budget;
    use apollo_engine::engine::types::{ActionBudgetState, OptimizationProfile, SafetyPolicy};

    fn throttle(pid: u32, name: impl Into<String>) -> RootAction {
        RootAction::ThrottleProcess {
            pid,
            name: name.into(),
            aggressive: false,
            reason: "test".to_string(),
            start_sec: 1,
            start_usec: 0,
            decision_reason: DecisionReason::PressureContext,
        }
    }

    #[test]
    fn secondary_producers_reuse_the_complete_protection_envelope() {
        let critical = HashSet::from([77]);

        assert!(is_protected_action_candidate(
            77,
            "new-macos-daemon",
            &critical
        ));
        assert!(is_protected_action_candidate(78, "trustd", &HashSet::new()));
        assert!(is_protected_action_candidate(
            79,
            "Brave Browser Helper (Renderer)",
            &HashSet::new(),
        ));
        assert!(!is_protected_action_candidate(
            80,
            "thirdparty-idle-worker",
            &HashSet::new(),
        ));
    }

    #[test]
    fn kernel_rejected_prefix_cannot_starve_later_executable_action() {
        let mut actions = Vec::new();
        for pid in 1_000..1_080 {
            actions.push(throttle(pid, format!("platform-job-{pid}")));
        }
        actions.push(throttle(9_999, "thirdparty-batch-worker"));

        let recent = RecentlyApplied::new();
        let (filtered, stats) = filter_dispatch_candidates(
            actions,
            &[],
            &recent,
            |pid| (1_000..1_080).contains(&pid),
            |_| true,
        );

        assert_eq!(stats.input, 81);
        assert_eq!(stats.apple_drops, 80);
        assert_eq!(stats.survivors, 1);
        let policy = SafetyPolicy::for_profile(OptimizationProfile::BalancedRoot);
        let admitted =
            enforce_limits_with_budget(filtered, &policy, &mut ActionBudgetState::default(), 80);
        assert_eq!(admitted.len(), 1);
        assert!(matches!(
            &admitted[0],
            RootAction::ThrottleProcess { pid: 9_999, .. }
        ));
    }

    #[test]
    fn reports_recent_protected_and_identity_rejections_separately() {
        let mut recent = RecentlyApplied::new();
        recent.record(10, CachedActionKind::Throttle);
        let actions = vec![
            throttle(10, "recent-thirdparty-job"),
            throttle(11, "WindowServer"),
            throttle(12, "identity-mismatch-job"),
            throttle(13, "surviving-thirdparty-job"),
        ];

        let (filtered, stats) = filter_dispatch_candidates(
            actions,
            &[],
            &recent,
            |_| false,
            |action| !matches!(action, RootAction::ThrottleProcess { pid: 12, .. }),
        );

        assert_eq!(stats.recent_drops, 1);
        assert_eq!(stats.protected_drops, 1);
        assert_eq!(stats.identity_drops, 1);
        assert_eq!(stats.survivors, 1);
        assert_eq!(filtered.len(), 1);
    }
}
