//! # Daemon Feature Gates
//!
//! Cohesive home for macOS-specific per-cycle feature blocks extracted from
//! `apollo-optimizerd`'s hot loop during the V1.1.0 Strangler Fig pass
//! [Fowler 2004].
//!
//! The five features share a common shape — read some cycle-local signals,
//! make a per-PID decision, push the result into `mach_qos`/Spotlight — but
//! are functionally independent of each other. Keeping them side-by-side here
//! makes their invariants easy to audit:
//!
//! - **F1 LLM Inference Mode**: detect ollama/llama.cpp/MLX/LM Studio and
//!   toggle Spotlight + emit a pressure boost.
//! - **F3 RT Boost**: apply THREAD_TIME_CONSTRAINT_POLICY to the foreground
//!   UI thread — unless thermal Phase3+ has requested E-core exile.
//! - **F4 Post-Wake Suppression**: if the inter-cycle gap jumps > 30s, the
//!   daemon assumes the system slept; opens a 60s grace window so foreground
//!   apps can restore state before background apps get CPU.
//! - **F5 Wakeup Budget**: graduated response to wake-storms — Critical/High
//!   → App Nap, Medium → Background tier, Low → monitor-only.
//! - **F2+F4 App-Nap Scheduling**: while LLM inference is active *or* we're
//!   inside the post-wake grace, aggressively App-Nap every non-protected
//!   non-foreground process.
//!
//! ## Inter-feature invariants (peer-review 2026-04-18)
//!
//! - **RT Boost persists across post-wake grace**: F3 runs every cycle; F4
//!   only *suppresses* background apps. Foreground fluidity is guaranteed
//!   immediately after wake.
//! - **Thermal Phase3+ vetoes RT Boost**: force_ecores wins — no P-core
//!   pinning while the package is hot.
//! - **LLM Mode respects freeze_protected**: App-Nap is applied only through
//!   `classify_protection`, which already honours OS/infra/user-interactive
//!   protection tiers.
//! - **Graduated wake-storm response**: StormSeverity::Medium routes to
//!   Background (E-cores) *without* App-Nap timer suppression.
//!   [Nygard 2018 "Release It!" Ch.5]
//! - **App-Nap release bypasses active storms**: when neither LLM nor wake
//!   suppression is active, we release every App-Nap *except* PIDs still in
//!   a wake storm — prevents F5's work from being undone by F2+F4.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use apollo_optimizer::collector::{SystemCollector, SystemSnapshot};
// spotlight_set_indexing import removed 2026-05-08: Apollo no longer
// touches mdutil automatically (user-reported Finder beachball regression).
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::llm_inference_mode::LlmInferenceDetector;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::mach_qos::SchedulingTier;
use apollo_optimizer::engine::process_classifier::ProcessSnapshot;
use apollo_optimizer::engine::safety::{
    classify_protection, infrastructure_processes, is_user_interactive_app, protected_processes,
    ProtectionLevel,
};
use apollo_optimizer::engine::thermal_bailout::{CoolingPhase, ThermalAction};
use apollo_optimizer::engine::wake_storm_detector::{
    StormSeverity, WakePattern, WakeStormDetector,
};

/// Result of the Feature 1 per-cycle tick.
pub struct LlmInferenceOutcome {
    /// Additive pressure boost (0.0..=0.20) contributed to the aggregator.
    pub llm_boost: f64,
    /// True when the detector currently classifies an LLM workload as active.
    pub llm_active: bool,
}

/// Feature 1: LLM Inference Mode tick.
///
/// Feeds the top-process iterator into `LlmInferenceDetector::observe`,
/// toggles Spotlight indexing when the active→inactive edge is crossed,
/// and returns the pressure boost + active flag for the aggregator.
///
/// ## Ordering
/// Runs AFTER the sensor tick (needs `snapshot.top_processes`) and BEFORE
/// `daemon_pressure_aggregator` (consumes `llm_boost`).
pub fn run_llm_inference_mode_tick(
    snapshot: &SystemSnapshot,
    llm_detector: &mut LlmInferenceDetector,
    llm_spotlight_disabled: &mut bool,
    is_root: bool,
) -> LlmInferenceOutcome {
    let llm_boost = {
        let proc_iter = snapshot
            .top_processes
            .iter()
            .map(|p| (p.pid, p.name.as_str(), p.cpu_usage));
        llm_detector.observe(proc_iter);
        llm_detector.pressure_boost()
    };
    let llm_active = llm_detector.is_active();

    // Spotlight toggle disabled (2026-05-08): user reported Finder beachball
    // when Apollo flipped mdutil during LLM bursts. The off→on edge invalidates
    // Spotlight metadata caches and any open Finder window stalls waiting on
    // mds. We keep the LLM detection + pressure boost, but never touch
    // `mdutil` automatically. Re-enable behind a config flag if ever needed.
    let _ = is_root;
    let _ = llm_spotlight_disabled;

    LlmInferenceOutcome {
        llm_boost,
        llm_active,
    }
}

/// Feature 3: RT Boost for foreground UI thread.
///
/// Applies THREAD_TIME_CONSTRAINT_POLICY (2ms/10ms) to the foreground PID
/// so UI frames hit their deadlines even while background CPU hogs run.
///
/// Skipped entirely during Phase3+ thermal bailout — pinning to P-cores
/// while the package is overheating would defeat the cooling strategy.
///
/// Clears any prior RT boost when foreground changes or disappears.
pub fn apply_rt_boost_foreground(
    state: &SharedState,
    thermal_action: &ThermalAction,
    foreground_pid: Option<u32>,
    rt_boosted_pid: &mut Option<u32>,
) {
    if thermal_action.phase >= CoolingPhase::Phase3Aggressive {
        return;
    }
    if let Some(fg_pid) = foreground_pid {
        if *rt_boosted_pid != Some(fg_pid) {
            // Clear RT boost from previous foreground.
            if let Some(old_pid) = *rt_boosted_pid {
                let mut qos = state.mach_qos.lock_recover();
                qos.clear_realtime_boost(old_pid);
            }
            // Apply RT boost to new foreground.
            let mut qos = state.mach_qos.lock_recover();
            if qos.set_realtime_boost(fg_pid) {
                *rt_boosted_pid = Some(fg_pid);
            } else {
                *rt_boosted_pid = None;
            }
        }
    } else if let Some(old_pid) = *rt_boosted_pid {
        // No foreground — clear boost.
        let mut qos = state.mach_qos.lock_recover();
        qos.clear_realtime_boost(old_pid);
        *rt_boosted_pid = None;
    }
}

/// Feature 4: Post-Wake Suppression tick.
///
/// If the last cycle was > 30s ago, the daemon assumes the system slept.
/// Opens a 60s grace window during which App-Nap is aggressively applied
/// to non-foreground processes so the foreground app restores state first,
/// then returns `in_wake_suppression` for downstream consumers (F2+F4).
///
/// This is complementary to `daemon_wake_handler::run_wake_tick` which
/// fires on a > 90s wall-clock jump: that path handles Kalman/outcome
/// reset + staggered SIGCONT drain. F4 is the lighter per-cycle App-Nap
/// heuristic and does NOT reset long-lived filter state.
pub fn apply_post_wake_suppression(
    state: &SharedState,
    last_cycle_instant: Instant,
    wake_suppression_until: &mut Option<Instant>,
) -> bool {
    let elapsed_since_last_cycle = last_cycle_instant.elapsed();
    if elapsed_since_last_cycle > Duration::from_secs(30) {
        *wake_suppression_until = Some(Instant::now() + Duration::from_secs(60));
        println!(
            "[wake] System woke from sleep ({}s gap) — 60s background suppression active",
            elapsed_since_last_cycle.as_secs()
        );
        // Release any App Nap set before sleep; re-evaluate fresh.
        let mut qos = state.mach_qos.lock_recover();
        qos.release_all_app_nap();
    }
    // NOTE: the caller must NOT reset last_cycle_instant here — it must
    // span the full inter-cycle interval so that cycle_dt_secs (computed
    // later) reflects the real wall-clock gap between cycles, not just
    // intra-cycle work time.
    wake_suppression_until
        .map(|t| Instant::now() < t)
        .unwrap_or(false)
}

/// Feature 5: Wakeup Budget Enforcer.
///
/// Graduated severity response to wake-storms:
/// - Critical / High → App Nap (timer suppression)
/// - Medium → SchedulingTier::Background (E-core routing, no suppression)
/// - Low → monitor-only
///
/// Also releases App-Nap for PIDs that have calmed down (were in a storm
/// but are no longer). Skips heuristic-critical PIDs and the current
/// foreground.
///
/// Returns the raw storms vector so F2+F4 can reuse the snapshot for
/// release filtering (avoids a double scan).
///
/// [Nygard 2018 "Release It!" Ch.5 — graduated response]
pub fn enforce_wakeup_budget(
    state: &SharedState,
    wake_storm: &mut WakeStormDetector,
    heuristic_critical_pids: &HashSet<u32>,
    foreground_pid: Option<u32>,
) -> Vec<WakePattern> {
    let storms = wake_storm.detect_storms();
    let storm_pids: HashSet<u32> = storms.iter().map(|s| s.pid).collect();
    let mut qos = state.mach_qos.lock_recover();

    // Apply severity-graduated mitigation.
    for storm in &storms {
        if heuristic_critical_pids.contains(&storm.pid) || Some(storm.pid) == foreground_pid {
            continue;
        }
        let severity = wake_storm.get_severity(storm.wakeups_per_second);
        match severity {
            StormSeverity::Critical | StormSeverity::High => {
                qos.set_app_nap(storm.pid, true);
            }
            StormSeverity::Medium => {
                // E-core routing without full App Nap suppression.
                qos.set_tier(storm.pid, SchedulingTier::Background);
            }
            StormSeverity::Low => {
                // Below threshold: monitor only, no intervention.
            }
        }
    }

    // Release App Nap for pids that are no longer in a storm.
    // (gc_dead_pids handles dead pids; this handles calmed pids)
    let app_napped_snapshot: Vec<u32> = qos
        .current_tier_keys()
        .iter()
        .filter(|(pid, _)| qos.is_app_napped(*pid))
        .map(|(pid, _)| *pid)
        .collect();
    for pid in app_napped_snapshot {
        if !storm_pids.contains(&pid) {
            qos.set_app_nap(pid, false);
        }
    }

    storms
}

/// Feature 2 + 4: App Nap scheduling for LLM mode and post-wake window.
///
/// Two modes:
///
/// 1. **llm_active || in_wake_suppression**: iterate every process, App-Nap
///    every non-protected non-foreground one (Apollo-self is always skipped).
///    Protected apps already napped get released.
/// 2. **Neither**: release any LLM/wake App-Nap that isn't also a live
///    wake-storm offender — the latter is the F5 invariant.
///
/// Protection classification uses `classify_protection` with hard/infra
/// pattern sets + the learned protected_patterns + behavioural
/// is_user_interactive_app (GUI + recent interaction + RSS) heuristic.
pub fn apply_app_nap_scheduling(
    state: &SharedState,
    collector: &SystemCollector,
    proc_snaps: &[ProcessSnapshot],
    foreground_pid: Option<u32>,
    llm_active: bool,
    in_wake_suppression: bool,
    storms: &[WakePattern],
) {
    if llm_active || in_wake_suppression {
        let appnap_hard = protected_processes();
        let appnap_infra = infrastructure_processes();
        let appnap_policy = state
            .policy
            .lock_recover()
            .learned_policy
            .protected_patterns
            .clone();
        let mut qos = state.mach_qos.lock_recover();
        for (pid, process) in collector.system().processes() {
            let pid_u32 = pid.as_u32();
            let name = process.name();
            let is_foreground = Some(pid_u32) == foreground_pid;
            // Evaluate behavioral signals for Tier-4 interactive detection.
            let snap = proc_snaps.iter().find(|s| s.pid == pid_u32);
            let has_gui = snap.map_or(false, |s| s.has_gui_window);
            let idle_s = snap.map_or(3600, |s| s.secs_since_user_interaction);
            let rss = snap.map_or(process.memory(), |s| s.rss_bytes);
            let is_interactive = is_user_interactive_app(has_gui, idle_s, rss, name);
            let protection = classify_protection(
                name,
                &appnap_hard,
                &appnap_infra,
                &appnap_policy,
                is_interactive,
            );
            // Apollo itself is never app-napped (self-protection).
            // Unconditional: OS/infra/policy — always skip.
            // ConditionalForeground: user-interactive apps — skip only when foreground.
            let should_protect = name == "apollo-optimizerd"
                || protection == ProtectionLevel::Unconditional
                || (protection == ProtectionLevel::ConditionalForeground && is_foreground);
            if should_protect {
                // Protected: ensure NOT app-napped.
                if qos.is_app_napped(pid_u32) {
                    qos.set_app_nap(pid_u32, false);
                }
                continue;
            }
            // Skip if already app-napped (dedup).
            if !qos.is_app_napped(pid_u32) {
                qos.set_app_nap(pid_u32, true);
            }
        }
    } else {
        // Neither LLM nor wake: release any LLM/wake App Naps that
        // aren't also wake-storm offenders.
        let storm_pids: HashSet<u32> = storms.iter().map(|s| s.pid).collect();
        let mut qos = state.mach_qos.lock_recover();
        let app_napped: Vec<u32> = qos
            .current_tier_keys()
            .iter()
            .filter(|(pid, _)| qos.is_app_napped(*pid) && !storm_pids.contains(pid))
            .map(|(pid, _)| *pid)
            .collect();
        for pid in app_napped {
            qos.set_app_nap(pid, false);
        }
    }
}
