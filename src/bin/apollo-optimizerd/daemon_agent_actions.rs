//! # Daemon Agent Actions
//!
//! Predictive agent action injection extracted from main.rs (Wave 19).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - PreThrottleNoise: renice top 3 noise processes (soft throttle, no SIGSTOP)
//! - ProactivePurge: send paging hints to top 3 background processes by RSS
//!
//! ## Ordering invariant
//! Must run AFTER agent_intervention is selected (decide_actions) and AFTER
//! paging hints (Wave 17) so per-PID dedup is correct.

use apollo_engine::collector::ProcessStats;
use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::decide_actions::is_interactive_app_name;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::predictive_agent::Intervention;
use apollo_engine::engine::types::RootAction;
use apollo_engine::engine::user_context::UserContext;

/// Inject predictive-agent soft actions for this cycle.
///
/// # Parameters
/// - `agent_intervention` — the intervention selected by the predictive agent
/// - `top_processes` — snapshot.top_processes for this cycle
/// - `state` — SharedState (policy lock for noise/interactive/protected patterns)
/// - `decide_interactive` — interactive process name patterns from decide_actions
/// - `user_ctx` — UserContext for media-active gate (audio_active /
///   call_in_progress / has_sleep_assertion). `ProactivePurge` is suppressed
///   when any media signal is held: `kern.memorystatus_vm_pressure_send`
///   forces the target to drop caches, which on graphics-/audio-handling
///   processes (e.g. WindowServer, coreaudiod helpers) causes stutter and
///   skipped audio frames during podcast/video playback. The maintenance
///   purge gate already enforces this for shell `purge`; the predictive
///   path was the missing twin.
///
/// Returns new actions to extend the main actions vec.
pub fn run_agent_actions(
    agent_intervention: &Intervention,
    top_processes: &[ProcessStats],
    state: &SharedState,
    decide_interactive: &[String],
    user_ctx: &UserContext,
) -> Vec<RootAction> {
    let mut new_actions: Vec<RootAction> = Vec::new();

    match agent_intervention {
        Intervention::PreThrottleNoise => {
            // Renice top 3 noise processes (soft throttle, no SIGSTOP).
            let noise_pats = state
                .policy
                .lock_recover()
                .learned_policy
                .noise_patterns
                .clone();
            let mut noise_procs: Vec<_> = top_processes
                .iter()
                .filter(|p| noise_pats.iter().any(|pat| p.name.contains(pat.as_str())))
                .collect();
            noise_procs.sort_by(|a, b| {
                b.cpu_usage
                    .partial_cmp(&a.cpu_usage)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for proc in noise_procs.iter().take(3) {
                new_actions.push(RootAction::throttle(
                    proc.pid,
                    proc.name.clone(),
                    false,
                    "predictive-agent: pre-throttle noise",
                    DecisionReason::PressureContext,
                ));
            }
        }
        Intervention::ProactivePurge => {
            // Media-active gate: suppress proactive paging hints during audio /
            // video / call playback. SetMemorystatus drops caches in the target
            // process; firing it on WindowServer or audio helpers during media
            // produces visible/audible stutter. Symmetric with the maintenance
            // purge gate in daemon_maintenance_tick.rs::should_fire.
            if user_ctx.audio_active
                || user_ctx.call_in_progress
                || user_ctx.has_sleep_assertion
            {
                return new_actions;
            }
            // Send paging hints to top 3 background processes by RSS.
            // SetMemorystatus priority -1 = voluntary cache release — no freeze, no kill.
            let protected_pats = state
                .policy
                .lock_recover()
                .learned_policy
                .protected_patterns
                .clone();
            let daemon_pid = std::process::id();
            let mut bg_procs: Vec<_> = top_processes
                .iter()
                .filter(|p| {
                    p.pid != daemon_pid
                        && !is_interactive_app_name(&p.name)
                        && !decide_interactive
                            .iter()
                            .any(|pat| p.name.contains(pat.as_str()))
                        && !protected_pats
                            .iter()
                            .any(|pat| p.name.contains(pat.as_str()))
                        && p.memory_usage > 50 * 1024 * 1024
                })
                .collect();
            bg_procs.sort_by(|a, b| b.memory_usage.cmp(&a.memory_usage));
            for proc in bg_procs.iter().take(3) {
                new_actions.push(RootAction::SetMemorystatus {
                    pid: proc.pid,
                    priority: -1,
                    reason: "predictive-agent: proactive purge hint".to_string(),
                    decision_reason: DecisionReason::PressureContext,
                });
            }
        }
        _ => {} // Observe, TightenThresholds, SuggestAggressive handled above
    }

    new_actions
}
