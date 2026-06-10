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
use apollo_engine::engine::active_coalition_envelope::ActiveCoalitionEnvelope;
use apollo_engine::engine::apple_owned::is_apple_owned;
use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::coalition::CoalitionTracker;
use apollo_engine::engine::companion_graph::CompanionGraph;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::decide_actions::is_interactive_app_name;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::predictive_agent::Intervention;
use apollo_engine::engine::safety::is_protected_name;
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
    foreground_app: Option<&str>,
    foreground_pid: Option<u32>,
    companion_graph: &CompanionGraph,
    coalition_tracker: &CoalitionTracker,
    active_coalitions: &ActiveCoalitionEnvelope,
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
            if user_ctx.audio_active || user_ctx.call_in_progress || user_ctx.has_sleep_assertion {
                return new_actions;
            }
            // Send paging hints to top 3 background processes by RSS.
            // SetMemorystatus priority -1 = voluntary cache release — no freeze, no kill.
            //
            // Active-coalition envelope check — protect every PID whose
            // coalition_id is in the recent fg envelope (current fg + last
            // 2 within a 5-min grace window). Closes the rapid-app-switch
            // gap: tabbing from Antigravity to Terminal for `git status`
            // does NOT immediately strip Antigravity helpers of protection.
            //
            // Coalition_id is computed per-PID via proc_pidinfo
            // (~200ns/call). top_processes is bounded so the per-cycle
            // cost is ≤ 10µs. Strictly subsumes the older single-fg
            // `family_of` call.
            let _ = foreground_pid; // not needed here; envelope already updated upstream
                                    //
                                    // Lock policy once and run the entire filter inside the guard so
                                    // process_relevance(name) can be queried without per-iteration
                                    // re-locking. top_processes is bounded (~50), so the critical
                                    // section stays short.
            let policy = state.policy.lock_recover();
            let protected_pats = &policy.learned_policy.protected_patterns;
            let user_profile = &policy.adaptive_governor.user_profile;
            // Threshold tuned conservatively: process_relevance returns
            //   1.0  → matches the current workload's signature directly
            //   0.8  → used in the last 5 minutes
            //   0.5  → used in the last hour
            //   0.2  → used in the last 24 hours
            //   0.0  → never seen / >24h ago
            // 0.30 protects "actively-used" apps (within ~1h) without
            // requiring the user to be in their dominant workload right now.
            const USAGE_RELEVANCE_PROTECT_THRESHOLD: f32 = 0.30;
            let daemon_pid = std::process::id();
            let mut bg_procs: Vec<_> = top_processes
                .iter()
                .filter(|p| {
                    // OS-essential guard: WindowServer / coreaudiod / launchd /
                    // loginwindow / configd live in safety::is_protected_name and
                    // are not user-interactive apps, so the INTERACTIVE_APPS list
                    // never matched them. Without this guard a high-RSS system
                    // process (observed: pid 422 WindowServer) ended up as a
                    // top-3 victim and received memorystatus_vm_pressure_send,
                    // forcing graphics-cache eviction and visible UI stutter.
                    //
                    // Future-proof guard: `apple_owned::is_apple_owned` classifies
                    // by SIP path prefix + codesign authority chain (cached). Any
                    // new Apple daemon shipped in a future macOS release is
                    // auto-protected without code change — no list to update.
                    //
                    // Personalised guard: `process_relevance` reflects what THIS
                    // user actually uses (foreground time, recency, current
                    // workload signature). Bridges the gap between the static
                    // INTERACTIVE_APPS list and the user's real habits, so apps
                    // the user touches every day are auto-protected even if
                    // they aren't in any hardcoded list.
                    // Companion-graph guard: if the user has put `fg_app` in
                    // foreground enough times for the graph to mature, processes
                    // that are reliably alive WITH that fg_app and lift > 2.0
                    // are considered satellites of the active workflow and
                    // protected from purge. Lift gating naturally rejects
                    // always-on daemons (kernel_task, mds, etc.) whose global
                    // base rate ≈ 1.0.
                    let is_companion = foreground_app
                        .map(|fg| companion_graph.is_companion_of(fg, &p.name))
                        .unwrap_or(false);
                    // Coalition guard: any PID whose coalition_id matches a
                    // recently-active fg coalition (current + 5-min grace
                    // for last 2) is a subprocess of an active workflow
                    // (Electron renderer, IDE LSP host, audio.mojom utility,
                    // XPC service spawned on demand). This is the
                    // name-stability fix the user asked for — the helper can
                    // rename across versions ("Antigravity Helper (Renderer)"
                    // → "Antigravity Helper v2") without losing protection.
                    let in_fg_family =
                        active_coalitions.is_active(coalition_tracker.get_coalition_id(p.pid));
                    p.pid != daemon_pid
                        && !in_fg_family
                        && !is_apple_owned(p.pid)
                        && !is_protected_name(&p.name)
                        && !is_interactive_app_name(&p.name)
                        && !is_companion
                        && user_profile.process_relevance(&p.name)
                            < USAGE_RELEVANCE_PROTECT_THRESHOLD
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
