//! # Daemon Stale Apps
//!
//! Stale background app freeze nomination extracted from main.rs (Wave 21).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - When pressure ≥ 0.50: nominate apps with no user interaction for >30min
//!   as freeze candidates. Only non-foreground, non-critical, non-already-acting.
//!
//! ## Ordering invariant
//! Must run AFTER heuristic_pass (so heuristic_critical_pids is populated) and
//! AFTER paging hints (so existing_pids dedup is complete).

use std::collections::HashSet;

use apollo_engine::collector::SystemCollector;
use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::daemon_helpers::pid_start_time;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::types::RootAction;

/// Nominate stale background apps as freeze candidates.
///
/// # Parameters
/// - `pressure_smooth` — EMA pressure from signal_digest
/// - `all_proc_names` — all process names in this cycle (for stale_apps lookup)
/// - `state` — SharedState (policy lock for adaptive_governor)
/// - `collector` — SystemCollector (process iterator)
/// - `foreground_pid` — current foreground PID (never frozen)
/// - `heuristic_critical_pids` — PIDs protected by heuristic pass (never frozen)
/// - `current_actions` — actions accumulated so far (for per-PID dedup)
///
/// Returns new freeze actions to extend the main actions vec.
pub fn run_stale_app_freeze(
    pressure_smooth: f64,
    all_proc_names: &[&str],
    state: &SharedState,
    collector: &SystemCollector,
    foreground_pid: Option<u32>,
    heuristic_critical_pids: &HashSet<u32>,
    current_actions: &[RootAction],
) -> Vec<RootAction> {
    let mut new_actions: Vec<RootAction> = Vec::new();

    if pressure_smooth < 0.50 {
        return new_actions;
    }

    let existing_pids: HashSet<u32> = current_actions
        .iter()
        .filter_map(|a| match a {
            RootAction::FreezeProcess { pid, .. }
            | RootAction::ThrottleProcess { pid, .. }
            | RootAction::BoostProcess { pid, .. } => Some(*pid),
            _ => None,
        })
        .collect();

    // Snapshot stale candidates AND the learned-interactive patterns under a
    // single lock. The interactive list protects user-facing GUI apps the
    // governor learned about (e.g. the editor LSP `language_server`) that are
    // NOT in safety.rs's static hard list.
    let (stale_names, interactive_lc): (Vec<String>, Vec<String>) = {
        let pg = state.policy.lock_recover();
        let stale = pg
            .adaptive_governor
            .user_profile
            .stale_apps(all_proc_names, 1800); // 30 min threshold
        let interactive = pg
            .learned_policy
            .interactive_patterns
            .iter()
            .map(|p| p.to_lowercase())
            .collect();
        (stale, interactive)
    };

    let sys = collector.system();
    for (pid, process) in sys.processes() {
        let pid_u32 = pid.as_u32();
        let name = process.name().to_string();
        if !stale_names.contains(&name)
            || Some(pid_u32) == foreground_pid
            || heuristic_critical_pids.contains(&pid_u32)
            || existing_pids.contains(&pid_u32)
            || process.memory() < 50 * 1024 * 1024
        {
            continue;
        }
        // Complete mediation: this nomination path previously bypassed safety
        // protection entirely (prod 2026-06-13: it froze the Antigravity LSP
        // `language_server` and could freeze hard-protected Apple apps like
        // Finder when idle >30min). Honor the single source of truth before
        // nominating: name+path+Apple-signing protection, plus the learned
        // interactive list (macOS 15-char-truncation-aware match).
        if apollo_engine::engine::safety::is_protected_pid(pid_u32) {
            continue;
        }
        let name_lc = name.to_lowercase();
        if interactive_lc.iter().any(|pat| {
            apollo_engine::engine::decide_actions::learned_pattern_matches(&name_lc, pat)
        }) {
            continue;
        }
        let (ss, su) = pid_start_time(pid_u32);
        new_actions.push(RootAction::freeze_full(
            pid_u32,
            name.clone(),
            format!(
                "stale-app: no user interaction for >30min, rss={}MB",
                process.memory() / 1024 / 1024
            ),
            ss,
            su,
            DecisionReason::PressureContext,
        ));
    }

    new_actions
}

#[cfg(test)]
mod tests {
    use apollo_engine::engine::decide_actions::learned_pattern_matches;

    /// Regression guard for the 2026-06-13 prod bug: the stale-app freezer
    /// nominated the Antigravity LSP because it consulted no protection. The
    /// guard now matches the runtime (macOS-truncated, 15-char) process name
    /// against the full learned-interactive pattern. `language_server` (the
    /// truncated comm-name) MUST match the stored `language_server_macos_arm`.
    #[test]
    fn stale_freeze_skips_interactive_lsp_with_macos_truncation() {
        let runtime_name = "language_server"; // macOS comm-name, 15 chars
        let learned_pattern = "language_server_macos_arm"; // stored full name
        assert!(
            learned_pattern_matches(runtime_name, learned_pattern),
            "truncated LSP name must match the full learned interactive pattern"
        );
        // A short, unrelated name must NOT spuriously prefix-match (the >=8 guard).
        assert!(
            !learned_pattern_matches("Dock", "dockerd"),
            "short name must not prefix-match an unrelated long pattern"
        );
    }
}
