//! # Daemon Paging Hints
//!
//! Per-cycle SetMemorystatus paging hint injection extracted from main.rs (Wave 17).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Direct pressure hints: when pressure_smooth ≥ 0.60, hint top 3 background procs
//! - ODE velocity hints: when ODE net_rate > 0.5 (leading indicator before threshold fires)
//!   [Hellerstein 2004 §9 — derivative control acts before integrator saturates]
//!
//! ## Ordering invariant
//! Must run AFTER signal_digest and reclaim_forecast are computed.
//! Must run BEFORE heuristic_pass (so hinted PIDs are visible for dedup).

use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::decide_actions::is_interactive_app_name;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::process_classifier::ProcessSnapshot;
use apollo_optimizer::engine::safety::{is_protected_name, is_user_interactive_app};
use apollo_optimizer::engine::swap_reclaim::{CyberPhysicalSignal, NetRateNorm};
use apollo_optimizer::engine::types::RootAction;

/// Emit direct pressure-driven and ODE-velocity paging hints for this cycle.
///
/// Returns new SetMemorystatus actions to append to the main actions vec.
/// Deduplicates against any pids that already have hints in `current_actions`.
///
/// # Parameters
/// - `proc_snaps` — full process snapshot list (not just top 10 by CPU)
/// - `state` — SharedState (policy lock for learned protection patterns)
/// - `pressure_smooth` — EMA pressure from signal_digest
/// - `ode_net_rate_bps` — ODE net rate from reclaim_forecast (bytes/sec)
/// - `foreground_app` — current foreground app name (hint filter)
/// - `current_actions` — actions accumulated so far (for per-PID dedup)
pub fn run_paging_hints(
    proc_snaps: &[ProcessSnapshot],
    state: &SharedState,
    pressure_smooth: f64,
    ode_net_rate_bps: f64,
    foreground_app: Option<&str>,
    current_actions: &[RootAction],
) -> Vec<RootAction> {
    let mut new_actions: Vec<RootAction> = Vec::new();

    // ── Direct pressure hints ────────────────────────────────────────────────
    // When pressure > 0.60, hint top 3 background memory consumers to release
    // caches voluntarily. SetMemorystatus priority -1 = voluntary cache release.
    // [Jiang & Zhang 2005] proactive beats reactive by 20-40%.
    // BUG #2 fix: per-PID dedup instead of "any SetMemorystatus → skip all".
    if pressure_smooth >= 0.60 {
        let hinted_pids: std::collections::HashSet<u32> = current_actions
            .iter()
            .filter_map(|a| {
                if let RootAction::SetMemorystatus { pid, .. } = a {
                    Some(*pid)
                } else {
                    None
                }
            })
            .collect();
        let protected_pats = state
            .policy
            .lock_recover()
            .learned_policy
            .protected_patterns
            .clone();
        let mut bg_procs: Vec<_> = proc_snaps
            .iter()
            .filter(|p| {
                if is_interactive_app_name(&p.name) {
                    return false;
                }
                let is_interactive = is_user_interactive_app(
                    p.has_gui_window,
                    p.secs_since_user_interaction,
                    p.rss_bytes,
                    &p.name,
                );
                !is_protected_name(&p.name)
                    && !is_interactive
                    && !protected_pats.iter().any(|pat| p.name.contains(pat.as_str()))
                    && p.rss_bytes > 80 * 1024 * 1024 // >80 MB RSS
                    && p.pid != std::process::id()
                    && !p.has_gui_window
                    && foreground_app.map(|fg| p.name != fg).unwrap_or(true)
                    && p.secs_since_user_interaction > 60
            })
            .collect();
        bg_procs.sort_by(|a, b| b.rss_bytes.cmp(&a.rss_bytes));
        let mut added = 0usize;
        for proc in bg_procs.iter() {
            if added >= 3 {
                break;
            }
            if hinted_pids.contains(&proc.pid) {
                continue;
            }
            new_actions.push(RootAction::set_memorystatus(
                proc.pid,
                -1,
                format!(
                    "pressure-driven hint (p={:.0}%): {} ({}MB)",
                    pressure_smooth * 100.0,
                    proc.name,
                    proc.rss_bytes / 1024 / 1024,
                ),
            ));
            added += 1;
        }
    }

    // ── G20: ODE velocity hints ──────────────────────────────────────────────
    // When ODE net_rate > 0.5 AND pressure < 0.60: proactively hint top 2 procs.
    // ODE is a leading indicator — rising compression rate predicts pressure before
    // the kernel threshold fires. [Hellerstein 2004 §9]
    let ode_rate_norm = NetRateNorm(ode_net_rate_bps).normalized();
    if ode_rate_norm > 0.5 && pressure_smooth < 0.60 {
        let hinted_pids_ode: std::collections::HashSet<u32> = current_actions
            .iter()
            .chain(new_actions.iter())
            .filter_map(|a| {
                if let RootAction::SetMemorystatus { pid, .. } = a {
                    Some(*pid)
                } else {
                    None
                }
            })
            .collect();
        let protected_pats = state
            .policy
            .lock_recover()
            .learned_policy
            .protected_patterns
            .clone();
        let mut bg_procs: Vec<_> = proc_snaps
            .iter()
            .filter(|p| {
                if is_interactive_app_name(&p.name) {
                    return false;
                }
                let is_interactive = is_user_interactive_app(
                    p.has_gui_window,
                    p.secs_since_user_interaction,
                    p.rss_bytes,
                    &p.name,
                );
                !is_protected_name(&p.name)
                    && !is_interactive
                    && !protected_pats.iter().any(|pat| p.name.contains(pat.as_str()))
                    && p.rss_bytes > 80 * 1024 * 1024
                    && p.pid != std::process::id()
                    && !p.has_gui_window
                    && foreground_app.map(|fg| p.name != fg).unwrap_or(true)
                    && p.secs_since_user_interaction > 60
            })
            .collect();
        bg_procs.sort_by(|a, b| b.rss_bytes.cmp(&a.rss_bytes));
        let mut added = 0usize;
        for proc in bg_procs.iter() {
            if added >= 2 {
                break;
            }
            if hinted_pids_ode.contains(&proc.pid) {
                continue;
            }
            new_actions.push(RootAction::set_memorystatus(
                proc.pid,
                -1,
                format!(
                    "ode-velocity hint (net_rate={:.0}%): {} ({}MB)",
                    ode_rate_norm * 100.0,
                    proc.name,
                    proc.rss_bytes / 1024 / 1024,
                ),
            ));
            added += 1;
        }
    }

    new_actions
}
