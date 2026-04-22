//! # Daemon Warn Limits
//!
//! Targeted non-fatal memorystatus warn-limit paging extracted from main.rs (Wave 23).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - When mem > 0.45 AND swap active AND root: set warn limits on idle hoarders,
//!   Rosetta background renderers, and Mach port leakers
//! - Clear expired warn limits after 3 cycles (process has had time to respond)
//!
//! ## Ordering invariant
//! Must run AFTER heuristic_pass (heuristic_critical_pids) and AFTER
//! daemon_feature_gates::apply_app_nap_scheduling.

use std::collections::{HashMap, HashSet};

use apollo_optimizer::engine::coalition::CoalitionTracker;
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::jetsam_control;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::process_classifier::ProcessSnapshot;
use apollo_optimizer::engine::process_tree::ProcessTree;

use crate::process_enrichment;

/// Apply targeted warn-limit paging hints to idle hoarders and port leakers.
///
/// # Parameters
/// - `memory_pressure` — effective memory pressure for this cycle
/// - `swap_used_bytes` — raw swap usage
/// - `is_root` — whether daemon runs as root (warn-limit requires root)
/// - `foreground_pid` — current foreground PID
/// - `process_tree` — process parent→child tree for foreground family detection
/// - `proc_snaps` — enriched process snapshots for this cycle
/// - `coalition_tracker` — for kernel-authoritative foreground family membership
/// - `state` — SharedState (policy + mach_qos locks)
/// - `heuristic_critical_pids` — PIDs protected by heuristic pass
/// - `warn_limit_pids` — per-PID countdown state (mutated in-place)
pub fn run_warn_limits(
    memory_pressure: f64,
    swap_used_bytes: u64,
    is_root: bool,
    foreground_pid: Option<u32>,
    process_tree: &ProcessTree,
    proc_snaps: &[ProcessSnapshot],
    coalition_tracker: &CoalitionTracker,
    state: &SharedState,
    heuristic_critical_pids: &HashSet<u32>,
    warn_limit_pids: &mut HashMap<u32, u8>,
) {
    let swap_active = swap_used_bytes > 256 * 1024 * 1024;
    if memory_pressure > 0.45 && swap_active && is_root {
        let mut fg_pids = process_enrichment::build_foreground_family(foreground_pid, process_tree);
        if let Some(fg_pid) = foreground_pid {
            let all_pids: Vec<u32> = proc_snaps.iter().map(|s| s.pid).collect();
            for coalition_pid in coalition_tracker.family_of(fg_pid, &all_pids) {
                fg_pids.insert(coalition_pid);
            }
        }
        let interactive_pats: Vec<String> = state
            .policy
            .lock_recover()
            .learned_policy
            .interactive_patterns
            .clone();

        for snap in proc_snaps.iter().take(100) {
            if heuristic_critical_pids.contains(&snap.pid) || fg_pids.contains(&snap.pid) {
                continue;
            }
            if interactive_pats
                .iter()
                .any(|p| snap.name.contains(p.as_str()))
            {
                continue;
            }
            let rss_threshold = if snap.is_translated {
                80 * 1024 * 1024 // Rosetta: lower threshold due to JIT overhead
            } else {
                120 * 1024 * 1024
            };
            let is_hoarder = snap.rss_bytes > rss_threshold
                && snap.secs_since_user_interaction > 120
                && !snap.has_gui_window;
            let is_bg_renderer = snap.rss_bytes > 60 * 1024 * 1024
                && snap.secs_since_user_interaction > 120
                && (snap.name.contains("Helper (Renderer)")
                    || snap.name.contains("Helper (Plugin)")
                    || snap.name.contains(" Renderer"));
            let is_port_leaker = if snap.rss_bytes > 50 * 1024 * 1024
                && snap.secs_since_user_interaction > 60
            {
                let qos = state.mach_qos.lock_recover();
                qos.get_mach_port_count(snap.pid)
                    .map(|c| c > 5000)
                    .unwrap_or(false)
            } else {
                false
            };
            if is_hoarder || is_bg_renderer || is_port_leaker {
                let ratio = if snap.is_translated { 3u64 } else { 4u64 };
                let warn_mb = (snap.rss_bytes * ratio / 5 / 1024 / 1024) as i32;
                let warn_mb = warn_mb.max(32);
                if jetsam_control::set_warn_limit(snap.pid, warn_mb).is_ok() {
                    warn_limit_pids.insert(snap.pid, 3);
                }
            }
        }
    }

    // Clear expired warn limits (process has had time to respond).
    warn_limit_pids.retain(|&pid, countdown| {
        *countdown -= 1;
        if *countdown == 0 {
            let _ = jetsam_control::set_warn_limit(pid, 0);
            false
        } else {
            true
        }
    });
}
