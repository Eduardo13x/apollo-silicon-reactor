//! # Daemon Context-Switch Tick
//!
//! Context-switch burst detector + reactive foreground unfreeze extracted
//! from main.rs (Wave 31). [Fowler 2004] Strangler Fig — pure move.
//!
//! ## Responsibilities
//! - Detect foreground change, push timestamp to ctx_switch_times ring buffer
//! - Reactively unfreeze foreground PID immediately on switch (before process_tree)
//! - Update last_fg_name + GC ctx_switch_times window (300 s)
//!
//! ## Ordering invariant
//! Must run AFTER ForegroundDetector (foreground_app/pid) and Markov tick,
//! and BEFORE process_tree build — reactive unfreeze needs only fg_pid, not
//! the full process family.

use std::collections::VecDeque;
use std::path::Path;
use std::time::{Duration, Instant};

use apollo_optimizer::engine::daemon_helpers::{unfreeze_pids, write_frozen_state};
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::lock_ext::LockRecover;

/// Run context-switch burst detector and reactive foreground unfreeze.
///
/// # Parameters
/// - `foreground_app` — current foreground app name (owned, stored as last_fg_name)
/// - `foreground_pid` — current foreground PID (reactive unfreeze target)
/// - `last_fg_name` — previous cycle fg name (detects transition)
/// - `ctx_switch_times` — ring buffer of recent switch timestamps
/// - `state` — SharedState (frozen_state + metrics)
/// - `frozen_state_path` — WAL path for write_frozen_state
pub fn run_ctx_switch_tick(
    foreground_app: Option<String>,
    foreground_pid: Option<u32>,
    last_fg_name: &mut Option<String>,
    ctx_switch_times: &mut VecDeque<Instant>,
    state: &SharedState,
    frozen_state_path: &Path,
) {
    let fg_changed = foreground_app.is_some()
        && last_fg_name.is_some()
        && foreground_app != *last_fg_name;

    if fg_changed {
        ctx_switch_times.push_back(Instant::now());
    }

    if let Some(fg_pid) = foreground_pid {
        let mut frozen_guard = state.frozen_state.lock_recover();
        if frozen_guard.remove(&fg_pid).is_some() {
            unfreeze_pids(std::iter::once(fg_pid));
            write_frozen_state(frozen_state_path, &frozen_guard);
            drop(frozen_guard);
            state.metrics.lock_recover().metrics.unfreezes_applied += 1;
        }
    }

    *last_fg_name = foreground_app;
    let cutoff = Instant::now() - Duration::from_secs(300);
    ctx_switch_times.retain(|t| *t > cutoff);
}
