//! # Daemon Wake Handler
//!
//! Per-cycle sleep/wake detection and post-wake grace window management.
//!
//! Detects system sleep/wake by comparing elapsed wall-clock time against
//! the expected cycle interval. A jump > 90s is treated as a wake event:
//! - Engages a 60s post-wake grace window (suppresses aggressive freezes)
//! - Resets volatile Kalman/OutcomeTracker state (stale from sleep)
//! - Queues all frozen PIDs for staggered SIGCONT (5/cycle drain loop)
//! - Clears turbo frozen PID set
//!
//! Staggered unfreeze avoids the 1-3GB decompression spike that a mass
//! SIGCONT produces on M1 8GB. [Nygard 2018] bulkhead pattern.

use std::collections::VecDeque;
use std::path::Path;

use apollo_engine::engine::daemon_helpers::write_wake_state;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::display_turbo::DisplayTurbo;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::outcome_tracker::OutcomeTracker;
use apollo_engine::engine::signal_intelligence::SignalIntelligence;
use chrono::{Duration as ChronoDuration, Utc};

/// Run one wake-detection tick.
///
/// Returns `grace_active`: true if currently inside a post-wake grace window.
///
/// # Parameters
/// - `state` — Shared daemon state (process, frozen_state, policy, metrics).
/// - `signal_intel` — Signal intelligence (reset on wake).
/// - `outcome_tracker` — Outcome tracker (reset on wake).
/// - `wake_unfreeze_queue` — Staggered-drain queue; frozen PIDs are appended here.
/// - `display_turbo` — Display-off turbo (turbo PIDs queued and cleared on wake).
/// - `wake_state_path` — Path to `wake_state.json` for persistence.
#[allow(clippy::too_many_arguments)]
pub fn run_wake_tick(
    state: &SharedState,
    signal_intel: &mut SignalIntelligence,
    outcome_tracker: &mut OutcomeTracker,
    wake_unfreeze_queue: &mut VecDeque<u32>,
    display_turbo: &mut DisplayTurbo,
    wake_state_path: &Path,
) -> bool {
    let now_wall = Utc::now();
    let mut process_guard = state.process.lock_recover();
    let wake_jump = now_wall - process_guard.wake_state.last_cycle_wallclock;
    let mut grace_active = process_guard
        .wake_state
        .post_wake_grace_until
        .map(|t| t > now_wall)
        .unwrap_or(false);
    // Clear expired grace so it doesn't carry across unrelated wakes.
    if !grace_active {
        process_guard.wake_state.post_wake_grace_until = None;
    }
    if wake_jump > ChronoDuration::seconds(90) {
        // Treat as wake: engage grace window and queue all frozen PIDs.
        process_guard.wake_state.last_wake_at = Some(now_wall);
        process_guard.wake_state.post_wake_grace_until =
            Some(now_wall + ChronoDuration::seconds(60));
        grace_active = true;

        // Reset volatile filter state. Pre-sleep Kalman position +
        // OutcomeTracker short-window deltas reflect a system that was
        // idle/paused for arbitrary time; without a reset they inject
        // phantom velocity/drift into the first post-wake decisions.
        // Long-term learned state (R, weights, EMAs) is preserved.
        signal_intel.reset_after_wake();
        outcome_tracker.reset_after_wake();

        // Staggered wake unfreeze: queue PIDs instead of mass SIGCONT.
        // Draining 5 PIDs/cycle avoids 1-3GB decompression spike on 8GB M1.
        // Priority: interactive PIDs first (user notices their latency).
        // [Nygard 2018 — bulkhead: bound blast radius of state transitions]
        let frozen_state = state.frozen_state.lock_recover();
        let total_queued = frozen_state.len() as u64;
        let interactive_pats = state
            .policy
            .lock_recover()
            .learned_policy
            .interactive_patterns
            .clone();
        let mut interactive_pids = Vec::new();
        let mut other_pids = Vec::new();
        for (&pid, entry) in frozen_state.iter() {
            let name = entry.process_name.as_deref().unwrap_or("");
            if interactive_pats
                .iter()
                .any(|pat| name.contains(pat.as_str()))
            {
                interactive_pids.push(pid);
            } else {
                other_pids.push(pid);
            }
        }
        // Interactive first, then the rest.
        // PIDs stay in frozen_state until actually SIGCONTed in the drain
        // loop — crash mid-drain won't orphan them.
        wake_unfreeze_queue.extend(interactive_pids);
        wake_unfreeze_queue.extend(other_pids);

        // Turbo PIDs also staggered.
        let turbo_pids = display_turbo.turbo_frozen_pids_snapshot();
        let turbo_count = turbo_pids.len() as u64;
        wake_unfreeze_queue.extend(turbo_pids.into_iter());
        display_turbo.clear_frozen();

        {
            let mut metrics = state.metrics.lock_recover();
            metrics.metrics.wake_events += 1;
            metrics.metrics.post_wake_grace_entries += 1;
            metrics.metrics.post_wake_defensive_unfreezes += total_queued + turbo_count;
            metrics.metrics.unfreezes_applied += total_queued + turbo_count;
            metrics.metrics.throttle_reverted += total_queued + turbo_count;
        }
    }
    process_guard.wake_state.last_cycle_wallclock = now_wall;
    write_wake_state(wake_state_path, &process_guard.wake_state);
    drop(process_guard);

    grace_active
}
