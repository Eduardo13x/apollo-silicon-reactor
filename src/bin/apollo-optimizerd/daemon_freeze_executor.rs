//! # Daemon Freeze Executor
//!
//! Per-cycle freeze lifecycle management extracted from the daemon main loop.
//!
//! Currently contains the TTL-based unfreeze sweep:
//! - Scans `frozen_state` for entries whose TTL elapsed and pressure is calm.
//! - FIFO-rotates the oldest frozen PID under sustained pressure on 8GB hardware
//!   to prevent unbounded resource hoarding.
//! - Skips PIDs currently held by the resource-interrupt handler.
//!
//! [Belady 1966] FIFO replacement is a well-behaved baseline under memory pressure.

use std::path::Path;

use apollo_optimizer::engine::daemon_helpers::{
    should_rotate_oldest, should_unfreeze, unfreeze_pids, write_frozen_state,
};
use apollo_optimizer::engine::daemon_state::{MetricsState, SharedState};
use apollo_optimizer::engine::lock_ext::LockRecover;
use chrono::Utc;

/// Sweep expired freezes and rotate the oldest under sustained pressure.
///
/// Caller must already hold the metrics lock and pass it as `&mut MetricsState`.
/// This preserves the lock-ordering invariant: metrics → frozen_state.
///
/// # Parameters
/// - `state` — Shared daemon state (resource_interrupt + frozen_state accessed).
/// - `frozen_state_path` — Path to `frozen_state.json` for atomic persistence.
/// - `current_pressure` — Live memory pressure; used by `should_unfreeze()` to
///   decide whether a TTL-expired freeze should actually be released.
/// - `metrics` — Caller's metrics guard; counters updated in-place.
pub fn run_ttl_unfreeze_sweep(
    state: &SharedState,
    frozen_state_path: &Path,
    current_pressure: f64,
    metrics: &mut MetricsState,
) {
    let now = Utc::now();
    let interrupt_pids = state
        .resource_interrupt
        .interrupt_frozen_pids
        .try_lock()
        .ok()
        .map(|g| g.clone())
        .unwrap_or_default();
    let mut frozen_state = state.frozen_state.lock_recover();
    let total_frozen = frozen_state.len();
    let mut expired: Vec<u32> = frozen_state
        .iter()
        .filter(|(pid, entry)| {
            let elapsed = now.signed_duration_since(entry.frozen_at).num_seconds();
            should_unfreeze(elapsed, entry.pressure_at_freeze, current_pressure)
                && !interrupt_pids.contains(pid)
        })
        .map(|(pid, _)| *pid)
        .collect();
    // FIFO rotation: on 8GB hardware, rotate oldest frozen process to
    // prevent resource hoarding under sustained pressure.
    if let Some((&oldest_pid, oldest_entry)) = frozen_state
        .iter()
        .filter(|(pid, _)| !interrupt_pids.contains(pid) && !expired.contains(pid))
        .min_by_key(|(_, e)| e.frozen_at)
    {
        let elapsed = now
            .signed_duration_since(oldest_entry.frozen_at)
            .num_seconds();
        if should_rotate_oldest(elapsed, total_frozen) {
            expired.push(oldest_pid);
        }
    }
    if !expired.is_empty() {
        let count = unfreeze_pids(expired.iter().copied());
        for pid in &expired {
            frozen_state.remove(pid);
        }
        write_frozen_state(frozen_state_path, &frozen_state);
        metrics.metrics.post_wake_defensive_unfreezes += count;
        metrics.metrics.unfreezes_applied += count;
        metrics.metrics.throttle_reverted += count;
    }
}
