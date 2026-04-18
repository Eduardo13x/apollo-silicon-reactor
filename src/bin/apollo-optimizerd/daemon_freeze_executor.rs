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

use std::collections::{HashMap, HashSet};
use std::path::Path;

use apollo_optimizer::engine::daemon_helpers::{
    should_rotate_oldest, should_unfreeze, unfreeze_pids, write_frozen_state,
};
use apollo_optimizer::engine::daemon_state::{MetricsState, SharedState};
use apollo_optimizer::engine::fluidity::FluidityState;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::types::RootAction;
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

/// Apply the 2-cycle freeze confirmation gate with fluidity-aware overrides.
///
/// Freezes are only emitted after a PID has been proposed for **N consecutive
/// cycles**, where N is:
/// - 2 under normal operation (filters short-lived transients)
/// - 1 when fluidity is predicted to drop below 0.60 within 3s (pre-emptive)
/// - 0 (all freezes deferred) while an app launch is active
///
/// Also performs per-cycle dedup (FreezeProcess can be proposed by multiple
/// upstream paths — stale-app, adaptive_governor, survival-mode) and decays
/// `freeze_candidates` entries no longer proposed this cycle.
///
/// [Welch & Bishop 2006] Kalman prediction enables anticipatory control.
/// [Shavit & Lotan 2000] pre-emptive action on predicted queue saturation.
/// [Selkowitz 1984] app launch is the user's primary interaction moment.
///
/// # Parameters
/// - `graced_actions` — actions after post-wake grace filter (consumed).
/// - `fluidity_state` — current fluidity snapshot (launch state, 3s prediction).
/// - `freeze_candidates` — persistent per-PID confirmation counter (mutated).
///
/// # Returns
/// The confirmed action list with non-confirmed FreezeProcess entries dropped.
pub fn apply_freeze_confirmation(
    graced_actions: Vec<RootAction>,
    fluidity_state: &FluidityState,
    freeze_candidates: &mut HashMap<u32, u8>,
) -> Vec<RootAction> {
    // Fluidity launch guard: defer ALL new freezes while an app launch is
    // in progress. App startup is a latency-sensitive critical path.
    let fluidity_launch_active = fluidity_state.launch_active;
    if fluidity_launch_active {
        tracing::debug!(
            launch = %fluidity_state.launch_name,
            cycles_remaining = fluidity_state.launch_cycles_remaining,
            "fluidity: launch active — deferring new freezes"
        );
    }

    // Pre-emptive: lower confirmation threshold from 2 to 1 cycle when
    // Kalman predicts fluidity < 0.60 within 3s AND velocity is negative.
    let fluidity_preemptive = !fluidity_launch_active
        && fluidity_state.fluidity_predicted_3s < 0.60
        && fluidity_state.fluidity_velocity < -0.05;
    if fluidity_preemptive {
        tracing::info!(
            predicted = fluidity_state.fluidity_predicted_3s,
            velocity = fluidity_state.fluidity_velocity,
            "fluidity: predicted drop to {:.2} — pre-emptive freeze threshold lowered",
            fluidity_state.fluidity_predicted_3s
        );
    }

    // Collect all PIDs proposed for freeze this cycle (before filtering).
    let proposed_freeze_pids: HashSet<u32> = graced_actions
        .iter()
        .filter_map(|a| {
            if let RootAction::FreezeProcess { pid, .. } = a {
                Some(*pid)
            } else {
                None
            }
        })
        .collect();

    let mut seen_freeze_pids: HashSet<u32> = HashSet::new();
    let confirmed_actions: Vec<RootAction> = graced_actions
        .into_iter()
        .filter(|a| {
            if let RootAction::FreezeProcess { pid, .. } = a {
                if fluidity_launch_active {
                    return false;
                }
                // Per-cycle dedup: FreezeProcess can be proposed by multiple
                // upstream paths without dedup, downstream deep-scan converts
                // each dup to a separate SetMemorystatus hit on the same PID.
                if !seen_freeze_pids.insert(*pid) {
                    return false;
                }
                let count = freeze_candidates.entry(*pid).or_insert(0);
                let required = if fluidity_preemptive { 1 } else { 2 };
                // Cap at required+1: PIDs proposed every cycle but skipped
                // downstream would otherwise accumulate count indefinitely.
                *count = (*count + 1).min(required + 1);
                *count >= required
            } else {
                true
            }
        })
        .collect();

    // Decay: remove PIDs no longer proposed for freeze this cycle.
    // Use all-proposals (not just confirmed) so first-cycle candidates
    // survive to reach count >= 2.
    freeze_candidates.retain(|pid, _| proposed_freeze_pids.contains(pid));

    confirmed_actions
}
