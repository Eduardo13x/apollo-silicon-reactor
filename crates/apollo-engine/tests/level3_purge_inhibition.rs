//! Sprint 12 Phase D — Purge → Predictor Inhibition end-to-end.
//!
//! When `vm_purge` fires, the swap signal drops by gigabytes in milliseconds.
//! A naive closed-loop predictor reads that as "load improved", cools its
//! OOM risk estimate, and disables itself just as the post-purge stabilisation
//! period is most fragile. Phase D wires
//! `MaintenanceState::is_in_purge_inhibition_window()` into
//! `SignalIntelligence::purge_inhibited`. While the flag is true, `tick()`
//! skips `kf_swap.update(...)` and bumps
//! `LSE_COUNTERS.purge_inhibition_skips_total`.
//!
//! This test closes the loop end-to-end:
//!   1. After `mark_purged()`, `tick()` increments the counter by 1.
//!   2. When the purge is outside the 5 s window, `tick()` does NOT increment.
//!   3. When no purge has ever fired, `tick()` does NOT increment.
//!
//! ## References
//! [Hellerstein 2004 §9] "Feedback Control of Computing Systems" — exogenous
//!     disturbances must not be learned as changes in the controlled variable.
//! [Welch & Bishop 2006] Kalman filter measurement model — when the
//!     measurement is corrupted by a known external event, the correct
//!     response is to skip the update, not increase the measurement
//!     covariance (which would still bias the state).

use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use apollo_engine::engine::lse_counters::LSE_COUNTERS;
use apollo_engine::engine::maintenance_state::{MaintenanceState, PURGE_INHIBITION_WINDOW_SECS};
use apollo_engine::engine::signal_intelligence::SignalIntelligence;

/// `LSE_COUNTERS` is process-static; serialise across this file's tests
/// so the delta assertions don't race.
static COUNTER_GUARD: Mutex<()> = Mutex::new(());

/// Run one `tick()` with synthetic-but-valid inputs. Returns the digest's
/// `pressure_smooth` for sanity checks downstream callers may want.
fn run_one_tick(signal_intel: &mut SignalIntelligence, dt: f64) {
    let cpu_vals: Vec<f64> = vec![10.0, 5.0];
    let mem_vals: Vec<f64> = vec![1_000_000.0, 500_000.0];
    let _digest = signal_intel.tick(
        0.55,      // memory_pressure (mid)
        500_000.0, // swap_delta_bps (some flow)
        0.20,      // swap_ratio
        0.55,      // compressor proxy
        &cpu_vals,
        &mem_vals,
        "dominant",
        2_000_000_000, // 2 GB dominant
        4_000_000_000, // 4 GB total used
        8_000_000_000, // 8 GB available
        dt,
    );
}

#[test]
fn purge_inhibition_counter_bumps_inside_window() {
    let _g = COUNTER_GUARD.lock().unwrap();
    let before = LSE_COUNTERS
        .purge_inhibition_skips_total
        .load(std::sync::atomic::Ordering::Relaxed);

    let mut maintenance = MaintenanceState::default();
    maintenance.mark_purged();
    assert!(
        maintenance.is_in_purge_inhibition_window(),
        "precondition: fresh mark_purged → in inhibition window"
    );

    let mut signal_intel = SignalIntelligence::new();
    signal_intel.purge_inhibited = maintenance.is_in_purge_inhibition_window();
    run_one_tick(&mut signal_intel, 1.0);

    let after = LSE_COUNTERS
        .purge_inhibition_skips_total
        .load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(after - before, 1, "purge inside window → counter += 1");

    // Auto-clear contract: a subsequent tick (without re-setting the flag)
    // must NOT bump the counter — fail-safe so a missed downstream clear
    // can never silence the swap track forever.
    let before2 = after;
    run_one_tick(&mut signal_intel, 1.0);
    let after2 = LSE_COUNTERS
        .purge_inhibition_skips_total
        .load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        after2 - before2,
        0,
        "auto-clear: second tick without re-set must not increment"
    );
}

#[test]
fn purge_inhibition_counter_silent_outside_window() {
    let _g = COUNTER_GUARD.lock().unwrap();
    let before = LSE_COUNTERS
        .purge_inhibition_skips_total
        .load(std::sync::atomic::Ordering::Relaxed);

    let mut maintenance = MaintenanceState::default();
    // Place the last_any_purge_at strictly outside the 5s window.
    maintenance.last_any_purge_at =
        Some(SystemTime::now() - Duration::from_secs(PURGE_INHIBITION_WINDOW_SECS + 1));
    assert!(
        !maintenance.is_in_purge_inhibition_window(),
        "precondition: stale purge → not in window"
    );

    let mut signal_intel = SignalIntelligence::new();
    signal_intel.purge_inhibited = maintenance.is_in_purge_inhibition_window();
    run_one_tick(&mut signal_intel, 1.0);

    let after = LSE_COUNTERS
        .purge_inhibition_skips_total
        .load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(after - before, 0, "outside window → counter flat");
}

#[test]
fn purge_inhibition_counter_silent_without_any_purge() {
    let _g = COUNTER_GUARD.lock().unwrap();
    let before = LSE_COUNTERS
        .purge_inhibition_skips_total
        .load(std::sync::atomic::Ordering::Relaxed);

    let maintenance = MaintenanceState::default();
    assert!(!maintenance.is_in_purge_inhibition_window());

    let mut signal_intel = SignalIntelligence::new();
    signal_intel.purge_inhibited = maintenance.is_in_purge_inhibition_window();
    run_one_tick(&mut signal_intel, 1.0);

    let after = LSE_COUNTERS
        .purge_inhibition_skips_total
        .load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(after - before, 0, "no purge ever → counter flat");
}
