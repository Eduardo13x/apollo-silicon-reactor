//! # Daemon Holt-Winters Tick
//!
//! Seasonal pressure forecasting per-cycle tick extracted from main.rs (Wave 30).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Accumulate pressure samples per cycle
//! - Feed hourly average to HoltWinters when the hour rolls over
//! - Use 1-hour-ahead forecast to proactively tighten overflow thresholds
//! - Cross-reference with UserProfile workload type for build-session multiplier
//!
//! ## Ordering invariant
//! Must run AFTER overflow_thresholds is computed (D-term PID) and signal_tick,
//! so the seasonal tightening stacks on top of the reactive adjustment.

use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::holt_winters::HoltWinters;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::overflow_guard::OverflowThresholds;

/// Apply Holt-Winters seasonal forecast: accumulate samples, observe on hour change,
/// and tighten overflow thresholds when next-hour pressure is predicted high.
///
/// # Parameters
/// - `memory_pressure` — current cycle memory pressure
/// - `hour_of_day` — current UTC hour
/// - `holt_winters` — mutable HoltWinters model (observe + forecast)
/// - `hw_pressure_accum` — running sum of pressure within current hour
/// - `hw_pressure_count` — cycle count within current hour
/// - `hw_last_hour` — last hour seen (detects rollover)
/// - `state` — SharedState (reads UserProfile for workload-type multiplier)
/// - `overflow_thresholds` — mutable thresholds tightened by seasonal forecast
pub fn run_holt_winters_tick(
    memory_pressure: f64,
    hour_of_day: u8,
    holt_winters: &mut HoltWinters,
    hw_pressure_accum: &mut f64,
    hw_pressure_count: &mut u32,
    hw_last_hour: &mut Option<u8>,
    state: &SharedState,
    overflow_thresholds: &mut OverflowThresholds,
) {
    *hw_pressure_accum += memory_pressure;
    *hw_pressure_count += 1;

    if *hw_last_hour != Some(hour_of_day) {
        if let Some(prev_hour) = *hw_last_hour {
            if *hw_pressure_count > 0 {
                let avg = *hw_pressure_accum / *hw_pressure_count as f64;
                holt_winters.observe(prev_hour, avg);
            }
        }
        *hw_last_hour = Some(hour_of_day);
        *hw_pressure_accum = 0.0;
        *hw_pressure_count = 0;
    }

    let (forecast_1h, confidence) = holt_winters.forecast(hour_of_day, 1);
    if confidence > 0.3 && forecast_1h > 0.75 {
        let hw_adjustment = (forecast_1h - 0.75) * confidence * 0.10;
        let hw_adjustment = hw_adjustment.min(0.04);

        let next_hour = (hour_of_day + 1) % 24;
        let next_workload = state
            .policy
            .lock_recover()
            .adaptive_governor
            .user_profile
            .likely_workload_at_hour(next_hour);
        let workload_multiplier = match next_workload {
            apollo_engine::engine::user_profile::WorkloadType::Coding => 1.5,
            apollo_engine::engine::user_profile::WorkloadType::VideoEdit => 1.3,
            _ => 1.0,
        };

        let final_adjustment = (hw_adjustment * workload_multiplier).min(0.06);
        overflow_thresholds.bg_pressure -= final_adjustment;
        overflow_thresholds.critical_pressure -= final_adjustment;
        overflow_thresholds.extreme_pressure -= final_adjustment;
    }
}
