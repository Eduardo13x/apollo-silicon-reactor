//! # Daemon Pressure Aggregator
//!
//! Per-cycle effective-pressure aggregation extracted from the daemon main
//! loop. Consumes the upstream boost factors produced by
//! [`daemon_sensor_tick`](crate::daemon_sensor_tick) plus the battery /
//! charging / thermal state sampled elsewhere in the cycle, and returns the
//! authoritative `effective_pressure` value together with its observability
//! breakdown.
//!
//! ## Responsibility
//!
//! 1. **Charging thermal stress boost.** On the fanless M1 Air, charging
//!    plus heavy compute causes SoC throttling; SMC `PSTR` > 8 W while
//!    charging adds +0.06 to effective pressure so we freeze backgrounds
//!    *before* the hardware throttles.
//! 2. **Battery low aggressiveness.** SMC `B0TE` < 20 min adds +0.08 so
//!    the system sheds load while the battery is the critical resource.
//! 3. **Aggregation.** Defers to
//!    [`apollo_engine::engine::effective_pressure::compute`] so the
//!    boost-cap (≤ 0.30) and clamp semantics stay in one place.
//! 4. **Cautious-mode adjustment.** Subtracts 0.10 for the first 50 cycles
//!    after a crash restart, so freeze / throttle gates trigger at higher
//!    real pressure (more headroom during the post-crash re-index storm).
//!
//! ## Design invariants
//!
//! - **Pure transform.** No lock acquisition, no I/O, no logging. The
//!   caller is responsible for the `cautious_mode_ended` audit-log side
//!   effect based on the returned [`CautiousModeState`].
//! - **Authoritative order.** The aggregation order matches the original
//!   main-loop sequence exactly:
//!   `base → additive boosts (cap 0.30) → clamp → cautious subtract → clamp`.
//!   Per the NotebookLM peer review (2026-04-18), re-ordering these layers
//!   would shift the Adaptive Governor zone activations.
//! - **`max(kernel, compressor)` is NOT recomputed here.** By the time the
//!   daemon calls into this helper, `snapshot.pressure.memory_pressure`
//!   already carries the compressor/kernel max (see the collector). This
//!   function treats that value as the base and adds boosts on top.
//!
//! Extracted from `main.rs` during the V1.1.0 Strangler Fig pass
//! [Fowler 2004 — *Refactoring*].

use apollo_engine::engine::effective_pressure::{self, PressureComponents};
use apollo_engine::engine::iokit_sensors::HardwareSnapshot;
use apollo_engine::engine::ioreport::IOReportSnapshot;
use apollo_engine::engine::smc_direct::SmcSnapshot;

/// Aggregated per-cycle pressure output.
#[derive(Debug, Clone)]
pub struct PressureAggregation {
    /// Final effective pressure, already clamped to `[0.0, 1.0]` and with
    /// the cautious-mode subtraction applied. Write this back into
    /// `snapshot.pressure.memory_pressure` so downstream consumers
    /// (`decide_actions`, `page_reclaim`, `io_shaper`, `skill_registry`)
    /// see the authoritative value.
    pub effective_pressure: f64,
    /// Per-boost breakdown for observability (`pressure_total_boost`,
    /// `pressure_dominant_factor`). The `effective` field is the
    /// pre-cautious-mode value; it is intentionally left as the compute()
    /// output so the dashboard shows the raw boost picture.
    pub components: PressureComponents,
    /// Cautious-mode bookkeeping — caller updates its mutable counter and
    /// logs the `cautious_mode_ended` audit event when this reports
    /// `ended = true`.
    pub cautious: CautiousModeState,
}

/// Cautious-mode transition report produced by [`aggregate_cycle_pressure`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CautiousModeState {
    /// `true` iff the cautious counter was active this cycle (i.e. the
    /// -0.10 subtraction was applied). Retained for observability / tests
    /// even though the main loop only consumes `remaining` + `ended`.
    #[allow(dead_code)]
    pub active: bool,
    /// Remaining cycles after this call returns. Assign this to the
    /// caller's `cautious_cycles_remaining` state.
    pub remaining: u32,
    /// `true` on the exact cycle the counter transitioned from 1 → 0.
    /// Emit the `cautious_mode_ended` audit entry when this is set.
    pub ended: bool,
}

/// Compute the effective per-cycle memory pressure and its breakdown.
///
/// Aggregates nine boost factors on top of the raw pressure, applies the
/// 0.30 boost cap [commit `bea73ed` — *eliminate RAM panic cycle*], then
/// subtracts 0.10 if we are in the post-crash cautious window
/// [Gray & Reuter 1992 §3 — *conservative restart after abnormal
/// termination*].
///
/// # Arguments
///
/// * `base_pressure` — `snapshot.pressure.memory_pressure` (already
///   `max(kernel, compressor)` from the collector).
/// * `hw_boost` — hw_predictor boost (0.0 / 0.15 / 0.30).
/// * `batt_boost` — power-manager battery boost.
/// * `thermal_pressure_boost` — thermal bailout phase boost (0.0 … 0.40).
/// * `llm_boost` — LLM inference detector boost.
/// * `mem_bw_boost`, `smc_thermal_boost`, `battery_overheat_boost` —
///   upstream sensor boosts from [`crate::daemon_sensor_tick`].
/// * `last_smc` — latest SMC sample (for PSTR, charger_watts, B0TE).
/// * `last_ioreport` — latest IOReport sample (fallback for system watts).
/// * `cycle_hw_snap` — IOKit hardware snapshot (fallback for charging
///   detection via `battery_watts < 0`).
/// * `cautious_cycles_remaining` — current counter value.
///
/// # Returns
///
/// A [`PressureAggregation`] whose `effective_pressure` must be written
/// back into `snapshot.pressure.memory_pressure`.
#[allow(clippy::too_many_arguments)]
pub fn aggregate_cycle_pressure(
    base_pressure: f64,
    hw_boost: f64,
    batt_boost: f64,
    thermal_pressure_boost: f64,
    llm_boost: f64,
    mem_bw_boost: f64,
    smc_thermal_boost: f64,
    battery_overheat_boost: f64,
    last_smc: Option<&SmcSnapshot>,
    last_ioreport: Option<&IOReportSnapshot>,
    cycle_hw_snap: Option<&HardwareSnapshot>,
    cautious_cycles_remaining: u32,
) -> PressureAggregation {
    let charging_stress_boost =
        compute_charging_stress_boost(last_smc, last_ioreport, cycle_hw_snap);
    let battery_low_boost = compute_battery_low_boost(last_smc);

    // Raw memory_pressure misses hardware stress (thermal, battery,
    // bandwidth saturation). effective_pressure::compute() is the
    // authoritative aggregator — caps total boost at 0.30 and clamps
    // to [0.0, 1.0].
    let (pressure_after_boosts, components) = effective_pressure::compute(
        base_pressure,
        hw_boost,
        batt_boost,
        thermal_pressure_boost,
        llm_boost,
        charging_stress_boost,
        battery_low_boost,
        mem_bw_boost,
        smc_thermal_boost,
        battery_overheat_boost,
    );

    // Cautious mode: during the first 50 cycles after a crash, lower the
    // effective pressure Apollo *sees* by 0.10 so freeze / throttle gates
    // trigger at a higher real pressure. This leaves more headroom during
    // the I/O-unstable period right after an abnormal shutdown (e.g.
    // Spotlight re-indexing).
    let (effective_pressure, cautious) = if cautious_cycles_remaining > 0 {
        let adjusted = (pressure_after_boosts - 0.10).max(0.0);
        let remaining = cautious_cycles_remaining - 1;
        (
            adjusted,
            CautiousModeState {
                active: true,
                remaining,
                ended: remaining == 0,
            },
        )
    } else {
        (
            pressure_after_boosts,
            CautiousModeState {
                active: false,
                remaining: 0,
                ended: false,
            },
        )
    };

    PressureAggregation {
        effective_pressure,
        components,
        cautious,
    }
}

/// Charging thermal stress boost: charging + system watts > 8 W on a
/// fanless M1 Air is a strong thermal-throttling precursor. Prefers SMC
/// `PSTR` (real-time, <100 µs) over IOReport `total_watts`, and uses
/// `cycle_hw_snap.battery_watts < 0` as a last-resort charging detector
/// when SMC `PDTR` is unavailable.
fn compute_charging_stress_boost(
    last_smc: Option<&SmcSnapshot>,
    last_ioreport: Option<&IOReportSnapshot>,
    cycle_hw_snap: Option<&HardwareSnapshot>,
) -> f64 {
    let system_watts = last_smc
        .and_then(|s| s.system_power_watts)
        .or_else(|| last_ioreport.map(|ir| ir.total_watts()));

    if let Some(watts) = system_watts {
        let is_charging = last_smc
            .and_then(|s| s.charger_watts)
            .map(|cw| cw > 0.0)
            .unwrap_or_else(|| {
                cycle_hw_snap
                    .and_then(|h| h.battery_watts)
                    .map(|w| w < 0.0) // negative = charging
                    .unwrap_or(false)
            });
        if is_charging && watts > 8.0 {
            0.06
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// Battery aggressiveness boost: SMC `B0TE` (time-to-empty) below 20 min
/// adds +0.08 so Apollo sheds load before the OS starts denying
/// allocations.
fn compute_battery_low_boost(last_smc: Option<&SmcSnapshot>) -> f64 {
    last_smc
        .and_then(|s| s.battery_time_to_empty_min)
        .filter(|&tte| tte < 20)
        .map(|_| 0.08)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_sensors_no_cautious_mode_returns_base_plus_boosts() {
        let out = aggregate_cycle_pressure(
            0.55, 0.15, 0.04, 0.07, 0.0, 0.0, 0.0, 0.0, None, None, None, 0,
        );
        // base + hw + batt + thermal = 0.55 + 0.15 + 0.04 + 0.07 = 0.81
        assert!((out.effective_pressure - 0.81).abs() < 1e-9);
        assert!(!out.cautious.active);
        assert!(!out.cautious.ended);
        assert_eq!(out.cautious.remaining, 0);
    }

    #[test]
    fn cautious_mode_subtracts_010() {
        let out = aggregate_cycle_pressure(
            0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, 50,
        );
        assert!((out.effective_pressure - 0.50).abs() < 1e-9);
        assert!(out.cautious.active);
        assert_eq!(out.cautious.remaining, 49);
        assert!(!out.cautious.ended);
    }

    #[test]
    fn cautious_mode_ends_on_final_cycle() {
        let out =
            aggregate_cycle_pressure(0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, 1);
        assert_eq!(out.cautious.remaining, 0);
        assert!(out.cautious.ended);
        assert!(out.cautious.active);
    }

    #[test]
    fn cautious_mode_cannot_drop_below_zero() {
        let out = aggregate_cycle_pressure(
            0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, 50,
        );
        assert_eq!(out.effective_pressure, 0.0);
    }

    #[test]
    fn charging_stress_requires_both_charging_and_high_watts() {
        let mut smc = dummy_smc();
        smc.system_power_watts = Some(10.0);
        smc.charger_watts = Some(30.0);
        assert_eq!(compute_charging_stress_boost(Some(&smc), None, None), 0.06);

        // Charging but low watts → no boost.
        smc.system_power_watts = Some(5.0);
        assert_eq!(compute_charging_stress_boost(Some(&smc), None, None), 0.0);

        // High watts but not charging → no boost.
        smc.system_power_watts = Some(10.0);
        smc.charger_watts = None;
        assert_eq!(compute_charging_stress_boost(Some(&smc), None, None), 0.0);
    }

    #[test]
    fn battery_low_boost_fires_below_20_min() {
        let mut smc = dummy_smc();
        smc.battery_time_to_empty_min = Some(15);
        assert_eq!(compute_battery_low_boost(Some(&smc)), 0.08);

        smc.battery_time_to_empty_min = Some(20);
        assert_eq!(compute_battery_low_boost(Some(&smc)), 0.0);

        smc.battery_time_to_empty_min = None;
        assert_eq!(compute_battery_low_boost(Some(&smc)), 0.0);
    }

    #[test]
    fn boost_cap_still_enforced_via_effective_pressure_compute() {
        // All boosts maxed: raw sum ≈ 1.74, compute caps at 0.30.
        // Then cautious subtracts 0.10 → 0.60 + 0.30 - 0.10 = 0.80.
        let out = aggregate_cycle_pressure(
            0.60, 0.30, 0.18, 0.40, 0.20, 0.10, 0.30, 0.12, None, None, None, 10,
        );
        assert!(
            (out.effective_pressure - 0.80).abs() < 1e-9,
            "expected 0.80 after cap+cautious, got {}",
            out.effective_pressure
        );
    }

    fn dummy_smc() -> SmcSnapshot {
        SmcSnapshot {
            system_power_watts: None,
            lid_closed: false,
            last_sleep_us: 0,
            last_wake_us: 0,
            battery_time_to_empty_min: None,
            battery_time_to_full_min: None,
            charger_watts: None,
            cpu_temp_celsius: None,
            gpu_temp_celsius: None,
            battery_temp_celsius: None,
            p_cluster_watts: None,
            gpu_watts: None,
            dc_in_current_amps: None,
            cpu_voltage: None,
        }
    }
}
