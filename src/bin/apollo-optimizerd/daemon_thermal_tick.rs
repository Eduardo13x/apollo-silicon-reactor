//! # Daemon Thermal Tick
//!
//! Per-cycle ThermalManager + GPUManager update extracted from main.rs (Wave 24).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Feed IOKit temperatures to ThermalManager → predictive throttle forecast
//! - Feed GPU utilization to GPUManager → power state + workload recommendations
//! - Engage fast-tick when GPU is thermally throttled (15s window)
//!
//! ## Ordering invariant
//! Must run AFTER cycle_hw_snap is populated (daemon_sensor_tick) and BEFORE
//! the thermal_action evaluation (overflow_thresholds boost).

use apollo_engine::engine::daemon_helpers::audit_log;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::gpu_manager::{GPUManager, GPUMetrics, GPUPowerState};
use apollo_engine::engine::iokit_sensors::{
    HardwareSnapshot, ThermalState as HardwareThermalState,
};
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::thermal_manager::ThermalManager;
use std::time::{Duration, Instant};

pub struct ThermalTickOutput {
    /// True if GPU power state is Throttled this cycle.
    pub gpu_thermal_throttled: bool,
    /// Predicted throttle level (0–100) from ThermalManager.
    pub thermal_predicted_throttle: u8,
    /// Estimated seconds until hardware throttle, if available.
    pub thermal_seconds_to_throttle: Option<i32>,
    /// Human-readable thermal trend string.
    pub thermal_trend_predicted: String,
}

/// Tick ThermalManager and GPUManager for this cycle.
///
/// # Parameters
/// - `cycle_hw_snap` — IOKit hardware snapshot for this cycle (may be None)
/// - `thermal_mgr` — mutable ThermalManager (predictive throttle state)
/// - `gpu_mgr` — mutable GPUManager (workload + power recommendations)
/// - `state` — SharedState (metrics lock for fast_tick_until + gpu_watts)
/// - `jitter_us` — hardware jitter from HwPredictor sample (fed to thermal model)
/// - `hardware_cores` — startup capability detection for GPU power normalization
pub fn run_thermal_tick(
    cycle_hw_snap: Option<&HardwareSnapshot>,
    thermal_mgr: &mut ThermalManager,
    gpu_mgr: &mut GPUManager,
    state: &SharedState,
    jitter_us: u64,
    hardware_cores: u32,
) -> ThermalTickOutput {
    let mut gpu_thermal_throttled = false;
    let mut thermal_predicted_throttle: u8 = 0;
    let mut thermal_seconds_to_throttle: Option<i32> = None;
    let mut thermal_trend_predicted = String::new();

    if let Some(hw) = cycle_hw_snap {
        let gpu_watts = hw.power.gpu_watts.unwrap_or(0.0);
        let gpu_util =
            (gpu_watts / gpu_power_reference_watts(hardware_cores) * 100.0).clamp(0.0, 100.0);
        let measured_cpu_temp = (!hw.temps_estimated)
            .then_some(hw.temps.p_cluster_celsius)
            .flatten();
        let measured_gpu_temp = (!hw.temps_estimated)
            .then_some(hw.temps.gpu_celsius)
            .flatten();

        if let Some(cpu_t) = measured_cpu_temp {
            let gpu_t = measured_gpu_temp.unwrap_or(cpu_t);
            let thermal_state = thermal_mgr.update(cpu_t, gpu_t, 0.0, 0, jitter_us);
            thermal_predicted_throttle = thermal_state.predicted_throttle_level;
            thermal_seconds_to_throttle = thermal_state.seconds_to_throttle;
            thermal_trend_predicted = format!("{:?}", thermal_state.thermal_trend);
        } else {
            thermal_predicted_throttle = match hw.thermal_state {
                HardwareThermalState::Normal => 0,
                HardwareThermalState::Moderate => 25,
                HardwareThermalState::Severe => 70,
                HardwareThermalState::Critical => 100,
            };
            thermal_trend_predicted = format!("System{:?}", hw.thermal_state);
        }

        let gpu_temp = measured_gpu_temp.or(measured_cpu_temp).unwrap_or(0.0);
        let forced_power_state = match hw.thermal_state {
            HardwareThermalState::Critical => Some(GPUPowerState::Throttled),
            HardwareThermalState::Severe => Some(GPUPowerState::Dynamic),
            _ => None,
        };
        let gpu_metrics = GPUMetrics {
            gpu_temp,
            gpu_utilization: gpu_util,
            gpu_frequency: 0,
            gpu_memory_used: 0,
            gpu_memory_total: 0,
            throttle_active: matches!(
                hw.thermal_state,
                HardwareThermalState::Severe | HardwareThermalState::Critical
            ) || gpu_mgr.needs_cooling(&GPUMetrics {
                gpu_temp,
                gpu_utilization: gpu_util,
                gpu_frequency: 0,
                gpu_memory_used: 0,
                gpu_memory_total: 0,
                throttle_active: false,
                power_state: GPUPowerState::Dynamic,
            }),
            power_state: forced_power_state
                .unwrap_or_else(|| gpu_mgr.recommend_power_state(gpu_util, gpu_temp)),
        };
        if gpu_metrics.power_state == GPUPowerState::Throttled {
            gpu_thermal_throttled = true;
            state.metrics.lock_recover().fast_tick_until =
                Some(Instant::now() + Duration::from_secs(15));
        }
        if gpu_metrics.throttle_active || gpu_metrics.power_state == GPUPowerState::Throttled {
            let recs = gpu_mgr.thermal_recommendations(&gpu_metrics);
            if !recs.is_empty() {
                audit_log(&serde_json::json!({
                    "event": "gpu_thermal",
                    "gpu_temp": gpu_temp,
                    "gpu_util": gpu_util,
                    "power_state": format!("{:?}", gpu_metrics.power_state),
                    "recommendations": recs,
                }));
            }
        }
        state.metrics.lock_recover().metrics.energy_gpu_watts =
            Some(hw.power.gpu_watts.unwrap_or(0.0) as f64);
    }

    ThermalTickOutput {
        gpu_thermal_throttled,
        thermal_predicted_throttle,
        thermal_seconds_to_throttle,
        thermal_trend_predicted,
    }
}

fn gpu_power_reference_watts(hardware_cores: u32) -> f32 {
    (hardware_cores.max(4) as f32 * 2.0).clamp(12.0, 64.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_power_reference_scales_with_core_count() {
        assert_eq!(gpu_power_reference_watts(8), 16.0);
        assert_eq!(gpu_power_reference_watts(10), 20.0);
        assert!(gpu_power_reference_watts(14) > gpu_power_reference_watts(10));
    }
}
