//! GPU monitoring and optimization for Apple Silicon
//!
//! Monitors GPU usage, temperature, and applies thermal management.

#[derive(Debug, Clone)]
pub struct GPUMetrics {
    pub gpu_temp: f32,         // Celsius
    pub gpu_utilization: f32,  // 0-100%
    pub gpu_frequency: u32,    // MHz
    pub gpu_memory_used: u64,  // bytes
    pub gpu_memory_total: u64, // bytes
    pub throttle_active: bool,
    pub power_state: GPUPowerState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPUPowerState {
    Off,       // GPU powered down
    Idle,      // Minimal power
    Dynamic,   // Variable frequency
    Maximum,   // Full frequency
    Throttled, // Thermally throttled
}

pub struct GPUManager {
    max_safe_temp: f32,
    throttle_threshold: f32,
}

impl GPUManager {
    pub fn new() -> Self {
        Self {
            max_safe_temp: 100.0, // Max safe temp on Apple Silicon (aligns with ThermalManager shutdown)
            throttle_threshold: 90.0, // Start throttling at 90°C (aligns with ThermalManager/Sentinel)
        }
    }

    /// Check if GPU needs cooling
    pub fn needs_cooling(&self, metrics: &GPUMetrics) -> bool {
        metrics.gpu_temp > self.throttle_threshold
    }

    /// Get recommended GPU power state based on usage
    pub fn recommend_power_state(&self, utilization: f32, temp: f32) -> GPUPowerState {
        if temp > self.max_safe_temp {
            GPUPowerState::Throttled
        } else if temp > self.throttle_threshold {
            GPUPowerState::Dynamic // Reduce frequency
        } else if utilization > 80.0 {
            GPUPowerState::Maximum
        } else if utilization > 20.0 {
            GPUPowerState::Dynamic
        } else {
            GPUPowerState::Idle
        }
    }

    /// Apply GPU optimization based on workload
    pub fn optimize_for_workload(&self, workload: &str) -> Vec<String> {
        let mut actions = Vec::new();

        match workload {
            "ai" | "ml" | "llm" => {
                // ML workloads benefit from maximum GPU
                actions.push("Enable GPU memory optimization for ML".to_string());
                actions.push("Set GPU to maximum frequency".to_string());
                actions.push("Allocate unified memory aggressively".to_string());
            }
            "rendering" | "video" => {
                // Rendering needs balanced power
                actions.push("Optimize GPU cache for sequential access".to_string());
                actions.push("Enable predictive prefetch".to_string());
            }
            "idle" => {
                // Save power when not needed
                actions.push("Reduce GPU frequency to idle".to_string());
                actions.push("Disable GPU memory prefetch".to_string());
            }
            _ => {
                actions.push("Use dynamic GPU frequency scaling".to_string());
            }
        }

        actions
    }

    /// Get GPU thermal recommendations
    pub fn thermal_recommendations(&self, metrics: &GPUMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();

        if metrics.gpu_temp > self.max_safe_temp {
            recommendations.push(format!(
                "🔴 CRITICAL: GPU at {:.1}°C - Immediate cooling needed",
                metrics.gpu_temp
            ));
        } else if metrics.gpu_temp > self.throttle_threshold {
            recommendations.push(format!(
                "🟡 GPU warming: {:.1}°C - Consider reducing load",
                metrics.gpu_temp
            ));
        }

        if metrics.throttle_active {
            recommendations.push("⚠️ GPU thermal throttling active".to_string());
        }

        recommendations
    }
}

impl Default for GPUManager {
    fn default() -> Self {
        Self::new()
    }
}
