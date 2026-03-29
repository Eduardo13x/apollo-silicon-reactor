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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(temp: f32, util: f32, throttle: bool, power: GPUPowerState) -> GPUMetrics {
        GPUMetrics {
            gpu_temp: temp,
            gpu_utilization: util,
            gpu_frequency: 1200,
            gpu_memory_used: 512 * 1024 * 1024,
            gpu_memory_total: 8 * 1024 * 1024 * 1024,
            throttle_active: throttle,
            power_state: power,
        }
    }

    #[test]
    fn default_matches_new() {
        let a = GPUManager::new();
        let b = GPUManager::default();
        assert_eq!(a.max_safe_temp, b.max_safe_temp);
        assert_eq!(a.throttle_threshold, b.throttle_threshold);
    }

    #[test]
    fn new_sets_expected_thresholds() {
        let mgr = GPUManager::new();
        assert_eq!(mgr.max_safe_temp, 100.0);
        assert_eq!(mgr.throttle_threshold, 90.0);
    }

    // --- needs_cooling ---

    #[test]
    fn needs_cooling_below_threshold() {
        let mgr = GPUManager::new();
        let m = make_metrics(85.0, 50.0, false, GPUPowerState::Dynamic);
        assert!(!mgr.needs_cooling(&m));
    }

    #[test]
    fn needs_cooling_at_threshold_exact() {
        let mgr = GPUManager::new();
        let m = make_metrics(90.0, 50.0, false, GPUPowerState::Dynamic);
        assert!(!mgr.needs_cooling(&m), "exactly at threshold should not need cooling");
    }

    #[test]
    fn needs_cooling_above_threshold() {
        let mgr = GPUManager::new();
        let m = make_metrics(91.0, 50.0, false, GPUPowerState::Dynamic);
        assert!(mgr.needs_cooling(&m));
    }

    #[test]
    fn needs_cooling_critical_temp() {
        let mgr = GPUManager::new();
        let m = make_metrics(105.0, 99.0, true, GPUPowerState::Throttled);
        assert!(mgr.needs_cooling(&m));
    }

    // --- recommend_power_state ---

    #[test]
    fn recommend_throttled_above_max_safe() {
        let mgr = GPUManager::new();
        assert_eq!(mgr.recommend_power_state(50.0, 101.0), GPUPowerState::Throttled);
    }

    #[test]
    fn recommend_dynamic_when_hot_but_below_max() {
        let mgr = GPUManager::new();
        // temp > throttle_threshold (90) but <= max_safe (100)
        assert_eq!(mgr.recommend_power_state(99.0, 95.0), GPUPowerState::Dynamic);
    }

    #[test]
    fn recommend_maximum_high_utilization_cool() {
        let mgr = GPUManager::new();
        assert_eq!(mgr.recommend_power_state(85.0, 70.0), GPUPowerState::Maximum);
    }

    #[test]
    fn recommend_dynamic_moderate_utilization() {
        let mgr = GPUManager::new();
        assert_eq!(mgr.recommend_power_state(50.0, 70.0), GPUPowerState::Dynamic);
    }

    #[test]
    fn recommend_idle_low_utilization_cool() {
        let mgr = GPUManager::new();
        assert_eq!(mgr.recommend_power_state(5.0, 40.0), GPUPowerState::Idle);
    }

    #[test]
    fn recommend_boundary_utilization_80() {
        let mgr = GPUManager::new();
        // utilization == 80 is NOT > 80, falls to Dynamic
        assert_eq!(mgr.recommend_power_state(80.0, 70.0), GPUPowerState::Dynamic);
    }

    #[test]
    fn recommend_boundary_utilization_20() {
        let mgr = GPUManager::new();
        // utilization == 20 is NOT > 20, falls to Idle
        assert_eq!(mgr.recommend_power_state(20.0, 70.0), GPUPowerState::Idle);
    }

    #[test]
    fn recommend_temp_takes_priority_over_utilization() {
        let mgr = GPUManager::new();
        // Even at 100% utilization, if temp > max_safe → Throttled
        assert_eq!(mgr.recommend_power_state(100.0, 105.0), GPUPowerState::Throttled);
    }

    // --- optimize_for_workload ---

    #[test]
    fn optimize_ai_workload() {
        let mgr = GPUManager::new();
        let actions = mgr.optimize_for_workload("ai");
        assert_eq!(actions.len(), 3);
        assert!(actions.iter().any(|a| a.contains("maximum frequency")));
        assert!(actions.iter().any(|a| a.contains("unified memory")));
    }

    #[test]
    fn optimize_ml_workload_same_as_ai() {
        let mgr = GPUManager::new();
        let ai = mgr.optimize_for_workload("ai");
        let ml = mgr.optimize_for_workload("ml");
        assert_eq!(ai, ml);
    }

    #[test]
    fn optimize_llm_workload_same_as_ai() {
        let mgr = GPUManager::new();
        let ai = mgr.optimize_for_workload("ai");
        let llm = mgr.optimize_for_workload("llm");
        assert_eq!(ai, llm);
    }

    #[test]
    fn optimize_rendering_workload() {
        let mgr = GPUManager::new();
        let actions = mgr.optimize_for_workload("rendering");
        assert_eq!(actions.len(), 2);
        assert!(actions.iter().any(|a| a.contains("sequential access")));
        assert!(actions.iter().any(|a| a.contains("prefetch")));
    }

    #[test]
    fn optimize_video_same_as_rendering() {
        let mgr = GPUManager::new();
        let rendering = mgr.optimize_for_workload("rendering");
        let video = mgr.optimize_for_workload("video");
        assert_eq!(rendering, video);
    }

    #[test]
    fn optimize_idle_workload() {
        let mgr = GPUManager::new();
        let actions = mgr.optimize_for_workload("idle");
        assert_eq!(actions.len(), 2);
        assert!(actions.iter().any(|a| a.contains("idle")));
        assert!(actions.iter().any(|a| a.contains("Disable")));
    }

    #[test]
    fn optimize_unknown_workload_uses_dynamic() {
        let mgr = GPUManager::new();
        let actions = mgr.optimize_for_workload("browsing");
        assert_eq!(actions.len(), 1);
        assert!(actions[0].contains("dynamic"));
    }

    // --- thermal_recommendations ---

    #[test]
    fn thermal_no_recommendations_cool() {
        let mgr = GPUManager::new();
        let m = make_metrics(70.0, 50.0, false, GPUPowerState::Dynamic);
        let recs = mgr.thermal_recommendations(&m);
        assert!(recs.is_empty());
    }

    #[test]
    fn thermal_warning_above_throttle_threshold() {
        let mgr = GPUManager::new();
        let m = make_metrics(95.0, 50.0, false, GPUPowerState::Dynamic);
        let recs = mgr.thermal_recommendations(&m);
        assert_eq!(recs.len(), 1);
        assert!(recs[0].contains("warming"));
        assert!(recs[0].contains("95.0"));
    }

    #[test]
    fn thermal_critical_above_max_safe() {
        let mgr = GPUManager::new();
        let m = make_metrics(105.0, 99.0, false, GPUPowerState::Throttled);
        let recs = mgr.thermal_recommendations(&m);
        assert_eq!(recs.len(), 1);
        assert!(recs[0].contains("CRITICAL"));
        assert!(recs[0].contains("105.0"));
    }

    #[test]
    fn thermal_throttle_active_adds_warning() {
        let mgr = GPUManager::new();
        let m = make_metrics(70.0, 50.0, true, GPUPowerState::Throttled);
        let recs = mgr.thermal_recommendations(&m);
        assert_eq!(recs.len(), 1);
        assert!(recs[0].contains("throttling active"));
    }

    #[test]
    fn thermal_critical_plus_throttle_two_recs() {
        let mgr = GPUManager::new();
        let m = make_metrics(105.0, 99.0, true, GPUPowerState::Throttled);
        let recs = mgr.thermal_recommendations(&m);
        assert_eq!(recs.len(), 2);
    }

    #[test]
    fn thermal_warm_plus_throttle_two_recs() {
        let mgr = GPUManager::new();
        let m = make_metrics(95.0, 80.0, true, GPUPowerState::Dynamic);
        let recs = mgr.thermal_recommendations(&m);
        assert_eq!(recs.len(), 2);
    }

    // --- GPUPowerState ---

    #[test]
    fn power_state_equality() {
        assert_eq!(GPUPowerState::Off, GPUPowerState::Off);
        assert_ne!(GPUPowerState::Off, GPUPowerState::Idle);
        assert_ne!(GPUPowerState::Dynamic, GPUPowerState::Maximum);
    }

    #[test]
    fn power_state_clone() {
        let s = GPUPowerState::Throttled;
        let s2 = s;
        assert_eq!(s, s2);
    }

    #[test]
    fn gpu_metrics_clone() {
        let m = make_metrics(75.0, 60.0, false, GPUPowerState::Dynamic);
        let m2 = m.clone();
        assert_eq!(m2.gpu_temp, 75.0);
        assert_eq!(m2.gpu_utilization, 60.0);
        assert!(!m2.throttle_active);
    }
}
