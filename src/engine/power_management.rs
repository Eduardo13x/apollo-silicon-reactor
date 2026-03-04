//! Power Management Optimization for macOS / Apple Silicon
//!
//! Manages power profiles, battery awareness, and pmset-based tuning.
//! On M1, the OS handles frequency scaling — our lever is QoS class routing
//! (see mach_qos.rs) plus pmset/sysctl parameters.

// ── Power Mode ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerMode {
    Performance,  // Max performance, high power
    Balanced,     // Balanced performance/power
    Efficiency,   // Maximize efficiency, lower power
    Battery,      // Maximum battery life
}

// ── Battery types (merged from battery_optimizer) ────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatteryMode {
    Normal,    // AC power or >50% battery
    LowPower,  // 20–50% battery
    Critical,  // <20% battery
}

#[derive(Debug, Clone)]
pub struct BatteryStatus {
    pub percentage: u32,
    pub time_remaining_minutes: u32,
    pub is_charging: bool,
    pub charge_rate_percent_per_hour: f32,
    pub discharge_rate_percent_per_hour: f32,
}

#[derive(Debug, Clone)]
pub struct BatteryOptimization {
    pub mode: BatteryMode,
    pub cpu_throttle: bool,
    pub gpu_limit_percent: u32,
    pub screen_brightness_percent: u32,
    pub network_optimization: String,
    pub disable_background_apps: bool,
    pub estimated_extension_minutes: u32,
}

// ── Power state types ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PowerState {
    pub cpu_frequency_mhz: u32,
    pub core_count_active: u32,
    pub power_draw_watts: f32,
    pub thermal_headroom: f32,
    pub idle_percentage: f32,
}

#[derive(Debug, Clone)]
pub struct PowerRecommendation {
    pub mode: PowerMode,
    pub target_frequency: u32,
    pub active_cores: u32,
    pub deep_sleep_enabled: bool,
    pub expected_power_mw: u32,
}

// ── Power Manager ────────────────────────────────────────────────────────────

pub struct PowerManager {
    current_mode: PowerMode,
    pub power_state: PowerState,
    // Battery sub-state
    battery_status: BatteryStatus,
    baseline_discharge_rate: f32,
}

impl PowerManager {
    pub fn new() -> Self {
        Self {
            current_mode: PowerMode::Balanced,
            power_state: PowerState {
                cpu_frequency_mhz: 2400,
                core_count_active: 8,
                power_draw_watts: 5.0,
                thermal_headroom: 15.0,
                idle_percentage: 50.0,
            },
            battery_status: BatteryStatus {
                percentage: 100,
                time_remaining_minutes: 600,
                is_charging: true,
                charge_rate_percent_per_hour: 50.0,
                discharge_rate_percent_per_hour: 10.0,
            },
            baseline_discharge_rate: 10.0,
        }
    }

    /// Get power recommendation for current mode
    pub fn get_recommendation(&self) -> PowerRecommendation {
        match self.current_mode {
            PowerMode::Performance => PowerRecommendation {
                mode: PowerMode::Performance,
                target_frequency: 3200,
                active_cores: 8,
                deep_sleep_enabled: false,
                expected_power_mw: 12000,
            },
            PowerMode::Balanced => PowerRecommendation {
                mode: PowerMode::Balanced,
                target_frequency: 2400,
                active_cores: 6,
                deep_sleep_enabled: true,
                expected_power_mw: 8000,
            },
            PowerMode::Efficiency => PowerRecommendation {
                mode: PowerMode::Efficiency,
                target_frequency: 1800,
                active_cores: 4,
                deep_sleep_enabled: true,
                expected_power_mw: 4000,
            },
            PowerMode::Battery => PowerRecommendation {
                mode: PowerMode::Battery,
                target_frequency: 1200,
                active_cores: 2,
                deep_sleep_enabled: true,
                expected_power_mw: 2000,
            },
        }
    }

    /// Set power mode
    pub fn set_mode(&mut self, mode: PowerMode) {
        self.current_mode = mode;
    }

    /// Optimize idle behaviour based on workload.
    /// On M1 macOS the OS controls C-states; we can only influence
    /// pmset settings like standbydelay and autopoweroff.
    pub fn optimize_idle_states(&mut self) {
        // High idle: encourage deeper sleep via pmset tuning
        // Low idle: keep cores responsive
        // The actual pmset changes are surfaced via get_pmset_recommendations()
    }

    /// Estimate power consumption for a workload
    pub fn estimate_power(
        &self,
        frequency_mhz: u32,
        core_count: u32,
        utilization_percent: f32,
    ) -> f32 {
        let base_power = 2.0; // Watts (SoC base)
        let freq_factor = (frequency_mhz as f32 / 2400.0) * 3.0;
        let core_factor = (core_count as f32 / 8.0) * 2.0;
        let util_factor = (utilization_percent / 100.0) * 2.0;

        base_power + freq_factor + core_factor + util_factor
    }

    /// macOS sysctl recommendations for power management.
    /// Note: `pm.powernap` is actually a `pmset` parameter, not a sysctl.
    /// We include both real sysctls and pmset recommendations here,
    /// tagged by their namespace.
    pub fn get_sysctl_recommendations(&self, mode: PowerMode) -> Vec<(String, String)> {
        let mut recommendations = Vec::new();

        match mode {
            PowerMode::Performance => {
                // Raise file descriptor limits for heavy workloads
                recommendations.push(("kern.maxfiles".to_string(), "200000".to_string()));
                // Disable Power Nap to prevent background activity
                recommendations.push(("pmset.powernap".to_string(), "0".to_string()));
                // Keep disks spinning
                recommendations.push(("pmset.disksleep".to_string(), "0".to_string()));
            }
            PowerMode::Balanced => {
                recommendations.push(("pmset.powernap".to_string(), "1".to_string()));
                recommendations.push(("pmset.disksleep".to_string(), "10".to_string()));
            }
            PowerMode::Efficiency => {
                recommendations.push(("pmset.powernap".to_string(), "1".to_string()));
                recommendations.push(("pmset.disksleep".to_string(), "5".to_string()));
                recommendations.push(("pmset.sleep".to_string(), "10".to_string()));
            }
            PowerMode::Battery => {
                recommendations.push(("pmset.powernap".to_string(), "0".to_string()));
                recommendations.push(("pmset.disksleep".to_string(), "2".to_string()));
                recommendations.push(("pmset.sleep".to_string(), "5".to_string()));
                recommendations.push(("pmset.lessbright".to_string(), "1".to_string()));
            }
        }

        recommendations
    }

    /// Check if CPU frequency scaling would be beneficial
    pub fn needs_frequency_scaling(&self) -> bool {
        self.power_state.idle_percentage > 70.0 && self.power_state.cpu_frequency_mhz > 1800
    }

    /// Get thermal headroom for increasing frequency
    pub fn get_thermal_headroom(&self) -> f32 {
        self.power_state.thermal_headroom
    }

    // ── Battery management (merged from battery_optimizer) ───────────────────

    /// Determine battery mode from percentage
    pub fn get_battery_mode(&self, percentage: u32) -> BatteryMode {
        match percentage {
            50..=100 => BatteryMode::Normal,
            20..=49 => BatteryMode::LowPower,
            _ => BatteryMode::Critical,
        }
    }

    /// Update battery status
    pub fn update_battery_status(&mut self, status: BatteryStatus) {
        self.battery_status = status;
    }

    /// Get optimization for current battery state
    pub fn get_battery_optimization(&self) -> BatteryOptimization {
        let mode = self.get_battery_mode(self.battery_status.percentage);

        match mode {
            BatteryMode::Normal => BatteryOptimization {
                mode,
                cpu_throttle: false,
                gpu_limit_percent: 100,
                screen_brightness_percent: 100,
                network_optimization: "balanced".to_string(),
                disable_background_apps: false,
                estimated_extension_minutes: 0,
            },
            BatteryMode::LowPower => BatteryOptimization {
                mode,
                cpu_throttle: true,
                gpu_limit_percent: 80,
                screen_brightness_percent: 70,
                network_optimization: "efficient".to_string(),
                disable_background_apps: true,
                estimated_extension_minutes: 30,
            },
            BatteryMode::Critical => BatteryOptimization {
                mode,
                cpu_throttle: true,
                gpu_limit_percent: 50,
                screen_brightness_percent: 40,
                network_optimization: "minimal".to_string(),
                disable_background_apps: true,
                estimated_extension_minutes: 60,
            },
        }
    }

    /// Predict time remaining with or without optimizations
    pub fn predict_time_remaining(&self, with_optimization: bool) -> u32 {
        if with_optimization {
            let reduced_rate = self.baseline_discharge_rate * 0.6;
            let remaining_percent = self.battery_status.percentage as f32;
            (remaining_percent / reduced_rate * 60.0) as u32
        } else {
            self.battery_status.time_remaining_minutes
        }
    }

    /// Aggressive power actions for critical battery
    pub fn get_critical_actions(&self) -> Vec<String> {
        vec![
            "Reduce CPU frequency to minimum".to_string(),
            "Disable GPU acceleration".to_string(),
            "Disable background app refresh".to_string(),
            "Reduce screen brightness to 40%".to_string(),
            "Disable Wi-Fi scanning".to_string(),
            "Disable Bluetooth".to_string(),
            "Close non-essential applications".to_string(),
        ]
    }

    /// Estimate power savings from battery mode
    pub fn estimate_power_savings_percent(&self, mode: BatteryMode) -> f32 {
        match mode {
            BatteryMode::Normal => 0.0,
            BatteryMode::LowPower => 25.0,
            BatteryMode::Critical => 50.0,
        }
    }

    /// macOS services that should be disabled in each battery mode
    pub fn get_apps_to_disable(&self, mode: BatteryMode) -> Vec<String> {
        match mode {
            BatteryMode::Normal => vec![],
            BatteryMode::LowPower => vec![
                "Spotlight indexing".to_string(),
                "Time Machine".to_string(),
                "iCloud sync".to_string(),
            ],
            BatteryMode::Critical => vec![
                "Spotlight indexing".to_string(),
                "Time Machine".to_string(),
                "iCloud sync".to_string(),
                "Dropbox sync".to_string(),
                "Mail fetch".to_string(),
                "Photo library sync".to_string(),
            ],
        }
    }

    /// Check if battery needs emergency intervention
    pub fn needs_emergency_intervention(&self) -> bool {
        self.battery_status.percentage < 5 || self.battery_status.time_remaining_minutes < 2
    }

    /// Calculate time until critical level (20%)
    pub fn time_to_critical(&self) -> u32 {
        if self.battery_status.percentage <= 20 {
            0
        } else {
            let percent_to_critical = self.battery_status.percentage - 20;
            ((percent_to_critical as f32) / self.baseline_discharge_rate * 60.0) as u32
        }
    }
}

impl Default for PowerManager {
    fn default() -> Self {
        Self::new()
    }
}
