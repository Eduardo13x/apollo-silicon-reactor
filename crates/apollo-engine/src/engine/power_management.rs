//! Power Management Optimization for macOS / Apple Silicon
//!
//! Manages power profiles, battery awareness, and real hardware detection.
//! On M1/M2/M3, the OS handles frequency scaling — our lever is QoS class routing
//! (see mach_qos.rs). This module is advisory: it reports real state and recommends
//! power profiles, but does NOT directly control CPU frequency.

use crate::engine::sysctl_direct;

// ── Power Mode ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerMode {
    Performance, // Max performance, high power
    Balanced,    // Balanced performance/power
    Efficiency,  // Maximize efficiency, lower power
    Battery,     // Maximum battery life
}

// ── Battery types (merged from battery_optimizer) ────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatteryMode {
    Normal,   // AC power or >50% battery
    LowPower, // 20–50% battery
    Critical, // <20% battery
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
    pub battery_status: BatteryStatus,
    baseline_discharge_rate: f32,
}

/// Detect real CPU power state from the system via sysctl.
///
/// - `core_count_active`: from `sysctl hw.ncpu` (logical CPUs available).
/// - `cpu_frequency_mhz`: from `sysctl hw.cpufrequency_max` (Hz, divided by 1_000_000).
///   Falls back to `sysctl hw.tbfrequency` if cpufrequency_max is unavailable.
/// - `power_draw_watts`: 0.0 (must be provided externally from IOKit/powermetrics).
/// - `thermal_headroom`: 100.0 (must be updated externally from IOKit thermal data).
/// - `idle_percentage`: 50.0 (must be updated externally from system metrics).
pub fn detect_power_state() -> PowerState {
    let core_count = read_sysctl_u64("hw.ncpu").unwrap_or(1) as u32;

    let cpu_freq_mhz = read_sysctl_u64("hw.cpufrequency_max")
        .map(|hz| (hz / 1_000_000) as u32)
        .or_else(|| read_sysctl_u64("hw.tbfrequency").map(|hz| (hz / 1_000_000) as u32))
        .unwrap_or(0);

    PowerState {
        cpu_frequency_mhz: cpu_freq_mhz,
        core_count_active: core_count,
        power_draw_watts: 0.0, // No fake value; must come from IOKit/powermetrics
        thermal_headroom: 100.0, // Assume full headroom until IOKit provides real data
        idle_percentage: 50.0, // Updated externally by daemon cycle
    }
}

fn read_sysctl_u64(key: &str) -> Option<u64> {
    sysctl_direct::read_u64(key)
}

impl PowerManager {
    pub fn new() -> Self {
        let power_state = detect_power_state();

        // Detect initial battery status; fall back to sensible defaults (AC power).
        let battery_status = detect_battery_status().unwrap_or(BatteryStatus {
            percentage: 100,
            time_remaining_minutes: 0,
            is_charging: true,
            charge_rate_percent_per_hour: 0.0,
            discharge_rate_percent_per_hour: 0.0,
        });

        // Initial baseline: use detected discharge rate if discharging, else 0.0.
        let baseline_discharge_rate = if battery_status.discharge_rate_percent_per_hour > 0.0 {
            battery_status.discharge_rate_percent_per_hour
        } else {
            0.0
        };

        Self {
            current_mode: PowerMode::Balanced,
            power_state,
            battery_status,
            baseline_discharge_rate,
        }
    }

    /// Get power recommendation for current mode.
    ///
    /// Advisory only — on Apple Silicon, macOS controls actual frequency.
    /// `target_frequency` is expressed as a percentage of the detected max frequency.
    /// `active_cores` is expressed as a fraction of detected core count.
    pub fn get_recommendation(&self) -> PowerRecommendation {
        let max_freq = self.power_state.cpu_frequency_mhz;
        let max_cores = self.power_state.core_count_active;

        match self.current_mode {
            PowerMode::Performance => PowerRecommendation {
                mode: PowerMode::Performance,
                // 100% of detected max frequency
                target_frequency: max_freq,
                active_cores: max_cores,
                deep_sleep_enabled: false,
                // Estimate: ~1.5W per core at full frequency
                expected_power_mw: (max_cores * 1500).max(2000),
            },
            PowerMode::Balanced => PowerRecommendation {
                mode: PowerMode::Balanced,
                // 75% of detected max frequency
                target_frequency: (max_freq as f32 * 0.75) as u32,
                // 75% of cores (at least 1)
                active_cores: ((max_cores as f32 * 0.75) as u32).max(1),
                deep_sleep_enabled: true,
                expected_power_mw: (max_cores * 1000).max(1000),
            },
            PowerMode::Efficiency => PowerRecommendation {
                mode: PowerMode::Efficiency,
                // 50% of detected max frequency
                target_frequency: (max_freq as f32 * 0.50) as u32,
                // 50% of cores (at least 1)
                active_cores: ((max_cores as f32 * 0.50) as u32).max(1),
                deep_sleep_enabled: true,
                expected_power_mw: (max_cores * 500).max(500),
            },
            PowerMode::Battery => PowerRecommendation {
                mode: PowerMode::Battery,
                // 30% of detected max frequency
                target_frequency: (max_freq as f32 * 0.30) as u32,
                // 25% of cores (at least 1)
                active_cores: ((max_cores as f32 * 0.25) as u32).max(1),
                deep_sleep_enabled: true,
                expected_power_mw: (max_cores * 250).max(250),
            },
        }
    }

    /// Set power mode
    pub fn set_mode(&mut self, mode: PowerMode) {
        self.current_mode = mode;
    }

    /// Estimate power consumption for a workload, parameterized by the real system state.
    ///
    /// Uses the detected `cpu_frequency_mhz` and `core_count_active` as the baseline
    /// instead of hardcoded constants. Returns estimated watts.
    pub fn estimate_power(
        &self,
        frequency_mhz: u32,
        core_count: u32,
        utilization_percent: f32,
    ) -> f32 {
        // Base SoC power floor (always consumed even at idle)
        let base_power: f32 = 1.0;

        // Frequency contribution: proportional to requested vs detected max
        let max_freq = self.power_state.cpu_frequency_mhz.max(1) as f32;
        let freq_factor = (frequency_mhz as f32 / max_freq) * 3.0;

        // Core contribution: proportional to requested vs detected total
        let max_cores = self.power_state.core_count_active.max(1) as f32;
        let core_factor = (core_count as f32 / max_cores) * 2.0;

        // Utilization contribution
        let util_factor = (utilization_percent / 100.0) * 2.0;

        base_power + freq_factor + core_factor + util_factor
    }

    // ── Battery management (merged from battery_optimizer) ───────────────────

    /// Determine battery mode from percentage.
    /// Uses open-ended range for Normal (50..) to handle any value >= 50.
    pub fn get_battery_mode(&self, percentage: u32) -> BatteryMode {
        match percentage {
            50.. => BatteryMode::Normal,
            20..=49 => BatteryMode::LowPower,
            _ => BatteryMode::Critical,
        }
    }

    /// Update battery status and recalibrate baseline discharge rate.
    ///
    /// When the system is discharging, the real discharge rate becomes the new baseline
    /// for time-remaining predictions.
    pub fn update_battery_status(&mut self, status: BatteryStatus) {
        if status.discharge_rate_percent_per_hour > 0.0 {
            self.baseline_discharge_rate = status.discharge_rate_percent_per_hour;
        }
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

    /// Predict time remaining with or without optimizations.
    ///
    /// Uses the dynamically-calibrated `baseline_discharge_rate` (updated from
    /// real battery readings) rather than a hardcoded constant.
    pub fn predict_time_remaining(&self, with_optimization: bool) -> u32 {
        if self.baseline_discharge_rate <= 0.0 {
            // No discharge data available (AC power or no battery)
            return 0;
        }
        if with_optimization {
            let reduced_rate = self.baseline_discharge_rate * 0.6;
            let remaining_percent = self.battery_status.percentage as f32;
            (remaining_percent / reduced_rate * 60.0) as u32
        } else {
            self.battery_status.time_remaining_minutes
        }
    }

    /// Returns true when the machine is running on battery (not charging).
    pub fn is_on_battery(&self) -> bool {
        !self.battery_status.is_charging
    }

    /// Returns the current BatteryMode derived from the current battery percentage.
    pub fn battery_mode_current(&self) -> BatteryMode {
        self.get_battery_mode(self.battery_status.percentage)
    }

    /// Check if battery needs emergency intervention
    pub fn needs_emergency_intervention(&self) -> bool {
        self.battery_status.percentage < 5 || self.battery_status.time_remaining_minutes < 2
    }

    /// Calculate time until critical level (20%)
    pub fn time_to_critical(&self) -> u32 {
        if self.battery_status.percentage <= 20 || self.baseline_discharge_rate <= 0.0 {
            0
        } else {
            let percent_to_critical = self.battery_status.percentage - 20;
            ((percent_to_critical as f32) / self.baseline_discharge_rate * 60.0) as u32
        }
    }

    /// Update thermal headroom from IOKit thermal level.
    ///
    /// `thermal_level` is a 0.0–1.0 value where 0.0 = cool, 1.0 = critical.
    /// Headroom = 100.0 - (thermal_level * 100.0), clamped to [0, 100].
    pub fn update_thermal_headroom(&mut self, thermal_level: f32) {
        self.power_state.thermal_headroom = (100.0 - thermal_level * 100.0).clamp(0.0, 100.0);
    }

    /// Update power draw from IOKit/powermetrics data.
    pub fn update_power_draw(&mut self, watts: f32) {
        self.power_state.power_draw_watts = watts;
    }
}

/// Detect real battery status from the system using IOKit power source APIs.
/// Returns `None` if battery info cannot be determined (e.g., desktop Mac).
pub fn detect_battery_status() -> Option<BatteryStatus> {
    #[cfg(not(target_os = "macos"))]
    {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn IOPSCopyPowerSourcesInfo() -> *const std::ffi::c_void;
            fn IOPSCopyPowerSourcesList(blob: *const std::ffi::c_void) -> *const std::ffi::c_void;
            fn IOPSGetPowerSourceDescription(
                blob: *const std::ffi::c_void,
                ps: *const std::ffi::c_void,
            ) -> *const std::ffi::c_void;
            fn CFArrayGetCount(array: *const std::ffi::c_void) -> i64;
            fn CFArrayGetValueAtIndex(
                array: *const std::ffi::c_void,
                idx: i64,
            ) -> *const std::ffi::c_void;
            fn CFDictionaryGetValue(
                dict: *const std::ffi::c_void,
                key: *const std::ffi::c_void,
            ) -> *const std::ffi::c_void;
            fn CFStringCreateWithCString(
                alloc: *const std::ffi::c_void,
                cstr: *const i8,
                encoding: u32,
            ) -> *const std::ffi::c_void;
            fn CFGetTypeID(cf: *const std::ffi::c_void) -> u64;
            fn CFNumberGetTypeID() -> u64;
            fn CFNumberGetValue(
                number: *const std::ffi::c_void,
                the_type: i64,
                value_ptr: *mut std::ffi::c_void,
            ) -> bool;
            fn CFStringGetTypeID() -> u64;
            fn CFStringGetCString(
                the_string: *const std::ffi::c_void,
                buffer: *mut i8,
                buffer_size: i64,
                encoding: u32,
            ) -> bool;
            fn CFBooleanGetTypeID() -> u64;
            fn CFBooleanGetValue(boolean: *const std::ffi::c_void) -> bool;
            fn CFRelease(cf: *const std::ffi::c_void);
        }

        const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
        const K_CF_NUMBER_SINT32_TYPE: i64 = 3;

        unsafe {
            let blob = IOPSCopyPowerSourcesInfo();
            if blob.is_null() {
                return None;
            }

            let sources = IOPSCopyPowerSourcesList(blob);
            if sources.is_null() {
                CFRelease(blob);
                return None;
            }

            let count = CFArrayGetCount(sources);
            if count <= 0 {
                CFRelease(sources);
                CFRelease(blob);
                return None;
            }

            // Use first power source (InternalBattery).
            let ps = CFArrayGetValueAtIndex(sources, 0);
            let desc = IOPSGetPowerSourceDescription(blob, ps);
            if desc.is_null() {
                CFRelease(sources);
                CFRelease(blob);
                return None;
            }

            let get_int = |key_name: &[u8]| -> Option<i32> {
                let cf_key = CFStringCreateWithCString(
                    std::ptr::null(),
                    key_name.as_ptr() as *const i8,
                    K_CF_STRING_ENCODING_UTF8,
                );
                if cf_key.is_null() {
                    return None;
                }
                let val = CFDictionaryGetValue(desc, cf_key);
                CFRelease(cf_key);
                if val.is_null() || CFGetTypeID(val) != CFNumberGetTypeID() {
                    return None;
                }
                let mut n: i32 = 0;
                if CFNumberGetValue(val, K_CF_NUMBER_SINT32_TYPE, &mut n as *mut _ as *mut _) {
                    Some(n)
                } else {
                    None
                }
            };

            let get_bool = |key_name: &[u8]| -> Option<bool> {
                let cf_key = CFStringCreateWithCString(
                    std::ptr::null(),
                    key_name.as_ptr() as *const i8,
                    K_CF_STRING_ENCODING_UTF8,
                );
                if cf_key.is_null() {
                    return None;
                }
                let val = CFDictionaryGetValue(desc, cf_key);
                CFRelease(cf_key);
                if val.is_null() || CFGetTypeID(val) != CFBooleanGetTypeID() {
                    return None;
                }
                Some(CFBooleanGetValue(val))
            };

            let get_string = |key_name: &[u8]| -> Option<String> {
                let cf_key = CFStringCreateWithCString(
                    std::ptr::null(),
                    key_name.as_ptr() as *const i8,
                    K_CF_STRING_ENCODING_UTF8,
                );
                if cf_key.is_null() {
                    return None;
                }
                let val = CFDictionaryGetValue(desc, cf_key);
                CFRelease(cf_key);
                if val.is_null() || CFGetTypeID(val) != CFStringGetTypeID() {
                    return None;
                }
                let mut buf = [0i8; 128];
                if CFStringGetCString(val, buf.as_mut_ptr(), 128, K_CF_STRING_ENCODING_UTF8) {
                    let s = std::ffi::CStr::from_ptr(buf.as_ptr())
                        .to_string_lossy()
                        .to_string();
                    Some(s)
                } else {
                    None
                }
            };

            let current_capacity = get_int(b"Current Capacity\0").unwrap_or(0);
            let max_capacity = get_int(b"Max Capacity\0").unwrap_or(100);
            let percentage = if max_capacity > 0 {
                ((current_capacity as f64 / max_capacity as f64) * 100.0) as u32
            } else {
                0
            };

            let is_charging = get_bool(b"Is Charging\0").unwrap_or(false);
            let time_to_empty = get_int(b"Time to Empty\0").unwrap_or(-1);
            let time_to_full = get_int(b"Time to Full Charge\0").unwrap_or(-1);

            let power_source_state = get_string(b"Power Source State\0").unwrap_or_default();
            let on_ac = power_source_state.contains("AC") || is_charging;

            let time_remaining_minutes = if is_charging {
                if time_to_full > 0 {
                    time_to_full as u32
                } else {
                    0
                }
            } else if time_to_empty > 0 {
                time_to_empty as u32
            } else {
                0
            };

            let (charge_rate, discharge_rate) = if time_remaining_minutes > 0 {
                if is_charging {
                    let remaining_to_full = 100u32.saturating_sub(percentage);
                    let rate = (remaining_to_full as f32 / time_remaining_minutes as f32) * 60.0;
                    (rate, 0.0)
                } else {
                    let rate = (percentage as f32 / time_remaining_minutes as f32) * 60.0;
                    (0.0, rate)
                }
            } else if on_ac {
                (20.0, 0.0)
            } else {
                (0.0, 10.0)
            };

            CFRelease(sources);
            CFRelease(blob);

            Some(BatteryStatus {
                percentage,
                time_remaining_minutes,
                is_charging: on_ac,
                charge_rate_percent_per_hour: charge_rate,
                discharge_rate_percent_per_hour: discharge_rate,
            })
        }
    }
}

#[cfg(test)]
/// Parse the text output of `pmset -g batt` into a `BatteryStatus`.
/// Returns `None` if no internal battery is found (desktop Mac) or output is unparseable.
fn parse_pmset_battery(text: &str) -> Option<BatteryStatus> {
    // Determine charging state from the first line.
    // "Now drawing from 'Battery Power'" => not charging
    // "Now drawing from 'AC Power'"      => charging (or charged)
    let first_line = text.lines().next().unwrap_or("");
    let drawing_from_battery = first_line.contains("Battery Power");

    // Find the battery detail line (contains "InternalBattery").
    let battery_line = text.lines().find(|l| l.contains("InternalBattery"))?;

    // Parse percentage: look for "XX%"
    let percentage = {
        let mut pct: Option<u32> = None;
        for token in battery_line.split([';', '\t', ' ']) {
            let trimmed = token.trim();
            if let Some(num_str) = trimmed.strip_suffix('%') {
                if let Ok(v) = num_str.parse::<u32>() {
                    pct = Some(v);
                    break;
                }
            }
        }
        pct?
    };

    // Parse state: "charging", "discharging", "charged", "finishing charge", "(no estimate)"
    let is_charging = !drawing_from_battery
        && (battery_line.contains("charging") || battery_line.contains("charged"))
        && !battery_line.contains("discharging");

    // Parse time remaining: "H:MM remaining"
    let time_remaining_minutes = parse_time_remaining(battery_line);

    // Estimate discharge/charge rates from percentage and time remaining.
    let (charge_rate, discharge_rate) = if let Some(mins) = time_remaining_minutes {
        if mins > 0 {
            if is_charging {
                let remaining_to_full = 100u32.saturating_sub(percentage);
                let rate = (remaining_to_full as f32 / mins as f32) * 60.0;
                (rate, 0.0)
            } else {
                let rate = (percentage as f32 / mins as f32) * 60.0;
                (0.0, rate)
            }
        } else {
            (0.0, 0.0)
        }
    } else {
        // No time estimate available — use conservative defaults.
        if is_charging {
            (20.0, 0.0)
        } else {
            (0.0, 10.0)
        }
    };

    Some(BatteryStatus {
        percentage,
        time_remaining_minutes: time_remaining_minutes.unwrap_or(0),
        is_charging,
        charge_rate_percent_per_hour: charge_rate,
        discharge_rate_percent_per_hour: discharge_rate,
    })
}

#[cfg(test)]
/// Parse "H:MM remaining" from a pmset battery line.
/// Returns total minutes, or None if no time estimate is available.
fn parse_time_remaining(line: &str) -> Option<u32> {
    // "(no estimate)" or "not charging" — no time info.
    if line.contains("no estimate") || line.contains("not charging") {
        return None;
    }
    // Look for a token matching "H:MM" followed by "remaining".
    let parts: Vec<&str> = line.split(';').collect();
    for part in &parts {
        let trimmed = part.trim();
        if trimmed.contains("remaining") {
            // Extract "H:MM" from e.g. "3:42 remaining"
            for token in trimmed.split_whitespace() {
                if token.contains(':') {
                    let hm: Vec<&str> = token.split(':').collect();
                    if hm.len() == 2 {
                        if let (Ok(h), Ok(m)) = (hm[0].parse::<u32>(), hm[1].parse::<u32>()) {
                            return Some(h * 60 + m);
                        }
                    }
                }
            }
        }
    }
    None
}

impl Default for PowerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Inputs to the adaptive daemon-cadence floor decision. Cheap, copyable
/// snapshot of the signals the main loop already has in scope.
#[derive(Debug, Clone, Copy)]
pub struct CadenceInputs {
    /// On battery power (any charge level), not on AC.
    pub on_battery: bool,
    /// Battery low (<20%) and discharging — the existing aggressive tier.
    pub battery_low: bool,
    /// Smoothed memory pressure [0,1] from the previous cycle.
    pub pressure_smooth: f64,
    /// HID idle seconds from the previous cycle's user context.
    pub idle_secs: u64,
    /// Sustained high pressure (>0.80) — daemon throttles its own footprint.
    pub high_pressure: bool,
    /// Hierarchical planner predicts a pressure spike / thrashing onset
    /// within <=120s (fresh, confident hint). Pre-arm: never relax the
    /// cadence while a storm is forecast — preparation beats reaction.
    /// (planner.rs Phase 1 consumer, 2026-06-11.)
    pub planner_spike_imminent: bool,
}

/// Minimum inter-cycle floor (ms) for the daemon main loop, BEFORE the
/// fast-tick / dry-run bypass (those force 0 upstream).
///
/// Evolve 2026-06-10 (battery footprint). The daemon polled at 3.3 Hz
/// (300 ms floor) whenever it wasn't battery-critical or thrashing — even
/// when the user had been idle for minutes on battery. At 3.3 Hz each
/// cycle spawns the per-PID enrichment storm + ioreg HID poll: pure
/// energy drain with zero responsiveness benefit while the user is away.
///
/// Real crises still preempt the floor: the reactor thread signals
/// `cycle_condvar` on kqueue Critical / hw_predictor events, waking the
/// loop immediately regardless of this value (and `is_fast_tick` forces
/// the floor to 0 upstream). Memory pressure that builds DURING idle is
/// slow — 0.2-0.33 Hz samples it with ample margin.
///
/// Tiers (first match wins; crisis tiers keep priority over idle tiers):
/// - high pressure (>0.80) OR battery-low → 1000 ms (existing aggressive)
/// - on battery + deep idle (≥10 min) + calm (<0.55) → 5000 ms (0.2 Hz)
/// - on battery + idle (≥2 min) + calm (<0.55)        → 3000 ms (0.33 Hz)
/// - AC + deep idle (≥10 min) + calm (<0.45)          → 1000 ms (mild —
///   fewer background wakeups leave headroom for foreground even plugged in)
/// - else → 300 ms (responsive default)
///
/// [Hellerstein 2004 §9] adaptive control: sample rate must reflect the
/// operating regime. Idle is the lowest-urgency regime.
pub fn adaptive_cycle_floor_ms(i: CadenceInputs) -> u64 {
    const IDLE_SECS: u64 = 120;
    const DEEP_IDLE_SECS: u64 = 600;
    const CALM: f64 = 0.55;
    const AC_CALM: f64 = 0.45;

    if i.high_pressure || i.battery_low {
        return 1000;
    }
    // Planner pre-arm: a forecast spike pins the fast floor BEFORE the
    // crisis arrives, overriding every idle slowdown tier below. Cost of
    // a false positive: ~60s of 3.3Hz sampling. Cost of a miss: the
    // daemon meets the spike asleep at 5000ms.
    if i.planner_spike_imminent {
        return 300;
    }
    if i.on_battery && i.pressure_smooth < CALM {
        if i.idle_secs >= DEEP_IDLE_SECS {
            return 5000;
        }
        if i.idle_secs >= IDLE_SECS {
            return 3000;
        }
    }
    if !i.on_battery && i.idle_secs >= DEEP_IDLE_SECS && i.pressure_smooth < AC_CALM {
        return 1000;
    }
    300
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ci(on_battery: bool, idle: u64, p: f64) -> CadenceInputs {
        CadenceInputs {
            on_battery,
            battery_low: false,
            pressure_smooth: p,
            idle_secs: idle,
            high_pressure: false,
            planner_spike_imminent: false,
        }
    }

    #[test]
    fn adaptive_cadence_idle_on_battery_slows_dramatically() {
        // Active on battery → responsive default.
        assert_eq!(adaptive_cycle_floor_ms(ci(true, 0, 0.30)), 300);
        // Idle 2 min on battery, calm → 0.33 Hz.
        assert_eq!(adaptive_cycle_floor_ms(ci(true, 120, 0.30)), 3000);
        // Deep idle 10 min on battery, calm → 0.2 Hz.
        assert_eq!(adaptive_cycle_floor_ms(ci(true, 600, 0.30)), 5000);
    }

    #[test]
    fn adaptive_cadence_crisis_overrides_idle() {
        // High pressure beats idle — full speed even if idle.
        let mut c = ci(true, 600, 0.90);
        c.high_pressure = true;
        assert_eq!(adaptive_cycle_floor_ms(c), 1000);
        // Battery-low beats idle.
        let mut c2 = ci(true, 600, 0.30);
        c2.battery_low = true;
        assert_eq!(adaptive_cycle_floor_ms(c2), 1000);
        // Idle but NOT calm (pressure ≥0.55) → stays responsive (pressure
        // could climb; don't under-sample).
        assert_eq!(adaptive_cycle_floor_ms(ci(true, 600, 0.60)), 300);
    }

    #[test]
    fn adaptive_cadence_ac_idle_mild_only() {
        // On AC active → default.
        assert_eq!(adaptive_cycle_floor_ms(ci(false, 0, 0.30)), 300);
        // On AC idle 2 min → still default (no battery to save).
        assert_eq!(adaptive_cycle_floor_ms(ci(false, 120, 0.30)), 300);
        // On AC deep idle + very calm → mild 1 Hz (background headroom).
        assert_eq!(adaptive_cycle_floor_ms(ci(false, 600, 0.30)), 1000);
        // On AC deep idle but pressure ≥0.45 → default.
        assert_eq!(adaptive_cycle_floor_ms(ci(false, 600, 0.50)), 300);
    }

    #[test]
    fn test_detect_power_state_returns_real_values() {
        let state = detect_power_state();
        // On any real macOS machine, hw.ncpu should return >= 1
        assert!(
            state.core_count_active >= 1,
            "core_count_active should be >= 1, got {}",
            state.core_count_active
        );
        // cpu_frequency_mhz: on Apple Silicon hw.cpufrequency_max may not exist,
        // but hw.tbfrequency should. Either way the result is >= 0.
        // power_draw_watts should be 0.0 (not invented)
        assert_eq!(state.power_draw_watts, 0.0);
        // thermal_headroom should be 100.0 (full headroom until real data)
        assert_eq!(state.thermal_headroom, 100.0);
    }

    #[test]
    fn test_parse_battery_power() {
        let output = "Now drawing from 'Battery Power'\n \
            -InternalBattery-0 (id=12345678)\t72%; discharging; 3:42 remaining present: true\n";
        let status = parse_pmset_battery(output).unwrap();
        assert_eq!(status.percentage, 72);
        assert!(!status.is_charging);
        assert_eq!(status.time_remaining_minutes, 222); // 3*60+42
        assert!(status.discharge_rate_percent_per_hour > 0.0);
        assert_eq!(status.charge_rate_percent_per_hour, 0.0);
    }

    #[test]
    fn test_parse_ac_power_charging() {
        let output = "Now drawing from 'AC Power'\n \
            -InternalBattery-0 (id=12345678)\t85%; charging; 0:45 remaining present: true\n";
        let status = parse_pmset_battery(output).unwrap();
        assert_eq!(status.percentage, 85);
        assert!(status.is_charging);
        assert_eq!(status.time_remaining_minutes, 45);
        assert!(status.charge_rate_percent_per_hour > 0.0);
        assert_eq!(status.discharge_rate_percent_per_hour, 0.0);
    }

    #[test]
    fn test_parse_ac_power_charged() {
        let output = "Now drawing from 'AC Power'\n \
            -InternalBattery-0 (id=12345678)\t100%; charged; 0:00 remaining present: true\n";
        let status = parse_pmset_battery(output).unwrap();
        assert_eq!(status.percentage, 100);
        assert!(status.is_charging);
        assert_eq!(status.time_remaining_minutes, 0);
    }

    #[test]
    fn test_parse_no_battery_desktop() {
        let output = "Now drawing from 'AC Power'\n";
        let status = parse_pmset_battery(output);
        assert!(status.is_none());
    }

    #[test]
    fn test_parse_no_estimate() {
        let output = "Now drawing from 'Battery Power'\n \
            -InternalBattery-0 (id=12345678)\t55%; discharging; (no estimate) present: true\n";
        let status = parse_pmset_battery(output).unwrap();
        assert_eq!(status.percentage, 55);
        assert!(!status.is_charging);
        assert_eq!(status.time_remaining_minutes, 0);
    }

    #[test]
    fn test_battery_mode_high_percentage() {
        // Verify that percentages > 100 (edge case) map to Normal, not panic
        let manager = PowerManager::new();
        assert_eq!(manager.get_battery_mode(150), BatteryMode::Normal);
        assert_eq!(manager.get_battery_mode(100), BatteryMode::Normal);
        assert_eq!(manager.get_battery_mode(50), BatteryMode::Normal);
    }

    #[test]
    fn test_update_battery_status_calibrates_baseline() {
        let mut manager = PowerManager::new();
        let status = BatteryStatus {
            percentage: 60,
            time_remaining_minutes: 180,
            is_charging: false,
            charge_rate_percent_per_hour: 0.0,
            discharge_rate_percent_per_hour: 15.0,
        };
        manager.update_battery_status(status);
        // baseline_discharge_rate should now be 15.0
        // Verify via time_to_critical which uses baseline_discharge_rate
        let ttc = manager.time_to_critical();
        // (60 - 20) / 15.0 * 60 = 160 minutes
        assert_eq!(ttc, 160);
    }

    #[test]
    fn planner_prearm_pins_fast_floor_over_idle_tiers() {
        // Deep idle on battery would normally relax to 5000ms…
        let mut i = ci(true, 700, 0.30);
        assert_eq!(adaptive_cycle_floor_ms(i.clone()), 5000);
        // …but a forecast spike pins 300ms: prepare BEFORE the storm.
        i.planner_spike_imminent = true;
        assert_eq!(adaptive_cycle_floor_ms(i.clone()), 300);
        // Crisis-now tiers still outrank the pre-arm (already throttling).
        i.high_pressure = true;
        assert_eq!(adaptive_cycle_floor_ms(i), 1000);
    }
}
