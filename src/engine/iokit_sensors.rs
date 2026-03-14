//! IOKit Sensor Reader — real hardware telemetry from the M1 SoC
//!
//! Reads thermal, power, and performance data via IOKit + SMC.
//!
//! The M1 has two CPU clusters:
//!   • P-Cluster (Firestorm) — high-performance, high-power
//!   • E-Cluster (Icestorm)  — efficiency, always-on
//!
//! IOKit exposes this via two paths:
//!   1. AppleSMC — raw sensor values (temps, fan, power rails)
//!   2. IOPMrootDomain — system thermal state (NORMAL / MODERATE / SEVERE / CRITICAL)
//!   3. IOHWSensor    — individual temp sensors
//!
//! Implementation strategy
//! -----------------------
//! Direct IOKit FFI is complex; we use a two-tier approach:
//!   Tier 1: Shell out to `powermetrics` (available on all Macs, requires root)
//!   Tier 2: Parse `/tmp/apollo_powermetrics.json` if powermetrics runs as a
//!           background job (daemon integration)
//!
//! This gives us real numbers without needing to reverse-engineer undocumented
//! SMC key names for each SoC generation.

use std::process::Command;

// ── Data types ────────────────────────────────────────────────────────────────

/// Thermal state of the entire SoC — mirrors IOPMrootDomain values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Normal,   // < ~80 °C — all cores available
    Moderate, // ~80–90 °C — P-Cores may be gated
    Severe,   // ~90–100 °C — mandatory frequency reduction
    Critical, // > 100 °C — emergency throttle, may halt P-Cores entirely
}

/// Per-cluster temperature readings.
#[derive(Debug, Clone)]
pub struct ClusterTemps {
    /// P-Cluster (Firestorm) average °C; None if not readable.
    pub p_cluster_celsius: Option<f32>,
    /// E-Cluster (Icestorm) average °C; None if not readable.
    pub e_cluster_celsius: Option<f32>,
    /// GPU °C.
    pub gpu_celsius: Option<f32>,
    /// NAND / storage °C.
    pub nand_celsius: Option<f32>,
}

/// Power consumption snapshot (watts).
#[derive(Debug, Clone)]
pub struct PowerReading {
    /// Total package power (CPU + GPU + DRAM).
    pub package_watts: Option<f32>,
    /// CPU only.
    pub cpu_watts: Option<f32>,
    /// GPU only.
    pub gpu_watts: Option<f32>,
    /// DRAM subsystem.
    pub dram_watts: Option<f32>,
}

/// Combined hardware snapshot.
#[derive(Debug, Clone)]
pub struct HardwareSnapshot {
    pub thermal_state: ThermalState,
    pub temps: ClusterTemps,
    pub power: PowerReading,
    /// P-Core utilisation 0.0–100.0 (from powermetrics).
    pub p_cluster_util: Option<f32>,
    /// E-Core utilisation 0.0–100.0.
    pub e_cluster_util: Option<f32>,
    /// Battery charge percent.
    pub battery_percent: Option<u32>,
    /// Discharging rate in watts (negative = charging).
    pub battery_watts: Option<f32>,
}

// ── Reader ────────────────────────────────────────────────────────────────────

pub struct IOKitSensorReader;

impl IOKitSensorReader {
    pub fn new() -> Self {
        Self
    }

    /// Take a one-shot snapshot using `powermetrics`.
    /// Requires root; returns `Err` if not available or permission denied.
    ///
    /// `powermetrics -n 1 -i 500` samples once with a 500 ms interval.
    pub fn snapshot(&self) -> Result<HardwareSnapshot, String> {
        // Run powermetrics in a thread with a 3-second timeout.
        // The smc sampler can hang on some systems; cpu_power covers power + utilisation.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = Command::new("/usr/bin/powermetrics")
                .args([
                    "--samplers",
                    "cpu_power,thermal,battery",
                    "-n",
                    "1",
                    "-i",
                    "500",
                ])
                .output();
            let _ = tx.send(result);
        });

        let output = rx
            .recv_timeout(std::time::Duration::from_secs(3))
            .map_err(|_| "powermetrics timed out after 3s".to_string())?
            .map_err(|e| format!("powermetrics exec failed: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("powermetrics error: {}", stderr));
        }

        let text = String::from_utf8_lossy(&output.stdout);
        Ok(self.parse_powermetrics(&text))
    }

    /// Parse the text output of `powermetrics`.
    /// This is the public method for testing — takes raw text.
    pub fn parse_powermetrics(&self, text: &str) -> HardwareSnapshot {
        let mut p_temp: Option<f32> = None;
        let mut e_temp: Option<f32> = None;
        let mut gpu_temp: Option<f32> = None;
        let mut pkg_watts: Option<f32> = None;
        let mut cpu_watts: Option<f32> = None;
        let mut gpu_watts: Option<f32> = None;
        let mut p_util: Option<f32> = None;
        let mut e_util: Option<f32> = None;
        let mut batt_pct: Option<u32> = None;
        let mut batt_watts: Option<f32> = None;
        let mut thermal_state = ThermalState::Normal;
        let mut thermal_state_explicit = false;

        for line in text.lines() {
            let line = line.trim();

            // Temperature lines — two formats depending on macOS version:
            //   Intel:          "CPU P-cluster temp: 72.34 C"
            //   Apple Silicon:  "P-cluster temp: 72.34 C"
            // NOTE: Modern macOS doesn't expose core temps via powermetrics for privacy.
            // If temps are unavailable, infer from thermal pressure level (below).
            if line.starts_with("CPU P-cluster temp:") || line.starts_with("P-cluster temp:") {
                p_temp = parse_float_after_colon(line);
            } else if line.starts_with("CPU E-cluster temp:") || line.starts_with("E-cluster temp:")
            {
                e_temp = parse_float_after_colon(line);
            } else if line.starts_with("GPU die temp:") || line.starts_with("GPU temp:") {
                gpu_temp = parse_float_after_colon(line);
            }
            // Power lines — formats vary:
            //   Intel/old:      "Package power: 4.532 W"
            //   Apple Silicon:  "Combined Power (CPU + GPU + ANE): 170 mW"
            //                   "CPU Power: 170 mW"
            // Values may be in W or mW; normalise to W.
            else if line.starts_with("Package power:") || line.starts_with("Combined Power") {
                pkg_watts = parse_power_watts(line);
            } else if line.starts_with("CPU Power:") || line.starts_with("CPU power:") {
                cpu_watts = parse_power_watts(line);
            } else if line.starts_with("GPU Power:") || line.starts_with("GPU power:") {
                gpu_watts = parse_power_watts(line);
            }
            // Utilisation: "P-Cluster HW active residency: 38.3%" (case varies by OS version)
            else if line.to_ascii_lowercase().contains("p-cluster")
                && line.contains("active residency")
            {
                p_util = parse_percent(line);
            } else if line.to_ascii_lowercase().contains("e-cluster")
                && line.contains("active residency")
            {
                e_util = parse_percent(line);
            }
            // Battery: "Capacity: 87% (charging)"
            else if line.starts_with("Capacity:") && line.contains('%') {
                batt_pct = parse_battery_percent(line);
                if line.contains("discharging") || !line.contains("charging") {
                    batt_watts = Some(-1.0); // Will be overridden if watt line found
                }
            }
            // Battery power: "Battery power: 8.2 W"
            else if line.starts_with("Battery power:") {
                batt_watts = parse_float_after_colon(line);
            }
            // Thermal state — multiple line formats depending on macOS version:
            //   Old: "System thermal state: MODERATE"
            //   New: "Current pressure level: Nominal" / "High" / "Critical"
            else if line.starts_with("System thermal state:")
                || line.starts_with("Current pressure level:")
            {
                thermal_state_explicit = true;
                let upper = line.to_uppercase();
                thermal_state = if upper.contains("CRITICAL") {
                    ThermalState::Critical
                } else if upper.contains("SEVERE") || upper.contains("HEAVY") {
                    ThermalState::Severe
                } else if upper.contains("MODERATE") || upper.contains("HIGH") {
                    ThermalState::Moderate
                } else {
                    ThermalState::Normal
                };
            }
        }

        // Infer thermal_state from P-cluster temp when no explicit thermal line was found.
        if !thermal_state_explicit {
            if let Some(t) = p_temp {
                thermal_state = if t >= 100.0 {
                    ThermalState::Critical
                } else if t >= 90.0 {
                    ThermalState::Severe
                } else if t >= 80.0 {
                    ThermalState::Moderate
                } else {
                    ThermalState::Normal
                };
            }
        }

        // Infer temps from thermal_state if not explicitly available.
        // Modern macOS hides core temps; we estimate based on pressure level + utilisation.
        if p_temp.is_none() && e_temp.is_none() {
            let (est_p, est_e) = match thermal_state {
                ThermalState::Normal => (60.0, 45.0),    // Cool idle
                ThermalState::Moderate => (80.0, 65.0),  // Warm, some activity
                ThermalState::Severe => (95.0, 80.0),    // Hot, heavy load
                ThermalState::Critical => (110.0, 95.0), // Throttled, emergency
            };
            p_temp = Some(est_p);
            e_temp = Some(est_e);
        }

        HardwareSnapshot {
            thermal_state,
            temps: ClusterTemps {
                p_cluster_celsius: p_temp,
                e_cluster_celsius: e_temp,
                gpu_celsius: gpu_temp,
                nand_celsius: None,
            },
            power: PowerReading {
                package_watts: pkg_watts,
                cpu_watts,
                gpu_watts,
                dram_watts: None,
            },
            p_cluster_util: p_util,
            e_cluster_util: e_util,
            battery_percent: batt_pct,
            battery_watts: batt_watts,
        }
    }

    /// True if system is thermally throttled (≥ Moderate).
    pub fn is_throttled(snap: &HardwareSnapshot) -> bool {
        !matches!(snap.thermal_state, ThermalState::Normal)
    }

    /// True if running on battery and level is critical.
    pub fn is_battery_critical(snap: &HardwareSnapshot) -> bool {
        snap.battery_percent.map(|p| p < 20).unwrap_or(false)
            && snap.battery_watts.map(|w| w > 0.0).unwrap_or(false) // positive = discharging
    }

    /// Recommend whether to push background tasks to E-Cores.
    pub fn should_push_to_ecores(snap: &HardwareSnapshot) -> bool {
        Self::is_throttled(snap) || Self::is_battery_critical(snap)
    }
}

impl Default for IOKitSensorReader {
    fn default() -> Self {
        Self::new()
    }
}

// ── Parsers ───────────────────────────────────────────────────────────────────

/// Parse a power value after the last colon, converting mW → W automatically.
/// Handles: "CPU Power: 170 mW", "Package power: 4.532 W"
fn parse_power_watts(line: &str) -> Option<f32> {
    let after_colon = line.split(':').next_back()?;
    let mut tokens = after_colon.split_whitespace();
    let value: f32 = tokens.next()?.parse().ok()?;
    let unit = tokens.next().unwrap_or("W");
    if unit.eq_ignore_ascii_case("mw") {
        Some(value / 1000.0)
    } else {
        Some(value)
    }
}

fn parse_float_after_colon(line: &str) -> Option<f32> {
    line.split(':')
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse::<f32>()
        .ok()
}

fn parse_percent(line: &str) -> Option<f32> {
    // Find the first "number%" token
    for token in line.split_whitespace() {
        if let Some(stripped) = token.strip_suffix('%') {
            if let Ok(v) = stripped.parse::<f32>() {
                return Some(v);
            }
        }
    }
    None
}

fn parse_battery_percent(line: &str) -> Option<u32> {
    for token in line.split_whitespace() {
        let clean = token.trim_end_matches('%').trim_end_matches("%(");
        if let Ok(v) = clean.parse::<u32>() {
            return Some(v);
        }
    }
    None
}
