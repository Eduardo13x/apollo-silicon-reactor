//! SMC Direct Read — sub-100µs power, thermal, lid, sleep/wake, and battery telemetry.
//!
//! Replaces `powermetrics` subprocess (500ms blocking) with direct IOKit
//! `IOConnectCallStructMethod` calls via a C bridge (`smc_bridge.c`).
//!
//! # Keys read
//!
//! | Key  | Type | Description                          |
//! |------|------|--------------------------------------|
//! | PSTR | flt  | System total power (watts)           |
//! | MSLD | ui8  | Lid state (0=open, 1=closed)         |
//! | CLSP | ui64 | Last sleep timestamp (µs)            |
//! | CLWK | ui64 | Last wake timestamp (µs)             |
//! | B0TE | ui16 | Battery time to empty (minutes)      |
//! | B0TF | ui16 | Battery time to full (minutes)       |
//! | PDTR | flt  | Charger delivery power (watts)       |
//! | TC0P | flt  | CPU proximity temperature (°C)       |
//! | TG0P | flt  | GPU proximity temperature (°C)       |
//! | TB0T | flt  | Battery cell temperature (°C)        |
//! | PCPC | flt  | P-cluster power (watts)              |
//! | PCPG | flt  | GPU power (watts)                    |
//! | ID0R | flt  | DC-In current draw (amps)            |
//! | VC0C | flt  | CPU core voltage (V)                 |
//!
//! # Safety
//!
//! The C bridge handles IOKit calls. Rust reads only safe value types.
//! Connection lives for the process lifetime (no Drop — same as IOReport).

// ── SMC key constants (ASCII → u32 big-endian) ─────────────────────────────

const KEY_PSTR: u32 = 0x5053_5452; // 'PSTR' — system total power
const KEY_MSLD: u32 = 0x4D53_4C44; // 'MSLD' — lid state
const KEY_CLSP: u32 = 0x434C_5350; // 'CLSP' — last sleep timestamp
const KEY_CLWK: u32 = 0x434C_574B; // 'CLWK' — last wake timestamp
const KEY_B0TE: u32 = 0x4230_5445; // 'B0TE' — battery time to empty
const KEY_B0TF: u32 = 0x4230_5446; // 'B0TF' — battery time to full
const KEY_PDTR: u32 = 0x5044_5452; // 'PDTR' — charger delivery power

// Thermal keys — Intel
const KEY_TC0P: u32 = 0x5443_3050; // 'TC0P' — CPU proximity temp
const KEY_TG0P: u32 = 0x5447_3050; // 'TG0P' — GPU proximity temp
const KEY_TB0T: u32 = 0x5442_3054; // 'TB0T' — battery cell temp
const KEY_TB1T: u32 = 0x5442_3154; // 'TB1T' — battery cell 2 temp

// Thermal keys — Apple Silicon (M1/M2/M3)
// These keys use 'sp78' (signed 7.8 fixed-point, 2 bytes) not 'flt'.
const KEY_TP01: u32 = 0x5470_3031; // 'Tp01' — CPU perf core 1 temp
const KEY_TP09: u32 = 0x5470_3039; // 'Tp09' — CPU efficiency core temp
const KEY_TG0F: u32 = 0x5467_3066; // 'Tg0f' — GPU die temp

// Per-component power keys
const KEY_PCPC: u32 = 0x5043_5043; // 'PCPC' — P-cluster power
const KEY_PCPG: u32 = 0x5043_5047; // 'PCPG' — GPU power
const KEY_ID0R: u32 = 0x4944_3052; // 'ID0R' — DC-In current (amps)
const KEY_VC0C: u32 = 0x5643_3043; // 'VC0C' — CPU core voltage

// SMC data type codes
const TYPE_FLT: u32 = 0x666C_7420; // 'flt ' — IEEE 754 float32
const TYPE_SP78: u32 = 0x7370_3738; // 'sp78' — signed 7.8 fixed-point
const TYPE_FP88: u32 = 0x6670_3838; // 'fp88' — unsigned 8.8 fixed-point

// ── C bridge FFI ────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
#[link(name = "smc_bridge", kind = "static")]
#[link(name = "IOKit", kind = "framework")]
extern "C" {
    fn apollo_smc_open() -> u32;
    #[allow(dead_code)]
    fn apollo_smc_close(conn: u32);
    fn apollo_smc_read_key(
        conn: u32,
        key: u32,
        out_bytes: *mut u8,
        out_size: *mut u32,
        out_type: *mut u32,
    ) -> i32;
}

// ── Snapshot ────────────────────────────────────────────────────────────────

/// Point-in-time SMC reading.
#[derive(Debug, Clone)]
pub struct SmcSnapshot {
    /// System total power consumption (watts). PSTR key.
    pub system_power_watts: Option<f64>,
    /// Whether the lid is closed. MSLD key.
    pub lid_closed: bool,
    /// Last sleep timestamp in microseconds (SMC epoch). CLSP key.
    pub last_sleep_us: u64,
    /// Last wake timestamp in microseconds (SMC epoch). CLWK key.
    pub last_wake_us: u64,
    /// Battery time to empty (minutes). B0TE key. None if charging or absent.
    pub battery_time_to_empty_min: Option<u16>,
    /// Battery time to full (minutes). B0TF key. None if discharging or absent.
    pub battery_time_to_full_min: Option<u16>,
    /// Charger delivery power (watts). PDTR key. None if no charger.
    pub charger_watts: Option<f64>,

    // ── Thermal (°C) ────────────────────────────────────────────────────
    /// CPU proximity temperature (°C). TC0P key.
    pub cpu_temp_celsius: Option<f64>,
    /// GPU proximity temperature (°C). TG0P key.
    pub gpu_temp_celsius: Option<f64>,
    /// Battery cell temperature (°C). Max of TB0T/TB1T.
    pub battery_temp_celsius: Option<f64>,

    // ── Per-component power (watts) ─────────────────────────────────────
    /// P-cluster (performance cores) power. PCPC key.
    pub p_cluster_watts: Option<f64>,
    /// GPU power. PCPG key.
    pub gpu_watts: Option<f64>,
    /// DC-In current draw (amps). ID0R key.
    pub dc_in_current_amps: Option<f64>,
    /// CPU core voltage (V). VC0C key. Drops indicate thermal throttling.
    pub cpu_voltage: Option<f64>,
}

impl SmcSnapshot {
    /// Whether CPU is in thermal danger zone (≥90°C).
    pub fn cpu_thermal_critical(&self) -> bool {
        self.cpu_temp_celsius.map(|t| t >= 90.0).unwrap_or(false)
    }

    /// Whether battery is overheating (≥45°C safety threshold).
    pub fn battery_overheating(&self) -> bool {
        self.battery_temp_celsius.map(|t| t >= 45.0).unwrap_or(false)
    }

    /// Whether CPU voltage has dropped below nominal (~0.7V idle, ~1.1V load).
    /// Voltage < 0.65V on M1 likely indicates firmware-level thermal throttle.
    pub fn voltage_throttled(&self) -> bool {
        self.cpu_voltage.map(|v| v > 0.0 && v < 0.65).unwrap_or(false)
    }
}

// ── Reader ──────────────────────────────────────────────────────────────────

/// Direct SMC reader. Holds an IOKit connection to AppleSMC.
pub struct SmcDirectReader {
    conn: u32,
    pub available: bool,
}

impl SmcDirectReader {
    /// Open a connection to the SMC. Safe to call without root (returns unavailable).
    pub fn new() -> Self {
        #[cfg(target_os = "macos")]
        {
            let conn = unsafe { apollo_smc_open() };
            Self {
                conn,
                available: conn != 0,
            }
        }

        #[cfg(not(target_os = "macos"))]
        Self {
            conn: 0,
            available: false,
        }
    }

    /// Diagnostic: probe a list of known SMC keys and return which ones are readable.
    /// Call at startup to discover which keys exist on this hardware.
    pub fn probe_available_keys(&self) -> Vec<(String, f64)> {
        if !self.available { return vec![]; }
        // Comprehensive list of known macOS SMC keys across Intel + Apple Silicon.
        let candidates: &[(u32, &str)] = &[
            // Intel thermal
            (KEY_TC0P, "TC0P"), (KEY_TG0P, "TG0P"),
            // Apple Silicon thermal (M1/M2 variants)
            (KEY_TP01, "Tp01"), (KEY_TP09, "Tp09"), (KEY_TG0F, "Tg0f"),
            (0x5470_3030, "Tp00"), (0x5470_3032, "Tp02"), (0x5470_3033, "Tp03"),
            (0x5470_3034, "Tp04"), (0x5470_3035, "Tp05"),
            (0x5467_3061, "Tg0a"), (0x5467_306A, "Tg0j"),
            // M1 Air specific: PMU temps
            (0x5470_3041, "TpOA"), // sometimes 'TpOA'
            // System/Package
            (KEY_PSTR, "PSTR"),
            (KEY_PCPC, "PCPC"), (KEY_PCPG, "PCPG"),
            // Battery
            (KEY_TB0T, "TB0T"), (KEY_TB1T, "TB1T"),
            (KEY_PDTR, "PDTR"),
            // Voltage/current
            (KEY_VC0C, "VC0C"), (KEY_ID0R, "ID0R"),
            // Die temps (Apple Silicon 'T' prefix variants)
            (0x5464_3030, "Td00"), (0x5464_3031, "Td01"),
            // NAND temp
            (0x544E_3054, "TN0T"),
            // PMU temp
            (0x5470_4D30, "TpM0"),
            // Airflow
            (0x5441_3050, "TA0P"), (0x5441_3150, "TA1P"),
            // Misc
            (KEY_MSLD, "MSLD"), (KEY_CLSP, "CLSP"), (KEY_CLWK, "CLWK"),
            (KEY_B0TE, "B0TE"), (KEY_B0TF, "B0TF"),
        ];
        let mut found = Vec::new();
        for &(key, name) in candidates {
            if let Some(val) = self.read_numeric(key) {
                found.push((name.to_string(), val));
            }
        }
        found
    }

    /// Read all SMC keys into a snapshot.
    pub fn read_snapshot(&self) -> Option<SmcSnapshot> {
        if !self.available {
            return None;
        }

        // Battery temp: max of two cells (TB0T / TB1T).
        let bt0 = self.read_float(KEY_TB0T).filter(|&v| v > -40.0 && v < 80.0);
        let bt1 = self.read_float(KEY_TB1T).filter(|&v| v > -40.0 && v < 80.0);
        let battery_temp = match (bt0, bt1) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            _ => None,
        };

        // CPU temp: try Intel key first (TC0P), then Apple Silicon fallbacks (Tp01, Tp09).
        let cpu_temp = self.read_numeric(KEY_TC0P)
            .or_else(|| self.read_numeric(KEY_TP01))
            .or_else(|| self.read_numeric(KEY_TP09))
            .filter(|&v| v > -40.0 && v < 150.0);

        // GPU temp: try Intel key first (TG0P), then Apple Silicon (Tg0f).
        let gpu_temp = self.read_numeric(KEY_TG0P)
            .or_else(|| self.read_numeric(KEY_TG0F))
            .filter(|&v| v > -40.0 && v < 150.0);

        Some(SmcSnapshot {
            system_power_watts: self.read_numeric(KEY_PSTR),
            lid_closed: self.read_u8(KEY_MSLD).map(|v| v != 0).unwrap_or(false),
            last_sleep_us: self.read_u64(KEY_CLSP).unwrap_or(0),
            last_wake_us: self.read_u64(KEY_CLWK).unwrap_or(0),
            battery_time_to_empty_min: self.read_u16(KEY_B0TE).filter(|&v| v > 0 && v < 1500),
            battery_time_to_full_min: self.read_u16(KEY_B0TF).filter(|&v| v > 0 && v < 1500),
            charger_watts: self.read_numeric(KEY_PDTR).filter(|&v| v > 0.0),
            cpu_temp_celsius: cpu_temp,
            gpu_temp_celsius: gpu_temp,
            battery_temp_celsius: battery_temp,
            p_cluster_watts: self.read_numeric(KEY_PCPC).filter(|&v| v >= 0.0 && v < 50.0),
            gpu_watts: self.read_numeric(KEY_PCPG).filter(|&v| v >= 0.0 && v < 50.0),
            dc_in_current_amps: self.read_numeric(KEY_ID0R).filter(|&v| v >= 0.0 && v < 10.0),
            cpu_voltage: self.read_numeric(KEY_VC0C).filter(|&v| v > 0.0 && v < 2.0),
        })
    }

    /// Read a numeric SMC value, auto-detecting the data type.
    /// Handles 'flt ' (IEEE float32), 'sp78' (signed 7.8 fixed-point),
    /// and 'fp88' (unsigned 8.8 fixed-point) — covers both Intel and
    /// Apple Silicon SMC key types.
    fn read_numeric(&self, key: u32) -> Option<f64> {
        #[cfg(target_os = "macos")]
        {
            let mut bytes = [0u8; 32];
            let mut size: u32 = 0;
            let mut data_type: u32 = 0;

            let ret = unsafe {
                apollo_smc_read_key(
                    self.conn,
                    key,
                    bytes.as_mut_ptr(),
                    &mut size,
                    &mut data_type,
                )
            };

            if ret != 0 {
                return None;
            }

            match data_type {
                TYPE_FLT if size >= 4 => {
                    let val = f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    if val.is_finite() { Some(val as f64) } else { None }
                }
                TYPE_SP78 if size >= 2 => {
                    // Signed 7.8 fixed-point: 1 sign bit + 7 integer + 8 fraction
                    let raw = i16::from_be_bytes([bytes[0], bytes[1]]);
                    Some(raw as f64 / 256.0)
                }
                TYPE_FP88 if size >= 2 => {
                    // Unsigned 8.8 fixed-point: 8 integer + 8 fraction
                    let raw = u16::from_be_bytes([bytes[0], bytes[1]]);
                    Some(raw as f64 / 256.0)
                }
                _ if size >= 4 => {
                    // Unknown type — try float interpretation as last resort
                    let val = f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    if val.is_finite() && val.abs() < 1e6 { Some(val as f64) } else { None }
                }
                _ => None,
            }
        }

        #[cfg(not(target_os = "macos"))]
        None
    }

    /// Legacy: read as flt only (used by read_float callers that are being migrated).
    fn read_float(&self, key: u32) -> Option<f64> {
        self.read_numeric(key)
    }

    /// Read a single unsigned byte (SMC type 'ui8 ').
    fn read_u8(&self, key: u32) -> Option<u8> {
        #[cfg(target_os = "macos")]
        {
            let mut bytes = [0u8; 32];
            let mut size: u32 = 0;
            let mut data_type: u32 = 0;

            let ret = unsafe {
                apollo_smc_read_key(
                    self.conn,
                    key,
                    bytes.as_mut_ptr(),
                    &mut size,
                    &mut data_type,
                )
            };

            if ret != 0 || size < 1 {
                return None;
            }

            Some(bytes[0])
        }

        #[cfg(not(target_os = "macos"))]
        None
    }

    /// Read a 2-byte big-endian unsigned integer (SMC type 'ui16').
    fn read_u16(&self, key: u32) -> Option<u16> {
        #[cfg(target_os = "macos")]
        {
            let mut bytes = [0u8; 32];
            let mut size: u32 = 0;
            let mut data_type: u32 = 0;

            let ret = unsafe {
                apollo_smc_read_key(
                    self.conn,
                    key,
                    bytes.as_mut_ptr(),
                    &mut size,
                    &mut data_type,
                )
            };

            if ret != 0 || size < 2 {
                return None;
            }

            Some(u16::from_be_bytes([bytes[0], bytes[1]]))
        }

        #[cfg(not(target_os = "macos"))]
        None
    }

    /// Read an 8-byte big-endian unsigned integer (SMC type 'ui64').
    fn read_u64(&self, key: u32) -> Option<u64> {
        #[cfg(target_os = "macos")]
        {
            let mut bytes = [0u8; 32];
            let mut size: u32 = 0;
            let mut data_type: u32 = 0;

            let ret = unsafe {
                apollo_smc_read_key(
                    self.conn,
                    key,
                    bytes.as_mut_ptr(),
                    &mut size,
                    &mut data_type,
                )
            };

            if ret != 0 || size < 8 {
                return None;
            }

            Some(u64::from_be_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]))
        }

        #[cfg(not(target_os = "macos"))]
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smc_opens_without_panic() {
        // Construction must be safe even without root / on CI.
        let reader = SmcDirectReader::new();
        // available may be true (macOS with SMC) or false — no panic either way.
        let _ = reader.available;
    }

    #[test]
    fn key_encoding() {
        assert_eq!(KEY_PSTR, 0x5053_5452);
        assert_eq!(KEY_MSLD, 0x4D53_4C44);
        assert_eq!(KEY_CLSP, 0x434C_5350);
        assert_eq!(KEY_CLWK, 0x434C_574B);
        assert_eq!(KEY_B0TE, 0x4230_5445);
        assert_eq!(KEY_B0TF, 0x4230_5446);
        assert_eq!(KEY_PDTR, 0x5044_5452);

        // Verify 'P'=0x50, 'S'=0x53, 'T'=0x54, 'R'=0x52
        let pstr = u32::from_be_bytes([b'P', b'S', b'T', b'R']);
        assert_eq!(pstr, KEY_PSTR);

        // Thermal keys
        assert_eq!(KEY_TC0P, u32::from_be_bytes([b'T', b'C', b'0', b'P']));
        assert_eq!(KEY_TG0P, u32::from_be_bytes([b'T', b'G', b'0', b'P']));
        assert_eq!(KEY_TB0T, u32::from_be_bytes([b'T', b'B', b'0', b'T']));
        assert_eq!(KEY_TB1T, u32::from_be_bytes([b'T', b'B', b'1', b'T']));

        // Power keys
        assert_eq!(KEY_PCPC, u32::from_be_bytes([b'P', b'C', b'P', b'C']));
        assert_eq!(KEY_PCPG, u32::from_be_bytes([b'P', b'C', b'P', b'G']));
        assert_eq!(KEY_ID0R, u32::from_be_bytes([b'I', b'D', b'0', b'R']));
        assert_eq!(KEY_VC0C, u32::from_be_bytes([b'V', b'C', b'0', b'C']));
    }

    #[test]
    fn thermal_critical_thresholds() {
        let snap = SmcSnapshot {
            system_power_watts: None,
            lid_closed: false,
            last_sleep_us: 0,
            last_wake_us: 0,
            battery_time_to_empty_min: None,
            battery_time_to_full_min: None,
            charger_watts: None,
            cpu_temp_celsius: Some(95.0),
            gpu_temp_celsius: Some(80.0),
            battery_temp_celsius: Some(46.0),
            p_cluster_watts: None,
            gpu_watts: None,
            dc_in_current_amps: None,
            cpu_voltage: Some(0.60),
        };
        assert!(snap.cpu_thermal_critical());
        assert!(snap.battery_overheating());
        assert!(snap.voltage_throttled());
    }

    #[test]
    fn thermal_normal_thresholds() {
        let snap = SmcSnapshot {
            system_power_watts: Some(5.0),
            lid_closed: false,
            last_sleep_us: 0,
            last_wake_us: 0,
            battery_time_to_empty_min: None,
            battery_time_to_full_min: None,
            charger_watts: None,
            cpu_temp_celsius: Some(65.0),
            gpu_temp_celsius: Some(55.0),
            battery_temp_celsius: Some(30.0),
            p_cluster_watts: Some(3.0),
            gpu_watts: Some(1.0),
            dc_in_current_amps: None,
            cpu_voltage: Some(1.05),
        };
        assert!(!snap.cpu_thermal_critical());
        assert!(!snap.battery_overheating());
        assert!(!snap.voltage_throttled());
    }

    #[test]
    fn float_parse_be() {
        // 3.14 as big-endian f32: 0x4048F5C3
        let bytes: [u8; 4] = [0x40, 0x48, 0xF5, 0xC3];
        let val = f32::from_be_bytes(bytes) as f64;
        assert!((val - 3.14).abs() < 0.01);
    }

    #[test]
    fn u16_parse_be() {
        let bytes: [u8; 2] = [0x01, 0x2C]; // 300
        let val = u16::from_be_bytes(bytes);
        assert_eq!(val, 300);
    }

    #[test]
    fn snapshot_reasonable() {
        let reader = SmcDirectReader::new();
        if reader.available {
            if let Some(snap) = reader.read_snapshot() {
                // Power should be positive and reasonable.
                if let Some(w) = snap.system_power_watts {
                    assert!(w > 0.0 && w < 200.0, "unreasonable power: {w}");
                }
                // Battery times should be < 1500 minutes if present.
                if let Some(tte) = snap.battery_time_to_empty_min {
                    assert!(tte < 1500);
                }
                if let Some(ttf) = snap.battery_time_to_full_min {
                    assert!(ttf < 1500);
                }
            }
        }
    }
}
