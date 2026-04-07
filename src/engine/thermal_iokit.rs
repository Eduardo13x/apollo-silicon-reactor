//! Direct Thermal State via IOPMrootDomain — no subprocess
//!
//! Reads the system thermal warning level directly from IOKit's
//! `IOPMrootDomain` service, eliminating the need for `powermetrics`
//! subprocess for thermal state detection.
//!
//! # How it works
//!
//! 1. `IOServiceGetMatchingService("IOPMrootDomain")` → service handle
//! 2. `IORegistryEntryCreateCFProperty("Thermal Warning Level")` → CFNumber
//! 3. Parse: 0=Normal, 5=Moderate, 10=Severe, 15=Critical
//!
//! Takes ~20µs vs 500ms for powermetrics.
//!
//! # Additional properties available on IOPMrootDomain
//!
//! - `"IOPMSystemSleepType"` — sleep vs hibernate vs standby
//! - `"CurrentPowerSource"` — "AC Power" or "Battery Power"
//! - `"HasBattery"` — boolean

/// Thermal warning level from IOPMrootDomain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThermalWarning {
    /// No thermal pressure.
    None,
    /// Moderate: system is warm, may begin throttling.
    Moderate,
    /// Severe: significant thermal throttling in effect.
    Severe,
    /// Critical: emergency thermal shutdown imminent.
    Critical,
}

/// Power source detected from IOPMrootDomain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerSource {
    AC,
    Battery,
    Unknown,
}

/// Combined IOPMrootDomain reading.
#[derive(Debug, Clone)]
pub struct IoPmSnapshot {
    /// Kernel thermal warning level.
    pub thermal_warning: ThermalWarning,
    /// Current power source.
    pub power_source: PowerSource,
    /// Whether the machine has a battery.
    pub has_battery: bool,
}

/// Read thermal and power state from IOPMrootDomain.
/// Returns None if the service is unavailable.
#[cfg(target_os = "macos")]
pub fn read_iopm_state() -> Option<IoPmSnapshot> {
    use std::ffi::CStr;

    // IOKit FFI types
    #[allow(dead_code)]
    type IOReturn = i32;

    extern "C" {
        fn IOServiceGetMatchingService(mainPort: u32, matching: *const std::ffi::c_void) -> u32;
        fn IOServiceMatching(name: *const i8) -> *mut std::ffi::c_void;
        fn IORegistryEntryCreateCFProperty(
            entry: u32,
            key: *const std::ffi::c_void,
            allocator: *const std::ffi::c_void,
            options: u32,
        ) -> *const std::ffi::c_void;
        fn IOObjectRelease(object: u32) -> IOReturn;

        // CoreFoundation
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
    // kIOMainPortDefault = 0 on modern macOS
    const K_IO_MAIN_PORT_DEFAULT: u32 = 0;

    unsafe {
        let matching = IOServiceMatching(b"IOPMrootDomain\0".as_ptr() as *const i8);
        if matching.is_null() {
            return None;
        }

        let service = IOServiceGetMatchingService(K_IO_MAIN_PORT_DEFAULT, matching);
        // Note: IOServiceMatching result is consumed by IOServiceGetMatchingService
        if service == 0 {
            return None;
        }

        // Helper: read a property
        let read_property = |key: &[u8]| -> *const std::ffi::c_void {
            let cf_key = CFStringCreateWithCString(
                std::ptr::null(),
                key.as_ptr() as *const i8,
                K_CF_STRING_ENCODING_UTF8,
            );
            if cf_key.is_null() {
                return std::ptr::null();
            }
            let val = IORegistryEntryCreateCFProperty(service, cf_key, std::ptr::null(), 0);
            CFRelease(cf_key);
            val
        };

        // Read thermal warning level
        let thermal_warning = {
            let prop = read_property(b"Thermal Warning Level\0");
            if !prop.is_null() && CFGetTypeID(prop) == CFNumberGetTypeID() {
                let mut level: i32 = 0;
                CFNumberGetValue(
                    prop,
                    K_CF_NUMBER_SINT32_TYPE,
                    &mut level as *mut _ as *mut _,
                );
                CFRelease(prop);
                match level {
                    0..=4 => ThermalWarning::None,
                    5..=9 => ThermalWarning::Moderate,
                    10..=14 => ThermalWarning::Severe,
                    _ => ThermalWarning::Critical,
                }
            } else {
                if !prop.is_null() {
                    CFRelease(prop);
                }
                ThermalWarning::None
            }
        };

        // Read power source
        let power_source = {
            let prop = read_property(b"CurrentPowerSource\0");
            if !prop.is_null() && CFGetTypeID(prop) == CFStringGetTypeID() {
                let mut buf = [0i8; 64];
                let ok = CFStringGetCString(prop, buf.as_mut_ptr(), 64, K_CF_STRING_ENCODING_UTF8);
                CFRelease(prop);
                if ok {
                    let s = CStr::from_ptr(buf.as_ptr()).to_string_lossy();
                    if s.contains("AC") {
                        PowerSource::AC
                    } else if s.contains("Battery") {
                        PowerSource::Battery
                    } else {
                        PowerSource::Unknown
                    }
                } else {
                    PowerSource::Unknown
                }
            } else {
                if !prop.is_null() {
                    CFRelease(prop);
                }
                PowerSource::Unknown
            }
        };

        // Read has_battery
        let has_battery = {
            let prop = read_property(b"HasBattery\0");
            if !prop.is_null() && CFGetTypeID(prop) == CFBooleanGetTypeID() {
                let val = CFBooleanGetValue(prop);
                CFRelease(prop);
                val
            } else {
                if !prop.is_null() {
                    CFRelease(prop);
                }
                false
            }
        };

        IOObjectRelease(service);

        Some(IoPmSnapshot {
            thermal_warning,
            power_source,
            has_battery,
        })
    }
}

#[cfg(not(target_os = "macos"))]
pub fn read_iopm_state() -> Option<IoPmSnapshot> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_without_panic() {
        // Should not crash even on CI / non-macOS.
        let result = read_iopm_state();
        // On macOS: Some with valid data. On others: None.
        let _ = result;
    }

    #[test]
    fn thermal_ordering() {
        assert!(ThermalWarning::None < ThermalWarning::Moderate);
        assert!(ThermalWarning::Moderate < ThermalWarning::Severe);
        assert!(ThermalWarning::Severe < ThermalWarning::Critical);
    }

    #[test]
    fn snapshot_fields_accessible() {
        if let Some(snap) = read_iopm_state() {
            // Just verify fields are readable.
            let _ = snap.thermal_warning;
            let _ = snap.power_source;
            let _ = snap.has_battery;
        }
    }
}
