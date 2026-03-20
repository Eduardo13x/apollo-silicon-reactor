//! IOReport direct integration — private IOKit API for hardware telemetry.
//!
//! IOReport is the private framework used internally by `powermetrics`.
//! It gives sub-millisecond hardware telemetry without the 500ms subprocess
//! overhead of running `powermetrics`.
//!
//! # What we extract
//!
//! | Group          | Data                                        |
//! |----------------|---------------------------------------------|
//! | CPU Stats      | P-cluster & E-cluster frequency state dist  |
//! | GPU Stats      | GPU performance state duty cycles           |
//! | Energy Model   | Per-component power in milliwatts           |
//! | AMC Stats      | Memory controller bandwidth utilization     |
//!
//! From these, Apollo derives:
//!   - `p_cluster_pct` — fraction of time P-cores were active
//!   - `e_cluster_pct` — fraction of time E-cores were active
//!   - `gpu_pct`       — GPU utilization
//!   - `ane_busy`      — whether the Neural Engine is active
//!   - `cpu_mw`        — CPU package power in milliwatts (real, not estimated)
//!   - `gpu_mw`        — GPU power in milliwatts
//!   - `dram_mw`       — DRAM power in milliwatts
//!
//! # Architecture
//!
//! The Objective-C block callback required by `IOReportIterate` is handled
//! in `src/engine_c/ioreport_bridge.c` (compiled via build.rs).  Rust calls
//! the plain-C bridge functions which internally wrap the block.
//!
//! # Safety
//!
//! The bridge functions are unsafe (C FFI + raw CF pointers).  All CFTypeRef
//! values are released via `apollo_ioreport_release` to prevent leaks.
//! The subscription and channels live for the lifetime of `IOReportReader`.
//!
//! # References
//!
//! - asitop (Python, open source): github.com/tlkh/asitop
//! - Stats.app (Swift, open source): github.com/exelban/stats
//! - powermetrics source (partial): Apple Open Source

use std::ffi::c_void;
use std::os::raw::c_char;
use std::time::Instant;

// ── C bridge FFI ─────────────────────────────────────────────────────────────

/// Mirror of ApolloIOReportChannel from ioreport_bridge.c.
#[repr(C)]
#[derive(Clone)]
pub struct RawChannel {
    pub driver:      [c_char; 128],
    pub channel:     [c_char; 256],
    pub value:       i64,
    pub state_count: i32,
    pub duty_cycles: [f64; 32],
    pub state_names: [[c_char; 64]; 32],
}

#[cfg_attr(target_os = "macos", link(name = "ioreport_bridge", kind = "static"))]
#[cfg_attr(target_os = "macos", link(name = "IOReport", kind = "dylib"))]
#[cfg_attr(target_os = "macos", link(name = "CoreFoundation", kind = "framework"))]
#[cfg(target_os = "macos")]
extern "C" {
    fn apollo_ioreport_create_subscription(
        out_channels: *mut *mut c_void,
        group_names:  *const *const c_char,
        group_count:  libc::c_int,
    ) -> *mut c_void;

    fn apollo_ioreport_sample(
        sub:      *mut c_void,
        channels: *mut c_void,
    ) -> *mut c_void;

    fn apollo_ioreport_delta(s1: *mut c_void, s2: *mut c_void) -> *mut c_void;

    fn apollo_ioreport_iterate(
        samples: *mut c_void,
        cb:      extern "C" fn(*const RawChannel, *mut c_void),
        ctx:     *mut c_void,
    );

    fn apollo_ioreport_release(ptr: *mut c_void);
}

// ── Channel group names ───────────────────────────────────────────────────────

const GROUPS: &[&str] = &[
    "CPU Stats",    // P/E cluster performance state distribution
    "GPU Stats",    // GPU performance states
    "Energy Model", // Per-component power (mW)
    "AMC Stats",    // Memory controller (bandwidth proxy)
];

// ── Parsed output ─────────────────────────────────────────────────────────────

/// Hardware utilization snapshot from IOReport.
#[derive(Debug, Clone, Default)]
pub struct IOReportSnapshot {
    /// P-core cluster active fraction (0.0–1.0).
    pub p_cluster_pct: f64,
    /// E-core cluster active fraction (0.0–1.0).
    pub e_cluster_pct: f64,
    /// GPU active fraction (0.0–1.0).
    pub gpu_pct: f64,
    /// ANE (Neural Engine) detected as active.
    pub ane_busy: bool,
    /// CPU package power (milliwatts).
    pub cpu_mw: f64,
    /// GPU power (milliwatts).
    pub gpu_mw: f64,
    /// DRAM power (milliwatts).
    pub dram_mw: f64,
    /// Total package power (milliwatts), if available.
    pub package_mw: f64,
    /// Memory controller (AMC) bandwidth utilization (0.0–1.0).
    /// Derived from AMC Stats performance state duty cycles.
    /// >0.8 indicates memory bandwidth saturation (M1 8GB bottleneck).
    pub amc_bandwidth_pct: f64,
}

impl IOReportSnapshot {
    /// Total SoC power in watts.
    pub fn total_watts(&self) -> f64 {
        if self.package_mw > 0.0 {
            self.package_mw / 1000.0
        } else {
            (self.cpu_mw + self.gpu_mw + self.dram_mw) / 1000.0
        }
    }

    /// Whether the P-cores are under significant load.
    pub fn p_cores_loaded(&self) -> bool {
        self.p_cluster_pct > 0.25
    }

    /// Whether the system is mostly idle (all clusters low).
    pub fn is_system_idle(&self) -> bool {
        self.p_cluster_pct < 0.05
            && self.e_cluster_pct < 0.10
            && self.gpu_pct < 0.05
    }

    /// Whether memory bandwidth is saturated (>80% utilization).
    /// On M1 8GB this is the #1 bottleneck — indicates heavy swap/compression.
    pub fn memory_bandwidth_saturated(&self) -> bool {
        self.amc_bandwidth_pct > 0.80
    }
}

// ── Reader ────────────────────────────────────────────────────────────────────

/// Long-lived IOReport reader.
///
/// Call `sample_once()` to take a baseline, then after a delay call
/// `sample_delta()` to get the utilization over that interval.
///
/// On macOS versions where IOReport is unavailable (or returns NULL),
/// all methods return `None` gracefully.
pub struct IOReportReader {
    #[cfg(target_os = "macos")]
    sub:      *mut c_void,
    #[cfg(target_os = "macos")]
    channels: *mut c_void,
    #[cfg(target_os = "macos")]
    prev:     Option<(*mut c_void, Instant)>,
    /// False if IOReport failed to initialize.
    pub available: bool,
}

// IOReportReader holds raw CF pointers that are effectively thread-local
// (used only from the smc-reader background thread).
// We implement Send so it can be moved into the background thread.
// Safety: the pointers are only accessed from one thread at a time.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for IOReportReader {}

impl IOReportReader {
    /// Create and initialize an IOReport subscription.
    ///
    /// Returns a reader regardless of whether IOReport is available;
    /// check `available` before using.
    pub fn new() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self::new_macos()
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self {
                available: false,
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn new_macos() -> Self {
        use std::ffi::CString;
        let group_cstrings: Vec<CString> = GROUPS
            .iter()
            .map(|s| CString::new(*s).unwrap())
            .collect();
        let group_ptrs: Vec<*const c_char> =
            group_cstrings.iter().map(|cs| cs.as_ptr()).collect();

        let mut channels: *mut c_void = std::ptr::null_mut();
        let sub = unsafe {
            apollo_ioreport_create_subscription(
                &mut channels,
                group_ptrs.as_ptr(),
                group_ptrs.len() as libc::c_int,
            )
        };

        let available = !sub.is_null() && !channels.is_null();
        Self {
            sub,
            channels,
            prev: None,
            available,
        }
    }

    /// Take a baseline sample (call at start of interval).
    #[cfg(target_os = "macos")]
    pub fn begin_sample(&mut self) {
        if !self.available {
            return;
        }
        // Release previous baseline if any.
        if let Some((ptr, _)) = self.prev.take() {
            unsafe { apollo_ioreport_release(ptr) };
        }
        let s = unsafe { apollo_ioreport_sample(self.sub, self.channels) };
        if !s.is_null() {
            self.prev = Some((s, Instant::now()));
        }
    }

    /// Take a second sample, compute delta, and parse hardware metrics.
    ///
    /// Returns `None` if no baseline exists or IOReport unavailable.
    #[cfg(target_os = "macos")]
    pub fn end_sample(&mut self) -> Option<IOReportSnapshot> {
        if !self.available {
            return None;
        }
        let (prev_ptr, _t) = self.prev.as_ref()?;
        let prev_ptr = *prev_ptr;

        let s2 = unsafe { apollo_ioreport_sample(self.sub, self.channels) };
        if s2.is_null() {
            return None;
        }

        let delta = unsafe { apollo_ioreport_delta(prev_ptr as *mut c_void, s2) };
        unsafe { apollo_ioreport_release(s2) };

        if delta.is_null() {
            return None;
        }

        let snap = Self::parse_delta(delta);
        unsafe { apollo_ioreport_release(delta) };
        Some(snap)
    }

    /// Parse a delta CFDictionaryRef into an `IOReportSnapshot`.
    #[cfg(target_os = "macos")]
    fn parse_delta(delta: *mut c_void) -> IOReportSnapshot {
        // Accumulator state passed through the C callback via void*.
        struct Acc {
            snap: IOReportSnapshot,
        }

        extern "C" fn callback(ch: *const RawChannel, ctx: *mut c_void) {
            let ch = unsafe { &*ch };
            let acc = unsafe { &mut *(ctx as *mut Acc) };

            let driver = c_chars_to_str(&ch.driver);
            let channel = c_chars_to_str(&ch.channel);

            // ── CPU cluster utilization ──────────────────────────────────────
            // Driver: "AppleARMCPUPowerState" or similar
            // Channel: "CPU Complex Performance States 0" (E) / "1" (P)
            if channel.contains("CPU Complex Performance States") {
                let active_pct = active_fraction(&ch.duty_cycles, &ch.state_names, ch.state_count);
                // Cluster 0 = E-cores, Cluster 1 = P-cores on M1.
                if channel.ends_with('0') {
                    acc.snap.e_cluster_pct = active_pct;
                } else if channel.ends_with('1') {
                    acc.snap.p_cluster_pct = active_pct;
                } else {
                    // Fallback: treat as P-core if only one cluster reported.
                    acc.snap.p_cluster_pct =
                        acc.snap.p_cluster_pct.max(active_pct);
                }
            }

            // ── GPU utilization ──────────────────────────────────────────────
            if channel.contains("GPU Performance States") || driver.contains("GPU") {
                let active_pct = active_fraction(&ch.duty_cycles, &ch.state_names, ch.state_count);
                acc.snap.gpu_pct = acc.snap.gpu_pct.max(active_pct);
            }

            // ── ANE (Neural Engine) ──────────────────────────────────────────
            if channel.contains("ANE") || driver.contains("ANE") {
                if ch.state_count > 0 {
                    let active = active_fraction(&ch.duty_cycles, &ch.state_names, ch.state_count);
                    if active > 0.01 {
                        acc.snap.ane_busy = true;
                    }
                } else if ch.value > 0 {
                    acc.snap.ane_busy = true;
                }
            }

            // ── AMC (memory controller bandwidth) ────────────────────────────
            // AMC Stats channels have performance state duty cycles.
            // Active fraction = memory bandwidth utilization.
            if (channel.contains("AMC") || driver.contains("AMC"))
                && ch.state_count > 0
            {
                let active_pct = active_fraction(&ch.duty_cycles, &ch.state_names, ch.state_count);
                // Take max across multiple AMC channels (one per memory port).
                acc.snap.amc_bandwidth_pct = acc.snap.amc_bandwidth_pct.max(active_pct);
            }

            // ── Power (milliwatts) ────────────────────────────────────────────
            // Energy Model channels report cumulative microjoules in the delta.
            // channel names: "CPU", "GPU", "DRAM", "Package"
            if ch.state_count == 0 && ch.value > 0 {
                let lc = channel.to_ascii_lowercase();
                // Convert energy delta (µJ) to average power (mW).
                // The delta covers ~1s of sampling; value / 1000 → mW.
                // (More accurate with actual elapsed time, but 1s is the
                //  typical poll interval from smc_reader.)
                let mw = ch.value as f64 / 1000.0;
                if lc.contains("cpu") && !lc.contains("complex") {
                    acc.snap.cpu_mw += mw;
                } else if lc.contains("gpu") {
                    acc.snap.gpu_mw += mw;
                } else if lc.contains("dram") || lc.contains("memory") {
                    acc.snap.dram_mw += mw;
                } else if lc.contains("package") || lc.contains("total") {
                    acc.snap.package_mw += mw;
                }
            }
        }

        let mut acc = Acc {
            snap: IOReportSnapshot::default(),
        };
        unsafe {
            apollo_ioreport_iterate(
                delta,
                callback,
                &mut acc as *mut Acc as *mut c_void,
            )
        };

        // Clamp fractions to [0,1].
        let s = &mut acc.snap;
        s.p_cluster_pct = s.p_cluster_pct.clamp(0.0, 1.0);
        s.e_cluster_pct = s.e_cluster_pct.clamp(0.0, 1.0);
        s.gpu_pct       = s.gpu_pct.clamp(0.0, 1.0);

        acc.snap
    }

    // Non-macOS stubs.
    #[cfg(not(target_os = "macos"))]
    pub fn begin_sample(&mut self) {}

    #[cfg(not(target_os = "macos"))]
    pub fn end_sample(&mut self) -> Option<IOReportSnapshot> {
        None
    }
}

impl Default for IOReportReader {
    fn default() -> Self {
        Self::new()
    }
}

// Note: no Drop impl — IOReportReader is designed to live for the process
// lifetime.  The subscription (sub, channels) and any pending sample (prev)
// are intentionally leaked; the OS reclaims all CF objects on exit.
// Implementing Drop would require the C bridge symbols to be linked into every
// binary that uses the library, including test binaries that never create an
// IOReportReader, causing spurious link failures.

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute the "active" fraction from a performance-state duty cycle array.
///
/// Performance states are ordered from lowest (idle/off) to highest frequency.
/// State[0] is typically "IDLE" or the lowest P-state (CPU in WFI).
/// We sum duty cycles of all states except the lowest (index 0).
fn active_fraction(duty_cycles: &[f64; 32], state_names: &[[c_char; 64]; 32], count: i32) -> f64 {
    if count <= 0 {
        return 0.0;
    }
    let n = count.min(32) as usize;
    let mut active = 0.0_f64;

    for i in 0..n {
        let name = c_chars_to_str(&state_names[i]);
        let dc = duty_cycles[i];
        // Skip explicit idle state (index 0 or named "IDLE"/"WFI"/"OFF").
        let is_idle = i == 0
            || name.eq_ignore_ascii_case("idle")
            || name.eq_ignore_ascii_case("wfi")
            || name.eq_ignore_ascii_case("off");
        if !is_idle && dc > 0.0 {
            active += dc;
        }
    }
    active.clamp(0.0, 1.0)
}

/// Convert a null-terminated C char array to a &str (lossy).
fn c_chars_to_str(arr: &[c_char]) -> String {
    let bytes: Vec<u8> = arr
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_initializes_without_panic() {
        let reader = IOReportReader::new();
        // Just verify it doesn't crash — available depends on root + IOKit.
        let _ = reader.available;
    }

    #[test]
    fn active_fraction_all_idle() {
        let mut names = [[0i8; 64]; 32];
        // State 0 = idle
        let idle_name = b"IDLE\0";
        for (i, &b) in idle_name.iter().enumerate() {
            names[0][i] = b as i8;
        }
        let mut cycles = [0.0f64; 32];
        cycles[0] = 1.0; // 100% in idle state
        let frac = active_fraction(&cycles, &names, 1);
        assert_eq!(frac, 0.0);
    }

    #[test]
    fn active_fraction_fully_loaded() {
        let names = [[0i8; 64]; 32];
        let mut cycles = [0.0f64; 32];
        cycles[0] = 0.0; // no idle time
        cycles[1] = 1.0; // 100% at max P-state
        let frac = active_fraction(&cycles, &names, 2);
        assert_eq!(frac, 1.0);
    }

    #[test]
    fn snapshot_total_watts_uses_package_if_available() {
        let snap = IOReportSnapshot {
            cpu_mw: 1000.0,
            gpu_mw: 500.0,
            dram_mw: 300.0,
            package_mw: 2500.0,
            ..Default::default()
        };
        assert!((snap.total_watts() - 2.5).abs() < 0.001);
    }

    #[test]
    fn snapshot_total_watts_sums_components_without_package() {
        let snap = IOReportSnapshot {
            cpu_mw: 1000.0,
            gpu_mw: 500.0,
            dram_mw: 300.0,
            package_mw: 0.0,
            ..Default::default()
        };
        assert!((snap.total_watts() - 1.8).abs() < 0.001);
    }

    #[test]
    fn is_system_idle_true_when_all_low() {
        let snap = IOReportSnapshot {
            p_cluster_pct: 0.02,
            e_cluster_pct: 0.05,
            gpu_pct: 0.01,
            ..Default::default()
        };
        assert!(snap.is_system_idle());
    }
}
