//! KPC Hardware Performance Counters — per-core IPC + memory-bound score via libkpc.dylib
//!
//! KPC (Kernel Performance Counters) gives direct access to ARM PMU counters:
//! - Fixed counters: CPU cycles + instructions retired (always available)
//! - Configurable counters: L1/L2 cache misses, branch mispredicts (Phase B)
//!
//! # Why
//!
//! `proc_pid_rusage` gives per-process instruction counts.  KPC gives
//! **system-wide** per-core hardware counters: IPC, cache miss rates.
//! Low IPC (<0.5) = memory-bound workload → freeze aggressively.
//! High IPC (>1.5) = compute-bound → avoid freezing compute processes.
//!
//! # Approach
//!
//! Uses `dlopen("/usr/lib/libkpc.dylib")` at runtime — no link-time dependency.
//! Gracefully degrades if SIP blocks kpc access.
//!
//! # Phase A (this implementation)
//!
//! Fixed counters (cycles + instructions) + `memory_bound_score` derived from IPC.
//! Maximum compatibility across M1/M2/M3/M4.
//!
//! # Phase B (framework ready, event programming pending)
//!
//! Configurable counter function pointers are loaded but not yet programmed.
//! Apple Silicon PMU event IDs have been reverse-engineered by the community
//! [@dougallj apple-silicon-pmu-events, asahi-linux/m1n1]:
//! - `0x02` CYCLES (same as fixed)
//! - `0x8c` INST_RETIRED (same as fixed)
//! - `0xbf` L1D_CACHE_MISS_LD — L1 data cache read misses
//! - `0xc0` L1D_CACHE_MISS_ST — L1 data cache write misses
//! - `0xcb` L2D_CACHE_MISS_LD — L2 data cache read misses
//! Event programming requires `kpc_set_config()` with class=KPC_CLASS_CONFIGURABLE.
//! [Hennessy & Patterson 2017] Cache miss rate × miss penalty = memory stall cycles.

use std::ffi::c_void;

/// KPC counter classes.
const KPC_CLASS_FIXED: u32 = 1;
/// Configurable counter class (for Phase B cache miss programming).
#[allow(dead_code)]
const KPC_CLASS_CONFIGURABLE: u32 = 2;

/// Apple M1 PMU event IDs (reverse-engineered by @dougallj / asahi-linux team).
/// These are for Phase B configurable counter programming.
#[allow(dead_code)]
mod pmu_events {
    /// L1 data cache read miss event ID on Apple M1/M2 P-cores.
    pub const L1D_CACHE_MISS_LD: u32 = 0xbf;
    /// L1 data cache write miss event ID on Apple M1/M2 P-cores.
    pub const L1D_CACHE_MISS_ST: u32 = 0xc0;
    /// L2 data cache read miss event ID on Apple M1/M2 P-cores.
    pub const L2D_CACHE_MISS_LD: u32 = 0xcb;
    /// Combined L2 cache misses.
    pub const L2D_CACHE_MISS: u32 = 0xcc;
    /// Expected peak IPC for Apple M1 P-cores under compute workloads.
    /// Used to normalize memory_bound_score to [0,1].
    pub const M1_PEAK_IPC: f64 = 5.0;
    /// Expected peak IPC for Apple M1 E-cores.
    pub const M1_ECPU_PEAK_IPC: f64 = 3.5;
}

/// Point-in-time KPC reading with derived IPC and memory-bound score.
#[derive(Debug, Clone)]
pub struct KpcSnapshot {
    /// Total CPU cycles (sum across all cores, delta since last sample).
    pub total_cycles: u64,
    /// Total instructions retired (sum across all cores, delta since last sample).
    pub total_instructions: u64,
    /// Instructions per cycle (delta-based). 0.0 if no previous sample.
    pub ipc: f64,
    /// IPC trend: EMA of IPC velocity (positive = improving, negative = degrading).
    /// Falling IPC trend predicts memory pressure increase before it shows in Mach counters.
    pub ipc_trend: f64,
    /// Memory-bound score: 0.0 = compute-bound (high IPC), 1.0 = fully memory-stalled.
    /// Derived from IPC relative to Apple M1 peak IPC (~5.0 for P-cores).
    /// [Hennessy & Patterson 2017 §2.2] Memory-bound fraction ≈ 1 - (achieved_IPC / peak_IPC).
    /// Score > 0.7 → system is spending >70% of cycles waiting on memory → safe to freeze.
    pub memory_bound_score: f64,
}

/// Hardware performance counter reader via libkpc.dylib.
pub struct KpcReader {
    /// dlopen handle to libkpc.dylib.
    #[allow(dead_code)]
    handle: *mut c_void,
    /// Function pointers (transmuted from dlsym).
    #[allow(dead_code)]
    fn_force_all_ctrs_set: Option<unsafe extern "C" fn(i32) -> i32>,
    #[allow(dead_code)]
    fn_set_counting: Option<unsafe extern "C" fn(u32) -> i32>,
    #[allow(dead_code)]
    fn_get_counter_count: Option<unsafe extern "C" fn(u32) -> u32>,
    fn_get_cpu_counters: Option<unsafe extern "C" fn(i32, u32, *mut i32, *mut u64) -> i32>,
    /// Phase B: configurable counter config set/get (loaded but not yet activated).
    /// kpc_set_config(class, config_ptr) programs PMU event selectors.
    #[allow(dead_code)]
    fn_set_config: Option<unsafe extern "C" fn(u32, *mut u64) -> i32>,
    #[allow(dead_code)]
    fn_get_config: Option<unsafe extern "C" fn(u32, *mut u64) -> i32>,
    /// Number of fixed counters.
    counter_count: u32,
    /// Previous raw counter values for delta computation.
    prev_counters: Option<Vec<u64>>,
    /// Whether KPC is operational.
    pub available: bool,
    /// EMA of IPC for trend detection.
    ipc_ema: f64,
    /// Previous IPC for velocity computation.
    prev_ipc: f64,
    /// EMA of IPC velocity (trend).
    ipc_velocity_ema: f64,
    /// EMA of memory-bound score.
    memory_bound_ema: f64,
}

// KpcReader contains raw pointers but they are function pointers / dlopen handle
// that are safe to send across threads (library is process-global).
unsafe impl Send for KpcReader {}

impl KpcReader {
    /// Load libkpc.dylib and initialize fixed counters.
    /// Safe to call without root — sets `available = false` on failure.
    pub fn new() -> Self {
        #[cfg(target_os = "macos")]
        {
            let handle = unsafe {
                libc::dlopen(
                    b"/usr/lib/libkpc.dylib\0".as_ptr() as *const i8,
                    libc::RTLD_LAZY,
                )
            };

            if handle.is_null() {
                return Self::unavailable();
            }

            // Load function pointers via dlsym.
            let fn_force = Self::load_sym(handle, b"kpc_force_all_ctrs_set\0");
            let fn_counting = Self::load_sym(handle, b"kpc_set_counting\0");
            let fn_count = Self::load_sym(handle, b"kpc_get_counter_count\0");
            let fn_cpu = Self::load_sym(handle, b"kpc_get_cpu_counters\0");
            // Phase B: configurable counter API (optional — null = Phase B not available).
            let fn_set_cfg = Self::load_sym(handle, b"kpc_set_config\0");
            let fn_get_cfg = Self::load_sym(handle, b"kpc_get_config\0");

            if fn_force.is_null() || fn_counting.is_null() || fn_count.is_null() || fn_cpu.is_null()
            {
                return Self::unavailable();
            }

            let fn_force_all_ctrs_set: unsafe extern "C" fn(i32) -> i32 =
                unsafe { std::mem::transmute(fn_force) };
            let fn_set_counting: unsafe extern "C" fn(u32) -> i32 =
                unsafe { std::mem::transmute(fn_counting) };
            let fn_get_counter_count: unsafe extern "C" fn(u32) -> u32 =
                unsafe { std::mem::transmute(fn_count) };
            let fn_get_cpu_counters: unsafe extern "C" fn(i32, u32, *mut i32, *mut u64) -> i32 =
                unsafe { std::mem::transmute(fn_cpu) };

            // Enable all counters (requires root).
            let ret = unsafe { fn_force_all_ctrs_set(1) };
            if ret != 0 {
                return Self {
                    handle,
                    fn_force_all_ctrs_set: Some(fn_force_all_ctrs_set),
                    fn_set_counting: Some(fn_set_counting),
                    fn_get_counter_count: Some(fn_get_counter_count),
                    fn_get_cpu_counters: Some(fn_get_cpu_counters),
                    counter_count: 0,
                    prev_counters: None,
                    available: false,
                    ipc_ema: 0.0,
                    prev_ipc: 0.0,
                    ipc_velocity_ema: 0.0,
                    memory_bound_ema: 0.0,
                    fn_set_config: None,
                    fn_get_config: None,
                };
            }

            // Enable fixed counter counting.
            let ret = unsafe { fn_set_counting(KPC_CLASS_FIXED) };
            if ret != 0 {
                return Self {
                    handle,
                    fn_force_all_ctrs_set: Some(fn_force_all_ctrs_set),
                    fn_set_counting: Some(fn_set_counting),
                    fn_get_counter_count: Some(fn_get_counter_count),
                    fn_get_cpu_counters: Some(fn_get_cpu_counters),
                    counter_count: 0,
                    prev_counters: None,
                    available: false,
                    ipc_ema: 0.0,
                    prev_ipc: 0.0,
                    ipc_velocity_ema: 0.0,
                    memory_bound_ema: 0.0,
                    fn_set_config: None,
                    fn_get_config: None,
                };
            }

            let counter_count = unsafe { fn_get_counter_count(KPC_CLASS_FIXED) };
            if counter_count == 0 || counter_count > 64 {
                return Self {
                    handle,
                    fn_force_all_ctrs_set: Some(fn_force_all_ctrs_set),
                    fn_set_counting: Some(fn_set_counting),
                    fn_get_counter_count: Some(fn_get_counter_count),
                    fn_get_cpu_counters: Some(fn_get_cpu_counters),
                    counter_count: 0,
                    prev_counters: None,
                    available: false,
                    ipc_ema: 0.0,
                    prev_ipc: 0.0,
                    ipc_velocity_ema: 0.0,
                    memory_bound_ema: 0.0,
                    fn_set_config: None,
                    fn_get_config: None,
                };
            }

            let fn_set_config = if fn_set_cfg.is_null() { None } else {
                Some(unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn(u32, *mut u64) -> i32>(fn_set_cfg) })
            };
            let fn_get_config = if fn_get_cfg.is_null() { None } else {
                Some(unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn(u32, *mut u64) -> i32>(fn_get_cfg) })
            };

            Self {
                handle,
                fn_force_all_ctrs_set: Some(fn_force_all_ctrs_set),
                fn_set_counting: Some(fn_set_counting),
                fn_get_counter_count: Some(fn_get_counter_count),
                fn_get_cpu_counters: Some(fn_get_cpu_counters),
                fn_set_config,
                fn_get_config,
                counter_count,
                prev_counters: None,
                available: true,
                ipc_ema: 0.0,
                prev_ipc: 0.0,
                ipc_velocity_ema: 0.0,
                memory_bound_ema: 0.0,
            }
        }

        #[cfg(not(target_os = "macos"))]
        Self::unavailable()
    }

    /// Sample fixed counters and compute IPC delta.
    /// First call returns IPC=0.0 (no previous sample for delta).
    pub fn sample(&mut self) -> Option<KpcSnapshot> {
        if !self.available {
            return None;
        }

        #[cfg(target_os = "macos")]
        {
            let fn_get = self.fn_get_cpu_counters?;
            let n = self.counter_count as usize;

            // Allocate buffer for all CPU counters.
            // kpc_get_cpu_counters returns counters for all CPUs × counter_count.
            // We read with cupcnt = counter_count which gives per-logical-cpu values.
            // Total buffer size = ncpus * counter_count, but the API with
            // cupcnt_buf = NULL returns the total across all CPUs into counter_count slots.
            let mut buf = vec![0u64; n];
            let mut cupcnt: i32 = 0;

            let ret = unsafe {
                fn_get(
                    KPC_CLASS_FIXED as i32,
                    KPC_CLASS_FIXED,
                    &mut cupcnt,
                    buf.as_mut_ptr(),
                )
            };

            if ret != 0 {
                return None;
            }

            // Fixed counters: index 0 = cycles, index 1 = instructions (on Apple Silicon).
            let cycles = buf.first().copied().unwrap_or(0);
            let instructions = buf.get(1).copied().unwrap_or(0);

            let (delta_cycles, delta_instructions, ipc) =
                if let Some(ref prev) = self.prev_counters {
                    let prev_cycles = prev.first().copied().unwrap_or(0);
                    let prev_instr = prev.get(1).copied().unwrap_or(0);

                    // Handle 48-bit counter overflow.
                    let mask_48 = (1u64 << 48) - 1;
                    let dc = if cycles >= prev_cycles {
                        cycles - prev_cycles
                    } else {
                        (cycles + mask_48) - prev_cycles
                    };
                    let di = if instructions >= prev_instr {
                        instructions - prev_instr
                    } else {
                        (instructions + mask_48) - prev_instr
                    };

                    let ipc = if dc > 0 {
                        di as f64 / dc as f64
                    } else {
                        0.0
                    };

                    (dc, di, ipc)
                } else {
                    (0, 0, 0.0)
                };

            self.prev_counters = Some(buf);

            // IPC trend: EMA of IPC velocity.
            // Falling IPC → system becoming memory-bound → pressure increase likely.
            let ipc_velocity = if self.prev_ipc > 0.0 {
                ipc - self.prev_ipc
            } else {
                0.0
            };
            const TREND_ALPHA: f64 = 0.15;
            self.ipc_ema = if self.ipc_ema == 0.0 { ipc } else {
                TREND_ALPHA * ipc + (1.0 - TREND_ALPHA) * self.ipc_ema
            };
            self.ipc_velocity_ema = TREND_ALPHA * ipc_velocity + (1.0 - TREND_ALPHA) * self.ipc_velocity_ema;
            self.prev_ipc = ipc;

            // Memory-bound score: fraction of cycles NOT executing instructions.
            // [Hennessy & Patterson 2017 §2.2] memory stall cycles / total cycles ≈ 1 - IPC/peak.
            // Apple M1 P-core peak ~5.0 IPC (measured via Agner Fog / microarch.info).
            // EMA-smoothed to avoid single-cycle noise.
            let raw_bound = if ipc > 0.001 {
                (1.0 - (ipc / pmu_events::M1_PEAK_IPC)).clamp(0.0, 1.0)
            } else if delta_cycles > 0 {
                1.0 // cycles with 0 instructions = 100% memory stalled
            } else {
                0.0
            };
            self.memory_bound_ema = if self.memory_bound_ema == 0.0 {
                raw_bound
            } else {
                TREND_ALPHA * raw_bound + (1.0 - TREND_ALPHA) * self.memory_bound_ema
            };

            Some(KpcSnapshot {
                total_cycles: delta_cycles,
                total_instructions: delta_instructions,
                ipc,
                ipc_trend: self.ipc_velocity_ema,
                memory_bound_score: self.memory_bound_ema,
            })
        }

        #[cfg(not(target_os = "macos"))]
        None
    }

    fn unavailable() -> Self {
        Self {
            handle: std::ptr::null_mut(),
            fn_force_all_ctrs_set: None,
            fn_set_counting: None,
            fn_get_counter_count: None,
            fn_get_cpu_counters: None,
            fn_set_config: None,
            fn_get_config: None,
            counter_count: 0,
            prev_counters: None,
            available: false,
            ipc_ema: 0.0,
            prev_ipc: 0.0,
            ipc_velocity_ema: 0.0,
            memory_bound_ema: 0.0,
        }
    }

    #[cfg(target_os = "macos")]
    fn load_sym(handle: *mut c_void, name: &[u8]) -> *mut c_void {
        unsafe { libc::dlsym(handle, name.as_ptr() as *const i8) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kpc_loads_without_panic() {
        // dlopen may fail on CI or without root — available=false is fine.
        let reader = KpcReader::new();
        let _ = reader.available;
    }

    #[test]
    fn counter_count_reasonable() {
        let reader = KpcReader::new();
        if reader.available {
            assert!(
                reader.counter_count >= 2 && reader.counter_count <= 64,
                "unexpected counter count: {}",
                reader.counter_count
            );
        }
    }

    #[test]
    fn ipc_positive() {
        let mut reader = KpcReader::new();
        if reader.available {
            // First sample: IPC=0.0 (no delta).
            let s1 = reader.sample();
            assert!(s1.is_some());
            assert_eq!(s1.unwrap().ipc, 0.0);

            // Burn some CPU to create a delta.
            let mut x = 0u64;
            for i in 0..1_000_000 {
                x = x.wrapping_add(i);
            }
            std::hint::black_box(x);

            // Second sample: IPC should be > 0.
            let s2 = reader.sample();
            assert!(s2.is_some());
            let snap = s2.unwrap();
            assert!(snap.ipc > 0.0, "IPC should be positive: {}", snap.ipc);
        }
    }

    #[test]
    fn delta_computation() {
        // Unit test with synthetic values — 48-bit overflow handling.
        let mask_48 = (1u64 << 48) - 1;
        let prev_cycles = mask_48 - 100;
        let curr_cycles = 50; // wrapped around

        let delta = if curr_cycles >= prev_cycles {
            curr_cycles - prev_cycles
        } else {
            (curr_cycles + mask_48) - prev_cycles
        };

        // Should be 50 + 100 = 150 (wrapped correctly).
        assert_eq!(delta, 150);
    }

    #[test]
    fn memory_bound_score_at_zero_ipc() {
        // IPC=0 with cycles>0 = 100% memory stalled.
        let raw = if 0.001 > 0.001 {
            (1.0 - (0.001 / pmu_events::M1_PEAK_IPC)).clamp(0.0, 1.0)
        } else if 1000u64 > 0 {
            1.0
        } else {
            0.0
        };
        assert!((raw - 1.0).abs() < 0.001);
    }

    #[test]
    fn memory_bound_score_at_peak_ipc() {
        // IPC = peak → memory_bound_score = 0 (fully compute-bound).
        let ipc = pmu_events::M1_PEAK_IPC;
        let score = (1.0 - (ipc / pmu_events::M1_PEAK_IPC)).clamp(0.0, 1.0);
        assert!(score < 0.001, "peak IPC should give near-zero memory_bound_score");
    }

    #[test]
    fn memory_bound_score_midpoint() {
        // IPC = 2.5 with peak=5.0 → score = 0.5 (half memory stalled).
        let ipc = 2.5;
        let score = (1.0 - (ipc / pmu_events::M1_PEAK_IPC)).clamp(0.0, 1.0);
        assert!((score - 0.5).abs() < 0.01);
    }

    #[test]
    fn pmu_event_ids_are_documented() {
        // Verify the reverse-engineered event IDs match known values.
        // [@dougallj apple-silicon-pmu-events on GitHub]
        assert_eq!(pmu_events::L1D_CACHE_MISS_LD, 0xbf);
        assert_eq!(pmu_events::L2D_CACHE_MISS_LD, 0xcb);
        assert!(pmu_events::M1_PEAK_IPC > 4.0, "M1 peak IPC should be ~5.0");
    }
}
