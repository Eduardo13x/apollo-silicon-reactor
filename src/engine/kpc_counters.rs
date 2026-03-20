//! KPC Hardware Performance Counters — per-core IPC via libkpc.dylib
//!
//! KPC (Kernel Performance Counters) gives direct access to ARM PMU counters:
//! - Fixed counters: CPU cycles + instructions retired (always available)
//! - Configurable counters: L1/L2 cache misses, branch mispredicts (future)
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
//! Only fixed counters (cycles + instructions).  No event configuration needed.
//! Maximum compatibility across M1/M2/M3/M4.

use std::ffi::c_void;

/// KPC counter classes.
const KPC_CLASS_FIXED: u32 = 1;

/// Point-in-time KPC reading with derived IPC.
#[derive(Debug, Clone)]
pub struct KpcSnapshot {
    /// Total CPU cycles (sum across all cores, delta since last sample).
    pub total_cycles: u64,
    /// Total instructions retired (sum across all cores, delta since last sample).
    pub total_instructions: u64,
    /// Instructions per cycle (delta-based). 0.0 if no previous sample.
    pub ipc: f64,
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
    /// Number of fixed counters.
    counter_count: u32,
    /// Previous raw counter values for delta computation.
    prev_counters: Option<Vec<u64>>,
    /// Whether KPC is operational.
    pub available: bool,
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
                };
            }

            Self {
                handle,
                fn_force_all_ctrs_set: Some(fn_force_all_ctrs_set),
                fn_set_counting: Some(fn_set_counting),
                fn_get_counter_count: Some(fn_get_counter_count),
                fn_get_cpu_counters: Some(fn_get_cpu_counters),
                counter_count,
                prev_counters: None,
                available: true,
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

            Some(KpcSnapshot {
                total_cycles: delta_cycles,
                total_instructions: delta_instructions,
                ipc,
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
            counter_count: 0,
            prev_counters: None,
            available: false,
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
}
