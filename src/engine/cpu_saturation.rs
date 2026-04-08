//! Per-core CPU utilisation via Mach `host_processor_info`.
//!
//! Apollo previously had no per-core load sensor — only per-process CPU%
//! (from sysinfo) and the aggregate run-queue signals in `RusageInfo`.
//! This module fills that gap with a direct Mach reading of the cumulative
//! per-core tick counters, deriving a busy ratio ∈ [0, 1] for every core
//! between two successive samples.
//!
//! Per-core is strictly more useful than load-average on Apple Silicon
//! because the P-cores and E-cores are scheduled asymmetrically: a load
//! of "4" on an M1 can mean "all 4 P-cores pegged, E-cores idle" (perf
//! workload) or "all 4 E-cores pegged, P-cores idle" (background work),
//! and apollo's freeze/boost decisions should react very differently to
//! the two cases.
//!
//! Costs: `host_processor_info` is ~5 µs on M1. Sampling every 2 s
//! (daemon cycle) adds negligible overhead to the hot path.
//!
//! ## References
//!
//! - Apple XNU `osfmk/mach/processor_info.h` —
//!   `PROCESSOR_CPU_LOAD_INFO` returns the cumulative tick counters
//!   (user/system/idle/nice) for every processor.
//! - Apple `Libc/gen/FreeBSD/getloadavg.c` — the `uptime` load average
//!   is derived from the same counters but smoothed; per-core samples
//!   give the raw instantaneous view apollo needs for Asymmetric
//!   Multiprocessing decisions.
//! - [Johnson 2017] "Mac OS X and iOS Internals" Vol I §5 — AMP
//!   scheduler treats P/E clusters as separate run queues; a single
//!   load average is insufficient for per-cluster pressure decisions.

/// Per-core cumulative tick counters, as returned by the kernel.
///
/// All four fields are monotonically increasing since boot. Consumers
/// compute busy ratios by taking deltas between successive samples —
/// see `CpuSaturation::compute`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PerCoreTicks {
    pub user: u64,
    pub system: u64,
    pub idle: u64,
    pub nice: u64,
}

impl PerCoreTicks {
    /// Total ticks on this core since boot.
    #[inline]
    pub fn total(&self) -> u64 {
        self.user
            .saturating_add(self.system)
            .saturating_add(self.idle)
            .saturating_add(self.nice)
    }

    /// Non-idle ticks (user + system + nice) on this core since boot.
    #[inline]
    pub fn busy(&self) -> u64 {
        self.user
            .saturating_add(self.system)
            .saturating_add(self.nice)
    }
}

/// Derived per-core saturation snapshot between two `PerCoreTicks` samples.
#[derive(Debug, Clone, Default)]
pub struct CpuSaturation {
    /// Per-core busy ratio ∈ [0, 1] for each online processor.
    pub per_core_busy: Vec<f64>,
    /// Mean busy ratio across all cores — rough aggregate saturation.
    pub mean_busy: f64,
    /// Maximum per-core busy ratio — "hottest core" signal.
    pub max_busy: f64,
    /// Fraction of cores with busy ratio ≥ 0.80 — pegged-core count / total.
    pub pegged_fraction: f64,
}

impl CpuSaturation {
    /// Compute saturation between two per-core tick samples.
    ///
    /// Lengths must match (same number of online cores). Returns
    /// `Self::default()` if either sample is empty or lengths differ —
    /// this lets callers treat the first cycle as "no signal yet"
    /// without special-casing.
    pub fn compute(prev: &[PerCoreTicks], curr: &[PerCoreTicks]) -> Self {
        if prev.is_empty() || prev.len() != curr.len() {
            return Self::default();
        }
        let n = prev.len();
        let mut per_core = Vec::with_capacity(n);
        let mut sum = 0.0_f64;
        let mut max = 0.0_f64;
        let mut pegged = 0_usize;
        for (p, c) in prev.iter().zip(curr.iter()) {
            let busy_delta = c.busy().saturating_sub(p.busy()) as f64;
            let total_delta = c.total().saturating_sub(p.total()) as f64;
            let ratio = if total_delta > 0.0 {
                (busy_delta / total_delta).clamp(0.0, 1.0)
            } else {
                0.0
            };
            per_core.push(ratio);
            sum += ratio;
            if ratio > max {
                max = ratio;
            }
            if ratio >= 0.80 {
                pegged += 1;
            }
        }
        Self {
            per_core_busy: per_core,
            mean_busy: sum / n as f64,
            max_busy: max,
            pegged_fraction: pegged as f64 / n as f64,
        }
    }
}

// ── Mach FFI ────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod ffi {
    use std::ffi::c_void;

    pub type MachPortT = u32;
    pub type KernReturnT = i32;
    pub type NaturalT = u32;
    pub type IntegerT = i32;
    pub type ProcessorFlavorT = i32;
    pub type ProcessorInfoArrayT = *mut IntegerT;

    pub const PROCESSOR_CPU_LOAD_INFO: ProcessorFlavorT = 2;
    pub const KERN_SUCCESS: i32 = 0;
    /// Number of `integer_t` words in `processor_cpu_load_info` — 4
    /// cumulative tick counters: user, system, idle, nice.
    pub const PROCESSOR_CPU_LOAD_INFO_COUNT: usize = 4;

    #[link(name = "System", kind = "dylib")]
    extern "C" {
        pub fn mach_host_self() -> MachPortT;
        pub fn host_processor_info(
            host: MachPortT,
            flavor: ProcessorFlavorT,
            out_processor_count: *mut NaturalT,
            out_processor_info: *mut ProcessorInfoArrayT,
            out_processor_infoCnt: *mut NaturalT,
        ) -> KernReturnT;
        pub fn vm_deallocate(
            target_task: MachPortT,
            address: *mut c_void,
            size: usize,
        ) -> KernReturnT;
        pub fn mach_task_self() -> MachPortT;
    }
}

/// Read per-core cumulative CPU tick counters from the kernel.
///
/// Returns one `PerCoreTicks` entry per online processor. On failure
/// returns an empty vec; consumers already handle that as "no signal
/// yet" via `CpuSaturation::compute`.
#[cfg(target_os = "macos")]
pub fn read_per_core_ticks() -> Vec<PerCoreTicks> {
    use ffi::*;
    let mut processor_count: NaturalT = 0;
    let mut info_array: ProcessorInfoArrayT = std::ptr::null_mut();
    let mut info_count: NaturalT = 0;
    // SAFETY: host_processor_info is a pure kernel query. We own the
    // returned vm_allocation and free it with vm_deallocate before
    // returning, ensuring no leak on any control-flow path.
    let kr = unsafe {
        host_processor_info(
            mach_host_self(),
            PROCESSOR_CPU_LOAD_INFO,
            &mut processor_count,
            &mut info_array,
            &mut info_count,
        )
    };
    if kr != KERN_SUCCESS || info_array.is_null() || processor_count == 0 {
        return Vec::new();
    }
    let n = processor_count as usize;
    let expected = n * PROCESSOR_CPU_LOAD_INFO_COUNT;
    let mut out = Vec::with_capacity(n);
    // SAFETY: info_array points to `processor_count * 4` i32 values
    // per the Mach contract; we bound our reads to that length.
    if (info_count as usize) >= expected {
        unsafe {
            let slice = std::slice::from_raw_parts(info_array, expected);
            for core_idx in 0..n {
                let base = core_idx * PROCESSOR_CPU_LOAD_INFO_COUNT;
                // Fields are declared as `integer_t` but conceptually
                // they're unsigned cumulative counters — Apple's own
                // top(1) casts them the same way we do here.
                out.push(PerCoreTicks {
                    user: slice[base] as u64,
                    system: slice[base + 1] as u64,
                    idle: slice[base + 2] as u64,
                    nice: slice[base + 3] as u64,
                });
            }
        }
    }
    // Free the kernel allocation.
    // SAFETY: vm_deallocate with the size returned by the kernel is
    // the documented pairing for host_processor_info.
    unsafe {
        let bytes = (info_count as usize) * std::mem::size_of::<IntegerT>();
        vm_deallocate(mach_task_self(), info_array as *mut _, bytes);
    }
    out
}

#[cfg(not(target_os = "macos"))]
pub fn read_per_core_ticks() -> Vec<PerCoreTicks> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturation_default_on_empty_samples() {
        let sat = CpuSaturation::compute(&[], &[]);
        assert!(sat.per_core_busy.is_empty());
        assert_eq!(sat.mean_busy, 0.0);
        assert_eq!(sat.max_busy, 0.0);
        assert_eq!(sat.pegged_fraction, 0.0);
    }

    #[test]
    fn saturation_default_on_mismatched_lengths() {
        let prev = vec![PerCoreTicks::default(); 4];
        let curr = vec![PerCoreTicks::default(); 8];
        let sat = CpuSaturation::compute(&prev, &curr);
        assert!(sat.per_core_busy.is_empty());
    }

    #[test]
    fn saturation_zero_when_idle() {
        let prev = vec![PerCoreTicks {
            idle: 1000,
            ..Default::default()
        }];
        let curr = vec![PerCoreTicks {
            idle: 2000,
            ..Default::default()
        }];
        let sat = CpuSaturation::compute(&prev, &curr);
        assert_eq!(sat.mean_busy, 0.0);
        assert_eq!(sat.max_busy, 0.0);
        assert_eq!(sat.pegged_fraction, 0.0);
    }

    #[test]
    fn saturation_fully_busy_single_core() {
        let prev = vec![PerCoreTicks::default()];
        let curr = vec![PerCoreTicks {
            user: 500,
            system: 500,
            idle: 0,
            nice: 0,
        }];
        let sat = CpuSaturation::compute(&prev, &curr);
        assert_eq!(sat.mean_busy, 1.0);
        assert_eq!(sat.max_busy, 1.0);
        assert_eq!(sat.pegged_fraction, 1.0);
    }

    #[test]
    fn saturation_mean_and_max_across_cores() {
        // 4 cores: 0%, 50%, 80% (pegged), 100% (pegged).
        let prev = vec![PerCoreTicks::default(); 4];
        let curr = vec![
            PerCoreTicks {
                idle: 100,
                ..Default::default()
            },
            PerCoreTicks {
                user: 50,
                idle: 50,
                ..Default::default()
            },
            PerCoreTicks {
                user: 80,
                idle: 20,
                ..Default::default()
            },
            PerCoreTicks {
                user: 100,
                idle: 0,
                ..Default::default()
            },
        ];
        let sat = CpuSaturation::compute(&prev, &curr);
        assert_eq!(sat.per_core_busy.len(), 4);
        assert!((sat.per_core_busy[0] - 0.0).abs() < 1e-9);
        assert!((sat.per_core_busy[1] - 0.5).abs() < 1e-9);
        assert!((sat.per_core_busy[2] - 0.8).abs() < 1e-9);
        assert!((sat.per_core_busy[3] - 1.0).abs() < 1e-9);
        assert!((sat.mean_busy - 0.575).abs() < 1e-9);
        assert!((sat.max_busy - 1.0).abs() < 1e-9);
        assert_eq!(sat.pegged_fraction, 0.5); // 2 of 4 cores ≥ 0.80
    }

    #[test]
    fn saturation_clamps_nonmonotonic_sample() {
        // Backwards read should not produce negative or > 1 values.
        let prev = vec![PerCoreTicks {
            user: 100,
            idle: 100,
            ..Default::default()
        }];
        let curr = vec![PerCoreTicks {
            user: 50,
            idle: 50,
            ..Default::default()
        }];
        let sat = CpuSaturation::compute(&prev, &curr);
        assert!(sat.mean_busy >= 0.0 && sat.mean_busy <= 1.0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn read_per_core_ticks_returns_nonempty_on_macos() {
        let ticks = read_per_core_ticks();
        assert!(
            !ticks.is_empty(),
            "host_processor_info should return at least one core on macOS"
        );
        // All counters should be non-zero-total on any running system.
        assert!(ticks.iter().any(|t| t.total() > 0));
    }
}
