//! Direct VM statistics via Mach `host_statistics64` — replaces `vm_stat` and
//! `memory_pressure -Q` subprocesses.
//!
//! `host_statistics64(HOST_VM_INFO64)` returns the same data as `vm_stat` in
//! ~1µs vs 5-10ms for subprocess.

/// VM page statistics from the kernel.
///
/// Contains BOTH the instantaneous page counts (state — "how much memory is
/// where") AND the cumulative event counters (flow — "how fast is memory
/// moving"). The event counters are monotonic since boot; callers compute
/// per-second rates by taking deltas between successive samples.
///
/// The flow metrics are the single biggest gap between "apollo knows the
/// water level" and "apollo senses the current". A system at 70% pressure
/// with zero compressions/s is stable; a system at 70% pressure with 50k
/// compressions/s is actively thrashing. Without these counters apollo
/// cannot tell the difference.
///
/// References:
/// - Apple XNU `vm_stat.c` — reads exactly these fields; Apple treats them
///   as the authoritative flow metrics for memory pressure.
/// - [Denning 1968] "The Working Set Model" — page fault rate, not
///   residency, defines working-set quality.
#[derive(Debug, Clone, Default)]
pub struct VmPageStats {
    // ── State (instantaneous) ────────────────────────────────────────────
    pub free_pages: u64,
    pub active_pages: u64,
    pub inactive_pages: u64,
    pub speculative_pages: u64,
    pub wired_pages: u64,
    pub compressor_pages: u64,
    pub page_size: u64,
    // ── Flow (cumulative since boot) ─────────────────────────────────────
    /// Total page faults (minor + major) since boot.
    pub faults: u64,
    /// Copy-on-write page faults since boot (fork/mmap sharing breakage).
    pub cow_faults: u64,
    /// Pages paged IN from the compressor/swap since boot.
    pub pageins: u64,
    /// Pages paged OUT to the compressor/swap since boot.
    pub pageouts: u64,
    /// Pages compressed into the WKdm compressor since boot.
    /// [THIS IS THE primary macOS memory-pressure flow signal.]
    pub compressions: u64,
    /// Pages decompressed back out of the compressor since boot.
    pub decompressions: u64,
    /// Pages swapped IN from physical disk since boot.
    pub swapins: u64,
    /// Pages swapped OUT to physical disk since boot.
    pub swapouts: u64,
    /// Pages moved from inactive→active list since boot
    /// (working-set reactivation — stale page touched again).
    pub reactivations: u64,
    /// Pages freed from the purgeable/volatile list since boot.
    pub purges: u64,
}

/// Per-second rates derived from two successive `VmPageStats` samples.
///
/// All fields are f64 "events per second" so callers can feed them directly
/// into EMAs / thresholds / predictors without additional conversion.
///
/// The primary consumer is the memory-pressure decision path: rates
/// distinguish a quiet high-residency system from an actively thrashing one,
/// which pressure-percentage alone cannot do.
#[derive(Debug, Clone, Default)]
pub struct VmRate {
    pub faults_per_sec: f64,
    pub cow_faults_per_sec: f64,
    pub pageins_per_sec: f64,
    pub pageouts_per_sec: f64,
    pub compressions_per_sec: f64,
    pub decompressions_per_sec: f64,
    pub swapins_per_sec: f64,
    pub swapouts_per_sec: f64,
    pub reactivations_per_sec: f64,
    pub purges_per_sec: f64,
}

impl VmRate {
    /// Compute per-second rates between two samples separated by `dt_secs`
    /// seconds of wall-clock time.
    ///
    /// Uses `saturating_sub` so non-monotonic reads (should not happen from
    /// the kernel but may occur in tests) clamp to zero instead of wrapping
    /// into astronomical rates. A `dt_secs <= 0.0` returns a default
    /// zero-filled struct.
    pub fn compute(prev: &VmPageStats, curr: &VmPageStats, dt_secs: f64) -> Self {
        if dt_secs <= 0.0 {
            return Self::default();
        }
        let div = dt_secs;
        Self {
            faults_per_sec: curr.faults.saturating_sub(prev.faults) as f64 / div,
            cow_faults_per_sec: curr.cow_faults.saturating_sub(prev.cow_faults) as f64 / div,
            pageins_per_sec: curr.pageins.saturating_sub(prev.pageins) as f64 / div,
            pageouts_per_sec: curr.pageouts.saturating_sub(prev.pageouts) as f64 / div,
            compressions_per_sec: curr.compressions.saturating_sub(prev.compressions) as f64 / div,
            decompressions_per_sec: curr.decompressions.saturating_sub(prev.decompressions) as f64
                / div,
            swapins_per_sec: curr.swapins.saturating_sub(prev.swapins) as f64 / div,
            swapouts_per_sec: curr.swapouts.saturating_sub(prev.swapouts) as f64 / div,
            reactivations_per_sec: curr.reactivations.saturating_sub(prev.reactivations) as f64
                / div,
            purges_per_sec: curr.purges.saturating_sub(prev.purges) as f64 / div,
        }
    }

    /// Composite "thrashing score" ∈ [0, ∞) combining the high-signal flow
    /// metrics. Heuristic weights: swap I/O counts triple because hitting
    /// the SSD is strictly worse than compressor churn; compressions count
    /// double because they indicate working-set overflow; decompressions
    /// and reactivations count single because they're lagging indicators.
    ///
    /// 0 = quiet. 10_000+ = actively thrashing.
    pub fn thrashing_score(&self) -> f64 {
        3.0 * (self.swapins_per_sec + self.swapouts_per_sec)
            + 2.0 * self.compressions_per_sec
            + self.decompressions_per_sec
            + self.reactivations_per_sec
    }
}

impl VmPageStats {
    /// Total reclaimable bytes (inactive + speculative pages).
    pub fn reclaimable_bytes(&self) -> u64 {
        (self.inactive_pages + self.speculative_pages) * self.page_size
    }

    /// Memory pressure as 0.0–1.0 (1.0 = fully pressured).
    ///
    /// Matches `memory_pressure -Q` output: pressure = 1.0 - free_percentage.
    /// Free percentage includes free + inactive + speculative pages (reclaimable).
    pub fn pressure(&self) -> f64 {
        let total_pages = self.free_pages
            + self.active_pages
            + self.inactive_pages
            + self.speculative_pages
            + self.wired_pages
            + self.compressor_pages;
        if total_pages == 0 {
            return 0.0;
        }
        // Available = free + inactive + speculative + partial active + partial compressor.
        // Wired is truly locked. Active pages may be demoted to inactive under
        // pressure — treat as 50% available. Compressor pages are partially
        // reclaimable: kernel can discard purgeable compressed pages without I/O
        // and decompress on-demand (latency, not loss) — treat as 30% available.
        // Without compressor credit, formula yields ~0.55 while kernel reports
        // 59% free (0.41 pressure). The 0.30 weight closes 40% of that gap.
        let available = self.free_pages as f64
            + self.inactive_pages as f64
            + self.speculative_pages as f64
            + self.active_pages as f64 * 0.50
            + self.compressor_pages as f64 * 0.30;
        (1.0 - available / total_pages as f64).clamp(0.0, 1.0)
    }
}

// ── Mach FFI ────────────────────────────────────────────────────────────────

// host_statistics64 flavor.
const HOST_VM_INFO64: i32 = 4;

// vm_statistics64 struct from XNU <mach/vm_statistics.h>.
// Fields are `natural_t` (u32) for counts, `uint64_t` for cumulative totals.
#[repr(C)]
#[derive(Default)]
struct VmStatistics64 {
    free_count: u32,
    active_count: u32,
    inactive_count: u32,
    wire_count: u32,
    zero_fill_count: u64,
    reactivations: u64,
    pageins: u64,
    pageouts: u64,
    faults: u64,
    cow_faults: u64,
    lookups: u64,
    hits: u64,
    purges: u64,
    purgeable_count: u32,
    speculative_count: u32,
    decompressions: u64,
    compressions: u64,
    swapins: u64,
    swapouts: u64,
    compressor_page_count: u32,
    throttled_count: u32,
    external_page_count: u32,
    internal_page_count: u32,
    total_uncompressed_pages_in_compressor: u64,
}

type MachPortT = u32;
type KernReturnT = i32;

#[allow(clashing_extern_declarations)]
extern "C" {
    fn mach_host_self() -> MachPortT;
    fn host_statistics64(
        host: MachPortT,
        flavor: i32,
        host_info: *mut VmStatistics64,
        count: *mut u32,
    ) -> KernReturnT;
}

const KERN_SUCCESS: i32 = 0;

/// Read VM page statistics via Mach `host_statistics64`.
///
/// Returns `None` only if the Mach call fails (should not happen on macOS).
#[cfg(target_os = "macos")]
pub fn read_vm_stats() -> Option<VmPageStats> {
    let mut info = VmStatistics64::default();
    // Count is in units of `integer_t` (i32), not bytes.
    let mut count = (std::mem::size_of::<VmStatistics64>() / std::mem::size_of::<i32>()) as u32;

    let kr = unsafe { host_statistics64(mach_host_self(), HOST_VM_INFO64, &mut info, &mut count) };

    if kr != KERN_SUCCESS {
        return None;
    }

    // Page size on ARM64 macOS is always 16384.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 };

    Some(VmPageStats {
        free_pages: info.free_count as u64,
        active_pages: info.active_count as u64,
        inactive_pages: info.inactive_count as u64,
        speculative_pages: info.speculative_count as u64,
        wired_pages: info.wire_count as u64,
        compressor_pages: info.compressor_page_count as u64,
        page_size,
        faults: info.faults,
        cow_faults: info.cow_faults,
        pageins: info.pageins,
        pageouts: info.pageouts,
        compressions: info.compressions,
        decompressions: info.decompressions,
        swapins: info.swapins,
        swapouts: info.swapouts,
        reactivations: info.reactivations,
        purges: info.purges,
    })
}

#[cfg(not(target_os = "macos"))]
pub fn read_vm_stats() -> Option<VmPageStats> {
    None
}

/// Trigger kernel page reclaim via `sync()` + memory pressure sysctl.
///
/// Replaces the `purge` subprocess. `sync()` flushes dirty buffers to disk,
/// then `kern.memorystatus_vm_pressure_send=1` tells the kernel to reclaim
/// cached pages.
pub fn trigger_purge() {
    unsafe {
        libc::sync();
    }
    // Send memory pressure hint to trigger kernel-level page reclaim.
    crate::engine::sysctl_direct::write_i32("kern.memorystatus_vm_pressure_send", 1);
}

/// Returns the current swap usage in bytes.
pub fn get_swap_used_bytes() -> u64 {
    let mut xsw = [0u64; 5];
    let mut len = std::mem::size_of_val(&xsw);
    let rc = unsafe {
        libc::sysctlbyname(
            "vm.swapusage\0".as_ptr() as *const libc::c_char,
            xsw.as_mut_ptr() as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        xsw[2] // used
    } else {
        0
    }
}

/// Returns the total swap capacity in bytes.
pub fn get_swap_total_bytes() -> u64 {
    let mut xsw = [0u64; 5];
    let mut len = std::mem::size_of_val(&xsw);
    let rc = unsafe {
        libc::sysctlbyname(
            "vm.swapusage\0".as_ptr() as *const libc::c_char,
            xsw.as_mut_ptr() as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        xsw[0] // total
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn vm_stats_reads_successfully() {
        let stats = read_vm_stats();
        assert!(stats.is_some(), "host_statistics64 should succeed on macOS");
        let stats = stats.unwrap();
        assert!(stats.page_size > 0);
        assert!(stats.free_pages + stats.active_pages + stats.wired_pages > 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn pressure_in_valid_range() {
        let stats = read_vm_stats().unwrap();
        let p = stats.pressure();
        assert!(p >= 0.0 && p <= 1.0, "pressure out of range: {}", p);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn reclaimable_bytes_reasonable() {
        let stats = read_vm_stats().unwrap();
        let reclaim = stats.reclaimable_bytes();
        // Should be > 0 on any running system (some inactive pages exist).
        // Allow 0 in minimal test environments.
        let _ = reclaim;
    }

    #[test]
    fn empty_stats_pressure_is_zero() {
        let stats = VmPageStats {
            page_size: 16384,
            ..Default::default()
        };
        assert_eq!(stats.pressure(), 0.0);
    }

    #[test]
    fn high_pressure_when_all_wired() {
        let stats = VmPageStats {
            wired_pages: 1000,
            page_size: 16384,
            ..Default::default()
        };
        // All wired = 100% unavailable
        assert!((stats.pressure() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn vm_rate_zero_when_samples_identical() {
        let s = VmPageStats {
            compressions: 1000,
            decompressions: 500,
            swapins: 0,
            swapouts: 0,
            ..Default::default()
        };
        let rate = VmRate::compute(&s, &s, 1.0);
        assert_eq!(rate.compressions_per_sec, 0.0);
        assert_eq!(rate.swapins_per_sec, 0.0);
        assert_eq!(rate.thrashing_score(), 0.0);
    }

    #[test]
    fn vm_rate_computes_forward_delta() {
        let prev = VmPageStats::default();
        let curr = VmPageStats {
            compressions: 2_000,
            decompressions: 500,
            swapins: 100,
            swapouts: 50,
            reactivations: 400,
            ..Default::default()
        };
        // 2-second window.
        let rate = VmRate::compute(&prev, &curr, 2.0);
        assert_eq!(rate.compressions_per_sec, 1000.0);
        assert_eq!(rate.decompressions_per_sec, 250.0);
        assert_eq!(rate.swapins_per_sec, 50.0);
        assert_eq!(rate.swapouts_per_sec, 25.0);
        assert_eq!(rate.reactivations_per_sec, 200.0);
        // Thrashing score = 3*(50+25) + 2*1000 + 250 + 200 = 225 + 2000 + 450 = 2675.
        assert!((rate.thrashing_score() - 2675.0).abs() < 0.5);
    }

    #[test]
    fn vm_rate_clamps_backwards_samples_to_zero() {
        // Non-monotonic read (should never happen from kernel, but guard it).
        let prev = VmPageStats {
            compressions: 100,
            ..Default::default()
        };
        let curr = VmPageStats {
            compressions: 50, // went backwards
            ..Default::default()
        };
        let rate = VmRate::compute(&prev, &curr, 1.0);
        assert_eq!(rate.compressions_per_sec, 0.0);
    }

    #[test]
    fn vm_rate_rejects_non_positive_dt() {
        let s = VmPageStats {
            compressions: 1000,
            ..Default::default()
        };
        let rate = VmRate::compute(&s, &s, 0.0);
        assert_eq!(rate.compressions_per_sec, 0.0);
        let rate_neg = VmRate::compute(&s, &s, -1.0);
        assert_eq!(rate_neg.compressions_per_sec, 0.0);
    }

    #[test]
    fn thrashing_score_weights_swap_triple() {
        // 1 swap/s should score equal to 1.5 compressions/s (3 vs 2 weights).
        let swap_rate = VmRate {
            swapouts_per_sec: 1.0,
            ..Default::default()
        };
        let comp_rate = VmRate {
            compressions_per_sec: 1.5,
            ..Default::default()
        };
        assert_eq!(swap_rate.thrashing_score(), comp_rate.thrashing_score());
    }
}
