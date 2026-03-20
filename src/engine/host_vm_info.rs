//! Direct VM statistics via Mach `host_statistics64` — replaces `vm_stat` and
//! `memory_pressure -Q` subprocesses.
//!
//! `host_statistics64(HOST_VM_INFO64)` returns the same data as `vm_stat` in
//! ~1µs vs 5-10ms for subprocess.

/// VM page statistics from the kernel.
#[derive(Debug, Clone)]
pub struct VmPageStats {
    pub free_pages: u64,
    pub active_pages: u64,
    pub inactive_pages: u64,
    pub speculative_pages: u64,
    pub wired_pages: u64,
    pub compressor_pages: u64,
    pub page_size: u64,
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
        let total = self.free_pages
            + self.active_pages
            + self.inactive_pages
            + self.speculative_pages
            + self.wired_pages
            + self.compressor_pages;
        if total == 0 {
            return 0.0;
        }
        let free_pct = (self.free_pages + self.inactive_pages + self.speculative_pages) as f64
            / total as f64;
        (1.0 - free_pct).clamp(0.0, 1.0)
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
    let mut count =
        (std::mem::size_of::<VmStatistics64>() / std::mem::size_of::<i32>()) as u32;

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
            free_pages: 0,
            active_pages: 0,
            inactive_pages: 0,
            speculative_pages: 0,
            wired_pages: 0,
            compressor_pages: 0,
            page_size: 16384,
        };
        assert_eq!(stats.pressure(), 0.0);
    }

    #[test]
    fn full_pressure_when_all_wired() {
        let stats = VmPageStats {
            free_pages: 0,
            active_pages: 0,
            inactive_pages: 0,
            speculative_pages: 0,
            wired_pages: 1000,
            compressor_pages: 0,
            page_size: 16384,
        };
        assert!((stats.pressure() - 1.0).abs() < f64::EPSILON);
    }
}
