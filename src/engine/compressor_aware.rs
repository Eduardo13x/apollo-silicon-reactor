//! Compressor-Aware Freeze Decisions — uses Mach task_info to query
//! compressed page counts before deciding SIGSTOP vs memory-pressure hint.
//!
//! macOS compresses RAM before swapping.  A process with a high compression
//! ratio (e.g. 3:1 for JSON/text data) is cheap to freeze — the kernel
//! keeps pages compressed in RAM and decompresses them instantly on SIGCONT.
//!
//! A process with low compression ratio (e.g. media, encrypted data) would
//! page out to swap on freeze, making SIGCONT expensive (disk I/O latency).

/// TASK_VM_INFO flavor for `task_info()`.
#[cfg(target_os = "macos")]
const TASK_VM_INFO: u32 = 22;
#[cfg(target_os = "macos")]
const KERN_SUCCESS: i32 = 0;

/// VM_REGION_TOP_INFO flavor for `mach_vm_region()`.
#[cfg(target_os = "macos")]
const VM_REGION_TOP_INFO: i32 = 12;
#[cfg(target_os = "macos")]
const VM_REGION_TOP_INFO_COUNT: u32 = 5;
/// Share modes from XNU osfmk/mach/vm_region.h.
#[cfg(target_os = "macos")]
const SM_PRIVATE: u8 = 1;
#[cfg(target_os = "macos")]
const SM_SHARED: u8 = 3;
#[cfg(target_os = "macos")]
const SM_TRUESHARED: u8 = 5;

#[cfg(target_os = "macos")]
extern "C" {
    fn mach_task_self() -> u32;
    fn task_for_pid(target: u32, pid: i32, t: *mut u32) -> i32;
    fn task_info(task: u32, flavor: u32, info: *mut i32, count: *mut u32) -> i32;
    fn mach_port_deallocate(task: u32, name: u32) -> i32;
    fn mach_vm_region(
        task: u32,
        address: *mut u64,
        size: *mut u64,
        flavor: i32,
        info: *mut i32,
        info_count: *mut u32,
        object_name: *mut u32,
    ) -> i32;
    fn mach_vm_read_overwrite(
        task: u32,
        address: u64,
        size: u64,
        data: u64,
        out_size: *mut u64,
    ) -> i32;
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
}

/// Mach timebase info for converting mach_absolute_time to nanoseconds.
#[cfg(target_os = "macos")]
#[repr(C)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

/// vm_region_top_info — compact per-region descriptor from XNU.
/// Flavor 12 (VM_REGION_TOP_INFO), count 5.
#[cfg(target_os = "macos")]
#[repr(C)]
struct VmRegionTopInfo {
    obj_id: u32,
    ref_count: u32,
    private_pages_resident: u32,
    shared_pages_resident: u32,
    share_mode: u8,
    _pad: [u8; 3],
}

/// Summary of a process's virtual memory region layout.
///
/// Produced by `scan_regions()` via `mach_vm_region_recurse`.
/// Cost: ~200-500µs per process. Only call on freeze candidates.
#[derive(Debug, Clone)]
pub struct RegionSummary {
    /// Number of non-submap regions.
    pub n_regions: u32,
    /// Resident bytes in private (SM_PRIVATE) regions.
    pub private_bytes: u64,
    /// Resident bytes in shared (SM_SHARED | SM_TRUESHARED) regions.
    pub shared_bytes: u64,
    /// Virtual size of the largest single private region.
    pub largest_anon_bytes: u64,
    /// Mean region virtual size (total / n_regions).
    pub mean_region_size: u64,
    /// Total virtual size across all regions.
    pub total_virtual: u64,
}

/// Enumerate memory regions of a process via `mach_vm_region` + `VM_REGION_TOP_INFO`.
///
/// Returns an aggregate summary without storing per-region details.
/// Cost: ~200-500µs depending on region count (Chrome ≈ 500+ regions).
#[cfg(target_os = "macos")]
pub fn scan_regions(pid: u32) -> Option<RegionSummary> {
    unsafe {
        let self_port = mach_task_self();
        let (task_port, need_dealloc) = if pid == std::process::id() {
            (self_port, false) // own task port — no dealloc needed
        } else {
            let mut port: u32 = 0;
            let kr = task_for_pid(self_port, pid as i32, &mut port);
            if kr != KERN_SUCCESS {
                return None;
            }
            (port, true)
        };

        let page_size: u64 = 16384; // ARM64

        let mut address: u64 = 0;
        let mut n_regions: u32 = 0;
        let mut private_bytes: u64 = 0;
        let mut shared_bytes: u64 = 0;
        let mut largest_anon: u64 = 0;
        let mut total_virtual: u64 = 0;

        loop {
            let mut size: u64 = 0;
            let mut info: VmRegionTopInfo = std::mem::zeroed();
            let mut count = VM_REGION_TOP_INFO_COUNT;
            let mut obj_name: u32 = 0;

            let kr = mach_vm_region(
                task_port,
                &mut address,
                &mut size,
                VM_REGION_TOP_INFO,
                &mut info as *mut _ as *mut i32,
                &mut count,
                &mut obj_name,
            );
            if kr != KERN_SUCCESS {
                break; // end of address space
            }

            n_regions += 1;
            total_virtual += size;

            // Use resident page counts for meaningful byte values.
            let priv_resident = info.private_pages_resident as u64 * page_size;
            let shr_resident = info.shared_pages_resident as u64 * page_size;

            match info.share_mode {
                SM_PRIVATE => {
                    private_bytes += priv_resident;
                    if size > largest_anon {
                        largest_anon = size;
                    }
                }
                SM_SHARED | SM_TRUESHARED => {
                    shared_bytes += shr_resident;
                }
                _ => {} // COW, empty, etc.
            }

            address += size;
            if address == 0 {
                break; // wrapped around
            }
        }

        if need_dealloc {
            mach_port_deallocate(self_port, task_port);
        }

        if n_regions == 0 {
            return None;
        }

        Some(RegionSummary {
            n_regions,
            private_bytes,
            shared_bytes,
            largest_anon_bytes: largest_anon,
            mean_region_size: total_virtual / n_regions as u64,
            total_virtual,
        })
    }
}

#[cfg(not(target_os = "macos"))]
pub fn scan_regions(_pid: u32) -> Option<RegionSummary> {
    None
}

// ── Page Temperature Oracle ──────────────────────────────────────────────────
//
// iLeakage (CCS'23): on M1, mach_absolute_time has ~42ns resolution.
// Access latency reveals where a page lives:
//   - L2/SLC (< 150ns) — hot, actively cached
//   - DRAM (150–600ns) — resident but not cached
//   - Compressed (> 600ns) — compressor must decompress
//
// This tells Apollo the real cost of freezing: hot pages = expensive to evict,
// compressed pages = already paying the penalty.

/// Where a page currently resides in the memory hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageTemp {
    /// L2 / SLC — cached, access < 150ns.
    SlcHot,
    /// DRAM — resident, not cached, 150–600ns.
    Dram,
    /// Compressed or swapped — > 600ns.
    Compressed,
    /// Could not read (permission, unmapped, etc.).
    Unreachable,
}

/// Aggregate temperature profile of a process's memory regions.
#[derive(Debug, Clone)]
pub struct TempProfile {
    /// Fraction of sampled pages in L2/SLC (0.0–1.0).
    pub pct_hot: f64,
    /// Fraction of sampled pages in DRAM (0.0–1.0).
    pub pct_dram: f64,
    /// Fraction of sampled pages compressed/swapped (0.0–1.0).
    pub pct_compressed: f64,
    /// Number of pages successfully sampled.
    pub sample_count: u32,
}

/// Sample up to 8 memory regions of a process and classify their temperature
/// using **relative timing** (percentile-based).
///
/// mach_vm_read_overwrite adds ~4µs Mach trap overhead, so absolute thresholds
/// don't work. Instead: read N pages, sort by latency, classify by percentile.
/// Bottom 25% → SlcHot, middle 50% → Dram, top 25% → Compressed.
/// Pages ≥2× median → definitely Compressed.
///
/// Cost: ~8 × vm_read ≈ 30–50µs total.
#[cfg(target_os = "macos")]
pub fn sample_process_temperature(pid: u32) -> Option<TempProfile> {
    let self_port = unsafe { mach_task_self() };
    let (task_port, need_dealloc) = if pid == std::process::id() {
        (self_port, false)
    } else {
        let mut port: u32 = 0;
        let kr = unsafe { task_for_pid(self_port, pid as i32, &mut port) };
        if kr != KERN_SUCCESS {
            return None;
        }
        (port, true)
    };

    // Enumerate private regions for probing.
    let mut regions: Vec<(u64, u64)> = Vec::new();
    unsafe {
        let mut address: u64 = 0;
        loop {
            let mut size: u64 = 0;
            let mut info: VmRegionTopInfo = std::mem::zeroed();
            let mut count = VM_REGION_TOP_INFO_COUNT;
            let mut obj: u32 = 0;

            let kr = mach_vm_region(
                task_port,
                &mut address,
                &mut size,
                VM_REGION_TOP_INFO,
                &mut info as *mut _ as *mut i32,
                &mut count,
                &mut obj,
            );
            if kr != KERN_SUCCESS {
                break;
            }
            if info.share_mode == SM_PRIVATE && size >= 16384 {
                regions.push((address, size));
            }
            address += size;
            if address == 0 {
                break;
            }
        }
    }

    // Sort by size descending, take top 8.
    regions.sort_by(|a, b| b.1.cmp(&a.1));
    regions.truncate(8);

    // Probe each region and record latency.
    let mut timings: Vec<u64> = Vec::new();
    unsafe {
        let mut tbi: MachTimebaseInfo = std::mem::zeroed();
        mach_timebase_info(&mut tbi);
        let numer = tbi.numer as u64;
        let denom = tbi.denom.max(1) as u64;

        for &(addr, size) in &regions {
            let probe_addr = (addr + size / 2) & !0x3FFF; // page-align
            if probe_addr < addr {
                continue;
            }

            let mut buf: u8 = 0;
            let mut out_size: u64 = 0;
            let t0 = mach_absolute_time();
            let kr = mach_vm_read_overwrite(
                task_port,
                probe_addr,
                1,
                &mut buf as *mut u8 as u64,
                &mut out_size,
            );
            let t1 = mach_absolute_time();
            if kr == KERN_SUCCESS && out_size > 0 {
                timings.push((t1 - t0) * numer / denom);
            }
        }
    }

    if need_dealloc {
        unsafe {
            mach_port_deallocate(self_port, task_port);
        }
    }

    let n = timings.len();
    if n == 0 {
        return None;
    }

    // Classify by percentile: sort timings, use median as reference.
    timings.sort_unstable();
    let median = timings[n / 2];
    // Threshold: anything ≥ 1.8× median is "compressed" (decompression adds latency).
    // Anything ≤ 0.7× median is "SLC hot" (cache hit is faster).
    let compressed_thresh = median * 18 / 10;
    let hot_thresh = median * 7 / 10;

    let mut hot = 0u32;
    let mut dram = 0u32;
    let mut compressed = 0u32;

    for &t in &timings {
        if t <= hot_thresh {
            hot += 1;
        } else if t >= compressed_thresh {
            compressed += 1;
        } else {
            dram += 1;
        }
    }

    let total = n as f64;
    Some(TempProfile {
        pct_hot: hot as f64 / total,
        pct_dram: dram as f64 / total,
        pct_compressed: compressed as f64 / total,
        sample_count: n as u32,
    })
}

#[cfg(not(target_os = "macos"))]
pub fn sample_process_temperature(_pid: u32) -> Option<TempProfile> {
    None
}

/// Subset of `task_vm_info` we care about.  Layout must match XNU exactly.
#[cfg(target_os = "macos")]
#[repr(C)]
struct TaskVmInfo {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
}

/// Per-process memory profile relevant to freeze and budget decisions.
#[derive(Debug, Clone)]
pub struct ProcessMemoryProfile {
    pub pid: u32,
    /// Physical footprint (bytes).
    pub phys_footprint: u64,
    /// Bytes currently in the compressor.
    pub compressed_bytes: u64,
    /// Purgeable volatile pages (bytes, can be discarded without I/O).
    pub purgeable_bytes: u64,
    /// Compression ratio: phys_footprint / (phys_footprint + compressed).
    /// Higher = more compressible = cheaper to freeze.
    pub compression_ratio: f64,
    /// Working set size estimate (bytes): internal + external - reusable pages.
    /// Proxy for Denning's WSS — pages the process actually needs in RAM.
    pub working_set_bytes: u64,
    /// Resident pages in physical RAM (bytes).
    pub resident_bytes: u64,
}

/// What action to take based on memory profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAction {
    /// SIGSTOP — high compression ratio, freeze is cheap.
    Freeze,
    /// Send `kern.memorystatus_vm_pressure_send` — process will
    /// release caches without needing to be stopped.
    PressureHint,
    /// Do not touch — freezing would cause heavy swap I/O.
    Skip,
}

/// Query the Mach kernel for a process's compressed-memory profile.
///
/// Cost: ~50 μs (task_for_pid + task_info).
/// Only call for freeze candidates, not on the hot path.
///
/// Falls back to proc_pid_rusage when task_for_pid fails (ad-hoc signing
/// on Apple Silicon lacks com.apple.system-task-ports entitlement).
#[cfg(target_os = "macos")]
pub fn query_memory_profile(pid: u32) -> Option<ProcessMemoryProfile> {
    query_memory_profile_mach(pid).or_else(|| query_memory_profile_rusage(pid))
}

/// Fallback: build a partial profile from proc_pid_rusage (always works as root).
#[cfg(target_os = "macos")]
fn query_memory_profile_rusage(pid: u32) -> Option<ProcessMemoryProfile> {
    let rusage = crate::engine::proc_taskinfo::get_rusage_info(pid)?;
    Some(ProcessMemoryProfile {
        pid,
        phys_footprint: rusage.phys_footprint,
        compressed_bytes: 0, // unavailable via rusage
        purgeable_bytes: 0,
        compression_ratio: 1.0, // assume uncompressed (conservative)
        working_set_bytes: rusage.resident_size,
        resident_bytes: rusage.resident_size,
    })
}

#[cfg(target_os = "macos")]
fn query_memory_profile_mach(pid: u32) -> Option<ProcessMemoryProfile> {
    unsafe {
        let mut task_port: u32 = 0;
        let kr = task_for_pid(mach_task_self(), pid as i32, &mut task_port);
        if kr != KERN_SUCCESS {
            return None;
        }

        let mut info: TaskVmInfo = std::mem::zeroed();
        let mut count = (std::mem::size_of::<TaskVmInfo>() / 4) as u32;
        let kr = task_info(
            task_port,
            TASK_VM_INFO,
            &mut info as *mut _ as *mut i32,
            &mut count,
        );
        mach_port_deallocate(mach_task_self(), task_port);

        if kr != KERN_SUCCESS {
            return None;
        }

        let page_size = if info.page_size > 0 {
            info.page_size as u64
        } else {
            16384 // ARM64 default
        };

        let compressed_bytes = info.compressed * page_size;
        let purgeable_bytes = info.purgeable_volatile_resident * page_size;
        let phys = info.phys_footprint;
        let resident_bytes = info.resident_size;

        let compression_ratio = if compressed_bytes > 0 {
            (phys + compressed_bytes) as f64 / phys.max(1) as f64
        } else {
            1.0
        };

        // WSS estimate: internal (private heap/stack) + external (shared/file-backed)
        // minus reusable (pages the kernel can reclaim without I/O).
        // All values are in pages → multiply by page_size.
        let internal_bytes = info.internal * page_size;
        let external_bytes = info.external * page_size;
        let reusable_bytes = info.reusable * page_size;
        let working_set_bytes = (internal_bytes + external_bytes).saturating_sub(reusable_bytes);

        Some(ProcessMemoryProfile {
            pid,
            phys_footprint: phys,
            compressed_bytes,
            purgeable_bytes,
            compression_ratio,
            working_set_bytes,
            resident_bytes,
        })
    }
}

#[cfg(not(target_os = "macos"))]
pub fn query_memory_profile(_pid: u32) -> Option<ProcessMemoryProfile> {
    None
}

/// Decide whether to freeze, hint, or skip based on memory profile.
///
/// Decision matrix (Denning & Schwartz 1972, adapted for macOS compressor):
///   - High ratio (≥2.0): data compresses well → freeze is cheap (~1-5µs decompress)
///   - Low ratio (<1.5) + large footprint: freeze → swap I/O → expensive SIGCONT
///   - Thrashing process (high page-in rate): compressor is in a decompress→recompress
///     loop — freezing breaks the loop (good), but only if ratio is decent
///   - Purgeable pages: kernel can discard without I/O → hint is enough
pub fn decide_memory_action(
    profile: &ProcessMemoryProfile,
    system_pressure: f64,
    major_faults_per_sec: f64,
) -> MemoryAction {
    // Lots of purgeable memory → a hint is enough, no need to freeze.
    if profile.purgeable_bytes > 50 * 1024 * 1024 {
        return MemoryAction::PressureHint;
    }

    // Process is actively thrashing (high page-in rate).
    // If compression ratio is decent, freezing breaks the thrash loop — do it.
    // If ratio is low, the compressor can't help — freeze would cause swap storm.
    if major_faults_per_sec > 50.0 {
        if profile.compression_ratio >= 1.5 {
            // Thrashing but compressible: freeze breaks the decompress→use→recompress loop.
            return MemoryAction::Freeze;
        }
        // Thrashing and incompressible: freezing would cause massive swap I/O.
        // Send pressure hint so the app releases caches voluntarily.
        return MemoryAction::PressureHint;
    }

    // High compression ratio → freeze is cheap (fast decompress on SIGCONT).
    if profile.compression_ratio >= 2.0 {
        return MemoryAction::Freeze;
    }

    // Low compression ratio + large footprint → freeze causes page-out to swap.
    if profile.compression_ratio < 1.5 && profile.phys_footprint > 200 * 1024 * 1024 {
        if system_pressure > 0.85 {
            // Emergency — freeze anyway, swap latency is the lesser evil.
            return MemoryAction::Freeze;
        }
        return MemoryAction::PressureHint;
    }

    MemoryAction::Freeze
}

/// Enhanced freeze decision using temperature + WSS + SLC knowledge.
///
/// Supplements `decide_memory_action` with deep-scan signals. Falls through
/// to the legacy decision when enhanced data is unavailable.
///
/// Key insight (MEMTIS SOSP'23 + M1 SLC architecture):
/// - M1 has 8MB System Level Cache shared by CPU+GPU+ANE
/// - A process with WSS < (SLC / active_clients) is free to freeze —
///   its pages remain cached in SLC, zero decompression on SIGCONT
/// - A mostly-compressed process is already paying the compression tax;
///   freezing would force swap and make things worse
pub fn decide_enhanced(
    profile: &ProcessMemoryProfile,
    temp: Option<&TempProfile>,
    damon_wss: Option<u64>,
    active_process_count: usize,
    system_pressure: f64,
    major_faults_per_sec: f64,
) -> MemoryAction {
    // SLC-aware fast path: if WSS fits in this process's SLC share, freeze is free.
    const SLC_BYTES: u64 = 8 * 1024 * 1024; // M1 SLC = 8 MB
    if let Some(wss) = damon_wss {
        let slc_share = SLC_BYTES / active_process_count.max(1) as u64;
        if wss > 0 && wss <= slc_share {
            return MemoryAction::Freeze;
        }
    }

    // Temperature-aware paths.
    if let Some(t) = temp {
        // Mostly compressed: freeze would cause swap (compressor pages move to disk).
        if t.pct_compressed > 0.60 {
            return MemoryAction::Skip;
        }
        // Actively hot: process is using its pages right now. Hint is gentler.
        if t.pct_hot > 0.80 {
            return MemoryAction::PressureHint;
        }
    }

    // Rusage-only paths: when temp and WSS are unavailable (task_for_pid fails
    // on ad-hoc signed binaries), use footprint heuristics for skip/hint decisions
    // instead of defaulting everything to Freeze.
    if temp.is_none() && damon_wss.is_none() {
        // Actively thrashing without compression data: skip (freeze is risky).
        if major_faults_per_sec > 50.0 {
            return MemoryAction::Skip;
        }
        // Large resident process under moderate pressure: hint is safer than freeze.
        if profile.phys_footprint > 200 * 1024 * 1024 && system_pressure < 0.80 {
            return MemoryAction::PressureHint;
        }
    }

    // Fall through to legacy decision.
    decide_memory_action(profile, system_pressure, major_faults_per_sec)
}

/// Assess how efficiently the compressor is serving this process.
/// Returns a score in [0.0, 1.0]:
///   - 1.0 = compressor is very effective (high ratio, worth keeping compressed)
///   - 0.0 = compressor is wasting effort (low ratio, large footprint)
///
/// Used for throttle prioritization: processes with low compressor efficiency
/// should be throttled earlier because they waste compressor bandwidth.
pub fn compressor_efficiency_score(profile: &ProcessMemoryProfile) -> f64 {
    if profile.compressed_bytes == 0 {
        return 1.0; // Nothing compressed → not burdening the compressor
    }

    // Ratio contribution: 1.0 is no compression, 4.0+ is excellent.
    // Map [1.0, 4.0] → [0.0, 1.0] linearly.
    let ratio_score = ((profile.compression_ratio - 1.0) / 3.0).clamp(0.0, 1.0);

    // Size contribution: larger compressed footprint = more compressor work.
    // Penalize processes with >500MB compressed (on 8GB system, that's significant).
    let size_penalty = (profile.compressed_bytes as f64 / (500.0 * 1024.0 * 1024.0)).min(1.0);

    // Combine: good ratio offsets large size, but very large size always penalizes.
    (ratio_score * 0.7 + (1.0 - size_penalty) * 0.3).clamp(0.0, 1.0)
}

// ── Cross-process memory reclaim ─────────────────────────────────────────────

/// Purgeable state constants from XNU <mach/vm_purgable.h>.
#[cfg(target_os = "macos")]
const VM_PURGABLE_SET_STATE: i32 = 0;
#[cfg(target_os = "macos")]
const VM_PURGABLE_VOLATILE: i32 = 1;

#[cfg(target_os = "macos")]
extern "C" {
    fn mach_vm_purgable_control(task: u32, address: u64, control: i32, state: *mut i32) -> i32;
}

/// Reclaim purgeable memory from another process without killing it.
///
/// Walks the process's VM regions, finds purgeable private regions, and marks
/// them volatile via `mach_vm_purgable_control`. The kernel can then reclaim
/// those pages on demand — the owning process gets zero-filled pages on next access.
///
/// This is the cross-process equivalent of `hint_free()` / `hint_dontneed()`:
/// reclaim RAM from a live process without SIGSTOP.
///
/// Returns the number of regions successfully marked volatile, or None on failure.
/// Cost: ~200-500µs (region enumeration + purgable_control per region).
#[cfg(target_os = "macos")]
pub fn purge_purgeable_regions(pid: u32) -> Option<u32> {
    unsafe {
        let self_port = mach_task_self();
        let (task_port, need_dealloc) = if pid == std::process::id() {
            (self_port, false)
        } else {
            let mut port: u32 = 0;
            let kr = task_for_pid(self_port, pid as i32, &mut port);
            if kr != KERN_SUCCESS {
                return None;
            }
            (port, true)
        };

        let mut purged = 0u32;
        let mut address: u64 = 0;

        loop {
            let mut size: u64 = 0;
            let mut info: VmRegionTopInfo = std::mem::zeroed();
            let mut count = VM_REGION_TOP_INFO_COUNT;
            let mut obj: u32 = 0;

            let kr = mach_vm_region(
                task_port,
                &mut address,
                &mut size,
                VM_REGION_TOP_INFO,
                &mut info as *mut _ as *mut i32,
                &mut count,
                &mut obj,
            );
            if kr != KERN_SUCCESS {
                break;
            }

            // Only target private regions with resident pages.
            // Purgeable memory is identified by attempting the purgable_control call —
            // non-purgeable regions return KERN_INVALID_ARGUMENT (harmlessly).
            if info.share_mode == SM_PRIVATE && info.private_pages_resident > 0 {
                let mut state = VM_PURGABLE_VOLATILE;
                let kr =
                    mach_vm_purgable_control(task_port, address, VM_PURGABLE_SET_STATE, &mut state);
                if kr == KERN_SUCCESS {
                    purged += 1;
                }
                // KERN_INVALID_ARGUMENT = not purgeable → harmless, skip.
            }

            address += size;
            if address == 0 {
                break;
            }
        }

        if need_dealloc {
            mach_port_deallocate(self_port, task_port);
        }

        Some(purged)
    }
}

#[cfg(not(target_os = "macos"))]
pub fn purge_purgeable_regions(_pid: u32) -> Option<u32> {
    None
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Region scanning tests ────────────────────────────────────────────

    #[test]
    fn scan_self_regions() {
        let pid = std::process::id();
        let summary = scan_regions(pid);
        assert!(summary.is_some(), "should scan own process (pid={})", pid);
        let s = summary.unwrap();
        assert!(
            s.n_regions > 10,
            "self should have many regions: {}",
            s.n_regions
        );
        assert!(s.private_bytes > 0, "self should have private memory");
        assert!(s.total_virtual > 0, "self should have virtual memory");
    }

    #[test]
    fn region_summary_has_sane_sizes() {
        let pid = std::process::id();
        if let Some(s) = scan_regions(pid) {
            // Private + shared should not exceed total virtual.
            assert!(
                s.private_bytes + s.shared_bytes <= s.total_virtual + 1024 * 1024,
                "private({}) + shared({}) should be <= total({})",
                s.private_bytes,
                s.shared_bytes,
                s.total_virtual
            );
            assert!(
                s.mean_region_size > 0,
                "mean region size should be positive"
            );
            assert!(
                s.largest_anon_bytes <= s.total_virtual,
                "largest anon should be <= total virtual"
            );
        }
    }

    #[test]
    fn scan_invalid_pid_returns_none() {
        // PID 99999 almost certainly doesn't exist.
        assert!(scan_regions(99999).is_none());
    }

    #[test]
    fn region_count_is_realistic() {
        let pid = std::process::id();
        if let Some(s) = scan_regions(pid) {
            // A Rust test binary should have between 20 and 5000 regions.
            assert!(
                s.n_regions >= 20 && s.n_regions <= 5000,
                "unexpected region count: {}",
                s.n_regions
            );
        }
    }

    // ── Page temperature tests ───────────────────────────────────────────

    // ── Enhanced freeze decision tests ──────────────────────────────────

    fn test_profile(phys: u64, compressed: u64) -> ProcessMemoryProfile {
        ProcessMemoryProfile {
            pid: 1,
            phys_footprint: phys,
            compressed_bytes: compressed,
            purgeable_bytes: 0,
            compression_ratio: if compressed > 0 {
                (phys + compressed) as f64 / phys.max(1) as f64
            } else {
                1.0
            },
            working_set_bytes: phys,
            resident_bytes: phys,
        }
    }

    #[test]
    fn decide_slc_fit_freezes() {
        let profile = test_profile(100_000_000, 0);
        // WSS = 1MB, 4 active processes → SLC share = 2MB → fits.
        let action = decide_enhanced(&profile, None, Some(1_000_000), 4, 0.50, 0.0);
        assert_eq!(action, MemoryAction::Freeze);
    }

    #[test]
    fn decide_hot_process_gets_hint() {
        let profile = test_profile(500_000_000, 0);
        let temp = TempProfile {
            pct_hot: 0.90,
            pct_dram: 0.10,
            pct_compressed: 0.0,
            sample_count: 8,
        };
        let action = decide_enhanced(&profile, Some(&temp), None, 10, 0.50, 0.0);
        assert_eq!(action, MemoryAction::PressureHint);
    }

    #[test]
    fn decide_mostly_compressed_skips() {
        let profile = test_profile(500_000_000, 300_000_000);
        let temp = TempProfile {
            pct_hot: 0.10,
            pct_dram: 0.20,
            pct_compressed: 0.70,
            sample_count: 8,
        };
        let action = decide_enhanced(&profile, Some(&temp), None, 10, 0.50, 0.0);
        assert_eq!(action, MemoryAction::Skip);
    }

    #[test]
    fn decide_falls_through_to_legacy() {
        // No enhanced data → should behave like decide_memory_action.
        let profile = test_profile(100_000_000, 0);
        let enhanced = decide_enhanced(&profile, None, None, 10, 0.50, 0.0);
        let legacy = decide_memory_action(&profile, 0.50, 0.0);
        assert_eq!(enhanced, legacy, "should fall through to legacy");
    }

    // ── Page temperature tests ───────────────────────────────────────────

    #[test]
    fn sample_own_process_returns_profile() {
        let pid = std::process::id();
        let profile = sample_process_temperature(pid);
        assert!(profile.is_some(), "should sample own process temperature");
        let p = profile.unwrap();
        assert!(p.sample_count > 0, "should have sampled at least 1 page");
        // Own process pages should be mostly hot or in DRAM.
        let hot_or_dram = p.pct_hot + p.pct_dram;
        assert!(
            hot_or_dram > 0.3,
            "own process should have hot/dram pages: hot={} dram={} compressed={}",
            p.pct_hot,
            p.pct_dram,
            p.pct_compressed
        );
    }

    #[test]
    fn temp_profile_fractions_sum_to_one() {
        let pid = std::process::id();
        if let Some(p) = sample_process_temperature(pid) {
            let sum = p.pct_hot + p.pct_dram + p.pct_compressed;
            assert!(
                (sum - 1.0).abs() < 0.01,
                "fractions should sum to 1.0: {}",
                sum
            );
        }
    }

    #[test]
    fn sample_invalid_pid_returns_none() {
        assert!(sample_process_temperature(99999).is_none());
    }
}
