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

#[cfg(target_os = "macos")]
extern "C" {
    fn mach_task_self() -> u32;
    fn task_for_pid(target: u32, pid: i32, t: *mut u32) -> i32;
    fn task_info(task: u32, flavor: u32, info: *mut i32, count: *mut u32) -> i32;
    fn mach_port_deallocate(task: u32, name: u32) -> i32;
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
#[cfg(target_os = "macos")]
pub fn query_memory_profile(pid: u32) -> Option<ProcessMemoryProfile> {
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
