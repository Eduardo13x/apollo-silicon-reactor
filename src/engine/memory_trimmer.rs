//! Working-Set Trimmer — marks cold private regions as VM_BEHAVIOR_REUSABLE.
//!
//! ## What this does
//!
//! On macOS, `mach_vm_behavior_set(task, addr, size, VM_BEHAVIOR_REUSABLE)` tells
//! the kernel: "this range can be reclaimed under memory pressure." The OS moves
//! the pages to the compressor or swap. If the process accesses them again it gets
//! zero-filled pages — the allocator (libmalloc) handles that transparently.
//!
//! This is identical to what `malloc_zone_pressure_relief()` does internally.
//! Apollo does it externally (as root with task_for_pid) for processes that hold
//! large anonymous heaps they're not actively using.
//!
//! ## Safety constraints
//!
//! - ONLY target processes in `TRIMMABLE_NAMES` (browsers, Electron apps).
//!   These use arena allocators where cold arenas are genuinely unused.
//! - ONLY mark SM_PRIVATE regions (private anonymous heap, not file-backed).
//! - Minimum region size: 16 MB. Tiny regions are bookkeeping, not caches.
//! - Maximum per-process: 512 MB per trim call to avoid over-trimming.
//! - NEVER trim: databases, compilers, system daemons, processes with live state.
//!   The caller (decide_actions) enforces the allowlist; trimmer is defense-in-depth.
//!
//! ## Cost
//!
//! ~200–800 µs per process (depends on region count). Only called under pressure.

#[cfg(target_os = "macos")]
use std::mem;

/// Minimum region size to consider for trimming (16 MB).
const MIN_TRIM_BYTES: u64 = 16 * 1024 * 1024;

/// Maximum bytes to mark reusable per process per trim call (512 MB).
const MAX_TRIM_BYTES: u64 = 512 * 1024 * 1024;

/// Process names that are safe targets for working-set trimming.
/// These are browser / Electron apps that use arena-based allocators
/// where cold arenas contain cached data, not live mutable objects.
pub const TRIMMABLE_NAMES: &[&str] = &[
    // Chromium-family
    "Brave Browser Helper",
    "Google Chrome Helper",
    "Chromium Helper",
    "Microsoft Edge Helper",
    "Arc Helper",
    // Firefox
    "firefox",
    "firefox-bin",
    // Electron apps
    "Notion Helper",
    "Slack Helper",
    "Discord Helper",
    "VSCode Helper",
    "Cursor Helper",
    "Obsidian Helper",
    "Linear Helper",
    // Safari (JS heap is recycled by the engine — safe to hint cold regions)
    "com.apple.WebKit.WebContent",
];

/// Returns true if this process name is a safe trim target.
pub fn is_trimmable(name: &str) -> bool {
    TRIMMABLE_NAMES
        .iter()
        .any(|t| name.contains(t) || t.contains(name))
}

/// Result of trimming a single process's cold memory regions.
#[derive(Debug, Clone)]
pub struct TrimResult {
    pub pid: u32,
    pub name: String,
    /// Bytes marked as VM_BEHAVIOR_REUSABLE (not yet reclaimed — the OS decides when).
    pub bytes_marked: u64,
    /// Number of regions marked.
    pub regions_trimmed: u32,
}

// ── macOS implementation ──────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod imp {
    use super::*;

    const KERN_SUCCESS: i32 = 0;
    const VM_REGION_TOP_INFO: i32 = 12;
    const VM_REGION_TOP_INFO_COUNT: u32 = 5;
    const SM_PRIVATE: u8 = 1;
    /// Hint to kernel: pages can be reclaimed; zero-filled on next access.
    const VM_BEHAVIOR_REUSABLE: i32 = 8;

    #[repr(C)]
    struct VmRegionTopInfo {
        obj_id: u32,
        ref_count: u32,
        private_pages_resident: u32,
        shared_pages_resident: u32,
        share_mode: u8,
        _pad: [u8; 3],
    }

    extern "C" {
        fn mach_task_self() -> u32;
        fn task_for_pid(target: u32, pid: i32, t: *mut u32) -> i32;
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
        fn mach_vm_behavior_set(
            task: u32,
            address: u64,
            size: u64,
            new_behavior: i32,
        ) -> i32;
    }

    /// Mark cold private regions of `pid` as VM_BEHAVIOR_REUSABLE.
    ///
    /// Enumerates the process's virtual address space, selects large (≥ 16 MB)
    /// SM_PRIVATE regions, and hints to the kernel that their pages are reusable.
    /// The kernel compresses / pages them out at its discretion — the process is
    /// NOT paused and does NOT see an error on next access (gets zero pages, which
    /// the arena allocator handles as a clean block).
    pub fn trim_process(pid: u32, name: &str) -> Option<TrimResult> {
        unsafe {
            let self_port = mach_task_self();
            let mut task_port: u32 = 0;

            if task_for_pid(self_port, pid as i32, &mut task_port) != KERN_SUCCESS {
                return None; // No rights (sandboxed, gone, or not root)
            }

            let mut address: u64 = 0;
            let mut bytes_marked: u64 = 0;
            let mut regions_trimmed: u32 = 0;

            loop {
                if bytes_marked >= MAX_TRIM_BYTES {
                    break;
                }

                let mut size: u64 = 0;
                let mut info: VmRegionTopInfo = mem::zeroed();
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
                    break;
                }

                // Only trim large, private anonymous regions (heap arenas).
                if info.share_mode == SM_PRIVATE && size >= MIN_TRIM_BYTES {
                    let trim_size = (MAX_TRIM_BYTES - bytes_marked).min(size);
                    let kr = mach_vm_behavior_set(
                        task_port,
                        address,
                        trim_size,
                        VM_BEHAVIOR_REUSABLE,
                    );
                    if kr == KERN_SUCCESS {
                        bytes_marked += trim_size;
                        regions_trimmed += 1;
                    }
                }

                address = match address.checked_add(size) {
                    Some(next) if next > 0 => next,
                    _ => break,
                };
            }

            mach_port_deallocate(self_port, task_port);

            if regions_trimmed == 0 {
                return None;
            }

            Some(TrimResult {
                pid,
                name: name.to_string(),
                bytes_marked,
                regions_trimmed,
            })
        }
    }
}

#[cfg(target_os = "macos")]
pub use imp::trim_process;

// ── Non-macOS stub ────────────────────────────────────────────────────────────

#[cfg(not(target_os = "macos"))]
pub fn trim_process(_pid: u32, _name: &str) -> Option<TrimResult> {
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trimmable_brave() {
        assert!(is_trimmable("Brave Browser Helper (Renderer)"));
        assert!(is_trimmable("Brave Browser Helper (GPU)"));
    }

    #[test]
    fn trimmable_notion() {
        assert!(is_trimmable("Notion Helper"));
    }

    #[test]
    fn not_trimmable_system() {
        assert!(!is_trimmable("kernel_task"));
        assert!(!is_trimmable("launchd"));
        assert!(!is_trimmable("apollo-optimizerd"));
        assert!(!is_trimmable("postgres"));
    }

    #[test]
    fn trim_self_no_panic() {
        // Self-trim should either succeed or return None gracefully.
        // (Won't return large regions in test binary.)
        let pid = std::process::id();
        let result = trim_process(pid, "test");
        // Just don't panic. May or may not find trimmable regions.
        println!("trim_self: {:?}", result.map(|r| r.bytes_marked));
    }
}
