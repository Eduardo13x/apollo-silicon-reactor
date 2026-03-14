//! VM Surgeon — surgical memory management at the mlock/madvise/mincore level.
//!
//! Provides direct control over the virtual memory subsystem:
//! - `pin_memory()` / `unpin_memory()` — mlock/munlock: keep pages in physical RAM
//! - `hint_willneed()` / `hint_free()` / `hint_sequential()` — madvise: guide the pager
//! - `check_resident()` / `resident_ratio()` — mincore: check page residency
//!
//! These are the lowest-level VM operations available in EL0 userspace.
//! Below this is the kernel pager (Mach VM) which we cannot touch.

use std::io;

// ── Page size ────────────────────────────────────────────────────────────────

/// macOS ARM64 page size: 16 KB (not 4 KB like x86).
/// Using the runtime value from sysconf for correctness.
pub fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

// ── mlock / munlock ──────────────────────────────────────────────────────────

/// Pin a memory region in physical RAM. Pages will not be paged out until
/// `unpin_memory()` is called or the process exits.
///
/// Use this for:
/// - Safety state (frozen_set, process tables) that MUST be accessible under
///   extreme memory pressure with zero-latency.
/// - Probe buffers before measurement to guarantee cache-warm reads.
///
/// # Errors
/// Fails if the region exceeds the process's RLIMIT_MEMLOCK (default 64 MB
/// on macOS, unlimited for root).
pub fn pin_memory(ptr: *const u8, len: usize) -> io::Result<()> {
    let rc = unsafe { libc::mlock(ptr as *const libc::c_void, len) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Unpin a previously pinned region. Pages become eligible for paging again.
pub fn unpin_memory(ptr: *const u8, len: usize) -> io::Result<()> {
    let rc = unsafe { libc::munlock(ptr as *const libc::c_void, len) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

// ── madvise ──────────────────────────────────────────────────────────────────

/// Tell the kernel we will need this memory soon. The pager will start
/// bringing pages into RAM asynchronously (readahead).
///
/// Use before hw_predictor probe measurements to ensure the buffer is hot.
pub fn hint_willneed(ptr: *const u8, len: usize) -> io::Result<()> {
    madvise_raw(ptr, len, libc::MADV_WILLNEED)
}

/// Tell the kernel these pages can be freed without writing back.
/// The pages remain mapped but their backing store is released.
/// Next access will zero-fill (for anonymous) or re-read (for file-backed).
///
/// Use after releasing process tracking state to immediately reclaim RAM.
pub fn hint_free(ptr: *mut u8, len: usize) -> io::Result<()> {
    madvise_raw(ptr, len, libc::MADV_FREE)
}

/// Tell the kernel we will access this region sequentially.
/// Enables aggressive readahead, useful for journal writes.
pub fn hint_sequential(ptr: *const u8, len: usize) -> io::Result<()> {
    madvise_raw(ptr, len, libc::MADV_SEQUENTIAL)
}

/// Tell the kernel we will access this region randomly.
/// Disables readahead, useful for hash table lookups.
pub fn hint_random(ptr: *const u8, len: usize) -> io::Result<()> {
    madvise_raw(ptr, len, libc::MADV_RANDOM)
}

/// Tell the kernel we don't need this region anymore. Pages are immediately
/// eligible for reclaim. More aggressive than MADV_FREE.
pub fn hint_dontneed(ptr: *const u8, len: usize) -> io::Result<()> {
    madvise_raw(ptr, len, libc::MADV_DONTNEED)
}

fn madvise_raw(ptr: *const u8, len: usize, advice: libc::c_int) -> io::Result<()> {
    let rc = unsafe { libc::madvise(ptr as *mut libc::c_void, len, advice) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

// ── mincore ──────────────────────────────────────────────────────────────────

/// Check which pages of a memory region are currently resident in RAM.
/// Returns a Vec<bool> with one entry per page (true = in RAM).
///
/// Use before deciding to freeze a process:
/// - If most pages are already paged out, the process is not consuming
///   physical RAM and freezing it is pointless.
pub fn check_resident(ptr: *const u8, len: usize) -> io::Result<Vec<bool>> {
    let ps = page_size();
    let n_pages = (len + ps - 1) / ps;
    if n_pages == 0 {
        return Ok(vec![]);
    }

    // mincore wants page-aligned address
    let aligned = (ptr as usize) & !(ps - 1);
    let adjusted_len = (ptr as usize + len) - aligned;
    let n_pages_adj = (adjusted_len + ps - 1) / ps;

    let mut vec = vec![0u8; n_pages_adj];
    let rc = unsafe {
        libc::mincore(
            aligned as *mut libc::c_void,
            adjusted_len,
            vec.as_mut_ptr() as *mut i8,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }

    Ok(vec.iter().map(|&b| b & 1 != 0).collect())
}

/// Fraction of pages in a region that are resident in RAM (0.0–1.0).
/// Returns 0.0 on error (safe default: assume not resident).
pub fn resident_ratio(ptr: *const u8, len: usize) -> f64 {
    match check_resident(ptr, len) {
        Ok(pages) if !pages.is_empty() => {
            let resident = pages.iter().filter(|&&b| b).count();
            resident as f64 / pages.len() as f64
        }
        _ => 0.0,
    }
}

// ── Allocator helper ─────────────────────────────────────────────────────────

/// Allocate a page-aligned buffer using mmap. This gives us direct control
/// over the pages — we can mlock, madvise, and mincore them precisely.
///
/// Returns a raw pointer and the actual allocated length (rounded up to page size).
/// Caller must `free_aligned()` when done.
pub fn alloc_aligned(size: usize) -> io::Result<(*mut u8, usize)> {
    let ps = page_size();
    let aligned_size = (size + ps - 1) & !(ps - 1);
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            aligned_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    Ok((ptr as *mut u8, aligned_size))
}

/// Free a buffer allocated with `alloc_aligned()`.
///
/// # Safety
/// `ptr` must have been returned by `alloc_aligned()` with the same `len`.
pub unsafe fn free_aligned(ptr: *mut u8, len: usize) {
    libc::munmap(ptr as *mut libc::c_void, len);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_size_is_16k_on_arm64() {
        let ps = page_size();
        // Apple Silicon uses 16KB pages
        assert_eq!(
            ps, 16384,
            "ARM64 macOS page size should be 16KB, got {}",
            ps
        );
    }

    #[test]
    fn alloc_aligned_and_free() {
        let (ptr, len) = alloc_aligned(1000).expect("mmap should work");
        assert!(!ptr.is_null());
        assert!(len >= 1000);
        assert_eq!(len % page_size(), 0, "should be page-aligned size");

        // Write to it — should not segfault
        unsafe {
            std::ptr::write_bytes(ptr, 0xAB, len);
        }

        unsafe { free_aligned(ptr, len) };
    }

    #[test]
    fn mlock_small_buffer() {
        let (ptr, len) = alloc_aligned(page_size()).unwrap();
        // Touch the page first
        unsafe { std::ptr::write_bytes(ptr, 0x42, len) };

        // Pin it
        let result = pin_memory(ptr, len);
        assert!(
            result.is_ok(),
            "mlock should succeed for 1 page: {:?}",
            result.err()
        );

        // Verify it's resident via mincore
        let resident = check_resident(ptr, len).unwrap();
        assert!(resident.iter().all(|&r| r), "locked pages must be resident");

        // Unpin
        unpin_memory(ptr, len).expect("munlock should work");

        unsafe { free_aligned(ptr, len) };
    }

    #[test]
    fn mincore_on_fresh_mmap() {
        let (ptr, len) = alloc_aligned(page_size() * 4).unwrap();

        // Fresh mmap pages may or may not be resident (macOS pre-faults some)
        let resident = check_resident(ptr, len);
        assert!(resident.is_ok(), "mincore should not fail on mmap'd memory");

        // Touch all pages
        unsafe { std::ptr::write_bytes(ptr, 0xFF, len) };

        // Now all should be resident
        let resident = check_resident(ptr, len).unwrap();
        let ratio = resident.iter().filter(|&&r| r).count() as f64 / resident.len() as f64;
        assert!(
            ratio > 0.5,
            "touched pages should mostly be resident, ratio={}",
            ratio,
        );

        unsafe { free_aligned(ptr, len) };
    }

    #[test]
    fn madvise_willneed_and_dontneed() {
        let (ptr, len) = alloc_aligned(page_size() * 8).unwrap();
        unsafe { std::ptr::write_bytes(ptr, 0xCC, len) };

        // Hint: we'll need it
        assert!(hint_willneed(ptr, len).is_ok());

        // Hint: we don't need it anymore
        assert!(hint_dontneed(ptr, len).is_ok());

        unsafe { free_aligned(ptr, len) };
    }

    #[test]
    fn resident_ratio_returns_sane_value() {
        let (ptr, len) = alloc_aligned(page_size() * 16).unwrap();
        unsafe { std::ptr::write_bytes(ptr, 0xDD, len) };

        let ratio = resident_ratio(ptr, len);
        assert!(
            (0.0..=1.0).contains(&ratio),
            "ratio must be in [0,1], got {}",
            ratio,
        );

        unsafe { free_aligned(ptr, len) };
    }

    #[test]
    fn mlock_too_large_may_fail() {
        // Try to lock 8 GB — should fail (exceeds rlimit for non-root)
        let huge = 8 * 1024 * 1024 * 1024_usize;
        let result = alloc_aligned(huge);
        // mmap might succeed (virtual address space) but mlock will fail
        if let Ok((ptr, len)) = result {
            let lock_result = pin_memory(ptr, len);
            // Don't assert failure — root might have unlimited mlock
            if lock_result.is_err() {
                // Expected for non-root
            } else {
                unpin_memory(ptr, len).ok();
            }
            unsafe { free_aligned(ptr, len) };
        }
        // If mmap fails, that's also acceptable
    }
}
