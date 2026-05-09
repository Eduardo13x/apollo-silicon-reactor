//! File Cache Warmer — pre-read executables into the buffer cache.
//!
//! When Apollo's Markov chain predicts an app switch, the predicted process
//! is unfrozen (SIGCONT) and its jetsam priority is boosted.  However, if
//! the process has been frozen for a while under memory pressure, its
//! **file-backed code pages** (executable + shared libraries) may have been
//! evicted from the buffer cache.  When the process resumes, each code page
//! triggers a page fault → SSD read → ~100μs per page, adding hundreds of
//! milliseconds of perceived latency on app switch.
//!
//! This module eliminates that latency by pre-reading the app's executable
//! file into the buffer cache before the user switches.  The kernel caches
//! the file pages, so when the process accesses them, they're already in RAM.
//!
//! On macOS, `fcntl(F_RDADVISE)` provides asynchronous advisory prefetch
//! that tells the kernel to read a file range without blocking the caller.
//!
//! ## References
//!
//! - Cao et al. 1994, "Implementation and Performance of Integrated
//!   Application-Controlled File Caching, Prefetching, and Disk Scheduling"
//! - Chang & Gibson 1999, "Automatic I/O Hint Generation through
//!   Speculative Execution"
//! - Apple `fcntl(2)` man page — `F_RDADVISE` advisory read

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

/// Maximum executable file size to pre-read (64 MB).
/// Larger binaries (e.g. Electron apps) are capped to avoid flooding the cache.
const MAX_PREFETCH_BYTES: u64 = 64 * 1024 * 1024;

/// Minimum interval between warming the same PID (avoid redundant reads).
const MIN_WARM_INTERVAL_SECS: u64 = 30;

/// macOS `F_RDADVISE` command for `fcntl`.
#[cfg(target_os = "macos")]
const F_RDADVISE: libc::c_int = 44;

/// `struct radvisory` for `fcntl(F_RDADVISE)`.
#[cfg(target_os = "macos")]
#[repr(C)]
struct Radvisory {
    ra_offset: libc::off_t,
    ra_count: libc::c_int,
}

/// Get the executable path for a PID via `proc_pidpath()`.
fn pid_exe_path(pid: u32) -> Option<PathBuf> {
    let mut buf = vec![0u8; libc::PATH_MAX as usize];
    let ret = unsafe {
        // proc_pidpath is declared in mach_qos.rs but we call it directly
        // from libproc (linked by default on macOS).
        extern "C" {
            fn proc_pidpath(pid: libc::pid_t, buffer: *mut u8, buffersize: u32) -> libc::c_int;
        }
        proc_pidpath(pid as libc::pid_t, buf.as_mut_ptr(), buf.len() as u32)
    };
    if ret <= 0 {
        return None;
    }
    let path = std::str::from_utf8(&buf[..ret as usize]).ok()?;
    Some(PathBuf::from(path))
}

/// Pre-read a file into the buffer cache using `fcntl(F_RDADVISE)`.
///
/// This is asynchronous — the kernel starts prefetching in the background
/// and the call returns immediately.  If `F_RDADVISE` fails (e.g. on a
/// non-macOS platform), falls back to a sequential `read()`.
#[cfg(target_os = "macos")]
fn advise_prefetch(path: &std::path::Path, max_bytes: u64) -> std::io::Result<u64> {
    use std::os::unix::io::AsRawFd;

    let file = std::fs::File::open(path)?;
    let file_size = file.metadata()?.len();
    let prefetch_size = file_size.min(max_bytes);

    // Try F_RDADVISE first (async, non-blocking).
    let advisory = Radvisory {
        ra_offset: 0,
        ra_count: prefetch_size as libc::c_int,
    };
    let ret = unsafe { libc::fcntl(file.as_raw_fd(), F_RDADVISE, &advisory as *const Radvisory) };

    if ret == 0 {
        return Ok(prefetch_size);
    }

    // Fallback: sequential read (synchronous but still warms cache).
    // Read in 256 KB chunks to avoid large stack allocations.
    let mut total_read = 0u64;
    let mut buf = vec![0u8; 256 * 1024];
    use std::io::Read;
    let mut reader = std::io::BufReader::new(file);
    while total_read < prefetch_size {
        let to_read = ((prefetch_size - total_read) as usize).min(buf.len());
        match reader.read(&mut buf[..to_read]) {
            Ok(0) => break,
            Ok(n) => total_read += n as u64,
            Err(_) => break,
        }
    }
    Ok(total_read)
}

#[cfg(not(target_os = "macos"))]
fn advise_prefetch(_path: &std::path::Path, _max_bytes: u64) -> std::io::Result<u64> {
    Ok(0)
}

/// Tracks recently warmed PIDs to avoid redundant prefetches.
pub struct CacheWarmer {
    /// PID → last warm time.
    last_warmed: HashMap<u32, Instant>,
}

impl CacheWarmer {
    pub fn new() -> Self {
        Self {
            last_warmed: HashMap::new(),
        }
    }

    /// Pre-warm the file cache for a process's executable.
    ///
    /// Returns the number of bytes prefetched, or 0 if skipped/failed.
    /// Skips if the same PID was warmed less than `MIN_WARM_INTERVAL_SECS` ago.
    pub fn warm_pid(&mut self, pid: u32) -> u64 {
        // Dedup: skip if recently warmed.
        let now = Instant::now();
        if let Some(&last) = self.last_warmed.get(&pid) {
            if now.duration_since(last).as_secs() < MIN_WARM_INTERVAL_SECS {
                return 0;
            }
        }

        let Some(exe_path) = pid_exe_path(pid) else {
            return 0;
        };

        match advise_prefetch(&exe_path, MAX_PREFETCH_BYTES) {
            Ok(bytes) => {
                self.last_warmed.insert(pid, now);
                bytes
            }
            Err(_) => 0,
        }
    }

    /// Clean up entries for PIDs that no longer exist.
    /// Call periodically to prevent unbounded growth.
    pub fn gc(&mut self) {
        let now = Instant::now();
        self.last_warmed
            .retain(|_, last| now.duration_since(*last).as_secs() < 300);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pid_exe_path_self() {
        // Our own PID should have a valid exe path.
        let pid = std::process::id();
        let path = pid_exe_path(pid);
        assert!(path.is_some(), "should resolve own exe path");
        let path = path.unwrap();
        assert!(path.exists(), "exe path should exist: {:?}", path);
    }

    #[test]
    fn pid_exe_path_invalid() {
        // PID 0 (kernel) should fail gracefully.
        let path = pid_exe_path(0);
        // May or may not return a path depending on permissions.
        // The important thing is it doesn't crash.
        let _ = path;
    }

    #[test]
    fn warmer_dedup() {
        let mut warmer = CacheWarmer::new();

        // First warm should work (returns bytes or 0 if no such PID).
        let pid = std::process::id();
        let first = warmer.warm_pid(pid);

        // Second warm within MIN_WARM_INTERVAL_SECS should be skipped.
        let second = warmer.warm_pid(pid);
        assert_eq!(second, 0, "should skip duplicate warm");

        // But first should have done something (our own exe exists).
        assert!(first > 0, "should have prefetched own exe");
    }

    #[test]
    fn warmer_gc() {
        let mut warmer = CacheWarmer::new();
        warmer.last_warmed.insert(99999, Instant::now());
        warmer.gc();
        // Recent entry should survive GC.
        assert!(warmer.last_warmed.contains_key(&99999));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn advise_prefetch_self_exe() {
        let pid = std::process::id();
        let path = pid_exe_path(pid).unwrap();
        let bytes = advise_prefetch(&path, MAX_PREFETCH_BYTES).unwrap();
        assert!(bytes > 0, "should prefetch at least some bytes");
    }
}
