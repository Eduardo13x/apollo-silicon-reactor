//! Activity sensor: detects processes that are actively doing work,
//! even when they are not in the foreground.
//!
//! Two signals:
//! 1. **Power assertions** — processes that told the OS "don't interrupt me"
//!    (audio playback, downloads, background tasks, etc.).
//! 2. **Active children** — processes with children consuming significant CPU
//!    (terminals running builds, scripts, long-running commands, etc.).

use std::collections::{HashMap, HashSet};

/// Return the set of PIDs that hold any active power assertion via IOKit.
/// These processes are actively doing something the user or system considers
/// important — freezing them would break it.
///
/// Cost: ~0.1ms (direct IOKit call, no subprocess). Cache the result per freeze cycle.
/// Note: IOPMCopyAssertionsByProcess can block indefinitely as root under kernel
/// contention. We run it in a thread with a 500ms timeout to avoid hanging the daemon.
pub fn pids_with_assertions() -> HashSet<u32> {
    #[cfg(not(target_os = "macos"))]
    {
        return HashSet::new();
    }

    #[cfg(target_os = "macos")]
    {
        // Run IOKit call in a thread with timeout to prevent indefinite blocking.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(pids_with_assertions_inner());
        });
        return rx
            .recv_timeout(std::time::Duration::from_millis(500))
            .unwrap_or_default();
    }
}

#[cfg(target_os = "macos")]
fn pids_with_assertions_inner() -> HashSet<u32> {
    {
        extern "C" {
            fn IOPMCopyAssertionsByProcess(assertions_by_pid: *mut *const std::ffi::c_void) -> i32;
            fn CFArrayGetCount(array: *const std::ffi::c_void) -> i64;
            #[allow(dead_code)]
            fn CFArrayGetValueAtIndex(
                array: *const std::ffi::c_void,
                idx: i64,
            ) -> *const std::ffi::c_void;
            fn CFDictionaryGetCount(dict: *const std::ffi::c_void) -> i64;
            fn CFDictionaryGetKeysAndValues(
                dict: *const std::ffi::c_void,
                keys: *mut *const std::ffi::c_void,
                values: *mut *const std::ffi::c_void,
            );
            fn CFGetTypeID(cf: *const std::ffi::c_void) -> u64;
            fn CFNumberGetTypeID() -> u64;
            fn CFNumberGetValue(
                number: *const std::ffi::c_void,
                the_type: i64,
                value_ptr: *mut std::ffi::c_void,
            ) -> bool;
            fn CFArrayGetTypeID() -> u64;
            fn CFRelease(cf: *const std::ffi::c_void);
        }

        const K_CF_NUMBER_SINT32_TYPE: i64 = 3;
        // IOPMCopyAssertionsByProcess returns a CFDictionary where:
        //   keys = CFNumber (PID)
        //   values = CFArray of assertion dictionaries
        // Any PID with a non-empty assertion array is "active".

        let mut dict_ref: *const std::ffi::c_void = std::ptr::null();
        let mut pids = HashSet::new();

        unsafe {
            let kr = IOPMCopyAssertionsByProcess(&mut dict_ref);
            if kr != 0 || dict_ref.is_null() {
                return pids;
            }

            let count = CFDictionaryGetCount(dict_ref);
            if count <= 0 {
                CFRelease(dict_ref);
                return pids;
            }

            let mut keys = vec![std::ptr::null(); count as usize];
            let mut values = vec![std::ptr::null(); count as usize];
            CFDictionaryGetKeysAndValues(dict_ref, keys.as_mut_ptr(), values.as_mut_ptr());

            for i in 0..count as usize {
                let key = keys[i];
                let value = values[i];

                // key is CFNumber (PID)
                if key.is_null() || CFGetTypeID(key) != CFNumberGetTypeID() {
                    continue;
                }
                // value is CFArray of assertions — if non-empty, PID is active
                if value.is_null() || CFGetTypeID(value) != CFArrayGetTypeID() {
                    continue;
                }
                if CFArrayGetCount(value) == 0 {
                    continue;
                }

                let mut pid: i32 = 0;
                if CFNumberGetValue(key, K_CF_NUMBER_SINT32_TYPE, &mut pid as *mut _ as *mut _) {
                    if pid > 0 {
                        pids.insert(pid as u32);
                    }
                }
            }

            CFRelease(dict_ref);
        }

        pids
    }
}

/// Return the set of parent PIDs whose children are collectively consuming
/// at least `threshold` percent CPU. A terminal running a build will show
/// up here even if the terminal process itself is idle.
///
/// Uses the already-refreshed `sysinfo::System` — no extra syscalls.
pub fn pids_with_active_children(
    processes: &HashMap<sysinfo::Pid, sysinfo::Process>,
    threshold: f32,
) -> HashSet<u32> {
    let mut child_cpu: HashMap<u32, f32> = HashMap::new();

    for (pid, proc_info) in processes {
        if let Some(parent) = proc_info.parent() {
            let entry = child_cpu.entry(parent.as_u32()).or_insert(0.0);
            *entry += proc_info.cpu_usage();
            let _ = pid; // suppress unused warning
        }
    }

    child_cpu
        .into_iter()
        .filter(|(_, total_cpu)| *total_cpu >= threshold)
        .map(|(pid, _)| pid)
        .collect()
}

/// Returns PIDs with at least `min_sockets` open socket file descriptors.
///
/// Detects processes actively doing network I/O (downloads via curl/wget/brew,
/// streaming apps, etc.) that may not hold a power assertion but should not
/// be frozen mid-transfer.
///
/// Uses `proc_pidinfo(PROC_PIDLISTFDS)` — ~2µs per process, no subprocess.
/// Called once per freeze cycle against candidate PIDs only.
pub fn pids_with_open_sockets(candidate_pids: &[u32], min_sockets: usize) -> HashSet<u32> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (candidate_pids, min_sockets);
        return HashSet::new();
    }

    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        #[derive(Copy, Clone, Default)]
        struct ProcFdInfo {
            proc_fd: i32,
            proc_fdtype: u32,
        }
        const PROC_PIDLISTFDS: i32 = 1;
        const PROX_FDTYPE_SOCKET: u32 = 2;

        extern "C" {
            fn proc_pidinfo(
                pid: i32,
                flavor: i32,
                arg: u64,
                buffer: *mut libc::c_void,
                buffersize: i32,
            ) -> i32;
        }

        let mut result = HashSet::new();
        for &pid in candidate_pids {
            // First call with null buf returns bytes needed.
            let needed =
                unsafe { proc_pidinfo(pid as i32, PROC_PIDLISTFDS, 0, std::ptr::null_mut(), 0) };
            if needed <= 0 {
                continue;
            }
            let capacity = (needed as usize / std::mem::size_of::<ProcFdInfo>()) + 8;
            let mut buf: Vec<ProcFdInfo> = vec![ProcFdInfo::default(); capacity];
            let written = unsafe {
                proc_pidinfo(
                    pid as i32,
                    PROC_PIDLISTFDS,
                    0,
                    buf.as_mut_ptr() as *mut libc::c_void,
                    (capacity * std::mem::size_of::<ProcFdInfo>()) as i32,
                )
            };
            if written <= 0 {
                continue;
            }
            let actual = written as usize / std::mem::size_of::<ProcFdInfo>();
            let socket_count = buf[..actual]
                .iter()
                .filter(|f| f.proc_fdtype == PROX_FDTYPE_SOCKET)
                .count();
            if socket_count >= min_sockets {
                result.insert(pid);
            }
        }
        result
    }
}

/// Combined: returns PIDs that should NOT be frozen because they are
/// actively doing work — via a power assertion, active children, or open
/// network sockets (downloads, streaming, background sync).
///
/// `processes` should come from a `sysinfo::System` that is already refreshed.
pub fn active_pids(processes: &HashMap<sysinfo::Pid, sysinfo::Process>) -> HashSet<u32> {
    let mut result = pids_with_assertions();
    result.extend(pids_with_active_children(processes, 10.0));
    // Detect network-active processes not covered by power assertions:
    // curl, wget, brew, rsync, cloud sync daemons, streaming apps, etc.
    // Threshold: >= 3 open sockets (1-2 sockets is normal for most daemons).
    let candidates: Vec<u32> = processes.keys().map(|p| p.as_u32()).collect();
    result.extend(pids_with_open_sockets(&candidates, 3));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assertions_returns_hashset() {
        // Just check it doesn't panic and returns something reasonable.
        let pids = pids_with_assertions();
        // IOPMCopyAssertionsByProcess may be empty when no assertions active.
        // The call must succeed without panicking.
        let _ = pids;
    }

    #[test]
    fn active_children_empty_system() {
        let processes = HashMap::new();
        let result = pids_with_active_children(&processes, 10.0);
        assert!(result.is_empty());
    }

    /// pids_with_open_sockets with empty candidates returns empty set.
    #[test]
    fn open_sockets_empty_candidates() {
        let result = pids_with_open_sockets(&[], 3);
        assert!(result.is_empty(), "no candidates → no results");
    }

    /// pids_with_open_sockets with a non-existent PID returns empty set.
    #[test]
    fn open_sockets_nonexistent_pid() {
        // PID 999999 almost certainly doesn't exist.
        let result = pids_with_open_sockets(&[999_999], 1);
        // Either returns empty (proc_pidinfo fails) or a valid set — must not panic.
        let _ = result;
    }

    /// pids_with_active_children: only parent PIDs with enough child CPU are returned.
    #[test]
    fn active_children_threshold_filtering() {
        // We can't easily create real sysinfo::Process objects, but we can
        // verify the contract: empty input → empty output at any threshold.
        let processes = HashMap::new();
        // Various thresholds — all return empty for empty input.
        for threshold in [0.0f32, 5.0, 10.0, 50.0, 100.0] {
            let result = pids_with_active_children(&processes, threshold);
            assert!(
                result.is_empty(),
                "threshold={}: empty system → empty result",
                threshold
            );
        }
    }

    /// active_pids with empty process map returns a set (possibly from IOKit assertions).
    #[test]
    fn active_pids_no_crash_empty_system() {
        let processes = HashMap::new();
        // Must not panic even with empty process map.
        let result = active_pids(&processes);
        // Result may be non-empty if the test runner holds power assertions (rare).
        let _ = result;
    }

    /// pids_with_open_sockets: min_sockets=usize::MAX returns empty (no process has that many).
    #[test]
    fn open_sockets_unreachable_threshold() {
        // Get current process PID — it definitely has some FDs open.
        let my_pid = std::process::id();
        // But requiring usize::MAX sockets should always return empty.
        let result = pids_with_open_sockets(&[my_pid], usize::MAX);
        assert!(result.is_empty(), "no process has usize::MAX sockets");
    }
}
