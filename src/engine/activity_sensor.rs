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
    { return HashSet::new(); }

    #[cfg(target_os = "macos")]
    {
        // Run IOKit call in a thread with timeout to prevent indefinite blocking.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(pids_with_assertions_inner());
        });
        return rx.recv_timeout(std::time::Duration::from_millis(500))
            .unwrap_or_default();
    }
}

#[cfg(target_os = "macos")]
fn pids_with_assertions_inner() -> HashSet<u32> {
    {
        extern "C" {
            fn IOPMCopyAssertionsByProcess(
                assertions_by_pid: *mut *const std::ffi::c_void,
            ) -> i32;
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

/// Combined: returns PIDs that should NOT be frozen because they are
/// actively doing work — either via a power assertion or via active children.
///
/// `processes` should come from a `sysinfo::System` that is already refreshed.
pub fn active_pids(processes: &HashMap<sysinfo::Pid, sysinfo::Process>) -> HashSet<u32> {
    let mut result = pids_with_assertions();
    result.extend(pids_with_active_children(processes, 10.0));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assertions_returns_hashset() {
        // Just check it doesn't panic and returns something reasonable.
        let pids = pids_with_assertions();
        // pmset should always be available on macOS; result may be empty if
        // no assertions are active, but the call must succeed.
        let _ = pids;
    }

    #[test]
    fn active_children_empty_system() {
        let processes = HashMap::new();
        let result = pids_with_active_children(&processes, 10.0);
        assert!(result.is_empty());
    }
}
