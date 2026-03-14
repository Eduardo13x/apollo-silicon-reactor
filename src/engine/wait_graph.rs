//! Wait-Graph Introspection — deadlock prevention for frozen processes.
//!
//! Before freezing a process (SIGSTOP), we must check whether any *running*
//! process holds a resource that a currently-frozen process is waiting on.
//! If we freeze the holder, the waiter can never wake up → deadlock.
//!
//! Strategy:
//!   1. Query thread states of frozen PIDs via `proc_pidinfo(PROC_PIDLISTTHREADS)`.
//!   2. For each frozen thread in TH_WAIT state, check if the wait channel
//!      overlaps with any thread in the freeze-candidate set.
//!   3. If overlap detected → veto the freeze OR unfreeze the waiter first.
//!
//! Limitations:
//!   - Mach wait channels are opaque kernel addresses; we cannot always map
//!     them to a specific resource.  We use a conservative heuristic: if a
//!     frozen process has threads in TH_WAIT, and we're about to freeze
//!     another process with shared port / IPC patterns, we skip the freeze.
//!   - Requires root for `proc_pidinfo` on other processes.

use std::collections::HashSet;

#[cfg(target_os = "macos")]
use std::ffi::c_void;

#[cfg(target_os = "macos")]
const PROC_PIDLISTTHREADS: i32 = 6;

/// Thread states from Mach (osfmk/kern/thread.h).
#[cfg(target_os = "macos")]
const _TH_STATE_RUNNING: i32 = 1;
#[cfg(target_os = "macos")]
const TH_STATE_WAITING: i32 = 2;
#[cfg(target_os = "macos")]
const _TH_STATE_STOPPED: i32 = 3;

#[cfg(target_os = "macos")]
const PROC_PIDTHREADINFO: i32 = 5;

#[cfg(target_os = "macos")]
extern "C" {
    fn proc_pidinfo(pid: i32, flavor: i32, arg: u64, buffer: *mut c_void, buffersize: i32) -> i32;
}

/// Per-thread info from `proc_pidinfo(PROC_PIDTHREADINFO)`.
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone)]
struct ProcThreadInfo {
    pth_user_time: u64,
    pth_system_time: u64,
    pth_cpu_usage: i32,
    pth_policy: i32,
    pth_run_state: i32,
    pth_flags: i32,
    pth_sleep_time: i32,
    pth_curpri: i32,
    pth_priority: i32,
    pth_maxpriority: i32,
    pth_name: [u8; 64],
}

/// Summary of a process's thread wait states.
#[derive(Debug, Clone)]
pub struct ProcessWaitState {
    pub pid: u32,
    pub total_threads: u32,
    pub waiting_threads: u32,
    pub stopped_threads: u32,
}

/// Check whether a frozen process has threads blocked in TH_WAIT state.
///
/// Cost: ~100 μs per process (list threads + query each).
#[cfg(target_os = "macos")]
pub fn query_wait_state(pid: u32) -> Option<ProcessWaitState> {
    unsafe {
        // Step 1: get thread IDs
        let mut thread_ids = [0u64; 256];
        let buf_size = (256 * std::mem::size_of::<u64>()) as i32;
        let ret = proc_pidinfo(
            pid as i32,
            PROC_PIDLISTTHREADS,
            0,
            thread_ids.as_mut_ptr() as *mut c_void,
            buf_size,
        );
        if ret <= 0 {
            return None;
        }
        let thread_count = ret as usize / std::mem::size_of::<u64>();

        let mut waiting = 0u32;
        let mut stopped = 0u32;

        // Step 2: query each thread's state
        for &tid in &thread_ids[..thread_count] {
            let mut info: ProcThreadInfo = std::mem::zeroed();
            let info_size = std::mem::size_of::<ProcThreadInfo>() as i32;
            let ret = proc_pidinfo(
                pid as i32,
                PROC_PIDTHREADINFO,
                tid,
                &mut info as *mut _ as *mut c_void,
                info_size,
            );
            if ret <= 0 {
                continue;
            }
            if info.pth_run_state == TH_STATE_WAITING {
                waiting += 1;
            } else if info.pth_run_state == _TH_STATE_STOPPED {
                stopped += 1;
            }
        }

        Some(ProcessWaitState {
            pid,
            total_threads: thread_count as u32,
            waiting_threads: waiting,
            stopped_threads: stopped,
        })
    }
}

#[cfg(not(target_os = "macos"))]
pub fn query_wait_state(_pid: u32) -> Option<ProcessWaitState> {
    None
}

/// Check whether freezing `candidate_pid` would risk deadlocking any
/// currently-frozen process.
///
/// Heuristic: if any frozen process has threads in TH_WAIT state AND
/// the candidate has threads actively running (potential lock holder),
/// we conservatively veto the freeze.
///
/// Returns `true` if freezing is safe, `false` if it should be skipped.
pub fn is_freeze_safe(candidate_pid: u32, frozen_pids: &HashSet<u32>) -> bool {
    if frozen_pids.is_empty() {
        return true;
    }

    // Check: does the candidate have running threads that might hold locks?
    let candidate_state = match query_wait_state(candidate_pid) {
        Some(s) => s,
        None => return true, // process already exited, safe to skip
    };

    // If the candidate itself is all-waiting, freezing it is low-risk.
    let candidate_active = candidate_state.total_threads
        - candidate_state.waiting_threads
        - candidate_state.stopped_threads;
    if candidate_active == 0 {
        return true;
    }

    // Check frozen processes: if any have waiting threads, the candidate
    // might be their lock holder.
    for &frozen_pid in frozen_pids {
        if let Some(frozen_state) = query_wait_state(frozen_pid) {
            if frozen_state.waiting_threads > 0 {
                // Conservative: a frozen process is waiting, and the candidate
                // has active threads — could be the holder.  Skip the freeze.
                return false;
            }
        }
    }

    true
}

/// Find frozen PIDs that should be unfrozen because they have threads
/// stuck in TH_WAIT (likely waiting on an IPC resource).
///
/// Call this periodically to prevent indefinite stalls.
pub fn find_stuck_frozen(frozen_pids: &HashSet<u32>) -> Vec<u32> {
    let mut stuck = Vec::new();
    for &pid in frozen_pids {
        if let Some(state) = query_wait_state(pid) {
            // If a frozen process has threads in TH_WAIT (not TH_STOPPED),
            // it was likely mid-IPC when frozen — unfreeze to prevent stall.
            if state.waiting_threads > state.total_threads / 2 {
                stuck.push(pid);
            }
        }
    }
    stuck
}
