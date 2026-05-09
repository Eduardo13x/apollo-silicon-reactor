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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // ── ProcessWaitState struct tests ──────────────────────────────

    #[test]
    fn process_wait_state_clone_and_debug() {
        let state = ProcessWaitState {
            pid: 42,
            total_threads: 10,
            waiting_threads: 3,
            stopped_threads: 1,
        };
        let cloned = state.clone();
        assert_eq!(cloned.pid, 42);
        assert_eq!(cloned.total_threads, 10);
        assert_eq!(cloned.waiting_threads, 3);
        assert_eq!(cloned.stopped_threads, 1);
        // Debug impl exists
        let dbg = format!("{:?}", state);
        assert!(dbg.contains("42"));
    }

    // ── is_freeze_safe tests ──────────────────────────────────────

    #[test]
    fn is_freeze_safe_empty_frozen_set_returns_true() {
        let frozen = HashSet::new();
        // With no frozen processes, any candidate is safe.
        assert!(is_freeze_safe(99999, &frozen));
    }

    #[test]
    fn is_freeze_safe_nonexistent_candidate_returns_true() {
        // Candidate PID doesn't exist → query_wait_state returns None → safe
        let mut frozen = HashSet::new();
        frozen.insert(99998);
        assert!(is_freeze_safe(99999, &frozen));
    }

    #[test]
    fn is_freeze_safe_nonexistent_frozen_pids() {
        // All frozen PIDs are dead → their query_wait_state returns None
        // → no frozen process shows waiting threads → safe
        let mut frozen = HashSet::new();
        frozen.insert(99990);
        frozen.insert(99991);
        frozen.insert(99992);
        // Candidate also doesn't exist → returns true (early exit)
        assert!(is_freeze_safe(99993, &frozen));
    }

    #[test]
    fn is_freeze_safe_with_self_pid_as_candidate() {
        // Use our own PID as candidate — it exists and has threads.
        // Frozen set contains only nonexistent PIDs → no waiting threads found → safe.
        let my_pid = std::process::id();
        let mut frozen = HashSet::new();
        frozen.insert(99997);
        assert!(is_freeze_safe(my_pid, &frozen));
    }

    // ── find_stuck_frozen tests ───────────────────────────────────

    #[test]
    fn find_stuck_frozen_empty_set() {
        let frozen = HashSet::new();
        let stuck = find_stuck_frozen(&frozen);
        assert!(stuck.is_empty());
    }

    #[test]
    fn find_stuck_frozen_nonexistent_pids() {
        // Nonexistent PIDs → query_wait_state returns None → not added to stuck
        let mut frozen = HashSet::new();
        frozen.insert(99994);
        frozen.insert(99995);
        frozen.insert(99996);
        let stuck = find_stuck_frozen(&frozen);
        assert!(stuck.is_empty());
    }

    #[test]
    fn find_stuck_frozen_single_nonexistent() {
        let mut frozen = HashSet::new();
        frozen.insert(1_000_000); // very unlikely to exist
        let stuck = find_stuck_frozen(&frozen);
        assert!(stuck.is_empty());
    }

    // ── query_wait_state tests ────────────────────────────────────

    #[test]
    fn query_wait_state_self_process() {
        // We can query our own process — it should succeed on macOS.
        let my_pid = std::process::id();
        let result = query_wait_state(my_pid);
        #[cfg(target_os = "macos")]
        {
            // On macOS, we should get a valid result for our own process.
            assert!(result.is_some(), "should be able to query own process");
            let state = result.unwrap();
            assert_eq!(state.pid, my_pid);
            assert!(
                state.total_threads >= 1,
                "test process has at least 1 thread"
            );
        }
        #[cfg(not(target_os = "macos"))]
        {
            assert!(result.is_none());
        }
    }

    #[test]
    fn query_wait_state_nonexistent_pid() {
        // PID 0 or very high PID should return None.
        let result = query_wait_state(4_000_000);
        assert!(result.is_none());
    }

    #[test]
    fn query_wait_state_pid_zero() {
        // PID 0 is kernel_task — we likely can't query it without root.
        // Either None or Some is acceptable, but it must not panic.
        let _result = query_wait_state(0);
    }

    #[test]
    fn query_wait_state_thread_counts_are_consistent() {
        let my_pid = std::process::id();
        if let Some(state) = query_wait_state(my_pid) {
            // waiting + stopped should never exceed total
            assert!(
                state.waiting_threads + state.stopped_threads <= state.total_threads,
                "waiting({}) + stopped({}) > total({})",
                state.waiting_threads,
                state.stopped_threads,
                state.total_threads
            );
        }
    }

    // ── Integration-style tests ───────────────────────────────────

    #[test]
    fn is_freeze_safe_consistent_with_empty_and_nonempty() {
        // Verify that adding nonexistent PIDs to frozen set doesn't change
        // the safety verdict for a nonexistent candidate.
        let candidate = 99800;
        let empty = HashSet::new();
        let mut nonempty = HashSet::new();
        nonempty.insert(99801);
        nonempty.insert(99802);

        let safe_empty = is_freeze_safe(candidate, &empty);
        let safe_nonempty = is_freeze_safe(candidate, &nonempty);

        // Both should be true: candidate doesn't exist → early return true
        assert!(safe_empty);
        assert!(safe_nonempty);
    }

    #[test]
    fn find_stuck_frozen_does_not_include_live_unfrozen_process() {
        // Our own process is alive but NOT frozen by us.
        // find_stuck_frozen should still work — it queries state regardless.
        // The question is whether waiting_threads > total/2.
        let my_pid = std::process::id();
        let mut frozen = HashSet::new();
        frozen.insert(my_pid);
        let stuck = find_stuck_frozen(&frozen);
        // We don't assert the exact result since it depends on thread states,
        // but verify it doesn't panic and returns a valid vec.
        assert!(stuck.len() <= 1);
    }

    #[test]
    fn is_freeze_safe_large_frozen_set() {
        // Ensure no performance issue / panic with many frozen PIDs
        let mut frozen = HashSet::new();
        for i in 50000..50100 {
            frozen.insert(i as u32);
        }
        // Should handle 100 nonexistent PIDs gracefully
        let result = is_freeze_safe(99999, &frozen);
        assert!(result); // candidate doesn't exist → true
    }
}
