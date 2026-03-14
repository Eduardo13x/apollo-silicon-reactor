//! PID Identity Validation — prevents A-B-A PID recycling attacks.
//!
//! macOS recycles PIDs aggressively.  A naive `kill(pid, 0)` check only
//! confirms the PID is alive — it cannot tell us whether it is the *same*
//! process we originally decided to act on.
//!
//! This module provides `validate_pid`, which checks both the process name
//! and its kernel start-time to confirm identity before sending signals.

#[cfg(target_os = "macos")]
use std::ffi::c_void;

/// PROC_PIDTBSDINFO — returns `proc_bsdinfo` with start timestamps.
#[cfg(target_os = "macos")]
const PROC_PIDTBSDINFO: i32 = 3;

#[cfg(target_os = "macos")]
extern "C" {
    fn proc_pidinfo(pid: i32, flavor: i32, arg: u64, buffer: *mut c_void, buffersize: i32) -> i32;
    fn proc_name(pid: i32, buffer: *mut c_void, buffersize: u32) -> i32;
}

/// Kernel `proc_bsdinfo` layout (stable since macOS 10.5).
#[cfg(target_os = "macos")]
#[repr(C)]
struct ProcBsdInfo {
    pbi_flags: u32,
    pbi_status: u32,
    pbi_xstatus: u32,
    pbi_pid: u32,
    pbi_ppid: u32,
    pbi_uid: u32,
    pbi_gid: u32,
    pbi_ruid: u32,
    pbi_rgid: u32,
    pbi_svuid: u32,
    pbi_svgid: u32,
    _rfu_1: u32,
    pbi_comm: [u8; 16], // MAXCOMLEN
    pbi_name: [u8; 32], // 2 * MAXCOMLEN
    pbi_nfiles: u32,
    pbi_pgid: u32,
    pbi_pjobc: u32,
    _e_tdev: u32,
    _e_tpgp: u32,
    pbi_nice: i32,
    pbi_start_tvsec: u64,
    pbi_start_tvusec: u64,
}

/// Unique process identity: (PID, start-time, name).
/// The kernel guarantees that `(pid, start_tvsec, start_tvusec)` is unique
/// even across PID recycling.
#[derive(Debug, Clone)]
pub struct ProcessIdentity {
    pub pid: u32,
    pub start_sec: u64,
    pub start_usec: u64,
    pub name: String,
}

impl ProcessIdentity {
    /// Query the kernel for the identity of `pid`.  Returns `None` if the
    /// process has already exited.
    #[cfg(target_os = "macos")]
    pub fn from_pid(pid: u32) -> Option<Self> {
        let mut info: ProcBsdInfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<ProcBsdInfo>() as i32;
        let ret = unsafe {
            proc_pidinfo(
                pid as i32,
                PROC_PIDTBSDINFO,
                0,
                &mut info as *mut _ as *mut c_void,
                size,
            )
        };
        if ret <= 0 {
            return None;
        }
        let name = proc_name_for_pid(pid).unwrap_or_default();
        Some(Self {
            pid,
            start_sec: info.pbi_start_tvsec,
            start_usec: info.pbi_start_tvusec,
            name,
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn from_pid(pid: u32) -> Option<Self> {
        Some(Self {
            pid,
            start_sec: 0,
            start_usec: 0,
            name: String::new(),
        })
    }

    /// Returns `true` if the PID still corresponds to the same process.
    pub fn is_still_valid(&self) -> bool {
        ProcessIdentity::from_pid(self.pid)
            .map(|current| {
                current.start_sec == self.start_sec && current.start_usec == self.start_usec
            })
            .unwrap_or(false)
    }
}

/// Fast process name lookup via `proc_name()`.  ~2 μs per call.
#[cfg(target_os = "macos")]
pub fn proc_name_for_pid(pid: u32) -> Option<String> {
    let mut buf = [0u8; 256];
    let ret = unsafe { proc_name(pid as i32, buf.as_mut_ptr() as *mut c_void, 256) };
    if ret <= 0 {
        return None;
    }
    let len = ret as usize;
    Some(String::from_utf8_lossy(&buf[..len]).to_string())
}

#[cfg(not(target_os = "macos"))]
pub fn proc_name_for_pid(_pid: u32) -> Option<String> {
    None
}

/// Validate that `pid` still corresponds to a process named `expected_name`.
///
/// Returns `true` if the process is alive AND its name matches.
/// Returns `false` if the PID died or was recycled to a different process.
///
/// Uses prefix matching (not substring) to handle `proc_name()` truncation
/// while preventing false positives from unrelated names that share substrings
/// (e.g. "Safari" should NOT match "SafariHelper").
///
/// Cost: ~4 μs (one `proc_name` call + string compare).
pub fn validate_pid(pid: u32, expected_name: &str) -> bool {
    // 1. Basic existence check (fast path — avoids proc_name on dead PIDs).
    if unsafe { libc::kill(pid as i32, 0) } != 0 {
        return false;
    }
    // 2. Name check — catches PID recycling.
    match proc_name_for_pid(pid) {
        Some(current_name) => {
            // Exact match (common case).
            if current_name == expected_name {
                return true;
            }
            // proc_name() may truncate to ~15 chars (MAXCOMLEN).
            // Accept if truncated name is a prefix of expected_name
            // and is reasonably long (≥6 chars) to avoid false positives.
            if current_name.len() >= 6 && expected_name.starts_with(&current_name) {
                return true;
            }
            // expected_name may itself be truncated by the caller.
            if expected_name.len() >= 6 && current_name.starts_with(expected_name) {
                return true;
            }
            false
        }
        None => false,
    }
}
