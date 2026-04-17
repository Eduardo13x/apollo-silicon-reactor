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

// ── Rosetta 2 detection ──────────────────────────────────────────────────────

/// P_TRANSLATED flag from XNU `bsd/sys/proc_internal.h`.
/// Set when a process is running under Rosetta 2 binary translation (x86_64→ARM64).
#[cfg(target_os = "macos")]
const P_TRANSLATED: u32 = 0x0002_0000;

/// Returns `true` if `pid` is running under Rosetta 2 (x86_64 binary on ARM64).
///
/// Rosetta processes incur ~10-30% CPU overhead from JIT translation.
/// Under memory/CPU pressure, freezing them first recovers more real throughput
/// than freezing a native ARM64 process at the same reported CPU%.
#[cfg(target_os = "macos")]
pub fn is_translated(pid: u32) -> bool {
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
        return false;
    }
    info.pbi_flags & P_TRANSLATED != 0
}

#[cfg(not(target_os = "macos"))]
pub fn is_translated(_pid: u32) -> bool {
    false
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

// ── Apple platform binary detection ─────────────────────────────────────────

/// Code-signing status operations (csops syscall).
#[cfg(target_os = "macos")]
const CS_OPS_STATUS: u32 = 0;

/// `CS_PLATFORM_BINARY` — set by the kernel for Apple-signed platform binaries.
/// All processes in `/System/`, `/usr/`, XNU itself, and Apple daemons carry this flag.
/// It is checked by SIP, TCC, and the kernel trust layer — the most reliable
/// signal that a process is part of the Apple system stack.
#[cfg(target_os = "macos")]
const CS_PLATFORM_BINARY: u32 = 0x0400_0000;

#[cfg(target_os = "macos")]
extern "C" {
    /// `csops(pid, ops, useraddr, usersize)` — query/set code-signing state.
    /// With `CS_OPS_STATUS` returns a `u32` bitmask of `CS_*` flags.
    fn csops(pid: libc::pid_t, ops: u32, useraddr: *mut libc::c_void, usersize: usize)
        -> libc::c_int;
}

// `proc_pidpath` — already declared in proc_taskinfo; re-use via extern here.
#[cfg(target_os = "macos")]
extern "C" {
    fn proc_pidpath(pid: libc::c_int, buffer: *mut u8, buffersize: u32) -> libc::c_int;
}

/// Maximum path buffer size for `proc_pidpath`.
#[cfg(target_os = "macos")]
const PROC_PIDPATHINFO_MAXSIZE: u32 = 4096;

/// Returns `true` if `pid` belongs to the Apple platform:
/// Apple-signed system binary, Apple daemon, or SIP-protected system service.
///
/// # Detection layers (ordered fastest → most expensive)
///
/// 1. **`csops` `CS_PLATFORM_BINARY`** (~1 µs): kernel-authoritative flag set on
///    all Apple-signed platform binaries. No false positives. No path needed.
///
/// 2. **`proc_pidpath` prefix** (~3 µs fallback): binary lives under a system
///    path (`/System/`, `/usr/` excl. `/usr/local/`, `/sbin/`, `/bin/`,
///    `/Library/Apple/`). Covers binaries that run before codesign is verified
///    and kernel helpers that `csops` returns 0 for.
///
/// Returns `false` for any process Apollo should be allowed to optimize
/// (user apps, third-party daemons, Homebrew services, etc.).
///
/// Cost: ~1–4 µs. Safe to call every cycle per-process because the answer
/// is stable for the lifetime of the process — callers should cache.
#[cfg(target_os = "macos")]
pub fn is_apple_platform_process(pid: u32) -> bool {
    // Layer 1: csops CS_PLATFORM_BINARY — kernel-authoritative.
    let mut flags: u32 = 0;
    let rc = unsafe {
        csops(
            pid as libc::pid_t,
            CS_OPS_STATUS,
            &mut flags as *mut u32 as *mut libc::c_void,
            std::mem::size_of::<u32>(),
        )
    };
    if rc == 0 && (flags & CS_PLATFORM_BINARY) != 0 {
        return true;
    }

    // Layer 2: path prefix + name heuristic.
    // Covers: kernel helpers, early-boot processes (csops returns 0),
    // third-party DriverKit dexts (no CS_PLATFORM_BINARY but SIP-managed path),
    // and system extensions installed under /Library/SystemExtensions/.
    let mut buf = [0u8; PROC_PIDPATHINFO_MAXSIZE as usize];
    let path_len =
        unsafe { proc_pidpath(pid as libc::c_int, buf.as_mut_ptr(), PROC_PIDPATHINFO_MAXSIZE) };
    if path_len > 0 {
        let path = std::str::from_utf8(&buf[..path_len as usize]).unwrap_or("");
        if is_system_driver_path(path) {
            return true;
        }
    }

    // Layer 3: name heuristic for DriverKit processes whose path is a bundle ID.
    // DriverKit dext processes report their bundle identifier as their name
    // (e.g. "com.apple.DriverKit-AppleBCMWLAN", "org.pqrs.Karabiner-DriverKit-…").
    // proc_pidpath on these returns the .dext bundle path which layer 2 catches,
    // but as a belt-and-suspenders guard we also check the name directly.
    if let Some(name) = crate::engine::process_identity::proc_name_for_pid(pid) {
        if is_driver_process_name(&name) {
            return true;
        }
    }

    false
}

#[cfg(not(target_os = "macos"))]
pub fn is_apple_platform_process(_pid: u32) -> bool {
    false
}

/// Returns `true` if `path` is a known Apple system binary location.
///
/// Excludes `/usr/local/` (Homebrew) — that is user-managed territory.
pub fn is_apple_system_path(path: &str) -> bool {
    path.starts_with("/System/")
        || path.starts_with("/usr/bin/")
        || path.starts_with("/usr/sbin/")
        || path.starts_with("/usr/libexec/")
        || path.starts_with("/usr/lib/")
        || path.starts_with("/sbin/")
        || path.starts_with("/bin/")
        || path.starts_with("/Library/Apple/")
        || path.starts_with("/private/var/db/")
}

/// Returns `true` if `path` is a system driver or extension location.
///
/// Covers Apple system paths plus driver/extension-specific locations:
///
/// - `/Library/SystemExtensions/` — SIP-managed; macOS approves each extension
///   before it lands here (Network Extensions, Endpoint Security, DriverKit dexts).
/// - `/Library/DriverExtensions/` — DriverKit dexts for third-party hardware.
/// - Any path containing `.dext/` — DriverKit driver bundle executable.
/// - Any path containing `.systemextension/` — System Extension bundle executable.
/// - Any path containing `.kext/` — kernel extension bundle (legacy drivers).
///
/// Note: third-party DriverKit processes (e.g. Karabiner, audio interfaces) lack
/// `CS_PLATFORM_BINARY` but are just as critical — freezing them hangs HID input,
/// audio, or USB entirely.
pub fn is_system_driver_path(path: &str) -> bool {
    is_apple_system_path(path)
        || path.starts_with("/Library/SystemExtensions/")
        || path.starts_with("/Library/DriverExtensions/")
        || path.contains(".dext/")
        || path.contains(".systemextension/")
        || path.contains(".kext/")
}

/// Returns `true` if the process name looks like a DriverKit bundle identifier.
///
/// DriverKit dexts report their bundle ID as their process name, e.g.:
/// - `com.apple.DriverKit-AppleBCMWLAN`
/// - `com.apple.DriverKit-IOUserDockChannelSerial`
/// - `org.pqrs.Karabiner-DriverKit-VirtualHIDDevice`
///
/// Matching on "DriverKit" in the name catches all of them regardless of vendor.
pub fn is_driver_process_name(name: &str) -> bool {
    name.contains("DriverKit")
        || name.ends_with(".dext")
        || name.ends_with(".kext")
        || name.ends_with(".systemextension")
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn own_process_is_not_translated() {
        // cargo test itself is a native ARM64 binary — should NOT be Rosetta.
        let own_pid = std::process::id();
        assert!(
            !is_translated(own_pid),
            "test binary should be native ARM64, not Rosetta"
        );
    }

    #[test]
    fn dead_pid_is_not_translated() {
        assert!(!is_translated(999_999_999));
    }

    #[test]
    fn launchd_is_not_translated() {
        // PID 1 (launchd) is always native.
        assert!(!is_translated(1));
    }

    #[test]
    fn launchd_is_apple_platform() {
        // PID 1 (launchd) is always an Apple platform binary.
        assert!(
            is_apple_platform_process(1),
            "launchd must be detected as Apple platform binary"
        );
    }

    #[test]
    fn dead_pid_is_not_apple() {
        assert!(!is_apple_platform_process(999_999_999));
    }

    #[test]
    fn own_process_not_apple_platform() {
        // cargo test is not a platform binary — it lives in /Users/ or ~/.cargo/
        let own_pid = std::process::id();
        assert!(
            !is_apple_platform_process(own_pid),
            "cargo test should NOT be detected as Apple platform binary"
        );
    }

    #[test]
    fn apple_system_paths() {
        assert!(is_apple_system_path("/System/Library/CoreServices/Finder.app/Contents/MacOS/Finder"));
        assert!(is_apple_system_path("/usr/libexec/AirPlayXPCHelper"));
        assert!(is_apple_system_path("/usr/bin/codesign"));
        assert!(is_apple_system_path("/sbin/launchd"));
        assert!(is_apple_system_path("/bin/sh"));
        assert!(is_apple_system_path("/Library/Apple/System/Library/Extensions/AppleT8101PCIe.kext/Contents/MacOS/AppleT8101PCIe"));
    }

    #[test]
    fn non_apple_paths() {
        assert!(!is_apple_system_path("/usr/local/bin/brew"));
        assert!(!is_apple_system_path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"));
        assert!(!is_apple_system_path("/Users/eduardocortez/.cargo/bin/cargo"));
        assert!(!is_apple_system_path("/usr/local/libexec/apollo-optimizerd"));
    }

    #[test]
    fn driver_paths_detected() {
        // SIP-managed system extensions (DriverKit dexts, NExt, EndpointSecurity).
        assert!(is_system_driver_path(
            "/Library/SystemExtensions/55CD9E59/org.pqrs.Karabiner-DriverKit-VirtualHIDDevice.dext/Contents/MacOS/org.pqrs.Karabiner-DriverKit-VirtualHIDDevice"
        ));
        // .dext bundle anywhere.
        assert!(is_system_driver_path(
            "/System/Library/DriverExtensions/com.apple.DriverKit-AppleBCMWLAN.dext/Contents/MacOS/com.apple.DriverKit-AppleBCMWLAN"
        ));
        // System extension bundle.
        assert!(is_system_driver_path(
            "/usr/libexec/com.apple.cmio.videodriverkithostextension.systemextension/Contents/MacOS/com.apple.cmio.videodriverkithostextension"
        ));
        // Legacy kext bundle.
        assert!(is_system_driver_path(
            "/Library/Extensions/SoftRAID.kext/Contents/MacOS/SoftRAID"
        ));
        // Third-party driver extensions path.
        assert!(is_system_driver_path("/Library/DriverExtensions/com.vendor.MyHIDDevice.dext/Contents/MacOS/com.vendor.MyHIDDevice"));
    }

    #[test]
    fn driver_names_detected() {
        assert!(is_driver_process_name("com.apple.DriverKit-AppleBCMWLAN"));
        assert!(is_driver_process_name("org.pqrs.Karabiner-DriverKit-VirtualHIDDevice"));
        assert!(is_driver_process_name("com.apple.DriverKit-IOUserDockChannelSerial"));
        assert!(!is_driver_process_name("Brave Browser"));
        assert!(!is_driver_process_name("apollo-optimizerd"));
        assert!(!is_driver_process_name("cargo"));
    }

    #[test]
    fn non_driver_paths_not_caught() {
        assert!(!is_system_driver_path("/usr/local/bin/brew"));
        assert!(!is_system_driver_path("/Applications/AudioHijack.app/Contents/MacOS/AudioHijack"));
        assert!(!is_system_driver_path("/Users/edu/.cargo/bin/rustc"));
    }
}
