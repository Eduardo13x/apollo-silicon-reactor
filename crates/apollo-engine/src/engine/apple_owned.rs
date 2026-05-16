//! Detect Apple/system/kernel-owned processes WITHOUT a hardcoded list.
//!
//! Hardcoded protection lists (`safety::is_protected_name`,
//! `decide_actions::INTERACTIVE_APPS`) drift behind every macOS release
//! that ships new daemons. This module classifies processes by *origin* —
//! SIP path prefix or Apple code-signing authority — so any new Apple
//! daemon is auto-protected without code change.
//!
//! Two layers:
//!
//! 1. **Path prefix** (free, deterministic): executable in
//!    `/System/Library`, `/usr/libexec`, `/usr/sbin`, `/sbin`, `/usr/bin`,
//!    `/Library/Apple/` → Apple. Covers ~95% of Apple-owned processes
//!    via SIP-protected directories. Zero subprocess.
//!
//! 2. **Code signature** (canonical, ~10-20ms): spawn `codesign -dv` and
//!    grep `Authority=` for `Apple Inc.` / `Software Signing` / `Apple Root`
//!    chain. Result cached per binary path so repeated lookups are free.
//!    Cache invalidates implicitly when the path changes (new binary →
//!    different cache key).
//!
//! kernel_task (pid 0) is short-circuited true — it has no executable path.
//!
//! Usage: call `is_apple_owned(pid)` before deciding to send any signal /
//! memorystatus pressure / SIGSTOP to a process. If true, skip — the
//! process belongs to the OS and breaking it breaks the user.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
extern "C" {
    fn proc_pidpath(pid: i32, buffer: *mut u8, buffersize: u32) -> i32;
}

/// Codesign verdict + timestamp. The `validated_at` is checked against
/// `CACHE_TTL` so a binary replacement (e.g. macOS update writes a new
/// executable to the same path) is re-validated within the TTL window
/// instead of carrying yesterday's verdict for tomorrow's binary.
/// Closes the split-brain risk NotebookLM peer-review flagged 2026-05-10
/// against IdentityCache (which invalidates on NOTE_EXIT / start-time
/// change). Path-keyed caches cannot use start_sec — TTL is the
/// canonical refresh primitive.
#[derive(Clone, Copy)]
struct CodesignEntry {
    is_apple: bool,
    validated_at: Instant,
}

/// Re-validate codesign verdict at most once every CACHE_TTL.
/// 6h is short enough to bound exposure to "binary replaced" edge cases,
/// long enough that hot-path queries are O(1) cache hits in steady state.
const CACHE_TTL: Duration = Duration::from_secs(6 * 3600);
/// Hard cap on cache size. Apple Silicon Macs run ~100-300 unique binary
/// paths in steady state; 1024 leaves comfortable headroom.
const MAX_CACHE_ENTRIES: usize = 1024;

static CODESIGN_CACHE: Mutex<Option<HashMap<String, CodesignEntry>>> = Mutex::new(None);

/// True when the process is owned by Apple (system, framework, or kernel).
///
/// Returns `false` only for clearly third-party / user binaries. Returns
/// `true` (conservative) on lookup failure: a missing path is more likely
/// a kernel/short-lived system task than a user app.
#[cfg(target_os = "macos")]
pub fn is_apple_owned(pid: u32) -> bool {
    if pid == 0 {
        return true; // kernel_task
    }
    let path = match resolve_pid_path(pid) {
        Some(p) => p,
        None => return true, // safer to protect on failure
    };

    // Layer 1: SIP / Apple path prefix.
    if is_apple_path(&path) {
        return true;
    }

    // Layer 2: codesign Authority chain (cached).
    is_apple_signed_cached(&path)
}

#[cfg(not(target_os = "macos"))]
pub fn is_apple_owned(_pid: u32) -> bool {
    false
}

/// Path-prefix check. Public so callers that already have a path string
/// can skip the syscall round-trip.
pub fn is_apple_path(path: &str) -> bool {
    path.starts_with("/System/")
        || path.starts_with("/usr/libexec/")
        || path.starts_with("/usr/sbin/")
        || path.starts_with("/usr/bin/")
        || path.starts_with("/sbin/")
        || path.starts_with("/Library/Apple/")
        || path.starts_with("/Library/DriverExtensions/")
        || path.starts_with("/Library/PrivilegedHelperTools/com.apple.")
}

#[cfg(target_os = "macos")]
pub(crate) fn resolve_pid_path(pid: u32) -> Option<String> {
    let mut buf = [0u8; 1024];
    let ret = unsafe { proc_pidpath(pid as i32, buf.as_mut_ptr(), buf.len() as u32) };
    if ret <= 0 || ret as usize > buf.len() {
        return None;
    }
    let bytes = &buf[..ret as usize];
    std::str::from_utf8(bytes).ok().map(|s| s.to_string())
}

#[cfg(target_os = "macos")]
fn is_apple_signed_cached(path: &str) -> bool {
    if let Ok(mut guard) = CODESIGN_CACHE.lock() {
        let cache = guard.get_or_insert_with(HashMap::new);
        let now = Instant::now();
        // Hit only if entry exists AND is within TTL.
        if let Some(entry) = cache.get(path) {
            if now.duration_since(entry.validated_at) < CACHE_TTL {
                return entry.is_apple;
            }
        }
        let v = codesign_authority_is_apple(path);
        // Hard cap: evict oldest entry when at capacity. O(n) on overflow
        // (rare: bounded by unique-binary-count), O(1) on hit.
        if cache.len() >= MAX_CACHE_ENTRIES && !cache.contains_key(path) {
            if let Some(oldest_key) = cache
                .iter()
                .min_by_key(|(_, e)| e.validated_at)
                .map(|(k, _)| k.clone())
            {
                cache.remove(&oldest_key);
            }
        }
        cache.insert(
            path.to_string(),
            CodesignEntry {
                is_apple: v,
                validated_at: now,
            },
        );
        return v;
    }
    // Mutex poisoned: skip cache, query live (still correct, just slower).
    codesign_authority_is_apple(path)
}

#[cfg(target_os = "macos")]
fn codesign_authority_is_apple(path: &str) -> bool {
    use std::process::Command;
    let out = match Command::new("/usr/bin/codesign")
        .args(["-dv", "--verbose=2", path])
        .output()
    {
        Ok(o) => o,
        Err(_) => return false,
    };
    // codesign writes signature info to stderr.
    let combined = String::from_utf8_lossy(&out.stderr).to_string()
        + &String::from_utf8_lossy(&out.stdout);
    // Apple's signing chain: Authority=Apple Code Signing Certification Authority,
    // Authority=Apple Root CA, or Authority=Software Signing for first-party
    // built-ins. TeamIdentifier=not set (Apple uses the special value).
    combined.contains("Authority=Apple Inc.")
        || combined.contains("Authority=Apple Root CA")
        || combined.contains("Authority=Software Signing")
}

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
fn resolve_pid_path(_pid: u32) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_prefix_classifies_system_binaries() {
        assert!(is_apple_path("/System/Library/Frameworks/Foo.framework/Foo"));
        assert!(is_apple_path("/usr/libexec/coreaudiod"));
        assert!(is_apple_path("/usr/sbin/spindump"));
        assert!(is_apple_path("/sbin/launchd"));
        assert!(is_apple_path("/usr/bin/codesign"));
        assert!(is_apple_path("/Library/Apple/System/Library/Extensions/Foo.kext"));
    }

    #[test]
    fn path_prefix_rejects_user_binaries() {
        assert!(!is_apple_path("/Applications/Brave Browser.app/Contents/MacOS/Brave"));
        assert!(!is_apple_path("/usr/local/bin/cargo"));
        assert!(!is_apple_path("/opt/homebrew/bin/git"));
        assert!(!is_apple_path("/Users/eduardo/projects/foo/target/release/foo"));
    }

    #[test]
    fn kernel_task_is_apple() {
        // pid 0 short-circuits without touching the syscalls.
        assert!(is_apple_owned(0));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn launchd_is_apple_owned() {
        // pid 1 = launchd, the canonical Apple system daemon.
        assert!(is_apple_owned(1));
    }
}
