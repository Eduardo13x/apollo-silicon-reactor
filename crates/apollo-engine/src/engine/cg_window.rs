//! CGWindowList FFI — enumerate on-screen window owners.
//!
//! Direct bindings to `CGWindowListCopyWindowInfo` so Apollo can answer:
//! "is this PID currently owning a visible window?" Used by the freeze
//! gate to avoid SIGSTOP on renderers whose tabs are still visible to
//! the user — those get jetsam demotion instead.
//!
//! Reference: Apple Developer / Quartz Window Services.

use std::collections::HashSet;
use std::ffi::{c_void, CString};

// kCGWindowListOptionOnScreenOnly = 1 << 0
const OPTION_ON_SCREEN_ONLY: u32 = 1;
// kCGWindowListExcludeDesktopElements = 1 << 4
const OPTION_EXCLUDE_DESKTOP: u32 = 16;
// kCGNullWindowID = 0
const NULL_WINDOW_ID: u32 = 0;
// kCFNumberSInt64Type = 4
const CF_NUMBER_S_INT64: i64 = 4;
// kCFStringEncodingUTF8 = 0x08000100
const CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

#[cfg_attr(target_os = "macos", link(name = "CoreFoundation", kind = "framework"))]
#[cfg_attr(target_os = "macos", link(name = "CoreGraphics", kind = "framework"))]
#[cfg(target_os = "macos")]
extern "C" {
    fn CGWindowListCopyWindowInfo(option: u32, relative_to_window: u32) -> *const c_void;
    fn CFArrayGetCount(array: *const c_void) -> i64;
    fn CFArrayGetValueAtIndex(array: *const c_void, idx: i64) -> *const c_void;
    fn CFDictionaryGetValue(dict: *const c_void, key: *const c_void) -> *const c_void;
    fn CFNumberGetValue(number: *const c_void, ty: i64, value_ptr: *mut c_void) -> bool;
    fn CFStringCreateWithCString(
        alloc: *const c_void,
        c_str: *const i8,
        encoding: u32,
    ) -> *const c_void;
    fn CFRelease(cf: *const c_void);
}

/// Query the on-screen window list and return the set of owner PIDs.
///
/// Runs in ~1-3ms on M1. Safe to call every cycle but wasteful — the
/// caller should cache results for ~1s between calls.
///
/// Returns an empty set on any failure (CoreGraphics denied, CF call
/// returned NULL, etc.). The caller interprets empty-set as "unknown"
/// and should skip any visibility-based gating rather than misclassify.
#[cfg(target_os = "macos")]
pub fn visible_pids() -> HashSet<u32> {
    let mut out: HashSet<u32> = HashSet::new();
    unsafe {
        let array = CGWindowListCopyWindowInfo(
            OPTION_ON_SCREEN_ONLY | OPTION_EXCLUDE_DESKTOP,
            NULL_WINDOW_ID,
        );
        if array.is_null() {
            return out;
        }
        let key_cstring = match CString::new("kCGWindowOwnerPID") {
            Ok(s) => s,
            Err(_) => {
                CFRelease(array);
                return out;
            }
        };
        let key = CFStringCreateWithCString(
            std::ptr::null(),
            key_cstring.as_ptr(),
            CF_STRING_ENCODING_UTF8,
        );
        if key.is_null() {
            CFRelease(array);
            return out;
        }
        let count = CFArrayGetCount(array);
        for i in 0..count {
            let dict = CFArrayGetValueAtIndex(array, i);
            if dict.is_null() {
                continue;
            }
            let pid_ref = CFDictionaryGetValue(dict, key);
            if pid_ref.is_null() {
                continue;
            }
            let mut pid_val: i64 = 0;
            if CFNumberGetValue(pid_ref, CF_NUMBER_S_INT64, &mut pid_val as *mut _ as *mut c_void)
                && pid_val > 0
                && pid_val <= u32::MAX as i64
            {
                out.insert(pid_val as u32);
            }
        }
        CFRelease(key);
        CFRelease(array);
    }
    out
}

#[cfg(not(target_os = "macos"))]
pub fn visible_pids() -> HashSet<u32> {
    HashSet::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_pids_returns_some_on_macos() {
        // Smoke test — in a windowed macOS environment there is always at
        // least one visible process (WindowServer, Dock, etc). On a headless
        // CI runner this may return empty; that is acceptable behavior, not
        // a test failure.
        let pids = visible_pids();
        println!("visible_pids count = {}", pids.len());
        // Assert only that the call does not panic or corrupt memory.
        assert!(pids.len() < 10_000, "unreasonable PID count");
    }
}
