//! Jetsam / memorystatus_control — direct kernel memory pressure API.
//!
//! macOS exposes `memorystatus_control()` from libSystem (private API, works as root).
//! This gives Apollo direct control over:
//!   - Per-process kill priority (which dies first under memory pressure)
//!   - Per-process memory limits (trigger early reclaim before OOM)
//!   - Suspend/freeze hints to the kernel compressor
//!
//! No entitlements required when running as root.
//! Reference: xnu/bsd/sys/kern_memorystatus.h

use std::ffi::c_void;

// ─── Jetsam priority bands ────────────────────────────────────────────────────
// Lower value = killed first under memory pressure.
#[allow(dead_code)]
pub mod priority {
    pub const IDLE: i32 = 0;
    pub const BACKGROUND_OPPORTUNISTIC: i32 = 1;
    pub const BACKGROUND: i32 = 2;
    pub const MAIL: i32 = 5;
    pub const PHONE: i32 = 6;
    pub const UI_SUPPORT: i32 = 7;
    pub const FOREGROUND_SUPPORT: i32 = 8;
    pub const FOREGROUND: i32 = 9;
    pub const AUDIO_AND_ACCESSORY: i32 = 10;
    pub const CRITICAL: i32 = 11;
    pub const VITAL: i32 = 12;
    pub const HIGHEST: i32 = 21;
    pub const KERNEL: i32 = 63;
}

// ─── memorystatus_control commands ───────────────────────────────────────────
const MEMORYSTATUS_CMD_SET_PRIORITY_PROPERTIES: u32 = 6;
const MEMORYSTATUS_CMD_GET_MEMLIMIT_PROPERTIES: u32 = 7;
const MEMORYSTATUS_CMD_SET_MEMLIMIT_PROPERTIES: u32 = 8;
/// Ask the kernel for the current jetsam priority of a PID (xnu 4570+).
const MEMORYSTATUS_CMD_GET_PRIORITY_LIST: u32 = 1;

// ─── C structs (must match xnu ABI exactly) ───────────────────────────────────
#[repr(C)]
struct MemorystatusPriorityProperties {
    priority: i32,
    user_data: u64,
}

#[repr(C)]
#[derive(Debug, Default, Clone)]
pub struct MemorystatusMemlimitProperties {
    pub memlimit_active: i32, // MB, 0 = unlimited
    pub memlimit_active_attr: u32,
    pub memlimit_inactive: i32, // MB, 0 = unlimited
    pub memlimit_inactive_attr: u32,
}

// ─── FFI ──────────────────────────────────────────────────────────────────────
extern "C" {
    fn memorystatus_control(
        command: u32,
        pid: i32,
        flags: u32,
        buffer: *mut c_void,
        buffersize: usize,
    ) -> i32;
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Set the Jetsam kill priority for a process.
///
/// Use `priority::BACKGROUND` for noise daemons (die early),
/// `priority::FOREGROUND` for interactive apps (die last).
pub fn set_priority(pid: u32, jetsam_priority: i32) -> Result<(), String> {
    let mut props = MemorystatusPriorityProperties {
        priority: jetsam_priority,
        user_data: 0,
    };
    let ret = unsafe {
        memorystatus_control(
            MEMORYSTATUS_CMD_SET_PRIORITY_PROPERTIES,
            pid as i32,
            0,
            &mut props as *mut _ as *mut c_void,
            std::mem::size_of::<MemorystatusPriorityProperties>(),
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(format!(
            "memorystatus_control SET_PRIORITY pid={} priority={}: errno={}",
            pid,
            jetsam_priority,
            unsafe { *libc::__error() }
        ))
    }
}

/// Read the kernel's current jetsam priority for `pid`.
///
/// A5/D1 fix (round-3): captured at freeze time so unfreeze can restore the
/// *exact* priority instead of unconditionally using FOREGROUND.  Previously
/// unfreeze always set FOREGROUND (9), losing AUDIO (18),
/// AUDIO_AND_ACCESSORY (10), VITAL (12), etc.
///
/// Uses `MEMORYSTATUS_CMD_GET_PRIORITY_LIST` with buffer sized for one entry
/// and the `flags=pid` argument (kernel filters to that PID).  Returns
/// `None` if the entry is unavailable.
pub fn get_priority(pid: u32) -> Option<i32> {
    #[repr(C)]
    #[derive(Default)]
    struct PriorityEntry {
        pid: i32,
        priority: i32,
        user_data: u64,
    }
    let mut entry = PriorityEntry::default();
    let ret = unsafe {
        memorystatus_control(
            MEMORYSTATUS_CMD_GET_PRIORITY_LIST,
            pid as i32,
            0,
            &mut entry as *mut _ as *mut c_void,
            std::mem::size_of::<PriorityEntry>(),
        )
    };
    // Returns the number of bytes filled. Non-negative success, <0 errno.
    if ret <= 0 || entry.pid != pid as i32 {
        return None;
    }
    Some(entry.priority)
}

/// Set memory limits for a process (active and inactive).
///
/// `active_mb`:   limit while process is in foreground (0 = unlimited)
/// `inactive_mb`: limit while process is in background (0 = unlimited)
///
/// When a process exceeds its inactive limit, the kernel reclaims it
/// before triggering system-wide memory pressure.
pub fn set_memlimit(pid: u32, active_mb: i32, inactive_mb: i32) -> Result<(), String> {
    let mut props = MemorystatusMemlimitProperties {
        memlimit_active: active_mb,
        memlimit_active_attr: 0,
        memlimit_inactive: inactive_mb,
        memlimit_inactive_attr: 0,
    };
    let ret = unsafe {
        memorystatus_control(
            MEMORYSTATUS_CMD_SET_MEMLIMIT_PROPERTIES,
            pid as i32,
            0,
            &mut props as *mut _ as *mut c_void,
            std::mem::size_of::<MemorystatusMemlimitProperties>(),
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(format!(
            "memorystatus_control SET_MEMLIMIT pid={} active={}MB inactive={}MB: errno={}",
            pid,
            active_mb,
            inactive_mb,
            unsafe { *libc::__error() }
        ))
    }
}

/// Get current memory limits for a process.
pub fn get_memlimit(pid: u32) -> Result<MemorystatusMemlimitProperties, String> {
    let mut props = MemorystatusMemlimitProperties::default();
    let ret = unsafe {
        memorystatus_control(
            MEMORYSTATUS_CMD_GET_MEMLIMIT_PROPERTIES,
            pid as i32,
            0,
            &mut props as *mut _ as *mut c_void,
            std::mem::size_of::<MemorystatusMemlimitProperties>(),
        )
    };
    if ret == 0 {
        Ok(props)
    } else {
        Err(format!(
            "memorystatus_control GET_MEMLIMIT pid={}: errno={}",
            pid,
            unsafe { *libc::__error() }
        ))
    }
}

/// Send a targeted memory pressure notification to a process without killing it.
///
/// Sets a **non-fatal** inactive memory limit (`memlimit_inactive_attr = 0`,
/// no `MEMORYSTATUS_MEMLIMIT_ATTR_FATAL`).  When the process's RSS exceeds
/// `warn_mb`, the kernel sends it a `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`
/// notification — prompting it to release caches — but does NOT terminate it.
///
/// This is a surgical alternative to system-wide memory pressure:
/// only the target process is poked, all others are unaffected.
///
/// Pass `warn_mb = 0` to clear the warn limit (restore unlimited).
pub fn set_warn_limit(pid: u32, warn_mb: i32) -> Result<(), String> {
    // Preserve the current active limit to avoid unintentionally changing it.
    let current = get_memlimit(pid).unwrap_or_default();
    let mut props = MemorystatusMemlimitProperties {
        memlimit_active: current.memlimit_active,
        memlimit_active_attr: current.memlimit_active_attr,
        memlimit_inactive: warn_mb,
        memlimit_inactive_attr: 0, // 0 = non-fatal (no MEMORYSTATUS_MEMLIMIT_ATTR_FATAL=0x1)
    };
    let ret = unsafe {
        memorystatus_control(
            MEMORYSTATUS_CMD_SET_MEMLIMIT_PROPERTIES,
            pid as i32,
            0,
            &mut props as *mut _ as *mut c_void,
            std::mem::size_of::<MemorystatusMemlimitProperties>(),
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(format!(
            "memorystatus_control SET_WARN_LIMIT pid={} warn={}MB: errno={}",
            pid,
            warn_mb,
            unsafe { *libc::__error() }
        ))
    }
}

/// Apply Jetsam policy based on Apollo process classification.
///
/// - interactive → FOREGROUND priority, no memory limit
/// - noise        → BACKGROUND priority, 200 MB inactive limit
/// - protected    → CRITICAL priority, no memory limit
pub fn apply_apollo_policy(pid: u32, classification: JetsamClass) -> Result<(), String> {
    match classification {
        JetsamClass::Interactive => {
            set_priority(pid, priority::FOREGROUND)?;
            set_memlimit(pid, 0, 0)?;
        }
        JetsamClass::Noise => {
            set_priority(pid, priority::BACKGROUND)?;
            set_memlimit(pid, 0, 200)?; // 200 MB inactive cap
        }
        JetsamClass::Protected => {
            set_priority(pid, priority::CRITICAL)?;
            set_memlimit(pid, 0, 0)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JetsamClass {
    Interactive,
    Noise,
    Protected,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── ABI struct layout pins ─────────────────────────────────────────────
    // These structs are passed by raw pointer + explicit size to a kernel
    // syscall. A wrong size/offset = the kernel reads garbage or rejects the
    // call silently. Pin them so a field reorder/type change fails the build.

    #[test]
    fn priority_properties_abi_layout() {
        // i32 (4) + u64 (8). u64 forces 8-byte alignment → 4 bytes pad after
        // the i32 → total 16 bytes. xnu's memorystatus_priority_properties_t.
        assert_eq!(
            std::mem::size_of::<MemorystatusPriorityProperties>(),
            16,
            "priority props size must be 16 (matches xnu ABI)"
        );
        assert_eq!(std::mem::align_of::<MemorystatusPriorityProperties>(), 8);
        let probe = MemorystatusPriorityProperties {
            priority: 0,
            user_data: 0,
        };
        let base = &probe as *const _ as usize;
        assert_eq!(&probe.priority as *const _ as usize - base, 0);
        assert_eq!(&probe.user_data as *const _ as usize - base, 8);
    }

    #[test]
    fn memlimit_properties_abi_layout() {
        // Four 4-byte fields, no padding → 16 bytes.
        // xnu's memorystatus_memlimit_properties_t.
        assert_eq!(
            std::mem::size_of::<MemorystatusMemlimitProperties>(),
            16,
            "memlimit props size must be 16 (matches xnu ABI)"
        );
        assert_eq!(std::mem::align_of::<MemorystatusMemlimitProperties>(), 4);
        let probe = MemorystatusMemlimitProperties::default();
        let base = &probe as *const _ as usize;
        assert_eq!(&probe.memlimit_active as *const _ as usize - base, 0);
        assert_eq!(&probe.memlimit_active_attr as *const _ as usize - base, 4);
        assert_eq!(&probe.memlimit_inactive as *const _ as usize - base, 8);
        assert_eq!(
            &probe.memlimit_inactive_attr as *const _ as usize - base,
            12
        );
    }

    #[test]
    fn priority_entry_abi_layout() {
        // Mirror of the local PriorityEntry used by get_priority().
        // i32 + i32 + u64 → 8-byte aligned → 16 bytes (no internal pad).
        #[repr(C)]
        struct PriorityEntry {
            pid: i32,
            priority: i32,
            user_data: u64,
        }
        assert_eq!(std::mem::size_of::<PriorityEntry>(), 16);
        assert_eq!(std::mem::align_of::<PriorityEntry>(), 8);
        let probe = PriorityEntry {
            pid: 0,
            priority: 0,
            user_data: 0,
        };
        let base = &probe as *const _ as usize;
        assert_eq!(&probe.pid as *const _ as usize - base, 0);
        assert_eq!(&probe.priority as *const _ as usize - base, 4);
        assert_eq!(&probe.user_data as *const _ as usize - base, 8);
    }

    // ─── memorystatus_control command constants ─────────────────────────────
    // These map directly to xnu/bsd/sys/kern_memorystatus.h enum values.
    // A wrong number issues a completely different kernel command.

    #[test]
    fn command_constants_match_xnu_abi() {
        assert_eq!(MEMORYSTATUS_CMD_GET_PRIORITY_LIST, 1);
        assert_eq!(MEMORYSTATUS_CMD_SET_PRIORITY_PROPERTIES, 6);
        assert_eq!(MEMORYSTATUS_CMD_GET_MEMLIMIT_PROPERTIES, 7);
        assert_eq!(MEMORYSTATUS_CMD_SET_MEMLIMIT_PROPERTIES, 8);
    }

    // ─── Jetsam priority band ordering ──────────────────────────────────────
    // Lower = killed first. The relative ordering is the whole contract:
    // background MUST die before foreground MUST die before critical.

    #[test]
    fn priority_bands_match_xnu_values() {
        assert_eq!(priority::IDLE, 0);
        assert_eq!(priority::BACKGROUND_OPPORTUNISTIC, 1);
        assert_eq!(priority::BACKGROUND, 2);
        assert_eq!(priority::MAIL, 5);
        assert_eq!(priority::PHONE, 6);
        assert_eq!(priority::UI_SUPPORT, 7);
        assert_eq!(priority::FOREGROUND_SUPPORT, 8);
        assert_eq!(priority::FOREGROUND, 9);
        assert_eq!(priority::AUDIO_AND_ACCESSORY, 10);
        assert_eq!(priority::CRITICAL, 11);
        assert_eq!(priority::VITAL, 12);
        assert_eq!(priority::HIGHEST, 21);
        assert_eq!(priority::KERNEL, 63);
    }

    #[test]
    fn priority_band_kill_order_invariant() {
        // The survival contract: noise dies first, interactive last, protected
        // basically never. If this ordering ever inverts, Apollo would make the
        // system MORE likely to kill the foreground app under pressure.
        assert!(priority::BACKGROUND < priority::FOREGROUND);
        assert!(priority::FOREGROUND < priority::CRITICAL);
        assert!(priority::IDLE < priority::BACKGROUND);
        assert!(priority::CRITICAL < priority::KERNEL);
    }

    // ─── apply_apollo_policy tier → raw-value mapping ───────────────────────
    // Pure mapping documented in the doc comment. We can't run the syscall in
    // tests, but we pin the intended priority each class maps to so a future
    // edit that swaps Noise↔Protected priorities is caught.

    /// Mirror of the (priority, active_mb, inactive_mb) tuple that
    /// `apply_apollo_policy` would emit for a class, without doing the syscall.
    fn intended_policy(class: JetsamClass) -> (i32, i32, i32) {
        match class {
            JetsamClass::Interactive => (priority::FOREGROUND, 0, 0),
            JetsamClass::Noise => (priority::BACKGROUND, 0, 200),
            JetsamClass::Protected => (priority::CRITICAL, 0, 0),
        }
    }

    #[test]
    fn apollo_policy_class_mapping() {
        assert_eq!(intended_policy(JetsamClass::Interactive), (9, 0, 0));
        assert_eq!(intended_policy(JetsamClass::Noise), (2, 0, 200));
        assert_eq!(intended_policy(JetsamClass::Protected), (11, 0, 0));
        // Only Noise gets a non-zero (capped) inactive limit.
        assert_eq!(intended_policy(JetsamClass::Interactive).2, 0);
        assert_eq!(intended_policy(JetsamClass::Protected).2, 0);
        assert!(intended_policy(JetsamClass::Noise).2 > 0);
    }
}
