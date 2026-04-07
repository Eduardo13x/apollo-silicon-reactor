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
