//! Granular I/O Tiering — 5-level disk bandwidth segmentation.
//!
//! macOS exposes 5 I/O priority levels via `taskpolicy -d`.  Apollo
//! currently uses only tiers 0 (boost) and 4 (throttle).  This module
//! maps ProcessTier to a fine-grained IOTier for better bandwidth
//! allocation between interactive apps and background indexing.

use std::process::Command;

use crate::engine::process_classifier::ProcessTier;

/// Darwin disk I/O priority tiers (mapped to `taskpolicy -d <N>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IOTier {
    /// Foreground interactive: swap paging, active compilation.
    Interactive = 0,
    /// Background visible: apps open but not focused.
    Standard = 1,
    /// Indexing tier: Spotlight, Time Machine, FSEvents.
    Utility = 2,
    /// Throttled: silent daemons, telemetry.
    Throttle = 3,
    /// Passive: completely deferrable, only when SSD is idle.
    Passive = 4,
}

/// Map a process classifier tier to the appropriate I/O priority.
pub fn io_tier_for_process(tier: ProcessTier) -> IOTier {
    match tier {
        ProcessTier::ActiveForeground => IOTier::Interactive,
        ProcessTier::SystemEssential => IOTier::Interactive,
        ProcessTier::BackgroundVisible => IOTier::Standard,
        ProcessTier::AppHelper => IOTier::Standard,
        ProcessTier::SilentDaemon => IOTier::Utility,
        ProcessTier::Stale => IOTier::Passive,
        ProcessTier::ZombieOrphan => IOTier::Passive,
        ProcessTier::Telemetry => IOTier::Throttle,
    }
}

/// Map throttle aggressiveness to an I/O tier.
pub fn io_tier_for_throttle(aggressive: bool) -> IOTier {
    if aggressive {
        IOTier::Passive
    } else {
        IOTier::Throttle
    }
}

/// Apply an I/O tier to a process via `taskpolicy -d`.
/// Best-effort: returns false on failure but does not panic.
pub fn apply_io_tier(pid: u32, tier: IOTier) -> bool {
    let tier_str = match tier {
        IOTier::Interactive => "0",
        IOTier::Standard => "1",
        IOTier::Utility => "2",
        IOTier::Throttle => "3",
        IOTier::Passive => "4",
    };
    Command::new("/usr/sbin/taskpolicy")
        .args(["-d", tier_str, "-p", &pid.to_string()])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Apply an I/O tier via direct Mach syscall through MachQoSManager.
/// Falls back to CLI `taskpolicy -d` when the direct path fails.
pub fn apply_io_tier_direct(
    pid: u32,
    tier: IOTier,
    qos_mgr: Option<&mut crate::engine::mach_qos::MachQoSManager>,
) -> bool {
    if let Some(mgr) = qos_mgr {
        let io_val = tier as i32;
        if mgr.set_io_tier(pid, io_val) {
            return true;
        }
    }
    // Fallback to CLI.
    apply_io_tier(pid, tier)
}
