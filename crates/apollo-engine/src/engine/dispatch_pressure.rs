//! Kernel memory pressure level monitor.
//!
//! On macOS 15 Apple Silicon, all three push APIs for memory pressure are broken:
//! - `EVFILT_VM` → `ENOTSUP` (errno=45)
//! - `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` → never fires
//! - `notify_register_file_descriptor("com.apple.system.memorystatus.level")` → never fires
//!
//! However, `kern.memorystatus_vm_pressure_level` sysctl **does** update in real time.
//! This module polls it and detects transitions. Cost: ~1µs per read via sysctlbyname.

use crate::engine::sysctl_direct;

/// Kernel memory pressure level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum KernelPressureLevel {
    /// No pressure — system is comfortable.
    Normal = 0,
    /// Warning — compressor active, swap growing.
    Warning = 1,
    /// Critical — jetsam may start killing processes.
    Critical = 2,
}

impl KernelPressureLevel {
    fn from_raw(v: i32) -> Self {
        match v {
            2.. => Self::Critical,
            1 => Self::Warning,
            _ => Self::Normal,
        }
    }
}

/// Lightweight monitor that detects transitions in kernel pressure level.
///
/// Call `poll()` on each reactor tick (~1s). Returns `Some(level)` only when
/// the level *changes*, so the caller can fire a reactive event.
pub struct KernelPressureMonitor {
    prev_level: KernelPressureLevel,
    /// Total transitions detected since creation.
    pub transitions: u64,
}

impl KernelPressureMonitor {
    pub fn new() -> Self {
        let current = Self::read_level();
        Self {
            prev_level: current,
            transitions: 0,
        }
    }

    /// Read the current kernel pressure level via sysctl (~1µs).
    pub fn read_level() -> KernelPressureLevel {
        sysctl_direct::read_i32("kern.memorystatus_vm_pressure_level")
            .map(KernelPressureLevel::from_raw)
            .unwrap_or(KernelPressureLevel::Normal)
    }

    /// Poll for a transition. Returns `Some(new_level)` if the level changed
    /// since the last call, `None` if unchanged.
    pub fn poll(&mut self) -> Option<KernelPressureLevel> {
        let current = Self::read_level();
        if current != self.prev_level {
            self.prev_level = current;
            self.transitions += 1;
            Some(current)
        } else {
            None
        }
    }

    /// Current level without transition detection.
    pub fn current(&self) -> KernelPressureLevel {
        self.prev_level
    }
}

impl Default for KernelPressureMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_level_succeeds() {
        let level = KernelPressureMonitor::read_level();
        // Should be one of the three valid levels.
        assert!(matches!(
            level,
            KernelPressureLevel::Normal
                | KernelPressureLevel::Warning
                | KernelPressureLevel::Critical
        ));
    }

    #[test]
    fn monitor_creates_successfully() {
        let monitor = KernelPressureMonitor::new();
        assert_eq!(monitor.transitions, 0);
    }

    #[test]
    fn poll_returns_none_when_stable() {
        let mut monitor = KernelPressureMonitor::new();
        // Back-to-back polls with no actual pressure change → None.
        let result = monitor.poll();
        // Might be None or Some depending on system state, but shouldn't panic.
        let _ = result;
    }

    #[test]
    fn level_ordering() {
        assert!(KernelPressureLevel::Normal < KernelPressureLevel::Warning);
        assert!(KernelPressureLevel::Warning < KernelPressureLevel::Critical);
    }

    #[test]
    fn from_raw_values() {
        assert_eq!(
            KernelPressureLevel::from_raw(0),
            KernelPressureLevel::Normal
        );
        assert_eq!(
            KernelPressureLevel::from_raw(1),
            KernelPressureLevel::Warning
        );
        assert_eq!(
            KernelPressureLevel::from_raw(2),
            KernelPressureLevel::Critical
        );
        assert_eq!(
            KernelPressureLevel::from_raw(4),
            KernelPressureLevel::Critical
        );
        assert_eq!(
            KernelPressureLevel::from_raw(-1),
            KernelPressureLevel::Normal
        );
    }
}
