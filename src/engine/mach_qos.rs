//! Mach QoS and Task Policy — M1-native process scheduling
//!
//! On M1/macOS, frequency scaling is not accessible from userspace.
//! The correct lever is the Mach scheduler's Quality of Service (QoS):
//!
//!  QOS_CLASS_USER_INTERACTIVE  → P-Cores (Firestorm), highest throughput
//!  QOS_CLASS_USER_INITIATED    → P-Cores, slightly lower priority
//!  QOS_CLASS_DEFAULT           → scheduler decides
//!  QOS_CLASS_UTILITY           → mix, reduced energy impact flag
//!  QOS_CLASS_BACKGROUND        → E-Cores (Icestorm) ONLY + throttled I/O
//!
//! Additionally, task_policy_set(TASK_CATEGORY_POLICY) can mark an entire
//! process as "background", which forces all its threads to E-Cores.
//!
//! Requirements:
//!   - The daemon must run as root to call task_for_pid() on other processes.
//!   - SIP does NOT block task_policy_set for root daemons.

use std::collections::HashMap;

// ── Low-level FFI ─────────────────────────────────────────────────────────────

/// Subset of Mach constants used here.
mod mach_sys {
    #![allow(non_upper_case_globals, dead_code)]

    pub const KERN_SUCCESS: i32 = 0;
    pub const MACH_PORT_NULL: u32 = 0;

    /// task_category_policy role values
    pub const TASK_UNSPECIFIED: i32 = 0;
    pub const TASK_FOREGROUND_APPLICATION: i32 = 1;
    pub const TASK_BACKGROUND_APPLICATION: i32 = 2;
    pub const TASK_CONTROL_APPLICATION: i32 = 3;
    pub const TASK_GRAPHICS_SERVER: i32 = 4;
    pub const TASK_THROTTLE_APPLICATION: i32 = 5;
    pub const TASK_NONUI_APPLICATION: i32 = 6;

    pub const TASK_CATEGORY_POLICY: i32 = 1;
    pub const TASK_CATEGORY_POLICY_COUNT: u32 = 1;
}

#[cfg(target_os = "macos")]
mod ffi {
    use libc::{c_int, c_uint, pid_t};

    pub type MachPortT = c_uint;
    pub type KernReturnT = c_int;
    pub type TaskPolicyFlavorT = c_int;
    pub type MachMsgTypeNumberT = c_uint;

    #[repr(C)]
    pub struct TaskCategoryPolicy {
        pub role: c_int,
    }

    #[allow(clashing_extern_declarations)]
    extern "C" {
        pub fn mach_task_self() -> MachPortT;
        pub fn task_for_pid(
            target_tport: MachPortT,
            pid: pid_t,
            t: *mut MachPortT,
        ) -> KernReturnT;
        pub fn task_policy_set(
            task: MachPortT,
            flavor: TaskPolicyFlavorT,
            policy_info: *const TaskCategoryPolicy,
            count: MachMsgTypeNumberT,
        ) -> KernReturnT;
    }
}

// ── QoS / scheduling tier ────────────────────────────────────────────────────

/// The tier we want to enforce for a process in the Mach scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingTier {
    /// Interactive foreground — routed to P-Cores (Firestorm).
    Foreground,
    /// Normal background — scheduler decides core assignment.
    Normal,
    /// Throttled background — routed to E-Cores (Icestorm) + reduced I/O.
    Background,
}

/// Result of applying a QoS policy change.
#[derive(Debug, Clone)]
pub struct QoSOutcome {
    pub pid: u32,
    pub tier: SchedulingTier,
    pub success: bool,
    pub error: Option<String>,
}

// ── Manager ───────────────────────────────────────────────────────────────────

/// Cycles to skip a PID after task_for_pid fails (reduces wasted syscalls).
const QOS_FAILURE_BACKOFF_CYCLES: u8 = 30;

pub struct MachQoSManager {
    /// Current tier per PID — we only issue a syscall when it changes.
    current_tier: HashMap<u32, SchedulingTier>,
    /// PIDs where task_for_pid failed → remaining cycles to skip.
    failed_backoff: HashMap<u32, u8>,
}

impl MachQoSManager {
    pub fn new() -> Self {
        Self {
            current_tier: HashMap::new(),
            failed_backoff: HashMap::new(),
        }
    }

    /// Call once per optimization cycle to decrement failure backoff counters.
    /// PIDs whose backoff reaches zero will be retried next time.
    pub fn tick_backoff(&mut self) {
        self.failed_backoff.retain(|_, remaining| {
            *remaining = remaining.saturating_sub(1);
            *remaining > 0
        });
    }

    /// Apply `tier` to `pid`.  Skips the syscall if already at that tier
    /// or if the PID is in the failure backoff window.
    pub fn set_tier(&mut self, pid: u32, tier: SchedulingTier) -> QoSOutcome {
        // Skip PIDs that consistently fail task_for_pid to avoid wasted syscalls.
        if self.failed_backoff.contains_key(&pid) {
            return QoSOutcome {
                pid,
                tier,
                success: true, // Treat as silent skip, not an error
                error: None,
            };
        }

        if self.current_tier.get(&pid) == Some(&tier) {
            return QoSOutcome {
                pid,
                tier,
                success: true,
                error: None,
            };
        }

        let result = self.apply_task_policy(pid, tier);

        if result.success {
            self.current_tier.insert(pid, tier);
        } else {
            // Back off — stop retrying this PID for N cycles
            self.failed_backoff.insert(pid, QOS_FAILURE_BACKOFF_CYCLES);
            self.current_tier.remove(&pid);
        }

        result
    }

    /// Apply QoS changes to many processes in one pass.
    pub fn apply_batch(&mut self, changes: &[(u32, SchedulingTier)]) -> Vec<QoSOutcome> {
        changes
            .iter()
            .map(|(pid, tier)| self.set_tier(*pid, *tier))
            .collect()
    }

    /// Remove tracking for a PID (process has exited).
    pub fn remove(&mut self, pid: u32) {
        self.current_tier.remove(&pid);
    }

    /// Current tier for a PID, or None if not tracked.
    pub fn current_tier(&self, pid: u32) -> Option<SchedulingTier> {
        self.current_tier.get(&pid).copied()
    }

    /// Return all currently tracked (pid, tier) pairs.
    pub fn current_tier_keys(&self) -> Vec<(u32, SchedulingTier)> {
        self.current_tier.iter().map(|(k, v)| (*k, *v)).collect()
    }

    /// Count of processes currently pushed to E-Cores.
    pub fn background_count(&self) -> usize {
        self.current_tier
            .values()
            .filter(|&&t| t == SchedulingTier::Background)
            .count()
    }

    // ── Private ───────────────────────────────────────────────────────────

    #[cfg(target_os = "macos")]
    fn apply_task_policy(&self, pid: u32, tier: SchedulingTier) -> QoSOutcome {
        use self::ffi::*;
        use self::mach_sys::*;

        let role = match tier {
            SchedulingTier::Foreground => TASK_FOREGROUND_APPLICATION,
            SchedulingTier::Normal => TASK_UNSPECIFIED,
            SchedulingTier::Background => TASK_BACKGROUND_APPLICATION,
        };

        unsafe {
            // 1. Get the Mach task port for the target PID
            let mut task_port: MachPortT = MACH_PORT_NULL;
            let kr = task_for_pid(mach_task_self(), pid as i32, &mut task_port);

            if kr != KERN_SUCCESS {
                return QoSOutcome {
                    pid,
                    tier,
                    success: false,
                    error: Some(format!("task_for_pid failed: kern_return={}", kr)),
                };
            }

            // 2. Apply the category policy
            let policy = TaskCategoryPolicy { role };
            let kr2 = task_policy_set(
                task_port,
                TASK_CATEGORY_POLICY,
                &policy as *const _,
                TASK_CATEGORY_POLICY_COUNT,
            );

            if kr2 != KERN_SUCCESS {
                return QoSOutcome {
                    pid,
                    tier,
                    success: false,
                    error: Some(format!("task_policy_set failed: kern_return={}", kr2)),
                };
            }
        }

        QoSOutcome {
            pid,
            tier,
            success: true,
            error: None,
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn apply_task_policy(&self, pid: u32, tier: SchedulingTier) -> QoSOutcome {
        // No-op on non-macOS platforms
        QoSOutcome {
            pid,
            tier,
            success: false,
            error: Some("task_policy_set only available on macOS".into()),
        }
    }
}

impl Default for MachQoSManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helper: map our ProcessTier → SchedulingTier ──────────────────────────────

use crate::engine::process_classifier::ProcessTier;

/// Map a classifier tier to the correct Mach scheduling tier.
pub fn tier_for_process(process_tier: ProcessTier) -> SchedulingTier {
    match process_tier {
        ProcessTier::ActiveForeground => SchedulingTier::Foreground,
        ProcessTier::BackgroundVisible => SchedulingTier::Normal,
        ProcessTier::AppHelper => SchedulingTier::Normal,
        ProcessTier::SystemEssential => SchedulingTier::Foreground,
        ProcessTier::SilentDaemon => SchedulingTier::Background,
        ProcessTier::Stale => SchedulingTier::Background,
        ProcessTier::ZombieOrphan => SchedulingTier::Background,
        ProcessTier::Telemetry => SchedulingTier::Background,
    }
}
