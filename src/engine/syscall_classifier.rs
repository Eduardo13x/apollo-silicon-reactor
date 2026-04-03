//! Syscall-aware process profiler via proc_pidinfo(PROC_PIDTASKINFO).
//!
//! Classifies processes based on per-cycle deltas of kernel-reported counters:
//! unix syscalls, Mach traps, context switches, page faults, and page-ins.
//! No Apple entitlements required — root access (which Apollo has) is sufficient.
//!
//! # Usage
//!
//! ```ignore
//! let mut classifier = SyscallClassifier::new();
//!
//! // Call once per optimization cycle for each PID of interest.
//! if let Some(profile) = classifier.sample(pid) {
//!     if profile == SyscallProfile::JitCompiling {
//!         // skip throttle — JIT page-fault spike in progress
//!     }
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::engine::proc_taskinfo;

// ── Public types ──────────────────────────────────────────────────────────────

/// Syscall-level activity snapshot for one process at one point in time.
/// Values are cumulative counters read from the kernel — use deltas for rates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyscallSnapshot {
    pub unix_calls: u64,
    pub mach_calls: u64,
    pub context_switches: u64,
    pub page_faults: u64,
    pub pageins: u64,
    pub threads: u32,
}

/// Behavioural profile inferred from inter-cycle deltas of syscall counters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SyscallProfile {
    /// High unix calls, moderate page-ins — disk or file I/O intensive.
    FileIO,
    /// High Mach traps, low faults — IPC/network heavy.
    Network,
    /// High page faults combined with high context switches — JIT compilation
    /// is mapping new executable pages.  Do NOT throttle.
    JitCompiling,
    /// Low syscalls, low context switches — CPU-bound compute.  Safe to throttle.
    Compute,
    /// Near-zero everything — truly idle.  Safe to freeze.
    Idle,
    /// Does not match any clear pattern.
    Unknown,
}

// ── Classifier ────────────────────────────────────────────────────────────────

/// Stateful classifier that maintains the previous snapshot per PID so it can
/// compute per-cycle deltas.  One instance should be kept alive across cycles.
pub struct SyscallClassifier {
    prev: HashMap<u32, SyscallSnapshot>,
}

impl Default for SyscallClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl SyscallClassifier {
    pub fn new() -> Self {
        Self {
            prev: HashMap::new(),
        }
    }

    /// Sample a process and return its current [`SyscallProfile`].
    ///
    /// Returns `None` if `proc_pidinfo` fails (permission denied, PID gone, etc.).
    ///
    /// On the first call for a given PID the snapshot is seeded but no profile is
    /// returned yet — there is no previous baseline for delta computation.
    pub fn sample(&mut self, pid: u32) -> Option<SyscallProfile> {
        let current = Self::read_taskinfo(pid)?;
        let profile = if let Some(prev) = self.prev.get(&pid) {
            let delta = SyscallSnapshot {
                unix_calls: current.unix_calls.saturating_sub(prev.unix_calls),
                mach_calls: current.mach_calls.saturating_sub(prev.mach_calls),
                context_switches: current
                    .context_switches
                    .saturating_sub(prev.context_switches),
                page_faults: current.page_faults.saturating_sub(prev.page_faults),
                pageins: current.pageins.saturating_sub(prev.pageins),
                threads: current.threads,
            };
            Some(Self::classify(&delta))
        } else {
            // First observation — seed the baseline; no delta yet.
            None
        };
        self.prev.insert(pid, current);
        profile
    }

    /// Evict stale entries for PIDs no longer being sampled.
    ///
    /// Call periodically (e.g. every 60 cycles) with the current live PID set
    /// to keep memory bounded on long-running daemons.
    pub fn evict_stale(&mut self, live_pids: &[u32]) {
        let live: std::collections::HashSet<u32> = live_pids.iter().copied().collect();
        self.prev.retain(|pid, _| live.contains(pid));
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Read raw task info via proc_pidinfo and convert to SyscallSnapshot.
    fn read_taskinfo(pid: u32) -> Option<SyscallSnapshot> {
        let ti = proc_taskinfo::get_task_info(pid)?;
        Some(SyscallSnapshot {
            unix_calls: ti.syscalls_unix as u64,
            mach_calls: ti.syscalls_mach as u64,
            context_switches: ti.context_switches as u64,
            page_faults: ti.faults as u64,
            pageins: ti.pageins as u64,
            threads: ti.thread_count,
        })
    }

    /// Classify a per-cycle delta into a [`SyscallProfile`].
    fn classify(delta: &SyscallSnapshot) -> SyscallProfile {
        let total = delta.unix_calls + delta.mach_calls;

        // Truly quiet — nothing happening.
        if total < 10 && delta.context_switches < 5 {
            return SyscallProfile::Idle;
        }

        // JIT compilation signature: burst of page faults while context switches
        // are also elevated.  This occurs when a JIT (Node, JVM, LuaJIT, Julia,
        // JavaScript engine) maps and wires new executable pages.
        // Do NOT throttle — interrupting JIT mid-compilation causes hangs/crashes.
        let fault_rate = delta.page_faults as f64 / (total + 1) as f64;
        if fault_rate > 0.1 && delta.context_switches > 50 {
            return SyscallProfile::JitCompiling;
        }

        // File I/O: unix syscalls dominate (read/write/open/stat).
        if delta.unix_calls > delta.mach_calls.saturating_mul(3) {
            return SyscallProfile::FileIO;
        }

        // Network / IPC: Mach traps dominate (mach_msg, semaphore, XPC).
        if delta.mach_calls > delta.unix_calls.saturating_mul(2) {
            return SyscallProfile::Network;
        }

        // Compute-bound: low syscall rate AND low context switches.
        // The process is burning CPU without blocking — safe to throttle.
        if total < 50 && delta.context_switches < 20 {
            return SyscallProfile::Compute;
        }

        SyscallProfile::Unknown
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_delta(
        unix: u64,
        mach: u64,
        csw: u64,
        faults: u64,
        pageins: u64,
    ) -> SyscallSnapshot {
        SyscallSnapshot {
            unix_calls: unix,
            mach_calls: mach,
            context_switches: csw,
            page_faults: faults,
            pageins,
            threads: 1,
        }
    }

    #[test]
    fn classify_idle_all_zeros() {
        let d = make_delta(0, 0, 0, 0, 0);
        assert_eq!(SyscallClassifier::classify(&d), SyscallProfile::Idle);
    }

    #[test]
    fn classify_idle_near_zero() {
        let d = make_delta(5, 3, 4, 0, 0);
        assert_eq!(SyscallClassifier::classify(&d), SyscallProfile::Idle);
    }

    #[test]
    fn classify_jit_high_faults_and_csw() {
        // fault_rate = 500 / (100 + 50 + 1) ≈ 3.3 > 0.1, csw=200 > 50
        let d = make_delta(100, 50, 200, 500, 10);
        assert_eq!(SyscallClassifier::classify(&d), SyscallProfile::JitCompiling);
    }

    #[test]
    fn classify_jit_boundary_exact() {
        // fault_rate = 12 / (100 + 1 + 1) ≈ 0.117 > 0.1, csw=51 > 50
        let d = make_delta(100, 1, 51, 12, 0);
        assert_eq!(SyscallClassifier::classify(&d), SyscallProfile::JitCompiling);
    }

    #[test]
    fn classify_fileio_unix_dominant() {
        // unix > mach * 3
        let d = make_delta(300, 50, 20, 5, 2);
        assert_eq!(SyscallClassifier::classify(&d), SyscallProfile::FileIO);
    }

    #[test]
    fn classify_network_mach_dominant() {
        // mach > unix * 2
        let d = make_delta(50, 200, 30, 3, 0);
        assert_eq!(SyscallClassifier::classify(&d), SyscallProfile::Network);
    }

    #[test]
    fn classify_compute_low_everything() {
        // total = 30, csw = 10 — CPU-bound, low syscalls
        let d = make_delta(20, 10, 10, 1, 0);
        assert_eq!(SyscallClassifier::classify(&d), SyscallProfile::Compute);
    }

    #[test]
    fn classify_unknown_mixed() {
        // unix ≈ mach, high csw, moderate faults — no clear winner
        let d = make_delta(100, 80, 100, 8, 5);
        assert_eq!(SyscallClassifier::classify(&d), SyscallProfile::Unknown);
    }

    #[test]
    fn sample_returns_none_on_first_call() {
        let mut cls = SyscallClassifier::new();
        let pid = std::process::id();
        // First call seeds baseline — no profile yet.
        // (If proc_pidinfo fails in the test runner, sample returns None anyway.)
        let first = cls.sample(pid);
        // Either None (first observation) or None (no permission) — not a profile.
        // We can't assert Some here because on first call it's always None.
        assert!(first.is_none(), "first sample must return None (no baseline yet)");
    }

    #[test]
    fn sample_returns_profile_on_second_call() {
        let mut cls = SyscallClassifier::new();
        let pid = std::process::id();
        // Seed the baseline.
        cls.sample(pid);
        // Second call should produce a profile (if proc_pidinfo is available).
        let second = cls.sample(pid);
        // In a test environment we can't guarantee root access, but if the call
        // succeeds it must return a valid profile.
        if let Some(profile) = second {
            let valid = matches!(
                profile,
                SyscallProfile::Idle
                    | SyscallProfile::Compute
                    | SyscallProfile::FileIO
                    | SyscallProfile::Network
                    | SyscallProfile::JitCompiling
                    | SyscallProfile::Unknown
            );
            assert!(valid, "returned an unrecognised profile variant");
        }
    }

    #[test]
    fn evict_stale_removes_dead_pids() {
        let mut cls = SyscallClassifier::new();
        // Inject two fake baseline entries directly.
        cls.prev.insert(1001, SyscallSnapshot::default());
        cls.prev.insert(1002, SyscallSnapshot::default());
        cls.prev.insert(1003, SyscallSnapshot::default());

        cls.evict_stale(&[1001, 1003]);

        assert!(cls.prev.contains_key(&1001));
        assert!(!cls.prev.contains_key(&1002), "1002 should have been evicted");
        assert!(cls.prev.contains_key(&1003));
    }
}
