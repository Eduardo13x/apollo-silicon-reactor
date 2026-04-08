//! Per-process CPU contention tracker — stateful cache of prev `RusageInfo`
//! samples so `cpu_contention_ratio` can be computed across cycles without
//! every caller having to maintain its own HashMap.
//!
//! The raw deriving helper `cpu_contention_ratio` (in `proc_taskinfo`) is
//! stateless: it takes `prev` and `curr` samples. This tracker owns the
//! `prev` side, keyed by pid, so consumers just call `observe(pid, curr)`
//! and get back the ratio against whatever sample the tracker last saw.
//!
//! ## Lifecycle
//!
//! - `observe(pid, curr)` — returns the contention ratio between the last
//!   cached sample for `pid` and the new one. Stores the new sample as the
//!   next baseline. Returns `None` if there was no prior sample, or if the
//!   process was idle across the interval.
//! - `latest(pid)` — returns the most recently computed ratio without
//!   inserting a new sample (useful for read-only consumers).
//! - `gc(live_pids)` — drops any cached entries for pids not in `live_pids`
//!   so the map doesn't grow unbounded over a long-running daemon session.
//!
//! ## Memory cost
//!
//! One `RusageInfo` entry per tracked pid (~200 bytes on M1). Apollo
//! typically tracks a few hundred pids at most, so < 100 KB steady state.
//! A hard cap (`MAX_TRACKED_PIDS`) enforces that the map can never exceed
//! this budget even if the pid stream is pathological.
//!
//! ## References
//!
//! - [Brown 2019] "Pressure Stall Information" — PSI is stateful in
//!   exactly this way: the kernel keeps per-task `psi_task_state` structs
//!   so user-space readers don't have to reconstruct history.
//! - [Mohan et al. 1992] "ARIES" §3 — separating the stateless recovery
//!   logic from the stateful cursor is the pattern that keeps the tracker
//!   testable in isolation.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::engine::proc_taskinfo::{cpu_contention_ratio, RusageInfo};

/// Global process-lifetime contention tracker. Populated lazily on first
/// access. Used by hot-path consumers (process_enrichment, decide_actions,
/// dashboard) that would otherwise have to thread a `&mut ContentionTracker`
/// through every function signature — a worse trade than a narrow global.
///
/// Safety: the inner Mutex is held only for the duration of individual
/// `observe`/`latest`/`gc` calls, each of which is O(1) or O(n) over the
/// tracked pid set. No I/O happens under the lock.
pub fn global() -> &'static Mutex<ContentionTracker> {
    static TRACKER: OnceLock<Mutex<ContentionTracker>> = OnceLock::new();
    TRACKER.get_or_init(|| Mutex::new(ContentionTracker::new()))
}

/// Hard cap on the number of pids we track at once. One RusageInfo is
/// ~200 bytes, so 2_000 entries ≈ 400 KB — comfortably bounded even on
/// an 8 GB machine.
pub const MAX_TRACKED_PIDS: usize = 2_000;

/// Stateful per-pid CPU contention tracker. See module docs.
#[derive(Debug, Default)]
pub struct ContentionTracker {
    /// Last RusageInfo sample seen for each pid.
    prev: HashMap<u32, RusageInfo>,
    /// Last contention ratio computed for each pid.
    last_ratio: HashMap<u32, f64>,
}

impl ContentionTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe a new rusage sample for `pid` and return the contention ratio
    /// against the previously cached sample. The new sample becomes the
    /// baseline for the next call.
    ///
    /// Returns `None` on the first call for a pid (no prior sample) or when
    /// the process did not want any CPU in the interval (idle → `cpu_contention_ratio`
    /// returns None).
    pub fn observe(&mut self, pid: u32, curr: RusageInfo) -> Option<f64> {
        // Enforce the hard cap before insertion. Eviction picks an arbitrary
        // pid via `keys().next()` — good enough because we only expect to
        // hit the cap under runaway-pid-churn pathologies, and fairness of
        // eviction is unimportant there.
        if self.prev.len() >= MAX_TRACKED_PIDS && !self.prev.contains_key(&pid) {
            if let Some(&drop_pid) = self.prev.keys().next() {
                self.prev.remove(&drop_pid);
                self.last_ratio.remove(&drop_pid);
            }
        }

        let ratio = self
            .prev
            .get(&pid)
            .and_then(|prev_sample| cpu_contention_ratio(prev_sample, &curr));
        if let Some(r) = ratio {
            self.last_ratio.insert(pid, r);
        }
        self.prev.insert(pid, curr);
        ratio
    }

    /// Most recent contention ratio computed for `pid`, if any. Does not
    /// modify tracker state.
    pub fn latest(&self, pid: u32) -> Option<f64> {
        self.last_ratio.get(&pid).copied()
    }

    /// Drop cached state for any pids not present in `live_pids`. Call once
    /// per cycle with the current known-alive pid set.
    pub fn gc(&mut self, live_pids: &std::collections::HashSet<u32>) {
        self.prev.retain(|pid, _| live_pids.contains(pid));
        self.last_ratio.retain(|pid, _| live_pids.contains(pid));
    }

    /// Number of pids currently tracked (for metrics/diagnostics).
    pub fn len(&self) -> usize {
        self.prev.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prev.is_empty()
    }

    /// Fraction of tracked pids with a contention ratio ≥ `threshold`
    /// in their most recent sample. This is the system-wide "how many
    /// processes are being starved right now" aggregate.
    ///
    /// Returns 0.0 if no pids are tracked.
    pub fn stall_fraction(&self, threshold: f64) -> f64 {
        if self.last_ratio.is_empty() {
            return 0.0;
        }
        let stalled = self
            .last_ratio
            .values()
            .filter(|&&r| r >= threshold)
            .count();
        stalled as f64 / self.last_ratio.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::proc_taskinfo::QoSBreakdown;

    fn mk(user: u64, system: u64, runnable: u64) -> RusageInfo {
        RusageInfo {
            pid: 1,
            user_time_ns: user,
            system_time_ns: system,
            idle_wakeups: 0,
            interrupt_wakeups: 0,
            pageins: 0,
            wired_size: 0,
            resident_size: 0,
            phys_footprint: 0,
            disk_read_bytes: 0,
            disk_write_bytes: 0,
            logical_writes: 0,
            instructions: 0,
            cycles: 0,
            billed_energy: 0,
            runnable_time_ns: runnable,
            proc_start_abstime: 0,
            qos_time: QoSBreakdown::default(),
        }
    }

    #[test]
    fn first_observe_returns_none() {
        let mut t = ContentionTracker::new();
        assert_eq!(t.observe(42, mk(1000, 0, 0)), None);
        assert_eq!(t.latest(42), None);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn subsequent_observe_returns_ratio() {
        let mut t = ContentionTracker::new();
        t.observe(42, mk(0, 0, 0));
        // Wanted CPU for 1 ms, got 500 μs → 50% contention.
        let ratio = t.observe(42, mk(250_000, 250_000, 500_000));
        assert!((ratio.unwrap() - 0.5).abs() < 1e-9);
        // latest() is now populated.
        assert!((t.latest(42).unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn gc_drops_dead_pids() {
        let mut t = ContentionTracker::new();
        t.observe(10, mk(0, 0, 0));
        t.observe(20, mk(0, 0, 0));
        t.observe(30, mk(0, 0, 0));
        assert_eq!(t.len(), 3);
        let live: std::collections::HashSet<u32> = [20u32].into_iter().collect();
        t.gc(&live);
        assert_eq!(t.len(), 1);
        assert!(t.latest(10).is_none());
    }

    #[test]
    fn stall_fraction_aggregates() {
        let mut t = ContentionTracker::new();
        // Prime 3 pids with zero ratios.
        t.observe(1, mk(0, 0, 0));
        t.observe(2, mk(0, 0, 0));
        t.observe(3, mk(0, 0, 0));
        // pid 1: fully starved (runnable delta only).
        t.observe(1, mk(0, 0, 1_000_000));
        // pid 2: fully satisfied (on-cpu only).
        t.observe(2, mk(500_000, 500_000, 0));
        // pid 3: stays idle → no new ratio, falls out of last_ratio.
        // stall_fraction threshold 0.5 → only pid 1 ≥ 0.5 ⇒ 0.5 of tracked.
        assert_eq!(t.stall_fraction(0.5), 0.5);
    }

    #[test]
    fn hard_cap_evicts_on_overflow() {
        let mut t = ContentionTracker::new();
        // Insert MAX + 5 pids. Map should never exceed MAX.
        for pid in 0..(MAX_TRACKED_PIDS as u32 + 5) {
            t.observe(pid, mk(0, 0, 0));
            assert!(t.len() <= MAX_TRACKED_PIDS);
        }
        assert_eq!(t.len(), MAX_TRACKED_PIDS);
    }

    #[test]
    fn reinsert_known_pid_does_not_trigger_eviction() {
        let mut t = ContentionTracker::new();
        // Fill to cap.
        for pid in 0..(MAX_TRACKED_PIDS as u32) {
            t.observe(pid, mk(0, 0, 0));
        }
        let len_before = t.len();
        // Re-observing an existing pid should not evict anyone.
        t.observe(5, mk(100, 100, 100));
        assert_eq!(t.len(), len_before);
    }
}
