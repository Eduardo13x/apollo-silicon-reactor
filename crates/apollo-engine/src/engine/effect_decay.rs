//! Stuck-Effect decay watchdog — Hellerstein 2004 §9.3 settling-time
//! observer.
//!
//! **STATUS — WIRED** (S10 follow-up, 2026-06-06). The detector is owned
//! by `SharedState.effect_decay`. Producer call sites in
//! `execute_actions.rs` record Jetsam + Sysctl effects post-Receipt; the
//! drain consumer in `daemon_cycle_tail.rs::drain_effect_decay` re-reads
//! the observable via `jetsam_control::get_priority` /
//! `sysctl_direct::read_i32` and bumps `effect_decay_detected_total` on
//! mismatch. MachPolicy producer is deferred — `MachQoSManager` exposes
//! no `get_policy(pid)` query API and adding one requires unverified
//! `thread_policy_get` FFI work outside the S10 budget.
//!
//! Once Jetsam/Sysctl observations start arriving the counter is the
//! "lying syscall" alarm described in the Sprint 9 telemetry-death
//! lesson — `kill(SIGSTOP)` succeeds but the process is observed
//! RUNNING 30 s later, etc.
//!
//! Wake-grace: the consumer guards the drain with a 30-s post-wake
//! grace; immediately after wake the kernel may not have reapplied
//! tier hints — false-positive disagreements would inflate the
//! counter.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// S10 cutover (2026-06-06): global handle so producer call sites in
/// `execute_actions.rs` can record observations without threading a new
/// parameter through the entire signature chain (mirrors the
/// LSE_COUNTERS / shadow_signals pattern). Initialised once at daemon
/// startup by `daemon_state::SharedState` construction; tests may use
/// `install_global_for_tests`. When uninitialised the producer helpers
/// no-op (test harnesses that don't bring up SharedState observe no
/// telemetry side-effect).
static GLOBAL_WATCHDOG: OnceLock<Arc<Mutex<DecayWatchdog>>> = OnceLock::new();

/// Install the shared watchdog handle. Idempotent: first call wins.
pub fn install_global(w: Arc<Mutex<DecayWatchdog>>) {
    let _ = GLOBAL_WATCHDOG.set(w);
}

/// Producer helper: enroll an observation against the global watchdog.
/// No-op when no global handle has been installed (test paths,
/// lib-only callers).
pub fn record_global(obs: PendingObservation) {
    if let Some(w) = GLOBAL_WATCHDOG.get() {
        let mut guard = w.lock().unwrap_or_else(|e| e.into_inner());
        guard.record(obs);
    }
}

/// Test-only: replace the global with a fresh watchdog. Unsafe under
/// concurrent test execution; gated behind cfg(test) on the caller.
pub fn install_global_for_tests(w: Arc<Mutex<DecayWatchdog>>) {
    // OnceLock has no public replace; tolerate the first-wins limitation
    // by ignoring duplicate installs. Tests that need isolation should
    // serialize via #[serial] or a test mutex.
    let _ = GLOBAL_WATCHDOG.set(w);
}

/// Type of observation a watchdog entry carries. Each kind has its own
/// post-settle re-read mechanism in the (future) consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObsKind {
    JetsamTier,
    MachPolicy,
    Sysctl,
}

/// A single pending settle-time observation.
#[derive(Debug, Clone)]
pub struct PendingObservation {
    pub effect_id: u64,
    pub pid: u32,
    pub kind: ObsKind,
    /// Sysctl key, populated only when `kind == ObsKind::Sysctl`. Required
    /// for the consumer's `sysctl_direct::read_i32(key)` re-read call.
    /// `None` for other kinds.
    pub key: Option<String>,
    /// Encoded post-syscall value (interpretation depends on `kind`):
    /// jetsam priority for `JetsamTier`, scheduling tier ord for
    /// `MachPolicy`, sysctl integer for `Sysctl`.
    pub value_post: i64,
    pub deadline: Instant,
}

const RING_CAP: usize = 64;
const SETTLE: Duration = Duration::from_secs(5);

/// FIFO ring of pending observations. Drains expired entries on demand.
#[derive(Default)]
pub struct DecayWatchdog {
    ring: VecDeque<PendingObservation>,
}

impl DecayWatchdog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Settling window used by [`Self::record`] to compute the deadline.
    pub fn settle_window() -> Duration {
        SETTLE
    }

    /// Bounded ring capacity.
    pub fn capacity() -> usize {
        RING_CAP
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Record a new observation. Older observations are dropped on cap.
    pub fn record(&mut self, obs: PendingObservation) {
        if self.ring.len() >= RING_CAP {
            self.ring.pop_front();
        }
        self.ring.push_back(obs);
    }

    /// Drain every observation whose `deadline <= now`. Returns the drained
    /// entries; caller is expected to re-read the observable (jetsam
    /// priority, sysctl, etc.) and call [`Self::report_disagreement`] for
    /// each entry whose re-read differs.
    pub fn drain_expired(&mut self, now: Instant) -> Vec<PendingObservation> {
        let mut out = Vec::new();
        while let Some(front) = self.ring.front() {
            if front.deadline <= now {
                if let Some(o) = self.ring.pop_front() {
                    out.push(o);
                }
            } else {
                break;
            }
        }
        out
    }

    /// Convenience: caller reports a post-settle re-read disagreed with
    /// the recorded value. Bumps the LSE counter exactly once per call.
    pub fn report_disagreement(&self) {
        crate::engine::lse_counters::LSE_COUNTERS.inc_effect_decay_detected();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(deadline: Instant) -> PendingObservation {
        PendingObservation {
            effect_id: 1,
            pid: std::process::id(),
            kind: ObsKind::JetsamTier,
            key: None,
            value_post: 9,
            deadline,
        }
    }

    #[test]
    fn record_respects_cap() {
        let mut w = DecayWatchdog::new();
        for _ in 0..(RING_CAP + 10) {
            w.record(obs(Instant::now() + Duration::from_secs(60)));
        }
        assert_eq!(w.len(), RING_CAP);
    }

    #[test]
    fn drain_returns_only_expired_entries() {
        let mut w = DecayWatchdog::new();
        let now = Instant::now();
        w.record(obs(now - Duration::from_secs(1))); // expired
        w.record(obs(now + Duration::from_secs(10))); // future
        let drained = w.drain_expired(now);
        assert_eq!(drained.len(), 1);
        assert_eq!(w.len(), 1, "future entry remains");
    }

    #[test]
    fn report_disagreement_bumps_counter() {
        use std::sync::atomic::Ordering;
        let w = DecayWatchdog::new();
        let pre = crate::engine::lse_counters::LSE_COUNTERS
            .effect_decay_detected_total
            .load(Ordering::Relaxed);
        w.report_disagreement();
        let post = crate::engine::lse_counters::LSE_COUNTERS
            .effect_decay_detected_total
            .load(Ordering::Relaxed);
        assert_eq!(post, pre + 1);
    }
}
