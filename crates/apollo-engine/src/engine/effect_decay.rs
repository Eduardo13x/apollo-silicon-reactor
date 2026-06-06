//! Stuck-Effect decay watchdog — Hellerstein 2004 §9.3 settling-time
//! observer.
//!
//! **STATUS — UNWIRED (follow-up #4, MED severity).** Module exists, has
//! unit tests, but NO producer wires it into the daemon main loop. The
//! counter `effect_decay_detected_total` reads 0 in
//! `/var/lib/apollo/runtime_metrics.json` and will continue to read 0 until
//! a follow-up commit threads sysctl re-reads and `memorystatus_control(GET)`
//! lookups into the post-settle path. Per CLAUDE.md Sprint 9 doctrine, a
//! dormant counter is indistinguishable from a broken one — DO NOT advertise
//! this as "wired MVP" until the producers land.
//!
//! Sprint patch (2026-06-05). Telemetry-only MVP — records observations
//! when an effect is applied with a known post-snapshot value, drains
//! expired observations on tick, and bumps an LSE counter when the
//! post-settle re-read returned by the caller disagrees with the recorded
//! value. The watchdog is NOT wired into the daemon main loop in this
//! patch — that follow-up has to thread sysctl re-reads and
//! `memorystatus_control(GET)` lookups, which are out of scope for a
//! type-level introduction.
//!
//! The counter (`effect_decay_detected_total`) stays at 0 until the wiring
//! lands; once it does, a ramping counter is the "lying syscall" alarm
//! described in the Sprint 9 telemetry-death lesson — `kill(SIGSTOP)`
//! succeeds but the process is observed RUNNING 30 s later, etc.
//!
//! Wake-grace: `record` accepts a `now: Instant` and the caller may guard
//! against false positives via the [`crate::engine::wake_state`] 30-s
//! post-wake grace, since immediately after wake the kernel may not have
//! reapplied tier hints.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

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
