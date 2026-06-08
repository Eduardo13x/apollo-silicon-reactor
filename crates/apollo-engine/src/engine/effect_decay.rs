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

impl ObsKind {
    /// Stable discriminant index into the per-kind ring array. MUST match
    /// the order in [`ALL_OBS_KINDS`]. Used for O(1) ring routing in
    /// [`DecayWatchdog::record`] / [`DecayWatchdog::drain_expired`].
    ///
    /// FIX-4-v2 (2026-06-07): the single 64-slot ring was vulnerable to
    /// Boost-burst eviction (79.5% of action volume) silently dropping
    /// the Jetsam/Sysctl observations that produce the working
    /// `effect_decay_detected_total` signal. Partition restores
    /// per-kind isolation: each kind keeps its own 64-slot ring.
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            ObsKind::JetsamTier => 0,
            ObsKind::MachPolicy => 1,
            ObsKind::Sysctl => 2,
        }
    }
}

/// Number of distinct [`ObsKind`] variants. Used to size the per-kind
/// ring array. MUST be kept in sync with the enum.
pub const N_OBS_KINDS: usize = 3;

/// All variants in stable-index order. Used by drain loops that need to
/// iterate every per-kind ring.
pub const ALL_OBS_KINDS: [ObsKind; N_OBS_KINDS] = [
    ObsKind::JetsamTier,
    ObsKind::MachPolicy,
    ObsKind::Sysctl,
];

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
    /// Approach-3 wire (2026-06-07): producer marks this observation as
    /// targeting a hard-protected process (per `safety::hard_protected_contains`).
    /// Consumer-side disagreements on hard-protected targets are
    /// accumulated into a 5-minute sliding window
    /// (`DecayWatchdog::hard_protected_decay_count_5min`) and used to
    /// trigger `PolicyRollbackGuard::evaluate_from_decay` when the count
    /// crosses the threshold.
    ///
    /// Sysctl observations have no process target — producers MUST set
    /// this to `false` for `ObsKind::Sysctl`. Backward-compat: missing
    /// in struct literals from older code paths is a compile error
    /// (intentional — forces explicit producer audit).
    #[doc(alias = "hp")]
    pub hard_protected: bool,
}

const RING_CAP: usize = 64;
const SETTLE: Duration = Duration::from_secs(5);

/// Approach-3 wire (2026-06-07): bounded ring of recent hard-protected
/// disagreement timestamps. Pruned to the 5-minute window on every push
/// and on every count query — bounded memory, bounded per-cycle work.
const HP_DECAY_RING_CAP: usize = 32;
/// Sliding window for `hard_protected_decay_count_5min`. Mirrors
/// `POLICY_ROLLBACK_RECENT_WINDOW` in `learned_state.rs` so the guard
/// and the producer agree on what "recent" means.
pub const HP_DECAY_WINDOW: Duration = Duration::from_secs(5 * 60);

/// FIFO ring of pending observations. Drains expired entries on demand.
///
/// FIX-4-v2 (2026-06-07): partitioned per-`ObsKind` rings. Each kind
/// owns an independent 64-slot ring so a Boost burst (MachPolicy
/// accounts for 79.5% of action volume) cannot evict the
/// Jetsam/Sysctl observations producing the live
/// `effect_decay_detected_total` signal. Total bounded memory:
/// `RING_CAP * N_OBS_KINDS` slots.
pub struct DecayWatchdog {
    /// Per-`ObsKind` FIFO rings, indexed by [`ObsKind::index`]. Each
    /// ring stays at [`RING_CAP`] slots; producers route via
    /// `obs.kind.index()`; drains iterate all kinds in
    /// [`ALL_OBS_KINDS`] order.
    rings: [VecDeque<PendingObservation>; N_OBS_KINDS],
    /// Approach-3 wire (2026-06-07): sliding window of recent
    /// hard-protected disagreement timestamps. Pruned to
    /// `HP_DECAY_WINDOW` on push/query. Capacity-bounded by
    /// `HP_DECAY_RING_CAP` for defense-in-depth against stalls.
    ///
    /// Producer-side data (PendingObservation.hard_protected) drives
    /// pushes; consumer-side queries via
    /// `hard_protected_decay_count_5min` drive the rollback trigger.
    hp_decays: VecDeque<Instant>,
    /// Track the PIDs that drove the recent disagreements so the log
    /// line on rollback names them. Same prune discipline as `hp_decays`.
    hp_decay_pids: VecDeque<(Instant, u32)>,
}

impl Default for DecayWatchdog {
    fn default() -> Self {
        Self {
            // VecDeque::new is not const, so we cannot use array
            // initialiser syntax with a literal. The array-of-arrays
            // builder via `from_fn` runs at construction time only —
            // negligible cost vs. per-cycle hot path.
            rings: std::array::from_fn(|_| VecDeque::new()),
            hp_decays: VecDeque::new(),
            hp_decay_pids: VecDeque::new(),
        }
    }
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

    /// Total pending observations across every per-kind ring.
    pub fn len(&self) -> usize {
        self.rings.iter().map(|r| r.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.rings.iter().all(|r| r.is_empty())
    }

    /// Per-`ObsKind` pending count. Used by tests verifying isolation
    /// (Boost burst must not evict Jetsam/Sysctl entries).
    pub fn len_for_kind(&self, kind: ObsKind) -> usize {
        self.rings[kind.index()].len()
    }

    /// Record a new observation. Older observations of the *same kind*
    /// are dropped on cap — kinds are isolated so a Boost burst cannot
    /// evict Jetsam/Sysctl entries (FIX-4-v2 2026-06-07).
    pub fn record(&mut self, obs: PendingObservation) {
        let idx = obs.kind.index();
        let ring = &mut self.rings[idx];
        if ring.len() >= RING_CAP {
            ring.pop_front();
        }
        ring.push_back(obs);
    }

    /// Drain every observation whose `deadline <= now` across every
    /// per-kind ring. Returns the drained entries; caller is expected
    /// to re-read the observable (jetsam priority, sysctl, etc.) and
    /// call [`Self::report_disagreement`] / [`Self::report_disagreement_with`]
    /// for each entry whose re-read differs.
    ///
    /// FIFO discipline within each kind preserved: we pop_front while
    /// the front entry is expired, breaking at the first non-expired
    /// front — same semantics as the pre-partition implementation,
    /// applied per ring.
    pub fn drain_expired(&mut self, now: Instant) -> Vec<PendingObservation> {
        let mut out = Vec::new();
        for ring in self.rings.iter_mut() {
            while let Some(front) = ring.front() {
                if front.deadline <= now {
                    if let Some(o) = ring.pop_front() {
                        out.push(o);
                    }
                } else {
                    break;
                }
            }
        }
        out
    }

    /// Convenience: caller reports a post-settle re-read disagreed with
    /// the recorded value. Bumps the LSE counter exactly once per call.
    ///
    /// Prefer [`Self::report_disagreement_with`] when the caller still
    /// has the originating observation in scope — it forwards the
    /// hard-protected bit into the rollback-trigger window. Retained
    /// for backward-compatibility with the existing daemon tail
    /// consumer.
    pub fn report_disagreement(&self) {
        crate::engine::lse_counters::LSE_COUNTERS.inc_effect_decay_detected();
    }

    /// Approach-3 wire (2026-06-07): caller reports a post-settle
    /// disagreement AND forwards the observation so we can track
    /// hard-protected disagreements in a sliding window. Bumps the
    /// `effect_decay_detected_total` counter exactly once, like
    /// [`Self::report_disagreement`].
    ///
    /// When the observation is marked hard-protected, also records the
    /// timestamp + pid into the sliding-window ring used by
    /// [`Self::hard_protected_decay_count_5min`] / [`Self::hard_protected_decay_pids`].
    pub fn report_disagreement_with(&mut self, obs: &PendingObservation) {
        crate::engine::lse_counters::LSE_COUNTERS.inc_effect_decay_detected();
        if obs.hard_protected {
            let now = Instant::now();
            // Defense in depth: cap before push to bound memory even
            // under pathological pressure where prune-by-time can't
            // keep up.
            if self.hp_decays.len() >= HP_DECAY_RING_CAP {
                self.hp_decays.pop_front();
            }
            if self.hp_decay_pids.len() >= HP_DECAY_RING_CAP {
                self.hp_decay_pids.pop_front();
            }
            self.hp_decays.push_back(now);
            self.hp_decay_pids.push_back((now, obs.pid));
            self.prune_hp_decays(now);
        }
    }

    /// Round-4 (2026-06-07). FIX-3-v2 unconditional HP MachPolicy
    /// forward path. Records the HP window entry (so
    /// `hard_protected_decay_count_5min` sees the attempt) and bumps
    /// the dedicated `effect_decay_hp_mach_attempts_total` counter,
    /// WITHOUT touching `effect_decay_detected_total` (preserves the
    /// Jetsam/Sysctl re-read-disagreement baseline 27 used by the
    /// regression watchdog).
    ///
    /// Caller must already have confirmed `obs.hard_protected == true`
    /// (the unconditional forward is gated upstream in daemon_cycle_tail).
    pub fn record_hp_mach_attempt(&mut self, obs: &PendingObservation) {
        crate::engine::lse_counters::LSE_COUNTERS.inc_effect_decay_hp_mach_attempt();
        let now = Instant::now();
        if self.hp_decays.len() >= HP_DECAY_RING_CAP {
            self.hp_decays.pop_front();
        }
        if self.hp_decay_pids.len() >= HP_DECAY_RING_CAP {
            self.hp_decay_pids.pop_front();
        }
        self.hp_decays.push_back(now);
        self.hp_decay_pids.push_back((now, obs.pid));
        self.prune_hp_decays(now);
    }

    /// Approach-3 wire (2026-06-07): count of hard-protected
    /// disagreement events within the [`HP_DECAY_WINDOW`] sliding
    /// window ending at `now`. Mutating because it prunes expired
    /// entries before counting — keeps memory bounded across the
    /// daemon's lifetime regardless of throughput.
    pub fn hard_protected_decay_count_5min(&mut self, now: Instant) -> usize {
        self.prune_hp_decays(now);
        self.hp_decays.len()
    }

    /// Approach-3 wire (2026-06-07): snapshot of PIDs in the
    /// 5-minute hard-protected window, newest-first. Used by the
    /// rollback-trigger log line so the operator sees which PIDs
    /// were stuck. Mutating for the same prune reason as
    /// [`Self::hard_protected_decay_count_5min`].
    pub fn hard_protected_decay_pids(&mut self, now: Instant) -> Vec<u32> {
        self.prune_hp_decays(now);
        self.hp_decay_pids
            .iter()
            .rev()
            .map(|(_, p)| *p)
            .collect()
    }

    /// Drop entries older than [`HP_DECAY_WINDOW`] from both
    /// hard-protected rings. `now.checked_duration_since(t)` is
    /// `None` if the clock moved backwards — treat that as
    /// out-of-window (matches the daemon "best-effort" discipline).
    fn prune_hp_decays(&mut self, now: Instant) {
        while let Some(front) = self.hp_decays.front() {
            let in_window = now
                .checked_duration_since(*front)
                .map(|elapsed| elapsed <= HP_DECAY_WINDOW)
                .unwrap_or(false);
            if in_window {
                break;
            }
            self.hp_decays.pop_front();
        }
        while let Some((front, _)) = self.hp_decay_pids.front() {
            let in_window = now
                .checked_duration_since(*front)
                .map(|elapsed| elapsed <= HP_DECAY_WINDOW)
                .unwrap_or(false);
            if in_window {
                break;
            }
            self.hp_decay_pids.pop_front();
        }
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
            hard_protected: false,
        }
    }

    fn hp_obs(pid: u32) -> PendingObservation {
        PendingObservation {
            effect_id: 1,
            pid,
            kind: ObsKind::JetsamTier,
            key: None,
            value_post: 2,
            deadline: Instant::now(),
            hard_protected: true,
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

    #[test]
    fn hp_disagreement_populates_window() {
        let mut w = DecayWatchdog::new();
        assert_eq!(w.hard_protected_decay_count_5min(Instant::now()), 0);
        for pid in [101, 102, 103] {
            w.report_disagreement_with(&hp_obs(pid));
        }
        assert_eq!(w.hard_protected_decay_count_5min(Instant::now()), 3);
        let pids = w.hard_protected_decay_pids(Instant::now());
        assert!(pids.contains(&101));
        assert!(pids.contains(&103));
        // Newest-first ordering: last push (103) at index 0.
        assert_eq!(pids.first().copied(), Some(103));
    }

    #[test]
    fn non_hp_disagreement_does_not_populate_window() {
        let mut w = DecayWatchdog::new();
        w.report_disagreement_with(&obs(Instant::now()));
        w.report_disagreement_with(&obs(Instant::now()));
        assert_eq!(w.hard_protected_decay_count_5min(Instant::now()), 0);
    }

    #[test]
    fn hp_window_prune_drops_old_entries() {
        let mut w = DecayWatchdog::new();
        // Inject an aged entry by hand: simulates 6 minutes ago.
        let aged = Instant::now()
            .checked_sub(Duration::from_secs(6 * 60))
            .expect("monotonic clock under test");
        w.hp_decays.push_back(aged);
        w.hp_decay_pids.push_back((aged, 7));
        // Push one fresh entry — prune should drop the aged one.
        w.report_disagreement_with(&hp_obs(8));
        assert_eq!(w.hard_protected_decay_count_5min(Instant::now()), 1);
        let pids = w.hard_protected_decay_pids(Instant::now());
        assert_eq!(pids, vec![8]);
    }

    /// FIX-4-v2 (2026-06-07): per-ObsKind partition prevents a Boost
    /// burst (MachPolicy) from evicting pre-existing Jetsam/Sysctl
    /// observations. Pre-partition: single ring, Boost burst silently
    /// dropped the working S10 signal.
    #[test]
    fn per_kind_partition_isolates_evictions() {
        let mut w = DecayWatchdog::new();
        let future = Instant::now() + Duration::from_secs(60);

        // Seed pre-existing Jetsam + Sysctl observations.
        w.record(PendingObservation {
            effect_id: 100,
            pid: 1,
            kind: ObsKind::JetsamTier,
            key: None,
            value_post: 9,
            deadline: future,
            hard_protected: false,
        });
        w.record(PendingObservation {
            effect_id: 200,
            pid: 2,
            kind: ObsKind::Sysctl,
            key: Some("vm.compressor_mode".to_string()),
            value_post: 4,
            deadline: future,
            hard_protected: false,
        });

        // Boost burst: 2 * RING_CAP MachPolicy entries — would have
        // evicted everything pre-partition.
        for i in 0..(RING_CAP as u64 * 2) {
            w.record(PendingObservation {
                effect_id: 1000 + i,
                pid: 100 + i as u32,
                kind: ObsKind::MachPolicy,
                key: None,
                value_post: 1,
                deadline: future,
                hard_protected: false,
            });
        }

        // Per-kind invariants:
        //   - MachPolicy capped at RING_CAP (FIFO eviction within kind).
        //   - Jetsam/Sysctl untouched by the burst.
        assert_eq!(w.len_for_kind(ObsKind::MachPolicy), RING_CAP);
        assert_eq!(w.len_for_kind(ObsKind::JetsamTier), 1);
        assert_eq!(w.len_for_kind(ObsKind::Sysctl), 1);
        // Total = RING_CAP + 2.
        assert_eq!(w.len(), RING_CAP + 2);
    }

    /// Drain visits every per-kind ring and returns expired entries
    /// from each. FIFO within each kind preserved.
    #[test]
    fn drain_iterates_all_kinds() {
        let mut w = DecayWatchdog::new();
        let now = Instant::now();
        let past = now - Duration::from_secs(1);
        let future = now + Duration::from_secs(60);

        // Expired entry per kind.
        for kind in ALL_OBS_KINDS {
            w.record(PendingObservation {
                effect_id: 1,
                pid: 1,
                kind,
                key: None,
                value_post: 0,
                deadline: past,
                hard_protected: false,
            });
            // Plus one fresh entry per kind that must survive.
            w.record(PendingObservation {
                effect_id: 2,
                pid: 2,
                kind,
                key: None,
                value_post: 0,
                deadline: future,
                hard_protected: false,
            });
        }

        let drained = w.drain_expired(now);
        assert_eq!(drained.len(), N_OBS_KINDS, "one per kind expired");
        // Surviving entries: one per kind.
        for kind in ALL_OBS_KINDS {
            assert_eq!(w.len_for_kind(kind), 1);
        }
    }

    #[test]
    fn hp_window_caps_at_ring_capacity() {
        let mut w = DecayWatchdog::new();
        for pid in 0..(HP_DECAY_RING_CAP as u32 + 10) {
            w.report_disagreement_with(&hp_obs(pid));
        }
        assert!(w.hard_protected_decay_count_5min(Instant::now()) <= HP_DECAY_RING_CAP);
    }
}
