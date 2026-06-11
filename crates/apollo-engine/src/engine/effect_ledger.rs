//! Effect Ledger — reversibility by construction for every kernel mutation.
//!
//! Evolve iter-3 (2026-06-10). The 2026-06-10 fight-hunt confirmed NINE
//! instances of the same disease: Apollo applies a kernel side-effect
//! (nice, Mach tier, jetsam priority, App Nap, memlimit, Darwin-BG,
//! RT-band, E-core migration, io-throttle) and nothing ever reverts it.
//! Each was patched with an ad-hoc tracking structure (`boost_ledger`,
//! `interrupt_migrated_pids`, `last_applied_limits`, `app_napped`,
//! `last_markov_prethaw`) — five bespoke implementations of one idea.
//!
//! This module is the consolidation: ONE ledger where every applied
//! effect is recorded WITH its undo semantics, a justification tag, a
//! TTL, and the PID-identity guard. A periodic reconcile pass reverts
//! anything whose justification expired. New effectors inherit
//! reversibility by recording here — ratchets become impossible by
//! construction instead of being hunted one at a time.
//!
//! [Saltzer & Schroeder 1975] complete mediation — applied to the
//! RETURN path: every privileged mutation passes through one
//! revert-capable chokepoint.
//!
//! Semantics:
//! - `record(...)` upserts by (pid, kind); re-applying refreshes the TTL
//!   (a continuously-justified effect never expires).
//! - `reconcile(...)` drains entries past TTL, verifies PID identity
//!   (start_sec — recycled PIDs keep their fresh state), skips the
//!   current foreground pid, executes the undo, and removes the entry.
//! - `cleanup(live)` drops entries for exited PIDs (kernel already
//!   reclaimed everything — nothing to undo).
//! - Global-handle pattern (mirrors `effect_decay` / `boost_ledger`):
//!   producers sit deep in execute paths where threading a parameter
//!   through the signature chain is disruptive.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::engine::lock_ext::LockRecover;
use crate::engine::mach_qos::{MachQoSManager, SchedulingTier};

/// What was mutated and how to undo it. Each variant carries the prior
/// state when it is cheaply capturable; otherwise the undo restores the
/// kernel-default rest state (documented per variant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppliedEffect {
    /// `setpriority(PRIO_PROCESS, pid, -N)` boost. Undo: restore 0
    /// (default). Prior nice is not captured — Apollo only ever boosts
    /// FROM the default, and a non-default prior means an external
    /// writer owns it (we still restore 0: the safe, kernel-default
    /// rest state).
    Nice { pid: u32 },
    /// Mach task scheduling tier (set_tier). Undo: Normal — the kernel /
    /// runningboard re-elevates genuinely-foreground work from there.
    MachTier { pid: u32 },
    /// Jetsam band override (memorystatus priority). Undo: restore the
    /// captured prior band verbatim. `prior < 0` = unreadable at apply
    /// time → undo skips the jetsam write (guessing a band would fight
    /// runningboard) and relies on the kernel's own lifecycle moves.
    JetsamPriority { pid: u32, prior: i32 },
    /// App Nap suppression-token (set_app_nap true). Undo: release.
    AppNap { pid: u32 },
    /// Jetsam memlimit (set_memlimit). Undo: 0/0 = kernel-default
    /// unlimited.
    Memlimit { pid: u32 },
    /// PRIO_DARWIN_BG demotion (fallback E-core path). Undo: clear flag.
    DarwinBg { pid: u32 },
}

impl AppliedEffect {
    pub fn pid(&self) -> u32 {
        match *self {
            AppliedEffect::Nice { pid }
            | AppliedEffect::MachTier { pid }
            | AppliedEffect::JetsamPriority { pid, .. }
            | AppliedEffect::AppNap { pid }
            | AppliedEffect::Memlimit { pid }
            | AppliedEffect::DarwinBg { pid } => pid,
        }
    }

    /// Stable kind index — (pid, kind) is the upsert key. MUST stay in
    /// sync with the variant count.
    fn kind(&self) -> u8 {
        match self {
            AppliedEffect::Nice { .. } => 0,
            AppliedEffect::MachTier { .. } => 1,
            AppliedEffect::JetsamPriority { .. } => 2,
            AppliedEffect::AppNap { .. } => 3,
            AppliedEffect::Memlimit { .. } => 4,
            AppliedEffect::DarwinBg { .. } => 5,
        }
    }
}

#[derive(Debug, Clone)]
struct LedgerEntry {
    effect: AppliedEffect,
    applied_at: Instant,
    ttl: Duration,
    /// PID-identity guard: live start_sec must match at undo time.
    start_sec: u64,
    /// Why the effect was applied — for the revert log line.
    justification: &'static str,
}

/// Default TTL when a producer has no domain-specific lifetime. 10 min
/// matches the boost-decay calibration (61267d3): long enough that an
/// actively-justified effect is refreshed many times over, short enough
/// that stale state never survives a session phase change.
pub const DEFAULT_TTL: Duration = Duration::from_secs(600);

pub struct EffectLedger {
    entries: HashMap<(u32, u8), LedgerEntry>,
}

impl EffectLedger {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    /// Upsert an applied effect. Re-applying refreshes the TTL clock —
    /// a continuously-qualifying effect never expires.
    pub fn record(
        &mut self,
        effect: AppliedEffect,
        ttl: Duration,
        start_sec: u64,
        justification: &'static str,
    ) {
        self.entries.insert(
            (effect.pid(), effect.kind()),
            LedgerEntry { effect, applied_at: Instant::now(), ttl, start_sec, justification },
        );
    }

    /// Remove an entry without undoing (the producer reverted it itself,
    /// or the justification is permanently settled).
    pub fn forget(&mut self, effect: &AppliedEffect) {
        self.entries.remove(&(effect.pid(), effect.kind()));
    }

    /// Entries past TTL, excluding the foreground pid. Returned entries
    /// are removed — the caller MUST execute the undo (or the effect is
    /// orphaned again). Use [`reconcile_global`] for the full loop.
    fn drain_expired(&mut self, foreground_pid: Option<u32>) -> Vec<LedgerEntry> {
        let now = Instant::now();
        let expired: Vec<(u32, u8)> = self
            .entries
            .iter()
            .filter(|((pid, _), e)| {
                Some(*pid) != foreground_pid && now.duration_since(e.applied_at) >= e.ttl
            })
            .map(|(k, _)| *k)
            .collect();
        expired.into_iter().filter_map(|k| self.entries.remove(&k)).collect()
    }

    /// Drop entries for PIDs that exited.
    pub fn cleanup(&mut self, live_pids: &[u32]) {
        let live: std::collections::HashSet<u32> = live_pids.iter().copied().collect();
        self.entries.retain(|(pid, _), _| live.contains(pid));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for EffectLedger {
    fn default() -> Self {
        Self::new()
    }
}

// ── Global handle ────────────────────────────────────────────────────────────

static GLOBAL: Mutex<Option<EffectLedger>> = Mutex::new(None);

fn with_global<R>(f: impl FnOnce(&mut EffectLedger) -> R) -> R {
    let mut guard = GLOBAL.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(EffectLedger::new))
}

/// Producer API: record an applied effect against the global ledger.
pub fn record_global(
    effect: AppliedEffect,
    ttl: Duration,
    start_sec: u64,
    justification: &'static str,
) {
    with_global(|l| l.record(effect, ttl, start_sec, justification));
}

/// Producer API: forget without undo (producer reverted it itself).
pub fn forget_global(effect: &AppliedEffect) {
    with_global(|l| l.forget(effect));
}

/// Observability: current ledger size.
pub fn len_global() -> usize {
    with_global(|l| l.len())
}

/// Drop entries for exited PIDs.
pub fn cleanup_global(live_pids: &[u32]) {
    with_global(|l| l.cleanup(live_pids));
}

/// The reconcile pass: drain expired entries, verify PID identity, and
/// execute each undo. Returns the number of effects actually reverted.
///
/// Call cadence: every ~30 cycles from the daemon main loop (dead weight
/// accumulates over minutes, not milliseconds — same cadence as the
/// zombie sweep). Cost is bounded: undo syscalls only for entries that
/// actually expired this window.
pub fn reconcile_global(
    foreground_pid: Option<u32>,
    qos_mgr: &Arc<Mutex<MachQoSManager>>,
) -> u64 {
    let expired = with_global(|l| l.drain_expired(foreground_pid));
    if expired.is_empty() {
        return 0;
    }
    const PRIO_DARWIN_BG: libc::c_int = 0x1000;
    let mut reverted = 0u64;
    for entry in expired {
        let pid = entry.effect.pid();
        // PID-identity guard: only undo on the same process we mutated.
        let (live_sec, _) = crate::engine::daemon_helpers::pid_start_time(pid);
        if live_sec == 0 || live_sec != entry.start_sec {
            continue;
        }
        match entry.effect {
            AppliedEffect::Nice { pid } => unsafe {
                libc::setpriority(libc::PRIO_PROCESS, pid, 0);
            },
            AppliedEffect::MachTier { pid } => {
                let mut qos = qos_mgr.lock_recover();
                qos.set_tier(pid, SchedulingTier::Normal);
            }
            AppliedEffect::JetsamPriority { pid, prior } => {
                if prior >= 0 {
                    let _ = crate::engine::jetsam_control::set_priority(pid, prior);
                }
            }
            AppliedEffect::AppNap { pid } => {
                let mut qos = qos_mgr.lock_recover();
                qos.set_app_nap(pid, false);
            }
            AppliedEffect::Memlimit { pid } => {
                let _ = crate::engine::jetsam_control::set_memlimit(pid, 0, 0);
            }
            AppliedEffect::DarwinBg { pid } => unsafe {
                libc::setpriority(PRIO_DARWIN_BG, pid, 0);
            },
        }
        crate::engine::lse_counters::LSE_COUNTERS.inc_effect_ledger_revert();
        tracing::debug!(
            pid,
            effect = ?entry.effect,
            justification = entry.justification,
            "effect-ledger: reverted expired effect"
        );
        reverted += 1;
    }
    reverted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nice(pid: u32) -> AppliedEffect {
        AppliedEffect::Nice { pid }
    }

    #[test]
    fn record_refreshes_ttl_on_reapply() {
        let mut l = EffectLedger::new();
        l.record(nice(901_001), Duration::from_secs(1), 42, "test");
        // Backdate, then re-record — the refresh must reset the clock.
        if let Some(e) = l.entries.get_mut(&(901_001, 0)) {
            e.applied_at = Instant::now() - Duration::from_secs(5);
        }
        l.record(nice(901_001), Duration::from_secs(60), 42, "test");
        let drained = l.drain_expired(None);
        assert!(drained.is_empty(), "refreshed entry must not expire");
        assert_eq!(l.len(), 1);
    }

    #[test]
    fn drain_respects_ttl_and_foreground() {
        let mut l = EffectLedger::new();
        l.record(nice(901_002), Duration::from_secs(0), 7, "test");
        // Foreground exclusion holds even when expired.
        assert!(l.drain_expired(Some(901_002)).is_empty());
        // Non-foreground expired entry drains exactly once.
        let d = l.drain_expired(None);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].start_sec, 7);
        assert!(l.drain_expired(None).is_empty());
    }

    #[test]
    fn upsert_key_is_pid_plus_kind() {
        let mut l = EffectLedger::new();
        l.record(nice(901_003), DEFAULT_TTL, 1, "a");
        l.record(AppliedEffect::MachTier { pid: 901_003 }, DEFAULT_TTL, 1, "b");
        assert_eq!(l.len(), 2, "different kinds for same pid coexist");
        l.record(nice(901_003), DEFAULT_TTL, 1, "a2");
        assert_eq!(l.len(), 2, "same (pid, kind) upserts");
    }

    #[test]
    fn cleanup_drops_dead_pids() {
        let mut l = EffectLedger::new();
        l.record(nice(901_004), DEFAULT_TTL, 1, "t");
        l.record(nice(901_005), DEFAULT_TTL, 1, "t");
        l.cleanup(&[901_005]);
        assert_eq!(l.len(), 1);
    }

    #[test]
    fn forget_removes_without_undo() {
        let mut l = EffectLedger::new();
        let e = nice(901_006);
        l.record(e, DEFAULT_TTL, 1, "t");
        l.forget(&e);
        assert!(l.is_empty());
    }
}
