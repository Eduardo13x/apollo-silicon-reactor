//! Per-PID cooldown after a throttle was refused at execution time.
//!
//! Problem observed 2026-08-18 in production: across 18 days the journal held
//! **1537 ThrottleProcess entries and not one success**. 1502 of them targeted
//! just two processes — `apollo-web-bridge` (795) and a ChatGPT child process
//! (707). Both refusals are correct: one is held by the coalition guard, the
//! other sits behind the execute-time protection classifier. Nothing, however,
//! carried that refusal back to the decision layer, so the same futile proposal
//! was re-emitted every cycle forever.
//!
//! The refusals are cheap early returns, so this costs little CPU. What it
//! costs is signal: the journal became 94% refused proposals and 6% real
//! actions, which silently biases anything computed over it.
//!
//! Boosts already have this feedback — the world-model utility gate learns
//! that a boost has negative measured utility and vetoes it. Throttles had no
//! equivalent. This module is that equivalent, deliberately dumber: it does not
//! learn a utility, it only remembers "execution said no, stop asking for a
//! while".
//!
//! Cooldown is **ephemeral** — it does not persist across a daemon restart, and
//! it is never permanent. A refusal reflects the state at one moment: the
//! coalition guard depends on which app is in the foreground, and the
//! protection lists change with a redeploy. Expiring the memory is what keeps a
//! transient refusal from hardening into a permanent blind spot.
//!
//! [Nygard 2018] §8.5 — hold-down windows after a refusal prevent retry storms.

use std::collections::HashMap;

/// How long a refusal is remembered, in daemon cycles.
///
/// Two windows, because the two refusal families decay differently. At the
/// ~0.9 cycles/s observed on the M4 development host these are roughly 11 min
/// and 65 min; both are stated in cycles because the tick rate varies with
/// load and no behaviour here depends on wall-clock time.
///
/// `TRANSIENT` covers refusals that follow the user around — the coalition
/// guard releases a PID once its app leaves the foreground envelope, so the
/// memory must expire soon enough to catch that.
pub const TRANSIENT_COOLDOWN_CYCLES: u16 = 600;

/// `STRUCTURAL` covers refusals rooted in a protection list or an OS
/// constraint. Those rarely change without a redeploy, so re-asking often buys
/// nothing — but the window still expires, because a permanent block would
/// outlive the policy that justified it.
pub const STRUCTURAL_COOLDOWN_CYCLES: u16 = 3600;

/// Bound on tracked PIDs. Past this the machine is not running more refusable
/// processes; something is churning PIDs, and an unbounded map would be the
/// real bug.
pub const MAX_TRACKED: usize = 256;

/// Which window a refusal earns. The caller maps its own reason enum onto this
/// so the module stays independent of `BlockReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalKind {
    /// Depends on user focus or momentary system state; re-check soon.
    Transient,
    /// Rooted in a protection list or OS constraint; re-check rarely.
    Structural,
}

impl RefusalKind {
    fn cycles(self) -> u16 {
        match self {
            Self::Transient => TRANSIENT_COOLDOWN_CYCLES,
            Self::Structural => STRUCTURAL_COOLDOWN_CYCLES,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct ThrottleRefusalCooldown {
    remaining: HashMap<u32, u16>,
}

impl ThrottleRefusalCooldown {
    pub fn new() -> Self {
        Self {
            remaining: HashMap::new(),
        }
    }

    /// Record that execution refused to throttle `pid`.
    ///
    /// A longer window always wins: a PID refused for a structural reason does
    /// not get its memory shortened by a later transient refusal.
    pub fn mark_refused(&mut self, pid: u32, kind: RefusalKind) {
        if self.remaining.len() >= MAX_TRACKED && !self.remaining.contains_key(&pid) {
            return;
        }
        let want = kind.cycles();
        let slot = self.remaining.entry(pid).or_insert(0);
        *slot = (*slot).max(want);
    }

    /// True when a throttle for this PID should not be proposed yet.
    pub fn is_in_cooldown(&self, pid: u32) -> bool {
        self.remaining.get(&pid).is_some_and(|&n| n > 0)
    }

    /// Decrement every window by one; drop entries that reach zero.
    /// Call once per daemon cycle.
    pub fn tick(&mut self) {
        self.remaining.retain(|_pid, n| {
            *n = n.saturating_sub(1);
            *n > 0
        });
    }

    /// PIDs currently held — for metrics.
    pub fn active_count(&self) -> usize {
        self.remaining.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refusal_is_remembered_then_forgotten() {
        let mut c = ThrottleRefusalCooldown::new();
        c.mark_refused(42, RefusalKind::Transient);
        assert!(c.is_in_cooldown(42));
        for _ in 0..TRANSIENT_COOLDOWN_CYCLES {
            c.tick();
        }
        assert!(
            !c.is_in_cooldown(42),
            "a refusal must expire — the coalition guard releases a PID once \
             its app leaves the foreground, and a permanent memory would never \
             notice"
        );
        assert_eq!(c.active_count(), 0);
    }

    #[test]
    fn a_structural_refusal_outlives_a_transient_one() {
        let mut c = ThrottleRefusalCooldown::new();
        c.mark_refused(7, RefusalKind::Structural);
        c.mark_refused(7, RefusalKind::Transient);
        for _ in 0..TRANSIENT_COOLDOWN_CYCLES {
            c.tick();
        }
        assert!(
            c.is_in_cooldown(7),
            "the later transient refusal must not shorten a structural window"
        );
    }

    #[test]
    fn the_map_is_bounded_but_keeps_refreshing_known_pids() {
        let mut c = ThrottleRefusalCooldown::new();
        for pid in 0..(MAX_TRACKED as u32 + 50) {
            c.mark_refused(pid, RefusalKind::Transient);
        }
        assert_eq!(c.active_count(), MAX_TRACKED);
        // A PID already tracked can still be refreshed past the bound.
        c.mark_refused(0, RefusalKind::Structural);
        for _ in 0..TRANSIENT_COOLDOWN_CYCLES {
            c.tick();
        }
        assert!(c.is_in_cooldown(0));
    }

    #[test]
    fn an_untracked_pid_is_never_held() {
        let mut c = ThrottleRefusalCooldown::new();
        c.mark_refused(9, RefusalKind::Transient);
        assert!(c.is_in_cooldown(9));
        assert!(
            !c.is_in_cooldown(10),
            "a refusal for one PID must not silence another"
        );
    }
}
