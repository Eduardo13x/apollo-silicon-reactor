//! Per-PID post-thaw cooldown for gate_e anti-flap.
//!
//! Problem: gate_e (decide_actions.rs) resurrects the freezer for M1 8GB by
//! firing on swap_pct >= 0.85 + memory_pressure >= 0.70. The existing TTL
//! (`MAX_FROZEN_CYCLES=150` in planner.rs) thaws frozen PIDs after ~5 min.
//! Without an explicit cooldown, a thawed PID may still meet gate_e thresholds
//! and be re-frozen on the next cycle — oscillation under sustained pressure.
//!
//! This module tracks recently-thawed PIDs and prevents gate_e from re-freezing
//! them within the cooldown window. The kernel needs ~10-30s to redistribute
//! pressure after a SIGCONT; the cooldown gives swap reclaim a chance to take
//! effect before another freeze decision.
//!
//! Cooldown is **ephemeral** — does not persist across daemon restart.
//! On restart, the freezer treats the system as fresh; if real pressure
//! still exists, the gate_e logic will reapply naturally.
//!
//! [Nygard 2018] §8.5 — circuit breakers must include hold-down windows
//! after recovery to prevent thrashing on slow-decaying load conditions.

use std::collections::HashMap;

/// Cooldown duration in daemon cycles. At ~2 Hz tick rate this is ~30 s.
/// Tuned to be longer than typical kernel swap-redistribution latency
/// (10-15 s observed) but shorter than MAX_FROZEN_CYCLES (150 ≈ 5 min)
/// so a process held in real, sustained pressure will eventually re-freeze.
pub const GATE_E_COOLDOWN_CYCLES: u8 = 60;

#[derive(Debug, Default, Clone)]
pub struct FreezeCooldown {
    remaining: HashMap<u32, u8>,
}

impl FreezeCooldown {
    pub fn new() -> Self {
        Self {
            remaining: HashMap::new(),
        }
    }

    /// Mark a PID as recently thawed. Resets its cooldown counter.
    pub fn mark_thawed(&mut self, pid: u32) {
        self.remaining.insert(pid, GATE_E_COOLDOWN_CYCLES);
    }

    /// Returns true if the PID is in cooldown and must not be re-frozen by gate_e.
    pub fn is_in_cooldown(&self, pid: u32) -> bool {
        self.remaining.get(&pid).map(|&n| n > 0).unwrap_or(false)
    }

    /// Decrement all cooldown counters by 1; remove entries that reach 0.
    /// Call once per daemon cycle.
    pub fn tick(&mut self) {
        self.remaining.retain(|_pid, n| {
            *n = n.saturating_sub(1);
            *n > 0
        });
    }

    /// Number of PIDs currently in cooldown — for observability/metrics.
    pub fn active_count(&self) -> usize {
        self.remaining.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newly_thawed_pid_is_in_cooldown() {
        let mut c = FreezeCooldown::new();
        c.mark_thawed(1234);
        assert!(c.is_in_cooldown(1234));
    }

    #[test]
    fn untracked_pid_is_not_in_cooldown() {
        let c = FreezeCooldown::new();
        assert!(!c.is_in_cooldown(9999));
    }

    #[test]
    fn cooldown_expires_after_n_ticks() {
        let mut c = FreezeCooldown::new();
        c.mark_thawed(1234);
        for _ in 0..GATE_E_COOLDOWN_CYCLES {
            c.tick();
        }
        assert!(!c.is_in_cooldown(1234));
    }

    #[test]
    fn cooldown_active_during_window() {
        let mut c = FreezeCooldown::new();
        c.mark_thawed(1234);
        for _ in 0..(GATE_E_COOLDOWN_CYCLES - 1) {
            c.tick();
        }
        assert!(c.is_in_cooldown(1234));
    }

    #[test]
    fn re_thaw_resets_cooldown() {
        let mut c = FreezeCooldown::new();
        c.mark_thawed(1234);
        for _ in 0..(GATE_E_COOLDOWN_CYCLES / 2) {
            c.tick();
        }
        c.mark_thawed(1234);
        for _ in 0..(GATE_E_COOLDOWN_CYCLES / 2) {
            c.tick();
        }
        assert!(c.is_in_cooldown(1234));
    }
}
