//! Work-session hysteresis — keeps "work mode" latched through a dev session.
//!
//! ## Problem
//! Apollo enters an aggressive "work mode" (boost dev cluster, freeze background
//! noise, AggressiveRoot profile) when it detects a dev session via
//! `workload_onset`. But the instant the user closes Claude Code / stops
//! compiling, the raw workload classifier snaps back to `Idle`, the governor
//! relaxes, background noise returns — "the zen mode deflates instantly."
//!
//! ## Design
//! [`WorkSession`] is a tiny, pure hysteresis latch. It records the timestamp of
//! the last observed dev activity and reports "active" for a grace window after
//! that — so the profile governor stays aggressive through the whole session and
//! decays gracefully instead of snapping to Idle.
//!
//! It is purely *additive*: the only lever it touches is the governor's
//! `workload_onset` input, and it can only ever turn that input ON (latch
//! aggression), never off. When the latch is inactive, behaviour is identical to
//! before this module existed.
//!
//! ## Battery safety
//! `battery_low` ALWAYS wins → the latch is forced inactive. We never hold zen
//! mode on a dying battery: survival of the user's battery beats the feel of a
//! fast system. On battery (but not low) the grace window is shortened
//! ([`GRACE_SECS_ON_BATTERY`]) to avoid draining when genuinely idle.
//!
//! No I/O, no allocation, no persistence — in-memory per-cycle daemon state only.

use std::time::{Duration, Instant};

/// Grace window on AC power: keep work mode latched for 5 min after the last
/// observed dev activity, so a brief pause (reading a diff, thinking, a failed
/// build) does not collapse the session.
pub const GRACE_SECS: u64 = 300;

/// Grace window on battery (not low): shortened to 90 s so we do not keep the
/// dev cluster boosted / background frozen for minutes while genuinely idle and
/// draining the battery.
pub const GRACE_SECS_ON_BATTERY: u64 = 90;

/// In-memory hysteresis latch for the dev work session. Cheap to copy; lives as
/// per-cycle daemon-loop state (like `prev_workload_mode`). Not persisted.
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkSession {
    /// Monotonic timestamp of the last cycle that observed dev activity.
    /// `None` until the first dev-active cycle is seen.
    last_dev_activity: Option<Instant>,
}

impl WorkSession {
    /// Construct an empty session (no activity ever observed → inactive).
    pub fn new() -> Self {
        Self {
            last_dev_activity: None,
        }
    }

    /// Record this cycle's dev-activity observation. Only refreshes the latch
    /// when `is_dev_active` — a non-dev cycle leaves the timestamp untouched so
    /// the grace window decays naturally from the *last* real activity.
    pub fn note_activity(&mut self, is_dev_active: bool, now: Instant) {
        if is_dev_active {
            self.last_dev_activity = Some(now);
        }
    }

    /// Whether the work-mode latch is currently active.
    ///
    /// - `battery_low` → always `false` (survival of the battery beats feel).
    /// - otherwise `true` iff the last dev activity is within the grace window,
    ///   where the window is [`GRACE_SECS_ON_BATTERY`] on battery else
    ///   [`GRACE_SECS`].
    pub fn is_active(&self, on_battery: bool, battery_low: bool, now: Instant) -> bool {
        if battery_low {
            return false;
        }
        let Some(last) = self.last_dev_activity else {
            return false;
        };
        let grace = if on_battery {
            GRACE_SECS_ON_BATTERY
        } else {
            GRACE_SECS
        };
        // saturating_duration_since: a clock that somehow went backwards yields
        // ZERO elapsed (treated as "just now" → active), never a panic.
        now.saturating_duration_since(last) < Duration::from_secs(grace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_never_active() {
        let ws = WorkSession::new();
        let now = Instant::now();
        assert!(!ws.is_active(false, false, now));
        assert!(!ws.is_active(true, false, now));
    }

    #[test]
    fn active_within_grace_inactive_past_grace_on_ac() {
        let mut ws = WorkSession::new();
        let t0 = Instant::now();
        ws.note_activity(true, t0);

        // Within the 300 s AC grace.
        let within = t0 + Duration::from_secs(200);
        assert!(ws.is_active(false, false, within));

        // Past the 300 s AC grace.
        let past = t0 + Duration::from_secs(301);
        assert!(!ws.is_active(false, false, past));
    }

    #[test]
    fn ac_power_uses_full_300s_grace() {
        let mut ws = WorkSession::new();
        let t0 = Instant::now();
        ws.note_activity(true, t0);
        // 120 s would be past the 90 s battery grace, but on AC the full 300 s
        // window keeps it active.
        let t = t0 + Duration::from_secs(120);
        assert!(ws.is_active(false, false, t));
    }

    #[test]
    fn battery_low_forces_inactive_even_within_grace() {
        let mut ws = WorkSession::new();
        let t0 = Instant::now();
        ws.note_activity(true, t0);
        let within = t0 + Duration::from_secs(10);
        // Within grace on AC → active.
        assert!(ws.is_active(false, false, within));
        // Same instant, but battery_low → forced inactive.
        assert!(!ws.is_active(false, true, within));
        assert!(!ws.is_active(true, true, within));
    }

    #[test]
    fn on_battery_shortens_grace() {
        let mut ws = WorkSession::new();
        let t0 = Instant::now();
        ws.note_activity(true, t0);

        // 60 s on battery → within the 90 s battery grace → active.
        let at_60 = t0 + Duration::from_secs(60);
        assert!(ws.is_active(true, false, at_60));

        // 120 s on battery → past the 90 s battery grace → inactive.
        let at_120 = t0 + Duration::from_secs(120);
        assert!(!ws.is_active(true, false, at_120));
    }

    #[test]
    fn note_activity_false_does_not_refresh_latch() {
        let mut ws = WorkSession::new();
        let t0 = Instant::now();
        ws.note_activity(true, t0);
        // A later non-dev cycle must NOT extend the window.
        let later = t0 + Duration::from_secs(100);
        ws.note_activity(false, later);
        // Past 300 s from the original activity → inactive despite the later
        // (non-dev) note.
        let past = t0 + Duration::from_secs(301);
        assert!(!ws.is_active(false, false, past));
    }
}
