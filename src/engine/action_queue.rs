//! Priority Action Queue — backpressure for the main optimization loop.
//!
//! Actions are classified into three priority tiers:
//!  - **Urgent**: Unfreeze / emergency — always execute first, no cap.
//!  - **Normal**: Freeze / Throttle / Boost — execute up to `max_per_cycle`.
//!  - **Background**: QoS hints, Sysctl, Spotlight — best-effort remainder.
//!
//! `drain_cycle()` returns at most `max_per_cycle` actions per call, draining
//! urgent first, then filling from normal, then background. This prevents a
//! burst of 50 deferred throttles from blocking the next cycle.
//!
//! `backpressure_ratio()` reports queue saturation [0.0, 1.0] for runtime
//! observability and adaptive aggressiveness decisions.

use std::collections::VecDeque;

use crate::engine::types::RootAction;

// ── Priority classification ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionPriority {
    /// Unfreeze (SIGCONT), emergency — never delayed.
    Urgent,
    /// Freeze (SIGSTOP), Throttle, Boost — core optimization work.
    Normal,
    /// QoS hints (SetThreadQoS), sysctl tuning, Spotlight toggle — best-effort.
    Background,
}

/// Classify a `RootAction` into its priority tier.
pub fn action_priority(action: &RootAction) -> ActionPriority {
    match action {
        RootAction::UnfreezeProcess { .. } => ActionPriority::Urgent,
        RootAction::FreezeProcess { .. }
        | RootAction::ThrottleProcess { .. }
        | RootAction::BoostProcess { .. } => ActionPriority::Normal,
        RootAction::SetSysctl { .. }
        | RootAction::SetMemorystatus { .. }
        | RootAction::ToggleSpotlight { .. }
        | RootAction::QuarantineDaemon { .. }
        | RootAction::SetThreadQoS { .. } => ActionPriority::Background,
    }
}

// ── ActionQueue ────────────────────────────────────────────────────────────

/// Bounded priority action queue with backpressure.
///
/// Typical usage (in main loop):
/// ```ignore
/// action_queue.push_all(final_actions);
/// let cycle_actions = action_queue.drain_cycle();
/// execute_actions(cycle_actions, ...);
/// let bp = action_queue.backpressure_ratio();
/// metrics.action_queue_backpressure = bp;
/// ```
pub struct ActionQueue {
    /// Urgent tier: Unfreeze, emergency — always execute first.
    urgent: VecDeque<RootAction>,
    /// Normal tier: Freeze, Throttle, Boost.
    normal: VecDeque<RootAction>,
    /// Background tier: QoS hints, sysctl, spotlight.
    background: VecDeque<RootAction>,
    /// Maximum actions dispatched per cycle (backpressure gate).
    /// Urgent actions are *not* counted against this limit.
    max_per_cycle: usize,
    /// Capacity for normal + background combined (soft cap for backpressure_ratio).
    capacity: usize,
}

impl ActionQueue {
    /// Create a new queue with the given per-cycle dispatch limit.
    ///
    /// `max_per_cycle`: typical value 10–20 for a 30s daemon cycle.
    /// `capacity`: soft cap used to compute `backpressure_ratio` (e.g. 100).
    pub fn new(max_per_cycle: usize, capacity: usize) -> Self {
        Self {
            urgent: VecDeque::new(),
            normal: VecDeque::new(),
            background: VecDeque::new(),
            max_per_cycle,
            capacity,
        }
    }

    /// Push a single action into the appropriate priority tier.
    pub fn push(&mut self, action: RootAction) {
        match action_priority(&action) {
            ActionPriority::Urgent => self.urgent.push_back(action),
            ActionPriority::Normal => self.normal.push_back(action),
            ActionPriority::Background => self.background.push_back(action),
        }
    }

    /// Push all actions from a `Vec` into the queue in order.
    pub fn push_all(&mut self, actions: Vec<RootAction>) {
        for a in actions {
            self.push(a);
        }
    }

    /// Drain up to `max_per_cycle` actions for this cycle.
    ///
    /// Ordering:
    /// 1. All urgent actions (no cap — safety invariant).
    /// 2. Up to `max_per_cycle` normal actions.
    /// 3. Fill remaining budget from background.
    ///
    /// Returns an owned `Vec<RootAction>` ready for `execute_actions`.
    pub fn drain_cycle(&mut self) -> Vec<RootAction> {
        let mut out = Vec::new();

        // 1. Drain all urgent actions unconditionally.
        while let Some(a) = self.urgent.pop_front() {
            out.push(a);
        }

        // 2. Fill normal up to max_per_cycle.
        let mut budget = self.max_per_cycle;
        while budget > 0 {
            match self.normal.pop_front() {
                Some(a) => {
                    out.push(a);
                    budget -= 1;
                }
                None => break,
            }
        }

        // 3. Fill background with whatever budget remains.
        while budget > 0 {
            match self.background.pop_front() {
                Some(a) => {
                    out.push(a);
                    budget -= 1;
                }
                None => break,
            }
        }

        out
    }

    /// Backpressure ratio [0.0, 1.0].
    ///
    /// 0.0 = queue empty (no backpressure).
    /// 1.0 = queue at or beyond `capacity` (fully backed up).
    ///
    /// Urgent actions do not contribute — they are never delayed.
    pub fn backpressure_ratio(&self) -> f64 {
        let queued = self.normal.len() + self.background.len();
        if self.capacity == 0 {
            return 0.0;
        }
        (queued as f64 / self.capacity as f64).min(1.0)
    }

    /// Total pending actions across all tiers.
    pub fn len(&self) -> usize {
        self.urgent.len() + self.normal.len() + self.background.len()
    }

    /// True if all tiers are empty.
    pub fn is_empty(&self) -> bool {
        self.urgent.is_empty() && self.normal.is_empty() && self.background.is_empty()
    }

    /// Number of pending urgent actions.
    pub fn urgent_len(&self) -> usize {
        self.urgent.len()
    }

    /// Number of pending normal actions.
    pub fn normal_len(&self) -> usize {
        self.normal.len()
    }

    /// Number of pending background actions.
    pub fn background_len(&self) -> usize {
        self.background.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_unfreeze(pid: u32) -> RootAction {
        RootAction::UnfreezeProcess {
            pid,
            name: format!("proc{}", pid),
        }
    }

    fn make_throttle(pid: u32) -> RootAction {
        RootAction::ThrottleProcess {
            pid,
            name: format!("proc{}", pid),
            aggressive: false,
            reason: "test".to_string(),
            start_sec: 0,
            start_usec: 0,
        }
    }

    fn make_sysctl(key: &str) -> RootAction {
        RootAction::SetSysctl {
            key: key.to_string(),
            value: "1".to_string(),
            reason: "test".to_string(),
        }
    }

    fn make_freeze(pid: u32) -> RootAction {
        RootAction::FreezeProcess {
            pid,
            name: format!("proc{}", pid),
            reason: "test".to_string(),
            start_sec: 0,
            start_usec: 0,
        }
    }

    #[test]
    fn urgent_actions_always_first() {
        let mut q = ActionQueue::new(2, 100);
        q.push(make_throttle(1));
        q.push(make_throttle(2));
        q.push(make_unfreeze(99)); // urgent pushed last
        q.push(make_sysctl("kern.test"));

        let cycle = q.drain_cycle();
        // Unfreeze must be first.
        assert!(matches!(cycle[0], RootAction::UnfreezeProcess { pid: 99, .. }));
        // Then up to max_per_cycle (2) normals.
        assert_eq!(cycle.len(), 3); // 1 urgent + 2 normal
        // Background (sysctl) stays in queue because budget=0.
        assert_eq!(q.background_len(), 1);
    }

    #[test]
    fn backpressure_ratio_at_half_capacity() {
        let mut q = ActionQueue::new(5, 10);
        for i in 0..5 {
            q.push(make_throttle(i));
        }
        // 5 queued, capacity 10 → 0.5
        assert!((q.backpressure_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn backpressure_ratio_empty_is_zero() {
        let q = ActionQueue::new(5, 10);
        assert_eq!(q.backpressure_ratio(), 0.0);
    }

    #[test]
    fn backpressure_ratio_caps_at_one() {
        let mut q = ActionQueue::new(2, 3);
        for i in 0..10 {
            q.push(make_throttle(i));
        }
        assert_eq!(q.backpressure_ratio(), 1.0);
    }

    #[test]
    fn urgent_not_counted_in_backpressure() {
        let mut q = ActionQueue::new(5, 10);
        for i in 0..10 {
            q.push(make_unfreeze(i)); // 10 urgent
        }
        // urgent not counted → backpressure still 0
        assert_eq!(q.backpressure_ratio(), 0.0);
    }

    #[test]
    fn drain_cycle_respects_max_per_cycle() {
        let mut q = ActionQueue::new(3, 100);
        for i in 0..10 {
            q.push(make_throttle(i));
        }
        let cycle = q.drain_cycle();
        assert_eq!(cycle.len(), 3);
        assert_eq!(q.normal_len(), 7);
    }

    #[test]
    fn drain_cycle_background_fills_remaining_budget() {
        let mut q = ActionQueue::new(4, 100);
        q.push(make_freeze(1)); // normal
        q.push(make_sysctl("a")); // background
        q.push(make_sysctl("b")); // background
        q.push(make_sysctl("c")); // background
        q.push(make_sysctl("d")); // background (over budget)

        let cycle = q.drain_cycle();
        // 1 normal + 3 background = 4 total (budget exhausted before 5th sysctl)
        assert_eq!(cycle.len(), 4);
        assert_eq!(q.background_len(), 1);
    }

    #[test]
    fn push_all_classifies_correctly() {
        let actions = vec![
            make_unfreeze(1),
            make_throttle(2),
            make_freeze(3),
            make_sysctl("kern.x"),
        ];
        let mut q = ActionQueue::new(10, 100);
        q.push_all(actions);
        assert_eq!(q.urgent_len(), 1);
        assert_eq!(q.normal_len(), 2);
        assert_eq!(q.background_len(), 1);
    }

    #[test]
    fn empty_drain_returns_empty_vec() {
        let mut q = ActionQueue::new(10, 100);
        assert!(q.drain_cycle().is_empty());
    }
}
