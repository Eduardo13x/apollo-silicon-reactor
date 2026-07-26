//! Priority Action Queue — backpressure for the main optimization loop.
//!
//! Actions are classified into three priority tiers:
//!  - **Urgent**: Unfreeze / emergency — always execute first, no cap.
//!  - **Normal**: Freeze / Throttle / Boost and latency/anomaly thread QoS.
//!  - **Background**: background-thread QoS, Sysctl, Spotlight.
//!
//! `drain_cycle()` returns at most `max_per_cycle` actions per call, draining
//! urgent first, then filling from normal, then background. This prevents a
//! burst of 50 deferred throttles from blocking the next cycle.
//!
//! `backpressure_ratio()` reports queue saturation [0.0, 1.0] for runtime
//! observability and adaptive aggressiveness decisions.

use std::collections::{HashMap, VecDeque};

use crate::engine::recently_applied::CachedActionKind;
use crate::engine::types::RootAction;

type PendingPidKey = (u32, CachedActionKind, u32);

fn pending_pid_key(action: &RootAction) -> Option<PendingPidKey> {
    match action {
        RootAction::SetThreadQoS {
            pid, thread_index, ..
        } => Some((*pid, CachedActionKind::SetThreadQoS, *thread_index)),
        _ => CachedActionKind::from_root_action(action),
    }
}

fn thread_qos_rank(action: &RootAction) -> u8 {
    match action {
        RootAction::SetThreadQoS { tier, .. } if tier == "interactive" => 3,
        RootAction::SetThreadQoS { tier, .. } if tier == "background" => 1,
        RootAction::SetThreadQoS { .. } => 2,
        _ => 0,
    }
}

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
        // Interactive promotions and runaway-thread utility demotions are
        // latency/anomaly control, not maintenance hints. Keeping all thread
        // QoS in Background let a sustained normal-action stream starve them.
        RootAction::SetThreadQoS { tier, .. } if tier != "background" => ActionPriority::Normal,
        RootAction::SetSysctl(_)
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
    /// Per-PID actions already waiting in any tier. Cross-cycle dedup belongs
    /// here: an admitted action is not "recently applied" until its syscall
    /// succeeds, but repeated producers must not fill the queue meanwhile.
    pending_pid_keys: HashMap<PendingPidKey, u8>,
    /// Maximum actions dispatched per cycle (backpressure gate).
    /// Urgent actions are *not* counted against this limit.
    max_per_cycle: usize,
    /// Hard capacity for normal + background combined. Urgent actions bypass it.
    capacity: usize,
    /// Non-urgent actions rejected because no queue slot was available.
    capacity_drops: u64,
    /// Background actions evicted to admit higher-priority normal work.
    background_evictions: u64,
}

impl ActionQueue {
    /// Create a new queue with the given per-cycle dispatch limit.
    ///
    /// `max_per_cycle`: typical value 10–20 for a 30s daemon cycle.
    /// `capacity`: hard cap for delayed normal/background work (e.g. 100).
    pub fn new(max_per_cycle: usize, capacity: usize) -> Self {
        Self {
            urgent: VecDeque::new(),
            normal: VecDeque::new(),
            background: VecDeque::new(),
            pending_pid_keys: HashMap::with_capacity(capacity.min(512)),
            max_per_cycle,
            capacity,
            capacity_drops: 0,
            background_evictions: 0,
        }
    }

    /// Current per-cycle dispatch budget (normal + background).
    /// C8 fix (round-3): callers reduce this on battery to save energy.
    pub fn max_per_cycle(&self) -> usize {
        self.max_per_cycle
    }

    /// Adjust the per-cycle dispatch budget.  Urgent (Unfreeze) actions are
    /// not counted against this limit.
    pub fn set_max_per_cycle(&mut self, n: usize) {
        self.max_per_cycle = n.max(1);
    }

    /// Push a single action into the appropriate priority tier.
    pub fn push(&mut self, action: RootAction) -> bool {
        let pending_key = pending_pid_key(&action);
        if let Some(key) = pending_key {
            let new_rank = thread_qos_rank(&action);
            if let Some(existing_rank) = self.pending_pid_keys.get(&key).copied() {
                if key.1 != CachedActionKind::SetThreadQoS || new_rank <= existing_rank {
                    return false;
                }
                // A latency-sensitive promotion supersedes a queued demotion
                // for the same thread, regardless of which tier held it.
                self.urgent
                    .retain(|queued| pending_pid_key(queued) != Some(key));
                self.normal
                    .retain(|queued| pending_pid_key(queued) != Some(key));
                self.background
                    .retain(|queued| pending_pid_key(queued) != Some(key));
                self.pending_pid_keys.remove(&key);
            }
        }

        let priority = action_priority(&action);
        if priority != ActionPriority::Urgent
            && self.normal.len() + self.background.len() >= self.capacity
        {
            if priority == ActionPriority::Normal {
                if let Some(evicted) = self.background.pop_back() {
                    self.release_pending_key(&evicted);
                    self.background_evictions = self.background_evictions.saturating_add(1);
                } else {
                    self.capacity_drops = self.capacity_drops.saturating_add(1);
                    return false;
                }
            } else {
                self.capacity_drops = self.capacity_drops.saturating_add(1);
                return false;
            }
        }

        if let Some(key) = pending_key {
            self.pending_pid_keys.insert(key, thread_qos_rank(&action));
        }
        match priority {
            ActionPriority::Urgent => self.urgent.push_back(action),
            ActionPriority::Normal => self.normal.push_back(action),
            ActionPriority::Background => self.background.push_back(action),
        }
        true
    }

    /// Push all actions from a `Vec` into the queue in order.
    pub fn push_all(&mut self, actions: Vec<RootAction>) {
        for a in actions {
            let _ = self.push(a);
        }
    }

    fn release_pending_key(&mut self, action: &RootAction) {
        if let Some(key) = pending_pid_key(action) {
            self.pending_pid_keys.remove(&key);
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
            self.release_pending_key(&a);
            out.push(a);
        }

        // 2. Fill normal up to max_per_cycle.
        let mut budget = self.max_per_cycle;
        while budget > 0 {
            match self.normal.pop_front() {
                Some(a) => {
                    self.release_pending_key(&a);
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
                    self.release_pending_key(&a);
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

    /// Cumulative non-urgent actions rejected at the hard capacity limit.
    pub fn capacity_drops(&self) -> u64 {
        self.capacity_drops
    }

    /// Cumulative background actions displaced by normal-priority work.
    pub fn background_evictions(&self) -> u64 {
        self.background_evictions
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::audit_types::DecisionReason;

    fn make_unfreeze(pid: u32) -> RootAction {
        RootAction::UnfreezeProcess {
            pid,
            name: format!("proc{}", pid),
            reason: "test".to_string(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
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
            decision_reason: DecisionReason::PressureContext,
        }
    }

    fn make_sysctl(key: &str) -> RootAction {
        RootAction::set_sysctl(
            key.to_string(),
            "1",
            "test",
            DecisionReason::PressureContext,
        )
    }

    fn make_freeze(pid: u32) -> RootAction {
        RootAction::FreezeProcess {
            pid,
            name: format!("proc{}", pid),
            reason: "test".to_string(),
            start_sec: 0,
            start_usec: 0,
            decision_reason: DecisionReason::PressureContext,
        }
    }

    fn make_thread_qos(pid: u32, tier: &str) -> RootAction {
        RootAction::SetThreadQoS {
            pid,
            name: format!("proc{}", pid),
            thread_index: 1,
            tier: tier.to_string(),
            reason: "test".to_string(),
            decision_reason: DecisionReason::ThreadQoSRouting,
            affinity_tag: None,
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
        assert!(matches!(
            cycle[0],
            RootAction::UnfreezeProcess { pid: 99, .. }
        ));
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
    fn backpressure_ratio_reaches_one_at_hard_capacity() {
        let mut q = ActionQueue::new(2, 3);
        for i in 0..10 {
            q.push(make_throttle(i));
        }
        assert_eq!(q.backpressure_ratio(), 1.0);
        assert_eq!(q.normal_len(), 3);
        assert_eq!(q.capacity_drops(), 7);
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
    fn latency_and_anomaly_thread_qos_cannot_starve_in_background() {
        assert_eq!(
            action_priority(&make_thread_qos(1, "interactive")),
            ActionPriority::Normal
        );
        assert_eq!(
            action_priority(&make_thread_qos(2, "utility")),
            ActionPriority::Normal
        );
        assert_eq!(
            action_priority(&make_thread_qos(3, "background")),
            ActionPriority::Background
        );
    }

    #[test]
    fn empty_drain_returns_empty_vec() {
        let mut q = ActionQueue::new(10, 100);
        assert!(q.drain_cycle().is_empty());
    }

    #[test]
    fn duplicate_pid_action_is_not_queued_twice() {
        let mut q = ActionQueue::new(10, 100);
        assert!(q.push(make_throttle(42)));
        assert!(!q.push(make_throttle(42)));
        assert_eq!(q.normal_len(), 1);
    }

    #[test]
    fn drained_pid_action_can_be_queued_again() {
        let mut q = ActionQueue::new(10, 100);
        assert!(q.push(make_throttle(42)));
        assert_eq!(q.drain_cycle().len(), 1);
        assert!(q.push(make_throttle(42)));
        assert_eq!(q.normal_len(), 1);
    }

    #[test]
    fn thread_qos_dedup_is_scoped_to_thread_index() {
        let mut first = make_thread_qos(42, "interactive");
        let mut second = make_thread_qos(42, "interactive");
        if let RootAction::SetThreadQoS { thread_index, .. } = &mut first {
            *thread_index = 1;
        }
        if let RootAction::SetThreadQoS { thread_index, .. } = &mut second {
            *thread_index = 2;
        }

        let mut q = ActionQueue::new(10, 100);
        assert!(q.push(first));
        assert!(q.push(second));
        assert_eq!(q.normal_len(), 2);
    }

    #[test]
    fn interactive_thread_qos_replaces_pending_background_route() {
        let mut q = ActionQueue::new(10, 100);
        assert!(q.push(make_thread_qos(42, "background")));
        assert!(q.push(make_thread_qos(42, "interactive")));
        assert_eq!(q.background_len(), 0);
        assert_eq!(q.normal_len(), 1);

        let drained = q.drain_cycle();
        assert!(matches!(
            &drained[0],
            RootAction::SetThreadQoS { tier, .. } if tier == "interactive"
        ));
    }

    #[test]
    fn background_thread_qos_cannot_replace_pending_interactive_route() {
        let mut q = ActionQueue::new(10, 100);
        assert!(q.push(make_thread_qos(42, "interactive")));
        assert!(!q.push(make_thread_qos(42, "background")));
        assert_eq!(q.normal_len(), 1);
        assert_eq!(q.background_len(), 0);
    }

    #[test]
    fn normal_work_evicts_background_at_capacity() {
        let mut q = ActionQueue::new(10, 2);
        assert!(q.push(make_sysctl("first")));
        assert!(q.push(make_sysctl("second")));
        assert!(q.push(make_throttle(42)));

        assert_eq!(q.len(), 2);
        assert_eq!(q.normal_len(), 1);
        assert_eq!(q.background_len(), 1);
        assert_eq!(q.background_evictions(), 1);
        assert_eq!(q.capacity_drops(), 0);
    }

    #[test]
    fn evicted_pid_key_can_be_queued_again() {
        let mut q = ActionQueue::new(10, 1);
        assert!(q.push(make_thread_qos(42, "background")));
        assert!(q.push(make_throttle(7)));
        assert_eq!(q.drain_cycle().len(), 1);
        assert!(q.push(make_thread_qos(42, "interactive")));

        assert_eq!(q.normal_len(), 1);
        assert_eq!(q.background_len(), 0);
        assert_eq!(q.background_evictions(), 1);
    }

    #[test]
    fn urgent_work_bypasses_zero_capacity() {
        let mut q = ActionQueue::new(10, 0);
        assert!(!q.push(make_throttle(1)));
        assert!(q.push(make_unfreeze(1)));
        assert_eq!(q.urgent_len(), 1);
        assert_eq!(q.capacity_drops(), 1);
    }
}
