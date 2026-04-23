//! BlockedActionJournal — dark-matter observability for actions gated out.
//!
//! Every safety/policy gate that prevents an action from executing emits a
//! `BlockedActionEvent` here. Downstream learning (OutcomeTracker,
//! RL reward) can then correlate blocked decisions with t+30s/t+120s outcomes
//! (e.g. OOM, thrashing spike) to discover gates that are too conservative.
//!
//! Candidate emitters (wired in a follow-up commit):
//!   • `user_context::UserContext::freeze_protected` — when it returns `true`,
//!     the caller should emit `BlockerKind::UserContextAssertion` (or
//!     `HardProtection` if `call_in_progress`). `freeze_protected` itself is a
//!     pure predicate with no journal handle, so emission stays at the call site.
//!   • `execute_actions` per-PID guards — PidInvalid, BudgetExhausted, thermal
//!     and interrupt phases.
//!   • `decide_actions` safety filters — ForegroundFamily, HardProtection.
//!
//! [Bengio 2013] Counterfactual reasoning requires observing the COUNTERFACTUAL,
//! not just the taken action. [Nygard 2018 §8.5] Adaptive capacity limits.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BlockerKind {
    /// Hard-protected name (classify_protection Unconditional).
    HardProtection,
    /// User-context sleep assertion, call, or recently-active.
    UserContextAssertion,
    /// Foreground family or foreground-app name match.
    ForegroundFamily,
    /// Thermal emergency or resource-interrupt phase.
    ThermalOrInterrupt,
    /// Per-cycle action budget exhausted.
    BudgetExhausted,
    /// PID validation failed (dead or recycled).
    PidInvalid,
    /// Epistemic uncertainty too high.
    EpistemicHigh,
    /// Other — free-form reason.
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockedActionEvent {
    pub t: DateTime<Utc>,
    pub action_kind: String, // "Freeze" / "Throttle" / "Boost"
    pub target_name: String,
    pub target_pid: Option<u32>,
    pub blocker: BlockerKind,
    /// Snapshot of relevant pressure indicators at block time.
    pub pressure: f64,
    pub swap_gb: f64,
    pub thrashing_score: f64,
    pub p_oom_30s: Option<f64>,
}

impl BlockedActionEvent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        action_kind: impl Into<String>,
        target_name: impl Into<String>,
        target_pid: Option<u32>,
        blocker: BlockerKind,
        pressure: f64,
        swap_gb: f64,
        thrashing_score: f64,
        p_oom_30s: Option<f64>,
    ) -> Self {
        Self {
            t: Utc::now(),
            action_kind: action_kind.into(),
            target_name: target_name.into(),
            target_pid,
            blocker,
            pressure,
            swap_gb,
            thrashing_score,
            p_oom_30s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_constructs_and_serializes() {
        let e = BlockedActionEvent::new(
            "Freeze",
            "firefox",
            Some(1234),
            BlockerKind::UserContextAssertion,
            0.66,
            1.07,
            29_599.0,
            Some(0.40),
        );
        let json = serde_json::to_string(&e).expect("serializes");
        assert!(json.contains("\"UserContextAssertion\""));
        assert!(json.contains("\"firefox\""));
        let back: BlockedActionEvent = serde_json::from_str(&json).expect("roundtrips");
        assert_eq!(back.blocker, BlockerKind::UserContextAssertion);
        assert_eq!(back.target_pid, Some(1234));
    }

    #[test]
    fn blocker_kind_other_holds_reason() {
        let b = BlockerKind::Other("custom-gate".to_string());
        let json = serde_json::to_string(&b).unwrap();
        assert!(json.contains("custom-gate"));
    }
}
