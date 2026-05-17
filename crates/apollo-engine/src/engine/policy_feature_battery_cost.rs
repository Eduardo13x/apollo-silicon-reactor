//! Phase 5.2 — Battery-aware cost feature for the PolicyScorer.
//!
//! Adds a `cost` contribution proportional to expected energy waste of the
//! candidate action when running on battery. Reuses the pure
//! `energy::battery_aware_cost_penalty` (range [0.0, 0.20]) and reports it
//! as `Contribution::cost` — so the scorer treats it like any other UX-cost
//! signal (Phase 5.1 user-presence multiplier on confidence is orthogonal:
//! that one modulates AT specialist voting time; this one modulates AT
//! scoring/dispatch time).
//!
//! **Wiring contract:**
//!
//! - All three inputs (`is_on_battery`, `wakeups_per_sec`,
//!   `ctx_switches_per_sec`) come from `ActionContext` and are `Option`.
//!   Whenever ANY of them is `None` the feature returns `Contribution::zero()`
//!   — never inject a false-positive cost on incomplete telemetry. This
//!   matches the Phase 5.1 shadow-signals "WRITTEN-flag" pattern and keeps
//!   `cargo test` paths quiet.
//! - On AC power (`is_on_battery == Some(false)`) the penalty function
//!   itself returns 0.0; this feature still reports `Contribution::zero()`.
//! - `inc_battery_aware_penalty_emission` LSE counter bumps ONLY when the
//!   computed penalty is strictly positive — dashboards see real emissions,
//!   not zero-cost calls.
//!
//! **Multiplicative vs additive:** NotebookLM 2026-05-16 verdict prescribed
//! `cost *= (1 + battery_penalty)` to keep this stack-compatible with
//! Phase 3.1/5.1's multiplicative confidence modulators. The PolicyScorer's
//! aggregation IS additive (`Σ benefit − Σ cost`), so we honour that by
//! returning the penalty AS a cost contribution rather than wrapping the
//! aggregate. Net effect: a 0.10 penalty subtracts 0.10 from the composite,
//! same shape as `UserDisruptionCostFeature`.

use crate::engine::action_policy::{ActionContext, Contribution, PolicyFeature};
use crate::engine::energy::battery_aware_cost_penalty;
use crate::engine::lse_counters::LSE_COUNTERS;
use crate::engine::types::RootAction;

/// Phase 5.2 wiring — battery-aware cost feature.
#[derive(Debug, Clone, Default)]
pub struct BatteryAwareCostFeature;

impl PolicyFeature for BatteryAwareCostFeature {
    fn name(&self) -> &'static str {
        "battery_aware_cost"
    }

    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution {
        // BoostProcess is always energy-neutral or energy-saving (re-promotes
        // an interactive app to P-cores) — no battery cost.
        if matches!(action, RootAction::BoostProcess { .. }) {
            return Contribution::zero();
        }

        let (Some(on_batt), Some(wakeups), Some(ctxsw)) = (
            ctx.is_on_battery,
            ctx.wakeups_per_sec,
            ctx.ctx_switches_per_sec,
        ) else {
            // Missing any signal → no penalty (avoid false-positive cost
            // on test paths / shadow-mode probes that don't publish).
            return Contribution::zero();
        };

        let penalty = battery_aware_cost_penalty(on_batt, wakeups, ctxsw, ctx.pressure);
        if penalty <= 0.0 {
            return Contribution::zero();
        }
        // Real emission — bump the LSE counter so dashboards see the
        // feature actually firing.
        LSE_COUNTERS.inc_battery_aware_penalty_emission();
        Contribution {
            benefit: 0.0,
            cost: penalty,
            uncertainty: 0.0,
            hard_veto: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::audit_types::DecisionReason;
    use crate::engine::safety::ProtectionLevel;

    fn ctx_with(
        on_battery: Option<bool>,
        wakeups: Option<f64>,
        ctxsw: Option<f64>,
        pressure: f64,
    ) -> ActionContext {
        ActionContext {
            pressure,
            swap_gb: 0.0,
            thrashing_score: 0.0,
            p_oom_30s: None,
            p_jank_60s: None,
            has_sleep_assertion: false,
            call_in_progress: false,
            idle_secs: 60.0,
            foreground_pid: None,
            is_foreground_family: false,
            is_recently_active: false,
            thermal_emergency: false,
            interrupt_phase: 0,
            protection_level: ProtectionLevel::Unprotected,
            hot_page_fraction: None,
            wss_mb: None,
            sensor_age_ms: None,
            epistemic_uncertainty: 0.0,
            is_on_battery: on_battery,
            wakeups_per_sec: wakeups,
            ctx_switches_per_sec: ctxsw,
        }
    }

    fn throttle_action() -> RootAction {
        RootAction::ThrottleProcess {
            pid: 100,
            name: "test".into(),
            aggressive: false,
            reason: "test".into(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }
    }

    fn boost_action() -> RootAction {
        RootAction::BoostProcess {
            pid: 100,
            name: "test".into(),
            reason: "test".into(),
            decision_reason: DecisionReason::PressureContext,
        }
    }

    #[test]
    fn battery_feature_missing_signals_returns_zero() {
        let f = BatteryAwareCostFeature;
        let c = f.contribute(&throttle_action(), &ctx_with(None, None, None, 0.5));
        assert_eq!(c.cost, 0.0);
        assert!(!c.hard_veto);
    }

    #[test]
    fn battery_feature_ac_power_returns_zero() {
        let f = BatteryAwareCostFeature;
        let c = f.contribute(
            &throttle_action(),
            &ctx_with(Some(false), Some(100.0), Some(1000.0), 0.4),
        );
        assert_eq!(c.cost, 0.0);
    }

    #[test]
    fn battery_feature_on_battery_with_noise_emits_cost() {
        let f = BatteryAwareCostFeature;
        // On battery + low pressure + high wakeups → penalty fires.
        let c = f.contribute(
            &throttle_action(),
            &ctx_with(Some(true), Some(400.0), Some(5000.0), 0.30),
        );
        assert!(c.cost > 0.0);
        assert!(c.cost <= 0.20); // hard cap
    }

    #[test]
    fn battery_feature_boost_action_always_zero() {
        let f = BatteryAwareCostFeature;
        // Even with maximum noise on battery, boost is always free.
        let c = f.contribute(
            &boost_action(),
            &ctx_with(Some(true), Some(10_000.0), Some(100_000.0), 0.10),
        );
        assert_eq!(c.cost, 0.0);
    }
}
