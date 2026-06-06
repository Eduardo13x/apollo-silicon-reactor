//! Predictive benefit feature — acts on p_oom_30s / p_jank_60s before crisis.
//!
//! The reactive gate tower acted only on current-state snapshots. This
//! feature adds benefit directly proportional to predicted OOM probability
//! and jank probability at 30s/60s horizons, so the scorer can authorize
//! freezes BEFORE thrashing saturates.
//!
//! Scaling:
//!   benefit = oom_weight × p_oom_30s + jank_weight × p_jank_60s
//!   (both terms gated by action kind — only adds benefit to Freeze/Throttle)
//!
//! When either prediction is None, the feature contributes uncertainty
//! (we're flying blind on the predictive horizon).
//!
//! Papers: [Camacho 2007] MPC; [Riedmiller 2005] anticipatory Q-learning.

use crate::engine::action_policy::{ActionContext, Contribution, PolicyFeature};
use crate::engine::types::RootAction;

/// Relative benefit multiplier for `ThrottleProcess` vs `FreezeProcess`.
/// Throttle reclaims less memory (it slows, not suspends), so 0.7× the
/// freeze weight — derived from empirical reclaim ratios in outcome_tracker.
const THROTTLE_BENEFIT_MULT: f64 = 0.7;

#[derive(Debug, Clone)]
pub struct PredictiveBenefitFeature {
    /// Benefit multiplier for predicted OOM probability (default 3.0).
    pub oom_weight: f64,
    /// Benefit multiplier for predicted jank probability (default 1.0).
    pub jank_weight: f64,
    /// Threshold above which predictions are considered actionable (default 0.15).
    /// Below this, contribution is zero — avoid acting on low-signal predictions.
    pub action_threshold: f64,
    /// Uncertainty when predictions are missing (default 0.3).
    pub missing_uncertainty: f64,
}

impl Default for PredictiveBenefitFeature {
    fn default() -> Self {
        Self {
            oom_weight: 3.0,
            jank_weight: 1.0,
            action_threshold: 0.15,
            missing_uncertainty: 0.3,
        }
    }
}

impl PolicyFeature for PredictiveBenefitFeature {
    fn name(&self) -> &'static str {
        "predictive_benefit"
    }

    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution {
        // Per-kind multiplier: Freeze = 1.0, Throttle = 0.7×, all others = 0.
        // Boost and system-level actions (Sysctl, Spotlight, QoS, …) have
        // their own benefit logic elsewhere — this feature stays silent.
        let kind_mult = match action {
            RootAction::FreezeProcess { .. } => 1.0,
            RootAction::ThrottleProcess { .. } => THROTTLE_BENEFIT_MULT,
            _ => return Contribution::zero(),
        };

        // Gather contributions from each horizon independently. Missing
        // predictions add uncertainty; sub-threshold predictions are silent.
        let mut benefit = 0.0f64;
        let mut uncertainty = 0.0f64;

        match ctx.p_oom_30s {
            Some(p) if p >= self.action_threshold => {
                benefit += self.oom_weight * p;
            }
            Some(_) => { /* below threshold — ignore as noise */ }
            None => {
                uncertainty += self.missing_uncertainty;
            }
        }

        match ctx.p_jank_60s {
            Some(p) if p >= self.action_threshold => {
                benefit += self.jank_weight * p;
            }
            Some(_) => { /* below threshold — ignore as noise */ }
            None => {
                uncertainty += self.missing_uncertainty;
            }
        }

        Contribution {
            benefit: benefit * kind_mult,
            cost: 0.0,
            uncertainty,
            hard_veto: false,
            ..Contribution::zero()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::audit_types::DecisionReason;
    use crate::engine::safety::ProtectionLevel;

    fn base_ctx() -> ActionContext {
        ActionContext {
            pressure: 0.40,
            swap_gb: 0.5,
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
            is_on_battery: None,
            wakeups_per_sec: None,
            ctx_switches_per_sec: None,
        }
    }

    fn freeze(pid: u32) -> RootAction {
        RootAction::freeze(
            pid,
            format!("p{pid}"),
            "test",
            DecisionReason::PressureContext,
        )
    }

    fn throttle(pid: u32) -> RootAction {
        RootAction::throttle(
            pid,
            format!("p{pid}"),
            false,
            "test",
            DecisionReason::PressureContext,
        )
    }

    fn boost(pid: u32) -> RootAction {
        RootAction::BoostProcess {
            pid,
            name: format!("p{pid}"),
            reason: "test".into(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }
    }

    #[test]
    fn default_feature_has_name() {
        let f = PredictiveBenefitFeature::default();
        assert_eq!(f.name(), "predictive_benefit");
    }

    #[test]
    fn high_p_oom_adds_large_benefit_to_freeze() {
        let f = PredictiveBenefitFeature::default();
        let mut ctx = base_ctx();
        ctx.p_oom_30s = Some(0.50);
        ctx.p_jank_60s = None; // uncertainty only
        let c = f.contribute(&freeze(1234), &ctx);
        // 3.0 × 0.50 = 1.50 → benefit > 1.0
        assert!(c.benefit > 1.0, "benefit={}", c.benefit);
        assert!(!c.hard_veto);
    }

    #[test]
    fn low_p_oom_below_threshold_contributes_zero() {
        let f = PredictiveBenefitFeature::default();
        let mut ctx = base_ctx();
        ctx.p_oom_30s = Some(0.10); // below 0.15
        ctx.p_jank_60s = Some(0.05); // also below
        let c = f.contribute(&freeze(1234), &ctx);
        assert_eq!(c.benefit, 0.0);
    }

    #[test]
    fn throttle_benefit_is_70pc_of_freeze_benefit() {
        let f = PredictiveBenefitFeature::default();
        let mut ctx = base_ctx();
        ctx.p_oom_30s = Some(0.60);
        ctx.p_jank_60s = Some(0.40);
        let cf = f.contribute(&freeze(1234), &ctx);
        let ct = f.contribute(&throttle(1234), &ctx);
        // ct.benefit should be ~0.7 × cf.benefit
        let ratio = ct.benefit / cf.benefit;
        assert!(
            (ratio - 0.7).abs() < 1e-9,
            "ratio={ratio} cf={} ct={}",
            cf.benefit,
            ct.benefit
        );
    }

    #[test]
    fn boost_ignores_predictive_signal() {
        let f = PredictiveBenefitFeature::default();
        let mut ctx = base_ctx();
        ctx.p_oom_30s = Some(0.90);
        ctx.p_jank_60s = Some(0.90);
        let c = f.contribute(&boost(1234), &ctx);
        // BoostProcess always returns Contribution::zero()
        assert_eq!(c.benefit, 0.0);
        assert_eq!(c.cost, 0.0);
        assert_eq!(c.uncertainty, 0.0);
        assert!(!c.hard_veto);
    }

    #[test]
    fn missing_both_predictions_adds_uncertainty() {
        let f = PredictiveBenefitFeature::default();
        let ctx = base_ctx(); // both p_* are None
        let c = f.contribute(&freeze(1234), &ctx);
        assert!(c.uncertainty > 0.0, "uncertainty={}", c.uncertainty);
        assert_eq!(c.benefit, 0.0);
    }

    #[test]
    fn jank_alone_still_contributes() {
        let f = PredictiveBenefitFeature::default();
        let mut ctx = base_ctx();
        ctx.p_oom_30s = None;
        ctx.p_jank_60s = Some(0.30); // above 0.15
        let c = f.contribute(&freeze(1234), &ctx);
        assert!(c.benefit > 0.0, "benefit={}", c.benefit);
        // missing p_oom should still add uncertainty
        assert!(c.uncertainty > 0.0);
    }

    #[test]
    fn benefit_scales_linearly_with_p_oom() {
        let f = PredictiveBenefitFeature::default();
        let mut ctx = base_ctx();
        ctx.p_jank_60s = Some(0.20); // constant

        ctx.p_oom_30s = Some(0.00);
        let b0 = f.contribute(&freeze(1234), &ctx).benefit;
        ctx.p_oom_30s = Some(0.30);
        let b30 = f.contribute(&freeze(1234), &ctx).benefit;
        ctx.p_oom_30s = Some(0.60);
        let b60 = f.contribute(&freeze(1234), &ctx).benefit;

        assert!(b60 > b30, "b60={b60} b30={b30}");
        assert!(b30 > b0, "b30={b30} b0={b0}");
    }

    #[test]
    fn no_hard_veto_ever() {
        let f = PredictiveBenefitFeature::default();
        // 100 pseudo-random combinations across all action kinds.
        // Simple LCG so the test is deterministic.
        let mut state: u64 = 0xCAFEBABEDEADBEEF;
        for i in 0..100 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let rnd = (state >> 33) as u32;

            let p_oom = if rnd & 1 == 0 {
                None
            } else {
                Some(((rnd >> 1) % 1001) as f64 / 1000.0) // 0.000..=1.000
            };
            let p_jank = if rnd & 2 == 0 {
                None
            } else {
                Some(((rnd >> 8) % 1001) as f64 / 1000.0)
            };

            let mut ctx = base_ctx();
            ctx.p_oom_30s = p_oom;
            ctx.p_jank_60s = p_jank;

            let action: RootAction = match i % 4 {
                0 => freeze(1000 + i as u32),
                1 => throttle(1000 + i as u32),
                2 => boost(1000 + i as u32),
                _ => RootAction::UnfreezeProcess {
                    pid: 1000 + i as u32,
                    name: format!("p{i}"),
                    reason: "test".to_string(),
                    decision_reason: DecisionReason::PressureContext,
                },
            };
            let c = f.contribute(&action, &ctx);
            assert!(!c.hard_veto, "iter {i}: hard_veto should always be false");
        }
    }
}
