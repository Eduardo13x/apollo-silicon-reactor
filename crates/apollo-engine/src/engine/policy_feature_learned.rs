//! Unification scaffold features (2026-06-11) — the inline decide_actions
//! gates (HRPO yield, world-model dominance) re-expressed as
//! [`PolicyFeature`]s so the scorer composes the SAME evidence the gates
//! act on. Phase: SHADOW — these features run per-candidate through
//! `ShadowEvaluator::evaluate_accepted` (log-only); the inline gates keep
//! final authority until the N>=500 disagreement-evidence mandate
//! (NotebookLM 2026-05-16, Candidate-C verdict) is met. The journal
//! disagreement stream this produces IS that evidence.
//!
//! [Saltzer & Kaashoek 2009 §3.3] complete mediation — one composition
//! point for every admission signal; [Sutton & Barto 2018 §2.6].

use crate::engine::action_policy::{ActionContext, Contribution, PolicyFeature};
use crate::engine::types::RootAction;

/// Cost from demonstrably low learned yield for this candidate's process
/// group. `ctx.learned_yield` = blend `0.5·effectiveness +
/// 0.5·predicted_effectiveness` (same blend as the inline graded gate).
/// None (class-level probe / no group data) → zero contribution.
pub struct LearnedYieldFeature;

impl PolicyFeature for LearnedYieldFeature {
    fn name(&self) -> &'static str {
        "learned_yield"
    }

    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution {
        let relevant = matches!(
            action,
            RootAction::FreezeProcess { .. }
                | RootAction::ThrottleProcess { .. }
                | RootAction::BoostProcess { .. }
        );
        let Some(y) = ctx.learned_yield.filter(|_| relevant) else {
            return Contribution::zero();
        };
        // Below ~0.55 blended yield the action class is historically poor
        // for this group; cost grows linearly, capped at 0.44 (strong but
        // not an unconditional veto — hard_veto stays the protection
        // feature's job).
        let cost = ((0.55 - y).max(0.0) * 0.8).min(0.44);
        Contribution {
            benefit: 0.0,
            cost,
            uncertainty: 0.05,
            hard_veto: false,
            ..Contribution::zero()
        }
    }
}

/// World-model imagination as scorer evidence. `ctx.imagined_margin`:
/// Some(m>0) = the calibrated model predicts the action beats do-nothing
/// by m pressure → benefit; Some(m<=0) = do-nothing dominates → cost;
/// None = Unknown/probe → zero (exploration must not be priced).
pub struct WorldModelFeature;

impl PolicyFeature for WorldModelFeature {
    fn name(&self) -> &'static str {
        "world_model"
    }

    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution {
        let relevant = matches!(
            action,
            RootAction::FreezeProcess { .. } | RootAction::ThrottleProcess { .. }
        );
        let Some(m) = ctx.imagined_margin.filter(|_| relevant) else {
            return Contribution::zero();
        };
        if m > 0.0 {
            // Margins are pressure deltas (~0.005–0.10); ×8 maps the
            // useful range onto [0, 0.4] of scorer benefit.
            Contribution {
                benefit: (m * 8.0).min(0.40),
                cost: 0.0,
                uncertainty: 0.05,
                hard_veto: false,
                ..Contribution::zero()
            }
        } else {
            // Dominated: flat cost — the model says this is side-effects
            // for nothing.
            Contribution {
                benefit: 0.0,
                cost: 0.30,
                uncertainty: 0.05,
                hard_veto: false,
                ..Contribution::zero()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::audit_types::DecisionReason;
    use crate::engine::safety::ProtectionLevel;

    fn ctx() -> ActionContext {
        ActionContext {
            pressure: 0.6,
            swap_gb: 2.0,
            learned_yield: None,
            imagined_margin: None,
            thrashing_score: 0.0,
            p_oom_30s: None,
            p_jank_60s: None,
            has_sleep_assertion: false,
            call_in_progress: false,
            idle_secs: 100.0,
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

    fn freeze() -> RootAction {
        RootAction::freeze_full(7, "x", "t", 0, 0, DecisionReason::PressureContext)
    }

    #[test]
    fn yield_feature_prices_low_yield_and_abstains_on_none() {
        let f = LearnedYieldFeature;
        assert_eq!(f.contribute(&freeze(), &ctx()).cost, 0.0, "None → zero");

        let mut c = ctx();
        c.learned_yield = Some(0.20); // Browser-like
        let contrib = f.contribute(&freeze(), &c);
        assert!((contrib.cost - 0.28).abs() < 1e-6, "low yield priced");

        c.learned_yield = Some(0.90); // healthy
        assert_eq!(f.contribute(&freeze(), &c).cost, 0.0, "good yield free");
    }

    #[test]
    fn world_model_feature_rewards_margin_and_prices_dominance() {
        let f = WorldModelFeature;
        assert_eq!(f.contribute(&freeze(), &ctx()).benefit, 0.0, "None → zero");

        let mut c = ctx();
        c.imagined_margin = Some(0.05);
        let win = f.contribute(&freeze(), &c);
        assert!((win.benefit - 0.40).abs() < 1e-6, "margin capped at 0.40");
        assert_eq!(win.cost, 0.0);

        c.imagined_margin = Some(-0.01);
        let lose = f.contribute(&freeze(), &c);
        assert_eq!(lose.cost, 0.30, "dominated priced flat");
        assert_eq!(lose.benefit, 0.0);
    }
}
