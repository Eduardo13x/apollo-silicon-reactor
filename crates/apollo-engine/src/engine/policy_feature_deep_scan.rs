//! Deep Scan cost feature — hot-page awareness for freeze/throttle cost.
//!
//! When `hot_page_fraction` indicates a memory-hot process, freezing is
//! expensive: resumption will cause a page-in storm proportional to
//! working-set size. Cost contribution is `hot_page_fraction × (WSS/100MB)`
//! times an action-kind scale. Cold processes contribute near-zero cost —
//! freeze aggressively.
//!
//! Unwired in this commit — feature is pluggable into `PolicyScorer` via
//! `.builder().feature(DeepScanCostFeature::default())`. Registration in
//! the daemon happens in a follow-up wiring commit once `ActionContext` is
//! populated with deep scan data.
//!
//! Papers:
//! - [Denning 1968] Working Set Model — resumption fault storm.
//! - [Jiang 2021 MEMTIS] page-tier classification for tiered memory.
//! - [Bergman 2010 DAMON] adaptive working-set sampling.
//!
//! Cost model:
//! ```text
//! if hot_page_fraction.is_some() && wss_mb.is_some():
//!     cost = hot_cost_scale * hot_fraction * (wss_mb / wss_ref_mb) * action_scale
//!     uncertainty = 0.0
//! else:
//!     cost = 0.0
//!     uncertainty = missing_uncertainty
//! ```
//!
//! Action scales: `FreezeProcess` = 1.0, `ThrottleProcess` = 0.3, all
//! others (including `BoostProcess`) = 0.0. No hard veto is ever issued —
//! cost is soft and may be outweighed by pressure benefits.

use crate::engine::action_policy::{ActionContext, Contribution, PolicyFeature};
use crate::engine::types::RootAction;

/// Hot-page cost feature. See module docs.
#[derive(Debug, Clone)]
pub struct DeepScanCostFeature {
    /// Cost multiplier for hot-page fraction (default 2.0).
    /// At `hot_fraction=1.0` and `wss_ref=100MB`, cost = `2.0 × wss_mb/100`.
    pub hot_cost_scale: f64,
    /// Reference WSS in MB for normalizing (default 100.0).
    pub wss_ref_mb: f64,
    /// Uncertainty when neither `hot_page_fraction` nor `wss_mb` is available (default 0.2).
    pub missing_uncertainty: f64,
}

impl Default for DeepScanCostFeature {
    fn default() -> Self {
        Self {
            hot_cost_scale: 2.0,
            wss_ref_mb: 100.0,
            missing_uncertainty: 0.2,
        }
    }
}

impl DeepScanCostFeature {
    /// Action-kind scale factor. Freeze is the full cost; throttle is cheaper
    /// (QoS Background doesn't evict pages); boost / any other action has
    /// no hot-page cost.
    fn action_scale(action: &RootAction) -> f64 {
        match action {
            RootAction::FreezeProcess { .. } => 1.0,
            RootAction::ThrottleProcess { .. } => 0.3,
            _ => 0.0,
        }
    }
}

impl PolicyFeature for DeepScanCostFeature {
    fn name(&self) -> &'static str {
        "deep_scan_cost"
    }

    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution {
        let scale = Self::action_scale(action);
        // For actions outside the freeze/throttle family, there is nothing
        // to penalize — and we don't want to inject uncertainty either.
        if scale == 0.0 {
            return Contribution::zero();
        }

        match (ctx.hot_page_fraction, ctx.wss_mb) {
            (Some(hot), Some(wss)) => {
                let hot = hot.clamp(0.0, 1.0);
                let wss = wss.max(0.0);
                let wss_ref = self.wss_ref_mb.max(f64::EPSILON);
                let cost = self.hot_cost_scale * hot * (wss / wss_ref) * scale;
                Contribution {
                    benefit: 0.0,
                    cost: cost.max(0.0),
                    uncertainty: 0.0,
                    hard_veto: false,
                }
            }
            _ => Contribution {
                benefit: 0.0,
                cost: 0.0,
                uncertainty: self.missing_uncertainty.max(0.0),
                hard_veto: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::audit_types::DecisionReason;
    use crate::engine::safety::ProtectionLevel;

    /// Build an `ActionContext` with sensible defaults; tests override what they need.
    fn make_ctx(hot_page_fraction: Option<f64>, wss_mb: Option<f64>) -> ActionContext {
        ActionContext {
            pressure: 0.5,
            swap_gb: 0.0,
            thrashing_score: 0.0,
            p_oom_30s: None,
            p_jank_60s: None,
            has_sleep_assertion: false,
            call_in_progress: false,
            idle_secs: 0.0,
            foreground_pid: None,
            is_foreground_family: false,
            is_recently_active: false,
            thermal_emergency: false,
            interrupt_phase: 0,
            protection_level: ProtectionLevel::Unprotected,
            hot_page_fraction,
            wss_mb,
            sensor_age_ms: None,
            epistemic_uncertainty: 0.0,
        }
    }

    fn freeze() -> RootAction {
        RootAction::FreezeProcess {
            pid: 1234,
            name: "test".into(),
            reason: "t".into(),
            start_sec: 0,
            start_usec: 0,
            decision_reason: DecisionReason::PressureContext,
        }
    }

    fn throttle() -> RootAction {
        RootAction::ThrottleProcess {
            pid: 1234,
            name: "test".into(),
            aggressive: false,
            reason: "t".into(),
            start_sec: 0,
            start_usec: 0,
            decision_reason: DecisionReason::PressureContext,
        }
    }

    fn boost() -> RootAction {
        RootAction::BoostProcess {
            pid: 1234,
            name: "test".into(),
            reason: "t".into(),
            decision_reason: DecisionReason::PressureContext,
        }
    }

    #[test]
    fn default_feature_has_name() {
        let f = DeepScanCostFeature::default();
        assert_eq!(f.name(), "deep_scan_cost");
    }

    #[test]
    fn cold_process_contributes_near_zero_cost() {
        // hot=0.05, wss=200 → 2.0 * 0.05 * 2.0 * 1.0 = 0.20  < 0.3
        let f = DeepScanCostFeature::default();
        let ctx = make_ctx(Some(0.05), Some(200.0));
        let c = f.contribute(&freeze(), &ctx);
        assert!(c.cost < 0.3, "cold cost should be < 0.3, got {}", c.cost);
        assert_eq!(c.uncertainty, 0.0);
        assert!(!c.hard_veto);
    }

    #[test]
    fn hot_process_contributes_large_cost() {
        // hot=0.90, wss=500 → 2.0 * 0.90 * 5.0 * 1.0 = 9.0  > 5.0
        let f = DeepScanCostFeature::default();
        let ctx = make_ctx(Some(0.90), Some(500.0));
        let c = f.contribute(&freeze(), &ctx);
        assert!(c.cost > 5.0, "hot cost should be > 5.0, got {}", c.cost);
        assert_eq!(c.uncertainty, 0.0);
    }

    #[test]
    fn throttle_cost_is_30pc_of_freeze_cost() {
        let f = DeepScanCostFeature::default();
        let ctx = make_ctx(Some(0.50), Some(400.0));
        let cf = f.contribute(&freeze(), &ctx).cost;
        let ct = f.contribute(&throttle(), &ctx).cost;
        assert!(cf > 0.0);
        let ratio = ct / cf;
        assert!(
            (ratio - 0.3).abs() < 1e-9,
            "throttle/freeze ratio should be 0.3, got {ratio}"
        );
    }

    #[test]
    fn boost_has_zero_cost() {
        let f = DeepScanCostFeature::default();
        let ctx = make_ctx(Some(0.95), Some(1000.0));
        let c = f.contribute(&boost(), &ctx);
        assert_eq!(c.cost, 0.0);
        assert_eq!(c.benefit, 0.0);
        assert_eq!(c.uncertainty, 0.0);
        assert!(!c.hard_veto);
    }

    #[test]
    fn missing_hot_page_fraction_adds_uncertainty() {
        let f = DeepScanCostFeature::default();
        let ctx = make_ctx(None, Some(200.0));
        let c = f.contribute(&freeze(), &ctx);
        assert!(c.uncertainty > 0.0);
        assert_eq!(c.cost, 0.0);
    }

    #[test]
    fn missing_wss_adds_uncertainty() {
        let f = DeepScanCostFeature::default();
        let ctx = make_ctx(Some(0.5), None);
        let c = f.contribute(&freeze(), &ctx);
        assert!(c.uncertainty > 0.0);
        assert_eq!(c.cost, 0.0);
    }

    #[test]
    fn both_present_no_uncertainty() {
        let f = DeepScanCostFeature::default();
        let ctx = make_ctx(Some(0.4), Some(150.0));
        let c = f.contribute(&freeze(), &ctx);
        assert_eq!(c.uncertainty, 0.0);
    }

    #[test]
    fn no_hard_veto_ever() {
        // Deterministic pseudo-random LCG — avoids a rand dep.
        let f = DeepScanCostFeature::default();
        let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        for _ in 0..100 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let hot_bits = (state >> 33) as u32;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let wss_bits = (state >> 33) as u32;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let pick_hot_none = (state >> 60) & 1 == 1;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let pick_wss_none = (state >> 60) & 1 == 1;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let action_kind = (state >> 60) % 3;

            let hot = if pick_hot_none {
                None
            } else {
                Some((hot_bits as f64 / u32::MAX as f64) * 1.5 - 0.25) // allow out-of-range to exercise clamp
            };
            let wss = if pick_wss_none {
                None
            } else {
                Some((wss_bits as f64 / u32::MAX as f64) * 2000.0 - 100.0) // allow negatives
            };
            let ctx = make_ctx(hot, wss);
            let action = match action_kind {
                0 => freeze(),
                1 => throttle(),
                _ => boost(),
            };
            let c = f.contribute(&action, &ctx);
            assert!(!c.hard_veto, "feature must never hard-veto");
            assert!(c.cost >= 0.0, "cost must be non-negative, got {}", c.cost);
            assert!(
                c.uncertainty >= 0.0,
                "uncertainty must be non-negative, got {}",
                c.uncertainty
            );
        }
    }
}
