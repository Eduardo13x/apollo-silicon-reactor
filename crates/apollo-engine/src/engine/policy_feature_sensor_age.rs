//! Sensor-age uncertainty feature — penalize decisions made on stale data.
//!
//! Every ActionContext carries `sensor_age_ms: Option<u64>` — time since
//! the slowest sensor in the fused snapshot was collected. Uncertainty
//! rises monotonically with age; above `stale_threshold_ms`, it saturates.
//!
//! This does NOT veto — stale data is often the ONLY data available. It
//! just raises the cost of confident action, so the scorer's threshold
//! becomes effectively harder to clear. If pressure is catastrophic,
//! benefit still wins. If it's marginal, staleness tips it toward Observe.
//!
//! Papers: [Hellerstein 2004 §9] sensor delay in feedback control;
//! [Lakshminarayanan 2017] epistemic uncertainty from incomplete observation.

use crate::engine::action_policy::{ActionContext, Contribution, PolicyFeature};
use crate::engine::types::RootAction;

/// F7 — SensorAgeFeature: scorer uncertainty scales with `sensor_age_ms`.
#[derive(Debug, Clone)]
pub struct SensorAgeFeature {
    /// Uncertainty contribution when age is 0ms (default 0.0 — fresh data, no penalty).
    pub baseline_uncertainty: f64,
    /// Age (ms) at which uncertainty saturates at `saturation_uncertainty` (default 2000ms).
    pub stale_threshold_ms: u64,
    /// Max uncertainty contribution (default 1.5).
    pub saturation_uncertainty: f64,
    /// Uncertainty when `sensor_age_ms` is None (we don't know the age, assume worst-case)
    /// (default 1.0).
    pub unknown_age_uncertainty: f64,
}

impl Default for SensorAgeFeature {
    fn default() -> Self {
        Self {
            baseline_uncertainty: 0.0,
            stale_threshold_ms: 2000,
            saturation_uncertainty: 1.5,
            unknown_age_uncertainty: 1.0,
        }
    }
}

impl PolicyFeature for SensorAgeFeature {
    fn name(&self) -> &'static str {
        "sensor_age"
    }

    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution {
        // BoostProcess: boost is safe regardless of age — zero contribution.
        if matches!(action, RootAction::BoostProcess { .. }) {
            return Contribution::zero();
        }

        // Freeze/Throttle and all other actions — scale uncertainty by age.
        let uncertainty = match ctx.sensor_age_ms {
            None => self.unknown_age_uncertainty,
            Some(age_ms) => {
                // Clamp age to stale_threshold; beyond that, saturate.
                let capped_age = age_ms.min(self.stale_threshold_ms);
                if self.stale_threshold_ms == 0 {
                    // Degenerate config: any age is "stale".
                    self.saturation_uncertainty
                } else {
                    let ratio = (capped_age as f64) / (self.stale_threshold_ms as f64);
                    let span = self.saturation_uncertainty - self.baseline_uncertainty;
                    let raw = self.baseline_uncertainty + ratio * span;
                    raw.clamp(self.baseline_uncertainty, self.saturation_uncertainty)
                }
            }
        };

        Contribution {
            benefit: 0.0,
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

    fn ctx_with_age(age: Option<u64>) -> ActionContext {
        ActionContext {
            pressure: 0.5,
            swap_gb: 0.0,
            learned_yield: None,
            imagined_margin: None,
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
            hot_page_fraction: None,
            wss_mb: None,
            sensor_age_ms: age,
            epistemic_uncertainty: 0.0,
            is_on_battery: None,
            wakeups_per_sec: None,
            ctx_switches_per_sec: None,
        }
    }

    fn freeze() -> RootAction {
        RootAction::freeze(
            1234,
            "testproc",
            "unit-test",
            DecisionReason::PressureContext,
        )
    }

    fn throttle() -> RootAction {
        RootAction::throttle(
            1234,
            "testproc",
            false,
            "unit-test",
            DecisionReason::PressureContext,
        )
    }

    fn boost() -> RootAction {
        RootAction::BoostProcess {
            pid: 1234,
            name: "testproc".into(),
            reason: "unit-test".into(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }
    }

    #[test]
    fn default_feature_has_name() {
        let f = SensorAgeFeature::default();
        assert_eq!(f.name(), "sensor_age");
    }

    #[test]
    fn fresh_data_zero_uncertainty() {
        let f = SensorAgeFeature::default();
        let c = f.contribute(&freeze(), &ctx_with_age(Some(0)));
        assert!(
            (c.uncertainty - f.baseline_uncertainty).abs() < 1e-9,
            "expected baseline {}, got {}",
            f.baseline_uncertainty,
            c.uncertainty
        );
    }

    #[test]
    fn stale_data_saturates_uncertainty() {
        let f = SensorAgeFeature::default();
        let c = f.contribute(&freeze(), &ctx_with_age(Some(5000)));
        assert!(
            (c.uncertainty - f.saturation_uncertainty).abs() < 1e-9,
            "expected saturation {}, got {}",
            f.saturation_uncertainty,
            c.uncertainty
        );
    }

    #[test]
    fn mid_age_linearly_interpolated() {
        let f = SensorAgeFeature::default();
        // 1000ms is half of stale_threshold_ms (2000ms).
        let c = f.contribute(&freeze(), &ctx_with_age(Some(1000)));
        let expected =
            f.baseline_uncertainty + 0.5 * (f.saturation_uncertainty - f.baseline_uncertainty);
        assert!(
            (c.uncertainty - expected).abs() < 1e-9,
            "expected {}, got {}",
            expected,
            c.uncertainty
        );
        // 0.75 given default params.
        assert!((c.uncertainty - 0.75).abs() < 1e-9);
    }

    #[test]
    fn unknown_age_uses_fallback() {
        let f = SensorAgeFeature::default();
        let c = f.contribute(&freeze(), &ctx_with_age(None));
        assert!(
            (c.uncertainty - f.unknown_age_uncertainty).abs() < 1e-9,
            "expected {}, got {}",
            f.unknown_age_uncertainty,
            c.uncertainty
        );
    }

    #[test]
    fn boost_has_zero_uncertainty_regardless_of_age() {
        let f = SensorAgeFeature::default();
        for age in [None, Some(0), Some(500), Some(2000), Some(99_999)] {
            let c = f.contribute(&boost(), &ctx_with_age(age));
            assert_eq!(c.benefit, 0.0);
            assert_eq!(c.cost, 0.0);
            assert_eq!(c.uncertainty, 0.0);
            assert!(!c.hard_veto);
        }
    }

    #[test]
    fn throttle_scales_same_as_freeze() {
        let f = SensorAgeFeature::default();
        for age in [None, Some(0), Some(500), Some(1500), Some(2000), Some(9999)] {
            let fc = f.contribute(&freeze(), &ctx_with_age(age));
            let tc = f.contribute(&throttle(), &ctx_with_age(age));
            assert!(
                (fc.uncertainty - tc.uncertainty).abs() < 1e-9,
                "freeze {} vs throttle {} diverged at age {:?}",
                fc.uncertainty,
                tc.uncertainty,
                age
            );
        }
    }

    #[test]
    fn no_benefit_no_cost_no_veto() {
        let f = SensorAgeFeature::default();
        // Deterministic "random" inputs — cycle over action kinds & ages.
        let actions = [freeze(), throttle(), boost()];
        for i in 0..100u64 {
            let age = if i % 7 == 0 {
                None
            } else {
                Some((i.wrapping_mul(131).wrapping_add(17)) % 6000)
            };
            let action = &actions[(i as usize) % actions.len()];
            let c = f.contribute(action, &ctx_with_age(age));
            assert_eq!(c.benefit, 0.0, "iter {i}: benefit nonzero");
            assert_eq!(c.cost, 0.0, "iter {i}: cost nonzero");
            assert!(!c.hard_veto, "iter {i}: veto set");
        }
    }

    #[test]
    fn monotonic_in_age() {
        let f = SensorAgeFeature::default();
        let ages: Vec<u64> = (0..=f.stale_threshold_ms).step_by(50).collect();
        let mut prev = f.baseline_uncertainty - 1e-9;
        for &a in &ages {
            let c = f.contribute(&freeze(), &ctx_with_age(Some(a)));
            assert!(
                c.uncertainty + 1e-9 >= prev,
                "non-monotonic at age {a}: prev={prev} got={}",
                c.uncertainty
            );
            prev = c.uncertainty;
        }
    }
}
