//! ActionPolicyScorer — composable decision surface replacing the gate tower.
//!
//! Every candidate `RootAction` is scored by a vector of `PolicyFeature`
//! implementations. Each feature contributes {benefit, cost, uncertainty,
//! hard_veto?}. The scorer aggregates and emits a `PolicyScore` with an
//! accept/reject decision and a reason string.
//!
//! **Shadow mode:** in the scaffold commit, the scorer is NOT wired into the
//! daemon main loop. Follow-up commits will run it alongside the existing
//! gate tower (logging disagreements to `BlockedActionJournal`), then cut
//! over once disagreement rate is bounded.
//!
//! # Motivation
//! The "Fix-N-minus-1" meta-pattern: every new failure mode spawned a new
//! bypass conditional. Gate C's thrashing-bypass (commit 69b0b8b) was the
//! 12th such fix in April. The root cause was that each gate was an
//! independent `if` branch with ad-hoc bypass chains — there was no
//! composable decision surface. Treating each gate as a *feature
//! contributing to a composite score* solves this structurally.
//!
//! # Theory
//! - [Puterman 1994] MDP — expected-utility decision frameworks: actions
//!   chosen by maximizing expected utility over composed value signals.
//! - [Ng 1999] potential-based reward shaping — safely inject new signals
//!   without changing the optimal policy.
//! - [Lakshminarayanan 2017] epistemic uncertainty estimation — weighted
//!   penalty for under-confident contributions.
//!
//! # Extension
//! Downstream features (F5 Deep Scan hot-page cost, F6 predictive
//! p_oom_30s, F7 sensor freshness) implement `PolicyFeature` and register
//! via `PolicyScorer::builder().feature(...)`. No core API change
//! required.

// Do NOT import from user_context, decide_actions, execute_actions — scaffold
// must be dependency-free of the current gate tower to keep migration clean.
use crate::engine::safety::ProtectionLevel;
use crate::engine::types::RootAction;

/// Context packaged for scoring one candidate action. All fields the scorer
/// needs for its decision. Fields here map 1-1 to current gate inputs.
#[derive(Debug, Clone)]
pub struct ActionContext {
    pub pressure: f64,
    pub swap_gb: f64,
    pub thrashing_score: f64,
    /// Predicted probability of OOM kill within the next 30s. F6 will wire.
    pub p_oom_30s: Option<f64>,
    /// Predicted probability of jank within the next 60s. F6 will wire.
    pub p_jank_60s: Option<f64>,
    pub has_sleep_assertion: bool,
    pub call_in_progress: bool,
    pub idle_secs: f64,
    pub foreground_pid: Option<u32>,
    pub is_foreground_family: bool,
    pub is_recently_active: bool,
    pub thermal_emergency: bool,
    /// 0..=3 — higher means deeper in an interrupt-driven grace period.
    pub interrupt_phase: u8,
    pub protection_level: ProtectionLevel,
    /// Fraction of process RSS that is "hot" (recently touched). F5 will wire.
    pub hot_page_fraction: Option<f64>,
    /// Working-set size in MiB from Deep Scan. F5 will wire.
    pub wss_mb: Option<f64>,
    /// Age of the freshest sensor sample feeding this decision, ms. F7 will wire.
    pub sensor_age_ms: Option<u64>,
    /// Global epistemic uncertainty ≥ 0; features may use or add to it.
    pub epistemic_uncertainty: f64,
}

/// A single feature's contribution to the composite score.
#[derive(Debug, Clone, Copy)]
pub struct Contribution {
    /// Expected reclaim / risk-reduction (≥0).
    pub benefit: f64,
    /// Expected user disruption (≥0).
    pub cost: f64,
    /// Feature's confidence penalty (≥0).
    pub uncertainty: f64,
    /// If true, the scorer MUST reject regardless of aggregate score.
    pub hard_veto: bool,
}

impl Contribution {
    pub fn zero() -> Self {
        Self {
            benefit: 0.0,
            cost: 0.0,
            uncertainty: 0.0,
            hard_veto: false,
        }
    }
}

/// Implement this trait to contribute to the action policy score.
///
/// Features are invoked in registration order. Their contributions are
/// summed (benefits, costs, uncertainties). Any feature may veto.
pub trait PolicyFeature: Send + Sync {
    /// Stable short name for logging (e.g., "protection", "pressure_benefit").
    fn name(&self) -> &'static str;
    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution;
}

/// Result of running the scorer on one action.
#[derive(Debug, Clone)]
pub struct PolicyScore {
    pub action_kind: &'static str,
    pub total_benefit: f64,
    pub total_cost: f64,
    pub total_uncertainty: f64,
    /// Feature name that fired `hard_veto`, if any.
    pub vetoed_by: Option<String>,
    pub accept: bool,
    /// Human-readable reason, for journal/dashboard.
    pub reason: String,
    pub per_feature: Vec<(&'static str, Contribution)>,
}

/// Composable decision surface. Construct via [`PolicyScorer::builder`].
pub struct PolicyScorer {
    features: Vec<Box<dyn PolicyFeature>>,
    /// Accept threshold: accept iff
    /// `benefit - λ_cost*cost - λ_unc*uncertainty ≥ threshold` AND no veto.
    threshold: f64,
    lambda_cost: f64,
    lambda_unc: f64,
}

impl PolicyScorer {
    pub fn builder() -> PolicyScorerBuilder {
        PolicyScorerBuilder::default()
    }

    /// Score an action against all registered features.
    pub fn score(&self, action: &RootAction, ctx: &ActionContext) -> PolicyScore {
        let action_kind = action_kind_name(action);
        let mut per_feature: Vec<(&'static str, Contribution)> =
            Vec::with_capacity(self.features.len());
        let mut total_benefit = 0.0f64;
        let mut total_cost = 0.0f64;
        let mut total_uncertainty = 0.0f64;
        let mut vetoed_by: Option<String> = None;

        for f in &self.features {
            let c = f.contribute(action, ctx);
            total_benefit += c.benefit.max(0.0);
            total_cost += c.cost.max(0.0);
            total_uncertainty += c.uncertainty.max(0.0);
            if c.hard_veto && vetoed_by.is_none() {
                vetoed_by = Some(f.name().to_string());
            }
            per_feature.push((f.name(), c));
        }

        let net =
            total_benefit - self.lambda_cost * total_cost - self.lambda_unc * total_uncertainty;
        let accept = vetoed_by.is_none() && net >= self.threshold;

        let reason = build_reason(
            action_kind,
            accept,
            net,
            self.threshold,
            total_benefit,
            total_cost,
            total_uncertainty,
            &vetoed_by,
            &per_feature,
        );

        PolicyScore {
            action_kind,
            total_benefit,
            total_cost,
            total_uncertainty,
            vetoed_by,
            accept,
            reason,
            per_feature,
        }
    }
}

/// Builder for [`PolicyScorer`].
#[derive(Default)]
pub struct PolicyScorerBuilder {
    features: Vec<Box<dyn PolicyFeature>>,
    threshold: Option<f64>,
    lambda_cost: Option<f64>,
    lambda_unc: Option<f64>,
}

impl PolicyScorerBuilder {
    /// Register a feature. Order is preserved; use for deterministic logging.
    pub fn feature<F: PolicyFeature + 'static>(mut self, f: F) -> Self {
        self.features.push(Box::new(f));
        self
    }

    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = Some(t);
        self
    }

    pub fn lambda_cost(mut self, l: f64) -> Self {
        self.lambda_cost = Some(l);
        self
    }

    pub fn lambda_unc(mut self, l: f64) -> Self {
        self.lambda_unc = Some(l);
        self
    }

    pub fn build(self) -> PolicyScorer {
        PolicyScorer {
            features: self.features,
            threshold: self.threshold.unwrap_or(0.0),
            lambda_cost: self.lambda_cost.unwrap_or(1.0),
            lambda_unc: self.lambda_unc.unwrap_or(0.5),
        }
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn action_kind_name(action: &RootAction) -> &'static str {
    match action {
        RootAction::BoostProcess { .. } => "BoostProcess",
        RootAction::ThrottleProcess { .. } => "ThrottleProcess",
        RootAction::FreezeProcess { .. } => "FreezeProcess",
        RootAction::UnfreezeProcess { .. } => "UnfreezeProcess",
        RootAction::SetSysctl { .. } => "SetSysctl",
        RootAction::SetMemorystatus { .. } => "SetMemorystatus",
        RootAction::ToggleSpotlight { .. } => "ToggleSpotlight",
        RootAction::QuarantineDaemon { .. } => "QuarantineDaemon",
        RootAction::SetThreadQoS { .. } => "SetThreadQoS",
    }
}

fn is_freeze_or_throttle(action: &RootAction) -> bool {
    matches!(
        action,
        RootAction::FreezeProcess { .. } | RootAction::ThrottleProcess { .. }
    )
}

#[allow(clippy::too_many_arguments)] // dedicated helper; fields are distinct scalars.
fn build_reason(
    action_kind: &'static str,
    accept: bool,
    net: f64,
    threshold: f64,
    total_benefit: f64,
    total_cost: f64,
    total_uncertainty: f64,
    vetoed_by: &Option<String>,
    per_feature: &[(&'static str, Contribution)],
) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(96);
    let verdict = if accept { "accept" } else { "reject" };
    let _ = write!(s, "{action_kind}:{verdict}");
    if let Some(v) = vetoed_by {
        let _ = write!(s, " veto={v}");
    }
    let _ = write!(
        s,
        " net={:.3} (b={:.3} c={:.3} u={:.3} thr={:.3})",
        net, total_benefit, total_cost, total_uncertainty, threshold
    );
    // Always list every feature (even zero contributors) so the reason string
    // deterministically mentions the full registered feature set — callers can
    // audit "silent" features as easily as active ones.
    if !per_feature.is_empty() {
        s.push_str(" [");
        for (i, (name, c)) in per_feature.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            let _ = write!(
                s,
                "{name}:b={:.2},c={:.2},u={:.2}{}",
                c.benefit,
                c.cost,
                c.uncertainty,
                if c.hard_veto { ",veto" } else { "" }
            );
        }
        s.push(']');
    }
    s
}

// -----------------------------------------------------------------------------
// Starter features
// -----------------------------------------------------------------------------

/// F1 — Protection. Veto-only feature: mirrors `classify_protection` semantics.
///
/// - `Unconditional` → veto always.
/// - `ConditionalForeground` + freeze/throttle + in foreground family → veto.
/// - Otherwise contributes zero.
pub struct ProtectionFeature;

impl PolicyFeature for ProtectionFeature {
    fn name(&self) -> &'static str {
        "protection"
    }

    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution {
        match ctx.protection_level {
            ProtectionLevel::Unconditional => Contribution {
                benefit: 0.0,
                cost: 0.0,
                uncertainty: 0.0,
                hard_veto: true,
            },
            ProtectionLevel::ConditionalForeground => {
                if is_freeze_or_throttle(action) && ctx.is_foreground_family {
                    Contribution {
                        benefit: 0.0,
                        cost: 0.0,
                        uncertainty: 0.0,
                        hard_veto: true,
                    }
                } else {
                    Contribution::zero()
                }
            }
            ProtectionLevel::Unprotected => Contribution::zero(),
        }
    }
}

/// F2 — PressureBenefit. Expected reclaim grows with pressure, thrashing, and
/// predicted OOM. Boosts receive a small positive baseline (cheap insurance).
pub struct PressureBenefitFeature;

impl PolicyFeature for PressureBenefitFeature {
    fn name(&self) -> &'static str {
        "pressure_benefit"
    }

    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution {
        match action {
            RootAction::FreezeProcess { .. } | RootAction::ThrottleProcess { .. } => {
                let mut benefit = ctx.pressure * 1.0;
                if ctx.thrashing_score > 5_000.0 {
                    benefit += 0.5;
                }
                if ctx.p_oom_30s.unwrap_or(0.0) > 0.30 {
                    benefit += 1.0;
                }
                Contribution {
                    benefit,
                    cost: 0.0,
                    uncertainty: 0.0,
                    hard_veto: false,
                }
            }
            RootAction::BoostProcess { .. } => Contribution {
                benefit: 0.1,
                cost: 0.0,
                uncertainty: 0.0,
                hard_veto: false,
            },
            _ => Contribution::zero(),
        }
    }
}

/// F3 — UserDisruptionCost. Charges cost for freeze/throttle during
/// interactive user phases. Reproduces current sleep-assertion bypass
/// semantics: cost contribution is zero when pressure/swap/thrashing are
/// high enough that the existing gate tower would bypass the assertion.
pub struct UserDisruptionCostFeature;

impl PolicyFeature for UserDisruptionCostFeature {
    fn name(&self) -> &'static str {
        "user_cost"
    }

    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution {
        match action {
            RootAction::FreezeProcess { .. } | RootAction::ThrottleProcess { .. } => {
                let mut cost = 0.0f64;
                let mut uncertainty = 0.0f64;
                if ctx.call_in_progress {
                    cost += 2.0;
                }
                // Sleep-assertion bypass: match existing freeze_protected/Gate C
                // logic — the assertion stops contributing cost when pressure,
                // swap, or thrashing cross their danger thresholds.
                let bypass_sleep =
                    ctx.pressure >= 0.70 || ctx.swap_gb >= 4.0 || ctx.thrashing_score >= 10_000.0;
                if ctx.has_sleep_assertion && !bypass_sleep {
                    cost += 1.0;
                }
                if ctx.is_recently_active {
                    cost += 0.5;
                }
                if ctx.idle_secs < 5.0 {
                    uncertainty += 0.3;
                }
                Contribution {
                    benefit: 0.0,
                    cost,
                    uncertainty,
                    hard_veto: false,
                }
            }
            RootAction::BoostProcess { .. } => Contribution::zero(),
            _ => Contribution::zero(),
        }
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
        }
    }

    fn freeze(pid: u32) -> RootAction {
        RootAction::freeze(pid, format!("p{pid}"), "test")
    }

    fn throttle(pid: u32) -> RootAction {
        RootAction::throttle(pid, format!("p{pid}"), false, "test")
    }

    fn boost(pid: u32) -> RootAction {
        RootAction::BoostProcess {
            pid,
            name: format!("p{pid}"),
            reason: "test".into(),
        }
    }

    #[test]
    fn zero_features_accepts_by_default() {
        let scorer = PolicyScorer::builder().build();
        let ctx = base_ctx();
        let s = scorer.score(&freeze(1), &ctx);
        assert!(s.accept, "empty scorer should accept (net=0 ≥ threshold=0)");
        assert_eq!(s.total_benefit, 0.0);
        assert_eq!(s.total_cost, 0.0);
        assert!(s.vetoed_by.is_none());
    }

    #[test]
    fn protection_unconditional_vetoes_freeze() {
        let scorer = PolicyScorer::builder()
            .feature(ProtectionFeature)
            .feature(PressureBenefitFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.protection_level = ProtectionLevel::Unconditional;
        ctx.pressure = 0.99;
        let s = scorer.score(&freeze(1), &ctx);
        assert!(!s.accept);
        assert_eq!(s.vetoed_by.as_deref(), Some("protection"));
    }

    #[test]
    fn protection_conditional_foreground_vetoes_only_when_in_family() {
        let scorer = PolicyScorer::builder()
            .feature(ProtectionFeature)
            .feature(PressureBenefitFeature)
            .build();
        // In foreground family → veto.
        let mut ctx = base_ctx();
        ctx.protection_level = ProtectionLevel::ConditionalForeground;
        ctx.is_foreground_family = true;
        ctx.pressure = 0.80;
        let s = scorer.score(&freeze(1), &ctx);
        assert!(!s.accept);
        assert_eq!(s.vetoed_by.as_deref(), Some("protection"));

        // Not in foreground family → no veto, high pressure → accept.
        ctx.is_foreground_family = false;
        let s = scorer.score(&freeze(1), &ctx);
        assert!(s.accept, "reason={}", s.reason);
        assert!(s.vetoed_by.is_none());
    }

    #[test]
    fn pressure_benefit_scales_with_thrashing() {
        let scorer = PolicyScorer::builder()
            .feature(PressureBenefitFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.50;
        let low = scorer.score(&freeze(1), &ctx);
        ctx.thrashing_score = 6_000.0;
        let hi = scorer.score(&freeze(1), &ctx);
        assert!(
            hi.total_benefit > low.total_benefit,
            "thrashing should raise benefit ({} !> {})",
            hi.total_benefit,
            low.total_benefit
        );
        // +0.5 from thrashing bonus.
        assert!((hi.total_benefit - low.total_benefit - 0.5).abs() < 1e-9);
    }

    #[test]
    fn pressure_benefit_fires_on_high_p_oom_30s() {
        let scorer = PolicyScorer::builder()
            .feature(PressureBenefitFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.10;
        ctx.p_oom_30s = Some(0.50);
        let s = scorer.score(&freeze(1), &ctx);
        // 0.10 + 1.0 = 1.10.
        assert!(
            (s.total_benefit - 1.10).abs() < 1e-9,
            "benefit={}",
            s.total_benefit
        );
    }

    #[test]
    fn user_cost_blocks_freeze_under_sleep_assertion_at_low_pressure() {
        let scorer = PolicyScorer::builder()
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.30; // benefit = 0.30
        ctx.has_sleep_assertion = true; // cost = 1.0
        let s = scorer.score(&freeze(1), &ctx);
        // net = 0.30 - 1.0*1.0 - 0.5*0 = -0.70 < 0.0 threshold → reject.
        assert!(!s.accept, "reason={}", s.reason);
        assert!(s.vetoed_by.is_none(), "cost-based reject, not veto");
        assert!(s.total_cost >= 1.0);
    }

    #[test]
    fn user_cost_bypassed_at_high_pressure() {
        let scorer = PolicyScorer::builder()
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.75; // crosses 0.70 threshold → bypass
        ctx.has_sleep_assertion = true;
        let s = scorer.score(&freeze(1), &ctx);
        // Sleep-assertion contribution suppressed; no other cost triggers.
        assert_eq!(s.total_cost, 0.0, "sleep-assertion should be bypassed");
    }

    #[test]
    fn user_cost_bypassed_at_high_thrashing() {
        let scorer = PolicyScorer::builder()
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.30;
        ctx.swap_gb = 0.5;
        ctx.thrashing_score = 15_000.0; // crosses 10k → bypass
        ctx.has_sleep_assertion = true;
        let s = scorer.score(&freeze(1), &ctx);
        assert_eq!(s.total_cost, 0.0, "thrashing>10k should bypass sleep cost");
    }

    #[test]
    fn call_in_progress_always_vetoes_via_cost_overflow() {
        // "veto via cost overflow": cost is so high benefit can't offset.
        let scorer = PolicyScorer::builder()
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.99; // max benefit ~0.99
        ctx.call_in_progress = true; // cost += 2.0
        let s = scorer.score(&freeze(1), &ctx);
        // net = 0.99 - 2.0 = -1.01 → reject.
        assert!(!s.accept, "reason={}", s.reason);
        assert!(s.total_cost >= 2.0);
        assert!(s.vetoed_by.is_none());
    }

    #[test]
    fn boost_always_accepted_even_under_call() {
        let scorer = PolicyScorer::builder()
            .feature(ProtectionFeature)
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.call_in_progress = true;
        ctx.has_sleep_assertion = true;
        ctx.is_recently_active = true;
        let s = scorer.score(&boost(1), &ctx);
        assert!(s.accept, "reason={}", s.reason);
        assert_eq!(s.total_cost, 0.0);
        assert!(s.total_benefit >= 0.1);
    }

    #[test]
    fn score_is_deterministic() {
        let scorer = PolicyScorer::builder()
            .feature(ProtectionFeature)
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.55;
        ctx.has_sleep_assertion = true;
        ctx.thrashing_score = 4_000.0;
        let first = scorer.score(&freeze(42), &ctx);
        for _ in 0..1_000 {
            let s = scorer.score(&freeze(42), &ctx);
            assert_eq!(s.accept, first.accept);
            assert!((s.total_benefit - first.total_benefit).abs() < 1e-12);
            assert!((s.total_cost - first.total_cost).abs() < 1e-12);
            assert!((s.total_uncertainty - first.total_uncertainty).abs() < 1e-12);
            assert_eq!(s.reason, first.reason);
        }
    }

    #[test]
    fn per_feature_contribs_sum_to_totals() {
        let scorer = PolicyScorer::builder()
            .feature(ProtectionFeature)
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.80;
        ctx.thrashing_score = 6_000.0;
        ctx.p_oom_30s = Some(0.40);
        ctx.has_sleep_assertion = true;
        ctx.idle_secs = 2.0;
        let s = scorer.score(&throttle(7), &ctx);
        let sum_b: f64 = s.per_feature.iter().map(|(_, c)| c.benefit.max(0.0)).sum();
        let sum_c: f64 = s.per_feature.iter().map(|(_, c)| c.cost.max(0.0)).sum();
        let sum_u: f64 = s
            .per_feature
            .iter()
            .map(|(_, c)| c.uncertainty.max(0.0))
            .sum();
        assert!((sum_b - s.total_benefit).abs() < 1e-9);
        assert!((sum_c - s.total_cost).abs() < 1e-9);
        assert!((sum_u - s.total_uncertainty).abs() < 1e-9);
    }

    #[test]
    fn reason_string_contains_all_contributors() {
        let scorer = PolicyScorer::builder()
            .feature(ProtectionFeature)
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.50;
        let s = scorer.score(&freeze(9), &ctx);
        assert!(s.reason.contains("protection"), "reason={}", s.reason);
        assert!(s.reason.contains("pressure_benefit"), "reason={}", s.reason);
        assert!(s.reason.contains("user_cost"), "reason={}", s.reason);
        assert!(s.reason.contains("FreezeProcess"), "reason={}", s.reason);
    }

    #[test]
    fn builder_composes_correctly() {
        let ctx = {
            let mut c = base_ctx();
            c.pressure = 0.60;
            c
        };
        let built = PolicyScorer::builder()
            .feature(PressureBenefitFeature)
            .threshold(0.5)
            .lambda_cost(2.0)
            .lambda_unc(0.25)
            .build();
        assert!((built.threshold - 0.5).abs() < 1e-12);
        assert!((built.lambda_cost - 2.0).abs() < 1e-12);
        assert!((built.lambda_unc - 0.25).abs() < 1e-12);
        // Functional check: benefit 0.60 passes threshold 0.5 → accept.
        let s = built.score(&freeze(1), &ctx);
        assert!(s.accept, "reason={}", s.reason);
    }

    #[test]
    fn threshold_negative_is_rejected_positive_accepted() {
        let scorer = PolicyScorer::builder()
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .threshold(0.0)
            .build();
        // Negative net: low pressure, sleep assertion costs 1.0.
        let mut ctx = base_ctx();
        ctx.pressure = 0.20;
        ctx.has_sleep_assertion = true;
        let s_neg = scorer.score(&freeze(1), &ctx);
        assert!(!s_neg.accept, "net should be negative");

        // Positive net: high pressure (bypasses sleep cost).
        ctx.pressure = 0.85;
        let s_pos = scorer.score(&freeze(1), &ctx);
        assert!(
            s_pos.accept,
            "net should be positive, reason={}",
            s_pos.reason
        );
    }
}
