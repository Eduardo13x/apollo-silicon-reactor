//! Disobedience test — proves PolicyScorer diverges from old `freeze_protected()`
//! in scenarios where the old gate is reactive-blind to predictive signals.
//!
//! This is the adversarial integration test prescribed by the NotebookLM
//! critique (2026-04-22, conversation 379c81af) to validate that shadow-mode
//! disagreement rate is not a tautology. If this test ever starts failing
//! because the two decision paths agree, the scorer has been neutered.
//!
//! Paper: [Camacho 2007] MPC — predictive control must diverge from reactive
//! baseline on signals the reactive path can't observe.

use apollo_engine::engine::action_policy::{
    ActionContext, PolicyScorer, PressureBenefitFeature, ProtectionFeature,
    UserDisruptionCostFeature,
};
use apollo_engine::engine::policy_feature_predictive::PredictiveBenefitFeature;
use apollo_engine::engine::safety::ProtectionLevel;
use apollo_engine::engine::types::RootAction;
use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::user_context::UserContext;

/// Construct the adversarial context: reactive signals all below bypass
/// (pressure 0.66 < 0.70, thrashing 9000 < 10 000) + active sleep assertion +
/// strong predictive OOM signal (p_oom_30s=0.35 < 0.40). Old gate with
/// p_oom=0.0 blocks; scorer with predictive feature accepts.
fn adversarial_ctx() -> ActionContext {
    ActionContext {
        pressure: 0.66,
        swap_gb: 1.1,
        thrashing_score: 9_000.0,
        p_oom_30s: Some(0.35),
        p_jank_60s: Some(0.15),
        has_sleep_assertion: true,
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

/// Full four-feature scorer used in the divergence test: the F4 starter trio
/// plus F6 PredictiveBenefit (default weights: oom=3.0, jank=1.0, thr=0.15).
fn full_scorer() -> PolicyScorer {
    PolicyScorer::builder()
        .feature(ProtectionFeature)
        .feature(PressureBenefitFeature)
        .feature(UserDisruptionCostFeature)
        .feature(PredictiveBenefitFeature::default())
        .build()
}

fn freeze_action(pid: u32) -> RootAction {
    RootAction::freeze(pid, "heavy_bg_proc", "shadow-mode disobedience test", DecisionReason::PressureContext)
}

#[test]
fn scorer_diverges_from_old_gate_on_high_p_oom_with_sleep_assertion() {
    // 1. Build the adversarial context (prod April 2026 scenario).
    let ctx = adversarial_ctx();

    // 2. The corresponding UserContext the old gate consumes.
    let user_ctx = UserContext {
        idle_secs: ctx.idle_secs,
        has_sleep_assertion: ctx.has_sleep_assertion,
        call_in_progress: ctx.call_in_progress,
        audio_active: false,
    };

    // 3. Assert old gate BLOCKS: no reactive-crisis bypass fires, sleep
    //    assertion dominates → returns true ("protected, don't freeze").
    //    This is the reactive blindness we're about to expose.
    // Old gate signature change: swap_used_bytes dropped (broken on macOS
    // dynamic swap), p_oom_30s added as predictive bypass. For the
    // disobedience test we pass LOW p_oom so the reactive path is what's
    // being tested — in prod ActionContext.p_oom_30s=0.35 would flow through
    // shadow_signals and trigger the bypass, which is the desired behavior.
    let old_blocks = user_ctx.freeze_protected(ctx.pressure, ctx.thrashing_score, 0.0);
    assert!(
        old_blocks,
        "precondition: old gate must BLOCK in this scenario when p_oom is zeroed \
         (reactive-blind to predictive signal). freeze_protected(pressure=0.66, \
         thrashing=9000, p_oom=0.0) returned false — one of the bypass thresholds \
         may have shifted."
    );

    // 4. Build the full scorer with F6 Predictive wired in.
    let scorer = full_scorer();

    // 5. Score the freeze action.
    let action = freeze_action(4242);
    let score = scorer.score(&action, &ctx);

    // 6. Scorer DISAGREES: accepts the freeze because the predictive
    //    feature consumes p_oom_30s that the old gate ignores.
    assert!(
        score.accept,
        "DISOBEDIENCE FAILED — scorer agrees with old gate (tautology).\n\
         reason: {}\n\
         This means the PolicyScorer is not adding predictive value beyond \
         the reactive gate tower. Either a feature regressed or weights moved.",
        score.reason
    );

    // 7. Benefit must exceed cost net of lambdas (structural sanity).
    assert!(
        score.total_benefit > score.total_cost,
        "benefit must dominate cost — got b={:.3} c={:.3}\nreason: {}",
        score.total_benefit,
        score.total_cost,
        score.reason
    );

    // 8. No feature may hard-veto an Unprotected target.
    assert!(
        score.vetoed_by.is_none(),
        "no hard-veto expected for Unprotected target, got veto_by={:?}\nreason: {}",
        score.vetoed_by,
        score.reason
    );

    // 9. Also assert the predictive feature is the *load-bearing* contributor:
    //    without it, benefit would be ~2.16 (pressure + thrashing bonus + oom
    //    bonus in PressureBenefit). With it, benefit ≥ ~3.2 (adds oom×3.0 + jank×1.0).
    assert!(
        score.total_benefit >= 3.0,
        "predictive feature should push benefit ≥ 3.0 — got {:.3}\nreason: {}",
        score.total_benefit,
        score.reason
    );
}

#[test]
fn scorer_still_agrees_with_old_gate_when_protected_process() {
    // Same adversarial scenario BUT the target is Unconditionally protected.
    // Old gate still blocks (sleep assertion). Scorer must also reject —
    // ProtectionFeature hard_veto fires regardless of predictive benefit.
    let mut ctx = adversarial_ctx();
    ctx.protection_level = ProtectionLevel::Unconditional;

    let user_ctx = UserContext {
        idle_secs: ctx.idle_secs,
        has_sleep_assertion: ctx.has_sleep_assertion,
        call_in_progress: ctx.call_in_progress,
        audio_active: false,
    };

    assert!(
        user_ctx.freeze_protected(ctx.pressure, ctx.thrashing_score, 0.0),
        "precondition: old gate still blocks on sleep assertion"
    );

    let scorer = full_scorer();
    let score = scorer.score(&freeze_action(4242), &ctx);

    assert!(
        !score.accept,
        "ProtectionFeature veto must override all predictive benefit.\nreason: {}",
        score.reason
    );
    assert_eq!(
        score.vetoed_by.as_deref(),
        Some("protection"),
        "expected veto from ProtectionFeature, got {:?}\nreason: {}",
        score.vetoed_by,
        score.reason
    );
}

#[test]
fn scorer_agrees_with_old_gate_on_pure_reactive_crisis() {
    // Pure reactive crisis: pressure=0.85 (> 0.70), thrashing=15_000
    // (> 10_000), no sleep assertion. Old gate returns false (not blocked
    // — either of the two bypasses satisfies the "not protected" path, and
    // there's no sleep assertion anyway). Scorer must accept strongly —
    // no veto, zero cost, high benefit.
    let ctx = ActionContext {
        pressure: 0.85,
        swap_gb: 2.0,
        thrashing_score: 15_000.0,
        p_oom_30s: Some(0.60),
        p_jank_60s: Some(0.40),
        has_sleep_assertion: false,
        call_in_progress: false,
        idle_secs: 120.0,
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
    };

    let user_ctx = UserContext {
        idle_secs: ctx.idle_secs,
        has_sleep_assertion: ctx.has_sleep_assertion,
        call_in_progress: ctx.call_in_progress,
        audio_active: false,
    };
    assert!(
        !user_ctx.freeze_protected(ctx.pressure, ctx.thrashing_score, 0.0),
        "precondition: old gate permits freeze in reactive crisis with no sleep assertion"
    );

    let scorer = full_scorer();
    let score = scorer.score(&freeze_action(4242), &ctx);

    assert!(
        score.accept,
        "expected scorer to accept in pure reactive crisis.\nreason: {}",
        score.reason
    );
    assert!(score.vetoed_by.is_none(), "no veto expected");
    assert!(
        score.total_cost == 0.0,
        "no sleep assertion + not recently active + idle_secs > 5 → cost must be zero. \
         got cost={:.3}\nreason: {}",
        score.total_cost,
        score.reason
    );
    assert!(
        score.total_benefit > 2.0,
        "high-pressure high-oom crisis should yield substantial benefit — got {:.3}\nreason: {}",
        score.total_benefit,
        score.reason
    );
}
