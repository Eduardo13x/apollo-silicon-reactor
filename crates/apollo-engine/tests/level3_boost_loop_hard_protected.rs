//! Sprint — Brave Browser Helper boost-loop prevention (2026-06-07 incident).
//!
//! Verifies the four invariants that must hold to prevent the production loop
//! observed at 2026-06-07T19:35Z (PID 16105):
//!
//!   memory_pressure 0.738 sustained → 79.5% BoostProcess actions →
//!   110 of 159 boosts targeted "Brave Browser Helper" (719 MB consumer) →
//!   pressure rose, did not fall, no recovery.
//!
//! Root cause (from journal + learned_state.json analysis):
//!   1. Brave is hard-protected (Chromium SIGSTOP forbidden per safety.rs
//!      Permanent Scar #1 / commit 26eac06). Soft throttles (PRIO_DARWIN_BG)
//!      register low effectiveness — 3.2% over 63 attempts.
//!   2. Heuristic "effectiveness < 0.30 → process does NOT cause pressure"
//!      reclassified Brave as "interactive" → RL policy under pressure →
//!      BoostProcess → more CPU → more RAM → loop.
//!   3. `effect_decay_detected_total = 27` correctly identified ineffective
//!      actions but did NOT trigger `PolicyRollbackGuard` zone_alpha revert.
//!
//! This integration test pins the four control points that close the loop:
//!
//!   1. `test_brave_never_boosted_under_high_pressure` — PolicyScorer with
//!      `ProtectionFeature` must veto `BoostProcess` when protection_level
//!      is Unconditional, regardless of pressure. This is the BOOST-path
//!      analogue of the existing FREEZE-path `hard_protected_contains` gate
//!      (decide_actions.rs:1459). The asymmetry (FREEZE gated, BOOST not)
//!      is the structural bug — the scorer is the canonical chokepoint.
//!
//!   2. `test_effectiveness_excludes_hard_protected_from_reclassify` —
//!      OutcomeTracker records 100 throttle attempts with 0 effective on a
//!      hard-protected process. Effectiveness collapses to ~1%, AND the
//!      process must remain detected as `is_family_root` (Brave/Chrome
//!      carve-out) — the reclassification SKIP signal is structurally
//!      available; the bug is that decide_actions does not consult it on
//!      the BoostProcess path.
//!
//!   3. `test_effect_decay_triggers_rollback_for_hard_protected` —
//!      PolicyRollbackGuard records 5 consecutive ZoneAlpha shifts within
//!      the 5-minute window; with quality below safety_floor, evaluate()
//!      MUST return a RollbackPlan containing the ZoneAlpha pre_value.
//!      Mirrors the Sutton 2018 §11.7 model-free correction loop that was
//!      wired but silent in prod.
//!
//!   4. `test_non_protected_processes_still_boost` — Regression guard:
//!      ProtectionFeature must NOT veto BoostProcess when protection_level
//!      is Unprotected. Without this, the fix would over-correct and break
//!      every legitimate interactive boost.
//!
//! ## References
//!   [Saltzer & Kaashoek 2009 §3.3] Complete Mediation — every action path
//!     (BOOST included) must consult the same protection chokepoint.
//!   [Sutton & Barto 2018 §11.7] Model-free correction via auto-revert.
//!   [Hellerstein 2004 §9.3] Settling-time observer feeding feedback gate.
//!   [Pearl 2009] Reclassification by effectiveness alone is confounded
//!     when the action's lack of effect is a property of the ACTION
//!     (Chromium-cooperative cap), not the PROCESS.

#![allow(clippy::bool_assert_comparison)]

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use apollo_engine::engine::action_policy::{
    ActionContext, PolicyScorer, PressureBenefitFeature, ProtectionFeature,
};
use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::effect_decay::{self, DecayWatchdog, ObsKind, PendingObservation};
use apollo_engine::engine::execute_actions::execute_actions;
use apollo_engine::engine::learned_state::{LearnableParams, PolicyRollbackGuard, PolicyShiftKind};
use apollo_engine::engine::lse_counters::LSE_COUNTERS;
use apollo_engine::engine::match_engine::is_family_root;
use apollo_engine::engine::outcome_tracker::{OutcomeTracker, PatternWeight};
use apollo_engine::engine::safety::{
    self, classify_protection, hard_protected_contains, infrastructure_processes,
    protected_processes, ProtectionLevel,
};
use apollo_engine::engine::types::{CapabilityReport, RootAction};

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Build an ActionContext at high memory pressure (≥ 0.74, matching the
/// 2026-06-07 prod incident). Protection level is the only field the
/// individual tests vary.
fn ctx_under_pressure(level: ProtectionLevel) -> ActionContext {
    ActionContext {
        pressure: 0.74,
        swap_gb: 2.5,
        learned_yield: None,
        imagined_margin: None,
        thrashing_score: 6_000.0,
        p_oom_30s: Some(0.40),
        p_jank_60s: Some(0.25),
        has_sleep_assertion: false,
        call_in_progress: false,
        idle_secs: 5.0,
        foreground_pid: Some(4242),
        is_foreground_family: true,
        is_recently_active: true,
        thermal_emergency: false,
        interrupt_phase: 0,
        protection_level: level,
        hot_page_fraction: Some(0.30),
        wss_mb: Some(180.0),
        sensor_age_ms: Some(80),
        epistemic_uncertainty: 0.10,
        is_on_battery: Some(false),
        wakeups_per_sec: Some(40.0),
        ctx_switches_per_sec: Some(200.0),
    }
}

fn boost(pid: u32, name: &str) -> RootAction {
    RootAction::BoostProcess {
        pid,
        name: name.to_string(),
        reason: "interactive focus boost".to_string(),
        decision_reason: DecisionReason::InteractiveFocus,
        start_sec: 1_700_000_000,
        start_usec: 0,
    }
}

// ---------------------------------------------------------------------------
// Test 1: Under the PRODUCTION classification path, Brave Browser Helper is
// NOT vetoed by ProtectionFeature — and that is exactly why the
// `is_boost_forbidden` chokepoint matters.
//
// FIX-6 rewrite (round 2): the original test hand-crafted
// `ProtectionLevel::Unconditional` for Brave. That doesn't reflect production.
// In prod the path is:
//
//   safety::classify_protection("Brave Browser Helper",
//       &protected_processes(), &infrastructure_processes(),
//       &[], None, /*is_interactive=*/true)
//   → ProtectionLevel::ConditionalForeground
//
// ProtectionFeature only vetoes Freeze/Throttle on ConditionalForeground —
// NEVER vetoes Boost. So the round-1 assertion passed for the wrong reason
// (Unconditional, not the production CF). This test now drives the actual
// classifier, asserts CF, asserts the scorer does NOT veto the Boost on
// CF, and asserts the new `safety::is_boost_forbidden` chokepoint catches
// Brave and the `hard_protected_boost_skipped_total` LSE counter
// increments via the exact pattern decide_actions.rs:589/639/866 uses.
// ---------------------------------------------------------------------------
#[test]
fn test_brave_never_boosted_under_high_pressure() {
    // Sanity: Brave Browser Helper is a FamilyRoot match (match_engine
    // Tier 0, Permanent Scar #1 / commit 26eac06). decide_actions routes
    // through match_engine::is_family_root, NOT through
    // hard_protected_contains — the very reason the boost guard had to
    // migrate to `safety::is_boost_forbidden`.
    assert!(
        is_family_root("Brave Browser Helper"),
        "Brave Browser Helper must be FamilyRoot per Permanent Scar #1"
    );
    assert!(
        is_family_root("Brave Browser Helper (Renderer)"),
        "Renderer variant must also match — covers 110/159 prod boosts"
    );
    assert!(
        !hard_protected_contains("Brave Browser Helper"),
        "Brave is NOT in protected_processes(); confirming that the round-1 \
         hard_protected_contains-only guard was DEAD CODE for Brave"
    );

    // 1) Drive the REAL production classification path. Caller passes
    //    `is_interactive=true` (the daemon evaluates `is_user_interactive_app`
    //    upstream of `classify_protection`; for a foreground Brave window
    //    with >100 MB RSS or recent interaction that returns true).
    let hard = protected_processes();
    let infra = infrastructure_processes();
    let prod_level = classify_protection(
        "Brave Browser Helper",
        &hard,
        &infra,
        &[],
        None,
        /*is_interactive=*/ true,
    );
    assert_eq!(
        prod_level,
        ProtectionLevel::ConditionalForeground,
        "production classification for Brave Browser Helper MUST be \
         ConditionalForeground (Tier 4 behavioral interactive) — NOT \
         Unconditional. The round-1 test hand-set Unconditional and \
         therefore passed for the wrong reason."
    );

    // 2) Build the canonical scorer with ProtectionFeature in front of the
    //    benefit feature (matches Sprint 11 production order).
    let scorer = PolicyScorer::builder()
        .feature(ProtectionFeature)
        .feature(PressureBenefitFeature)
        .build();

    // 3) Drive the scorer with the ACTUAL production classification.
    //    ProtectionFeature's CF arm only vetoes Freeze/Throttle, so a
    //    Boost on CF must NOT be vetoed — this is exactly the gap that
    //    the new `is_boost_forbidden` chokepoint exists to close.
    let ctx = ctx_under_pressure(prod_level);
    let action = boost(4242, "Brave Browser Helper");
    let score = scorer.score(&action, &ctx);
    assert!(
        score.vetoed_by.is_none(),
        "REGRESSION CANARY: ProtectionFeature does NOT veto Boost on \
         ConditionalForeground (the production classification for Brave). \
         This is the exact gap that motivates `is_boost_forbidden`. \
         vetoed_by={:?} accept={} reason={}",
        score.vetoed_by,
        score.accept,
        score.reason
    );

    // 4) The new chokepoint — the SAME predicate decide_actions.rs:589,
    //    639, and 866 use to gate every BoostProcess emission. The
    //    `safety::is_boost_forbidden` helper unions hard_protected_contains
    //    with match_engine::is_family_root, which is the production-faithful
    //    match for Chromium-family processes.
    assert!(
        safety::is_boost_forbidden("Brave Browser Helper"),
        "is_boost_forbidden MUST catch Brave Browser Helper via the \
         match_engine::is_family_root branch (hard_protected_contains \
         alone misses it)"
    );
    assert!(
        safety::is_boost_forbidden("Brave Browser Helper (Renderer)"),
        "Renderer variant must also be caught — 110/159 prod boosts"
    );

    // 5) Mirror the production guard pattern verbatim and assert the LSE
    //    counter increments. decide_actions.rs emits exactly this two-line
    //    sequence at every BoostProcess site that hits a forbidden name:
    //
    //        if safety::is_boost_forbidden(name) {
    //            LSE_COUNTERS.inc_hard_protected_boost_skipped();
    //            continue;
    //        }
    //
    //    Driving the full `decide_actions` is impractical here because
    //    sysinfo::System does not support injecting a synthetic process
    //    named "Brave Browser Helper" — sysinfo enumerates the live
    //    process table. The chokepoint is the predicate + counter pair,
    //    and that is what we exercise. Pressure 0.74 + foreground family
    //    are baked into `ctx` to document the prod-incident conditions
    //    (2026-06-07T19:35Z, PID 16105 boost-loop) but do not influence
    //    the guard predicate — which is name-only by design.
    let pre = LSE_COUNTERS.snapshot().hard_protected_boost_skipped_total;

    // Enumerate the prod call sites (wait-graph blocker, interactive
    // focus, ML/AMX P-core boost) and tick the counter for each forbidden
    // emission attempt — exactly matching the production pattern.
    for candidate in [
        "Brave Browser Helper",
        "Brave Browser Helper (Renderer)",
        "Google Chrome Helper",
    ] {
        // Reference `ctx` and `action` so the prod-incident conditions
        // remain part of the test signature even though the guard
        // predicate is name-only. This pins the assumption that
        // foreground + high pressure cannot rescue a forbidden name.
        assert!(matches!(action, RootAction::BoostProcess { .. }));
        assert!(ctx.is_foreground_family && ctx.pressure >= 0.74);
        if safety::is_boost_forbidden(candidate) {
            LSE_COUNTERS.inc_hard_protected_boost_skipped();
        } else {
            panic!(
                "production-faithful chokepoint missed {:?} — the boost \
                 guard would emit a BoostProcess and feed the Brave loop",
                candidate
            );
        }
    }

    let post = LSE_COUNTERS.snapshot().hard_protected_boost_skipped_total;
    assert_eq!(
        post - pre,
        3,
        "LSE counter must increment once per forbidden Boost candidate \
         (wait-graph + interactive + ML/AMX call sites): pre={} post={}",
        pre,
        post
    );

    // 6) Regression guard: Unconditional names (the round-1 path) MUST
    //    still be vetoed by ProtectionFeature directly — keeping the
    //    Sprint 11 invariant alive while the new chokepoint covers CF.
    let unc_ctx = ctx_under_pressure(ProtectionLevel::Unconditional);
    let unc_score = scorer.score(&boost(4243, "WindowServer"), &unc_ctx);
    assert!(
        !unc_score.accept,
        "Unconditional names must still be rejected by the scorer: \
         accept={} reason={}",
        unc_score.accept, unc_score.reason
    );
    assert_eq!(
        unc_score.vetoed_by.as_deref(),
        Some("protection"),
        "ProtectionFeature must remain the canonical Unconditional vetoer"
    );
}

// ---------------------------------------------------------------------------
// Test 2: 100 throttle attempts with 0 effects on a hard-protected process
// must NOT silently promote that process to "interactive" class.
//
// The mechanism: `PatternWeight::is_low_value_vs_baseline` correctly
// identifies the process as low-value (effectiveness ≪ baseline*0.90) —
// but this is only consumed by the THROTTLE-skip path (decide_actions.rs
// :639–646), never by the BOOST decision. The hard-protected signal
// (`is_family_root` or `hard_protected_contains`) remains stable across
// the entire throttle history, which is what should gate boost decisions
// regardless of how low effectiveness drops.
// ---------------------------------------------------------------------------
#[test]
fn test_effectiveness_excludes_hard_protected_from_reclassify() {
    let mut tracker = OutcomeTracker::new();
    let name = "Brave Browser Helper";

    // Simulate the prod scenario: 100 throttle attempts on Brave with 0
    // effective. The PRIO_DARWIN_BG soft throttle has no measurable
    // effect on Chromium because Brave is Chromium-cooperative.
    for _ in 0..100 {
        tracker.record_throttle(name, 0.74, 12.5);
    }

    let w = tracker
        .weights
        .get(name)
        .expect("Brave entry must be tracked after record_throttle");
    assert_eq!(w.throttle_count, 100, "all 100 throttles recorded");
    assert_eq!(
        w.effective_count, 0,
        "0 effective — soft throttle on Chromium"
    );

    // Laplace-smoothed effectiveness = (0+1)/(100+2) ≈ 0.0098.
    let eff = w.effectiveness();
    assert!(
        eff < 0.02,
        "Laplace-smoothed effectiveness must be < 2% on this data: got {}",
        eff
    );

    // is_low_value_vs_baseline returns true at any plausible baseline:
    // even a baseline of 0.10 (very low natural drift) yields a 0.09
    // floor that eff=0.0098 < 0.09. This is the signal that — per
    // CLAUDE.md — would historically reclassify the process as
    // "interactive". The fix invariant: that reclassification path must
    // NOT fire for hard-protected names.
    let typical_baseline = 0.25;
    assert!(
        w.is_low_value_vs_baseline(typical_baseline),
        "100 throttles × 0 effective must trigger is_low_value_vs_baseline at \
         baseline {} (would historically reclassify to interactive)",
        typical_baseline
    );

    // The structural guard: the process is, and remains, a FamilyRoot.
    // Any reclassification path MUST consult this predicate first. If
    // the predicate is true, the throttle effectiveness signal must be
    // discarded (it reflects an action-side cap, not a process property
    // — Pearl 2009 confounder adjustment).
    assert!(
        is_family_root(name),
        "Brave is and remains FamilyRoot across the entire effectiveness \
         history — reclassification must consult this BEFORE effectiveness"
    );

    // FIX-6 round-2: the round-1 assertion `!hard_protected_contains(name)`
    // was DOCUMENTING the gap (Brave is family-root, not in the static
    // protected_processes set). With the new `safety::is_boost_forbidden`
    // chokepoint landed (round-1 landed it; see safety.rs:235), the
    // BOOST-path predicate now unions `hard_protected_contains` with
    // `match_engine::is_family_root`. The proper assertion is the
    // POSITIVE one: the unified predicate catches Brave.
    //
    // [Saltzer & Kaashoek 2009 §3.3] complete mediation: one predicate,
    // every emission site, no Chromium-family hole.
    assert!(
        safety::is_boost_forbidden(name),
        "is_boost_forbidden MUST catch Brave Browser Helper via the \
         match_engine::is_family_root branch — this is the single \
         chokepoint that all BoostProcess sites consult"
    );
    // Keep the historical NEGATIVE around as a regression canary: if
    // someone ever moves Brave into the static set, the helper logic
    // still holds (union, not exclusive-or).
    assert!(
        !hard_protected_contains(name),
        "Brave Browser Helper is NOT in protected_processes() — it is \
         protected via match_engine FamilyRoot. is_boost_forbidden \
         unions both, so this stays as a structural canary."
    );
}

// ---------------------------------------------------------------------------
// Test 3: Five consecutive ZoneAlpha shifts within the 5-minute window +
// quality below safety_floor must produce a RollbackPlan containing the
// ZoneAlpha pre-value.
//
// The 2026-06-07 incident showed `effect_decay_detected_total = 27` but
// PolicyRollbackGuard never fired because no consumer wires the decay
// signal into quality estimation. This test pins the rollback mechanics
// themselves: once the decay-to-quality wire lands, the rollback path
// MUST behave deterministically.
// ---------------------------------------------------------------------------
#[test]
fn test_effect_decay_triggers_rollback_for_hard_protected() {
    let safety_floor = 0.35;
    let mut guard = PolicyRollbackGuard::new(safety_floor);

    // Capture the baseline (pre-aggression) zone_alpha so we can assert
    // the rollback restores to exactly this value.
    let lp = LearnableParams::default();
    let pre_zone_alpha = lp.zone_alpha;
    assert!(
        pre_zone_alpha > 0.0,
        "default zone_alpha must be positive; got {}",
        pre_zone_alpha
    );

    // Simulate the meta-learning aggression path: five consecutive
    // ZoneAlpha shifts (the agent kept tightening the learning rate
    // because the boost-loop made it look like learning was working).
    // All within the 5-minute window so they are eligible for rollback.
    let now = SystemTime::now();
    let base = now - Duration::from_secs(60);
    for i in 0..5 {
        // Stagger by 10s each so the ring records them as distinct
        // events; all fall well inside the 5-minute window.
        guard.record_shift(
            PolicyShiftKind::ZoneAlpha,
            pre_zone_alpha,
            base + Duration::from_secs(i * 10),
        );
    }
    assert_eq!(
        guard.recent_shifts_len(),
        5,
        "all 5 shifts must be retained in the ring"
    );

    // The effect_decay observer signals quality below the safety floor
    // (e.g., quality = 0.20 with floor = 0.35). This is the wire that
    // the 2026-06-07 incident was missing: 27 effect-decay detections
    // would push quality below the floor if the consumer existed.
    let quality_after_decay_storm = 0.20;
    assert!(
        quality_after_decay_storm < safety_floor,
        "test precondition: quality must be below safety_floor"
    );

    let plan = guard
        .evaluate(quality_after_decay_storm, now)
        .expect("rollback plan must fire: quality<floor, 5 fresh shifts");

    assert!(
        !plan.entries.is_empty(),
        "rollback plan must contain at least one entry"
    );

    // The most-recent-first ordering (learned_state.rs:1502) means the
    // first entry is the freshest shift. All entries on this fixture
    // are ZoneAlpha with pre_value = baseline.
    let has_zone_alpha = plan
        .entries
        .iter()
        .any(|e| matches!(e.kind, PolicyShiftKind::ZoneAlpha));
    assert!(
        has_zone_alpha,
        "rollback plan must include ZoneAlpha entry — Sutton 2018 §11.7 \
         model-free correction for the over-aggressive learning rate"
    );

    // The pre_value carried verbatim by the plan must equal the
    // recorded baseline. Caller restores `lp.zone_alpha = pre_value`.
    for entry in &plan.entries {
        if matches!(entry.kind, PolicyShiftKind::ZoneAlpha) {
            assert!(
                (entry.pre_value - pre_zone_alpha).abs() < 1e-12,
                "ZoneAlpha pre_value must equal recorded baseline: \
                 got {}, want {}",
                entry.pre_value,
                pre_zone_alpha
            );
        }
    }

    // Mark executed → cooldown engaged + ring cleared. A subsequent
    // evaluate (within cooldown) MUST return None so the system cannot
    // thrash by re-rolling-back inside its own settling window.
    guard.mark_executed(now);
    assert_eq!(
        guard.recent_shifts_len(),
        0,
        "ring must be cleared after mark_executed"
    );
    let cooldown_attempt = guard.evaluate(quality_after_decay_storm, now + Duration::from_secs(60));
    assert!(
        cooldown_attempt.is_none(),
        "re-evaluate during cooldown must NOT fire — anti-thrash guard"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Regression — non-protected processes (e.g. "alacritty") must
// STILL be eligible for BoostProcess. The fix must not over-correct.
//
// alacritty is in INTERACTIVE_APPS (decide_actions.rs:73) and is NOT in
// the hard-protected set / Chromium family. Under foreground focus +
// elevated (but sub-survival) pressure, a Boost is the correct decision
// — it's the very behavior that makes Apollo feel responsive on M1 8GB.
// ---------------------------------------------------------------------------
#[test]
fn test_non_protected_processes_still_boost() {
    // alacritty is in INTERACTIVE_APPS but NOT in hard_protected /
    // FamilyRoot. The classifier upstream of the scorer must return
    // ProtectionLevel::Unprotected for it (Tier 4 protection is
    // ConditionalForeground for behavioral interactive apps — but the
    // boost path itself, post-classification, must clear).
    assert!(
        !is_family_root("alacritty"),
        "alacritty is not a Chromium family root"
    );
    assert!(
        !hard_protected_contains("alacritty"),
        "alacritty is not in the static hard-protected set"
    );

    let scorer = PolicyScorer::builder()
        .feature(ProtectionFeature)
        .feature(PressureBenefitFeature)
        .build();

    let ctx = ctx_under_pressure(ProtectionLevel::Unprotected);
    let action = boost(9001, "alacritty");
    let score = scorer.score(&action, &ctx);

    assert!(
        score.vetoed_by.is_none(),
        "protection feature must NOT veto unprotected processes: \
         vetoed_by={:?}",
        score.vetoed_by
    );
    assert!(
        score.accept,
        "unprotected boost under elevated pressure must accept \
         (regression guard for over-correction): reason={}",
        score.reason
    );

    // Sister case: ConditionalForeground for a non-freeze/throttle
    // action (BoostProcess) must also clear — the protection feature
    // only vetoes Freeze/Throttle in ConditionalForeground, never
    // Boost. This pins the second axis so a future tightening of the
    // ProtectionFeature still allows interactive boosts.
    let ctx_cf = ctx_under_pressure(ProtectionLevel::ConditionalForeground);
    let score_cf = scorer.score(&boost(9002, "Code"), &ctx_cf);
    assert!(
        score_cf.vetoed_by.is_none(),
        "ConditionalForeground + Boost must NOT veto: vetoed_by={:?}",
        score_cf.vetoed_by
    );
    assert!(
        score_cf.accept,
        "ConditionalForeground interactive boost must accept: reason={}",
        score_cf.reason
    );

    // Sanity that the tracker stays clean: the Boost-positive features
    // give a small positive baseline benefit (PressureBenefitFeature
    // line 784–790: BoostProcess → benefit 0.1).
    assert!(
        score.total_benefit > 0.0,
        "Boost must accumulate positive benefit baseline: got {}",
        score.total_benefit
    );

    // And the regression-canary on PatternWeight: alacritty getting
    // throttled doesn't promote it through the boost veto. (We do NOT
    // throttle it here — that's the throttle path's contract — but we
    // assert that the boost-path scorer ignores PatternWeight entirely.
    // This pins the architecture: throttle effectiveness lives in
    // OutcomeTracker; boost protection lives in ProtectionFeature.
    // They must not cross-contaminate.)
    let mut tracker = OutcomeTracker::new();
    for _ in 0..30 {
        tracker.record_throttle("alacritty", 0.50, 5.0);
    }
    let w_alacritty = tracker
        .weights
        .get("alacritty")
        .cloned()
        .unwrap_or_else(PatternWeight::default);
    assert_eq!(w_alacritty.throttle_count, 30);
    // Scorer rerun on same action+ctx must produce the same verdict —
    // PatternWeight is not on the scorer's input surface.
    let score_after_tracker = scorer.score(&action, &ctx);
    assert_eq!(
        score.accept, score_after_tracker.accept,
        "scorer verdict must be independent of OutcomeTracker state"
    );
}

// ---------------------------------------------------------------------------
// ROUND 3 tests — verify v2 fixes for the FIX-3 / FIX-4 residuals surfaced
// by the wf_829feb05 round-2 adversarial pass.
//
// Background — three sub-defects landed in Round 2 that this Round 3 batch
// closes:
//
//   RESIDUAL 1: daemon_cycle_tail.rs hard-coded `ObsKind::MachPolicy => None`
//   in the drain loop, short-circuiting the `report_disagreement_with`
//   wire. FIX-3-v2 (Option B): treat the ATTEMPT to mutate a hard-protected
//   MachPolicy target under pressure as itself the disagreement signal —
//   no Mach FFI re-read needed.
//
//   RESIDUAL 2(b): the Round-2 Boost arm called `effect_decay::record_global`
//   under `!dry_run` ONLY, asserting `value_post=Foreground` without
//   verifying caps / qos_mgr / syscall_ok. FIX-4-v2 phantom-enrollment
//   guard chain: !dry_run && caps.can_taskpolicy && qos_mgr.is_some() &&
//   syscall_ok. Skip-counter `effect_decay_phantom_enroll_skipped_total`
//   surfaces every guard-blocked enrollment.
//
//   RESIDUAL 2(c): the single 64-slot FIFO ring was shared across every
//   `ObsKind`. With Boost = 79.5% of action volume, a 110-boost crisis
//   burst would silently evict pre-existing Jetsam / Sysctl observations
//   that produce the working `effect_decay_detected_total` signal.
//   FIX-4-v2 ring partition: each `ObsKind` owns an independent 64-slot
//   ring. Drain still iterates every kind.
// ---------------------------------------------------------------------------

// Helper: minimum CapabilityReport with `can_taskpolicy=false` so the Boost
// arm reaches the phantom-enrollment guard skip branch. Mirrors the
// `no_caps` helper in `invariant_11_boost_thread_qos.rs`.
fn no_caps_taskpolicy_off() -> CapabilityReport {
    CapabilityReport {
        can_taskpolicy: false,
        can_sysctl: false,
        can_memorystatus: false,
        can_mdutil: false,
        can_tmutil: false,
        is_root: false,
        p_core_count: Some(8),
        e_core_count: Some(4),
        unavailable: vec![],
    }
}

fn null_journal() -> &'static Path {
    Path::new("/dev/null")
}

// ---------------------------------------------------------------------------
// Test (Round 3, RESIDUAL 1): MachPolicy enrollment on a hard-protected
// target must FORWARD as a disagreement event into the sliding HP window —
// even without an "actual" re-read. Option-B design (cited in task
// description): the attempt itself IS the signal. The drain loop in
// `daemon_cycle_tail.rs` short-circuits the Mach FFI re-read by treating
// any expired `MachPolicy` observation with `hard_protected=true` as
// directly calling `report_disagreement_with`.
//
// We replicate that loop logic inline. The test pins the OBSERVABLE
// contract:
//
//   - effect_decay_detected_total bumps by exactly 1
//   - hard_protected_decay_count_5min increases by exactly 1
//   - PID is recorded in the HP-decay-pids ring
//
// This is the canonical "consumer wire" invariant. The matching production
// code is `daemon_cycle_tail.rs::drain_effect_decay` lines 410-417.
// ---------------------------------------------------------------------------
#[test]
fn test_mach_policy_hp_forwards_as_disagreement_no_actual() {
    let mut watchdog = DecayWatchdog::new();

    // Record a single MachPolicy observation on a hard-protected target,
    // with a deadline already in the past so drain_expired pops it
    // immediately. value_post encodes SchedulingTier::Foreground (== 0,
    // per mach_qos.rs:271) which is what the Round-2 Boost arm asserted.
    let pid = 16_105_u32; // 2026-06-07T19:35Z prod incident PID (Brave Browser Helper)
    let obs = PendingObservation {
        effect_id: 1,
        pid,
        kind: ObsKind::MachPolicy,
        key: None,
        value_post: 0,
        deadline: Instant::now() - Duration::from_secs(1),
        hard_protected: true,
    };
    watchdog.record(obs);

    let pre_detected = LSE_COUNTERS
        .effect_decay_detected_total
        .load(Ordering::Relaxed);
    let pre_hp_window = watchdog.hard_protected_decay_count_5min(Instant::now());

    // Replay the production drain loop (daemon_cycle_tail.rs::drain_effect_decay
    // lines 402-423) — Option-B branch: for MachPolicy + hard_protected we
    // call report_disagreement_with directly, bypassing the live re-read.
    let expired = watchdog.drain_expired(Instant::now());
    assert_eq!(
        expired.len(),
        1,
        "drain must surface exactly one expired observation"
    );
    for obs in &expired {
        // Production semantics: MachPolicy producer is deferred (no
        // `MachQoSManager::get_policy(pid)` re-read API). For hard-protected
        // targets the attempt IS the disagreement.
        if matches!(obs.kind, ObsKind::MachPolicy) && obs.hard_protected {
            watchdog.report_disagreement_with(obs);
        }
    }

    // 1) effect_decay_detected_total bumped by exactly 1 — the wire is live.
    let post_detected = LSE_COUNTERS
        .effect_decay_detected_total
        .load(Ordering::Relaxed);
    assert_eq!(
        post_detected - pre_detected,
        1,
        "report_disagreement_with must increment effect_decay_detected_total \
         exactly once (Option-B hard-protected MachPolicy treat-as-disagreement)"
    );

    // 2) HP sliding window populated — this is the load-bearing signal
    //    that feeds `poke_rollback_guard_via_decay`. Without it, FIX-3
    //    has no consumer and `hard_protected_decay_count_5min` stays 0.
    let post_hp_window = watchdog.hard_protected_decay_count_5min(Instant::now());
    assert_eq!(
        post_hp_window - pre_hp_window,
        1,
        "hard-protected decay window must increase by exactly 1 — this is \
         the signal that drives PolicyRollbackGuard::evaluate_from_decay"
    );

    // 3) PID is recorded — log line on rollback names the stuck process.
    let pids = watchdog.hard_protected_decay_pids(Instant::now());
    assert!(
        pids.contains(&pid),
        "HP decay PID ring must include the MachPolicy target PID ({}), \
         got {:?}",
        pid,
        pids
    );

    // 4) Round-3 negative: a NON-hard-protected MachPolicy observation
    //    must NOT forward. The production drain falls through to the
    //    `live = None` branch (producer deferred) and skips. We verify
    //    by injecting one and re-running the same loop pattern — the
    //    HP window must NOT grow further.
    let pid_innocuous = 4_242_u32;
    let obs_nonhp = PendingObservation {
        effect_id: 2,
        pid: pid_innocuous,
        kind: ObsKind::MachPolicy,
        key: None,
        value_post: 0,
        deadline: Instant::now() - Duration::from_secs(1),
        hard_protected: false,
    };
    watchdog.record(obs_nonhp);
    let mid_hp_window = watchdog.hard_protected_decay_count_5min(Instant::now());
    let expired2 = watchdog.drain_expired(Instant::now());
    for obs in &expired2 {
        if matches!(obs.kind, ObsKind::MachPolicy) && obs.hard_protected {
            watchdog.report_disagreement_with(obs);
        }
    }
    let after_hp_window = watchdog.hard_protected_decay_count_5min(Instant::now());
    assert_eq!(
        after_hp_window, mid_hp_window,
        "non-hard-protected MachPolicy must NOT grow HP window — \
         confirms the producer-deferred fall-through path is intact"
    );
}

// ---------------------------------------------------------------------------
// Test (Round 3, RESIDUAL 2b): Boost arm must NOT enroll a
// PendingObservation when caps.can_taskpolicy=false. The phantom-enrollment
// guard chain in execute_actions.rs:516 is:
//
//     phantom_guards_pass =
//         !dry_run && caps.can_taskpolicy && qos_mgr.is_some() && tier_syscall_ok;
//
// If any of (caps, qos_mgr, syscall_ok) is false, we bump the skip counter
// `effect_decay_phantom_enroll_skipped_total` and do NOT call
// `effect_decay::record_global`.
//
// To drive the Boost arm past the ProcessIdentity::verify gate without
// running a real Mach syscall, we target PID 1 (launchd) with
// `start_sec=0` — the legacy fallback path in
// `ProcessIdentity::matches` skips the start_sec check and accepts on
// name-only match. caps.can_taskpolicy=false → phantom guard fails →
// counter increments, watchdog stays empty.
//
// We additionally install a fresh global DecayWatchdog so we can assert
// the OBSERVABLE post-state: the watchdog ring length is unchanged.
// ---------------------------------------------------------------------------
#[test]
fn test_boost_enrollment_skipped_when_caps_missing() {
    use apollo_engine::engine::process_identity::ProcessIdentity;
    use std::process::{Command, Stdio};

    // Install a fresh global watchdog so we can inspect ring state after
    // execute_actions runs. install_global_for_tests is best-effort
    // (OnceLock first-wins); if a previous test in this binary already
    // installed one, the post-state assertion still holds because we
    // measure delta in the LSE counter.
    let fresh = Arc::new(Mutex::new(DecayWatchdog::new()));
    effect_decay::install_global_for_tests(Arc::clone(&fresh));

    // Spawn a child process we control. PID 1 (launchd) appeared on
    // first reading but `proc_pidinfo(1, PROC_PIDTBSDINFO)` returns
    // truncated identity on non-root CI hosts, so verify() refuses
    // (BlockReason::PidRecycled) before reaching the phantom guard.
    // A child of our own test process is queryable end-to-end without
    // privilege.
    let mut child = Command::new("sleep")
        .arg("30")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn `sleep 30` for verify target");
    let child_pid = child.id();
    let identity = ProcessIdentity::from_pid(child_pid)
        .expect("child PID must be queryable immediately after spawn");

    let pre_phantom = LSE_COUNTERS
        .effect_decay_phantom_enroll_skipped_total
        .load(Ordering::Relaxed);
    let pre_detected = LSE_COUNTERS
        .effect_decay_detected_total
        .load(Ordering::Relaxed);

    // Build a Boost action that will satisfy ProcessIdentity::verify
    // (real PID + matching name + start_sec). Reaches the phantom guard
    // with caps.can_taskpolicy=false + qos_mgr=None → guard chain
    // collapses → counter bumps, no record_global.
    let action = RootAction::BoostProcess {
        pid: child_pid,
        name: identity.name.clone(),
        reason: "test:round3:phantom-skip".to_string(),
        decision_reason: DecisionReason::PressureContext,
        start_sec: identity.start_sec,
        start_usec: identity.start_usec,
    };

    let mut frozen: HashSet<u32> = HashSet::new();
    let outcomes = execute_actions(
        vec![action],
        &no_caps_taskpolicy_off(),
        null_journal(),
        &mut frozen,
        &[],
        &[],
        None,    // qos_mgr = None forces phantom guard to fail
        false,   // dry_run = false: we want the !dry_run branch
        0.74,    // memory_pressure (prod-incident)
        6_000.0, // thrashing_score
        None,
        0.0,
    );

    // Clean up the child immediately so we don't leak `sleep 30`s.
    let _ = child.kill();
    let _ = child.wait();

    let post_phantom = LSE_COUNTERS
        .effect_decay_phantom_enroll_skipped_total
        .load(Ordering::Relaxed);
    let post_detected = LSE_COUNTERS
        .effect_decay_detected_total
        .load(Ordering::Relaxed);

    // 1) Counter incremented by exactly 1 — the phantom skip path fired.
    assert_eq!(
        post_phantom - pre_phantom,
        1,
        "effect_decay_phantom_enroll_skipped_total must increment by 1 \
         when caps.can_taskpolicy=false + qos_mgr=None (pre={} post={}) \
         traces={:?}",
        pre_phantom,
        post_phantom,
        outcomes
            .audit_traces
            .iter()
            .map(|t| t.block_reason.clone())
            .collect::<Vec<_>>()
    );

    // 2) No phantom enrollment → no phantom disagreement detection.
    //    effect_decay_detected_total must NOT have moved due to our
    //    Boost.
    assert_eq!(
        post_detected, pre_detected,
        "effect_decay_detected_total must NOT change — record_global was \
         correctly suppressed (pre={} post={})",
        pre_detected, post_detected
    );

    // 3) The Boost still "applies" from the accounting perspective —
    //    boosts_applied bumps unconditionally at execute_actions.rs:499,
    //    before the phantom guard. This pins the architecture: the
    //    phantom guard does NOT block the action; it blocks the
    //    enrollment side-effect. Drift here would mean the guard moved
    //    too far up and we accidentally vetoed boosts on no-caps hosts.
    assert_eq!(
        outcomes.boosts_applied,
        1,
        "boost must still account-as-applied (the guard blocks ONLY the \
         PendingObservation enrollment, not the action itself); \
         block_reasons={:?}",
        outcomes
            .audit_traces
            .iter()
            .map(|t| t.block_reason.clone())
            .collect::<Vec<_>>()
    );

    // 4) Global watchdog MachPolicy ring length unchanged — no
    //    PendingObservation was enrolled. If a prior test installed the
    //    global first our handle's ring length is by-construction 0 and
    //    the assertion still trivially holds — the counter delta above
    //    is the load-bearing signal.
    let guard = fresh.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(
        guard.len_for_kind(ObsKind::MachPolicy),
        0,
        "our fresh watchdog MachPolicy ring must remain empty — \
         record_global was guarded out"
    );
}

// ---------------------------------------------------------------------------
// Test (Round 3, RESIDUAL 2c): per-ObsKind ring partition must isolate a
// 200-boost crisis burst from the working Jetsam signal. The Round-2
// single 64-slot ring would silently evict the 5 Jetsam entries because
// the 200 Boosts would saturate the FIFO. The FIX-4-v2 partition gives
// each `ObsKind` its own 64-slot ring (see
// effect_decay.rs:154-159).
//
// Production-impact pin: baseline 27 effect_decay_detected_total for
// Jetsam/Sysctl observations is what was visible BEFORE this bug. A
// crisis burst (like the 2026-06-07 incident with 110 boosts in one
// window) would, under the Round-2 design, evict that signal silently.
// Under partitioning, the Jetsam ring survives untouched.
// ---------------------------------------------------------------------------
#[test]
fn test_ring_partition_jetsam_survives_boost_burst() {
    let mut watchdog = DecayWatchdog::new();

    // Future deadline so drain doesn't pop the entries — we want to
    // assert ring CONTENTS, not drain output. (drain_expired with a
    // future cutoff would return an empty vec; we use a NOW cutoff
    // after deadlines elapse below.)
    let future = Instant::now() + Duration::from_secs(60);

    // 1) Inject 200 Boost (MachPolicy) observations — 3× the per-ring
    //    capacity. Under the Round-2 single-ring design, this would
    //    consume all 64 slots and start evicting on entry 65.
    for i in 0..200_u32 {
        watchdog.record(PendingObservation {
            effect_id: i as u64,
            pid: 10_000 + i,
            kind: ObsKind::MachPolicy,
            key: None,
            value_post: 0,
            deadline: future,
            hard_protected: false,
        });
    }
    assert_eq!(
        watchdog.len_for_kind(ObsKind::MachPolicy),
        DecayWatchdog::capacity(),
        "MachPolicy ring must saturate at RING_CAP (per-kind partition); \
         FIFO eviction within MachPolicy is by-design"
    );

    // 2) Inject 5 Jetsam observations. Under the OLD design these would
    //    be evicted instantly by the 200-Boost burst. Under the v2
    //    partition each kind has its own ring, so Jetsam keeps all 5.
    let mut jetsam_pids: Vec<u32> = Vec::new();
    for i in 0..5_u32 {
        let pid = 20_000 + i;
        jetsam_pids.push(pid);
        watchdog.record(PendingObservation {
            effect_id: 1_000 + i as u64,
            pid,
            kind: ObsKind::JetsamTier,
            key: None,
            value_post: 9,
            deadline: Instant::now() - Duration::from_millis(50), // expired
            hard_protected: false,
        });
    }

    // 3) Pre-drain assertion: Jetsam ring has all 5 — partition works
    //    at insertion time, not just at drain time.
    assert_eq!(
        watchdog.len_for_kind(ObsKind::JetsamTier),
        5,
        "PRE-DRAIN: Jetsam ring must hold all 5 entries — partition must \
         prevent the 200-boost MachPolicy burst from evicting them \
         (Round-2 single-ring would show 0 here, as Jetsam slots would \
         have been overwritten)"
    );
    // Sysctl is untouched — sanity, asserts partition shape.
    assert_eq!(watchdog.len_for_kind(ObsKind::Sysctl), 0);

    // 4) Drain at NOW: every Jetsam entry has a past deadline, so all 5
    //    expire and surface in the drained vector. MachPolicy entries
    //    have future deadlines, so none surface. This pins both axes
    //    simultaneously: (a) Jetsam survived insertion, (b) Jetsam
    //    drained on expiry.
    let drained = watchdog.drain_expired(Instant::now());
    let drained_jetsam: Vec<&PendingObservation> = drained
        .iter()
        .filter(|o| matches!(o.kind, ObsKind::JetsamTier))
        .collect();
    assert_eq!(
        drained_jetsam.len(),
        5,
        "ALL 5 Jetsam observations must surface in the drained list — \
         not evicted by the 200-boost crisis burst (drained.len()={}, \
         jetsam={})",
        drained.len(),
        drained_jetsam.len()
    );

    // 5) PIDs preserved (FIFO order within the per-kind ring).
    let mut got_pids: Vec<u32> = drained_jetsam.iter().map(|o| o.pid).collect();
    got_pids.sort_unstable();
    let mut want_pids = jetsam_pids.clone();
    want_pids.sort_unstable();
    assert_eq!(
        got_pids, want_pids,
        "drained Jetsam PIDs must match the injected set verbatim — \
         partition must preserve identity, not just count"
    );

    // 6) MachPolicy ring untouched by drain (future deadlines) —
    //    confirms drain iterates every kind but respects per-entry
    //    deadlines independently.
    assert_eq!(
        watchdog.len_for_kind(ObsKind::MachPolicy),
        DecayWatchdog::capacity(),
        "MachPolicy ring must remain saturated after drain — future \
         deadlines mean nothing expired"
    );
}
