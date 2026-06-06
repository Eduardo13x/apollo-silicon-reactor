//! Sprint 11 Phase E — Phase 3.2 (Arousal-Modulated NARS Decay) end-to-end stress.
//!
//! Phase 3.2 wires `DriftDetector::arousal_modulated_decay_factor(...)` into
//! `LearnedState::self_improve()` (see `learned_state.rs:791-811`). When the
//! global arousal EMA enters the Crisis tier (level ≥ 0.80), the per-persist
//! NARS decay factor is reduced by 0.10 (Stressed: 0.05), accelerating
//! Bayesian forgetting so the post-crisis re-learning regime dominates.
//! The observability counter `arousal_decay_accelerations_total` bumps by
//! exactly 1 every cycle the effective factor drops below the base.
//!
//! ## Empirical gap this test closes
//! As of master `8d0f2d2`, `arousal_decay_accelerations_total = 0` across
//! every production session — synthetic Crisis arousal (level ≥ 0.80) has
//! never been observed empirically. The unit tests in `nars_belief.rs`
//! verify the helper math, but no test drives the full
//! `LearnedState::self_improve()` path with a Crisis arousal AND populated
//! beliefs to prove that:
//!   1. The counter actually bumps end-to-end.
//!   2. At least one belief's confidence drops *more* than it would have
//!      under the base decay factor.
//!   3. The path is silent when arousal is Calm (counter does not move).
//!   4. The path is silent when no `arousal_state` is set (None branch).
//!
//! ## References
//! [McGaugh 2004] "The amygdala modulates the consolidation of memories of
//! emotionally arousing experiences" — Annual Review of Neuroscience.
//! [Yerkes & Dodson 1908] inverted-U arousal vs. learning efficiency.

use std::collections::HashMap;
use std::sync::Mutex;

use apollo_engine::engine::learned_state::LearnedState;
use apollo_engine::engine::lse_counters::LSE_COUNTERS;
use apollo_engine::engine::nars_belief::{ArousalState, DriftDetector};
use apollo_engine::engine::outcome_tracker::OutcomeTrackerPersisted;

// ── Test isolation ──────────────────────────────────────────────────────────
//
// `LSE_COUNTERS` is a process-wide static `LockFreeMetrics`. Within a single
// integration-test binary `cargo` runs tests in parallel by default, so two
// tests in this file would race on the counter and the assertion `delta == 1`
// would flake. Serializing through this mutex keeps the three tests
// deterministic without requiring `--test-threads=1` at the command-line.
static COUNTER_GUARD: Mutex<()> = Mutex::new(());

// ── Fixture helpers ─────────────────────────────────────────────────────────

/// Build a `DriftDetector` with `n_beliefs` synthetic beliefs, each driven
/// past the 5-observation floor required by the test contract. We rely on
/// the public `observe()` API (the inner `BeliefEntry` is `pub(crate)`).
///
/// Mixing successes and failures keeps `frequency` mid-band so revision is
/// non-trivial — a pathological all-success or all-failure stream would
/// short-circuit the revision rule.
fn populated_drift_detector(n_beliefs: usize, obs_per_belief: u32) -> DriftDetector {
    assert!(
        obs_per_belief > 5,
        "test contract requires >5 observations per belief"
    );
    let mut dd = DriftDetector::new();
    for i in 0..n_beliefs {
        let key = format!("synthetic_action_{}", i);
        for k in 0..obs_per_belief {
            // 80% success — keeps frequency well above 0.5 and confidence
            // climbing monotonically. Concrete numbers don't matter; we
            // just need a belief with confidence > 0 to observe decay on.
            dd.observe(&key, k % 5 != 0);
        }
    }
    dd
}

/// Construct an `OutcomeTrackerPersisted` carrying the supplied drift detector
/// and otherwise-empty Bayesian state. Keeps the fixture surface minimal so
/// failures point at the decay path, not at unrelated `self_improve` branches.
fn ot_persisted_with_drift(dd: DriftDetector) -> OutcomeTrackerPersisted {
    OutcomeTrackerPersisted {
        weights: HashMap::new(),
        total_effective: 0,
        total_resolved: 0,
        baseline_drop_ema: 0.0,
        baseline_samples: 0,
        experience_records: Vec::new(),
        co_occurrence: Vec::new(),
        natural_drift_ema: 0.0,
        hop_groups: HashMap::new(),
        drift_detector: Some(dd),
        blocked_patterns: HashMap::new(),
    }
}

/// Build a `LearnedState` skeleton with all `None` slots except the two we
/// care about — `outcome_tracker` and (optionally) `arousal_state`. Default
/// `LearnableParams` means `nars_decay_factor = 0.95` (the production
/// baseline at boot).
fn fresh_state(ot: OutcomeTrackerPersisted, arousal: Option<ArousalState>) -> LearnedState {
    LearnedState {
        version: 1,
        signal_intelligence: None,
        outcome_tracker: Some(ot),
        specialist_accuracy: None,
        persist_generations: 0,
        last_restore_quality: None,
        pending_trial_skill: None,
        skill_registry: None,
        overflow_guard_history: None,
        frozen_pids: None,
        effectiveness_tracker: None,
        arousal_state: arousal,
        causal_graph_edges: None,
        process_baselines: None,
        // Explicitly Some(default) so `learnable_params.as_ref().map(...)`
        // in self_improve takes the real-prod branch (base=0.95) rather
        // than the fallback default (also 0.95 but masks the wire bug).
        learnable_params: Some(Default::default()),
        nested_learner: None,
        teacher_consolidator: None,
        unfreeze_decay_tau: None,
        neuro_state: None,
        meta_cognition: None,
        last_any_purge_at: None,
        last_cli_purge_at: None,
        companion_graph: None,
        policy_aggregator_mode: None,
    }
}

/// Construct an `ArousalState` at an arbitrary level. The struct's `alpha`
/// and `samples` fields are private-default; `level` is the only knob the
/// arousal-decay path reads (`learned_state.rs:797-801`).
fn arousal_at(level: f32) -> ArousalState {
    let mut a = ArousalState::default();
    a.level = level;
    a
}

/// Sum the per-belief confidences in a drift detector. Decay is multiplicative
/// per belief, so a smaller-overall sum after `self_improve()` is the
/// behavioural witness that decay happened.
fn confidence_total(dd: &DriftDetector, keys: &[&str]) -> f64 {
    keys.iter()
        .filter_map(|k| dd.belief(k).map(|tv| tv.confidence as f64))
        .sum()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn arousal_decay_crisis_flushes_beliefs_faster() {
    let _g = COUNTER_GUARD.lock().unwrap_or_else(|e| e.into_inner());

    // Two parallel detectors with identical seeded beliefs so we can A/B the
    // Crisis path against the Calm baseline within a single test run.
    let crisis_dd = populated_drift_detector(5, 8);
    let calm_dd = populated_drift_detector(5, 8);

    let keys: Vec<String> = (0..5).map(|i| format!("synthetic_action_{}", i)).collect();
    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();

    // Pre-decay confidence — same for both detectors (deterministic seeding).
    let crisis_pre = confidence_total(&crisis_dd, &key_refs);
    let calm_pre = confidence_total(&calm_dd, &key_refs);
    assert!(crisis_pre > 0.0, "fixture must seed non-zero confidence");
    assert!(
        (crisis_pre - calm_pre).abs() < 1e-9,
        "A/B detectors must start identical (got crisis={:.6}, calm={:.6})",
        crisis_pre,
        calm_pre
    );

    // ── Crisis run ──────────────────────────────────────────────────────────
    let before = LSE_COUNTERS.snapshot().arousal_decay_accelerations_total;

    let mut crisis_state = fresh_state(
        ot_persisted_with_drift(crisis_dd),
        Some(arousal_at(0.85)), // Crisis tier (≥ 0.80)
    );
    crisis_state.self_improve();

    let after = LSE_COUNTERS.snapshot().arousal_decay_accelerations_total;
    let delta = after - before;
    println!(
        "[crisis] arousal_decay_accelerations_total: before={} after={} delta={}",
        before, after, delta
    );
    assert_eq!(
        delta, 1,
        "Crisis arousal (0.85) must bump counter exactly once per self_improve cycle"
    );

    // ── Baseline (Calm) run on a separate state ─────────────────────────────
    // Calm = 0.10 → arousal < 0.30 → effective factor == base, counter no-op.
    let before_calm = LSE_COUNTERS.snapshot().arousal_decay_accelerations_total;
    let mut calm_state = fresh_state(ot_persisted_with_drift(calm_dd), Some(arousal_at(0.10)));
    calm_state.self_improve();
    let after_calm = LSE_COUNTERS.snapshot().arousal_decay_accelerations_total;
    assert_eq!(
        after_calm - before_calm,
        0,
        "Calm arousal must NOT accelerate decay (counter must stay flat)"
    );

    // ── Behavioural witness: Crisis confidence < Calm confidence ────────────
    let crisis_post = confidence_total(
        crisis_state
            .outcome_tracker
            .as_ref()
            .unwrap()
            .drift_detector
            .as_ref()
            .unwrap(),
        &key_refs,
    );
    let calm_post = confidence_total(
        calm_state
            .outcome_tracker
            .as_ref()
            .unwrap()
            .drift_detector
            .as_ref()
            .unwrap(),
        &key_refs,
    );
    println!(
        "[decay] crisis_post_confidence_sum={:.6} calm_post_confidence_sum={:.6} (pre={:.6})",
        crisis_post, calm_post, crisis_pre
    );
    assert!(
        crisis_post < calm_post,
        "Crisis decay (factor=base-0.10) must shrink confidence MORE than \
         Calm decay (factor=base): crisis_post={:.6} calm_post={:.6}",
        crisis_post,
        calm_post
    );
    // Both runs decayed, so both totals should be < pre. Defensive check that
    // the calm path isn't silently a no-op (which would make the comparison
    // above misleading).
    assert!(
        calm_post < crisis_pre,
        "Calm decay should still reduce confidence \
         (factor 0.95 < 1.0): calm_post={:.6} pre={:.6}",
        calm_post,
        crisis_pre
    );
}

#[test]
fn arousal_decay_neutral_when_calm() {
    let _g = COUNTER_GUARD.lock().unwrap_or_else(|e| e.into_inner());

    let dd = populated_drift_detector(5, 8);
    let keys: Vec<String> = (0..5).map(|i| format!("synthetic_action_{}", i)).collect();
    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    let pre = confidence_total(&dd, &key_refs);
    assert!(pre > 0.0, "fixture must seed non-zero confidence");

    let before = LSE_COUNTERS.snapshot().arousal_decay_accelerations_total;
    // 0.30 sits exactly at the Optimal-zone floor where the helper still
    // returns `base_factor` unchanged (see `nars_belief.rs:664-672`). The
    // counter is wired to bump strictly when `effective < base`, so this
    // must be a no-op.
    let mut state = fresh_state(ot_persisted_with_drift(dd), Some(arousal_at(0.30)));
    state.self_improve();
    let after = LSE_COUNTERS.snapshot().arousal_decay_accelerations_total;
    println!(
        "[calm] arousal=0.30 counter: before={} after={} delta={}",
        before,
        after,
        after - before
    );
    assert_eq!(
        after - before,
        0,
        "Optimal-zone arousal (0.30) must not accelerate decay"
    );

    // Beliefs still decay at the base 0.95 rate.
    let post = confidence_total(
        state
            .outcome_tracker
            .as_ref()
            .unwrap()
            .drift_detector
            .as_ref()
            .unwrap(),
        &key_refs,
    );
    assert!(
        post < pre,
        "base-rate decay should still shrink confidence: pre={:.6} post={:.6}",
        pre,
        post
    );
}

#[test]
fn arousal_decay_idempotent_no_arousal_state() {
    let _g = COUNTER_GUARD.lock().unwrap_or_else(|e| e.into_inner());

    let dd = populated_drift_detector(5, 8);
    let keys: Vec<String> = (0..5).map(|i| format!("synthetic_action_{}", i)).collect();
    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    let pre = confidence_total(&dd, &key_refs);

    let before = LSE_COUNTERS.snapshot().arousal_decay_accelerations_total;
    // No arousal_state → `arousal_level` falls back to 0.0 → Idle zone →
    // effective == base. Bump must NOT fire.
    let mut state = fresh_state(ot_persisted_with_drift(dd), None);
    state.self_improve();
    let after = LSE_COUNTERS.snapshot().arousal_decay_accelerations_total;
    println!(
        "[no-arousal] counter: before={} after={} delta={}",
        before,
        after,
        after - before
    );
    assert_eq!(
        after - before,
        0,
        "Missing arousal_state must not accelerate decay (None branch)"
    );

    let post = confidence_total(
        state
            .outcome_tracker
            .as_ref()
            .unwrap()
            .drift_detector
            .as_ref()
            .unwrap(),
        &key_refs,
    );
    assert!(
        post < pre,
        "base-rate decay still applies even without arousal_state: \
         pre={:.6} post={:.6}",
        pre,
        post
    );
}
