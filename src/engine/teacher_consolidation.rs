//! Teacher consolidation — S2 → S1 affective memory transfer.
//!
//! Wires the Gemma 4 teacher loop into Apollo's fast-path pattern_weights and
//! NARS beliefs via affective consolidation (dopamine burst on success,
//! acetylcholine spike on failure). The effect is that Apollo learns from
//! Gemma's suggestions *directly into its S1 reflexes* instead of needing to
//! re-consult Gemma for similar patterns.
//!
//! Papers:
//! - [McGaugh 2004] "The amygdala modulates the consolidation of memories of
//!   emotionally arousing experiences" — high-arousal events receive stronger,
//!   more durable memory encoding via dopamine + glucocorticoids.
//! - [Yerkes-Dodson 1908] inverted-U relationship between arousal and learning
//!   efficiency — we clamp consolidation to the optimal band (0.25–0.85).
//! - [Rubin 1974] Potential Outcomes framework — use counterfactual drift to
//!   separate Gemma's real causal contribution from natural fluctuation.
//! - [Kahneman 2011] "Thinking, Fast and Slow" — System 2 repeated insights
//!   should compile down to System 1 intuitions over time.
//!
//! Architecture:
//!
//!   Gemma suggests → Apollo applies → OutcomeTracker measures → SuggestionOutcome
//!        │
//!        └─→ TeacherConsolidator::consolidate()
//!             ├─ Compute causal_effect (subtract natural drift)
//!             ├─ Build Salience (arousal + valence)
//!             ├─ Dopamine burst if valence > 0:
//!             │   ├─ pattern_weights: amplify effective_count by salience
//!             │   ├─ NARS beliefs: observe_salient with amplified evidence
//!             │   └─ GemmaTrust[category]: reward
//!             ├─ Acetylcholine spike if valence < 0:
//!             │   ├─ pattern_weights: demote (throttle_count up, effective down)
//!             │   ├─ NARS beliefs: negative evidence
//!             │   └─ GemmaTrust[category]: penalty
//!             └─ Arousal state update (LTI)

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::engine::llm::{LlmSuggestion, SuggestionOutcome};
use crate::engine::nars_belief::{ArousalState, DriftDetector, Salience};
use crate::engine::outcome_tracker::PatternWeight;

/// Amplification ceiling for dopamine burst. Prevents a single very-good
/// outcome from dominating the Bayesian weights forever.
const MAX_DOPAMINE_BURST: f32 = 4.0;

/// Minimum pressure drop magnitude to trigger consolidation. Below this,
/// the outcome is considered noise and no update happens.
/// Calibrated against the natural-drift noise floor (~0.01 on M1 8GB).
const CONSOLIDATION_DEADBAND: f64 = 0.015;

/// EMA alpha for GemmaTrust scores. α=0.20 → half-life ≈ 3 observations.
/// Trust reacts quickly so bad streaks get flagged before many bad calls land.
const TRUST_ALPHA: f32 = 0.20;

/// Categories of Gemma suggestion. Tracked separately because Gemma may be
/// reliable at one class of advice (reclassification) but unreliable at
/// another (profile changes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SuggestionCategory {
    /// Added processes to interactive_patterns.
    Interactive,
    /// Added processes to noise_patterns.
    Noise,
    /// Added processes to protected_patterns.
    Protected,
    /// Changed optimization profile (balanced/aggressive/safe).
    Profile,
    /// Changed latency target (low/normal/max).
    Latency,
}

impl SuggestionCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Noise => "noise",
            Self::Protected => "protected",
            Self::Profile => "profile",
            Self::Latency => "latency",
        }
    }
}

/// Trust score for Gemma per suggestion category.
///
/// trust ∈ [0.0, 1.0]
///   0.5 = neutral (default)
///   >0.7 = reliable (Apollo trusts this class of advice)
///   <0.3 = unreliable (Apollo should skip or double-check)
///
/// Updated by EMA: trust_new = α·observation + (1-α)·trust_old
///   observation = 1.0 if outcome IMPROVED, 0.0 if WORSENED, 0.5 if NO_EFFECT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmaTrust {
    scores: HashMap<String, f32>,
    /// Count of observations per category (for evidence-based weighting).
    counts: HashMap<String, u32>,
}

impl Default for GemmaTrust {
    fn default() -> Self {
        Self {
            scores: HashMap::new(),
            counts: HashMap::new(),
        }
    }
}

impl GemmaTrust {
    /// Current trust for a category. Returns 0.5 (neutral) if never observed.
    pub fn trust(&self, cat: SuggestionCategory) -> f32 {
        self.scores.get(cat.as_str()).copied().unwrap_or(0.5)
    }

    /// Number of times this category has been observed.
    pub fn count(&self, cat: SuggestionCategory) -> u32 {
        self.counts.get(cat.as_str()).copied().unwrap_or(0)
    }

    /// Update trust EMA for a category. `observation` ∈ [0, 1].
    pub fn update(&mut self, cat: SuggestionCategory, observation: f32) {
        let key = cat.as_str().to_string();
        let prev = self.scores.get(&key).copied().unwrap_or(0.5);
        let new = TRUST_ALPHA * observation + (1.0 - TRUST_ALPHA) * prev;
        self.scores.insert(key.clone(), new.clamp(0.0, 1.0));
        *self.counts.entry(key).or_insert(0) += 1;
    }

    /// True if this category has enough evidence AND trust is reliable.
    /// Apollo can use this to gate application of Gemma's advice.
    pub fn is_reliable(&self, cat: SuggestionCategory) -> bool {
        self.count(cat) >= 3 && self.trust(cat) >= 0.70
    }
}

/// Summary of what the consolidator updated — for observability and tests.
#[derive(Debug, Clone, Default)]
pub struct ConsolidationReport {
    /// Computed salience from the outcome.
    pub arousal: f32,
    pub valence: f32,
    /// Natural drift that was subtracted (counterfactual correction).
    pub natural_drift_subtracted: f64,
    /// Causal effect after drift subtraction (what Gemma actually contributed).
    pub causal_effect: f64,
    /// Final dopamine burst amplification factor (1.0 = neutral, >1 = boost).
    pub amplification: f32,
    /// Processes whose pattern_weights were updated.
    pub weights_updated: Vec<String>,
    /// Processes whose NARS beliefs were updated.
    pub beliefs_updated: Vec<String>,
    /// Categories of Gemma trust that were updated.
    pub trust_updated: Vec<SuggestionCategory>,
    /// Verdict: IMPROVED, WORSENED, NO_EFFECT, BELOW_DEADBAND.
    pub verdict: &'static str,
}

/// The consolidator: stateless over its inputs. Owns only the GemmaTrust map
/// because that is long-lived and persists across calls.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TeacherConsolidator {
    pub gemma_trust: GemmaTrust,
    /// Total consolidations performed (for stats).
    pub total_consolidations: u64,
    /// Running count of how many were improvements.
    pub total_improvements: u64,
}

impl TeacherConsolidator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Main entry point. Takes the outcome Apollo measured, the original
    /// suggestion Gemma sent, and mutable refs to the learning structures
    /// that should receive the consolidation signal.
    ///
    /// `natural_drift_ema` comes from `OutcomeTracker::natural_drift()` and
    /// is used to subtract the counterfactual (what would have happened with
    /// no action). This prevents Gemma from getting credit for pressure that
    /// would have dropped naturally.
    pub fn consolidate(
        &mut self,
        outcome: &SuggestionOutcome,
        suggestion: &LlmSuggestion,
        natural_drift_ema: f64,
        pattern_weights: &mut HashMap<String, PatternWeight>,
        drift_detector: &mut DriftDetector,
        arousal_state: &mut ArousalState,
    ) -> ConsolidationReport {
        self.total_consolidations += 1;

        // Observed drop = pressure_before - pressure_after = -delta
        let observed_drop = -outcome.pressure_delta;
        // Causal effect: subtract natural drift. If pressure would have
        // dropped 0.03 on its own and it dropped 0.05, Gemma only gets
        // credit for 0.02.
        let causal_effect = observed_drop - natural_drift_ema;

        let mut report = ConsolidationReport::default();
        report.natural_drift_subtracted = natural_drift_ema;
        report.causal_effect = causal_effect;

        // Deadband: too small to be signal. Skip update entirely.
        if causal_effect.abs() < CONSOLIDATION_DEADBAND {
            report.verdict = "BELOW_DEADBAND";
            report.amplification = 1.0;
            return report;
        }

        // Build salience from the causal effect (not raw delta).
        // p_oom estimated from pressure_before — high pressure = high OOM risk.
        let p_oom = ((outcome.pressure_before - 0.70) / 0.30).clamp(0.0, 1.0);
        let salience = Salience::compute(
            outcome.pressure_before,
            causal_effect,
            p_oom,
            0.0, // swap_gb not tracked in outcome; use 0 as conservative default
        );
        report.arousal = salience.arousal;
        report.valence = salience.valence;

        // Yerkes-Dodson gate: consolidation is most effective in 0.20–0.70
        // arousal band. Outside that band, amplification is dampened.
        // Bounds calibrated against Salience::compute() which maxes near
        // 0.80 for worst-case pressure+p_oom (swap_gb unavailable in outcome).
        let yerkes_factor = if salience.arousal < 0.20 {
            salience.arousal / 0.20 // ramp up from 0
        } else if salience.arousal > 0.70 {
            (1.0 - (salience.arousal - 0.70) / 0.10).max(0.2) // ramp down to floor 0.2
        } else {
            1.0
        };

        // Dopamine burst amplification. Valence ∈ [-1, 1], amplification
        // applied asymmetrically: positive outcomes get boosted more.
        let base_amp = if salience.valence > 0.0 {
            // Dopamine: 1.0 + valence·arousal·yerkes·3.0 → up to 4.0x
            1.0 + (salience.valence * salience.arousal * yerkes_factor * 3.0)
        } else if salience.valence < 0.0 {
            // Acetylcholine: slight shrinkage, 0.7–1.0
            1.0 + (salience.valence * 0.3) // negative valence pulls toward 0.7
        } else {
            1.0
        };
        let amplification = base_amp.clamp(0.0, MAX_DOPAMINE_BURST);
        report.amplification = amplification;

        // Determine verdict string for the report.
        report.verdict = if salience.valence > 0.0 {
            self.total_improvements += 1;
            "IMPROVED"
        } else if salience.valence < 0.0 {
            "WORSENED"
        } else {
            "NO_EFFECT"
        };

        // ── Collect all processes mentioned in the suggestion ───────────────
        // Each category gets its own trust update because Gemma may be
        // reliable at one type of advice but not another.
        let mut touched_processes: Vec<(String, SuggestionCategory)> = Vec::new();
        for p in &suggestion.add_interactive_patterns {
            touched_processes.push((p.clone(), SuggestionCategory::Interactive));
        }
        for p in &suggestion.add_noise_patterns {
            touched_processes.push((p.clone(), SuggestionCategory::Noise));
        }
        for p in &suggestion.add_protected_patterns {
            touched_processes.push((p.clone(), SuggestionCategory::Protected));
        }

        // Track which categories had any contribution — deduped for trust update.
        let mut seen_cats: Vec<SuggestionCategory> = Vec::new();
        for (_, cat) in &touched_processes {
            if !seen_cats.contains(cat) {
                seen_cats.push(*cat);
            }
        }
        // Profile and latency changes are also categories of advice.
        if suggestion.suggested_profile.is_some() {
            seen_cats.push(SuggestionCategory::Profile);
        }
        if suggestion.suggested_latency_target.is_some() {
            seen_cats.push(SuggestionCategory::Latency);
        }

        // ── Apply consolidation to pattern_weights ──────────────────────────
        for (proc_name, _cat) in &touched_processes {
            let w = pattern_weights.entry(proc_name.clone()).or_default();
            w.throttle_count = w.throttle_count.saturating_add(1);

            if salience.valence > 0.0 {
                // Dopamine: boost effective_count by amplification factor.
                // Use saturating add to avoid overflow.
                let boost = amplification.ceil() as u32;
                w.effective_count = w.effective_count.saturating_add(boost);
            }
            // Negative valence: don't increment effective_count. The
            // throttle_count increase alone drops the Bayesian effectiveness.
            report.weights_updated.push(proc_name.clone());
        }

        // ── Apply consolidation to NARS beliefs ─────────────────────────────
        let success = salience.valence > 0.0;
        for (proc_name, _) in &touched_processes {
            drift_detector.observe_salient(proc_name, success, salience);
            report.beliefs_updated.push(proc_name.clone());
        }

        // ── Update Arousal state (global LTI) ───────────────────────────────
        arousal_state.update(salience);

        // ── Update GemmaTrust per category ──────────────────────────────────
        let observation = match report.verdict {
            "IMPROVED" => 1.0,
            "WORSENED" => 0.0,
            _ => 0.5,
        };
        for cat in &seen_cats {
            self.gemma_trust.update(*cat, observation);
            report.trust_updated.push(*cat);
        }

        report
    }

    /// Returns the running improvement ratio [0, 1]. Useful for dashboards.
    pub fn improvement_ratio(&self) -> f32 {
        if self.total_consolidations == 0 {
            return 0.0;
        }
        self.total_improvements as f32 / self.total_consolidations as f32
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn mk_outcome(before: f64, after: f64) -> SuggestionOutcome {
        SuggestionOutcome {
            applied_at: Utc::now(),
            pressure_before: before,
            pressure_after: after,
            pressure_delta: after - before,
            rationale_snippet: "test".to_string(),
        }
    }

    fn mk_suggestion() -> LlmSuggestion {
        LlmSuggestion {
            suggested_profile: None,
            suggested_latency_target: None,
            add_interactive_patterns: vec![],
            add_noise_patterns: vec!["cfprefsd".to_string()],
            add_protected_patterns: vec!["log".to_string()],
            confidence: 0.8,
            rationale: "test".to_string(),
        }
    }

    #[test]
    fn positive_outcome_boosts_pattern_weights() {
        let mut c = TeacherConsolidator::new();
        let mut weights = HashMap::new();
        let mut detector = DriftDetector::new();
        let mut arousal = ArousalState::default();

        // pressure dropped from 0.80 → 0.65 = big improvement
        let outcome = mk_outcome(0.80, 0.65);
        let suggestion = mk_suggestion();

        let report = c.consolidate(
            &outcome,
            &suggestion,
            0.0, // no natural drift
            &mut weights,
            &mut detector,
            &mut arousal,
        );

        assert_eq!(report.verdict, "IMPROVED");
        assert!(report.amplification > 1.0, "expected dopamine burst");
        assert!(weights.contains_key("cfprefsd"));
        assert!(weights.contains_key("log"));
        // effective_count should have been boosted
        let cfprefsd = &weights["cfprefsd"];
        assert_eq!(cfprefsd.throttle_count, 1);
        assert!(cfprefsd.effective_count >= 1);
    }

    #[test]
    fn negative_outcome_reduces_effectiveness() {
        let mut c = TeacherConsolidator::new();
        let mut weights = HashMap::new();
        let mut detector = DriftDetector::new();
        let mut arousal = ArousalState::default();

        // pressure went UP after Gemma's advice — bad
        let outcome = mk_outcome(0.60, 0.80);
        let suggestion = mk_suggestion();

        let report = c.consolidate(
            &outcome,
            &suggestion,
            0.0,
            &mut weights,
            &mut detector,
            &mut arousal,
        );

        assert_eq!(report.verdict, "WORSENED");
        // throttle_count went up but effective_count stayed at 0
        let cfprefsd = &weights["cfprefsd"];
        assert_eq!(cfprefsd.throttle_count, 1);
        assert_eq!(cfprefsd.effective_count, 0);
        // effectiveness should be low
        assert!(cfprefsd.effectiveness() < 0.5);
    }

    #[test]
    fn below_deadband_no_update() {
        let mut c = TeacherConsolidator::new();
        let mut weights = HashMap::new();
        let mut detector = DriftDetector::new();
        let mut arousal = ArousalState::default();

        // pressure barely changed — within noise floor
        let outcome = mk_outcome(0.60, 0.595);
        let suggestion = mk_suggestion();

        let report = c.consolidate(
            &outcome,
            &suggestion,
            0.0,
            &mut weights,
            &mut detector,
            &mut arousal,
        );

        assert_eq!(report.verdict, "BELOW_DEADBAND");
        assert!(weights.is_empty(), "no weights should be updated");
        assert!(report.weights_updated.is_empty());
    }

    #[test]
    fn counterfactual_reduces_credit_when_drift_is_large() {
        let mut c = TeacherConsolidator::new();
        let mut weights = HashMap::new();
        let mut detector = DriftDetector::new();
        let mut arousal = ArousalState::default();

        // Pressure dropped 0.10, but natural drift was 0.09 (most of the
        // drop would have happened anyway). Causal effect = 0.01 = deadband.
        let outcome = mk_outcome(0.80, 0.70);
        let suggestion = mk_suggestion();

        let report = c.consolidate(
            &outcome,
            &suggestion,
            0.09, // huge natural drift
            &mut weights,
            &mut detector,
            &mut arousal,
        );

        // After subtracting drift, effect is 0.01 < deadband 0.015
        assert_eq!(report.verdict, "BELOW_DEADBAND");
        assert!((report.causal_effect - 0.01).abs() < 1e-9);
    }

    #[test]
    fn gemma_trust_tracked_per_category() {
        let mut c = TeacherConsolidator::new();
        let mut weights = HashMap::new();
        let mut detector = DriftDetector::new();
        let mut arousal = ArousalState::default();

        let outcome = mk_outcome(0.80, 0.65);
        let suggestion = LlmSuggestion {
            suggested_profile: None,
            suggested_latency_target: None,
            add_interactive_patterns: vec![],
            add_noise_patterns: vec!["a".to_string()],
            add_protected_patterns: vec!["b".to_string()],
            confidence: 0.8,
            rationale: "test".to_string(),
        };

        c.consolidate(&outcome, &suggestion, 0.0, &mut weights, &mut detector, &mut arousal);

        // Noise and Protected should both be updated, Interactive not.
        assert!(c.gemma_trust.count(SuggestionCategory::Noise) > 0);
        assert!(c.gemma_trust.count(SuggestionCategory::Protected) > 0);
        assert_eq!(c.gemma_trust.count(SuggestionCategory::Interactive), 0);
        // Both should have trust > 0.5 after a positive outcome.
        assert!(c.gemma_trust.trust(SuggestionCategory::Noise) > 0.5);
        assert!(c.gemma_trust.trust(SuggestionCategory::Protected) > 0.5);
    }

    #[test]
    fn is_reliable_requires_minimum_evidence() {
        let mut trust = GemmaTrust::default();
        // One good observation is not enough.
        trust.update(SuggestionCategory::Noise, 1.0);
        assert!(!trust.is_reliable(SuggestionCategory::Noise));
        // After 3 good observations, trust should be ≥ 0.70
        trust.update(SuggestionCategory::Noise, 1.0);
        trust.update(SuggestionCategory::Noise, 1.0);
        // With α=0.20: trust progression 0.5 → 0.60 → 0.68 → 0.744
        assert!(trust.is_reliable(SuggestionCategory::Noise));
    }

    #[test]
    fn trust_decays_toward_failure() {
        let mut trust = GemmaTrust::default();
        // Build up trust
        for _ in 0..10 {
            trust.update(SuggestionCategory::Profile, 1.0);
        }
        let high = trust.trust(SuggestionCategory::Profile);
        assert!(high > 0.85, "expected high trust, got {}", high);

        // Hit with 5 failures
        for _ in 0..5 {
            trust.update(SuggestionCategory::Profile, 0.0);
        }
        let after = trust.trust(SuggestionCategory::Profile);
        assert!(after < high, "trust should decay");
        assert!(after < 0.50, "should be below neutral after bad streak");
    }

    #[test]
    fn s2_to_s1_convergence_simulation() {
        // Simulates the core thesis: repeated positive Gemma advice should
        // compile into Apollo's pattern_weights without needing Gemma again.
        let mut c = TeacherConsolidator::new();
        let mut weights = HashMap::new();
        let mut detector = DriftDetector::new();
        let mut arousal = ArousalState::default();

        // 5 consecutive successful consolidations
        for _ in 0..5 {
            let outcome = mk_outcome(0.80, 0.65);
            let suggestion = mk_suggestion();
            c.consolidate(&outcome, &suggestion, 0.0, &mut weights, &mut detector, &mut arousal);
        }

        // After 5 positive bursts, the pattern_weight for cfprefsd should
        // reflect high effectiveness — Apollo's S1 has internalized the lesson.
        let cfprefsd = &weights["cfprefsd"];
        assert_eq!(cfprefsd.throttle_count, 5);
        // With amplification, effective_count should be much higher than
        // throttle_count would normally allow — this IS the consolidation effect.
        assert!(
            cfprefsd.effective_count >= 5,
            "expected S1 reinforcement, got eff={}",
            cfprefsd.effective_count
        );
        // Effectiveness should be very high → is_high_value() triggers
        assert!(cfprefsd.is_high_value());
        // And Gemma trust for Noise category should be high (EMA α=0.20
        // converges toward 1.0 at rate ~0.2 per observation, so 5 good
        // observations bring it to ~0.836).
        assert!(
            c.gemma_trust.trust(SuggestionCategory::Noise) > 0.80,
            "got trust {}",
            c.gemma_trust.trust(SuggestionCategory::Noise)
        );
        assert_eq!(c.improvement_ratio(), 1.0);
    }

    #[test]
    fn yerkes_dodson_dampens_extreme_arousal() {
        let mut c = TeacherConsolidator::new();
        let mut weights = HashMap::new();
        let mut detector = DriftDetector::new();
        let mut arousal = ArousalState::default();

        // Extreme crisis: pressure 0.98, huge drop to 0.70 — should NOT
        // give maximum amplification because the system is near-OOM.
        let outcome = mk_outcome(0.98, 0.70);
        let suggestion = mk_suggestion();

        let report = c.consolidate(
            &outcome,
            &suggestion,
            0.0,
            &mut weights,
            &mut detector,
            &mut arousal,
        );

        // Arousal should be high (>0.70, within reachable range given swap_gb=0).
        assert!(report.arousal > 0.70, "got arousal {}", report.arousal);
        // Amplification should be less than theoretical max (Yerkes damping).
        assert!(
            report.amplification < MAX_DOPAMINE_BURST,
            "got amp {}",
            report.amplification
        );
    }

    // ── Benchmark ─────────────────────────────────────────────────────────
    //
    // Inline latency check: consolidate() must run in well under 100µs so
    // it can sit on the daemon hot path without budget impact.

    #[test]
    fn bench_consolidate_hot_path_cost() {
        use std::time::Instant;

        let mut c = TeacherConsolidator::new();
        let mut weights = HashMap::new();
        let mut detector = DriftDetector::new();
        let mut arousal = ArousalState::default();

        let outcome = mk_outcome(0.80, 0.65);
        let suggestion = LlmSuggestion {
            suggested_profile: Some(crate::engine::types::OptimizationProfile::BalancedRoot),
            suggested_latency_target: None,
            add_interactive_patterns: vec!["a".to_string(), "b".to_string()],
            add_noise_patterns: vec!["c".to_string(), "d".to_string(), "e".to_string()],
            add_protected_patterns: vec!["f".to_string()],
            confidence: 0.8,
            rationale: "bench".to_string(),
        };

        const ITERS: u32 = 10_000;
        let t0 = Instant::now();
        for _ in 0..ITERS {
            c.consolidate(
                &outcome,
                &suggestion,
                0.01,
                &mut weights,
                &mut detector,
                &mut arousal,
            );
        }
        let elapsed = t0.elapsed();
        let per_call_us = elapsed.as_micros() as f64 / ITERS as f64;
        eprintln!(
            "bench_consolidate: {} iterations in {:?} = {:.2} µs/call",
            ITERS, elapsed, per_call_us
        );
        // Hot-path budget: consolidate must be <100µs/call.
        assert!(
            per_call_us < 100.0,
            "consolidate too slow: {:.2} µs/call",
            per_call_us
        );
    }

    #[test]
    fn bench_learning_speed_vs_baseline() {
        // Measures effectiveness resilience under mixed outcomes.
        //
        // Setup: 10 iterations, 7 successful + 3 failed. Baseline (naive)
        // treats each outcome equally. Consolidation amplifies successes by
        // dopamine burst and dampens failures. Result: consolidation should
        // produce a higher effectiveness score from the same evidence — the
        // S2-informed S1 has stronger conviction after the same observations.
        let mut c = TeacherConsolidator::new();
        let mut weights_consol: HashMap<String, PatternWeight> = HashMap::new();
        let mut weights_naive: HashMap<String, PatternWeight> = HashMap::new();
        let mut detector = DriftDetector::new();
        let mut arousal = ArousalState::default();

        let good = mk_outcome(0.80, 0.62); // clear drop, above deadband
        let bad = mk_outcome(0.60, 0.66); // clear increase
        let suggestion = LlmSuggestion {
            suggested_profile: None,
            suggested_latency_target: None,
            add_interactive_patterns: vec![],
            add_noise_patterns: vec!["target".to_string()],
            add_protected_patterns: vec![],
            confidence: 0.8,
            rationale: "bench".to_string(),
        };

        // Interleave 7 good + 3 bad outcomes.
        let sequence = [true, true, false, true, true, false, true, true, false, true];
        for &is_good in &sequence {
            let outcome = if is_good { &good } else { &bad };
            c.consolidate(
                outcome,
                &suggestion,
                0.0,
                &mut weights_consol,
                &mut detector,
                &mut arousal,
            );
            // Naive baseline: +1 throttle, +1 effective only if good.
            let nw = weights_naive.entry("target".to_string()).or_default();
            nw.throttle_count += 1;
            if is_good {
                nw.effective_count += 1;
            }
        }

        let consol_eff = weights_consol["target"].effectiveness();
        let naive_eff = weights_naive["target"].effectiveness();
        let consol_eff_count = weights_consol["target"].effective_count;
        let naive_eff_count = weights_naive["target"].effective_count;

        eprintln!(
            "learning_speed: after 7 good + 3 bad outcomes | \
             consolidation eff={:.3} (effective_count={}) | naive eff={:.3} (effective_count={})",
            consol_eff, naive_eff, consol_eff_count, naive_eff_count
        );

        // Consolidation's effective_count should be higher than naive's
        // because of dopamine amplification on the successes.
        assert!(
            consol_eff_count > naive_eff_count,
            "consolidation should amplify successes: consol={}, naive={}",
            consol_eff_count,
            naive_eff_count
        );
        // Consolidation effectiveness should also be higher.
        assert!(
            consol_eff > naive_eff,
            "consolidation eff {:.3} not greater than naive eff {:.3}",
            consol_eff,
            naive_eff
        );
        // And Gemma trust should reflect the 70% success rate.
        let trust = c.gemma_trust.trust(SuggestionCategory::Noise);
        eprintln!("gemma_trust after mixed outcomes: {:.3}", trust);
        assert!(trust > 0.40 && trust < 0.90);
    }
}
