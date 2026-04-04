//! NARS-inspired belief revision with affective salience weighting.
//!
//! Implements TruthValue + Revision Rule from Pei Wang's NARS (2013), extended
//! with arousal-based salience weighting from cognitive neuroscience:
//!
//! # Salience & Emotional Memory
//! High-arousal events (swap=12GB, p_oom=1.0, massive pressure spike) are
//! remembered more strongly than low-arousal routine observations. This mirrors
//! the amygdala's role in memory consolidation: stress hormones (norepinephrine,
//! cortisol) strengthen synaptic encoding proportional to arousal intensity.
//!
//! [McGaugh 2004] "The amygdala modulates the consolidation of memories of
//! emotionally arousing experiences" — Annual Review of Neuroscience.
//! [Yerkes & Dodson 1908] "The relation of strength of stimulus to rapidity of
//! habit-formation" — arousal modulates learning rate.
//! [OCC model, Ortony-Clore-Collins 1988] — arousal + valence as independent
//! dimensions; valence = positive/negative outcome, arousal = intensity.
//!
//! # Implementation
//! - `Salience { arousal, valence }` — computed from pressure metrics
//! - High arousal → higher evidence weight in NARS Revision
//!   (acts like N observations instead of 1, where N ∝ arousal)
//! - `BeliefEntry.lti` (Long-Term Importance, OpenCog-inspired):
//!   high-arousal beliefs decay slower — emotionally significant memories persist
//! - Low-arousal routine observations use the standard single-observation weight
//!
//! # Revision Rule (Pei Wang 2013, §3.3.3)
//! w = c / (1 - c)
//! f_new = (w1·f1 + w2·f2) / (w1 + w2)
//! c_new = (w1 + w2) / (w1 + w2 + 1)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Constants ────────────────────────────────────────────────────────────────

/// Frequency shift that triggers a drift alert for a single belief.
/// A 20pp shift means the action's effectiveness profile has materially changed.
/// Inspired by Population Stability Index threshold (PSI ≥ 0.20 = major shift).
const DRIFT_THRESHOLD: f32 = 0.20;

/// Minimum confidence before drift can be declared (need enough evidence).
const MIN_CONFIDENCE_FOR_DRIFT: f32 = 0.30;

/// EMA alpha for aggregate drift score (slow-decaying: half-life ≈ 69 ticks).
const DRIFT_SCORE_ALPHA: f64 = 0.01;

/// Max equivalent observations for a maximum-arousal event.
/// arousal=1.0 → evidence weight = confidence_from_count(MAX_SALIENT_OBS).
/// This means a crisis event (swap=12GB, p_oom=1.0) counts as strongly as
/// 4 normal observations — 4× faster belief update under maximum stress.
/// [McGaugh 2004] amygdala modulation: arousal boosts memory consolidation.
const MAX_SALIENT_OBS: u32 = 4;

/// LTI decay protection: high-arousal beliefs decay at this slower rate.
/// Standard decay = 0.95; LTI-protected decay = 0.985 (3× slower fading).
/// Equivalent to long-term potentiation (LTP) in neuroscience: strong stimuli
/// produce lasting synaptic changes. [Bliss & Lømo 1973] LTP paper.
const LTI_DECAY_FACTOR: f32 = 0.985;

/// Arousal threshold above which LTI protection is granted.
const LTI_AROUSAL_THRESHOLD: f32 = 0.60;

// ── TruthValue ───────────────────────────────────────────────────────────────

/// NARS TruthValue: (frequency, confidence).
///
/// frequency ∈ [0,1]: P(proposition is true | all evidence)
/// confidence ∈ [0,1): evidence weight; approaches 1 asymptotically
///
/// [Pei Wang 2013] "Non-Axiomatic Reasoning System", §3.3
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct TruthValue {
    /// How often this proposition was true. 0 = never, 1 = always.
    pub frequency: f32,
    /// Evidence weight. Starts near 0, grows toward 1 with each observation.
    pub confidence: f32,
}

impl Default for TruthValue {
    /// Ignorance prior: no evidence either way.
    fn default() -> Self {
        Self { frequency: 0.5, confidence: 0.0 }
    }
}

impl TruthValue {
    pub fn new(frequency: f32, confidence: f32) -> Self {
        Self {
            frequency: frequency.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 0.9999),
        }
    }

    /// Expected value: P(true) weighted by confidence.
    /// Unconfident beliefs regress toward 0.5 (maximum uncertainty).
    /// [Pei Wang 2013] §3.3.1 — expectation = f·c + 0.5·(1-c)
    pub fn expectation(&self) -> f32 {
        self.frequency * self.confidence + 0.5 * (1.0 - self.confidence)
    }

    /// Apply the NARS Revision Rule: merge two independent observations.
    ///
    /// Returns updated TruthValue after incorporating new evidence.
    /// Revision is symmetric and commutative.
    ///
    /// [Pei Wang 2013] §3.3.3 — Revision
    pub fn revise(self, new_evidence: TruthValue) -> TruthValue {
        let eps = 1e-6_f32;
        let w1 = self.confidence / (1.0 - self.confidence + eps);
        let w2 = new_evidence.confidence / (1.0 - new_evidence.confidence + eps);
        let w = w1 + w2;
        if w < eps {
            return self;
        }
        let f_new = (w1 * self.frequency + w2 * new_evidence.frequency) / w;
        let c_new = w / (w + 1.0);
        TruthValue::new(f_new, c_new)
    }

    /// Confidence from evidence count n using the NARS formula: c = n / (n + k)
    /// where k = 1 (Laplace-like prior strength).
    pub fn confidence_from_count(n: u32) -> f32 {
        n as f32 / (n as f32 + 1.0)
    }
}

// ── Salience ─────────────────────────────────────────────────────────────────

/// Affective salience of an observation event.
///
/// Captures the emotional intensity (arousal) and outcome quality (valence)
/// of a system event, inspired by the VAD model (Valence-Arousal-Dominance).
///
/// [Russell 1980] "A circumplex model of affect" — Journal of Personality and
/// Social Psychology. Arousal and valence are orthogonal dimensions.
/// [OCC model, Ortony-Clore-Collins 1988] — appraisal theory of emotion.
///
/// In Apollo's context:
/// - arousal = how intense/critical the event was (pressure level, OOM risk, swap size)
/// - valence = was the outcome good (+1) or bad (-1)?
///   Positive valence = action reduced pressure. Negative = did nothing or worse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Salience {
    /// Event intensity ∈ [0,1]. 0 = routine, 1 = maximum crisis.
    /// Computed from: pressure_before, pressure_drop, p_oom, swap_gb.
    pub arousal: f32,
    /// Outcome quality ∈ [-1,1]. +1 = highly effective, -1 = counterproductive.
    pub valence: f32,
}

impl Salience {
    /// Neutral observation: no special emotional weight.
    pub fn neutral() -> Self {
        Self { arousal: 0.0, valence: 0.0 }
    }

    /// Compute salience from system metrics.
    ///
    /// Arousal formula:
    ///   arousal = clamp(0.4·pressure_before + 0.4·p_oom + 0.2·swap_factor, 0, 1)
    ///   swap_factor = min(swap_gb / 8.0, 1.0)  (8GB = full saturation)
    ///
    /// Valence formula:
    ///   +1.0 if pressure_drop >= 0.10 (large effective drop)
    ///   +0.5 if pressure_drop >= 0.01 (small but effective)
    ///    0.0 if pressure_drop == 0    (no effect)
    ///   -0.5 if pressure_drop <  0   (pressure increased)
    pub fn compute(
        pressure_before: f64,
        pressure_drop: f64,
        p_oom: f64,
        swap_gb: f64,
    ) -> Self {
        let swap_factor = (swap_gb / 8.0).min(1.0) as f32;
        let arousal = (0.4 * pressure_before as f32
            + 0.4 * p_oom as f32
            + 0.2 * swap_factor)
            .clamp(0.0, 1.0);

        let valence = if pressure_drop >= 0.10 {
            1.0_f32
        } else if pressure_drop >= 0.01 {
            0.5
        } else if pressure_drop < 0.0 {
            -0.5
        } else {
            0.0
        };

        Self { arousal, valence }
    }

    /// Evidence count equivalent for NARS Revision.
    /// Maps arousal [0,1] → [1, MAX_SALIENT_OBS] observations.
    /// A maximum-crisis event counts as MAX_SALIENT_OBS independent observations.
    pub fn evidence_count(&self) -> u32 {
        1 + (self.arousal * (MAX_SALIENT_OBS - 1) as f32).round() as u32
    }

    /// True if this event warrants Long-Term Importance protection.
    /// High-arousal events form durable memories. [Bliss & Lømo 1973] LTP.
    pub fn grants_lti(&self) -> bool {
        self.arousal >= LTI_AROUSAL_THRESHOLD
    }
}

impl Default for Salience {
    fn default() -> Self {
        Self::neutral()
    }
}

// ── BeliefEntry ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BeliefEntry {
    tv: TruthValue,
    /// Frequency value before the last revision — used to compute drift delta.
    freq_before_last_revision: f32,
    /// Total observations that fed this belief.
    observations: u32,
    /// Long-Term Importance ∈ [0,1]. Accumulates when high-arousal events
    /// grant LTI protection. High LTI → slower confidence decay (durable memory).
    /// [OpenCog STI/LTI, Goertzel 2010] + [Bliss & Lømo 1973] LTP.
    #[serde(default)]
    lti: f32,
}

impl BeliefEntry {
    fn new(initial_freq: f32) -> Self {
        Self {
            tv: TruthValue::new(initial_freq, TruthValue::confidence_from_count(1)),
            freq_before_last_revision: initial_freq,
            observations: 1,
            lti: 0.0,
        }
    }

    /// Incorporate a new observation with neutral salience (standard weight).
    fn observe(&mut self, success: bool) -> f32 {
        self.observe_salient(success, Salience::neutral())
    }

    /// Incorporate a new observation with explicit salience weighting.
    ///
    /// High arousal → higher evidence weight → faster belief update.
    /// High arousal + LTI threshold → LTI credit accumulated.
    ///
    /// [McGaugh 2004] amygdala modulation of memory consolidation.
    fn observe_salient(&mut self, success: bool, salience: Salience) -> f32 {
        self.observations += 1;
        let new_freq = if success { 1.0_f32 } else { 0.0_f32 };
        // Evidence weight scaled by arousal: high-arousal events count as
        // multiple observations (up to MAX_SALIENT_OBS).
        let evidence_n = salience.evidence_count();
        let new_conf = TruthValue::confidence_from_count(evidence_n);
        let new_evidence = TruthValue::new(new_freq, new_conf);
        self.freq_before_last_revision = self.tv.frequency;
        self.tv = self.tv.revise(new_evidence);
        // Accumulate LTI for high-arousal events (long-term potentiation).
        if salience.grants_lti() {
            self.lti = (self.lti + 0.05).min(1.0);
        }
        (self.tv.frequency - self.freq_before_last_revision).abs()
    }

    /// Effective decay factor for this belief.
    /// High-LTI beliefs use LTI_DECAY_FACTOR instead of the caller's factor.
    /// Models the durability of emotionally significant memories.
    fn effective_decay(&self, base_factor: f32) -> f32 {
        if self.lti > 0.3 {
            // LTI protection: decay slower. Blend based on LTI strength.
            base_factor + (LTI_DECAY_FACTOR - base_factor) * self.lti
        } else {
            base_factor
        }
    }

    /// True if this belief has shifted significantly since last calibration.
    fn is_drifted(&self) -> bool {
        let delta = (self.tv.frequency - self.freq_before_last_revision).abs();
        self.tv.confidence >= MIN_CONFIDENCE_FOR_DRIFT && delta >= DRIFT_THRESHOLD
    }
}

// ── DriftDetector ────────────────────────────────────────────────────────────

/// Tracks effectiveness beliefs for a set of named actions/specialists.
/// Detects concept drift via NARS Revision: when frequency shifts ≥20pp
/// with sufficient confidence, the learned model no longer matches reality.
///
/// Drift score ∈ [0,1]: 0 = stable, 1 = total model invalidation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DriftDetector {
    beliefs: HashMap<String, BeliefEntry>,
    /// EMA of per-belief drift deltas. High = model is drifting.
    pub drift_score: f64,
    /// Number of beliefs currently in a drifted state.
    pub drifted_count: usize,
}

impl DriftDetector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an observation with neutral salience (standard weight).
    /// `success` = did the action produce a good outcome?
    /// Returns the local frequency delta from revision.
    pub fn observe(&mut self, key: &str, success: bool) -> f32 {
        self.observe_salient(key, success, Salience::neutral())
    }

    /// Record an observation with explicit affective salience.
    ///
    /// High-arousal events (memory crisis, OOM risk) update beliefs faster
    /// and may earn LTI protection (slower decay). This is the primary
    /// mechanism for emotionally-weighted memory in Apollo.
    ///
    /// [McGaugh 2004] amygdala modulates memory consolidation proportional
    /// to arousal intensity — high stress → stronger, more durable memories.
    pub fn observe_salient(&mut self, key: &str, success: bool, salience: Salience) -> f32 {
        let delta = if let Some(entry) = self.beliefs.get_mut(key) {
            entry.observe_salient(success, salience)
        } else {
            let initial_freq = if success { 1.0 } else { 0.0 };
            let mut entry = BeliefEntry::new(initial_freq);
            // Apply salience to the first observation too (not just subsequent ones).
            if salience.grants_lti() {
                entry.lti = 0.05;
            }
            self.beliefs.insert(key.to_string(), entry);
            0.0 // first observation: no drift yet
        };

        // Arousal amplifies the drift EMA signal too — a crisis-level regime
        // change is more alarming than a routine one.
        let arousal_amp = 1.0 + salience.arousal as f64;
        self.drift_score = DRIFT_SCORE_ALPHA * delta as f64 * arousal_amp
            + (1.0 - DRIFT_SCORE_ALPHA) * self.drift_score;

        // Recount drifted beliefs
        self.drifted_count = self.beliefs.values().filter(|e| e.is_drifted()).count();

        delta
    }

    /// Current arousal-weighted drift score [0,1].
    /// EMA of per-belief drift deltas, amplified by event arousal.
    pub fn score(&self) -> f64 {
        self.drift_score
    }

    /// True if model drift is significant enough to warrant recalibration.
    /// Threshold: ≥2 beliefs drifted OR aggregate EMA score > 0.08.
    pub fn needs_recalibration(&self) -> bool {
        self.needs_recalibration_at(0.08)
    }

    /// Like `needs_recalibration()` but with a caller-supplied EMA threshold.
    ///
    /// Used by `ArousalState::adjusted_drift_threshold()` to dynamically tighten
    /// or loosen the recalibration trigger per Yerkes-Dodson:
    /// - High arousal → lower threshold → faster recalibration response
    /// - Low arousal  → higher threshold → conservative (avoids false alarms)
    pub fn needs_recalibration_at(&self, score_threshold: f64) -> bool {
        self.drifted_count >= 2 || self.drift_score > score_threshold
    }

    /// Get current TruthValue for a key (for diagnostics).
    pub fn belief(&self, key: &str) -> Option<TruthValue> {
        self.beliefs.get(key).map(|e| e.tv)
    }

    /// Reset drift signals after recalibration has been applied.
    /// Does NOT reset the beliefs themselves — keeps accumulated evidence.
    pub fn acknowledge_recalibration(&mut self) {
        self.drift_score *= 0.1; // decay but don't erase
        for entry in self.beliefs.values_mut() {
            entry.freq_before_last_revision = entry.tv.frequency;
        }
        self.drifted_count = 0;
    }

    /// Decay confidence of all beliefs by `factor` (0 < factor < 1).
    ///
    /// Simulates Bayesian forgetting: old evidence becomes less certain over time.
    /// With factor=0.95, confidence halves every ~14 persist cycles.
    /// After decay, new observations will have proportionally more influence.
    ///
    /// Beliefs with confidence < 0.05 after decay are pruned (noise floor).
    /// [Bayesian forgetting: Pfau et al. 2010, Streaming Bayesian Updates]
    /// Decay confidence of all beliefs by `factor` (0 < factor < 1).
    ///
    /// High-LTI beliefs (formed during crisis events) decay slower —
    /// they use `effective_decay()` which blends toward LTI_DECAY_FACTOR.
    /// This models long-term potentiation: emotionally significant memories
    /// persist longer than routine ones. [Bliss & Lømo 1973] LTP.
    ///
    /// Standard: 0.95/cycle → half-life ≈ 14 cycles.
    /// LTI-protected: 0.985/cycle → half-life ≈ 46 cycles (3× more durable).
    pub fn decay_confidence(&mut self, factor: f32) {
        let factor = factor.clamp(0.0, 1.0);
        let mut to_remove = Vec::new();
        for (key, entry) in &mut self.beliefs {
            let eff = entry.effective_decay(factor);
            entry.tv.confidence *= eff;
            // LTI itself also decays slowly (1% per persist cycle).
            entry.lti = (entry.lti * 0.99).max(0.0);
            if entry.tv.confidence < 0.05 {
                to_remove.push(key.clone());
            }
        }
        for key in to_remove {
            self.beliefs.remove(&key);
        }
        // Recount drifted beliefs after pruning
        self.drifted_count = self.beliefs.values().filter(|e| e.is_drifted()).count();
    }

    /// Number of tracked beliefs.
    pub fn len(&self) -> usize {
        self.beliefs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.beliefs.is_empty()
    }
}

// ── ArousalState ─────────────────────────────────────────────────────────────

/// EMA-based global arousal tracker for the daemon.
///
/// Tracks the daemon's current system-wide stress level as a continuous signal
/// ∈ [0,1]. High arousal = system under crisis (swap full, p_oom elevated).
/// Low arousal = system idle / healthy.
///
/// Yerkes-Dodson (1908): arousal modulates learning efficiency — too low =
/// inattentive, too high = overwhelmed, optimal band (0.3–0.7) = peak learning.
///
/// Used in learning_tick.rs to:
/// - Tighten recalibration threshold under high arousal (faster drift response)
/// - Expand recalibration threshold under low arousal (avoid false positives)
/// - Gate expensive subsystems when arousal is trivially low
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArousalState {
    /// Current EMA arousal level ∈ [0,1].
    pub level: f32,
    /// EMA decay factor. α=0.15 → half-life ≈ 4 samples (fast-reacting).
    alpha: f32,
    /// Count of observations fed into this EMA.
    pub samples: u64,
}

impl Default for ArousalState {
    fn default() -> Self {
        Self { level: 0.0, alpha: 0.15, samples: 0 }
    }
}

impl ArousalState {
    /// Update arousal EMA with a new salience observation.
    ///
    /// The EMA reacts quickly to spikes (α=0.15) but decays slowly when inputs
    /// are low — mimicking the lingering effect of stress hormones (cortisol
    /// half-life ≈ 60–90 min, modeled here as persistent EMA memory).
    pub fn update(&mut self, salience: Salience) {
        self.level = self.alpha * salience.arousal + (1.0 - self.alpha) * self.level;
        self.samples += 1;
    }

    /// Drift recalibration threshold adjusted by Yerkes-Dodson inverted-U.
    ///
    /// At low arousal: raise threshold (don't recalibrate on noise).
    /// At high arousal: lower threshold (fast response to real crises).
    /// At optimal arousal (0.5): return the base threshold unchanged.
    ///
    /// Formula: threshold × (1.0 + 0.5 × (0.5 - arousal))
    ///   arousal=0.0 → threshold × 1.25 (sluggish, conservative)
    ///   arousal=0.5 → threshold × 1.00 (baseline)
    ///   arousal=1.0 → threshold × 0.75 (hair-trigger, aggressive)
    pub fn adjusted_drift_threshold(&self, base: f64) -> f64 {
        let arousal = self.level as f64;
        base * (1.0 + 0.5 * (0.5 - arousal))
    }

    /// Yerkes-Dodson zone label for dashboard display.
    pub fn zone(&self) -> &'static str {
        match self.level {
            a if a < 0.25 => "Idle",
            a if a < 0.45 => "Calm",
            a if a < 0.65 => "Optimal",
            a if a < 0.80 => "Stressed",
            _ => "Crisis",
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TruthValue ────────────────────────────────────────────────────────────

    #[test]
    fn truth_value_defaults_to_ignorance_prior() {
        let tv = TruthValue::default();
        assert_eq!(tv.frequency, 0.5);
        assert_eq!(tv.confidence, 0.0);
        // Expectation of ignorance = 0.5
        assert!((tv.expectation() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn revision_symmetric_equal_evidence() {
        // Two equal observations → same result
        let tv1 = TruthValue::new(0.8, 0.6);
        let tv2 = TruthValue::new(0.2, 0.6);
        let r1 = tv1.revise(tv2);
        let r2 = tv2.revise(tv1);
        // Symmetric
        assert!((r1.frequency - r2.frequency).abs() < 1e-5, "revision should be symmetric");
        assert!((r1.confidence - r2.confidence).abs() < 1e-5);
        // Midpoint
        assert!((r1.frequency - 0.5).abs() < 0.01, "equal evidence → midpoint");
    }

    #[test]
    fn revision_higher_confidence_dominates() {
        // High-confidence belief should pull result toward its frequency
        let strong = TruthValue::new(0.9, 0.9);
        let weak = TruthValue::new(0.1, 0.1);
        let result = strong.revise(weak);
        assert!(result.frequency > 0.7, "strong belief should dominate: got {}", result.frequency);
        assert!(result.confidence > strong.confidence, "confidence should grow after revision");
    }

    #[test]
    fn revision_confidence_grows_monotonically() {
        let mut tv = TruthValue::new(0.5, 0.3);
        for _ in 0..10 {
            let prev_conf = tv.confidence;
            tv = tv.revise(TruthValue::new(0.5, 0.1));
            assert!(tv.confidence > prev_conf, "confidence must grow with each observation");
        }
    }

    #[test]
    fn confidence_from_count_approaches_one() {
        assert!((TruthValue::confidence_from_count(1) - 0.5).abs() < 1e-5);
        assert!((TruthValue::confidence_from_count(9) - 0.9).abs() < 1e-5);
        assert!(TruthValue::confidence_from_count(999) > 0.99);
        assert!(TruthValue::confidence_from_count(9999) > 0.999);
    }

    #[test]
    fn expectation_regresses_toward_half_for_low_confidence() {
        let tv = TruthValue::new(1.0, 0.0);
        // With zero confidence, expectation = 0.5 regardless of frequency
        assert!((tv.expectation() - 0.5).abs() < 1e-5);
    }

    // ── DriftDetector ─────────────────────────────────────────────────────────

    #[test]
    fn drift_detector_no_drift_on_consistent_outcomes() {
        let mut dd = DriftDetector::new();
        // 20 consistent successes → stable model
        for _ in 0..20 {
            dd.observe("proc_A", true);
        }
        assert!(!dd.needs_recalibration(), "consistent outcomes → no drift");
        assert!(dd.drift_score < 0.05);
    }

    #[test]
    fn drift_detector_detects_regime_change() {
        let mut dd = DriftDetector::new();
        // Phase 1: process always effective
        for _ in 0..30 {
            dd.observe("proc_X", true);
        }
        let score_before = dd.drift_score;
        // Phase 2: suddenly never effective (regime change)
        for _ in 0..30 {
            dd.observe("proc_X", false);
        }
        // Drift score should increase
        assert!(
            dd.drift_score > score_before || dd.drifted_count >= 1,
            "regime change should increase drift signal"
        );
    }

    #[test]
    fn drift_detector_acknowledge_resets_signal() {
        let mut dd = DriftDetector::new();
        for _ in 0..30 {
            dd.observe("proc_A", true);
        }
        for _ in 0..30 {
            dd.observe("proc_A", false);
        }
        let drift_before = dd.drift_score;
        dd.acknowledge_recalibration();
        assert!(dd.drift_score < drift_before * 0.5, "acknowledge should reduce drift score");
        assert_eq!(dd.drifted_count, 0);
    }

    #[test]
    fn drift_detector_multiple_beliefs_tracked_independently() {
        let mut dd = DriftDetector::new();
        // proc_A: stable
        for _ in 0..20 {
            dd.observe("proc_A", true);
        }
        // proc_B: unstable
        for _ in 0..10 {
            dd.observe("proc_B", true);
        }
        for _ in 0..20 {
            dd.observe("proc_B", false);
        }
        assert_eq!(dd.len(), 2);
        let tv_a = dd.belief("proc_A").unwrap();
        let tv_b = dd.belief("proc_B").unwrap();
        assert!(tv_a.frequency > 0.7, "proc_A should have high frequency");
        assert!(tv_b.frequency < 0.5, "proc_B should have lower frequency after failures");
    }

    #[test]
    fn drift_detector_first_observation_no_drift() {
        let mut dd = DriftDetector::new();
        let delta = dd.observe("new_process", true);
        assert_eq!(delta, 0.0, "first observation produces no drift delta");
        assert!(!dd.needs_recalibration());
    }

    #[test]
    fn drift_detector_decay_prunes_low_confidence_beliefs() {
        let mut dd = DriftDetector::new();
        for _ in 0..5 {
            dd.observe("proc_A", true);
        }
        assert_eq!(dd.len(), 1);
        // Decay 20 times at 0.5 factor: 0.5^20 → effectively 0
        for _ in 0..20 {
            dd.decay_confidence(0.5);
        }
        // proc_A should be pruned (confidence < 0.05)
        assert_eq!(dd.len(), 0, "fully decayed belief should be pruned");
    }

    #[test]
    fn drift_detector_decay_reduces_confidence_not_frequency() {
        let mut dd = DriftDetector::new();
        for _ in 0..10 {
            dd.observe("proc_B", true);
        }
        let tv_before = dd.belief("proc_B").unwrap();
        dd.decay_confidence(0.95);
        let tv_after = dd.belief("proc_B").unwrap();
        // Confidence decays
        assert!(tv_after.confidence < tv_before.confidence, "confidence must decay");
        // Frequency is preserved (decay only affects evidence weight, not outcome)
        assert!((tv_after.frequency - tv_before.frequency).abs() < 1e-4,
            "frequency must not change after decay");
    }

    #[test]
    fn revision_rule_math_from_paper() {
        let tv1 = TruthValue::new(0.8, 0.6);
        let tv2 = TruthValue::new(0.8, 0.6);
        let result = tv1.revise(tv2);
        assert!((result.frequency - 0.8).abs() < 0.001, "same freq → no change: {}", result.frequency);
        assert!((result.confidence - 0.75).abs() < 0.001, "c_new=0.75: {}", result.confidence);
    }

    // ── Salience tests ────────────────────────────────────────────────────────

    #[test]
    fn salience_neutral_has_arousal_zero() {
        let s = Salience::neutral();
        assert_eq!(s.arousal, 0.0);
        assert_eq!(s.valence, 0.0);
        assert_eq!(s.evidence_count(), 1, "neutral → 1 observation equivalent");
        assert!(!s.grants_lti(), "neutral → no LTI protection");
    }

    #[test]
    fn salience_max_crisis_grants_lti_and_max_evidence() {
        // Maximum crisis: full pressure, full OOM risk, 8+ GB swap
        let s = Salience::compute(1.0, 0.15, 1.0, 8.0);
        assert!(s.arousal > 0.9, "full crisis → near-max arousal: {}", s.arousal);
        assert!(s.grants_lti(), "high arousal → LTI protection");
        assert_eq!(s.evidence_count(), MAX_SALIENT_OBS,
            "max arousal → MAX_SALIENT_OBS evidence equivalent");
        assert_eq!(s.valence, 1.0, "large drop → positive valence");
    }

    #[test]
    fn salience_routine_low_pressure_no_lti() {
        // Routine: low pressure, effective small drop, no swap
        let s = Salience::compute(0.20, 0.02, 0.05, 0.1);
        assert!(s.arousal < 0.3, "low pressure → low arousal: {}", s.arousal);
        assert!(!s.grants_lti(), "low arousal → no LTI");
        assert_eq!(s.evidence_count(), 1, "low arousal → single observation weight");
        assert_eq!(s.valence, 0.5, "small drop → moderate positive valence");
    }

    #[test]
    fn salience_negative_valence_when_pressure_increased() {
        let s = Salience::compute(0.80, -0.05, 0.70, 2.0);
        assert_eq!(s.valence, -0.5, "pressure increase → negative valence");
    }

    #[test]
    fn high_arousal_observation_updates_belief_faster() {
        let mut dd_normal = DriftDetector::new();
        let mut dd_crisis = DriftDetector::new();

        // Both see the same false outcome, but crisis has high arousal
        let crisis = Salience::compute(0.95, 0.0, 0.98, 10.0);

        // Start both with 10 true observations (high confidence)
        for _ in 0..10 {
            dd_normal.observe("proc", true);
            dd_crisis.observe_salient("proc", true, crisis);
        }

        // Now one failure arrives
        dd_normal.observe("proc", false);
        dd_crisis.observe_salient("proc", false, crisis);

        let tv_normal = dd_normal.belief("proc").unwrap();
        let tv_crisis = dd_crisis.belief("proc").unwrap();

        // Crisis belief should have moved further from 1.0 (faster update)
        let normal_drop = 1.0 - tv_normal.frequency;
        let crisis_drop = 1.0 - tv_crisis.frequency;
        assert!(
            crisis_drop > normal_drop,
            "high-arousal failure should update belief faster: normal_drop={:.3} crisis_drop={:.3}",
            normal_drop, crisis_drop
        );
    }

    #[test]
    fn lti_protection_slows_decay() {
        let mut dd_normal = DriftDetector::new();
        let mut dd_crisis = DriftDetector::new();

        let crisis = Salience::compute(0.95, 0.12, 0.95, 9.0);

        // Crisis tracker gets LTI via high-arousal observations
        for _ in 0..10 {
            dd_normal.observe("proc", true);
            dd_crisis.observe_salient("proc", true, crisis);
        }

        // Apply 20 decay cycles
        for _ in 0..20 {
            dd_normal.decay_confidence(0.95);
            dd_crisis.decay_confidence(0.95);
        }

        let tv_normal = dd_normal.belief("proc");
        let tv_crisis = dd_crisis.belief("proc").unwrap();

        // Crisis belief should survive with higher confidence (LTI protection)
        let crisis_conf = tv_crisis.confidence;
        let normal_conf = tv_normal.map(|tv| tv.confidence).unwrap_or(0.0);
        assert!(
            crisis_conf > normal_conf,
            "LTI-protected belief should decay slower: crisis={:.3} normal={:.3}",
            crisis_conf, normal_conf
        );
    }

    #[test]
    fn arousal_amplifies_drift_score_ema() {
        let mut dd_neutral = DriftDetector::new();
        let mut dd_crisis = DriftDetector::new();

        // Build up beliefs
        for _ in 0..10 {
            dd_neutral.observe("proc", true);
            dd_crisis.observe_salient("proc", true, Salience::compute(0.9, 0.1, 0.9, 8.0));
        }

        // Regime change
        dd_neutral.observe("proc", false);
        dd_crisis.observe_salient("proc", false, Salience::compute(0.9, -0.05, 0.9, 8.0));

        // Crisis drift score should be higher due to arousal amplification
        assert!(
            dd_crisis.drift_score >= dd_neutral.drift_score,
            "arousal should amplify drift score EMA: crisis={:.4} neutral={:.4}",
            dd_crisis.drift_score, dd_neutral.drift_score
        );
    }

    // ── ArousalState ──────────────────────────────────────────────────────────

    #[test]
    fn arousal_state_starts_at_zero() {
        let a = ArousalState::default();
        assert_eq!(a.level, 0.0);
        assert_eq!(a.samples, 0);
        assert_eq!(a.zone(), "Idle");
    }

    #[test]
    fn arousal_state_ema_rises_under_crisis() {
        let mut a = ArousalState::default();
        let crisis = Salience::compute(0.9, -0.05, 0.9, 8.0); // high swap, high p_oom
        for _ in 0..30 {
            a.update(crisis);
        }
        // After 30 updates with high-arousal input, EMA should converge near crisis.arousal
        assert!(a.level > 0.60, "EMA should approach crisis arousal: got {:.3}", a.level);
        assert!(matches!(a.zone(), "Stressed" | "Crisis"));
    }

    #[test]
    fn arousal_state_decays_back_to_idle() {
        let mut a = ArousalState::default();
        let crisis = Salience::compute(0.9, -0.05, 0.9, 8.0);
        // Build up arousal
        for _ in 0..30 { a.update(crisis); }
        assert!(a.level > 0.50);
        // Feed zero-arousal inputs — EMA decays
        let calm = Salience::compute(0.1, 0.01, 0.0, 0.0);
        for _ in 0..60 { a.update(calm); }
        assert!(a.level < 0.20, "EMA should decay toward calm: got {:.3}", a.level);
    }

    #[test]
    fn arousal_adjusted_threshold_follows_yerkes_dodson() {
        let base = 0.08_f64;
        let mut a = ArousalState::default();

        // Low arousal → threshold raised (conservative)
        let low = Salience::compute(0.0, 0.0, 0.0, 0.0);
        for _ in 0..50 { a.update(low); }
        let t_low = a.adjusted_drift_threshold(base);
        assert!(t_low > base, "low arousal should raise threshold: {:.4}", t_low);

        // High arousal → threshold lowered (aggressive)
        let mut b = ArousalState::default();
        let crisis = Salience::compute(0.9, -0.05, 0.9, 8.0);
        for _ in 0..50 { b.update(crisis); }
        let t_high = b.adjusted_drift_threshold(base);
        assert!(t_high < base, "high arousal should lower threshold: {:.4}", t_high);

        // High arousal is more aggressive than low arousal
        assert!(t_high < t_low);
    }

    #[test]
    fn arousal_zone_labels_are_correct() {
        let cases = [
            (0.0_f32, "Idle"),
            (0.20, "Idle"),
            (0.30, "Calm"),
            (0.50, "Optimal"),
            (0.70, "Stressed"),
            (0.85, "Crisis"),
        ];
        for (level, expected) in cases {
            let a = ArousalState { level, alpha: 0.15, samples: 1 };
            assert_eq!(a.zone(), expected, "level={level} → expected {expected}");
        }
    }
}
