//! Unified process effectiveness tracker — merges three independent signal sources
//! into one blended score per process.
//!
//! ## Problem
//!
//! Apollo previously maintained three independent learning loops:
//! - `OutcomeTracker`  — Bayesian weights (correlational, 30s resolution)
//! - `CausalGraph`     — causal confidence edges (Pearl-style, eval_delay cycles)
//! - `SkillRegistry`   — recipe success rates (action-level, immediate)
//!
//! Each loop updated independently, sometimes disagreeing by >40pp on the same
//! process. The coordinated freeze (and future action-gating) had no authoritative
//! single number to consult.
//!
//! ## Solution: credibility-weighted EMA blend (F3)
//!
//! ```text
//!                      Bayesian (Laplace)        Causal (EMA)          Skill (ratio)
//!                      ┌──────────────────┐   ┌───────────────┐   ┌──────────────────┐
//! per-process score =  │ credibility_b × b │ + │ credibility_c × c │ + │ credibility_s × s │
//!                      └──────────────────┘   └───────────────┘   └──────────────────┘
//!                      ─────────────────────────────────────────────────────────────────
//!                                    credibility_b + credibility_c + credibility_s
//! ```
//!
//! Credibility saturates at 1.0 as observation count grows (Beta posterior logic).
//! Missing sources contribute 0 credibility — effectively excluded from the blend.
//! Cold start (0 observations from all sources) → neutral score 0.5.
//!
//! ## Theoretical grounding
//!
//! This is a **Thompson Sampling analog with multi-source Beta posteriors**:
//! - Each source maintains an independent Beta-like posterior over "this process
//!   is effective", with credibility = min(obs / saturation, 1.0).
//! - The blend is the credibility-weighted mean of the three posteriors.
//! - Corresponds to multi-armed bandit feedback where each "arm" is a
//!   source type (Bayesian / causal / skill), weighted by evidence quality.
//! - When the causal source has ≥5 observations, Pearl's do-calculus intuition
//!   applies: causal evidence naturally dominates because causal credibility
//!   saturates faster (saturation=5 vs 20 for Bayesian).
//!
//! ## Apollo constraints
//!
//! - 8 GB M1, 300 ms cycles, processes appear/disappear.
//! - Processes with no data → 0.5 (neither target nor protect).
//! - GC removes entries below `min_observations` to keep memory bounded.
//!
//! ## References
//!
//! - Thompson (1933) "On the likelihood that one unknown probability exceeds another"
//! - Russo et al. (2018) "A Tutorial on Thompson Sampling" arXiv:1707.02038
//! - Pearl (2009) "Causality: Models, Reasoning and Inference", Ch. 3
//! - Auer et al. (2002) "Finite-time Analysis of the Multiarmed Bandit Problem"

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Per-process effectiveness record ─────────────────────────────────────────

/// Snapshot of per-source signals and the blended score for one process.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessEffectiveness {
    /// The winning blended score — the authoritative single number.
    ///
    /// Range [0, 1]. 0.5 = neutral/unknown. >0.6 = worth targeting. <0.4 = skip.
    pub blended_score: f64,

    /// Bayesian effectiveness from OutcomeTracker: (effective+1) / (total+2).
    /// Updated via `update_from_outcome()`. Neutral prior = 0.5.
    pub bayesian_eff: f64,
    /// Number of Bayesian observations (throttle_count).
    pub bayesian_obs: u32,

    /// Causal confidence from CausalGraph: EMA-updated confidence in action→pressure_drop.
    /// Updated via `update_from_causal()`. Absent until ≥3 causal observations.
    pub causal_confidence: f64,
    /// Number of causal evidence observations.
    pub causal_obs: u32,

    /// Skill success rate from SkillRegistry: success_count / apply_count.
    /// Updated via `update_from_skill()`. Absent when no matching skill exists.
    pub skill_rate: f64,
    /// Number of skill applications.
    pub skill_obs: u32,

    /// Total observations from all sources combined.
    pub observation_count: u32,

    /// Daemon cycle when this record was last touched.
    pub last_updated_cycle: u64,
}

impl Default for ProcessEffectiveness {
    fn default() -> Self {
        Self {
            blended_score: 0.5,
            bayesian_eff: 0.5,
            bayesian_obs: 0,
            causal_confidence: 0.5,
            causal_obs: 0,
            skill_rate: 0.5,
            skill_obs: 0,
            observation_count: 0,
            last_updated_cycle: 0,
        }
    }
}

impl ProcessEffectiveness {
    /// Recompute `blended_score` from current per-source values.
    ///
    /// ## Formula
    ///
    /// Each source contributes with credibility = min(obs / saturation, 1.0).
    /// Saturation constants are tuned so causal evidence (Pearl-style) dominates
    /// faster than correlational Bayesian evidence, which in turn dominates faster
    /// than infrequent skill observations.
    ///
    /// | Source   | Saturation | Rationale                                      |
    /// |----------|-----------|------------------------------------------------|
    /// | Causal   | 5         | ≥5 causal obs → near-full confidence (Pearl)   |
    /// | Bayesian | 20        | 20 obs for Laplace-smoothed Bayesian stability |
    /// | Skill    | 10        | skill.apply_count cap in SkillRegistry         |
    ///
    /// Cold start (all obs = 0): all credibilities = 0 → return 0.5 (neutral).
    fn recompute_blend(&mut self) {
        let cred_bayesian = (self.bayesian_obs as f64 / 20.0).min(1.0);
        let cred_causal = (self.causal_obs as f64 / 5.0).min(1.0);
        let cred_skill = (self.skill_obs as f64 / 10.0).min(1.0);

        let total_cred = cred_bayesian + cred_causal + cred_skill;
        if total_cred < 1e-9 {
            // Cold start: no data from any source.
            self.blended_score = 0.5;
            return;
        }

        let weighted_sum = cred_bayesian * self.bayesian_eff
            + cred_causal * self.causal_confidence
            + cred_skill * self.skill_rate;

        // Guard: clamp to [0, 1] to prevent NaN/inf from propagating.
        let score = (weighted_sum / total_cred).clamp(0.0, 1.0);
        // Final NaN guard (should be unreachable given clamp, but defensive).
        self.blended_score = if score.is_nan() { 0.5 } else { score };
    }
}

// ── EffectivenessTracker ──────────────────────────────────────────────────────

/// Unified per-process effectiveness tracker.
///
/// Holds one `ProcessEffectiveness` per process name. Each of the three
/// learning subsystems writes its signal via `update_from_*()`. Any reader
/// consults `blended_score()` — a single float that integrates all sources.
///
/// The tracker is intentionally cheap: no locks (single-threaded daemon), no
/// allocations on the hot path (HashMap updates are O(1) amortized).
pub struct EffectivenessTracker {
    scores: HashMap<String, ProcessEffectiveness>,
}

impl EffectivenessTracker {
    /// Create a new empty tracker. All unknown processes return 0.5.
    pub fn new() -> Self {
        Self {
            scores: HashMap::new(),
        }
    }

    // ── Update from OutcomeTracker ────────────────────────────────────────────

    /// Update the Bayesian signal for a process.
    ///
    /// `bayesian_eff` is `(effective_count + 1.0) / (throttle_count + 2.0)` —
    /// the Laplace-smoothed posterior from `PatternWeight::effectiveness()`.
    /// `obs_count` is `throttle_count` (number of times process was throttled).
    pub fn update_from_outcome(&mut self, name: &str, bayesian_eff: f64, obs_count: u32, cycle: u64) {
        let entry = self.scores.entry(name.to_string()).or_default();
        entry.bayesian_eff = bayesian_eff.clamp(0.0, 1.0);
        entry.bayesian_obs = obs_count;
        entry.last_updated_cycle = cycle;
        entry.observation_count = entry
            .bayesian_obs
            .saturating_add(entry.causal_obs)
            .saturating_add(entry.skill_obs);
        entry.recompute_blend();
    }

    // ── Update from CausalGraph ───────────────────────────────────────────────

    /// Update the causal signal for a process.
    ///
    /// `confidence` is `CausalEdge::confidence` for the `"throttle:<name>" → pressure_drop`
    /// edge. `evidence_count` is `CausalEdge::evidence_count`.
    ///
    /// Only call this when `evidence_count >= 3` (the CausalGraph's own gate);
    /// it is safe to call regardless — credibility naturally stays near 0 with
    /// fewer observations (cred = obs / 5.0 ≈ 0 for obs ≤ 1).
    pub fn update_from_causal(&mut self, name: &str, confidence: f64, evidence_count: u32, cycle: u64) {
        let entry = self.scores.entry(name.to_string()).or_default();
        entry.causal_confidence = confidence.clamp(0.0, 1.0);
        entry.causal_obs = evidence_count;
        entry.last_updated_cycle = cycle;
        entry.observation_count = entry
            .bayesian_obs
            .saturating_add(entry.causal_obs)
            .saturating_add(entry.skill_obs);
        entry.recompute_blend();
    }

    // ── Update from SkillRegistry ─────────────────────────────────────────────

    /// Update the skill signal for a process.
    ///
    /// `rate` is `OptimizationSkill::success_rate` (f32, cast to f64 here).
    /// `apply_count` is `OptimizationSkill::apply_count`.
    pub fn update_from_skill(&mut self, name: &str, rate: f64, apply_count: u32, cycle: u64) {
        let entry = self.scores.entry(name.to_string()).or_default();
        entry.skill_rate = rate.clamp(0.0, 1.0);
        entry.skill_obs = apply_count;
        entry.last_updated_cycle = cycle;
        entry.observation_count = entry
            .bayesian_obs
            .saturating_add(entry.causal_obs)
            .saturating_add(entry.skill_obs);
        entry.recompute_blend();
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /// The authoritative blended effectiveness score for a process.
    ///
    /// Returns 0.5 (neutral) when no data has been recorded for this process.
    /// Values above 0.6 indicate the process is a reliable pressure-reduction target.
    /// Values below 0.4 indicate throttling this process tends to be ineffective.
    pub fn blended_score(&self, name: &str) -> f64 {
        self.scores
            .get(name)
            .map(|e| e.blended_score)
            .unwrap_or(0.5)
    }

    /// Full effectiveness record for a process. Returns None if not tracked.
    pub fn get(&self, name: &str) -> Option<&ProcessEffectiveness> {
        self.scores.get(name)
    }

    /// Number of processes currently tracked.
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// True if no processes are tracked.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    // ── Garbage collection ────────────────────────────────────────────────────

    /// Remove entries with fewer than `min_observations` total observations
    /// and not updated within `max_stale_cycles` cycles of `current_cycle`.
    ///
    /// Prevents unbounded HashMap growth in long-running daemons. Safe to call
    /// every 500 cycles. Entries at the neutral prior (0.5) carry no signal.
    pub fn gc(&mut self, min_observations: u32, max_stale_cycles: u64, current_cycle: u64) {
        self.scores.retain(|_, e| {
            // Keep if recently updated (within the staleness window).
            let age = current_cycle.saturating_sub(e.last_updated_cycle);
            if age <= max_stale_cycles {
                return true;
            }
            // Keep if it has enough observations to carry real signal.
            e.observation_count >= min_observations
        });
    }

    /// Snapshot of all scores — for persistence in `LearnedState`.
    pub fn snapshot(&self) -> HashMap<String, ProcessEffectiveness> {
        self.scores.clone()
    }

    /// Restore from a persisted snapshot.
    pub fn restore_from_map(&mut self, map: HashMap<String, ProcessEffectiveness>) {
        self.scores = map;
        // Re-clamp all values on restore to guard against corrupt state.
        for entry in self.scores.values_mut() {
            entry.bayesian_eff = entry.bayesian_eff.clamp(0.0, 1.0);
            entry.causal_confidence = entry.causal_confidence.clamp(0.0, 1.0);
            entry.skill_rate = entry.skill_rate.clamp(0.0, 1.0);
            // Recompute blend from sanitized values.
            entry.recompute_blend();
        }
    }
}

// ── Default ───────────────────────────────────────────────────────────────────

impl Default for EffectivenessTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Cold-start behaviour ──────────────────────────────────────────────────

    /// An unregistered process must return neutral 0.5 — never panic, never NaN.
    #[test]
    fn test_cold_start_unknown_process() {
        let tracker = EffectivenessTracker::new();
        let score = tracker.blended_score("nonexistent_process");
        assert!(
            (score - 0.5).abs() < 1e-9,
            "unknown process should return neutral 0.5, got {}",
            score
        );
    }

    /// A process with zero observations from all sources must return 0.5.
    #[test]
    fn test_cold_start_zero_observations() {
        let mut tracker = EffectivenessTracker::new();
        // Update with 0 obs — should stay neutral.
        tracker.update_from_outcome("Safari", 0.5, 0, 1);
        let score = tracker.blended_score("Safari");
        assert!(
            (score - 0.5).abs() < 1e-9,
            "zero Bayesian obs should keep score at 0.5, got {}",
            score
        );
    }

    // ── Single-source updates ─────────────────────────────────────────────────

    /// Bayesian-only update: score should track the Bayesian signal when causal
    /// and skill credibilities are zero.
    #[test]
    fn test_single_source_bayesian_only() {
        let mut tracker = EffectivenessTracker::new();
        // 20 throttles, all effective → Bayesian near 1.0, credibility = 1.0.
        tracker.update_from_outcome("Firefox", 21.0 / 22.0, 20, 10);
        let score = tracker.blended_score("Firefox");
        assert!(
            score > 0.85,
            "high-effectiveness Bayesian-only should yield >0.85, got {}",
            score
        );
    }

    /// Causal-only update: 5 causal obs saturates credibility → score ≈ confidence.
    #[test]
    fn test_single_source_causal_only() {
        let mut tracker = EffectivenessTracker::new();
        tracker.update_from_causal("Dropbox", 0.90, 5, 20);
        let score = tracker.blended_score("Dropbox");
        // Causal credibility = 5/5 = 1.0, bayesian and skill credibilities = 0.
        // → score = 1.0 × 0.90 / 1.0 = 0.90
        assert!(
            (score - 0.90).abs() < 1e-6,
            "causal-only at full credibility should ≈ confidence, got {}",
            score
        );
    }

    /// Skill-only update: success_rate drives score, credibility saturates at apply_count=10.
    #[test]
    fn test_single_source_skill_only() {
        let mut tracker = EffectivenessTracker::new();
        tracker.update_from_skill("cloud_throttle", 0.80, 10, 30);
        let score = tracker.blended_score("cloud_throttle");
        // Skill credibility = 10/10 = 1.0 → score = 0.80.
        assert!(
            (score - 0.80).abs() < 1e-6,
            "skill-only at full credibility should ≈ rate, got {}",
            score
        );
    }

    // ── Convergence (multi-source) ────────────────────────────────────────────

    /// When causal evidence (≥5 obs) says effective and Bayesian says ineffective
    /// with fewer observations, causal should dominate.
    #[test]
    fn test_causal_dominates_weak_bayesian() {
        let mut tracker = EffectivenessTracker::new();
        // Causal: 5 obs → credibility 1.0, confidence 0.90 (very effective).
        tracker.update_from_causal("Safari", 0.90, 5, 50);
        // Bayesian: 2 obs → credibility 2/20 = 0.10, eff = 0.30 (seems ineffective).
        tracker.update_from_outcome("Safari", 0.30, 2, 50);

        let score = tracker.blended_score("Safari");
        // Weighted: (0.10 × 0.30 + 1.0 × 0.90) / (0.10 + 1.0) = (0.03 + 0.90) / 1.10 ≈ 0.845
        assert!(
            score > 0.70,
            "causal should dominate weak Bayesian, got {}",
            score
        );
    }

    /// After enough observations all three sources agree → score converges near consensus.
    #[test]
    fn test_convergence_all_three_sources() {
        let mut tracker = EffectivenessTracker::new();
        // All three sources say ~0.80 effective with full credibility.
        tracker.update_from_outcome("SomeApp", 21.0 / 22.0, 20, 100); // ≈ 0.955
        tracker.update_from_causal("SomeApp", 0.80, 5, 100);
        tracker.update_from_skill("SomeApp", 0.80, 10, 100);

        let score = tracker.blended_score("SomeApp");
        // All credibilities at max → simple mean of (0.955, 0.80, 0.80) ≈ 0.852
        assert!(
            score > 0.75,
            "three-source consensus at 0.80 should yield >0.75, got {}",
            score
        );
    }

    // ── NaN/infinity guards ───────────────────────────────────────────────────

    /// Extreme values must never produce NaN or values outside [0, 1].
    #[test]
    fn test_no_nan_on_extreme_inputs() {
        let mut tracker = EffectivenessTracker::new();
        // Feed extreme but valid-ish inputs.
        tracker.update_from_outcome("ProcA", 0.0, 1000, 1);
        tracker.update_from_causal("ProcA", 1.0, 1000, 1);
        tracker.update_from_skill("ProcA", 0.5, 1000, 1);

        let score = tracker.blended_score("ProcA");
        assert!(
            score.is_finite(),
            "score must be finite, got {}",
            score
        );
        assert!(
            (0.0..=1.0).contains(&score),
            "score must be in [0,1], got {}",
            score
        );
    }

    /// Restore from a map with out-of-range values must not panic and must clamp.
    #[test]
    fn test_restore_clamping() {
        let mut tracker = EffectivenessTracker::new();
        let mut corrupt = ProcessEffectiveness::default();
        corrupt.bayesian_eff = 2.5;   // out of range
        corrupt.causal_confidence = -0.3; // out of range
        corrupt.bayesian_obs = 20;
        corrupt.causal_obs = 5;

        let mut map = HashMap::new();
        map.insert("CorruptProc".to_string(), corrupt);
        tracker.restore_from_map(map);

        let score = tracker.blended_score("CorruptProc");
        assert!(
            score.is_finite() && (0.0..=1.0).contains(&score),
            "restored score must be clamped to [0,1], got {}",
            score
        );
    }

    // ── GC ────────────────────────────────────────────────────────────────────

    /// GC should remove stale entries with few observations.
    #[test]
    fn test_gc_removes_stale_entries() {
        let mut tracker = EffectivenessTracker::new();
        // Entry with 1 obs, last updated at cycle 1.
        tracker.update_from_outcome("StaleProc", 0.5, 1, 1);
        assert_eq!(tracker.len(), 1);

        // GC at cycle 1000 with stale window = 500 and min_obs = 5.
        // Entry: age = 999 > 500 (stale), obs = 1 < 5 (below min) → remove.
        tracker.gc(5, 500, 1000);
        assert_eq!(
            tracker.len(),
            0,
            "stale low-observation entry should be removed"
        );
    }

    /// GC should keep entries with sufficient observations even if stale.
    #[test]
    fn test_gc_keeps_well_observed_stale_entries() {
        let mut tracker = EffectivenessTracker::new();
        tracker.update_from_outcome("SolidProc", 0.85, 25, 1);

        // GC at cycle 2000 with stale window = 500, but min_obs = 5.
        // Entry: age = 1999 > 500 (stale), BUT obs = 25 >= 5 → keep.
        tracker.gc(5, 500, 2000);
        assert_eq!(
            tracker.len(),
            1,
            "well-observed entry kept even when stale"
        );
    }

    /// GC should keep recently-updated entries regardless of observation count.
    #[test]
    fn test_gc_keeps_recently_updated() {
        let mut tracker = EffectivenessTracker::new();
        tracker.update_from_outcome("FreshProc", 0.5, 1, 990);

        // GC at cycle 1000, window = 500. Age = 10 ≤ 500 → keep (recently updated).
        tracker.gc(5, 500, 1000);
        assert_eq!(
            tracker.len(),
            1,
            "recently-updated entry should be kept regardless of observation count"
        );
    }
}
