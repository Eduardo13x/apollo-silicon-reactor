//! Learning Pipeline — unified mini-batch coordinator for the three learning subsystems.
//!
//! Apollo has three independent learning systems that each track action effectiveness:
//!   - `OutcomeTracker` — Bayesian weights per process, experience memory, co-occurrence
//!   - `CausalGraph`    — causal edges with confidence EMA (action → pressure_drop)
//!   - `SkillRegistry`  — success rates per skill (throttle recipe)
//!
//! Previously these systems diverged: a process might score 0.85 in OutcomeTracker but
//! only 0.40 in CausalGraph because each system observed different subsets of events.
//!
//! This module provides:
//!   1. A `Learner` trait that all subsystems implement (observe / observe_batch / effectiveness / gc).
//!   2. `LearningObservation` — a single normalized event shared across all subsystems.
//!   3. `LearningPipeline` — mini-batch accumulator (default batch_size=8) that:
//!      a. Fans out each observation to all three learners coherently.
//!      b. Applies cross-feed rules when the batch is flushed.
//!
//! ## Cross-feed rules (applied at flush time)
//!
//! **OutcomeTracker → SkillRegistry**: when a process has Bayesian effectiveness > 0.7
//! (≥3 throttles), boost the corresponding skill's success_count by one to seed the
//! skill's empirical rate toward the causal evidence.
//!
//! **CausalGraph → SkillRegistry**: solid causal edges (confidence > 0.7, ≥5 evidence)
//! for which the matching skill has success_rate < 0.5 receive one artificial success
//! to correct trials that saw anomalous failures.
//!
//! **SkillRegistry → OutcomeTracker (prior)**: if a skill has success_rate > 0.8 with
//! ≥20 applications, its strong empirical signal is used to seed the Bayesian
//! effective_count so new OutcomeTracker entries benefit from skill experience.
//!
//! ## Feature flag
//!
//! Set `LearningPipeline::enabled = false` (or construct with `disabled()`) to fall back
//! to the existing per-system update paths. This allows A/B comparison in production.
//!
//! ## Mini-batch rationale
//!
//! A batch size of 8–16 observations reduces per-event overhead:
//! - Better cache locality: sort by `process_name` before updating weights so the
//!   same `HashMap` bucket is hit multiple times in sequence.
//! - Lower variance: cross-feed boosts are computed over the whole batch, not
//!   per observation, so a single noisy event can't spike a skill's confidence.
//! - Structurally identical to batch_size=1 for correctness; if batch_size=1
//!   gives the same results, switch via `with_batch_size(1)`.

use crate::engine::causal_graph::CausalGraph;
use crate::engine::effectiveness_tracker::EffectivenessTracker;
use crate::engine::optimization_skills::SkillRegistry;
use crate::engine::outcome_tracker::OutcomeTracker;

// ── Learner trait ─────────────────────────────────────────────────────────────

/// Unified learning interface implemented by all subsystems.
///
/// Generic over `Observation` so each subsystem can use the shared
/// `LearningObservation` or define its own if needed.
pub trait Learner {
    type Observation;

    /// Process a single observation.
    fn observe(&mut self, obs: &Self::Observation);

    /// Process a batch (default: iterate and call observe).
    /// Override for subsystems that benefit from batch-level logic.
    fn observe_batch(&mut self, batch: &[Self::Observation]) {
        for obs in batch {
            self.observe(obs);
        }
    }

    /// Query effectiveness [0,1] for a given key (process name or action key).
    /// Returns `None` when insufficient data exists.
    fn effectiveness(&self, key: &str) -> Option<f64>;

    /// Prune stale entries. Called after each flush.
    fn gc(&mut self);
}

// ── LearningObservation ───────────────────────────────────────────────────────

/// A single learning event shared across all three subsystems.
///
/// Produced once per pressure-reduction action, consumed coherently by all learners.
#[derive(Clone, Debug)]
pub struct LearningObservation {
    pub process_name: String,
    pub skill_name: Option<String>,
    pub pre_pressure: f64,
    pub post_pressure: f64,
    pub workload: String,
    pub cycle: u64,
    pub action_type: ActionKind,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum ActionKind {
    Throttle,
    Freeze,
    Memorystatus,
}

impl LearningObservation {
    pub fn delta(&self) -> f64 {
        self.pre_pressure - self.post_pressure
    }

    pub fn effective(&self) -> bool {
        self.delta() >= 0.01
    }

    pub fn causal_action_key(&self) -> String {
        let prefix = match self.action_type {
            ActionKind::Throttle => "throttle",
            ActionKind::Freeze => "freeze",
            ActionKind::Memorystatus => "memorystatus",
        };
        format!("{}:{}", prefix, self.process_name)
    }

    pub fn registry_key(&self) -> String {
        self.causal_action_key()
    }
}

// ── LearningPipeline ──────────────────────────────────────────────────────────

/// Mini-batch coordinator for the three learning subsystems.
///
/// Accumulates observations until `batch_size` is reached, then flushes them
/// to all subsystems coherently, applying cross-feed rules at flush time.
pub struct LearningPipeline {
    /// Accumulated observations awaiting flush.
    batch: Vec<LearningObservation>,
    /// Number of observations to accumulate before auto-flushing. Default: 8.
    batch_size: usize,
    /// Feature flag: when false, `push()` is a no-op (fall back to legacy paths).
    pub enabled: bool,
}

impl LearningPipeline {
    /// Create a new pipeline with the default batch size (8).
    pub fn new() -> Self {
        Self {
            batch: Vec::with_capacity(8),
            batch_size: 8,
            enabled: true,
        }
    }

    /// Create a disabled pipeline (all pushes are no-ops).
    /// Use for A/B testing: instantiate this in the daemon, check `enabled` before push.
    pub fn disabled() -> Self {
        Self {
            batch: Vec::new(),
            batch_size: 8,
            enabled: false,
        }
    }

    /// Override batch size. `1` gives single-observation semantics with the same
    /// cross-feed logic — useful to verify correctness matches batch mode.
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    /// Push one observation. Auto-flushes when batch reaches `batch_size`.
    ///
    /// `outcome_tracker`, `causal_graph`, and `skill_registry` are passed by
    /// mutable reference so callers keep ownership (no Rc/RefCell needed).
    pub fn push(
        &mut self,
        obs: LearningObservation,
        outcome_tracker: &mut OutcomeTracker,
        causal_graph: &mut CausalGraph,
        skill_registry: &mut SkillRegistry,
        effectiveness_tracker: &mut EffectivenessTracker,
    ) {
        if !self.enabled {
            return;
        }
        self.batch.push(obs);
        if self.batch.len() >= self.batch_size {
            self.flush(
                outcome_tracker,
                causal_graph,
                skill_registry,
                effectiveness_tracker,
            );
        }
    }

    /// Flush pending observations to all subsystems.
    ///
    /// Order of operations:
    ///   1. Sort batch by `process_name` for cache locality (same HashMap bucket).
    ///   2. Fan out to each subsystem.
    ///   3. Apply cross-feed rules.
    ///   4. Clear the batch.
    pub fn flush(
        &mut self,
        outcome_tracker: &mut OutcomeTracker,
        causal_graph: &mut CausalGraph,
        skill_registry: &mut SkillRegistry,
        effectiveness_tracker: &mut EffectivenessTracker,
    ) {
        if self.batch.is_empty() {
            return;
        }

        // Sort by process_name for cache locality when updating HashMaps.
        self.batch
            .sort_unstable_by(|a, b| a.process_name.cmp(&b.process_name));

        // ── Step 1: Fan-out to each subsystem ────────────────────────────────

        for obs in &self.batch {
            // OutcomeTracker: record each action+outcome directly (post-evaluation path).
            // Note: record_throttle() + tick() is the live path; here we update the
            // Bayesian weight directly for observations that already have post_pressure.
            {
                let w = outcome_tracker
                    .weights
                    .entry(obs.process_name.clone())
                    .or_default();
                w.throttle_count += 1;
                if obs.effective() {
                    w.effective_count += 1;
                }
            }

            // CausalGraph: record + evaluate in the same cycle using pre/post pressure.
            // We use a synthetic cycle so evaluate() triggers immediately.
            let synthetic_cycle = obs.cycle;
            causal_graph.record_action(
                &obs.causal_action_key(),
                obs.pre_pressure as f32,
                synthetic_cycle,
            );
            // Evaluate 0 cycles later with post_pressure (we already have the outcome).
            // This bypasses the normal eval_delay but is correct since we supply the
            // resolved outcome directly. We call evaluate() with a cycle offset of
            // eval_delay so the pending entry is eligible.
            causal_graph.evaluate(
                obs.post_pressure as f32,
                synthetic_cycle + causal_graph_eval_delay(),
            );

            // SkillRegistry: record result for the named skill, if any.
            if let Some(skill_name) = &obs.skill_name {
                skill_registry.record_result_with_pressure(
                    skill_name,
                    obs.effective(),
                    obs.pre_pressure as f32,
                );
            }
        }

        // ── Step 2: Cross-feed rules ──────────────────────────────────────────

        // Cross-feed A: OutcomeTracker → SkillRegistry
        //
        // When a process has strong Bayesian evidence (effectiveness > 0.7, ≥3 throttles),
        // boost the corresponding skill's success_count by one.
        //
        // This seeds the skill's empirical rate toward the causal evidence, so skills
        // that have seen few trials but strong process-level evidence converge faster.
        for obs in &self.batch {
            if let Some(w) = outcome_tracker.weights.get(&obs.process_name) {
                if w.throttle_count >= 3 && w.effectiveness() > 0.7 {
                    let skill_key = obs.registry_key();
                    let evidence_rate = w.effectiveness();
                    // Boost if skill rate lags evidence by ≥0.15 with ≥3 applications.
                    skill_registry.cross_feed_boost(&skill_key, evidence_rate, 0.15, 3);
                }
            }
        }

        // Cross-feed B: CausalGraph → SkillRegistry
        //
        // Solid causal edges (confidence > 0.7, evidence ≥ 5) whose matching skill
        // has success_rate < 0.5 receive one artificial success.
        //
        // Rationale: trial data can be noisy early on; if the causal graph has strong
        // evidence from a *different* measurement path, trust it over sparse trials.
        for edge in causal_graph.solid_edges() {
            if !edge.cause.starts_with("throttle:") {
                continue;
            }
            let current_rate = skill_registry.success_rate(&edge.cause).unwrap_or_else(|| {
                eprintln!(
                    "[learning_pipeline] cross-feed B: no skill rate for {:?}, defaulting to 1.0",
                    edge.cause
                );
                1.0
            });
            if current_rate < 0.5 {
                // Boost: lift rate toward causal evidence. min_apply_count=3, min_gap=0.
                skill_registry.cross_feed_boost(
                    &edge.cause,
                    edge.confidence as f64,
                    0.0, // any positive gap triggers a boost when rate < 0.5
                    3,
                );
            }
        }

        // Cross-feed C: SkillRegistry → OutcomeTracker (prior seeding)
        //
        // When a skill has strong empirical evidence (success_rate > 0.8, ≥20 applications),
        // seed the corresponding OutcomeTracker Bayesian weight so new daemon restarts
        // benefit from crystallised skill knowledge without waiting 20+ throttle cycles.
        for obs in &self.batch {
            let skill_key = obs.registry_key();
            let skill_apply_count = skill_registry.apply_count(&skill_key).unwrap_or_else(|| {
                eprintln!(
                    "[learning_pipeline] cross-feed C: no apply_count for {:?}, defaulting to 0",
                    skill_key
                );
                0
            });
            let skill_rate = skill_registry.success_rate(&skill_key).unwrap_or_else(|| {
                eprintln!(
                    "[learning_pipeline] cross-feed C: no success_rate for {:?}, defaulting to 0.0",
                    skill_key
                );
                0.0
            });
            if skill_apply_count >= 20 && skill_rate > 0.8 {
                let w = outcome_tracker
                    .weights
                    .entry(obs.process_name.clone())
                    .or_default();
                // Only seed if OutcomeTracker has less evidence than the skill.
                // Use one synthetic throttle+effective pair to shift the prior.
                if w.throttle_count < skill_apply_count / 2 {
                    w.throttle_count = w.throttle_count.saturating_add(1);
                    w.effective_count = w.effective_count.saturating_add(1);
                }
            }
        }

        // ── Step 3: Update EffectivenessTracker (F3 Blend) ───────────────────

        // After all learners have updated, feed their new signals into the
        // EffectivenessTracker to recompute the blended scores for all
        // processes touched in this batch.
        for obs in &self.batch {
            let cycle = obs.cycle;

            // 1. Bayesian signal
            if let Some(w) = outcome_tracker.weights.get(&obs.process_name) {
                effectiveness_tracker.update_from_outcome(
                    &obs.process_name,
                    w.effectiveness(),
                    w.throttle_count,
                    cycle,
                );
            }

            // 2. Causal signal
            let causal_key = obs.causal_action_key();
            if let Some(edge) = causal_graph.get_edge(&causal_key, "pressure_drop") {
                effectiveness_tracker.update_from_causal(
                    &obs.process_name,
                    edge.confidence as f64,
                    edge.evidence_count,
                    cycle,
                );
            }

            // 3. Skill signal
            let skill_key = obs.skill_name.clone().unwrap_or_else(|| obs.registry_key());
            if let Some(rate) = skill_registry.success_rate(&skill_key) {
                let apps = skill_registry.apply_count(&skill_key).unwrap_or_else(|| {
                    eprintln!(
                        "[learning_pipeline] F3 blend: no apply_count for {:?}, defaulting to 0",
                        skill_key
                    );
                    0
                });
                effectiveness_tracker.update_from_skill(
                    &obs.process_name,
                    rate as f64,
                    apps,
                    cycle,
                );
            }
        }

        // ── Step 4: Clear batch ───────────────────────────────────────────────
        self.batch.clear();
    }

    /// Number of observations currently buffered (not yet flushed).
    pub fn pending_count(&self) -> usize {
        self.batch.len()
    }

    /// Force-flush any remaining observations (call at daemon shutdown or persist time).
    pub fn flush_remaining(
        &mut self,
        outcome_tracker: &mut OutcomeTracker,
        causal_graph: &mut CausalGraph,
        skill_registry: &mut SkillRegistry,
        effectiveness_tracker: &mut EffectivenessTracker,
    ) {
        self.flush(
            outcome_tracker,
            causal_graph,
            skill_registry,
            effectiveness_tracker,
        );
    }
}

impl Default for LearningPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// The CausalGraph eval_delay value (3 cycles). Exposed as a const so the
/// pipeline can synthesize the correct cycle offset for immediate evaluation.
/// This matches `CausalGraph::new()` where `eval_delay = 3`.
const fn causal_graph_eval_delay() -> u64 {
    3
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_obs(process: &str, pre: f64, post: f64, cycle: u64) -> LearningObservation {
        make_obs_with_kind(process, pre, post, cycle, ActionKind::Throttle)
    }

    fn make_obs_with_kind(
        process: &str,
        pre: f64,
        post: f64,
        cycle: u64,
        action_type: ActionKind,
    ) -> LearningObservation {
        LearningObservation {
            process_name: process.to_string(),
            skill_name: None,
            pre_pressure: pre,
            post_pressure: post,
            workload: "any".to_string(),
            cycle,
            action_type,
        }
    }

    fn make_obs_with_skill(
        process: &str,
        skill: &str,
        pre: f64,
        post: f64,
        cycle: u64,
    ) -> LearningObservation {
        LearningObservation {
            process_name: process.to_string(),
            skill_name: Some(skill.to_string()),
            pre_pressure: pre,
            post_pressure: post,
            workload: "any".to_string(),
            cycle,
            action_type: ActionKind::Throttle,
        }
    }

    #[test]
    fn test_observation_delta_and_effective() {
        let obs = make_obs("Dropbox", 0.75, 0.70, 1);
        assert!((obs.delta() - 0.05).abs() < 1e-9);
        assert!(obs.effective());

        let obs2 = make_obs("Dropbox", 0.75, 0.75, 1);
        assert!(!obs2.effective()); // no change

        let obs3 = make_obs("Dropbox", 0.75, 0.745, 1);
        assert!(!obs3.effective()); // delta < 0.01
    }

    #[test]
    fn test_causal_action_key() {
        let obs = make_obs("Safari", 0.80, 0.70, 1);
        assert_eq!(obs.causal_action_key(), "throttle:Safari");

        let freeze_obs = make_obs_with_kind("Safari", 0.80, 0.70, 1, ActionKind::Freeze);
        assert_eq!(freeze_obs.causal_action_key(), "freeze:Safari");

        let mem_obs = make_obs_with_kind("pid:30", 0.80, 0.70, 1, ActionKind::Memorystatus);
        assert_eq!(mem_obs.causal_action_key(), "memorystatus:pid:30");
    }

    #[test]
    fn non_throttle_observation_updates_matching_causal_edge_only() {
        let mut pipeline = LearningPipeline::new().with_batch_size(1);
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        let obs = make_obs_with_kind("Safari", 0.80, 0.72, 0, ActionKind::Freeze);
        pipeline.push(obs, &mut ot, &mut cg, &mut sr, &mut eff);

        assert!(cg.get_edge("freeze:Safari", "pressure_drop").is_some());
        assert!(cg.get_edge("throttle:Safari", "pressure_drop").is_none());
    }

    #[test]
    fn test_pipeline_disabled_noop() {
        let mut pipeline = LearningPipeline::disabled();
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        let obs = make_obs("Dropbox", 0.75, 0.70, 1);
        pipeline.push(obs, &mut ot, &mut cg, &mut sr, &mut eff);

        assert_eq!(pipeline.pending_count(), 0);
        assert!(
            ot.weights.is_empty(),
            "disabled pipeline should not update OutcomeTracker"
        );
    }

    #[test]
    fn test_pipeline_accumulates_and_flushes() {
        let mut pipeline = LearningPipeline::new().with_batch_size(4);
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        // Push 3 observations — should not flush yet.
        for i in 0..3u64 {
            let obs = make_obs("Dropbox", 0.75, 0.70, i * 4);
            pipeline.push(obs, &mut ot, &mut cg, &mut sr, &mut eff);
        }
        assert_eq!(pipeline.pending_count(), 3);
        // OutcomeTracker not yet updated.
        assert!(ot.weights.is_empty());

        // Push 4th — triggers auto-flush.
        let obs = make_obs("Dropbox", 0.75, 0.70, 12);
        pipeline.push(obs, &mut ot, &mut cg, &mut sr, &mut eff);

        assert_eq!(pipeline.pending_count(), 0);
        let w = ot
            .weights
            .get("Dropbox")
            .expect("weight should exist after flush");
        assert_eq!(w.throttle_count, 4);
        assert_eq!(w.effective_count, 4); // all effective (delta=0.05 ≥ 0.01)
    }

    #[test]
    fn test_pipeline_effective_updates_outcome_tracker() {
        let mut pipeline = LearningPipeline::new().with_batch_size(1);
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        let obs = make_obs("Dropbox", 0.80, 0.74, 0); // delta=0.06, effective
        pipeline.push(obs, &mut ot, &mut cg, &mut sr, &mut eff);

        let w = ot.weights.get("Dropbox").unwrap();
        assert_eq!(w.throttle_count, 1);
        assert_eq!(w.effective_count, 1);
    }

    #[test]
    fn test_pipeline_ineffective_no_effective_count() {
        let mut pipeline = LearningPipeline::new().with_batch_size(1);
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        let obs = make_obs("contactsd", 0.70, 0.70, 0); // delta=0, not effective
        pipeline.push(obs, &mut ot, &mut cg, &mut sr, &mut eff);

        let w = ot.weights.get("contactsd").unwrap();
        assert_eq!(w.throttle_count, 1);
        assert_eq!(w.effective_count, 0);
    }

    #[test]
    fn test_pipeline_skill_result_recorded() {
        let mut pipeline = LearningPipeline::new().with_batch_size(1);
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        // Register a skill first.
        sr.learn("cloud_throttle", 0.70, "any", vec!["Dropbox".into()]);

        let obs = make_obs_with_skill("Dropbox", "cloud_throttle", 0.75, 0.68, 0);
        pipeline.push(obs, &mut ot, &mut cg, &mut sr, &mut eff);

        assert_eq!(sr.apply_count("cloud_throttle"), Some(1));
        // delta=0.07 ≥ 0.01 → effective, so success_count should be 1.
        // success_rate = success_count / apply_count = 1.0.
        assert_eq!(sr.success_rate("cloud_throttle"), Some(1.0));
    }

    #[test]
    fn test_crossfeed_outcome_to_skill() {
        // Set up: OutcomeTracker has strong Bayesian evidence for "Safari".
        // SkillRegistry has a matching skill with low success_rate.
        // After flush, the skill rate should be boosted toward evidence.

        let mut pipeline = LearningPipeline::new().with_batch_size(1);
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        // Seed OutcomeTracker with 5 effective throttles (effectiveness ≈ 0.86).
        {
            let w = ot.weights.entry("Safari".to_string()).or_default();
            w.throttle_count = 5;
            w.effective_count = 5;
        }

        // Add a skill with very low rate (1 success out of 5) via learn + record.
        sr.learn("throttle:Safari", 0.65, "any", vec!["Safari".to_string()]);
        // Force the skill to have 5 apps and 1 success to simulate low trial rate.
        for _ in 0..4 {
            sr.record_result("throttle:Safari", false);
        }
        sr.record_result("throttle:Safari", true);
        // Verify setup: success_rate ≈ 0.2 (1/5).
        let rate_before = sr.success_rate("throttle:Safari").unwrap();
        assert!(
            rate_before < 0.3,
            "setup: initial rate should be low (got {})",
            rate_before
        );

        // Push one observation for Safari — triggers cross-feed at flush.
        let obs = make_obs("Safari", 0.75, 0.70, 0);
        pipeline.push(obs, &mut ot, &mut cg, &mut sr, &mut eff);

        // The skill should have been boosted (success_rate increased).
        let rate_after = sr.success_rate("throttle:Safari").unwrap();
        assert!(
            rate_after > rate_before,
            "cross-feed should boost skill success_rate (was {}, now {})",
            rate_before,
            rate_after
        );
    }

    #[test]
    fn test_crossfeed_skill_to_outcome_tracker_prior() {
        // When skill has strong evidence (>0.8, ≥20 apps), seed OutcomeTracker.

        let mut pipeline = LearningPipeline::new().with_batch_size(1);
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        // Create a skill with strong evidence via learn + record.
        sr.learn("throttle:Dropbox", 0.65, "any", vec!["Dropbox".to_string()]);
        for _ in 0..17 {
            sr.record_result("throttle:Dropbox", true);
        }
        for _ in 0..3 {
            sr.record_result("throttle:Dropbox", false);
        }
        // 17 successes / 20 = 0.85 success_rate.
        let rate = sr.success_rate("throttle:Dropbox").unwrap();
        assert!(
            rate > 0.8,
            "setup: skill should have >0.8 rate (got {})",
            rate
        );

        // OutcomeTracker has no data for Dropbox yet.
        assert!(ot.weights.get("Dropbox").is_none());

        let obs = make_obs("Dropbox", 0.75, 0.68, 0);
        pipeline.push(obs, &mut ot, &mut cg, &mut sr, &mut eff);

        // After flush, OutcomeTracker should have been seeded by the skill.
        let w = ot
            .weights
            .get("Dropbox")
            .expect("weight should exist after prior seeding");
        // throttle_count comes from the fan-out (1) + possibly a seed (1 more).
        // effective_count should be ≥1 (effective obs) + seed boost.
        assert!(w.throttle_count >= 1);
        assert!(w.effective_count >= 1);
    }

    #[test]
    fn test_flush_remaining_clears_partial_batch() {
        let mut pipeline = LearningPipeline::new().with_batch_size(10);
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        // Push fewer than batch_size observations.
        for i in 0..3u64 {
            let obs = make_obs("Dropbox", 0.75, 0.70, i * 4);
            pipeline.push(obs, &mut ot, &mut cg, &mut sr, &mut eff);
        }
        assert_eq!(pipeline.pending_count(), 3);

        // Explicit flush_remaining should process them.
        pipeline.flush_remaining(&mut ot, &mut cg, &mut sr, &mut eff);
        assert_eq!(pipeline.pending_count(), 0);

        let w = ot
            .weights
            .get("Dropbox")
            .expect("weight after flush_remaining");
        assert_eq!(w.throttle_count, 3);
    }

    #[test]
    fn test_causal_graph_receives_observations() {
        let mut pipeline = LearningPipeline::new().with_batch_size(1);
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        // Run enough observations to build causal evidence.
        for i in 0..10u64 {
            let obs = make_obs("Firefox", 0.80, 0.74, i * 4);
            pipeline.push(obs, &mut ot, &mut cg, &mut sr, &mut eff);
        }

        // CausalGraph should have evidence for throttle:Firefox.
        let eff_score = cg.effectiveness("throttle:Firefox");
        assert!(
            eff_score.is_some(),
            "causal graph should have evidence for Firefox"
        );
        assert!(
            eff_score.unwrap() > 0.5,
            "effectiveness should be > 0.5 for consistently effective actions"
        );
    }

    #[test]
    fn test_batch_size_one_semantics_match_larger_batch() {
        // Verify correctness invariant: batch_size=1 and batch_size=8 produce
        // the same OutcomeTracker weights for the same observations.

        let observations: Vec<LearningObservation> = (0..8u64)
            .map(|i| make_obs("Dropbox", 0.75, if i % 2 == 0 { 0.70 } else { 0.75 }, i * 4))
            .collect();

        // Run with batch_size=1.
        let mut ot1 = OutcomeTracker::new();
        {
            let mut cg = CausalGraph::new();
            let mut sr = SkillRegistry::new();
            let mut eff = EffectivenessTracker::new();
            let mut pipeline = LearningPipeline::new().with_batch_size(1);
            for obs in &observations {
                pipeline.push(obs.clone(), &mut ot1, &mut cg, &mut sr, &mut eff);
            }
        }

        // Run with batch_size=8.
        let mut ot8 = OutcomeTracker::new();
        {
            let mut cg = CausalGraph::new();
            let mut sr = SkillRegistry::new();
            let mut eff = EffectivenessTracker::new();
            let mut pipeline = LearningPipeline::new().with_batch_size(8);
            for obs in &observations {
                pipeline.push(obs.clone(), &mut ot8, &mut cg, &mut sr, &mut eff);
            }
        }

        // Core weights should match.
        let w1 = ot1.weights.get("Dropbox").unwrap();
        let w8 = ot8.weights.get("Dropbox").unwrap();
        assert_eq!(
            w1.throttle_count, w8.throttle_count,
            "throttle_count should match"
        );
        assert_eq!(
            w1.effective_count, w8.effective_count,
            "effective_count should match"
        );
    }

    #[test]
    fn test_pending_count_starts_at_zero() {
        let pipeline = LearningPipeline::new();
        assert_eq!(pipeline.pending_count(), 0);
    }

    #[test]
    fn test_flush_empty_batch_is_noop() {
        let mut pipeline = LearningPipeline::new();
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        // flush with nothing buffered should not panic or mutate state
        pipeline.flush(&mut ot, &mut cg, &mut sr, &mut eff);
        assert!(ot.weights.is_empty());
        assert_eq!(pipeline.pending_count(), 0);
    }

    #[test]
    fn test_with_batch_size_zero_clamped_to_one() {
        // with_batch_size(0) must clamp to 1 so push() always flushes immediately.
        let mut pipeline = LearningPipeline::new().with_batch_size(0);
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        let obs = make_obs("Dropbox", 0.75, 0.70, 0);
        pipeline.push(obs, &mut ot, &mut cg, &mut sr, &mut eff);

        // Batch flushed immediately (batch_size clamped to 1).
        assert_eq!(pipeline.pending_count(), 0);
        assert!(ot.weights.contains_key("Dropbox"));
    }

    #[test]
    fn test_default_pipeline_is_enabled() {
        let pipeline = LearningPipeline::default();
        assert!(pipeline.enabled, "default pipeline should be enabled");
    }

    #[test]
    fn test_delta_negative_not_effective() {
        // Pressure went UP — bad action, delta < 0.
        let obs = make_obs("Slack", 0.60, 0.65, 0);
        assert!(obs.delta() < 0.0);
        assert!(!obs.effective());
    }

    #[test]
    fn test_effective_threshold_exact_boundary() {
        // delta = exactly 0.01 → effective (≥ 0.01 is true).
        let obs_at = make_obs("Arc", 0.70, 0.69, 0);
        assert!((obs_at.delta() - 0.01).abs() < 1e-9);
        assert!(obs_at.effective());

        // delta = 0.009 → not effective.
        let obs_below = make_obs("Arc", 0.70, 0.691, 0);
        assert!(!obs_below.effective());
    }

    #[test]
    fn test_multiple_processes_tracked_independently() {
        let mut pipeline = LearningPipeline::new().with_batch_size(2);
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        // One effective, one not.
        let obs_a = make_obs("Dropbox", 0.75, 0.70, 0); // effective
        let obs_b = make_obs("Slack", 0.60, 0.61, 1); // not effective (delta=-0.01)

        pipeline.push(obs_a, &mut ot, &mut cg, &mut sr, &mut eff);
        pipeline.push(obs_b, &mut ot, &mut cg, &mut sr, &mut eff);

        let w_dropbox = ot.weights.get("Dropbox").unwrap();
        assert_eq!(w_dropbox.throttle_count, 1);
        assert_eq!(w_dropbox.effective_count, 1);

        let w_slack = ot.weights.get("Slack").unwrap();
        assert_eq!(w_slack.throttle_count, 1);
        assert_eq!(w_slack.effective_count, 0);
    }

    #[test]
    fn test_flush_remaining_on_empty_is_noop() {
        let mut pipeline = LearningPipeline::new().with_batch_size(10);
        let mut ot = OutcomeTracker::new();
        let mut cg = CausalGraph::new();
        let mut sr = SkillRegistry::new();
        let mut eff = EffectivenessTracker::new();

        // Nothing pushed — flush_remaining should be silent.
        pipeline.flush_remaining(&mut ot, &mut cg, &mut sr, &mut eff);
        assert_eq!(pipeline.pending_count(), 0);
        assert!(ot.weights.is_empty());
    }
}
