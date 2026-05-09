//! # LearningContext
//!
//! Groups all mutable learning-subsystem references that are needed on *every*
//! decision cycle.  This is the "spine" that connects the decision stage to the
//! observation stage, and that unblocks the extraction of both
//! `observation_stage` and `decision_stage` from the main daemon loop.
//!
//! ## Why this grouping?
//!
//! The daemon main loop holds nine independent `mut` local variables that every
//! hot-path stage needs:
//!
//! | Variable               | Type                       | Used in              |
//! |------------------------|----------------------------|----------------------|
//! | `outcome_tracker`      | `OutcomeTracker`           | decision + observe   |
//! | `signal_intel`         | `SignalIntelligence`       | decision + observe   |
//! | `predictive_agent`     | `PredictiveAgent`          | decision + observe   |
//! | `specialist_accuracy`  | `SpecialistAccuracyTracker`| decision             |
//! | `overflow_guard`       | `OverflowGuard`            | decision + observe   |
//! | `causal_graph`         | `CausalGraph`              | decision + observe   |
//! | `skill_registry`       | `SkillRegistry`            | decision + observe   |
//! | `neuromod`             | `ApolloNeuromodulator`     | decision             |
//! | `energy_tracker`       | `EnergyTracker`            | decision + observe   |
//!
//! Passing all nine as separate `&mut` parameters to extracted functions would
//! produce signatures with 9+ mutably-borrowed parameters before any
//! stage-specific inputs are added.  `LearningContext` collapses them into one
//! grouped reference.
//!
//! ## What does NOT belong here
//!
//! Variables that appear only in `if cycle_count % N` blocks belong in
//! `PeriodicContext` (see `periodic_stage.rs`), not here.  This includes:
//! `holt_winters`, `focus_markov`, `cache_warmer`, `io_shaper`,
//! `temporal_predictor`.
//!
//! ## Borrow-checker note
//!
//! All fields are `&'a mut` references, so `LearningContext<'a>` holds exactly
//! one lifetime.  The struct does not own any data — it is a temporary lens
//! over locals that live in `run_daemon`.  Rust's borrow checker allows this
//! because each field borrows a *different* local variable; there are no
//! aliasing concerns.
//!
//! If split-borrow issues arise in practice (e.g., a helper needing two fields
//! independently), the caller can destructure the struct:
//!
//! ```rust,ignore
//! let LearningContext { outcome_tracker, causal_graph, .. } = &mut ctx;
//! some_fn(outcome_tracker, causal_graph);
//! ```
//!
//! Rust performs split-borrow on struct fields automatically through the
//! destructuring syntax.

use crate::engine::causal_graph::CausalGraph;
use crate::engine::energy::EnergyTracker;
use crate::engine::neuromodulator::ApolloNeuromodulator;
use crate::engine::optimization_skills::SkillRegistry;
use crate::engine::outcome_tracker::OutcomeTracker;
use crate::engine::overflow_guard::OverflowGuard;
use crate::engine::predictive_agent::{PredictiveAgent, SpecialistAccuracyTracker};
use crate::engine::signal_intelligence::SignalIntelligence;

/// Mutable references to every learning subsystem consulted on each decision cycle.
///
/// Construct with [`LearningContext::new`] at the top of the main loop body,
/// then pass `&mut ctx` to `decision_stage::decide(…)` and
/// `observation_stage::observe(…)`.
///
/// The struct is intentionally **not** `Clone` or `Copy` — it holds exclusive
/// mutable references and must not be duplicated within a single borrow scope.
pub struct LearningContext<'a> {
    /// Bayesian throttle-outcome weights and experience memory.
    pub outcome_tracker: &'a mut OutcomeTracker,

    /// Kalman + CUSUM + Entropy + Hazard + LV + MPC signal processing.
    pub signal_intel: &'a mut SignalIntelligence,

    /// LinUCB contextual bandit for proactive interventions.
    pub predictive_agent: &'a mut PredictiveAgent,

    /// Per-specialist EMA confidence weights (HAZARD / MONOPOLY / KALMAN / LINUCB).
    pub specialist_accuracy: &'a mut SpecialistAccuracyTracker,

    /// RL-augmented overflow guard; tracks OOM history and dynamic threshold offsets.
    pub overflow_guard: &'a mut OverflowGuard,

    /// Pearl-style causal action → outcome graph.
    pub causal_graph: &'a mut CausalGraph,

    /// Self-improving skill registry (Hermes pattern).
    pub skill_registry: &'a mut SkillRegistry,

    /// Bio-inspired neuromodulator (DA / NA / SE / ACh parameter modulation).
    pub neuromod: &'a mut ApolloNeuromodulator,

    /// Per-app energy estimation and savings accounting.
    pub energy_tracker: &'a mut EnergyTracker,
}

impl<'a> LearningContext<'a> {
    /// Construct a `LearningContext` from the nine per-cycle learning subsystems.
    ///
    /// Call this once at the top of the main loop body, before the decision
    /// and observation stages, and drop it before any `Arc<Mutex<…>>` lock is
    /// acquired (so the exclusive borrows are released).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        outcome_tracker: &'a mut OutcomeTracker,
        signal_intel: &'a mut SignalIntelligence,
        predictive_agent: &'a mut PredictiveAgent,
        specialist_accuracy: &'a mut SpecialistAccuracyTracker,
        overflow_guard: &'a mut OverflowGuard,
        causal_graph: &'a mut CausalGraph,
        skill_registry: &'a mut SkillRegistry,
        neuromod: &'a mut ApolloNeuromodulator,
        energy_tracker: &'a mut EnergyTracker,
    ) -> Self {
        Self {
            outcome_tracker,
            signal_intel,
            predictive_agent,
            specialist_accuracy,
            overflow_guard,
            causal_graph,
            skill_registry,
            neuromod,
            energy_tracker,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::causal_graph::CausalGraph;
    use crate::engine::energy::EnergyTracker;
    use crate::engine::neuromodulator::ApolloNeuromodulator;
    use crate::engine::optimization_skills::SkillRegistry;
    use crate::engine::outcome_tracker::OutcomeTracker;
    use crate::engine::overflow_guard::OverflowGuard;
    use crate::engine::predictive_agent::{PredictiveAgent, SpecialistAccuracyTracker};
    use crate::engine::signal_intelligence::SignalIntelligence;

    /// Verify that `LearningContext::new` compiles and all fields are
    /// accessible, confirming the borrow grouping is valid.
    #[test]
    fn learning_context_constructs_and_fields_accessible() {
        let mut outcome_tracker = OutcomeTracker::new();
        let mut signal_intel = SignalIntelligence::new();
        let mut predictive_agent =
            PredictiveAgent::load_or_default(std::path::Path::new("/dev/null"));
        let mut specialist_accuracy = SpecialistAccuracyTracker::new();
        let mut overflow_guard =
            OverflowGuard::load_or_default(std::path::Path::new("/dev/null"), None);
        let mut causal_graph = CausalGraph::new();
        let mut skill_registry = SkillRegistry::new();
        let mut neuromod = ApolloNeuromodulator::new();
        let mut energy_tracker = EnergyTracker::new();

        let ctx = LearningContext::new(
            &mut outcome_tracker,
            &mut signal_intel,
            &mut predictive_agent,
            &mut specialist_accuracy,
            &mut overflow_guard,
            &mut causal_graph,
            &mut skill_registry,
            &mut neuromod,
            &mut energy_tracker,
        );

        // Fields are accessible through the context.
        assert_eq!(ctx.outcome_tracker.total_resolved, 0);
        assert_eq!(ctx.causal_graph.edge_count(), 0);
        assert_eq!(ctx.skill_registry.len(), 0);
    }

    /// Verify split-borrow destructuring compiles cleanly — this is the
    /// pattern callers must use when a helper needs two fields independently.
    #[test]
    fn learning_context_split_borrow_pattern() {
        let mut outcome_tracker = OutcomeTracker::new();
        let mut signal_intel = SignalIntelligence::new();
        let mut predictive_agent =
            PredictiveAgent::load_or_default(std::path::Path::new("/dev/null"));
        let mut specialist_accuracy = SpecialistAccuracyTracker::new();
        let mut overflow_guard =
            OverflowGuard::load_or_default(std::path::Path::new("/dev/null"), None);
        let mut causal_graph = CausalGraph::new();
        let mut skill_registry = SkillRegistry::new();
        let mut neuromod = ApolloNeuromodulator::new();
        let mut energy_tracker = EnergyTracker::new();

        let mut ctx = LearningContext::new(
            &mut outcome_tracker,
            &mut signal_intel,
            &mut predictive_agent,
            &mut specialist_accuracy,
            &mut overflow_guard,
            &mut causal_graph,
            &mut skill_registry,
            &mut neuromod,
            &mut energy_tracker,
        );

        // Split-borrow: destructure to get two independent `&mut` references.
        let LearningContext {
            outcome_tracker: ot,
            causal_graph: cg,
            ..
        } = &mut ctx;
        // Both can be used independently (no aliasing).
        let _ = ot.total_resolved;
        let _ = cg.edge_count();
    }
}
