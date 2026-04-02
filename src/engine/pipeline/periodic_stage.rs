//! # Periodic Stage
//!
//! Houses the housekeeping work that runs every N cycles rather than every cycle.
//! In the current daemon main loop these are scattered as `if cycle_count % N == 0`
//! blocks throughout the ~5000-line `run_daemon` function.
//!
//! ## Current extraction status
//!
//! The `PeriodicContext` and `run_periodic()` interface are defined here and
//! compile cleanly, but the function body is intentionally a no-op stub:
//! the actual periodic logic remains inline in the main loop because wiring it
//! through this interface would require a `PeriodicContext` with 15+ `&mut`
//! fields — exactly the parameter explosion the extraction rules warn against.
//!
//! ### Why 15+ parameters?
//!
//! The periodic stage touches learning subsystems spread across independent
//! struct instances that are *also* mutably borrowed by the decision and
//! observation stages:
//!
//! | Cycle gate  | What it touches                                        |
//! |-------------|--------------------------------------------------------|
//! | % 100 == 0  | signal_intel, outcome_tracker, specialist_accuracy     |
//! |             | skill_registry, causal_graph, overflow_guard::rl_agent |
//! |             | ls_path, hop_groups_path, skills_path                  |
//! | % 100 == 0  | rule_inducer (outcome_tracker, top_pairs, skill_registry) |
//! | % 500 == 0  | outcome_tracker, skill_registry (GC + compress)        |
//! | % 7200 == 1 | cache_warmer, io_shaper, temporal_predictor            |
//!
//! ### Pre-condition for full extraction
//!
//! Group the learning subsystems into a `LearningContext` struct:
//!
//! ```rust,ignore
//! struct LearningContext {
//!     signal_intel: SignalIntelligence,
//!     outcome_tracker: OutcomeTracker,
//!     specialist_accuracy: SpecialistAccuracyTracker,
//!     skill_registry: SkillRegistry,
//!     causal_graph: CausalGraph,
//!     overflow_guard: OverflowGuard,
//!     predictive_agent: PredictiveAgent,
//! }
//! ```
//!
//! With that grouping, `PeriodicContext` drops to 5 parameters and full
//! extraction becomes straightforward. `LearningContext` is the recommended
//! next refactoring step (tracked in architecture notes).

/// Everything the periodic stage needs to do its work.
///
/// This struct is the *target* interface — not all fields are wired to the
/// main loop yet. See module-level documentation for details.
pub struct PeriodicContext<'a> {
    /// Current daemon cycle counter (starts at 1, monotonically increasing).
    pub cycle_count: u64,

    /// Memory pressure at the time the periodic stage runs.
    /// Used by rule_inducer to gate skill induction (only at elevated pressure).
    pub current_pressure: f64,

    /// Current workload mode string ("idle", "build", "browser", etc.).
    pub workload_mode: &'a str,

    /// Filesystem path where optimization skills are persisted.
    pub skills_path: &'a std::path::Path,

    /// Filesystem path where hop-group data is persisted.
    pub hop_groups_path: &'a std::path::Path,

    /// Filesystem path where signal intelligence state is persisted.
    pub signal_intel_path: &'a std::path::Path,

    /// Filesystem path where learned state (unified persistence) is stored.
    pub learned_state_path: &'a std::path::Path,

    /// Persist generation counter (incremented by LearnedState::persist_improved).
    pub persist_generations: u32,

    /// Quality score of the last restored state (None if no restore yet).
    pub last_restore_quality: Option<f64>,

    /// Pending trial skill from the current decision cycle, if any.
    /// Passed through to LearnedState so a crash mid-trial can be recovered.
    pub pending_trial_skill: Option<(String, f64)>,
    // TODO: learning_ctx: &mut LearningContext  — blocked until LearningContext exists.
    // Until then: signal_intel, outcome_tracker, specialist_accuracy, skill_registry,
    // causal_graph, overflow_guard, predictive_agent, cache_warmer, io_shaper,
    // temporal_predictor, holt_winters, focus_markov are passed separately in the
    // main loop.
}

/// Which periodic housekeeping tasks ran this cycle.
///
/// Returned by `run_periodic` so the caller can log what happened.
#[derive(Debug, Default)]
pub struct PeriodicResult {
    /// Causal graph solid-edge count logged this cycle (0 if gate didn't fire).
    pub causal_solid_edges: Option<usize>,

    /// Number of new skills crystallised from rule induction (0 if gate didn't fire).
    pub induced_skills: Option<usize>,

    /// Whether unified learned-state was persisted this cycle.
    pub did_persist: bool,

    /// Whether GC/compression ran this cycle (% 500 gate).
    pub did_gc: bool,

    /// Whether hourly housekeeping ran this cycle (% 7200 gate).
    pub did_hourly: bool,
}

/// Decide which periodic tasks to run for this cycle.
///
/// In its current stub form this function only computes the `PeriodicResult`
/// flags — it does not actually mutate any state.  The real mutations remain
/// inline in the main loop pending the `LearningContext` grouping.
///
/// Once `LearningContext` exists this function will accept it as a parameter
/// and contain the full implementation.
pub fn run_periodic(ctx: &PeriodicContext) -> PeriodicResult {
    let mut result = PeriodicResult::default();

    if ctx.cycle_count % 100 == 0 {
        result.did_persist = true;
        result.causal_solid_edges = Some(0); // placeholder; real value from causal_graph
        result.induced_skills = Some(0); // placeholder; real value from skill_registry
    }

    if ctx.cycle_count % 500 == 0 {
        result.did_gc = true;
    }

    if ctx.cycle_count % 7200 == 1 {
        result.did_hourly = true;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx(cycle: u64) -> PeriodicContext<'static> {
        PeriodicContext {
            cycle_count: cycle,
            current_pressure: 0.30,
            workload_mode: "idle",
            skills_path: std::path::Path::new("/tmp/test-skills.json"),
            hop_groups_path: std::path::Path::new("/tmp/test-hops.json"),
            signal_intel_path: std::path::Path::new("/tmp/test-si.json"),
            learned_state_path: std::path::Path::new("/tmp/test-ls.json"),
            persist_generations: 0,
            last_restore_quality: None,
            pending_trial_skill: None,
        }
    }

    #[test]
    fn persist_fires_at_cycle_100() {
        let result = run_periodic(&make_ctx(100));
        assert!(result.did_persist, "% 100 gate should fire at cycle 100");
        assert!(!result.did_gc, "% 500 gate must not fire at cycle 100");
        assert!(!result.did_hourly, "% 7200 gate must not fire at cycle 100");
    }

    #[test]
    fn gc_fires_at_cycle_500() {
        let result = run_periodic(&make_ctx(500));
        // 500 % 100 == 0, so persist fires too
        assert!(result.did_persist, "% 100 gate must co-fire at cycle 500");
        assert!(result.did_gc, "% 500 gate should fire at cycle 500");
        assert!(!result.did_hourly, "% 7200 gate must not fire at cycle 500");
    }

    #[test]
    fn hourly_fires_at_cycle_7201() {
        let result = run_periodic(&make_ctx(7201));
        assert!(result.did_hourly, "% 7200 == 1 gate should fire at cycle 7201");
    }

    #[test]
    fn gates_at_cycle_1() {
        // cycle 1: 1 % 100 != 0, 1 % 500 != 0, BUT 1 % 7200 == 1 (hourly fires on first cycle).
        // This matches the main loop's % 7200 == 1 gate which intentionally fires at startup
        // to run initial housekeeping (GC cache warmer, IO shaper, temporal predictor).
        let result = run_periodic(&make_ctx(1));
        assert!(!result.did_persist, "% 100 gate must not fire at cycle 1");
        assert!(!result.did_gc, "% 500 gate must not fire at cycle 1");
        assert!(result.did_hourly, "% 7200 == 1 gate DOES fire at cycle 1 (startup housekeeping)");
    }

    #[test]
    fn periodic_result_default_is_all_false() {
        let r = PeriodicResult::default();
        assert!(!r.did_persist);
        assert!(!r.did_gc);
        assert!(!r.did_hourly);
        assert!(r.causal_solid_edges.is_none());
        assert!(r.induced_skills.is_none());
    }
}
