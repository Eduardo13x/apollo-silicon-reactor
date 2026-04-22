//! # Daemon Teacher Tick
//!
//! Teacher consolidation (S2 → S1 memory transfer) per-cycle tick extracted
//! from main.rs (Wave 34). [Fowler 2004] Strangler Fig — pure move.
//!
//! ## Responsibilities
//! - Read last_suggestion_outcome + last_suggestion from SharedState LLM lock
//! - Run TeacherConsolidator::consolidate() once per unique outcome (applied_at gate)
//! - Increment teacher_consolidations / teacher_improvements metrics
//!
//! ## Ordering invariant
//! Must run AFTER llm_reactive_tick resolves a pending outcome and BEFORE
//! the reactor_weight / decision pass — arousal_state is modified here.

use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::nars_belief::ArousalState;
use apollo_optimizer::engine::outcome_tracker::OutcomeTracker;
use apollo_optimizer::engine::teacher_consolidation::TeacherConsolidator;
use chrono::{DateTime, Utc};

/// Run teacher consolidation: compile Gemma suggestion into pattern_weights + NARS beliefs.
///
/// # Parameters
/// - `state` — SharedState (reads LLM last_suggestion_outcome + last_suggestion;
///   writes teacher_consolidations / teacher_improvements metrics)
/// - `outcome_tracker` — mutable: natural_drift() read + weights/drift_detector written
/// - `teacher_consolidator` — mutable consolidator state
/// - `last_consolidated_at` — prevents re-consolidating the same outcome
/// - `arousal_state` — mutable: updated by consolidate() (acetylcholine modulation)
pub fn run_teacher_consolidation(
    state: &SharedState,
    outcome_tracker: &mut OutcomeTracker,
    teacher_consolidator: &mut TeacherConsolidator,
    last_consolidated_at: &mut Option<DateTime<Utc>>,
    arousal_state: &mut ArousalState,
) {
    let (new_outcome, matching_suggestion) = {
        let guard = state.llm.lock_recover();
        let outcome = guard.llm_state.last_suggestion_outcome.clone();
        let suggestion = guard.llm_state.last_suggestion.clone();
        (outcome, suggestion)
    };
    if let (Some(outcome), Some(suggestion)) = (new_outcome, matching_suggestion) {
        if *last_consolidated_at != Some(outcome.applied_at) {
            let natural_drift = outcome_tracker.natural_drift();
            let report = teacher_consolidator.consolidate(
                &outcome,
                &suggestion,
                natural_drift,
                &mut outcome_tracker.weights,
                &mut outcome_tracker.drift_detector,
                arousal_state,
            );
            *last_consolidated_at = Some(outcome.applied_at);
            if !matches!(report.verdict, "BELOW_DEADBAND") {
                let mut mx = state.metrics.lock_recover();
                mx.metrics.teacher_consolidations += 1;
                if report.verdict == "IMPROVED" {
                    mx.metrics.teacher_improvements += 1;
                }
            }
        }
    }
}
