//! # Apollo Optimizer — Pipeline Stage Abstraction
//!
//! This module defines the conceptual pipeline of a single optimization cycle.
//! The daemon main loop in `src/bin/apollo-optimizerd/main.rs` executes these
//! stages sequentially every cycle (~0.3–2 s depending on reactor events).
//!
//! ## Pipeline anatomy (single cycle)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  1. PressureStage   — snapshot + all boost factors → EffectivePressure
//! │  2. DecisionStage   — snapshot + pressure → Vec<RootAction>
//! │  3. ExecutionStage  — Vec<RootAction> → ExecOutcomes          (execute_actions)
//! │  4. ObservationStage — ExecOutcomes + pre/post pressure → learning feedback
//! │  5. PeriodicStage   — every N cycles: persist, GC, rule induction
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Extraction status
//!
//! Each stage is documented below with its current extraction readiness and
//! the specific blockers that prevent a clean separation.
//!
//! ### Stage 1 — PressureStage (`pressure_stage.rs`)
//!
//! **Status: NOT extracted — logic is already in `effective_pressure::compute`.**
//!
//! All boost factors (hw_boost, batt_boost, thermal_pressure_boost, llm_boost,
//! charging_stress_boost, battery_low_boost, mem_bw_boost, smc_thermal_boost,
//! battery_overheat_boost) are computed inline and then forwarded to
//! `effective_pressure::compute()` which already encapsulates the aggregation.
//! There is no duplicate logic; the "stage" is already as clean as it can be
//! given that each boost factor requires its own hardware read (IOReport, SMC,
//! ThermalBailout, LlmDetector) that happens naturally in the surrounding cycle
//! code.
//!
//! **Blocker**: boost factors are side-outputs of hardware reads that each serve
//! multiple purposes (not just pressure). Extracting them into a single stage
//! struct would require bundling 12+ heterogeneous sensor outputs — a struct that
//! exists implicitly as the set of cycle-local variables.
//!
//! ### Stage 2 — DecisionStage (`decision_stage.rs`)
//!
//! **Status: NOT extracted — requires >10 parameters.**
//!
//! `DecisionContext` would need at minimum:
//! - `snapshot: &SystemSnapshot`
//! - `collector: &sysinfo::System`
//! - `current_profile: OptimizationProfile`
//! - `latency_target: LatencyTarget`
//! - `reactor_weight: f64`
//! - `decide_interactive: &[String]`
//! - `decide_noise: &[String]`
//! - `overflow_thresholds: OverflowThresholds`
//! - `decide_weights: &HashMap<String, PatternWeight>`
//! - `outcome_baseline: f64`
//! - `behavior_interactive_pids: &HashSet<u32>`
//! - `ipc_hints: &HashMap<u32, f64>`
//! - `hop_groups: &HopGroups`
//! - `habituated_pids: &HashSet<u32>`
//! - `causal_confidence: &HashMap<String, f64>`
//! - `skill_registry: &mut SkillRegistry`
//! - `outcome_tracker: &mut OutcomeTracker`
//! - `predictive_agent: &mut PredictiveAgent`
//! - `causal_graph: &CausalGraph`
//! - `foreground_pid: Option<u32>`
//! - `foreground_app: &Option<String>`
//! - `agent_intervention: Intervention`
//! - `signal_digest: &SignalDigest`
//! - `workload_mode: WorkloadMode`
//! - `state: &SharedState`
//!
//! That is well over 10 parameters. The decision stage is the most entangled
//! part of the loop precisely because it must consult every piece of system
//! intelligence accumulated in prior stages.
//!
//! **Pre-condition for extraction**: group related inputs into sub-structs:
//! - `PressureContext { pressure_ram, overflow_thresholds, signal_digest, workload_mode }`
//! - `PolicyContext { decide_interactive, decide_noise, decide_weights, outcome_baseline }`
//! - `LearningContext { skill_registry, outcome_tracker, predictive_agent, causal_graph }`
//!
//! Once those groupings exist, DecisionContext becomes 3–4 references and
//! extraction becomes clean. This is the recommended next refactoring step.
//!
//! ### Stage 3 — ExecutionStage
//!
//! **Status: already extracted** — `execute_actions()` in `execute_actions.rs`.
//!
//! ### Stage 4 — ObservationStage (`observation_stage.rs`)
//!
//! **Status: NOT extracted — ~12 parameters at minimum.**
//!
//! Inputs:
//! - `throttle_names: &[String]`
//! - `pre_pressure: f64`
//! - `post_pressure: f64` (from next snapshot)
//! - `exec_outcomes: &ExecOutcomes`
//! - `cycle_hw_snap: &Option<HardwareSnapshot>`
//! - `snapshot: &SystemSnapshot`
//! - `cycle_count: u64`
//! - `outcome_tracker: &mut OutcomeTracker`
//! - `causal_graph: &mut CausalGraph`
//! - `overflow_guard: &mut OverflowGuard`
//! - `signal_intel: &mut SignalIntelligence`
//! - `signal_digest: &SignalDigest`
//! - `energy_tracker: &mut EnergyTracker`
//! - `predictive_agent: &mut PredictiveAgent`
//! - `specialist_accuracy: &mut SpecialistAccuracyTracker`
//! - `state: &SharedState`
//!
//! **Pre-condition for extraction**: same sub-struct groupings as DecisionStage.
//! Specifically `LearningContext` would reduce this from 16 to ~8 parameters.
//!
//! ### Stage 5 — PeriodicStage (`periodic_stage.rs`)
//!
//! **Status: DEFINED in `periodic_stage.rs`** but not yet wired to main loop.
//!
//! The `PeriodicContext` and `run_periodic()` function are defined. Main loop
//! wiring is blocked by the same parameter-grouping pre-conditions:
//! `outcome_tracker`, `signal_intel`, `specialist_accuracy`, and `skill_registry`
//! are separate unrelated `mut` borrows that cannot be cleanly packed without
//! a grouping struct.
//!
//! See `periodic_stage.rs` for the full interface.

pub mod periodic_stage;

/// Trait implemented by each pipeline stage.
///
/// A `PipelineStage` transforms some `Input` into some `Output` while
/// potentially mutating owned context state. The trait is intentionally minimal
/// so that stages with very different input/output shapes can share a common
/// conceptual interface without requiring complex generic bounds.
///
/// ## Why a trait and not just free functions?
///
/// Future work (v0.8+) may dispatch between different strategy implementations
/// at runtime (e.g., swap the decision stage for a neural policy). Having the
/// trait defined now costs nothing and documents the intended boundary.
pub trait PipelineStage {
    type Input<'a>;
    type Output;

    /// Execute one cycle of this stage.
    fn run<'a>(&mut self, input: Self::Input<'a>) -> Self::Output;
}

/// Aggregated inputs to the decision stage.
///
/// This struct does NOT yet match the full parameter list required by the
/// current inline decision code. It represents the *target* interface once
/// the sub-struct groupings described in the module doc are implemented.
///
/// Fields that are still inline in main.rs are marked `// TODO`.
#[derive(Debug)]
pub struct DecisionContext<'a> {
    /// Kalman-smoothed and boosted memory pressure [0.0, 1.0].
    pub pressure_ram: f64,

    /// Names of processes behaviorally identified as interactive.
    pub interactive_patterns: &'a [String],

    /// Names of processes identified as low-signal noise.
    pub noise_patterns: &'a [String],

    /// Whether the system is in survival mode (pressure > 0.85 or swap thrash).
    pub survival_mode: bool,

    /// Foreground application PID, if known.
    pub foreground_pid: Option<u32>,

    /// Foreground application name, if known.
    pub foreground_app: Option<&'a str>,
    // TODO: overflow_thresholds, signal_digest, workload_mode, skill_registry,
    // outcome_tracker, predictive_agent, causal_graph, collector, snapshot, state...
    // These will be added as the sub-struct groupings are introduced.
}

/// Summary of actions produced by the decision stage.
#[derive(Debug, Default)]
pub struct DecisionOutput {
    /// Actions ready for execution (after skill matching and policy application).
    pub action_count: usize,
    /// Names of processes whose throttle was coordinated via causal clustering.
    pub coordinated_names: Vec<String>,
    /// Whether survival mode was active this cycle.
    pub survival_mode: bool,
}

/// Aggregated inputs to the observation / learning stage.
#[derive(Debug)]
pub struct ObservationContext<'a> {
    /// Memory pressure at the start of the cycle (before execute_actions).
    pub pre_pressure: f64,

    /// Memory pressure observed after execute_actions (same cycle, post-snapshot).
    /// In practice this is the pressure from the *next* cycle's snapshot since we
    /// only have one snapshot per cycle. Pass `pre_pressure` initially and update
    /// on the following cycle.
    pub post_pressure: f64,

    /// Names of processes that were throttled this cycle.
    pub throttle_names: &'a [String],

    /// Current cycle index (used for causal graph timing).
    pub cycle_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_context_has_required_fields() {
        let patterns: Vec<String> = vec!["Chrome".to_string()];
        let ctx = DecisionContext {
            pressure_ram: 0.45,
            interactive_patterns: &patterns,
            noise_patterns: &[],
            survival_mode: false,
            foreground_pid: Some(1234),
            foreground_app: Some("Chrome"),
        };
        assert!(!ctx.survival_mode);
        assert_eq!(ctx.foreground_pid, Some(1234));
        assert_eq!(ctx.pressure_ram, 0.45);
    }

    #[test]
    fn observation_context_pressure_field() {
        let names: Vec<String> = vec!["photoanalysisd".to_string()];
        let ctx = ObservationContext {
            pre_pressure: 0.70,
            post_pressure: 0.65,
            throttle_names: &names,
            cycle_count: 42,
        };
        // post < pre means the action helped
        assert!(ctx.post_pressure < ctx.pre_pressure);
        assert_eq!(ctx.cycle_count, 42);
    }
}
