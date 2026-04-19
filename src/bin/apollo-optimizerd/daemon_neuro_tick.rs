//! # Daemon Neuro Tick
//!
//! Late-cycle bio-inspired + neurocognitive processing.  Extracted from
//! the daemon main loop as part of the V1.1.0 Strangler Fig pass
//! [Fowler 2004].
//!
//! ## Two blocks, one cohesive phase
//!
//! 1. **Neuromodulator parameter modulation** — builds a
//!    [`NeuroSignals`] bundle from the fresh signal digest, outcome
//!    tracker RL penalty, stability-oracle instability signal, thermal
//!    phase, and overflow-guard state, then ticks
//!    [`ApolloNeuromodulator`] and pushes derived parameters
//!    (`alpha_multiplier`, `epsilon_bonus`, `dyna_steps`,
//!    `serotonin_shift`) into the RL agent and
//!    [`SignalIntelligence`].  See [McGaugh 2004] for the neuroscience
//!    grounding of the four-signal bio-inspired model.
//!
//! 2. **Neurocognitive tick** — derives real epistemic signals from
//!    subsystems ([Lakshminarayanan 2017] predictive uncertainty from
//!    ensemble variance; LinUCB upper-confidence bound; full causal
//!    confidence map) then runs the 8-module cognitive pipeline via
//!    [`cognitive_tick::run_cognitive_tick`].  Result is returned so the
//!    caller can cache it in `prev_cog_decision` for next-cycle gating
//!    and current-cycle metrics.
//!
//! ## Ordering invariant (peer-review 2026-04-18)
//!
//! `Neuromod → Neurocognitive → Metrics/QoS elevation`.  UCHS pulls
//! `MetaCognition.meta_confidence` (D1) and `DriftDetector.score()` (D3)
//! that are updated by `run_cognitive_tick`, so metrics reporting MUST
//! stay downstream.  Fluidity QoS elevation also reads serotonin-shifted
//! state via `signal_intel`, so it must stay downstream of the
//! neuromodulator tick.  Do not reorder without re-running the NotebookLM
//! peer review.
//!
//! ## Purity
//!
//! The epistemic-signals derivation is a **pure function** of subsystem
//! read-only views (arm avg-rewards, arm pulls, total cycles, causal
//! confidence map).  All mutation happens inside
//! [`cognitive_tick::run_cognitive_tick`] and inside the neuromodulator
//! `tick` itself.
//!
//! ## Shared-state carry-overs
//!
//! Every mutable cross-cycle carrier is passed in by `&mut` from the
//! caller — nothing is smuggled through globals or statics.

use crate::cognitive_tick::{self, CognitiveDecision, CognitiveState, CognitiveTickInputs};
use apollo_optimizer::engine::neuromodulator::NeuroSignals;
use apollo_optimizer::engine::nars_belief::DriftDetector;
use apollo_optimizer::engine::pipeline::learning_context::LearningContext;
use apollo_optimizer::engine::signal_intelligence::SignalDigest;
use apollo_optimizer::engine::stability_oracle::StabilityOracle;
use apollo_optimizer::engine::thermal_bailout::{CoolingPhase, ThermalAction};

/// Apply the bio-inspired neuromodulator: build NeuroSignals, tick the
/// neuromodulator, then push derived parameters into the RL agent and
/// signal_intel.
///
/// Pre-conditions:
/// - `lctx.neuromod` is the per-daemon `ApolloNeuromodulator`.
/// - `stability_oracle` has already been updated this cycle.
///
/// Post-conditions:
/// - `lctx.overflow_guard.rl_agent` (if present) has its
///   `neuro_alpha_mult`, `neuro_epsilon_bonus`, `dyna_steps` synced and
///   `enforce_constraints()` called (Hermes infrastructure lock).
/// - `lctx.signal_intel.neuro_serotonin_shift` reflects the new tick.
pub fn apply_neuromodulator(
    lctx: &mut LearningContext<'_>,
    signal_digest: &SignalDigest,
    stability_oracle: &StabilityOracle,
    thermal_action: &ThermalAction,
    process_count: usize,
    cpu_temp_celsius: Option<f64>,
) {
    // Graded thermal stress [0, 1]: 0 at ≤60°C, 0.5 at 80°C, 1.0 at ≥100°C.
    // Falls back to thermal phase estimate when SMC/IOKit temperature is unavailable.
    // [Wilson & Dayan 2004] — graded neuromodulatory signals reduce policy variance.
    let thermal_stress = if let Some(temp) = cpu_temp_celsius {
        ((temp - 60.0) / 40.0).clamp(0.0, 1.0)
    } else {
        // Binary fallback from cooling phase — equivalent to the previous bool gate.
        if thermal_action.phase >= CoolingPhase::Phase2Moderate {
            1.0
        } else {
            0.0
        }
    };
    let overflow_occurred = lctx.overflow_guard.history.total_overflows > 0;
    let neuro_signals = NeuroSignals {
        pressure_drop: signal_digest.pressure_smooth as f64 * -1.0
            * signal_digest.pressure_velocity,
        // Combine outcome-tracker RL penalty with stability oracle signal.
        // rl_penalty ∈ [-3, 0]; instability_penalty ∈ [0, 1] scaled by 0.5
        // → max additional penalty = -0.5, keeping the existing penalty
        // dominant while letting stability shape policy at the margin.
        // [Sutton & Barto 2018] §17.4 — reward shaping must preserve scale
        // hierarchy or it inverts the optimal policy.
        outcome_penalty: lctx.outcome_tracker.rl_penalty()
            - 0.5
                * stability_oracle.instability_penalty_attenuated(
                    apollo_optimizer::engine::daemon_helpers::system_uptime_secs(),
                ),
        overflow_occurred,
        urgency: signal_digest.urgency,
        regime_shift_up: signal_digest.regime_shift_up,
        pressure_velocity: signal_digest.pressure_velocity,
        thermal_stress,
        pressure_smooth: signal_digest.pressure_smooth as f64,
        regime_shift_down: signal_digest.regime_shift_down,
        process_count,
        entropy_anomaly: signal_digest.entropy_anomaly as f64,
        rl_exploring: lctx
            .overflow_guard
            .rl_agent
            .as_ref()
            .map_or(false, |rl| rl.total_ticks() < 200),
    };
    lctx.neuromod.tick(&neuro_signals);

    // Push derived params to subsystems + enforce constraints.
    if let Some(rl) = &mut lctx.overflow_guard.rl_agent {
        rl.neuro_alpha_mult = lctx.neuromod.alpha_multiplier;
        rl.neuro_epsilon_bonus = lctx.neuromod.epsilon_bonus;
        rl.dyna_steps = lctx.neuromod.dyna_steps;
        rl.enforce_constraints(); // Infrastructure-locked (Hermes)
    }
    lctx.signal_intel.neuro_serotonin_shift = lctx.neuromod.serotonin_shift;
}

/// Run the full neurocognitive tick: derive real epistemic signals from
/// subsystems, build [`CognitiveTickInputs`], and invoke
/// [`cognitive_tick::run_cognitive_tick`].
///
/// Returns the fresh [`CognitiveDecision`] for the caller to cache in
/// `prev_cog_decision`.
///
/// Ordering: MUST run after `learning_tick` (so drift/causal/arousal are
/// fresh) and BEFORE `metrics_reporter` / UCHS / fluidity QoS elevation
/// — those consumers read state mutated inside this tick.
pub fn run_neurocognitive_tick(
    lctx: &mut LearningContext<'_>,
    cognitive_state: &mut CognitiveState,
    cycle_count: u64,
    signal_digest: &SignalDigest,
    throttle_names_for_outcome: &[String],
    workload_mode_str: &str,
) -> CognitiveDecision {
    // ── Derive real epistemic signals from subsystems ─────────────────
    // [Lakshminarayanan 2017] predictive uncertainty from ensemble variance.
    // RL Q-value variance: std-dev across arm avg-rewards → spread = uncertainty.
    let rl_q_variance = {
        let avg = lctx.predictive_agent.arm_avg_rewards();
        let n = avg.len() as f64;
        let mean = avg.iter().sum::<f64>() / n;
        let var = avg.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n;
        (var.sqrt() as f32).clamp(0.0, 1.0)
    };
    // LinUCB exploration: UCB for the most-pulled arm (lower = more exploited).
    let linucb_exploration = {
        let pulls = lctx.predictive_agent.arm_pulls();
        let total = lctx.predictive_agent.total_cycles();
        if total > 1 {
            let best = pulls.iter().copied().max().unwrap_or(1).max(1);
            ((2.0 * (total as f64).ln() / best as f64).sqrt().min(1.0) as f32).clamp(0.0, 1.0)
        } else {
            1.0 // maximum uncertainty on cold start
        }
    };
    // Full causal confidence map — lets SelfRewardingEvaluator look up
    // any past action's confidence, not just the current one.
    // [Yuan 2024 §3 DR-ZERO]: CausalGraph as internal oracle for JuicyScore.
    let causal_confidence_map: Vec<(String, f32)> =
        lctx.causal_graph.confidence_map().into_iter().collect();
    let top_causal = lctx
        .causal_graph
        .solid_edges_by_impact()
        .first()
        .map(|e| e.confidence)
        .unwrap_or(0.0);
    let cog_inputs = CognitiveTickInputs {
        cycle: cycle_count,
        pressure: signal_digest.pressure_smooth,
        drift_score: lctx.outcome_tracker.nars_drift_score(),
        rl_q_variance,
        linucb_exploration,
        nars_min_confidence: (1.0 - lctx.outcome_tracker.nars_drift_score() as f32)
            .clamp(0.0, 1.0),
        outcome_effectiveness: lctx.outcome_tracker.overall_effectiveness(),
        causal_confidence: top_causal,
        causal_confidence_map,
        latest_action: throttle_names_for_outcome
            .first()
            .map(|n| format!("throttle:{}", n)),
        predicted_score: lctx
            .predictive_agent
            .arm_avg_rewards()
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .clamp(0.0, 1.0) as f32,
        workload_fingerprint: workload_mode_str
            .bytes()
            .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64)),
        rl_state_idx: 0,
        rl_q_delta: 0.0,
        linucb_arm_idx: 0,
        linucb_delta: 0.0,
    };
    let drift: &mut DriftDetector = &mut lctx.outcome_tracker.drift_detector;
    cognitive_tick::run_cognitive_tick(cognitive_state, &cog_inputs, Some(drift))
}
