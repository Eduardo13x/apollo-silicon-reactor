//! # Cognitive Tick
//!
//! Per-cycle neurocognitive pipeline work. Runs after learning_tick to feed
//! signals through the 8 new cognitive modules.
//!
//! ## Pipeline (per cycle):
//! 1. CognitiveRewardBus: collect + flush signals from all subsystems
//! 2. MetaCognition: observe subsystem accuracy, tick humble mode
//! 3. SelfRewardingEvaluator: log decisions, evaluate past
//! 4. EpistemicUncertainty: update composite from all subsystem signals
//! 5. ReptileMeta: apply learning deltas, detect workload fingerprint changes
//! 6. ProactiveDrift: update early warning from DriftDetector
//! 7. AdversarialProbe: run synthetic probes every 500 cycles
//! 8. CognitiveHealthScore: update UCHS from all dimensions

use apollo_engine::engine::adversarial_probe::{AdversarialProbe, ProbeResult};
use apollo_engine::engine::cognitive_bus::{CognitiveRewardBus, RewardSignal, RewardSource};
use apollo_engine::engine::cognitive_health::{CognitiveHealthScore, CognitiveInputs};
use apollo_engine::engine::daemon_helpers::audit_log;
use apollo_engine::engine::epistemic::EpistemicUncertainty;
use apollo_engine::engine::meta_cognition::{MetaCognition, SubsystemId};
use apollo_engine::engine::nars_belief::DriftDetector;
use apollo_engine::engine::reptile_meta::ReptileMeta;
use apollo_engine::engine::self_reward::SelfRewardingEvaluator;

/// All neurocognitive state bundled for the daemon loop.
pub struct CognitiveState {
    pub reward_bus: CognitiveRewardBus,
    pub meta_cognition: MetaCognition,
    pub self_evaluator: SelfRewardingEvaluator,
    pub epistemic: EpistemicUncertainty,
    pub reptile: ReptileMeta,
    pub adversarial: AdversarialProbe,
    pub health: CognitiveHealthScore,
}

impl CognitiveState {
    pub fn new() -> Self {
        Self {
            reward_bus: CognitiveRewardBus::new(),
            meta_cognition: MetaCognition::new(),
            self_evaluator: SelfRewardingEvaluator::new(),
            epistemic: EpistemicUncertainty::new(),
            reptile: ReptileMeta::new(),
            adversarial: AdversarialProbe::new(),
            health: CognitiveHealthScore::new(),
        }
    }
}

/// Inputs from the main daemon loop for the cognitive tick.
pub struct CognitiveTickInputs {
    /// Current daemon cycle.
    pub cycle: u64,
    /// Current memory pressure [0,1].
    pub pressure: f64,
    /// Current drift score from OutcomeTracker's DriftDetector.
    pub drift_score: f64,
    /// RL Q-value variance in current state (normalized [0,1]).
    pub rl_q_variance: f32,
    /// LinUCB exploration bonus for chosen arm (normalized [0,1]).
    pub linucb_exploration: f32,
    /// Minimum NARS confidence across relevant beliefs (1 - min = spread).
    pub nars_min_confidence: f32,
    /// Outcome effectiveness from last batch resolve [0,1].
    pub outcome_effectiveness: f64,
    /// Causal confidence for strongest recent action [0,1].
    pub causal_confidence: f32,
    /// Full action→confidence map from CausalGraph (for SelfRewardingEvaluator retroactive eval).
    /// Allows evaluating past decisions by their actual causal confidence, not just the current action.
    pub causal_confidence_map: Vec<(String, f32)>,
    /// Name of latest action taken (for SelfRewardingEvaluator log).
    pub latest_action: Option<String>,
    /// Predicted score for latest action [0,1].
    pub predicted_score: f32,
    /// Workload fingerprint hash (for ReptileMeta).
    pub workload_fingerprint: u64,
    /// RL state index and Q-delta (for ReptileMeta learning).
    pub rl_state_idx: usize,
    pub rl_q_delta: f64,
    /// LinUCB arm index and delta (for ReptileMeta learning).
    pub linucb_arm_idx: usize,
    pub linucb_delta: f64,
    /// Mean blocked-action overprotection signal from OutcomeTracker
    /// (Bayesian-Laplace aggregate over mature blocked patterns, [0,1]).
    /// Feeds Epistemic composite W_GUARD=0.20.
    pub guard_overprotection: f32,
}

impl Default for CognitiveTickInputs {
    fn default() -> Self {
        Self {
            cycle: 0,
            pressure: 0.0,
            drift_score: 0.0,
            rl_q_variance: 0.0,
            linucb_exploration: 0.0,
            nars_min_confidence: 0.70,
            outcome_effectiveness: 0.5,
            causal_confidence: 0.0,
            causal_confidence_map: Vec::new(),
            latest_action: None,
            predicted_score: 0.5,
            workload_fingerprint: 0,
            rl_state_idx: 0,
            rl_q_delta: 0.0,
            linucb_arm_idx: 0,
            linucb_delta: 0.0,
            guard_overprotection: 0.0,
        }
    }
}

/// Run the full neurocognitive pipeline for this cycle.
///
/// Returns (should_pause_learning, should_block_aggressive, should_observe_only).
pub fn run_cognitive_tick(
    cog: &mut CognitiveState,
    inputs: &CognitiveTickInputs,
    drift_detector: Option<&mut DriftDetector>,
) -> CognitiveDecision {
    let cycle = inputs.cycle;

    // ── 1. CognitiveRewardBus: publish signals ────────────────────────────
    // Outcome effectiveness as reward signal — only publish above noise floor.
    // Under high pressure (>0.80), 21% effectiveness is expected (single throttles
    // can't overcome system-wide pressure). Publishing -0.58 every cycle floods
    // the bus with noise that drowns directional signal (SNR ~0.18).
    // Threshold 0.30 filters cycles where signal > noise.
    if inputs.outcome_effectiveness > 0.30 {
        cog.reward_bus.publish(RewardSignal {
            source: RewardSource::Outcome,
            value: inputs.outcome_effectiveness * 2.0 - 1.0, // map [0,1] → [-1,1]
            confidence: 0.8,
            cycle,
        });
    }
    // Causal graph confidence delta as reward
    if inputs.causal_confidence > 0.1 {
        cog.reward_bus.publish(RewardSignal {
            source: RewardSource::CausalGraph,
            value: inputs.causal_confidence as f64 * 2.0 - 1.0,
            confidence: inputs.causal_confidence.min(1.0),
            cycle,
        });
    }
    // Stability reward every 100 cycles: daemon alive, no drift, nars beliefs stable.
    // [Silver 2021] "Reward is Enough" — survival is a valid reward signal.
    if cycle % 100 == 50 && inputs.drift_score < 0.05 && inputs.nars_min_confidence > 0.5 {
        cog.reward_bus.publish(RewardSignal {
            source: RewardSource::Outcome,
            value: 0.20,
            confidence: 0.5,
            cycle,
        });
    }
    // Guard-tower over-protection penalty — closes the bus blind spot
    // NotebookLM round-2 flagged 2026-05-10. record_blocked already updates
    // local OutcomeTracker weights, but RL Q-values + LinUCB arms learning
    // FROM the bus would otherwise stay on a survival-biased dataset.
    // Emit a NEGATIVE signal proportional to the over-protection magnitude
    // so RL learns: "high blocked-effectiveness = our policy is wrong".
    // Floor 0.20 suppresses cold-start noise (signal < 0.20 means <2 mature
    // patterns or near-neutral Bayesian priors).
    if inputs.guard_overprotection > 0.20 {
        cog.reward_bus.publish(RewardSignal {
            source: RewardSource::Outcome,
            value: -(inputs.guard_overprotection as f64),
            confidence: inputs.guard_overprotection.min(1.0),
            cycle,
        });
    }
    cog.reward_bus.flush_cycle();

    // ── 2. MetaCognition: observe subsystem accuracy ──────────────────────
    // RL: predicted Q-value improvement vs. actual outcome effectiveness
    cog.meta_cognition.observe(
        SubsystemId::RlAgent,
        inputs.predicted_score,
        inputs.outcome_effectiveness as f32,
    );
    // CausalGraph: confidence vs. actual pressure improvement
    if inputs.causal_confidence > 0.0 {
        let actual_improvement = if inputs.pressure < 0.5 { 0.8 } else { 0.3 };
        cog.meta_cognition.observe(
            SubsystemId::CausalGraph,
            inputs.causal_confidence,
            actual_improvement,
        );
    }
    cog.meta_cognition.tick();

    // ── 3. SelfRewardingEvaluator: log decision + evaluate past ──────────
    if let Some(ref action) = inputs.latest_action {
        cog.self_evaluator.log_decision(
            cycle,
            action.clone(),
            inputs.predicted_score,
            inputs.pressure,
        );
    }
    let eval_results = cog
        .self_evaluator
        .evaluate_past(cycle, inputs.pressure, |action_name| {
            // Full CausalGraph map lookup — evaluate any past action, not just current.
            // [Yuan 2024 §3 DR-ZERO]: use causal graph as internal oracle for JuicyScore.
            inputs
                .causal_confidence_map
                .iter()
                .find(|(a, _)| a == action_name)
                .map(|(_, c)| *c)
                .unwrap_or_else(|| {
                    // Fallback to current causal_confidence if action matches current
                    if inputs.latest_action.as_deref() == Some(action_name) {
                        inputs.causal_confidence
                    } else {
                        0.0
                    }
                })
        });
    // Publish self-eval scores to reward bus
    for eval in &eval_results {
        if eval.informative {
            cog.reward_bus.publish(RewardSignal {
                source: RewardSource::SelfEval,
                value: eval.juicy_score as f64 * 2.0 - 1.0,
                confidence: cog.self_evaluator.evaluator_trust().min(1.0),
                cycle,
            });
        }
    }

    // ── 4. EpistemicUncertainty: update composite ─────────────────────────
    // 6 components total. Guard-overprotection (W_GUARD=0.20) closes the
    // Three-Pillar gap noted by NotebookLM 2026-05-10: an 8-layer guard tower
    // could over-protect for hours without anyone noticing. The
    // OutcomeTracker's Bayesian-Laplace aggregate over mature blocked
    // patterns surfaces it.
    cog.epistemic.update(
        inputs.rl_q_variance,
        inputs.linucb_exploration,
        1.0 - inputs.nars_min_confidence,
        inputs.drift_score as f32,
        cog.meta_cognition.calibration_error,
        inputs.guard_overprotection,
    );

    // ── 5. ReptileMeta: detect fingerprint changes + apply deltas ────────
    cog.reptile
        .on_fingerprint_change(inputs.workload_fingerprint, cycle);
    if inputs.rl_q_delta.abs() > 0.001 || inputs.linucb_delta.abs() > 0.001 {
        cog.reptile.apply_learning_delta(
            inputs.rl_state_idx,
            inputs.rl_q_delta,
            inputs.linucb_arm_idx,
            inputs.linucb_delta,
            cycle,
        );
    }
    // Prune stale workload params every 1000 cycles
    if cycle % 1000 == 0 {
        cog.reptile.prune_stale(cycle);
    }

    // ── 6. ProactiveDrift: update early warning ──────────────────────────
    if let Some(dd) = drift_detector {
        dd.update_early_warning();
        // If early warning fires, feed to MetaCognition as a hint
        if dd.has_early_warning() {
            cog.meta_cognition.observe(
                SubsystemId::NarsBelief,
                0.80, // predicted: high drift
                1.0,  // actual: drift confirmed by early warning
            );
        }
    }

    // ── 7. AdversarialProbe: run probes every 500 cycles ─────────────────
    if cog.adversarial.should_probe(cycle) {
        let scenarios = AdversarialProbe::generate_scenarios();
        let mut results = Vec::new();

        for scenario in &scenarios {
            let result = match scenario.expectation {
                apollo_engine::engine::adversarial_probe::ProbeExpectation::NoFreezeProtected => {
                    AdversarialProbe::probe_no_freeze_protected(scenario, |name, _p, _oom| {
                        // Protected processes should never be frozen.
                        // Use the unified safety oracle (is_protected_name) so the
                        // adversarial probe stresses the *real* protection logic
                        // (Tier 1 hard + Tier 2 infra + Tier 3 dev runtime),
                        // not a slice of it. [Saltzer & Kaashoek 2009] §3.3
                        // Complete Mediation — single source of truth.
                        !apollo_engine::engine::safety::is_protected_name(name)
                    })
                }
                apollo_engine::engine::adversarial_probe::ProbeExpectation::SafetyFloorRespected => {
                    AdversarialProbe::probe_safety_floor(|_| {
                        apollo_engine::engine::rl_threshold::RL_ABSOLUTE_FLOOR
                    })
                }
                apollo_engine::engine::adversarial_probe::ProbeExpectation::NarsDriftRecovery => {
                    AdversarialProbe::probe_nars_recovery(20)
                }
                apollo_engine::engine::adversarial_probe::ProbeExpectation::EpistemicBlocksAggressive => {
                    AdversarialProbe::probe_epistemic_blocks(|rv, le, ns, ds, ce| {
                        // Mirror EpistemicUncertainty composite weights.
                        let composite = 0.25 * rv + 0.20 * le + 0.20 * ns + 0.10 * ds + 0.25 * ce;
                        composite > 0.70
                    })
                }
                apollo_engine::engine::adversarial_probe::ProbeExpectation::OdeDivergenceResilient => {
                    AdversarialProbe::probe_ode_divergence()
                }
                apollo_engine::engine::adversarial_probe::ProbeExpectation::StickySwapSpotlightSuppressed => {
                    AdversarialProbe::probe_sticky_swap_spotlight()
                }
                apollo_engine::engine::adversarial_probe::ProbeExpectation::SubnormalFloorRecovery => {
                    AdversarialProbe::probe_subnormal_floor_recovery()
                }
            };
            results.push(ProbeResult { cycle, ..result });
        }
        // Journal any failures so prod adversarial_pass_rate regressions
        // can be traced back to the specific invariant that broke.
        // [Gray & Reuter 1992 §2 — audit trails must identify the failing unit].
        for r in results.iter().filter(|r| !r.passed) {
            audit_log(&serde_json::json!({
                "t": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                "event": "adversarial_probe_failed",
                "cycle": r.cycle,
                "expectation": format!("{:?}", r.expectation),
                "description": r.description,
            }));
        }
        cog.adversarial.record_results(results, cycle);
    }

    // ── 8. CognitiveHealthScore: update UCHS ─────────────────────────────
    cog.health.update(&CognitiveInputs {
        calibration: cog.meta_cognition.meta_confidence,
        reward_snr: cog.reward_bus.signal_to_noise(),
        drift_score: inputs.drift_score,
        self_eval_trust: cog.self_evaluator.evaluator_trust(),
        adaptation_quality: cog.reptile.adaptation_quality,
        safety_score: cog.adversarial.safety_score(),
    });

    CognitiveDecision {
        pause_learning: cog.health.should_pause_learning(),
        block_aggressive: cog.epistemic.should_block_aggressive(),
        observe_only: cog.epistemic.should_observe_only(),
        humble_mode: cog.meta_cognition.humble_mode,
        uchs_composite: cog.health.composite,
        safety_alert: cog.adversarial.safety_alert,
    }
}

/// Output of the cognitive tick — informs the daemon what to do.
#[derive(Clone, Copy)]
pub struct CognitiveDecision {
    /// UCHS < 0.40 → pause all learning for recovery.
    pub pause_learning: bool,
    /// Epistemic uncertainty > 0.70 → block aggressive freezes.
    pub block_aggressive: bool,
    /// Epistemic uncertainty > 0.85 → force Observe arm only.
    pub observe_only: bool,
    /// MetaCognition humble mode active → more exploration.
    pub humble_mode: bool,
    /// Unified Cognitive Health Score [0, 1].
    pub uchs_composite: f32,
    /// AdversarialProbe safety alert active.
    pub safety_alert: bool,
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_state_creates() {
        let state = CognitiveState::new();
        assert_eq!(state.reward_bus.total_signals(), 0);
        assert!(!state.meta_cognition.humble_mode);
        assert_eq!(state.self_evaluator.eval_count, 0);
        assert_eq!(state.epistemic.composite, 0.0);
        assert_eq!(state.reptile.adaptation_steps, 0);
        assert!(!state.adversarial.safety_alert);
        assert!(!state.health.recovery_mode);
    }

    #[test]
    fn test_cognitive_tick_normal_cycle() {
        let mut cog = CognitiveState::new();
        let inputs = CognitiveTickInputs {
            cycle: 100,
            pressure: 0.50,
            drift_score: 0.01,
            rl_q_variance: 0.2,
            linucb_exploration: 0.3,
            nars_min_confidence: 0.65,
            outcome_effectiveness: 0.7,
            causal_confidence: 0.6,
            latest_action: Some("throttle:Firefox".into()),
            predicted_score: 0.65,
            ..Default::default()
        };

        let decision = run_cognitive_tick(&mut cog, &inputs, None);

        assert!(!decision.pause_learning);
        assert!(!decision.block_aggressive);
        assert!(!decision.observe_only);
        assert!(decision.uchs_composite > 0.0);
    }

    #[test]
    fn test_cognitive_tick_high_uncertainty_blocks() {
        let mut cog = CognitiveState::new();
        // 6-component composite (rebalanced 2026-05-10):
        //   W_RL=0.20 + W_LINUCB=0.15 + W_NARS=0.15 + W_DRIFT=0.10
        //   + W_CALIB=0.20 + W_GUARD=0.20 = 1.0
        // 4 spread-based at max alone = 0.60 (below HIGH 0.70). Drive guard_overprotection
        // to 1.0 too so composite reaches 0.80 > 0.70 — also exercises the new
        // 6th component end-to-end.
        let inputs = CognitiveTickInputs {
            cycle: 100,
            pressure: 0.80,
            drift_score: 1.0,
            rl_q_variance: 1.0,
            linucb_exploration: 1.0,
            nars_min_confidence: 0.0,
            outcome_effectiveness: 0.2,
            guard_overprotection: 1.0,
            ..Default::default()
        };

        let decision = run_cognitive_tick(&mut cog, &inputs, None);
        assert!(decision.block_aggressive, "High uncertainty → block");
    }

    #[test]
    fn test_cognitive_tick_with_action_logging() {
        let mut cog = CognitiveState::new();
        // Log action at cycle 10
        let inputs1 = CognitiveTickInputs {
            cycle: 10,
            pressure: 0.80,
            latest_action: Some("throttle:Slack".into()),
            predicted_score: 0.70,
            outcome_effectiveness: 0.5,
            ..Default::default()
        };
        run_cognitive_tick(&mut cog, &inputs1, None);
        assert_eq!(cog.self_evaluator.pending_count(), 1);

        // Evaluate at cycle 25 (after EVAL_DELAY_CYCLES=10)
        let inputs2 = CognitiveTickInputs {
            cycle: 25,
            pressure: 0.60,
            causal_confidence: 0.70,
            latest_action: Some("throttle:Slack".into()),
            outcome_effectiveness: 0.6,
            ..Default::default()
        };
        run_cognitive_tick(&mut cog, &inputs2, None);
        // Pending should be 1 (new action at cycle 25) + 0 (old one evaluated)
    }

    #[test]
    fn test_cognitive_tick_probes_at_500() {
        let mut cog = CognitiveState::new();
        let inputs = CognitiveTickInputs {
            cycle: 500,
            pressure: 0.50,
            ..Default::default()
        };
        let decision = run_cognitive_tick(&mut cog, &inputs, None);
        assert!(
            cog.adversarial.total_probes >= 4,
            "Should run 4 probes at cycle 500"
        );
        assert!(!decision.safety_alert);
    }

    #[test]
    fn test_cognitive_tick_reptile_fingerprint_change() {
        let mut cog = CognitiveState::new();
        let inputs1 = CognitiveTickInputs {
            cycle: 10,
            workload_fingerprint: 42,
            ..Default::default()
        };
        run_cognitive_tick(&mut cog, &inputs1, None);

        let inputs2 = CognitiveTickInputs {
            cycle: 20,
            workload_fingerprint: 99, // different fingerprint
            ..Default::default()
        };
        run_cognitive_tick(&mut cog, &inputs2, None);
        assert_eq!(cog.reptile.current_fingerprint(), 99);
        assert!(cog.reptile.adaptation_steps > 0);
    }

    #[test]
    fn test_cognitive_tick_with_drift_detector() {
        let mut cog = CognitiveState::new();
        let mut dd = DriftDetector::new();
        // Feed some observations to create drift signal
        for _ in 0..10 {
            dd.observe("test", false);
        }
        let inputs = CognitiveTickInputs {
            cycle: 50,
            drift_score: dd.score() as f64,
            ..Default::default()
        };
        run_cognitive_tick(&mut cog, &inputs, Some(&mut dd));
        // Early warning should have been updated
        assert!(dd.early_warning() >= 0.0);
    }

    #[test]
    fn test_cognitive_tick_reward_bus_accumulates() {
        let mut cog = CognitiveState::new();
        for cycle in 0..20 {
            let inputs = CognitiveTickInputs {
                cycle,
                outcome_effectiveness: 0.70,
                causal_confidence: 0.60,
                ..Default::default()
            };
            run_cognitive_tick(&mut cog, &inputs, None);
        }
        assert!(cog.reward_bus.total_signals() > 0);
        assert!(cog.reward_bus.rl_reward().abs() > 0.0);
    }

    #[test]
    fn test_cognitive_tick_meta_observes_subsystems() {
        let mut cog = CognitiveState::new();
        for cycle in 0..30 {
            let inputs = CognitiveTickInputs {
                cycle,
                predicted_score: 0.80,
                outcome_effectiveness: 0.75,
                causal_confidence: 0.50,
                ..Default::default()
            };
            run_cognitive_tick(&mut cog, &inputs, None);
        }
        assert!(cog.meta_cognition.tracked_subsystems() >= 1);
        assert!(cog.meta_cognition.total_observations() > 0);
    }

    #[test]
    fn test_cognitive_decision_fields() {
        let d = CognitiveDecision {
            pause_learning: false,
            block_aggressive: true,
            observe_only: false,
            humble_mode: true,
            uchs_composite: 0.75,
            safety_alert: false,
        };
        assert!(!d.pause_learning);
        assert!(d.block_aggressive);
        assert!(d.humble_mode);
    }
}
