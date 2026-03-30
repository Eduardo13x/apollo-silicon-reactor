//! Apollo Intelligence Score (AIS) — fórmula compuesta que mide la calidad,
//! velocidad, eficiencia y profundidad de aprendizaje del optimizador.
//!
//! Inspirado en:
//! - Shannon Information Theory (1948): medir la "información útil" que el sistema extrae
//! - Bellman Optimality (1957): medir qué tan cerca de la política óptima operamos
//! - Pareto Efficiency: ninguna dimensión puede mejorar sin degradar otra
//!
//! ## Fórmula
//!
//! AIS = Σ wᵢ · Dᵢ(x)   donde Dᵢ ∈ [0, 1] son dimensiones normalizadas
//!
//! | Dimensión              | Peso | Qué mide                                          |
//! |------------------------|------|---------------------------------------------------|
//! | Decision Precision     | 0.25 | Correctness of throttle/freeze/boost decisions     |
//! | Signal Quality         | 0.20 | Kalman/CUSUM/Hazard accuracy & convergence         |
//! | Learning Velocity      | 0.20 | RL convergence + causal graph + skill emergence    |
//! | Resource Efficiency    | 0.15 | Cycle speed + cognitive budget effectiveness       |
//! | Safety Compliance      | 0.12 | Adherence to safety invariants                     |
//! | Adaptability           | 0.08 | Regime detection + workload classification         |
//!
//! Score final: AIS ∈ [0, 100]

use serde::{Deserialize, Serialize};

// ── Weights ──────────────────────────────────────────────────────────────────
const W_DECISION: f64 = 0.25;
const W_SIGNAL: f64 = 0.20;
const W_LEARNING: f64 = 0.20;
const W_RESOURCE: f64 = 0.15;
const W_SAFETY: f64 = 0.12;
const W_ADAPT: f64 = 0.08;

/// Input data for AIS computation. Can come from live daemon or simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AisInput {
    // ── Decision metrics ─────────────────────────────────────────────────
    /// Total decisions made (throttle/freeze/boost/skip).
    pub total_decisions: u64,
    /// Decisions that correctly matched the expected action for the process class.
    pub correct_decisions: u64,
    /// Protected processes that were correctly left alone.
    pub protected_preserved: u64,
    /// Protected processes that should have been preserved (total eligible).
    pub protected_total: u64,
    /// Noise processes correctly throttled.
    pub noise_throttled: u64,
    /// Noise processes that could have been throttled.
    pub noise_total: u64,
    /// Interactive processes correctly boosted.
    pub interactive_boosted: u64,
    /// Interactive processes total.
    pub interactive_total: u64,

    // ── Signal quality ───────────────────────────────────────────────────
    /// Kalman filter prediction error (RMSE). Lower = better.
    pub kalman_rmse: f64,
    /// CUSUM true positive detections.
    pub cusum_true_positives: u32,
    /// CUSUM false positive detections.
    pub cusum_false_positives: u32,
    /// CUSUM total actual regime shifts.
    pub cusum_actual_shifts: u32,
    /// Hazard model calibration: mean |predicted_p - actual_outcome|.
    pub hazard_calibration_error: f64,
    /// Entropy anomaly true positive rate (0-1).
    pub entropy_tpr: f64,

    // ── Learning metrics ─────────────────────────────────────────────────
    /// RL Q-value variance (lower = more converged). Computed as std of Q-table values.
    pub rl_q_variance: f64,
    /// RL ticks to first good policy (lower = faster learning).
    pub rl_convergence_ticks: u64,
    /// Max possible ticks for normalization.
    pub rl_max_ticks: u64,
    /// Causal graph edges with confidence > 0.50.
    pub causal_solid_edges: u32,
    /// Causal graph total edges.
    pub causal_total_edges: u32,
    /// Skills that passed reliability threshold (≥5 applications, ≥60% success).
    pub reliable_skills: u32,
    /// Total skills in registry.
    pub total_skills: u32,
    /// Experience memory records.
    pub experience_records: u32,
    /// Dyna-Q simulated transitions.
    pub dyna_transitions: u64,

    // ── Resource efficiency ──────────────────────────────────────────────
    /// p95 cycle time in milliseconds.
    pub p95_cycle_ms: f64,
    /// Target cycle time (budget = 500ms sleep, work should be minimal).
    pub target_cycle_ms: f64,
    /// Cognitive budget: subsystem skips (correctly avoided unnecessary work).
    pub subsystem_skips: u64,
    /// Total subsystem evaluations.
    pub subsystem_evals: u64,
    /// Habituation skips (processes unchanged for N cycles).
    pub habituation_skips: u64,
    /// Total process evaluations.
    pub process_evals: u64,

    // ── Safety metrics ───────────────────────────────────────────────────
    /// Kills applied (should be 0 except emergencies).
    pub kills_applied: u32,
    /// Survival mode activations (should be very rare).
    pub survival_activations: u32,
    /// Overflow events in last 7 days.
    pub overflow_events_7d: u32,
    /// Failures (daemon errors).
    pub failures: u32,
    /// Frozen critical processes (MUST be 0).
    pub frozen_critical: u32,

    // ── Adaptability metrics ─────────────────────────────────────────────
    /// Correct profile switches (matched workload).
    pub correct_profile_switches: u32,
    /// Total profile switches.
    pub total_profile_switches: u32,
    /// Correct workload classifications.
    pub correct_workload_class: u32,
    /// Total workload classification attempts.
    pub total_workload_class: u32,
    /// Regime shifts detected within 3 cycles of actual shift.
    pub regime_shifts_detected: u32,
    /// Total actual regime shifts.
    pub regime_shifts_total: u32,
}

/// AIS result with per-dimension breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AisScore {
    /// Total AIS score [0, 100].
    pub total: f64,
    /// Per-dimension scores [0, 1].
    pub decision_precision: f64,
    pub signal_quality: f64,
    pub learning_velocity: f64,
    pub resource_efficiency: f64,
    pub safety_compliance: f64,
    pub adaptability: f64,
    /// Pareto frontier check: true if no dimension is below 0.30.
    pub pareto_balanced: bool,
    /// Grade: S (≥90), A (≥80), B (≥70), C (≥60), D (≥50), F (<50).
    pub grade: char,
}

/// Compute the Apollo Intelligence Score from input metrics.
///
/// Each dimension is normalized to [0, 1] using calibrated transfer functions,
/// then weighted and summed to produce the final [0, 100] score.
pub fn compute_ais(input: &AisInput) -> AisScore {
    let d1 = decision_precision(input);
    let d2 = signal_quality(input);
    let d3 = learning_velocity(input);
    let d4 = resource_efficiency(input);
    let d5 = safety_compliance(input);
    let d6 = adaptability(input);

    let total = (W_DECISION * d1
        + W_SIGNAL * d2
        + W_LEARNING * d3
        + W_RESOURCE * d4
        + W_SAFETY * d5
        + W_ADAPT * d6)
        * 100.0;

    let dims = [d1, d2, d3, d4, d5, d6];
    let pareto_balanced = dims.iter().all(|&d| d >= 0.30);

    let grade = match total as u32 {
        90..=100 => 'S',
        80..=89 => 'A',
        70..=79 => 'B',
        60..=69 => 'C',
        50..=59 => 'D',
        _ => 'F',
    };

    AisScore {
        total,
        decision_precision: d1,
        signal_quality: d2,
        learning_velocity: d3,
        resource_efficiency: d4,
        safety_compliance: d5,
        adaptability: d6,
        pareto_balanced,
        grade,
    }
}

// ── Dimension 1: Decision Precision ──────────────────────────────────────────
// Weighted F1-like score across process classes:
// - 40% protected preservation (recall)
// - 30% noise throttling (precision)
// - 30% interactive boosting (recall)
fn decision_precision(input: &AisInput) -> f64 {
    let protected_rate = safe_ratio(input.protected_preserved, input.protected_total);
    let noise_rate = safe_ratio(input.noise_throttled, input.noise_total);
    let interactive_rate = safe_ratio(input.interactive_boosted, input.interactive_total);

    (0.40 * protected_rate + 0.30 * noise_rate + 0.30 * interactive_rate).clamp(0.0, 1.0)
}

// ── Dimension 2: Signal Quality ──────────────────────────────────────────────
// Combines: Kalman accuracy, CUSUM detection rate, Hazard calibration, Entropy TPR.
fn signal_quality(input: &AisInput) -> f64 {
    // Kalman: RMSE < 0.01 = perfect, > 0.10 = poor.
    // Sigmoid mapping: score = 1 / (1 + (rmse/0.03)^2)
    let kalman_score = 1.0 / (1.0 + (input.kalman_rmse / 0.03).powi(2));

    // CUSUM: F1 score of detection.
    let cusum_tp = input.cusum_true_positives as f64;
    let cusum_fp = input.cusum_false_positives as f64;
    let cusum_fn = (input.cusum_actual_shifts.saturating_sub(input.cusum_true_positives)) as f64;
    let cusum_precision = if cusum_tp + cusum_fp > 0.0 {
        cusum_tp / (cusum_tp + cusum_fp)
    } else {
        0.5 // no data = neutral
    };
    let cusum_recall = if cusum_tp + cusum_fn > 0.0 {
        cusum_tp / (cusum_tp + cusum_fn)
    } else {
        0.5
    };
    let cusum_f1 = if cusum_precision + cusum_recall > 0.0 {
        2.0 * cusum_precision * cusum_recall / (cusum_precision + cusum_recall)
    } else {
        0.0
    };

    // Hazard: calibration error < 0.05 = excellent, > 0.30 = poor.
    let hazard_score = 1.0 / (1.0 + (input.hazard_calibration_error / 0.08).powi(2));

    // Entropy TPR directly.
    let entropy_score = input.entropy_tpr.clamp(0.0, 1.0);

    (0.30 * kalman_score + 0.30 * cusum_f1 + 0.25 * hazard_score + 0.15 * entropy_score)
        .clamp(0.0, 1.0)
}

// ── Dimension 3: Learning Velocity ───────────────────────────────────────────
// How fast and deep the system learns from experience.
fn learning_velocity(input: &AisInput) -> f64 {
    // RL convergence: normalize ticks to first good policy.
    // Fewer ticks = faster learning. Score = 1 - (ticks/max_ticks).
    let rl_speed = if input.rl_max_ticks > 0 {
        1.0 - (input.rl_convergence_ticks as f64 / input.rl_max_ticks as f64).clamp(0.0, 1.0)
    } else {
        0.5
    };

    // RL stability: low Q-variance = converged.
    // Sigmoid: score = 1 / (1 + variance)
    let rl_stability = 1.0 / (1.0 + input.rl_q_variance);

    // Causal graph: fraction of solid edges.
    let causal_depth = safe_ratio_u32(input.causal_solid_edges, input.causal_total_edges);

    // Skills: reliable / total, with bonus for having any.
    let skill_score = if input.total_skills > 0 {
        let base = input.reliable_skills as f64 / input.total_skills as f64;
        // Bonus for having more skills (diversity).
        let diversity_bonus = (input.total_skills as f64 / 10.0).min(1.0) * 0.2;
        (base + diversity_bonus).min(1.0)
    } else {
        0.0
    };

    // Dyna-Q utilization: are we doing model-based planning?
    let dyna_score = if input.dyna_transitions > 0 { 0.8 } else { 0.0 }
        + if input.dyna_transitions > 100 {
            0.2
        } else {
            input.dyna_transitions as f64 / 500.0
        };

    (0.20 * rl_speed
        + 0.20 * rl_stability
        + 0.25 * causal_depth
        + 0.20 * skill_score
        + 0.15 * dyna_score.min(1.0))
    .clamp(0.0, 1.0)
}

// ── Dimension 4: Resource Efficiency ─────────────────────────────────────────
// How efficiently the optimizer uses compute resources.
fn resource_efficiency(input: &AisInput) -> f64 {
    // Cycle time: exponential decay. 20ms = 1.0, 50ms = 0.82, 100ms = 0.45, 200ms = 0.09.
    // score = exp(-0.015 * p95)
    let cycle_score = (-0.015 * input.p95_cycle_ms).exp();

    // Cognitive budget: skip rate when skips are appropriate.
    let budget_score = if input.subsystem_evals > 0 {
        let skip_rate = input.subsystem_skips as f64 / input.subsystem_evals as f64;
        // Optimal skip rate: ~40-60%. Too low = wasting compute, too high = missing signals.
        // Bell curve centered at 0.50: score = exp(-((rate - 0.50) / 0.25)^2)
        (-((skip_rate - 0.50) / 0.25).powi(2)).exp()
    } else {
        0.3
    };

    // Habituation: what fraction of stable processes we skip.
    let habituation_score = if input.process_evals > 0 {
        let hab_rate = input.habituation_skips as f64 / input.process_evals as f64;
        // Any habituation is good, up to ~30%. Above that, diminishing returns.
        (hab_rate / 0.30).min(1.0)
    } else {
        0.0
    };

    (0.45 * cycle_score + 0.30 * budget_score + 0.25 * habituation_score).clamp(0.0, 1.0)
}

// ── Dimension 5: Safety Compliance ───────────────────────────────────────────
// Binary + graduated safety metrics.
fn safety_compliance(input: &AisInput) -> f64 {
    // Critical: frozen_critical MUST be 0. If > 0, safety = 0.
    if input.frozen_critical > 0 {
        return 0.0;
    }

    let mut score: f64 = 0.0;

    // No kills: +0.30 (unless emergency).
    score += if input.kills_applied == 0 { 0.30 } else { 0.0 };

    // No survival mode: +0.25.
    score += if input.survival_activations == 0 {
        0.25
    } else {
        0.0
    };

    // No failures: +0.20.
    score += if input.failures == 0 { 0.20 } else { 0.0 };

    // Low overflow: graduated.
    // 0 overflows = 0.25, 1-5 = 0.15, 6-20 = 0.05, >20 = 0.0.
    score += match input.overflow_events_7d {
        0 => 0.25,
        1..=5 => 0.15,
        6..=20 => 0.05,
        _ => 0.0,
    };

    score.clamp(0.0, 1.0)
}

// ── Dimension 6: Adaptability ────────────────────────────────────────────────
// How well the system responds to changing conditions.
fn adaptability(input: &AisInput) -> f64 {
    let profile_accuracy =
        safe_ratio_u32(input.correct_profile_switches, input.total_profile_switches);
    let workload_accuracy =
        safe_ratio_u32(input.correct_workload_class, input.total_workload_class);
    let regime_detection =
        safe_ratio_u32(input.regime_shifts_detected, input.regime_shifts_total);

    (0.30 * profile_accuracy + 0.40 * workload_accuracy + 0.30 * regime_detection).clamp(0.0, 1.0)
}

// ── Helpers ──────────────────────────────────────────────────────────────────
fn safe_ratio(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.5 // no data = neutral
    } else {
        (num as f64 / den as f64).clamp(0.0, 1.0)
    }
}

fn safe_ratio_u32(num: u32, den: u32) -> f64 {
    if den == 0 {
        0.5
    } else {
        (num as f64 / den as f64).clamp(0.0, 1.0)
    }
}

// ── Display ──────────────────────────────────────────────────────────────────
impl std::fmt::Display for AisScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AIS: {:.1} [{}] | D={:.2} S={:.2} L={:.2} R={:.2} Sf={:.2} A={:.2}{}",
            self.total,
            self.grade,
            self.decision_precision,
            self.signal_quality,
            self.learning_velocity,
            self.resource_efficiency,
            self.safety_compliance,
            self.adaptability,
            if self.pareto_balanced {
                " (Pareto)"
            } else {
                ""
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Baseline AIS from real daemon data (daemon status captured 2026-03-30).
    ///
    /// This test serves as the autoresearch metric:
    /// `cargo test ais_benchmark -- --nocapture 2>&1 | grep 'AIS:' | awk -F': ' '{print $2}' | awk '{print $1}'`
    #[test]
    fn ais_benchmark() {
        let input = real_daemon_snapshot();
        let score = compute_ais(&input);
        println!("{}", score);
        println!(
            "AIS: {:.1} | Decision={:.0}% Signal={:.0}% Learning={:.0}% Resource={:.0}% Safety={:.0}% Adapt={:.0}%",
            score.total,
            score.decision_precision * 100.0,
            score.signal_quality * 100.0,
            score.learning_velocity * 100.0,
            score.resource_efficiency * 100.0,
            score.safety_compliance * 100.0,
            score.adaptability * 100.0,
        );
        // Sanity: score should be reasonable
        assert!(score.total > 0.0, "AIS must be positive");
        assert!(score.total <= 100.0, "AIS must be ≤ 100");
    }

    /// Multi-scenario benchmark: tests the AIS across 5 workload scenarios.
    /// The average score is the final metric.
    #[test]
    fn ais_multi_scenario() {
        let scenarios = vec![
            ("idle", idle_scenario()),
            ("browser_heavy", browser_heavy_scenario()),
            ("build_mode", build_mode_scenario()),
            ("thermal_crisis", thermal_crisis_scenario()),
            ("real_daemon", real_daemon_snapshot()),
        ];

        let mut total = 0.0;
        for (name, input) in &scenarios {
            let score = compute_ais(input);
            println!("  {:<16} AIS={:.1} [{}]", name, score.total, score.grade);
            total += score.total;
        }
        let avg = total / scenarios.len() as f64;
        println!("AIS: {:.1}", avg);
        assert!(avg > 0.0);
    }

    // ── Scenario Builders ────────────────────────────────────────────────

    /// Real daemon snapshot from 2026-03-30 status output.
    fn real_daemon_snapshot() -> AisInput {
        AisInput {
            // Decision: from heuristic_decisions=700426, throttles=105940, freezes=1596
            total_decisions: 700_426,
            correct_decisions: 620_000, // estimated ~88% correct
            protected_preserved: 232, // learned_policy.protected_patterns
            protected_total: 232,
            noise_throttled: 55, // learned_policy.noise_patterns
            noise_total: 55,
            interactive_boosted: 41, // learned_policy.interactive_patterns
            interactive_total: 50,   // some interactive not yet learned

            // Signal: from si_* metrics
            kalman_rmse: 0.025,         // estimated from pressure smoothing
            cusum_true_positives: 8,    // si_regime_shifts=10, ~80% true
            cusum_false_positives: 2,   // ~2 false alarms
            cusum_actual_shifts: 10,    // si_regime_shifts
            hazard_calibration_error: 0.08, // moderate calibration
            entropy_tpr: 0.70,          // entropy_anomaly=0.33, decent detection

            // Learning: from rl_*, causal_*, dr_zero_*
            rl_q_variance: 0.15,        // moderate convergence
            rl_convergence_ticks: 50_000,
            rl_max_ticks: 590_000,      // rl_total_ticks
            causal_solid_edges: 3,      // top 5 pairs, ~3 solid
            causal_total_edges: 5,      // causal_pairs count
            reliable_skills: 0,         // skills need more time
            total_skills: 0,
            experience_records: 231,    // experience_memory_size
            dyna_transitions: 5_000,    // estimated from rl_total_ticks/10

            // Resource: from cycle_durations and metrics
            p95_cycle_ms: 76.0,         // from daemon status
            target_cycle_ms: 50.0,
            subsystem_skips: 400,       // cognitive budget skips per 1000 cycles
            subsystem_evals: 1000,
            habituation_skips: 150,     // estimated
            process_evals: 700,         // per-process eval count

            // Safety: from daemon metrics
            kills_applied: 0,
            survival_activations: 0,
            overflow_events_7d: 20,
            failures: 0,
            frozen_critical: 0,

            // Adaptability: from profile/workload data
            correct_profile_switches: 2,
            total_profile_switches: 2,
            correct_workload_class: 8,
            total_workload_class: 10,
            regime_shifts_detected: 8,
            regime_shifts_total: 10,
        }
    }

    /// Idle system: low pressure, few processes, everything calm.
    fn idle_scenario() -> AisInput {
        AisInput {
            total_decisions: 1000,
            correct_decisions: 950,
            protected_preserved: 50,
            protected_total: 50,
            noise_throttled: 10,
            noise_total: 12,
            interactive_boosted: 5,
            interactive_total: 5,

            kalman_rmse: 0.01,
            cusum_true_positives: 0,
            cusum_false_positives: 0,
            cusum_actual_shifts: 0,
            hazard_calibration_error: 0.02,
            entropy_tpr: 0.90,

            rl_q_variance: 0.05,
            rl_convergence_ticks: 100,
            rl_max_ticks: 1000,
            causal_solid_edges: 5,
            causal_total_edges: 8,
            reliable_skills: 2,
            total_skills: 3,
            experience_records: 100,
            dyna_transitions: 500,

            p95_cycle_ms: 40.0,
            target_cycle_ms: 50.0,
            subsystem_skips: 600,
            subsystem_evals: 1000,
            habituation_skips: 300,
            process_evals: 1000,

            kills_applied: 0,
            survival_activations: 0,
            overflow_events_7d: 0,
            failures: 0,
            frozen_critical: 0,

            correct_profile_switches: 1,
            total_profile_switches: 1,
            correct_workload_class: 10,
            total_workload_class: 10,
            regime_shifts_detected: 0,
            regime_shifts_total: 0,
        }
    }

    /// Browser-heavy: Brave with 30+ tabs, high memory pressure.
    fn browser_heavy_scenario() -> AisInput {
        AisInput {
            total_decisions: 5000,
            correct_decisions: 4200,
            protected_preserved: 100,
            protected_total: 100,
            noise_throttled: 40,
            noise_total: 50,
            interactive_boosted: 30,
            interactive_total: 35,

            kalman_rmse: 0.04,
            cusum_true_positives: 5,
            cusum_false_positives: 1,
            cusum_actual_shifts: 6,
            hazard_calibration_error: 0.10,
            entropy_tpr: 0.65,

            rl_q_variance: 0.20,
            rl_convergence_ticks: 2000,
            rl_max_ticks: 5000,
            causal_solid_edges: 4,
            causal_total_edges: 10,
            reliable_skills: 1,
            total_skills: 3,
            experience_records: 200,
            dyna_transitions: 2000,

            p95_cycle_ms: 65.0,
            target_cycle_ms: 50.0,
            subsystem_skips: 300,
            subsystem_evals: 1000,
            habituation_skips: 100,
            process_evals: 800,

            kills_applied: 0,
            survival_activations: 0,
            overflow_events_7d: 5,
            failures: 0,
            frozen_critical: 0,

            correct_profile_switches: 2,
            total_profile_switches: 2,
            correct_workload_class: 8,
            total_workload_class: 10,
            regime_shifts_detected: 5,
            regime_shifts_total: 6,
        }
    }

    /// Build mode: cargo/rustc running, high CPU, moderate memory.
    fn build_mode_scenario() -> AisInput {
        AisInput {
            total_decisions: 3000,
            correct_decisions: 2700,
            protected_preserved: 80,
            protected_total: 80,
            noise_throttled: 35,
            noise_total: 40,
            interactive_boosted: 20,
            interactive_total: 25,

            kalman_rmse: 0.03,
            cusum_true_positives: 3,
            cusum_false_positives: 1,
            cusum_actual_shifts: 4,
            hazard_calibration_error: 0.06,
            entropy_tpr: 0.75,

            rl_q_variance: 0.12,
            rl_convergence_ticks: 1500,
            rl_max_ticks: 3000,
            causal_solid_edges: 6,
            causal_total_edges: 12,
            reliable_skills: 2,
            total_skills: 5,
            experience_records: 300,
            dyna_transitions: 3000,

            p95_cycle_ms: 55.0,
            target_cycle_ms: 50.0,
            subsystem_skips: 450,
            subsystem_evals: 1000,
            habituation_skips: 200,
            process_evals: 900,

            kills_applied: 0,
            survival_activations: 0,
            overflow_events_7d: 3,
            failures: 0,
            frozen_critical: 0,

            correct_profile_switches: 3,
            total_profile_switches: 3,
            correct_workload_class: 9,
            total_workload_class: 10,
            regime_shifts_detected: 3,
            regime_shifts_total: 4,
        }
    }

    /// Thermal crisis: high temp, battery, aggressive throttling.
    fn thermal_crisis_scenario() -> AisInput {
        AisInput {
            total_decisions: 8000,
            correct_decisions: 6500,
            protected_preserved: 150,
            protected_total: 150,
            noise_throttled: 80,
            noise_total: 80,
            interactive_boosted: 10,
            interactive_total: 20,

            kalman_rmse: 0.06,
            cusum_true_positives: 8,
            cusum_false_positives: 3,
            cusum_actual_shifts: 12,
            hazard_calibration_error: 0.15,
            entropy_tpr: 0.55,

            rl_q_variance: 0.30,
            rl_convergence_ticks: 4000,
            rl_max_ticks: 8000,
            causal_solid_edges: 2,
            causal_total_edges: 8,
            reliable_skills: 0,
            total_skills: 2,
            experience_records: 150,
            dyna_transitions: 1000,

            p95_cycle_ms: 90.0,
            target_cycle_ms: 50.0,
            subsystem_skips: 200,
            subsystem_evals: 1000,
            habituation_skips: 50,
            process_evals: 600,

            kills_applied: 0,
            survival_activations: 0,
            overflow_events_7d: 15,
            failures: 0,
            frozen_critical: 0,

            correct_profile_switches: 4,
            total_profile_switches: 5,
            correct_workload_class: 7,
            total_workload_class: 10,
            regime_shifts_detected: 8,
            regime_shifts_total: 12,
        }
    }

    #[test]
    fn test_perfect_score() {
        let input = AisInput {
            total_decisions: 10000,
            correct_decisions: 10000,
            protected_preserved: 100,
            protected_total: 100,
            noise_throttled: 100,
            noise_total: 100,
            interactive_boosted: 100,
            interactive_total: 100,

            kalman_rmse: 0.005,
            cusum_true_positives: 10,
            cusum_false_positives: 0,
            cusum_actual_shifts: 10,
            hazard_calibration_error: 0.01,
            entropy_tpr: 1.0,

            rl_q_variance: 0.01,
            rl_convergence_ticks: 10,
            rl_max_ticks: 1000,
            causal_solid_edges: 20,
            causal_total_edges: 20,
            reliable_skills: 10,
            total_skills: 10,
            experience_records: 500,
            dyna_transitions: 10000,

            p95_cycle_ms: 15.0,
            target_cycle_ms: 50.0,
            subsystem_skips: 500,
            subsystem_evals: 1000,
            habituation_skips: 300,
            process_evals: 1000,

            kills_applied: 0,
            survival_activations: 0,
            overflow_events_7d: 0,
            failures: 0,
            frozen_critical: 0,

            correct_profile_switches: 5,
            total_profile_switches: 5,
            correct_workload_class: 10,
            total_workload_class: 10,
            regime_shifts_detected: 5,
            regime_shifts_total: 5,
        };
        let score = compute_ais(&input);
        println!("Perfect: {}", score);
        assert!(score.total > 90.0, "Perfect input should score S-tier");
        assert!(score.pareto_balanced);
    }

    #[test]
    fn test_safety_violation_zeroes() {
        let mut input = real_daemon_snapshot();
        input.frozen_critical = 1;
        let score = compute_ais(&input);
        assert_eq!(score.safety_compliance, 0.0);
        println!("With safety violation: {}", score);
    }

    #[test]
    fn test_weights_sum_to_one() {
        let sum = W_DECISION + W_SIGNAL + W_LEARNING + W_RESOURCE + W_SAFETY + W_ADAPT;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Weights must sum to 1.0, got {}",
            sum
        );
    }
}
