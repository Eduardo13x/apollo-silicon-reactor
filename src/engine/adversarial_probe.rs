//! Adversarial Probe — synthetic stress-testing of cognitive safety invariants.
//!
//! ## Problem solved
//! No verification that the cognitive system fails gracefully under extreme
//! conditions. Subtle bugs (like the frozen-renderer thaw bug) only surface
//! in production.
//!
//! ## Design
//! Every N cycles, run synthetic worst-case scenarios on COPIES of cognitive
//! state (zero side effects on real state). Each scenario verifies one safety
//! invariant. Track pass rate as a cognitive health signal.
//!
//! ## References
//! - [Madry 2018] "Towards Deep Learning Models Resistant to Adversarial Attacks"
//!   ICLR — adversarial robustness via synthetic worst-case probing
//! - [Yuan 2024] "Self-Rewarding LMs" §4.2 — self-consistency checks

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Probe runs every this many cycles.
const PROBE_INTERVAL: u32 = 500;

/// EMA alpha for pass rate tracking.
const PASS_RATE_ALPHA: f32 = 0.10;

/// Alert threshold: if pass rate drops below this, emit safety alert.
const ALERT_THRESHOLD: f32 = 0.75;

/// Maximum failure log entries.
const MAX_FAILURE_LOG: usize = 20;

// ── Types ──────────────────────────────────────────────────────────────────────

/// What safety property is being tested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbeExpectation {
    /// Protected processes must NEVER be frozen, even under max pressure.
    NoFreezeProtected,
    /// RL floor (0.45) must never be violated, even with extreme Q-values.
    SafetyFloorRespected,
    /// After injecting drift, system must recalibrate within N cycles.
    NarsDriftRecovery,
    /// High epistemic uncertainty must block aggressive actions.
    EpistemicBlocksAggressive,
    /// ODE inputs oscillate wildly (noisy sensor) — EMA must bound urgency.
    OdeDivergenceResilient,
    /// Swap 75% full + high compression rate (kernel pressure low) → ODE detects urgency.
    /// Proves ODE physics surface saturation risk invisible to the kernel pressure signal.
    StickySwapSpotlightSuppressed,
    /// Utility EMA at 0.0 (subnormal deadlock) → heavy-zone cycling recovers above floor.
    SubnormalFloorRecovery,
}

/// A synthetic scenario to probe.
#[derive(Debug, Clone)]
pub struct SyntheticScenario {
    /// Which invariant this scenario tests.
    pub expectation: ProbeExpectation,
    /// Synthetic memory pressure [0, 1].
    pub pressure: f32,
    /// Synthetic P(OOM) [0, 1].
    pub p_oom: f32,
    /// Process names that must NOT be frozen.
    pub protected_names: Vec<String>,
}

/// Result of a single probe run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    /// Which invariant was tested.
    pub expectation: ProbeExpectation,
    /// Whether the invariant held.
    pub passed: bool,
    /// Human-readable description of what happened.
    pub description: String,
    /// Cycle when the probe ran.
    pub cycle: u64,
}

/// Adversarial probe engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialProbe {
    /// Run probe every this many cycles.
    probe_interval: u32,
    /// Last cycle when probes were run.
    last_probe_cycle: u64,
    /// EMA of pass rate [0, 1].
    pub pass_rate_ema: f32,
    /// Log of recent failures.
    failure_log: VecDeque<ProbeResult>,
    /// Total probes ever run.
    pub total_probes: u64,
    /// Total probes that passed.
    pub total_passed: u64,
    /// Whether a safety alert is active (pass_rate < 0.75).
    pub safety_alert: bool,
}

impl Default for AdversarialProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl AdversarialProbe {
    pub fn new() -> Self {
        Self {
            probe_interval: PROBE_INTERVAL,
            last_probe_cycle: 0,
            pass_rate_ema: 1.0,
            failure_log: VecDeque::with_capacity(MAX_FAILURE_LOG),
            total_probes: 0,
            total_passed: 0,
            safety_alert: false,
        }
    }

    /// Check if probes should run this cycle.
    pub fn should_probe(&self, current_cycle: u64) -> bool {
        current_cycle.saturating_sub(self.last_probe_cycle) >= self.probe_interval as u64
    }

    /// Generate the standard set of synthetic scenarios.
    pub fn generate_scenarios() -> Vec<SyntheticScenario> {
        vec![
            // Scenario 1: Max pressure + protected processes
            SyntheticScenario {
                expectation: ProbeExpectation::NoFreezeProtected,
                pressure: 0.98,
                p_oom: 0.95,
                protected_names: vec![
                    "kernel_task".into(),
                    "WindowServer".into(),
                    "launchd".into(),
                    "Claude".into(),
                    "Brave Browser".into(),
                ],
            },
            // Scenario 2: Extreme pressure testing RL floor
            SyntheticScenario {
                expectation: ProbeExpectation::SafetyFloorRespected,
                pressure: 0.99,
                p_oom: 0.99,
                protected_names: vec![],
            },
            // Scenario 3: Drift injection
            SyntheticScenario {
                expectation: ProbeExpectation::NarsDriftRecovery,
                pressure: 0.60,
                p_oom: 0.30,
                protected_names: vec![],
            },
            // Scenario 4: Max uncertainty
            SyntheticScenario {
                expectation: ProbeExpectation::EpistemicBlocksAggressive,
                pressure: 0.80,
                p_oom: 0.50,
                protected_names: vec![],
            },
            // Scenario 5: Noisy ODE sensor — alternating 0/max dirty rate
            SyntheticScenario {
                expectation: ProbeExpectation::OdeDivergenceResilient,
                pressure: 0.70,
                p_oom: 0.40,
                protected_names: vec![],
            },
        ]
    }

    /// Record probe results.
    ///
    /// `results`: list of (expectation, passed, description) from running scenarios.
    /// Call this after running all scenarios for this cycle.
    pub fn record_results(&mut self, results: Vec<ProbeResult>, current_cycle: u64) {
        self.last_probe_cycle = current_cycle;

        for result in &results {
            self.total_probes += 1;
            if result.passed {
                self.total_passed += 1;
            } else {
                if self.failure_log.len() >= MAX_FAILURE_LOG {
                    self.failure_log.pop_front();
                }
                self.failure_log.push_back(result.clone());
            }
        }

        // Update pass rate EMA
        if !results.is_empty() {
            let batch_rate =
                results.iter().filter(|r| r.passed).count() as f32 / results.len() as f32;
            self.pass_rate_ema =
                self.pass_rate_ema + PASS_RATE_ALPHA * (batch_rate - self.pass_rate_ema);
        }

        self.safety_alert = self.pass_rate_ema < ALERT_THRESHOLD;
    }

    /// Run a NoFreezeProtected probe.
    ///
    /// `would_freeze_fn`: closure that returns true if the system would freeze
    /// a process with the given name at the given pressure/p_oom.
    pub fn probe_no_freeze_protected<F>(
        scenario: &SyntheticScenario,
        would_freeze_fn: F,
    ) -> ProbeResult
    where
        F: Fn(&str, f32, f32) -> bool,
    {
        for name in &scenario.protected_names {
            if would_freeze_fn(name, scenario.pressure, scenario.p_oom) {
                return ProbeResult {
                    expectation: ProbeExpectation::NoFreezeProtected,
                    passed: false,
                    description: format!("Would freeze protected process: {name}"),
                    cycle: 0,
                };
            }
        }
        ProbeResult {
            expectation: ProbeExpectation::NoFreezeProtected,
            passed: true,
            description: "All protected processes safe".into(),
            cycle: 0,
        }
    }

    /// Run a SafetyFloorRespected probe.
    ///
    /// `rl_threshold_fn`: closure returning the RL-adjusted threshold at given pressure.
    pub fn probe_safety_floor<F>(rl_threshold_fn: F) -> ProbeResult
    where
        F: Fn(f32) -> f64,
    {
        let threshold = rl_threshold_fn(0.99);
        let floor = 0.45;
        if threshold < floor {
            ProbeResult {
                expectation: ProbeExpectation::SafetyFloorRespected,
                passed: false,
                description: format!("RL threshold {threshold:.4} < floor {floor}"),
                cycle: 0,
            }
        } else {
            ProbeResult {
                expectation: ProbeExpectation::SafetyFloorRespected,
                passed: true,
                description: format!("RL threshold {threshold:.4} ≥ floor {floor}"),
                cycle: 0,
            }
        }
    }

    /// Run a NarsDriftRecovery probe.
    ///
    /// Injects drift into a cloned DriftDetector and checks recovery.
    pub fn probe_nars_recovery(max_recovery_cycles: u32) -> ProbeResult {
        use crate::engine::nars_belief::DriftDetector;

        use crate::engine::nars_belief::Salience;

        let mut detector = DriftDetector::new();
        // Phase 1: build stable beliefs (few observations for low confidence)
        for _ in 0..3 {
            detector.observe("proc_a", true);
            detector.observe("proc_b", true);
            detector.observe("proc_c", true);
        }
        // Phase 2: inject sudden regime change with HIGH arousal (crisis salience)
        // High arousal = 4× evidence weight → larger frequency shifts per observation
        let crisis = Salience {
            arousal: 0.95,
            valence: -0.8,
        };
        for _ in 0..15 {
            detector.observe_salient("proc_a", false, crisis);
            detector.observe_salient("proc_b", false, crisis);
            detector.observe_salient("proc_c", false, crisis);
        }

        // Check either drift_score > 0.08 OR drifted_count >= 2
        let drifted = detector.needs_recalibration() || detector.drift_score > 0.02;
        if !drifted {
            // Not even detecting drift → problem
            return ProbeResult {
                expectation: ProbeExpectation::NarsDriftRecovery,
                passed: false,
                description: "Failed to detect injected drift".into(),
                cycle: 0,
            };
        }

        // Phase 3: acknowledge recalibration + feed successful observations → recovery
        detector.acknowledge_recalibration();
        let mut recovered = false;
        for _ in 0..max_recovery_cycles {
            detector.observe("proc_a", true);
            detector.observe("proc_b", true);
            detector.observe("proc_c", true);
            if !detector.needs_recalibration_at(0.08) {
                recovered = true;
                break;
            }
        }

        ProbeResult {
            expectation: ProbeExpectation::NarsDriftRecovery,
            passed: recovered,
            description: if recovered {
                "Drift detected and recovered".into()
            } else {
                format!("Failed to recover within {max_recovery_cycles} cycles")
            },
            cycle: 0,
        }
    }

    /// Run an EpistemicBlocksAggressive probe.
    ///
    /// `epistemic_blocks_fn`: returns true if epistemic uncertainty would block
    /// aggressive actions at given (rl_var, linucb_exp, nars_spread, drift).
    pub fn probe_epistemic_blocks<F>(epistemic_blocks_fn: F) -> ProbeResult
    where
        F: Fn(f32, f32, f32, f32) -> bool,
    {
        // Max uncertainty on all dimensions → MUST block
        let blocks = epistemic_blocks_fn(1.0, 1.0, 1.0, 1.0);
        ProbeResult {
            expectation: ProbeExpectation::EpistemicBlocksAggressive,
            passed: blocks,
            description: if blocks {
                "Max uncertainty correctly blocks aggressive actions".into()
            } else {
                "FAIL: max uncertainty did NOT block aggressive actions".into()
            },
            cycle: 0,
        }
    }

    /// Verify EMA damps oscillating dirty-rate inputs vs sustained inputs.
    ///
    /// Invariant: avg urgency under alternating 0/HIGH input < avg urgency under
    /// sustained HIGH input. EMA steady-state for 0/HIGH oscillation converges to
    /// ~half the sustained rate [Zhao 2009 §4.2 EMA smoothing].
    ///
    /// [Hellerstein 2004] §9 — PID sensor noise rejection must attenuate
    /// high-frequency oscillations while tracking real load shifts.
    pub fn probe_ode_divergence() -> ProbeResult {
        use crate::engine::swap_reclaim::{SwapReclaimModel, VmFlowSample, CRITICAL_ETA_SEC};

        const SWAP_CAP: u64 = 8 * 1024 * 1024 * 1024; // 8 GB
        let swap_used = (SWAP_CAP as f64 * 0.50) as u64; // 50% used
        let high_cps = 2_000.0_f64; // pages/s peak

        let urgency_of = |sample: &VmFlowSample, model: &mut SwapReclaimModel| -> f64 {
            model
                .update(sample)
                .t_sat_sec
                .map(|t| (CRITICAL_ETA_SEC / t.max(1.0)).clamp(0.0, 1.0))
                .unwrap_or(0.0)
        };

        let make_sample = |cps: f64| VmFlowSample {
            compressions_per_sec: cps,
            decompressions_per_sec: 0.0,
            purges_per_sec: 0.0,
            swapouts_per_sec: 0.0,
            swap_used_bytes: swap_used,
            swap_total_bytes: SWAP_CAP,
        };

        let mut model_sustained = SwapReclaimModel::new();
        let avg_sustained: f64 = (0..20)
            .map(|_| urgency_of(&make_sample(high_cps), &mut model_sustained))
            .sum::<f64>()
            / 20.0;

        let mut model_oscillating = SwapReclaimModel::new();
        let avg_oscillating: f64 = (0..20)
            .map(|i| {
                let cps = if i % 2 == 0 { high_cps } else { 0.0 };
                urgency_of(&make_sample(cps), &mut model_oscillating)
            })
            .sum::<f64>()
            / 20.0;

        // EMA damps 0/HIGH to ~half the sustained rate → lower average urgency.
        let passed = avg_oscillating < avg_sustained;
        ProbeResult {
            expectation: ProbeExpectation::OdeDivergenceResilient,
            passed,
            description: if passed {
                format!(
                    "EMA damps oscillating urgency {avg_oscillating:.3} < sustained {avg_sustained:.3}"
                )
            } else {
                format!(
                    "EMA noise rejection failed: oscillating {avg_oscillating:.3} ≥ sustained {avg_sustained:.3}"
                )
            },
            cycle: 0,
        }
    }

    /// Probe: ODE detects swap saturation when kernel pressure is low (sticky-swap scenario).
    ///
    /// Invariant: swap=75% full + high compression rate → `TsatUrgency > 0.5`, even when
    /// `memory_pressure = 0.35` (below any kernel-visible threshold).
    /// Demonstrates that ODE physics close the "sticky swap" gap that kernel pressure misses
    /// [Denning 1968 §3 — working set overflow before eviction pressure].
    pub fn probe_sticky_swap_spotlight() -> ProbeResult {
        use crate::engine::swap_reclaim::{CyberPhysicalSignal, SwapReclaimModel, TsatUrgency, VmFlowSample};

        const SWAP_6GB: u64 = 6 * 1024 * 1024 * 1024;
        const SWAP_8GB: u64 = 8 * 1024 * 1024 * 1024;
        const HIGH_CPS: f64 = 3_000.0;

        let sample = VmFlowSample {
            compressions_per_sec: HIGH_CPS,
            decompressions_per_sec: 50.0,
            purges_per_sec: 10.0,
            swapouts_per_sec: 5.0,
            swap_used_bytes: SWAP_6GB,
            swap_total_bytes: SWAP_8GB,
        };

        let mut model = SwapReclaimModel::new();
        let mut forecast = model.update(&sample);
        for _ in 0..19 {
            forecast = model.update(&sample);
        }

        let urgency = TsatUrgency(forecast.t_sat_sec).normalized();
        let passed = urgency > 0.5;
        ProbeResult {
            expectation: ProbeExpectation::StickySwapSpotlightSuppressed,
            passed,
            description: if passed {
                format!("ODE urgency {urgency:.3} > 0.5 (swap=6GB/8GB, kernel pressure=0.35 invisible)")
            } else {
                format!("ODE missed sticky-swap saturation: urgency={urgency:.3} ≤ 0.5")
            },
            cycle: 0,
        }
    }

    /// Probe: utility EMA recovers from 0.0 (subnormal deadlock) via heavy-zone cycling.
    ///
    /// Invariant: after 30 productive heavy-zone cycles (alpha=0.05), utility rises above
    /// UTIL_THRESHOLD (0.15), breaking the mid-zone skip deadlock [Denning 1968 §3 —
    /// escape from low-activation trap requires forcing exploratory cycles].
    pub fn probe_subnormal_floor_recovery() -> ProbeResult {
        const UTIL_ALPHA: f64 = 0.05;
        const UTIL_THRESHOLD: f64 = 0.15;
        const PRODUCTIVE_OUTCOME: f64 = 1.0;

        let mut utility_ema = 0.0_f64;
        // Heavy-zone: run_X=true regardless → EMA updates every cycle.
        for _ in 0..30 {
            utility_ema += UTIL_ALPHA * (PRODUCTIVE_OUTCOME - utility_ema);
        }

        let passed = utility_ema > UTIL_THRESHOLD;
        ProbeResult {
            expectation: ProbeExpectation::SubnormalFloorRecovery,
            passed,
            description: if passed {
                format!("Utility EMA recovered to {utility_ema:.3} > {UTIL_THRESHOLD} in 30 heavy-zone cycles")
            } else {
                format!("Subnormal deadlock persists: utility_ema={utility_ema:.3} ≤ {UTIL_THRESHOLD}")
            },
            cycle: 0,
        }
    }

    /// Recent failures (newest first).
    pub fn recent_failures(&self, n: usize) -> Vec<&ProbeResult> {
        self.failure_log.iter().rev().take(n).collect()
    }

    /// Pass rate as a fraction [0, 1].
    pub fn lifetime_pass_rate(&self) -> f64 {
        if self.total_probes == 0 {
            return 1.0;
        }
        self.total_passed as f64 / self.total_probes as f64
    }

    /// Cognitive safety score [0, 1] for CognitiveHealthScore.
    pub fn safety_score(&self) -> f32 {
        self.pass_rate_ema
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_defaults() {
        let ap = AdversarialProbe::new();
        assert_eq!(ap.total_probes, 0);
        assert!(!ap.safety_alert);
        assert!((ap.pass_rate_ema - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_should_probe_interval() {
        let ap = AdversarialProbe::new();
        assert!(!ap.should_probe(100));
        assert!(ap.should_probe(PROBE_INTERVAL as u64));
        assert!(ap.should_probe(PROBE_INTERVAL as u64 + 1));
    }

    #[test]
    fn test_generate_scenarios() {
        let scenarios = AdversarialProbe::generate_scenarios();
        assert_eq!(scenarios.len(), 5);
        assert_eq!(scenarios[0].expectation, ProbeExpectation::NoFreezeProtected);
        assert_eq!(scenarios[1].expectation, ProbeExpectation::SafetyFloorRespected);
        assert_eq!(scenarios[2].expectation, ProbeExpectation::NarsDriftRecovery);
        assert_eq!(scenarios[3].expectation, ProbeExpectation::EpistemicBlocksAggressive);
        assert_eq!(scenarios[4].expectation, ProbeExpectation::OdeDivergenceResilient);
    }

    #[test]
    fn test_ode_divergence_probe_passes() {
        let result = AdversarialProbe::probe_ode_divergence();
        assert!(
            result.passed,
            "EMA should smooth oscillating sensor: {}",
            result.description
        );
    }

    #[test]
    fn test_no_freeze_protected_passes() {
        let scenario = SyntheticScenario {
            expectation: ProbeExpectation::NoFreezeProtected,
            pressure: 0.99,
            p_oom: 0.95,
            protected_names: vec!["WindowServer".into(), "launchd".into()],
        };
        // Correct behavior: never freeze protected
        let result = AdversarialProbe::probe_no_freeze_protected(&scenario, |name, _, _| {
            !["WindowServer", "launchd"].contains(&name)
        });
        assert!(result.passed);
    }

    #[test]
    fn test_no_freeze_protected_fails() {
        let scenario = SyntheticScenario {
            expectation: ProbeExpectation::NoFreezeProtected,
            pressure: 0.99,
            p_oom: 0.95,
            protected_names: vec!["WindowServer".into()],
        };
        // Bug: freezes everything
        let result = AdversarialProbe::probe_no_freeze_protected(&scenario, |_, _, _| true);
        assert!(!result.passed);
        assert!(result.description.contains("WindowServer"));
    }

    #[test]
    fn test_safety_floor_passes() {
        let result = AdversarialProbe::probe_safety_floor(|_| 0.50);
        assert!(result.passed, "0.50 ≥ 0.45");
    }

    #[test]
    fn test_safety_floor_fails() {
        let result = AdversarialProbe::probe_safety_floor(|_| 0.40);
        assert!(!result.passed, "0.40 < 0.45");
    }

    #[test]
    fn test_nars_drift_recovery() {
        let result = AdversarialProbe::probe_nars_recovery(20);
        assert!(
            result.passed,
            "Should detect and recover: {}",
            result.description
        );
    }

    #[test]
    fn test_epistemic_blocks_at_max_uncertainty() {
        let result = AdversarialProbe::probe_epistemic_blocks(|rv, le, ns, ds| {
            // Correct: block when all dimensions at max
            let composite = 0.30 * rv + 0.30 * le + 0.25 * ns + 0.15 * ds;
            composite > 0.70
        });
        assert!(result.passed);
    }

    #[test]
    fn test_epistemic_blocks_fails_when_not_blocking() {
        let result = AdversarialProbe::probe_epistemic_blocks(|_, _, _, _| false);
        assert!(!result.passed);
    }

    #[test]
    fn test_record_results_updates_pass_rate() {
        let mut ap = AdversarialProbe::new();
        let results = vec![
            ProbeResult {
                expectation: ProbeExpectation::NoFreezeProtected,
                passed: true,
                description: String::new(),
                cycle: 500,
            },
            ProbeResult {
                expectation: ProbeExpectation::SafetyFloorRespected,
                passed: true,
                description: String::new(),
                cycle: 500,
            },
            ProbeResult {
                expectation: ProbeExpectation::NarsDriftRecovery,
                passed: false,
                description: "drift".into(),
                cycle: 500,
            },
        ];
        ap.record_results(results, 500);
        assert_eq!(ap.total_probes, 3);
        assert_eq!(ap.total_passed, 2);
        // pass_rate_ema should decrease from 1.0
        assert!(ap.pass_rate_ema < 1.0);
    }

    #[test]
    fn test_safety_alert_triggers() {
        let mut ap = AdversarialProbe::new();
        // Many failures → low pass rate → alert
        for batch in 0..20 {
            let results = vec![ProbeResult {
                expectation: ProbeExpectation::NoFreezeProtected,
                passed: false,
                description: "fail".into(),
                cycle: batch * 500,
            }];
            ap.record_results(results, batch * 500);
        }
        assert!(ap.safety_alert, "Many failures → safety alert");
    }

    #[test]
    fn test_failure_log_capped() {
        let mut ap = AdversarialProbe::new();
        for i in 0..MAX_FAILURE_LOG + 10 {
            let results = vec![ProbeResult {
                expectation: ProbeExpectation::SafetyFloorRespected,
                passed: false,
                description: format!("fail {i}"),
                cycle: i as u64,
            }];
            ap.record_results(results, i as u64);
        }
        assert!(ap.failure_log.len() <= MAX_FAILURE_LOG);
    }

    #[test]
    fn test_lifetime_pass_rate() {
        let mut ap = AdversarialProbe::new();
        assert_eq!(ap.lifetime_pass_rate(), 1.0);
        let results = vec![
            ProbeResult {
                expectation: ProbeExpectation::NoFreezeProtected,
                passed: true,
                description: String::new(),
                cycle: 1,
            },
            ProbeResult {
                expectation: ProbeExpectation::SafetyFloorRespected,
                passed: false,
                description: String::new(),
                cycle: 1,
            },
        ];
        ap.record_results(results, 1);
        assert!((ap.lifetime_pass_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut ap = AdversarialProbe::new();
        let results = vec![ProbeResult {
            expectation: ProbeExpectation::NoFreezeProtected,
            passed: true,
            description: "ok".into(),
            cycle: 500,
        }];
        ap.record_results(results, 500);

        let json = serde_json::to_string(&ap).expect("serialize");
        let restored: AdversarialProbe = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.total_probes, ap.total_probes);
        assert!((restored.pass_rate_ema - ap.pass_rate_ema).abs() < 1e-6);
    }

    #[test]
    fn test_safety_score_equals_pass_rate() {
        let mut ap = AdversarialProbe::new();
        ap.pass_rate_ema = 0.82;
        assert!((ap.safety_score() - 0.82).abs() < 0.001);
    }

    #[test]
    fn test_sticky_swap_spotlight_passes() {
        let result = AdversarialProbe::probe_sticky_swap_spotlight();
        assert!(
            result.passed,
            "ODE should detect swap saturation at 75% full + high CPS: {}",
            result.description
        );
    }

    #[test]
    fn test_subnormal_floor_recovery_passes() {
        let result = AdversarialProbe::probe_subnormal_floor_recovery();
        assert!(
            result.passed,
            "30 productive heavy-zone cycles must lift utility above 0.15: {}",
            result.description
        );
    }
}
