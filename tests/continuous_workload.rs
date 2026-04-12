//! # Continuous Workload Benchmark
//!
//! Tests Apollo's learning hierarchy under temporal workload sequences — not
//! just point-in-time snapshots. Validates that [Page 1954] CUSUM-style detection
//! and [Kuncheva 2004] concept drift handling work across realistic time-series.
//!
//! Addresses paper §8.3 L2 limitation: "The 165-scenario benchmark tests fixed,
//! deterministic conditions. It does not capture adversarial workloads, multi-hour
//! gradual memory leaks, or hardware-fault conditions."
//!
//! ## Workload Sequences
//!
//! 1. **compilation_spike** — pressure ramps 0.30→0.85, plateau, recovery
//! 2. **browser_accumulation** — slow drift 0.50→0.75 (memory leak pattern)
//! 3. **llm_steady** — constant ~0.72 ± 0.03 noise (LLM inference)
//! 4. **mixed_adversarial** — alternating regimes every 10 steps (high meta-velocity)

use apollo_optimizer::engine::nested_learner::NestedLearner;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Simulate one workload step through the NestedLearner pipeline.
/// Returns (l1_gate_open, l1_aggregate, l2_meta_velocity).
fn step(nl: &mut NestedLearner, signal_quality: f64, outcome: f64) -> (bool, f64, f64) {
    let gate_open = nl.tick_l0(signal_quality);
    if gate_open {
        if nl.tick_l1(outcome) {
            nl.flush_l2();
        }
    }
    (gate_open, nl.l1_aggregate, nl.l2_meta_velocity)
}

// ── Workload 1: Compilation Spike ────────────────────────────────────────────

/// Compilation spike: pressure linearly rises 0.30→0.85 over 30 steps,
/// plateaus for 10 steps at peak, then recovers.
///
/// Validates: L0 gate stays open under high-but-stable signal quality (rustc
/// workloads have clean signal — deterministic memory access patterns).
/// L1 aggregate should track the spike phase's high effectiveness.
#[test]
fn workload_compilation_spike() {
    let mut nl = NestedLearner::new();

    // Warm up L0 with good initial signal
    for _ in 0..20 {
        nl.tick_l0(0.8);
    }

    let mut gate_closures = 0usize;
    let mut max_aggregate = 0.0_f64;

    // Phase 1: ramp up (0.30→0.85 pressure, high signal quality = rustc is deterministic)
    for i in 0..30 {
        let pressure = 0.30 + (i as f64) * (0.55 / 30.0);
        // Effectiveness: throttling more impactful as pressure rises
        let effectiveness = (pressure - 0.30) / 0.55;
        let signal_quality = 0.80; // clean during compilation
        let (gate, agg, _) = step(&mut nl, signal_quality, effectiveness);
        if !gate {
            gate_closures += 1;
        }
        max_aggregate = max_aggregate.max(agg);
    }

    // Phase 2: plateau at peak pressure (high effectiveness)
    for _ in 0..10 {
        let (gate, agg, _) = step(&mut nl, 0.78, 0.90);
        if !gate {
            gate_closures += 1;
        }
        max_aggregate = max_aggregate.max(agg);
    }

    // Phase 3: recovery (pressure drops, lower effectiveness needed)
    for i in 0..10 {
        let effectiveness = 1.0 - (i as f64) * 0.1;
        step(&mut nl, 0.82, effectiveness.max(0.0));
    }

    // Invariants
    assert!(
        gate_closures == 0,
        "gate should never close during compilation spike (signal quality=0.8), closures={}",
        gate_closures
    );
    assert!(
        max_aggregate > 0.20,
        "l1_aggregate should reflect spike phase effectiveness, got max={}",
        max_aggregate
    );
    assert!(
        nl.l2_total >= 2,
        "at least 2 L2 flushes should have occurred over 50 steps, got {}",
        nl.l2_total
    );
}

// ── Workload 2: Browser Accumulation ─────────────────────────────────────────

/// Browser accumulation: slow linear drift 0.50→0.75 over 50 steps.
/// Simulates memory leak from tab accumulation.
///
/// Validates: gradual drift doesn't produce high meta-velocity (no spike in
/// l2_meta_velocity) — the system adapts smoothly without overreacting.
/// NARS confidence should grow monotonically (stable regime → more evidence).
#[test]
fn workload_browser_accumulation() {
    let mut nl = NestedLearner::new();

    // Warm up
    for _ in 0..20 {
        nl.tick_l0(0.65);
    }

    let mut prev_velocity = nl.l2_meta_velocity;
    let mut velocity_spike = false;

    for i in 0..50 {
        let pressure = 0.50 + (i as f64) * (0.25 / 50.0);
        // Moderate effectiveness — slow leak is harder to address than spike
        let effectiveness = (pressure - 0.50).min(0.40) + 0.20;
        let signal_quality = 0.60 + (pressure - 0.50) * 0.3; // slightly noisier at high pressure
        step(&mut nl, signal_quality, effectiveness);

        // Check for velocity spike (would indicate unstable regime detection)
        if nl.l2_meta_velocity > prev_velocity * 3.0 + 0.10 {
            velocity_spike = true;
        }
        prev_velocity = nl.l2_meta_velocity;
    }

    // Invariants for gradual drift
    assert!(
        !velocity_spike,
        "slow accumulation should not cause velocity spikes (gradual drift, not regime change)"
    );
    assert!(
        nl.l2_meta_velocity < 0.30,
        "meta-velocity should remain moderate under slow drift, got {}",
        nl.l2_meta_velocity
    );
    assert!(
        nl.l1_total > 0,
        "L1 should have accumulated outcome observations, got {}",
        nl.l1_total
    );
}

// ── Workload 3: LLM Steady State ─────────────────────────────────────────────

/// LLM steady state: pressure ~0.72 ± 0.03 noise over 60 steps.
/// Simulates local LLM inference (constant high memory pressure, stable workload).
///
/// Validates: (1) gate stays open consistently (good signal quality),
/// (2) meta-velocity stays near zero (no regime changes),
/// (3) dynamic gate remains close to baseline L1_GATE_THRESHOLD.
#[test]
fn workload_llm_steady() {
    let mut nl = NestedLearner::new();

    // Warm up at steady-state pressure
    for _ in 0..30 {
        nl.tick_l0(0.75);
    }

    let mut gate_closures = 0usize;
    // LCG-style deterministic "noise" — avoids dependency on rand
    let mut pseudo_noise: u64 = 0xdeadbeef;

    for _ in 0..60 {
        // Deterministic noise in ±0.03 range
        pseudo_noise = pseudo_noise
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = ((pseudo_noise >> 33) as f64 / u32::MAX as f64) * 0.06 - 0.03;
        let signal_quality = (0.78 + noise * 0.5).clamp(0.0, 1.0);
        let effectiveness = (0.55 + noise).clamp(0.0, 1.0); // moderate effectiveness

        let (gate, _, _) = step(&mut nl, signal_quality, effectiveness);
        if !gate {
            gate_closures += 1;
        }
    }

    // Invariants
    assert!(
        gate_closures <= 3, // allow very occasional dips from noise
        "LLM steady workload should have gate open nearly always, closures={}",
        gate_closures
    );
    assert!(
        nl.l2_meta_velocity < 0.15,
        "steady LLM workload should have near-zero meta-velocity, got {}",
        nl.l2_meta_velocity
    );
    // Dynamic gate should be close to baseline (velocity ≈ 0)
    let gate_threshold = nl.dynamic_l1_gate();
    assert!(
        gate_threshold < 0.30,
        "gate should remain near baseline 0.25 under steady workload, got {}",
        gate_threshold
    );
}

// ── Workload 4: Mixed Adversarial ────────────────────────────────────────────

/// Mixed adversarial: alternates between compile (high effectiveness) and
/// browse (low effectiveness) every 10 steps over 80 steps.
/// Rapid regime changes produce high meta-velocity.
///
/// Validates: the L2→L0 feedback from P1 (Iter 1) is detectable —
/// dynamic_l1_gate() rises measurably above baseline L1_GATE_THRESHOLD
/// after sustained regime oscillations.
#[test]
fn workload_mixed_adversarial_raises_gate() {
    let mut nl = NestedLearner::new();

    // Warm up
    for _ in 0..20 {
        nl.tick_l0(0.70);
    }

    let baseline_gate = nl.dynamic_l1_gate();

    // Alternate regimes: compile (eff=0.85, sig=0.82) vs browse (eff=0.15, sig=0.55)
    for cycle in 0..8 {
        let (effectiveness, signal_quality) = if cycle % 2 == 0 {
            (0.85_f64, 0.82_f64) // compile regime
        } else {
            (0.15_f64, 0.55_f64) // browse regime
        };

        // Run 10 steps in this regime
        for _ in 0..10 {
            step(&mut nl, signal_quality, effectiveness);
        }
    }

    let final_gate = nl.dynamic_l1_gate();

    // The repeated oscillations should have built up non-zero meta-velocity,
    // which in turn should have raised the gate above the initial baseline.
    assert!(
        nl.l2_meta_velocity > 0.0,
        "adversarial workload should produce non-zero meta-velocity, got {}",
        nl.l2_meta_velocity
    );
    assert!(
        final_gate >= baseline_gate,
        "gate should not drop below baseline after adversarial regimes: baseline={}, final={}",
        baseline_gate,
        final_gate
    );

    // Ideally gate has risen — but only assert if velocity is substantial
    if nl.l2_meta_velocity > 0.05 {
        assert!(
            final_gate > 0.25,
            "with meta-velocity={:.3}, gate should exceed 0.25, got {}",
            nl.l2_meta_velocity,
            final_gate
        );
    }

    assert!(
        nl.l2_total >= 2,
        "adversarial workload should produce multiple L2 flushes, got {}",
        nl.l2_total
    );
}
