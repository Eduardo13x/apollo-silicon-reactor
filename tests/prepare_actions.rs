//! ══════════════════════════════════════════════════════════════════════════════
//! Apollo AutoResearch — Decision Actions Benchmark
//! ══════════════════════════════════════════════════════════════════════════════
//!
//! THIS FILE IS READ-ONLY. The agent must NEVER modify it.
//!
//! Tests the pure decision functions in decide_actions.rs:
//!   - Context classification (pressure → InteractiveContext)
//!   - Blocker score formula (weights, threshold, ranking)
//!
//! These require `context_from_pressure` and `blocker_score_formula` to be
//! exposed as `pub` in decide_actions.rs.
//!
//! Target file: src/engine/decide_actions.rs

#[cfg(test)]
mod scenarios {
    use apollo_optimizer::collector::{
        CpuStats, MemoryStats, PressureStats, SystemSnapshot,
    };
    use apollo_optimizer::engine::decide_actions::context_from_pressure;
    use apollo_optimizer::engine::decide_actions::blocker_score_formula;
    use apollo_optimizer::engine::overflow_guard::OverflowThresholds;
    use apollo_optimizer::engine::types::InteractiveContext;
    use chrono::Utc;

    fn make_snapshot(cpu_usage: f32, mem_pressure: f64, swap_delta: f64) -> SystemSnapshot {
        SystemSnapshot {
            timestamp: Utc::now(),
            cpu: CpuStats {
                global_usage: cpu_usage,
                core_count: 8,
            },
            memory: MemoryStats {
                total_ram: 8 * 1024 * 1024 * 1024,
                used_ram: 5 * 1024 * 1024 * 1024,
                free_ram: 3 * 1024 * 1024 * 1024,
                total_swap: 4 * 1024 * 1024 * 1024,
                used_swap: 0,
            },
            pressure: PressureStats {
                memory_pressure: mem_pressure,
                swap_used_bytes: 0,
                swap_total_bytes: 4 * 1024 * 1024 * 1024,
                swap_delta_bytes_per_sec: swap_delta,
                thermal_level: "nominal".to_string(),
                compressor_pressure: 0.0,
            },
            disks: vec![],
            networks: vec![],
            top_processes: vec![],
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 1: CONTEXT CLASSIFICATION
    // The three contexts determine overall system behavior.
    // Getting this wrong means throttling when we should boost, or vice versa.
    // ══════════════════════════════════════════════════════════════════════════

    /// A01: Low pressure → InteractiveFocus (normal operation, boost foreground).
    #[test]
    fn a01_low_pressure_is_interactive() {
        let snap = make_snapshot(30.0, 0.40, 0.0);
        let ctx = context_from_pressure(&snap, &OverflowThresholds::default());
        assert!(
            matches!(ctx, InteractiveContext::InteractiveFocus),
            "CPU 30% + mem 0.40 = normal → InteractiveFocus. Got {:?}", ctx
        );
    }

    /// A02: Moderate CPU (75%) with low memory → BackgroundPressure.
    /// CPU is above 72% threshold — should start throttling background.
    #[test]
    fn a02_moderate_cpu_is_background_pressure() {
        let snap = make_snapshot(75.0, 0.40, 0.0);
        let ctx = context_from_pressure(&snap, &OverflowThresholds::default());
        assert!(
            matches!(ctx, InteractiveContext::BackgroundPressure),
            "CPU 75% > 72% → BackgroundPressure. Got {:?}", ctx
        );
    }

    /// A03: High CPU (90%) → ThermalConstrained.
    #[test]
    fn a03_high_cpu_is_thermal() {
        let snap = make_snapshot(90.0, 0.40, 0.0);
        let ctx = context_from_pressure(&snap, &OverflowThresholds::default());
        assert!(
            matches!(ctx, InteractiveContext::ThermalConstrained),
            "CPU 90% > 88% → ThermalConstrained. Got {:?}", ctx
        );
    }

    /// A04: Normal CPU but high memory pressure → ThermalConstrained.
    /// Memory pressure alone at critical level should escalate.
    #[test]
    fn a04_high_memory_is_thermal() {
        let snap = make_snapshot(50.0, 0.92, 0.0);
        let ctx = context_from_pressure(&snap, &OverflowThresholds::default());
        assert!(
            matches!(ctx, InteractiveContext::ThermalConstrained),
            "Mem pressure 0.92 > critical → ThermalConstrained. Got {:?}", ctx
        );
    }

    /// A05: CPU at exactly 72% — should be the boundary for BackgroundPressure.
    /// At exactly the threshold, we should still be InteractiveFocus (> not >=).
    #[test]
    fn a05_cpu_at_boundary_72() {
        let snap = make_snapshot(72.0, 0.40, 0.0);
        let ctx = context_from_pressure(&snap, &OverflowThresholds::default());
        // 72.0 is NOT > 72.0, so should remain InteractiveFocus
        assert!(
            matches!(ctx, InteractiveContext::InteractiveFocus),
            "CPU at exactly 72% should be InteractiveFocus (> not >=). Got {:?}", ctx
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 2: BLOCKER SCORE FORMULA
    // Tests the weight balance for identifying system blockers.
    // ══════════════════════════════════════════════════════════════════════════

    /// A06: High interactive wait ratio should dominate the score.
    /// When 4+ interactive apps are waiting, blocker score must be > 0.30.
    #[test]
    fn a06_high_interactive_wait_scores_above_threshold() {
        // interactive_wait_ratio = 1.0 (5+ apps waiting), cpu_spike = 0.1,
        // seen_recently = false, reactor = 0.0
        let score = blocker_score_formula(1.0, 0.1, false, 0.0);
        assert!(score > 0.30, "High interactive wait should score > threshold. Got {}", score);
    }

    /// A07: High CPU spike from blocker should score above threshold.
    #[test]
    fn a07_cpu_spike_scores_above_threshold() {
        // interactive_wait = 0.0, cpu_spike = 80% (0.8), seen_recently = true, reactor = 0.5
        let score = blocker_score_formula(0.0, 0.8, true, 0.5);
        assert!(score > 0.30, "80% CPU spike from blocker should score > threshold. Got {}", score);
    }

    /// A08: Everything low → score below threshold (no action needed).
    #[test]
    fn a08_low_everything_below_threshold() {
        let score = blocker_score_formula(0.0, 0.05, false, 0.0);
        assert!(score < 0.30, "Quiet system → no blocker action. Got {}", score);
    }

    /// A09: Interactive wait should outweigh CPU spike in scoring.
    /// User waiting is worse than a process being busy.
    #[test]
    fn a09_interactive_wait_outweighs_cpu() {
        let wait_heavy = blocker_score_formula(0.8, 0.1, false, 0.0);
        let cpu_heavy = blocker_score_formula(0.1, 0.8, false, 0.0);
        assert!(
            wait_heavy > cpu_heavy,
            "Interactive wait ({}) should score higher than CPU spike ({})",
            wait_heavy, cpu_heavy
        );
    }

    /// A10: Reactor events should contribute when other signals are moderate.
    #[test]
    fn a10_reactor_contributes_to_score() {
        let without_reactor = blocker_score_formula(0.3, 0.3, false, 0.0);
        let with_reactor = blocker_score_formula(0.3, 0.3, false, 0.8);
        assert!(
            with_reactor > without_reactor,
            "Reactor events should increase score: {} vs {}", with_reactor, without_reactor
        );
    }
}
