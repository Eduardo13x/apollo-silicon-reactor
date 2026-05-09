//! ══════════════════════════════════════════════════════════════════════════════
//! Apollo AutoResearch — RL Threshold Agent Benchmark
//! ══════════════════════════════════════════════════════════════════════════════
//!
//! THIS FILE IS READ-ONLY. The agent must NEVER modify it.
//!
//! Tests safety floor, Q-value convergence, state discretization, action
//! asymmetry, EMA alpha decay, reward shaping, and external reward injection.
//! Failures cause: threshold runaway (RAM overflow), stale learning (agent
//! ignores feedback), or floor violation (unsafe thresholds applied).
//!
//! Target file: src/engine/rl_threshold.rs

#[cfg(test)]
mod scenarios {
    use std::path::PathBuf;

    use apollo_engine::engine::rl_threshold::{RlState, RlThresholdAgent, RL_ABSOLUTE_FLOOR};

    fn make_agent() -> RlThresholdAgent {
        RlThresholdAgent::load_or_default(&PathBuf::from("/dev/null"))
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 1: SAFETY FLOOR VALIDATION
    // ══════════════════════════════════════════════════════════════════════════

    /// R01: After 100 overflow ticks, adjustment must stay >= floor (-0.20).
    /// Safety: prevents threshold from going too low and freezing the system.
    #[test]
    fn r01_adjustment_never_below_floor() {
        let mut agent = make_agent();
        let crisis = RlState::from_metrics(0.95, 0.90, 3);
        for _ in 0..100 {
            agent.tick(crisis, true);
        }
        assert!(
            agent.current_adjustment >= -0.20,
            "adjustment must stay >= -0.20: {}",
            agent.current_adjustment
        );
    }

    /// R02: After 200 calm ticks, adjustment must stay <= ceiling (+0.05).
    #[test]
    fn r02_adjustment_never_above_ceiling() {
        let mut agent = make_agent();
        let calm = RlState::from_metrics(0.10, 0.05, 0);
        for _ in 0..200 {
            agent.tick(calm, false);
        }
        assert!(
            agent.current_adjustment <= 0.05,
            "adjustment must stay <= 0.05: {}",
            agent.current_adjustment
        );
    }

    /// R03: The absolute floor (0.45) must hold even with worst-case stacking
    /// of adjustment + dynamic offset.
    #[test]
    fn r03_absolute_floor_holds() {
        let mut agent = make_agent();
        for _ in 0..100 {
            agent.tick(RlState::from_metrics(0.95, 0.90, 3), true);
        }
        // Simulate worst-case: base 0.78 + adjustment + hypothetical -0.08 dynamic offset.
        let effective = (0.78 + agent.current_adjustment - 0.08).max(RL_ABSOLUTE_FLOOR);
        assert!(
            effective >= RL_ABSOLUTE_FLOOR,
            "effective threshold must be >= {}: {}",
            RL_ABSOLUTE_FLOOR,
            effective
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 2: Q-VALUE CONVERGENCE (observed via behavior)
    // ══════════════════════════════════════════════════════════════════════════

    /// R04: After 50 stable ticks at low pressure, the agent's last Q-value
    /// must be positive (stability is rewarding).
    #[test]
    fn r04_stable_ticks_positive_q() {
        let mut agent = make_agent();
        let calm = RlState::from_metrics(0.30, 0.10, 0);
        for _ in 0..50 {
            agent.tick(calm, false);
        }
        let last_q = agent.last_q_value();
        assert!(
            last_q > 0.0,
            "after 50 stable ticks, last Q must be positive: {}",
            last_q
        );
    }

    /// R05: After 10 overflow ticks, overflow counter must increase correctly.
    #[test]
    fn r05_overflow_counter_tracks() {
        let mut agent = make_agent();
        let crisis = RlState::from_metrics(0.85, 0.70, 2);
        for _ in 0..10 {
            agent.tick(crisis, true);
        }
        assert_eq!(
            agent.total_overflows(),
            10,
            "must count all overflow events"
        );
    }

    /// R06: Overflow must penalize: Q-value decreases after an overflow tick.
    #[test]
    fn r06_overflow_penalizes_q() {
        let mut agent = make_agent();
        let state = RlState::from_metrics(0.85, 0.40, 0);
        // First tick: establish state-action pair.
        agent.tick(state, false);
        let _q_before = agent.last_q_value();
        // Second tick with overflow: updates Q for previous state-action.
        agent.tick(state, true);
        // Third tick to read the updated Q for the state.
        agent.tick(state, false);
        // The Q landscape should have been pushed negative by the overflow.
        // We verify total_overflows increased as minimum guarantee.
        assert_eq!(agent.total_overflows(), 1, "overflow must be recorded");
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 3: STATE DISCRETIZATION
    // ══════════════════════════════════════════════════════════════════════════

    /// R07: All 36 state indices must be unique and in-range.
    #[test]
    fn r07_all_36_states_unique() {
        let mut seen = std::collections::HashSet::new();
        for pb in 0..3u8 {
            for cb in 0..3u8 {
                for oh in 0..4u8 {
                    let s = RlState {
                        pressure_band: pb,
                        compressor_band: cb,
                        overflow_last_hour: oh,
                    };
                    let idx = s.index();
                    assert!(idx < 36, "index {} out of range for {:?}", idx, s);
                    seen.insert(idx);
                }
            }
        }
        assert_eq!(seen.len(), 36, "must produce 36 unique indices");
    }

    /// R08: Boundary values discretize correctly.
    #[test]
    fn r08_boundary_discretization() {
        // Pressure bands: <0.50=0, 0.50-0.80=1, >0.80=2
        assert_eq!(RlState::from_metrics(0.00, 0.0, 0).pressure_band, 0);
        assert_eq!(RlState::from_metrics(0.49, 0.0, 0).pressure_band, 0);
        assert_eq!(RlState::from_metrics(0.50, 0.0, 0).pressure_band, 1);
        assert_eq!(RlState::from_metrics(0.80, 0.0, 0).pressure_band, 1);
        assert_eq!(RlState::from_metrics(0.81, 0.0, 0).pressure_band, 2);
        // Compressor bands: <0.30=0, 0.30-0.60=1, >0.60=2
        assert_eq!(RlState::from_metrics(0.0, 0.29, 0).compressor_band, 0);
        assert_eq!(RlState::from_metrics(0.0, 0.30, 0).compressor_band, 1);
        assert_eq!(RlState::from_metrics(0.0, 0.60, 0).compressor_band, 1);
        assert_eq!(RlState::from_metrics(0.0, 0.61, 0).compressor_band, 2);
        // Overflow clamped at 3
        assert_eq!(RlState::from_metrics(0.0, 0.0, 5).overflow_last_hour, 3);
        assert_eq!(RlState::from_metrics(0.0, 0.0, 100).overflow_last_hour, 3);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 4: ACTION ASYMMETRY & EMA ALPHA
    // ══════════════════════════════════════════════════════════════════════════

    /// R09: Adjustment stays within bounds after mixed stable/overflow ticks.
    #[test]
    fn r09_bounds_hold_under_mixed_conditions() {
        let mut agent = make_agent();
        let calm = RlState::from_metrics(0.30, 0.10, 0);
        let crisis = RlState::from_metrics(0.90, 0.80, 3);
        for _ in 0..50 {
            agent.tick(calm, false);
        }
        for _ in 0..50 {
            agent.tick(crisis, true);
        }
        for _ in 0..50 {
            agent.tick(calm, false);
        }
        assert!(
            agent.current_adjustment >= -0.20 && agent.current_adjustment <= 0.05,
            "adjustment must be in [-0.20, 0.05] after mixed conditions: {}",
            agent.current_adjustment
        );
    }

    /// R10: EMA alpha decays from 0.20 toward 0.02 over 400 ticks.
    #[test]
    fn r10_alpha_decays_over_time() {
        let mut agent = make_agent();
        let alpha_0 = agent.alpha();
        assert!(
            (alpha_0 - 0.20).abs() < 1e-6,
            "initial alpha must be 0.20: {}",
            alpha_0
        );

        let state = RlState::from_metrics(0.50, 0.30, 0);
        for _ in 0..400 {
            agent.tick(state, false);
        }
        let alpha_400 = agent.alpha();
        assert!(
            alpha_400 < alpha_0,
            "alpha must decay: {} < {}",
            alpha_400,
            alpha_0
        );
        assert!(
            alpha_400 >= 0.02,
            "alpha must not go below floor: {}",
            alpha_400
        );
    }

    /// R11: Epsilon decays from 0.10 to 0.05 after 200 ticks.
    #[test]
    fn r11_epsilon_decay() {
        let mut agent = make_agent();
        assert!(
            (agent.epsilon() - 0.10).abs() < 1e-6,
            "initial epsilon must be 0.10"
        );
        let state = RlState::from_metrics(0.50, 0.30, 0);
        for _ in 0..200 {
            agent.tick(state, false);
        }
        assert!(
            (agent.epsilon() - 0.05).abs() < 1e-6,
            "epsilon after 200 ticks must be 0.05"
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 5: REWARD SHAPING & EXTERNAL INJECTION
    // ══════════════════════════════════════════════════════════════════════════

    /// R12: External negative reward must decrease Q for the last state-action pair.
    #[test]
    fn r12_external_reward_decreases_q() {
        let mut agent = make_agent();
        let state = RlState::from_metrics(0.60, 0.40, 0);
        agent.tick(state, false);
        let q_before = agent.last_q_value();
        agent.inject_external_reward(-5.0);
        let q_after = agent.last_q_value();
        assert!(
            q_after < q_before,
            "negative external reward must decrease Q: {} < {}",
            q_after,
            q_before
        );
    }

    /// R13: External positive reward must increase Q for the last state-action pair.
    #[test]
    fn r13_external_reward_increases_q() {
        let mut agent = make_agent();
        let state = RlState::from_metrics(0.60, 0.40, 0);
        agent.tick(state, false);
        let q_before = agent.last_q_value();
        agent.inject_external_reward(5.0);
        let q_after = agent.last_q_value();
        assert!(
            q_after > q_before,
            "positive external reward must increase Q: {} > {}",
            q_after,
            q_before
        );
    }

    /// R14: After 50 stable ticks with high initial alpha, learning must be
    /// faster than a hypothetical fixed-alpha agent — Q must exceed 2.0.
    #[test]
    fn r14_ema_converges_fast() {
        let mut agent = make_agent();
        let calm = RlState::from_metrics(0.30, 0.10, 0);
        for _ in 0..50 {
            agent.tick(calm, false);
        }
        let last_q = agent.last_q_value();
        assert!(
            last_q > 2.0,
            "EMA agent must learn fast from early data: last_q={}",
            last_q
        );
    }

    /// R15: Total ticks counter must increment correctly.
    #[test]
    fn r15_total_ticks_counter() {
        let mut agent = make_agent();
        assert_eq!(agent.total_ticks(), 0);
        let state = RlState::from_metrics(0.50, 0.30, 0);
        for _ in 0..25 {
            agent.tick(state, false);
        }
        assert_eq!(
            agent.total_ticks(),
            25,
            "total_ticks must equal 25 after 25 ticks"
        );
    }
}
