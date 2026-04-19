//! Neuromodulator — bio-inspired parameter modulation for Apollo.
//!
//! Adapted from memoria-core/src/cognition/neuromodulator.rs.
//! Four "neurochemical" signals modulate Apollo's behavior parameters:
//!
//! - **Dopamine**: reward flowing → increase RL learning rate
//! - **Noradrenaline**: stress/urgency → more Dyna-Q planning, narrow focus
//! - **Serotonin**: stability → raise router thresholds, conserve CPU
//! - **Acetylcholine**: novelty → boost exploration epsilon
//!
//! All levels are [0.0, 1.0] with leaky integration (tau ~10 ticks).
//! At baseline (all 0.5), derived parameters equal current hardcoded values.
//! Cost: ~50ns per cycle, 0 allocations, 0 dependencies.

/// Decay rate per tick. With tau=10, levels return to baseline in ~10 ticks.
const DECAY: f64 = 0.10;

/// Input signals from Apollo's subsystems, collected each cycle.
pub struct NeuroSignals {
    // Dopamine inputs
    pub pressure_drop: f64,   // prev_pressure - current (positive = good)
    pub outcome_penalty: f64, // from OutcomeTracker (negative = bad)
    pub overflow_occurred: bool,

    // Noradrenaline inputs
    pub urgency: f64, // signal_digest.urgency
    pub regime_shift_up: bool,
    pub pressure_velocity: f64, // positive = rising pressure
    /// Graded thermal stress [0.0, 1.0]: 0 at 60°C, 0.5 at 80°C, 1.0 at ≥100°C.
    pub thermal_stress: f64,
    /// ODE swap saturation urgency [0.0, 1.0]: (CRITICAL_ETA_SEC / t_sat_sec).clamp(0,1).
    /// 0 = safe, 1 = saturation within CRITICAL_ETA_SEC. Leading indicator — rises
    /// before memory pressure changes so NA responds predictively, not reactively.
    pub ode_swap_urgency: f64,

    // Serotonin inputs
    pub pressure_smooth: f64, // for streak tracking
    pub regime_shift_down: bool,

    // Acetylcholine inputs
    pub process_count: usize,
    pub entropy_anomaly: f64,
    pub rl_exploring: bool,
}

pub struct ApolloNeuromodulator {
    // Raw levels [0.0, 1.0]
    dopamine: f64,
    noradrenaline: f64,
    serotonin: f64,
    acetylcholine: f64,

    // Derived parameters (computed each tick)
    /// DA → RL alpha multiplier [0.5, 1.5]. Baseline=1.0.
    pub alpha_multiplier: f64,
    /// NA → Dyna-Q planning steps [4, 20]. Baseline=10.
    pub dyna_steps: usize,
    /// SE → Router zone shift [-0.05, +0.05]. Baseline=0.0.
    pub serotonin_shift: f64,
    /// ACh → Epsilon exploration bonus [0.0, 0.05]. Baseline=0.025.
    pub epsilon_bonus: f64,

    // Internal state
    low_pressure_streak: u32,
    last_process_count: usize,
}

impl ApolloNeuromodulator {
    pub fn new() -> Self {
        Self {
            dopamine: 0.5,
            noradrenaline: 0.5,
            serotonin: 0.5,
            acetylcholine: 0.5,
            alpha_multiplier: 1.0,
            dyna_steps: 10,
            serotonin_shift: 0.0,
            epsilon_bonus: 0.025,
            low_pressure_streak: 0,
            last_process_count: 0,
        }
    }

    /// Update all neurotransmitter levels and recompute derived parameters.
    pub fn tick(&mut self, s: &NeuroSignals) {
        // ── Dopamine: reward signal ──────────────────────────────────
        let da_reward = if s.overflow_occurred { 0.0 } else { 0.3 };
        let da_drop = s.pressure_drop.clamp(0.0, 0.5) * 0.8;
        let da_outcome = (1.0 + s.outcome_penalty / 5.0).clamp(0.0, 1.0) * 0.2;
        let da_signal = (da_reward + da_drop + da_outcome).clamp(0.0, 1.0);
        self.dopamine = (self.dopamine * (1.0 - DECAY) + da_signal * DECAY).clamp(0.0, 1.0);

        // ── Noradrenaline: stress/urgency ────────────────────────────
        // [Deacon 2013] predictive NA: ode_swap_urgency is a leading indicator
        // that rises before pressure changes — reduces urgency weight slightly
        // to keep total NA scale stable while adding anticipatory ODE signal.
        let na_urgency = s.urgency.clamp(0.0, 1.0) * 0.35;
        let na_regime = if s.regime_shift_up { 0.3 } else { 0.0 };
        let na_velocity = (s.pressure_velocity * 2.0).clamp(0.0, 0.3);
        let na_thermal = s.thermal_stress.clamp(0.0, 1.0) * 0.2;
        let na_ode = s.ode_swap_urgency.clamp(0.0, 1.0) * 0.15;
        let na_signal = (na_urgency + na_regime + na_velocity + na_thermal + na_ode).clamp(0.0, 1.0);
        self.noradrenaline =
            (self.noradrenaline * (1.0 - DECAY) + na_signal * DECAY).clamp(0.0, 1.0);

        // ── Serotonin: stability ─────────────────────────────────────
        if s.pressure_smooth < 0.30 {
            self.low_pressure_streak += 1;
        } else {
            self.low_pressure_streak = self.low_pressure_streak.saturating_sub(1);
        }
        let se_streak = (self.low_pressure_streak as f64 / 20.0).clamp(0.0, 0.5);
        let se_calm = (1.0 - s.urgency) * 0.3;
        let se_regime = if s.regime_shift_down { 0.15 } else { 0.0 };
        let se_no_overflow = if !s.overflow_occurred { 0.1 } else { 0.0 };
        let se_signal = (se_streak + se_calm + se_regime + se_no_overflow).clamp(0.0, 1.0);
        self.serotonin = (self.serotonin * (1.0 - DECAY) + se_signal * DECAY).clamp(0.0, 1.0);

        // ── Acetylcholine: novelty ───────────────────────────────────
        let churn = (self.last_process_count as isize - s.process_count as isize).unsigned_abs();
        self.last_process_count = s.process_count;
        let ach_churn = (churn as f64 / 20.0).clamp(0.0, 0.4);
        let ach_entropy = (s.entropy_anomaly.abs() / 3.0).clamp(0.0, 0.3);
        let ach_explore = if s.rl_exploring { 0.2 } else { 0.05 };
        let ach_signal = (ach_churn + ach_entropy + ach_explore).clamp(0.0, 1.0);
        self.acetylcholine =
            (self.acetylcholine * (1.0 - DECAY) + ach_signal * DECAY).clamp(0.0, 1.0);

        // ── Derive parameters ────────────────────────────────────────
        self.alpha_multiplier = 0.5 + self.dopamine; // [0.5, 1.5]
        self.dyna_steps = (4.0 + self.noradrenaline * 16.0).round() as usize; // [4, 20]
        self.serotonin_shift = (self.serotonin - 0.5) * 0.10; // [-0.05, +0.05]
        self.epsilon_bonus = self.acetylcholine * 0.05; // [0.0, 0.05]
    }

    /// Current levels for observability.
    pub fn levels(&self) -> (f64, f64, f64, f64) {
        (
            self.dopamine,
            self.noradrenaline,
            self.serotonin,
            self.acetylcholine,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_signals() -> NeuroSignals {
        NeuroSignals {
            pressure_drop: 0.0,
            outcome_penalty: 0.0,
            overflow_occurred: false,
            urgency: 0.4,
            regime_shift_up: false,
            pressure_velocity: 0.0,
            thermal_stress: 0.0,
            ode_swap_urgency: 0.0,
            pressure_smooth: 0.50,
            regime_shift_down: false,
            process_count: 400,
            entropy_anomaly: 0.0,
            rl_exploring: false,
        }
    }

    #[test]
    fn test_baseline_derived_params() {
        let nm = ApolloNeuromodulator::new();
        assert_eq!(nm.alpha_multiplier, 1.0);
        assert_eq!(nm.dyna_steps, 10);
        assert!((nm.serotonin_shift).abs() < 1e-9);
        assert!((nm.epsilon_bonus - 0.025).abs() < 1e-9);
    }

    #[test]
    fn test_dopamine_rises_on_pressure_drop() {
        let mut nm = ApolloNeuromodulator::new();
        let mut s = default_signals();
        s.pressure_drop = 0.10; // good drop
        for _ in 0..20 {
            nm.tick(&s);
        }
        assert!(nm.dopamine > 0.5, "DA should rise: {}", nm.dopamine);
        assert!(
            nm.alpha_multiplier > 1.0,
            "alpha mult should increase: {}",
            nm.alpha_multiplier
        );
    }

    #[test]
    fn test_noradrenaline_rises_on_urgency() {
        let mut nm = ApolloNeuromodulator::new();
        let mut s = default_signals();
        s.urgency = 0.9;
        s.thermal_stress = 1.0; // full thermal emergency (≥100°C equivalent)
        for _ in 0..20 {
            nm.tick(&s);
        }
        assert!(
            nm.noradrenaline > 0.5,
            "NA should rise: {}",
            nm.noradrenaline
        );
        assert!(
            nm.dyna_steps > 10,
            "dyna steps should increase: {}",
            nm.dyna_steps
        );
    }

    #[test]
    fn test_serotonin_rises_on_calm() {
        let mut nm = ApolloNeuromodulator::new();
        let mut s = default_signals();
        s.pressure_smooth = 0.20; // low pressure
        s.urgency = 0.1;
        s.regime_shift_down = true;
        for _ in 0..30 {
            nm.tick(&s);
        }
        assert!(nm.serotonin > 0.5, "SE should rise: {}", nm.serotonin);
        assert!(
            nm.serotonin_shift > 0.0,
            "shift should be positive: {}",
            nm.serotonin_shift
        );
    }

    #[test]
    fn test_acetylcholine_rises_on_novelty() {
        let mut nm = ApolloNeuromodulator::new();
        let mut s = default_signals();
        s.entropy_anomaly = 2.0;
        s.rl_exploring = true;
        s.process_count = 420; // churn from default 400
        for _ in 0..20 {
            nm.tick(&s);
        }
        assert!(
            nm.acetylcholine > 0.5,
            "ACh should rise: {}",
            nm.acetylcholine
        );
        assert!(
            nm.epsilon_bonus > 0.025,
            "epsilon bonus should increase: {}",
            nm.epsilon_bonus
        );
    }

    #[test]
    fn test_levels_clamped() {
        let mut nm = ApolloNeuromodulator::new();
        let mut s = default_signals();
        // Extreme stress
        s.urgency = 1.0;
        s.thermal_stress = 1.0;
        s.regime_shift_up = true;
        s.pressure_velocity = 1.0;
        s.overflow_occurred = true;
        s.entropy_anomaly = 10.0;
        s.process_count = 1000;
        for _ in 0..100 {
            nm.tick(&s);
        }
        let (da, na, se, ach) = nm.levels();
        assert!(da >= 0.0 && da <= 1.0);
        assert!(na >= 0.0 && na <= 1.0);
        assert!(se >= 0.0 && se <= 1.0);
        assert!(ach >= 0.0 && ach <= 1.0);
    }

    #[test]
    fn test_decay_returns_to_baseline() {
        let mut nm = ApolloNeuromodulator::new();
        let mut s = default_signals();
        // Spike noradrenaline
        s.urgency = 1.0;
        s.thermal_stress = 1.0;
        for _ in 0..20 {
            nm.tick(&s);
        }
        let na_high = nm.noradrenaline;
        // Remove stress
        s.urgency = 0.1;
        s.thermal_stress = 0.0;
        for _ in 0..50 {
            nm.tick(&s);
        }
        assert!(
            nm.noradrenaline < na_high,
            "NA should decay: {} < {}",
            nm.noradrenaline,
            na_high
        );
    }

    #[test]
    fn test_graded_thermal_stress_proportional() {
        // thermal_stress=0.5 (80°C) should produce NA between cold(0.0) and hot(1.0).
        let mut nm_cold = ApolloNeuromodulator::new();
        let mut nm_warm = ApolloNeuromodulator::new();
        let mut nm_hot = ApolloNeuromodulator::new();
        let mut s_cold = default_signals();
        let mut s_warm = default_signals();
        let mut s_hot = default_signals();
        s_cold.thermal_stress = 0.0;
        s_warm.thermal_stress = 0.5;
        s_hot.thermal_stress = 1.0;
        for _ in 0..20 {
            nm_cold.tick(&s_cold);
            nm_warm.tick(&s_warm);
            nm_hot.tick(&s_hot);
        }
        assert!(
            nm_cold.noradrenaline < nm_warm.noradrenaline,
            "warm NA({}) should exceed cold NA({})",
            nm_warm.noradrenaline,
            nm_cold.noradrenaline
        );
        assert!(
            nm_warm.noradrenaline < nm_hot.noradrenaline,
            "hot NA({}) should exceed warm NA({})",
            nm_hot.noradrenaline,
            nm_warm.noradrenaline
        );
    }

    #[test]
    fn test_ode_swap_urgency_raises_na() {
        // ODE urgency=1.0 (swap saturating now) should raise NA above baseline.
        let mut nm_safe = ApolloNeuromodulator::new();
        let mut nm_critical = ApolloNeuromodulator::new();
        let mut s_safe = default_signals();
        let mut s_critical = default_signals();
        s_safe.ode_swap_urgency = 0.0;
        s_critical.ode_swap_urgency = 1.0;
        for _ in 0..20 {
            nm_safe.tick(&s_safe);
            nm_critical.tick(&s_critical);
        }
        assert!(
            nm_critical.noradrenaline > nm_safe.noradrenaline,
            "critical ODE NA({}) should exceed safe NA({})",
            nm_critical.noradrenaline,
            nm_safe.noradrenaline
        );
        assert!(
            nm_critical.dyna_steps > nm_safe.dyna_steps,
            "critical ODE dyna_steps({}) should exceed safe({})",
            nm_critical.dyna_steps,
            nm_safe.dyna_steps
        );
    }
}
