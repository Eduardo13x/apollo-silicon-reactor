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

use serde::{Deserialize, Serialize};

/// Decay rate per tick. With tau=10, levels return to baseline in ~10 ticks.
const DECAY: f64 = 0.10;

/// Serializable snapshot of the four raw neurotransmitter levels and the
/// low-pressure streak counter — everything needed to resume leaky integration
/// across a daemon restart without cold-starting at baseline 0.5.
///
/// Derived parameters (`alpha_multiplier`, `dyna_steps`, `serotonin_shift`,
/// `epsilon_bonus`) are NOT persisted: they are recomputed from the raw levels
/// on the next `tick()` call and their in-flight values are safe to reconstruct.
///
/// [Schultz 1997] — reward prediction error signals require continuity;
/// cold restarts erase the entire prediction history accumulated since startup.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NeuroState {
    /// Dopamine level [0.0, 1.0]. DA → alpha_multiplier [0.5, 1.5].
    pub dopamine: f64,
    /// Noradrenaline level [0.0, 1.0]. NA → dyna_steps [4, 20].
    pub noradrenaline: f64,
    /// Serotonin level [0.0, 1.0]. 5-HT → serotonin_shift [-0.05, +0.05].
    pub serotonin: f64,
    /// Acetylcholine level [0.0, 1.0]. ACh → epsilon_bonus [0.0, 0.05].
    pub acetylcholine: f64,
    /// Consecutive cycles where pressure_smooth < 0.30.
    /// Persisted so serotonin warm-start resumes the streak correctly.
    pub low_pressure_streak: u32,
}

/// Input signals from Apollo's subsystems, collected each cycle.
pub struct NeuroSignals {
    // Dopamine inputs
    pub pressure_drop: f64,   // prev_pressure - current (positive = good)
    pub outcome_penalty: f64, // from OutcomeTracker (negative = bad)
    pub overflow_occurred: bool,

    /// ODE prediction error [0.0, 1.0]: positive when swap was feared but pressure fell.
    /// DA encodes RPE [Schultz 1997] — better-than-predicted outcome boosts reward signal.
    pub ode_rss_surprise: f64,

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
    /// τ-divergence [0.0, 1.0]: mean relative deviation of learned τ from default.
    /// ACh tracks novelty [Schultz 1997]; heterogeneous τ = diverse reload behaviors.
    pub tau_divergence: f64,
    /// G11 — L2 Contention: fraction of cycles with L2 cache stalls [0.0, 1.0].
    /// High contention = unexpected memory-access pattern = ACh novelty signal.
    /// [Hasler & Mahowald 1994] — attentional gating amplifies novel stimuli.
    pub contention_stall_fraction: f64,
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
        // [Schultz 1997] RPE: ODE feared saturation but pressure fell → positive surprise.
        let da_ode = s.ode_rss_surprise.clamp(0.0, 1.0) * 0.10;
        let da_signal = (da_reward + da_drop + da_outcome + da_ode).clamp(0.0, 1.0);
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
        // [Schultz 1997] ACh novelty: heterogeneous τ across apps = diverse physical behaviors.
        let ach_tau = s.tau_divergence.clamp(0.0, 1.0) * 0.15;
        // G11: L2 contention novelty — unexpected cache pressure signals access-pattern shift.
        let ach_contention = s.contention_stall_fraction.clamp(0.0, 1.0) * 0.10;
        let ach_signal = (ach_churn + ach_entropy + ach_explore + ach_tau + ach_contention)
            .clamp(0.0, 1.0);
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

    /// Capture raw neurotransmitter levels for persistence.
    ///
    /// Derived parameters are omitted — they are recomputed from levels on the
    /// next `tick()`. The `last_process_count` internal counter is not persisted
    /// because it only affects ACh churn on the FIRST tick post-restore, a
    /// single-cycle artefact that is safe to absorb.
    pub fn snapshot(&self) -> NeuroState {
        NeuroState {
            dopamine: self.dopamine,
            noradrenaline: self.noradrenaline,
            serotonin: self.serotonin,
            acetylcholine: self.acetylcholine,
            low_pressure_streak: self.low_pressure_streak,
        }
    }

    /// Restore neurotransmitter levels from a persisted snapshot.
    ///
    /// All values are clamped to their valid ranges before being applied so
    /// corrupted or extreme disk state cannot destabilise the policy.  A NaN
    /// or Inf in any field is silently replaced with the neutral baseline (0.5).
    ///
    /// # Safety
    /// Even at extreme restored values (e.g. dopamine=1.0, noradrenaline=1.0)
    /// the worst outcome is an elevated alpha_multiplier and more Dyna-Q steps
    /// for the first few cycles.  The leaky integrator (τ≈10 ticks) returns
    /// signals to their ambient equilibrium within ~30 cycles regardless of
    /// starting point, bounding any transient distortion.
    pub fn restore(&mut self, s: NeuroState) {
        self.dopamine = if s.dopamine.is_finite() {
            s.dopamine.clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.noradrenaline = if s.noradrenaline.is_finite() {
            s.noradrenaline.clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.serotonin = if s.serotonin.is_finite() {
            s.serotonin.clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.acetylcholine = if s.acetylcholine.is_finite() {
            s.acetylcholine.clamp(0.0, 1.0)
        } else {
            0.5
        };
        // Streak is bounded by how many ticks the daemon has been running; cap
        // at 1000 to prevent an absurdly high persisted value from pinning
        // serotonin at its ceiling until the streak naturally decays.
        self.low_pressure_streak = s.low_pressure_streak.min(1000);
        // Recompute derived parameters immediately from restored raw levels so
        // the first read of alpha_multiplier / dyna_steps / etc. is correct
        // even before the next tick() call.
        self.alpha_multiplier = 0.5 + self.dopamine;
        self.dyna_steps = (4.0 + self.noradrenaline * 16.0).round() as usize;
        self.serotonin_shift = (self.serotonin - 0.5) * 0.10;
        self.epsilon_bonus = self.acetylcholine * 0.05;
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
            ode_rss_surprise: 0.0,
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
            tau_divergence: 0.0,
            contention_stall_fraction: 0.0,
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
    fn test_ode_rss_surprise_raises_da() {
        let mut nm_flat = ApolloNeuromodulator::new();
        let mut nm_surprised = ApolloNeuromodulator::new();
        let mut s_flat = default_signals();
        let mut s_surprised = default_signals();
        s_flat.ode_rss_surprise = 0.0;
        s_surprised.ode_rss_surprise = 1.0;
        for _ in 0..20 {
            nm_flat.tick(&s_flat);
            nm_surprised.tick(&s_surprised);
        }
        assert!(
            nm_surprised.dopamine > nm_flat.dopamine,
            "DA with surprise({}) should exceed no-surprise({})",
            nm_surprised.dopamine,
            nm_flat.dopamine
        );
    }

    #[test]
    fn test_tau_divergence_raises_ach() {
        let mut nm_uniform = ApolloNeuromodulator::new();
        let mut nm_diverse = ApolloNeuromodulator::new();
        let mut s_uniform = default_signals();
        let mut s_diverse = default_signals();
        s_uniform.tau_divergence = 0.0;
        s_diverse.tau_divergence = 1.0;
        for _ in 0..20 {
            nm_uniform.tick(&s_uniform);
            nm_diverse.tick(&s_diverse);
        }
        assert!(
            nm_diverse.acetylcholine > nm_uniform.acetylcholine,
            "ACh with τ-divergence({}) should exceed uniform({})",
            nm_diverse.acetylcholine,
            nm_uniform.acetylcholine
        );
    }

    // ── Warm-start tests ────────────────────────────────────────────────

    #[test]
    fn neuro_warm_start_survives_roundtrip() {
        // Tick neuromod to a non-neutral state by driving all signals hard.
        let mut neuro = ApolloNeuromodulator::new();
        let mut s = default_signals();
        s.pressure_drop = 0.40;        // drives DA up
        s.urgency = 0.90;              // NA urgency component
        s.regime_shift_up = true;      // NA regime component (+0.30)
        s.pressure_velocity = 0.50;    // NA velocity component
        s.thermal_stress = 1.0;        // NA thermal component (+0.20)
        s.pressure_smooth = 0.15;      // drives SE streak up (low pressure)
        s.entropy_anomaly = 2.0;       // ACh novelty
        s.rl_exploring = true;
        for _ in 0..30 {
            neuro.tick(&s);
        }
        // DA and NA should have drifted above 0.5 baseline under these inputs.
        // (NA steady-state ≈ urgency*0.35 + regime*0.30 + velocity*0.30 + thermal*0.20 ≈ 1.0)
        assert!(neuro.dopamine > 0.5, "DA should rise above baseline: {}", neuro.dopamine);
        assert!(neuro.noradrenaline > 0.5, "NA should rise above baseline: {}", neuro.noradrenaline);

        // Snapshot and restore into a fresh instance.
        let snap = neuro.snapshot();
        let mut neuro2 = ApolloNeuromodulator::new();
        neuro2.restore(snap.clone());

        // Raw levels must match exactly.
        assert!((neuro2.dopamine - snap.dopamine).abs() < 1e-10,
            "dopamine mismatch: {} vs {}", neuro2.dopamine, snap.dopamine);
        assert!((neuro2.noradrenaline - snap.noradrenaline).abs() < 1e-10,
            "noradrenaline mismatch: {} vs {}", neuro2.noradrenaline, snap.noradrenaline);
        assert!((neuro2.serotonin - snap.serotonin).abs() < 1e-10,
            "serotonin mismatch: {} vs {}", neuro2.serotonin, snap.serotonin);
        assert!((neuro2.acetylcholine - snap.acetylcholine).abs() < 1e-10,
            "acetylcholine mismatch: {} vs {}", neuro2.acetylcholine, snap.acetylcholine);

        // Derived parameters are recomputed on restore — verify they match
        // the source instance.
        assert!((neuro2.alpha_multiplier - neuro.alpha_multiplier).abs() < 1e-10,
            "alpha_multiplier mismatch after restore");
        assert_eq!(neuro2.dyna_steps, neuro.dyna_steps,
            "dyna_steps mismatch after restore");
        assert!((neuro2.serotonin_shift - neuro.serotonin_shift).abs() < 1e-10,
            "serotonin_shift mismatch after restore");
        assert!((neuro2.epsilon_bonus - neuro.epsilon_bonus).abs() < 1e-10,
            "epsilon_bonus mismatch after restore");
    }

    #[test]
    fn neuro_restore_clamps_corrupted_state() {
        let mut neuro = ApolloNeuromodulator::new();
        // Corrupted disk state with out-of-range values.
        neuro.restore(NeuroState {
            dopamine: 999.0,      // way above [0.0, 1.0]
            noradrenaline: -5.0,  // below [0.0, 1.0]
            serotonin: f64::NAN,  // NaN → neutral baseline
            acetylcholine: f64::INFINITY, // Inf → clamp
            low_pressure_streak: u32::MAX, // cap to 1000
        });
        assert!(neuro.dopamine >= 0.0 && neuro.dopamine <= 1.0,
            "dopamine out of range: {}", neuro.dopamine);
        assert!(neuro.noradrenaline >= 0.0 && neuro.noradrenaline <= 1.0,
            "noradrenaline out of range: {}", neuro.noradrenaline);
        assert!(neuro.serotonin.is_finite() && neuro.serotonin >= 0.0 && neuro.serotonin <= 1.0,
            "serotonin should be finite and clamped: {}", neuro.serotonin);
        assert!(neuro.acetylcholine >= 0.0 && neuro.acetylcholine <= 1.0,
            "acetylcholine out of range: {}", neuro.acetylcholine);
        assert!(neuro.low_pressure_streak <= 1000,
            "streak should be capped at 1000: {}", neuro.low_pressure_streak);
        // Derived params must also be in their valid ranges.
        assert!(neuro.alpha_multiplier >= 0.5 && neuro.alpha_multiplier <= 1.5,
            "alpha_multiplier out of range: {}", neuro.alpha_multiplier);
        assert!(neuro.dyna_steps >= 4 && neuro.dyna_steps <= 20,
            "dyna_steps out of range: {}", neuro.dyna_steps);
        assert!(neuro.serotonin_shift >= -0.05 && neuro.serotonin_shift <= 0.05,
            "serotonin_shift out of range: {}", neuro.serotonin_shift);
        assert!(neuro.epsilon_bonus >= 0.0 && neuro.epsilon_bonus <= 0.05,
            "epsilon_bonus out of range: {}", neuro.epsilon_bonus);
    }

    #[test]
    fn neuro_warm_start_serde_roundtrip() {
        // The NeuroState must survive JSON serialization cleanly.
        let original = NeuroState {
            dopamine: 0.72,
            noradrenaline: 0.61,
            serotonin: 0.44,
            acetylcholine: 0.55,
            low_pressure_streak: 7,
        };
        let json = serde_json::to_string(&original).expect("serialize NeuroState");
        let restored: NeuroState = serde_json::from_str(&json).expect("deserialize NeuroState");
        assert!((restored.dopamine - original.dopamine).abs() < 1e-10);
        assert!((restored.noradrenaline - original.noradrenaline).abs() < 1e-10);
        assert!((restored.serotonin - original.serotonin).abs() < 1e-10);
        assert!((restored.acetylcholine - original.acetylcholine).abs() < 1e-10);
        assert_eq!(restored.low_pressure_streak, original.low_pressure_streak);
    }

    #[test]
    fn neuro_default_snapshot_is_neutral() {
        // A freshly constructed neuromodulator should snapshot at 0.5 baseline.
        let neuro = ApolloNeuromodulator::new();
        let snap = neuro.snapshot();
        assert!((snap.dopamine - 0.5).abs() < 1e-10);
        assert!((snap.noradrenaline - 0.5).abs() < 1e-10);
        assert!((snap.serotonin - 0.5).abs() < 1e-10);
        assert!((snap.acetylcholine - 0.5).abs() < 1e-10);
        assert_eq!(snap.low_pressure_streak, 0);
    }

    #[test]
    fn neuro_restore_recomputes_derived_immediately() {
        // After restore(), derived params must be correct without a tick().
        let mut neuro = ApolloNeuromodulator::new();
        neuro.restore(NeuroState {
            dopamine: 1.0,
            noradrenaline: 1.0,
            serotonin: 1.0,
            acetylcholine: 1.0,
            low_pressure_streak: 0,
        });
        assert!((neuro.alpha_multiplier - 1.5).abs() < 1e-10,
            "alpha_multiplier should be 1.5 when dopamine=1.0, got {}", neuro.alpha_multiplier);
        assert_eq!(neuro.dyna_steps, 20,
            "dyna_steps should be 20 when noradrenaline=1.0, got {}", neuro.dyna_steps);
        assert!((neuro.serotonin_shift - 0.05).abs() < 1e-10,
            "serotonin_shift should be 0.05 when serotonin=1.0, got {}", neuro.serotonin_shift);
        assert!((neuro.epsilon_bonus - 0.05).abs() < 1e-10,
            "epsilon_bonus should be 0.05 when acetylcholine=1.0, got {}", neuro.epsilon_bonus);
    }

    #[test]
    fn neuro_restore_then_tick_is_stable() {
        // Restoring at non-neutral values should not cause divergence — the
        // leaky integrator damps back to ambient equilibrium within 50 ticks.
        let mut neuro = ApolloNeuromodulator::new();
        neuro.restore(NeuroState {
            dopamine: 1.0,
            noradrenaline: 1.0,
            serotonin: 0.0,
            acetylcholine: 0.0,
            low_pressure_streak: 0,
        });
        let mut s = default_signals(); // neutral inputs
        s.urgency = 0.2;
        for _ in 0..50 {
            neuro.tick(&s);
        }
        // After 50 ticks under neutral pressure, extreme values should have decayed.
        // Not asserting specific values — just that nothing is NaN or out of [0,1].
        let (da, na, se, ach) = neuro.levels();
        assert!(da.is_finite() && (0.0..=1.0).contains(&da), "dopamine not in range: {}", da);
        assert!(na.is_finite() && (0.0..=1.0).contains(&na), "noradrenaline not in range: {}", na);
        assert!(se.is_finite() && (0.0..=1.0).contains(&se), "serotonin not in range: {}", se);
        assert!(ach.is_finite() && (0.0..=1.0).contains(&ach), "acetylcholine not in range: {}", ach);
    }

    #[test]
    fn neuro_warm_start_better_than_cold_start() {
        // A warm-started neuromod should be closer to the steady-state of a
        // continuously-running neuromod than a cold-started one, measured after
        // a further 5 ticks.  This verifies the entire warm-start value proposition.
        let mut reference = ApolloNeuromodulator::new();
        let mut s = default_signals();
        s.pressure_drop = 0.30;
        s.urgency = 0.70;
        for _ in 0..30 {
            reference.tick(&s);
        }
        let steady_state_da = reference.dopamine;

        // Cold start.
        let mut cold = ApolloNeuromodulator::new();
        for _ in 0..5 {
            cold.tick(&s);
        }

        // Warm start from reference snapshot.
        let snap = reference.snapshot();
        let mut warm = ApolloNeuromodulator::new();
        warm.restore(snap);
        for _ in 0..5 {
            warm.tick(&s);
        }

        let cold_err = (cold.dopamine - steady_state_da).abs();
        let warm_err = (warm.dopamine - steady_state_da).abs();
        assert!(warm_err < cold_err,
            "warm-start (err={:.4}) should be closer to steady-state than cold-start (err={:.4})",
            warm_err, cold_err);
    }

    #[test]
    fn neuro_restore_streak_is_capped() {
        let mut neuro = ApolloNeuromodulator::new();
        neuro.restore(NeuroState {
            low_pressure_streak: u32::MAX,
            ..Default::default()
        });
        assert!(neuro.low_pressure_streak <= 1000,
            "streak not capped: {}", neuro.low_pressure_streak);
    }

    #[test]
    fn neuro_snapshot_restore_cold_is_neutral() {
        // Default snapshot (all zeros from Default) restores to safe values.
        let mut neuro = ApolloNeuromodulator::new();
        neuro.restore(NeuroState::default());
        // dopamine=0.0 → alpha_multiplier=0.5 (floor of range, not baseline but valid).
        assert!(neuro.alpha_multiplier >= 0.5,
            "alpha_multiplier below floor: {}", neuro.alpha_multiplier);
        assert!(neuro.dyna_steps >= 4,
            "dyna_steps below floor: {}", neuro.dyna_steps);
    }

    #[test]
    fn neuro_restore_nan_in_one_field_uses_neutral() {
        // If only one field is NaN, only that field falls back; others stay.
        let mut neuro = ApolloNeuromodulator::new();
        neuro.restore(NeuroState {
            dopamine: f64::NAN, // NaN → 0.5
            noradrenaline: 0.8,
            serotonin: 0.3,
            acetylcholine: 0.6,
            low_pressure_streak: 5,
        });
        assert_eq!(neuro.dopamine, 0.5, "NaN dopamine should default to 0.5");
        assert!((neuro.noradrenaline - 0.8).abs() < 1e-10, "noradrenaline should be preserved");
        assert!((neuro.serotonin - 0.3).abs() < 1e-10, "serotonin should be preserved");
        assert!((neuro.acetylcholine - 0.6).abs() < 1e-10, "acetylcholine should be preserved");
        assert_eq!(neuro.low_pressure_streak, 5);
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
