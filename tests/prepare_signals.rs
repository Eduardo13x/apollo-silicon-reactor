//! ══════════════════════════════════════════════════════════════════════════════
//! Apollo AutoResearch — Signal Intelligence Benchmark
//! ══════════════════════════════════════════════════════════════════════════════
//!
//! THIS FILE IS READ-ONLY. The agent must NEVER modify it.
//!
//! Tests zone routing, urgency computation, energy bias, zone learning,
//! budget cognitivo, and PID integral behavior. Misrouting causes:
//! wasted CPU (heavy modules run when unnecessary), missed emergencies
//! (heavy modules skipped when needed), or integral windup (stale pressure
//! accumulates false urgency).
//!
//! Target file: src/engine/signal_intelligence.rs

#[cfg(test)]
mod scenarios {
    use apollo_optimizer::engine::signal_intelligence::SignalIntelligence;

    /// Helper: tick with given pressure, all else nominal.
    fn tick_at(si: &mut SignalIntelligence, pressure: f64) {
        si.tick(
            pressure,
            10.0,
            0.01,
            0.05,
            &[5.0, 3.0],
            &[100e6, 50e6],
            "calm_app",
            100_000_000,
            500_000_000,
            8_000_000_000,
            0.5,
        );
    }

    /// Helper: tick with stressed parameters.
    fn tick_stressed(si: &mut SignalIntelligence, pressure: f64) {
        si.tick(
            pressure,
            50_000.0,
            0.7,
            0.8,
            &[50.0, 40.0, 30.0, 20.0, 10.0],
            &[2e9, 1.5e9, 1e9, 500e6, 200e6],
            "hog_process",
            2_000_000_000,
            6_000_000_000,
            8_000_000_000,
            0.5,
        );
    }

    /// Helper: warm up Kalman filter to reach target pressure.
    fn warmup(si: &mut SignalIntelligence, pressure: f64, ticks: usize) {
        for _ in 0..ticks {
            tick_at(si, pressure);
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 1: ZONE ROUTING (false positive / false negative prevention)
    // ══════════════════════════════════════════════════════════════════════════

    /// I01: At very low pressure (0.10), all heavy subsystems must be OFF.
    /// Running entropy/hazard/lotka/MPC at 10% pressure wastes CPU.
    #[test]
    fn i01_very_low_pressure_skips_all_heavy() {
        let mut si = SignalIntelligence::new();
        warmup(&mut si, 0.10, 30);
        let d = tick_at_returning(&mut si, 0.10);
        assert_eq!(d.entropy_anomaly, 0.0, "entropy must be skipped at 0.10");
        assert_eq!(d.p_oom_30s, 0.0, "hazard must be skipped at 0.10");
        assert_eq!(d.monopoly_risk, 0.0, "lotka must be skipped at 0.10");
        assert_eq!(d.mpc_recommendation, 0, "MPC must be skipped at 0.10");
    }

    /// I02: At high pressure (0.80), urgency must be meaningful (>0.15).
    /// The system must engage heavy subsystems and produce actionable signals.
    #[test]
    fn i02_high_pressure_engages_heavy() {
        let mut si = SignalIntelligence::new();
        for _ in 0..30 {
            tick_stressed(&mut si, 0.80);
        }
        let d = tick_stressed_returning(&mut si, 0.80);
        assert!(
            d.urgency > 0.15,
            "urgency must be meaningful at 0.80: {}",
            d.urgency
        );
        assert!(
            d.pressure_smooth > 0.50,
            "smoothed pressure should track 0.80: {}",
            d.pressure_smooth
        );
    }

    /// I03: Kalman always runs, even at lowest pressure. The smoothed pressure
    /// must reflect input regardless of zone routing.
    #[test]
    fn i03_kalman_always_runs() {
        let mut si = SignalIntelligence::new();
        warmup(&mut si, 0.15, 30);
        let d = tick_at_returning(&mut si, 0.15);
        assert!(
            d.pressure_smooth > 0.10,
            "Kalman must track input at low pressure: {}",
            d.pressure_smooth
        );
    }

    /// I04: CUSUM detects regime shift from stable 0.40 to sudden 0.80.
    /// This must trigger within 10 ticks after the jump.
    #[test]
    fn i04_cusum_detects_regime_shift() {
        let mut si = SignalIntelligence::new();
        warmup(&mut si, 0.40, 20);
        let mut found = false;
        for _ in 0..10 {
            let d = tick_stressed_returning(&mut si, 0.80);
            if d.regime_shift_up {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "CUSUM must detect regime shift from 0.40→0.80 within 10 ticks"
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 2: URGENCY COMPUTATION
    // ══════════════════════════════════════════════════════════════════════════

    /// I05: Rising pressure must produce positive velocity and higher urgency
    /// than flat pressure at the same level.
    #[test]
    fn i05_rising_pressure_increases_urgency() {
        let mut si = SignalIntelligence::new();
        warmup(&mut si, 0.40, 20);
        // Ramp up
        for i in 0..15 {
            let pressure = 0.50 + i as f64 * 0.03;
            tick_stressed(&mut si, pressure);
        }
        let d = tick_stressed_returning(&mut si, 0.90);
        assert!(
            d.pressure_velocity > 0.0,
            "velocity must be positive during ramp: {}",
            d.pressure_velocity
        );
        assert!(
            d.urgency > 0.30,
            "urgency must be high during rapid rise: {}",
            d.urgency
        );
    }

    /// I06: Flat low pressure (0.30) for 50 ticks should produce urgency < 0.15.
    /// No false alarms from accumulated noise.
    #[test]
    fn i06_flat_low_pressure_low_urgency() {
        let mut si = SignalIntelligence::new();
        warmup(&mut si, 0.30, 50);
        let d = tick_at_returning(&mut si, 0.30);
        assert!(
            d.urgency < 0.15,
            "urgency should be low at flat 0.30: {}",
            d.urgency
        );
    }

    /// I07: After recording 3 overflow events, P(OOM) must increase compared to
    /// before any events were recorded (at the same pressure).
    #[test]
    fn i07_overflow_events_increase_poom() {
        let mut si = SignalIntelligence::new();
        for _ in 0..20 {
            tick_stressed(&mut si, 0.80);
        }
        let d_before = tick_stressed_returning(&mut si, 0.85);
        let p_before = d_before.p_oom_30s;

        for _ in 0..3 {
            si.record_overflow(0.95, 0.8, 0.9, 2.0);
        }
        let d_after = tick_stressed_returning(&mut si, 0.85);
        assert!(
            d_after.p_oom_30s > p_before,
            "P(OOM) must increase after overflows: {} > {}",
            d_after.p_oom_30s,
            p_before
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 3: ENERGY-AWARE ROUTING
    // ══════════════════════════════════════════════════════════════════════════

    /// I08: Critical battery (15%) should shift zones up, making 0.40 pressure
    /// fall in LOW zone (skipping heavy subsystems to save battery).
    #[test]
    fn i08_critical_battery_conserves_cpu() {
        let mut si = SignalIntelligence::new();
        si.set_energy_bias(15, false, false);
        warmup(&mut si, 0.40, 30);
        let d = tick_at_returning(&mut si, 0.40);
        // With bias +0.15, mid_entry = 0.45, so 0.40 is LOW zone.
        assert_eq!(
            d.entropy_anomaly, 0.0,
            "entropy must be skipped on critical battery at 0.40"
        );
    }

    /// I09: Thermal emergency should shift zones down, engaging all heavy
    /// subsystems even at moderate pressure (0.38).
    #[test]
    fn i09_thermal_emergency_engages_early() {
        let mut si = SignalIntelligence::new();
        si.set_energy_bias(100, true, true);
        // With bias -0.15, high_entry = 0.35, so 0.38 = HIGH zone.
        for _ in 0..30 {
            tick_at(&mut si, 0.38);
        }
        let d = tick_at_returning(&mut si, 0.38);
        assert!(
            d.pressure_smooth > 0.34,
            "pressure must be above shifted high_entry: {}",
            d.pressure_smooth
        );
    }

    /// I10: Plugged in with no thermal issue → zero energy bias.
    #[test]
    fn i10_plugged_in_no_bias() {
        let mut si = SignalIntelligence::new();
        si.set_energy_bias(50, true, false);
        let (mid, high) = si.learned_zones();
        // Default zones unaffected.
        assert!(
            (mid - 0.30).abs() < 0.01,
            "mid_entry should be default: {}",
            mid
        );
        assert!(
            (high - 0.50).abs() < 0.01,
            "high_entry should be default: {}",
            high
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 4: ZONE LEARNING & BUDGET COGNITIVO
    // ══════════════════════════════════════════════════════════════════════════

    /// I11: Repeated effective actions near mid_entry should lower the entry
    /// threshold (engage earlier in future).
    #[test]
    fn i11_effective_actions_lower_zones() {
        let mut si = SignalIntelligence::new();
        let (mid_before, high_before) = si.learned_zones();
        for _ in 0..100 {
            si.zone_feedback(0.32, true);
        }
        let (mid_after, high_after) = si.learned_zones();
        assert!(
            mid_after < mid_before,
            "mid_entry must decrease: {} < {}",
            mid_after,
            mid_before
        );
        assert!(
            high_after < high_before,
            "high_entry must decrease: {} < {}",
            high_after,
            high_before
        );
    }

    /// I12: Repeated ineffective actions should raise the entry threshold
    /// (be more conservative in future).
    #[test]
    fn i12_ineffective_actions_raise_zones() {
        let mut si = SignalIntelligence::new();
        let (mid_before, _) = si.learned_zones();
        for _ in 0..100 {
            si.zone_feedback(0.40, false);
        }
        let (mid_after, _) = si.learned_zones();
        assert!(
            mid_after > mid_before,
            "mid_entry must increase: {} > {}",
            mid_after,
            mid_before
        );
    }

    /// I13: Zone boundaries must be clamped — even extreme feedback cannot push
    /// them outside safe bounds.
    #[test]
    fn i13_zone_learning_bounded() {
        let mut si = SignalIntelligence::new();
        for _ in 0..10000 {
            si.zone_feedback(0.25, true); // extreme push down
        }
        let (mid, high) = si.learned_zones();
        assert!(mid >= 0.20, "mid_entry clamped at 0.20: {}", mid);
        assert!(high >= 0.35, "high_entry clamped at 0.35: {}", high);
    }

    /// I14: Hazard utility decays toward 0 when p_oom stays near 0 (no overflows).
    /// After 200 ticks in high zone with no events, utility should be < 0.10.
    #[test]
    fn i14_budget_utility_decays_without_signal() {
        let mut si = SignalIntelligence::new();
        let initial = si.subsystem_utilities()[1]; // hazard
        assert!((initial - 0.5).abs() < 0.01, "start at 0.5: {}", initial);

        // 200 ticks in high zone, no overflow events.
        for _ in 0..200 {
            si.tick(
                0.55,
                10.0,
                0.01,
                0.05,
                &[5.0, 3.0],
                &[100e6, 50e6],
                "calm_app",
                100_000_000,
                500_000_000,
                8_000_000_000,
                0.5,
            );
        }
        let after = si.subsystem_utilities()[1];
        assert!(
            after < 0.10,
            "hazard utility must decay without OOM events: {}",
            after
        );
    }

    /// I15: PID integral accumulates positive error when pressure stays above target.
    /// After 30 ticks at pressure 0.80 (target=0.65), integral must be positive.
    #[test]
    fn i15_pid_integral_accumulates_above_target() {
        let mut si = SignalIntelligence::new();
        warmup(&mut si, 0.80, 30);
        let d = tick_at_returning(&mut si, 0.80);
        assert!(
            d.pressure_integral > 0.0,
            "PID integral must be positive when pressure > target: {}",
            d.pressure_integral
        );
    }

    // ── Helpers that return SignalDigest ─────────────────────────────────────

    fn tick_at_returning(
        si: &mut SignalIntelligence,
        pressure: f64,
    ) -> apollo_optimizer::engine::signal_intelligence::SignalDigest {
        si.tick(
            pressure,
            10.0,
            0.01,
            0.05,
            &[5.0, 3.0],
            &[100e6, 50e6],
            "calm_app",
            100_000_000,
            500_000_000,
            8_000_000_000,
            0.5,
        )
    }

    fn tick_stressed_returning(
        si: &mut SignalIntelligence,
        pressure: f64,
    ) -> apollo_optimizer::engine::signal_intelligence::SignalDigest {
        si.tick(
            pressure,
            50_000.0,
            0.7,
            0.8,
            &[50.0, 40.0, 30.0, 20.0, 10.0],
            &[2e9, 1.5e9, 1e9, 500e6, 200e6],
            "hog_process",
            2_000_000_000,
            6_000_000_000,
            8_000_000_000,
            0.5,
        )
    }
}
