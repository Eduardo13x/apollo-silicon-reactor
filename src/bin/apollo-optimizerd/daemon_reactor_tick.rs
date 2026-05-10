use apollo_engine::engine::build_tracker::{BuildPhase, BuildTracker};
use apollo_engine::engine::holt_winters::HoltWinters;
use apollo_engine::engine::signal_intelligence::SignalDigest;
use apollo_engine::engine::temporal_predictor::TemporalPredictor;
use apollo_engine::engine::window_sensor::{SessionPhase, WorkloadIntent};

pub struct ReactorTickInput<'a> {
    pub signal_digest: &'a SignalDigest,
    pub holt_winters: &'a HoltWinters,
    pub window_relief_cycles: &'a mut u32,
    pub win_session_phase: &'a SessionPhase,
    pub win_pressure_floor: f64,
    pub win_workload_intent: &'a WorkloadIntent,
    pub raw_pressure: f64,
    pub temporal_predictor: &'a TemporalPredictor,
    pub temporal_hour: u8,
    pub temporal_weekday: u8,
    pub build_tracker: &'a BuildTracker,
    pub amx_available: bool,
    pub llm_active: bool,
    pub base_reactor_weight: f64,
}

/// Compute the adaptive reactor weight based on session context and signal intelligence.
/// Ported from main.rs (lines 2594-2736) as part of Wave 39.
pub fn run_reactor_tick(input: ReactorTickInput) -> f64 {
    let mut reactor_weight = input.base_reactor_weight;
    let signal_digest = input.signal_digest;

    // Signal intelligence → reactor_weight boosting.
    // CUSUM regime shift: pressure drifting up significantly.
    if signal_digest.regime_shift_up {
        reactor_weight = (reactor_weight + 0.15).min(1.0);
    }
    // High composite urgency: multiple signals converging on danger.
    if signal_digest.urgency > 0.7 {
        reactor_weight = (reactor_weight + 0.1).min(1.0);
    }
    // Entropy anomaly: chaotic process distribution change.
    if signal_digest.entropy_anomaly > 2.0 {
        reactor_weight = (reactor_weight + 0.07).min(1.0);
    }
    // SDE sticky-swap: high σ at moderate pressure = oscillation harbinger.
    if signal_digest.swap_net_rate_volatility > 1_000_000.0
        && signal_digest.pressure_smooth > 0.35
        && signal_digest.pressure_smooth < 0.65
    {
        reactor_weight = (reactor_weight + 0.10).min(1.0);
    }
    // Lyapunov chaos: positive FTLE = exponential divergence in pressure trajectory.
    if signal_digest.lyapunov_exponent > 0.5 && signal_digest.pressure_smooth > 0.40 {
        reactor_weight = (reactor_weight + 0.08).min(1.0);
    }
    // Cumulative stress: chronic high urgency boosts reactor even at moderate snapshots.
    if signal_digest.cumulative_stress > 0.55 {
        reactor_weight = (reactor_weight + 0.07).min(1.0);
    }
    // HW seasonal anomaly: pressure far above what's normal for this hour.
    if signal_digest.hw_seasonal_anomaly > 1.5 && input.holt_winters.observations() >= 24 {
        reactor_weight = (reactor_weight + 0.06).min(1.0);
    }
    // Darwin-Boltzmann anomaly: learned pattern deviation.
    if signal_digest.transformer_anomaly > 0.5 {
        reactor_weight = (reactor_weight + 0.1).min(1.0);
    }
    // Feed-forward pressure relief: tabs closed or heavy app terminated.
    if *input.window_relief_cycles > 0 {
        reactor_weight = (reactor_weight - 0.25).max(0.0);
        *input.window_relief_cycles -= 1;
    }

    // Session phase feed-forward [Pirolli & Card 1999].
    if *input.win_session_phase == SessionPhase::Ramping {
        reactor_weight = (reactor_weight + 0.15).min(1.0);
    }

    // Pressure floor correction [Denning 1968].
    if input.win_pressure_floor > 0.08 && input.raw_pressure < input.win_pressure_floor + 0.15 {
        reactor_weight = (reactor_weight - input.win_pressure_floor * 0.5).max(0.0);
    }

    // Workload intent adjustments.
    match input.win_workload_intent {
        WorkloadIntent::AISession => {
            if input.raw_pressure < 0.85 {
                reactor_weight = (reactor_weight - 0.20).max(0.0);
            }
        }
        WorkloadIntent::ResearchSession => {
            if input.raw_pressure < 0.80 {
                reactor_weight = (reactor_weight - 0.10).max(0.0);
            }
        }
        WorkloadIntent::BuildSession => {
            reactor_weight = (reactor_weight + 0.10).min(1.0);
        }
        WorkloadIntent::MediaSession => {
            if input.raw_pressure < 0.75 {
                reactor_weight = (reactor_weight - 0.08).max(0.0);
            }
        }
        WorkloadIntent::General => {}
    }

    // Temporal pre-positioning [Denning 1968 Working Set Model].
    let temporal_headroom = input
        .temporal_predictor
        .pressure_headroom_for_incoming(input.temporal_hour, input.temporal_weekday);
    if temporal_headroom > 0.02 && !input.build_tracker.build_active {
        reactor_weight = (reactor_weight + temporal_headroom).min(1.0);
    }

    // Build progress [McKenney 2004].
    match input.build_tracker.phase {
        BuildPhase::Starting => {
            reactor_weight = (reactor_weight + 0.15).min(1.0);
        }
        BuildPhase::Finishing => {
            if input.raw_pressure < 0.80 {
                reactor_weight = (reactor_weight - 0.12).max(0.0);
            }
        }
        _ => {}
    }

    // G10 — AMX Proactive Steering.
    if input.amx_available && input.llm_active {
        reactor_weight = (reactor_weight + 0.15).min(1.0);
    }

    reactor_weight.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::engine::signal_intelligence::SignalIntelligence;
    use std::path::PathBuf;

    #[test]
    fn test_reactor_weight_clamping() {
        let mut si = SignalIntelligence::new();
        let digest = si.tick(
            0.5,
            0.0,
            0.05,
            0.1,
            &[10.0],
            &[500e6],
            "app",
            500_000_000,
            2_000_000_000,
            8_000_000_000,
            0.5,
        );
        let hw = HoltWinters::new();
        let tp = TemporalPredictor::new(PathBuf::from("/tmp/apollo_test_tp.json"));
        let bt = BuildTracker::new();
        let mut relief = 0;

        let input = ReactorTickInput {
            signal_digest: &digest,
            holt_winters: &hw,
            window_relief_cycles: &mut relief,
            win_session_phase: &SessionPhase::Settled,
            win_pressure_floor: 0.0,
            win_workload_intent: &WorkloadIntent::General,
            raw_pressure: 0.5,
            temporal_predictor: &tp,
            temporal_hour: 12,
            temporal_weekday: 1,
            build_tracker: &bt,
            amx_available: false,
            llm_active: false,
            base_reactor_weight: 0.5,
        };

        // Test base case
        assert_eq!(run_reactor_tick(input), 0.5);
    }
}
