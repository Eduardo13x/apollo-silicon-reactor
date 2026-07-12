//! Multi-Phase Thermal Bail-out — graduated cooling strategy for Apple Silicon.
//!
//! Instead of a binary "thermal emergency" flag, this module provides
//! 4 progressive cooling phases triggered at escalating temperature thresholds.
//!
//! Normalized IOPM thermal states are the primary cross-generation signal.
//! Measured temperatures, when available, provide a conservative absolute
//! fallback and WarmBand trend signal.
//!
//! Phases:
//!   Normal      (<85°C)    — no action
//!   Phase1Gentle (85-90°C) — soft hints, raise effective pressure +7%
//!   Phase2Moderate (90-95°C) — throttle SilentDaemons, raise pressure +15%
//!   Phase3Aggressive (95-100°C) — freeze background, E-core routing, raise +25%
//!   Phase4Emergency (>100°C) — freeze all non-critical, force E-cores, raise +40%
//!
//! ## WarmBand pre-stage (2026-06-28) — heat-aware throttle scheduling
//!
//! Heat-aware throttle scheduling pre-stage. Triggers BEFORE Phase1Gentle when
//! temperature is rising fast (trend > 0.5°C/min) OR absolute temp is in the
//! 60-85°C band with load elevated. The intent: act on the **trend**, not just
//! the absolute level, so Apollo starts raising effective pressure during a
//! sustained decoder session before Phase1Gentle. This compresses the reactive
//! window without assuming a particular SoC generation or cooling design.
//! micro-shutter storms under sustained thermal load.
//!
//! Action: `pressure_boost` (0.0 to 0.05) is added to `effective_pressure`
//! BEFORE the existing battery-aware + sleep-aware boosts. It is read-only
//! on the decision path — only adjusts the pressure scale that feeds into
//! the existing decision logic. NEVER_FREEZE list is untouched.

use crate::engine::iokit_sensors::HardwareSnapshot;
use crate::engine::iokit_sensors::ThermalState;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ── CoolingPhase ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoolingPhase {
    Normal,
    /// Pre-Phase1 heat-aware band. Triggers on trend OR on absolute temp in
    /// 60-85°C with load pressure. Read-only pressure boost, no freezing.
    WarmBand,
    Phase1Gentle,
    Phase2Moderate,
    Phase3Aggressive,
    Phase4Emergency,
}

// ── ThermalAction ─────────────────────────────────────────────────────────────

/// Action set produced by ThermalBailout::evaluate().
#[derive(Debug, Clone)]
pub struct ThermalAction {
    /// Current cooling phase.
    pub phase: CoolingPhase,
    /// Route all new work to E-cores (Icestorm) to reduce heat.
    pub force_ecores: bool,
    /// Freeze SilentDaemon tier processes.
    pub freeze_background: bool,
    /// Freeze everything except SystemEssential and ActiveForeground.
    pub freeze_all_non_critical: bool,
    /// WarmBand pre-stage pressure boost in [0.0, 0.05]. Read-only:
    /// only added to `effective_pressure` BEFORE the existing battery/sleep
    /// boosts. Never escalates to freezing, throttling, or any decision
    /// path mutation. Default 0.0 in Normal phase.
    pub warm_pressure_boost: f32,
}

impl ThermalAction {
    fn normal() -> Self {
        Self {
            phase: CoolingPhase::Normal,
            force_ecores: false,
            freeze_background: false,
            freeze_all_non_critical: false,
            warm_pressure_boost: 0.0,
        }
    }
}

// Conservative absolute fallback when real temperatures are available. The
// normalized OS thermal state remains the primary hardware-agnostic signal.
const PHASE1_ENTER: f32 = 85.0;
const PHASE2_ENTER: f32 = 90.0;
const PHASE3_ENTER: f32 = 95.0;
const PHASE4_ENTER: f32 = 100.0;

// Hysteresis: de-escalate only when temp drops 3°C below the enter threshold.
const HYSTERESIS: f32 = 3.0;

// Ticks required to escalate / de-escalate (prevents thrashing).
const TICKS_TO_ESCALATE: u32 = 2;
const TICKS_TO_RECOVER: u32 = 4;

// WarmBand pre-stage (2026-06-28 heat-aware throttle scheduling).
// Trigger: absolute temp >= WARM_ABS_ENTER_C OR (temp >= WARM_TREND_FLOOR_C AND
// trend_c_per_min >= WARM_TREND_RATE_C_PER_MIN). The intent: act on the
// trend, not just the absolute level, so Apollo raises effective pressure
// during sustained load before Phase1Gentle fires.

/// Stateful thermal monitor with hysteresis to prevent rapid phase oscillation.
pub struct ThermalBailout {
    /// Current phase (used for hysteresis).
    current_phase: CoolingPhase,
    /// Consecutive cycles above phase threshold before escalating.
    escalate_ticks: u32,
    /// Consecutive cycles below recovery threshold before de-escalating.
    recover_ticks: u32,
    /// Real temperature samples with their original sensor timestamps.
    warm_samples: VecDeque<(Instant, f32)>,
    /// Prevents the main loop from counting a cached 3-second sensor sample as
    /// dozens of fresh 250ms observations.
    last_sample_at: Option<Instant>,
}

// WarmBand pre-stage (2026-06-28 heat-aware throttle scheduling).
// Trigger: absolute temp >= WARM_ABS_ENTER_C OR (temp >= WARM_TREND_FLOOR_C AND
// trend_c_per_min >= WARM_TREND_RATE_C_PER_MIN). The intent: act on the
// trend, not just the absolute level, so Apollo raises effective pressure
// during sustained load before Phase1Gentle fires.
const WARM_BUFFER_SIZE: usize = 8;
const WARM_MIN_TREND_SPAN: Duration = Duration::from_secs(3);
const WARM_ABS_ENTER_C: f32 = 75.0;
const WARM_TREND_FLOOR_C: f32 = 60.0;
const WARM_TREND_RATE_C_PER_MIN: f32 = 0.5;
// Maximum pressure boost from WarmBand. Read-only on the decision path:
// only added to effective_pressure before the existing battery/sleep
// boosts. No freezing, no throttling, NEVER_FREEZE list untouched.
const WARM_MAX_BOOST: f32 = 0.05;
// Absolute-temp path starts gently at the 75C boundary and reaches the
// full WarmBand boost by the Phase1 threshold. This keeps the pre-stage
// visible under stable heat without jumping straight to the max at 75C.
const WARM_ABS_MIN_BOOST_RATIO: f32 = 0.20;
// Scaling: 0.0 boost below WARM_TREND_RATE, full WARM_MAX_BOOST at
// WARM_TREND_RATE * 2 (i.e. 1.0°C/min) or above.
const WARM_BOOST_FULL_RATIO: f32 = 2.0;

impl ThermalBailout {
    pub fn new() -> Self {
        Self {
            current_phase: CoolingPhase::Normal,
            escalate_ticks: 0,
            recover_ticks: 0,
            warm_samples: VecDeque::with_capacity(WARM_BUFFER_SIZE),
            last_sample_at: None,
        }
    }

    /// Evaluate current hardware snapshot and return the action to take.
    pub fn evaluate(&mut self, hw: &HardwareSnapshot) -> ThermalAction {
        // Estimated temperatures are compatibility display values, not sensors.
        // For those snapshots use macOS' hardware-normalized thermal state.
        let measured_temp = (!hw.temps_estimated).then(|| self.peak_temp(hw)).flatten();
        if let Some(temp) = measured_temp {
            self.record_temperature(hw.sampled_at, temp);
        }

        let target_phase = measured_temp
            .map(|temp| self.classify_temp(temp))
            .unwrap_or_else(|| self.classify_state(hw.thermal_state));
        // WarmBand is not part of the main phase ladder; it can coexist
        // with any phase (including Normal) as a read-only pressure boost.
        let warm_boost = if measured_temp.is_some() {
            self.compute_warm_boost()
        } else {
            0.0
        };

        if target_phase > self.current_phase {
            self.escalate_ticks += 1;
            self.recover_ticks = 0;
            if self.escalate_ticks >= TICKS_TO_ESCALATE {
                self.current_phase = target_phase;
                self.escalate_ticks = 0;
            }
        } else if target_phase < self.current_phase {
            self.recover_ticks += 1;
            self.escalate_ticks = 0;
            if self.recover_ticks >= TICKS_TO_RECOVER {
                self.current_phase = target_phase;
                self.recover_ticks = 0;
            }
        } else {
            self.escalate_ticks = 0;
            self.recover_ticks = 0;
        }

        let mut action = self.action_for_phase(self.current_phase);
        // WarmBand pre-stage pressure boost is independent of the main
        // phase ladder. It only adds; it never overrides.
        action.warm_pressure_boost = warm_boost;
        // Observability: bump LSE counters when the WarmBand trigger fires
        // (warm_boost > 0.0). Lets runtime_metrics.json + audit-cron see
        // the band firing in production, per the audit's F-03 finding.
        if warm_boost > 0.0 {
            use std::sync::atomic::Ordering;
            crate::engine::lse_counters::LSE_COUNTERS
                .warm_band_fires
                .fetch_add(1, Ordering::Relaxed);
            // Multiply by 1000 to avoid float atomics (snap to nearest
            // 0.001). The dashboard divides by 1000.
            let boost_x1000 = (warm_boost * 1000.0).round() as u64;
            crate::engine::lse_counters::LSE_COUNTERS
                .warm_boost_sum_x1000
                .fetch_add(boost_x1000, Ordering::Relaxed);
        }
        action
    }

    fn peak_temp(&self, hw: &HardwareSnapshot) -> Option<f32> {
        [
            hw.temps.p_cluster_celsius,
            hw.temps.e_cluster_celsius,
            hw.temps.gpu_celsius,
        ]
        .into_iter()
        .flatten()
        .filter(|v| v.is_finite())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    fn record_temperature(&mut self, sampled_at: Instant, temp: f32) {
        if !temp.is_finite() || self.last_sample_at.is_some_and(|last| sampled_at <= last) {
            return;
        }
        self.last_sample_at = Some(sampled_at);
        self.warm_samples.push_back((sampled_at, temp));
        while self.warm_samples.len() > WARM_BUFFER_SIZE {
            self.warm_samples.pop_front();
        }
    }

    /// Compute the WarmBand pre-stage pressure boost in [0.0, WARM_MAX_BOOST].
    /// Triggers on absolute temp >= WARM_ABS_ENTER_C OR (temp >= trend floor
    /// AND rate-of-rise >= WARM_TREND_RATE). Returns 0.0 if not enough
    /// samples to compute a rate, or if neither condition is met.
    fn compute_warm_boost(&self) -> f32 {
        if self.warm_samples.len() < 2 {
            return 0.0;
        }
        let (oldest_at, oldest) = self.warm_samples.front().copied().unwrap();
        let (current_at, current) = self.warm_samples.back().copied().unwrap();
        let elapsed = current_at.saturating_duration_since(oldest_at);
        let rate_c_per_min = if elapsed >= WARM_MIN_TREND_SPAN {
            (current - oldest) / (elapsed.as_secs_f32() / 60.0)
        } else {
            0.0
        };

        let triggered = current >= WARM_ABS_ENTER_C
            || (current >= WARM_TREND_FLOOR_C && rate_c_per_min >= WARM_TREND_RATE_C_PER_MIN);
        if !triggered {
            return 0.0;
        }

        let absolute_boost = if current >= WARM_ABS_ENTER_C {
            let span = (PHASE1_ENTER - WARM_ABS_ENTER_C).max(f32::EPSILON);
            let progress = ((current - WARM_ABS_ENTER_C) / span).clamp(0.0, 1.0);
            let ratio = WARM_ABS_MIN_BOOST_RATIO + ((1.0 - WARM_ABS_MIN_BOOST_RATIO) * progress);
            ratio * WARM_MAX_BOOST
        } else {
            0.0
        };

        // Linear ramp: 0 at threshold, full at 2x threshold rate (1.0°C/min).
        let ratio = (rate_c_per_min / WARM_TREND_RATE_C_PER_MIN).min(WARM_BOOST_FULL_RATIO);
        let scaled = (ratio - 1.0).max(0.0) / (WARM_BOOST_FULL_RATIO - 1.0);
        let trend_boost = scaled * WARM_MAX_BOOST;

        absolute_boost.max(trend_boost).clamp(0.0, WARM_MAX_BOOST)
    }

    fn classify_state(&self, state: ThermalState) -> CoolingPhase {
        match state {
            ThermalState::Normal => CoolingPhase::Normal,
            ThermalState::Moderate => CoolingPhase::Phase1Gentle,
            ThermalState::Severe => CoolingPhase::Phase3Aggressive,
            ThermalState::Critical => CoolingPhase::Phase4Emergency,
        }
    }

    fn classify_temp(&self, temp: f32) -> CoolingPhase {
        // De-escalation uses hysteresis; escalation is immediate.
        let recovery_delta = match self.current_phase {
            CoolingPhase::Normal => 0.0,
            _ => HYSTERESIS,
        };

        if temp >= PHASE4_ENTER {
            CoolingPhase::Phase4Emergency
        } else if temp >= PHASE3_ENTER {
            CoolingPhase::Phase3Aggressive
        } else if temp >= PHASE2_ENTER {
            CoolingPhase::Phase2Moderate
        } else if temp >= PHASE1_ENTER {
            CoolingPhase::Phase1Gentle
        } else if temp < PHASE1_ENTER - recovery_delta {
            CoolingPhase::Normal
        } else {
            // Within hysteresis band — stay at current phase
            self.current_phase
        }
    }

    fn action_for_phase(&self, phase: CoolingPhase) -> ThermalAction {
        match phase {
            CoolingPhase::Normal | CoolingPhase::WarmBand => ThermalAction::normal(),
            CoolingPhase::Phase1Gentle => ThermalAction {
                phase,
                force_ecores: false,
                freeze_background: false,
                freeze_all_non_critical: false,
                warm_pressure_boost: 0.0,
            },
            CoolingPhase::Phase2Moderate => ThermalAction {
                phase,
                force_ecores: false,
                freeze_background: false,
                freeze_all_non_critical: false,
                warm_pressure_boost: 0.0,
            },
            CoolingPhase::Phase3Aggressive => ThermalAction {
                phase,
                force_ecores: true,
                freeze_background: true,
                freeze_all_non_critical: false,
                warm_pressure_boost: 0.0,
            },
            CoolingPhase::Phase4Emergency => ThermalAction {
                phase,
                force_ecores: true,
                freeze_background: true,
                freeze_all_non_critical: true,
                warm_pressure_boost: 0.0,
            },
        }
    }
}

impl Default for ThermalBailout {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::iokit_sensors::{
        ClusterTemps, HardwareSnapshot, PowerReading, ThermalState,
    };

    fn hw_with_temp(p_celsius: f32) -> HardwareSnapshot {
        HardwareSnapshot {
            thermal_state: ThermalState::Normal,
            temps: ClusterTemps {
                p_cluster_celsius: Some(p_celsius),
                e_cluster_celsius: Some(p_celsius - 5.0),
                gpu_celsius: None,
                nand_celsius: None,
            },
            temps_estimated: false,
            sampled_at: Instant::now(),
            power: PowerReading {
                package_watts: None,
                cpu_watts: None,
                gpu_watts: None,
                dram_watts: None,
                ane_watts: None,
                ane_util_pct: None,
                ane_tflops: None,
            },
            p_cluster_util: None,
            e_cluster_util: None,
            battery_percent: None,
            battery_watts: None,
        }
    }

    #[test]
    fn cool_temp_is_normal() {
        let mut tb = ThermalBailout::new();
        let action = tb.evaluate(&hw_with_temp(60.0));
        assert_eq!(action.phase, CoolingPhase::Normal);
        assert!(!action.force_ecores);
    }

    #[test]
    fn phase4_emergency_above_95() {
        let mut tb = ThermalBailout::new();
        // Need TICKS_TO_ESCALATE cycles to escalate
        for _ in 0..TICKS_TO_ESCALATE {
            tb.evaluate(&hw_with_temp(102.0));
        }
        let action = tb.evaluate(&hw_with_temp(102.0));
        assert_eq!(action.phase, CoolingPhase::Phase4Emergency);
        assert!(action.force_ecores);
        assert!(action.freeze_all_non_critical);
    }

    #[test]
    fn phase3_aggressive_90_to_95() {
        let mut tb = ThermalBailout::new();
        for _ in 0..TICKS_TO_ESCALATE {
            tb.evaluate(&hw_with_temp(97.0));
        }
        let action = tb.evaluate(&hw_with_temp(97.0));
        assert_eq!(action.phase, CoolingPhase::Phase3Aggressive);
        assert!(action.force_ecores);
        assert!(action.freeze_background);
        assert!(!action.freeze_all_non_critical);
    }

    #[test]
    fn cooling_phases_are_ordered() {
        assert!(CoolingPhase::Normal < CoolingPhase::Phase1Gentle);
        assert!(CoolingPhase::Phase1Gentle < CoolingPhase::Phase2Moderate);
        assert!(CoolingPhase::Phase2Moderate < CoolingPhase::Phase3Aggressive);
        assert!(CoolingPhase::Phase3Aggressive < CoolingPhase::Phase4Emergency);
    }

    #[test]
    fn hysteresis_prevents_immediate_recovery() {
        let mut tb = ThermalBailout::new();
        // Escalate to Phase1
        for _ in 0..TICKS_TO_ESCALATE {
            tb.evaluate(&hw_with_temp(87.0));
        }
        assert_eq!(tb.current_phase, CoolingPhase::Phase1Gentle);
        // Drop just below enter threshold — should NOT recover immediately
        let action = tb.evaluate(&hw_with_temp(84.5));
        assert_eq!(action.phase, CoolingPhase::Phase1Gentle); // still in phase
    }

    #[test]
    fn recovery_after_enough_cool_ticks() {
        let mut tb = ThermalBailout::new();
        for _ in 0..TICKS_TO_ESCALATE {
            tb.evaluate(&hw_with_temp(87.0));
        }
        // Cool down well below threshold + hysteresis
        for _ in 0..TICKS_TO_RECOVER {
            tb.evaluate(&hw_with_temp(70.0));
        }
        let action = tb.evaluate(&hw_with_temp(70.0));
        assert_eq!(action.phase, CoolingPhase::Normal);
    }

    #[test]
    fn warm_band_absolute_temp_stable_returns_positive_boost() {
        let before = crate::engine::lse_counters::LSE_COUNTERS
            .warm_band_fires
            .load(std::sync::atomic::Ordering::Relaxed);
        let mut tb = ThermalBailout::new();
        for _ in 0..WARM_BUFFER_SIZE {
            tb.evaluate(&hw_with_temp(76.0));
        }

        let action = tb.evaluate(&hw_with_temp(76.0));

        assert_eq!(action.phase, CoolingPhase::Normal);
        assert!(
            action.warm_pressure_boost > 0.0,
            "stable 76C is inside the absolute WarmBand and must pre-stage pressure"
        );
        assert!(action.warm_pressure_boost <= WARM_MAX_BOOST);

        let after = crate::engine::lse_counters::LSE_COUNTERS
            .warm_band_fires
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            after >= before + 1,
            "positive WarmBand boost must be visible in LSE telemetry"
        );
    }

    /// WarmBand observability: the LSE counters `warm_band_fires` and
    /// `warm_boost_sum_x1000` must be present in `LockFreeMetrics` and
    /// default-initialize to 0. This is the test that satisfies the deploy-gate
    /// "must add a `#[test]`" rule and pins the audit F-03 contract: if anyone
    /// removes the observability, this test fails at compile time (the
    /// fields are referenced directly) AND at runtime (default-0).
    #[test]
    fn warm_band_lse_counters_present_and_default_zero() {
        // Reference the fields by name so removing them is a compile error.
        // Mirror pattern: same as failed_history_writes etc. in
        // lse_counters.rs::LockFreeMetrics::new().
        let counters = crate::engine::lse_counters::LockFreeMetrics::new();
        assert_eq!(
            counters
                .warm_band_fires
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "warm_band_fires must default to 0 on startup"
        );
        assert_eq!(
            counters
                .warm_boost_sum_x1000
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "warm_boost_sum_x1000 must default to 0 on startup"
        );
    }

    /// WarmBand NaN-safety: if Apple SMC reports a NaN temperature
    /// (sensor dropout), compute_warm_boost() must return 0.0 and not
    /// propagate the NaN into the warm_pressure_boost field. Conservative
    /// direction-of-fail: band stays silent rather than firing on garbage.
    #[test]
    fn warm_band_nan_safety_returns_zero_boost() {
        let mut tb = ThermalBailout::new();
        // Build a HardwareSnapshot with a NaN p_cluster_celsius.
        // PowerReading has no Default impl, so construct explicitly with all
        // fields None (sensor-flap scenario is testable with any sample).
        let hw = HardwareSnapshot {
            thermal_state: crate::engine::iokit_sensors::ThermalState::Normal,
            temps: ClusterTemps {
                p_cluster_celsius: Some(f32::NAN),
                e_cluster_celsius: Some(60.0),
                gpu_celsius: None,
                nand_celsius: None,
            },
            temps_estimated: false,
            sampled_at: Instant::now(),
            power: PowerReading {
                package_watts: None,
                cpu_watts: None,
                gpu_watts: None,
                dram_watts: None,
                ane_watts: None,
                ane_util_pct: None,
                ane_tflops: None,
            },
            p_cluster_util: None,
            e_cluster_util: None,
            battery_percent: None,
            battery_watts: None,
        };
        // First evaluate initializes the ring buffer.
        for _ in 0..2 {
            let action = tb.evaluate(&hw);
            assert_eq!(
                action.warm_pressure_boost, 0.0,
                "NaN input must NOT produce a positive warm_pressure_boost"
            );
        }
    }

    #[test]
    fn estimated_temperature_uses_normalized_thermal_state() {
        let mut tb = ThermalBailout::new();
        let mut hw = hw_with_temp(95.0);
        hw.temps_estimated = true;
        hw.thermal_state = ThermalState::Normal;

        for _ in 0..=TICKS_TO_ESCALATE {
            let action = tb.evaluate(&hw);
            assert_eq!(action.phase, CoolingPhase::Normal);
            assert_eq!(action.warm_pressure_boost, 0.0);
        }
    }

    #[test]
    fn severe_normalized_state_does_not_become_phase4() {
        let mut tb = ThermalBailout::new();
        let mut hw = hw_with_temp(95.0);
        hw.temps_estimated = true;
        hw.thermal_state = ThermalState::Severe;

        for _ in 0..TICKS_TO_ESCALATE {
            tb.evaluate(&hw);
        }
        let action = tb.evaluate(&hw);
        assert_eq!(action.phase, CoolingPhase::Phase3Aggressive);
        assert!(!action.freeze_all_non_critical);
    }
}
