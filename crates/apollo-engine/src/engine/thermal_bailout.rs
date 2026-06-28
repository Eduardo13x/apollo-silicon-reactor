//! Multi-Phase Thermal Bail-out — graduated cooling strategy for Apple Silicon.
//!
//! Instead of a binary "thermal emergency" flag, this module provides
//! 4 progressive cooling phases triggered at escalating temperature thresholds.
//!
//! M1 Air has no fan — acting 5-10°C before the hardware ceiling prevents visible
//! stutter caused by hardware-level frequency reduction at ~95°C.
//!
//! Phases:
//!   Normal      (<80°C)   — no action
//!   Phase1Gentle (80-85°C) — soft hints, raise effective pressure +7%
//!   Phase2Moderate (85-90°C) — throttle SilentDaemons, raise pressure +15%
//!   Phase3Aggressive (90-95°C) — freeze background, E-core routing, raise +25%
//!   Phase4Emergency (>95°C)  — freeze all non-critical, force E-cores, raise +40%
//!
//! ## WarmBand pre-stage (2026-06-28) — heat-aware throttle scheduling
//!
//! Heat-aware throttle scheduling pre-stage. Triggers BEFORE Phase1Gentle when
//! temperature is rising fast (trend > 0.5°C/min) OR absolute temp is in the
//! 60-80°C band with load elevated. The intent: act on the **trend**, not just
//! the absolute level, so Apollo starts raising effective pressure during a
//! 4K-decoder session before the M1 hits 80°C. This compresses the reactive
//! window of Phase1Gentle (which only fires at 80°C) and reduces the
//! micro-shutter storms under sustained thermal load.
//!
//! Action: `pressure_boost` (0.0 to 0.05) is added to `effective_pressure`
//! BEFORE the existing battery-aware + sleep-aware boosts. It is read-only
//! on the decision path — only adjusts the pressure scale that feeds into
//! the existing decision logic. NEVER_FREEZE list is untouched.

use crate::engine::iokit_sensors::HardwareSnapshot;

// ── CoolingPhase ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoolingPhase {
    Normal,
    /// Pre-Phase1 heat-aware band. Triggers on trend OR on absolute temp in
    /// 60-80°C with load pressure. Read-only pressure boost, no freezing.
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

// Temperature thresholds (°C) — tuned for M1 Air (fanless).
const PHASE1_ENTER: f32 = 80.0;
const PHASE2_ENTER: f32 = 85.0;
const PHASE3_ENTER: f32 = 90.0;
const PHASE4_ENTER: f32 = 95.0;

// Hysteresis: de-escalate only when temp drops 3°C below the enter threshold.
const HYSTERESIS: f32 = 3.0;

// Ticks required to escalate / de-escalate (prevents thrashing).
const TICKS_TO_ESCALATE: u32 = 2;
const TICKS_TO_RECOVER: u32 = 4;

// WarmBand pre-stage (2026-06-28 heat-aware throttle scheduling).
// Trigger: absolute temp >= WARM_ABS_ENTER_C OR (temp >= WARM_TREND_FLOOR_C AND
// trend_c_per_min >= WARM_TREND_RATE_C_PER_MIN). The intent: act on the
// trend, not just the absolute level, so Apollo raises effective pressure
// during a 4K-decoder session before Phase1Gentle fires at 80°C.

/// Stateful thermal monitor with hysteresis to prevent rapid phase oscillation.
pub struct ThermalBailout {
    /// Current phase (used for hysteresis).
    current_phase: CoolingPhase,
    /// Consecutive cycles above phase threshold before escalating.
    escalate_ticks: u32,
    /// Consecutive cycles below recovery threshold before de-escalating.
    recover_ticks: u32,
    /// WarmBand temperature ring buffer (last 8 samples, ~4s at 500ms cadence).
    /// Used to compute the rate-of-rise that triggers the pre-Phase1 band.
    warm_temps: [f32; 8],
    /// Next write index in the warm_temps ring (wraps).
    warm_idx: usize,
    /// Number of valid samples currently in the buffer (0..=8).
    warm_filled: usize,
}

// WarmBand pre-stage (2026-06-28 heat-aware throttle scheduling).
// Trigger: absolute temp >= WARM_ABS_ENTER_C OR (temp >= WARM_TREND_FLOOR_C AND
// trend_c_per_min >= WARM_TREND_RATE_C_PER_MIN). The intent: act on the
// trend, not just the absolute level, so Apollo raises effective pressure
// during a 4K-decoder session before Phase1Gentle fires at 80°C.
const WARM_ABS_ENTER_C: f32 = 75.0;
const WARM_TREND_FLOOR_C: f32 = 60.0;
const WARM_TREND_RATE_C_PER_MIN: f32 = 0.5;
// Maximum pressure boost from WarmBand. Read-only on the decision path:
// only added to effective_pressure before the existing battery/sleep
// boosts. No freezing, no throttling, NEVER_FREEZE list untouched.
const WARM_MAX_BOOST: f32 = 0.05;
// Scaling: 0.0 boost below WARM_TREND_RATE, full WARM_MAX_BOOST at
// WARM_TREND_RATE * 2 (i.e. 1.0°C/min) or above.
const WARM_BOOST_FULL_RATIO: f32 = 2.0;

impl ThermalBailout {
    pub fn new() -> Self {
        Self {
            current_phase: CoolingPhase::Normal,
            escalate_ticks: 0,
            recover_ticks: 0,
            // WarmBand trend buffer: keeps the last 8 temp samples (~4s at
            // 500ms cadence) to compute rate-of-rise. Empty on startup.
            warm_temps: [0.0; 8],
            warm_idx: 0,
            warm_filled: 0,
        }
    }

    /// Evaluate current hardware snapshot and return the action to take.
    pub fn evaluate(&mut self, hw: &HardwareSnapshot) -> ThermalAction {
        // Use the maximum of P-cluster and GPU temperature.
        let temp = self.peak_temp(hw);

        // Update the WarmBand temperature ring buffer.
        self.warm_temps[self.warm_idx] = temp;
        self.warm_idx = (self.warm_idx + 1) % self.warm_temps.len();
        if self.warm_filled < self.warm_temps.len() {
            self.warm_filled += 1;
        }

        let target_phase = self.classify_temp(temp);
        // WarmBand is not part of the main phase ladder; it can coexist
        // with any phase (including Normal) as a read-only pressure boost.
        let warm_boost = self.compute_warm_boost();

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
        action
    }

    fn peak_temp(&self, hw: &HardwareSnapshot) -> f32 {
        let p = hw.temps.p_cluster_celsius.unwrap_or(0.0);
        let e = hw.temps.e_cluster_celsius.unwrap_or(0.0);
        let g = hw.temps.gpu_celsius.unwrap_or(0.0);
        p.max(e).max(g)
    }

    /// Compute the WarmBand pre-stage pressure boost in [0.0, WARM_MAX_BOOST].
    /// Triggers on absolute temp >= WARM_ABS_ENTER_C OR (temp >= trend floor
    /// AND rate-of-rise >= WARM_TREND_RATE). Returns 0.0 if not enough
    /// samples to compute a rate, or if neither condition is met.
    fn compute_warm_boost(&self) -> f32 {
        if self.warm_filled < 2 {
            return 0.0;
        }
        let (current, oldest) = self.warm_trend_endpoints();
        let rate_per_cycle = (current - oldest) / ((self.warm_filled - 1) as f32).max(1.0);
        // Approximate cycles-per-minute from a typical 250-300ms cadence.
        // We use 250ms (2.4 Hz) for a slightly-conservative rate; actual
        // cadence is at least that.
        let rate_c_per_min = rate_per_cycle * 240.0_f32;

        let triggered = current >= WARM_ABS_ENTER_C
            || (current >= WARM_TREND_FLOOR_C && rate_c_per_min >= WARM_TREND_RATE_C_PER_MIN);
        if !triggered {
            return 0.0;
        }

        // Linear ramp: 0 at threshold, full at 2x threshold rate (1.0°C/min).
        let ratio = (rate_c_per_min / WARM_TREND_RATE_C_PER_MIN).min(WARM_BOOST_FULL_RATIO);
        let scaled = (ratio - 1.0).max(0.0) / (WARM_BOOST_FULL_RATIO - 1.0);
        (scaled * WARM_MAX_BOOST).clamp(0.0, WARM_MAX_BOOST)
    }

    /// Returns (newest, oldest) sample in the ring buffer.
    fn warm_trend_endpoints(&self) -> (f32, f32) {
        // Newest = slot BEFORE the next write index (warm_idx points to the
        // slot to overwrite next). Oldest = slot warm_filled steps back from
        // warm_idx.
        let n = self.warm_temps.len();
        let newest_idx = (self.warm_idx + n - 1) % n;
        let oldest_idx = (self.warm_idx + n - self.warm_filled) % n;
        (self.warm_temps[newest_idx], self.warm_temps[oldest_idx])
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
            tb.evaluate(&hw_with_temp(97.0));
        }
        let action = tb.evaluate(&hw_with_temp(97.0));
        assert_eq!(action.phase, CoolingPhase::Phase4Emergency);
        assert!(action.force_ecores);
        assert!(action.freeze_all_non_critical);
    }

    #[test]
    fn phase3_aggressive_90_to_95() {
        let mut tb = ThermalBailout::new();
        for _ in 0..TICKS_TO_ESCALATE {
            tb.evaluate(&hw_with_temp(92.0));
        }
        let action = tb.evaluate(&hw_with_temp(92.0));
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
            tb.evaluate(&hw_with_temp(82.0));
        }
        assert_eq!(tb.current_phase, CoolingPhase::Phase1Gentle);
        // Drop just below enter threshold — should NOT recover immediately
        let action = tb.evaluate(&hw_with_temp(79.5));
        assert_eq!(action.phase, CoolingPhase::Phase1Gentle); // still in phase
    }

    #[test]
    fn recovery_after_enough_cool_ticks() {
        let mut tb = ThermalBailout::new();
        for _ in 0..TICKS_TO_ESCALATE {
            tb.evaluate(&hw_with_temp(82.0));
        }
        // Cool down well below threshold + hysteresis
        for _ in 0..TICKS_TO_RECOVER {
            tb.evaluate(&hw_with_temp(70.0));
        }
        let action = tb.evaluate(&hw_with_temp(70.0));
        assert_eq!(action.phase, CoolingPhase::Normal);
    }
}
