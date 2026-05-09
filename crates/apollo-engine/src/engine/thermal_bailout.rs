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

use crate::engine::iokit_sensors::HardwareSnapshot;

// ── CoolingPhase ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoolingPhase {
    Normal,
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
}

impl ThermalAction {
    fn normal() -> Self {
        Self {
            phase: CoolingPhase::Normal,
            force_ecores: false,
            freeze_background: false,
            freeze_all_non_critical: false,
        }
    }
}

// ── ThermalBailout ────────────────────────────────────────────────────────────

/// Stateful thermal monitor with hysteresis to prevent rapid phase oscillation.
pub struct ThermalBailout {
    /// Current phase (used for hysteresis).
    current_phase: CoolingPhase,
    /// Consecutive cycles above phase threshold before escalating.
    escalate_ticks: u32,
    /// Consecutive cycles below recovery threshold before de-escalating.
    recover_ticks: u32,
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

impl ThermalBailout {
    pub fn new() -> Self {
        Self {
            current_phase: CoolingPhase::Normal,
            escalate_ticks: 0,
            recover_ticks: 0,
        }
    }

    /// Evaluate current hardware snapshot and return the action to take.
    pub fn evaluate(&mut self, hw: &HardwareSnapshot) -> ThermalAction {
        // Use the maximum of P-cluster and GPU temperature.
        let temp = self.peak_temp(hw);

        let target_phase = self.classify_temp(temp);

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

        self.action_for_phase(self.current_phase)
    }

    fn peak_temp(&self, hw: &HardwareSnapshot) -> f32 {
        let p = hw.temps.p_cluster_celsius.unwrap_or(0.0);
        let e = hw.temps.e_cluster_celsius.unwrap_or(0.0);
        let g = hw.temps.gpu_celsius.unwrap_or(0.0);
        p.max(e).max(g)
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
            CoolingPhase::Normal => ThermalAction::normal(),
            CoolingPhase::Phase1Gentle => ThermalAction {
                phase,
                force_ecores: false,
                freeze_background: false,
                freeze_all_non_critical: false,
            },
            CoolingPhase::Phase2Moderate => ThermalAction {
                phase,
                force_ecores: false,
                freeze_background: false,
                freeze_all_non_critical: false,
            },
            CoolingPhase::Phase3Aggressive => ThermalAction {
                phase,
                force_ecores: true,
                freeze_background: true,
                freeze_all_non_critical: false,
            },
            CoolingPhase::Phase4Emergency => ThermalAction {
                phase,
                force_ecores: true,
                freeze_background: true,
                freeze_all_non_critical: true,
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
