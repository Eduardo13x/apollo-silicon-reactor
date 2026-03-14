//! Multi-Phase Thermal Bail-out — graduated cooling strategy for Apple Silicon.
//!
//! Instead of a binary "thermal emergency" flag at 95 °C, this module provides
//! 4 progressive cooling phases with different strategies:
//!
//!   Phase 1 (Gentle,    80–85 °C): Reduce background I/O, hint purgeable memory.
//!   Phase 2 (Moderate,  85–90 °C): Route all background to E-Cores, throttle GPU.
//!   Phase 3 (Aggressive, 90–95 °C): Freeze non-essential daemons, cut P-Core allocation.
//!   Phase 4 (Emergency,   >95 °C): Freeze everything non-critical, disable GPU compute.
//!
//! Each phase builds on the previous one (cumulative).  The module also provides
//! a hysteresis band (5 °C) to prevent rapid oscillation between phases.

use crate::engine::iokit_sensors::HardwareSnapshot;

/// Cooling phases in order of aggressiveness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CoolingPhase {
    /// Below 80 °C — no cooling intervention needed.
    Normal,
    /// 80–85 °C — gentle: reduce background I/O, send memory pressure hints.
    Phase1Gentle,
    /// 85–90 °C — moderate: force background to E-Cores, throttle GPU workloads.
    Phase2Moderate,
    /// 90–95 °C — aggressive: freeze non-essential daemons, cut P-Core allocation.
    Phase3Aggressive,
    /// >95 °C — emergency: freeze everything non-critical, disable GPU compute.
    Phase4Emergency,
}

/// Actions recommended by the thermal bail-out module.
#[derive(Debug, Clone)]
pub struct ThermalAction {
    pub phase: CoolingPhase,
    /// Force all non-foreground processes to E-Cores.
    pub force_ecores: bool,
    /// Send memory pressure hints to purgeable-heavy processes.
    pub send_pressure_hints: bool,
    /// Freeze background daemons (those in Stale/SilentDaemon tier).
    pub freeze_background: bool,
    /// Freeze everything except protected + foreground.
    pub freeze_all_non_critical: bool,
    /// Throttle GPU compute workloads (reduce IOTier for GPU-bound processes).
    pub throttle_gpu: bool,
    /// Recommended max P-Core utilisation target (0.0–1.0).
    /// The scheduler should try to keep P-Core load below this.
    pub p_core_cap: f32,
}

impl ThermalAction {
    fn normal() -> Self {
        Self {
            phase: CoolingPhase::Normal,
            force_ecores: false,
            send_pressure_hints: false,
            freeze_background: false,
            freeze_all_non_critical: false,
            throttle_gpu: false,
            p_core_cap: 1.0,
        }
    }
}

/// Stateful thermal bail-out evaluator with hysteresis.
pub struct ThermalBailout {
    current_phase: CoolingPhase,
    /// Hysteresis band in °C — must cool this much below threshold to de-escalate.
    hysteresis: f32,
}

impl ThermalBailout {
    pub fn new() -> Self {
        Self {
            current_phase: CoolingPhase::Normal,
            hysteresis: 5.0,
        }
    }

    /// Evaluate the current thermal state and return recommended actions.
    ///
    /// Uses P-cluster temperature as the primary signal.
    /// Falls back to thermal_state from the snapshot if temps are unavailable.
    pub fn evaluate(&mut self, snapshot: &HardwareSnapshot) -> ThermalAction {
        let p_temp = snapshot
            .temps
            .p_cluster_celsius
            .unwrap_or(self.fallback_temp(snapshot));

        // Escalation thresholds (going up).
        let new_phase = if p_temp >= 95.0 {
            CoolingPhase::Phase4Emergency
        } else if p_temp >= 90.0 {
            CoolingPhase::Phase3Aggressive
        } else if p_temp >= 85.0 {
            CoolingPhase::Phase2Moderate
        } else if p_temp >= 80.0 {
            CoolingPhase::Phase1Gentle
        } else {
            CoolingPhase::Normal
        };

        // De-escalation with hysteresis (going down).
        // Only drop phase if temp is hysteresis-band below the current phase's lower threshold.
        let effective_phase = if new_phase < self.current_phase {
            let deescalation_temp = match self.current_phase {
                CoolingPhase::Phase4Emergency => 95.0 - self.hysteresis,
                CoolingPhase::Phase3Aggressive => 90.0 - self.hysteresis,
                CoolingPhase::Phase2Moderate => 85.0 - self.hysteresis,
                CoolingPhase::Phase1Gentle => 80.0 - self.hysteresis,
                CoolingPhase::Normal => 0.0,
            };
            if p_temp <= deescalation_temp {
                new_phase
            } else {
                self.current_phase // hold current phase (hysteresis)
            }
        } else {
            new_phase
        };

        self.current_phase = effective_phase;
        self.action_for_phase(effective_phase)
    }

    /// Current cooling phase (for observability / metrics).
    pub fn current_phase(&self) -> CoolingPhase {
        self.current_phase
    }

    fn action_for_phase(&self, phase: CoolingPhase) -> ThermalAction {
        match phase {
            CoolingPhase::Normal => ThermalAction::normal(),

            CoolingPhase::Phase1Gentle => ThermalAction {
                phase,
                force_ecores: false,
                send_pressure_hints: true,
                freeze_background: false,
                freeze_all_non_critical: false,
                throttle_gpu: false,
                p_core_cap: 0.9,
            },

            CoolingPhase::Phase2Moderate => ThermalAction {
                phase,
                force_ecores: true,
                send_pressure_hints: true,
                freeze_background: false,
                freeze_all_non_critical: false,
                throttle_gpu: true,
                p_core_cap: 0.7,
            },

            CoolingPhase::Phase3Aggressive => ThermalAction {
                phase,
                force_ecores: true,
                send_pressure_hints: true,
                freeze_background: true,
                freeze_all_non_critical: false,
                throttle_gpu: true,
                p_core_cap: 0.4,
            },

            CoolingPhase::Phase4Emergency => ThermalAction {
                phase,
                force_ecores: true,
                send_pressure_hints: true,
                freeze_background: true,
                freeze_all_non_critical: true,
                throttle_gpu: true,
                p_core_cap: 0.1,
            },
        }
    }

    /// Estimate temperature from thermal_state when direct sensor data is unavailable.
    fn fallback_temp(&self, snapshot: &HardwareSnapshot) -> f32 {
        use crate::engine::iokit_sensors::ThermalState;
        match snapshot.thermal_state {
            ThermalState::Normal => 60.0,
            ThermalState::Moderate => 82.0,
            ThermalState::Severe => 92.0,
            ThermalState::Critical => 100.0,
        }
    }
}

impl Default for ThermalBailout {
    fn default() -> Self {
        Self::new()
    }
}
