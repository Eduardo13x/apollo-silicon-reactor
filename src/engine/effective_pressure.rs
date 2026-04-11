/// Effective system memory pressure — the authoritative value for all decision-making.
///
/// # Problem
/// The raw `memory_pressure` from the kernel reflects only the compressor/wired ratio.
/// It misses hardware stress (bandwidth saturation, thermal throttling, battery state)
/// that makes the system *behaviorally* more constrained even at the same raw number.
///
/// # Solution
/// `compute()` aggregates all boost factors via additive sum with clamp(0.0, 1.0)
/// that were previously computed inline in `apollo-optimizerd/main.rs` and scattered
/// across separate decision sites. The resulting `effective` value is the one all
/// decision-makers should use.
///
/// # Threshold impact example
/// Default `bg_pressure` threshold = 0.78, `critical_pressure` = 0.88.
/// At raw pressure = 0.60 with moderate load (Phase1 thermal + Warning hw + battery):
///   0.60 + 0.07 + 0.15 + 0.04 = **0.86** → BackgroundPressure, not InteractiveFocus.
/// Without the boosts, `decide_actions` would classify this as InteractiveFocus and
/// skip all throttling — systemically under-aggressive in the 0.55–0.70 raw range.

/// All individual pressure contributions, retained for observability and debugging.
#[derive(Debug, Clone, Default)]
pub struct PressureComponents {
    /// Raw kernel memory pressure (compressor/wired ratio).
    pub base: f64,
    /// Hardware predictor boost: Warning=0.15, Critical=0.30.
    pub hardware: f64,
    /// Battery mode boost: Normal=0.04, LowPower=0.10, Critical=0.18.
    pub battery: f64,
    /// Thermal bailout phase: Phase1=0.07 … Phase4=0.40.
    pub thermal: f64,
    /// LLM inference detection boost (e.g. ollama/llama.cpp/MLX active).
    pub llm_workload: f64,
    /// Charging under high system power (>8W while charging): 0.06.
    pub charging_stress: f64,
    /// Battery time-to-empty < 20 min: 0.08.
    pub battery_low: f64,
    /// AMC memory bandwidth saturated (>80%): 0.10.
    pub memory_bandwidth: f64,
    /// SMC CPU temp: ≥80°C=0.05, ≥90°C=0.15, ≥100°C=0.30.
    pub smc_thermal: f64,
    /// Battery overheating flag: 0.12.
    pub battery_overheat: f64,
    /// Final effective pressure, clamped to [0.0, 1.0].
    pub effective: f64,
}

impl PressureComponents {
    /// Sum of all boost factors (not including base).
    pub fn total_boost(&self) -> f64 {
        self.hardware
            + self.battery
            + self.thermal
            + self.llm_workload
            + self.charging_stress
            + self.battery_low
            + self.memory_bandwidth
            + self.smc_thermal
            + self.battery_overheat
    }

    /// Name of the largest active boost factor, or "none" if all are zero.
    /// Useful for observability: shows WHY effective pressure exceeds raw pressure.
    pub fn dominant_factor(&self) -> &str {
        let factors = [
            (self.hardware, "hardware"),
            (self.battery, "battery"),
            (self.thermal, "thermal"),
            (self.llm_workload, "llm_workload"),
            (self.charging_stress, "charging_stress"),
            (self.battery_low, "battery_low"),
            (self.memory_bandwidth, "memory_bandwidth"),
            (self.smc_thermal, "smc_thermal"),
            (self.battery_overheat, "battery_overheat"),
        ];
        factors
            .iter()
            .filter(|(v, _)| *v >= 0.01)
            .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, name)| *name)
            .unwrap_or("none")
    }
}

/// Compute the effective system memory pressure including all boost factors.
///
/// This is the **authoritative** pressure value. All decision-making subsystems
/// (`decide_actions`, `page_reclaim`, `io_shaper`, `skill_registry`) should use
/// `effective` instead of `snapshot.pressure.memory_pressure` directly.
///
/// Returns `(effective_pressure, components)`. The components breakdown is provided
/// for observability (metrics, logging, AIS scoring) without additional computation cost.
///
/// # Arguments
/// * `base` — raw kernel memory pressure from `snapshot.pressure.memory_pressure`
/// * `hardware` — hw_predictor boost (0.0 / 0.15 / 0.30)
/// * `battery` — battery mode boost from `battery_pressure_boost()`
/// * `thermal` — thermal bailout phase boost (0.0 … 0.40)
/// * `llm_workload` — LLM inference detector boost
/// * `charging_stress` — charging + high wattage boost (0.0 or 0.06)
/// * `battery_low` — near-empty battery boost (0.0 or 0.08)
/// * `memory_bandwidth` — AMC bandwidth saturation boost (0.0 or 0.10)
/// * `smc_thermal` — SMC direct CPU temperature boost (0.0 / 0.05 / 0.15 / 0.30)
/// * `battery_overheat` — battery overheating boost (0.0 or 0.12)
pub fn compute(
    base: f64,
    hardware: f64,
    battery: f64,
    thermal: f64,
    llm_workload: f64,
    charging_stress: f64,
    battery_low: f64,
    memory_bandwidth: f64,
    smc_thermal: f64,
    battery_overheat: f64,
) -> (f64, PressureComponents) {
    debug_assert!(
        (0.0..=1.0).contains(&base),
        "base pressure out of range: {base}"
    );
    let effective = (base
        + hardware
        + battery
        + thermal
        + llm_workload
        + charging_stress
        + battery_low
        + memory_bandwidth
        + smc_thermal
        + battery_overheat)
        .clamp(0.0, 1.0);

    let components = PressureComponents {
        base,
        hardware,
        battery,
        thermal,
        llm_workload,
        charging_stress,
        battery_low,
        memory_bandwidth,
        smc_thermal,
        battery_overheat,
        effective,
    };

    (effective, components)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_boosts_returns_base() {
        let (eff, comp) = compute(0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(eff, 0.55);
        assert_eq!(comp.effective, 0.55);
        assert_eq!(comp.total_boost(), 0.0);
        assert_eq!(comp.base, 0.55);
    }

    #[test]
    fn all_boosts_maxed_clamps_to_one() {
        // hw=0.30, batt=0.18, thermal=0.40, llm=0.20, charging=0.06,
        // batt_low=0.08, mem_bw=0.10, smc_thermal=0.30, overheat=0.12
        // sum = 0.60 + 1.74 = 2.34 → clamped to 1.0
        let (eff, comp) = compute(0.60, 0.30, 0.18, 0.40, 0.20, 0.06, 0.08, 0.10, 0.30, 0.12);
        assert_eq!(eff, 1.0);
        assert_eq!(comp.effective, 1.0);
        // base is preserved even though effective is clamped
        assert_eq!(comp.base, 0.60);
        assert!(
            comp.total_boost() > 1.0,
            "boosts sum should exceed 1.0 in worst case"
        );
    }

    #[test]
    fn typical_moderate_scenario() {
        // base=0.60, Phase1Gentle thermal=0.07, Warning hw=0.15, Normal battery=0.04
        // → effective ≈ 0.86 (above bg_pressure=0.78 threshold)
        let (eff, comp) = compute(0.60, 0.15, 0.04, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let expected = 0.60 + 0.15 + 0.04 + 0.07;
        assert!(
            (eff - expected).abs() < 1e-9,
            "expected {expected}, got {eff}"
        );
        assert!(
            eff > 0.78,
            "effective pressure should exceed bg_pressure threshold (0.78), got {eff}"
        );
        assert_eq!(comp.hardware, 0.15);
        assert_eq!(comp.thermal, 0.07);
        assert_eq!(comp.battery, 0.04);
    }

    #[test]
    fn clamp_lower_bound() {
        // Negative inputs are not expected but must not produce sub-zero output.
        let (eff, _) = compute(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(eff, 0.0);
    }

    #[test]
    fn components_boost_sum_matches_manual() {
        let (_, comp) = compute(0.50, 0.15, 0.10, 0.25, 0.15, 0.06, 0.08, 0.10, 0.15, 0.12);
        let manual_sum = 0.15 + 0.10 + 0.25 + 0.15 + 0.06 + 0.08 + 0.10 + 0.15 + 0.12;
        assert!(
            (comp.total_boost() - manual_sum).abs() < 1e-9,
            "total_boost()={} manual={manual_sum}",
            comp.total_boost()
        );
    }

    #[test]
    fn hardware_boost_only() {
        let (eff, comp) = compute(0.50, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((eff - 0.65).abs() < 1e-9);
        assert_eq!(comp.hardware, 0.15);
    }

    #[test]
    fn thermal_boost_only() {
        let (eff, comp) = compute(0.40, 0.0, 0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((eff - 0.80).abs() < 1e-9);
        assert_eq!(comp.thermal, 0.40);
    }

    #[test]
    fn llm_boost_only() {
        let (eff, comp) = compute(0.55, 0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((eff - 0.75).abs() < 1e-9);
        assert_eq!(comp.llm_workload, 0.20);
    }

    #[test]
    fn battery_low_boost_only() {
        let (eff, comp) = compute(0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0);
        assert!((eff - 0.68).abs() < 1e-9);
        assert_eq!(comp.battery_low, 0.08);
    }

    #[test]
    fn smc_thermal_boost_only() {
        let (eff, comp) = compute(0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30, 0.0);
        assert!((eff - 0.80).abs() < 1e-9);
        assert_eq!(comp.smc_thermal, 0.30);
    }

    #[test]
    fn memory_bandwidth_boost_only() {
        let (eff, comp) = compute(0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0);
        assert!((eff - 0.65).abs() < 1e-9);
        assert_eq!(comp.memory_bandwidth, 0.10);
    }

    #[test]
    fn base_already_at_max_stays_one() {
        let (eff, comp) = compute(1.0, 0.30, 0.18, 0.40, 0.20, 0.06, 0.08, 0.10, 0.30, 0.12);
        assert_eq!(eff, 1.0);
        assert_eq!(comp.base, 1.0);
        // effective is clamped but base is preserved
        assert!(comp.total_boost() > 0.0);
    }

    #[test]
    fn additive_semantics_not_max() {
        // If compute used max(base, boost) this would return 0.30.
        // Additive semantics: 0.20 + 0.30 = 0.50
        let (eff, _) = compute(0.20, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((eff - 0.50).abs() < 1e-9, "must be additive: got {eff}");
    }

    #[test]
    fn base_preserved_in_components_when_clamped() {
        let base = 0.70;
        let (eff, comp) = compute(base, 0.30, 0.18, 0.40, 0.20, 0.06, 0.08, 0.10, 0.30, 0.12);
        assert_eq!(eff, 1.0, "should be clamped");
        assert_eq!(comp.base, base, "base must be preserved in components");
    }

    #[test]
    fn total_boost_zero_when_no_boosts() {
        let (_, comp) = compute(0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(comp.total_boost(), 0.0);
    }

    #[test]
    fn effective_matches_component_field() {
        let (eff, comp) = compute(0.45, 0.15, 0.04, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(
            eff, comp.effective,
            "returned value must match component field"
        );
    }
}
