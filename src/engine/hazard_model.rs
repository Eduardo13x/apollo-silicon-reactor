//! Hazard Model — análisis de supervivencia para predecir P(OOM | estado actual).
//!
//! ## Modelo
//! Usa un hazard proporcional simplificado (inspirado en Cox, 1972):
//!
//!   h(t | x) = h₀(t) · exp(β · x)
//!
//! Donde:
//!   - h₀(t) = hazard base (tasa de overflows por hora, aprendida del historial)
//!   - x = vector de riesgo [pressure, swap_velocity, compressor_ratio]
//!   - β = pesos de riesgo (aprendidos online vía gradiente)
//!
//! La función de supervivencia:
//!   S(T | x) = exp(-∫₀ᵀ h(t|x) dt) ≈ exp(-h(x) · T)
//!
//! Esto da P(OOM en T segundos | x) = 1 - S(T | x).
//!
//! ## Ventaja sobre extrapolación lineal
//! - Calibrada con datos reales de overflows (no asume linealidad)
//! - Los pesos β aprenden qué señales realmente preceden a un OOM
//! - La probabilidad resultante es 0–1, más útil que "12 segundos" para tomar decisiones

use serde::{Deserialize, Serialize};

/// Número de features de riesgo.
const N_RISK: usize = 4;

/// Modelo de hazard proporcional para estimar P(OOM).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HazardModel {
    /// Tasa base de overflows por segundo (h₀). Aprendida del historial.
    base_rate: f64,
    /// Pesos de riesgo (β). Multiplicadores exponenciales por feature.
    beta: [f64; N_RISK],
    /// Learning rate para actualización online de β.
    lr: f64,
    /// Total de eventos observados (para estimar base_rate).
    total_events: u64,
    /// Total de horas de observación.
    total_hours: f64,
    /// Último hazard calculado (para diagnóstico).
    last_hazard: f64,
}

impl Default for HazardModel {
    fn default() -> Self {
        Self::new()
    }
}

impl HazardModel {
    /// Crea un modelo nuevo con priors conservadores.
    pub fn new() -> Self {
        Self {
            // Prior: 1 overflow cada 24 horas como baseline.
            base_rate: 1.0 / (24.0 * 3600.0),
            // Pesos iniciales: presión y swap tienen más importancia a priori.
            beta: [2.0, 1.5, 1.0, 0.5],
            lr: 0.01,
            total_events: 0,
            total_hours: 0.0,
            last_hazard: 0.0,
        }
    }

    /// Calcula el vector de riesgo a partir de señales del sistema.
    ///
    /// Todas las features se normalizan a ~0–1 para que los β sean comparables.
    pub fn risk_features(
        memory_pressure: f64,
        pressure_velocity: f64, // del Kalman, unidades/segundo
        swap_ratio: f64,        // swap_used / swap_total, 0–1
        compressor_ratio: f64,  // de collector.rs, 0–1
    ) -> [f64; N_RISK] {
        [
            memory_pressure.clamp(0.0, 1.0),
            // Velocidad: normalizar a 0–1 (0.1/s = bastante rápido para presión)
            (pressure_velocity / 0.1).clamp(0.0, 1.0),
            swap_ratio.clamp(0.0, 1.0),
            compressor_ratio.clamp(0.0, 1.0),
        ]
    }

    /// Calcula h(x) = h₀ · exp(β · x).
    fn hazard_rate(&self, features: &[f64; N_RISK]) -> f64 {
        let dot: f64 = self
            .beta
            .iter()
            .zip(features.iter())
            .map(|(b, x)| b * x)
            .sum();
        // Clamp el exponente para evitar overflow.
        let exp_val = dot.clamp(-10.0, 10.0);
        self.base_rate * exp_val.exp()
    }

    /// P(OOM en los próximos T segundos | estado actual).
    ///
    /// Retorna un valor calibrado 0–1.
    pub fn probability_oom(&mut self, features: &[f64; N_RISK], horizon_secs: f64) -> f64 {
        let h = self.hazard_rate(features);
        self.last_hazard = h;
        // S(T) = exp(-h · T), P(OOM) = 1 - S(T)
        let survival = (-h * horizon_secs).exp();
        1.0 - survival
    }

    /// Actualiza el modelo tras un overflow observado.
    ///
    /// - `features_at_event`: las features de riesgo en el momento del overflow.
    /// - `hours_since_last_event`: horas desde el overflow anterior (para actualizar base_rate).
    pub fn record_event(&mut self, features_at_event: &[f64; N_RISK], hours_since_last_event: f64) {
        self.total_events += 1;
        self.total_hours += hours_since_last_event;

        // Actualizar tasa base: MLE con prior suavizado (Laplace).
        if self.total_hours > 0.0 {
            // (events + 1) / (hours + 24) — prior de 1 evento cada 24h.
            self.base_rate =
                (self.total_events as f64 + 1.0) / ((self.total_hours + 24.0) * 3600.0);
        }

        // Gradient update para β: en un overflow, las features con valor alto
        // deberían tener β más alto (contribuyeron al riesgo).
        // ∂log L / ∂βⱼ = xⱼ - E[xⱼ] ≈ xⱼ - 0.5 (asumiendo media ~0.5 para features normalizadas)
        for (b, x) in self.beta.iter_mut().zip(features_at_event.iter()) {
            *b += self.lr * (x - 0.5);
            *b = b.clamp(0.0, 5.0); // mantener β positivo y acotado
        }
    }

    /// Tick de "no-evento": pasó un ciclo sin overflow.
    /// Actualiza total_hours para mantener base_rate calibrado.
    pub fn tick_no_event(&mut self, dt_secs: f64) {
        self.total_hours += dt_secs / 3600.0;
        // Recalcular base_rate periódicamente.
        if self.total_hours > 0.0 {
            self.base_rate =
                (self.total_events as f64 + 1.0) / ((self.total_hours + 24.0) * 3600.0);
        }
    }

    /// Último hazard rate calculado (para métricas/diagnóstico).
    pub fn last_hazard_rate(&self) -> f64 {
        self.last_hazard
    }

    /// Pesos β actuales (para observabilidad).
    pub fn beta_weights(&self) -> [f64; N_RISK] {
        self.beta
    }

    /// Tasa base actual (overflows/segundo).
    pub fn base_rate(&self) -> f64 {
        self.base_rate
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_risk_low_probability() {
        let mut model = HazardModel::new();
        let features = HazardModel::risk_features(0.3, 0.0, 0.0, 0.1);
        let p = model.probability_oom(&features, 30.0);
        assert!(p < 0.01, "low risk should give low P(OOM), got {}", p);
    }

    #[test]
    fn test_high_risk_high_probability() {
        let mut model = HazardModel::new();
        // Record some events to raise base_rate.
        for _ in 0..5 {
            let feat = HazardModel::risk_features(0.9, 0.08, 0.7, 0.8);
            model.record_event(&feat, 2.0);
        }
        let features = HazardModel::risk_features(0.95, 0.1, 0.8, 0.9);
        let p = model.probability_oom(&features, 30.0);
        assert!(
            p > 0.01,
            "high risk should give meaningful P(OOM), got {}",
            p
        );
    }

    #[test]
    fn test_longer_horizon_higher_probability() {
        let mut model = HazardModel::new();
        let features = HazardModel::risk_features(0.7, 0.05, 0.4, 0.5);
        let p5 = model.probability_oom(&features, 5.0);
        let p30 = model.probability_oom(&features, 30.0);
        let p120 = model.probability_oom(&features, 120.0);
        assert!(p5 < p30, "P(5s) < P(30s): {} < {}", p5, p30);
        assert!(p30 < p120, "P(30s) < P(120s): {} < {}", p30, p120);
    }

    #[test]
    fn test_record_event_increases_beta() {
        let mut model = HazardModel::new();
        let beta_before = model.beta;
        let feat = HazardModel::risk_features(0.9, 0.08, 0.7, 0.8);
        model.record_event(&feat, 1.0);
        // Feature 0 (pressure=0.9) > 0.5, so beta[0] should increase.
        assert!(
            model.beta[0] > beta_before[0],
            "beta[0] should increase for high-risk feature"
        );
    }

    #[test]
    fn test_tick_no_event_lowers_base_rate() {
        let mut model = HazardModel::new();
        let rate_before = model.base_rate();
        // Simulate 100 hours without events.
        model.tick_no_event(100.0 * 3600.0);
        assert!(
            model.base_rate() < rate_before,
            "base rate should decrease with observation time without events"
        );
    }

    #[test]
    fn test_probability_bounded_0_1() {
        let mut model = HazardModel::new();
        for _ in 0..20 {
            let feat = HazardModel::risk_features(0.99, 0.1, 0.99, 0.99);
            model.record_event(&feat, 0.5);
        }
        let features = HazardModel::risk_features(1.0, 0.1, 1.0, 1.0);
        let p = model.probability_oom(&features, 300.0);
        assert!(p >= 0.0 && p <= 1.0, "P should be in [0,1], got {}", p);
    }
}
