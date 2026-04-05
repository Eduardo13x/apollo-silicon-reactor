//! CUSUM — Cumulative Sum change-point detector (Page, 1954).
//!
//! Detecta cambios en la media de una señal antes que un threshold fijo.
//! Mantiene dos acumuladores: uno para subidas (S⁺) y otro para bajadas (S⁻).
//!
//! Cuando S⁺ > h → la señal ha subido significativamente (regime shift up).
//! Cuando S⁻ > h → la señal ha bajado significativamente (regime shift down).
//!
//! ## Parámetros
//! - `target`: media esperada de la señal en régimen normal.
//! - `allowance` (k): desviación mínima para acumular. Filtra ruido pequeño.
//! - `threshold` (h): cuánta acumulación antes de alarma. Más alto = menos falsos positivos, más delay.
//!
//! ## Uso
//! ```ignore
//! let mut cs = Cusum::new(0.50, 0.02, 0.15); // target=0.50, k=0.02, h=0.15
//! cs.update(0.55); // slight increase, accumulates
//! if cs.alarm_high() { /* regime shift detected! */ }
//! ```

use serde::{Deserialize, Serialize};

/// Detector CUSUM bidireccional.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cusum {
    /// Media de referencia (régimen normal).
    target: f64,
    /// Allowance: desviación mínima antes de acumular (filtra ruido).
    k: f64,
    /// Threshold de alarma.
    h: f64,
    /// Acumulador positivo (detecta subidas).
    s_pos: f64,
    /// Acumulador negativo (detecta bajadas).
    s_neg: f64,
    /// Número de observaciones desde la última alarma o inicio.
    run_length: u32,
}

impl Cusum {
    /// Crea un detector CUSUM.
    ///
    /// - `target`: media esperada de la señal en estado normal.
    /// - `allowance` (k): tolerancia mínima (típico: 0.5 × desviación que quieres detectar).
    /// - `threshold` (h): umbral de alarma (típico: 4–5 × desviación estándar del ruido).
    pub fn new(target: f64, allowance: f64, threshold: f64) -> Self {
        Self {
            target,
            k: allowance,
            h: threshold,
            s_pos: 0.0,
            s_neg: 0.0,
            run_length: 0,
        }
    }

    /// Alimenta una nueva observación.
    pub fn update(&mut self, value: f64) {
        // Guard: reject NaN/Infinity to prevent accumulator corruption.
        // A single NaN poisons s_pos/s_neg permanently (NaN + x = NaN).
        if !value.is_finite() {
            return;
        }
        // S⁺ = max(0, S⁺ + (x - target - k))
        self.s_pos = (self.s_pos + (value - self.target - self.k)).max(0.0);
        // S⁻ = max(0, S⁻ + (target - k - x))  [equivalente: detecta bajadas]
        self.s_neg = (self.s_neg + (self.target - self.k - value)).max(0.0);
        self.run_length += 1;
    }

    /// ¿Alarma de subida? La señal ha drifteado significativamente por encima del target.
    pub fn alarm_high(&self) -> bool {
        self.s_pos > self.h
    }

    /// ¿Alarma de bajada? La señal ha drifteado significativamente por debajo del target.
    pub fn alarm_low(&self) -> bool {
        self.s_neg > self.h
    }

    /// Resetea los acumuladores (llamar después de actuar sobre una alarma).
    pub fn reset(&mut self) {
        self.s_pos = 0.0;
        self.s_neg = 0.0;
        self.run_length = 0;
    }

    /// Resetea y actualiza el target (cuando aprendemos un nuevo régimen normal).
    pub fn reset_target(&mut self, new_target: f64) {
        self.target = new_target;
        self.reset();
    }

    /// Acumulador positivo actual (diagnóstico).
    pub fn score_high(&self) -> f64 {
        self.s_pos
    }

    /// Acumulador negativo actual (diagnóstico).
    pub fn score_low(&self) -> f64 {
        self.s_neg
    }

    /// Observaciones desde último reset.
    pub fn run_length(&self) -> u32 {
        self.run_length
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_signal_no_alarm() {
        let mut cs = Cusum::new(0.50, 0.02, 0.15);
        for _ in 0..100 {
            cs.update(0.50);
        }
        assert!(!cs.alarm_high());
        assert!(!cs.alarm_low());
    }

    #[test]
    fn test_gradual_rise_triggers_alarm() {
        let mut cs = Cusum::new(0.50, 0.02, 0.15);
        // Feed small rises: 0.50, 0.53, 0.56, 0.59, ...
        // Each step adds (0.03 - 0.02) = 0.01 to S⁺. After 15+ steps: S⁺ > 0.15.
        let mut triggered_at = None;
        for i in 0..30 {
            cs.update(0.50 + i as f64 * 0.03);
            if cs.alarm_high() && triggered_at.is_none() {
                triggered_at = Some(i);
            }
        }
        assert!(
            triggered_at.is_some(),
            "CUSUM should trigger alarm on rising signal"
        );
        assert!(
            triggered_at.unwrap() < 15,
            "alarm should trigger early, got {}",
            triggered_at.unwrap()
        );
    }

    #[test]
    fn test_sudden_drop_triggers_low_alarm() {
        let mut cs = Cusum::new(0.50, 0.02, 0.10);
        // Suddenly drop to 0.30
        for _ in 0..10 {
            cs.update(0.30);
        }
        assert!(cs.alarm_low(), "Should detect drop below target");
        assert!(!cs.alarm_high(), "Should not detect rise");
    }

    #[test]
    fn test_reset_clears_state() {
        let mut cs = Cusum::new(0.50, 0.02, 0.10);
        for _ in 0..20 {
            cs.update(0.70);
        }
        assert!(cs.alarm_high());
        cs.reset();
        assert!(!cs.alarm_high());
        assert_eq!(cs.run_length(), 0);
    }

    #[test]
    fn test_noise_within_allowance_no_alarm() {
        let mut cs = Cusum::new(0.50, 0.03, 0.15);
        // Noise within ±0.03 of target → below allowance → no accumulation.
        let noise = [0.52, 0.48, 0.51, 0.49, 0.53, 0.47, 0.50, 0.52, 0.48, 0.51];
        for &v in &noise {
            cs.update(v);
        }
        assert!(!cs.alarm_high());
        assert!(!cs.alarm_low());
    }

    /// NaN input must not corrupt CUSUM accumulators.
    /// A single NaN poisons s_pos/s_neg permanently (NaN + x = NaN),
    /// causing all subsequent alarm_high()/alarm_low() to return false forever.
    #[test]
    fn test_nan_input_rejected() {
        let mut cs = Cusum::new(0.50, 0.02, 0.15);
        // Build up some accumulation.
        for _ in 0..5 {
            cs.update(0.60);
        }
        let s_pos_before = cs.score_high();
        assert!(s_pos_before > 0.0, "should have accumulated");

        // NaN should be silently rejected.
        cs.update(f64::NAN);
        assert!(
            cs.score_high().is_finite(),
            "NaN corrupted s_pos: {}",
            cs.score_high()
        );
        assert_eq!(cs.score_high(), s_pos_before, "s_pos should not change on NaN");
        assert_eq!(cs.run_length(), 5, "run_length should not increment on NaN");
    }

    /// Infinity input must not corrupt CUSUM accumulators.
    #[test]
    fn test_infinity_input_rejected() {
        let mut cs = Cusum::new(0.50, 0.02, 0.15);
        cs.update(0.55);
        let s_pos_before = cs.score_high();
        cs.update(f64::INFINITY);
        assert!(cs.score_high().is_finite());
        assert_eq!(cs.score_high(), s_pos_before);
    }

    /// reset_target should update the target and clear accumulators.
    #[test]
    fn test_reset_target_changes_baseline() {
        let mut cs = Cusum::new(0.50, 0.02, 0.15);
        for _ in 0..10 {
            cs.update(0.70);
        }
        assert!(cs.alarm_high());
        cs.reset_target(0.70);
        assert!(!cs.alarm_high());
        // Now feeding 0.70 should NOT trigger alarm (it's the new target).
        for _ in 0..20 {
            cs.update(0.70);
        }
        assert!(!cs.alarm_high(), "new target should prevent alarm");
    }
}
