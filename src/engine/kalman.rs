//! Kalman Filter 1D con modelo posición + velocidad.
//!
//! Modelo de estado:
//!   x = [posición, velocidad]'
//!   F = [[1, dt], [0, 1]]     (transición: posición += velocidad * dt)
//!   H = [1, 0]                (observamos solo posición)
//!
//! Filtra ruido de medición y estima la derivada (velocidad de cambio)
//! de señales como memory_pressure, swap_delta, jitter_us.
//!
//! ## Uso
//! ```ignore
//! let mut kf = Kalman1D::new(0.01, 0.1); // process_noise, measurement_noise
//! kf.update(0.72, 0.5); // (measurement, dt_seconds)
//! let smoothed = kf.position();
//! let rate = kf.velocity(); // derivada: unidades/segundo
//! let predicted = kf.predict_ahead(5.0); // ¿dónde estará en 5s?
//! ```

use serde::{Deserialize, Serialize};

/// Filtro de Kalman 1D: estado = [posición, velocidad].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kalman1D {
    // ── Estado ───────────────────────────────────────────────────────────
    /// Posición estimada (valor suavizado de la señal).
    x: f64,
    /// Velocidad estimada (derivada: unidades/segundo).
    v: f64,

    // ── Covarianza (2×2 simétrica, almacenada como 3 escalares) ─────────
    /// P[0,0]: varianza de posición.
    p00: f64,
    /// P[0,1] = P[1,0]: covarianza posición-velocidad.
    p01: f64,
    /// P[1,1]: varianza de velocidad.
    p11: f64,

    // ── Parámetros del modelo ────────────────────────────────────────────
    /// Varianza del ruido de proceso (q). Controla cuánto confiamos en el modelo
    /// vs las mediciones. Valor bajo = filtro más suave, más lag.
    q: f64,
    /// Varianza del ruido de medición (r). Valor alto = mediciones ruidosas,
    /// el filtro confía más en el modelo.
    r: f64,

    /// true una vez que recibimos al menos una observación.
    initialized: bool,
}

impl Kalman1D {
    /// Crea un filtro nuevo.
    ///
    /// - `process_noise` (q): cuánto esperamos que la señal cambie por segundo².
    ///   Típico: 0.001–0.05 para señales lentas (presión), 0.1–1.0 para rápidas (jitter).
    /// - `measurement_noise` (r): varianza del ruido de medición.
    ///   Típico: 0.01–0.1 para presión (rango 0–1), 100–10000 para jitter (rango 0–50000).
    pub fn new(process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            x: 0.0,
            v: 0.0,
            p00: 1.0,
            p01: 0.0,
            p11: 1.0,
            q: process_noise,
            r: measurement_noise,
            initialized: false,
        }
    }

    /// Predict + update con una nueva observación.
    ///
    /// - `measurement`: valor observado crudo.
    /// - `dt`: tiempo desde la última observación (segundos). Usar el dt real del ciclo.
    pub fn update(&mut self, measurement: f64, dt: f64) {
        if !self.initialized {
            // Primera observación: inicializar estado directamente.
            self.x = measurement;
            self.v = 0.0;
            self.p00 = self.r;
            self.p01 = 0.0;
            self.p11 = 1.0;
            self.initialized = true;
            return;
        }

        let dt = dt.max(0.001); // floor para evitar dt=0

        // ── Predict ──────────────────────────────────────────────────────
        // x_pred = F * x = [x + v*dt, v]
        let x_pred = self.x + self.v * dt;
        let v_pred = self.v;

        // P_pred = F * P * F' + Q
        // Q se escala con dt: bloques proporcionales a [dt³/3, dt²/2; dt²/2, dt]
        let q_dt3 = self.q * dt * dt * dt / 3.0;
        let q_dt2 = self.q * dt * dt / 2.0;
        let q_dt1 = self.q * dt;

        let p00_pred = self.p00 + dt * (self.p01 + self.p01 + dt * self.p11) + q_dt3;
        let p01_pred = self.p01 + dt * self.p11 + q_dt2;
        let p11_pred = self.p11 + q_dt1;

        // ── Update ───────────────────────────────────────────────────────
        // Innovation: y = z - H * x_pred = measurement - x_pred
        let y = measurement - x_pred;

        // S = H * P_pred * H' + R = p00_pred + R
        let s = p00_pred + self.r;
        if s.abs() < 1e-30 {
            return; // degenerate — skip update
        }
        let s_inv = 1.0 / s;

        // Kalman gain: K = P_pred * H' / S = [p00_pred/S, p01_pred/S]
        let k0 = p00_pred * s_inv;
        let k1 = p01_pred * s_inv;

        // x_new = x_pred + K * y
        self.x = x_pred + k0 * y;
        self.v = v_pred + k1 * y;

        // P_new = (I - K*H) * P_pred
        self.p00 = p00_pred - k0 * p00_pred;
        self.p01 = p01_pred - k0 * p01_pred;
        self.p11 = p11_pred - k1 * p01_pred;
    }

    /// Posición estimada (valor suavizado).
    pub fn position(&self) -> f64 {
        self.x
    }

    /// Velocidad estimada (derivada: unidades por segundo).
    /// Positivo = señal subiendo, negativo = bajando.
    pub fn velocity(&self) -> f64 {
        self.v
    }

    /// Predice el valor en `dt_ahead` segundos usando el modelo lineal.
    pub fn predict_ahead(&self, dt_ahead: f64) -> f64 {
        self.x + self.v * dt_ahead
    }

    /// ¿Se ha inicializado con al menos una observación?
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Dynamically adjust measurement noise R.
    /// Lower R = trust measurements more. Higher R = trust predictions more.
    /// Used by KPC IPC modulation: low IPC (memory-bound) → lower R.
    pub fn set_measurement_noise(&mut self, r: f64) {
        self.r = r.max(1e-6); // safety floor
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_observation_sets_position() {
        let mut kf = Kalman1D::new(0.01, 0.1);
        assert!(!kf.is_initialized());
        kf.update(0.72, 0.5);
        assert!(kf.is_initialized());
        assert!((kf.position() - 0.72).abs() < 1e-10);
        assert!((kf.velocity()).abs() < 1e-10);
    }

    #[test]
    fn test_constant_signal_converges() {
        let mut kf = Kalman1D::new(0.01, 0.1);
        // Feed constant signal with noise.
        for _ in 0..100 {
            kf.update(0.50, 0.5);
        }
        assert!((kf.position() - 0.50).abs() < 0.01);
        assert!(kf.velocity().abs() < 0.01);
    }

    #[test]
    fn test_rising_signal_positive_velocity() {
        let mut kf = Kalman1D::new(0.01, 0.05);
        // Feed linearly rising signal: 0.5, 0.52, 0.54, ...
        for i in 0..50 {
            let val = 0.5 + i as f64 * 0.02;
            kf.update(val, 0.5);
        }
        // Velocity should be positive (~0.04/s since Δ=0.02 per 0.5s).
        assert!(
            kf.velocity() > 0.03,
            "velocity={} should be > 0.03",
            kf.velocity()
        );
    }

    #[test]
    fn test_predict_ahead_rising() {
        let mut kf = Kalman1D::new(0.01, 0.05);
        for i in 0..30 {
            let val = 0.5 + i as f64 * 0.02;
            kf.update(val, 0.5);
        }
        let pred_5s = kf.predict_ahead(5.0);
        // Should be higher than current position.
        assert!(pred_5s > kf.position());
    }

    #[test]
    fn test_noisy_signal_smoothed() {
        let mut kf = Kalman1D::new(0.005, 0.1);
        let mut raw_devs = 0.0;
        let mut filt_devs = 0.0;
        let true_val = 0.60;
        // Noisy signal around 0.60.
        let noise = [
            0.05, -0.08, 0.12, -0.03, 0.09, -0.11, 0.04, -0.06, 0.07, -0.02, 0.10, -0.07, 0.03,
            -0.09, 0.06, -0.04, 0.08, -0.05, 0.01, -0.10,
        ];
        for &n in &noise {
            let raw = true_val + n;
            kf.update(raw, 0.5);
            raw_devs += (raw - true_val).abs();
            filt_devs += (kf.position() - true_val).abs();
        }
        // Filtered should have less total deviation than raw.
        assert!(
            filt_devs < raw_devs,
            "filtered {} should be < raw {}",
            filt_devs,
            raw_devs
        );
    }
}
