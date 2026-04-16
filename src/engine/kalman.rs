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

    // ── Innovation tracking (for auto-tuning R) ───────────────────────
    /// EMA of squared innovation y² (for R auto-tune: R_opt ≈ Var(y) - P[0,0]).
    /// Not serialized — rebuilds from live data after restart.
    #[serde(default)]
    residual_var_ema: f64,
    /// Number of innovation samples observed (for warm-up gating).
    #[serde(default)]
    residual_samples: u64,
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
            residual_var_ema: 0.0,
            residual_samples: 0,
        }
    }

    /// Predict + update con una nueva observación.
    ///
    /// - `measurement`: valor observado crudo.
    /// - `dt`: tiempo desde la última observación (segundos). Usar el dt real del ciclo.
    pub fn update(&mut self, measurement: f64, dt: f64) {
        // Guard: reject NaN/Infinity inputs to prevent permanent filter corruption.
        // [Crassidis & Junkins 2012] "Optimal Estimation" §3.6: a single NaN observation
        // propagates through Riccati recursion and corrupts all subsequent estimates.
        // Silently skip the update — last valid state is always better than NaN state.
        if !measurement.is_finite() || !dt.is_finite() {
            return;
        }

        if !self.initialized {
            // Primera observación: inicializar estado directamente.
            // p11 initialized to q (process noise) rather than 1.0 — more confident
            // that velocity starts near 0, reducing noise in early stable-period estimates.
            self.x = measurement;
            self.v = 0.0;
            self.p00 = self.r;
            self.p01 = 0.0;
            self.p11 = self.q;
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

        // Track innovation variance for R auto-tuning.
        // EMA α=0.05 → half-life ≈14 samples (~28s at 2s/cycle).
        let y_sq = y * y;
        if self.residual_samples < 5 {
            // Cold start: seed with first observations.
            self.residual_var_ema = y_sq;
        } else {
            self.residual_var_ema = 0.95 * self.residual_var_ema + 0.05 * y_sq;
        }
        self.residual_samples += 1;

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

        // Enforce positive semi-definiteness of the covariance matrix.
        // Numerical drift can push p00/p11 slightly negative, which causes
        // the Kalman gain to diverge on subsequent steps.
        // [Crassidis & Junkins 2012 §3.5] — clamp diagonal to a small positive floor.
        self.p00 = self.p00.max(1e-8);
        self.p11 = self.p11.max(1e-8);
        // p01 must satisfy |p01| ≤ sqrt(p00 * p11) (Cauchy-Schwarz for covariance).
        let p_cross_max = (self.p00 * self.p11).sqrt();
        self.p01 = self.p01.clamp(-p_cross_max, p_cross_max);
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

    /// Dynamically adjust process noise Q.
    /// Higher Q = filter adapts faster to signal changes (less lag, more noise).
    /// Lower Q = smoother but slower to track genuine regime shifts.
    /// Mirrors set_measurement_noise for symmetric adaptive tuning.
    pub fn set_process_noise(&mut self, q: f64) {
        self.q = q.max(1e-8); // safety floor
    }

    /// Auto-tune R from innovation variance.
    ///
    /// In a well-tuned Kalman filter, the innovation variance S = P[0,0] + R.
    /// If the empirical innovation variance (EMA of y²) exceeds S, the filter
    /// underestimates measurement noise → increase R. If it's below S, we can
    /// trust measurements more → decrease R.
    ///
    /// Returns the suggested R value, or None if not enough samples (< 20).
    /// The caller is responsible for clamping to a safe range.
    pub fn auto_tune_r(&self) -> Option<f64> {
        if self.residual_samples < 20 {
            return None;
        }
        // R_suggested = Var(innovation) - P[0,0]
        // Var(innovation) ≈ residual_var_ema (EMA of y²)
        let r_suggested = (self.residual_var_ema - self.p00).max(1e-6);
        Some(r_suggested)
    }

    /// Number of innovation samples collected (for warm-up gating).
    pub fn residual_samples(&self) -> u64 {
        self.residual_samples
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

    /// NaN measurement must not corrupt filter state.
    /// [Crassidis & Junkins 2012] §3.6: a single NaN propagates through Riccati
    /// recursion and permanently corrupts all subsequent estimates.
    #[test]
    fn test_nan_measurement_rejected() {
        let mut kf = Kalman1D::new(0.01, 0.1);
        kf.update(0.50, 0.5);
        assert!(kf.is_initialized());

        // Feed NaN — should be silently rejected.
        kf.update(f64::NAN, 0.5);
        assert!(kf.position().is_finite(), "NaN corrupted position");
        assert!(kf.velocity().is_finite(), "NaN corrupted velocity");
        assert!(
            (kf.position() - 0.50).abs() < 1e-10,
            "position changed after NaN"
        );
    }

    /// Infinity measurement must not corrupt filter state.
    #[test]
    fn test_infinity_measurement_rejected() {
        let mut kf = Kalman1D::new(0.01, 0.1);
        kf.update(0.50, 0.5);
        kf.update(0.52, 0.5);
        let pos_before = kf.position();

        kf.update(f64::INFINITY, 0.5);
        assert!(kf.position().is_finite(), "Inf corrupted position");
        assert!((kf.position() - pos_before).abs() < 1e-10);
    }

    /// NaN dt must not corrupt filter state.
    #[test]
    fn test_nan_dt_rejected() {
        let mut kf = Kalman1D::new(0.01, 0.1);
        kf.update(0.50, 0.5);
        let pos_before = kf.position();

        kf.update(0.55, f64::NAN);
        assert!(kf.position().is_finite());
        assert!((kf.position() - pos_before).abs() < 1e-10);
    }

    /// NaN on first observation must not initialize the filter.
    #[test]
    fn test_nan_first_observation_not_initialized() {
        let mut kf = Kalman1D::new(0.01, 0.1);
        kf.update(f64::NAN, 0.5);
        assert!(!kf.is_initialized(), "NaN should not initialize filter");
        // Subsequent valid observation should initialize correctly.
        kf.update(0.60, 0.5);
        assert!(kf.is_initialized());
        assert!((kf.position() - 0.60).abs() < 1e-10);
    }

    #[test]
    fn test_auto_tune_r_not_ready_before_warmup() {
        let mut kf = Kalman1D::new(0.005, 0.02);
        for i in 0..15 {
            kf.update(0.50 + (i as f64 * 0.001), 0.5);
        }
        assert!(kf.auto_tune_r().is_none(), "should need ≥20 samples");
    }

    #[test]
    fn test_auto_tune_r_produces_reasonable_value() {
        let mut kf = Kalman1D::new(0.005, 0.02);
        // Feed constant signal — innovation should be small.
        for _ in 0..50 {
            kf.update(0.50, 0.5);
        }
        let r = kf.auto_tune_r().unwrap();
        // R should be positive and reasonably small for a constant signal.
        assert!(r > 0.0, "R must be positive");
        assert!(r < 0.5, "R should be small for constant signal");
    }

    #[test]
    fn test_residual_samples_increment() {
        let mut kf = Kalman1D::new(0.01, 0.1);
        assert_eq!(kf.residual_samples(), 0);
        kf.update(0.50, 0.5);
        // First update initializes — no residual computed.
        assert_eq!(kf.residual_samples(), 0);
        kf.update(0.52, 0.5);
        assert_eq!(kf.residual_samples(), 1);
        kf.update(0.51, 0.5);
        assert_eq!(kf.residual_samples(), 2);
    }
}
