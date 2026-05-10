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

    /// Clear estimated state (position/velocity/covariance) but preserve
    /// learned q/r noise parameters. Use after a wake from sleep where the
    /// pre-sleep state is stale and would inject false velocity into the
    /// next measurement [Crassidis & Junkins 2012, §3.7].
    pub fn reset_state(&mut self) {
        self.x = 0.0;
        self.v = 0.0;
        self.p00 = 1.0;
        self.p01 = 0.0;
        self.p11 = 1.0;
        self.initialized = false;
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

// ── KalmanMV8 — 8-dimensional multivariate Kalman filter ────────────────────

const MV8_D: usize = 8;

/// Process noise diagonal Q.
/// [memory_pressure, pressure_velocity, swap_norm, thrashing_norm,
///  ode_net_rate, ode_t_sat, cpu_saturation, thermal_stress]
const MV8_Q: [f64; MV8_D] = [0.005, 0.005, 0.015, 0.015, 0.015, 0.015, 0.020, 0.001];
/// Measurement noise diagonal R.
const MV8_R: [f64; MV8_D] = [0.020, 0.020, 0.050, 0.050, 0.015, 0.015, 0.080, 0.010];
/// swap_norm → pressure_velocity coupling in F.
const MV8_ALPHA: f64 = 0.05;
/// thrashing_norm → pressure_velocity coupling in F.
const MV8_BETA: f64 = 0.05;
/// t_sat_urgency → ode_net_rate coupling in F.
const MV8_GAMMA: f64 = 0.10;

fn mv8_default_p() -> [f64; 64] {
    let mut p = [0.0f64; 64];
    for i in 0..MV8_D {
        p[i * MV8_D + i] = MV8_Q[i];
    }
    p
}

fn mv8_default_r_scale() -> f64 {
    1.0
}

#[inline(always)]
fn mv8_mat_mul(a: &[f64; 64], b: &[f64; 64]) -> [f64; 64] {
    let mut c = [0.0f64; 64];
    for i in 0..MV8_D {
        for k in 0..MV8_D {
            let aik = a[i * MV8_D + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..MV8_D {
                c[i * MV8_D + j] += aik * b[k * MV8_D + j];
            }
        }
    }
    c
}

#[inline(always)]
fn mv8_mat_transpose(a: &[f64; 64]) -> [f64; 64] {
    let mut t = [0.0f64; 64];
    for i in 0..MV8_D {
        for j in 0..MV8_D {
            t[j * MV8_D + i] = a[i * MV8_D + j];
        }
    }
    t
}

/// S = a + diag(d). Only touches diagonal elements.
#[inline(always)]
fn mv8_add_diag(a: &[f64; 64], d: &[f64; MV8_D]) -> [f64; 64] {
    let mut s = *a;
    for i in 0..MV8_D {
        s[i * MV8_D + i] += d[i];
    }
    s
}

#[inline(always)]
fn mv8_mat_vec_mul(m: &[f64; 64], v: &[f64; MV8_D]) -> [f64; MV8_D] {
    let mut r = [0.0f64; MV8_D];
    for i in 0..MV8_D {
        for j in 0..MV8_D {
            r[i] += m[i * MV8_D + j] * v[j];
        }
    }
    r
}

fn mv8_identity() -> [f64; 64] {
    let mut m = [0.0f64; 64];
    for i in 0..MV8_D {
        m[i * MV8_D + i] = 1.0;
    }
    m
}

/// Gauss-Jordan inversion of 8×8 matrix. Returns None if near-singular (pivot < 1e-12).
fn mv8_mat_inv(m: &[f64; 64]) -> Option<[f64; 64]> {
    // Augmented [m | I], 8 rows × 16 cols.
    let cols = MV8_D * 2;
    let mut aug = [0.0f64; MV8_D * MV8_D * 2];
    for i in 0..MV8_D {
        for j in 0..MV8_D {
            aug[i * cols + j] = m[i * MV8_D + j];
        }
        aug[i * cols + MV8_D + i] = 1.0;
    }
    for col in 0..MV8_D {
        // Partial pivot.
        let mut max_row = col;
        let mut max_val = aug[col * cols + col].abs();
        for row in (col + 1)..MV8_D {
            let v = aug[row * cols + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        if max_row != col {
            for j in 0..cols {
                aug.swap(col * cols + j, max_row * cols + j);
            }
        }
        let pivot = aug[col * cols + col];
        for j in 0..cols {
            aug[col * cols + j] /= pivot;
        }
        for row in 0..MV8_D {
            if row == col {
                continue;
            }
            let factor = aug[row * cols + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..cols {
                let sub = aug[col * cols + j] * factor;
                aug[row * cols + j] -= sub;
            }
        }
    }
    let mut inv = [0.0f64; 64];
    for i in 0..MV8_D {
        for j in 0..MV8_D {
            inv[i * MV8_D + j] = aug[i * cols + MV8_D + j];
        }
    }
    Some(inv)
}

/// Build F matrix for time step dt.
/// F = I + kinematic and cross-signal coupling terms.
fn mv8_build_f(dt: f64) -> [f64; 64] {
    let mut f = mv8_identity();
    f[1] = dt; // [0,1] pressure_smooth += velocity * dt
    f[MV8_D + 2] = MV8_ALPHA; // [1,2] swap_norm → pressure_velocity
    f[MV8_D + 3] = MV8_BETA; // [1,3] thrashing_norm → pressure_velocity
    f[4 * MV8_D + 5] = MV8_GAMMA; // [4,5] t_sat_urgency → ode_net_rate
    f
}

/// 8-dimensional multivariate Kalman filter fusing memory pressure + ODE signals.
///
/// State vector indices:
///   0: memory_pressure      1: pressure_velocity    2: swap_norm
///   3: thrashing_norm       4: ode_net_rate         5: ode_t_sat_urgency
///   6: cpu_saturation       7: thermal_stress
///
/// H = I₈ (all states directly observable each cycle).
/// Zero heap allocation — all matrices stored as `[f64; 64]` on the stack.
///
/// Q diagonal: [0.005, 0.005, 0.015, 0.015, 0.015, 0.015, 0.020, 0.001]
/// R diagonal: [0.020, 0.020, 0.050, 0.050, 0.015, 0.015, 0.080, 0.010]
/// P init = diag(Q). Covariance (P) is not serialized — reconverges in ~10 cycles.
///
/// [Welch & Bishop 2006] "An Introduction to the Kalman Filter"
/// [Kalman 1960] "A New Approach to Linear Filtering and Prediction Problems"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanMV8 {
    x: [f64; MV8_D],
    #[serde(skip, default = "mv8_default_p")]
    p: [f64; 64],
    initialized: bool,
    /// Cycles since first observation. Not serialized — warmup re-runs after restart.
    #[serde(skip)]
    warmup_cycles: u64,
    /// KPC IPC-derived R scale for pressure dimensions [0,1]. Default=1.0.
    #[serde(skip, default = "mv8_default_r_scale")]
    kpc_r_scale: f64,
}

impl Default for KalmanMV8 {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanMV8 {
    pub fn new() -> Self {
        Self {
            x: [0.0; MV8_D],
            p: mv8_default_p(),
            initialized: false,
            warmup_cycles: 0,
            kpc_r_scale: 1.0,
        }
    }

    /// Predict: x = F*x, P = F*P*F' + Q.
    pub fn predict(&mut self, dt: f64) {
        if !self.initialized || !dt.is_finite() {
            return;
        }
        let f = mv8_build_f(dt.max(0.001));
        let ft = mv8_mat_transpose(&f);
        self.x = mv8_mat_vec_mul(&f, &self.x);
        let fp = mv8_mat_mul(&f, &self.p);
        self.p = mv8_mat_mul(&fp, &ft);
        // P += Q (diagonal process noise)
        for i in 0..MV8_D {
            self.p[i * MV8_D + i] += MV8_Q[i];
            self.p[i * MV8_D + i] = self.p[i * MV8_D + i].max(1e-9);
        }
    }

    /// Update with observation z (H=I, so y = z - x).
    /// Returns false if z contains non-finite values or S is singular.
    pub fn update(&mut self, z: &[f64; MV8_D]) -> bool {
        if z.iter().any(|v| !v.is_finite()) {
            return false;
        }
        if !self.initialized {
            self.x = *z;
            // P stays as diag(Q) from new()
            self.initialized = true;
            return true;
        }
        self.warmup_cycles += 1;
        // y = z - x (H=I)
        let mut y = [0.0f64; MV8_D];
        for i in 0..MV8_D {
            y[i] = z[i] - self.x[i];
        }
        // S = P + diag(R), with KPC IPC scaling on pressure dimensions [0,1].
        let mut r = MV8_R;
        r[0] *= self.kpc_r_scale;
        r[1] *= self.kpc_r_scale;
        let s = mv8_add_diag(&self.p, &r);
        let s_inv = match mv8_mat_inv(&s) {
            Some(inv) => inv,
            None => return false,
        };
        // K = P * S^{-1}
        let k = mv8_mat_mul(&self.p, &s_inv);
        // x += K * y
        let ky = mv8_mat_vec_mul(&k, &y);
        for i in 0..MV8_D {
            self.x[i] += ky[i];
        }
        // P = (I - K) * P
        let mut ik = mv8_identity();
        for i in 0..MV8_D {
            for j in 0..MV8_D {
                ik[i * MV8_D + j] -= k[i * MV8_D + j];
            }
        }
        self.p = mv8_mat_mul(&ik, &self.p);
        // Enforce PSD: symmetrize + clamp diagonal.
        for i in 0..MV8_D {
            self.p[i * MV8_D + i] = self.p[i * MV8_D + i].max(1e-9);
            for j in (i + 1)..MV8_D {
                let sym = (self.p[i * MV8_D + j] + self.p[j * MV8_D + i]) * 0.5;
                self.p[i * MV8_D + j] = sym;
                self.p[j * MV8_D + i] = sym;
            }
        }
        true
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn state(&self) -> &[f64; MV8_D] {
        &self.x
    }

    /// Fused memory pressure estimate. Clamped to [0,1].
    pub fn memory_pressure(&self) -> f64 {
        self.x[0].clamp(0.0, 1.0)
    }

    /// Fused pressure velocity (cross-informed by swap + ODE coupling).
    pub fn pressure_velocity(&self) -> f64 {
        self.x[1]
    }

    pub fn swap_norm(&self) -> f64 {
        self.x[2].clamp(0.0, 1.0)
    }

    pub fn ode_net_rate(&self) -> f64 {
        self.x[4].clamp(0.0, 1.0)
    }

    pub fn ode_t_sat_urgency(&self) -> f64 {
        self.x[5].clamp(0.0, 1.0)
    }

    pub fn cpu_saturation(&self) -> f64 {
        self.x[6].clamp(0.0, 1.0)
    }

    pub fn thermal_stress(&self) -> f64 {
        self.x[7].clamp(0.0, 1.0)
    }

    /// Posterior variance (uncertainty) for state dimension i.
    pub fn variance(&self, i: usize) -> f64 {
        self.p[i * MV8_D + i]
    }

    /// Tr(P) / D — normalized Riccati trace.
    /// < 0.10 indicates filter has reconciled sensor noise with process noise.
    pub fn trace_per_dim(&self) -> f64 {
        let tr: f64 = (0..MV8_D).map(|i| self.p[i * MV8_D + i]).sum();
        tr / MV8_D as f64
    }

    /// P[1,1]: velocity dimension variance. Gate criterion for D-term switch.
    pub fn velocity_variance(&self) -> f64 {
        self.p[MV8_D + 1]
    }

    /// True when warmup ≥ 50 cycles AND Tr(P)/D < 0.10.
    /// [NotebookLM KalmanMV8 spec; mirrors RestoreQualityMonitor 50-cycle warmup]
    pub fn is_converged(&self) -> bool {
        self.initialized && self.warmup_cycles >= 50 && self.trace_per_dim() < 0.10
    }

    /// True when warmup ≥ 50 cycles AND P[1,1] ≤ Q[1].
    /// Gate for switching D-term PID from 1D Kalman velocity to MV8 velocity.
    pub fn velocity_converged(&self) -> bool {
        self.initialized && self.warmup_cycles >= 50 && self.velocity_variance() <= MV8_Q[1]
    }

    /// Linear blend factor: 0.0 (start) → 1.0 (full MV8) over 200 cycles.
    /// Use: `(1 - α)*x_1d + α*x_mv8` to avoid LinUCB feature shock.
    pub fn blend_alpha(&self) -> f64 {
        (self.warmup_cycles as f64 / 200.0).min(1.0)
    }

    /// Modulate R[0,1] (pressure dimensions) based on KPC IPC.
    /// Low IPC (memory-bound) → pressure signal more reliable → scale R down.
    /// [03b78fa wiring pattern; NotebookLM IPC-aware R scaling spec]
    pub fn set_kpc_ipc(&mut self, ipc: f64) {
        self.kpc_r_scale = if ipc <= 0.0 {
            1.0
        } else if ipc < 0.5 {
            0.4 // memory-bound: trust pressure measurements more
        } else if ipc > 1.5 {
            2.5 // compute-bound: pressure is noisy
        } else {
            1.0
        };
    }

    pub fn reset_state(&mut self) {
        self.x = [0.0; MV8_D];
        self.p = mv8_default_p();
        self.initialized = false;
        self.warmup_cycles = 0;
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

    // ── KalmanMV8 tests ──────────────────────────────────────────────────────

    #[test]
    fn test_mv8_new_uninitialized() {
        let kf = KalmanMV8::new();
        assert!(!kf.is_initialized());
        assert_eq!(kf.memory_pressure(), 0.0);
    }

    #[test]
    fn test_mv8_first_update_initializes() {
        let mut kf = KalmanMV8::new();
        let z = [0.5, 0.01, 0.1, 0.2, 0.1, 0.05, 0.3, 0.0];
        let ok = kf.update(&z);
        assert!(ok);
        assert!(kf.is_initialized());
        assert!((kf.memory_pressure() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mv8_constant_signal_converges() {
        let mut kf = KalmanMV8::new();
        let z = [0.60, 0.0, 0.1, 0.1, 0.1, 0.05, 0.3, 0.0];
        for _ in 0..100 {
            kf.predict(2.0);
            kf.update(&z);
        }
        assert!(
            (kf.memory_pressure() - 0.60).abs() < 0.02,
            "pressure converged to {} (expected 0.60)",
            kf.memory_pressure()
        );
        // Velocity should converge near 0 for constant input.
        assert!(
            kf.pressure_velocity().abs() < 0.05,
            "velocity {} should be near 0",
            kf.pressure_velocity()
        );
    }

    #[test]
    fn test_mv8_nan_rejected() {
        let mut kf = KalmanMV8::new();
        let z_good = [0.5, 0.0, 0.1, 0.1, 0.1, 0.05, 0.3, 0.0];
        kf.update(&z_good);
        let pos_before = kf.memory_pressure();

        let mut z_nan = z_good;
        z_nan[0] = f64::NAN;
        let ok = kf.update(&z_nan);
        assert!(!ok, "NaN update should return false");
        assert!((kf.memory_pressure() - pos_before).abs() < 1e-10);
    }

    #[test]
    fn test_mv8_pressure_velocity_cross_propagates() {
        // Rising memory pressure should push pressure_velocity positive via F kinematics.
        let mut kf = KalmanMV8::new();
        // Seed with some cycles then feed rising pressure.
        for i in 0..30 {
            let p = 0.40 + i as f64 * 0.01;
            let z = [p, 0.01 * i as f64, 0.05, 0.05, 0.05, 0.02, 0.3, 0.0];
            kf.predict(2.0);
            kf.update(&z);
        }
        // After 30 cycles of rising pressure, velocity estimate should be positive.
        assert!(
            kf.pressure_velocity() > 0.0,
            "velocity {} should be > 0 with rising pressure",
            kf.pressure_velocity()
        );
    }

    #[test]
    fn test_mv8_reset_clears_state() {
        let mut kf = KalmanMV8::new();
        let z = [0.7, 0.05, 0.2, 0.3, 0.1, 0.1, 0.4, 0.33];
        for _ in 0..10 {
            kf.predict(2.0);
            kf.update(&z);
        }
        assert!(kf.is_initialized());
        kf.reset_state();
        assert!(!kf.is_initialized());
        assert_eq!(kf.memory_pressure(), 0.0);
    }

    #[test]
    fn test_mv8_not_converged_before_warmup() {
        let mut kf = KalmanMV8::new();
        let z = [0.5, 0.0, 0.1, 0.1, 0.1, 0.05, 0.3, 0.0];
        // First update initializes but does NOT increment warmup_cycles.
        // Subsequent predict+update increments by 1 each.
        // Need 50 predict+update pairs to reach warmup_cycles=50.
        kf.update(&z);
        // Feed 49 predict+update → warmup_cycles=49: not yet converged.
        for _ in 0..49 {
            kf.predict(2.0);
            kf.update(&z);
        }
        assert!(
            !kf.is_converged(),
            "warmup={} should be < 50",
            kf.warmup_cycles
        );
        // 50th predict+update → warmup_cycles=50: converged.
        kf.predict(2.0);
        kf.update(&z);
        assert!(
            kf.is_converged(),
            "trace_per_dim={:.4} warmup={}",
            kf.trace_per_dim(),
            kf.warmup_cycles
        );
    }

    #[test]
    fn test_mv8_blend_alpha_ramps() {
        let mut kf = KalmanMV8::new();
        assert_eq!(kf.blend_alpha(), 0.0);
        let z = [0.5, 0.0, 0.1, 0.1, 0.1, 0.05, 0.3, 0.0];
        kf.update(&z);
        for _ in 0..99 {
            kf.predict(2.0);
            kf.update(&z);
        }
        // 100 update cycles → α = 0.5.
        let alpha = kf.blend_alpha();
        assert!(
            (alpha - 0.5).abs() < 0.01,
            "expected α≈0.5, got {:.3}",
            alpha
        );
    }

    #[test]
    fn test_mv8_kpc_ipc_scales_r() {
        let mut kf = KalmanMV8::new();
        // Memory-bound: R scale should drop to 0.4.
        kf.set_kpc_ipc(0.3);
        assert!((kf.kpc_r_scale - 0.4).abs() < 1e-10);
        // Compute-bound: R scale up.
        kf.set_kpc_ipc(2.0);
        assert!((kf.kpc_r_scale - 2.5).abs() < 1e-10);
        // Unknown: default.
        kf.set_kpc_ipc(0.0);
        assert!((kf.kpc_r_scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mv8_gauss_jordan_identity() {
        // Inverting the identity matrix should return identity.
        let i8 = mv8_identity();
        let inv = mv8_mat_inv(&i8).expect("identity is invertible");
        for r in 0..MV8_D {
            for c in 0..MV8_D {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (inv[r * MV8_D + c] - expected).abs() < 1e-10,
                    "inv[{},{}]={} expected {}",
                    r,
                    c,
                    inv[r * MV8_D + c],
                    expected
                );
            }
        }
    }
}
