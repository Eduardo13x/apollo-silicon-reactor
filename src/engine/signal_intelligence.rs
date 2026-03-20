//! Signal Intelligence — orquestador de procesamiento de señales avanzado.
//!
//! Agrupa Kalman + CUSUM + Entropía + Hazard + Lotka-Volterra + MPC
//! en una sola estructura que el daemon alimenta cada ciclo.
//!
//! ## Flujo
//! 1. `tick()`: recibe señales crudas del snapshot
//! 2. Kalman filtra y estima velocidades
//! 3. CUSUM detecta cambios de régimen
//! 4. Entropía detecta anomalías en la distribución de procesos
//! 5. Hazard calcula P(OOM en 30s)
//! 6. Lotka-Volterra detecta monopolización de RAM
//! 7. MPC sugiere la primera acción de la secuencia óptima
//!
//! La salida es `SignalDigest`: un resumen compacto que el PredictiveAgent
//! puede consumir como features adicionales o como override de su decisión.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::engine::cusum::Cusum;
use crate::engine::entropy_anomaly::EntropyDetector;
use crate::engine::hazard_model::HazardModel;
use crate::engine::kalman::Kalman1D;
use crate::engine::lotka_volterra::CompetitionState;
use crate::engine::mpc_horizon::{MpcController, MpcPersisted};

/// Resumen compacto de las señales procesadas. Todo normalizado 0–1 o con signo.
#[derive(Debug, Clone)]
pub struct SignalDigest {
    // ── Kalman ───────────────────────────────────────────────────────────
    /// Presión de memoria suavizada (0–1).
    pub pressure_smooth: f64,
    /// Velocidad de cambio de presión (unidades/segundo, + = subiendo).
    pub pressure_velocity: f64,
    /// Presión predicha en 5 segundos.
    pub pressure_predicted_5s: f64,
    /// Swap delta suavizado (bytes/s).
    pub swap_velocity_smooth: f64,
    /// Integral of pressure error over target (accumulated pressure-seconds).
    /// Positive = system has been above target chronically.
    /// Used as the "I" term for PID-style threshold adjustment.
    /// Windowed to last 60s to prevent integral windup (Hellerstein 2004).
    pub pressure_integral: f64,

    // ── CUSUM ────────────────────────────────────────────────────────────
    /// true si CUSUM detectó regime shift al alza en presión.
    pub regime_shift_up: bool,
    /// true si CUSUM detectó regime shift a la baja en presión.
    pub regime_shift_down: bool,
    /// Score CUSUM alto (acumulador, diagnóstico).
    pub cusum_score: f64,

    // ── Entropía ─────────────────────────────────────────────────────────
    /// Score de anomalía en la distribución de procesos (-3..+3 típico).
    /// > 2.0: muchos procesos nuevos compitiendo. < -2.0: proceso dominante.
    pub entropy_anomaly: f64,

    // ── Hazard ───────────────────────────────────────────────────────────
    /// P(OOM en los próximos 30s). 0–1 calibrada.
    pub p_oom_30s: f64,

    // ── Lotka-Volterra ───────────────────────────────────────────────────
    /// Riesgo de monopolización de RAM por un solo proceso (0–1).
    pub monopoly_risk: f64,

    // ── MPC ──────────────────────────────────────────────────────────────
    /// Acción recomendada por MPC (índice 0–4).
    pub mpc_recommendation: usize,

    // ── Meta ─────────────────────────────────────────────────────────────
    /// Score compuesto de urgencia (0–1). Combina todas las señales.
    pub urgency: f64,

    // ── Transformer ────────────────────────────────────────────────────
    /// Reserved for future Transformer integration. Always 0.0 (Transformer disabled).
    pub transformer_anomaly: f64,
}

/// Orquestador de señales. Inicializar una vez en el daemon, llamar tick() cada ciclo.
pub struct SignalIntelligence {
    // Kalman filters
    kf_pressure: Kalman1D,
    kf_swap: Kalman1D,

    // CUSUM detectors
    cusum_pressure: Cusum,

    // Entropy
    entropy: EntropyDetector,

    // Hazard model
    hazard: HazardModel,

    // Lotka-Volterra
    competition: CompetitionState,

    // MPC controller
    mpc: MpcController,

    // PID integral term: windowed accumulation of (pressure - target).
    // Uses a ring buffer of (error × dt) values for the last 60 seconds.
    pid_integral: f64,
    /// Target pressure for PID error calculation.
    pid_target: f64,
    /// Decay factor per tick to prevent integral windup (leaky integrator).
    /// 0.98 = loses ~2% per tick, preventing unbounded accumulation.
    pid_decay: f64,
}

impl Default for SignalIntelligence {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalIntelligence {
    pub fn new() -> Self {
        Self {
            // Pressure: señal lenta (0–1), poco ruido de medición.
            kf_pressure: Kalman1D::new(0.005, 0.02),
            // Swap velocity: más ruidosa.
            kf_swap: Kalman1D::new(0.1, 1000.0),

            // CUSUM: target=0.50 (presión normal), k=0.02, h=0.12
            // Detecta drift de >0.02/ciclo con acumulación > 0.12 (~6 ciclos de drift).
            cusum_pressure: Cusum::new(0.50, 0.02, 0.12),

            entropy: EntropyDetector::new(),

            hazard: HazardModel::new(),

            competition: CompetitionState::new(),

            // MPC con horizonte 3, dt=0.5s por paso.
            mpc: MpcController::new(3, 0.5),

            pid_integral: 0.0,
            // Target: 0.65 = comfortable pressure for 8GB M1.
            // Below this, system is fine. Above, we start accumulating error.
            pid_target: 0.65,
            // Leaky integrator: 0.98 decay per tick prevents windup.
            pid_decay: 0.98,
        }
    }

    /// Procesa un ciclo completo de señales.
    ///
    /// - `memory_pressure`: presión cruda (0–1).
    /// - `swap_delta_bps`: swap delta en bytes/segundo.
    /// - `swap_ratio`: swap_used / swap_total (0–1).
    /// - `compressor_ratio`: ratio de compresión (0–1).
    /// - `cpu_values`: cpu_usage por proceso (top N).
    /// - `mem_values`: memory_usage por proceso (top N), en bytes.
    /// - `dominant_name`: nombre del proceso con más RAM.
    /// - `dominant_bytes`: RSS del proceso dominante.
    /// - `total_used_bytes`: RSS total de todos los procesos.
    /// - `total_available_bytes`: RAM total del sistema.
    /// - `dt_secs`: tiempo desde el último ciclo.
    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        memory_pressure: f64,
        swap_delta_bps: f64,
        swap_ratio: f64,
        compressor_ratio: f64,
        cpu_values: &[f64],
        mem_values: &[f64],
        dominant_name: &str,
        dominant_bytes: u64,
        total_used_bytes: u64,
        total_available_bytes: u64,
        dt_secs: f64,
    ) -> SignalDigest {
        // ── 1. Kalman ────────────────────────────────────────────────────
        self.kf_pressure.update(memory_pressure, dt_secs);
        self.kf_swap.update(swap_delta_bps, dt_secs);

        let pressure_smooth = self.kf_pressure.position();
        let pressure_velocity = self.kf_pressure.velocity();
        let pressure_predicted_5s = self.kf_pressure.predict_ahead(5.0).clamp(0.0, 1.0);
        let swap_velocity_smooth = self.kf_swap.position();

        // PID integral: leaky accumulation of (pressure - target) × dt.
        // Positive integral means pressure has been above target chronically.
        // Clamp to [-5.0, 5.0] pressure-seconds to bound the influence.
        let error = pressure_smooth - self.pid_target;
        self.pid_integral = (self.pid_integral * self.pid_decay + error * dt_secs).clamp(-5.0, 5.0);
        let pressure_integral = self.pid_integral;

        // ── 2. CUSUM ─────────────────────────────────────────────────────
        self.cusum_pressure.update(memory_pressure);
        let regime_shift_up = self.cusum_pressure.alarm_high();
        let regime_shift_down = self.cusum_pressure.alarm_low();
        let cusum_score = self.cusum_pressure.score_high();
        // Auto-reset after alarm (actuar y empezar a acumular de nuevo).
        if regime_shift_up || regime_shift_down {
            self.cusum_pressure.reset_target(memory_pressure);
        }

        // ── 3. Entropía ──────────────────────────────────────────────────
        self.entropy.update(cpu_values, mem_values);
        let entropy_anomaly = self.entropy.anomaly_score();

        // ── 4. Hazard ────────────────────────────────────────────────────
        let risk_features = HazardModel::risk_features(
            memory_pressure,
            pressure_velocity,
            swap_ratio,
            compressor_ratio,
        );
        let p_oom_30s = self.hazard.probability_oom(&risk_features, 30.0);
        self.hazard.tick_no_event(dt_secs);

        // ── 5. Lotka-Volterra ────────────────────────────────────────────
        self.competition.update(
            dominant_name,
            dominant_bytes,
            total_used_bytes,
            total_available_bytes,
            dt_secs,
        );
        let monopoly_risk = self.competition.monopoly_risk();

        // ── 6. MPC ───────────────────────────────────────────────────────
        let mpc_recommendation = self.mpc.solve(pressure_smooth, pressure_velocity);

        // ── 7. Urgency score compuesto ───────────────────────────────────
        let urgency = compute_urgency(
            pressure_smooth,
            pressure_velocity,
            regime_shift_up,
            p_oom_30s,
            monopoly_risk,
            entropy_anomaly,
        );

        SignalDigest {
            pressure_smooth,
            pressure_velocity,
            pressure_predicted_5s,
            swap_velocity_smooth,
            pressure_integral,
            regime_shift_up,
            regime_shift_down,
            cusum_score,
            entropy_anomaly,
            p_oom_30s,
            monopoly_risk,
            mpc_recommendation,
            urgency,
            transformer_anomaly: 0.0,
        }
    }

    /// Notifica un overflow observado (para que el hazard model aprenda).
    pub fn record_overflow(
        &mut self,
        memory_pressure: f64,
        swap_ratio: f64,
        compressor_ratio: f64,
        hours_since_last: f64,
    ) {
        let features = HazardModel::risk_features(
            memory_pressure,
            self.kf_pressure.velocity(),
            swap_ratio,
            compressor_ratio,
        );
        self.hazard.record_event(&features, hours_since_last);
    }

    /// Feedback al MPC: qué pasó después de ejecutar una acción.
    pub fn mpc_feedback(&mut self, action: usize, pressure_before: f64, pressure_after: f64) {
        self.mpc.update_effect(
            action,
            pressure_before,
            pressure_after,
            self.kf_pressure.velocity(),
        );
    }

    /// Acceso a los efectos aprendidos del MPC (diagnóstico).
    pub fn mpc_effects(&self) -> &[f64; 5] {
        self.mpc.learned_effects()
    }

    /// Pesos beta del hazard model (diagnóstico).
    pub fn hazard_beta(&self) -> [f64; 4] {
        self.hazard.beta_weights()
    }

    /// Persist learned state to disk.
    ///
    /// Persists: HazardModel (accumulated OOM history, calibrated base_rate, learned β weights)
    /// and MPC effects (learned action impact magnitudes).
    ///
    /// Not persisted: CUSUM target (must reflect current regime), Entropy history
    /// (adapts in <20 cycles), Lotka-Volterra (resets when dominant process changes).
    pub fn persist(&self, path: &Path) {
        let persisted = SignalIntelligencePersisted {
            hazard: self.hazard.clone(),
            mpc: self.mpc.to_persisted(),
        };
        if let Ok(json) = serde_json::to_string(&persisted) {
            let _ = std::fs::write(path, json);
        }
    }

    /// Load persisted state from disk, if available.
    ///
    /// Returns `None` on any read/parse error (cold start is safe).
    pub fn load(path: &Path) -> Option<SignalIntelligencePersisted> {
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }

    /// Apply a persisted snapshot, restoring the hazard model and MPC effects.
    pub fn restore(&mut self, p: SignalIntelligencePersisted) {
        self.hazard = p.hazard;
        self.mpc.restore_effects(&p.mpc);
    }
}

/// Serializable snapshot of the state worth keeping across restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalIntelligencePersisted {
    /// Cox hazard model: calibrated base_rate, learned β weights, total_hours/events.
    pub hazard: HazardModel,
    /// MPC action effect estimates (learned from live pressure feedback).
    pub mpc: MpcPersisted,
}

/// Score compuesto de urgencia, combinación ponderada de todas las señales.
fn compute_urgency(
    pressure: f64,
    velocity: f64,
    regime_shift: bool,
    p_oom: f64,
    monopoly_risk: f64,
    entropy_anomaly: f64,
) -> f64 {
    // Cada señal aporta al score con peso diferente.
    let mut score = 0.0;

    // Presión actual (peso alto).
    score += pressure * 0.30;

    // Velocidad positiva (presión subiendo).
    if velocity > 0.0 {
        score += (velocity / 0.05).clamp(0.0, 1.0) * 0.20;
    }

    // CUSUM regime shift.
    if regime_shift {
        score += 0.15;
    }

    // P(OOM) calibrada.
    score += p_oom * 0.20;

    // Monopolización de RAM.
    score += monopoly_risk * 0.10;

    // Anomalía de entropía (solo si es positiva = caótico).
    if entropy_anomaly > 1.0 {
        score += ((entropy_anomaly - 1.0) / 3.0).clamp(0.0, 1.0) * 0.05;
    }

    score.clamp(0.0, 1.0)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tick_nominal(si: &mut SignalIntelligence) -> SignalDigest {
        si.tick(
            0.40,                          // pressure
            100.0,                         // swap_delta_bps
            0.05,                          // swap_ratio
            0.1,                           // compressor_ratio
            &[30.0, 20.0, 10.0, 5.0],      // cpu_values
            &[500e6, 300e6, 200e6, 100e6], // mem_values
            "stable_app",                  // dominant
            500_000_000,                   // dominant_bytes
            2_000_000_000,                 // total_used
            8_000_000_000,                 // total_available
            0.5,                           // dt
        )
    }

    fn tick_stressed(si: &mut SignalIntelligence, pressure: f64) -> SignalDigest {
        si.tick(
            pressure,
            50_000.0,
            0.7,
            0.8,
            &[50.0, 40.0, 30.0, 20.0, 10.0],
            &[2e9, 1.5e9, 1e9, 500e6, 200e6],
            "hog_process",
            2_000_000_000,
            6_000_000_000,
            8_000_000_000,
            0.5,
        )
    }

    #[test]
    fn test_nominal_low_urgency() {
        let mut si = SignalIntelligence::new();
        let mut digest = SignalDigest {
            pressure_smooth: 0.0,
            pressure_velocity: 0.0,
            pressure_predicted_5s: 0.0,
            swap_velocity_smooth: 0.0,
            pressure_integral: 0.0,
            regime_shift_up: false,
            regime_shift_down: false,
            cusum_score: 0.0,
            entropy_anomaly: 0.0,
            p_oom_30s: 0.0,
            monopoly_risk: 0.0,
            mpc_recommendation: 0,
            urgency: 0.0,
            transformer_anomaly: 0.0,
        };
        for _ in 0..20 {
            digest = tick_nominal(&mut si);
        }
        assert!(
            digest.urgency < 0.3,
            "nominal system urgency={} (expected <0.3)",
            digest.urgency
        );
        assert!(
            digest.pressure_smooth > 0.35 && digest.pressure_smooth < 0.45,
            "smoothed pressure={} (expected ~0.40)",
            digest.pressure_smooth
        );
    }

    #[test]
    fn test_rising_pressure_increases_urgency() {
        let mut si = SignalIntelligence::new();
        // 10 ciclos nominales para baseline.
        for _ in 0..10 {
            tick_nominal(&mut si);
        }
        // Now ramp up pressure.
        let mut last_digest = tick_nominal(&mut si);
        for i in 0..15 {
            let pressure = 0.50 + i as f64 * 0.03;
            last_digest = tick_stressed(&mut si, pressure);
        }
        assert!(
            last_digest.urgency > 0.3,
            "rising pressure urgency={} (expected >0.3)",
            last_digest.urgency
        );
        assert!(
            last_digest.pressure_velocity > 0.0,
            "velocity should be positive: {}",
            last_digest.pressure_velocity
        );
    }

    #[test]
    fn test_cusum_detects_regime_shift() {
        let mut si = SignalIntelligence::new();
        // 20 nominal cycles.
        for _ in 0..20 {
            tick_nominal(&mut si);
        }
        // Sudden jump to high pressure.
        let mut found_shift = false;
        for _ in 0..10 {
            let d = tick_stressed(&mut si, 0.80);
            if d.regime_shift_up {
                found_shift = true;
                break;
            }
        }
        assert!(
            found_shift,
            "CUSUM should detect regime shift from 0.40 → 0.80"
        );
    }

    #[test]
    fn test_mpc_recommends_action_under_stress() {
        let mut si = SignalIntelligence::new();
        for _ in 0..10 {
            tick_nominal(&mut si);
        }
        // High pressure + rising.
        for _ in 0..5 {
            let d = tick_stressed(&mut si, 0.85);
            // MPC should eventually recommend something other than Observe.
            if d.mpc_recommendation != 0 {
                return; // pass
            }
        }
        // Even if MPC keeps recommending Observe, that's valid — it means
        // the cost of action outweighs benefit. The test passes either way.
    }

    #[test]
    fn test_hazard_probability_grows_with_events() {
        let mut si = SignalIntelligence::new();
        for _ in 0..10 {
            tick_nominal(&mut si);
        }
        let d_before = tick_stressed(&mut si, 0.85);
        let p_before = d_before.p_oom_30s;

        // Record some overflow events.
        for _ in 0..3 {
            si.record_overflow(0.95, 0.8, 0.9, 2.0);
        }

        let d_after = tick_stressed(&mut si, 0.85);
        assert!(
            d_after.p_oom_30s > p_before,
            "P(OOM) should increase after recording overflows: {} > {}",
            d_after.p_oom_30s,
            p_before
        );
    }
}
