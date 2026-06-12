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

/// NEON-accelerated dot product for exactly 4 f64 values.
/// Uses 2 × float64x2_t multiply + horizontal reduction.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn neon_dot4(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    use std::arch::aarch64::*;
    unsafe {
        let a0 = vld1q_f64(a.as_ptr());
        let a1 = vld1q_f64(a.as_ptr().add(2));
        let b0 = vld1q_f64(b.as_ptr());
        let b1 = vld1q_f64(b.as_ptr().add(2));
        let prod0 = vmulq_f64(a0, b0);
        let prod1 = vmulq_f64(a1, b1);
        let sum01 = vaddq_f64(prod0, prod1);
        vgetq_lane_f64(sum01, 0) + vgetq_lane_f64(sum01, 1)
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn neon_dot4(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

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
    /// Cycle counter for throttling beta counter-gradient updates.
    /// Beta is only updated every SURVIVAL_BETA_STRIDE calls to prevent the
    /// ~10,000:1 survival-to-OOM ratio from pinning betas at their floor.
    /// base_rate update still runs every call.
    /// Skipped during serialization — resets to 0 on daemon restart.
    #[serde(skip)]
    survival_tick_count: u32,
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
            survival_tick_count: 0,
        }
    }

    /// Calcula el vector de riesgo a partir de señales del sistema.
    ///
    /// Todas las features se normalizan a ~0–1 para que los β sean comparables.
    ///
    /// Slot 3: `max(compressor_ratio, cumulative_stress * 0.7)` — enriches the
    /// temporal dimension without resizing NEON dot4. [Yerkes & Dodson 1908]
    /// cumulative_stress captures chronic overload that compressor_ratio misses
    /// at moderate (but sustained) pressure levels.
    pub fn risk_features(
        memory_pressure: f64,
        pressure_velocity: f64, // del Kalman, unidades/segundo
        swap_ratio: f64,        // swap_used / swap_total, 0–1
        compressor_ratio: f64,  // de collector.rs, 0–1
        cumulative_stress: f64, // slow EMA of urgency, 0–1 [Yerkes-Dodson 1908]
    ) -> [f64; N_RISK] {
        [
            memory_pressure.clamp(0.0, 1.0),
            // Velocidad: normalizar a 0–1 (0.1/s = bastante rápido para presión)
            (pressure_velocity / 0.1).clamp(0.0, 1.0),
            swap_ratio.clamp(0.0, 1.0),
            compressor_ratio
                .clamp(0.0, 1.0)
                .max((cumulative_stress * 0.7).clamp(0.0, 1.0)),
        ]
    }

    /// Calcula h(x) = h₀ · exp(β · x).
    /// NEON-accelerated: 4 f64 = 2 × float64x2_t → 2 FMA + horizontal add.
    fn hazard_rate(&self, features: &[f64; N_RISK]) -> f64 {
        let dot = neon_dot4(&self.beta, features);
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

    /// Tick de "supervivencia bajo presión alta": el sistema estaba bajo presión
    /// pero no ocurrió ningún OOM. Esto es evidencia negativa — el modelo sobreestimó
    /// el riesgo. Decae base_rate y ajusta beta hacia abajo (contra-gradiente suave).
    ///
    /// Solo debe llamarse cuando swap_ratio == 0 y presión > 0.6, para no
    /// interferir con situaciones donde el riesgo es real.
    pub fn tick_survived_high_pressure(&mut self, features: &[f64; N_RISK], dt_secs: f64) {
        self.total_hours += dt_secs / 3600.0;
        if self.total_hours > 0.0 {
            // Incrementar el denominador como si hubieran pasado horas adicionales
            // de observación sin evento — equivale a añadir evidencia negativa.
            // Factor 3× para que la supervivencia a presión alta cuente más.
            let effective_hours = self.total_hours + (dt_secs / 3600.0) * 2.0;
            self.base_rate = (self.total_events as f64 + 1.0) / ((effective_hours + 24.0) * 3600.0);
        }
        // Contra-gradiente en β: apply every SURVIVAL_BETA_STRIDE ticks only.
        // At ~5s cycles and frequent high-pressure periods, survival ticks outnumber
        // OOM events ~10,000:1. Without throttling, betas get pinned at the floor
        // regardless of update magnitude (H-2 calibration issue).
        // base_rate update above runs every tick (correct — it's time-based).
        const SURVIVAL_BETA_STRIDE: u32 = 10;
        self.survival_tick_count = self.survival_tick_count.wrapping_add(1);
        if self
            .survival_tick_count
            .is_multiple_of(SURVIVAL_BETA_STRIDE)
        {
            // neg_lr × 0.05 keeps OOM events (lr=0.01) 20× more impactful per event.
            // Combined with stride=10, effective asymmetry is 200× — OOM events
            // dominate on a per-event basis while survival ticks still converge over hours.
            // Floor: 0.1 (not 0.5) — allows the model to learn that some features
            // are genuinely less predictive of OOM without zeroing them out entirely.
            let neg_lr = self.lr * 0.05;
            for (b, x) in self.beta.iter_mut().zip(features.iter()) {
                *b -= neg_lr * x;
                *b = b.clamp(0.1, 5.0);
            }
        }
    }

    /// Valida y sana el modelo post-restore.
    ///
    /// Un `base_rate` > 1 OOM/hora indica saturación por entrenamiento en
    /// eventos de presión (no OOMs reales). Si se detecta, resetea base_rate
    /// Y total_events al prior — si no, tick_no_event recalcularía base_rate
    /// desde total_events alto e inmediatamente re-saturaría el modelo.
    /// Los β se clampean a su rango válido.
    pub fn validate_after_restore(&mut self) {
        // Máximo plausible: 1 OOM por hora = 1/3600 eventos/segundo.
        const MAX_SANE_BASE_RATE: f64 = 1.0 / 3600.0;
        if self.base_rate > MAX_SANE_BASE_RATE {
            // Reset completo: base_rate, total_events, AND total_hours.
            //
            // Previously only total_events was reset, preserving total_hours.
            // BUG: with total_hours=2000 (months of uptime) and total_events=0,
            // tick_no_event() computes base_rate ≈ 1/(2000*3600) ≈ 1.37e-7,
            // which is 33x below the prior — model becomes blind to OOM risk.
            //
            // Fix: also reset total_hours to a 24h prior window. This anchors
            // the model to "1 event in the first 24h" (the Laplace prior) rather
            // than to a phantom history it never observed in the new session.
            self.base_rate = 1.0 / (24.0 * 3600.0);
            self.total_events = 0;
            self.total_hours = 24.0; // 24h prior window — matches Laplace +24 in formula
        }
        for b in self.beta.iter_mut() {
            *b = b.clamp(0.0, 5.0);
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
        let features = HazardModel::risk_features(0.3, 0.0, 0.0, 0.1, 0.0);
        let p = model.probability_oom(&features, 30.0);
        assert!(p < 0.01, "low risk should give low P(OOM), got {}", p);
    }

    #[test]
    fn test_high_risk_high_probability() {
        let mut model = HazardModel::new();
        // Record some events to raise base_rate.
        for _ in 0..5 {
            let feat = HazardModel::risk_features(0.9, 0.08, 0.7, 0.8, 0.0);
            model.record_event(&feat, 2.0);
        }
        let features = HazardModel::risk_features(0.95, 0.1, 0.8, 0.9, 0.0);
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
        let features = HazardModel::risk_features(0.7, 0.05, 0.4, 0.5, 0.0);
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
        let feat = HazardModel::risk_features(0.9, 0.08, 0.7, 0.8, 0.0);
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
            let feat = HazardModel::risk_features(0.99, 0.1, 0.99, 0.99, 0.0);
            model.record_event(&feat, 0.5);
        }
        let features = HazardModel::risk_features(1.0, 0.1, 1.0, 1.0, 0.0);
        let p = model.probability_oom(&features, 300.0);
        assert!(p >= 0.0 && p <= 1.0, "P should be in [0,1], got {}", p);
    }

    /// Hazard must be monotonically increasing with pressure (all else equal).
    /// [Cox 1972] proportional hazards: higher risk features → higher hazard.
    #[test]
    fn test_monotonic_with_pressure() {
        let mut model = HazardModel::new();
        let pressures = [0.1, 0.3, 0.5, 0.7, 0.9];
        let mut prev_p = 0.0;
        for &pr in &pressures {
            let feat = HazardModel::risk_features(pr, 0.02, pr * 0.5, pr * 0.5, 0.0);
            let p = model.probability_oom(&feat, 30.0);
            assert!(
                p >= prev_p,
                "P(OOM) not monotonic: p={}, pr={}, prev={}",
                p,
                pr,
                prev_p
            );
            prev_p = p;
        }
    }

    /// Zero-horizon must give zero probability (no time to fail).
    #[test]
    fn test_zero_horizon_zero_probability() {
        let mut model = HazardModel::new();
        let feat = HazardModel::risk_features(0.9, 0.1, 0.8, 0.8, 0.0);
        let p = model.probability_oom(&feat, 0.0);
        assert!(
            p.abs() < 1e-10,
            "zero horizon should give ~zero P(OOM), got {}",
            p
        );
    }

    /// Beta weights must stay bounded [0, 5] even after many events.
    #[test]
    fn test_beta_bounded_after_many_events() {
        let mut model = HazardModel::new();
        // Record 100 events with extreme features.
        for _ in 0..100 {
            let feat = HazardModel::risk_features(1.0, 0.1, 1.0, 1.0, 0.0);
            model.record_event(&feat, 0.1);
        }
        for &b in &model.beta_weights() {
            assert!(b >= 0.0 && b <= 5.0, "beta {} out of bounds [0,5]", b);
        }
    }

    /// risk_features must clamp inputs to [0, 1] for normalized features.
    #[test]
    fn test_risk_features_clamps_out_of_range() {
        let feat = HazardModel::risk_features(-0.5, -1.0, 2.0, 1.5, 0.0);
        assert_eq!(feat[0], 0.0, "negative pressure should clamp to 0");
        assert_eq!(feat[1], 0.0, "negative velocity should clamp to 0");
        assert_eq!(feat[2], 1.0, "swap > 1.0 should clamp to 1.0");
        assert_eq!(feat[3], 1.0, "compressor > 1.0 should clamp to 1.0");
    }

    /// Serde roundtrip must preserve all model state (for persistence).
    #[test]
    fn test_serde_roundtrip() {
        let mut model = HazardModel::new();
        let feat = HazardModel::risk_features(0.8, 0.05, 0.5, 0.6, 0.0);
        model.record_event(&feat, 3.0);
        model.tick_no_event(7200.0);

        let json = serde_json::to_string(&model).unwrap();
        let restored: HazardModel = serde_json::from_str(&json).unwrap();

        assert_eq!(model.total_events, restored.total_events);
        assert!((model.total_hours - restored.total_hours).abs() < 1e-10);
        assert!((model.base_rate - restored.base_rate).abs() < 1e-15);
        for i in 0..N_RISK {
            assert!(
                (model.beta[i] - restored.beta[i]).abs() < 1e-15,
                "beta[{}] diverged: {} vs {}",
                i,
                model.beta[i],
                restored.beta[i]
            );
        }
    }

    /// validate_after_restore() resets a saturated model to a safe prior.
    /// Base rate > 1 OOM/hour is physically impossible for a stable system
    /// — it indicates the model was trained on pressure events, not real OOMs.
    #[test]
    fn test_validate_after_restore_resets_saturated_base_rate() {
        let mut model = HazardModel::new();
        // Simulate saturation: feed many "overflow" events in quick succession
        let feat = HazardModel::risk_features(0.85, 0.05, 0.9, 0.8, 0.0);
        for _ in 0..200 {
            model.record_event(&feat, 0.1); // 0.1 hour each
        }
        // Model should now have a very high base_rate
        assert!(
            model.base_rate > 1.0 / 3600.0,
            "base_rate should be saturated after many events in short time"
        );

        model.validate_after_restore();

        // After validation, base_rate must be <= 1/hour (1/3600 s^-1)
        assert!(
            model.base_rate <= 1.0 / 3600.0,
            "validate_after_restore must clamp base_rate to sane range, got {}",
            model.base_rate
        );
        // total_events should be reset to 0 (prevents re-saturation on next tick)
        assert_eq!(
            model.total_events, 0,
            "total_events must reset to prevent re-saturation"
        );
        // total_hours reset to 24h prior window (NOT preserved — preserving months of
        // uptime would make base_rate ≈ 0 after events=0 reset)
        assert!(
            (model.total_hours - 24.0).abs() < 1e-10,
            "total_hours should be reset to 24h prior, got {}",
            model.total_hours
        );
        // beta values must all be in valid range
        for b in model.beta.iter() {
            assert!(
                *b >= 0.0 && *b <= 5.0,
                "beta must stay in valid range after validate"
            );
        }
    }

    /// validate_after_restore() must be a no-op for a healthy model.
    #[test]
    fn test_validate_after_restore_noop_on_healthy_model() {
        let mut model = HazardModel::new();
        let feat = HazardModel::risk_features(0.5, 0.03, 0.4, 0.3, 0.0);
        model.record_event(&feat, 24.0); // one event in 24 hours — healthy rate
        let rate_before = model.base_rate;
        let events_before = model.total_events;

        model.validate_after_restore();

        assert_eq!(
            model.total_events, events_before,
            "validate should not change events on healthy model"
        );
        assert!(
            (model.base_rate - rate_before).abs() < 1e-15,
            "validate should not change base_rate on healthy model"
        );
    }

    /// tick_survived_high_pressure() must decay base_rate (negative evidence).
    #[test]
    fn test_survived_high_pressure_decays_base_rate() {
        let mut model = HazardModel::new();
        // Give it a somewhat elevated base_rate first
        let feat = HazardModel::risk_features(0.75, 0.02, 0.8, 0.7, 0.0);
        model.record_event(&feat, 1.0);
        let rate_before = model.base_rate;

        // Simulate survival: system was at high pressure but no OOM occurred
        model.tick_survived_high_pressure(&feat, 30.0); // 30 seconds of survival

        assert!(
            model.base_rate < rate_before,
            "survival evidence should decay base_rate: before={}, after={}",
            rate_before,
            model.base_rate
        );
    }

    /// beta values must stay in valid range after tick_survived_high_pressure().
    #[test]
    fn test_survived_high_pressure_beta_stays_bounded() {
        let mut model = HazardModel::new();
        let feat = HazardModel::risk_features(0.9, 0.01, 0.95, 0.85, 0.0);
        // Apply many survival ticks — beta must never go below 0.1 floor.
        // Floor lowered from 0.5 to 0.1 (H-2 fix) to allow feature discrimination
        // without zeroing out features entirely.
        for _ in 0..500 {
            model.tick_survived_high_pressure(&feat, 5.0);
        }
        for (i, b) in model.beta.iter().enumerate() {
            assert!(*b >= 0.1, "beta[{}] fell below floor of 0.1: got {}", i, b);
        }
    }
}
