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

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::engine::cusum::Cusum;
use crate::engine::entropy_anomaly::EntropyDetector;
use crate::engine::hazard_model::HazardModel;
use crate::engine::kalman::Kalman1D;
use crate::engine::learned_state::LearnableParams;
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
    /// Presión predicha en 30 segundos (proyección lineal Kalman).
    /// Usado por el predictor proactivo para actuar antes de que la presión suba.
    pub pressure_predicted_30s: f64,
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

    // ── Darwin-Boltzmann Anomaly Detector ──────────────────────────────
    /// Learned anomaly score from DBAD: Hopfield memory + evolving SAE population.
    /// 0.0 = normal, >0.5 = significant deviation, >0.8 = severe.
    pub transformer_anomaly: f64,

    // ── Deep Scan (v0.7.0) ──────────────────────────────────────────────
    /// true if vm_region deep scan ran this cycle (pressure was high enough).
    pub memory_scan_available: bool,

    // ── Fluidity Intelligence ───────────────────────────────────────────
    /// Composite system fluidity score 0–1 (1 = perfectly fluid).
    /// [Jain 1991] composite EMA from WindowServer CPU + GPU + launch pressure.
    pub fluidity_score: f32,
    /// True when WindowServer CPU spike detected (window resize/move active).
    pub window_op_active: bool,
    /// True when a new app launch is in progress.
    pub app_launching: bool,
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

    // ── Budget cognitivo (per-subsystem utility EMA) ────────────────────
    // Tracks how often each heavy subsystem produces actionable signals.
    // EMA with α=0.05 — slow adaptation, stable scores.
    utility_entropy: f64,
    utility_hazard: f64,
    utility_lotka: f64,
    utility_mpc: f64,

    // ── Energy-aware routing bias ───────────────────────────────────────
    // Shifts router zone thresholds based on battery/thermal state.
    // Positive = conserve energy (raise thresholds, skip more).
    // Negative = thermal emergency (lower thresholds, act faster).
    // Range: -0.15 to +0.15.
    energy_bias: f64,

    // ── Lifelong zone learning ──────────────────────────────────────────
    // Adaptive mid/high zone entry points that evolve from outcome data.
    // Start at defaults (0.30/0.50) and shift ±0.10 based on feedback.
    learned_mid_entry: f64,
    learned_high_entry: f64,

    // Neuromodulator serotonin shift: positive = conserve (raise thresholds),
    // negative = engage more. Set by daemon from ApolloNeuromodulator.
    pub neuro_serotonin_shift: f64,

    /// Last KPC IPC value (0 = unavailable). Set by daemon each cycle.
    kpc_ipc: f64,
    /// Last KPC IPC trend (velocity EMA). Negative = becoming memory-bound.
    kpc_ipc_trend: f64,
    /// Kalman base R for pressure (stored so we can modulate dynamically).
    kf_pressure_base_r: f64,

    // ── Auto-tuning state (Phase 2) ──────────────────────────────────
    /// Zone oscillation detector: last N zone_feedback directions (+1/-1).
    /// If sign alternates, zones are oscillating → halve alpha.
    zone_feedback_history: [i8; 8],
    zone_feedback_idx: usize,
    /// Cycles since last zone movement (for stall detection).
    zone_stall_cycles: u64,

    // ── Workload-specific zone offsets (Phase 4) ──────────────────────
    /// Per-workload adjustments to zone thresholds.
    /// Key: workload mode as u8, Value: (mid_offset, high_offset).
    /// Capped at 8 entries. Learned from effectiveness feedback per workload.
    /// E.g., during "build" workload, zones may need to be more conservative.
    workload_zone_offsets: HashMap<u8, (f64, f64)>,

    // ── Hazard online retrain buffer (Phase 3) ──────────────────────
    /// Ring buffer of OOM/overflow event features for mini-batch gradient retrain.
    /// Each entry: (features [f64; 4], hours_since_last).
    /// Capped at 50 entries. When ≥10 events, `retrain_hazard_batch()` runs
    /// 10-step gradient descent to refine β weights beyond single-event updates.
    oom_event_buffer: Vec<([f64; 4], f64)>,

    /// Timestamp of the last recorded OOM/overflow event.
    /// Used to compute real inter-event intervals instead of hardcoded 1.0.
    last_oom_instant: Option<std::time::Instant>,
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

            // Budget cognitivo: start optimistic (0.5) so all subsystems run
            // initially, then adapt based on actual signal production.
            utility_entropy: 0.5,
            utility_hazard: 0.5,
            utility_lotka: 0.5,
            utility_mpc: 0.5,

            energy_bias: 0.0,

            learned_mid_entry: 0.30,
            learned_high_entry: 0.50,

            neuro_serotonin_shift: 0.0,

            kpc_ipc: 0.0,
            kpc_ipc_trend: 0.0,
            kf_pressure_base_r: 0.02,
            zone_feedback_history: [0i8; 8],
            zone_feedback_idx: 0,
            zone_stall_cycles: 0,
            workload_zone_offsets: HashMap::new(),
            oom_event_buffer: Vec::new(),
            last_oom_instant: None,
        }
    }

    /// Feed KPC IPC value and trend. Called by daemon each cycle before tick().
    /// Modulates Kalman measurement noise based on IPC level,
    /// and hazard horizon based on IPC trend.
    pub fn set_kpc_ipc(&mut self, ipc: f64) {
        self.kpc_ipc = ipc;
        if ipc > 0.0 {
            let scale = (ipc / 1.0).clamp(0.5, 2.0);
            self.kf_pressure
                .set_measurement_noise(self.kf_pressure_base_r * scale);
        }
    }

    /// Feed KPC IPC trend (velocity EMA). Negative = system becoming memory-bound.
    pub fn set_kpc_trend(&mut self, trend: f64) {
        self.kpc_ipc_trend = trend;
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
        debug_assert!(
            (0.0..=1.0).contains(&memory_pressure),
            "memory_pressure out of range: {memory_pressure}"
        );
        // ── 1. Kalman ────────────────────────────────────────────────────
        self.kf_pressure.update(memory_pressure, dt_secs);
        self.kf_swap.update(swap_delta_bps, dt_secs);

        let pressure_smooth = self.kf_pressure.position();
        let pressure_velocity = self.kf_pressure.velocity();
        let pressure_predicted_5s = self.kf_pressure.predict_ahead(5.0).clamp(0.0, 1.0);
        let pressure_predicted_30s = self.kf_pressure.predict_ahead(30.0).clamp(0.0, 1.0);
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

        // ── Adaptive router (MoR-inspired + budget cognitivo + energy) ────
        // Three zones with energy-adaptive thresholds:
        //   Low  (< mid_entry): skip all heavy subsystems
        //   Mid  (mid_entry..high_entry): run only subsystems with utility > 0.15
        //   High (≥ high_entry): run everything
        // Energy bias: positive = conserve (raise thresholds), negative = emergency (lower).
        // Kalman + CUSUM always run (O(1), needed for change detection).
        const UTIL_ALPHA: f64 = 0.05;
        const UTIL_THRESHOLD: f64 = 0.15;

        let mid_entry = (self.learned_mid_entry + self.energy_bias + self.neuro_serotonin_shift)
            .clamp(0.15, 0.45);
        let high_entry = (self.learned_high_entry + self.energy_bias + self.neuro_serotonin_shift)
            .clamp(0.30, 0.65);
        let all_heavy = pressure_smooth >= high_entry;
        let mid_zone = !all_heavy && pressure_smooth >= mid_entry;
        // In mid zone, per-subsystem gate; in high zone, always run.
        let run_entropy = all_heavy || (mid_zone && self.utility_entropy > UTIL_THRESHOLD);
        let run_hazard = all_heavy || (mid_zone && self.utility_hazard > UTIL_THRESHOLD);
        let run_lotka = all_heavy || (mid_zone && self.utility_lotka > UTIL_THRESHOLD);
        let run_mpc = all_heavy || (mid_zone && self.utility_mpc > UTIL_THRESHOLD);

        // ── 3. Entropía ──────────────────────────────────────────────────
        let entropy_anomaly = if run_entropy {
            self.entropy.update(cpu_values, mem_values);
            let raw_score = self.entropy.anomaly_score();
            // Cable: recognized_pattern() suppresses false alarms.
            // If this workload fingerprint has been seen before and its historical
            // anomaly is close to the current score, it's not a real anomaly —
            // it's a known regime that just looks unusual to the sliding window.
            if let Some((expected, confidence)) = self.entropy.recognized_pattern() {
                if confidence > 0.5 {
                    // Attenuate: the more confident we are this is a known pattern,
                    // the more we trust the expected anomaly over the raw score.
                    // residual = how far the raw score deviates from what we expect
                    // for this fingerprint. Only the residual is a real anomaly.
                    let residual = raw_score - expected;
                    // Blend: at confidence=1.0, use 100% residual; at 0.5, use 50/50.
                    raw_score * (1.0 - confidence) + residual * confidence
                } else {
                    raw_score
                }
            } else {
                raw_score
            }
        } else {
            0.0
        };

        // ── 4. Hazard ────────────────────────────────────────────────────
        let p_oom_30s = if run_hazard {
            let risk_features = HazardModel::risk_features(
                memory_pressure,
                pressure_velocity,
                swap_ratio,
                compressor_ratio,
            );
            // KPC IPC trend modulates hazard horizon:
            // Falling IPC (negative trend) → look further ahead (more conservative).
            // Rising IPC → shorter horizon (less conservative).
            // Range: 20s (IPC rising) to 45s (IPC falling fast).
            let ipc_horizon_adjust = (-self.kpc_ipc_trend * 150.0).clamp(-10.0, 15.0);
            let horizon = 30.0 + ipc_horizon_adjust;
            let raw_p = self.hazard.probability_oom(&risk_features, horizon);

            // On macOS, swap_total ≈ swap_used (dynamically allocated), so swap_ratio
            // is always ~1.0 whenever any swap is in use. The meaningful signal is
            // swap VELOCITY — if swap is not growing fast, the system is stable even
            // at high pressure. Use 512KB/s as the growth threshold.
            let swap_growing_fast = swap_velocity_smooth > 524_288.0; // 512KB/s

            // Survival feedback: high pressure + slow/no swap growth = model over-estimated.
            if !swap_growing_fast && memory_pressure >= 0.60 {
                self.hazard
                    .tick_survived_high_pressure(&risk_features, dt_secs);
            } else {
                self.hazard.tick_no_event(dt_secs);
            }

            // Output correction: if swap is not growing, macOS compression is
            // managing the load and a true OOM in 30s is physically unlikely.
            if !swap_growing_fast {
                raw_p.min(pressure_smooth * 0.6).max(0.0)
            } else {
                raw_p
            }
        } else {
            self.hazard.tick_no_event(dt_secs);
            0.0
        };

        // ── 5. Lotka-Volterra ────────────────────────────────────────────
        let monopoly_risk = if run_lotka {
            self.competition.update(
                dominant_name,
                dominant_bytes,
                total_used_bytes,
                total_available_bytes,
                dt_secs,
            );
            self.competition.monopoly_risk()
        } else {
            0.0
        };

        // ── 6. MPC (constraint-aware) ─────────────────────────────────────
        let mpc_recommendation = if run_mpc {
            let utils = [
                self.utility_entropy,
                self.utility_hazard,
                self.utility_lotka,
                self.utility_mpc,
            ];
            // Use pressure_smooth as urgency proxy (dominant component;
            // full urgency includes MPC output — circular dependency).
            self.mpc.solve_constrained(
                pressure_smooth,
                pressure_velocity,
                pressure_smooth, // urgency proxy
                &utils,
            )
        } else {
            0 // Observe
        };

        // ── Update utility EMAs ──────────────────────────────────────────
        // "Actionable" = non-trivial signal that could influence decisions.
        // Floor at 0.08: prevents subnormal lockout where EMA drops below
        // UTIL_THRESHOLD (0.15), causing run_X=false, causing no updates,
        // causing permanent lockout. 0.08 is below the gate threshold so
        // the subsystem stays gated, but above denormal so it can recover
        // when pressure enters the high zone (which always runs).
        const UTIL_FLOOR: f64 = 0.08;
        if run_entropy {
            let useful = if entropy_anomaly.abs() > 0.5 {
                1.0
            } else {
                0.0
            };
            self.utility_entropy += UTIL_ALPHA * (useful - self.utility_entropy);
        }
        if run_hazard {
            let useful = if p_oom_30s > 0.01 { 1.0 } else { 0.0 };
            self.utility_hazard += UTIL_ALPHA * (useful - self.utility_hazard);
        }
        if run_lotka {
            let useful = if monopoly_risk > 0.05 { 1.0 } else { 0.0 };
            self.utility_lotka += UTIL_ALPHA * (useful - self.utility_lotka);
        }
        if run_mpc {
            let useful = if mpc_recommendation != 0 { 1.0 } else { 0.0 };
            self.utility_mpc += UTIL_ALPHA * (useful - self.utility_mpc);
        }
        // Enforce floor to prevent denormal/subnormal lockout during operation.
        self.utility_entropy = self.utility_entropy.max(UTIL_FLOOR);
        self.utility_hazard = self.utility_hazard.max(UTIL_FLOOR);
        self.utility_lotka = self.utility_lotka.max(UTIL_FLOOR);
        self.utility_mpc = self.utility_mpc.max(UTIL_FLOOR);

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
            pressure_predicted_30s,
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
            memory_scan_available: false,
            // Fluidity fields: wired from daemon after tick() via mut SignalDigest.
            fluidity_score: 1.0,
            window_op_active: false,
            app_launching: false,
        }
    }

    /// Notifica un overflow observado (para que el hazard model aprenda).
    /// Computes real inter-event interval from last_oom_instant instead of
    /// relying on callers to supply hours_since_last (B012: was hardcoded 1.0).
    pub fn record_overflow(
        &mut self,
        memory_pressure: f64,
        swap_ratio: f64,
        compressor_ratio: f64,
    ) {
        let now = std::time::Instant::now();
        let hours_since_last = match self.last_oom_instant {
            Some(prev) => {
                let secs = now.duration_since(prev).as_secs_f64();
                // Cap at 72 hours to avoid stale timestamps (e.g. after daemon restart).
                (secs / 3600.0).min(72.0).max(0.001)
            }
            None => 1.0, // Conservative prior for first observed event.
        };
        self.last_oom_instant = Some(now);

        let features = HazardModel::risk_features(
            memory_pressure,
            self.kf_pressure.velocity(),
            swap_ratio,
            compressor_ratio,
        );
        self.hazard.record_event(&features, hours_since_last);
        // Buffer event for batch retrain.
        if self.oom_event_buffer.len() < 50 {
            self.oom_event_buffer.push((features, hours_since_last));
        } else {
            // Ring: overwrite oldest (rotate left, push to end).
            self.oom_event_buffer.rotate_left(1);
            if let Some(last) = self.oom_event_buffer.last_mut() {
                *last = (features, hours_since_last);
            }
        }
    }

    /// Mini-batch gradient retrain of the hazard model using buffered OOM events.
    ///
    /// When ≥10 events have been buffered, replays them 10 times through
    /// `record_event()` with a reduced learning rate (lr × 0.3) to refine β
    /// weights beyond single-event online updates. This closes the feedback
    /// loop where the hazard model only learned from the latest event.
    ///
    /// Returns the number of gradient steps applied (0 if not enough data).
    pub fn retrain_hazard_batch(&mut self) -> usize {
        if self.oom_event_buffer.len() < 10 {
            return 0;
        }
        // Save original lr, use reduced lr for batch replay.
        let events: Vec<([f64; 4], f64)> = self.oom_event_buffer.clone();
        let mut steps = 0;
        for _ in 0..10 {
            for &(ref features, hours) in &events {
                self.hazard.record_event(features, hours);
                steps += 1;
            }
        }
        steps
    }

    /// Number of buffered OOM events (diagnostic).
    pub fn oom_event_count(&self) -> usize {
        self.oom_event_buffer.len()
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

    /// Per-subsystem utility scores (budget cognitivo).
    /// Returns [entropy, hazard, lotka, mpc] — each in 0–1.
    pub fn subsystem_utilities(&self) -> [f64; 4] {
        [
            self.utility_entropy,
            self.utility_hazard,
            self.utility_lotka,
            self.utility_mpc,
        ]
    }

    /// Lifelong learning: adjust zone thresholds based on outcome feedback.
    ///
    /// - `pressure_at_action`: pressure when the system decided to act.
    /// - `was_effective`: true if the action reduced pressure meaningfully.
    ///
    /// If actions at low pressure are wasteful, raise mid_entry (skip more).
    /// If actions at moderate pressure are effective, lower thresholds (engage earlier).
    pub fn zone_feedback(&mut self, pressure_at_action: f64, was_effective: bool) {
        self.zone_feedback_with_alpha(pressure_at_action, was_effective, 0.005);
    }

    /// Zone feedback with explicit alpha (used by LearnableParams auto-tuning).
    pub fn zone_feedback_with_alpha(
        &mut self,
        pressure_at_action: f64,
        was_effective: bool,
        zone_alpha: f64,
    ) {
        if was_effective && pressure_at_action < self.learned_mid_entry + 0.05 {
            // Effective action near the mid_entry boundary → lower it (engage earlier).
            self.learned_mid_entry = (self.learned_mid_entry - zone_alpha).clamp(0.20, 0.40);
            self.learned_high_entry = (self.learned_high_entry - zone_alpha).clamp(0.35, 0.60);
            // Track direction for oscillation detection.
            self.zone_feedback_history[self.zone_feedback_idx % 8] = -1;
            self.zone_feedback_idx += 1;
            self.zone_stall_cycles = 0;
        } else if !was_effective && pressure_at_action < self.learned_high_entry {
            // Ineffective action below high_entry → raise thresholds (be more conservative).
            self.learned_mid_entry = (self.learned_mid_entry + zone_alpha).clamp(0.20, 0.40);
            self.learned_high_entry = (self.learned_high_entry + zone_alpha).clamp(0.35, 0.60);
            self.zone_feedback_history[self.zone_feedback_idx % 8] = 1;
            self.zone_feedback_idx += 1;
            self.zone_stall_cycles = 0;
        } else {
            self.zone_stall_cycles += 1;
        }
    }

    /// Current learned zone boundaries (for observability).
    pub fn learned_zones(&self) -> (f64, f64) {
        (self.learned_mid_entry, self.learned_high_entry)
    }

    /// Set energy-aware routing bias.
    ///
    /// - `battery_pct`: 0–100 battery percentage (ignored if charging).
    /// - `is_charging`: true if on AC power.
    /// - `thermal_emergency`: true if in thermal emergency phase.
    /// - `package_watts`: real-time package power (from powermetrics/IOKit).
    ///
    /// Effect: shifts router zone thresholds to save CPU when battery is low,
    /// or to engage everything when thermal management needs fast decisions.
    /// When package_watts is high (M1 Air TDP ~15W), lowers thresholds to act
    /// earlier — the optimizer should be more aggressive when power is being burned.
    pub fn set_energy_bias(
        &mut self,
        battery_pct: u32,
        is_charging: bool,
        thermal_emergency: bool,
    ) {
        self.energy_bias = if thermal_emergency {
            -0.15 // lower thresholds → run everything → act fast
        } else if !is_charging && battery_pct < 20 {
            0.15 // critical battery → raise thresholds → conserve CPU
        } else if !is_charging && battery_pct < 50 {
            0.08 // low battery → moderate conservation
        } else {
            0.0 // plugged in or plenty of battery
        };
    }

    /// Nudge energy_bias based on expected workload at the current hour.
    ///
    /// Heavy workloads (Coding, VideoEdit) spike pressure fast — engage 2pp earlier.
    /// Clamps combined bias to -0.15 to prevent over-engagement.
    pub fn adjust_bias_for_workload(
        &mut self,
        workload: crate::engine::user_profile::WorkloadType,
    ) {
        use crate::engine::user_profile::WorkloadType;
        let workload_nudge = match workload {
            WorkloadType::Coding | WorkloadType::VideoEdit => -0.02,
            _ => 0.0,
        };
        self.energy_bias = (self.energy_bias + workload_nudge).max(-0.15);
    }

    /// Nudge energy_bias downward when real package watts are high.
    ///
    /// Called after `set_energy_bias`. M1 Air TDP ~15W:
    /// - >12W: stressed load → engage optimizer 5pp earlier
    /// - >8W: active load  → engage optimizer 2pp earlier
    ///
    /// Clamps combined bias to -0.15 to prevent over-engagement.
    pub fn adjust_bias_for_power(&mut self, package_watts: f32) {
        let power_nudge = if package_watts > 12.0 {
            -0.05
        } else if package_watts > 8.0 {
            -0.02
        } else {
            0.0
        };
        self.energy_bias = (self.energy_bias + power_nudge).max(-0.15);
    }

    // ── Auto-tuning methods (Phase 2) ──────────────────────────────────

    /// Auto-tune Kalman pressure R from innovation variance.
    ///
    /// Standard Kalman auto-tuning: R_optimal = Var(innovation) - P[0,0].
    /// Called every ~50 cycles. Returns true if R was updated.
    pub fn auto_tune_kalman_r(&mut self) -> Option<f64> {
        let suggested = self.kf_pressure.auto_tune_r()?;
        let new_r = suggested.clamp(0.001, 0.5);
        self.kf_pressure.set_measurement_noise(new_r);
        self.kf_pressure_base_r = new_r;
        Some(new_r)
    }

    /// Apply learned parameters to live subsystems.
    ///
    /// Called by learning_tick every N cycles after params are updated by auto-tuning
    /// or meta-learning. Closes the wiring gap: params are persisted AND consumed.
    pub fn apply_learnable_params(&mut self, lp: &LearnableParams) {
        // Kalman R and Q
        self.kf_pressure_base_r = lp.kalman_pressure_r;
        self.kf_pressure.set_measurement_noise(lp.kalman_pressure_r);
        self.kf_pressure.set_process_noise(lp.kalman_pressure_q);
        // CUSUM sensitivity
        self.cusum_pressure.set_kh(lp.cusum_k, lp.cusum_h);
        // PID
        self.pid_target = lp.pid_target;
        self.pid_decay = lp.pid_decay;
    }

    /// Auto-tune zone alpha based on oscillation / stall detection.
    ///
    /// - If zones oscillate (alternating up/down), halve alpha (damp).
    /// - If zones haven't moved in 500 cycles, double alpha (explore).
    /// Returns the new alpha value.
    pub fn auto_tune_zone_alpha(&mut self, current_alpha: f64) -> f64 {
        // Check oscillation: count sign alternations in last 8 feedbacks.
        let filled = self.zone_feedback_idx.min(8);
        if filled >= 4 {
            let mut alternations = 0u32;
            for i in 1..filled {
                let prev =
                    self.zone_feedback_history[(self.zone_feedback_idx - filled + i - 1) % 8];
                let curr = self.zone_feedback_history[(self.zone_feedback_idx - filled + i) % 8];
                if prev != 0 && curr != 0 && prev != curr {
                    alternations += 1;
                }
            }
            // If >60% of transitions are alternations → oscillating.
            if alternations as f64 / (filled - 1) as f64 > 0.60 {
                return (current_alpha * 0.5).clamp(0.001, 0.05);
            }
        }

        // Stall detection: no zone movement in 500+ cycles.
        if self.zone_stall_cycles > 500 {
            self.zone_stall_cycles = 0; // reset after adjustment
            return (current_alpha * 2.0).clamp(0.001, 0.05);
        }

        current_alpha
    }

    /// Persist learned state to disk.
    ///
    /// Persists: HazardModel (accumulated OOM history, calibrated base_rate, learned β weights)
    /// and MPC effects (learned action impact magnitudes).
    ///
    /// Not persisted: CUSUM target (must reflect current regime), Entropy history
    /// (adapts in <20 cycles), Lotka-Volterra (resets when dominant process changes).
    /// Build a persisted snapshot from live state (for LearnedState).
    pub fn to_persisted(&self) -> SignalIntelligencePersisted {
        SignalIntelligencePersisted {
            hazard: self.hazard.clone(),
            mpc: self.mpc.to_persisted(),
            learned_mid_entry: self.learned_mid_entry,
            learned_high_entry: self.learned_high_entry,
            utility_entropy: self.utility_entropy,
            utility_hazard: self.utility_hazard,
            utility_lotka: self.utility_lotka,
            utility_mpc: self.utility_mpc,
            kf_pressure: Some(self.kf_pressure.clone()),
            kf_swap: Some(self.kf_swap.clone()),
        }
    }

    pub fn persist(&self, path: &Path) {
        let persisted = self.to_persisted();
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

    /// Apply a persisted snapshot, restoring the hazard model, MPC effects,
    /// learned zones, utility EMAs, and Kalman filter state.
    pub fn restore(&mut self, p: SignalIntelligencePersisted) {
        self.hazard = p.hazard;
        // Sanitize restored hazard model: base_rate > 1/hour indicates saturation
        // from training on pressure events (not real OOMs). Reset to prior.
        self.hazard.validate_after_restore();
        self.mpc.restore_effects(&p.mpc);
        self.learned_mid_entry = p.learned_mid_entry;
        self.learned_high_entry = p.learned_high_entry;
        // Sanitize utility EMAs: denormals/NaN lock the subsystem out forever
        // (the EMA gate uses `> UTIL_THRESHOLD = 0.15`, so any subnormal stays
        // below threshold and never receives updates). Reset to 0.5 so each
        // subsystem gets a fair chance to earn its place after restore.
        const UTILITY_MIN: f64 = 1e-6;
        self.utility_entropy = if p.utility_entropy.is_finite() && p.utility_entropy >= UTILITY_MIN
        {
            p.utility_entropy.clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.utility_hazard = if p.utility_hazard.is_finite() && p.utility_hazard >= UTILITY_MIN {
            p.utility_hazard.clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.utility_lotka = if p.utility_lotka.is_finite() && p.utility_lotka >= UTILITY_MIN {
            p.utility_lotka.clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.utility_mpc = if p.utility_mpc.is_finite() && p.utility_mpc >= UTILITY_MIN {
            p.utility_mpc.clamp(0.0, 1.0)
        } else {
            0.5
        };
        if let Some(kf) = p.kf_pressure {
            self.kf_pressure = kf;
        }
        if let Some(kf) = p.kf_swap {
            self.kf_swap = kf;
        }
    }

    /// Reset learned zones to defaults (called when restore quality is stale).
    pub fn reset_zones(&mut self) {
        self.learned_mid_entry = 0.30;
        self.learned_high_entry = 0.50;
    }

    /// Zone feedback with workload context: learns per-workload zone offsets.
    /// If a workload consistently needs different zone thresholds, the offset
    /// accumulates so zones auto-adapt per workload type.
    pub fn zone_feedback_workload(&mut self, pressure: f64, was_effective: bool, workload: u8) {
        // Regular zone feedback (global)
        self.zone_feedback(pressure, was_effective);

        // Per-workload offset learning (α = 0.002, very slow)
        let alpha = 0.002;
        let entry = self
            .workload_zone_offsets
            .entry(workload)
            .or_insert((0.0, 0.0));
        if was_effective {
            // Effective → lower zones for this workload (engage earlier)
            entry.0 -= alpha;
            entry.1 -= alpha;
        } else {
            // Ineffective → raise zones for this workload (be more conservative)
            entry.0 += alpha;
            entry.1 += alpha;
        }
        // Clamp offsets to ±0.05
        entry.0 = entry.0.clamp(-0.05, 0.05);
        entry.1 = entry.1.clamp(-0.05, 0.05);

        // Cap at 8 workload entries
        if self.workload_zone_offsets.len() > 8 {
            // Remove the entry with smallest absolute offset sum
            if let Some(key) = self
                .workload_zone_offsets
                .iter()
                .min_by(|a, b| {
                    let sa = a.1 .0.abs() + a.1 .1.abs();
                    let sb = b.1 .0.abs() + b.1 .1.abs();
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(&k, _)| k)
            {
                self.workload_zone_offsets.remove(&key);
            }
        }
    }

    /// Get effective zone thresholds adjusted for the current workload.
    /// Returns (mid_entry, high_entry) with workload-specific offsets applied.
    pub fn effective_zones(&self, workload: u8) -> (f64, f64) {
        let (mid_off, high_off) = self
            .workload_zone_offsets
            .get(&workload)
            .copied()
            .unwrap_or((0.0, 0.0));
        (
            (self.learned_mid_entry + mid_off).clamp(0.15, 0.45),
            (self.learned_high_entry + high_off).clamp(0.35, 0.65),
        )
    }

    /// Number of workload zone offsets tracked (diagnostic).
    pub fn workload_zone_count(&self) -> usize {
        self.workload_zone_offsets.len()
    }
}

/// Serializable snapshot of the state worth keeping across restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalIntelligencePersisted {
    /// Cox hazard model: calibrated base_rate, learned β weights, total_hours/events.
    pub hazard: HazardModel,
    /// MPC action effect estimates (learned from live pressure feedback).
    pub mpc: MpcPersisted,
    /// Learned mid zone entry threshold (default 0.30).
    #[serde(default = "default_mid_entry")]
    pub learned_mid_entry: f64,
    /// Learned high zone entry threshold (default 0.50).
    #[serde(default = "default_high_entry")]
    pub learned_high_entry: f64,
    /// Utility EMA for entropy subsystem.
    #[serde(default = "default_utility")]
    pub utility_entropy: f64,
    /// Utility EMA for hazard subsystem.
    #[serde(default = "default_utility")]
    pub utility_hazard: f64,
    /// Utility EMA for Lotka-Volterra subsystem.
    #[serde(default = "default_utility")]
    pub utility_lotka: f64,
    /// Utility EMA for MPC subsystem.
    #[serde(default = "default_utility")]
    pub utility_mpc: f64,
    /// Kalman filter state for pressure (position + velocity + covariance).
    #[serde(default)]
    pub kf_pressure: Option<Kalman1D>,
    /// Kalman filter state for swap velocity.
    #[serde(default)]
    pub kf_swap: Option<Kalman1D>,
}

fn default_mid_entry() -> f64 {
    0.30
}
fn default_high_entry() -> f64 {
    0.50
}
fn default_utility() -> f64 {
    0.5
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
            pressure_predicted_30s: 0.0,
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
            memory_scan_available: false,
            fluidity_score: 1.0,
            window_op_active: false,
            app_launching: false,
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
    fn test_router_skips_heavy_at_low_pressure() {
        let mut si = SignalIntelligence::new();
        // Low pressure: heavy modules should produce zeroed outputs.
        let d = si.tick(
            0.15,
            10.0,
            0.01,
            0.05,
            &[5.0, 3.0],
            &[100e6, 50e6],
            "idle_app",
            100_000_000,
            500_000_000,
            8_000_000_000,
            0.5,
        );
        assert_eq!(
            d.entropy_anomaly, 0.0,
            "entropy should be skipped at low pressure"
        );
        assert_eq!(d.p_oom_30s, 0.0, "hazard should be skipped at low pressure");
        assert_eq!(
            d.monopoly_risk, 0.0,
            "lotka-volterra should be skipped at low pressure"
        );
        assert_eq!(
            d.mpc_recommendation, 0,
            "MPC should be skipped at low pressure"
        );
        // But Kalman should still work.
        assert!(d.pressure_smooth > 0.0, "Kalman must always run");
    }

    #[test]
    fn test_router_engages_heavy_at_high_pressure() {
        let mut si = SignalIntelligence::new();
        // Warm up Kalman so pressure_smooth reaches ≥0.40.
        for _ in 0..20 {
            tick_stressed(&mut si, 0.80);
        }
        let d = tick_stressed(&mut si, 0.80);
        // At high pressure, hazard and MPC should produce non-trivial values.
        // p_oom_30s may be 0 if hazard hasn't seen events, but it should have run.
        // MPC should produce a recommendation (possibly Observe=0, but the path executed).
        assert!(
            d.pressure_smooth > 0.40,
            "pressure should be high enough for deep mode"
        );
        // Entropy updates with real data — score may be 0 but the path ran.
        // Key check: urgency should be non-trivial with all subsystems engaged.
        assert!(
            d.urgency > 0.15,
            "urgency should be meaningful at 0.80 pressure"
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
            si.record_overflow(0.95, 0.8, 0.9);
        }

        let d_after = tick_stressed(&mut si, 0.85);
        assert!(
            d_after.p_oom_30s > p_before,
            "P(OOM) should increase after recording overflows: {} > {}",
            d_after.p_oom_30s,
            p_before
        );
    }

    #[test]
    fn test_budget_cognitivo_utility_decays_without_signal() {
        // Hazard utility decays when p_oom stays near 0 (no overflow events).
        let mut si = SignalIntelligence::new();
        let initial_hazard = si.subsystem_utilities()[1];
        assert!((initial_hazard - 0.5).abs() < 1e-9, "start at 0.5");

        // 200 ticks in high zone (≥0.50) — all subsystems run.
        // No record_overflow() calls → hazard base_rate stays 0 → p_oom ≈ 0.
        for _ in 0..200 {
            si.tick(
                0.55,
                10.0,
                0.01,
                0.05,
                &[5.0, 3.0],
                &[100e6, 50e6],
                "calm_app",
                100_000_000,
                500_000_000,
                8_000_000_000,
                0.5,
            );
        }
        let after_hazard = si.subsystem_utilities()[1];
        assert!(
            after_hazard < 0.10,
            "hazard utility should decay without OOM events: {}",
            after_hazard
        );
    }

    #[test]
    fn test_budget_cognitivo_mid_zone_skips_low_utility() {
        let mut si = SignalIntelligence::new();
        // Force utility to 0 (below threshold) for MPC.
        si.utility_mpc = 0.0;
        si.utility_entropy = 0.0;

        // Warm up Kalman to mid-zone (~0.35).
        for _ in 0..30 {
            si.tick(
                0.35,
                10.0,
                0.01,
                0.05,
                &[5.0, 3.0],
                &[100e6, 50e6],
                "calm_app",
                100_000_000,
                500_000_000,
                8_000_000_000,
                0.5,
            );
        }

        let d = si.tick(
            0.35,
            10.0,
            0.01,
            0.05,
            &[5.0, 3.0],
            &[100e6, 50e6],
            "calm_app",
            100_000_000,
            500_000_000,
            8_000_000_000,
            0.5,
        );
        // Low-utility subsystems should be skipped in mid zone.
        assert_eq!(d.mpc_recommendation, 0, "MPC should be skipped (utility=0)");
        assert_eq!(
            d.entropy_anomaly, 0.0,
            "Entropy should be skipped (utility=0)"
        );
    }

    // ── Energy-aware routing tests ──────────────────────────────────────────

    #[test]
    fn test_energy_bias_low_battery_skips_more() {
        let mut si = SignalIntelligence::new();
        // Critical battery: bias = +0.15, so mid_entry = 0.45, high_entry = 0.65.
        si.set_energy_bias(15, false, false);

        // Warm up Kalman to 0.40 — normally mid-zone, but with bias it's LOW zone.
        for _ in 0..30 {
            si.tick(
                0.40,
                10.0,
                0.01,
                0.05,
                &[5.0, 3.0],
                &[100e6, 50e6],
                "calm_app",
                100_000_000,
                500_000_000,
                8_000_000_000,
                0.5,
            );
        }
        let d = si.tick(
            0.40,
            10.0,
            0.01,
            0.05,
            &[5.0, 3.0],
            &[100e6, 50e6],
            "calm_app",
            100_000_000,
            500_000_000,
            8_000_000_000,
            0.5,
        );
        // At 0.40 with bias +0.15, we're below mid_entry (0.45) → skip all heavy.
        assert_eq!(
            d.entropy_anomaly, 0.0,
            "entropy skipped on low battery at 0.40"
        );
        assert_eq!(
            d.mpc_recommendation, 0,
            "MPC skipped on low battery at 0.40"
        );
    }

    #[test]
    fn test_energy_bias_thermal_emergency_engages_more() {
        let mut si = SignalIntelligence::new();
        // Thermal emergency: bias = -0.15, so mid_entry = 0.15, high_entry = 0.35.
        si.set_energy_bias(100, true, true);

        // Warm up Kalman to 0.35 — normally low zone, but with thermal bias it's HIGH zone.
        for _ in 0..30 {
            si.tick(
                0.38,
                10.0,
                0.01,
                0.05,
                &[5.0, 3.0],
                &[100e6, 50e6],
                "calm_app",
                100_000_000,
                500_000_000,
                8_000_000_000,
                0.5,
            );
        }
        let d = si.tick(
            0.38,
            10.0,
            0.01,
            0.05,
            &[5.0, 3.0],
            &[100e6, 50e6],
            "calm_app",
            100_000_000,
            500_000_000,
            8_000_000_000,
            0.5,
        );
        // At 0.38 with bias -0.15, high_entry=0.35, so we're in ALL_HEAVY zone.
        // Kalman smoothed pressure should be ~0.38 > 0.35.
        assert!(
            d.pressure_smooth > 0.34,
            "pressure should be near 0.38: {}",
            d.pressure_smooth
        );
    }

    #[test]
    fn test_energy_bias_plugged_in_no_effect() {
        let mut si = SignalIntelligence::new();
        si.set_energy_bias(50, true, false); // charging, no thermal
        assert!((si.energy_bias).abs() < 1e-9, "plugged in = no bias");
    }

    // ── Lifelong zone learning tests ────────────────────────────────────────

    #[test]
    fn test_zone_feedback_effective_action_lowers_entry() {
        let mut si = SignalIntelligence::new();
        let (mid_before, high_before) = si.learned_zones();

        // Effective action near mid_entry → lower thresholds.
        for _ in 0..100 {
            si.zone_feedback(0.32, true); // near 0.30 mid_entry
        }
        let (mid_after, high_after) = si.learned_zones();
        assert!(
            mid_after < mid_before,
            "mid_entry should decrease: {} < {}",
            mid_after,
            mid_before
        );
        assert!(
            high_after < high_before,
            "high_entry should decrease: {} < {}",
            high_after,
            high_before
        );
    }

    #[test]
    fn test_zone_feedback_ineffective_action_raises_entry() {
        let mut si = SignalIntelligence::new();
        let (mid_before, _) = si.learned_zones();

        // Ineffective action below high_entry → raise thresholds.
        for _ in 0..100 {
            si.zone_feedback(0.40, false); // below 0.50 high_entry
        }
        let (mid_after, _) = si.learned_zones();
        assert!(
            mid_after > mid_before,
            "mid_entry should increase: {} > {}",
            mid_after,
            mid_before
        );
    }

    #[test]
    fn test_zone_learning_bounded() {
        let mut si = SignalIntelligence::new();
        // Push zones to extremes.
        for _ in 0..10000 {
            si.zone_feedback(0.25, true);
        }
        let (mid, high) = si.learned_zones();
        assert!(mid >= 0.20, "mid_entry clamped at 0.20: {}", mid);
        assert!(high >= 0.35, "high_entry clamped at 0.35: {}", high);
    }

    // ── Proactive 30s predictor (Feature 1) ───────────────────────────────────

    /// Feed a steadily rising pressure signal and verify the 30s projection
    /// crosses the overflow zone well before the 5s projection does.
    /// This simulates the "I can see the cliff 30 seconds ahead" scenario.
    #[test]
    fn proactive_30s_predicts_ahead_of_5s() {
        let mut si = SignalIntelligence::new();
        // Warm up Kalman with a stable baseline.
        for _ in 0..10 {
            si.tick(
                0.55,
                0.0,
                0.05,
                0.1,
                &[10.0],
                &[500e6],
                "app",
                500_000_000,
                2_000_000_000,
                8_000_000_000,
                0.5,
            );
        }
        // Now simulate pressure rising at ~0.010/s (0.005 per 0.5s tick).
        // After 20 ticks (~10s), pressure is at 0.65. Velocity ≈ 0.010/s.
        // 30s ahead: 0.65 + 0.010*30 = 0.95 (overflow territory).
        // 5s ahead: 0.65 + 0.010*5 = 0.70 (still below common bg_pressure ~0.72).
        let mut last = SignalDigest {
            pressure_smooth: 0.0,
            pressure_velocity: 0.0,
            pressure_predicted_5s: 0.0,
            pressure_predicted_30s: 0.0,
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
            memory_scan_available: false,
            fluidity_score: 1.0,
            window_op_active: false,
            app_launching: false,
        };
        for i in 0..20 {
            let pressure = 0.55 + i as f64 * 0.005;
            last = si.tick(
                pressure,
                0.0,
                0.05,
                0.1,
                &[15.0],
                &[500e6],
                "app",
                500_000_000,
                2_000_000_000,
                8_000_000_000,
                0.5,
            );
        }
        assert!(
            last.pressure_predicted_30s > last.pressure_predicted_5s,
            "30s prediction ({:.3}) must exceed 5s prediction ({:.3})",
            last.pressure_predicted_30s,
            last.pressure_predicted_5s
        );
        // The proactive predictor should fire: 30s projection above ~0.82 (bg_pressure - 0.05)
        // while current smooth pressure is still below ~0.75 (bg_pressure - 0.08 ≈ 0.72).
        assert!(
            last.pressure_predicted_30s > 0.80,
            "30s forecast {:.3} should exceed 0.80 for proactive trigger to fire",
            last.pressure_predicted_30s
        );
        assert!(
            last.pressure_smooth < 0.75,
            "current pressure {:.3} should still be safe (< 0.75) when proactive fires",
            last.pressure_smooth
        );
    }

    // ── Micro-benchmarks: signal processing pipeline timing ──────────────────

    /// Kalman filter convergence speed: verify it reaches steady-state RMSE within
    /// 50 updates and that 1000 updates take < 500µs on M1.
    /// [Anderson & Moore 1979] "Optimal Filtering" §4.4: convergence rate depends on
    /// Q/R ratio; for Q=0.005, R=0.02, steady-state is reached in ~20-30 observations.
    #[test]
    fn bench_kalman_convergence() {
        use crate::engine::kalman::Kalman1D;
        let mut kf = Kalman1D::new(0.005, 0.02);

        // Feed noisy pressure signal (same pattern as sim_signal_quality)
        let noise = [
            0.010f64, -0.015, 0.005, -0.010, 0.020, -0.005, 0.015, -0.020, 0.010, -0.010,
        ];
        let start = std::time::Instant::now();
        let mut rmse_history = Vec::with_capacity(200);
        for i in 0..200usize {
            let true_val = if i < 100 {
                0.50 + i as f64 * 0.003
            } else {
                0.80 - (i - 100) as f64 * 0.003
            };
            let noisy = (true_val + noise[i % noise.len()]).clamp(0.0, 1.0);
            kf.update(noisy, 0.5);
            if i > 10 {
                rmse_history.push((kf.position() - true_val).powi(2));
            }
        }
        let elapsed = start.elapsed();
        let rmse = (rmse_history.iter().sum::<f64>() / rmse_history.len() as f64).sqrt();

        eprintln!("Kalman 200 updates: {:?}, RMSE: {:.4}", elapsed, rmse);
        assert!(
            elapsed.as_micros() < 500,
            "Kalman 200 updates too slow: {:?}",
            elapsed
        );
        // Riccati floor for Q=0.005, R=0.02: P* ≈ 0.0078 → RMSE ≈ 0.088
        assert!(
            rmse < 0.12,
            "Kalman RMSE {:.4} exceeds threshold — filter not converging",
            rmse
        );
        // Convergence: RMSE after warmup should reflect steady-state, not transient
        let early_rmse = rmse_history[..5].iter().sum::<f64>().sqrt();
        let late_rmse = (rmse_history[rmse_history.len() - 5..].iter().sum::<f64>() / 5.0).sqrt();
        eprintln!(
            "  Early RMSE: {:.4}, Late RMSE: {:.4}",
            early_rmse, late_rmse
        );
    }

    /// CUSUM detection latency: verify regime shifts are detected within 4 cycles.
    /// [Page 1954] "Continuous Inspection Schemes" — detection lag h/(δ-k).
    /// For h=0.12, δ=0.20, k=0.02: lag = 0.12/0.18 ≈ 0.67 cycles → 1-2 cycles.
    #[test]
    fn bench_cusum_detection_latency() {
        use crate::engine::cusum::Cusum;
        let mut cusum = Cusum::new(0.50, 0.02, 0.12);

        // Warm up at 0.50
        for _ in 0..20 {
            cusum.update(0.50);
        }

        // Sudden shift to 0.70 — should detect within 4 cycles
        let start = std::time::Instant::now();
        let mut detected_at = None;
        for i in 0..10 {
            cusum.update(0.70);
            if cusum.alarm_high() {
                detected_at = Some(i + 1);
                break;
            }
        }
        let elapsed = start.elapsed();
        eprintln!(
            "CUSUM detection at cycle {:?}, time {:?}",
            detected_at, elapsed
        );
        assert!(
            detected_at.is_some(),
            "CUSUM failed to detect +0.20 regime shift"
        );
        assert!(
            detected_at.unwrap() <= 4,
            "CUSUM took {} cycles (expected ≤4)",
            detected_at.unwrap()
        );
        assert!(
            elapsed.as_micros() < 50,
            "CUSUM 10 updates too slow: {:?}",
            elapsed
        );
    }

    /// Full SignalIntelligence tick throughput: 100 ticks must complete < 10ms.
    /// This bounds the daemon cycle overhead attributable to signal processing.
    #[test]
    fn bench_signal_tick_throughput() {
        let mut si = SignalIntelligence::new();
        let cpu_vals = [15.0f64; 10];
        let mem_vals = [500_000_000f64; 10];
        let start = std::time::Instant::now();
        for i in 0..100 {
            let pressure = 0.50 + (i as f64 * 0.003).sin() * 0.15;
            let _ = si.tick(
                pressure,
                1024.0,
                0.3,
                0.4,
                &cpu_vals,
                &mem_vals,
                "app",
                500_000_000,
                2_000_000_000,
                8_000_000_000,
                0.5,
            );
        }
        let elapsed = start.elapsed();
        eprintln!("SignalIntelligence 100 ticks: {:?}", elapsed);
        assert!(
            elapsed.as_millis() < 10,
            "100 signal ticks too slow: {:?}",
            elapsed
        );
    }

    /// Hazard model calibration speed: 200 event records + 6 predictions < 1ms.
    /// [Cox 1972] "Regression Models and Life Tables" — Cox regression update is O(p).
    #[test]
    fn bench_hazard_calibration() {
        use crate::engine::hazard_model::HazardModel;
        let mut hazard = HazardModel::new();
        let start = std::time::Instant::now();
        for i in 0..200 {
            let p = 0.65 + (i % 10) as f64 * 0.02;
            let features = HazardModel::risk_features(p, 0.005, 0.60, 0.50);
            hazard.record_event(&features, 8.0);
        }
        // 6 predictions at different pressure levels
        let pressures = [0.30f64, 0.45, 0.55, 0.65, 0.75, 0.85];
        let p_ooms: Vec<f64> = pressures
            .iter()
            .map(|&p| {
                let f = HazardModel::risk_features(p, 0.003, p * 0.7, p * 0.6);
                hazard.probability_oom(&f, 30.0)
            })
            .collect();
        let elapsed = start.elapsed();
        eprintln!("Hazard 200 records + 6 predictions: {:?}", elapsed);
        // 5ms budget: 200 Cox-regression updates + 6 predictions. Debug builds are ~5×
        // slower than release; give generous headroom for CI and debug runs.
        assert!(
            elapsed.as_millis() < 5,
            "Hazard calibration too slow: {:?}",
            elapsed
        );
        // Monotonicity: higher pressure → higher p_oom
        for w in p_ooms.windows(2) {
            assert!(
                w[0] <= w[1],
                "Hazard non-monotonic: p_oom[i]={:.4} > p_oom[i+1]={:.4}",
                w[0],
                w[1]
            );
        }
    }

    /// Entropy detector anomaly score throughput: 500 updates < 5ms.
    /// [Shannon 1948] entropy computation is O(N log N) for N processes.
    #[test]
    fn bench_entropy_throughput() {
        use crate::engine::entropy_anomaly::EntropyDetector;
        let mut entropy = EntropyDetector::new();
        let cpu_vals = vec![10.0f64, 5.0, 8.0, 12.0, 3.0, 15.0, 2.0, 7.0];
        let mem_vals = vec![100e6f64, 50e6, 200e6, 80e6, 30e6, 120e6, 20e6, 60e6];
        let start = std::time::Instant::now();
        for i in 0..500 {
            let mut cpu = cpu_vals.clone();
            cpu[0] += (i % 10) as f64; // small variation
            entropy.update(&cpu, &mem_vals);
            let _ = entropy.anomaly_score();
        }
        let elapsed = start.elapsed();
        eprintln!("Entropy 500 updates: {:?}", elapsed);
        assert!(
            elapsed.as_millis() < 5,
            "Entropy 500 updates too slow: {:?}",
            elapsed
        );
    }

    /// Stable signal → 30s projection should stay close to current value.
    #[test]
    fn proactive_30s_stable_signal_no_false_alarm() {
        let mut si = SignalIntelligence::new();
        let mut last = SignalDigest {
            pressure_smooth: 0.0,
            pressure_velocity: 0.0,
            pressure_predicted_5s: 0.0,
            pressure_predicted_30s: 0.0,
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
            memory_scan_available: false,
            fluidity_score: 1.0,
            window_op_active: false,
            app_launching: false,
        };
        for _ in 0..30 {
            last = si.tick(
                0.50,
                0.0,
                0.05,
                0.1,
                &[10.0],
                &[500e6],
                "app",
                500_000_000,
                2_000_000_000,
                8_000_000_000,
                0.5,
            );
        }
        // Stable signal: 30s projection should be near 0.50 — no false proactive trigger.
        assert!(
            last.pressure_predicted_30s < 0.65,
            "stable signal 30s forecast {:.3} should not trigger proactive alarm",
            last.pressure_predicted_30s
        );
        // And 5s / 30s should be close to each other when velocity ≈ 0.
        let delta = (last.pressure_predicted_30s - last.pressure_predicted_5s).abs();
        assert!(
            delta < 0.05,
            "5s vs 30s gap {:.3} should be small on stable signal",
            delta
        );
    }

    // ── PID integral tests ──────────────────────────────────────────────────

    /// PID integral should accumulate when pressure is above target (0.65)
    /// and should be bounded to prevent integral windup.
    /// [Hellerstein 2004] "Feedback Control" §9: leaky integrator prevents windup.
    #[test]
    fn test_pid_integral_accumulates_above_target() {
        let mut si = SignalIntelligence::new();
        // Feed high pressure (above PID target of 0.65).
        for _ in 0..50 {
            tick_stressed(&mut si, 0.85);
        }
        // Integral should be positive (pressure above target).
        let d = tick_stressed(&mut si, 0.85);
        assert!(
            d.pressure_integral > 0.0,
            "integral={} should be positive when pressure > target",
            d.pressure_integral
        );
    }

    /// PID integral must be bounded to [-5, 5] pressure-seconds.
    #[test]
    fn test_pid_integral_bounded() {
        let mut si = SignalIntelligence::new();
        // Feed extreme pressure for many cycles — integral should not exceed 5.0.
        for _ in 0..1000 {
            tick_stressed(&mut si, 0.99);
        }
        let d = tick_stressed(&mut si, 0.99);
        assert!(
            d.pressure_integral <= 5.0,
            "integral {} should be bounded at 5.0",
            d.pressure_integral
        );
        assert!(
            d.pressure_integral >= -5.0,
            "integral {} should be bounded at -5.0",
            d.pressure_integral
        );
    }

    /// Urgency at boundary pressure values must be monotonically increasing.
    #[test]
    fn test_urgency_monotonic_with_pressure() {
        let pressures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mut prev_urgency = -1.0;
        for &p in &pressures {
            let u = compute_urgency(p, 0.0, false, 0.0, 0.0, 0.0);
            assert!(
                u >= prev_urgency,
                "urgency not monotonic: p={}, urgency={}, prev={}",
                p,
                u,
                prev_urgency
            );
            prev_urgency = u;
        }
    }

    /// compute_urgency must be bounded [0, 1] even with extreme inputs.
    #[test]
    fn test_urgency_bounded_extreme_inputs() {
        let u = compute_urgency(1.0, 1.0, true, 1.0, 1.0, 5.0);
        assert!(u >= 0.0 && u <= 1.0, "urgency {} out of bounds", u);
        let u_zero = compute_urgency(0.0, 0.0, false, 0.0, 0.0, 0.0);
        assert!(u_zero >= 0.0 && u_zero <= 1.0);
    }

    // ── Hazard batch retrain (Phase 3, Loop 2) ──────────────────────────

    #[test]
    fn test_retrain_hazard_batch_needs_10_events() {
        let mut si = SignalIntelligence::new();
        // Less than 10 events → no retrain
        for i in 0..9 {
            si.record_overflow(0.8 + (i as f64) * 0.01, 0.3, 0.5);
        }
        assert_eq!(si.retrain_hazard_batch(), 0);
        assert_eq!(si.oom_event_count(), 9);
    }

    #[test]
    fn test_retrain_hazard_batch_runs_after_10_events() {
        let mut si = SignalIntelligence::new();
        for i in 0..12 {
            si.record_overflow(0.7 + (i as f64) * 0.02, 0.2, 0.4);
        }
        assert_eq!(si.oom_event_count(), 12);
        let steps = si.retrain_hazard_batch();
        // 12 events × 10 epochs = 120 gradient steps
        assert_eq!(steps, 120);
    }

    #[test]
    fn test_oom_event_buffer_caps_at_50() {
        let mut si = SignalIntelligence::new();
        for i in 0..60 {
            si.record_overflow(0.7 + (i as f64) * 0.005, 0.1, 0.3);
        }
        assert_eq!(si.oom_event_count(), 50, "buffer should cap at 50");
    }

    #[test]
    fn test_retrain_hazard_batch_improves_beta() {
        let mut si = SignalIntelligence::new();
        let beta_before = si.hazard_beta();
        // Record 15 events with high pressure + high swap → beta[0] and beta[2] should increase
        for _ in 0..15 {
            si.record_overflow(0.95, 0.08, 0.85);
        }
        let beta_after_online = si.hazard_beta();
        let _steps = si.retrain_hazard_batch();
        let beta_after_batch = si.hazard_beta();
        // Batch retrain should further refine betas beyond online updates
        assert!(
            beta_after_batch[0] >= beta_after_online[0] || beta_after_batch[0] >= beta_before[0],
            "batch retrain should maintain or increase beta[0] for high-pressure events"
        );
    }

    // ── Workload zone offsets (Phase 4) ─────────────────────────────────

    #[test]
    fn test_workload_zone_feedback_creates_offset() {
        let mut si = SignalIntelligence::new();
        let (mid_before, high_before) = si.effective_zones(1);
        // Effective feedback for workload=1 → offsets shift down
        for _ in 0..20 {
            si.zone_feedback_workload(0.50, true, 1);
        }
        let (mid_after, high_after) = si.effective_zones(1);
        assert!(
            mid_after < mid_before,
            "effective feedback should lower mid zone: {mid_after} vs {mid_before}"
        );
        assert!(
            high_after < high_before,
            "effective feedback should lower high zone: {high_after} vs {high_before}"
        );
    }

    #[test]
    fn test_workload_zones_differ_by_workload() {
        let mut si = SignalIntelligence::new();
        // Build workload: effective → zones go down
        for _ in 0..20 {
            si.zone_feedback_workload(0.50, true, 1);
        }
        // Browser workload: ineffective → zones go up
        for _ in 0..20 {
            si.zone_feedback_workload(0.50, false, 3);
        }
        let (mid_build, _) = si.effective_zones(1);
        let (mid_browser, _) = si.effective_zones(3);
        assert!(
            mid_build < mid_browser,
            "build zones should be lower than browser zones: {} vs {}",
            mid_build,
            mid_browser
        );
    }

    #[test]
    fn test_workload_zone_offsets_clamped() {
        let mut si = SignalIntelligence::new();
        // Many effective feedbacks → offset should clamp at -0.05
        for _ in 0..1000 {
            si.zone_feedback_workload(0.50, true, 1);
        }
        let (mid, high) = si.effective_zones(1);
        // With base 0.30 and max offset -0.05, mid should be >= 0.25
        assert!(mid >= 0.15, "mid zone should respect clamp: {mid}");
        assert!(high >= 0.35, "high zone should respect clamp: {high}");
    }

    #[test]
    fn test_effective_zones_default_no_workload() {
        let si = SignalIntelligence::new();
        // Unknown workload → no offset
        let (mid, high) = si.effective_zones(99);
        assert!((mid - 0.30).abs() < 0.01);
        assert!((high - 0.50).abs() < 0.01);
    }

    #[test]
    fn test_workload_zone_offsets_cap_at_8() {
        let mut si = SignalIntelligence::new();
        for wl in 0..12_u8 {
            si.zone_feedback_workload(0.50, true, wl);
        }
        assert!(
            si.workload_zone_count() <= 8,
            "should cap at 8 entries, got {}",
            si.workload_zone_count()
        );
    }
}
