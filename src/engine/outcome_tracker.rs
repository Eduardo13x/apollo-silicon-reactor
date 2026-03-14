//! Outcome tracker — cierra el ciclo de retroalimentación del heurístico.
//!
//! Cuando Apollo throttlea un proceso, registra la presión de memoria antes.
//! 30 segundos después mide si bajó ≥5%. Si bajó: el throttle fue efectivo.
//! Si no bajó: el heurístico está gastando budget en algo inútil.
//!
//! Los resultados alimentan pesos Bayesianos por proceso (`PatternWeight`),
//! que a su vez informan al LLM cuándo el heurístico está fallando.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ── Tipos públicos ────────────────────────────────────────────────────────────

/// Peso Bayesiano de un patrón de proceso.
/// Bayesian estimate: effectiveness = (effective + 1) / (total + 2)  [Laplace smoothing]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PatternWeight {
    /// Veces que se throttleó este proceso.
    pub throttle_count: u32,
    /// Veces que el throttle fue efectivo (presión bajó ≥5% en 30s).
    pub effective_count: u32,
}

impl PatternWeight {
    /// Score Bayesiano [0,1]. Valores altos = este proceso sí causa presión.
    pub fn effectiveness(&self) -> f64 {
        (self.effective_count as f64 + 1.0) / (self.throttle_count as f64 + 2.0)
    }

    /// Umbral fijo (legacy). Usado en tests y cuando no hay baseline disponible.
    pub fn is_low_value(&self) -> bool {
        self.throttle_count >= 5 && self.effectiveness() < 0.30
    }

    /// Umbral calibrado contra la tasa base de fluctuación natural de presión.
    ///
    /// Un proceso es low-value solo si su efectividad es < 90% del baseline.
    /// Si baseline ≈ 0.25 (fluctuación natural), el umbral queda en ≈ 0.225.
    /// Requiere ≥20 throttles para tener suficiente señal estadística.
    pub fn is_low_value_vs_baseline(&self, baseline: f64) -> bool {
        self.throttle_count >= 20 && self.effectiveness() < baseline * 0.90
    }

    /// Proceso con ≥3 throttles y efectividad >75% — patrón de ruido confirmado.
    pub fn is_high_value(&self) -> bool {
        self.throttle_count >= 3 && self.effectiveness() > 0.75
    }
}

/// Throttle pendiente de resolución de outcome.
struct PendingOutcome {
    process_name: String,
    throttled_at: Instant,
    pressure_before: f64,
    /// Watts estimados del proceso en el momento del throttle (para record_savings).
    watts_before: f64,
}

/// Resumen de la resolución de un batch de outcomes.
pub struct OutcomeBatch {
    /// Nombres de procesos cuyo throttle fue efectivo esta ronda.
    pub effective_names: Vec<String>,
    /// Watts totales ahorrados por outcomes efectivos (para EnergyTracker).
    pub savings_watts: f64,
    /// Nombres de procesos marcados como low-value (heurístico fallando).
    pub low_value_names: Vec<String>,
}

// ── OutcomeTracker ────────────────────────────────────────────────────────────

pub struct OutcomeTracker {
    pending: VecDeque<PendingOutcome>,
    /// Pesos Bayesianos por nombre de proceso.
    pub weights: HashMap<String, PatternWeight>,
    /// Total de throttles que resultaron efectivos.
    pub total_effective: u32,
    /// Total de throttles resueltos.
    pub total_resolved: u32,
    /// EMA de tasa de caída de presión natural (≥2% en ventana de 30s),
    /// independientemente de qué proceso se throttleó. Calibra el umbral
    /// de is_low_value_vs_baseline contra la fluctuación de fondo.
    /// alpha ≈ 0.01 → half-life ≈ 69 observaciones.
    baseline_drop_ema: f64,
    /// Número de outcomes resueltos que alimentan el baseline.
    baseline_samples: u32,
}

impl OutcomeTracker {
    pub fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            weights: HashMap::new(),
            total_effective: 0,
            total_resolved: 0,
            baseline_drop_ema: 0.0,
            baseline_samples: 0,
        }
    }

    /// Umbral calibrado para is_low_value_vs_baseline.
    ///
    /// Requiere ≥50 muestras para estar bien establecido. Antes de eso
    /// retorna 0.15 (conservador — casi nada se skipea).
    /// Con baseline ≈ 0.25: threshold = 0.225.
    pub fn calibrated_threshold(&self) -> f64 {
        if self.baseline_samples < 50 {
            return 0.15; // sin datos suficientes: umbral conservador
        }
        (self.baseline_drop_ema * 0.90).max(0.10)
    }

    /// Registra un throttle aplicado. Llamar justo después de ejecutar la acción.
    pub fn record_throttle(&mut self, process_name: &str, pressure_before: f64, watts_before: f64) {
        // Actualiza contador de throttles para el peso Bayesiano.
        let w = self.weights.entry(process_name.to_string()).or_default();
        w.throttle_count += 1;

        self.pending.push_back(PendingOutcome {
            process_name: process_name.to_string(),
            throttled_at: Instant::now(),
            pressure_before,
            watts_before,
        });

        // Cap: si la cola crece demasiado, descarta los más viejos sin resolver.
        if self.pending.len() > 300 {
            self.pending.drain(..100);
        }
    }

    /// Resuelve los outcomes pendientes con más de 30s de antigüedad.
    /// Retorna un batch con los resultados para que el llamador actualice
    /// el EnergyTracker y la LearnedPolicy.
    pub fn tick(&mut self, current_pressure: f64) -> OutcomeBatch {
        const BASELINE_ALPHA: f64 = 0.01; // half-life ≈ 69 observaciones
        let check_after = Duration::from_secs(30);
        let mut effective_names = Vec::new();
        let mut savings_watts = 0.0_f64;

        while let Some(front) = self.pending.front() {
            if front.throttled_at.elapsed() < check_after {
                break;
            }
            let outcome = self.pending.pop_front().unwrap();
            let pressure_drop = outcome.pressure_before - current_pressure;
            let effective = pressure_drop >= 0.02;

            // Actualiza el baseline de fluctuación natural: ¿bajó la presión ≥2%
            // en esta ventana de 30s, independientemente de qué proceso causó qué?
            // Este EMA nos dice cuán frecuentemente la presión cae sola.
            let dropped = if effective { 1.0 } else { 0.0 };
            self.baseline_drop_ema =
                self.baseline_drop_ema * (1.0 - BASELINE_ALPHA) + dropped * BASELINE_ALPHA;
            self.baseline_samples = self.baseline_samples.saturating_add(1);

            if let Some(w) = self.weights.get_mut(&outcome.process_name) {
                if effective {
                    w.effective_count += 1;
                }
            }

            self.total_resolved += 1;
            if effective {
                self.total_effective += 1;
                effective_names.push(outcome.process_name.clone());
                savings_watts += outcome.watts_before;
            }
        }

        // Detecta patrones que ya tienen suficientes datos y están por debajo
        // del baseline calibrado — throttlearlos no aporta más que la fluctuación natural.
        let threshold = self.calibrated_threshold();
        let low_value_names: Vec<String> = self
            .weights
            .iter()
            .filter(|(_, w)| w.is_low_value_vs_baseline(threshold))
            .map(|(name, _)| name.clone())
            .collect();

        OutcomeBatch {
            effective_names,
            savings_watts,
            low_value_names,
        }
    }

    /// Efectividad global del heurístico [0,1].
    /// < 0.40 indica que el heurístico está fallando y conviene llamar al LLM.
    pub fn overall_effectiveness(&self) -> f64 {
        if self.total_resolved < 5 {
            return 0.5; // sin datos suficientes, asumir neutral
        }
        (self.total_effective as f64 + 1.0) / (self.total_resolved as f64 + 2.0)
    }

    /// True si el heurístico tiene patrones confirmados como low-value
    /// y la efectividad global es baja — señal para llamar al LLM.
    pub fn heuristic_is_struggling(&self) -> bool {
        self.total_resolved >= 10
            && self.overall_effectiveness() < 0.35
            && self.weights.values().any(|w| w.is_low_value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PatternWeight unit tests ──────────────────────────────────────────────

    #[test]
    fn pattern_weight_default_is_neutral() {
        let w = PatternWeight::default();
        // Laplace smoothing: (0+1)/(0+2) = 0.5
        assert!((w.effectiveness() - 0.5).abs() < 1e-6);
        assert!(!w.is_low_value(), "fresh weight must not be low-value");
        assert!(!w.is_high_value(), "fresh weight must not be high-value");
    }

    #[test]
    fn pattern_weight_low_value_threshold() {
        // ≥5 throttles, <30% effectiveness → low_value (legacy method)
        let mut w = PatternWeight {
            throttle_count: 5,
            effective_count: 0,
        };
        // effectiveness = (0+1)/(5+2) ≈ 0.143 < 0.30
        assert!(w.is_low_value());
        assert!(!w.is_high_value());

        // One effective result pushes it above 30% at count=5
        w.effective_count = 1;
        // effectiveness = (1+1)/(5+2) ≈ 0.286 < 0.30 → still low
        assert!(w.is_low_value());

        w.effective_count = 2;
        // effectiveness = (2+1)/(5+2) ≈ 0.429 → no longer low-value
        assert!(!w.is_low_value());
    }

    #[test]
    fn pattern_weight_low_value_vs_baseline_calibrated() {
        // baseline = 0.25 (fluctuación natural ~25%) → threshold = 0.225
        let baseline = 0.25_f64;

        // ≥20 throttles requeridos; con 19 nunca es low_value
        let w_insufficient = PatternWeight {
            throttle_count: 19,
            effective_count: 0,
        };
        assert!(!w_insufficient.is_low_value_vs_baseline(baseline));

        // 20 throttles, efectividad ≈ 0.143 < 0.225 → low_value
        let w_low = PatternWeight {
            throttle_count: 20,
            effective_count: 0,
        };
        // (0+1)/(20+2) ≈ 0.045 < 0.225
        assert!(w_low.is_low_value_vs_baseline(baseline));

        // 100 throttles, efectividad ≈ 0.248 > 0.225 → NOT low_value
        // (simula el caso real de nsurlsessiond con baseline ≈ 0.25)
        let w_borderline = PatternWeight {
            throttle_count: 100,
            effective_count: 24,
        };
        // (24+1)/(100+2) ≈ 0.245 > 0.225 → sigue throttleándose
        assert!(!w_borderline.is_low_value_vs_baseline(baseline));

        // 100 throttles, efectividad ≈ 0.158 < 0.225 → definitivamente low_value
        let w_confirmed = PatternWeight {
            throttle_count: 100,
            effective_count: 15,
        };
        // (15+1)/(100+2) ≈ 0.157 < 0.225
        assert!(w_confirmed.is_low_value_vs_baseline(baseline));
    }

    #[test]
    fn calibrated_threshold_conservative_until_enough_samples() {
        let mut tracker = OutcomeTracker::new();
        // < 50 muestras → umbral conservador 0.15
        assert!((tracker.calibrated_threshold() - 0.15).abs() < 1e-6);

        // Simular 50 muestras con baseline ≈ 0.25
        // EMA converge desde 0.0 — necesitamos muchas para llegar a 0.25
        // En lugar de eso, seteamos directamente para testear la lógica
        tracker.baseline_drop_ema = 0.25;
        tracker.baseline_samples = 50;
        // threshold = 0.25 * 0.90 = 0.225, max(0.10, 0.225) = 0.225
        let t = tracker.calibrated_threshold();
        assert!((t - 0.225).abs() < 1e-6, "expected 0.225, got {}", t);
    }

    #[test]
    fn calibrated_threshold_never_below_floor() {
        let mut tracker = OutcomeTracker::new();
        // baseline muy bajo (presión casi nunca fluctúa) → threshold = max(0.10, baseline*0.90)
        tracker.baseline_drop_ema = 0.05;
        tracker.baseline_samples = 100;
        // 0.05 * 0.90 = 0.045 → se aplica el floor: 0.10
        assert!((tracker.calibrated_threshold() - 0.10).abs() < 1e-6);
    }

    #[test]
    fn pattern_weight_high_value_threshold() {
        // ≥3 throttles, >75% effectiveness → high_value
        let w = PatternWeight {
            throttle_count: 3,
            effective_count: 3,
        };
        // effectiveness = (3+1)/(3+2) = 0.8 > 0.75
        assert!(w.is_high_value());
        assert!(!w.is_low_value());
    }

    #[test]
    fn pattern_weight_not_enough_data() {
        // <5 throttles → never low_value, regardless of effectiveness
        let w = PatternWeight {
            throttle_count: 4,
            effective_count: 0,
        };
        assert!(!w.is_low_value(), "need ≥5 throttles for low_value verdict");
    }

    // ── OutcomeTracker integration tests ─────────────────────────────────────

    #[test]
    fn record_throttle_increments_count() {
        let mut tracker = OutcomeTracker::new();
        tracker.record_throttle("Dropbox", 0.70, 1.5);
        tracker.record_throttle("Dropbox", 0.70, 1.5);

        let w = tracker.weights.get("Dropbox").unwrap();
        assert_eq!(w.throttle_count, 2);
        assert_eq!(w.effective_count, 0);
    }

    #[test]
    fn tick_marks_effective_when_pressure_drops() {
        let mut tracker = OutcomeTracker::new();
        // Simulate a throttle that happened 31s ago by manipulating pending directly.
        tracker.pending.push_back(super::PendingOutcome {
            process_name: "Dropbox".to_string(),
            throttled_at: Instant::now() - Duration::from_secs(31),
            pressure_before: 0.80,
            watts_before: 2.0,
        });
        // Also add throttle_count so weights exist.
        tracker.weights.insert(
            "Dropbox".to_string(),
            PatternWeight {
                throttle_count: 1,
                effective_count: 0,
            },
        );

        // Pressure dropped by 0.05 (≥ 0.02 threshold) → effective.
        let batch = tracker.tick(0.75);
        assert_eq!(batch.effective_names, vec!["Dropbox"]);
        assert!(batch.savings_watts > 0.0);

        let w = tracker.weights.get("Dropbox").unwrap();
        assert_eq!(w.effective_count, 1);
    }

    #[test]
    fn tick_does_not_mark_effective_when_pressure_stable() {
        let mut tracker = OutcomeTracker::new();
        tracker.pending.push_back(super::PendingOutcome {
            process_name: "Dropbox".to_string(),
            throttled_at: Instant::now() - Duration::from_secs(31),
            pressure_before: 0.80,
            watts_before: 2.0,
        });
        tracker.weights.insert(
            "Dropbox".to_string(),
            PatternWeight {
                throttle_count: 1,
                effective_count: 0,
            },
        );

        // Pressure barely dropped (< 0.02) → ineffective.
        let batch = tracker.tick(0.79);
        assert!(batch.effective_names.is_empty());

        let w = tracker.weights.get("Dropbox").unwrap();
        assert_eq!(w.effective_count, 0);
    }

    #[test]
    fn low_value_names_reported_after_enough_ineffective_throttles() {
        let mut tracker = OutcomeTracker::new();
        // El método calibrado requiere ≥20 throttles y baseline establecido.
        tracker.weights.insert(
            "suggestd".to_string(),
            PatternWeight {
                throttle_count: 25,
                effective_count: 0,
            },
        );
        // Establecer baseline con ≥50 muestras para que calibrated_threshold() sea activo.
        tracker.baseline_drop_ema = 0.25;
        tracker.baseline_samples = 50;
        // threshold = 0.225; suggestd effectiveness = (0+1)/(25+2) ≈ 0.037 < 0.225

        let batch = tracker.tick(0.50);
        assert!(
            batch.low_value_names.contains(&"suggestd".to_string()),
            "suggestd should be reported as low-value (25 throttles, 0 effective)"
        );
    }

    #[test]
    fn overall_effectiveness_neutral_with_few_resolved() {
        let tracker = OutcomeTracker::new();
        // < 5 resolved → returns neutral 0.5
        assert!((tracker.overall_effectiveness() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn heuristic_not_struggling_with_insufficient_data() {
        let mut tracker = OutcomeTracker::new();
        // Only 9 resolved — below the 10 required
        tracker.total_resolved = 9;
        tracker.total_effective = 0;
        tracker.weights.insert(
            "some_proc".to_string(),
            PatternWeight {
                throttle_count: 9,
                effective_count: 0,
            },
        );
        assert!(!tracker.heuristic_is_struggling());
    }
}
