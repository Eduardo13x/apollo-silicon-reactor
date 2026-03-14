//! Gaussian Naive Bayes con features de hardware ARM64.
//!
//! El clasificador actual de Apollo es un acumulador de pesos — no actualiza
//! priors ni aprende distribuciones reales. Este módulo implementa NB correcto:
//!
//!   log P(workload | hw) ∝ log P(workload)
//!                         + Σᵢ log P(featureᵢ | workload)
//!
//! Features continuas → distribución Gaussiana por (workload, feature).
//! Update online via algoritmo de Welford (O(1), sin buffer de historia).
//! Persiste en disco como JSON entre reinicios del daemon.
//!
//! Integración con el clasificador existente:
//!   score_final = score_texto (existente) + log_likelihood_hardware (este módulo)

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::engine::user_profile::WorkloadType;

// ─── Workloads como índice ────────────────────────────────────────────────────

const N_WL: usize = 8;

const WL_ORDER: [WorkloadType; N_WL] = [
    WorkloadType::Coding,
    WorkloadType::VideoCall,
    WorkloadType::MediaPlayback,
    WorkloadType::VideoEdit,
    WorkloadType::OfficeWork,
    WorkloadType::CommandLine,
    WorkloadType::Idle,
    WorkloadType::General,
];

fn wl_idx(wl: WorkloadType) -> usize {
    WL_ORDER.iter().position(|w| *w == wl).unwrap_or(N_WL - 1)
}

// ─── Features de hardware ─────────────────────────────────────────────────────

const N_FEAT: usize = 3;

/// Feature 0: throughput de instrucciones (MIPS medido con cntvct_el0).
/// P-cores: ~800-1200. E-cores o bajo carga: ~200-500.
const F_THROUGHPUT: usize = 0;

/// Feature 1: jitter de scheduling (µs, medido con cntvct_el0).
/// Sistema nominal: ~0-50. Presión térmica: >200.
const F_JITTER: usize = 1;

/// Feature 2: tiempo total del pointer-chase de 16 MB (µs).
/// El timer cntvct_el0 corre a 24 MHz (≈42 ns/tick) — sin resolución de ns.
/// Nominal: ~3000-9000. Presión de memoria: ~9000-24000. Swap: >24000.
const F_CACHE: usize = 2;

pub struct HwFeatures {
    pub throughput_mips: f64,
    pub jitter_us: f64,
    pub cache_latency_us: f64,
}

impl HwFeatures {
    fn as_array(&self) -> [f64; N_FEAT] {
        [self.throughput_mips, self.jitter_us, self.cache_latency_us]
    }
}

// ─── Parámetros Gaussianos con update online (Welford) ───────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianParams {
    /// Media corriente (Welford).
    pub mean: f64,
    /// Suma de cuadrados de diferencias (Welford M2).
    m2: f64,
    /// Número de observaciones vistas.
    pub count: u64,
    /// Varianza mínima para evitar división por cero y overfitting.
    min_var: f64,
}

impl GaussianParams {
    fn new(seed_mean: f64, seed_std: f64) -> Self {
        // Sembramos con observaciones sintéticas basadas en conocimiento del hardware.
        // count=10 da peso razonable al prior sin dominar las observaciones reales.
        let seed_count = 10u64;
        Self {
            mean: seed_mean,
            m2: seed_std * seed_std * (seed_count - 1) as f64,
            count: seed_count,
            min_var: (seed_std * 0.1).powi(2), // mínimo 10% de la desviación seed
        }
    }

    /// Varianza estimada de la distribución.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            self.m2.max(self.min_var)
        } else {
            (self.m2 / (self.count - 1) as f64).max(self.min_var)
        }
    }

    /// Update online: O(1), sin guardar historia.
    /// Algoritmo de Welford — numéricamente estable.
    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Log-likelihood: log P(x | esta distribución Gaussiana).
    /// log N(x; μ, σ²) = -0.5 * (x-μ)²/σ² - 0.5 * log(2πσ²)
    pub fn log_likelihood(&self, x: f64) -> f64 {
        let var = self.variance();
        let diff = x - self.mean;
        -0.5 * (diff * diff / var) - 0.5 * (2.0 * PI * var).ln()
    }
}

// ─── Clasificador ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HwBayesClassifier {
    /// params[workload_idx][feature_idx] = distribución Gaussiana.
    params: [[GaussianParams; N_FEAT]; N_WL],
    /// Conteo de observaciones por workload (para prior).
    prior_counts: [u64; N_WL],
}

impl HwBayesClassifier {
    /// Crea el clasificador con seeds basados en comportamiento conocido del hardware.
    /// El clasificador mejora automáticamente con cada ciclo de Apollo.
    pub fn new() -> Self {
        // Seeds: (mean_throughput, std) (mean_jitter, std) (mean_cache, std)
        // Valores calibrados para Apple Silicon M-series.
        // cache_mean/std en µs totales del pointer-chase de 16 MB (262 144 accesos).
        // Conversión: latencia_ns_por_acceso × 262 = µs_totales
        let seeds: [(f64, f64, f64, f64, f64, f64); N_WL] = [
            // (tput_mean, tput_std, jitter_mean, jitter_std, cache_mean_us, cache_std_us)
            (900.0, 200.0, 40.0, 60.0, 9_000.0, 4_000.0), // Coding
            (650.0, 200.0, 120.0, 80.0, 8_000.0, 4_000.0), // VideoCall
            (500.0, 150.0, 60.0, 60.0, 6_500.0, 3_000.0), // MediaPlayback
            (800.0, 200.0, 80.0, 70.0, 16_000.0, 6_000.0), // VideoEdit: alta presión de mem
            (450.0, 150.0, 40.0, 50.0, 6_000.0, 2_500.0), // OfficeWork
            (850.0, 200.0, 50.0, 60.0, 10_000.0, 5_000.0), // CommandLine
            (250.0, 100.0, 20.0, 30.0, 3_000.0, 1_500.0), // Idle: E-cores, caches frías
            (550.0, 250.0, 60.0, 80.0, 7_000.0, 4_000.0), // General
        ];

        let mut params: [[GaussianParams; N_FEAT]; N_WL] =
            std::array::from_fn(|_| std::array::from_fn(|_| GaussianParams::new(0.0, 1.0)));

        for (i, (tm, ts, jm, js, cm, cs)) in seeds.iter().enumerate() {
            params[i][F_THROUGHPUT] = GaussianParams::new(*tm, *ts);
            params[i][F_JITTER] = GaussianParams::new(*jm, *js);
            params[i][F_CACHE] = GaussianParams::new(*cm, *cs);
        }

        // Prior uniforme inicial.
        let prior_counts = [10u64; N_WL];

        Self {
            params,
            prior_counts,
        }
    }

    /// Calcula log P(workload | hw_features) para todos los workloads.
    ///
    /// Retorna un array de (WorkloadType, log_posterior) ordenado por probabilidad.
    pub fn log_posteriors(&self, features: &HwFeatures) -> [(WorkloadType, f64); N_WL] {
        let feat = features.as_array();
        let total_prior: f64 = self.prior_counts.iter().sum::<u64>() as f64;

        let mut result = [(WorkloadType::General, 0.0f64); N_WL];

        for (wi, wl) in WL_ORDER.iter().enumerate() {
            // log prior: log P(workload)
            let log_prior =
                ((self.prior_counts[wi] as f64 + 1.0) / (total_prior + N_WL as f64)).ln();

            // log likelihood: Σ log P(featureᵢ | workload) — Naive Bayes (features independientes)
            let log_lik: f64 = feat
                .iter()
                .enumerate()
                .map(|(fi, &x)| self.params[wi][fi].log_likelihood(x))
                .sum();

            result[wi] = (*wl, log_prior + log_lik);
        }

        result
    }

    /// Clasifica y retorna el workload más probable + probabilidad normalizada.
    ///
    /// Normalización via log-sum-exp para estabilidad numérica.
    pub fn classify(&self, features: &HwFeatures) -> (WorkloadType, f64) {
        let posts = self.log_posteriors(features);

        // Log-sum-exp trick: max para estabilidad
        let max_log = posts
            .iter()
            .map(|(_, lp)| *lp)
            .fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = posts.iter().map(|(_, lp)| (lp - max_log).exp()).sum();

        let (best_wl, best_log) = posts
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or((WorkloadType::General, max_log));

        let probability = ((best_log - max_log).exp() / sum_exp).clamp(0.0, 1.0);

        (best_wl, probability)
    }

    /// Entrena con una observación etiquetada.
    ///
    /// Llamar cuando Apollo tiene alta confianza en el workload actual
    /// (ej: foreground app conocida + confidence > 0.70).
    /// Esto mejora el clasificador continuamente en background.
    pub fn observe(&mut self, features: &HwFeatures, workload: WorkloadType) {
        let wi = wl_idx(workload);
        let feat = features.as_array();

        for (fi, &x) in feat.iter().enumerate() {
            self.params[wi][fi].update(x);
        }
        self.prior_counts[wi] += 1;
    }

    /// Serializa los parámetros aprendidos para persistencia en disco.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Restaura parámetros aprendidos desde disco.
    /// Si el JSON es inválido, retorna un clasificador nuevo (no falla).
    pub fn from_json(json: &str) -> Self {
        serde_json::from_str(json).unwrap_or_else(|_| Self::new())
    }

    /// Diagnóstico: muestra qué aprendió el modelo de cada workload.
    pub fn summary(&self) -> Vec<WorkloadSummary> {
        WL_ORDER
            .iter()
            .enumerate()
            .map(|(wi, wl)| WorkloadSummary {
                workload: *wl,
                observations: self.prior_counts[wi],
                throughput_mean: self.params[wi][F_THROUGHPUT].mean,
                throughput_std: self.params[wi][F_THROUGHPUT].variance().sqrt(),
                jitter_mean: self.params[wi][F_JITTER].mean,
                cache_mean: self.params[wi][F_CACHE].mean,
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct WorkloadSummary {
    pub workload: WorkloadType,
    pub observations: u64,
    pub throughput_mean: f64,
    pub throughput_std: f64,
    pub jitter_mean: f64,
    pub cache_mean: f64,
}

impl Default for HwBayesClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_classify_correctly() {
        let clf = HwBayesClassifier::new();

        // M1 en compilación: P-cores a tope, poco jitter, cache normal
        let coding_hw = HwFeatures {
            throughput_mips: 950.0,
            jitter_us: 30.0,
            cache_latency_us: 8_400.0,
        };
        let (wl, conf) = clf.classify(&coding_hw);
        println!("Coding hw → {:?} conf={:.2}", wl, conf);
        assert!(
            matches!(wl, WorkloadType::Coding | WorkloadType::CommandLine),
            "throughput alto + jitter bajo debe predecir Coding, got {:?}",
            wl
        );

        // Sistema idle: E-cores, sin jitter, cache fría
        let idle_hw = HwFeatures {
            throughput_mips: 220.0,
            jitter_us: 15.0,
            cache_latency_us: 2_900.0,
        };
        let (wl, conf) = clf.classify(&idle_hw);
        println!("Idle hw    → {:?} conf={:.2}", wl, conf);
        assert_eq!(
            wl,
            WorkloadType::Idle,
            "throughput bajo + cache fría debe predecir Idle"
        );

        // Video edit: P-cores + alta presión de memoria
        let video_hw = HwFeatures {
            throughput_mips: 780.0,
            jitter_us: 90.0,
            cache_latency_us: 17_000.0,
        };
        let (wl, conf) = clf.classify(&video_hw);
        println!("VideoEdit  → {:?} conf={:.2}", wl, conf);
        assert!(matches!(wl, WorkloadType::VideoEdit | WorkloadType::Coding));
    }

    #[test]
    fn online_learning_updates_params() {
        let mut clf = HwBayesClassifier::new();

        // Entrenar con 20 observaciones de Idle con throughput muy bajo
        for _ in 0..20 {
            clf.observe(
                &HwFeatures {
                    throughput_mips: 100.0,
                    jitter_us: 5.0,
                    cache_latency_us: 2_100.0,
                },
                WorkloadType::Idle,
            );
        }

        let idle_idx = wl_idx(WorkloadType::Idle);
        let learned_mean = clf.params[idle_idx][F_THROUGHPUT].mean;
        println!("Idle throughput mean after 20 obs: {:.1}", learned_mean);

        // La media debe haberse movido hacia 100 (desde el seed 250)
        assert!(
            learned_mean < 250.0,
            "mean debe bajar hacia 100, got {:.1}",
            learned_mean
        );
        assert!(
            clf.prior_counts[idle_idx] > 10,
            "debe haber observaciones acumuladas"
        );
    }

    #[test]
    fn log_sum_exp_sums_to_one() {
        let clf = HwBayesClassifier::new();
        let hw = HwFeatures {
            throughput_mips: 600.0,
            jitter_us: 50.0,
            cache_latency_us: 6_500.0,
        };
        let posts = clf.log_posteriors(&hw);

        let max_log = posts
            .iter()
            .map(|(_, lp)| *lp)
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = posts.iter().map(|(_, lp)| (lp - max_log).exp()).sum();
        let probs: Vec<f64> = posts
            .iter()
            .map(|(_, lp)| (lp - max_log).exp() / sum)
            .collect();

        let total: f64 = probs.iter().sum();
        println!(
            "Probabilidades normalizadas: {:?}",
            probs
                .iter()
                .map(|p| format!("{:.3}", p))
                .collect::<Vec<_>>()
        );
        assert!(
            (total - 1.0).abs() < 1e-9,
            "probabilidades deben sumar 1, sum={}",
            total
        );
    }

    #[test]
    fn persistence_roundtrip() {
        let mut clf = HwBayesClassifier::new();
        clf.observe(
            &HwFeatures {
                throughput_mips: 900.0,
                jitter_us: 40.0,
                cache_latency_us: 7_800.0,
            },
            WorkloadType::Coding,
        );

        let json = clf.to_json();
        let restored = HwBayesClassifier::from_json(&json);

        let coding_idx = wl_idx(WorkloadType::Coding);
        assert_eq!(
            clf.prior_counts[coding_idx], restored.prior_counts[coding_idx],
            "prior_counts deben sobrevivir la serialización"
        );
        assert!(
            (clf.params[coding_idx][F_THROUGHPUT].mean
                - restored.params[coding_idx][F_THROUGHPUT].mean)
                .abs()
                < 1e-10
        );
    }
}
