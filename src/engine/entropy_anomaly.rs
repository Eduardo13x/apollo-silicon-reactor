//! Entropy Anomaly — detección de cambios en la distribución de procesos.
//!
//! ## Idea (Shannon, 1948)
//! H = -Σ pᵢ log₂(pᵢ) mide la "sorpresa" de una distribución.
//!
//! - Sistema estable (3 procesos dominan): H bajo (~2 bits)
//! - Sistema caótico (20 procesos compiten): H alto (~4+ bits)
//!
//! Un cambio rápido en H indica un regime shift en el workload:
//! - H sube: muchos procesos nuevos compitiendo (ej. build paralelo, tabs de browser)
//! - H baja: un proceso dominante (ej. compilación single-threaded, video rendering)
//!
//! ## Uso
//! ```ignore
//! let mut detector = EntropyDetector::new();
//! detector.update(&process_stats); // cada ciclo
//! if detector.anomaly_score() > 0.5 { /* workload changed significantly */ }
//! ```

/// Detector de anomalía basado en entropía de Shannon.
#[derive(Debug)]
pub struct EntropyDetector {
    /// Entropía suavizada (EWMA).
    smoothed_entropy: f64,
    /// Historial reciente de entropías para calcular desviación.
    history: Vec<f64>,
    /// Máximo tamaño del historial.
    max_history: usize,
    /// Peso EWMA (0–1). Más alto = más peso a la observación reciente.
    alpha: f64,
    /// true una vez que tengamos suficientes observaciones.
    initialized: bool,
}

impl EntropyDetector {
    pub fn new() -> Self {
        Self {
            smoothed_entropy: 0.0,
            history: Vec::with_capacity(60),
            max_history: 60, // ~30s de historia a 0.5s/ciclo
            alpha: 0.1,
            initialized: false,
        }
    }
}

impl Default for EntropyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl EntropyDetector {
    /// Calcula la entropía de una distribución de recursos (CPU o RAM).
    ///
    /// Recibe un slice de valores positivos (cpu_usage o memory_usage por proceso).
    /// Retorna H en bits.
    pub fn shannon_entropy(values: &[f64]) -> f64 {
        let total: f64 = values.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        let mut h = 0.0;
        for &v in values {
            if v > 0.0 {
                let p = v / total;
                h -= p * p.log2();
            }
        }
        h
    }

    /// Actualiza con las distribuciones de CPU y RAM de los procesos actuales.
    ///
    /// - `cpu_values`: cpu_usage de cada proceso (top N).
    /// - `mem_values`: memory_usage de cada proceso (top N).
    ///
    /// Usa la media de ambas entropías como señal combinada.
    pub fn update(&mut self, cpu_values: &[f64], mem_values: &[f64]) {
        let h_cpu = Self::shannon_entropy(cpu_values);
        let h_mem = Self::shannon_entropy(mem_values);
        let h_combined = (h_cpu + h_mem) / 2.0;

        if !self.initialized {
            self.smoothed_entropy = h_combined;
            self.initialized = true;
        } else {
            self.smoothed_entropy =
                self.alpha * h_combined + (1.0 - self.alpha) * self.smoothed_entropy;
        }

        self.history.push(h_combined);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Score de anomalía: cuántas desviaciones estándar está la entropía actual
    /// respecto a su media reciente.
    ///
    /// - > 0: entropía por encima de lo normal (más procesos compitiendo)
    /// - < 0: entropía por debajo de lo normal (proceso dominante)
    /// - |score| > 2.0: anomalía significativa
    pub fn anomaly_score(&self) -> f64 {
        if self.history.len() < 5 {
            return 0.0;
        }
        let mean: f64 = self.history.iter().sum::<f64>() / self.history.len() as f64;
        let variance: f64 = self.history.iter().map(|h| (h - mean).powi(2)).sum::<f64>()
            / self.history.len() as f64;
        let std_dev = variance.sqrt();
        if std_dev < 1e-6 {
            return 0.0;
        }
        let current = *self.history.last().unwrap_or(&mean);
        (current - mean) / std_dev
    }

    /// Entropía suavizada actual.
    pub fn smoothed(&self) -> f64 {
        self.smoothed_entropy
    }

    /// Entropía actual (última observación sin suavizar).
    pub fn current(&self) -> f64 {
        *self.history.last().unwrap_or(&0.0)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution_max_entropy() {
        // 8 procesos iguales → H = log₂(8) = 3.0
        let values = vec![1.0; 8];
        let h = EntropyDetector::shannon_entropy(&values);
        assert!((h - 3.0).abs() < 0.01, "uniform 8 → H={} (expected 3.0)", h);
    }

    #[test]
    fn test_single_dominant_low_entropy() {
        // 1 proceso con 99% + 9 procesos con ~0.1% cada uno
        let mut values = vec![0.01; 10];
        values[0] = 99.0;
        let h = EntropyDetector::shannon_entropy(&values);
        assert!(h < 0.5, "single dominant → H={} (expected <0.5)", h);
    }

    #[test]
    fn test_anomaly_stable_near_zero() {
        let mut det = EntropyDetector::new();
        // 20 ciclos con distribución estable.
        for _ in 0..20 {
            det.update(
                &[30.0, 20.0, 15.0, 10.0, 5.0],
                &[500.0, 300.0, 200.0, 100.0, 50.0],
            );
        }
        assert!(
            det.anomaly_score().abs() < 1.0,
            "stable system → low anomaly score"
        );
    }

    #[test]
    fn test_anomaly_spike_on_change() {
        let mut det = EntropyDetector::new();
        // 20 ciclos estables.
        for _ in 0..20 {
            det.update(
                &[30.0, 20.0, 15.0, 10.0, 5.0],
                &[500.0, 300.0, 200.0, 100.0, 50.0],
            );
        }
        // Cambio súbito: 20 procesos iguales (build paralelo).
        det.update(&vec![5.0; 20], &vec![100.0; 20]);
        assert!(
            det.anomaly_score() > 1.0,
            "sudden change → anomaly_score > 1.0, got {}",
            det.anomaly_score()
        );
    }

    #[test]
    fn test_empty_values_zero_entropy() {
        assert_eq!(EntropyDetector::shannon_entropy(&[]), 0.0);
        assert_eq!(EntropyDetector::shannon_entropy(&[0.0, 0.0]), 0.0);
    }
}
