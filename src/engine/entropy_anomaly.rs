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

use std::collections::{HashMap, VecDeque};

// ── NEON-accelerated f64 reductions ─────────────────────────────────────────
// Processes 2×f64 per cycle via float64x2_t on Apple Silicon.
// Falls back to scalar iterator on non-aarch64.

/// Sum a contiguous f64 slice using NEON 2-wide accumulation.
#[cfg(target_arch = "aarch64")]
fn neon_sum(data: &[f64]) -> f64 {
    use std::arch::aarch64::*;
    if data.len() < 4 {
        return data.iter().sum();
    }
    unsafe {
        let mut acc = vdupq_n_f64(0.0);
        let chunks = data.len() / 2;
        let remainder = data.len() % 2;
        for i in 0..chunks {
            let v = vld1q_f64(data.as_ptr().add(i * 2));
            acc = vaddq_f64(acc, v);
        }
        let mut total = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
        for i in (chunks * 2)..(chunks * 2 + remainder) {
            total += data[i];
        }
        total
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn neon_sum(data: &[f64]) -> f64 {
    data.iter().sum()
}

/// Sum of squared deviations from mean: Σ(x - mean)² using NEON.
#[cfg(target_arch = "aarch64")]
fn neon_sum_sq_dev(data: &[f64], mean: f64) -> f64 {
    use std::arch::aarch64::*;
    if data.len() < 4 {
        return data.iter().map(|x| (x - mean).powi(2)).sum();
    }
    unsafe {
        let mean_v = vdupq_n_f64(mean);
        let mut acc = vdupq_n_f64(0.0);
        let chunks = data.len() / 2;
        let remainder = data.len() % 2;
        for i in 0..chunks {
            let v = vld1q_f64(data.as_ptr().add(i * 2));
            let diff = vsubq_f64(v, mean_v);
            acc = vfmaq_f64(acc, diff, diff); // acc += diff * diff (FMA)
        }
        let mut total = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
        for i in (chunks * 2)..(chunks * 2 + remainder) {
            let d = data[i] - mean;
            total += d * d;
        }
        total
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn neon_sum_sq_dev(data: &[f64], mean: f64) -> f64 {
    data.iter().map(|x| (x - mean).powi(2)).sum()
}

/// Sum over a VecDeque using NEON (handles two-slice layout).
fn vecdeque_neon_sum(dq: &VecDeque<f64>) -> f64 {
    let (a, b) = dq.as_slices();
    neon_sum(a) + neon_sum(b)
}

/// Sum of squared deviations over a VecDeque using NEON.
fn vecdeque_neon_sq_dev(dq: &VecDeque<f64>, mean: f64) -> f64 {
    let (a, b) = dq.as_slices();
    neon_sum_sq_dev(a, mean) + neon_sum_sq_dev(b, mean)
}

/// Compact fingerprint of a workload distribution.
/// Buckets entropy into 0.25-bit bands and counts processes.
/// Two distributions with the same fingerprint produce similar anomaly scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkloadFingerprint {
    /// Entropy bucketed to 0.25 bits (e.g., 2.75 → 11).
    entropy_bucket: u16,
    /// Number of active processes (capped at 255).
    process_count: u8,
}

impl WorkloadFingerprint {
    fn from_entropy_and_count(entropy: f64, count: usize) -> Self {
        Self {
            entropy_bucket: (entropy / 0.25).round() as u16,
            process_count: count.min(255) as u8,
        }
    }
}

/// Cached knowledge about a fingerprint.
#[derive(Debug, Clone)]
struct FingerprintEntry {
    /// Running average anomaly score for this fingerprint.
    avg_anomaly: f64,
    /// How many times we've seen this fingerprint.
    hits: u32,
}

/// Detector de anomalía basado en entropía de Shannon.
#[derive(Debug)]
pub struct EntropyDetector {
    /// Entropía suavizada (EWMA).
    smoothed_entropy: f64,
    /// Historial reciente de entropías para calcular desviación (ring buffer O(1) push/pop).
    history: VecDeque<f64>,
    /// Máximo tamaño del historial.
    max_history: usize,
    /// Peso EWMA (0–1). Más alto = más peso a la observación reciente.
    alpha: f64,
    /// true una vez que tengamos suficientes observaciones.
    initialized: bool,
    /// Fingerprint cache: recognized workload patterns → expected anomaly score.
    fingerprints: HashMap<WorkloadFingerprint, FingerprintEntry>,
    /// Last fingerprint (for external query).
    last_fingerprint: Option<WorkloadFingerprint>,
    /// Cached anomaly score (reused when entropy is stable).
    cached_anomaly: f64,
    /// Last entropy value for which anomaly_score was computed.
    last_entropy_computed: f64,
}

impl EntropyDetector {
    pub fn new() -> Self {
        Self {
            smoothed_entropy: 0.0,
            history: VecDeque::with_capacity(61),
            max_history: 60, // ~30s de historia a 0.5s/ciclo
            alpha: 0.1,
            initialized: false,
            fingerprints: HashMap::new(),
            last_fingerprint: None,
            cached_anomaly: 0.0,
            last_entropy_computed: f64::NAN,
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
        let total: f64 = neon_sum(values);
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
    /// Also updates the fingerprint cache for pattern recognition.
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

        if self.history.len() >= self.max_history {
            self.history.pop_front(); // O(1) vs O(N) Vec::remove(0)
        }
        self.history.push_back(h_combined);

        // Generate fingerprint for this workload state.
        let process_count = cpu_values.len().max(mem_values.len());
        self.last_fingerprint =
            Some(WorkloadFingerprint::from_entropy_and_count(h_combined, process_count));
    }

    /// Score de anomalía: cuántas desviaciones estándar está la entropía actual
    /// respecto a su media reciente.
    ///
    /// - > 0: entropía por encima de lo normal (más procesos compitiendo)
    /// - < 0: entropía por debajo de lo normal (proceso dominante)
    /// - |score| > 2.0: anomalía significativa
    ///
    /// Also updates the fingerprint cache with the computed score.
    pub fn anomaly_score(&mut self) -> f64 {
        // Short-circuit: if entropy hasn't changed meaningfully, reuse cached score.
        // Saves O(N) mean+variance iteration over 60 samples when signal is stable.
        let current_entropy = *self.history.back().unwrap_or(&0.0);
        if (current_entropy - self.last_entropy_computed).abs() < 1e-4
            && !self.last_entropy_computed.is_nan()
        {
            // Still update fingerprint cache so recognition keeps working.
            if let Some(fp) = self.last_fingerprint {
                let score = self.cached_anomaly;
                let entry = self.fingerprints.entry(fp).or_insert(FingerprintEntry {
                    avg_anomaly: score,
                    hits: 0,
                });
                entry.hits += 1;
                entry.avg_anomaly += 0.1 * (score - entry.avg_anomaly);
            }
            return self.cached_anomaly;
        }

        let score = if self.history.len() < 5 {
            0.0
        } else {
            let n = self.history.len() as f64;
            let mean: f64 = vecdeque_neon_sum(&self.history) / n;
            let variance: f64 = vecdeque_neon_sq_dev(&self.history, mean) / n;
            let std_dev = variance.sqrt();
            if std_dev < 1e-6 {
                0.0
            } else {
                (current_entropy - mean) / std_dev
            }
        };
        self.cached_anomaly = score;
        self.last_entropy_computed = current_entropy;

        // Update fingerprint cache with this observation (even if score is 0).
        if let Some(fp) = self.last_fingerprint {
            let entry = self.fingerprints.entry(fp).or_insert(FingerprintEntry {
                avg_anomaly: score,
                hits: 0,
            });
            entry.hits += 1;
            entry.avg_anomaly += 0.1 * (score - entry.avg_anomaly);

            // GC: if cache grows too large, evict least-seen entries.
            if self.fingerprints.len() > 200 {
                let min_hits = self.fingerprints.values().map(|e| e.hits).min().unwrap_or(0);
                self.fingerprints.retain(|_, e| e.hits > min_hits);
            }
        }

        score
    }

    /// Entropía suavizada actual.
    pub fn smoothed(&self) -> f64 {
        self.smoothed_entropy
    }

    /// Entropía actual (última observación sin suavizar).
    pub fn current(&self) -> f64 {
        *self.history.back().unwrap_or(&0.0)
    }

    /// Check if the current workload fingerprint has been seen before.
    /// Returns (expected_anomaly, confidence) if recognized, None if new pattern.
    /// Confidence saturates at 1.0 after 50 observations.
    pub fn recognized_pattern(&self) -> Option<(f64, f64)> {
        let fp = self.last_fingerprint?;
        let entry = self.fingerprints.get(&fp)?;
        if entry.hits < 3 {
            return None; // not enough data
        }
        let confidence = (entry.hits as f64 / 50.0).min(1.0);
        Some((entry.avg_anomaly, confidence))
    }

    /// Number of unique fingerprints in the cache.
    pub fn fingerprint_count(&self) -> usize {
        self.fingerprints.len()
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
        det.update(&[5.0; 20], &[100.0; 20]);
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

    // ── Fingerprinting tests ────────────────────────────────────────────────

    #[test]
    fn test_fingerprint_built_on_update() {
        let mut det = EntropyDetector::new();
        assert!(det.last_fingerprint.is_none());
        det.update(&[30.0, 20.0, 10.0], &[500.0, 300.0, 100.0]);
        assert!(det.last_fingerprint.is_some());
    }

    #[test]
    fn test_fingerprint_recognized_after_repeated_pattern() {
        let mut det = EntropyDetector::new();
        let cpu = [30.0, 20.0, 15.0, 10.0, 5.0];
        let mem = [500.0, 300.0, 200.0, 100.0, 50.0];

        // 10 cycles to build history for z-score calculation.
        for _ in 0..10 {
            det.update(&cpu, &mem);
            det.anomaly_score();
        }

        // Same distribution → same fingerprint, should be recognized.
        assert!(det.fingerprint_count() > 0);
        // After enough hits, recognized_pattern should return Some.
        let result = det.recognized_pattern();
        assert!(result.is_some(), "pattern should be recognized after 10 hits");
        let (avg_anomaly, confidence) = result.unwrap();
        // Stable pattern → anomaly near 0.
        assert!(avg_anomaly.abs() < 1.5, "stable pattern anomaly should be low: {}", avg_anomaly);
        assert!(confidence > 0.0, "should have some confidence");
    }

    #[test]
    fn test_fingerprint_different_workloads_different_fingerprints() {
        let mut det = EntropyDetector::new();
        // Workload A: 5 processes.
        for _ in 0..5 {
            det.update(&[30.0, 20.0, 15.0, 10.0, 5.0], &[500.0, 300.0, 200.0, 100.0, 50.0]);
            det.anomaly_score();
        }
        let count_after_a = det.fingerprint_count();

        // Workload B: 20 uniform processes (very different entropy).
        for _ in 0..5 {
            det.update(&[5.0; 20], &[100.0; 20]);
            det.anomaly_score();
        }
        let count_after_b = det.fingerprint_count();

        assert!(
            count_after_b > count_after_a,
            "different workloads should produce different fingerprints: {} > {}",
            count_after_b, count_after_a
        );
    }

    #[test]
    fn test_fingerprint_cache_gc_limits_size() {
        let mut det = EntropyDetector::new();
        // Generate 250 distinct fingerprints (vary process count).
        for i in 1..=250 {
            let cpu: Vec<f64> = (0..i).map(|j| (j as f64 + 1.0) * 0.1).collect();
            let mem: Vec<f64> = (0..i).map(|j| (j as f64 + 1.0) * 100.0).collect();
            det.update(&cpu, &mem);
            det.anomaly_score(); // triggers GC when >200
        }
        assert!(
            det.fingerprint_count() <= 200,
            "cache should be bounded: {}",
            det.fingerprint_count()
        );
    }

    // ── NEON correctness tests ──────────────────────────────────────────

    #[test]
    fn test_neon_sum_matches_scalar() {
        let data: Vec<f64> = (0..60).map(|i| i as f64 * 0.1 + 0.5).collect();
        let scalar_sum: f64 = data.iter().sum();
        let neon_result = neon_sum(&data);
        assert!(
            (scalar_sum - neon_result).abs() < 1e-10,
            "neon_sum({}) != scalar_sum({})",
            neon_result,
            scalar_sum
        );
    }

    #[test]
    fn test_neon_sum_sq_dev_matches_scalar() {
        let data: Vec<f64> = (0..60).map(|i| i as f64 * 0.1 + 0.5).collect();
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let scalar: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
        let neon_result = neon_sum_sq_dev(&data, mean);
        assert!(
            (scalar - neon_result).abs() < 1e-8,
            "neon_sq_dev({}) != scalar({})",
            neon_result,
            scalar
        );
    }

    #[test]
    fn test_neon_sum_small_input() {
        // Edge case: fewer than 4 elements (falls back to scalar).
        assert!((neon_sum(&[1.0, 2.0]) - 3.0).abs() < 1e-15);
        assert!((neon_sum(&[]) - 0.0).abs() < 1e-15);
        assert!((neon_sum(&[42.0]) - 42.0).abs() < 1e-15);
    }
}
