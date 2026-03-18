//! Transformer Predictor — ONNX inference for time-series anomaly detection.
//!
//! Loads a pre-trained Transformer model (exported from `scripts/train_transformer.py`)
//! and runs inference each daemon cycle to detect anomalous system behaviour that
//! Apollo's existing univariate models miss.
//!
//! When no ONNX model file is present (e.g. before first training), all methods
//! gracefully return 0.0 — Apollo works exactly as before.  Once the nightly
//! retrain job produces a model, the daemon hot-reloads it automatically.
//!
//! ## Anomaly score
//!
//! The model predicts the next system state vector given the last 120 observations.
//! The anomaly score is the normalised reconstruction error (MSE between predicted
//! and actual state).  High score = system behaving in a way the model hasn't
//! seen during training = potential emerging problem.
//!
//! Score normalisation uses an EWMA of recent errors (Holt 1957) so the score
//! is relative to the system's typical prediction error, not absolute.
//!
//! ## References
//!
//! - Vaswani et al. 2017, "Attention Is All You Need"
//! - Tuli et al. 2022, "TranAD" — reconstruction error as anomaly score
//! - Holt 1957 — EWMA for adaptive baseline tracking

use std::collections::VecDeque;
use std::path::Path;

use tract_onnx::prelude::*;

use crate::engine::telemetry_logger::{TelemetryVector, N_FEATURES};

/// Transformer context window length (must match training seq_len).
const SEQ_LEN: usize = 120;

/// EWMA smoothing factor for error baseline (α=0.05 → slow adaptation).
const EWMA_ALPHA: f32 = 0.05;

/// Feature normalisation statistics (mean and std per feature).
/// Loaded from `feature_stats.json` produced by the training script.
#[derive(Debug, Clone)]
pub struct FeatureStats {
    pub mean: [f32; N_FEATURES],
    pub std: [f32; N_FEATURES],
}

impl FeatureStats {
    /// Load from a JSON file with `{"mean": [...], "std": [...]}`.
    pub fn load(path: &Path) -> Option<Self> {
        let data = std::fs::read_to_string(path).ok()?;
        let parsed: serde_json::Value = serde_json::from_str(&data).ok()?;

        let mean_arr = parsed.get("mean")?.as_array()?;
        let std_arr = parsed.get("std")?.as_array()?;

        if mean_arr.len() != N_FEATURES || std_arr.len() != N_FEATURES {
            return None;
        }

        let mut mean = [0.0f32; N_FEATURES];
        let mut std = [1.0f32; N_FEATURES];

        for (i, v) in mean_arr.iter().enumerate() {
            mean[i] = v.as_f64()? as f32;
        }
        for (i, v) in std_arr.iter().enumerate() {
            let s = v.as_f64()? as f32;
            std[i] = if s.abs() < 1e-8 { 1.0 } else { s };
        }

        Some(FeatureStats { mean, std })
    }

    /// Z-score normalise a raw feature vector.
    fn normalise(&self, raw: &[f32; N_FEATURES]) -> [f32; N_FEATURES] {
        let mut out = [0.0f32; N_FEATURES];
        for i in 0..N_FEATURES {
            out[i] = (raw[i] - self.mean[i]) / self.std[i];
        }
        out
    }
}

impl Default for FeatureStats {
    fn default() -> Self {
        Self {
            mean: [0.0; N_FEATURES],
            std: [1.0; N_FEATURES],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// tract-onnx inference backend
// ═══════════════════════════════════════════════════════════════════════════

/// Compiled ONNX model ready for inference.
type RunModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

struct OnnxBackend {
    model: RunModel,
}

impl OnnxBackend {
    fn load(model_path: &Path) -> Option<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .ok()?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![1, SEQ_LEN as i64, N_FEATURES as i64],
                ),
            )
            .ok()?
            .into_optimized()
            .ok()?
            .into_runnable()
            .ok()?;
        Some(OnnxBackend { model })
    }

    /// Run inference on a normalised sequence [SEQ_LEN, N_FEATURES].
    /// Returns predicted sequence [SEQ_LEN, N_FEATURES].
    fn predict(&self, input: &[[f32; N_FEATURES]; SEQ_LEN]) -> Option<Vec<f32>> {
        let flat: Vec<f32> = input.iter().flat_map(|row| row.iter().copied()).collect();

        let tensor = tract_ndarray::Array3::from_shape_vec((1, SEQ_LEN, N_FEATURES), flat).ok()?;

        let result = self.model.run(tvec!(tensor.into_tensor().into())).ok()?;
        let output = result[0].to_array_view::<f32>().ok()?;
        Some(output.iter().copied().collect())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public predictor
// ═══════════════════════════════════════════════════════════════════════════

/// Transformer-based anomaly predictor.
///
/// If no ONNX model is present, `score()` returns 0.0.  Once a model appears
/// (via nightly retrain + hot-reload), inference activates automatically.
pub struct TransformerPredictor {
    /// Ring buffer of normalised feature vectors (last SEQ_LEN observations).
    ring: VecDeque<[f32; N_FEATURES]>,
    /// Feature normalisation statistics.
    stats: FeatureStats,
    /// EWMA of reconstruction error (baseline for normalisation).
    error_ewma: f32,
    /// Whether the predictor has a loaded model and is ready.
    ready: bool,
    /// Compiled ONNX backend.
    backend: Option<OnnxBackend>,
}

impl TransformerPredictor {
    /// Create a new predictor.  Attempts to load the ONNX model and feature stats.
    ///
    /// If either file is missing, the predictor operates in no-op mode
    /// (returns 0.0 for all scores) until `reload()` succeeds.
    pub fn new(model_path: &Path, stats_path: &Path) -> Self {
        let stats = FeatureStats::load(stats_path).unwrap_or_default();
        let backend = OnnxBackend::load(model_path);
        let ready = backend.is_some();

        if ready {
            eprintln!("[transformer] Model loaded from {}", model_path.display());
        }

        TransformerPredictor {
            ring: VecDeque::with_capacity(SEQ_LEN),
            stats,
            error_ewma: 0.0,
            ready,
            backend,
        }
    }

    /// Record a new telemetry vector and return the anomaly score.
    ///
    /// Returns a value in `[0.0, 1.0]`:
    /// - `0.0`: no anomaly (or model not loaded yet)
    /// - `> 0.5`: significant deviation from learned normal behaviour
    /// - `> 0.8`: severe anomaly — system behaving in unprecedented ways
    pub fn score(&mut self, vec: &TelemetryVector) -> f64 {
        // Normalise and push to ring buffer.
        let raw = *vec.as_f32_slice();
        let normalised = self.stats.normalise(&raw);

        if self.ring.len() >= SEQ_LEN {
            self.ring.pop_front();
        }
        self.ring.push_back(normalised);

        // Need full window + model to produce a score.
        if self.ring.len() < SEQ_LEN || !self.ready {
            return 0.0;
        }

        self.run_inference()
    }

    /// Run the actual model inference and compute anomaly score.
    fn run_inference(&mut self) -> f64 {
        let Some(ref backend) = self.backend else {
            return 0.0;
        };

        // Build input tensor from ring buffer.
        let mut input = [[0.0f32; N_FEATURES]; SEQ_LEN];
        for (i, row) in self.ring.iter().enumerate() {
            input[i] = *row;
        }

        let Some(output) = backend.predict(&input) else {
            return 0.0;
        };

        // Compute MSE between the last predicted step and the actual last observation.
        let pred_offset = (SEQ_LEN - 1) * N_FEATURES;
        if output.len() < pred_offset + N_FEATURES {
            return 0.0;
        }

        let actual = &self.ring[SEQ_LEN - 1];
        let mut mse = 0.0f32;
        for i in 0..N_FEATURES {
            let diff = output[pred_offset + i] - actual[i];
            mse += diff * diff;
        }
        mse /= N_FEATURES as f32;

        // Normalise with EWMA baseline (Holt 1957).
        self.error_ewma = EWMA_ALPHA * mse + (1.0 - EWMA_ALPHA) * self.error_ewma;
        let ratio = if self.error_ewma > 1e-8 {
            mse / self.error_ewma
        } else {
            0.0
        };

        // Map ratio to 0-1 score.  ratio=1 means "average error" → ~0.33.
        // ratio=3 means "3x normal error" → ~1.0.
        (ratio / 3.0).min(1.0) as f64
    }

    /// Whether the predictor has a loaded model ready for inference.
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Number of observations in the ring buffer.
    pub fn buffered(&self) -> usize {
        self.ring.len()
    }

    /// Try to hot-reload the model (e.g. after retraining).
    /// Returns true if the model was successfully loaded.
    pub fn reload(&mut self, model_path: &Path, stats_path: &Path) -> bool {
        if let Some(new_stats) = FeatureStats::load(stats_path) {
            self.stats = new_stats;
        }
        if let Some(new_backend) = OnnxBackend::load(model_path) {
            self.backend = Some(new_backend);
            self.ready = true;
            self.error_ewma = 0.0; // Reset baseline for new model.
            eprintln!(
                "[transformer] Model hot-reloaded from {}",
                model_path.display()
            );
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(pressure: f32) -> TelemetryVector {
        TelemetryVector {
            pressure_smooth: pressure,
            pressure_velocity: 0.0,
            pressure_predicted_5s: pressure,
            swap_velocity_smooth: 0.0,
            pressure_integral: 0.0,
            cusum_score: 0.0,
            entropy_anomaly: 0.0,
            p_oom_30s: 0.0,
            monopoly_risk: 0.0,
            urgency: 0.0,
            cpu_total: 0.3,
            compressor_ratio: 0.0,
            dominant_share: 0.0,
            latency_score: 0.0,
            active_proc_count: 0.1,
            thermal_score: 0.0,
        }
    }

    #[test]
    fn predictor_no_model_returns_zero() {
        let model_path = Path::new("/nonexistent/model.onnx");
        let stats_path = Path::new("/nonexistent/stats.json");
        let mut predictor = TransformerPredictor::new(model_path, stats_path);

        for _ in 0..SEQ_LEN {
            let score = predictor.score(&make_vec(0.5));
            assert!((score - 0.0).abs() < 1e-6);
        }

        assert!(!predictor.is_ready());
    }

    #[test]
    fn predictor_buffers_correctly() {
        let model_path = Path::new("/nonexistent/model.onnx");
        let stats_path = Path::new("/nonexistent/stats.json");
        let mut predictor = TransformerPredictor::new(model_path, stats_path);

        for i in 0..200 {
            predictor.score(&make_vec(i as f32 / 200.0));
        }

        assert_eq!(predictor.buffered(), SEQ_LEN);
    }

    #[test]
    fn feature_stats_load_valid() {
        let dir = std::env::temp_dir().join("apollo_test_stats");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("feature_stats.json");

        let mean_arr: Vec<f64> = vec![0.0; N_FEATURES];
        let std_arr: Vec<f64> = vec![1.0; N_FEATURES];
        let json = serde_json::json!({
            "mean": mean_arr,
            "std": std_arr,
            "feature_names": []
        });
        std::fs::write(&path, json.to_string()).unwrap();

        let stats = FeatureStats::load(&path);
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert!((stats.mean[0] - 0.0).abs() < 1e-6);
        assert!((stats.std[0] - 1.0).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn feature_stats_normalise() {
        let stats = FeatureStats {
            mean: [0.5; N_FEATURES],
            std: [0.1; N_FEATURES],
        };
        let raw = [0.6f32; N_FEATURES];
        let norm = stats.normalise(&raw);
        for v in &norm {
            assert!((*v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn feature_stats_load_missing_file() {
        let stats = FeatureStats::load(Path::new("/nonexistent/stats.json"));
        assert!(stats.is_none());
    }

    #[test]
    fn feature_stats_default() {
        let stats = FeatureStats::default();
        assert!((stats.mean[0] - 0.0).abs() < 1e-6);
        assert!((stats.std[0] - 1.0).abs() < 1e-6);
    }
}
