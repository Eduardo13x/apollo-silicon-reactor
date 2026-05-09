//! ML Ligero — Lightweight local Bayesian workload classifier.
//!
//! Combines:
//!   1. LLM-learned pattern weights (updated via `update_learned_policy`)
//!   2. Foreground app matching
//!   3. Hour-of-day prior from UserProfile
//!   4. App recency from UserProfile
//!   5. Background process mix
//!
//! No network calls. Runs in <1 ms per classification.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::engine::{
    hw_bayes::{HwBayesClassifier, HwFeatures},
    llm::LearnedPolicy,
    user_profile::{workload_signatures, AppStats, HourProfile, WorkloadType},
};

use crate::engine::compressor_aware::RegionSummary;

/// Memory layout classification based on vm_region scan.
///
/// Each workload type has a distinctive memory region fingerprint:
/// - LLM inference: few huge anonymous regions (model weights + KV cache)
/// - Browser: many regions with significant shared memory (web content, JIT)
/// - Build system: many small file-backed regions (object files, headers)
/// - Database: large shared buffers, few total regions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryLayoutHint {
    LlmInference,
    BrowserCache,
    BuildSystem,
    DatabaseEngine,
    General,
}

/// Classify workload by memory region layout.
///
/// Returns `(hint, confidence)` if the layout matches a known pattern,
/// `None` if the layout is ambiguous.
pub fn classify_by_memory(summary: &RegionSummary) -> Option<(MemoryLayoutHint, f32)> {
    const MB: u64 = 1024 * 1024;

    // LLM: few regions, one huge anonymous private region (model weights).
    // llama.cpp / MLX typically mmap the full model as one region.
    if summary.largest_anon_bytes > 512 * MB && summary.n_regions < 100 {
        return Some((MemoryLayoutHint::LlmInference, 0.85));
    }

    // Browser: many regions (per-tab isolates, JIT segments) + large shared.
    if summary.n_regions > 300 && summary.shared_bytes > 50 * MB {
        return Some((MemoryLayoutHint::BrowserCache, 0.75));
    }

    // Build system: many small regions (object files, headers, temp allocs).
    if summary.n_regions > 200 && summary.mean_region_size < MB {
        return Some((MemoryLayoutHint::BuildSystem, 0.70));
    }

    // Database: large shared buffers (buffer pool), relatively few regions.
    if summary.shared_bytes > 200 * MB && summary.n_regions < 80 {
        return Some((MemoryLayoutHint::DatabaseEngine, 0.65));
    }

    None // ambiguous — return None, caller uses other classifiers
}

/// How a particular piece of evidence contributed to the classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClassifierSource {
    ForegroundApp,
    HourPrior,
    AppRecency(String),
    ProcessMix(u32), // # of matching process names
    LlmLearned,
    MemoryLayout(MemoryLayoutHint),
}

/// Result of one classify() call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadClassification {
    pub workload: WorkloadType,
    pub confidence: f32, // 0.0–1.0
    pub sources: Vec<ClassifierSource>,
}

impl WorkloadClassification {
    /// Returns the evidence sources as human-readable strings for logging/status.
    pub fn sources_summary(&self) -> Vec<String> {
        self.sources
            .iter()
            .map(|s| match s {
                ClassifierSource::ForegroundApp => "foreground-app".to_string(),
                ClassifierSource::HourPrior => "hour-prior".to_string(),
                ClassifierSource::AppRecency(app) => format!("recency:{}", app),
                ClassifierSource::ProcessMix(n) => format!("process-mix:{}", n),
                ClassifierSource::LlmLearned => "llm-learned".to_string(),
                ClassifierSource::MemoryLayout(hint) => format!("memory-layout:{:?}", hint),
            })
            .collect()
    }
}

/// A pattern learned from the LLM teacher, mapped to a workload type.
struct PatternWeight {
    pattern: String,
    workload: WorkloadType,
    weight: f32,
}

pub struct WorkloadClassifier {
    learned_weights: Vec<PatternWeight>,
    /// Gaussian NB sobre features de hardware ARM64.
    /// Se actualiza online con cada ciclo donde la confianza de texto es alta.
    pub hw_bayes: HwBayesClassifier,
}

impl WorkloadClassifier {
    pub fn new() -> Self {
        Self {
            learned_weights: Vec::new(),
            hw_bayes: HwBayesClassifier::new(),
        }
    }

    /// Call this whenever LearnedPolicy is updated (LLM retraining result).
    pub fn update_learned_policy(&mut self, policy: &LearnedPolicy) {
        self.learned_weights.clear();
        let sigs = workload_signatures();

        for pattern in &policy.interactive_patterns {
            // Map pattern to a workload type by checking which signature it matches.
            let workload = sigs
                .iter()
                .find(|(_, patterns)| patterns.iter().any(|p| pattern.contains(p)))
                .map(|(wl, _)| *wl)
                .unwrap_or(WorkloadType::General);
            self.learned_weights.push(PatternWeight {
                pattern: pattern.clone(),
                workload,
                weight: 1.5,
            });
        }
        // Noise patterns: de-boost (negative weight toward General)
        for pattern in &policy.noise_patterns {
            self.learned_weights.push(PatternWeight {
                pattern: pattern.clone(),
                workload: WorkloadType::General,
                weight: -0.5,
            });
        }
    }

    /// Bayesian workload classification.
    ///
    /// `hw`: features de hardware medidas con assembly (throughput, jitter, cache).
    /// Si es None, solo usa texto/hora (comportamiento anterior).
    pub fn classify(
        &self,
        foreground_app: Option<&str>,
        all_proc_names: &[&str],
        hour_model: &[HourProfile; 24],
        app_stats: &HashMap<String, AppStats>,
        hour_of_day: u8,
    ) -> WorkloadClassification {
        self.classify_with_hw(
            foreground_app,
            all_proc_names,
            hour_model,
            app_stats,
            hour_of_day,
            None,
        )
    }

    /// Versión con features de hardware. Fusiona el score de texto con el
    /// log-posterior del Gaussian NB de hardware en log-space.
    pub fn classify_with_hw(
        &self,
        foreground_app: Option<&str>,
        all_proc_names: &[&str],
        hour_model: &[HourProfile; 24],
        app_stats: &HashMap<String, AppStats>,
        hour_of_day: u8,
        hw: Option<&HwFeatures>,
    ) -> WorkloadClassification {
        // Score accumulator: WorkloadType → f32
        let mut scores: HashMap<WorkloadType, f32> = HashMap::new();
        let mut sources: Vec<ClassifierSource> = Vec::new();
        let sigs = workload_signatures();

        // 1. Foreground app — weight 2.0 (0.8 for launcher workloads like CommandLine).
        //
        // A terminal emulator is a launcher: it tells us the user has a shell open,
        // not what they're doing inside it.  CommandLine as a foreground signal has
        // low information content compared to e.g. "zoom.us" (definitely a video call).
        // We halve the foreground weight for CommandLine so background process signals
        // can override it — no hardcoded app names needed, uses existing WorkloadType.
        let mut fg_is_launcher = false;
        if let Some(fg) = foreground_app {
            for (wl, patterns) in &sigs {
                if patterns.iter().any(|p| fg.contains(p)) {
                    let fg_weight = if *wl == WorkloadType::CommandLine {
                        fg_is_launcher = true;
                        0.8 // launcher: low-info foreground, process mix can override
                    } else {
                        2.0
                    };
                    *scores.entry(*wl).or_insert(0.0) += fg_weight;
                    sources.push(ClassifierSource::ForegroundApp);
                    break;
                }
            }
            // LLM-learned boost for foreground
            for pw in &self.learned_weights {
                if fg.contains(pw.pattern.as_str()) {
                    *scores.entry(pw.workload).or_insert(0.0) += pw.weight;
                    sources.push(ClassifierSource::LlmLearned);
                }
            }
        }

        // 2. Hour-of-day prior — weight 0.30
        let hour_profile = &hour_model[hour_of_day as usize];
        let hour_total: f32 = hour_profile.values().sum::<f32>().max(1.0);
        for (wl, count) in hour_profile {
            *scores.entry(*wl).or_insert(0.0) += (*count / hour_total) * 0.30;
        }
        if !hour_profile.is_empty() {
            sources.push(ClassifierSource::HourPrior);
        }

        // 3. App recency — variable weight from app_stats
        if let Some(fg) = foreground_app {
            if let Some(stats) = app_stats.get(fg) {
                let recency_weight = match stats.secs_since_last_use {
                    0..=300 => 0.8,
                    301..=3600 => 0.5,
                    _ => 0.1,
                };
                if let Some(wl) = stats.dominant_workload {
                    *scores.entry(wl).or_insert(0.0) += recency_weight;
                    sources.push(ClassifierSource::AppRecency(fg.to_string()));
                }
            }
        }

        // 4. Process mix — 0.04 per match, capped at 50
        // Process mix weight: higher when foreground is a launcher (CommandLine).
        // A terminal running cargo/rustc has process signals as the PRIMARY evidence;
        // standard 0.04 is calibrated for apps where foreground already has 2.0.
        let proc_mix_weight = if fg_is_launcher { 0.30 } else { 0.04 };

        let mut mix_count = 0u32;
        for proc in all_proc_names.iter().take(50) {
            let proc_name = *proc;
            for (wl, patterns) in &sigs {
                if patterns.iter().any(|p| proc_name.contains(*p)) {
                    *scores.entry(*wl).or_insert(0.0) += proc_mix_weight;
                    mix_count += 1;
                    break;
                }
            }
            // LLM boost for background processes
            for pw in &self.learned_weights {
                if proc_name.contains(pw.pattern.as_str()) {
                    *scores.entry(pw.workload).or_insert(0.0) += pw.weight * 0.3;
                }
            }
        }
        if mix_count > 0 {
            sources.push(ClassifierSource::ProcessMix(mix_count));
        }

        // Fusión con Gaussian NB de hardware (si hay features disponibles).
        // Suma el log-posterior normalizado como boost adicional al score de texto.
        if let Some(hw_features) = hw {
            let (hw_wl, hw_prob) = self.hw_bayes.classify(hw_features);
            // Peso del hardware: 0.5 si es muy seguro (>0.8), 0.2 si es débil (<0.5)
            let hw_weight = if hw_prob > 0.80 {
                0.5f32
            } else if hw_prob > 0.60 {
                0.35
            } else if hw_prob > 0.40 {
                0.20
            } else {
                0.10
            };
            *scores.entry(hw_wl).or_insert(0.0) += hw_weight * hw_prob as f32;
        }

        // Winner + AIS-style multi-dimensional confidence.
        //
        // Old formula: best_score / total — structurally weak because hour-prior
        // spreads evidence across all workloads, diluting the winner even when
        // strong signals (foreground app, recency) agree.
        //
        // New formula: weighted composite of 3 orthogonal dimensions, each with
        // a sigmoid transfer function (same approach as intelligence_score.rs).
        //
        //   D1 (0.40) — Margin: how far ahead is the winner vs runner-up.
        //               sigmoid: 1 / (1 + exp(-6 * (margin - 0.3)))
        //   D2 (0.35) — Evidence depth: how many distinct source types contributed.
        //               sigmoid: 1 / (1 + exp(-3 * (n_sources - 1.5)))
        //   D3 (0.25) — Signal strength: absolute winner score (saturates at ~4).
        //               sigmoid: 1 / (1 + exp(-2 * (score - 2.0)))

        let mut sorted_scores: Vec<f32> = scores.values().copied().collect();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let best_score = sorted_scores.first().copied().unwrap_or(0.0);
        let runner_up = sorted_scores.get(1).copied().unwrap_or(0.0);
        let best_wl = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(wl, _)| *wl)
            .unwrap_or(WorkloadType::General);

        // D1: Margin — separation between winner and runner-up.
        // Only meaningful when there is actual competition (runner_up > 0).
        // A single weak signal with no alternatives is NOT high-margin evidence.
        let margin = if best_score > 0.0 && runner_up > 0.0 {
            (best_score - runner_up) / best_score
        } else {
            0.0
        };
        let d1 = 1.0 / (1.0 + (-6.0_f32 * (margin - 0.3)).exp());

        // D2: Evidence depth — count of distinct source types.
        let n_source_types = {
            let mut seen = [false; 6]; // ForegroundApp, HourPrior, AppRecency, ProcessMix, LlmLearned, MemoryLayout
            for s in &sources {
                match s {
                    ClassifierSource::ForegroundApp => seen[0] = true,
                    ClassifierSource::HourPrior => seen[1] = true,
                    ClassifierSource::AppRecency(_) => seen[2] = true,
                    ClassifierSource::ProcessMix(_) => seen[3] = true,
                    ClassifierSource::LlmLearned => seen[4] = true,
                    ClassifierSource::MemoryLayout(_) => seen[5] = true,
                }
            }
            seen.iter().filter(|&&b| b).count() as f32
        };
        let d2 = 1.0 / (1.0 + (-3.0_f32 * (n_source_types - 1.5)).exp());

        // D3: Signal strength — absolute winner score.
        let d3 = 1.0 / (1.0 + (-2.0_f32 * (best_score - 2.0)).exp());

        let confidence = (0.40 * d1 + 0.35 * d2 + 0.25 * d3).clamp(0.0, 1.0);

        // Emit Idle when no foreground app and evidence is very weak.
        let final_workload = if foreground_app.is_none() && confidence < 0.25 {
            WorkloadType::Idle
        } else {
            best_wl
        };

        WorkloadClassification {
            workload: final_workload,
            confidence,
            sources,
        }
    }

    /// Aprendizaje online: si la clasificación de texto tiene confianza alta
    /// Y hay features de hardware, entrena el Gaussian NB con esta observación.
    ///
    /// Llamar después de classify_with_hw cuando confidence > 0.70.
    pub fn maybe_observe(&mut self, hw: &HwFeatures, workload: WorkloadType, text_confidence: f32) {
        if text_confidence >= 0.70 {
            self.hw_bayes.observe(hw, workload);
        }
    }
}

impl Default for WorkloadClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ── Phase 3: Feature-based Workload Mode Classifier ─────────────────────────
//
// Nearest-centroid classification (Cover & Hart, 1967).
// Uses a 5-dimensional feature vector and 4 pre-defined centroids tuned
// for Apple Silicon laptops with 8 GB RAM.
//
// Orthogonal to the text-based WorkloadClassifier above — this produces
// a coarser WorkloadMode used for overflow_guard threshold tuning.

/// Coarse workload mode for threshold tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkloadMode {
    Build,
    LlmInference,
    Browsing,
    Idle,
}

impl WorkloadMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::LlmInference => "llm-inference",
            Self::Browsing => "browsing",
            Self::Idle => "idle",
        }
    }

    /// Threshold bonus for overflow_guard. Negative = more conservative.
    pub fn threshold_bonus(&self) -> f64 {
        match self {
            Self::Build => -0.08,
            Self::LlmInference => -0.04,
            Self::Browsing => 0.0,
            Self::Idle => 0.03,
        }
    }
}

impl Default for WorkloadMode {
    fn default() -> Self {
        Self::Idle
    }
}

/// Feature vector for nearest-centroid classification.
pub struct WorkloadFeatures {
    /// Mean cpu_wall_ratio across all processes (0.0–1.0).
    pub avg_cpu_wall_ratio: f64,
    /// Count of active build tool processes (rustc, cargo, clang, etc.).
    pub build_tool_count: f64,
    /// 1.0 if GPU power > 2W, else 0.0.
    pub gpu_active: f64,
    /// Total RSS of all processes in GB.
    pub total_rss_gb: f64,
    /// Count of behavior-interactive PIDs (Phase 2).
    pub interactive_count: f64,
}

/// Pre-defined centroids tuned for M1 8GB MacBook Air.
/// Dimensions: [avg_cpu_wall_ratio, build_tool_count, gpu_active, total_rss_gb, interactive_count]
const CENTROIDS: [(WorkloadMode, [f64; 5]); 4] = [
    (WorkloadMode::Build, [0.75, 3.0, 0.0, 5.0, 2.0]),
    (WorkloadMode::LlmInference, [0.60, 0.0, 1.0, 4.0, 1.0]),
    (WorkloadMode::Browsing, [0.03, 0.0, 0.3, 3.0, 5.0]),
    (WorkloadMode::Idle, [0.01, 0.0, 0.0, 1.0, 0.0]),
];

/// Normalization ranges per feature dimension (prevents scale bias).
const FEATURE_RANGES: [f64; 5] = [1.0, 4.0, 1.0, 6.0, 6.0];

/// Classify the current system state into one of 4 workload modes.
/// Returns (mode, confidence) where confidence ∈ [0, 1].
pub fn classify_workload_mode(features: &WorkloadFeatures) -> (WorkloadMode, f64) {
    let fv = [
        features.avg_cpu_wall_ratio,
        features.build_tool_count,
        features.gpu_active,
        features.total_rss_gb,
        features.interactive_count,
    ];

    let mut distances: Vec<(WorkloadMode, f64)> = CENTROIDS
        .iter()
        .map(|(mode, centroid)| {
            let d: f64 = fv
                .iter()
                .zip(centroid.iter())
                .zip(FEATURE_RANGES.iter())
                .map(|((v, c), r)| ((v - c) / r).powi(2))
                .sum::<f64>()
                .sqrt();
            (*mode, d)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Hard constraint: Build mode requires at least one build tool running.
    // Without this, high RSS alone (≈5 GB) matches the Build centroid and
    // incorrectly applies -8pp threshold bonus on a browsing/idle system.
    let best = if distances[0].0 == WorkloadMode::Build && features.build_tool_count < 1.0 {
        // Fall back to next-closest mode that isn't Build.
        distances
            .iter()
            .find(|(m, _)| *m != WorkloadMode::Build)
            .copied()
            .unwrap_or(distances[0])
    } else {
        distances[0]
    };

    let d_min = best.1;
    let d_second = distances
        .iter()
        .find(|(m, _)| *m != best.0)
        .map(|(_, d)| *d)
        .unwrap_or(d_min);
    let confidence = if d_min + d_second > 0.0 {
        1.0 - d_min / (d_min + d_second)
    } else {
        0.5
    };

    (best.0, confidence.clamp(0.0, 1.0))
}

#[cfg(test)]
mod memory_layout_tests {
    use super::*;

    fn make_summary(n: u32, private: u64, shared: u64, largest: u64, total: u64) -> RegionSummary {
        RegionSummary {
            n_regions: n,
            private_bytes: private,
            shared_bytes: shared,
            largest_anon_bytes: largest,
            mean_region_size: if n > 0 { total / n as u64 } else { 0 },
            total_virtual: total,
        }
    }

    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * MB;

    #[test]
    fn classify_llm_layout() {
        // Few regions, one huge anonymous (model weights).
        let s = make_summary(40, 4 * GB, 100 * MB, 3 * GB, 5 * GB);
        let result = classify_by_memory(&s);
        assert!(result.is_some());
        let (hint, conf) = result.unwrap();
        assert_eq!(hint, MemoryLayoutHint::LlmInference);
        assert!(conf > 0.7);
    }

    #[test]
    fn classify_browser_layout() {
        // Many regions + large shared.
        let s = make_summary(500, 200 * MB, 300 * MB, 100 * MB, 2 * GB);
        let result = classify_by_memory(&s);
        assert!(result.is_some());
        let (hint, _) = result.unwrap();
        assert_eq!(hint, MemoryLayoutHint::BrowserCache);
    }

    #[test]
    fn classify_build_layout() {
        // Many small regions (object files).
        let s = make_summary(400, 300 * MB, 50 * MB, 50 * MB, 350 * MB);
        let result = classify_by_memory(&s);
        assert!(result.is_some());
        let (hint, _) = result.unwrap();
        assert_eq!(hint, MemoryLayoutHint::BuildSystem);
    }

    #[test]
    fn classify_general_layout() {
        // Average process — doesn't match any pattern.
        let s = make_summary(50, 100 * MB, 20 * MB, 30 * MB, 2 * GB);
        let result = classify_by_memory(&s);
        assert!(result.is_none(), "should be ambiguous");
    }
}

#[cfg(test)]
mod feature_tests {
    use super::*;

    #[test]
    fn build_centroid_match() {
        let (mode, conf) = classify_workload_mode(&WorkloadFeatures {
            avg_cpu_wall_ratio: 0.80,
            build_tool_count: 4.0,
            gpu_active: 0.0,
            total_rss_gb: 5.5,
            interactive_count: 1.0,
        });
        assert_eq!(mode, WorkloadMode::Build);
        assert!(conf > 0.5, "confidence={}", conf);
    }

    #[test]
    fn idle_centroid_match() {
        let (mode, _) = classify_workload_mode(&WorkloadFeatures {
            avg_cpu_wall_ratio: 0.01,
            build_tool_count: 0.0,
            gpu_active: 0.0,
            total_rss_gb: 0.8,
            interactive_count: 0.0,
        });
        assert_eq!(mode, WorkloadMode::Idle);
    }

    #[test]
    fn browsing_centroid_match() {
        let (mode, _) = classify_workload_mode(&WorkloadFeatures {
            avg_cpu_wall_ratio: 0.02,
            build_tool_count: 0.0,
            gpu_active: 0.2,
            total_rss_gb: 3.5,
            interactive_count: 6.0,
        });
        assert_eq!(mode, WorkloadMode::Browsing);
    }

    #[test]
    fn llm_centroid_match() {
        let (mode, _) = classify_workload_mode(&WorkloadFeatures {
            avg_cpu_wall_ratio: 0.55,
            build_tool_count: 0.0,
            gpu_active: 1.0,
            total_rss_gb: 4.5,
            interactive_count: 1.0,
        });
        assert_eq!(mode, WorkloadMode::LlmInference);
    }

    #[test]
    fn threshold_bonus_values() {
        assert!((WorkloadMode::Build.threshold_bonus() - (-0.08)).abs() < 1e-6);
        assert!((WorkloadMode::LlmInference.threshold_bonus() - (-0.04)).abs() < 1e-6);
        assert!((WorkloadMode::Browsing.threshold_bonus() - 0.0).abs() < 1e-6);
        assert!((WorkloadMode::Idle.threshold_bonus() - 0.03).abs() < 1e-6);
    }
}
