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
    llm::LearnedPolicy,
    user_profile::{AppStats, HourProfile, WorkloadType, workload_signatures},
};

/// How a particular piece of evidence contributed to the classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClassifierSource {
    ForegroundApp,
    HourPrior,
    AppRecency(String),
    ProcessMix(u32), // # of matching process names
    LlmLearned,
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
}

impl WorkloadClassifier {
    pub fn new() -> Self {
        Self {
            learned_weights: Vec::new(),
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
    pub fn classify(
        &self,
        foreground_app: Option<&str>,
        all_proc_names: &[&str],
        hour_model: &[HourProfile; 24],
        app_stats: &HashMap<String, AppStats>,
        hour_of_day: u8,
    ) -> WorkloadClassification {
        // Score accumulator: WorkloadType → f32
        let mut scores: HashMap<WorkloadType, f32> = HashMap::new();
        let mut sources: Vec<ClassifierSource> = Vec::new();
        let sigs = workload_signatures();

        // 1. Foreground app — weight 2.0
        if let Some(fg) = foreground_app {
            for (wl, patterns) in &sigs {
                if patterns.iter().any(|p| fg.contains(p)) {
                    *scores.entry(*wl).or_insert(0.0) += 2.0;
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
        let mut mix_count = 0u32;
        for proc in all_proc_names.iter().take(50) {
            let proc_name = *proc;
            for (wl, patterns) in &sigs {
                if patterns.iter().any(|p| proc_name.contains(*p)) {
                    *scores.entry(*wl).or_insert(0.0) += 0.04;
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

        // Winner + confidence
        let total: f32 = scores.values().map(|v| v.max(0.0)).sum::<f32>().max(1.0);
        let (best_wl, best_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(wl, s)| (*wl, *s))
            .unwrap_or((WorkloadType::General, 0.0));

        let confidence = (best_score.max(0.0) / total).clamp(0.0, 1.0);

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
}

impl Default for WorkloadClassifier {
    fn default() -> Self {
        Self::new()
    }
}
