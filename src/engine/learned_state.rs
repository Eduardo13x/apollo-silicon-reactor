//! Unified persistence layer for all learned state.
//!
//! Single file, single struct. To add a new persisted component:
//! 1. Add a field to [`LearnedState`] with `#[serde(default)]`
//! 2. Populate it in [`LearnedState::collect`]
//! 3. Restore it in [`LearnedState::apply`]
//! That's it. No daemon wiring changes needed.
//!
//! ## Self-improvement
//! Before each persist, `self_improve()` prunes stale data and decays old signals.
//! After each restore, `validate()` detects corrupt/out-of-range state and resets it.
//! The `RestoreQualityMonitor` tracks whether restored state helps or hurts,
//! and can trigger partial resets if the restored state is stale.

use std::path::Path;

use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use crate::engine::causal_graph::{CausalEdge, CausalGraph};
use crate::engine::process_baseline::ProcessBaselineMap;
use crate::engine::effectiveness_tracker::{EffectivenessTracker, ProcessEffectiveness};
use crate::engine::nars_belief::ArousalState;
use crate::engine::optimization_skills::{OptimizationSkill, SkillRegistry};
use crate::engine::outcome_tracker::{OutcomeTracker, OutcomeTrackerPersisted};
use crate::engine::overflow_guard::OverflowHistory;
use crate::engine::predictive_agent::SpecialistAccuracyTracker;
use crate::engine::signal_intelligence::{SignalIntelligence, SignalIntelligencePersisted};
use crate::engine::types::FrozenStatePersisted;

/// Everything Apollo learns at runtime, in one serializable struct.
///
/// All fields use `#[serde(default)]` so old files missing new fields
/// deserialize cleanly — components fall back to cold-start defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedState {
    /// Schema version for forward compatibility.
    #[serde(default = "default_version")]
    pub version: u32,

    /// Signal intelligence: hazard model, MPC, Kalman, zones, utility EMAs.
    #[serde(default)]
    pub signal_intelligence: Option<SignalIntelligencePersisted>,

    /// Outcome tracker: Bayesian weights, experience memory, causal graph, HRPO.
    #[serde(default)]
    pub outcome_tracker: Option<OutcomeTrackerPersisted>,

    /// Specialist voting accuracy weights (4 floats).
    #[serde(default)]
    pub specialist_accuracy: Option<SpecialistAccuracyTracker>,

    /// Metadata: how many persist cycles this state has survived.
    #[serde(default)]
    pub persist_generations: u32,

    /// Restore quality: effectiveness in the first 50 cycles after last restore.
    /// Persisted so we can compare across restarts.
    #[serde(default)]
    pub last_restore_quality: Option<f64>,

    /// Pending trial skill: (skill_name, pressure_before_trial).
    /// If the daemon restarts mid-trial, this lets the next cycle record the
    /// trial result instead of silently dropping it.
    #[serde(default)]
    pub pending_trial_skill: Option<(String, f64)>,

    /// Optimization skills — persisted here so a single file captures all learned state.
    ///
    /// When present, this field takes precedence over `optimization_skills.json`.
    /// During the transition period, both files are written (dual-write).
    /// `None` means no skills were persisted — cold start or old file format.
    #[serde(default)]
    pub skill_registry: Option<HashMap<String, OptimizationSkill>>,

    /// Overflow guard history — persisted here so overflow events and adaptive
    /// thresholds survive crashes and reboots without depending solely on
    /// `overflow_history.json`.  Dual-write: the guard still writes its own
    /// file as a fallback.  `None` on first run or old file format.
    #[serde(default)]
    pub overflow_guard_history: Option<OverflowHistory>,

    /// Frozen process state — persisted here so a daemon crash leaves the
    /// system consistent: on restart, Apollo knows which PIDs were frozen and
    /// can SIGCONT them before any new freeze decisions are made.  Stored as
    /// the same `FrozenStatePersisted` format used by `frozen_state.json`
    /// (dual-write preserved).  `None` on first run or old file format.
    #[serde(default)]
    pub frozen_pids: Option<FrozenStatePersisted>,

    /// Unified effectiveness scores per process.
    #[serde(default)]
    pub effectiveness_tracker: Option<HashMap<String, ProcessEffectiveness>>,

    /// Global arousal EMA state — persisted so crisis context survives restarts.
    /// Without this, the daemon starts cold (arousal=0.0) after a restart even
    /// if the system was under heavy load. [Yerkes & Dodson 1908]
    #[serde(default)]
    pub arousal_state: Option<ArousalState>,

    /// Causal graph edges — persisted so learned causal relationships (confidence,
    /// mechanism attribution, slow-horizon data) survive daemon restarts.
    /// Without this, Apollo restarts with no causal knowledge and wastes cycles
    /// re-learning which throttles are effective. [Pearl 2009]
    #[serde(default)]
    pub causal_graph_edges: Option<Vec<((String, String), CausalEdge)>>,

    /// Per-process hardware counter baselines for behavioral anomaly detection.
    /// EMA + EMA-MAD per {ipc, wakeup_rate, disk_mbps} per process name.
    /// Persisted so warm baselines (≥ 5 obs) survive daemon restarts — without this,
    /// every restart discards learned behavioral norms and cold-starts anomaly detection.
    /// [Holt 1957] exponential smoothing; [Chandola 2009] EMA-MAD anomaly detection.
    #[serde(default)]
    pub process_baselines: Option<ProcessBaselineMap>,
}

fn default_version() -> u32 {
    1
}

// ── Self-improvement constants ──────────────────────────────────────────────

/// Co-occurrence decay factor per persist (×0.90 = 10% decay per save).
const CO_OCC_DECAY: f64 = 0.90;
/// Co-occurrence entries below this count after decay are pruned.
const CO_OCC_PRUNE_THRESHOLD: u32 = 2;
/// Bayesian weights with fewer than this many throttles AND effectiveness
/// indistinguishable from prior (0.5) are pruned on persist.
const WEIGHT_MIN_THROTTLES: u32 = 3;
/// Experience records cap after compression.
const EXPERIENCE_CAP: usize = 300;

impl LearnedState {
    /// Collect snapshots from all live components into a single struct.
    pub fn collect(
        signal_intel: &SignalIntelligence,
        outcome_tracker: &OutcomeTracker,
        specialist_accuracy: &SpecialistAccuracyTracker,
        skill_registry: &SkillRegistry,
        effectiveness_tracker: &EffectivenessTracker,
        overflow_history: Option<OverflowHistory>,
        frozen_state: Option<FrozenStatePersisted>,
        arousal_state: Option<ArousalState>,
        causal_graph: Option<&CausalGraph>,
        process_baselines: Option<ProcessBaselineMap>,
    ) -> Self {
        Self {
            version: 1,
            signal_intelligence: Some(signal_intel.to_persisted()),
            outcome_tracker: Some(outcome_tracker.to_persisted()),
            specialist_accuracy: Some(specialist_accuracy.clone()),
            persist_generations: 0,
            last_restore_quality: None,
            pending_trial_skill: None,
            skill_registry: Some(skill_registry.snapshot()),
            effectiveness_tracker: Some(effectiveness_tracker.snapshot()),
            overflow_guard_history: overflow_history,
            frozen_pids: frozen_state,
            arousal_state,
            causal_graph_edges: causal_graph.map(|cg| cg.to_persisted()),
            process_baselines,
        }
    }

    /// Apply persisted state back to live components.
    /// Runs `validate()` first to sanitize corrupt or out-of-range data.
    /// Each component handles missing data gracefully (keeps defaults).
    ///
    /// Returns `(overflow_history, frozen_pids)` — the caller is responsible
    /// for wiring these into `OverflowGuard::import_history()` and the frozen
    /// state map respectively.  Returning `None` in either slot means the
    /// field was absent in the file (old format or cold start); the caller
    /// should fall back to the legacy single-purpose file.
    pub fn apply(
        mut self,
        signal_intel: &mut SignalIntelligence,
        outcome_tracker: &mut OutcomeTracker,
        specialist_accuracy: &mut SpecialistAccuracyTracker,
        skill_registry: &mut SkillRegistry,
        effectiveness_tracker: &mut EffectivenessTracker,
        causal_graph: Option<&mut CausalGraph>,
    ) -> (Option<OverflowHistory>, Option<FrozenStatePersisted>, Option<ArousalState>, Option<ProcessBaselineMap>) {
        self.validate();
        if let Some(si) = self.signal_intelligence {
            signal_intel.restore(si);
        }
        if let Some(ot) = self.outcome_tracker {
            outcome_tracker.restore(ot);
        }
        if let Some(sa) = self.specialist_accuracy {
            *specialist_accuracy = sa;
        }
        // Restore skills only if the field is present — backwards compat:
        // old learned_state.json files (field absent) fall through to the
        // legacy optimization_skills.json load that the caller performs after.
        if let Some(skills) = self.skill_registry {
            skill_registry.restore_from_map(skills);
        }
        if let Some(eff) = self.effectiveness_tracker {
            effectiveness_tracker.restore_from_map(eff);
        }
        if let Some(edges) = self.causal_graph_edges {
            if let Some(cg) = causal_graph {
                cg.restore(edges);
            }
        }
        (self.overflow_guard_history, self.frozen_pids, self.arousal_state, self.process_baselines)
    }

    // ── Self-improvement: called before persist ─────────────────────────

    /// Prune stale data, decay old signals, compress bloated sections.
    /// Called automatically by `persist_improved()`.
    pub fn self_improve(&mut self) {
        // NOTE: persist_generations is incremented by persist_improved() before
        // calling self_improve(), so we must NOT increment here too.
        // Double-incrementing causes all half-life / decay calculations to run
        // at 2× the intended speed (beliefs forgotten prematurely).
        // [Hamilton 2007] — version counters must increment exactly once per operation.

        if let Some(ot) = &mut self.outcome_tracker {
            // 1. Decay co-occurrence counts — old pairs fade out.
            for entry in &mut ot.co_occurrence {
                entry.2 = ((entry.2 as f64) * CO_OCC_DECAY).round() as u32;
            }
            ot.co_occurrence.retain(|e| e.2 >= CO_OCC_PRUNE_THRESHOLD);

            // 2. Prune Bayesian weights that carry no signal.
            //    Processes with <3 throttles and effectiveness ~0.5 (prior)
            //    are noise — discard them to keep the file lean.
            ot.weights
                .retain(|_, w| w.throttle_count >= WEIGHT_MIN_THROTTLES || w.effective_count > 0);

            // 3. Compress experience memory: keep last EXPERIENCE_CAP records.
            //    Older records are less relevant as workload patterns shift.
            if ot.experience_records.len() > EXPERIENCE_CAP {
                let drain = ot.experience_records.len() - EXPERIENCE_CAP;
                ot.experience_records.drain(..drain);
            }

            // 4. Prune HRPO groups with <2 throttles — not enough signal.
            ot.hop_groups.retain(|_, g| g.throttle_count >= 2);

            // 5. NARS belief confidence decay: old evidence becomes less certain.
            //    Processes not observed recently lose confidence → new observations
            //    have more influence, preventing stale beliefs from dominating.
            //    [Bayesian forgetting] factor=0.95 → half-life ≈ 14 persist cycles.
            if let Some(dd) = &mut ot.drift_detector {
                dd.decay_confidence(0.95);
            }
        }

        // 6. Process baseline prune: remove entries with 0 observations (defensive).
        if let Some(pb) = &mut self.process_baselines {
            pb.prune_stale();
        }

        // 7. Causal graph decay: stale edges lose confidence over time.
        //    [Bayesian forgetting] factor=0.97 → half-life ≈ 23 persist cycles.
        //    Prune edges with near-zero impact AND low evidence.
        if let Some(edges) = &mut self.causal_graph_edges {
            for (_, edge) in edges.iter_mut() {
                // Decay both fast and slow confidence toward uninformed prior (0.5).
                edge.confidence = 0.5 + (edge.confidence - 0.5) * 0.97;
                edge.slow_confidence = 0.5 + (edge.slow_confidence - 0.5) * 0.97;
                // Decay mechanism attribution EMAs.
                edge.mechanism.rss_delta_mb *= 0.95;
                edge.mechanism.cpu_delta_pct *= 0.95;
                edge.mechanism.swap_delta_mb *= 0.95;
            }
            // Prune edges near uninformed prior with low evidence — no signal.
            edges.retain(|(_, e)| {
                let near_prior = (e.confidence - 0.5).abs() < 0.05
                    && (e.slow_confidence - 0.5).abs() < 0.05;
                !(near_prior && e.evidence_count < 10)
            });
        }
    }

    // ── Validation: called before apply ─────────────────────────────────

    /// Sanitize restored state: detect out-of-range values and reset them
    /// to cold-start defaults rather than letting corrupt data propagate.
    pub fn validate(&mut self) {
        if let Some(si) = &mut self.signal_intelligence {
            // Zone entries must be in their clamp ranges.
            si.learned_mid_entry = si.learned_mid_entry.clamp(0.20, 0.40);
            si.learned_high_entry = si.learned_high_entry.clamp(0.35, 0.60);
            // Mid must be < high.
            if si.learned_mid_entry >= si.learned_high_entry {
                si.learned_mid_entry = 0.30;
                si.learned_high_entry = 0.50;
            }
            // Utility EMAs must be in [0, 1].
            si.utility_entropy = si.utility_entropy.clamp(0.0, 1.0);
            si.utility_hazard = si.utility_hazard.clamp(0.0, 1.0);
            si.utility_lotka = si.utility_lotka.clamp(0.0, 1.0);
            si.utility_mpc = si.utility_mpc.clamp(0.0, 1.0);
            // Kalman pressure position should be in [0, 1].
            if let Some(kf) = &si.kf_pressure {
                if kf.position() < -0.1 || kf.position() > 1.5 {
                    si.kf_pressure = None; // let it re-initialize from live data
                }
            }
        }

        if let Some(ot) = &mut self.outcome_tracker {
            // natural_drift_ema should be small (typical: -0.05 to +0.05).
            ot.natural_drift_ema = ot.natural_drift_ema.clamp(-0.2, 0.2);
            // baseline_drop_ema is a probability-like value in [0, 1].
            ot.baseline_drop_ema = ot.baseline_drop_ema.clamp(0.0, 1.0);
        }

        if let Some(sa) = &mut self.specialist_accuracy {
            // All accuracy weights must be in [0, 1].
            for w in sa.weights_mut() {
                *w = w.clamp(0.0, 1.0);
            }
        }

        // Causal graph: clamp confidence values to [0, 1].
        if let Some(edges) = &mut self.causal_graph_edges {
            for (_, edge) in edges.iter_mut() {
                edge.confidence = edge.confidence.clamp(0.0, 1.0);
                edge.slow_confidence = edge.slow_confidence.clamp(0.0, 1.0);
                edge.avg_delta = edge.avg_delta.clamp(0.0, 1.0);
                edge.slow_avg_delta = edge.slow_avg_delta.clamp(0.0, 1.0);
            }
        }
    }

    // ── Persist with self-improvement ───────────────────────────────────

    /// Collect + self-improve + persist in one call.
    /// This is the recommended way to persist — replaces raw `collect().persist()`.
    #[allow(clippy::too_many_arguments)]
    pub fn persist_improved(
        signal_intel: &SignalIntelligence,
        outcome_tracker: &OutcomeTracker,
        specialist_accuracy: &SpecialistAccuracyTracker,
        skill_registry: &SkillRegistry,
        effectiveness_tracker: &EffectivenessTracker,
        overflow_history: Option<OverflowHistory>,
        frozen_state: Option<FrozenStatePersisted>,
        path: &Path,
        prev_generations: u32,
        last_quality: Option<f64>,
        pending_trial_skill: Option<(String, f64)>,
        arousal_state: Option<ArousalState>,
        causal_graph: Option<&CausalGraph>,
        process_baselines: Option<ProcessBaselineMap>,
    ) {
        let mut state = Self::collect(
            signal_intel,
            outcome_tracker,
            specialist_accuracy,
            skill_registry,
            effectiveness_tracker,
            overflow_history,
            frozen_state,
            arousal_state,
            causal_graph,
            process_baselines,
        );
        state.persist_generations = prev_generations.saturating_add(1);
        state.last_restore_quality = last_quality;
        state.pending_trial_skill = pending_trial_skill;
        // If no baselines were passed (periodic persist), preserve the previously
        // persisted baselines so we don't erase them on every cycle persist.
        if state.process_baselines.is_none() {
            state.process_baselines = Self::load(path).and_then(|old| old.process_baselines);
        }
        state.self_improve();
        state.persist(path);
    }

    /// Persist to disk (best-effort, never panics).
    pub fn persist(&self, path: &Path) {
        if let Ok(json) = serde_json::to_string(self) {
            let _ = std::fs::write(path, json);
        }
    }

    /// Patch only the `process_baselines` field of an existing persisted file.
    /// Reads the file, updates the field, writes back. No-op if file is missing.
    /// Used by periodic persist which doesn't have access to the baseline map.
    pub fn patch_process_baselines(path: &Path, baselines: ProcessBaselineMap) {
        let Some(mut state) = Self::load(path) else { return };
        state.process_baselines = Some(baselines);
        state.persist(path);
    }

    /// Load from disk. Returns None on any error (cold start is safe).
    pub fn load(path: &Path) -> Option<Self> {
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }
}

// ── Restore Quality Monitor ─────────────────────────────────────────────────

/// Tracks whether restored state is helping or hurting.
///
/// Usage: create after restore, call `observe()` each cycle for the first 50 cycles.
/// After 50 cycles, `verdict()` returns the quality score.
/// If quality < cold-start baseline (0.5), the restored state was stale — the daemon
/// should partially reset (e.g., clear experience memory, reset zones to defaults).
pub struct RestoreQualityMonitor {
    /// Number of cycles observed since restore.
    cycles: u32,
    /// Number of effective throttles in the observation window.
    effective: u32,
    /// Total throttles resolved in the observation window.
    resolved: u32,
    /// Whether this monitor has already fired its verdict.
    fired: bool,
}

/// Observation window: 50 cycles (~100s at 2s/cycle).
const QUALITY_WINDOW: u32 = 50;
/// If post-restore effectiveness is below this, restored state is hurting.
const QUALITY_THRESHOLD: f64 = 0.35;

impl RestoreQualityMonitor {
    /// Create a new monitor. Call right after `LearnedState::apply()`.
    pub fn new() -> Self {
        Self {
            cycles: 0,
            effective: 0,
            resolved: 0,
            fired: false,
        }
    }

    /// Feed an outcome observation. Call each cycle with the batch results.
    pub fn observe(&mut self, batch_effective: u32, batch_resolved: u32) {
        if self.fired {
            return;
        }
        self.cycles += 1;
        self.effective += batch_effective;
        self.resolved += batch_resolved;
    }

    /// Check if the observation window is complete and return a verdict.
    /// Returns `Some(quality)` once, where quality is the effectiveness ratio.
    /// Returns `None` if still observing or already fired.
    pub fn verdict(&mut self) -> Option<RestoreVerdict> {
        if self.fired || self.cycles < QUALITY_WINDOW {
            return None;
        }
        self.fired = true;
        let quality = if self.resolved < 5 {
            // Not enough data to judge — assume OK.
            return Some(RestoreVerdict {
                quality: 0.5,
                stale: false,
            });
        } else {
            (self.effective as f64 + 1.0) / (self.resolved as f64 + 2.0)
        };
        Some(RestoreVerdict {
            quality,
            stale: quality < QUALITY_THRESHOLD,
        })
    }

    /// True if the monitor has already produced a verdict.
    pub fn is_done(&self) -> bool {
        self.fired
    }
}

/// Result of the restore quality assessment.
#[derive(Debug, Clone)]
pub struct RestoreVerdict {
    /// Effectiveness ratio [0, 1] in the first 50 cycles post-restore.
    pub quality: f64,
    /// True if restored state appears stale (quality < threshold).
    /// Daemon should partially reset learned state.
    pub stale: bool,
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::outcome_tracker::{
        ExperienceRecord, HopGroupWeight, PatternWeight, WorkloadHop,
    };
    use std::collections::HashMap;

    fn make_ot_persisted() -> OutcomeTrackerPersisted {
        let mut weights = HashMap::new();
        // Process with signal: keep.
        weights.insert(
            "brave".to_string(),
            PatternWeight {
                throttle_count: 10,
                effective_count: 7,
            },
        );
        // Process without signal: prune (1 throttle, 0 effective).
        weights.insert(
            "noise".to_string(),
            PatternWeight {
                throttle_count: 1,
                effective_count: 0,
            },
        );

        let co_occurrence = vec![
            ("a".into(), "b".into(), 20), // will decay to 18, kept
            ("c".into(), "d".into(), 2),  // will decay to ~2, borderline
            ("e".into(), "f".into(), 1),  // will decay to ~1, pruned
        ];

        let mut experience_records = Vec::new();
        for i in 0..400 {
            experience_records.push(ExperienceRecord {
                process_name: format!("proc_{}", i % 10),
                pressure_at_action: 0.6,
                pressure_drop: 0.03,
                effective: i % 3 == 0,
            });
        }

        OutcomeTrackerPersisted {
            weights,
            total_effective: 50,
            total_resolved: 100,
            baseline_drop_ema: 0.25,
            baseline_samples: 100,
            experience_records,
            co_occurrence,
            natural_drift_ema: 0.01,
            hop_groups: HashMap::new(),
            drift_detector: None,
        }
    }

    #[test]
    fn self_improve_decays_co_occurrence() {
        let mut state = LearnedState {
            version: 1,
            signal_intelligence: None,
            outcome_tracker: Some(make_ot_persisted()),
            specialist_accuracy: None,
            persist_generations: 0,
            last_restore_quality: None,
            pending_trial_skill: None,
            skill_registry: None,
            overflow_guard_history: None,
            frozen_pids: None,
            effectiveness_tracker: None,
            arousal_state: None,
            causal_graph_edges: None,
            process_baselines: None,
        };
        state.self_improve();
        let ot = state.outcome_tracker.as_ref().unwrap();
        // (a, b, 20) → 18 after decay, kept.
        assert!(ot.co_occurrence.iter().any(|e| e.0 == "a" && e.2 == 18));
        // (e, f, 1) → ~1 after decay, pruned.
        assert!(!ot.co_occurrence.iter().any(|e| e.0 == "e"));
    }

    #[test]
    fn self_improve_prunes_noisy_weights() {
        let mut state = LearnedState {
            version: 1,
            signal_intelligence: None,
            outcome_tracker: Some(make_ot_persisted()),
            specialist_accuracy: None,
            persist_generations: 0,
            last_restore_quality: None,
            pending_trial_skill: None,
            skill_registry: None,
            overflow_guard_history: None,
            frozen_pids: None,
            effectiveness_tracker: None,
            arousal_state: None,
            causal_graph_edges: None,
            process_baselines: None,
        };
        state.self_improve();
        let ot = state.outcome_tracker.as_ref().unwrap();
        assert!(ot.weights.contains_key("brave"), "high-signal weight kept");
        assert!(!ot.weights.contains_key("noise"), "no-signal weight pruned");
    }

    #[test]
    fn self_improve_caps_experience() {
        let mut state = LearnedState {
            version: 1,
            signal_intelligence: None,
            outcome_tracker: Some(make_ot_persisted()),
            specialist_accuracy: None,
            persist_generations: 0,
            last_restore_quality: None,
            pending_trial_skill: None,
            skill_registry: None,
            overflow_guard_history: None,
            frozen_pids: None,
            effectiveness_tracker: None,
            arousal_state: None,
            causal_graph_edges: None,
            process_baselines: None,
        };
        assert_eq!(
            state
                .outcome_tracker
                .as_ref()
                .unwrap()
                .experience_records
                .len(),
            400
        );
        state.self_improve();
        assert_eq!(
            state
                .outcome_tracker
                .as_ref()
                .unwrap()
                .experience_records
                .len(),
            EXPERIENCE_CAP
        );
    }

    #[test]
    fn self_improve_increments_generations() {
        let mut state = LearnedState {
            version: 1,
            signal_intelligence: None,
            outcome_tracker: None,
            specialist_accuracy: None,
            persist_generations: 5,
            last_restore_quality: None,
            pending_trial_skill: None,
            skill_registry: None,
            overflow_guard_history: None,
            frozen_pids: None,
            effectiveness_tracker: None,
            arousal_state: None,
            causal_graph_edges: None,
            process_baselines: None,
        };
        state.self_improve();
        assert_eq!(state.persist_generations, 6);
    }

    #[test]
    fn validate_clamps_zones() {
        let si = SignalIntelligencePersisted {
            hazard: crate::engine::hazard_model::HazardModel::new(),
            mpc: crate::engine::mpc_horizon::MpcController::new(3, 0.5).to_persisted(),
            learned_mid_entry: 0.99,  // way out of range
            learned_high_entry: 0.10, // below mid
            utility_entropy: 5.0,     // out of [0,1]
            utility_hazard: -1.0,
            utility_lotka: 0.7,
            utility_mpc: 0.3,
            kf_pressure: None,
            kf_swap: None,
        };
        let mut state = LearnedState {
            version: 1,
            signal_intelligence: Some(si),
            outcome_tracker: None,
            specialist_accuracy: None,
            persist_generations: 0,
            last_restore_quality: None,
            pending_trial_skill: None,
            skill_registry: None,
            overflow_guard_history: None,
            frozen_pids: None,
            effectiveness_tracker: None,
            arousal_state: None,
            causal_graph_edges: None,
            process_baselines: None,
        };
        state.validate();
        let si = state.signal_intelligence.as_ref().unwrap();
        // Zones reset to defaults because mid >= high after clamping.
        assert_eq!(si.learned_mid_entry, 0.30);
        assert_eq!(si.learned_high_entry, 0.50);
        // Utilities clamped.
        assert_eq!(si.utility_entropy, 1.0);
        assert_eq!(si.utility_hazard, 0.0);
    }

    #[test]
    fn validate_clamps_drift() {
        let ot = OutcomeTrackerPersisted {
            weights: HashMap::new(),
            total_effective: 0,
            total_resolved: 0,
            baseline_drop_ema: 2.0, // out of range
            baseline_samples: 0,
            experience_records: vec![],
            co_occurrence: vec![],
            natural_drift_ema: -0.5, // out of range
            hop_groups: HashMap::new(),
            drift_detector: None,
        };
        let mut state = LearnedState {
            version: 1,
            signal_intelligence: None,
            outcome_tracker: Some(ot),
            specialist_accuracy: None,
            persist_generations: 0,
            last_restore_quality: None,
            pending_trial_skill: None,
            skill_registry: None,
            overflow_guard_history: None,
            frozen_pids: None,
            effectiveness_tracker: None,
            arousal_state: None,
            causal_graph_edges: None,
            process_baselines: None,
        };
        state.validate();
        let ot = state.outcome_tracker.as_ref().unwrap();
        assert_eq!(ot.baseline_drop_ema, 1.0);
        assert_eq!(ot.natural_drift_ema, -0.2);
    }

    #[test]
    fn restore_quality_monitor_detects_stale() {
        let mut monitor = RestoreQualityMonitor::new();
        // Simulate 50 cycles of terrible effectiveness (2/50 effective).
        for i in 0..QUALITY_WINDOW {
            let eff = if i < 2 { 1 } else { 0 };
            monitor.observe(eff, 1);
        }
        let verdict = monitor.verdict().expect("should have verdict");
        assert!(
            verdict.stale,
            "low effectiveness should be flagged as stale"
        );
        assert!(verdict.quality < QUALITY_THRESHOLD);
    }

    #[test]
    fn restore_quality_monitor_approves_good_state() {
        let mut monitor = RestoreQualityMonitor::new();
        // Simulate 50 cycles of good effectiveness (40/50 effective).
        for i in 0..QUALITY_WINDOW {
            let eff = if i < 40 { 1 } else { 0 };
            monitor.observe(eff, 1);
        }
        let verdict = monitor.verdict().expect("should have verdict");
        assert!(!verdict.stale, "good effectiveness should not be stale");
        assert!(verdict.quality > 0.7);
    }

    #[test]
    fn restore_quality_fires_only_once() {
        let mut monitor = RestoreQualityMonitor::new();
        for _ in 0..QUALITY_WINDOW {
            monitor.observe(1, 1);
        }
        assert!(monitor.verdict().is_some());
        assert!(
            monitor.verdict().is_none(),
            "second call should return None"
        );
        assert!(monitor.is_done());
    }
}
