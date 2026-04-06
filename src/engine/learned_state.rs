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
use crate::engine::effectiveness_tracker::{EffectivenessTracker, ProcessEffectiveness};
use crate::engine::nars_belief::ArousalState;
use crate::engine::optimization_skills::{OptimizationSkill, SkillRegistry};
use crate::engine::outcome_tracker::{OutcomeTracker, OutcomeTrackerPersisted};
use crate::engine::overflow_guard::OverflowHistory;
use crate::engine::predictive_agent::SpecialistAccuracyTracker;
use crate::engine::process_baseline::ProcessBaselineMap;
use crate::engine::signal_intelligence::{SignalIntelligence, SignalIntelligencePersisted};
use crate::engine::types::FrozenStatePersisted;

/// Adaptive parameters that replace hardcoded thresholds.
///
/// Every field has a safe default matching the original hardcoded value,
/// a valid range enforced by `validate()`, and a learning pathway that
/// adjusts it from outcome data.  Persisted via `LearnedState` so learned
/// values survive daemon restarts.
///
/// ## Adding a new parameter
/// 1. Add a `#[serde(default = "default_X")]` field here
/// 2. Add a clamp rule in `LearnableParams::validate()`
/// 3. Wire the consumer to read `learnable.X` instead of its hardcoded constant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnableParams {
    // ── Kalman filter tuning ──────────────────────────────────────────
    /// Kalman measurement noise for pressure filter.
    /// Lower = trusts measurements more. Auto-tuned from innovation variance.
    #[serde(default = "lp_kalman_pressure_r")]
    pub kalman_pressure_r: f64,

    /// Kalman process noise for pressure filter.
    #[serde(default = "lp_kalman_pressure_q")]
    pub kalman_pressure_q: f64,

    // ── RL state discretization ───────────────────────────────────────
    /// Pressure band boundaries for Q-table state discretization.
    /// Auto-tuned from pressure histogram quantiles (33rd/66th/90th).
    #[serde(default = "lp_rl_pressure_bands")]
    pub rl_pressure_bands: [f64; 3],

    /// Compressor band boundaries for Q-table state discretization.
    #[serde(default = "lp_rl_compressor_bands")]
    pub rl_compressor_bands: [f64; 2],

    // ── Zone learning ─────────────────────────────────────────────────
    /// Zone feedback learning rate. Auto-tuned: halved on oscillation, doubled on stall.
    #[serde(default = "lp_zone_alpha")]
    pub zone_alpha: f64,

    // ── Outcome tracker ───────────────────────────────────────────────
    /// Seconds to wait before checking outcome (per-process adaptive in Phase 7).
    #[serde(default = "lp_outcome_wait_secs")]
    pub outcome_wait_secs: u64,

    /// Minimum pressure drop to count as effective.
    #[serde(default = "lp_outcome_effective_threshold")]
    pub outcome_effective_threshold: f64,

    /// Pressure similarity band for experience memory queries.
    #[serde(default = "lp_experience_pressure_band")]
    pub experience_pressure_band: f64,

    // ── NARS belief system ────────────────────────────────────────────
    /// Frequency shift threshold for drift detection.
    #[serde(default = "lp_nars_drift_threshold")]
    pub nars_drift_threshold: f64,

    /// Per-persist confidence decay factor (Bayesian forgetting).
    #[serde(default = "lp_nars_decay_factor")]
    pub nars_decay_factor: f32,

    // ── Signal intelligence ───────────────────────────────────────────
    /// CUSUM drift magnitude parameter.
    #[serde(default = "lp_cusum_k")]
    pub cusum_k: f64,

    /// CUSUM threshold parameter.
    #[serde(default = "lp_cusum_h")]
    pub cusum_h: f64,

    /// PID target pressure (below = fine, above = error accumulates).
    #[serde(default = "lp_pid_target")]
    pub pid_target: f64,

    /// PID leaky integrator decay (prevents windup).
    #[serde(default = "lp_pid_decay")]
    pub pid_decay: f64,

    // ── Fluidity ──────────────────────────────────────────────────────
    /// WindowServer CPU spike threshold (%).
    #[serde(default = "lp_ws_spike_threshold")]
    pub ws_spike_threshold: f32,

    /// Fluidity degraded threshold (0–1).
    #[serde(default = "lp_fluidity_degraded_threshold")]
    pub fluidity_degraded_threshold: f32,

    // ── Hazard model ──────────────────────────────────────────────────
    /// Online hazard retrain learning rate.
    #[serde(default = "lp_hazard_lr")]
    pub hazard_lr: f64,

    // ── Memory budget ─────────────────────────────────────────────────
    /// Max fraction of allocatable RAM for foreground processes.
    #[serde(default = "lp_max_foreground_share")]
    pub max_foreground_share: f64,

    /// Max fraction of allocatable RAM for background processes.
    #[serde(default = "lp_max_background_share")]
    pub max_background_share: f64,

    // ── Meta-learning (Phase 6) ───────────────────────────────────────
    /// EMA of global effectiveness (for meta-learning velocity detection).
    #[serde(default)]
    pub meta_effectiveness_ema: f64,

    /// EMA of |param_delta|/cycle (learning velocity).
    #[serde(default)]
    pub meta_learning_velocity: f64,

    // ── Provenance ────────────────────────────────────────────────────
    /// Total tuning cycles that have contributed to these parameters.
    #[serde(default)]
    pub tuning_cycles: u64,
}

// ── LearnableParams defaults (match original hardcoded values) ─────────
fn lp_kalman_pressure_r() -> f64 {
    0.02
}
fn lp_kalman_pressure_q() -> f64 {
    0.005
}
fn lp_rl_pressure_bands() -> [f64; 3] {
    [0.50, 0.80, 0.92]
}
fn lp_rl_compressor_bands() -> [f64; 2] {
    [0.30, 0.60]
}
fn lp_zone_alpha() -> f64 {
    0.005
}
fn lp_outcome_wait_secs() -> u64 {
    30
}
fn lp_outcome_effective_threshold() -> f64 {
    0.01
}
fn lp_experience_pressure_band() -> f64 {
    0.10
}
fn lp_nars_drift_threshold() -> f64 {
    0.20
}
fn lp_nars_decay_factor() -> f32 {
    0.95
}
fn lp_cusum_k() -> f64 {
    0.02
}
fn lp_cusum_h() -> f64 {
    0.12
}
fn lp_pid_target() -> f64 {
    0.65
}
fn lp_pid_decay() -> f64 {
    0.98
}
fn lp_ws_spike_threshold() -> f32 {
    25.0
}
fn lp_fluidity_degraded_threshold() -> f32 {
    0.65
}
fn lp_hazard_lr() -> f64 {
    0.01
}
fn lp_max_foreground_share() -> f64 {
    0.40
}
fn lp_max_background_share() -> f64 {
    0.15
}

impl Default for LearnableParams {
    fn default() -> Self {
        Self {
            kalman_pressure_r: lp_kalman_pressure_r(),
            kalman_pressure_q: lp_kalman_pressure_q(),
            rl_pressure_bands: lp_rl_pressure_bands(),
            rl_compressor_bands: lp_rl_compressor_bands(),
            zone_alpha: lp_zone_alpha(),
            outcome_wait_secs: lp_outcome_wait_secs(),
            outcome_effective_threshold: lp_outcome_effective_threshold(),
            experience_pressure_band: lp_experience_pressure_band(),
            nars_drift_threshold: lp_nars_drift_threshold(),
            nars_decay_factor: lp_nars_decay_factor(),
            cusum_k: lp_cusum_k(),
            cusum_h: lp_cusum_h(),
            pid_target: lp_pid_target(),
            pid_decay: lp_pid_decay(),
            ws_spike_threshold: lp_ws_spike_threshold(),
            fluidity_degraded_threshold: lp_fluidity_degraded_threshold(),
            hazard_lr: lp_hazard_lr(),
            max_foreground_share: lp_max_foreground_share(),
            max_background_share: lp_max_background_share(),
            meta_effectiveness_ema: 0.0,
            meta_learning_velocity: 0.0,
            tuning_cycles: 0,
        }
    }
}

impl LearnableParams {
    /// Clamp all values to their safe ranges.
    pub fn validate(&mut self) {
        self.kalman_pressure_r = self.kalman_pressure_r.clamp(0.001, 0.5);
        self.kalman_pressure_q = self.kalman_pressure_q.clamp(0.001, 0.1);

        // RL pressure bands must be monotonically increasing in safe ranges.
        self.rl_pressure_bands[0] = self.rl_pressure_bands[0].clamp(0.30, 0.60);
        self.rl_pressure_bands[1] = self.rl_pressure_bands[1].clamp(0.55, 0.85);
        self.rl_pressure_bands[2] = self.rl_pressure_bands[2].clamp(0.80, 0.97);
        // Enforce monotonicity.
        if self.rl_pressure_bands[1] <= self.rl_pressure_bands[0] + 0.05 {
            self.rl_pressure_bands[1] = self.rl_pressure_bands[0] + 0.05;
        }
        if self.rl_pressure_bands[2] <= self.rl_pressure_bands[1] + 0.05 {
            self.rl_pressure_bands[2] = self.rl_pressure_bands[1] + 0.05;
        }

        self.rl_compressor_bands[0] = self.rl_compressor_bands[0].clamp(0.10, 0.50);
        self.rl_compressor_bands[1] = self.rl_compressor_bands[1].clamp(0.40, 0.80);
        if self.rl_compressor_bands[1] <= self.rl_compressor_bands[0] + 0.05 {
            self.rl_compressor_bands[1] = self.rl_compressor_bands[0] + 0.05;
        }

        self.zone_alpha = self.zone_alpha.clamp(0.001, 0.05);
        self.outcome_wait_secs = self.outcome_wait_secs.clamp(10, 60);
        self.outcome_effective_threshold = self.outcome_effective_threshold.clamp(0.005, 0.05);
        self.experience_pressure_band = self.experience_pressure_band.clamp(0.02, 0.25);
        self.nars_drift_threshold = self.nars_drift_threshold.clamp(0.05, 0.40);
        self.nars_decay_factor = self.nars_decay_factor.clamp(0.80, 0.99);
        self.cusum_k = self.cusum_k.clamp(0.005, 0.10);
        self.cusum_h = self.cusum_h.clamp(0.05, 0.30);
        self.pid_target = self.pid_target.clamp(0.40, 0.85);
        self.pid_decay = self.pid_decay.clamp(0.90, 0.999);
        self.ws_spike_threshold = self.ws_spike_threshold.clamp(10.0, 50.0);
        self.fluidity_degraded_threshold = self.fluidity_degraded_threshold.clamp(0.30, 0.90);
        self.hazard_lr = self.hazard_lr.clamp(0.001, 0.1);
        self.max_foreground_share = self.max_foreground_share.clamp(0.20, 0.60);
        self.max_background_share = self.max_background_share.clamp(0.05, 0.30);
        self.meta_effectiveness_ema = self.meta_effectiveness_ema.clamp(0.0, 1.0);
        self.meta_learning_velocity = self.meta_learning_velocity.clamp(0.0, 1.0);
    }
}

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

    /// Adaptive parameters replacing hardcoded thresholds.
    /// Auto-tuned from outcome data, persisted across restarts.
    /// `None` on old file format → falls back to `LearnableParams::default()`.
    #[serde(default)]
    pub learnable_params: Option<LearnableParams>,
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
    #[allow(clippy::too_many_arguments)]
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
        learnable_params: Option<LearnableParams>,
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
            learnable_params,
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
    ) -> (
        Option<OverflowHistory>,
        Option<FrozenStatePersisted>,
        Option<ArousalState>,
        Option<ProcessBaselineMap>,
        LearnableParams,
    ) {
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
        // Restore learnable params — validated + default-fallback.
        let mut lp = self.learnable_params.unwrap_or_default();
        lp.validate();
        (
            self.overflow_guard_history,
            self.frozen_pids,
            self.arousal_state,
            self.process_baselines,
            lp,
        )
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
                let near_prior =
                    (e.confidence - 0.5).abs() < 0.05 && (e.slow_confidence - 0.5).abs() < 0.05;
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

        // LearnableParams: clamp all adaptive thresholds to safe ranges.
        if let Some(lp) = &mut self.learnable_params {
            lp.validate();
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
        learnable_params: Option<LearnableParams>,
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
            learnable_params,
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
        let Some(mut state) = Self::load(path) else {
            return;
        };
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
                workload: 0,
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
            learnable_params: None,
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
            learnable_params: None,
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
            learnable_params: None,
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
    fn self_improve_does_not_increment_generations() {
        // persist_generations is incremented exactly once per persist cycle by
        // persist_improved() BEFORE calling self_improve(). self_improve() must
        // NOT increment it again — doing so would advance the counter by 2 per
        // cycle, making all decay / half-life calculations run at 2× intended rate.
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
            learnable_params: None,
        };
        state.self_improve();
        assert_eq!(
            state.persist_generations, 5,
            "self_improve must not touch persist_generations"
        );
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
            learnable_params: None,
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
            learnable_params: None,
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

    // ── LearnableParams tests ───────────────────────────────────────────

    #[test]
    fn learnable_params_defaults_match_hardcoded() {
        let lp = LearnableParams::default();
        assert_eq!(lp.kalman_pressure_r, 0.02);
        assert_eq!(lp.kalman_pressure_q, 0.005);
        assert_eq!(lp.rl_pressure_bands, [0.50, 0.80, 0.92]);
        assert_eq!(lp.rl_compressor_bands, [0.30, 0.60]);
        assert_eq!(lp.zone_alpha, 0.005);
        assert_eq!(lp.outcome_wait_secs, 30);
        assert_eq!(lp.outcome_effective_threshold, 0.01);
        assert_eq!(lp.experience_pressure_band, 0.10);
        assert_eq!(lp.nars_drift_threshold, 0.20);
        assert_eq!(lp.nars_decay_factor, 0.95);
        assert_eq!(lp.cusum_k, 0.02);
        assert_eq!(lp.cusum_h, 0.12);
        assert_eq!(lp.pid_target, 0.65);
        assert_eq!(lp.pid_decay, 0.98);
        assert_eq!(lp.ws_spike_threshold, 25.0);
        assert_eq!(lp.fluidity_degraded_threshold, 0.65);
        assert_eq!(lp.hazard_lr, 0.01);
        assert_eq!(lp.max_foreground_share, 0.40);
        assert_eq!(lp.max_background_share, 0.15);
        assert_eq!(lp.tuning_cycles, 0);
    }

    #[test]
    fn learnable_params_serde_roundtrip() {
        let lp = LearnableParams {
            kalman_pressure_r: 0.03,
            rl_pressure_bands: [0.45, 0.75, 0.90],
            zone_alpha: 0.01,
            tuning_cycles: 42,
            ..Default::default()
        };
        let json = serde_json::to_string(&lp).unwrap();
        let restored: LearnableParams = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.kalman_pressure_r, 0.03);
        assert_eq!(restored.rl_pressure_bands, [0.45, 0.75, 0.90]);
        assert_eq!(restored.zone_alpha, 0.01);
        assert_eq!(restored.tuning_cycles, 42);
    }

    #[test]
    fn learnable_params_validate_clamps_out_of_range() {
        let mut lp = LearnableParams {
            kalman_pressure_r: 999.0,
            kalman_pressure_q: -1.0,
            rl_pressure_bands: [0.01, 0.02, 0.03], // way below range
            rl_compressor_bands: [0.99, 0.01],     // inverted
            zone_alpha: 0.0,
            outcome_wait_secs: 0,
            nars_decay_factor: 0.0,
            pid_target: 0.0,
            meta_effectiveness_ema: 5.0,
            ..Default::default()
        };
        lp.validate();
        assert_eq!(lp.kalman_pressure_r, 0.5);
        assert_eq!(lp.kalman_pressure_q, 0.001);
        assert!(lp.rl_pressure_bands[0] >= 0.30);
        assert!(lp.rl_pressure_bands[1] > lp.rl_pressure_bands[0]);
        assert!(lp.rl_pressure_bands[2] > lp.rl_pressure_bands[1]);
        assert!(lp.rl_compressor_bands[1] > lp.rl_compressor_bands[0]);
        assert_eq!(lp.zone_alpha, 0.001);
        assert_eq!(lp.outcome_wait_secs, 10);
        assert_eq!(lp.nars_decay_factor, 0.80);
        assert_eq!(lp.pid_target, 0.40);
        assert_eq!(lp.meta_effectiveness_ema, 1.0);
    }

    #[test]
    fn learnable_params_backward_compat_missing_field() {
        // Simulate old learned_state.json without learnable_params field.
        let json = r#"{"version":1}"#;
        let state: LearnedState = serde_json::from_str(json).unwrap();
        assert!(state.learnable_params.is_none());
        // Default fallback works.
        let lp = state.learnable_params.unwrap_or_default();
        assert_eq!(lp.pid_target, 0.65);
    }

    #[test]
    fn learnable_params_monotonicity_enforcement() {
        let mut lp = LearnableParams {
            rl_pressure_bands: [0.55, 0.55, 0.55], // all same
            ..Default::default()
        };
        lp.validate();
        // After validation, must be strictly increasing with ≥0.05 gap.
        assert!(lp.rl_pressure_bands[1] > lp.rl_pressure_bands[0]);
        assert!(lp.rl_pressure_bands[2] > lp.rl_pressure_bands[1]);
    }
}
