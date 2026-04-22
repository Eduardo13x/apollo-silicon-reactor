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
use crate::engine::neuromodulator::NeuroState;
use crate::engine::effectiveness_tracker::{EffectivenessTracker, ProcessEffectiveness};
use crate::engine::nars_belief::ArousalState;
use crate::engine::nested_learner::NestedLearner;
use crate::engine::optimization_skills::{OptimizationSkill, SkillRegistry};
use crate::engine::outcome_tracker::{OutcomeTracker, OutcomeTrackerPersisted};
use crate::engine::overflow_guard::OverflowHistory;
use crate::engine::predictive_agent::SpecialistAccuracyTracker;
use crate::engine::process_baseline::ProcessBaselineMap;
use crate::engine::signal_intelligence::{SignalIntelligence, SignalIntelligencePersisted};
use crate::engine::teacher_consolidation::TeacherConsolidator;
use crate::engine::types::FrozenStatePersisted;
use crate::engine::unfreeze_decay::TauEstimate;

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

    // ── Diagnostics ───────────────────────────────────────────────────
    /// True when `nars_decay_factor` is stuck near its 0.90 floor (B5).
    /// At the floor, beliefs decay maximally fast and are therefore unreliable.
    /// Exposed as a diagnostic flag so future decision logic can reduce
    /// confidence in NARS outputs when this is set.
    #[serde(default)]
    pub nars_beliefs_stale: bool,
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
            nars_beliefs_stale: false,
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
        // B5 fix (round-3): raise decay floor 0.80 → 0.90.
        // Previously 7 persist cycles collapsed confidence to 0.80^7 ≈ 0.21
        // (79% evidence lost). New floor 0.90^7 ≈ 0.48 retains half the mass.
        self.nars_decay_factor = self.nars_decay_factor.clamp(0.90, 0.99);
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

    /// Second-order meta-learning: adjust learning rates based on system behavior.
    ///
    /// Called every 500 cycles. Tracks two signals:
    /// - `meta_effectiveness_ema`: EMA of overall optimization effectiveness
    /// - `meta_learning_velocity`: EMA of |param_delta| per tuning cycle
    ///
    /// Decision matrix:
    /// - Velocity low + effectiveness falling → stuck: multiply rates ×1.5 (explore more)
    /// - Velocity low + effectiveness stable  → converged: multiply rates ×0.8 (slow down)
    /// - Velocity high → actively adapting: no change
    ///
    /// Safety: only adjusts learning *rates*, never safety thresholds. All clamped.
    pub fn meta_learn(&mut self, current_effectiveness: f64, param_delta: f64) {
        // Update meta EMA trackers
        let alpha = 0.01; // very slow: half-life ≈ 69 cycles at 500-cycle intervals
        self.meta_effectiveness_ema =
            (1.0 - alpha) * self.meta_effectiveness_ema + alpha * current_effectiveness;
        self.meta_learning_velocity =
            (1.0 - alpha) * self.meta_learning_velocity + alpha * param_delta.abs();
        self.tuning_cycles += 1;

        // Need at least 3 meta-learning cycles before acting
        if self.tuning_cycles < 3 {
            return;
        }

        let velocity_low = self.meta_learning_velocity < 0.005;
        let effectiveness_falling = current_effectiveness < self.meta_effectiveness_ema - 0.02;
        let effectiveness_stable =
            (current_effectiveness - self.meta_effectiveness_ema).abs() < 0.02;

        if velocity_low && effectiveness_falling {
            // Stuck: increase exploration — multiply learning rates ×1.5.
            //
            // B4 fix (round-3): cap at an *interim* ceiling (half of the hard
            // clamp in `validate()`) before assignment so a crash between the
            // multiply and the next `validate()` call leaves rates in a still-
            // sane range.  Hard clamps: zone_alpha ≤ 0.05, hazard_lr ≤ 0.1.
            const ZONE_ALPHA_INTERIM_MAX: f64 = 0.025; // 0.05 / 2
            const HAZARD_LR_INTERIM_MAX: f64 = 0.05; // 0.1 / 2
            self.zone_alpha = (self.zone_alpha * 1.5).min(ZONE_ALPHA_INTERIM_MAX);
            self.hazard_lr = (self.hazard_lr * 1.5).min(HAZARD_LR_INTERIM_MAX);
            self.nars_decay_factor = (self.nars_decay_factor * 0.98).max(0.90); // faster forgetting (bounded by new 0.90 floor — B5)
        } else if velocity_low && effectiveness_stable {
            // Converged: slow down — multiply learning rates ×0.8
            self.zone_alpha *= 0.8;
            self.hazard_lr *= 0.8;
            self.nars_decay_factor = (self.nars_decay_factor * 1.005).min(0.99);
            // slower forgetting
        }
        // High velocity → actively adapting, no change needed

        // Re-validate after adjustment
        self.validate();

        // When decay is at floor, beliefs are unreliable — mark stale so
        // decision makers can reduce confidence in NARS outputs.
        // Threshold shifted with the new 0.90 floor (B5): mark stale when
        // within 2pp of the floor, indicating decay is stuck.
        self.nars_beliefs_stale = self.nars_decay_factor <= 0.92;
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

    /// NestedLearner L0/L1/L2 hierarchy state.
    /// Persisted so EMA quality signals survive restarts — otherwise the daemon
    /// cold-starts at 0.5 neutral and needs ~50 cycles to stabilize the L1 gate.
    /// [Google Nested Learning 2025] — multi-level context flow state.
    #[serde(default)]
    pub nested_learner: Option<NestedLearner>,

    /// GemmaTrust EMA per suggestion category (Interactive / Noise / Protected /
    /// Profile / Latency) + total consolidations + improvement count.  Without
    /// this, trust resets to 0.5 neutral on every daemon restart and the
    /// is_reliable() gate needs ≥3 fresh observations before Apollo re-accepts
    /// advice it already proved reliable pre-restart.  [McGaugh 2004] long-term
    /// consolidation; [Gray & Reuter 1992] atomic persistence of learned state.
    #[serde(default)]
    pub teacher_consolidator: Option<TeacherConsolidator>,

    /// Per-app learned τ for the unfreeze-decay ODE.  Without this, a daemon
    /// restart cold-starts every app's decay model and the predictive thaw
    /// gate falls back to `DEFAULT_TAU_SEC` for ~3 samples per app post-thaw.
    /// [Strogatz 2015 §2.3 — learned time constants of linear relaxation]
    #[serde(default)]
    pub unfreeze_decay_tau: Option<HashMap<String, TauEstimate>>,

    /// Neuromodulator raw signal levels — DA/ACh/NA/5-HT plus the
    /// low-pressure streak counter.  Without this, every daemon restart
    /// cold-starts at neutral (0.5) regardless of the system state before
    /// shutdown, discarding all accumulated reward-prediction history.
    ///
    /// [Schultz 1997] — reward prediction error signals require continuity;
    /// cold restarts erase the entire prediction history.
    ///
    /// `None` = first run or old file format → `ApolloNeuromodulator::new()` baseline.
    #[serde(default)]
    pub neuro_state: Option<NeuroState>,
}

/// Current schema version for [`LearnedState`].
///
/// Bump this constant whenever a structural change is made to `LearnedState`
/// that cannot be handled by `#[serde(default)]` alone (e.g., a field whose
/// absence must trigger a data-shape migration, not just a default value).
/// The migration logic lives in [`try_migrate`].
///
/// Version history:
/// - 0: implicit (files written before versioning was added — no `version` key)
/// - 1: first versioned baseline; no structural changes from v0
/// - 2: KalmanMV8 slot 3 semantics changed (pressure proxy → lyapunov_norm);
///       kf_mv is reset to None so the filter reconverges cleanly [Wolf 1985 FTLE]
pub const CURRENT_SCHEMA_VERSION: u32 = 2;

fn default_version() -> u32 {
    // Files that pre-date schema versioning have no `version` key.
    // Deserializing them yields 0, so `try_migrate` can handle the upgrade path.
    0
}

/// Migrate a [`LearnedState`] from `schema_version` up to [`CURRENT_SCHEMA_VERSION`].
///
/// This is a pure function: it never performs I/O and never panics.
/// Each `match` arm must leave `state.version` set to the version it produces.
///
/// # Adding a new migration
/// 1. Bump `CURRENT_SCHEMA_VERSION`.
/// 2. Add a `match` arm for the old version that transforms `state` and sets
///    `state.version` to the new version, then `continue`s the loop.
///
/// [Gray & Reuter 1992 §11] — write-ahead versioning prevents crash-recovery
/// from reading structurally-stale data.
pub fn try_migrate(schema_version: u32, mut state: LearnedState) -> LearnedState {
    let mut v = schema_version;
    loop {
        match v {
            // v0 → v1: first versioned baseline; no structural changes needed —
            // all fields already carry `#[serde(default)]`. Just stamp the version.
            0 => {
                state.version = 1;
                v = 1;
            }
            // v1 → v2: KalmanMV8 slot 3 changed from pressure proxy to lyapunov_norm.
            // Stale x[3] carries wrong-domain state; reset so the filter reconverges
            // in ~10 cycles rather than starting with a corrupted initial estimate.
            // [Wolf et al. 1985 Physica D §3] — FTLE slot is orthogonal to pressure.
            1 => {
                if let Some(si) = state.signal_intelligence.as_mut() {
                    si.kf_mv = None;
                }
                state.version = 2;
                v = 2;
            }
            // Up to date — nothing left to migrate.
            _ if v >= CURRENT_SCHEMA_VERSION => {
                state.version = CURRENT_SCHEMA_VERSION;
                return state;
            }
            // Unknown future version loaded by an older binary. Keep as-is so
            // we do not destroy data the older binary cannot understand.
            _ => return state,
        }
    }
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
        nested_learner: Option<NestedLearner>,
    ) -> Self {
        Self {
            version: CURRENT_SCHEMA_VERSION,
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
            nested_learner,
            teacher_consolidator: None,
            unfreeze_decay_tau: None,
            neuro_state: None,
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
        Option<NestedLearner>,
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
            self.nested_learner,
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
            //    Factor from LearnableParams (default 0.95 → half-life ≈ 14 persist cycles).
            //    Meta-learning adjusts it: stuck→faster forgetting, converged→slower.
            if let Some(dd) = &mut ot.drift_detector {
                let decay = self
                    .learnable_params
                    .as_ref()
                    .map(|lp| lp.nars_decay_factor)
                    .unwrap_or(0.95);
                dd.decay_confidence(decay);
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
            // Prune edges that have lost signal:
            //   a) near-prior confidence + low evidence (cold edge, never converged), OR
            //   b) mechanism EMAs all decayed to near-zero (high-evidence edge that
            //      hasn't been updated in many persists — staleness gate).
            // Without (b) a stale edge with e.g. evidence_count=200 from a workload
            // that no longer runs survives forever and corrupts ranking.
            edges.retain(|(_, e)| {
                let near_prior =
                    (e.confidence - 0.5).abs() < 0.05 && (e.slow_confidence - 0.5).abs() < 0.05;
                let mech_dead = e.mechanism.rss_delta_mb.abs() < 0.5
                    && e.mechanism.cpu_delta_pct.abs() < 0.5
                    && e.mechanism.swap_delta_mb.abs() < 0.5;
                let cold_unconverged = near_prior && e.evidence_count < 10;
                let stale_high_evidence = near_prior && mech_dead;
                !(cold_unconverged || stale_high_evidence)
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
        nested_learner: Option<NestedLearner>,
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
            nested_learner,
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
    ///
    /// Uses atomic write (tmp → rename) so a crash mid-write leaves the
    /// PREVIOUS state intact rather than a truncated/empty file. Without
    /// this, a kernel panic, OOM kill or power loss during the write
    /// would corrupt learned_state.json and destroy ALL learned state
    /// (RL thresholds, NARS beliefs, causal graph, experience memory,
    /// learnable params, arousal state).
    ///
    /// [Gray & Reuter 1992] §10 — WAL/atomic-replace: the previous
    /// committed state must survive any single-point failure.
    pub fn persist(&self, path: &Path) {
        let json = match serde_json::to_string(self) {
            Ok(j) => j,
            Err(_) => return,
        };
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, &json).is_ok() {
            let _ = std::fs::rename(&tmp, path);
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

    /// Patch only the `teacher_consolidator` field of an existing persisted file.
    /// Same rationale as `patch_process_baselines` — persist_improved() does not
    /// thread the TeacherConsolidator through its signature; callers invoke this
    /// after persist_improved() to snapshot Gemma trust EMA + consolidation totals.
    /// No-op if file is missing (cold start is safe).
    pub fn patch_teacher_consolidator(path: &Path, tc: TeacherConsolidator) {
        let Some(mut state) = Self::load(path) else {
            return;
        };
        state.teacher_consolidator = Some(tc);
        state.persist(path);
    }

    /// Patch only the `unfreeze_decay_tau` field of an existing persisted file.
    /// Same pattern as `patch_teacher_consolidator` — persist_improved() does
    /// not thread the UnfreezeDecayModel through its signature; callers invoke
    /// this after persist_improved() to snapshot learned τ per app.
    /// No-op if file is missing (cold start is safe).
    pub fn patch_unfreeze_decay(path: &Path, tau_map: HashMap<String, TauEstimate>) {
        let Some(mut state) = Self::load(path) else {
            return;
        };
        state.unfreeze_decay_tau = Some(tau_map);
        state.persist(path);
    }

    /// Patch only the `neuro_state` field of an existing persisted file.
    ///
    /// Same pattern as `patch_unfreeze_decay` — `persist_improved()` does not
    /// thread `ApolloNeuromodulator` through its signature.  Callers invoke this
    /// after `persist_improved()` to snapshot the four neurotransmitter levels so
    /// DA/ACh/NA/5-HT state survives daemon restarts without cold-starting at 0.5.
    ///
    /// No-op if the file is missing (cold start is safe — neuromodulator
    /// initialises at baseline on the first ever run).
    ///
    /// [Schultz 1997] — reward prediction error signals require continuity.
    pub fn patch_neuro_state(path: &Path, ns: NeuroState) {
        let Some(mut state) = Self::load(path) else {
            return;
        };
        state.neuro_state = Some(ns);
        state.persist(path);
    }

    /// Load only the `neuro_state` field from disk (cold-start safe).
    /// Returns `None` if the file is missing, unreadable, malformed, or the
    /// field is absent (old file format pre-dating NeuroState persistence).
    pub fn load_neuro_state(path: &Path) -> Option<NeuroState> {
        Self::load(path)?.neuro_state
    }

    /// Load only the `teacher_consolidator` field from disk (cold-start safe).
    /// Returns `None` if the file is missing, unreadable, malformed, or the
    /// field is absent (old file format pre-dating GemmaTrust persistence).
    pub fn load_teacher_consolidator(path: &Path) -> Option<TeacherConsolidator> {
        Self::load(path)?.teacher_consolidator
    }

    /// Load from disk. Returns None on any error (cold start is safe).
    ///
    /// Automatically runs [`try_migrate`] so callers always receive a struct
    /// at [`CURRENT_SCHEMA_VERSION`], regardless of how old the on-disk file is.
    pub fn load(path: &Path) -> Option<Self> {
        let data = std::fs::read_to_string(path).ok()?;
        let state: Self = serde_json::from_str(&data).ok()?;
        Some(try_migrate(state.version, state))
    }
}

// ── Restore Quality Monitor ─────────────────────────────────────────────────

/// Tracks whether restored state is helping or hurting.
///
/// Two-phase measurement:
///   1. Warmup (20 cycles, ~40s): observations are discarded because post-restart
///      data is contaminated by stale pending outcomes and startup scan noise.
///   2. Observation (50 cycles, ~100s): clean measurement of effectiveness.
///
/// The verdict compares the measured quality against the long-term steady-state
/// effective rate (supplied by the caller via `overall_effectiveness()`), not
/// against a hardcoded threshold. Stale = quality dropped to <50% of baseline.
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

/// Warmup cycles to skip after restore (~60s at 2s/cycle).
///
/// Observation at cycle N judges a throttle started at cycle N - (outcome_wait
/// /cycle_interval) ≈ N - 15. So a 30-cycle warmup ensures we observe only
/// throttles that were started after cycle 15 — past the initial daemon scan
/// and log ingester burst. Combined with the 50-cycle observation window, the
/// full (warmup + observation) window is 80 cycles ≈ 160 seconds post-restart.
const WARMUP_CYCLES: u32 = 30;
/// Observation window: 50 cycles of *clean* data after warmup (~100s).
const QUALITY_WINDOW: u32 = 50;
/// Minimum resolved outcomes required for a statistically meaningful verdict.
/// Below this, we assume OK (can't judge with too few samples).
const MIN_RESOLVED: u32 = 30;
/// Stale detection: quality must drop below this fraction of the long-term
/// steady-state effectiveness rate. A 50% drop from baseline is clearly broken;
/// smaller fluctuations are normal variance.
///
/// Why relative, not absolute: the previous code used an absolute threshold
/// of 0.35, but real steady-state effective rate is ~0.20 (19.67% in production).
/// An absolute threshold of 0.35 would flag *any* healthy system as stale.
const STALE_RATIO: f64 = 0.5;

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
    /// During the warmup period the call is counted toward cycle progress but
    /// the effective/resolved accumulators are NOT touched — post-restart noise
    /// must not pollute the quality measurement.
    pub fn observe(&mut self, batch_effective: u32, batch_resolved: u32) {
        if self.fired {
            return;
        }
        self.cycles += 1;
        if self.cycles <= WARMUP_CYCLES {
            return;
        }
        self.effective += batch_effective;
        self.resolved += batch_resolved;
    }

    /// Check if the observation window is complete and return a verdict.
    ///
    /// The verdict compares the measured quality against the caller's long-term
    /// steady-state effective rate, NOT against a hardcoded constant. Stale is
    /// defined as a drop to less than `STALE_RATIO` of steady-state.
    ///
    /// Returns `Some(verdict)` once the full (warmup + observation) window has
    /// elapsed; `None` while still observing or after the monitor has fired.
    pub fn verdict(&mut self, long_term_rate: f64) -> Option<RestoreVerdict> {
        if self.fired || self.cycles < WARMUP_CYCLES + QUALITY_WINDOW {
            return None;
        }
        self.fired = true;
        if self.resolved < MIN_RESOLVED {
            // Not enough clean samples to judge — assume OK.
            return Some(RestoreVerdict {
                quality: 0.5,
                stale: false,
            });
        }
        let quality = (self.effective as f64 + 1.0) / (self.resolved as f64 + 2.0);
        // Stale threshold: half of steady-state. Floored at 0.02 so a cold-start
        // system (long_term_rate ≈ 0) still has a sane comparison point.
        let stale_threshold = (long_term_rate * STALE_RATIO).max(0.02);
        Some(RestoreVerdict {
            quality,
            stale: quality < stale_threshold,
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
        ExperienceRecord, PatternWeight,
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
            nested_learner: None,
            teacher_consolidator: None,
            unfreeze_decay_tau: None,
            neuro_state: None,
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
            nested_learner: None,
            teacher_consolidator: None,
            unfreeze_decay_tau: None,
            neuro_state: None,
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
            nested_learner: None,
            teacher_consolidator: None,
            unfreeze_decay_tau: None,
            neuro_state: None,
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
            nested_learner: None,
            teacher_consolidator: None,
            unfreeze_decay_tau: None,
            neuro_state: None,
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
            kf_mv: None,
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
            nested_learner: None,
            teacher_consolidator: None,
            unfreeze_decay_tau: None,
            neuro_state: None,
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
            nested_learner: None,
            teacher_consolidator: None,
            unfreeze_decay_tau: None,
            neuro_state: None,
        };
        state.validate();
        let ot = state.outcome_tracker.as_ref().unwrap();
        assert_eq!(ot.baseline_drop_ema, 1.0);
        assert_eq!(ot.natural_drift_ema, -0.2);
    }

    #[test]
    fn restore_quality_monitor_detects_stale() {
        let mut monitor = RestoreQualityMonitor::new();
        // Feed WARMUP_CYCLES + QUALITY_WINDOW total. Within the observation
        // window (post-warmup), only 2 of 50 outcomes are effective — that is
        // well below half of a 20% steady-state.
        for i in 0..(WARMUP_CYCLES + QUALITY_WINDOW) {
            let in_observation = i >= WARMUP_CYCLES;
            let eff = if in_observation && i < WARMUP_CYCLES + 2 {
                1
            } else {
                0
            };
            monitor.observe(eff, 1);
        }
        // Simulate a healthy steady-state of 20% (matches real production).
        let verdict = monitor.verdict(0.20).expect("should have verdict");
        assert!(
            verdict.stale,
            "effectiveness < 50% of steady-state must be flagged stale"
        );
        // quality ≈ 3/52 ≈ 0.058; stale_threshold = 0.20 * 0.5 = 0.10.
        assert!(verdict.quality < 0.10);
    }

    #[test]
    fn restore_quality_monitor_approves_good_state() {
        let mut monitor = RestoreQualityMonitor::new();
        // 40/50 effective in the observation window — clearly above baseline.
        for i in 0..(WARMUP_CYCLES + QUALITY_WINDOW) {
            let in_observation = i >= WARMUP_CYCLES;
            let eff = if in_observation && i < WARMUP_CYCLES + 40 {
                1
            } else {
                0
            };
            monitor.observe(eff, 1);
        }
        let verdict = monitor.verdict(0.20).expect("should have verdict");
        assert!(!verdict.stale, "good effectiveness should not be stale");
        assert!(verdict.quality > 0.7);
    }

    #[test]
    fn restore_quality_fires_only_once() {
        let mut monitor = RestoreQualityMonitor::new();
        for _ in 0..(WARMUP_CYCLES + QUALITY_WINDOW) {
            monitor.observe(1, 1);
        }
        assert!(monitor.verdict(0.20).is_some());
        assert!(
            monitor.verdict(0.20).is_none(),
            "second call should return None"
        );
        assert!(monitor.is_done());
    }

    #[test]
    fn restore_quality_monitor_ignores_warmup_noise() {
        // This test reproduces the production bug: the first cycles post-restart
        // have 0 effective outcomes (pending actions resolving with stale data).
        // Before the fix, this contaminated the measurement; after the fix, the
        // warmup is skipped and the clean observation window sees good data.
        let mut monitor = RestoreQualityMonitor::new();

        // Warmup: 20 cycles of pure noise (0 effective, 1 resolved each).
        for _ in 0..WARMUP_CYCLES {
            monitor.observe(0, 1);
        }
        // Observation: 50 cycles at healthy 20% effectiveness.
        for i in 0..QUALITY_WINDOW {
            let eff = if i < 10 { 1 } else { 0 };
            monitor.observe(eff, 1);
        }

        let verdict = monitor.verdict(0.20).expect("should have verdict");
        // Measured quality should reflect ONLY the observation window.
        // 10/50 effective → (10+1)/(50+2) = 11/52 ≈ 0.212 (healthy).
        assert!(
            !verdict.stale,
            "warmup noise must NOT contaminate verdict (quality={})",
            verdict.quality
        );
        assert!(verdict.quality > 0.18 && verdict.quality < 0.25);
    }

    #[test]
    fn restore_quality_monitor_waits_for_minimum_samples() {
        // Even after the full window, if too few outcomes resolved (sparse
        // traffic), the monitor returns neutral 0.5 instead of a noisy verdict.
        let mut monitor = RestoreQualityMonitor::new();
        for i in 0..(WARMUP_CYCLES + QUALITY_WINDOW) {
            // Only 10 resolved outcomes total in the observation window —
            // below MIN_RESOLVED = 30.
            let resolved = if i >= WARMUP_CYCLES && i < WARMUP_CYCLES + 10 {
                1
            } else {
                0
            };
            monitor.observe(0, resolved);
        }
        let verdict = monitor.verdict(0.20).expect("should have verdict");
        assert!(!verdict.stale, "too-few-samples verdict must not be stale");
        assert_eq!(verdict.quality, 0.5);
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
        assert_eq!(lp.nars_decay_factor, 0.90);
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

    // ── Meta-learning tests (Phase 6) ───────────────────────────────────────

    #[test]
    fn meta_learn_stuck_increases_learning_rates() {
        let mut lp = LearnableParams::default();
        // Build up a solid meta_effectiveness_ema by running many cycles
        // α=0.01 → need ~100 cycles to converge
        for _ in 0..200 {
            lp.meta_learn(0.60, 0.001);
        }
        let alpha_before = lp.zone_alpha;
        let lr_before = lp.hazard_lr;
        // Now effectiveness drops below the EMA - 0.02 → stuck
        lp.meta_learn(0.30, 0.001);
        assert!(
            lp.zone_alpha > alpha_before || lp.hazard_lr > lr_before,
            "stuck system should increase learning rates: zone_alpha {} vs {}, hazard_lr {} vs {}",
            lp.zone_alpha,
            alpha_before,
            lp.hazard_lr,
            lr_before
        );
    }

    #[test]
    fn meta_learn_converged_decreases_learning_rates() {
        let mut lp = LearnableParams::default();
        // Build up stable effectiveness EMA
        for _ in 0..200 {
            lp.meta_learn(0.50, 0.001);
        }
        let alpha_before = lp.zone_alpha;
        let lr_before = lp.hazard_lr;
        // Simulate converged: low velocity, stable effectiveness (within ±0.02)
        lp.meta_learn(0.50, 0.001);
        assert!(
            lp.zone_alpha <= alpha_before && lp.hazard_lr <= lr_before,
            "converged system should decrease learning rates: zone_alpha {} vs {}, hazard_lr {} vs {}",
            lp.zone_alpha, alpha_before, lp.hazard_lr, lr_before
        );
    }

    #[test]
    fn meta_learn_respects_clamps_after_many_stuck_cycles() {
        let mut lp = LearnableParams::default();
        // Many stuck cycles → rates should be clamped
        for i in 0..100 {
            lp.meta_learn(0.50 - (i as f64) * 0.005, 0.001);
        }
        assert!(
            lp.zone_alpha <= 0.05,
            "zone_alpha should be clamped: {}",
            lp.zone_alpha
        );
        assert!(
            lp.hazard_lr <= 0.1,
            "hazard_lr should be clamped: {}",
            lp.hazard_lr
        );
    }

    #[test]
    fn meta_learn_no_action_before_warmup() {
        let mut lp = LearnableParams::default();
        let alpha_before = lp.zone_alpha;
        // Only 2 cycles → no adjustment yet
        lp.meta_learn(0.30, 0.001);
        lp.meta_learn(0.30, 0.001);
        assert_eq!(
            lp.zone_alpha, alpha_before,
            "should not adjust before warmup"
        );
    }

    #[test]
    fn meta_learn_tuning_cycles_increment() {
        let mut lp = LearnableParams::default();
        assert_eq!(lp.tuning_cycles, 0);
        lp.meta_learn(0.50, 0.01);
        assert_eq!(lp.tuning_cycles, 1);
        lp.meta_learn(0.50, 0.01);
        assert_eq!(lp.tuning_cycles, 2);
    }

    #[test]
    fn teacher_consolidator_default_absent_on_collect() {
        use crate::engine::effectiveness_tracker::EffectivenessTracker;
        use crate::engine::optimization_skills::SkillRegistry;
        use crate::engine::outcome_tracker::OutcomeTracker;
        use crate::engine::predictive_agent::SpecialistAccuracyTracker;
        use crate::engine::signal_intelligence::SignalIntelligence;

        let si = SignalIntelligence::new();
        let ot = OutcomeTracker::new();
        let sa = SpecialistAccuracyTracker::new();
        let sr = SkillRegistry::new();
        let et = EffectivenessTracker::new();
        let state = LearnedState::collect(
            &si, &ot, &sa, &sr, &et, None, None, None, None, None, None, None,
        );
        assert!(state.teacher_consolidator.is_none(),
            "collect() leaves teacher_consolidator None; callers must patch post-persist");
    }

    #[test]
    fn patch_teacher_consolidator_roundtrip() {
        use crate::engine::teacher_consolidation::{SuggestionCategory, TeacherConsolidator};
        let tmp = std::env::temp_dir().join(format!(
            "apollo_tc_patch_{}.json",
            std::process::id()
        ));
        // Seed a minimal file so load() succeeds.
        let seed = LearnedState {
            version: 1,
            signal_intelligence: None,
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
            nested_learner: None,
            teacher_consolidator: None,
            unfreeze_decay_tau: None,
            neuro_state: None,
        };
        seed.persist(&tmp);

        let mut tc = TeacherConsolidator::new();
        // Drive one IMPROVED observation on Noise so trust > 0.5.
        tc.gemma_trust.update(SuggestionCategory::Noise, 1.0);
        tc.total_consolidations = 7;
        tc.total_improvements = 5;

        LearnedState::patch_teacher_consolidator(&tmp, tc.clone());

        let loaded = LearnedState::load_teacher_consolidator(&tmp)
            .expect("patched field must survive round-trip");
        assert_eq!(loaded.total_consolidations, 7);
        assert_eq!(loaded.total_improvements, 5);
        assert_eq!(loaded.gemma_trust.count(SuggestionCategory::Noise), 1);
        assert!(loaded.gemma_trust.trust(SuggestionCategory::Noise) > 0.5);
        // Untouched categories fall back to the neutral 0.5 default.
        assert!(
            (loaded.gemma_trust.trust(SuggestionCategory::Interactive) - 0.5).abs() < 1e-9
        );

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn load_teacher_consolidator_missing_file_returns_none() {
        let tmp = std::env::temp_dir()
            .join(format!("apollo_tc_missing_{}.json", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        assert!(LearnedState::load_teacher_consolidator(&tmp).is_none());
    }

    #[test]
    fn patch_teacher_consolidator_noop_when_file_missing() {
        use crate::engine::teacher_consolidation::TeacherConsolidator;
        let tmp = std::env::temp_dir()
            .join(format!("apollo_tc_noop_{}.json", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        // Must not panic, must not create the file.
        LearnedState::patch_teacher_consolidator(&tmp, TeacherConsolidator::new());
        assert!(!tmp.exists(), "patch is no-op when the state file is absent");
    }

    #[test]
    fn teacher_consolidator_serde_backward_compat_missing_field() {
        // Old file format: no teacher_consolidator key. Must deserialize cleanly
        // with the field defaulting to None, so upgrades do not erase state.
        let old_json = r#"{"version":1}"#;
        let state: LearnedState = serde_json::from_str(old_json)
            .expect("missing teacher_consolidator must default to None");
        assert!(state.teacher_consolidator.is_none());
    }

    // ── Schema versioning tests ─────────────────────────────────────────────

    #[test]
    fn test_schema_version_default_is_zero() {
        // JSON with no `version` key represents a pre-versioning file.
        // `default_version()` must return 0 so `try_migrate` can upgrade it.
        let state: LearnedState = serde_json::from_str("{}").expect("empty object must deserialize");
        assert_eq!(
            state.version, 0,
            "missing version key must deserialize as 0 (pre-versioning baseline)"
        );
    }

    #[test]
    fn test_migrate_v0_to_current() {
        // v0 → v1 is a no-op baseline: no structural changes, just stamps version.
        let state = LearnedState {
            version: 0,
            signal_intelligence: None,
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
            nested_learner: None,
            teacher_consolidator: None,
            unfreeze_decay_tau: None,
            neuro_state: None,
        };
        let migrated = try_migrate(0, state);
        assert_eq!(
            migrated.version, CURRENT_SCHEMA_VERSION,
            "try_migrate(0, _) must stamp version == CURRENT_SCHEMA_VERSION"
        );
    }

    #[test]
    fn test_migrate_v1_resets_kf_mv() {
        // v1 → v2: kf_mv slot 3 changed semantics (pressure proxy → lyapunov_norm).
        // Migration must clear kf_mv so the filter reconverges cleanly rather than
        // inheriting stale slot-3 state from the previous signal assignment.
        use crate::engine::signal_intelligence::SignalIntelligence;
        let si_persisted = SignalIntelligence::new().to_persisted();
        assert!(si_persisted.kf_mv.is_some(), "precondition: kf_mv present");
        let state = LearnedState {
            version: 1,
            signal_intelligence: Some(si_persisted),
            outcome_tracker: None,
            specialist_accuracy: None,
            persist_generations: 0,
            last_restore_quality: None,
            pending_trial_skill: None,
            skill_registry: None,
            effectiveness_tracker: None,
            overflow_guard_history: None,
            frozen_pids: None,
            arousal_state: None,
            causal_graph_edges: None,
            process_baselines: None,
            learnable_params: None,
            nested_learner: None,
            teacher_consolidator: None,
            unfreeze_decay_tau: None,
            neuro_state: None,
        };
        let migrated = try_migrate(1, state);
        assert_eq!(migrated.version, CURRENT_SCHEMA_VERSION);
        assert!(
            migrated
                .signal_intelligence
                .as_ref()
                .and_then(|si| si.kf_mv.as_ref())
                .is_none(),
            "v1→v2 must clear kf_mv to None"
        );
    }
}
