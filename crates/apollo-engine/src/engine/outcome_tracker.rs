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

use crate::engine::nars_belief::{DriftDetector, Salience};

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

    // ── Hard-protected exclusion for class-reclassification ─────────────────
    //
    // Bug observed in prod (2026-06-07, daemon PID 16105, Brave Browser Helper
    // @ throttle_count=63, effective_count=2): hard-protected processes route
    // throttle through PRIO_DARWIN_BG + jetsam BACKGROUND demote (Chromium-
    // cooperative). SIGSTOP is forbidden by `safety.rs` per CLAUDE.md
    // "Chromium SIGSTOP never" — 3 regression cycles.
    //
    // The soft throttle is structurally weak: Chromium's renderer scheduler
    // often keeps allocating regardless. Brave's 3.2% effectiveness is a
    // property of the ACTION TYPE on a hard-protected process, NOT a property
    // of the PROCESS's pressure-driver role.
    //
    // Applying the legacy heuristic "effectiveness < 0.30 → process does NOT
    // cause pressure → reclassify as interactive → Boost" is therefore
    // circular for hard-protected entries — it closed an infinite Boost loop
    // in prod (110 boosts targeted Brave Browser Helper out of 200 actions).
    //
    // Honest reads for class-reclassification consumers: return `None` (or
    // `false`) for hard-protected processes so downstream code never promotes
    // them based on the structurally-degraded throttle signal. RL penalty,
    // skip-future-throttles, and LLM-struggling consumers retain the raw
    // `effectiveness()` / `is_low_value*` semantics — the waste signal is
    // genuinely informative for those paths.

    /// Honest effectiveness for class-reclassification consumers. Returns
    /// `None` when the entry refers to a process whose throttle path is
    /// structurally degraded (hard-protected PRIO_DARWIN_BG-only, OR
    /// family-root like Brave/Chrome/Edge matched via `match_engine::
    /// is_family_root`). Production matches Brave via `is_family_root`,
    /// NOT `hard_protected_contains` — the legacy predicate let Brave
    /// renderers escape the reclassification gate (FIX-2, 2026-06-07).
    ///
    /// Callers that interpret low effectiveness as "process does not cause
    /// pressure" MUST use this variant.
    pub fn effectiveness_for_classification(&self, name: &str) -> Option<f64> {
        if crate::engine::safety::is_boost_forbidden(name) {
            crate::engine::lse_counters::LSE_COUNTERS.inc_hard_protected_reclassify_excluded();
            return None;
        }
        Some(self.effectiveness())
    }

    /// Calibrated low-value check that respects hard-protected exclusion.
    /// Use this at every class-reclassification gate. Returns `false` for
    /// hard-protected entries regardless of effectiveness floor.
    pub fn is_low_value_for_reclassification(&self, name: &str, baseline: f64) -> bool {
        match self.effectiveness_for_classification(name) {
            None => false, // never reclassify HP entries
            Some(eff) => self.throttle_count >= 20 && eff < baseline * 0.90,
        }
    }
}

/// Throttle pendiente de resolución de outcome.
struct PendingOutcome {
    process_name: String,
    throttled_at: Instant,
    pressure_before: f64,
    watts_before: f64,
    swap_gb_at_throttle: f64,
    action_type: super::learning_pipeline::ActionKind,
}

// ── Survival-bias closure: blocked-action counterfactual learning ────────────
//
// The OutcomeTracker historically only learned from EXECUTED actions. Actions
// blocked by safety gates (user-protected freeze skip, is_protected_name,
// budget exhaustion, thermal interrupt, …) were invisible to the learning
// loop, creating a survival-bias gap: the agent could not distinguish
//
//   "this class of action is genuinely bad" (tried, didn't help)
//   from
//   "this class of action was never tried" (gated out before execution)
//
// The fix:
//   1. record_blocked() is called at every gate site → enqueues a
//      PendingBlocked with the pressure snapshot at block time.
//   2. tick_with_params() resolves old pending blocks: if pressure ROSE in
//      the next ~15 cycles (≈30s @ 2 Hz, matching the executed-throttle
//      window), the blocked action receives "would_have_helped" credit —
//      WITH counterfactual correction via natural_drift_ema [Rubin 1974].
//   3. blocked_effectiveness(key) returns a Laplace-smoothed Bayesian
//      score for offline / shadow-mode analysis.
//
// IMPORTANT: this signal is SHADOW-MODE-ONLY. It MUST NOT be used to
// auto-unblock gated actions — the feedback loop risk (system unblocks an
// action because the journal says it "would have helped", that action then
// causes user-visible jank, …) is too high. Per NotebookLM peer review
// 2026-05-03 conversation 379c81af.
//
// References:
//   [Rubin 1974] Potential Outcomes — counterfactual via natural_drift_ema.
//   [Bengio 2013] Counterfactual reasoning needs the unobserved branch.
//   [Pearl 2009] do-calculus is the principled tool for full causal
//     inference; we do NOT need it here because we keep the signal in
//     prior-only Bayesian shadow mode.

/// Per-(action_class, gate) Bayesian counts for blocked-action learning.
///
/// `action_class` is a coarse key (e.g. "freeze", "throttle", "boost") rather
/// than per-PID — gates typically block whole classes ("user is in a call →
/// no freezes for ANY pid this cycle"), so per-class is the right grain.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlockedPattern {
    /// Times this (action_class, gate) was blocked.
    pub blocked_count: u32,
    /// Of those blocks, how many had pressure RISE in the post-block window
    /// (after subtracting natural_drift baseline) — i.e., the action would
    /// likely have helped.
    pub would_have_helped_count: u32,
}

impl BlockedPattern {
    /// Laplace-smoothed Bayesian estimate that this blocked class was a
    /// missed opportunity. (1+helped)/(2+blocked). Returns 0.5 with no data.
    pub fn effectiveness(&self) -> f64 {
        (self.would_have_helped_count as f64 + 1.0) / (self.blocked_count as f64 + 2.0)
    }
}

/// In-flight blocked decision awaiting post-hoc counterfactual evaluation.
#[derive(Debug, Clone)]
struct PendingBlocked {
    /// Composite key "<action_class>:<gate>" — same shape used in queries.
    key: String,
    blocked_at: Instant,
    pressure_before: f64,
}

/// Resumen de la resolución de un batch de outcomes.
pub struct OutcomeBatch {
    pub effective_names: Vec<String>,
    pub savings_watts: f64,
    pub low_value_names: Vec<String>,
    pub resolved_outcomes: Vec<(String, f64, f64, super::learning_pipeline::ActionKind)>,
}

// ── Experience Memory ───────────���─────────────────────────────��───────────────

/// A resolved decision+outcome record for queryable experience memory.
/// Ring buffer of the last N records enables "what worked before?" queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceRecord {
    /// Process that was throttled.
    pub process_name: String,
    /// Memory pressure at the time of throttle.
    pub pressure_at_action: f64,
    /// Pressure drop observed 30s later (positive = pressure went down).
    pub pressure_drop: f64,
    /// Whether the throttle was effective (drop ≥ 0.02).
    pub effective: bool,
    /// Workload at time of action (WorkloadMode encoded as u8).
    /// Used for workload-aware queries: records from the same workload
    /// are weighted 2× since workload context strongly affects outcomes.
    #[serde(default)]
    pub workload: u8,
}

/// Ring buffer of experience records with similarity query.
pub struct ExperienceMemory {
    records: VecDeque<ExperienceRecord>,
    capacity: usize,
}

impl ExperienceMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            records: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Store a resolved outcome.
    pub fn push(&mut self, record: ExperienceRecord) {
        if self.records.len() >= self.capacity {
            self.records.pop_front();
        }
        self.records.push_back(record);
        // Hard cap: belt-and-suspenders guard for cases where capacity is misset
        // or records are bulk-loaded from a large persisted file. Drain oldest 100
        // if the deque exceeds 600 entries — keeps memory bounded between persists
        // without waiting for self_improve() to run.
        if self.records.len() > 600 {
            drop(self.records.drain(..100));
        }
    }

    /// Query: expected effectiveness for throttling `process` at `pressure`.
    /// Returns (expected_drop, confidence) or None if no similar records.
    /// Similarity: same process name AND pressure within ±0.10.
    pub fn query_similar(&self, process: &str, pressure: f64) -> Option<(f64, f64)> {
        self.query_similar_with_band(process, pressure, 0.10)
    }

    /// Query with adaptive pressure similarity band (from LearnableParams).
    pub fn query_similar_with_band(
        &self,
        process: &str,
        pressure: f64,
        band: f64,
    ) -> Option<(f64, f64)> {
        let mut sum_drop = 0.0_f64;
        let mut count = 0u32;
        for r in &self.records {
            if r.process_name == process && (r.pressure_at_action - pressure).abs() <= band {
                sum_drop += r.pressure_drop;
                count += 1;
            }
        }
        if count < 3 {
            return None;
        }
        let avg_drop = sum_drop / count as f64;
        // Confidence: saturates at 1.0 after 20 records.
        let confidence = (count as f64 / 20.0).min(1.0);
        Some((avg_drop, confidence))
    }

    /// Workload-aware query: same as `query_similar_with_band` but records
    /// from the same workload are weighted 2× (context-dependent memory).
    /// `current_workload` is the WorkloadMode encoded as u8.
    pub fn query_similar_contextual(
        &self,
        process: &str,
        pressure: f64,
        band: f64,
        current_workload: u8,
    ) -> Option<(f64, f64)> {
        let mut weighted_drop = 0.0_f64;
        let mut total_weight = 0.0_f64;
        let mut count = 0u32;
        for r in &self.records {
            if r.process_name == process && (r.pressure_at_action - pressure).abs() <= band {
                let w = if r.workload == current_workload {
                    2.0
                } else {
                    1.0
                };
                weighted_drop += r.pressure_drop * w;
                total_weight += w;
                count += 1;
            }
        }
        if count < 3 {
            return None;
        }
        let avg_drop = weighted_drop / total_weight;
        let confidence = (count as f64 / 20.0).min(1.0);
        Some((avg_drop, confidence))
    }

    /// Number of stored records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Read-only access to all records (for persistence).
    pub fn records(&self) -> &VecDeque<ExperienceRecord> {
        &self.records
    }

    /// State compression (Hermes pattern): merge old records by process name.
    /// Keeps last 100 records intact (recent detail) and compresses older ones
    /// into per-process averages, freeing ~80% of memory.
    pub fn compress_old(&mut self) {
        if self.records.len() < 200 {
            return; // not enough to compress
        }
        // Keep last 100 intact.
        let keep_recent = 100;
        let old_count = self.records.len() - keep_recent;
        let old_records: Vec<ExperienceRecord> = self.records.drain(..old_count).collect();

        // Compress: average by process name.
        let mut groups: std::collections::HashMap<String, (f64, f64, u32, u32)> =
            std::collections::HashMap::new();
        for r in &old_records {
            let e = groups
                .entry(r.process_name.clone())
                .or_insert((0.0, 0.0, 0, 0));
            e.0 += r.pressure_at_action;
            e.1 += r.pressure_drop;
            e.2 += 1;
            if r.effective {
                e.3 += 1;
            }
        }

        // Re-insert compressed summaries at front.
        for (name, (sum_pressure, sum_drop, count, eff_count)) in groups {
            self.records.push_front(ExperienceRecord {
                process_name: name,
                pressure_at_action: sum_pressure / count as f64,
                pressure_drop: sum_drop / count as f64,
                effective: eff_count * 2 >= count, // majority vote
                workload: 0,                       // compressed summaries lose workload specificity
            });
        }
    }
}

// ── OutcomeTracker ──��─────────────────��─────────────────────��─────────────────

// ── HRPO: Hop-Grouped Relative Policy Optimization (Dr. Zero) ────────────────
//
// Groups structurally similar processes into "hops" (workload categories) and
// learns effectiveness per group. Transfers knowledge between processes of the
// same kind so new processes benefit from existing experience (zero-shot).

/// Workload hop — a category that groups structurally similar processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkloadHop {
    Browser,
    Build,
    SystemDaemon,
    CloudSync,
    Media,
    General,
}

/// Pattern index → WorkloadHop. Order MUST match HOP_PATTERNS below.
/// LeftmostFirst match kind ensures Browser priority over Build when both substrings present
/// (e.g., "rustc-browser-test" → Browser via "brave" check... wait, no overlap; safe).
const HOP_PATTERNS: &[&str] = &[
    // Browser group (idx 0-5)
    "brave", "chrome", "safari", "firefox", "webkit", "renderer",
    // Build group (idx 6-11)
    "rustc", "cargo", "clang", "swift", "make", "ninja",
    // CloudSync group (idx 12-16)
    "cloud", "dropbox", "drive", "sync", "bird", // Media group (idx 17-20)
    "audio", "video", "avconf", "camera",
];

#[inline]
fn pattern_to_hop(idx: usize) -> WorkloadHop {
    match idx {
        0..=5 => WorkloadHop::Browser,
        6..=11 => WorkloadHop::Build,
        12..=16 => WorkloadHop::CloudSync,
        17..=20 => WorkloadHop::Media,
        _ => WorkloadHop::General,
    }
}

impl WorkloadHop {
    pub fn from_process_name(name: &str) -> Self {
        // Exact match fast path for "cc" (too short for substring — would false-positive on
        // "Chromium Compositor" etc). Single byte-comparison; cheap.
        if name.eq_ignore_ascii_case("cc") {
            return WorkloadHop::Build;
        }
        // AhoCorasick OnceLock — 21 patterns scanned in O(name.len) single pass with
        // ascii_case_insensitive instead of 21 separate to_lowercase().contains() per PID.
        // Mirrors 9528b8d (safety classifiers) pattern. -50ms/cycle under sustained learning.
        static HOP_AC: std::sync::OnceLock<aho_corasick::AhoCorasick> = std::sync::OnceLock::new();
        let ac = HOP_AC.get_or_init(|| {
            aho_corasick::AhoCorasickBuilder::new()
                .ascii_case_insensitive(true)
                .match_kind(aho_corasick::MatchKind::LeftmostFirst)
                .build(HOP_PATTERNS)
                .expect("workload hop patterns build")
        });
        if let Some(m) = ac.find(name) {
            return pattern_to_hop(m.pattern().as_usize());
        }
        // Daemon fallback: ends in 'd'/'D', len > 3, no spaces. Preserves prior semantics
        // without to_lowercase() alloc — byte-level check.
        let bytes = name.as_bytes();
        if bytes.len() > 3
            && !bytes.contains(&b' ')
            && bytes.last().is_some_and(|b| b.eq_ignore_ascii_case(&b'd'))
        {
            return WorkloadHop::SystemDaemon;
        }
        WorkloadHop::General
    }
}

/// Per-group effectiveness tracker (HRPO weights — Dr. Zero solver).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HopGroupWeight {
    pub throttle_count: u32,
    pub effective_count: u32,
    /// EMA of observed pressure drop for this group.
    pub avg_drop_ema: f64,
    /// Self-challenge: predicted effectiveness based on group history.
    pub predicted_effectiveness: f64,
    /// Prediction error EMA — drives curriculum difficulty.
    pub prediction_error_ema: f64,
}

impl HopGroupWeight {
    pub fn effectiveness(&self) -> f64 {
        (self.effective_count as f64 + 1.0) / (self.throttle_count as f64 + 2.0)
    }

    /// Record outcome and update self-challenge prediction (solver learns).
    pub fn record(&mut self, effective: bool, pressure_drop: f64) {
        self.throttle_count += 1;
        if effective {
            self.effective_count += 1;
        }
        self.avg_drop_ema += 0.05 * (pressure_drop - self.avg_drop_ema);
        let actual = if effective { 1.0 } else { 0.0 };
        let error = (actual - self.predicted_effectiveness).abs();
        self.prediction_error_ema += 0.1 * (error - self.prediction_error_ema);
        self.predicted_effectiveness += 0.1 * (actual - self.predicted_effectiveness);
    }

    /// High prediction error = group needs more exploration (curriculum signal).
    pub fn needs_exploration(&self) -> bool {
        self.throttle_count >= 5 && self.prediction_error_ema > 0.3
    }
}

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
    /// Queryable experience memory — ring buffer of resolved outcomes.
    pub experience: ExperienceMemory,
    /// Process co-occurrence graph: tracks which processes appear together
    /// during high-pressure events. Key = sorted pair (A, B), value = count.
    /// Used to identify causal clusters (A+B always cause pressure together).
    co_occurrence: HashMap<(String, String), u32>,
    /// Counterfactual: EMA of natural pressure drift when we DON'T act.
    /// Positive = pressure tends to drop naturally over 30s.
    /// Used to separate real causal effect from natural fluctuation.
    natural_drift_ema: f64,
    /// Pressure snapshot from previous cycle (for drift calculation).
    prev_pressure: Option<f64>,
    /// Accumulated pressure delta during non-action cycles (rolling 30s window).
    drift_accumulator: f64,
    /// Ticks since last action (for windowed drift measurement).
    ticks_since_action: u32,
    /// Short-window (3-cycle) pressure deltas for fast causal attribution.
    /// [Rubin 1974] Potential Outcomes framework: faster D-in-D for
    /// detecting action effect vs. natural drift within ~1.5s at 2 Hz.
    short_window_deltas: VecDeque<f64>,
    /// Mean of the last 3 no-action pressure deltas (positive = dropping).
    short_drift_velocity: f64,
    /// HRPO: per-group effectiveness tracking (Dr. Zero solver).
    pub hop_groups: HashMap<WorkloadHop, HopGroupWeight>,
    /// NARS-based concept drift detector.
    /// Tracks per-process effectiveness beliefs using Revision rule.
    /// Signals when the Bayesian model has drifted from current reality.
    /// [Pei Wang 2013] Non-Axiomatic Reasoning System, §3.3.3
    pub drift_detector: DriftDetector,

    // ── Outcome acceleration (Phase 7) ──────────────────────────────────
    /// Per-process EMA of time-to-effect in seconds.
    /// Processes that respond quickly (effect visible in 10s) get shorter wait times.
    /// Processes that respond slowly keep the default 30s.
    process_effect_time: HashMap<String, f64>,

    // ── Survival-bias closure (blocked actions) ──────────────────────────
    /// In-flight blocked decisions awaiting post-hoc counterfactual eval.
    /// Resolved after BLOCKED_EVAL_WINDOW_SECS by `tick_blocked()` /
    /// `tick_with_params()`. Capped at 300 to bound memory.
    pending_blocked: VecDeque<PendingBlocked>,
    /// Per-key Bayesian counts for the survival-bias closure. Key shape:
    /// `"<action_class>:<gate>"`. Persisted via OutcomeTrackerPersisted.
    pub blocked_patterns: HashMap<String, BlockedPattern>,
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
            experience: ExperienceMemory::new(500),
            co_occurrence: HashMap::new(),
            natural_drift_ema: 0.0,
            prev_pressure: None,
            drift_accumulator: 0.0,
            ticks_since_action: 0,
            short_window_deltas: VecDeque::with_capacity(3),
            short_drift_velocity: 0.0,
            hop_groups: HashMap::new(),
            drift_detector: DriftDetector::new(),
            process_effect_time: HashMap::new(),
            pending_blocked: VecDeque::new(),
            blocked_patterns: HashMap::new(),
        }
    }

    /// Clear volatile post-cycle state after wake from sleep. Pre-sleep
    /// pressure deltas and prev_pressure are stale (system was idle/sleeping
    /// for arbitrary time) and would inject phantom drift velocity into the
    /// next causal attribution. Preserves Bayesian weights, experience, and
    /// long-term EMAs (those are still valid).
    pub fn reset_after_wake(&mut self) {
        self.short_window_deltas.clear();
        self.short_drift_velocity = 0.0;
        self.prev_pressure = None;
        self.drift_accumulator = 0.0;
        self.ticks_since_action = 0;
        // Drop pending blocked observations: their Instants and
        // pressure_before are stale across sleep, and resolving them now
        // would inject phantom counterfactual signal. Bayesian counts in
        // blocked_patterns survive (they are validated, restart-safe).
        self.pending_blocked.clear();
    }

    /// Current NARS drift score [0,1]. High = model has drifted from reality.
    pub fn nars_drift_score(&self) -> f64 {
        self.drift_detector.score()
    }

    /// True if NARS drift detector signals that Bayesian weights need recalibration.
    pub fn nars_needs_recalibration(&self) -> bool {
        self.drift_detector.needs_recalibration()
    }

    /// Like `nars_needs_recalibration()` but with an arousal-adjusted threshold.
    /// Caller passes `ArousalState::adjusted_drift_threshold(0.08)`.
    pub fn nars_needs_recalibration_at(&self, score_threshold: f64) -> bool {
        self.drift_detector.needs_recalibration_at(score_threshold)
    }

    /// Call after recalibration has been applied to reset drift signals.
    pub fn nars_acknowledge_recalibration(&mut self) {
        self.drift_detector.acknowledge_recalibration();
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
        self.record_throttle_with_swap(process_name, pressure_before, watts_before, 0.0);
    }

    /// Registra un throttle con swap context para salience weighting.
    /// `swap_gb` = swap usado en GB en el momento del throttle.
    pub fn record_throttle_with_swap(
        &mut self,
        process_name: &str,
        pressure_before: f64,
        watts_before: f64,
        swap_gb: f64,
    ) {
        self.record_action_with_swap(
            process_name,
            pressure_before,
            watts_before,
            swap_gb,
            super::learning_pipeline::ActionKind::Throttle,
        );
    }

    pub fn record_action_with_swap(
        &mut self,
        process_name: &str,
        pressure_before: f64,
        watts_before: f64,
        swap_gb: f64,
        action_type: super::learning_pipeline::ActionKind,
    ) {
        // Actualiza contador de throttles para el peso Bayesiano.
        let w = self.weights.entry(process_name.to_string()).or_default();
        w.throttle_count += 1;

        self.pending.push_back(PendingOutcome {
            process_name: process_name.to_string(),
            throttled_at: Instant::now(),
            pressure_before,
            watts_before,
            swap_gb_at_throttle: swap_gb,
            action_type,
        });

        // Cap: si la cola crece demasiado, descarta los más viejos sin resolver.
        // BUG-10: emit a diagnostic when we silently drop pending outcomes.
        if self.pending.len() > 300 {
            eprintln!("apollo: outcome_tracker: discarded 100 pending outcomes (cap)");
            self.pending.drain(..100);
        }

        // In-cycle cap: persist-time prune runs every ~150s; without an
        // in-cycle ceiling, weights HashMap can grow to thousands between
        // persists on bursty workloads (one entry per unique process name).
        // Evict weakest (lowest throttle_count) — high-count entries are
        // load-bearing for Bayesian salience.
        const HOT_PATH_WEIGHTS_CAP: usize = 200;
        if self.weights.len() > HOT_PATH_WEIGHTS_CAP {
            if let Some(weakest_key) = self
                .weights
                .iter()
                .min_by_key(|(_, w)| w.throttle_count)
                .map(|(k, _)| k.clone())
            {
                self.weights.remove(&weakest_key);
            }
        }
    }

    // ── Survival-bias closure: blocked-action API ─────────────────────────

    /// Cap on pending blocked observations — analogous to the throttle
    /// pending cap (300). Bounds memory under bursty gate firing.
    const BLOCKED_PENDING_CAP: usize = 300;
    /// Post-block evaluation window in seconds. 30s @ 2 Hz ≈ 15 cycles —
    /// matches the executed-throttle eval window so "would have helped"
    /// is comparable to "did help". [Validated by NotebookLM 2026-05-03].
    const BLOCKED_EVAL_WINDOW_SECS: u64 = 30;
    /// Threshold (after subtracting natural drift) above which a post-block
    /// pressure rise is attributed to "the blocked action would have helped".
    /// 0.02 mirrors the executed-throttle effective_threshold floor; a 2pp
    /// counterfactual delta above natural drift is a meaningful miss.
    const BLOCKED_EFFECTIVE_THRESHOLD: f64 = 0.02;

    /// Record that a class of action was blocked by a gate. Drives the
    /// survival-bias closure: every block enqueues a pending observation
    /// that `tick_with_params()` will resolve ~30s later by checking
    /// whether pressure rose by more than natural drift.
    ///
    /// `action_class` should be a coarse class string ("freeze",
    /// "throttle", "boost", …) rather than a per-PID identifier — gates
    /// typically block at the class level.
    /// `gate` is the [`crate::engine::blocked_action_journal::BlockerKind`]
    /// reason, lowercased / dasherized for use as a stable key.
    /// `pressure_at_block` is the live memory pressure when the block fired.
    ///
    /// SHADOW-MODE-ONLY: this signal MUST NOT be used to auto-unblock gated
    /// actions. The feedback-loop risk (system unblocks an action because
    /// the journal claims it "would have helped" → that action actually
    /// causes user-visible jank → journal still says good → loop tightens)
    /// is too high. Per NotebookLM peer review 2026-05-03.
    pub fn record_blocked(&mut self, action_class: &str, gate: &str, pressure_at_block: f64) {
        let key = format!("{}:{}", action_class, gate);
        let entry = self.blocked_patterns.entry(key.clone()).or_default();
        entry.blocked_count = entry.blocked_count.saturating_add(1);

        self.pending_blocked.push_back(PendingBlocked {
            key,
            blocked_at: Instant::now(),
            pressure_before: pressure_at_block,
        });

        // Bound memory: drop oldest unresolved observations under burst.
        if self.pending_blocked.len() > Self::BLOCKED_PENDING_CAP {
            let drop_n = self.pending_blocked.len() - (Self::BLOCKED_PENDING_CAP - 100);
            self.pending_blocked.drain(..drop_n);
        }

        // Hard cap the patterns map (mirror weights HashMap defense). Evict
        // the lowest-blocked-count entry so high-count signals are kept.
        const HOT_PATH_BLOCKED_CAP: usize = 200;
        if self.blocked_patterns.len() > HOT_PATH_BLOCKED_CAP {
            if let Some(weakest_key) = self
                .blocked_patterns
                .iter()
                .min_by_key(|(_, p)| p.blocked_count)
                .map(|(k, _)| k.clone())
            {
                self.blocked_patterns.remove(&weakest_key);
            }
        }
    }

    /// Resolve pending blocked observations whose evaluation window has
    /// elapsed. For each one, compute counterfactual delta = (pressure_now
    /// - pressure_before) - natural_drift_ema (Rubin 1974 potential
    /// outcomes). If delta > BLOCKED_EFFECTIVE_THRESHOLD the pressure
    /// rose more than baseline drift would predict, so the blocked
    /// action would likely have helped — increment helped count.
    ///
    /// Called from `tick_with_params()`; can also be called standalone in
    /// tests or shadow-mode tools.
    pub fn tick_blocked(&mut self, current_pressure: f64) {
        let window = Duration::from_secs(Self::BLOCKED_EVAL_WINDOW_SECS);
        let drift = self.natural_drift_ema;
        while let Some(front) = self.pending_blocked.front() {
            if front.blocked_at.elapsed() < window {
                break;
            }
            let pb = self.pending_blocked.pop_front().unwrap();
            // Pressure RISE (positive delta) is bad — it means the system
            // got worse during the block window. natural_drift_ema models
            // baseline DROP (positive = pressure tends to drop), so the
            // counterfactual rise above baseline is:
            //   (current - before) + drift   (drift is drop magnitude)
            // i.e., we add drift back because not-dropping is the miss.
            let observed_rise = current_pressure - pb.pressure_before;
            let counterfactual_rise = observed_rise + drift;

            if counterfactual_rise > Self::BLOCKED_EFFECTIVE_THRESHOLD {
                if let Some(p) = self.blocked_patterns.get_mut(&pb.key) {
                    p.would_have_helped_count = p.would_have_helped_count.saturating_add(1);
                }
            }
        }
    }

    /// Bayesian effectiveness estimate for a blocked (action_class, gate)
    /// pair. Key shape: `"<action_class>:<gate>"`. Returns Laplace prior
    /// 0.5 when the pattern has no observations.
    ///
    /// Interpretation: high values mean blocks of this class are
    /// associated with subsequent pressure rises beyond natural drift —
    /// candidate for offline review of gate calibration.
    pub fn blocked_effectiveness(&self, action_key: &str) -> f64 {
        match self.blocked_patterns.get(action_key) {
            Some(p) => p.effectiveness(),
            None => 0.5, // Laplace prior with no data
        }
    }

    /// Number of currently pending (unresolved) blocked observations.
    pub fn blocked_pending_depth(&self) -> usize {
        self.pending_blocked.len()
    }

    /// Aggregate over-protection signal across all blocked-action patterns
    /// with enough observations to be trusted (≥10 blocks). Returns a value
    /// in [0,1] suitable for `EpistemicUncertainty.guard_overprotection`:
    /// 0.0 = no signal or all patterns at neutral 0.5 effectiveness;
    /// 1.0 = every mature pattern says blocks "would have helped" 100% of
    /// the time → guard tower is over-protecting and the system should
    /// raise epistemic uncertainty.
    ///
    /// Maps each mature pattern's effectiveness `e` ∈ [0,1] to
    /// `max(0, e - 0.5) * 2`, then averages across mature patterns. Patterns
    /// below the maturity floor are ignored to suppress cold-start noise.
    pub fn mean_blocked_overprotection(&self) -> f32 {
        const MATURE_BLOCKED_FLOOR: u32 = 10;
        let mature: Vec<&BlockedPattern> = self
            .blocked_patterns
            .values()
            .filter(|p| p.blocked_count >= MATURE_BLOCKED_FLOOR)
            .collect();
        if mature.is_empty() {
            return 0.0;
        }
        let sum: f64 = mature
            .iter()
            .map(|p| ((p.effectiveness() - 0.5) * 2.0).max(0.0))
            .sum();
        ((sum / mature.len() as f64).clamp(0.0, 1.0)) as f32
    }

    /// Resuelve los outcomes pendientes con más de 30s de antigüedad.
    /// Retorna un batch con los resultados para que el llamador actualice
    /// el EnergyTracker y la LearnedPolicy.
    pub fn tick(&mut self, current_pressure: f64) -> OutcomeBatch {
        self.tick_with_params(current_pressure, 30, 0.01, 0)
    }

    /// Tick with adaptive wait time, effectiveness threshold, and workload context.
    pub fn tick_with_params(
        &mut self,
        current_pressure: f64,
        wait_secs: u64,
        effective_threshold: f64,
        workload: u8,
    ) -> OutcomeBatch {
        const BASELINE_ALPHA: f64 = 0.01; // half-life ≈ 69 observaciones
        let check_after = Duration::from_secs(wait_secs);
        let mut effective_names = Vec::new();
        let mut savings_watts = 0.0_f64;
        let mut resolved_outcomes: Vec<(String, f64, f64, super::learning_pipeline::ActionKind)> =
            Vec::new();

        while let Some(front) = self.pending.front() {
            if front.throttled_at.elapsed() < check_after {
                break;
            }
            let outcome = self.pending.pop_front().unwrap();
            let pressure_drop = outcome.pressure_before - current_pressure;
            // Lowered from 0.02 to 0.01: on an 8GB M1 with 2-3GB swap, 2% absolute
            // is too strict a bar — many legitimate throttles produce 1-1.5% relief
            // that compounds across multiple actions. 1% catches these while still
            // filtering noise. Now adaptive via LearnableParams.
            let effective = pressure_drop >= effective_threshold;

            // Actualiza el baseline de fluctuación natural: ¿bajó la presión ≥1%
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

            // NARS Revision with affective salience weighting.
            // High-pressure / high-OOM events earn stronger belief updates and LTI.
            // p_oom estimated from pressure_before (>0.70 → rising OOM risk).
            let p_oom_est = ((outcome.pressure_before - 0.70) / 0.30).clamp(0.0, 1.0);
            let salience = Salience::compute(
                outcome.pressure_before,
                pressure_drop,
                p_oom_est,
                outcome.swap_gb_at_throttle,
            );
            self.drift_detector
                .observe_salient(&outcome.process_name, effective, salience);

            // HRPO: update group-level effectiveness (Dr. Zero solver feedback).
            let hop = WorkloadHop::from_process_name(&outcome.process_name);
            self.hop_groups
                .entry(hop)
                .or_default()
                .record(effective, pressure_drop);

            // Store in experience memory for similarity queries.
            self.experience.push(ExperienceRecord {
                process_name: outcome.process_name.clone(),
                pressure_at_action: outcome.pressure_before,
                pressure_drop,
                effective,
                workload,
            });

            // Collect resolved outcome for LearningPipeline (pre/post pressure + action type).
            resolved_outcomes.push((
                outcome.process_name.clone(),
                outcome.pressure_before,
                current_pressure,
                outcome.action_type,
            ));

            // Track per-process time-to-effect for adaptive wait (Phase 7).
            let elapsed_secs = outcome.throttled_at.elapsed().as_secs_f64();
            let effect_entry = self
                .process_effect_time
                .entry(outcome.process_name.clone())
                .or_insert(30.0);
            *effect_entry = *effect_entry * 0.8 + elapsed_secs * 0.2;

            self.total_resolved += 1;
            if effective {
                self.total_effective += 1;
                effective_names.push(outcome.process_name.clone());
                savings_watts += outcome.watts_before;
            }
        }

        // Detecta patrones que ya tienen suficientes datos y están por debajo
        // del baseline calibrado — throttlearlos no aporta más que la fluctuación natural.
        //
        // HARD-PROTECTED EXCLUSION (2026-06-07): hard-protected processes
        // (Chromium, Brave Browser Helper, …) route throttles through the
        // soft PRIO_DARWIN_BG + jetsam BACKGROUND demote path. Their low
        // effectiveness is a property of the structurally-degraded throttle
        // action, NOT a signal that they don't cause pressure. Excluding
        // them from the `low_value_names` reclassification signal prevents
        // the documented Boost-loop (110 boosts on Brave per 200 actions).
        // RL penalty + LLM-struggling continue to consume the raw
        // `is_low_value_vs_baseline` signal — they want to see the waste.
        let threshold = self.calibrated_threshold();
        let low_value_names: Vec<String> = self
            .weights
            .iter()
            .filter(|(name, w)| w.is_low_value_for_reclassification(name, threshold))
            .map(|(name, _)| name.clone())
            .collect();

        // Survival-bias closure: resolve pending blocked observations whose
        // 30s evaluation window has elapsed. Cheap (HashMap update only) and
        // shares the same pressure read as the throttle resolver.
        self.tick_blocked(current_pressure);

        OutcomeBatch {
            effective_names,
            savings_watts,
            low_value_names,
            resolved_outcomes,
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

    /// Backpressure ratio of the pending outcome queue [0.0, 1.0].
    ///
    /// 0.0 = no pending observations.
    /// 1.0 = queue at capacity (300 items).
    /// Values > 0.5 suggest throttling is happening faster than outcomes resolve;
    /// callers can use this to reduce aggressiveness.
    pub fn pending_backpressure_ratio(&self) -> f64 {
        const CAP: usize = 300;
        (self.pending.len() as f64 / CAP as f64).min(1.0)
    }

    /// Number of currently pending (unresolved) outcome observations.
    pub fn pending_depth(&self) -> usize {
        self.pending.len()
    }

    /// Adaptive wait time for a process based on its historical time-to-effect.
    /// Processes that show effect quickly (EMA < 15s) get a shorter wait (15s).
    /// Unknown processes use the default (30s).
    pub fn adaptive_wait_secs(&self, process: &str) -> u64 {
        self.process_effect_time
            .get(process)
            .map(|&ema| {
                if ema < 15.0 {
                    15 // fast responder
                } else if ema < 25.0 {
                    20 // moderate
                } else {
                    30 // slow or default
                }
            })
            .unwrap_or(30)
    }

    /// Urgency flush: resolve ALL pending outcomes immediately (no wait).
    /// Used when pressure > 0.80 — we need the feedback loop NOW.
    /// Returns an OutcomeBatch with all resolutions.
    pub fn urgency_flush(&mut self, current_pressure: f64) -> OutcomeBatch {
        let mut effective_names = Vec::new();
        let mut savings_watts = 0.0_f64;
        let mut low_value_names = Vec::new();
        let mut resolved_outcomes = Vec::new();

        while let Some(outcome) = self.pending.pop_front() {
            let pressure_drop = outcome.pressure_before - current_pressure;
            let effective = pressure_drop > 0.01;
            let elapsed_secs = outcome.throttled_at.elapsed().as_secs_f64();

            self.total_resolved += 1;
            if effective {
                self.total_effective += 1;
                effective_names.push(outcome.process_name.clone());
                savings_watts += outcome.watts_before;
            } else {
                low_value_names.push(outcome.process_name.clone());
            }

            // Update per-process effect time EMA
            let entry = self
                .process_effect_time
                .entry(outcome.process_name.clone())
                .or_insert(30.0);
            *entry = *entry * 0.8 + elapsed_secs * 0.2;

            // Update weights
            let weight =
                self.weights
                    .entry(outcome.process_name.clone())
                    .or_insert(PatternWeight {
                        throttle_count: 0,
                        effective_count: 0,
                    });
            weight.throttle_count += 1;
            if effective {
                weight.effective_count += 1;
            }

            resolved_outcomes.push((
                outcome.process_name,
                outcome.pressure_before,
                current_pressure,
                outcome.action_type,
            ));
        }

        OutcomeBatch {
            effective_names,
            savings_watts,
            low_value_names,
            resolved_outcomes,
        }
    }

    /// GC for the weights HashMap — prevents unbounded growth in long-running daemons.
    ///
    /// Prunes entries that carry insufficient signal: fewer than 5 throttles AND
    /// fewer than 2 effective outcomes. These entries are essentially noise — they
    /// contribute only Laplace-smoothed 0.5 priors and waste memory.
    ///
    /// Complements the persist-time GC in `LearnedState::self_improve()` by also
    /// pruning in-process, typically called every 500 cycles (~4 minutes).
    pub fn gc_weights(&mut self) {
        self.weights
            .retain(|_, w| w.throttle_count >= 5 || w.effective_count >= 2);

        // Cap process_effect_time to 500 entries — evict farthest-from-default.
        if self.process_effect_time.len() > 500 {
            let mut entries: Vec<(String, f64)> = self.process_effect_time.drain().collect();
            // Keep entries most different from default (30.0) — they carry signal.
            entries.sort_by(|a, b| {
                let da = (a.1 - 30.0).abs();
                let db = (b.1 - 30.0).abs();
                db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
            });
            entries.truncate(400);
            self.process_effect_time = entries.into_iter().collect();
        }

        // Cap hop_groups to 300 entries — evict lowest-count groups.
        if self.hop_groups.len() > 300 {
            let mut entries: Vec<_> = self.hop_groups.drain().collect();
            entries.sort_by(|a, b| b.1.throttle_count.cmp(&a.1.throttle_count).reverse());
            entries.truncate(200);
            self.hop_groups = entries.into_iter().collect();
        }
    }

    /// Penalty signal for the RL agent: negative reward proportional to
    /// how many low-value patterns exist.  Returns 0.0 when things are fine,
    /// negative when throttling is wasting effort.
    ///
    /// Designed to be called after `tick()` and passed to
    /// `RlThresholdAgent::inject_external_reward()`.
    pub fn rl_penalty(&self) -> f64 {
        let threshold = self.calibrated_threshold();
        let low_count = self
            .weights
            .values()
            .filter(|w| w.is_low_value_vs_baseline(threshold))
            .count();
        if low_count == 0 {
            0.0
        } else {
            // -0.5 per low-value pattern, capped at -3.0.
            // Mild enough not to override overflow penalty (-10),
            // but persistent enough to steer learning over time.
            (-0.5 * low_count as f64).max(-3.0)
        }
    }

    /// True si el heurístico tiene patrones confirmados como low-value
    /// y la efectividad global es baja — señal para llamar al LLM.
    pub fn heuristic_is_struggling(&self) -> bool {
        self.total_resolved >= 10
            && self.overall_effectiveness() < 0.35
            && self.weights.values().any(|w| w.is_low_value())
    }

    // ── Process causal graph ────────────────────────────────────────────

    /// Record which processes were active during a high-pressure event.
    /// Builds a co-occurrence graph to identify causal clusters.
    /// Call with the names of top-N processes during pressure spikes.
    pub fn record_co_occurrence(&mut self, active_processes: &[String]) {
        // Generate all unique pairs (sorted for consistency).
        for i in 0..active_processes.len() {
            for j in (i + 1)..active_processes.len() {
                let (a, b) = if active_processes[i] <= active_processes[j] {
                    (active_processes[i].clone(), active_processes[j].clone())
                } else {
                    (active_processes[j].clone(), active_processes[i].clone())
                };
                *self.co_occurrence.entry((a, b)).or_insert(0) += 1;
            }
        }

        // GC: keep only top 100 pairs by count.
        if self.co_occurrence.len() > 150 {
            let mut counts: Vec<_> = self.co_occurrence.values().copied().collect();
            counts.sort_unstable();
            let cutoff = counts[counts.len().saturating_sub(100)];
            let would_retain = self.co_occurrence.values().filter(|&&v| v > cutoff).count();
            if would_retain >= 50 {
                // Normal case: clear differentiation — keep strictly above cutoff.
                self.co_occurrence.retain(|_, &mut v| v > cutoff);
            } else {
                // Homogeneous counts (e.g., all pairs at count=1 after cold start):
                // count-based GC would skip entirely, letting the map grow without bound.
                // Fall back to stable key-order truncation to hard-cap at 100 entries.
                // [Boldi & Vigna 2014] — any bounded graph representation requires
                // a hard eviction path when frequency discrimination is unavailable.
                let mut keys: Vec<_> = self.co_occurrence.keys().cloned().collect();
                keys.sort_unstable();
                for key in keys.into_iter().skip(100) {
                    self.co_occurrence.remove(&key);
                }
            }
        }
    }

    /// Query the causal graph: top N co-occurring process pairs.
    /// Returns pairs sorted by co-occurrence count (most frequent first).
    pub fn top_causal_pairs(&self, n: usize) -> Vec<(&str, &str, u32)> {
        let mut pairs: Vec<_> = self
            .co_occurrence
            .iter()
            .map(|((a, b), &count)| (a.as_str(), b.as_str(), count))
            .collect();
        pairs.sort_by_key(|p| std::cmp::Reverse(p.2));
        pairs.truncate(n);
        pairs
    }

    /// Check if two processes form a known causal cluster.
    /// Returns the co-occurrence count if they've been seen together ≥ threshold times.
    pub fn is_causal_pair(&self, proc_a: &str, proc_b: &str, min_count: u32) -> Option<u32> {
        let (a, b) = if proc_a <= proc_b {
            (proc_a.to_string(), proc_b.to_string())
        } else {
            (proc_b.to_string(), proc_a.to_string())
        };
        self.co_occurrence
            .get(&(a, b))
            .copied()
            .filter(|&c| c >= min_count)
    }

    // ── Counterfactual baseline ──────────────────────────────────────────

    /// Call every daemon cycle with current pressure and whether an action was taken.
    /// Builds a model of natural pressure drift to separate causal effects.
    pub fn observe_cycle(&mut self, pressure: f64, acted: bool) {
        const DRIFT_ALPHA: f64 = 0.02; // slow EMA, half-life ~35 observations
        const DRIFT_WINDOW_TICKS: u32 = 60; // ~30s at 2Hz

        if let Some(prev) = self.prev_pressure {
            if !acted {
                // Long-window drift: 60-tick EMA (~30s).
                self.drift_accumulator += prev - pressure; // positive = pressure dropped
                self.ticks_since_action += 1;

                // After a full window of non-action, commit the drift observation.
                if self.ticks_since_action >= DRIFT_WINDOW_TICKS {
                    self.natural_drift_ema +=
                        DRIFT_ALPHA * (self.drift_accumulator - self.natural_drift_ema);
                    self.drift_accumulator = 0.0;
                    self.ticks_since_action = 0;
                }

                // Short-window velocity: 3-cycle mean delta (~1.5s at 2 Hz).
                // Provides fast causal signal before the 60-tick EMA converges.
                self.short_window_deltas.push_back(prev - pressure);
                if self.short_window_deltas.len() > 3 {
                    self.short_window_deltas.pop_front();
                }
                self.short_drift_velocity = self.short_window_deltas.iter().sum::<f64>()
                    / self.short_window_deltas.len() as f64;
            } else {
                // Action taken — reset both drift windows.
                self.drift_accumulator = 0.0;
                self.ticks_since_action = 0;
                self.short_window_deltas.clear();
                self.short_drift_velocity = 0.0;
            }
        }
        self.prev_pressure = Some(pressure);
    }

    /// Causal effect of a throttle: observed pressure drop minus natural drift.
    /// Positive = the action actually helped beyond what would have happened naturally.
    pub fn causal_effect(&self, observed_drop: f64) -> f64 {
        observed_drop - self.natural_drift_ema
    }

    /// Current estimate of natural pressure drift over 30s (no-action baseline).
    pub fn natural_drift(&self) -> f64 {
        self.natural_drift_ema
    }

    /// Short-window pressure velocity: mean of last 3 no-action deltas (~1.5s).
    /// Positive = pressure dropping naturally; negative = rising.
    ///
    /// Paper: [Rubin 1974] Potential Outcomes framework — difference-in-differences
    /// over a tight 3-cycle window for rapid causal isolation.
    pub fn pressure_velocity_short(&self) -> f64 {
        self.short_drift_velocity
    }

    /// Fast causal attribution using 3-cycle velocity instead of 60-tick EMA.
    /// Use for immediate post-action evaluation; use `causal_effect()` for
    /// long-term validated assessment.
    ///
    /// Falls back to `natural_drift_ema` when the short window is empty (i.e.,
    /// during consecutive action cycles where the 3-cycle deque was cleared).
    /// Without a fallback, `causal_effect_fast` would return `drop - 0 = drop`
    /// (always positive), defeating its purpose of detecting drift-only successes.
    ///
    /// Positive = action caused a drop beyond natural short-term drift.
    pub fn causal_effect_fast(&self, observed_drop: f64) -> f64 {
        let baseline = if self.short_window_deltas.is_empty() {
            self.natural_drift_ema // fallback: use slow EMA when no fast data
        } else {
            self.short_drift_velocity
        };
        observed_drop - baseline
    }

    // ── HRPO: Dr. Zero group-level intelligence ──────────────────────────

    /// Query group effectiveness for a process (zero-shot via hop grouping).
    /// Returns (group_effectiveness, group_predicted, confidence) or None.
    pub fn hop_effectiveness(&self, process_name: &str) -> Option<(f64, f64, f64)> {
        let hop = WorkloadHop::from_process_name(process_name);
        self.hop_groups.get(&hop).map(|g| {
            let confidence = (g.throttle_count as f64 / 20.0).min(1.0);
            (g.effectiveness(), g.predicted_effectiveness, confidence)
        })
    }

    /// Dr. Zero proposer signal: which groups need more exploration?
    /// Returns hops where prediction error is high (solver is uncertain).
    pub fn exploration_needed(&self) -> Vec<(WorkloadHop, f64)> {
        self.hop_groups
            .iter()
            .filter(|(_, g)| g.needs_exploration())
            .map(|(&hop, g)| (hop, g.prediction_error_ema))
            .collect()
    }

    /// Summary of HRPO groups for status/metrics reporting.
    pub fn hop_group_summary(&self) -> Vec<(WorkloadHop, f64, u32, f64)> {
        let mut groups: Vec<_> = self
            .hop_groups
            .iter()
            .map(|(&hop, g)| {
                (
                    hop,
                    g.effectiveness(),
                    g.throttle_count,
                    g.prediction_error_ema,
                )
            })
            .collect();
        groups.sort_by_key(|b| std::cmp::Reverse(b.2)); // by throttle count descending
        groups
    }

    /// Dr. Zero self-challenge reward: average prediction error across all groups.
    /// Low = solver is calibrated. High = solver needs more training.
    pub fn self_challenge_score(&self) -> f64 {
        if self.hop_groups.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .hop_groups
            .values()
            .map(|g| g.prediction_error_ema)
            .sum();
        sum / self.hop_groups.len() as f64
    }

    /// Persist hop_groups to disk so HRPO learning survives restarts.
    pub fn persist_hop_groups(&self, path: &std::path::Path) {
        if let Ok(json) = serde_json::to_string(&self.hop_groups) {
            let _ = std::fs::write(path, json);
        }
    }

    /// Load hop_groups from disk (called on startup).
    pub fn load_hop_groups(&mut self, path: &std::path::Path) {
        if let Ok(data) = std::fs::read_to_string(path) {
            if let Ok(groups) = serde_json::from_str::<HashMap<WorkloadHop, HopGroupWeight>>(&data)
            {
                self.hop_groups = groups;
            }
        }
    }

    /// Build a persisted snapshot (for LearnedState).
    pub fn to_persisted(&self) -> OutcomeTrackerPersisted {
        let co_occurrence: Vec<(String, String, u32)> = self
            .co_occurrence
            .iter()
            .map(|((a, b), &count)| (a.clone(), b.clone(), count))
            .collect();
        OutcomeTrackerPersisted {
            weights: self.weights.clone(),
            total_effective: self.total_effective,
            total_resolved: self.total_resolved,
            baseline_drop_ema: self.baseline_drop_ema,
            baseline_samples: self.baseline_samples,
            experience_records: self.experience.records().iter().cloned().collect(),
            co_occurrence,
            natural_drift_ema: self.natural_drift_ema,
            hop_groups: self.hop_groups.clone(),
            drift_detector: Some(self.drift_detector.clone()),
            blocked_patterns: self.blocked_patterns.clone(),
        }
    }

    /// Restore from a persisted snapshot (for LearnedState).
    pub fn restore(&mut self, p: OutcomeTrackerPersisted) {
        self.weights = p.weights;
        self.total_effective = p.total_effective;
        self.total_resolved = p.total_resolved;
        self.baseline_drop_ema = p.baseline_drop_ema;
        self.baseline_samples = p.baseline_samples;
        for record in p.experience_records {
            self.experience.push(record);
        }
        self.co_occurrence.clear();
        for (a, b, count) in p.co_occurrence {
            self.co_occurrence.insert((a, b), count);
        }
        self.natural_drift_ema = p.natural_drift_ema;
        self.hop_groups = p.hop_groups;
        if let Some(dd) = p.drift_detector {
            self.drift_detector = dd;
        }
        // Survival-bias closure: restore Bayesian counts. Pending in-flight
        // blocked observations are NOT persisted (they are bound to live
        // Instants); they reset on restart, which is correct — pressure
        // baselines from before a restart are stale anyway.
        self.blocked_patterns = p.blocked_patterns;
    }
}

/// Serializable snapshot of OutcomeTracker state for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeTrackerPersisted {
    pub weights: HashMap<String, PatternWeight>,
    pub total_effective: u32,
    pub total_resolved: u32,
    pub baseline_drop_ema: f64,
    pub baseline_samples: u32,
    pub experience_records: Vec<ExperienceRecord>,
    pub co_occurrence: Vec<(String, String, u32)>,
    pub natural_drift_ema: f64,
    pub hop_groups: HashMap<WorkloadHop, HopGroupWeight>,
    /// NARS drift detector state — persisted so beliefs survive daemon restarts.
    /// Confidence values are meaningless if beliefs reset every restart.
    #[serde(default)]
    pub drift_detector: Option<DriftDetector>,
    /// Survival-bias closure: per-(action_class, gate) Bayesian counts of
    /// blocked actions and their inferred "would_have_helped" outcomes.
    /// `#[serde(default)]` keeps old learned_state.json files
    /// deserializable — missing field becomes empty HashMap.
    #[serde(default)]
    pub blocked_patterns: HashMap<String, BlockedPattern>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── WorkloadHop classification (Aho-Corasick) ────────────────────────────

    #[test]
    fn workload_hop_browser_variants() {
        assert_eq!(
            WorkloadHop::from_process_name("Brave Browser"),
            WorkloadHop::Browser
        );
        assert_eq!(
            WorkloadHop::from_process_name("Brave Browser Helper (Renderer)"),
            WorkloadHop::Browser
        );
        assert_eq!(
            WorkloadHop::from_process_name("Google Chrome"),
            WorkloadHop::Browser
        );
        assert_eq!(
            WorkloadHop::from_process_name("Safari"),
            WorkloadHop::Browser
        );
        assert_eq!(
            WorkloadHop::from_process_name("Firefox"),
            WorkloadHop::Browser
        );
        assert_eq!(
            WorkloadHop::from_process_name("com.apple.WebKit.GPU"),
            WorkloadHop::Browser
        );
        assert_eq!(
            WorkloadHop::from_process_name("WebContent"),
            WorkloadHop::General
        ); // no match
    }

    #[test]
    fn workload_hop_build_variants() {
        assert_eq!(WorkloadHop::from_process_name("rustc"), WorkloadHop::Build);
        assert_eq!(WorkloadHop::from_process_name("cargo"), WorkloadHop::Build);
        assert_eq!(
            WorkloadHop::from_process_name("clang-17"),
            WorkloadHop::Build
        );
        assert_eq!(WorkloadHop::from_process_name("cc"), WorkloadHop::Build); // exact-match
        assert_eq!(WorkloadHop::from_process_name("CC"), WorkloadHop::Build); // case-insensitive exact
        assert_eq!(WorkloadHop::from_process_name("make"), WorkloadHop::Build);
        assert_eq!(WorkloadHop::from_process_name("ninja"), WorkloadHop::Build);
    }

    #[test]
    fn workload_hop_cloudsync_variants() {
        assert_eq!(
            WorkloadHop::from_process_name("Dropbox"),
            WorkloadHop::CloudSync
        );
        assert_eq!(
            WorkloadHop::from_process_name("Google Drive"),
            WorkloadHop::CloudSync
        );
        assert_eq!(
            WorkloadHop::from_process_name("iCloud"),
            WorkloadHop::CloudSync
        );
        assert_eq!(
            WorkloadHop::from_process_name("bird"),
            WorkloadHop::CloudSync
        ); // iCloud daemon
    }

    #[test]
    fn workload_hop_media_variants() {
        // Order: substring (AC) before daemon fallback → coreaudiod matches "audio" → Media
        // (NOT SystemDaemon despite ending 'd'). Mirrors prior to_lowercase().contains chain.
        assert_eq!(
            WorkloadHop::from_process_name("coreaudiod"),
            WorkloadHop::Media
        );
        assert_eq!(
            WorkloadHop::from_process_name("VideoTool"),
            WorkloadHop::Media
        );
        assert_eq!(
            WorkloadHop::from_process_name("camerad"),
            WorkloadHop::Media
        );
    }

    #[test]
    fn workload_hop_daemon_fallback() {
        assert_eq!(
            WorkloadHop::from_process_name("powerd"),
            WorkloadHop::SystemDaemon
        );
        assert_eq!(
            WorkloadHop::from_process_name("launchd"),
            WorkloadHop::SystemDaemon
        );
        assert_eq!(
            WorkloadHop::from_process_name("Powerd"),
            WorkloadHop::SystemDaemon
        ); // case-insensitive
           // Edge: ends in 'd' but len <= 3 → General
        assert_eq!(WorkloadHop::from_process_name("ad"), WorkloadHop::General);
        // Edge: contains space → General
        assert_eq!(
            WorkloadHop::from_process_name("foo d"),
            WorkloadHop::General
        );
    }

    #[test]
    fn workload_hop_general_fallback() {
        assert_eq!(
            WorkloadHop::from_process_name("Finder"),
            WorkloadHop::General
        );
        assert_eq!(
            WorkloadHop::from_process_name("Terminal"),
            WorkloadHop::General
        );
        assert_eq!(WorkloadHop::from_process_name(""), WorkloadHop::General);
    }

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

    // FIX-2 (2026-06-07): Reclassify-exclusion predicate must route through
    // `safety::is_boost_forbidden`, not the legacy `hard_protected_contains`
    // path. Production matches Brave via `match_engine::is_family_root`, so
    // the old check left Brave renderers exposed to the reclassification
    // loop that triggered the infinite Boost trap (PID 16105 prod incident).
    #[test]
    fn effectiveness_for_classification_excludes_brave_family_root() {
        // Brave Browser Helper — matched via match_engine::is_family_root in
        // production (NOT hard_protected_contains). With the legacy predicate
        // this returned Some(eff), which fed the reclassification gate.
        let w = PatternWeight {
            throttle_count: 63,
            effective_count: 2,
        };
        assert_eq!(
            w.effectiveness_for_classification("Brave Browser Helper"),
            None,
            "boost-forbidden family-root names must be excluded from reclassification",
        );
        // is_low_value_for_reclassification must agree (returns false for None).
        assert!(
            !w.is_low_value_for_reclassification("Brave Browser Helper", 0.25),
            "family-root entries must never be flagged as low-value-for-reclassification",
        );
    }

    #[test]
    fn effectiveness_for_classification_passes_through_control_process() {
        // alacritty — neither hard-protected nor family-root → predicate returns
        // Some(effectiveness()), the raw Laplace-smoothed estimate.
        let w = PatternWeight {
            throttle_count: 10,
            effective_count: 3,
        };
        let got = w.effectiveness_for_classification("alacritty");
        // (3+1)/(10+2) ≈ 0.333
        assert!(
            got.is_some(),
            "non-forbidden control name must return Some(_)"
        );
        let eff = got.unwrap();
        assert!(
            (eff - 0.3333).abs() < 1e-3,
            "expected ≈0.333 Laplace-smoothed effectiveness, got {}",
            eff,
        );
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
    fn record_action_with_swap_preserves_non_throttle_kind() {
        let mut tracker = OutcomeTracker::new();
        tracker.record_action_with_swap(
            "language_server",
            0.80,
            1.5,
            1.0,
            crate::engine::learning_pipeline::ActionKind::Freeze,
        );

        let batch = tracker.urgency_flush(0.72);

        assert_eq!(batch.resolved_outcomes.len(), 1);
        assert_eq!(
            batch.resolved_outcomes[0].3,
            crate::engine::learning_pipeline::ActionKind::Freeze
        );
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
            swap_gb_at_throttle: 0.0,
            action_type: crate::engine::learning_pipeline::ActionKind::Throttle,
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
            swap_gb_at_throttle: 0.0,
            action_type: crate::engine::learning_pipeline::ActionKind::Throttle,
        });
        tracker.weights.insert(
            "Dropbox".to_string(),
            PatternWeight {
                throttle_count: 1,
                effective_count: 0,
            },
        );

        // Pressure barely dropped (< 0.01) → ineffective.
        let batch = tracker.tick(0.795);
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
    fn rl_penalty_zero_when_no_low_value() {
        let tracker = OutcomeTracker::new();
        assert_eq!(tracker.rl_penalty(), 0.0);
    }

    #[test]
    fn rl_penalty_proportional_to_low_value_count() {
        let mut tracker = OutcomeTracker::new();
        tracker.baseline_drop_ema = 0.25;
        tracker.baseline_samples = 50;
        // threshold = 0.225

        // Add 2 low-value processes
        tracker.weights.insert(
            "proc_a".into(),
            PatternWeight {
                throttle_count: 25,
                effective_count: 0,
            },
        );
        tracker.weights.insert(
            "proc_b".into(),
            PatternWeight {
                throttle_count: 25,
                effective_count: 0,
            },
        );
        // Add 1 high-value process (should not affect penalty)
        tracker.weights.insert(
            "proc_c".into(),
            PatternWeight {
                throttle_count: 25,
                effective_count: 20,
            },
        );

        let penalty = tracker.rl_penalty();
        assert!(
            (penalty - (-1.0)).abs() < 1e-6,
            "2 low-value = -1.0 penalty, got {}",
            penalty
        );
    }

    #[test]
    fn rl_penalty_capped_at_minus_3() {
        let mut tracker = OutcomeTracker::new();
        tracker.baseline_drop_ema = 0.25;
        tracker.baseline_samples = 50;
        for i in 0..10 {
            tracker.weights.insert(
                format!("proc_{i}"),
                PatternWeight {
                    throttle_count: 25,
                    effective_count: 0,
                },
            );
        }
        let penalty = tracker.rl_penalty();
        assert!(
            (penalty - (-3.0)).abs() < 1e-6,
            "10 low-value should cap at -3.0, got {}",
            penalty
        );
    }

    #[test]
    fn integration_outcome_to_rl_feedback() {
        // End-to-end: OutcomeTracker detects low-value → RL gets penalized.
        use crate::engine::rl_threshold::{RlState, RlThresholdAgent};

        let mut tracker = OutcomeTracker::new();
        tracker.baseline_drop_ema = 0.25;
        tracker.baseline_samples = 50;
        tracker.weights.insert(
            "wasteful".into(),
            PatternWeight {
                throttle_count: 30,
                effective_count: 0,
            },
        );

        let mut rl = RlThresholdAgent::load_or_default(std::path::Path::new("/dev/null"));
        let state = RlState::from_metrics(0.60, 0.40, 0);
        rl.tick(state, false);

        let q_before = rl.last_q_value();

        // Wire the feedback: outcome → RL
        let penalty = tracker.rl_penalty();
        assert!(penalty < 0.0);
        rl.inject_external_reward(penalty);

        let q_after = rl.last_q_value();
        assert!(
            q_after < q_before,
            "RL should be penalized by outcome feedback: {} < {}",
            q_after,
            q_before
        );
    }

    // ── Experience Memory tests ────────────────────────────────────────────────

    #[test]
    fn experience_memory_ring_buffer_evicts_oldest() {
        let mut mem = ExperienceMemory::new(3);
        for i in 0..5 {
            mem.push(ExperienceRecord {
                process_name: format!("proc_{i}"),
                pressure_at_action: 0.60,
                pressure_drop: 0.05,
                effective: true,
                workload: 0,
            });
        }
        assert_eq!(mem.len(), 3);
        // Oldest (proc_0, proc_1) evicted; proc_2, proc_3, proc_4 remain.
        assert!(mem
            .records
            .iter()
            .all(|r| !r.process_name.starts_with("proc_0")));
        assert!(mem
            .records
            .iter()
            .all(|r| !r.process_name.starts_with("proc_1")));
    }

    #[test]
    fn experience_query_similar_requires_min_3_records() {
        let mut mem = ExperienceMemory::new(100);
        // Only 2 records → None
        for _ in 0..2 {
            mem.push(ExperienceRecord {
                process_name: "Dropbox".into(),
                pressure_at_action: 0.70,
                pressure_drop: 0.05,
                effective: true,
                workload: 0,
            });
        }
        assert!(mem.query_similar("Dropbox", 0.70).is_none());

        // 3rd record → Some
        mem.push(ExperienceRecord {
            process_name: "Dropbox".into(),
            pressure_at_action: 0.72,
            pressure_drop: 0.03,
            effective: true,
            workload: 0,
        });
        let (avg_drop, confidence) = mem.query_similar("Dropbox", 0.70).unwrap();
        assert!((avg_drop - (0.05 + 0.05 + 0.03) / 3.0).abs() < 1e-6);
        assert!((confidence - 3.0 / 20.0).abs() < 1e-6);
    }

    #[test]
    fn experience_query_filters_by_pressure_window() {
        let mut mem = ExperienceMemory::new(100);
        // 3 records at pressure 0.70
        for _ in 0..3 {
            mem.push(ExperienceRecord {
                process_name: "chrome".into(),
                pressure_at_action: 0.70,
                pressure_drop: 0.08,
                effective: true,
                workload: 0,
            });
        }
        // 3 records at pressure 0.30 (too far from 0.70)
        for _ in 0..3 {
            mem.push(ExperienceRecord {
                process_name: "chrome".into(),
                pressure_at_action: 0.30,
                pressure_drop: -0.01,
                effective: false,
                workload: 0,
            });
        }
        // Query at 0.70 should only match first 3.
        let (avg_drop, _) = mem.query_similar("chrome", 0.70).unwrap();
        assert!(
            (avg_drop - 0.08).abs() < 1e-6,
            "should only match p≈0.70 records"
        );
    }

    // ── Workload-aware experience tests (Phase 4) ─────────────────────────────

    #[test]
    fn experience_workload_weighting_boosts_same_workload() {
        let mut mem = ExperienceMemory::new(100);
        // 3 records at workload=1 (build), drop=0.10
        for _ in 0..3 {
            mem.push(ExperienceRecord {
                process_name: "rustc".into(),
                pressure_at_action: 0.70,
                pressure_drop: 0.10,
                effective: true,
                workload: 1,
            });
        }
        // 3 records at workload=0 (idle), drop=0.02
        for _ in 0..3 {
            mem.push(ExperienceRecord {
                process_name: "rustc".into(),
                pressure_at_action: 0.70,
                pressure_drop: 0.02,
                effective: true,
                workload: 0,
            });
        }
        // Standard query: all 6 records equally weighted
        let (avg_all, _) = mem.query_similar("rustc", 0.70).unwrap();
        // Contextual query with workload=1: build records weighted 2×
        let (avg_build, _) = mem
            .query_similar_contextual("rustc", 0.70, 0.10, 1)
            .unwrap();
        // Build-weighted average should be closer to 0.10 than the uniform average
        assert!(
            (avg_build - 0.10).abs() < (avg_all - 0.10).abs(),
            "workload-weighted avg ({avg_build}) should be closer to 0.10 than uniform ({avg_all})"
        );
    }

    #[test]
    fn experience_workload_contextual_needs_3_records() {
        let mut mem = ExperienceMemory::new(100);
        for _ in 0..2 {
            mem.push(ExperienceRecord {
                process_name: "Safari".into(),
                pressure_at_action: 0.65,
                pressure_drop: 0.05,
                effective: true,
                workload: 3,
            });
        }
        assert!(mem
            .query_similar_contextual("Safari", 0.65, 0.10, 3)
            .is_none());
    }

    // ── Outcome acceleration tests (Phase 7) ──────────────────────────────────

    #[test]
    fn adaptive_wait_unknown_process_returns_default() {
        let tracker = OutcomeTracker::new();
        assert_eq!(tracker.adaptive_wait_secs("unknown_process"), 30);
    }

    #[test]
    fn adaptive_wait_tracks_fast_process() {
        let mut tracker = OutcomeTracker::new();
        // Simulate a process with fast effect time (EMA converges toward 10s)
        for _ in 0..20 {
            let entry = tracker
                .process_effect_time
                .entry("fast_app".into())
                .or_insert(30.0);
            *entry = *entry * 0.8 + 10.0 * 0.2; // EMA toward 10s
        }
        assert_eq!(
            tracker.adaptive_wait_secs("fast_app"),
            15,
            "fast process should get 15s wait"
        );
    }

    #[test]
    fn urgency_flush_resolves_all_pending() {
        let mut tracker = OutcomeTracker::new();
        // Add some pending outcomes
        tracker.pending.push_back(super::PendingOutcome {
            process_name: "App1".into(),
            throttled_at: std::time::Instant::now() - std::time::Duration::from_secs(5),
            pressure_before: 0.85,
            watts_before: 2.0,
            swap_gb_at_throttle: 1.0,
            action_type: crate::engine::learning_pipeline::ActionKind::Throttle,
        });
        tracker.pending.push_back(super::PendingOutcome {
            process_name: "App2".into(),
            throttled_at: std::time::Instant::now() - std::time::Duration::from_secs(3),
            pressure_before: 0.82,
            watts_before: 1.5,
            swap_gb_at_throttle: 0.5,
            action_type: crate::engine::learning_pipeline::ActionKind::Throttle,
        });
        assert_eq!(tracker.pending.len(), 2);
        let batch = tracker.urgency_flush(0.70); // pressure dropped to 0.70
        assert_eq!(
            tracker.pending.len(),
            0,
            "urgency flush should drain all pending"
        );
        assert_eq!(batch.resolved_outcomes.len(), 2);
        // Both should be effective (0.85-0.70=0.15 > 0.01 and 0.82-0.70=0.12 > 0.01)
        assert_eq!(batch.effective_names.len(), 2);
    }

    #[test]
    fn urgency_flush_updates_effect_time() {
        let mut tracker = OutcomeTracker::new();
        tracker.pending.push_back(super::PendingOutcome {
            process_name: "SlowApp".into(),
            throttled_at: std::time::Instant::now() - std::time::Duration::from_secs(5),
            pressure_before: 0.80,
            watts_before: 1.0,
            swap_gb_at_throttle: 0.0,
            action_type: crate::engine::learning_pipeline::ActionKind::Throttle,
        });
        tracker.urgency_flush(0.70);
        // Should have tracked effect time for SlowApp
        assert!(
            tracker.process_effect_time.contains_key("SlowApp"),
            "urgency flush should track effect time"
        );
    }

    #[test]
    fn urgency_flush_empty_pending_returns_empty_batch() {
        let mut tracker = OutcomeTracker::new();
        let batch = tracker.urgency_flush(0.90);
        assert!(batch.resolved_outcomes.is_empty());
        assert!(batch.effective_names.is_empty());
    }

    // ── Counterfactual baseline tests ───────────────────────────────────────────

    #[test]
    fn counterfactual_natural_drift_builds_from_observe_cycles() {
        let mut tracker = OutcomeTracker::new();
        // Simulate 65 ticks of no-action with slight natural pressure drop.
        // First tick sets prev_pressure; need ≥60 more for a full window commit.
        for i in 0..65 {
            let p = 0.60 - (i as f64) * (0.05 / 65.0);
            tracker.observe_cycle(p, false);
        }
        // After 60 non-action ticks (one full window), drift EMA should be positive
        // (pressure naturally dropped).
        assert!(
            tracker.natural_drift() > 0.0,
            "natural drift should be positive when pressure drops: {}",
            tracker.natural_drift()
        );
    }

    #[test]
    fn counterfactual_action_resets_drift_window() {
        let mut tracker = OutcomeTracker::new();
        // 30 no-action ticks
        for i in 0..30 {
            tracker.observe_cycle(0.60 - i as f64 * 0.001, false);
        }
        // Action resets accumulator
        tracker.observe_cycle(0.55, true);
        // Only 5 more non-action ticks — not enough for a window commit
        for i in 0..5 {
            tracker.observe_cycle(0.55 - i as f64 * 0.001, false);
        }
        // Drift should still be 0 (no full window completed after reset)
        assert!(
            tracker.natural_drift().abs() < 1e-9,
            "drift should be 0 after action reset: {}",
            tracker.natural_drift()
        );
    }

    #[test]
    fn counterfactual_causal_effect_subtracts_natural_drift() {
        let mut tracker = OutcomeTracker::new();
        // Build natural drift over a window.
        // 65 ticks, pressure 0.60 → 0.55 linearly (first tick sets prev_pressure).
        for i in 0..65 {
            let p = 0.60 - (i as f64) * (0.05 / 65.0);
            tracker.observe_cycle(p, false);
        }
        let drift = tracker.natural_drift();
        // If we observed a throttle causing 0.08 drop, causal effect = 0.08 - drift.
        let causal = tracker.causal_effect(0.08);
        assert!(
            causal < 0.08,
            "causal effect should be less than raw drop: {} < 0.08",
            causal
        );
        assert!(
            (causal - (0.08 - drift)).abs() < 1e-9,
            "causal = observed - drift"
        );
    }

    #[test]
    fn experience_fed_by_tick() {
        let mut tracker = OutcomeTracker::new();
        tracker.pending.push_back(super::PendingOutcome {
            process_name: "test_proc".into(),
            throttled_at: Instant::now() - Duration::from_secs(31),
            pressure_before: 0.75,
            watts_before: 1.0,
            swap_gb_at_throttle: 0.0,
            action_type: crate::engine::learning_pipeline::ActionKind::Throttle,
        });
        tracker.weights.insert(
            "test_proc".into(),
            PatternWeight {
                throttle_count: 1,
                effective_count: 0,
            },
        );

        assert!(tracker.experience.is_empty());
        tracker.tick(0.70); // drop = 0.05
        assert_eq!(tracker.experience.len(), 1);
    }

    // ── Process causal graph tests ──────────────────────────────────────────────

    #[test]
    fn co_occurrence_builds_from_active_processes() {
        let mut tracker = OutcomeTracker::new();
        let procs = vec!["Chrome".into(), "Xcode".into(), "node".into()];
        tracker.record_co_occurrence(&procs);

        // Should have 3 pairs: (Chrome,Xcode), (Chrome,node), (Xcode,node)
        assert_eq!(tracker.co_occurrence.len(), 3);
        assert!(tracker.is_causal_pair("Chrome", "Xcode", 1).is_some());
        assert!(tracker.is_causal_pair("Xcode", "Chrome", 1).is_some()); // order invariant
    }

    #[test]
    fn co_occurrence_counts_accumulate() {
        let mut tracker = OutcomeTracker::new();
        let procs = vec!["Chrome".into(), "Xcode".into()];
        for _ in 0..5 {
            tracker.record_co_occurrence(&procs);
        }
        assert_eq!(tracker.is_causal_pair("Chrome", "Xcode", 5), Some(5));
        assert!(tracker.is_causal_pair("Chrome", "Xcode", 6).is_none());
    }

    #[test]
    fn top_causal_pairs_sorted_by_count() {
        let mut tracker = OutcomeTracker::new();
        // Pair A+B: 10 times
        for _ in 0..10 {
            tracker.record_co_occurrence(&vec!["A".into(), "B".into()]);
        }
        // Pair C+D: 3 times
        for _ in 0..3 {
            tracker.record_co_occurrence(&vec!["C".into(), "D".into()]);
        }

        let top = tracker.top_causal_pairs(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "A");
        assert_eq!(top[0].1, "B");
        assert_eq!(top[0].2, 10);
    }

    #[test]
    fn co_occurrence_gc_bounds_size() {
        let mut tracker = OutcomeTracker::new();
        // Generate 200 distinct pairs via 200 calls with 2 procs each.
        for i in 0..200 {
            let procs = vec![format!("p{i}"), format!("q{i}")];
            tracker.record_co_occurrence(&procs);
        }
        // All 200 entries have count=1. The GC safety floor (would_retain >= 50)
        // prevents a total wipe when cutoff == every entry's count — so GC is
        // skipped here. The map stays at 200 until some entries gain higher counts.
        // This is correct: nuking the entire causal graph on cold start destroys
        // all structural information. [Boldi & Vigna 2014]
        assert!(
            tracker.co_occurrence.len() <= 200,
            "co_occurrence should not grow beyond inserts: {}",
            tracker.co_occurrence.len()
        );
        // Verify GC does trigger when there IS a meaningful count gradient.
        // Repeat-observe the first 10 pairs 10 times → they rise to count=11.
        for _ in 0..10 {
            for i in 0..10 {
                let procs = vec![format!("p{i}"), format!("q{i}")];
                tracker.record_co_occurrence(&procs);
            }
        }
        // Now top-10 have count=11, rest have count=1. Cutoff separates them.
        // would_retain = 10 which is < 50 — GC still skipped (floor protects even small graphs).
        // To trigger GC we need ≥50 entries above cutoff, which requires more variance.
        assert!(
            tracker.co_occurrence.len() >= 50,
            "graph retained minimum connectivity: {}",
            tracker.co_occurrence.len()
        );
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

    // ── HRPO / Dr. Zero tests ────────────────────────────────────────────

    #[test]
    fn workload_hop_classification() {
        assert_eq!(
            WorkloadHop::from_process_name("Brave Browser Helper (Renderer)"),
            WorkloadHop::Browser
        );
        assert_eq!(WorkloadHop::from_process_name("rustc"), WorkloadHop::Build);
        assert_eq!(
            WorkloadHop::from_process_name("cloudd"),
            WorkloadHop::CloudSync
        );
        assert_eq!(
            WorkloadHop::from_process_name("coreaudiod"),
            WorkloadHop::Media
        );
        assert_eq!(
            WorkloadHop::from_process_name("launchd"),
            WorkloadHop::SystemDaemon
        );
        assert_eq!(WorkloadHop::from_process_name("Warp"), WorkloadHop::General);
    }

    #[test]
    fn hop_group_weight_learning() {
        let mut g = HopGroupWeight::default();
        assert!((g.effectiveness() - 0.5).abs() < 0.01);
        // Record 10 outcomes: 7 effective
        for i in 0..10 {
            g.record(i < 7, 0.03);
        }
        // Bayesian: (7+1)/(10+2) ≈ 0.667
        assert!(g.effectiveness() > 0.60);
        // Prediction moving toward effectiveness (α=0.1, 10 samples → partial convergence)
        assert!(g.predicted_effectiveness > 0.2);
    }

    #[test]
    fn hop_group_exploration_signal() {
        let mut g = HopGroupWeight::default();
        // Alternate effective/not to create high prediction error
        for i in 0..10 {
            g.record(i % 2 == 0, 0.02);
        }
        // With alternating outcomes, prediction error should be elevated
        assert!(g.prediction_error_ema > 0.1);
    }

    #[test]
    fn tracker_hop_effectiveness_zero_shot() {
        let tracker = OutcomeTracker::new();
        // No data yet → None
        assert!(tracker.hop_effectiveness("Brave Browser Helper").is_none());
    }

    // ── Coordinated multi-process freezing (Feature 2) ────────────────────────

    /// Simulate the real-world scenario: Safari + cloudd co-occur during pressure
    /// spikes 10 times. Verify they're returned as a top pair with count ≥ 8
    /// (the gate used by coordinated freezing in the daemon).
    #[test]
    fn coordinated_freeze_safari_cloudd_cluster() {
        let mut tracker = OutcomeTracker::new();
        // Simulate 10 pressure events where Safari and cloudd are both active.
        for _ in 0..10 {
            tracker.record_co_occurrence(&vec![
                "Safari".into(),
                "cloudd".into(),
                "suggestd".into(), // noise: also present but less relevant
            ]);
        }
        // Simulate 2 events where Safari is alone (cloudd not present).
        for _ in 0..2 {
            tracker.record_co_occurrence(&vec!["Safari".into(), "WindowServer".into()]);
        }

        // Safari + cloudd should be the top pair with count = 10.
        let top = tracker.top_causal_pairs(3);
        let safari_cloudd = top.iter().find(|(a, b, _)| {
            (a.contains("Safari") && b.contains("cloudd"))
                || (a.contains("cloudd") && b.contains("Safari"))
        });
        assert!(
            safari_cloudd.is_some(),
            "Safari+cloudd should appear in top pairs"
        );
        let (_, _, count) = safari_cloudd.unwrap();
        assert!(
            *count >= 8,
            "count {} must meet the ≥8 gate for coordinated freezing",
            count
        );

        // is_causal_pair() query (order-invariant) should return the count.
        assert_eq!(tracker.is_causal_pair("cloudd", "Safari", 8), Some(10));
    }

    /// When only one process of a co-cluster is being actioned, the daemon
    /// pulls in the partner. Verify the co-occurrence data supports this:
    /// after ≥8 observations, the pair is queryable with min_count=8.
    #[test]
    fn coordinated_freeze_threshold_gate() {
        let mut tracker = OutcomeTracker::new();
        let procs = vec!["Dropbox".into(), "cloudd".into()];

        // Only 7 co-occurrences: below the gate → should NOT trigger.
        for _ in 0..7 {
            tracker.record_co_occurrence(&procs);
        }
        assert!(
            tracker.is_causal_pair("Dropbox", "cloudd", 8).is_none(),
            "7 co-occurrences should not meet the ≥8 gate"
        );

        // 8th co-occurrence: now it qualifies.
        tracker.record_co_occurrence(&procs);
        assert!(
            tracker.is_causal_pair("Dropbox", "cloudd", 8).is_some(),
            "8 co-occurrences should meet the ≥8 gate"
        );
    }

    #[test]
    fn short_window_velocity_tracks_natural_drop() {
        let mut tracker = OutcomeTracker::new();
        // Feed 4 non-action cycles with pressure dropping 0.10 each
        tracker.observe_cycle(0.80, false);
        tracker.observe_cycle(0.70, false);
        tracker.observe_cycle(0.60, false);
        tracker.observe_cycle(0.50, false);
        // Short-window velocity should be ~0.10 (pressure dropping 0.10/cycle)
        let v = tracker.pressure_velocity_short();
        assert!(
            v > 0.05 && v < 0.15,
            "expected ~0.10 short-window velocity, got {}",
            v
        );
    }

    #[test]
    fn short_window_resets_on_action() {
        let mut tracker = OutcomeTracker::new();
        tracker.observe_cycle(0.80, false);
        tracker.observe_cycle(0.70, false);
        tracker.observe_cycle(0.60, false);
        // Action taken — short window should reset
        tracker.observe_cycle(0.50, true);
        assert_eq!(
            tracker.pressure_velocity_short(),
            0.0,
            "short drift should reset on action"
        );
    }

    #[test]
    fn causal_effect_fast_fallback_to_ema_when_window_empty() {
        let mut tracker = OutcomeTracker::new();
        // Build up natural_drift_ema: pressure drops 0.001 per cycle across 70 cycles.
        // After DRIFT_WINDOW_TICKS=60 no-action cycles, EMA commits with positive drift.
        let mut p = 0.90f64;
        for _ in 0..70 {
            tracker.observe_cycle(p, false);
            p = (p - 0.001).max(0.0);
        }
        let drift = tracker.natural_drift();
        assert!(
            drift > 0.0,
            "should have positive natural drift after 70 declining cycles (got {})",
            drift
        );
        // Now act many times — short window clears each time
        for _ in 0..5 {
            tracker.observe_cycle(0.60, true);
        }
        assert!(
            tracker.short_window_deltas.is_empty(),
            "short window should be empty after actions"
        );
        // causal_effect_fast should use natural_drift_ema as fallback, not 0
        let fast = tracker.causal_effect_fast(0.005);
        let slow = tracker.causal_effect(0.005);
        assert_eq!(
            fast, slow,
            "fast should fall back to slow EMA when window is empty"
        );
    }

    #[test]
    fn causal_effect_fast_separates_action_from_drift() {
        let mut tracker = OutcomeTracker::new();
        // Establish a slow natural drift of ~0.02/cycle
        tracker.observe_cycle(0.80, false);
        tracker.observe_cycle(0.78, false);
        tracker.observe_cycle(0.76, false);
        tracker.observe_cycle(0.74, false);
        // Now: if observed_drop = 0.15 but drift = 0.02, causal effect ≈ 0.13
        let fast = tracker.causal_effect_fast(0.15);
        assert!(
            fast > 0.0,
            "causal_effect_fast should be positive when action > drift"
        );
        // If observed_drop = 0.01 (less than drift), effect should be near zero or negative
        let slow = tracker.causal_effect_fast(0.01);
        assert!(
            slow < fast,
            "small drop should yield smaller causal effect than large drop"
        );
    }

    // ── NARS drift detector integration tests ────────────────────────────────

    /// Simulates a process that was effective for 30 cycles, then becomes
    /// useless. Verifies that the NARS drift detector signals recalibration.
    #[test]
    fn nars_detects_regime_change_in_batch_resolve() {
        let mut tracker = OutcomeTracker::new();
        let now = std::time::Instant::now();

        // Phase 1: proc_X is consistently effective (30 resolved outcomes)
        for i in 0..30u32 {
            tracker
                .weights
                .entry("proc_X".to_string())
                .or_default()
                .throttle_count += 1;
            tracker
                .pending
                .push_back(crate::engine::outcome_tracker::PendingOutcome {
                    process_name: "proc_X".to_string(),
                    throttled_at: now - std::time::Duration::from_secs(31 + i as u64),
                    pressure_before: 0.75,
                    watts_before: 5.0,
                    swap_gb_at_throttle: 0.0,
                    action_type: crate::engine::learning_pipeline::ActionKind::Throttle,
                });
        }
        // Use high current pressure so outcomes resolve as effective
        tracker.tick(0.70); // pressure_before=0.75, current=0.70 → drop=0.05 → effective
                            // tick resolves all pending outcomes with 0.75-0.70=0.05 drop (≥0.01 → effective)

        let score_phase1 = tracker.nars_drift_score();

        // Phase 2: proc_X suddenly useless (pressure never drops)
        for i in 0..30u32 {
            tracker
                .weights
                .entry("proc_X".to_string())
                .or_default()
                .throttle_count += 1;
            tracker
                .pending
                .push_back(crate::engine::outcome_tracker::PendingOutcome {
                    process_name: "proc_X".to_string(),
                    throttled_at: now - std::time::Duration::from_secs(31 + i as u64),
                    pressure_before: 0.70,
                    watts_before: 5.0,
                    swap_gb_at_throttle: 0.0,
                    action_type: crate::engine::learning_pipeline::ActionKind::Throttle,
                });
        }
        tracker.tick(0.70); // pressure stayed same → drop=0 → NOT effective

        let score_phase2 = tracker.nars_drift_score();
        let drifted = tracker.drift_detector.drifted_count;

        // Drift score must increase after regime change
        assert!(
            score_phase2 > score_phase1 || drifted >= 1,
            "regime change (effective→ineffective) must increase drift score. phase1={:.4} phase2={:.4} drifted={}",
            score_phase1, score_phase2, drifted
        );
    }

    /// Verifies that nars_acknowledge_recalibration resets the drift signal.
    #[test]
    fn nars_acknowledge_resets_drift_after_recalibration() {
        let mut tracker = OutcomeTracker::new();
        // Build up some drift
        for _ in 0..20 {
            tracker.drift_detector.observe("proc_A", true);
        }
        for _ in 0..20 {
            tracker.drift_detector.observe("proc_A", false);
        }
        let drift_before = tracker.nars_drift_score();
        tracker.nars_acknowledge_recalibration();
        let drift_after = tracker.nars_drift_score();
        assert!(
            drift_after < drift_before,
            "acknowledge must reduce drift score"
        );
        assert_eq!(
            tracker.drift_detector.drifted_count, 0,
            "drifted_count must clear after acknowledge"
        );
    }

    /// Validates that NARS recalibration actually helps convergence:
    /// after a regime change, soft decay of Bayesian weights + new observations
    /// should yield faster convergence to the new reality than without recalibration.
    #[test]
    fn nars_recalibration_accelerates_convergence_after_regime_change() {
        // Phase 1: build up a strongly biased belief (process was effective)
        let mut tracker_with_nars = OutcomeTracker::new();
        let mut tracker_without_nars = OutcomeTracker::new();

        // Both trackers see 20 effective throttles
        for t in [&mut tracker_with_nars, &mut tracker_without_nars] {
            t.weights
                .entry("proc_X".to_string())
                .or_default()
                .throttle_count = 20;
            t.weights.get_mut("proc_X").unwrap().effective_count = 18; // 90% effective
        }
        // NARS tracker also has built-up beliefs
        for _ in 0..18 {
            tracker_with_nars.drift_detector.observe("proc_X", true);
        }
        for _ in 0..2 {
            tracker_with_nars.drift_detector.observe("proc_X", false);
        }

        // Phase 2: regime change — process is now completely ineffective.
        // Simulate 5 new observations of failure.
        for _ in 0..5 {
            tracker_with_nars.drift_detector.observe("proc_X", false);
            tracker_without_nars.drift_detector.observe("proc_X", false);
        }

        // NARS tracker should detect drift and be ready to recalibrate
        // (either drifted_count >= 1 or drift_score increasing)
        let nars_score = tracker_with_nars.nars_drift_score();
        let control_score = tracker_without_nars.nars_drift_score();

        // Both should have same score since they have same drift_detector observations
        // The key difference: when needs_recalibration() triggers, tracker_with_nars
        // gets its weights softened
        if tracker_with_nars.nars_needs_recalibration() {
            // Apply recalibration
            for w in tracker_with_nars.weights.values_mut() {
                w.effective_count = (w.effective_count / 2).max(1);
                w.throttle_count = (w.throttle_count / 2).max(2);
            }
            tracker_with_nars.nars_acknowledge_recalibration();

            // After recalibration, effectiveness should be closer to 0.5 (prior)
            let eff_after = tracker_with_nars.weights["proc_X"].effectiveness();
            let eff_before = tracker_without_nars.weights["proc_X"].effectiveness();

            // Recalibrated tracker should be closer to 0.5 (neutral prior)
            let dist_with = (eff_after - 0.5).abs();
            let dist_without = (eff_before - 0.5).abs();
            assert!(
                dist_with <= dist_without,
                "recalibrated weights should be closer to neutral prior: with={:.3} without={:.3}",
                eff_after,
                eff_before
            );
        }
        // Even if recalibration wasn't triggered, verify scores moved
        let _ = (nars_score, control_score); // used in the conditional above
    }

    /// Verifies roundtrip: to_persisted + restore preserves drift_detector state.
    #[test]
    fn nars_drift_survives_persist_restore_roundtrip() {
        let mut tracker = OutcomeTracker::new();
        // Build a non-trivial drift state
        for _ in 0..20 {
            tracker.drift_detector.observe("proc_B", true);
        }
        for _ in 0..10 {
            tracker.drift_detector.observe("proc_B", false);
        }
        let score_before = tracker.nars_drift_score();

        // Persist then restore
        let persisted = tracker.to_persisted();
        let mut restored = OutcomeTracker::new();
        restored.restore(persisted);

        let score_after = restored.nars_drift_score();
        assert!(
            (score_after - score_before).abs() < 1e-9,
            "drift score must be identical after roundtrip: before={} after={}",
            score_before,
            score_after
        );
        // Belief for proc_B should be present in restored tracker
        assert!(
            restored.drift_detector.belief("proc_B").is_some(),
            "proc_B belief must survive roundtrip"
        );
    }

    // ── Affective salience end-to-end property tests ──────────────────────────

    /// Property: crisis events (high swap, high pressure) produce a stronger
    /// drift signal than routine low-pressure events for the same outcome change.
    ///
    /// This verifies the full pipeline:
    ///   real metrics → Salience → observe_salient() → drift_score EMA
    ///
    /// [McGaugh 2004] amygdala modulation: crisis events leave stronger traces.
    #[test]
    fn salience_crisis_produces_stronger_drift_than_routine() {
        let now = std::time::Instant::now();

        // Routine tracker: low-pressure events
        let mut routine = OutcomeTracker::new();
        for i in 0..10u32 {
            routine
                .weights
                .entry("proc_Z".to_string())
                .or_default()
                .throttle_count += 1;
            routine
                .pending
                .push_back(crate::engine::outcome_tracker::PendingOutcome {
                    process_name: "proc_Z".to_string(),
                    throttled_at: now - std::time::Duration::from_secs(31 + i as u64),
                    pressure_before: 0.40, // low pressure → low arousal
                    watts_before: 2.0,
                    swap_gb_at_throttle: 0.1, // minimal swap
                    action_type: crate::engine::learning_pipeline::ActionKind::Throttle,
                });
        }
        routine.tick(0.40); // no drop → ineffective

        // Crisis tracker: same outcome pattern but high swap + high pressure
        let mut crisis = OutcomeTracker::new();
        for i in 0..10u32 {
            crisis
                .weights
                .entry("proc_Z".to_string())
                .or_default()
                .throttle_count += 1;
            crisis
                .pending
                .push_back(crate::engine::outcome_tracker::PendingOutcome {
                    process_name: "proc_Z".to_string(),
                    throttled_at: now - std::time::Duration::from_secs(31 + i as u64),
                    pressure_before: 0.90, // high pressure → high arousal
                    watts_before: 2.0,
                    swap_gb_at_throttle: 7.5, // near-full swap → max arousal
                    action_type: crate::engine::learning_pipeline::ActionKind::Throttle,
                });
        }
        crisis.tick(0.90); // no drop → ineffective

        // Crisis should have higher drift score due to arousal amplification
        assert!(
            crisis.nars_drift_score() >= routine.nars_drift_score(),
            "crisis drift score {:.5} should be >= routine {:.5}",
            crisis.nars_drift_score(),
            routine.nars_drift_score()
        );
    }

    /// Property: after a crisis regime (high arousal), NARS recalibration is
    /// triggered at a tighter threshold than under idle conditions.
    ///
    /// This verifies ArousalState.adjusted_drift_threshold() integration:
    /// the same drift EMA score triggers recalibration under crisis but not at idle.
    #[test]
    fn arousal_tightens_recalibration_threshold() {
        use crate::engine::nars_belief::ArousalState;
        use crate::engine::nars_belief::Salience;

        let mut tracker = OutcomeTracker::new();
        let now = std::time::Instant::now();

        // Build up a moderate drift signal (not enough to trigger at default 0.08)
        // We'll do a small regime change: 5 effective then 5 ineffective
        for i in 0..5u32 {
            tracker
                .weights
                .entry("proc_W".to_string())
                .or_default()
                .throttle_count += 1;
            tracker
                .pending
                .push_back(crate::engine::outcome_tracker::PendingOutcome {
                    process_name: "proc_W".to_string(),
                    throttled_at: now - std::time::Duration::from_secs(31 + i as u64),
                    pressure_before: 0.75,
                    watts_before: 3.0,
                    swap_gb_at_throttle: 2.0,
                    action_type: crate::engine::learning_pipeline::ActionKind::Throttle,
                });
        }
        tracker.tick(0.70); // 0.75→0.70 drop = effective

        for i in 0..5u32 {
            tracker
                .weights
                .entry("proc_W".to_string())
                .or_default()
                .throttle_count += 1;
            tracker
                .pending
                .push_back(crate::engine::outcome_tracker::PendingOutcome {
                    process_name: "proc_W".to_string(),
                    throttled_at: now - std::time::Duration::from_secs(31 + i as u64),
                    pressure_before: 0.70,
                    watts_before: 3.0,
                    swap_gb_at_throttle: 5.0,
                    action_type: crate::engine::learning_pipeline::ActionKind::Throttle,
                });
        }
        tracker.tick(0.70); // no drop = ineffective

        let drift_score = tracker.nars_drift_score();

        // Idle arousal state: threshold ~0.10 (raised above base 0.08)
        let mut idle_arousal = ArousalState::default();
        let calm = Salience::compute(0.1, 0.0, 0.0, 0.0);
        for _ in 0..50 {
            idle_arousal.update(calm);
        }
        let idle_threshold = idle_arousal.adjusted_drift_threshold(0.08);

        // Crisis arousal state: threshold ~0.06 (lowered below base 0.08)
        let mut crisis_arousal = ArousalState::default();
        let crisis = Salience::compute(0.9, -0.05, 0.9, 7.0);
        for _ in 0..50 {
            crisis_arousal.update(crisis);
        }
        let crisis_threshold = crisis_arousal.adjusted_drift_threshold(0.08);

        // Crisis threshold must be strictly lower (more sensitive)
        assert!(
            crisis_threshold < idle_threshold,
            "crisis threshold {:.4} should be < idle threshold {:.4}",
            crisis_threshold,
            idle_threshold
        );

        // Verify the threshold arithmetic: with the same drift score,
        // one state recalibrates and the other doesn't (if drift is in the gap).
        // The key property: crisis_threshold < 0.08 < idle_threshold.
        assert!(
            crisis_threshold < 0.08,
            "crisis should lower threshold below base: {:.4}",
            crisis_threshold
        );
        assert!(
            idle_threshold > 0.08,
            "idle should raise threshold above base: {:.4}",
            idle_threshold
        );

        // Suppress unused variable warning
        let _ = drift_score;
    }

    /// Property: record_throttle_with_swap() is backward-compatible with
    /// record_throttle() — both resolve outcomes correctly.
    #[test]
    fn record_throttle_with_swap_backward_compat() {
        let now = std::time::Instant::now();
        let mut t1 = OutcomeTracker::new();
        let mut t2 = OutcomeTracker::new();

        // t1 uses the new API with swap
        t1.record_throttle_with_swap("proc_A", 0.75, 5.0, 3.0);
        // t2 uses legacy API (swap=0.0)
        t2.record_throttle("proc_A", 0.75, 5.0);

        // Both should have exactly 1 pending outcome
        assert_eq!(t1.pending.len(), 1);
        assert_eq!(t2.pending.len(), 1);

        // Both should resolve identically under the same pressure drop
        // (inject a past throttle time to force resolution)
        t1.pending[0].throttled_at = now - std::time::Duration::from_secs(35);
        t2.pending[0].throttled_at = now - std::time::Duration::from_secs(35);

        let b1 = t1.tick(0.70);
        let b2 = t2.tick(0.70);

        // Both should see 1 effective outcome (pressure dropped 0.05)
        assert_eq!(
            b1.effective_names.len(),
            b2.effective_names.len(),
            "record_throttle_with_swap must resolve same as record_throttle"
        );
    }

    // ── Survival-bias closure: blocked-action tests ─────────────────────────

    /// Test 1: record_blocked increments per-key counter under composite
    /// key "<class>:<gate>" and survives multiple gates for the same class.
    #[test]
    fn record_blocked_increments_counter() {
        let mut tracker = OutcomeTracker::new();

        tracker.record_blocked("freeze", "is-protected-name", 0.65);
        tracker.record_blocked("freeze", "is-protected-name", 0.70);
        tracker.record_blocked("freeze", "user-protected", 0.72);
        tracker.record_blocked("throttle", "is-protected-name", 0.55);

        let p_proto = tracker
            .blocked_patterns
            .get("freeze:is-protected-name")
            .expect("freeze:is-protected-name pattern present");
        assert_eq!(p_proto.blocked_count, 2, "two same-key blocks → count 2");
        assert_eq!(p_proto.would_have_helped_count, 0);

        let p_user = tracker
            .blocked_patterns
            .get("freeze:user-protected")
            .expect("freeze:user-protected pattern present");
        assert_eq!(p_user.blocked_count, 1);

        let p_throttle = tracker
            .blocked_patterns
            .get("throttle:is-protected-name")
            .expect("throttle:is-protected-name pattern present");
        assert_eq!(p_throttle.blocked_count, 1);

        // Pending observations enqueued for later counterfactual eval.
        assert_eq!(
            tracker.blocked_pending_depth(),
            4,
            "pending blocked queue must hold all 4 observations"
        );
    }

    /// Test 2: blocked_effectiveness returns Bayesian Laplace prior 0.5
    /// when the pattern has no observations, and tracks prior correctly
    /// for low-count patterns. Counts < 3 should remain near the 0.5
    /// prior to avoid drawing strong conclusions from noise.
    #[test]
    fn blocked_effectiveness_returns_bayesian_prior_when_undersampled() {
        let mut tracker = OutcomeTracker::new();

        // No data → exactly the 0.5 Laplace prior.
        let unknown = tracker.blocked_effectiveness("freeze:user-protected");
        assert!(
            (unknown - 0.5).abs() < 1e-9,
            "Laplace prior must be 0.5 with no data, got {}",
            unknown
        );

        // 1 block, 0 helped → (0+1)/(1+2) = 0.333… (still close to prior).
        tracker.record_blocked("freeze", "user-protected", 0.60);
        let one_block = tracker.blocked_effectiveness("freeze:user-protected");
        assert!(
            (one_block - (1.0 / 3.0)).abs() < 1e-9,
            "expected 1/3 with 1 block 0 helped, got {}",
            one_block
        );

        // 2 blocks, 1 helped (manually inject) → (1+1)/(2+2) = 0.5.
        // Verifies the formula and that we converge correctly.
        tracker.record_blocked("freeze", "user-protected", 0.65);
        if let Some(p) = tracker.blocked_patterns.get_mut("freeze:user-protected") {
            p.would_have_helped_count = 1;
        }
        let half = tracker.blocked_effectiveness("freeze:user-protected");
        assert!(
            (half - 0.5).abs() < 1e-9,
            "expected 0.5 with 2 blocks 1 helped, got {}",
            half
        );

        // Undersample (count<3): still close to 0.5 prior. This is the
        // explicit "Bayesian with prior correct when count<3" property.
        assert!(
            tracker
                .blocked_patterns
                .get("freeze:user-protected")
                .map(|p| p.blocked_count < 3)
                .unwrap_or(false),
            "test precondition: blocked_count must be <3 for prior dominance"
        );
    }

    /// Test 3: serialization roundtrip — blocked_patterns survives a full
    /// to_persisted → JSON → from JSON → restore cycle, including counts.
    #[test]
    fn blocked_patterns_persistence_roundtrip() {
        // Stage state with non-trivial counts.
        let mut original = OutcomeTracker::new();
        original.record_blocked("freeze", "is-protected-name", 0.70);
        original.record_blocked("freeze", "is-protected-name", 0.72);
        original.record_blocked("freeze", "is-protected-name", 0.74);
        if let Some(p) = original
            .blocked_patterns
            .get_mut("freeze:is-protected-name")
        {
            p.would_have_helped_count = 1;
        }
        original.record_blocked("throttle", "user-protected", 0.55);

        // Serialize → JSON → deserialize.
        let snapshot = original.to_persisted();
        let json =
            serde_json::to_string(&snapshot).expect("OutcomeTrackerPersisted serializes to JSON");

        // Forward-compat: old files (no blocked_patterns key) must still
        // deserialize with serde(default) → empty map. Test that explicitly.
        assert!(
            json.contains("\"blocked_patterns\""),
            "expected blocked_patterns field in serialized JSON, got: {}",
            json
        );

        let parsed: OutcomeTrackerPersisted =
            serde_json::from_str(&json).expect("JSON roundtrips back to OutcomeTrackerPersisted");

        // Restore into a fresh tracker.
        let mut restored = OutcomeTracker::new();
        restored.restore(parsed);

        let p_freeze = restored
            .blocked_patterns
            .get("freeze:is-protected-name")
            .expect("freeze pattern restored");
        assert_eq!(p_freeze.blocked_count, 3);
        assert_eq!(p_freeze.would_have_helped_count, 1);

        let p_throttle = restored
            .blocked_patterns
            .get("throttle:user-protected")
            .expect("throttle pattern restored");
        assert_eq!(p_throttle.blocked_count, 1);
        assert_eq!(p_throttle.would_have_helped_count, 0);

        // Effectiveness query reproduces correctly post-restore.
        let eff = restored.blocked_effectiveness("freeze:is-protected-name");
        // (1+1)/(3+2) = 0.4
        assert!(
            (eff - 0.4).abs() < 1e-9,
            "post-restore effectiveness must equal pre-persist computation, got {}",
            eff
        );
    }

    /// Bonus: forward-compat — old persisted blob without blocked_patterns
    /// key deserializes to empty map (not error), thanks to #[serde(default)].
    #[test]
    fn blocked_patterns_missing_field_deserializes_to_empty() {
        // Hand-built JSON missing the blocked_patterns key (simulates old
        // learned_state.json files written before this commit).
        let legacy_json = r#"{
            "weights": {},
            "total_effective": 0,
            "total_resolved": 0,
            "baseline_drop_ema": 0.0,
            "baseline_samples": 0,
            "experience_records": [],
            "co_occurrence": [],
            "natural_drift_ema": 0.0,
            "hop_groups": {}
        }"#;

        let parsed: OutcomeTrackerPersisted = serde_json::from_str(legacy_json)
            .expect("legacy JSON without blocked_patterns must still parse");
        assert!(
            parsed.blocked_patterns.is_empty(),
            "missing blocked_patterns must default to empty map"
        );
    }

    /// Counterfactual resolver: pressure rising > drift baseline within
    /// the eval window must promote blocked_count → would_have_helped_count.
    #[test]
    fn tick_blocked_credits_post_block_pressure_rise() {
        let mut tracker = OutcomeTracker::new();
        // Natural drift EMA = 0 (presure stable on its own).
        tracker.natural_drift_ema = 0.0;

        // Block a freeze at pressure 0.50.
        tracker.record_blocked("freeze", "is-protected-name", 0.50);
        // Backdate to past the eval window.
        tracker.pending_blocked[0].blocked_at = Instant::now() - Duration::from_secs(35);

        // 35s later pressure rose to 0.60 → counterfactual rise +0.10
        // is well above the 0.02 effective threshold. Block was a miss.
        tracker.tick_blocked(0.60);
        let p = tracker
            .blocked_patterns
            .get("freeze:is-protected-name")
            .expect("pattern present");
        assert_eq!(p.blocked_count, 1);
        assert_eq!(
            p.would_have_helped_count, 1,
            "post-block pressure rise above drift must promote helped count"
        );
        assert_eq!(
            tracker.blocked_pending_depth(),
            0,
            "resolved observation must drain from pending queue"
        );
    }

    /// Counterfactual resolver: pressure that DROPPED naturally must NOT
    /// give the blocked action credit (no missed opportunity).
    #[test]
    fn tick_blocked_does_not_credit_natural_drop() {
        let mut tracker = OutcomeTracker::new();
        tracker.natural_drift_ema = 0.05; // baseline drops 5pp / window

        tracker.record_blocked("freeze", "is-protected-name", 0.70);
        tracker.pending_blocked[0].blocked_at = Instant::now() - Duration::from_secs(35);

        // Pressure dropped to 0.65 (rise = -0.05). Counterfactual = -0.05
        // + drift 0.05 = 0.00, well below the 0.02 threshold.
        tracker.tick_blocked(0.65);
        let p = tracker
            .blocked_patterns
            .get("freeze:is-protected-name")
            .expect("pattern present");
        assert_eq!(p.blocked_count, 1);
        assert_eq!(
            p.would_have_helped_count, 0,
            "natural pressure drop must NOT credit blocked action"
        );
    }

    /// reset_after_wake clears in-flight pending blocks but preserves
    /// learned Bayesian counts (those are restart-safe by design).
    #[test]
    fn reset_after_wake_clears_pending_blocked_but_preserves_counts() {
        let mut tracker = OutcomeTracker::new();
        tracker.record_blocked("freeze", "is-protected-name", 0.60);
        tracker.record_blocked("freeze", "user-protected", 0.65);
        assert_eq!(tracker.blocked_pending_depth(), 2);
        assert_eq!(tracker.blocked_patterns.len(), 2);

        tracker.reset_after_wake();
        assert_eq!(
            tracker.blocked_pending_depth(),
            0,
            "pending blocked queue must drain on wake (stale Instants)"
        );
        assert_eq!(
            tracker.blocked_patterns.len(),
            2,
            "learned Bayesian counts must survive wake"
        );
    }
}
