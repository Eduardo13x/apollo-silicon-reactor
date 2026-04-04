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

use crate::engine::nars_belief::DriftDetector;

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
}

/// Throttle pendiente de resolución de outcome.
struct PendingOutcome {
    process_name: String,
    throttled_at: Instant,
    pressure_before: f64,
    /// Watts estimados del proceso en el momento del throttle (para record_savings).
    watts_before: f64,
}

/// Resumen de la resolución de un batch de outcomes.
pub struct OutcomeBatch {
    /// Nombres de procesos cuyo throttle fue efectivo esta ronda.
    pub effective_names: Vec<String>,
    /// Watts totales ahorrados por outcomes efectivos (para EnergyTracker).
    pub savings_watts: f64,
    /// Nombres de procesos marcados como low-value (heurístico fallando).
    pub low_value_names: Vec<String>,
    /// All resolved outcomes this tick: (process_name, pre_pressure, post_pressure).
    /// Includes both effective and ineffective resolutions — used by LearningPipeline.
    pub resolved_outcomes: Vec<(String, f64, f64)>,
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
    }

    /// Query: expected effectiveness for throttling `process` at `pressure`.
    /// Returns (expected_drop, confidence) or None if no similar records.
    /// Similarity: same process name AND pressure within ±0.10.
    pub fn query_similar(&self, process: &str, pressure: f64) -> Option<(f64, f64)> {
        let mut sum_drop = 0.0_f64;
        let mut count = 0u32;
        for r in &self.records {
            if r.process_name == process && (r.pressure_at_action - pressure).abs() <= 0.10 {
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
            let e = groups.entry(r.process_name.clone()).or_insert((0.0, 0.0, 0, 0));
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

impl WorkloadHop {
    pub fn from_process_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("brave") || lower.contains("chrome") || lower.contains("safari")
            || lower.contains("firefox") || lower.contains("webkit") || lower.contains("renderer")
        {
            WorkloadHop::Browser
        } else if lower.contains("rustc") || lower.contains("cargo") || lower.contains("clang")
            || lower.contains("swift") || lower == "cc" || lower.contains("make")
            || lower.contains("ninja")
        {
            WorkloadHop::Build
        } else if lower.contains("cloud") || lower.contains("dropbox") || lower.contains("drive")
            || lower.contains("sync") || lower.contains("bird")
        {
            WorkloadHop::CloudSync
        } else if lower.contains("audio") || lower.contains("video")
            || lower.contains("avconf") || lower.contains("camera")
        {
            WorkloadHop::Media
        } else if lower.ends_with('d') && lower.len() > 3 && !lower.contains(' ') {
            WorkloadHop::SystemDaemon
        } else {
            WorkloadHop::General
        }
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
        }
    }

    /// Current NARS drift score [0,1]. High = model has drifted from reality.
    pub fn nars_drift_score(&self) -> f64 {
        self.drift_detector.score()
    }

    /// True if NARS drift detector signals that Bayesian weights need recalibration.
    pub fn nars_needs_recalibration(&self) -> bool {
        self.drift_detector.needs_recalibration()
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
        // Actualiza contador de throttles para el peso Bayesiano.
        let w = self.weights.entry(process_name.to_string()).or_default();
        w.throttle_count += 1;

        self.pending.push_back(PendingOutcome {
            process_name: process_name.to_string(),
            throttled_at: Instant::now(),
            pressure_before,
            watts_before,
        });

        // Cap: si la cola crece demasiado, descarta los más viejos sin resolver.
        // BUG-10: emit a diagnostic when we silently drop pending outcomes.
        if self.pending.len() > 300 {
            eprintln!("apollo: outcome_tracker: discarded 100 pending outcomes (cap)");
            self.pending.drain(..100);
        }
    }

    /// Resuelve los outcomes pendientes con más de 30s de antigüedad.
    /// Retorna un batch con los resultados para que el llamador actualice
    /// el EnergyTracker y la LearnedPolicy.
    pub fn tick(&mut self, current_pressure: f64) -> OutcomeBatch {
        const BASELINE_ALPHA: f64 = 0.01; // half-life ≈ 69 observaciones
        let check_after = Duration::from_secs(30);
        let mut effective_names = Vec::new();
        let mut savings_watts = 0.0_f64;
        let mut resolved_outcomes: Vec<(String, f64, f64)> = Vec::new();

        while let Some(front) = self.pending.front() {
            if front.throttled_at.elapsed() < check_after {
                break;
            }
            let outcome = self.pending.pop_front().unwrap();
            let pressure_drop = outcome.pressure_before - current_pressure;
            // Lowered from 0.02 to 0.01: on an 8GB M1 with 2-3GB swap, 2% absolute
            // is too strict a bar — many legitimate throttles produce 1-1.5% relief
            // that compounds across multiple actions. 1% catches these while still
            // filtering noise.
            let effective = pressure_drop >= 0.01;

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

            // NARS Revision: update drift belief for this process.
            // Detects when a process's effectiveness profile has changed regime.
            self.drift_detector.observe(&outcome.process_name, effective);

            // HRPO: update group-level effectiveness (Dr. Zero solver feedback).
            let hop = WorkloadHop::from_process_name(&outcome.process_name);
            self.hop_groups.entry(hop).or_default().record(effective, pressure_drop);

            // Store in experience memory for similarity queries.
            self.experience.push(ExperienceRecord {
                process_name: outcome.process_name.clone(),
                pressure_at_action: outcome.pressure_before,
                pressure_drop,
                effective,
            });

            // Collect resolved outcome for LearningPipeline (pre/post pressure).
            resolved_outcomes.push((
                outcome.process_name.clone(),
                outcome.pressure_before,
                current_pressure,
            ));

            self.total_resolved += 1;
            if effective {
                self.total_effective += 1;
                effective_names.push(outcome.process_name.clone());
                savings_watts += outcome.watts_before;
            }
        }

        // Detecta patrones que ya tienen suficientes datos y están por debajo
        // del baseline calibrado — throttlearlos no aporta más que la fluctuación natural.
        let threshold = self.calibrated_threshold();
        let low_value_names: Vec<String> = self
            .weights
            .iter()
            .filter(|(_, w)| w.is_low_value_vs_baseline(threshold))
            .map(|(name, _)| name.clone())
            .collect();

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

    /// GC for the weights HashMap — prevents unbounded growth in long-running daemons.
    ///
    /// Prunes entries that carry insufficient signal: fewer than 5 throttles AND
    /// fewer than 2 effective outcomes. These entries are essentially noise — they
    /// contribute only Laplace-smoothed 0.5 priors and waste memory.
    ///
    /// Complements the persist-time GC in `LearnedState::self_improve()` by also
    /// pruning in-process, typically called every 500 cycles (~4 minutes).
    pub fn gc_weights(&mut self) {
        self.weights.retain(|_, w| w.throttle_count >= 5 || w.effective_count >= 2);
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
            // Use > to ensure entries at the cutoff boundary are evicted, keeping only
            // entries strictly above the Nth-largest count. This guarantees the map
            // shrinks to ≤100 entries even when many pairs share the same count.
            self.co_occurrence.retain(|_, &mut v| v > cutoff);
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
        self.co_occurrence.get(&(a, b)).copied().filter(|&c| c >= min_count)
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
                    self.natural_drift_ema += DRIFT_ALPHA
                        * (self.drift_accumulator - self.natural_drift_ema);
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
        let mut groups: Vec<_> = self.hop_groups
            .iter()
            .map(|(&hop, g)| (hop, g.effectiveness(), g.throttle_count, g.prediction_error_ema))
            .collect();
        groups.sort_by(|a, b| b.2.cmp(&a.2)); // by throttle count descending
        groups
    }

    /// Dr. Zero self-challenge reward: average prediction error across all groups.
    /// Low = solver is calibrated. High = solver needs more training.
    pub fn self_challenge_score(&self) -> f64 {
        if self.hop_groups.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.hop_groups.values().map(|g| g.prediction_error_ema).sum();
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
            if let Ok(groups) = serde_json::from_str::<HashMap<WorkloadHop, HopGroupWeight>>(&data) {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn tick_marks_effective_when_pressure_drops() {
        let mut tracker = OutcomeTracker::new();
        // Simulate a throttle that happened 31s ago by manipulating pending directly.
        tracker.pending.push_back(super::PendingOutcome {
            process_name: "Dropbox".to_string(),
            throttled_at: Instant::now() - Duration::from_secs(31),
            pressure_before: 0.80,
            watts_before: 2.0,
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
        tracker.weights.insert("proc_a".into(), PatternWeight { throttle_count: 25, effective_count: 0 });
        tracker.weights.insert("proc_b".into(), PatternWeight { throttle_count: 25, effective_count: 0 });
        // Add 1 high-value process (should not affect penalty)
        tracker.weights.insert("proc_c".into(), PatternWeight { throttle_count: 25, effective_count: 20 });

        let penalty = tracker.rl_penalty();
        assert!((penalty - (-1.0)).abs() < 1e-6, "2 low-value = -1.0 penalty, got {}", penalty);
    }

    #[test]
    fn rl_penalty_capped_at_minus_3() {
        let mut tracker = OutcomeTracker::new();
        tracker.baseline_drop_ema = 0.25;
        tracker.baseline_samples = 50;
        for i in 0..10 {
            tracker.weights.insert(format!("proc_{i}"), PatternWeight { throttle_count: 25, effective_count: 0 });
        }
        let penalty = tracker.rl_penalty();
        assert!((penalty - (-3.0)).abs() < 1e-6, "10 low-value should cap at -3.0, got {}", penalty);
    }

    #[test]
    fn integration_outcome_to_rl_feedback() {
        // End-to-end: OutcomeTracker detects low-value → RL gets penalized.
        use crate::engine::rl_threshold::{RlThresholdAgent, RlState};

        let mut tracker = OutcomeTracker::new();
        tracker.baseline_drop_ema = 0.25;
        tracker.baseline_samples = 50;
        tracker.weights.insert("wasteful".into(), PatternWeight { throttle_count: 30, effective_count: 0 });

        let mut rl = RlThresholdAgent::load_or_default(std::path::Path::new("/dev/null"));
        let state = RlState::from_metrics(0.60, 0.40, 0);
        rl.tick(state, false);

        let q_before = rl.last_q_value();

        // Wire the feedback: outcome → RL
        let penalty = tracker.rl_penalty();
        assert!(penalty < 0.0);
        rl.inject_external_reward(penalty);

        let q_after = rl.last_q_value();
        assert!(q_after < q_before,
            "RL should be penalized by outcome feedback: {} < {}", q_after, q_before);
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
            });
        }
        assert_eq!(mem.len(), 3);
        // Oldest (proc_0, proc_1) evicted; proc_2, proc_3, proc_4 remain.
        assert!(mem.records.iter().all(|r| !r.process_name.starts_with("proc_0")));
        assert!(mem.records.iter().all(|r| !r.process_name.starts_with("proc_1")));
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
            });
        }
        assert!(mem.query_similar("Dropbox", 0.70).is_none());

        // 3rd record → Some
        mem.push(ExperienceRecord {
            process_name: "Dropbox".into(),
            pressure_at_action: 0.72,
            pressure_drop: 0.03,
            effective: true,
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
            });
        }
        // 3 records at pressure 0.30 (too far from 0.70)
        for _ in 0..3 {
            mem.push(ExperienceRecord {
                process_name: "chrome".into(),
                pressure_at_action: 0.30,
                pressure_drop: -0.01,
                effective: false,
            });
        }
        // Query at 0.70 should only match first 3.
        let (avg_drop, _) = mem.query_similar("chrome", 0.70).unwrap();
        assert!((avg_drop - 0.08).abs() < 1e-6, "should only match p≈0.70 records");
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
        });
        tracker.weights.insert("test_proc".into(), PatternWeight { throttle_count: 1, effective_count: 0 });

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
        // After GC at 151 entries, all count=1 entries are evicted.
        // Subsequent inserts re-grow but won't hit 150 again.
        assert!(
            tracker.co_occurrence.len() < 200,
            "GC should have pruned: {}",
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
        assert_eq!(WorkloadHop::from_process_name("Brave Browser Helper (Renderer)"), WorkloadHop::Browser);
        assert_eq!(WorkloadHop::from_process_name("rustc"), WorkloadHop::Build);
        assert_eq!(WorkloadHop::from_process_name("cloudd"), WorkloadHop::CloudSync);
        assert_eq!(WorkloadHop::from_process_name("coreaudiod"), WorkloadHop::Media);
        assert_eq!(WorkloadHop::from_process_name("launchd"), WorkloadHop::SystemDaemon);
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
                "suggestd".into(),  // noise: also present but less relevant
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
        assert!(safari_cloudd.is_some(), "Safari+cloudd should appear in top pairs");
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
        assert!(v > 0.05 && v < 0.15, "expected ~0.10 short-window velocity, got {}", v);
    }

    #[test]
    fn short_window_resets_on_action() {
        let mut tracker = OutcomeTracker::new();
        tracker.observe_cycle(0.80, false);
        tracker.observe_cycle(0.70, false);
        tracker.observe_cycle(0.60, false);
        // Action taken — short window should reset
        tracker.observe_cycle(0.50, true);
        assert_eq!(tracker.pressure_velocity_short(), 0.0, "short drift should reset on action");
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
        assert!(drift > 0.0, "should have positive natural drift after 70 declining cycles (got {})", drift);
        // Now act many times — short window clears each time
        for _ in 0..5 {
            tracker.observe_cycle(0.60, true);
        }
        assert!(tracker.short_window_deltas.is_empty(), "short window should be empty after actions");
        // causal_effect_fast should use natural_drift_ema as fallback, not 0
        let fast = tracker.causal_effect_fast(0.005);
        let slow = tracker.causal_effect(0.005);
        assert_eq!(fast, slow, "fast should fall back to slow EMA when window is empty");
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
        assert!(fast > 0.0, "causal_effect_fast should be positive when action > drift");
        // If observed_drop = 0.01 (less than drift), effect should be near zero or negative
        let slow = tracker.causal_effect_fast(0.01);
        assert!(slow < fast, "small drop should yield smaller causal effect than large drop");
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
            tracker.weights.entry("proc_X".to_string()).or_default().throttle_count += 1;
            tracker.pending.push_back(crate::engine::outcome_tracker::PendingOutcome {
                process_name: "proc_X".to_string(),
                throttled_at: now - std::time::Duration::from_secs(31 + i as u64),
                pressure_before: 0.75,
                watts_before: 5.0,
            });
        }
        // Use high current pressure so outcomes resolve as effective
        tracker.tick(0.70); // pressure_before=0.75, current=0.70 → drop=0.05 → effective
        // tick resolves all pending outcomes with 0.75-0.70=0.05 drop (≥0.01 → effective)

        let score_phase1 = tracker.nars_drift_score();

        // Phase 2: proc_X suddenly useless (pressure never drops)
        for i in 0..30u32 {
            tracker.weights.entry("proc_X".to_string()).or_default().throttle_count += 1;
            tracker.pending.push_back(crate::engine::outcome_tracker::PendingOutcome {
                process_name: "proc_X".to_string(),
                throttled_at: now - std::time::Duration::from_secs(31 + i as u64),
                pressure_before: 0.70,
                watts_before: 5.0,
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
        assert!(drift_after < drift_before, "acknowledge must reduce drift score");
        assert_eq!(tracker.drift_detector.drifted_count, 0, "drifted_count must clear after acknowledge");
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
            score_before, score_after
        );
        // Belief for proc_B should be present in restored tracker
        assert!(
            restored.drift_detector.belief("proc_B").is_some(),
            "proc_B belief must survive roundtrip"
        );
    }
}
