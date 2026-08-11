//! Focus Markov Chain — predicts which app the user will switch to next.
//!
//! # Model
//!
//! First-order Markov chain over foreground app transitions:
//!   P(next = B | current = A) = count(A→B) / Σ_x count(A→x)
//!
//! Reference: Norris, J.R. (1997). "Markov Chains." Cambridge University Press.
//!
//! # Pre-warming
//!
//! When Apollo predicts the next app with high confidence (≥ threshold),
//! it can pre-warm that app by:
//!   1. Raising its QoS tier (route to P-cores before the switch)
//!   2. Unfreezing it if frozen (SIGCONT before user clicks)
//!   3. Boosting its Jetsam priority (kernel keeps its pages resident)
//!
//! # Persistence
//!
//! The transition matrix survives reboots via `/var/lib/apollo/markov_transitions.json`.
//! Cold start: no predictions until ≥ 5 transitions from a given app are observed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ── Configuration ────────────────────────────────────────────────────────────

/// Minimum transitions from an app before we trust predictions from it.
const MIN_TRANSITIONS_FOR_PREDICTION: u32 = 5;

/// Minimum probability to consider a prediction actionable.
const MIN_CONFIDENCE: f64 = 0.30;

/// Maximum number of source apps to track (evict least-recent on overflow).
const MAX_TRACKED_APPS: usize = 100;

/// Maximum transition targets per source app.
const MAX_TARGETS_PER_SOURCE: usize = 30;

/// Calibration semantics version. Version 2 separates cold calibration from
/// trusted kernel acceleration and replaces lifetime dwell timing with a
/// recency-adaptive estimate. Old outcome scores are not comparable.
const PREWARM_CALIBRATION_SCHEMA: u32 = 2;
const PREWARM_MIN_TRIALS: u32 = 5;
const PREWARM_QUARANTINE_THRESHOLD: f64 = 0.35;
const PREWARM_RECOVERY_THRESHOLD: f64 = 0.40;
const PREWARM_OUTCOME_ALPHA: f64 = 0.25;
const PREWARM_PROBE_BASE_TRANSITIONS: u64 = 16;
const PREWARM_MAX_BACKOFF_SHIFT: u8 = 4;
const PREWARM_MIN_CONTEXT_TRIALS: u32 = 5;
const PREWARM_CONTEXT_PRIOR_WEIGHT: f64 = 4.0;
const MAX_PREWARM_CONTEXTS_PER_TRANSITION: usize = 12;
const PREWARM_CONTEXT_KEY_LIMIT: u16 = 144;
const DWELL_EWMA_ALPHA: f64 = 0.55;
const MAX_DWELL_SAMPLE_SECS: f64 = 3_600.0;
const MAX_LEGACY_DWELL_ESTIMATE_SECS: f64 = 120.0;

fn default_prewarm_reliability() -> f64 {
    0.5
}

/// Bounded operating context for speculative pre-warm calibration. The source
/// app already carries the previous-app signal, so this key adds orthogonal
/// workload, time, pressure, and interaction state without unbounded strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrewarmContext {
    key: u16,
}

impl PrewarmContext {
    pub fn new(workload: &str, hour: u8, pressure: f64, interactive: bool) -> Self {
        let workload_bucket = match workload {
            "idle" => 0,
            "browsing" => 1,
            "build" | "coding" => 2,
            "llm-inference" | "ai-session" => 3,
            "mediaplayback" | "media-session" => 4,
            _ => 5,
        };
        let daypart = (hour.min(23) / 6) as u16;
        let pressure_bucket = if !pressure.is_finite() || pressure >= 0.65 {
            2
        } else if pressure >= 0.45 {
            1
        } else {
            0
        };
        let key =
            (((workload_bucket * 4 + daypart) * 3 + pressure_bucket) * 2) + u16::from(interactive);
        Self { key }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrewarmCalibrationStats {
    #[serde(default)]
    pub trials: u32,
    #[serde(default)]
    pub hits: u32,
    #[serde(default = "default_prewarm_reliability")]
    pub reliability: f64,
    #[serde(default)]
    pub quarantine_until_transition: u64,
    #[serde(default)]
    pub backoff_level: u8,
}

impl PrewarmCalibrationStats {
    fn new(prior_reliability: f64) -> Self {
        Self {
            trials: 0,
            hits: 0,
            reliability: prior_reliability.clamp(0.0, 1.0),
            quarantine_until_transition: 0,
            backoff_level: 0,
        }
    }
}

// ── Data structures ──────────────────────────────────────────────────────────

/// Statistics for a single A→B transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionStats {
    /// How many times this transition was observed.
    pub count: u32,
    /// Sum of dwell times in the source app before this transition (seconds).
    /// Used to compute average time before switching to this target.
    pub total_dwell_secs: f64,
    /// Recent dwell estimate for this transition. Lifetime means react too
    /// slowly after a hardware migration or a workflow change.
    #[serde(default)]
    pub recent_dwell_secs: f64,
    /// EWMA absolute deviation used for observability and future uncertainty
    /// calibration without retaining raw activity history.
    #[serde(default)]
    pub recent_dwell_deviation_secs: f64,
    #[serde(default)]
    pub recent_dwell_observations: u32,
    /// Outcome calibration for the speculative accelerator only. Transition
    /// probabilities continue learning even while acceleration is quarantined.
    #[serde(default)]
    pub prewarm_trials: u32,
    #[serde(default)]
    pub prewarm_hits: u32,
    #[serde(default = "default_prewarm_reliability")]
    pub prewarm_reliability: f64,
    #[serde(default)]
    pub prewarm_quarantine_until_transition: u64,
    #[serde(default)]
    pub prewarm_backoff_level: u8,
    /// Context-local outcome calibration. Keys are bounded categorical IDs,
    /// never process names or free-form telemetry.
    #[serde(default)]
    pub prewarm_contexts: HashMap<u16, PrewarmCalibrationStats>,
}

impl TransitionStats {
    fn new(dwell_secs: f64) -> Self {
        let dwell_secs = sanitize_dwell_sample(dwell_secs);
        Self {
            count: 1,
            total_dwell_secs: dwell_secs,
            recent_dwell_secs: dwell_secs,
            recent_dwell_deviation_secs: 0.0,
            recent_dwell_observations: 1,
            prewarm_trials: 0,
            prewarm_hits: 0,
            prewarm_reliability: default_prewarm_reliability(),
            prewarm_quarantine_until_transition: 0,
            prewarm_backoff_level: 0,
            prewarm_contexts: HashMap::new(),
        }
    }

    /// Average seconds spent in source app before switching to this target.
    pub fn avg_dwell_secs(&self) -> f64 {
        if self.count > 0 {
            self.total_dwell_secs / self.count as f64
        } else {
            0.0
        }
    }

    /// Timing estimate used by the live prewarmer. New observations adapt in
    /// a few transitions; legacy files fall back to a bounded lifetime mean
    /// until the first local observation arrives.
    pub fn predicted_dwell_secs(&self) -> f64 {
        if self.recent_dwell_observations > 0
            && self.recent_dwell_secs.is_finite()
            && self.recent_dwell_secs >= 0.0
        {
            self.recent_dwell_secs
        } else {
            self.avg_dwell_secs()
                .clamp(0.0, MAX_LEGACY_DWELL_ESTIMATE_SECS)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SourceDwellStats {
    #[serde(default)]
    pub recent_dwell_secs: f64,
    #[serde(default)]
    pub recent_dwell_deviation_secs: f64,
    #[serde(default)]
    pub observations: u32,
}

impl SourceDwellStats {
    fn observe(&mut self, dwell_secs: f64) {
        let dwell_secs = sanitize_dwell_sample(dwell_secs);
        if self.observations == 0 {
            self.recent_dwell_secs = dwell_secs;
            self.recent_dwell_deviation_secs = 0.0;
        } else {
            let previous = self.recent_dwell_secs;
            self.recent_dwell_secs = previous + DWELL_EWMA_ALPHA * (dwell_secs - previous);
            let absolute_error = (dwell_secs - previous).abs();
            self.recent_dwell_deviation_secs +=
                DWELL_EWMA_ALPHA * (absolute_error - self.recent_dwell_deviation_secs);
        }
        self.observations = self.observations.saturating_add(1);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrewarmAdmission {
    Ready,
    Probe,
    Quarantined,
}

impl PrewarmAdmission {
    pub fn allows_acceleration(self) -> bool {
        !matches!(self, Self::Quarantined)
    }

    pub fn allows_kernel_acceleration(self) -> bool {
        matches!(self, Self::Ready)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Probe => "probe",
            Self::Quarantined => "quarantined",
        }
    }
}

/// A prediction: which app is most likely next and with what confidence.
#[derive(Debug, Clone)]
pub struct FocusPrediction {
    /// Name of the predicted next app.
    pub app_name: String,
    /// Probability [0.0, 1.0].
    pub probability: f64,
    /// Average dwell time before this transition (seconds).
    pub avg_dwell_secs: f64,
    /// Number of local recency samples backing `avg_dwell_secs`. Zero means
    /// the estimate came from a bounded legacy lifetime average.
    pub dwell_observations: u32,
    pub dwell_deviation_secs: f64,
    /// PID of the predicted app (if currently running).
    pub pid: Option<u32>,
}

impl FocusPrediction {
    /// Probability that this destination becomes foreground within `horizon`.
    /// Destination probability and timing uncertainty remain separate: a very
    /// likely app can still have low near-term confidence when its ETA is far.
    pub fn confidence_within(&self, elapsed_secs: f64, horizon_secs: f64) -> f64 {
        if !elapsed_secs.is_finite() || !horizon_secs.is_finite() || horizon_secs <= 0.0 {
            return 0.0;
        }
        let eta = self.avg_dwell_secs - elapsed_secs.max(0.0);
        let timing_scale = self
            .dwell_deviation_secs
            .max(self.avg_dwell_secs.abs() * 0.15)
            .max(2.0);
        let exponent = ((eta - horizon_secs) / timing_scale).clamp(-60.0, 60.0);
        let timing_probability = 1.0 / (1.0 + exponent.exp());
        (self.probability.clamp(0.0, 1.0) * timing_probability).clamp(0.0, 1.0)
    }
}

/// Persisted state of the Markov chain.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MarkovState {
    /// transitions[source_app][target_app] = TransitionStats
    pub transitions: HashMap<String, HashMap<String, TransitionStats>>,
    /// Recent time-to-switch per foreground source app. This adapts timing
    /// even when the most likely destination changes.
    #[serde(default)]
    pub source_dwell: HashMap<String, SourceDwellStats>,
    /// Total transitions observed (lifetime counter).
    pub total_transitions: u64,
    #[serde(default)]
    pub prewarm_calibration_schema: u32,
}

// ── Markov Tracker ───────────────────────────────────────────────────────────

pub struct FocusMarkov {
    state: MarkovState,
    persist_path: PathBuf,
    /// Name of the app that was in the foreground last cycle.
    last_app: Option<String>,
    /// When the current foreground app became active (for dwell time).
    last_switch_at: Option<std::time::Instant>,
    /// Dirty flag: state changed since last persist.
    dirty: bool,
    /// How many transitions since last persist (batch writes).
    transitions_since_persist: u32,
}

impl FocusMarkov {
    /// Create a new tracker, loading persisted state if available.
    pub fn new(persist_path: PathBuf) -> Self {
        let mut state = Self::load_state(&persist_path).unwrap_or_default();
        let migrated = sanitize_state(&mut state);
        let mut tracker = Self {
            state,
            persist_path,
            last_app: None,
            last_switch_at: None,
            dirty: migrated,
            transitions_since_persist: 0,
        };
        if migrated {
            tracker.persist();
        }
        tracker
    }

    /// Record a foreground app observation. Call every daemon cycle.
    ///
    /// If the foreground app changed since last call, records the transition.
    /// Returns the predicted next app (if confidence is sufficient).
    pub fn observe(&mut self, current_app: Option<&str>) -> Option<FocusPrediction> {
        let now = std::time::Instant::now();

        let current = match current_app {
            Some(name) if !name.is_empty() => name,
            _ => {
                // Screen locked or no app — don't record, but keep state.
                return None;
            }
        };

        match &self.last_app {
            Some(prev) if prev != current => {
                // Transition detected: prev → current
                let dwell_secs = self
                    .last_switch_at
                    .map(|t| now.duration_since(t).as_secs_f64())
                    .unwrap_or(0.0);

                self.record_transition(prev.clone(), current.to_string(), dwell_secs);

                self.last_app = Some(current.to_string());
                self.last_switch_at = Some(now);
            }
            None => {
                // First observation — initialize.
                self.last_app = Some(current.to_string());
                self.last_switch_at = Some(now);
            }
            _ => {
                // Same app as last cycle — no transition.
            }
        }

        // Return prediction for what comes after the current app.
        self.predict(current)
    }

    /// Record a transition from `from` to `to` with the given dwell time.
    fn record_transition(&mut self, from: String, to: String, dwell_secs: f64) {
        self.state
            .source_dwell
            .entry(from.clone())
            .or_default()
            .observe(dwell_secs);
        let targets = self.state.transitions.entry(from).or_default();

        if let Some(stats) = targets.get_mut(&to) {
            let dwell_secs = sanitize_dwell_sample(dwell_secs);
            stats.count = stats.count.saturating_add(1);
            stats.total_dwell_secs += dwell_secs;
            if stats.recent_dwell_observations == 0 {
                stats.recent_dwell_secs = dwell_secs;
                stats.recent_dwell_deviation_secs = 0.0;
            } else {
                let previous = stats.recent_dwell_secs;
                stats.recent_dwell_secs = previous + DWELL_EWMA_ALPHA * (dwell_secs - previous);
                let absolute_error = (dwell_secs - previous).abs();
                stats.recent_dwell_deviation_secs = stats.recent_dwell_deviation_secs
                    + DWELL_EWMA_ALPHA * (absolute_error - stats.recent_dwell_deviation_secs);
            }
            stats.recent_dwell_observations = stats.recent_dwell_observations.saturating_add(1);
        } else {
            // Evict least-used target if at capacity.
            if targets.len() >= MAX_TARGETS_PER_SOURCE {
                if let Some(min_key) = targets
                    .iter()
                    .min_by_key(|(_, v)| v.count)
                    .map(|(k, _)| k.clone())
                {
                    targets.remove(&min_key);
                }
            }
            targets.insert(to, TransitionStats::new(dwell_secs));
        }

        self.state.total_transitions += 1;
        self.dirty = true;
        self.transitions_since_persist += 1;

        // Evict least-used source app if at capacity.
        if self.state.transitions.len() > MAX_TRACKED_APPS {
            if let Some(min_key) = self
                .state
                .transitions
                .iter()
                .min_by_key(|(_, targets)| targets.values().map(|t| t.count).sum::<u32>())
                .map(|(k, _)| k.clone())
            {
                self.state.source_dwell.remove(&min_key);
                self.state.transitions.remove(&min_key);
            }
        }

        // Batch persist every 10 transitions (not every cycle).
        if self.transitions_since_persist >= 10 {
            self.persist();
        }
    }

    /// Predict the most likely next app given the current foreground.
    pub fn predict(&self, current_app: &str) -> Option<FocusPrediction> {
        let targets = self.state.transitions.get(current_app)?;

        let total: u32 = targets.values().map(|t| t.count).sum();
        if total < MIN_TRANSITIONS_FOR_PREDICTION {
            return None; // Not enough data.
        }

        // Find the most likely target.
        let (best_name, best_stats) = targets.iter().max_by_key(|(_, v)| v.count)?;

        let probability = best_stats.count as f64 / total as f64;
        if probability < MIN_CONFIDENCE {
            return None; // Not confident enough.
        }

        let source_timing = self
            .state
            .source_dwell
            .get(current_app)
            .filter(|timing| timing.observations > 0);
        Some(FocusPrediction {
            app_name: best_name.clone(),
            probability,
            avg_dwell_secs: source_timing
                .map(|timing| timing.recent_dwell_secs)
                .unwrap_or_else(|| best_stats.predicted_dwell_secs()),
            dwell_observations: source_timing
                .map(|timing| timing.observations)
                .unwrap_or(best_stats.recent_dwell_observations),
            dwell_deviation_secs: source_timing
                .map(|timing| timing.recent_dwell_deviation_secs)
                .unwrap_or(best_stats.recent_dwell_deviation_secs),
            pid: None, // Caller fills this in from the process table.
        })
    }

    /// Get top-N predictions for the current app (for observability/logging).
    pub fn predict_top_n(&self, current_app: &str, n: usize) -> Vec<FocusPrediction> {
        if n == 0 {
            return Vec::new();
        }
        let targets = match self.state.transitions.get(current_app) {
            Some(t) => t,
            None => return vec![],
        };

        let total: u32 = targets.values().map(|t| t.count).sum();
        if total < MIN_TRANSITIONS_FOR_PREDICTION {
            return vec![];
        }

        let source_timing = self
            .state
            .source_dwell
            .get(current_app)
            .filter(|timing| timing.observations > 0);
        let mut predictions: Vec<FocusPrediction> = targets
            .iter()
            .map(|(name, stats)| FocusPrediction {
                app_name: name.clone(),
                probability: stats.count as f64 / total as f64,
                avg_dwell_secs: source_timing
                    .map(|timing| timing.recent_dwell_secs)
                    .unwrap_or_else(|| stats.predicted_dwell_secs()),
                dwell_observations: source_timing
                    .map(|timing| timing.observations)
                    .unwrap_or(stats.recent_dwell_observations),
                dwell_deviation_secs: source_timing
                    .map(|timing| timing.recent_dwell_deviation_secs)
                    .unwrap_or(stats.recent_dwell_deviation_secs),
                pid: None,
            })
            .collect();

        let by_probability = |a: &FocusPrediction, b: &FocusPrediction| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        };
        if predictions.len() > n {
            predictions.select_nth_unstable_by(n, by_probability);
            predictions.truncate(n);
        }
        predictions.sort_by(by_probability);
        predictions
    }

    /// Total transitions observed (lifetime).
    pub fn total_transitions(&self) -> u64 {
        self.state.total_transitions
    }

    /// Number of unique source apps tracked.
    pub fn tracked_apps(&self) -> usize {
        self.state.transitions.len()
    }

    /// Admission for one source→target speculative acceleration. A poor pair
    /// is isolated without suppressing the Markov prediction itself or other
    /// transitions, and receives exponentially spaced probes after quarantine.
    pub fn prewarm_admission(&self, source: &str, target: &str) -> PrewarmAdmission {
        let Some(stats) = self
            .state
            .transitions
            .get(source)
            .and_then(|targets| targets.get(target))
        else {
            return PrewarmAdmission::Ready;
        };
        calibration_admission(
            stats.prewarm_trials,
            stats.prewarm_reliability,
            stats.prewarm_quarantine_until_transition,
            self.state.total_transitions,
            PREWARM_MIN_TRIALS,
        )
    }

    pub fn prewarm_admission_with_context(
        &self,
        source: &str,
        target: &str,
        context: PrewarmContext,
    ) -> PrewarmAdmission {
        let Some(stats) = self
            .state
            .transitions
            .get(source)
            .and_then(|targets| targets.get(target))
        else {
            return PrewarmAdmission::Ready;
        };
        let global_admission = self.prewarm_admission(source, target);
        let Some(context_stats) = stats.prewarm_contexts.get(&context.key) else {
            // A new context may collect bounded cache-only probes even when an
            // old global calibration is quarantined.
            return if global_admission == PrewarmAdmission::Quarantined {
                PrewarmAdmission::Probe
            } else {
                global_admission
            };
        };
        if context_stats.trials < PREWARM_MIN_CONTEXT_TRIALS {
            return PrewarmAdmission::Probe;
        }
        calibration_admission(
            context_stats.trials,
            context_stats.reliability,
            context_stats.quarantine_until_transition,
            self.state.total_transitions,
            PREWARM_MIN_CONTEXT_TRIALS,
        )
    }

    pub fn prewarm_reliability(&self, source: &str, target: &str) -> f64 {
        self.state
            .transitions
            .get(source)
            .and_then(|targets| targets.get(target))
            .map(|stats| stats.prewarm_reliability.clamp(0.0, 1.0))
            .unwrap_or(1.0)
    }

    pub fn prewarm_reliability_with_context(
        &self,
        source: &str,
        target: &str,
        context: PrewarmContext,
    ) -> f64 {
        let Some(stats) = self
            .state
            .transitions
            .get(source)
            .and_then(|targets| targets.get(target))
        else {
            return 1.0;
        };
        let global = stats.prewarm_reliability.clamp(0.0, 1.0);
        let Some(context_stats) = stats.prewarm_contexts.get(&context.key) else {
            return global;
        };
        let weight = context_stats.trials as f64
            / (context_stats.trials as f64 + PREWARM_CONTEXT_PRIOR_WEIGHT);
        (global * (1.0 - weight) + context_stats.reliability.clamp(0.0, 1.0) * weight)
            .clamp(0.0, 1.0)
    }

    pub fn prewarm_context_trials(
        &self,
        source: &str,
        target: &str,
        context: PrewarmContext,
    ) -> u32 {
        self.state
            .transitions
            .get(source)
            .and_then(|targets| targets.get(target))
            .and_then(|stats| stats.prewarm_contexts.get(&context.key))
            .map(|stats| stats.trials)
            .unwrap_or(0)
    }

    /// Remaining foreground transitions before a quarantined pair can run a
    /// bounded probe. Zero means ready/probe-now or no calibration exists.
    pub fn prewarm_probe_transitions_remaining(
        &self,
        source: &str,
        target: &str,
        context: PrewarmContext,
    ) -> u64 {
        let Some(stats) = self
            .state
            .transitions
            .get(source)
            .and_then(|targets| targets.get(target))
        else {
            return 0;
        };
        let until = stats
            .prewarm_contexts
            .get(&context.key)
            .filter(|context_stats| context_stats.trials >= PREWARM_MIN_CONTEXT_TRIALS)
            .map(|context_stats| context_stats.quarantine_until_transition)
            .unwrap_or(stats.prewarm_quarantine_until_transition);
        until.saturating_sub(self.state.total_transitions)
    }

    /// Feed a resolved acceleration lease back into its transition-local
    /// calibration. This does not alter transition probabilities.
    pub fn record_prewarm_outcome(&mut self, source: &str, target: &str, hit: bool) {
        self.record_prewarm_outcome_inner(source, target, hit, None);
    }

    pub fn record_prewarm_outcome_with_context(
        &mut self,
        source: &str,
        target: &str,
        hit: bool,
        context: PrewarmContext,
    ) {
        self.record_prewarm_outcome_inner(source, target, hit, Some(context));
    }

    fn record_prewarm_outcome_inner(
        &mut self,
        source: &str,
        target: &str,
        hit: bool,
        context: Option<PrewarmContext>,
    ) {
        let global_admission_before = self.prewarm_admission(source, target);
        let context_admission_before = context
            .map(|context| self.prewarm_admission_with_context(source, target, context))
            .unwrap_or(global_admission_before);
        let total_transitions = self.state.total_transitions;
        let Some(stats) = self
            .state
            .transitions
            .get_mut(source)
            .and_then(|targets| targets.get_mut(target))
        else {
            return;
        };

        let prior_reliability = stats.prewarm_reliability.clamp(0.0, 1.0);
        update_calibration(
            &mut stats.prewarm_trials,
            &mut stats.prewarm_hits,
            &mut stats.prewarm_reliability,
            &mut stats.prewarm_quarantine_until_transition,
            &mut stats.prewarm_backoff_level,
            hit,
            global_admission_before,
            total_transitions,
            PREWARM_MIN_TRIALS,
        );

        if let Some(context) = context {
            if !stats.prewarm_contexts.contains_key(&context.key)
                && stats.prewarm_contexts.len() >= MAX_PREWARM_CONTEXTS_PER_TRANSITION
            {
                if let Some(evict) = stats
                    .prewarm_contexts
                    .iter()
                    .min_by_key(|(_, calibration)| calibration.trials)
                    .map(|(key, _)| *key)
                {
                    stats.prewarm_contexts.remove(&evict);
                }
            }
            let context_stats = stats
                .prewarm_contexts
                .entry(context.key)
                .or_insert_with(|| PrewarmCalibrationStats::new(prior_reliability));
            update_calibration(
                &mut context_stats.trials,
                &mut context_stats.hits,
                &mut context_stats.reliability,
                &mut context_stats.quarantine_until_transition,
                &mut context_stats.backoff_level,
                hit,
                context_admission_before,
                total_transitions,
                PREWARM_MIN_CONTEXT_TRIALS,
            );
        }
        self.dirty = true;
    }

    /// How long the current foreground app has been in focus (seconds).
    /// Returns 0.0 if no foreground app has been observed yet.
    pub fn elapsed_dwell_secs(&self) -> f64 {
        self.last_switch_at
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Persist state to disk (if dirty).
    pub fn persist(&mut self) {
        if !self.dirty {
            return;
        }
        if let Ok(json) = serde_json::to_string(&self.state) {
            let _ = std::fs::write(&self.persist_path, json);
        }
        self.dirty = false;
        self.transitions_since_persist = 0;
    }

    fn load_state(path: &Path) -> Option<MarkovState> {
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }
}

fn calibration_admission(
    trials: u32,
    reliability: f64,
    quarantine_until_transition: u64,
    total_transitions: u64,
    min_trials: u32,
) -> PrewarmAdmission {
    if trials < min_trials {
        PrewarmAdmission::Probe
    } else if reliability >= PREWARM_RECOVERY_THRESHOLD {
        PrewarmAdmission::Ready
    } else if quarantine_until_transition == 0 || total_transitions >= quarantine_until_transition {
        PrewarmAdmission::Probe
    } else {
        PrewarmAdmission::Quarantined
    }
}

#[allow(clippy::too_many_arguments)]
fn update_calibration(
    trials: &mut u32,
    hits: &mut u32,
    reliability: &mut f64,
    quarantine_until_transition: &mut u64,
    backoff_level: &mut u8,
    hit: bool,
    admission_before: PrewarmAdmission,
    total_transitions: u64,
    min_trials: u32,
) {
    let was_mature = *trials >= min_trials;
    *trials = trials.saturating_add(1);
    *hits = hits.saturating_add(u32::from(hit));
    *reliability = (PREWARM_OUTCOME_ALPHA * f64::from(hit)
        + (1.0 - PREWARM_OUTCOME_ALPHA) * reliability.clamp(0.0, 1.0))
    .clamp(0.0, 1.0);

    if hit && admission_before == PrewarmAdmission::Probe && was_mature {
        // A spaced probe hit is fresh evidence under the current workload.
        // Let the next eligible lease be trusted, while later misses can
        // immediately demote it again.
        *reliability = reliability.max(PREWARM_RECOVERY_THRESHOLD);
        *quarantine_until_transition = 0;
        *backoff_level = backoff_level.saturating_sub(1);
    } else if hit && *reliability >= PREWARM_RECOVERY_THRESHOLD {
        *quarantine_until_transition = 0;
        *backoff_level = backoff_level.saturating_sub(1);
    } else if *trials >= min_trials && *reliability < PREWARM_QUARANTINE_THRESHOLD {
        if admission_before == PrewarmAdmission::Probe && was_mature {
            *backoff_level = backoff_level
                .saturating_add(1)
                .min(PREWARM_MAX_BACKOFF_SHIFT);
        }
        let backoff = PREWARM_PROBE_BASE_TRANSITIONS.saturating_mul(1_u64 << *backoff_level);
        *quarantine_until_transition = total_transitions.saturating_add(backoff);
    }
}

fn sanitize_dwell_sample(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, MAX_DWELL_SAMPLE_SECS)
    } else {
        0.0
    }
}

fn sanitize_state(state: &mut MarkovState) -> bool {
    let reset_calibration = state.prewarm_calibration_schema < PREWARM_CALIBRATION_SCHEMA;
    for targets in state.transitions.values_mut() {
        for stats in targets.values_mut() {
            if !stats.total_dwell_secs.is_finite() || stats.total_dwell_secs < 0.0 {
                stats.total_dwell_secs = 0.0;
            }
            if !stats.recent_dwell_secs.is_finite() || stats.recent_dwell_secs < 0.0 {
                stats.recent_dwell_secs = 0.0;
                stats.recent_dwell_observations = 0;
            }
            if !stats.recent_dwell_deviation_secs.is_finite()
                || stats.recent_dwell_deviation_secs < 0.0
            {
                stats.recent_dwell_deviation_secs = 0.0;
            }
            stats.recent_dwell_secs = stats.recent_dwell_secs.min(MAX_DWELL_SAMPLE_SECS);
            stats.recent_dwell_deviation_secs =
                stats.recent_dwell_deviation_secs.min(MAX_DWELL_SAMPLE_SECS);
            if reset_calibration {
                stats.prewarm_trials = 0;
                stats.prewarm_hits = 0;
                stats.prewarm_reliability = default_prewarm_reliability();
                stats.prewarm_quarantine_until_transition = 0;
                stats.prewarm_backoff_level = 0;
                stats.prewarm_contexts.clear();
                continue;
            }
            stats.prewarm_hits = stats.prewarm_hits.min(stats.prewarm_trials);
            stats.prewarm_reliability = if stats.prewarm_reliability.is_finite() {
                stats.prewarm_reliability.clamp(0.0, 1.0)
            } else {
                default_prewarm_reliability()
            };
            stats.prewarm_backoff_level =
                stats.prewarm_backoff_level.min(PREWARM_MAX_BACKOFF_SHIFT);
            let global_reliability = stats.prewarm_reliability;
            stats.prewarm_contexts.retain(|key, calibration| {
                if *key >= PREWARM_CONTEXT_KEY_LIMIT {
                    return false;
                }
                calibration.hits = calibration.hits.min(calibration.trials);
                calibration.reliability = if calibration.reliability.is_finite() {
                    calibration.reliability.clamp(0.0, 1.0)
                } else {
                    global_reliability
                };
                calibration.backoff_level =
                    calibration.backoff_level.min(PREWARM_MAX_BACKOFF_SHIFT);
                true
            });
            while stats.prewarm_contexts.len() > MAX_PREWARM_CONTEXTS_PER_TRANSITION {
                let Some(evict) = stats
                    .prewarm_contexts
                    .iter()
                    .min_by_key(|(_, calibration)| calibration.trials)
                    .map(|(key, _)| *key)
                else {
                    break;
                };
                stats.prewarm_contexts.remove(&evict);
            }
        }
    }
    state.source_dwell.retain(|source, timing| {
        if source.is_empty()
            || !timing.recent_dwell_secs.is_finite()
            || timing.recent_dwell_secs < 0.0
            || !timing.recent_dwell_deviation_secs.is_finite()
            || timing.recent_dwell_deviation_secs < 0.0
        {
            return false;
        }
        timing.recent_dwell_secs = timing.recent_dwell_secs.min(MAX_DWELL_SAMPLE_SECS);
        timing.recent_dwell_deviation_secs = timing
            .recent_dwell_deviation_secs
            .min(MAX_DWELL_SAMPLE_SECS);
        true
    });
    state.prewarm_calibration_schema = PREWARM_CALIBRATION_SCHEMA;
    reset_calibration
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_markov() -> FocusMarkov {
        // Use a unique path per test invocation to avoid cross-test contamination.
        let path = std::env::temp_dir().join(format!(
            "test_markov_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        FocusMarkov::new(path)
    }

    #[test]
    fn no_prediction_cold_start() {
        let m = test_markov();
        assert!(m.predict("Claude").is_none());
    }

    #[test]
    fn learns_transitions() {
        let mut m = test_markov();

        // Simulate: Claude → Brave → Claude → Brave → Claude → Brave (5 transitions)
        for i in 0..10 {
            let app = if i % 2 == 0 { "Claude" } else { "Brave" };
            m.observe(Some(app));
        }

        let pred = m.predict("Claude");
        assert!(pred.is_some(), "should predict after 5 transitions");
        let pred = pred.unwrap();
        assert_eq!(pred.app_name, "Brave");
        assert!(pred.probability >= 0.9, "Claude→Brave should be ~100%");
    }

    #[test]
    fn mixed_transitions() {
        let mut m = test_markov();

        // Claude → Brave (3x), Claude → Terminal (2x)
        let sequence = [
            "Claude", "Brave", "Claude", "Brave", "Claude", "Terminal", "Claude", "Brave",
            "Claude", "Terminal", "Claude",
        ];
        for app in &sequence {
            m.observe(Some(app));
        }

        let pred = m.predict("Claude").unwrap();
        assert_eq!(pred.app_name, "Brave", "Brave should win (3 vs 2)");
        assert!(
            pred.probability > 0.5 && pred.probability < 0.7,
            "probability should be ~0.6, got {}",
            pred.probability
        );
    }

    #[test]
    fn repeated_prewarm_misses_quarantine_only_that_transition() {
        let mut m = test_markov();
        m.state.transitions.insert(
            "Finder".to_string(),
            HashMap::from([("Terminal".to_string(), TransitionStats::new(5.0))]),
        );
        m.state.transitions.insert(
            "Safari".to_string(),
            HashMap::from([("Mail".to_string(), TransitionStats::new(5.0))]),
        );

        for _ in 0..5 {
            m.record_prewarm_outcome("Finder", "Terminal", false);
        }

        assert_eq!(
            m.prewarm_admission("Finder", "Terminal"),
            PrewarmAdmission::Quarantined
        );
        assert_eq!(
            m.prewarm_admission("Safari", "Mail"),
            PrewarmAdmission::Probe,
            "cold transitions may cache-probe but cannot mutate kernel QoS"
        );
    }

    #[test]
    fn quarantined_transition_gets_a_spaced_probe_and_can_recover() {
        let mut m = test_markov();
        m.state.total_transitions = 100;
        m.state.transitions.insert(
            "Finder".to_string(),
            HashMap::from([("Terminal".to_string(), TransitionStats::new(5.0))]),
        );
        for _ in 0..5 {
            m.record_prewarm_outcome("Finder", "Terminal", false);
        }
        assert_eq!(
            m.prewarm_admission("Finder", "Terminal"),
            PrewarmAdmission::Quarantined
        );

        m.state.total_transitions += PREWARM_PROBE_BASE_TRANSITIONS;
        assert_eq!(
            m.prewarm_admission("Finder", "Terminal"),
            PrewarmAdmission::Probe
        );
        m.record_prewarm_outcome("Finder", "Terminal", true);
        assert_eq!(
            m.prewarm_admission("Finder", "Terminal"),
            PrewarmAdmission::Ready
        );
    }

    #[test]
    fn mature_contexts_calibrate_independently_without_bypassing_warmup() {
        let mut m = test_markov();
        m.state.total_transitions = 100;
        m.state.transitions.insert(
            "Finder".to_string(),
            HashMap::from([("Terminal".to_string(), TransitionStats::new(5.0))]),
        );
        let poor = PrewarmContext::new("idle", 2, 0.20, false);
        let useful = PrewarmContext::new("build", 14, 0.52, true);

        for _ in 0..5 {
            m.record_prewarm_outcome_with_context("Finder", "Terminal", false, poor);
        }
        assert_eq!(
            m.prewarm_admission_with_context("Finder", "Terminal", poor),
            PrewarmAdmission::Quarantined
        );

        for _ in 0..4 {
            m.record_prewarm_outcome_with_context("Finder", "Terminal", true, useful);
        }
        assert_eq!(m.prewarm_context_trials("Finder", "Terminal", useful), 4);
        assert_eq!(
            m.prewarm_admission_with_context("Finder", "Terminal", useful),
            PrewarmAdmission::Probe,
            "an immature context gets bounded cache-only probes"
        );
        m.record_prewarm_outcome_with_context("Finder", "Terminal", true, useful);

        assert_eq!(
            m.prewarm_admission_with_context("Finder", "Terminal", poor),
            PrewarmAdmission::Quarantined
        );
        assert_eq!(
            m.prewarm_admission_with_context("Finder", "Terminal", useful),
            PrewarmAdmission::Ready
        );
        assert!(
            m.prewarm_reliability_with_context("Finder", "Terminal", useful)
                > m.prewarm_reliability_with_context("Finder", "Terminal", poor)
        );
    }

    #[test]
    fn context_calibration_is_bounded_per_transition() {
        let mut m = test_markov();
        m.state.transitions.insert(
            "Finder".to_string(),
            HashMap::from([("Terminal".to_string(), TransitionStats::new(5.0))]),
        );
        for index in 0..24 {
            let workload = match index % 6 {
                0 => "idle",
                1 => "browsing",
                2 => "build",
                3 => "llm-inference",
                4 => "mediaplayback",
                _ => "other",
            };
            let context = PrewarmContext::new(
                workload,
                (index % 24) as u8,
                match index % 3 {
                    0 => 0.20,
                    1 => 0.50,
                    _ => 0.75,
                },
                index % 2 == 0,
            );
            m.record_prewarm_outcome_with_context("Finder", "Terminal", true, context);
        }
        let stats = &m.state.transitions["Finder"]["Terminal"];
        assert!(stats.prewarm_contexts.len() <= MAX_PREWARM_CONTEXTS_PER_TRANSITION);
    }

    #[test]
    fn cold_transition_starts_as_probe_not_full_acceleration() {
        let mut m = test_markov();
        m.state.transitions.insert(
            "Finder".to_string(),
            HashMap::from([("Terminal".to_string(), TransitionStats::new(5.0))]),
        );
        assert_eq!(
            m.prewarm_admission("Finder", "Terminal"),
            PrewarmAdmission::Probe
        );
        assert!(!m
            .prewarm_admission("Finder", "Terminal")
            .allows_kernel_acceleration());
    }

    #[test]
    fn intent_horizon_confidence_is_bounded_and_monotonic() {
        let prediction = FocusPrediction {
            app_name: "Terminal".to_string(),
            probability: 0.80,
            avg_dwell_secs: 90.0,
            dwell_deviation_secs: 12.0,
            dwell_observations: 40,
            pid: Some(42),
        };
        let horizons =
            [5.0, 30.0, 120.0, 600.0].map(|horizon| prediction.confidence_within(20.0, horizon));
        assert!(horizons.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(horizons.iter().all(|value| (0.0..=0.80).contains(value)));
        assert!(horizons[0] < 0.10);
        assert!(horizons[3] > 0.79);
        assert_eq!(prediction.confidence_within(f64::NAN, 30.0), 0.0);
        assert_eq!(prediction.confidence_within(0.0, 0.0), 0.0);
    }

    #[test]
    fn probe_hit_recovers_even_from_a_saturated_old_penalty() {
        let mut m = test_markov();
        m.state.total_transitions = 200;
        let mut stats = TransitionStats::new(5.0);
        stats.prewarm_trials = 100;
        stats.prewarm_reliability = 1e-12;
        stats.prewarm_quarantine_until_transition = 200;
        stats.prewarm_backoff_level = PREWARM_MAX_BACKOFF_SHIFT;
        m.state.transitions.insert(
            "Finder".to_string(),
            HashMap::from([("Terminal".to_string(), stats)]),
        );
        assert_eq!(
            m.prewarm_admission("Finder", "Terminal"),
            PrewarmAdmission::Probe
        );
        m.record_prewarm_outcome("Finder", "Terminal", true);
        assert_eq!(
            m.prewarm_admission("Finder", "Terminal"),
            PrewarmAdmission::Ready
        );
    }

    #[test]
    fn schema_migration_preserves_transitions_and_resets_stale_calibration() {
        let mut stats = TransitionStats::new(210.0);
        stats.count = 20;
        stats.total_dwell_secs = 4_200.0;
        stats.recent_dwell_observations = 0;
        stats.prewarm_trials = 40;
        stats.prewarm_hits = 1;
        stats.prewarm_reliability = 1e-8;
        stats.prewarm_quarantine_until_transition = 9_999;
        stats.prewarm_backoff_level = 4;
        let mut state = MarkovState {
            transitions: HashMap::from([(
                "Finder".to_string(),
                HashMap::from([("Terminal".to_string(), stats)]),
            )]),
            source_dwell: HashMap::new(),
            total_transitions: 500,
            prewarm_calibration_schema: 1,
        };

        assert!(sanitize_state(&mut state));
        let migrated = &state.transitions["Finder"]["Terminal"];
        assert_eq!(migrated.count, 20);
        assert_eq!(migrated.total_dwell_secs, 4_200.0);
        assert_eq!(migrated.prewarm_trials, 0);
        assert_eq!(migrated.prewarm_reliability, default_prewarm_reliability());
        assert_eq!(migrated.prewarm_quarantine_until_transition, 0);
        assert_eq!(state.prewarm_calibration_schema, PREWARM_CALIBRATION_SCHEMA);
    }

    #[test]
    fn first_local_dwell_sample_replaces_legacy_timing() {
        let mut m = test_markov();
        let mut legacy = TransitionStats::new(210.0);
        legacy.count = 20;
        legacy.total_dwell_secs = 4_200.0;
        legacy.recent_dwell_secs = 0.0;
        legacy.recent_dwell_observations = 0;
        m.state.transitions.insert(
            "Finder".to_string(),
            HashMap::from([("Terminal".to_string(), legacy)]),
        );

        m.record_transition("Finder".to_string(), "Terminal".to_string(), 12.0);
        let prediction = m.predict("Finder").expect("prediction");
        assert_eq!(prediction.dwell_observations, 1);
        assert!((prediction.avg_dwell_secs - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn source_timing_adapts_even_when_the_destination_differs() {
        let mut m = test_markov();
        let mut legacy = TransitionStats::new(210.0);
        legacy.count = 20;
        legacy.total_dwell_secs = 4_200.0;
        legacy.recent_dwell_secs = 0.0;
        legacy.recent_dwell_observations = 0;
        m.state.transitions.insert(
            "Finder".to_string(),
            HashMap::from([("Terminal".to_string(), legacy)]),
        );

        m.record_transition("Finder".to_string(), "Safari".to_string(), 9.0);
        let prediction = m.predict("Finder").expect("legacy destination still wins");
        assert_eq!(prediction.app_name, "Terminal");
        assert_eq!(prediction.dwell_observations, 1);
        assert!((prediction.avg_dwell_secs - 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn respects_min_confidence() {
        let mut m = test_markov();

        // 5 transitions all to different apps — no clear winner
        let sequence = ["A", "B", "A", "C", "A", "D", "A", "E", "A", "F", "A"];
        for app in &sequence {
            m.observe(Some(app));
        }

        let pred = m.predict("A");
        // Each target has 1/5 = 0.20 probability < MIN_CONFIDENCE (0.30)
        assert!(pred.is_none(), "no prediction when too spread out");
    }

    #[test]
    fn predict_top_n() {
        let mut m = test_markov();

        // Build up: Claude→Brave(4), Claude→Terminal(3), Claude→Finder(1)
        let sequence = [
            "Claude", "Brave", "Claude", "Brave", "Claude", "Terminal", "Claude", "Brave",
            "Claude", "Terminal", "Claude", "Brave", "Claude", "Terminal", "Claude", "Finder",
            "Claude",
        ];
        for app in &sequence {
            m.observe(Some(app));
        }

        let top = m.predict_top_n("Claude", 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].app_name, "Brave");
        assert_eq!(top[1].app_name, "Terminal");
        assert_eq!(top[2].app_name, "Finder");
    }

    #[test]
    fn idle_does_not_break_chain() {
        let mut m = test_markov();

        m.observe(Some("Claude"));
        m.observe(None); // Screen locked
        m.observe(None); // Still locked
        m.observe(Some("Brave")); // Back — should NOT record None→Brave

        // Only 1 transition: Claude→Brave
        assert_eq!(m.state.total_transitions, 1);
    }

    #[test]
    fn persistence_roundtrip() {
        let path = PathBuf::from("/tmp/test_markov_persist.json");
        let _ = std::fs::remove_file(&path);

        {
            let mut m = FocusMarkov::new(path.clone());
            for i in 0..12 {
                let app = if i % 2 == 0 { "A" } else { "B" };
                m.observe(Some(app));
            }
            m.persist();
        }

        // Reload
        let m2 = FocusMarkov::new(path.clone());
        assert!(m2.state.total_transitions >= 5);
        let pred = m2.predict("A");
        assert!(pred.is_some());

        let _ = std::fs::remove_file(&path);
    }

    // ── Additional untested paths ─────────────────────────────────────────────

    #[test]
    fn elapsed_dwell_secs_zero_before_first_observation() {
        let m = test_markov();
        // No observation yet — elapsed should be 0.0
        assert_eq!(m.elapsed_dwell_secs(), 0.0);
    }

    #[test]
    fn elapsed_dwell_secs_positive_after_observation() {
        let mut m = test_markov();
        m.observe(Some("Claude"));
        // After observing, dwell time should be non-negative
        assert!(m.elapsed_dwell_secs() >= 0.0);
    }

    #[test]
    fn tracked_apps_zero_initially() {
        let m = test_markov();
        assert_eq!(m.tracked_apps(), 0);
    }

    #[test]
    fn tracked_apps_increases_after_transitions() {
        let mut m = test_markov();
        // Claude → Brave transition
        m.observe(Some("Claude"));
        m.observe(Some("Brave"));
        // At least one source app should now be tracked
        assert!(m.tracked_apps() >= 1, "should track at least 1 source app");
    }

    #[test]
    fn total_transitions_zero_on_first_observation() {
        let mut m = test_markov();
        m.observe(Some("Claude")); // first observation → no transition yet
        assert_eq!(m.total_transitions(), 0);
    }

    #[test]
    fn total_transitions_increments_on_each_switch() {
        let mut m = test_markov();
        m.observe(Some("A"));
        m.observe(Some("B"));
        m.observe(Some("A"));
        m.observe(Some("B"));
        assert_eq!(m.total_transitions(), 3);
    }

    #[test]
    fn predict_top_n_empty_when_no_data() {
        let m = test_markov();
        let top = m.predict_top_n("Claude", 5);
        assert!(top.is_empty(), "should return empty when no transitions");
    }

    #[test]
    fn predict_top_zero_is_empty() {
        let mut m = test_markov();
        for app in ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A"] {
            m.observe(Some(app));
        }
        assert!(m.predict_top_n("A", 0).is_empty());
    }

    #[test]
    fn predict_top_n_truncates_to_n() {
        let mut m = test_markov();
        // Build many transitions from X to A,B,C,D,E,F
        for &target in &["A", "B", "C", "D", "E", "F"] {
            for _ in 0..2 {
                m.observe(Some("X"));
                m.observe(Some(target));
            }
        }
        // 12 transitions from X, 6 distinct targets — top 3 requested
        let top = m.predict_top_n("X", 3);
        assert!(top.len() <= 3, "predict_top_n should truncate to n");
    }

    #[test]
    fn transition_stats_avg_dwell_secs() {
        let stats = TransitionStats {
            count: 4,
            total_dwell_secs: 100.0,
            ..TransitionStats::new(0.0)
        };
        assert!((stats.avg_dwell_secs() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn transition_stats_avg_dwell_zero_count() {
        let stats = TransitionStats {
            count: 0,
            total_dwell_secs: 0.0,
            ..TransitionStats::new(0.0)
        };
        assert_eq!(stats.avg_dwell_secs(), 0.0);
    }
}
