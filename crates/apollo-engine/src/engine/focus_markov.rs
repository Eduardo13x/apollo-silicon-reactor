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

const PREWARM_MIN_TRIALS: u32 = 5;
const PREWARM_QUARANTINE_THRESHOLD: f64 = 0.35;
const PREWARM_RECOVERY_THRESHOLD: f64 = 0.40;
const PREWARM_OUTCOME_ALPHA: f64 = 0.25;
const PREWARM_PROBE_BASE_TRANSITIONS: u64 = 16;
const PREWARM_MAX_BACKOFF_SHIFT: u8 = 4;

fn default_prewarm_reliability() -> f64 {
    1.0
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
}

impl TransitionStats {
    fn new(dwell_secs: f64) -> Self {
        Self {
            count: 1,
            total_dwell_secs: dwell_secs,
            prewarm_trials: 0,
            prewarm_hits: 0,
            prewarm_reliability: default_prewarm_reliability(),
            prewarm_quarantine_until_transition: 0,
            prewarm_backoff_level: 0,
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
    /// PID of the predicted app (if currently running).
    pub pid: Option<u32>,
}

/// Persisted state of the Markov chain.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MarkovState {
    /// transitions[source_app][target_app] = TransitionStats
    pub transitions: HashMap<String, HashMap<String, TransitionStats>>,
    /// Total transitions observed (lifetime counter).
    pub total_transitions: u64,
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
        let state = Self::load_state(&persist_path).unwrap_or_default();
        Self {
            state,
            persist_path,
            last_app: None,
            last_switch_at: None,
            dirty: false,
            transitions_since_persist: 0,
        }
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
        let targets = self.state.transitions.entry(from).or_default();

        if let Some(stats) = targets.get_mut(&to) {
            stats.count += 1;
            stats.total_dwell_secs += dwell_secs;
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

        Some(FocusPrediction {
            app_name: best_name.clone(),
            probability,
            avg_dwell_secs: best_stats.avg_dwell_secs(),
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

        let mut predictions: Vec<FocusPrediction> = targets
            .iter()
            .map(|(name, stats)| FocusPrediction {
                app_name: name.clone(),
                probability: stats.count as f64 / total as f64,
                avg_dwell_secs: stats.avg_dwell_secs(),
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
        if stats.prewarm_trials < PREWARM_MIN_TRIALS
            || stats.prewarm_reliability >= PREWARM_RECOVERY_THRESHOLD
            || stats.prewarm_quarantine_until_transition == 0
        {
            PrewarmAdmission::Ready
        } else if self.state.total_transitions >= stats.prewarm_quarantine_until_transition {
            PrewarmAdmission::Probe
        } else {
            PrewarmAdmission::Quarantined
        }
    }

    pub fn prewarm_reliability(&self, source: &str, target: &str) -> f64 {
        self.state
            .transitions
            .get(source)
            .and_then(|targets| targets.get(target))
            .map(|stats| stats.prewarm_reliability.clamp(0.0, 1.0))
            .unwrap_or(1.0)
    }

    /// Feed a resolved acceleration lease back into its transition-local
    /// calibration. This does not alter transition probabilities.
    pub fn record_prewarm_outcome(&mut self, source: &str, target: &str, hit: bool) {
        let admission_before = self.prewarm_admission(source, target);
        let total_transitions = self.state.total_transitions;
        let Some(stats) = self
            .state
            .transitions
            .get_mut(source)
            .and_then(|targets| targets.get_mut(target))
        else {
            return;
        };

        stats.prewarm_trials = stats.prewarm_trials.saturating_add(1);
        stats.prewarm_hits = stats.prewarm_hits.saturating_add(u32::from(hit));
        let observed = f64::from(hit);
        stats.prewarm_reliability = (PREWARM_OUTCOME_ALPHA * observed
            + (1.0 - PREWARM_OUTCOME_ALPHA) * stats.prewarm_reliability)
            .clamp(0.0, 1.0);

        if hit && stats.prewarm_reliability >= PREWARM_RECOVERY_THRESHOLD {
            stats.prewarm_quarantine_until_transition = 0;
            stats.prewarm_backoff_level = stats.prewarm_backoff_level.saturating_sub(1);
        } else if stats.prewarm_trials >= PREWARM_MIN_TRIALS
            && stats.prewarm_reliability < PREWARM_QUARANTINE_THRESHOLD
        {
            if admission_before == PrewarmAdmission::Probe {
                stats.prewarm_backoff_level = stats
                    .prewarm_backoff_level
                    .saturating_add(1)
                    .min(PREWARM_MAX_BACKOFF_SHIFT);
            }
            let backoff =
                PREWARM_PROBE_BASE_TRANSITIONS.saturating_mul(1_u64 << stats.prewarm_backoff_level);
            stats.prewarm_quarantine_until_transition = total_transitions.saturating_add(backoff);
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
            PrewarmAdmission::Ready
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
