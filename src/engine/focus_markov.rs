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

// ── Data structures ──────────────────────────────────────────────────────────

/// Statistics for a single A→B transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionStats {
    /// How many times this transition was observed.
    pub count: u32,
    /// Sum of dwell times in the source app before this transition (seconds).
    /// Used to compute average time before switching to this target.
    pub total_dwell_secs: f64,
}

impl TransitionStats {
    fn new(dwell_secs: f64) -> Self {
        Self {
            count: 1,
            total_dwell_secs: dwell_secs,
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
        let (best_name, best_stats) = targets
            .iter()
            .max_by_key(|(_, v)| v.count)?;

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

        predictions.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        predictions.truncate(n);
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
            "Claude", "Brave", "Claude", "Brave", "Claude", "Terminal",
            "Claude", "Brave", "Claude", "Terminal", "Claude",
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
            "Claude", "Brave", "Claude", "Brave", "Claude", "Terminal",
            "Claude", "Brave", "Claude", "Terminal", "Claude", "Brave",
            "Claude", "Terminal", "Claude", "Finder", "Claude",
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
}
