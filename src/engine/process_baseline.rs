//! Per-process hardware counter baseline learning.
//!
//! Accumulates streaming baselines for `{ipc, wakeup_rate, disk_mbps}` per
//! process **name** using EMA + EMA-MAD (mean absolute deviation).
//!
//! # Anomaly scoring
//!
//! For each signal:
//!   `score = |current - ema| / (mad + ε)`
//!
//! This is a scale-free z-score equivalent for streaming data.
//! [Chandola et al. 2009 ACM CSUR "Anomaly Detection: A Survey" §3.1]
//! Composite anomaly = max across all signals (a process anomalous in *any*
//! dimension is interesting; OR-semantics matches battery vampire / I/O burst).
//!
//! # Design choices
//!
//! - Keyed by **name** (not PID): same semantic process across restarts/forks.
//! - `MIN_OBS = 5`: don't score until we've seen the process at least 5 times
//!   to avoid false positives during cold start.
//! - `ALPHA = 0.10`: slow learner → stable baseline; a sudden spike scores high
//!   without immediately collapsing the baseline.
//! - `ANOMALY_THRESHOLD = 3.0`: ~3 MADs above baseline = anomalous. Chosen
//!   empirically: typical process noise is <1.5 MADs; genuine anomalies (backup
//!   starting, JIT burst) appear at 4-10×.
//!
//! # Persistence
//!
//! The full `ProcessBaselineMap` is serializable and stored in `LearnedState`
//! so baselines survive daemon restarts.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Minimum observations before anomaly scoring is active.
/// Prevents false positives during cold start.
const MIN_OBS: u32 = 5;

/// EMA smoothing factor. Low = stable baseline, high = fast adaptation.
/// 0.10 → half-life ≈ 6.6 samples. [Holt 1957] exponential smoothing.
const ALPHA: f64 = 0.10;

/// Anomaly threshold in MAD units.
/// score >= ANOMALY_THRESHOLD → process is anomalous for that signal.
pub const ANOMALY_THRESHOLD: f64 = 3.0;

/// Single-signal streaming baseline: EMA value + EMA of absolute deviation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalBaseline {
    /// Exponential moving average of the signal.
    pub ema: f64,
    /// EMA of |x - ema| (mean absolute deviation, streaming estimate).
    pub mad: f64,
    /// Total observations seen.
    pub obs: u32,
}

impl SignalBaseline {
    fn new() -> Self {
        Self { ema: 0.0, mad: 0.0, obs: 0 }
    }

    /// Update baseline with a new observation.
    /// First observation bootstraps EMA to the value (no cold-start bias from 0.0).
    /// Subsequent updates: compute deviation from OLD ema, then update ema and mad.
    fn update(&mut self, value: f64) {
        if self.obs == 0 {
            // Bootstrap: set EMA to first value to avoid 0→value ramp-up bias.
            self.ema = value;
            self.mad = 0.0;
        } else {
            let dev = (value - self.ema).abs();
            self.ema = ALPHA * value + (1.0 - ALPHA) * self.ema;
            self.mad = ALPHA * dev + (1.0 - ALPHA) * self.mad;
        }
        self.obs += 1;
    }

    /// Deviation score for `current` against this baseline.
    /// Returns 0.0 if not enough observations (cold start).
    /// Scale-free: 1.0 = one MAD away; 3.0 = three MADs (anomalous).
    fn score(&self, current: f64) -> f64 {
        if self.obs < MIN_OBS {
            return 0.0;
        }
        let dev = (current - self.ema).abs();
        dev / (self.mad + 1e-6)
    }
}

impl Default for SignalBaseline {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-process baseline across all tracked hardware counter signals.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessSignals {
    /// IPC (instructions per cycle) from KPC / proc_pid_rusage.
    pub ipc: SignalBaseline,
    /// CPU wakeup rate (idle + interrupt wakeups/s).
    pub wakeup_rate: SignalBaseline,
    /// Disk write rate (MB/s).
    pub disk_mbps: SignalBaseline,
}

impl ProcessSignals {
    /// Update all three baselines with one observation.
    pub fn update(&mut self, ipc: f64, wakeup_rate: f64, disk_mbps: f64) {
        self.ipc.update(ipc);
        self.wakeup_rate.update(wakeup_rate);
        self.disk_mbps.update(disk_mbps);
    }

    /// Composite anomaly score: max deviation across all signals.
    ///
    /// OR-semantics: a process anomalous in *any* dimension is a priority target.
    /// If all signals are below MIN_OBS, returns 0.0 (cold start).
    pub fn anomaly_score(&self, ipc: f64, wakeup_rate: f64, disk_mbps: f64) -> f64 {
        let s_ipc = self.ipc.score(ipc);
        let s_wk = self.wakeup_rate.score(wakeup_rate);
        let s_disk = self.disk_mbps.score(disk_mbps);
        s_ipc.max(s_wk).max(s_disk)
    }

    /// Which signal is the primary driver of the anomaly.
    pub fn dominant_signal(&self, ipc: f64, wakeup_rate: f64, disk_mbps: f64) -> &'static str {
        let s_ipc = self.ipc.score(ipc);
        let s_wk = self.wakeup_rate.score(wakeup_rate);
        let s_disk = self.disk_mbps.score(disk_mbps);
        if s_disk >= s_ipc && s_disk >= s_wk {
            "disk"
        } else if s_wk >= s_ipc {
            "wakeup"
        } else {
            "ipc"
        }
    }

    /// Total observations (minimum across signals — weakest link).
    pub fn min_obs(&self) -> u32 {
        self.ipc.obs.min(self.wakeup_rate.obs).min(self.disk_mbps.obs)
    }
}

/// Map of process name → learned signal baseline.
///
/// Persisted in `LearnedState` so baselines survive daemon restarts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessBaselineMap {
    /// Process name → per-signal baseline.
    pub entries: HashMap<String, ProcessSignals>,
}

impl ProcessBaselineMap {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    /// Update the baseline for `name` with a new observation.
    /// Creates an entry on first observation (cold start).
    pub fn observe(&mut self, name: &str, ipc: f64, wakeup_rate: f64, disk_mbps: f64) {
        let entry = self.entries.entry(name.to_string()).or_default();
        entry.update(ipc, wakeup_rate, disk_mbps);
    }

    /// Anomaly score for a process given its current readings.
    /// Returns 0.0 if not enough history (< MIN_OBS cycles).
    pub fn anomaly_score(&self, name: &str, ipc: f64, wakeup_rate: f64, disk_mbps: f64) -> f64 {
        self.entries
            .get(name)
            .map(|e| e.anomaly_score(ipc, wakeup_rate, disk_mbps))
            .unwrap_or(0.0)
    }

    /// Dominant anomaly signal for `name`.
    pub fn dominant_signal(&self, name: &str, ipc: f64, wakeup_rate: f64, disk_mbps: f64) -> Option<&'static str> {
        self.entries.get(name).map(|e| e.dominant_signal(ipc, wakeup_rate, disk_mbps))
    }

    /// Build a pid → anomaly_score map for all processes in `results`.
    ///
    /// Also updates the baselines as a side effect (observe + score in one pass).
    /// Returns only entries with score > 0 (warm baselines, actual anomalies).
    pub fn update_and_score(
        &mut self,
        results: &[crate::engine::energy_pid::ProcessEnergyDelta],
    ) -> HashMap<u32, f64> {
        let mut scores = HashMap::new();
        for r in results {
            self.observe(&r.name, r.ipc, r.wakeup_rate, r.disk_write_mbps);
            let score = self.anomaly_score(&r.name, r.ipc, r.wakeup_rate, r.disk_write_mbps);
            if score > 0.0 {
                scores.insert(r.pid, score);
            }
        }
        scores
    }

    /// Prune entries for processes not seen in the last `max_unseen_cycles` persist cycles.
    /// Called from `LearnedState::self_improve()` to bound map size.
    /// For now: prune entries with 0 observations (should not exist but defensive).
    pub fn prune_stale(&mut self) {
        self.entries.retain(|_, v| v.min_obs() > 0);
    }

    /// Number of entries with warm baselines (>= MIN_OBS).
    pub fn warm_count(&self) -> usize {
        self.entries.values().filter(|v| v.min_obs() >= MIN_OBS).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_baseline_cold_start_returns_zero() {
        let b = SignalBaseline::new();
        // Only 0 obs — should return 0 regardless of value.
        assert_eq!(b.score(999.0), 0.0);
    }

    #[test]
    fn signal_baseline_warms_after_min_obs() {
        let mut b = SignalBaseline::new();
        for _ in 0..MIN_OBS {
            b.update(1.0);
        }
        // After MIN_OBS observations all at 1.0, a current value of 1.0 = no anomaly.
        let score = b.score(1.0);
        assert!(score < 0.5, "stable signal should score near 0, got {}", score);
    }

    #[test]
    fn signal_baseline_spike_scores_high() {
        let mut b = SignalBaseline::new();
        // Warm up on stable value.
        for _ in 0..20 {
            b.update(1.0);
        }
        // A spike of 100× should score very high.
        let score = b.score(100.0);
        assert!(score > ANOMALY_THRESHOLD, "spike should score >{}, got {}", ANOMALY_THRESHOLD, score);
    }

    #[test]
    fn signal_baseline_ema_converges() {
        let mut b = SignalBaseline::new();
        for _ in 0..50 {
            b.update(5.0);
        }
        assert!((b.ema - 5.0).abs() < 0.5, "EMA should converge to 5.0, got {}", b.ema);
    }

    #[test]
    fn process_baseline_map_observe_and_score() {
        let mut map = ProcessBaselineMap::new();
        // Warm up Safari with stable readings.
        for _ in 0..20 {
            map.observe("Safari", 2.0, 50.0, 0.1);
        }
        // Normal reading = low anomaly.
        let normal = map.anomaly_score("Safari", 2.0, 50.0, 0.1);
        assert!(normal < ANOMALY_THRESHOLD, "normal reading should not be anomalous");

        // Disk burst anomaly.
        let anomalous = map.anomaly_score("Safari", 2.0, 50.0, 200.0);
        assert!(anomalous >= ANOMALY_THRESHOLD, "disk burst should be anomalous, got {}", anomalous);
    }

    #[test]
    fn process_baseline_map_unknown_process_returns_zero() {
        let map = ProcessBaselineMap::new();
        assert_eq!(map.anomaly_score("unknown", 1.0, 100.0, 5.0), 0.0);
    }

    #[test]
    fn process_baseline_map_warm_count() {
        let mut map = ProcessBaselineMap::new();
        // 3 observations — not warm yet.
        for _ in 0..3 {
            map.observe("proc_a", 1.0, 10.0, 0.0);
        }
        assert_eq!(map.warm_count(), 0);
        // 5 observations — warm.
        for _ in 0..2 {
            map.observe("proc_a", 1.0, 10.0, 0.0);
        }
        assert_eq!(map.warm_count(), 1);
    }

    #[test]
    fn dominant_signal_identifies_disk_burst() {
        let mut map = ProcessBaselineMap::new();
        for _ in 0..20 {
            map.observe("backup", 2.0, 30.0, 0.5);
        }
        // Disk burst, other signals normal.
        let dom = map.dominant_signal("backup", 2.0, 30.0, 100.0);
        assert_eq!(dom, Some("disk"));
    }

    #[test]
    fn dominant_signal_identifies_wakeup_burst() {
        let mut map = ProcessBaselineMap::new();
        for _ in 0..20 {
            map.observe("some_daemon", 1.5, 20.0, 0.1);
        }
        // Wakeup explosion, others normal.
        let dom = map.dominant_signal("some_daemon", 1.5, 1000.0, 0.1);
        assert_eq!(dom, Some("wakeup"));
    }

    #[test]
    fn update_and_score_returns_warm_only() {
        use crate::engine::energy_pid::ProcessEnergyDelta;
        let mut map = ProcessBaselineMap::new();

        // Pre-warm "Safari" manually.
        for _ in 0..20 {
            map.observe("Safari", 2.0, 50.0, 0.1);
        }

        let results = vec![
            ProcessEnergyDelta {
                pid: 1,
                name: "Safari".into(),
                delta_nj: 1000,
                power_mw: 5.0,
                ipc: 2.0,
                wakeup_rate: 50.0,
                phys_footprint_mb: 300.0,
                disk_write_mbps: 0.1,
                anomaly_score: 0.0,
            },
            ProcessEnergyDelta {
                pid: 2,
                name: "new_proc".into(),
                delta_nj: 500,
                power_mw: 2.0,
                ipc: 1.0,
                wakeup_rate: 10.0,
                phys_footprint_mb: 50.0,
                disk_write_mbps: 0.0,
                anomaly_score: 0.0,
            },
        ];

        let scores = map.update_and_score(&results);
        // Safari reading is exactly on-baseline → score = 0.0 → NOT in map (not anomalous).
        assert!(!scores.contains_key(&1), "Safari at baseline should not be in scores");
        // "new_proc" has only 1 obs — cold start, also not in scores.
        assert!(!scores.contains_key(&2), "new_proc cold start should not be in scores");
        // Warm baseline for Safari is recorded.
        assert!(map.warm_count() >= 1, "Safari baseline should be warm after 20+ obs");
    }

    #[test]
    fn prune_stale_does_not_remove_active_entries() {
        let mut map = ProcessBaselineMap::new();
        map.observe("chrome", 2.0, 30.0, 0.5);
        assert_eq!(map.entries.len(), 1);
        map.prune_stale();
        assert_eq!(map.entries.len(), 1, "active entry should survive prune");
    }

    #[test]
    fn full_pipeline_warm_then_anomaly() {
        // Simulate the full path:
        // 1. Warm up process baseline (20 stable cycles)
        // 2. Observe an anomalous cycle (disk burst)
        // 3. Verify anomaly_score ≥ ANOMALY_THRESHOLD
        // 4. Verify build_anomaly_hints would include this pid
        use crate::engine::energy_pid::ProcessEnergyDelta;

        let mut map = ProcessBaselineMap::new();

        // Step 1: warm baseline on stable values
        for _ in 0..20 {
            map.observe("Spotlight", 1.5, 20.0, 0.1);
        }
        assert_eq!(map.warm_count(), 1, "Spotlight baseline should be warm");

        // Step 2: normal reading — no anomaly
        let normal_score = map.anomaly_score("Spotlight", 1.5, 20.0, 0.1);
        assert!(normal_score < ANOMALY_THRESHOLD, "normal reading should not trigger anomaly");

        // Step 3: disk burst — Spotlight suddenly indexing heavy content
        let burst_score = map.anomaly_score("Spotlight", 1.5, 20.0, 80.0);
        assert!(burst_score >= ANOMALY_THRESHOLD,
            "disk burst should trigger anomaly, got {}", burst_score);

        // Step 4: build_anomaly_hints filters correctly
        let results = vec![
            ProcessEnergyDelta {
                pid: 42,
                name: "Spotlight".into(),
                delta_nj: 0,
                power_mw: 0.0,
                ipc: 1.5,
                wakeup_rate: 20.0,
                phys_footprint_mb: 100.0,
                disk_write_mbps: 80.0,
                anomaly_score: burst_score,
            },
        ];
        // Simulated anomaly_hints build (same logic as main.rs)
        let hints: std::collections::HashMap<u32, f64> = results.iter()
            .filter(|r| r.anomaly_score >= ANOMALY_THRESHOLD)
            .map(|r| (r.pid, r.anomaly_score))
            .collect();
        assert!(hints.contains_key(&42), "Spotlight should appear in anomaly_hints");
        assert!(hints[&42] >= ANOMALY_THRESHOLD);
    }
}
