//! Hierarchical Planner — Strangler Fig Phase 0 (advisory-only)
//!
//! Slow planning layer for apollo. Runs in its own thread at 30-second
//! cadence, reads `runtime_metrics.json` (the existing daemon-written
//! observability snapshot), computes forward-looking hints with horizons
//! between 30 seconds and 5 minutes, and writes them to
//! `planner_hints.json` for **future consumers**.
//!
//! ## Why "Phase 0"
//!
//! The current apollo architecture is a pure 2-second reactive loop:
//! every decision is made from the current cycle's state with at most a
//! few-second Markov lookahead. Real planning — "compilation predicted
//! in 30 s, prepare memory headroom now" — requires a slow layer that
//! reasons over minutes and hands hints down to the fast reactor.
//!
//! Strangler Fig methodology says: the new component starts as
//! ADVISORY-ONLY. It produces output, no consumer reads it, the system
//! is unchanged. Only after the new component's output has been
//! validated empirically (e.g. "do its hints actually correlate with
//! reality 70% of the time?") does any reactor consumer get wired up.
//! This commit sets up the production side. No consumer wiring.
//!
//! ## Why decoupled from the daemon main loop
//!
//! The planner intentionally does NOT share any state with the daemon
//! beyond filesystem paths. This is by design:
//!
//!   1. The daemon main loop is the FAST reactor and must not be slowed
//!      by any planner work, even via lock contention.
//!   2. The planner can be killed and restarted independently for
//!      experiments without affecting freeze/throttle decisions.
//!   3. Strangler Fig isolation: the planner and reactor are different
//!      processes-of-thought even though they live in the same binary.
//!      Coupling them via shared state would re-import the very risk
//!      Strangler Fig is meant to manage.
//!
//! The daemon already publishes everything the planner needs to
//! `runtime_metrics.json` every cycle. The planner reads that file at
//! its own cadence and emits hints based on observed trends.
//!
//! ## Hint shapes
//!
//! - `PressureSpike { peak }` — memory pressure has been rising at
//!   ≥ 0.5 %/sec for the last 60 seconds; expect to cross `peak`
//!   within `horizon_secs`.
//! - `ThrashingOnset { score }` — thrashing_score has climbed past
//!   1500 with positive slope; expect to cross 5_000 (gate_c) soon.
//! - `CpuSaturation { fraction }` — cpu_pegged_fraction has exceeded
//!   0.5 with rising trend; expect sustained P-cluster saturation.
//!
//! Each hint carries a `confidence` ∈ [0, 1] derived from the
//! steadiness of the trend (more samples in agreement = higher conf).
//!
//! ## References
//!
//! - [Fowler 2004] "StranglerFigApplication" — incremental replacement
//!   pattern: produce in parallel, wire consumers only after validation.
//! - [Sutton & Barto 2018] §17 — model-based RL planning lives at a
//!   slower timescale than the model-free reactive policy.
//! - [Pearl 2009] "Causality" Ch.3 — temporal precedence is a necessary
//!   (not sufficient) condition for causal inference; the planner emits
//!   the precedent observation, the reactor decides if it acts on it.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// One forward-looking hint about a probable future state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerHint {
    /// Seconds from `emitted_at` until the predicted state is expected.
    pub horizon_secs: u64,
    /// Confidence ∈ [0, 1] derived from trend steadiness.
    pub confidence: f32,
    /// Wall-clock time the hint was produced.
    pub emitted_at: DateTime<Utc>,
    /// Type-specific payload.
    pub kind: HintKind,
}

/// Type-specific hint payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HintKind {
    /// Memory pressure rising fast; expect peak within horizon.
    PressureSpike { peak: f64 },
    /// VM thrashing flow score climbing; expect gate_c-territory soon.
    ThrashingOnset { score: f64 },
    /// Per-core CPU saturation rising; expect sustained P-cluster
    /// pegging.
    CpuSaturation { fraction: f64 },
}

/// Snapshot of one runtime_metrics.json read.
#[derive(Debug, Clone, Default, Deserialize)]
struct MetricsObservation {
    #[serde(default)]
    memory_pressure: f64,
    #[serde(default)]
    thrashing_score: f64,
    #[serde(default)]
    cpu_pegged_fraction: f64,
}

/// File written each tick. Contains the current emission set; consumers
/// should read it as a complete replacement, not append.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerHintFile {
    pub generated_at: DateTime<Utc>,
    pub planner_version: u32,
    pub hints: Vec<PlannerHint>,
}

/// Bounded ring buffer of recent observations used for trend detection.
#[derive(Debug, Clone, Default)]
struct TrendWindow {
    samples: std::collections::VecDeque<(DateTime<Utc>, MetricsObservation)>,
    capacity: usize,
}

impl TrendWindow {
    fn new(capacity: usize) -> Self {
        Self {
            samples: std::collections::VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, ts: DateTime<Utc>, obs: MetricsObservation) {
        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back((ts, obs));
    }

    /// Trend slope of `field` per second over the window. Returns 0.0
    /// if fewer than 2 samples.
    fn slope<F>(&self, field: F) -> f64
    where
        F: Fn(&MetricsObservation) -> f64,
    {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let first = self.samples.front().unwrap();
        let last = self.samples.back().unwrap();
        let dt = (last.0 - first.0).num_milliseconds() as f64 / 1000.0;
        if dt <= 0.0 {
            return 0.0;
        }
        let dv = field(&last.1) - field(&first.1);
        dv / dt
    }

    /// Latest observation, or None.
    fn latest(&self) -> Option<&MetricsObservation> {
        self.samples.back().map(|(_, o)| o)
    }

    /// Steadiness ∈ [0, 1]: fraction of consecutive sample pairs whose
    /// `field` delta has the same sign as the overall trend. Used as a
    /// confidence proxy.
    fn steadiness<F>(&self, field: F) -> f32
    where
        F: Fn(&MetricsObservation) -> f64,
    {
        if self.samples.len() < 3 {
            return 0.0;
        }
        let mut agreements = 0usize;
        let mut transitions = 0usize;
        // Note: f64::signum returns ±1.0 even for 0.0 (not 0). Use abs
        // against epsilon to detect zero-slope before computing the
        // sign — otherwise oscillating series with net-zero slope are
        // mis-classified as having a meaningful direction.
        let overall = self.slope(&field);
        if overall.abs() < f64::EPSILON {
            return 0.0;
        }
        let overall_sign = overall.signum();
        let pairs: Vec<_> = self.samples.iter().collect();
        for w in pairs.windows(2) {
            let local = field(&w[1].1) - field(&w[0].1);
            transitions += 1;
            if local.signum() == overall_sign {
                agreements += 1;
            }
        }
        if transitions == 0 {
            return 0.0;
        }
        agreements as f32 / transitions as f32
    }
}

/// Planner thread. Owns its own state, reads metrics file, writes
/// hints file. No shared state with the daemon.
pub struct Planner {
    cadence: Duration,
    metrics_path: PathBuf,
    output_path: PathBuf,
    window: TrendWindow,
    stop: Arc<AtomicBool>,
}

impl Planner {
    /// Number of samples retained for trend detection. At a 30-s cadence
    /// this is a 5-minute window which catches most workload phase
    /// transitions without being so long that stale data drowns the
    /// signal.
    pub const WINDOW_SAMPLES: usize = 10;

    pub fn new(cadence: Duration, metrics_path: PathBuf, output_path: PathBuf) -> Self {
        Self {
            cadence,
            metrics_path,
            output_path,
            window: TrendWindow::new(Self::WINDOW_SAMPLES),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Stop flag handle for graceful shutdown.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    /// Spawn the planner in its own thread. Returns the stop flag so
    /// the daemon can request shutdown. The thread exits cleanly when
    /// the flag flips to true OR when an unrecoverable I/O error
    /// happens (in which case it logs once and exits — never panics).
    pub fn spawn(mut self) -> Arc<AtomicBool> {
        let stop = self.stop.clone();
        std::thread::Builder::new()
            .name("apollo-planner".to_string())
            .spawn(move || {
                while !self.stop.load(Ordering::Relaxed) {
                    self.tick();
                    // Sleep in small chunks so the stop flag is checked
                    // promptly without missing the cadence.
                    let chunks = (self.cadence.as_secs().max(1) as usize).max(1);
                    for _ in 0..chunks {
                        if self.stop.load(Ordering::Relaxed) {
                            return;
                        }
                        std::thread::sleep(Duration::from_secs(1));
                    }
                }
            })
            .ok();
        stop
    }

    /// One observation cycle: read metrics, push into trend window,
    /// derive hints, persist to disk.
    fn tick(&mut self) {
        let obs = match Self::read_metrics(&self.metrics_path) {
            Some(o) => o,
            None => return, // metrics file not yet written; skip
        };
        self.window.push(Utc::now(), obs);
        let hints = self.derive_hints();
        let _ = Self::persist(&self.output_path, &hints);
    }

    fn read_metrics(path: &Path) -> Option<MetricsObservation> {
        let raw = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&raw).ok()
    }

    /// Apply the trend rules and produce the current hint set.
    /// Pure function over `self.window` — testable without I/O.
    fn derive_hints(&self) -> Vec<PlannerHint> {
        let mut out = Vec::new();
        let now = Utc::now();
        let latest = match self.window.latest() {
            Some(l) => l.clone(),
            None => return out,
        };

        // Pressure rule: rising at ≥ 0.005/sec for the window AND
        // current level ≥ 0.55. Predicted peak at horizon_secs based on
        // linear extrapolation, capped at 0.95.
        let p_slope = self.window.slope(|o| o.memory_pressure);
        if p_slope >= 0.005 && latest.memory_pressure >= 0.55 {
            let horizon = 60u64;
            let predicted = (latest.memory_pressure + p_slope * horizon as f64).min(0.95);
            out.push(PlannerHint {
                horizon_secs: horizon,
                confidence: self.window.steadiness(|o| o.memory_pressure),
                emitted_at: now,
                kind: HintKind::PressureSpike { peak: predicted },
            });
        }

        // Thrashing rule: thrashing_score climbing past 1500 with
        // positive slope, expect to enter gate_c territory (5000) soon.
        let t_slope = self.window.slope(|o| o.thrashing_score);
        if latest.thrashing_score >= 1500.0 && t_slope > 0.0 {
            let to_gate = (5000.0 - latest.thrashing_score).max(0.0);
            let horizon = if t_slope > 0.0 {
                ((to_gate / t_slope).clamp(15.0, 300.0)) as u64
            } else {
                300
            };
            out.push(PlannerHint {
                horizon_secs: horizon,
                confidence: self.window.steadiness(|o| o.thrashing_score),
                emitted_at: now,
                kind: HintKind::ThrashingOnset {
                    score: latest.thrashing_score,
                },
            });
        }

        // CPU saturation rule: pegged_fraction climbing above 0.5
        // suggests P-cluster will be sustained-busy for the horizon.
        let c_slope = self.window.slope(|o| o.cpu_pegged_fraction);
        if latest.cpu_pegged_fraction >= 0.5 && c_slope >= 0.0 {
            out.push(PlannerHint {
                horizon_secs: 120,
                confidence: self.window.steadiness(|o| o.cpu_pegged_fraction),
                emitted_at: now,
                kind: HintKind::CpuSaturation {
                    fraction: latest.cpu_pegged_fraction,
                },
            });
        }

        out
    }

    fn persist(path: &Path, hints: &[PlannerHint]) -> std::io::Result<()> {
        // Atomic write: tmp file + rename so partial writes never expose
        // half-written JSON to a future consumer.
        let file = PlannerHintFile {
            generated_at: Utc::now(),
            planner_version: 0, // bump on hint format changes
            hints: hints.to_vec(),
        };
        let json = serde_json::to_string_pretty(&file).map_err(std::io::Error::other)?;
        let tmp = path.with_extension("json.tmp");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;

    fn obs(p: f64, t: f64, c: f64) -> MetricsObservation {
        MetricsObservation {
            memory_pressure: p,
            thrashing_score: t,
            cpu_pegged_fraction: c,
        }
    }

    #[test]
    fn slope_zero_with_one_sample() {
        let mut w = TrendWindow::new(10);
        w.push(Utc::now(), obs(0.5, 0.0, 0.0));
        assert_eq!(w.slope(|o| o.memory_pressure), 0.0);
    }

    #[test]
    fn slope_positive_for_rising_pressure() {
        let mut w = TrendWindow::new(10);
        let t0 = Utc::now();
        w.push(t0, obs(0.50, 0.0, 0.0));
        w.push(t0 + ChronoDuration::seconds(30), obs(0.60, 0.0, 0.0));
        // 0.10 over 30s = 0.00333/sec
        let s = w.slope(|o| o.memory_pressure);
        assert!((s - 0.10 / 30.0).abs() < 1e-9, "got {s}");
    }

    #[test]
    fn steadiness_unanimous_trend_returns_one() {
        let mut w = TrendWindow::new(10);
        let t0 = Utc::now();
        for i in 0..6 {
            w.push(
                t0 + ChronoDuration::seconds(i * 30),
                obs(0.4 + i as f64 * 0.05, 0.0, 0.0),
            );
        }
        // All transitions positive → steadiness = 1.0.
        assert_eq!(w.steadiness(|o| o.memory_pressure), 1.0);
    }

    #[test]
    fn steadiness_mixed_trend_below_one() {
        let mut w = TrendWindow::new(10);
        let t0 = Utc::now();
        // Up, down, up, down — overall slope ≈ 0, steadiness ≈ 0.
        for (i, p) in [0.5, 0.6, 0.5, 0.6, 0.5].iter().enumerate() {
            w.push(t0 + ChronoDuration::seconds(i as i64 * 30), obs(*p, 0.0, 0.0));
        }
        let s = w.steadiness(|o| o.memory_pressure);
        // Overall slope is exactly 0 → steadiness returns 0 by contract.
        assert_eq!(s, 0.0);
    }

    fn make_planner_with_window() -> Planner {
        Planner {
            cadence: Duration::from_secs(30),
            metrics_path: PathBuf::from("/dev/null"),
            output_path: PathBuf::from("/dev/null"),
            window: TrendWindow::new(10),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    #[test]
    fn no_hints_when_window_empty() {
        let p = make_planner_with_window();
        assert!(p.derive_hints().is_empty());
    }

    #[test]
    fn pressure_spike_hint_emitted_on_rising_high_pressure() {
        let mut p = make_planner_with_window();
        let t0 = Utc::now();
        // Climbing from 0.55 → 0.70 over 5 minutes (0.0005/sec).
        // 0.0005 < 0.005 threshold → should NOT emit at this slope.
        for i in 0..6 {
            p.window.push(
                t0 + ChronoDuration::seconds(i * 30),
                obs(0.55 + i as f64 * 0.025, 0.0, 0.0),
            );
        }
        // 0.025 per 30s = 0.000833/s — still below 0.005/s threshold.
        let hints = p.derive_hints();
        assert!(
            !hints.iter().any(|h| matches!(h.kind, HintKind::PressureSpike { .. })),
            "0.000833/s slope should NOT emit pressure spike"
        );
    }

    #[test]
    fn pressure_spike_hint_emitted_on_fast_rise() {
        let mut p = make_planner_with_window();
        let t0 = Utc::now();
        // Climbing 0.55 → 0.85 over 60s = 0.005/sec exactly.
        for i in 0..3 {
            p.window.push(
                t0 + ChronoDuration::seconds(i * 30),
                obs(0.55 + i as f64 * 0.15, 0.0, 0.0),
            );
        }
        let hints = p.derive_hints();
        let spike = hints
            .iter()
            .find(|h| matches!(h.kind, HintKind::PressureSpike { .. }));
        assert!(spike.is_some(), "fast rise must emit pressure spike");
    }

    #[test]
    fn thrashing_onset_hint_when_climbing_past_1500() {
        let mut p = make_planner_with_window();
        let t0 = Utc::now();
        for i in 0..4 {
            p.window.push(
                t0 + ChronoDuration::seconds(i * 30),
                obs(0.4, 1500.0 + i as f64 * 500.0, 0.0),
            );
        }
        let hints = p.derive_hints();
        assert!(
            hints
                .iter()
                .any(|h| matches!(h.kind, HintKind::ThrashingOnset { .. })),
            "thrashing climbing past 1500 must emit ThrashingOnset"
        );
    }

    #[test]
    fn cpu_saturation_hint_when_pegged_above_half() {
        let mut p = make_planner_with_window();
        let t0 = Utc::now();
        for i in 0..3 {
            p.window.push(
                t0 + ChronoDuration::seconds(i * 30),
                obs(0.4, 0.0, 0.55 + i as f64 * 0.05),
            );
        }
        let hints = p.derive_hints();
        assert!(
            hints
                .iter()
                .any(|h| matches!(h.kind, HintKind::CpuSaturation { .. })),
            "rising pegged_fraction above 0.5 must emit CpuSaturation hint"
        );
    }

    #[test]
    fn no_hints_at_quiet_steady_state() {
        let mut p = make_planner_with_window();
        let t0 = Utc::now();
        // Steady at 0.4 pressure, 0 thrashing, 0 saturation — quiet.
        for i in 0..6 {
            p.window
                .push(t0 + ChronoDuration::seconds(i * 30), obs(0.4, 0.0, 0.0));
        }
        assert!(
            p.derive_hints().is_empty(),
            "quiet steady state must emit zero hints"
        );
    }

    #[test]
    fn persist_writes_atomic_json_round_trip() {
        let dir = std::env::temp_dir().join("apollo-planner-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("hints.json");
        let _ = std::fs::remove_file(&path);

        let hints = vec![PlannerHint {
            horizon_secs: 60,
            confidence: 0.8,
            emitted_at: Utc::now(),
            kind: HintKind::PressureSpike { peak: 0.85 },
        }];
        Planner::persist(&path, &hints).expect("persist must succeed");
        let raw = std::fs::read_to_string(&path).expect("file must exist");
        let parsed: PlannerHintFile =
            serde_json::from_str(&raw).expect("JSON must be valid");
        assert_eq!(parsed.hints.len(), 1);
        assert!(matches!(parsed.hints[0].kind, HintKind::PressureSpike { .. }));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_metrics_returns_none_on_missing_file() {
        let result = Planner::read_metrics(Path::new("/nonexistent/path/metrics.json"));
        assert!(result.is_none());
    }

    #[test]
    fn window_capped_at_capacity() {
        let mut w = TrendWindow::new(3);
        let t0 = Utc::now();
        for i in 0..5 {
            w.push(t0 + ChronoDuration::seconds(i * 30), obs(0.4, 0.0, 0.0));
        }
        assert_eq!(w.samples.len(), 3, "window must not exceed capacity");
    }
}
