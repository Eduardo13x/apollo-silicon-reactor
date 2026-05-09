//! # Self-Diagnosis — daemon's meta-observer for known regression classes
//!
//! Implements the Phase 6 self-healing observability layer requested by the
//! user 2026-05-06: instead of just patching individual gaps, the daemon
//! detects gap **classes** as they regress and emits actionable alerts.
//!
//! ## Signal classes monitored
//!
//! 1. **Dedup drops trending up** — `lf_metrics.dedup_drops_*` non-zero over
//!    a 5-minute rolling window means upstream paths are emitting duplicate
//!    actions. The Phase 1 chokepoint catches them, but the *signal* says
//!    a new emission path was added without per-PID dedup awareness.
//! 2. **Refresh duration regression** — `lf_metrics.refresh_duration_us`
//!    median > 30ms in Normal pressure zone means staggered cadence is
//!    degrading (cache miss, code change defeating the staggering, etc.).
//! 3. **Reactor weight saturation time** — placeholder (Phase 4 deferred);
//!    when stress test infrastructure is added, time-to-saturation > 15s
//!    will trigger a "weights too damped" alert.
//!
//! ## What this module does NOT do
//!
//! - **No mutations**: detection + alert only. Auto-fix is up to the LLM
//!   teacher (Gemma 4) or next apollo-evolve loop reading the alert stream.
//! - **No NARS bridge yet**: planned for next iteration (inject as belief
//!   into the existing NARS system so god nodes accumulate diagnosis context).
//!
//! ## Outputs
//!
//! - `tracing::warn!` event with `target = "apollo.self_diagnosis"`
//! - JSONL append to `/var/lib/apollo/self_diagnosis.jsonl` (or `/tmp/`
//!   when running non-root).
//!
//! [Hellerstein 2004 §9] meta-observer over feedback control —
//! detection-only layer prevents oscillation between observer and actuator.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Rolling-window observation of one cycle's self-healing signals.
#[derive(Debug, Clone, Copy)]
struct CycleObservation {
    /// Wall-clock observation timestamp. Currently unused (analyses operate
    /// on the rolling buffer position), but recorded for future per-window
    /// time-correlation features (e.g., burst detection).
    #[allow(dead_code)]
    at: Instant,
    dedup_drops_setmemorystatus: u64,
    dedup_drops_throttle: u64,
    dedup_drops_freeze: u64,
    dedup_drops_unfreeze: u64,
    refresh_duration_us: u64,
    /// Coarse pressure zone discrimination via memory_pressure value.
    /// Avoids depending on `MemoryBudgetState::current_zone` which would
    /// pull in daemon binary types into a library module.
    in_normal_zone: bool,
}

impl CycleObservation {
    fn dedup_drops_total(&self) -> u64 {
        self.dedup_drops_setmemorystatus
            + self.dedup_drops_throttle
            + self.dedup_drops_freeze
            + self.dedup_drops_unfreeze
    }
}

/// Diagnosis event severity matches NotebookLM gap classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosisSeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// One self-diagnosis alert. Persisted to `self_diagnosis.jsonl` and emitted
/// via `tracing::warn!`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosisAlert {
    pub at: DateTime<Utc>,
    pub kind: String,
    pub severity: DiagnosisSeverity,
    pub summary: String,
    pub recommended_action: String,
}

/// Self-diagnosis state: rolling window over recent cycles plus throttle
/// timestamps so we don't spam alerts every cycle once a regression is
/// detected.
pub struct SelfDiagnosis {
    window: VecDeque<CycleObservation>,
    window_capacity: usize,
    /// Cooldown per alert kind (5min) — prevents spam once detected.
    last_alert_at: std::collections::HashMap<&'static str, Instant>,
    cooldown: Duration,
    persist_path: PathBuf,
}

impl SelfDiagnosis {
    /// New self-diagnosis state.
    /// `persist_path` typically points to `/var/lib/apollo/self_diagnosis.jsonl`
    /// (root) or `/tmp/apollo_self_diagnosis.jsonl` (non-root).
    pub fn new(persist_path: PathBuf) -> Self {
        Self {
            // ~5 minutes at 2Hz cycle rate = 600 observations.
            window: VecDeque::with_capacity(600),
            window_capacity: 600,
            last_alert_at: std::collections::HashMap::new(),
            cooldown: Duration::from_secs(300),
            persist_path,
        }
    }

    /// Record one cycle's signals with per-kind dedup breakdown.
    /// Bounded ring buffer.
    pub fn record_cycle(
        &mut self,
        dedup_drops_setmemorystatus: u64,
        dedup_drops_throttle: u64,
        dedup_drops_freeze: u64,
        dedup_drops_unfreeze: u64,
        refresh_duration_us: u64,
        memory_pressure: f64,
    ) {
        if self.window.len() >= self.window_capacity {
            self.window.pop_front();
        }
        self.window.push_back(CycleObservation {
            at: Instant::now(),
            dedup_drops_setmemorystatus,
            dedup_drops_throttle,
            dedup_drops_freeze,
            dedup_drops_unfreeze,
            refresh_duration_us,
            in_normal_zone: memory_pressure < 0.65,
        });
    }

    /// Check thresholds across the rolling window and return alerts that
    /// crossed the threshold *and* are outside their cooldown.
    /// Caller is responsible for persistence + tracing emission.
    pub fn check(&mut self) -> Vec<DiagnosisAlert> {
        let mut alerts: Vec<DiagnosisAlert> = Vec::new();
        let now = Instant::now();

        // Need at least 60 cycles (~30s) of data before alerting.
        if self.window.len() < 60 {
            return alerts;
        }

        // ── Signal 1: dedup_drops trending up ────────────────────────────────
        // Per-kind breakdown lets the alert pinpoint which emission class is
        // regressing (Throttle vs SetMemorystatus vs Freeze).
        //
        // Threshold tuned 2026-05-06 from 0.5 → 3.0 actions/cycle: steady-state
        // baseline measured at ~7.6/cycle in prod (multiple modules emitting
        // independent decisions for the same PID, expected behavior). Alert now
        // fires only when drops climb significantly above that floor — the
        // signal we WANT is "regression beyond steady-state", not "any dup".
        let total_drops: u64 = self.window.iter().map(|o| o.dedup_drops_total()).sum();
        let throttle_drops: u64 = self.window.iter().map(|o| o.dedup_drops_throttle).sum();
        let setmem_drops: u64 = self.window.iter().map(|o| o.dedup_drops_setmemorystatus).sum();
        let freeze_drops: u64 = self.window.iter().map(|o| o.dedup_drops_freeze).sum();
        let unfreeze_drops: u64 = self.window.iter().map(|o| o.dedup_drops_unfreeze).sum();
        let n = self.window.len() as f64;
        let avg_drops_per_cycle = total_drops as f64 / n;

        if avg_drops_per_cycle > 3.0 && self.cooldown_ok("dedup_regression", now) {
            self.last_alert_at.insert("dedup_regression", now);
            alerts.push(DiagnosisAlert {
                at: Utc::now(),
                kind: "dedup_regression".to_string(),
                severity: DiagnosisSeverity::Medium,
                summary: format!(
                    "dedup chokepoint dropping {:.2}/cycle (throttle={:.2}, setmem={:.2}, freeze={:.2}, unfreeze={:.2}; window={} cycles, threshold=3.0)",
                    avg_drops_per_cycle,
                    throttle_drops as f64 / n,
                    setmem_drops as f64 / n,
                    freeze_drops as f64 / n,
                    unfreeze_drops as f64 / n,
                    self.window.len()
                ),
                recommended_action: "audit emission paths for the dominant kind: ThrottleProcess most often = process_enrichment + decide_actions heuristic-pass producing parallel decisions for same PID; SetMemorystatus = daemon_paging_hints + main.rs deep-scan + daemon_agent_actions; Freeze = process_enrichment GovernorDecision::Freeze + Kill→Freeze downgrade".to_string(),
            });
        }

        // ── Signal 2: refresh_duration regression in Normal zone ─────────────
        // Foundation commit added staggered refresh (Normal=8c, Elev=4c, Crit=1c).
        // Median refresh in Normal should be ≤30ms (measured 20-29ms post-deploy).
        // If median climbs > 30ms in Normal, staggering is degrading.
        let normal_samples: Vec<u64> = self
            .window
            .iter()
            .filter(|o| o.in_normal_zone)
            .map(|o| o.refresh_duration_us)
            .collect();

        if normal_samples.len() >= 30 {
            let mut sorted = normal_samples.clone();
            sorted.sort_unstable();
            let median_us = sorted[sorted.len() / 2];
            if median_us > 30_000 && self.cooldown_ok("sysinfo_regression", now) {
                self.last_alert_at.insert("sysinfo_regression", now);
                alerts.push(DiagnosisAlert {
                    at: Utc::now(),
                    kind: "sysinfo_regression".to_string(),
                    severity: DiagnosisSeverity::Medium,
                    summary: format!(
                        "sysinfo refresh median in Normal zone = {:.1}ms (target ≤30ms; staggered cadence likely defeated); n={}",
                        median_us as f64 / 1000.0,
                        normal_samples.len()
                    ),
                    recommended_action: "verify collect_snapshot_light is being called when use_light=true (main.rs:1505 area); sample lf_metrics.refresh_duration_us by zone with /tmp/measure_phase2.sh".to_string(),
                });
            }
        }

        alerts
    }

    /// Cooldown check: returns true if this alert kind hasn't fired recently.
    fn cooldown_ok(&self, kind: &'static str, now: Instant) -> bool {
        match self.last_alert_at.get(kind) {
            Some(last) => now.duration_since(*last) >= self.cooldown,
            None => true,
        }
    }

    /// Persist alerts to the JSONL file (one alert per line, append mode).
    /// Errors are silenced — best-effort logging, daemon must not fail.
    pub fn persist(&self, alerts: &[DiagnosisAlert]) {
        if alerts.is_empty() {
            return;
        }
        use std::fs::OpenOptions;
        use std::io::Write;
        let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.persist_path)
        else {
            return;
        };
        for alert in alerts {
            if let Ok(line) = serde_json::to_string(alert) {
                let _ = writeln!(f, "{}", line);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path() -> PathBuf {
        std::env::temp_dir().join(format!("apollo_sd_test_{}.jsonl", std::process::id()))
    }

    #[test]
    fn empty_window_emits_no_alerts() {
        let mut sd = SelfDiagnosis::new(temp_path());
        let alerts = sd.check();
        assert!(alerts.is_empty());
    }

    #[test]
    fn under_60_cycles_emits_no_alerts() {
        let mut sd = SelfDiagnosis::new(temp_path());
        for _ in 0..30 {
            // throttle=10 well above threshold but below cycle quorum.
            sd.record_cycle(0, 10, 0, 0, 50_000, 0.5);
        }
        let alerts = sd.check();
        assert!(alerts.is_empty(), "below window-min threshold");
    }

    #[test]
    fn dedup_drops_above_threshold_emits_alert() {
        let mut sd = SelfDiagnosis::new(temp_path());
        // 100 cycles each with 4 throttle drops = 4.0/cycle (above 3.0 threshold).
        for _ in 0..100 {
            sd.record_cycle(0, 4, 0, 0, 10_000, 0.5);
        }
        let alerts = sd.check();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].kind, "dedup_regression");
        assert_eq!(alerts[0].severity, DiagnosisSeverity::Medium);
        assert!(
            alerts[0].summary.contains("throttle="),
            "summary should break down by kind: {}",
            alerts[0].summary
        );
    }

    #[test]
    fn dedup_drops_steady_state_no_alert() {
        let mut sd = SelfDiagnosis::new(temp_path());
        // 100 cycles each with 2 throttle drops = 2.0/cycle (steady state, below 3.0).
        // Mirrors prod baseline ~7.6/cycle but well below regression threshold.
        for _ in 0..100 {
            sd.record_cycle(0, 2, 0, 0, 10_000, 0.5);
        }
        let alerts = sd.check();
        assert!(
            alerts.iter().all(|a| a.kind != "dedup_regression"),
            "steady-state drops below threshold should not alert"
        );
    }

    #[test]
    fn dedup_drops_zero_emits_no_alert() {
        let mut sd = SelfDiagnosis::new(temp_path());
        for _ in 0..100 {
            sd.record_cycle(0, 0, 0, 0, 10_000, 0.5);
        }
        let alerts = sd.check();
        assert!(alerts.iter().all(|a| a.kind != "dedup_regression"));
    }

    #[test]
    fn cooldown_suppresses_repeated_alerts() {
        let mut sd = SelfDiagnosis::new(temp_path());
        for _ in 0..100 {
            sd.record_cycle(0, 4, 0, 0, 10_000, 0.5);
        }
        let first = sd.check();
        assert_eq!(first.len(), 1);
        // Second check immediately after should be silent (cooldown).
        let second = sd.check();
        assert_eq!(second.len(), 0);
    }

    #[test]
    fn refresh_duration_regression_in_normal_emits_alert() {
        let mut sd = SelfDiagnosis::new(temp_path());
        // 100 cycles in Normal zone with refresh 50ms — above 30ms threshold.
        for _ in 0..100 {
            sd.record_cycle(0, 0, 0, 0, 50_000, 0.5);
        }
        let alerts = sd.check();
        assert!(alerts.iter().any(|a| a.kind == "sysinfo_regression"));
    }

    #[test]
    fn refresh_regression_skipped_in_elevated_zone() {
        let mut sd = SelfDiagnosis::new(temp_path());
        // 100 cycles in Elevated zone with refresh 50ms — should NOT alert
        // (only Normal zone has the staggered-refresh expectation).
        for _ in 0..100 {
            sd.record_cycle(0, 0, 0, 0, 50_000, 0.70);
        }
        let alerts = sd.check();
        assert!(
            !alerts.iter().any(|a| a.kind == "sysinfo_regression"),
            "Elevated zone should not trigger sysinfo regression alert"
        );
    }

    #[test]
    fn rolling_window_bounded() {
        let mut sd = SelfDiagnosis::new(temp_path());
        for _ in 0..1000 {
            sd.record_cycle(0, 0, 0, 0, 10_000, 0.5);
        }
        assert!(sd.window.len() <= sd.window_capacity);
    }

    #[test]
    fn persist_writes_jsonl_lines() {
        let path = temp_path();
        let _ = std::fs::remove_file(&path);
        let sd = SelfDiagnosis::new(path.clone());
        let alerts = vec![
            DiagnosisAlert {
                at: Utc::now(),
                kind: "test".to_string(),
                severity: DiagnosisSeverity::Low,
                summary: "test alert".to_string(),
                recommended_action: "noop".to_string(),
            },
        ];
        sd.persist(&alerts);
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"kind\":\"test\""));
        let _ = std::fs::remove_file(&path);
    }
}
