//! Wake storm detection - prevents constant process wakeups
//!
//! Detects when a process is being woken up excessively and applies throttling.

use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct WakePattern {
    pub pid: u32,
    pub name: String,
    pub wakeup_count: u32,
    pub time_window: Duration,
    pub wakeups_per_second: f32,
    pub is_storm: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StormSeverity {
    Low,      // 10-50 wakeups/sec
    Medium,   // 50-200 wakeups/sec
    High,     // 200-1000 wakeups/sec
    Critical, // 1000+ wakeups/sec
}

pub struct WakeStormDetector {
    processes: HashMap<u32, ProcessWakeData>,
    storm_threshold: f32, // 10 wakeups/sec
    detection_window: Duration,
}

#[derive(Debug, Clone)]
struct ProcessWakeData {
    name: String,
    wakeup_times: Vec<Instant>,
    last_check: Instant,
}

impl WakeStormDetector {
    pub fn new() -> Self {
        Self {
            processes: HashMap::new(),
            storm_threshold: 10.0, // 10 wakeups/sec
            detection_window: Duration::from_secs(1),
        }
    }

    /// Record a wakeup event for a process
    pub fn record_wakeup(&mut self, pid: u32, name: String) {
        self.processes
            .entry(pid)
            .and_modify(|data| {
                let now = Instant::now();
                data.wakeup_times.push(now);
                data.last_check = now;
                data.name = name.clone();
                // Keep only recent wakeups within detection window
                data.wakeup_times
                    .retain(|t| now.duration_since(*t) < self.detection_window);
                // Cap to prevent memory growth from pathological wakeup rates
                if data.wakeup_times.len() > 10_000 {
                    data.wakeup_times.drain(..data.wakeup_times.len() - 10_000);
                }
            })
            .or_insert(ProcessWakeData {
                name,
                wakeup_times: vec![Instant::now()],
                last_check: Instant::now(),
            });
    }

    /// Detect wake storms for all monitored processes
    pub fn detect_storms(&self) -> Vec<WakePattern> {
        let mut storms = Vec::new();

        for (pid, data) in &self.processes {
            let wakeup_count = data.wakeup_times.len() as u32;
            if wakeup_count == 0 {
                continue;
            }

            let wakeups_per_sec = wakeup_count as f32 / self.detection_window.as_secs_f32();

            if wakeups_per_sec > self.storm_threshold {
                storms.push(WakePattern {
                    pid: *pid,
                    name: data.name.clone(),
                    wakeup_count,
                    time_window: self.detection_window,
                    wakeups_per_second: wakeups_per_sec,
                    is_storm: true,
                });
            }
        }

        storms.sort_by(|a, b| {
            b.wakeups_per_second
                .partial_cmp(&a.wakeups_per_second)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        storms
    }

    /// Get severity level for a wake rate
    pub fn get_severity(&self, wakeups_per_sec: f32) -> StormSeverity {
        match wakeups_per_sec {
            w if w >= 1000.0 => StormSeverity::Critical,
            w if w >= 200.0 => StormSeverity::High,
            w if w >= 50.0 => StormSeverity::Medium,
            _ => StormSeverity::Low,
        }
    }

    /// Get mitigation actions for a wake storm
    pub fn get_mitigation_actions(severity: StormSeverity) -> Vec<String> {
        let mut actions = Vec::new();

        match severity {
            StormSeverity::Critical => {
                actions.push("🔴 CRITICAL: Suspend process immediately".to_string());
                actions.push("🔴 Disable all event sources for process".to_string());
            }
            StormSeverity::High => {
                actions.push("🟠 HIGH: Reduce process priority (raise nice)".to_string());
                actions.push("🟠 Disable network polling for process".to_string());
            }
            StormSeverity::Medium => {
                actions.push("🟡 MEDIUM: Throttle process CPU time".to_string());
                actions.push("🟡 Reduce timer resolution for process".to_string());
            }
            StormSeverity::Low => {
                actions.push("⚠️ LOW: Monitor for escalation".to_string());
            }
        }

        actions
    }

    /// Clean up stale process data
    pub fn cleanup_stale(&mut self, max_age: Duration) {
        let now = Instant::now();
        self.processes
            .retain(|_, data| now.duration_since(data.last_check) < max_age);
    }
}

impl Default for WakeStormDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn new_detector_starts_empty() {
        let det = WakeStormDetector::new();
        assert!(det.detect_storms().is_empty());
    }

    #[test]
    fn default_is_same_as_new() {
        let det = WakeStormDetector::default();
        assert!(det.detect_storms().is_empty());
    }

    #[test]
    fn no_storm_below_threshold() {
        let mut det = WakeStormDetector::new();
        // Record only a few wakeups — well below the 10/sec threshold
        for _ in 0..3 {
            det.record_wakeup(100, "quiet-proc".to_string());
        }
        let storms = det.detect_storms();
        assert!(
            storms.is_empty(),
            "3 wakeups should not trigger a storm, got: {:?}",
            storms
        );
    }

    #[test]
    fn storm_detected_above_threshold() {
        let mut det = WakeStormDetector::new();
        // Record many wakeups in a tight burst — well above 10/sec
        for _ in 0..500 {
            det.record_wakeup(200, "noisy-proc".to_string());
        }
        let storms = det.detect_storms();
        assert!(
            !storms.is_empty(),
            "500 wakeups within the detection window should trigger a storm"
        );
        assert!(storms[0].is_storm);
        assert_eq!(storms[0].pid, 200);
    }

    #[test]
    fn get_severity_classification() {
        let det = WakeStormDetector::new();
        assert_eq!(det.get_severity(5.0), StormSeverity::Low);
        assert_eq!(det.get_severity(75.0), StormSeverity::Medium);
        assert_eq!(det.get_severity(300.0), StormSeverity::High);
        assert_eq!(det.get_severity(2000.0), StormSeverity::Critical);
    }

    #[test]
    fn get_mitigation_actions_non_empty() {
        for severity in [
            StormSeverity::Low,
            StormSeverity::Medium,
            StormSeverity::High,
            StormSeverity::Critical,
        ] {
            let actions = WakeStormDetector::get_mitigation_actions(severity);
            assert!(
                !actions.is_empty(),
                "expected at least one mitigation action for {severity:?}"
            );
        }
    }

    #[test]
    fn cleanup_stale_removes_old_processes() {
        let mut det = WakeStormDetector::new();
        det.record_wakeup(99, "stale-proc".to_string());
        // Sleep just long enough that cleanup_stale removes the entry
        std::thread::sleep(Duration::from_millis(5));
        det.cleanup_stale(Duration::from_millis(1));
        // After cleanup the process should be gone
        assert!(det.detect_storms().is_empty());
    }
}
