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
    storm_threshold: f32,  // 10 wakeups/sec
    detection_window: Duration,
}

#[derive(Debug, Clone)]
struct ProcessWakeData {
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
    pub fn record_wakeup(&mut self, pid: u32, _name: String) {
        self.processes
            .entry(pid)
            .and_modify(|data| {
                let now = Instant::now();
                data.wakeup_times.push(now);
                // Keep only recent wakeups within detection window
                data.wakeup_times
                    .retain(|t| now.duration_since(*t) < self.detection_window);
            })
            .or_insert(ProcessWakeData {
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
                    name: String::new(), // Would be filled from process list
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
        self.processes.retain(|_, data| {
            now.duration_since(data.last_check) < max_age
        });
    }
}

impl Default for WakeStormDetector {
    fn default() -> Self {
        Self::new()
    }
}
