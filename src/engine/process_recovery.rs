//! Automatic process recovery - Kill and restart memory-leaking processes
//!
//! Detects memory leaks and automatically recovers by killing/restarting them.

use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct LeakingProcess {
    pub pid: u32,
    pub name: String,
    pub leak_probability: f64,
    pub rss_bytes: u64,
    pub first_detected_at: Instant,
    pub recovery_attempts: u32,
}

pub struct ProcessRecoveryManager {
    leaking_processes: HashMap<u32, LeakingProcess>,
    max_recovery_attempts: u32,
    recovery_cooldown: Duration,
}

impl ProcessRecoveryManager {
    pub fn new() -> Self {
        Self {
            leaking_processes: HashMap::new(),
            max_recovery_attempts: 3,
            recovery_cooldown: Duration::from_secs(300), // 5 min cooldown
        }
    }

    /// Register a detected memory leak
    pub fn register_leak(&mut self, pid: u32, name: String, leak_prob: f64, rss: u64) {
        if leak_prob < 0.75 {
            return; // Only track high-confidence leaks
        }

        self.leaking_processes
            .entry(pid)
            .and_modify(|p| {
                p.leak_probability = leak_prob;
                p.rss_bytes = rss;
            })
            .or_insert(LeakingProcess {
                pid,
                name,
                leak_probability: leak_prob,
                rss_bytes: rss,
                first_detected_at: Instant::now(),
                recovery_attempts: 0,
            });
    }

    /// Check if a process should be killed (leaked for too long)
    pub fn should_kill_process(&self, pid: u32) -> bool {
        if let Some(proc) = self.leaking_processes.get(&pid) {
            // Kill if: leaked for > 30min AND attempts < max
            let elapsed = proc.first_detected_at.elapsed();
            elapsed > Duration::from_secs(1800) && proc.recovery_attempts < self.max_recovery_attempts
        } else {
            false
        }
    }

    /// Record a kill attempt
    pub fn record_kill_attempt(&mut self, pid: u32) {
        if let Some(proc) = self.leaking_processes.get_mut(&pid) {
            proc.recovery_attempts += 1;
        }
    }

    /// Clear resolved processes (no longer leaking)
    pub fn cleanup_resolved(&mut self) {
        self.leaking_processes.retain(|_, proc| {
            // Keep if: still leaking OR within cooldown
            let cooldown_active = proc.first_detected_at.elapsed() < self.recovery_cooldown;
            proc.leak_probability > 0.7 || cooldown_active
        });
    }

    /// Get processes to recover (kill + restart)
    pub fn get_recovery_targets(&self) -> Vec<LeakingProcess> {
        let mut targets: Vec<_> = self
            .leaking_processes
            .values()
            .filter(|p| self.should_kill_process(p.pid))
            .cloned()
            .collect();

        targets.sort_by(|a, b| b.rss_bytes.cmp(&a.rss_bytes)); // Kill highest memory first
        targets.truncate(3); // Max 3 per cycle
        targets
    }

    /// Estimate recovery cost (time + resources)
    pub fn estimate_recovery_cost(proc: &LeakingProcess) -> f64 {
        // Cost = probability * (attempts + 1) * rss_ratio
        let attempt_penalty = 1.0 + (proc.recovery_attempts as f64 * 0.5);
        let memory_ratio = (proc.rss_bytes as f64) / (1024.0 * 1024.0 * 1024.0); // GB
        proc.leak_probability * attempt_penalty * memory_ratio
    }
}

impl Default for ProcessRecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}
