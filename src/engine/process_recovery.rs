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
}

impl ProcessRecoveryManager {
    pub fn new() -> Self {
        Self {
            leaking_processes: HashMap::new(),
            max_recovery_attempts: 3,
        }
    }

    /// Register a detected memory leak
    pub fn register_leak(&mut self, pid: u32, name: String, leak_prob: f64, rss: u64) {
        if !leak_prob.is_finite() || !(0.75..=1.0).contains(&leak_prob) {
            return; // Only track high-confidence, valid leaks
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
            elapsed > Duration::from_secs(1800)
                && proc.recovery_attempts < self.max_recovery_attempts
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
        let max_attempts = self.max_recovery_attempts;
        self.leaking_processes.retain(|_, proc| {
            // Remove entries older than 1 hour or that exhausted recovery attempts
            proc.first_detected_at.elapsed() < Duration::from_secs(3600)
                && proc.recovery_attempts < max_attempts
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── register_leak ────────────────────────────────────────────────────────

    #[test]
    fn register_leak_tracks_high_confidence_only() {
        let mut mgr = ProcessRecoveryManager::new();
        // Valid high-confidence leak (0.75–1.0).
        mgr.register_leak(100, "leaker".into(), 0.90, 256 * 1024 * 1024);
        assert!(mgr.leaking_processes.contains_key(&100));
    }

    #[test]
    fn register_leak_rejects_low_confidence() {
        let mut mgr = ProcessRecoveryManager::new();
        mgr.register_leak(200, "borderline".into(), 0.74, 100 * 1024 * 1024);
        assert!(!mgr.leaking_processes.contains_key(&200), "below threshold should be ignored");
    }

    #[test]
    fn register_leak_rejects_nan_and_inf() {
        let mut mgr = ProcessRecoveryManager::new();
        mgr.register_leak(300, "nan".into(), f64::NAN, 1024);
        mgr.register_leak(301, "inf".into(), f64::INFINITY, 1024);
        assert!(!mgr.leaking_processes.contains_key(&300));
        assert!(!mgr.leaking_processes.contains_key(&301));
    }

    #[test]
    fn register_leak_updates_existing_entry() {
        let mut mgr = ProcessRecoveryManager::new();
        mgr.register_leak(400, "grower".into(), 0.80, 100 * 1024 * 1024);
        mgr.register_leak(400, "grower".into(), 0.95, 200 * 1024 * 1024);
        let proc = &mgr.leaking_processes[&400];
        assert!((proc.leak_probability - 0.95).abs() < 1e-9);
        assert_eq!(proc.rss_bytes, 200 * 1024 * 1024);
    }

    // ── should_kill_process ───────────────────────────────────────────────────

    #[test]
    fn should_kill_process_returns_false_for_unknown_pid() {
        let mgr = ProcessRecoveryManager::new();
        assert!(!mgr.should_kill_process(999));
    }

    #[test]
    fn should_kill_process_returns_false_when_no_time_elapsed() {
        let mut mgr = ProcessRecoveryManager::new();
        mgr.register_leak(500, "fresh".into(), 0.90, 256 * 1024 * 1024);
        // Freshly registered — has NOT been leaking for 30 min yet.
        assert!(!mgr.should_kill_process(500));
    }

    // ── record_kill_attempt ───────────────────────────────────────────────────

    #[test]
    fn record_kill_attempt_increments_counter() {
        let mut mgr = ProcessRecoveryManager::new();
        mgr.register_leak(600, "leaker".into(), 0.85, 256 * 1024 * 1024);
        assert_eq!(mgr.leaking_processes[&600].recovery_attempts, 0);
        mgr.record_kill_attempt(600);
        assert_eq!(mgr.leaking_processes[&600].recovery_attempts, 1);
        mgr.record_kill_attempt(600);
        assert_eq!(mgr.leaking_processes[&600].recovery_attempts, 2);
    }

    #[test]
    fn record_kill_attempt_no_op_for_unknown_pid() {
        let mut mgr = ProcessRecoveryManager::new();
        // Should not panic.
        mgr.record_kill_attempt(999_999);
    }

    // ── cleanup_resolved ─────────────────────────────────────────────────────

    #[test]
    fn cleanup_resolved_removes_exhausted_attempts() {
        let mut mgr = ProcessRecoveryManager::new();
        mgr.register_leak(700, "exhausted".into(), 0.90, 256 * 1024 * 1024);
        // Exhaust recovery attempts.
        for _ in 0..3 {
            mgr.record_kill_attempt(700);
        }
        mgr.cleanup_resolved();
        assert!(!mgr.leaking_processes.contains_key(&700),
            "exhausted process should be removed");
    }

    #[test]
    fn cleanup_resolved_retains_fresh_leaks() {
        let mut mgr = ProcessRecoveryManager::new();
        mgr.register_leak(800, "fresh_leak".into(), 0.80, 256 * 1024 * 1024);
        mgr.cleanup_resolved();
        assert!(mgr.leaking_processes.contains_key(&800),
            "fresh leak with remaining attempts should be retained");
    }

    // ── get_recovery_targets ──────────────────────────────────────────────────

    #[test]
    fn get_recovery_targets_empty_when_no_leaks() {
        let mgr = ProcessRecoveryManager::new();
        assert!(mgr.get_recovery_targets().is_empty());
    }

    #[test]
    fn get_recovery_targets_fresh_processes_not_included() {
        let mut mgr = ProcessRecoveryManager::new();
        mgr.register_leak(900, "leaker".into(), 0.90, 256 * 1024 * 1024);
        // Fresh registrations never appear in recovery targets (need 30 min).
        assert!(mgr.get_recovery_targets().is_empty());
    }

    // ── estimate_recovery_cost ────────────────────────────────────────────────

    #[test]
    fn estimate_recovery_cost_increases_with_attempts() {
        let base = LeakingProcess {
            pid: 1, name: "test".into(),
            leak_probability: 0.90,
            rss_bytes: 1024 * 1024 * 1024, // 1 GB
            first_detected_at: std::time::Instant::now(),
            recovery_attempts: 0,
        };
        let with_attempts = LeakingProcess { recovery_attempts: 2, ..base.clone() };
        let cost0 = ProcessRecoveryManager::estimate_recovery_cost(&base);
        let cost2 = ProcessRecoveryManager::estimate_recovery_cost(&with_attempts);
        assert!(cost2 > cost0, "more attempts → higher recovery cost");
    }

    #[test]
    fn estimate_recovery_cost_increases_with_rss() {
        let small = LeakingProcess {
            pid: 2, name: "small".into(),
            leak_probability: 0.90,
            rss_bytes: 128 * 1024 * 1024, // 128 MB
            first_detected_at: std::time::Instant::now(),
            recovery_attempts: 0,
        };
        let large = LeakingProcess { rss_bytes: 4 * 1024 * 1024 * 1024, ..small.clone() };
        let cost_small = ProcessRecoveryManager::estimate_recovery_cost(&small);
        let cost_large = ProcessRecoveryManager::estimate_recovery_cost(&large);
        assert!(cost_large > cost_small, "higher RSS → higher recovery cost");
    }

    #[test]
    fn estimate_recovery_cost_formula_is_correct() {
        // cost = prob × (1 + attempts×0.5) × rss_in_gb
        let proc = LeakingProcess {
            pid: 3, name: "exact".into(),
            leak_probability: 0.80,
            rss_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
            first_detected_at: std::time::Instant::now(),
            recovery_attempts: 1,
        };
        let cost = ProcessRecoveryManager::estimate_recovery_cost(&proc);
        let expected = 0.80 * 1.5 * 2.0; // 2.40
        assert!((cost - expected).abs() < 0.01, "cost={cost:.4} expected={expected:.4}");
    }

    // ── Default impl ─────────────────────────────────────────────────────────

    #[test]
    fn process_recovery_manager_default_is_clean() {
        let mgr = ProcessRecoveryManager::default();
        assert!(mgr.leaking_processes.is_empty());
        assert!(mgr.get_recovery_targets().is_empty());
    }
}
