//! Advanced memory analysis - Working Set Size (WSS), memory leak detection
//!
//! Provides deeper insights into process memory usage patterns.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MemoryProfile {
    pub pid: u32,
    pub name: String,
    pub rss_bytes: u64,           // Resident set size
    pub vms_bytes: u64,           // Virtual memory size
    pub wss_bytes: u64,           // Working set size (estimated)
    pub page_faults_per_sec: f64,
    pub memory_leak_probability: f64, // 0.0-1.0
    pub memory_efficiency: f64,   // WSS / RSS ratio (0.0-1.0)
}

pub struct MemoryAnalyzer {
    process_history: HashMap<u32, Vec<MemorySnapshot>>,
    history_limit: usize,
}

#[derive(Debug, Clone)]
struct MemorySnapshot {
    timestamp: std::time::Instant,
    rss: u64,
    page_faults: u64,
}

impl MemoryAnalyzer {
    pub fn new() -> Self {
        Self {
            process_history: HashMap::new(),
            history_limit: 60, // Keep 60 samples
        }
    }

    /// Analyze a process for memory leaks based on growth pattern
    pub fn analyze_process(
        &mut self,
        pid: u32,
        name: &str,
        rss_bytes: u64,
        vms_bytes: u64,
        page_faults: u64,
    ) -> MemoryProfile {
        let now = std::time::Instant::now();
        let snapshot = MemorySnapshot {
            timestamp: now,
            rss: rss_bytes,
            page_faults,
        };

        // Store history
        self.process_history
            .entry(pid)
            .or_default()
            .push(snapshot.clone());

        // Trim history
        if let Some(history) = self.process_history.get_mut(&pid) {
            if history.len() > self.history_limit {
                let _ = history.remove(0);
            }
        }

        let history = self.process_history.get(&pid).unwrap();
        let leak_prob = self.detect_memory_leak(pid, history);
        let page_faults_per_sec = self.calculate_page_fault_rate(history);
        let wss = self.estimate_wss(rss_bytes, page_faults_per_sec);
        let efficiency = (wss as f64 / rss_bytes.max(1) as f64).clamp(0.0, 1.0);

        MemoryProfile {
            pid,
            name: name.to_string(),
            rss_bytes,
            vms_bytes,
            wss_bytes: wss,
            page_faults_per_sec,
            memory_leak_probability: leak_prob,
            memory_efficiency: efficiency,
        }
    }

    fn detect_memory_leak(&self, _pid: u32, history: &[MemorySnapshot]) -> f64 {
        if history.len() < 5 {
            return 0.0; // Not enough samples
        }

        // Simple heuristic: is RSS consistently growing?
        let recent = &history[history.len().saturating_sub(10)..];
        let mut growth_count = 0;

        for i in 1..recent.len() {
            if recent[i].rss > recent[i - 1].rss {
                growth_count += 1;
            }
        }

        let growth_rate = growth_count as f64 / (recent.len() - 1).max(1) as f64;

        // If growing in > 70% of samples, likely a leak
        if growth_rate > 0.7 {
            growth_rate // Return leak probability 0.7-1.0
        } else {
            0.0
        }
    }

    fn calculate_page_fault_rate(&self, history: &[MemorySnapshot]) -> f64 {
        if history.len() < 2 {
            return 0.0;
        }

        let first = &history[0];
        let last = &history[history.len() - 1];
        let time_delta = last
            .timestamp
            .duration_since(first.timestamp)
            .as_secs_f64()
            .max(1.0);
        let fault_delta = last.page_faults.saturating_sub(first.page_faults);

        fault_delta as f64 / time_delta
    }

    fn estimate_wss(&self, rss: u64, page_faults_per_sec: f64) -> u64 {
        // Heuristic: WSS ≈ RSS * (1 - page_fault_ratio)
        // More page faults = less efficient WSS
        let fault_impact = (page_faults_per_sec / 1000.0).min(1.0);
        let wss_ratio = 1.0 - (fault_impact * 0.3); // 30% max impact

        ((rss as f64) * wss_ratio) as u64
    }

    /// Identify memory-inefficient processes
    pub fn find_inefficient_processes(&self, threshold: f64) -> Vec<(u32, f64)> {
        let mut results = Vec::new();

        for (pid, history) in &self.process_history {
            if history.is_empty() {
                continue;
            }

            let last = &history[history.len() - 1];
            let page_faults_per_sec = self.calculate_page_fault_rate(history);
            let wss = self.estimate_wss(last.rss, page_faults_per_sec);
            let efficiency = (wss as f64 / last.rss.max(1) as f64).clamp(0.0, 1.0);

            if efficiency < threshold {
                results.push((*pid, efficiency));
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get processes with probable memory leaks
    pub fn find_memory_leaks(&self, threshold: f64) -> Vec<(u32, f64)> {
        let mut results = Vec::new();

        for (pid, history) in &self.process_history {
            if history.is_empty() {
                continue;
            }

            let leak_prob = self.detect_memory_leak(*pid, history);
            if leak_prob >= threshold {
                results.push((*pid, leak_prob));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

impl Default for MemoryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
