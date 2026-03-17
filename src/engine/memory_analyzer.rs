//! Advanced memory analysis — Working Set Size (WSS), thrashing detection, leak detection.
//!
//! Key insight (Denning 1968, "The Working Set Model for Program Behavior"):
//! A process thrashes when its working set exceeds available physical pages.
//! The observable signal is **major page faults (page-ins)** — pages fetched
//! from disk/swap/compressor.  Minor faults (soft remaps) are cheap and normal.
//!
//! On macOS, `PROC_PIDTASKINFO` gives us:
//!   - `pti_faults`  — total VM faults (major + minor), cumulative counter
//!   - `pti_pageins` — page-ins from backing store (major faults only)
//!
//! Therefore: `minor_faults = faults - pageins`, `major_faults = pageins`.
//! We track page-in rate (Δpageins / Δt) as the primary thrashing signal.

use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct MemoryProfile {
    pub pid: u32,
    pub name: String,
    pub rss_bytes: u64,
    pub vms_bytes: u64,
    pub wss_bytes: u64,
    /// Major page faults per second (page-ins from disk/swap/compressor).
    /// This is the thrashing signal: > 50/s = moderate, > 200/s = severe.
    pub major_faults_per_sec: f64,
    /// Minor page faults per second (soft remaps, cheap).
    pub minor_faults_per_sec: f64,
    pub memory_leak_probability: f64,
    /// WSS / RSS ratio.  Low (<0.5) = process has lots of cold pages in RAM.
    pub memory_efficiency: f64,
    /// True if process is actively thrashing (major faults > threshold).
    pub is_thrashing: bool,
    /// True if WSS was measured via Mach TASK_VM_INFO (not heuristic).
    pub wss_is_measured: bool,
}

pub struct MemoryAnalyzer {
    process_history: HashMap<u32, VecDeque<MemorySnapshot>>,
    history_limit: usize,
}

#[derive(Debug, Clone)]
struct MemorySnapshot {
    timestamp: std::time::Instant,
    rss: u64,
    /// Cumulative page-ins (major faults) from `pti_pageins`.
    pageins: u64,
}

/// Thrashing thresholds (page-ins per second).
/// Calibrated for Apple M1 with 16 KB pages:
///   50 page-ins/s × 16 KB = 800 KB/s of swap/compressor I/O
///   200 page-ins/s × 16 KB = 3.2 MB/s — noticeable latency
const THRASHING_MODERATE: f64 = 50.0;
const THRASHING_SEVERE: f64 = 200.0;

impl MemoryAnalyzer {
    pub fn new() -> Self {
        Self {
            process_history: HashMap::new(),
            history_limit: 60,
        }
    }

    /// Analyze a process's memory behavior.
    ///
    /// `page_faults` should be the cumulative `pti_pageins` (major faults)
    /// from `PROC_PIDTASKINFO`.  If unavailable, pass 0 (degrades gracefully).
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
            pageins: page_faults,
        };

        let history = self.process_history.entry(pid).or_default();
        history.push_back(snapshot);
        while history.len() > self.history_limit {
            history.pop_front();
        }

        let history = self.process_history.get(&pid).unwrap();
        let leak_prob = self.detect_memory_leak(history);
        let major_faults_per_sec = Self::calculate_pagein_rate(history);

        // Denning WSS estimation:
        // If major fault rate is near zero → WSS ≈ RSS (everything fits in RAM).
        // If major fault rate is high → WSS > RSS, estimate overshoot.
        // WSS ≈ RSS × (1 + major_faults_rate / THRASHING_SEVERE)
        // Clamped so WSS ≥ RSS (by definition, working set can exceed resident set).
        let wss = if major_faults_per_sec > 1.0 {
            let overshoot = (major_faults_per_sec / THRASHING_SEVERE).min(2.0);
            ((rss_bytes as f64) * (1.0 + overshoot)) as u64
        } else {
            rss_bytes
        };

        let efficiency = (rss_bytes as f64 / wss.max(1) as f64).clamp(0.0, 1.0);
        let is_thrashing = major_faults_per_sec >= THRASHING_MODERATE;

        MemoryProfile {
            pid,
            name: name.to_string(),
            rss_bytes,
            vms_bytes,
            wss_bytes: wss,
            major_faults_per_sec,
            minor_faults_per_sec: 0.0, // Not tracked separately (would need pti_faults too)
            memory_leak_probability: leak_prob,
            memory_efficiency: efficiency,
            is_thrashing,
            wss_is_measured: false, // Heuristic; caller can override via refine_wss()
        }
    }

    /// Page-in rate (Δpageins / Δt) using the most recent vs earliest snapshot.
    fn calculate_pagein_rate(history: &VecDeque<MemorySnapshot>) -> f64 {
        if history.len() < 2 {
            return 0.0;
        }

        let first = &history[0];
        let last = &history[history.len() - 1];
        let dt = last
            .timestamp
            .duration_since(first.timestamp)
            .as_secs_f64()
            .max(0.1);
        let delta = last.pageins.saturating_sub(first.pageins);

        delta as f64 / dt
    }

    fn detect_memory_leak(&self, history: &VecDeque<MemorySnapshot>) -> f64 {
        if history.len() < 5 {
            return 0.0;
        }

        let start = history.len().saturating_sub(10);
        let recent_len = history.len() - start;
        let mut growth_count = 0;

        for i in 1..recent_len {
            if history[start + i].rss > history[start + i - 1].rss {
                growth_count += 1;
            }
        }

        let growth_rate = growth_count as f64 / (recent_len - 1).max(1) as f64;
        if growth_rate > 0.7 {
            growth_rate
        } else {
            0.0
        }
    }

    pub fn find_inefficient_processes(&self, threshold: f64) -> Vec<(u32, f64)> {
        let mut results = Vec::new();

        for (pid, history) in &self.process_history {
            if history.is_empty() {
                continue;
            }

            let last = &history[history.len() - 1];
            let major_rate = Self::calculate_pagein_rate(history);
            let wss = if major_rate > 1.0 {
                let overshoot = (major_rate / THRASHING_SEVERE).min(2.0);
                ((last.rss as f64) * (1.0 + overshoot)) as u64
            } else {
                last.rss
            };
            let efficiency = (last.rss as f64 / wss.max(1) as f64).clamp(0.0, 1.0);

            if efficiency < threshold {
                results.push((*pid, efficiency));
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn find_memory_leaks(&self, threshold: f64) -> Vec<(u32, f64)> {
        let mut results = Vec::new();

        for (pid, history) in &self.process_history {
            if history.is_empty() {
                continue;
            }

            let leak_prob = self.detect_memory_leak(history);
            if leak_prob >= threshold {
                results.push((*pid, leak_prob));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Remove entries for PIDs not present in the live set.
    pub fn cleanup_dead_pids(&mut self, live_pids: &[u32]) {
        let live: std::collections::HashSet<u32> = live_pids.iter().copied().collect();
        self.process_history.retain(|pid, _| live.contains(pid));
    }

    /// Refine a MemoryProfile with real WSS data from Mach TASK_VM_INFO.
    /// Call after `analyze_process` when `query_memory_profile` data is available.
    pub fn refine_wss(profile: &mut MemoryProfile, measured_wss_bytes: u64) {
        profile.wss_bytes = measured_wss_bytes;
        profile.wss_is_measured = true;
        profile.memory_efficiency =
            (profile.rss_bytes as f64 / measured_wss_bytes.max(1) as f64).clamp(0.0, 1.0);
    }

    /// Get the current major page-in rate for a specific process.
    /// Returns 0.0 if the process has no history.
    pub fn major_fault_rate(&self, pid: u32) -> f64 {
        self.process_history
            .get(&pid)
            .map(|h| Self::calculate_pagein_rate(h))
            .unwrap_or(0.0)
    }

    /// Return processes currently thrashing (major faults > moderate threshold).
    pub fn find_thrashing_processes(&self) -> Vec<(u32, f64)> {
        let mut results = Vec::new();

        for (pid, history) in &self.process_history {
            let rate = Self::calculate_pagein_rate(history);
            if rate >= THRASHING_MODERATE {
                results.push((*pid, rate));
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
