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

// ── DAMON-style Adaptive WSS Estimator ──────────────────────────────────────
//
// DAMON (Data Access MONitor), SeongJae Park, arXiv:2303.05919.
//
// Instead of tracking every page, divide the address space into N adaptive
// regions and sample 1 page per region per cycle. Hot regions split for
// finer tracking; cold regions merge to save overhead.
//
// Cost: O(N) probes per cycle (N ≤ 64). At ~4µs per probe → ~256µs max.

/// Maximum number of tracked regions per process.
const DAMON_MAX_REGIONS: usize = 64;
/// Minimum number of regions (floor after merges).
const DAMON_MIN_REGIONS: usize = 8;
/// Cycles without access before a region is considered cold.
const COLD_STREAK_THRESHOLD: u32 = 5;
/// Cycles of consecutive access before a region is considered "very hot" and splits.
const HOT_SPLIT_THRESHOLD: u32 = 3;

/// A single monitored memory region.
#[derive(Debug, Clone)]
pub struct DamonRegion {
    /// Start address (inclusive).
    pub start: u64,
    /// End address (exclusive).
    pub end: u64,
    /// How many cycles this region was accessed since last cold reset.
    pub hot_count: u32,
    /// Consecutive cycles without access.
    pub cold_streak: u32,
}

impl DamonRegion {
    fn size(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    fn midpoint(&self) -> u64 {
        self.start + self.size() / 2
    }
}

/// Per-process DAMON-style working set estimator.
///
/// Initialize once with `init_from_regions`, then call `sample_cycle` each
/// daemon tick with a function that probes whether a page was recently accessed.
#[derive(Debug, Clone)]
pub struct DamonEstimator {
    regions: Vec<DamonRegion>,
    cycle_count: u32,
    last_wss: u64,
}

impl DamonEstimator {
    /// Create an estimator from a RegionSummary by dividing the total virtual
    /// space into `max_regions` equal-sized regions.
    ///
    /// Only call once per process; subsequent calls reset the estimator.
    pub fn init_from_summary(total_virtual: u64, n_initial: usize) -> Self {
        let n = n_initial.clamp(DAMON_MIN_REGIONS, DAMON_MAX_REGIONS);
        let region_size = total_virtual / n as u64;
        let regions: Vec<DamonRegion> = (0..n)
            .map(|i| {
                let start = i as u64 * region_size;
                let end = if i == n - 1 {
                    total_virtual
                } else {
                    start + region_size
                };
                DamonRegion {
                    start,
                    end,
                    hot_count: 0,
                    cold_streak: 0,
                }
            })
            .collect();

        Self {
            regions,
            cycle_count: 0,
            last_wss: 0,
        }
    }

    /// Run one sampling cycle. `is_accessed` should return `true` if the page
    /// at the given address was recently accessed (e.g., timing probe shows hot).
    ///
    /// Returns the estimated WSS in bytes.
    pub fn sample_cycle(&mut self, is_accessed: impl Fn(u64) -> bool) -> u64 {
        self.cycle_count += 1;

        // Phase 1: sample each region at its midpoint.
        for region in &mut self.regions {
            let probe = region.midpoint();
            if is_accessed(probe) {
                region.hot_count += 1;
                region.cold_streak = 0;
            } else {
                region.cold_streak += 1;
                // Reset hot count after sustained cold.
                if region.cold_streak >= COLD_STREAK_THRESHOLD {
                    region.hot_count = 0;
                }
            }
        }

        // Phase 2: split very hot regions (consecutive access ≥ threshold).
        if self.regions.len() < DAMON_MAX_REGIONS {
            let mut splits = Vec::new();
            for i in 0..self.regions.len() {
                if self.regions[i].hot_count >= HOT_SPLIT_THRESHOLD
                    && self.regions[i].cold_streak == 0
                    && self.regions[i].size() >= 2 * 16384
                    && self.regions.len() + splits.len() < DAMON_MAX_REGIONS
                {
                    splits.push(i);
                }
            }
            // Split from end to preserve indices.
            for &idx in splits.iter().rev() {
                let mid = self.regions[idx].midpoint();
                let new_region = DamonRegion {
                    start: mid,
                    end: self.regions[idx].end,
                    hot_count: self.regions[idx].hot_count,
                    cold_streak: 0,
                };
                self.regions[idx].end = mid;
                self.regions.insert(idx + 1, new_region);
            }
        }

        // Phase 3: merge adjacent cold regions.
        if self.regions.len() > DAMON_MIN_REGIONS {
            let mut i = 0;
            while i + 1 < self.regions.len() && self.regions.len() > DAMON_MIN_REGIONS {
                let both_cold = self.regions[i].cold_streak >= COLD_STREAK_THRESHOLD
                    && self.regions[i + 1].cold_streak >= COLD_STREAK_THRESHOLD;
                if both_cold {
                    self.regions[i].end = self.regions[i + 1].end;
                    self.regions.remove(i + 1);
                    // Don't advance i — check the merged region against next neighbor.
                } else {
                    i += 1;
                }
            }
        }

        // Phase 4: compute WSS = sum of hot region sizes.
        let wss: u64 = self
            .regions
            .iter()
            .filter(|r| r.hot_count > 0)
            .map(|r| r.size())
            .sum();

        self.last_wss = wss;
        wss
    }

    /// Last computed WSS.
    pub fn wss_bytes(&self) -> u64 {
        self.last_wss
    }

    /// Current number of tracked regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Total cycles elapsed.
    pub fn cycles(&self) -> u32 {
        self.cycle_count
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MemoryAnalyzer tests ─────────────────────────────────────────────

    #[test]
    fn analyze_basic() {
        let mut analyzer = MemoryAnalyzer::new();
        let profile = analyzer.analyze_process(100, "test", 100_000_000, 200_000_000, 0);
        assert_eq!(profile.pid, 100);
        assert_eq!(profile.rss_bytes, 100_000_000);
        assert!(!profile.is_thrashing);
    }

    // ── DAMON estimator tests ────────────────────────────────────────────

    #[test]
    fn damon_init_creates_regions() {
        let est = DamonEstimator::init_from_summary(1024 * 1024 * 1024, 32);
        assert_eq!(est.region_count(), 32);
        assert_eq!(est.wss_bytes(), 0);
        assert_eq!(est.cycles(), 0);
    }

    #[test]
    fn damon_init_clamps_regions() {
        let est = DamonEstimator::init_from_summary(1024 * 1024, 200);
        assert_eq!(est.region_count(), DAMON_MAX_REGIONS);
        let est = DamonEstimator::init_from_summary(1024 * 1024, 2);
        assert_eq!(est.region_count(), DAMON_MIN_REGIONS);
    }

    #[test]
    fn damon_all_hot_wss_equals_total() {
        let total = 1024 * 1024 * 1024u64; // 1 GB
        let mut est = DamonEstimator::init_from_summary(total, 16);
        // All regions accessed every cycle.
        let wss = est.sample_cycle(|_| true);
        assert_eq!(wss, total, "all hot → WSS = total");
    }

    #[test]
    fn damon_all_cold_wss_drops() {
        let total = 1024 * 1024 * 1024u64;
        let mut est = DamonEstimator::init_from_summary(total, 16);

        // Warm up: all hot for 3 cycles.
        for _ in 0..3 {
            est.sample_cycle(|_| true);
        }
        assert!(est.wss_bytes() > 0);

        // Cool down: all cold for COLD_STREAK_THRESHOLD cycles.
        for _ in 0..COLD_STREAK_THRESHOLD {
            est.sample_cycle(|_| false);
        }
        assert_eq!(est.wss_bytes(), 0, "all cold after streak → WSS = 0");
    }

    #[test]
    fn damon_cold_regions_merge() {
        let total = 1024 * 1024 * 1024u64;
        let mut est = DamonEstimator::init_from_summary(total, 32);
        let initial = est.region_count();

        // All cold for many cycles → regions should merge.
        for _ in 0..(COLD_STREAK_THRESHOLD + 2) {
            est.sample_cycle(|_| false);
        }

        assert!(
            est.region_count() < initial,
            "cold regions should merge: {} < {}",
            est.region_count(),
            initial
        );
        assert!(
            est.region_count() >= DAMON_MIN_REGIONS,
            "should not go below minimum: {}",
            est.region_count()
        );
    }

    #[test]
    fn damon_hot_regions_split() {
        let total = 1024 * 1024 * 1024u64;
        let mut est = DamonEstimator::init_from_summary(total, 16);
        let initial = est.region_count();

        // All hot for HOT_SPLIT_THRESHOLD cycles → hot regions split.
        for _ in 0..HOT_SPLIT_THRESHOLD {
            est.sample_cycle(|_| true);
        }

        assert!(
            est.region_count() > initial,
            "hot regions should split: {} > {}",
            est.region_count(),
            initial
        );
    }

    #[test]
    fn damon_wss_decreases_when_cooling() {
        let total = 1024 * 1024 * 1024u64;
        let mut est = DamonEstimator::init_from_summary(total, 16);

        // Warm up.
        for _ in 0..3 {
            est.sample_cycle(|_| true);
        }
        let warm_wss = est.wss_bytes();

        // Half the regions go cold.
        for _ in 0..COLD_STREAK_THRESHOLD {
            est.sample_cycle(|addr| addr < total / 2);
        }
        let cooled_wss = est.wss_bytes();

        assert!(
            cooled_wss < warm_wss,
            "WSS should decrease: {} < {}",
            cooled_wss,
            warm_wss
        );
    }
}
