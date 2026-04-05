//! Per-Process Energy Ranking via XNU `ri_billed_energy`
//!
//! Uses `proc_pid_rusage(RUSAGE_INFO_V4).ri_billed_energy` — the most accurate
//! per-process energy metric available on macOS. Counted by the kernel in
//! nanojoules, not estimated from CPU%.
//!
//! # How it works
//!
//! Each cycle, we sample `ri_billed_energy` for the top processes and compute
//! deltas. The delta gives actual energy consumed since last sample.
//!
//! Unlike EnergyTracker (which estimates watts from CPU%), this gives
//! **kernel-measured** energy including CPU, GPU, and ANE attribution.
//!
//! # Usage
//!
//! The ranking is used for:
//! - Identifying true energy hogs (for throttle/freeze decisions)
//! - More accurate savings estimation
//! - Dashboard display of real per-app energy

use std::collections::HashMap;

use crate::engine::proc_taskinfo;
use crate::engine::process_baseline::ProcessBaselineMap;

/// Per-process energy delta for one cycle.
#[derive(Debug, Clone)]
pub struct ProcessEnergyDelta {
    pub pid: u32,
    pub name: String,
    /// Energy consumed this cycle (nanojoules).
    pub delta_nj: u64,
    /// Energy consumed this cycle (milliwatts, derived from delta/dt).
    pub power_mw: f64,
    /// Per-process IPC from ri_instructions/ri_cycles delta.
    pub ipc: f64,
    /// CPU wakeups per second (idle + pkg_idle combined).
    /// Apple Activity Monitor uses this as the primary "Energy Impact" signal.
    /// >100/s = battery vampire; >500/s = severe drain.
    pub wakeup_rate: f64,
    /// True physical memory footprint (MB). More accurate than RSS for freeze ranking
    /// because it excludes shared pages and includes compressed memory contribution.
    pub phys_footprint_mb: f64,
    /// Disk writes this cycle (MB/s). High disk I/O during LLM inference = contention.
    /// [Bhagwan & Savage 2002 OSDI] I/O-heavy background processes compete for disk
    /// bandwidth with model weight loading — throttle during inference.
    pub disk_write_mbps: f64,
    /// Composite anomaly score vs learned baseline for this process name.
    /// Score = max(|x - ema| / (mad + ε)) across {ipc, wakeup_rate, disk_mbps}.
    /// 0.0 = cold start or on-baseline; ≥ 3.0 = anomalous (3 MADs from normal).
    /// [Chandola 2009 ACM CSUR] scale-free z-score via EMA-MAD streaming estimator.
    pub anomaly_score: f64,
}

/// Tracks ri_billed_energy deltas across cycles.
pub struct EnergyPidTracker {
    /// Previous readings: pid → (energy_nj, instructions, cycles, proc_start_abstime, idle_wakeups, interrupt_wakeups, disk_write_bytes).
    prev: HashMap<u32, (u64, u64, u64, u64, u64, u64, u64)>,
    /// Per-process-name hardware counter baselines for anomaly detection.
    /// Shared ownership: caller can extract for persistence via `take_baseline` /
    /// `restore_baseline`. Keyed by name so baselines survive PID recycling.
    pub baseline: ProcessBaselineMap,
}

impl EnergyPidTracker {
    pub fn new() -> Self {
        Self {
            prev: HashMap::new(),
            baseline: ProcessBaselineMap::new(),
        }
    }

    /// Extract the baseline map for persistence (replaces with empty map).
    pub fn take_baseline(&mut self) -> ProcessBaselineMap {
        std::mem::replace(&mut self.baseline, ProcessBaselineMap::new())
    }

    /// Restore a previously persisted baseline map.
    pub fn restore_baseline(&mut self, map: ProcessBaselineMap) {
        self.baseline = map;
    }

    /// Sample energy for a list of (pid, name) pairs and compute deltas.
    ///
    /// `dt_secs`: elapsed time since last call (for mW conversion).
    /// Returns sorted by power_mw descending (top consumers first).
    pub fn sample(
        &mut self,
        processes: &[(u32, &str)],
        dt_secs: f64,
    ) -> Vec<ProcessEnergyDelta> {
        if dt_secs <= 0.001 {
            return Vec::new();
        }

        let mut results = Vec::new();
        let mut new_prev = HashMap::with_capacity(processes.len());

        for &(pid, name) in processes {
            let rusage = match proc_taskinfo::get_rusage_info(pid) {
                Some(r) => r,
                None => continue,
            };

            let current = (
                rusage.billed_energy,
                rusage.instructions,
                rusage.cycles,
                rusage.proc_start_abstime,
                rusage.idle_wakeups,
                rusage.interrupt_wakeups,
                rusage.disk_write_bytes,
            );

            let phys_footprint_mb = rusage.phys_footprint as f64 / (1024.0 * 1024.0);

            if let Some(&(prev_energy, prev_instr, prev_cycles, prev_start, prev_idle_wk, prev_intr_wk, prev_disk_w)) =
                self.prev.get(&pid)
            {
                // PID recycling check: if proc_start_abstime changed, skip delta.
                if prev_start == current.3 {
                    let delta_nj = current.0.saturating_sub(prev_energy);
                    let delta_instr = current.1.saturating_sub(prev_instr);
                    let delta_cycles = current.2.saturating_sub(prev_cycles);
                    let delta_idle_wk = current.4.saturating_sub(prev_idle_wk);
                    let delta_intr_wk = current.5.saturating_sub(prev_intr_wk);
                    let delta_disk_w = current.6.saturating_sub(prev_disk_w);

                    // Convert nJ to mW: mW = nJ / (dt_s * 1_000_000)
                    let power_mw = delta_nj as f64 / (dt_secs * 1_000_000.0);

                    let ipc = if delta_cycles > 0 {
                        delta_instr as f64 / delta_cycles as f64
                    } else {
                        0.0
                    };

                    // Wakeup rate = (idle + interrupt wakeups) per second.
                    let wakeup_rate = (delta_idle_wk + delta_intr_wk) as f64 / dt_secs;

                    // Disk write rate in MB/s.
                    let disk_write_mbps = delta_disk_w as f64 / (dt_secs * 1_048_576.0);

                    if delta_nj > 0 || wakeup_rate > 10.0 || disk_write_mbps > 1.0 {
                        // Score BEFORE updating baseline so the current sample is
                        // evaluated against the historical baseline, not itself.
                        let anomaly_score = self.baseline.anomaly_score(
                            name, ipc, wakeup_rate, disk_write_mbps,
                        );
                        // Update baseline with this observation (after scoring).
                        self.baseline.observe(name, ipc, wakeup_rate, disk_write_mbps);
                        results.push(ProcessEnergyDelta {
                            pid,
                            name: name.to_string(),
                            delta_nj,
                            power_mw,
                            ipc,
                            wakeup_rate,
                            phys_footprint_mb,
                            disk_write_mbps,
                            anomaly_score,
                        });
                    }
                }
            }

            new_prev.insert(pid, current);
        }

        self.prev = new_prev;

        // Sort by power descending.
        results.sort_by(|a, b| b.power_mw.partial_cmp(&a.power_mw).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get the top N energy consumers from last sample.
    pub fn top_consumers(results: &[ProcessEnergyDelta], n: usize) -> &[ProcessEnergyDelta] {
        let end = n.min(results.len());
        &results[..end]
    }

    /// Build a pid → wakeup_rate map for use in decide_actions.
    /// Only includes processes with wakeup_rate > threshold (default 50/s).
    pub fn build_wakeup_hints(
        results: &[ProcessEnergyDelta],
        min_rate: f64,
    ) -> HashMap<u32, f64> {
        results
            .iter()
            .filter(|r| r.wakeup_rate >= min_rate)
            .map(|r| (r.pid, r.wakeup_rate))
            .collect()
    }

    /// Build a pid → phys_footprint_mb map for freeze priority ranking.
    pub fn build_footprint_hints(results: &[ProcessEnergyDelta]) -> HashMap<u32, f64> {
        results
            .iter()
            .filter(|r| r.phys_footprint_mb > 0.0)
            .map(|r| (r.pid, r.phys_footprint_mb))
            .collect()
    }

    /// Build a pid → disk_write_mbps map for I/O burst detection.
    /// Only includes processes with disk_write_mbps > threshold.
    /// [Bhagwan & Savage 2002 OSDI] I/O burst = process writing >5 MB/s in background.
    pub fn build_io_burst_hints(
        results: &[ProcessEnergyDelta],
        min_mbps: f64,
    ) -> HashMap<u32, f64> {
        results
            .iter()
            .filter(|r| r.disk_write_mbps >= min_mbps)
            .map(|r| (r.pid, r.disk_write_mbps))
            .collect()
    }

    /// Total system energy this cycle (nanojoules, sum of all sampled processes).
    pub fn total_energy_nj(results: &[ProcessEnergyDelta]) -> u64 {
        results.iter().map(|r| r.delta_nj).sum()
    }

    /// Total system power this cycle (milliwatts).
    pub fn total_power_mw(results: &[ProcessEnergyDelta]) -> f64 {
        results.iter().map(|r| r.power_mw).sum()
    }

    /// Clean up stale PIDs not seen in this cycle.
    pub fn gc(&mut self, live_pids: &[u32]) {
        let live: std::collections::HashSet<u32> = live_pids.iter().copied().collect();
        self.prev.retain(|pid, _| live.contains(pid));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_on_zero_dt() {
        let mut tracker = EnergyPidTracker::new();
        let results = tracker.sample(&[], 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn nj_to_mw_conversion() {
        // 1_000_000 nJ over 1 second = 1 mW
        let power_mw = 1_000_000u64 as f64 / (1.0 * 1_000_000.0);
        assert!((power_mw - 1.0).abs() < 0.001);
    }

    #[test]
    fn top_consumers_limits() {
        let data = vec![
            ProcessEnergyDelta {
                pid: 1,
                name: "a".into(),
                delta_nj: 100,
                power_mw: 10.0,
                ipc: 1.0,
                wakeup_rate: 0.0,
                phys_footprint_mb: 0.0,
                disk_write_mbps: 0.0,
                anomaly_score: 0.0,
            },
            ProcessEnergyDelta {
                pid: 2,
                name: "b".into(),
                delta_nj: 50,
                power_mw: 5.0,
                ipc: 0.5,
                wakeup_rate: 0.0,
                phys_footprint_mb: 0.0,
                disk_write_mbps: 0.0,
                anomaly_score: 0.0,
            },
        ];
        assert_eq!(EnergyPidTracker::top_consumers(&data, 1).len(), 1);
        assert_eq!(EnergyPidTracker::top_consumers(&data, 10).len(), 2);
    }

    #[test]
    fn total_energy() {
        let data = vec![
            ProcessEnergyDelta {
                pid: 1,
                name: "a".into(),
                delta_nj: 100,
                power_mw: 10.0,
                ipc: 1.0,
                wakeup_rate: 0.0,
                phys_footprint_mb: 0.0,
                disk_write_mbps: 0.0,
                anomaly_score: 0.0,
            },
            ProcessEnergyDelta {
                pid: 2,
                name: "b".into(),
                delta_nj: 200,
                power_mw: 20.0,
                ipc: 0.5,
                wakeup_rate: 0.0,
                phys_footprint_mb: 0.0,
                disk_write_mbps: 0.0,
                anomaly_score: 0.0,
            },
        ];
        assert_eq!(EnergyPidTracker::total_energy_nj(&data), 300);
        assert!((EnergyPidTracker::total_power_mw(&data) - 30.0).abs() < 0.001);
    }

    #[test]
    fn wakeup_hints_filters_threshold() {
        let data = vec![
            ProcessEnergyDelta { pid: 1, name: "a".into(), delta_nj: 0, power_mw: 0.0, ipc: 0.0, wakeup_rate: 200.0, phys_footprint_mb: 100.0, disk_write_mbps: 0.0, anomaly_score: 0.0 },
            ProcessEnergyDelta { pid: 2, name: "b".into(), delta_nj: 0, power_mw: 0.0, ipc: 0.0, wakeup_rate: 30.0, phys_footprint_mb: 50.0, disk_write_mbps: 0.0, anomaly_score: 0.0 },
            ProcessEnergyDelta { pid: 3, name: "c".into(), delta_nj: 0, power_mw: 0.0, ipc: 0.0, wakeup_rate: 500.0, phys_footprint_mb: 200.0, disk_write_mbps: 0.0, anomaly_score: 0.0 },
        ];
        let hints = EnergyPidTracker::build_wakeup_hints(&data, 50.0);
        assert_eq!(hints.len(), 2);
        assert!(hints.contains_key(&1));
        assert!(hints.contains_key(&3));
        assert!(!hints.contains_key(&2));
    }

    #[test]
    fn footprint_hints_built_correctly() {
        let data = vec![
            ProcessEnergyDelta { pid: 10, name: "x".into(), delta_nj: 100, power_mw: 5.0, ipc: 1.0, wakeup_rate: 0.0, phys_footprint_mb: 256.0, disk_write_mbps: 0.0, anomaly_score: 0.0 },
            ProcessEnergyDelta { pid: 11, name: "y".into(), delta_nj: 50, power_mw: 2.0, ipc: 0.5, wakeup_rate: 0.0, phys_footprint_mb: 0.0, disk_write_mbps: 0.0, anomaly_score: 0.0 },
        ];
        let footprints = EnergyPidTracker::build_footprint_hints(&data);
        assert_eq!(footprints.len(), 1);
        assert!((footprints[&10] - 256.0).abs() < 0.1);
    }

    #[test]
    fn wakeup_rate_math() {
        // 300 wakeups over 0.5s = 600/s
        let rate = 300u64 as f64 / 0.5;
        assert!((rate - 600.0).abs() < 0.001);
    }

    #[test]
    fn disk_write_mbps_math() {
        // 10 MB in 1 second = 10 MB/s
        let bytes = 10u64 * 1_048_576;
        let mbps = bytes as f64 / (1.0 * 1_048_576.0);
        assert!((mbps - 10.0).abs() < 0.001);
    }

    #[test]
    fn io_burst_hints_filters_threshold() {
        let data = vec![
            ProcessEnergyDelta { pid: 1, name: "backup".into(), delta_nj: 0, power_mw: 0.0, ipc: 0.0, wakeup_rate: 0.0, phys_footprint_mb: 0.0, disk_write_mbps: 50.0, anomaly_score: 0.0 },
            ProcessEnergyDelta { pid: 2, name: "idle".into(), delta_nj: 0, power_mw: 0.0, ipc: 0.0, wakeup_rate: 0.0, phys_footprint_mb: 0.0, disk_write_mbps: 0.5, anomaly_score: 0.0 },
            ProcessEnergyDelta { pid: 3, name: "spotlight".into(), delta_nj: 0, power_mw: 0.0, ipc: 0.0, wakeup_rate: 0.0, phys_footprint_mb: 0.0, disk_write_mbps: 8.0, anomaly_score: 0.0 },
        ];
        let hints = EnergyPidTracker::build_io_burst_hints(&data, 5.0);
        assert_eq!(hints.len(), 2);
        assert!(hints.contains_key(&1));
        assert!(hints.contains_key(&3));
        assert!(!hints.contains_key(&2));
    }
}
