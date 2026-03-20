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
}

/// Tracks ri_billed_energy deltas across cycles.
pub struct EnergyPidTracker {
    /// Previous readings: pid → (billed_energy_nj, instructions, cycles, proc_start_abstime).
    prev: HashMap<u32, (u64, u64, u64, u64)>,
}

impl EnergyPidTracker {
    pub fn new() -> Self {
        Self {
            prev: HashMap::new(),
        }
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
            );

            if let Some(&(prev_energy, prev_instr, prev_cycles, prev_start)) = self.prev.get(&pid)
            {
                // PID recycling check: if proc_start_abstime changed, skip delta.
                if prev_start == current.3 {
                    let delta_nj = current.0.saturating_sub(prev_energy);
                    let delta_instr = current.1.saturating_sub(prev_instr);
                    let delta_cycles = current.2.saturating_sub(prev_cycles);

                    // Convert nJ to mW: mW = nJ / (dt_s * 1_000_000)
                    let power_mw = delta_nj as f64 / (dt_secs * 1_000_000.0);

                    let ipc = if delta_cycles > 0 {
                        delta_instr as f64 / delta_cycles as f64
                    } else {
                        0.0
                    };

                    if delta_nj > 0 {
                        results.push(ProcessEnergyDelta {
                            pid,
                            name: name.to_string(),
                            delta_nj,
                            power_mw,
                            ipc,
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
            },
            ProcessEnergyDelta {
                pid: 2,
                name: "b".into(),
                delta_nj: 50,
                power_mw: 5.0,
                ipc: 0.5,
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
            },
            ProcessEnergyDelta {
                pid: 2,
                name: "b".into(),
                delta_nj: 200,
                power_mw: 20.0,
                ipc: 0.5,
            },
        ];
        assert_eq!(EnergyPidTracker::total_energy_nj(&data), 300);
        assert!((EnergyPidTracker::total_power_mw(&data) - 30.0).abs() < 0.001);
    }
}
