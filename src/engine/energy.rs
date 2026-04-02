//! Per-App Energy Estimation — proportional attribution of system power draw.
//!
//! The daemon knows total CPU/GPU/package power from `HardwareSnapshot` (via
//! `SmcReader` / `powermetrics`) and per-process CPU usage from `sysinfo`.
//! This module attributes power consumption to individual processes using
//! proportional CPU usage and accumulates energy (Wh) over time.
//!
//! # Attribution model
//!
//! For each daemon cycle of duration `dt` seconds:
//!
//! ```text
//! process_watts = (process_cpu% / total_cpu%) * cpu_watts
//! process_wh   += process_watts * (dt / 3600)
//! ```
//!
//! Where `process_cpu%` is from `sysinfo` (per-core percentage: a process fully
//! using one core reports 100.0 regardless of core count) and `total_cpu%` is
//! the sum of all processes' `cpu_usage`.
//!
//! # Known limitations
//!
//! 1. **P-core vs E-core asymmetry**: On Apple Silicon, P-cores consume roughly
//!    3x the power of E-cores per unit of CPU%. A process pinned to P-cores
//!    will use more energy than one on E-cores at the same CPU%, but we cannot
//!    determine per-process core affinity from `sysinfo`. The proportional model
//!    therefore over-estimates E-core-bound processes and under-estimates P-core
//!    heavy ones. This is documented as an inherent limitation of user-space
//!    energy estimation without per-core scheduling data (which requires kernel
//!    tracing or `powermetrics --show-process-energy`).
//!
//! 2. **GPU attribution**: GPU power is reported as a lump sum. Without per-process
//!    GPU usage data (which macOS does not expose via public APIs), we cannot
//!    attribute GPU watts to individual processes. GPU energy is tracked as a
//!    separate unattributed total.
//!
//! 3. **DRAM / ANE**: Not attributed to processes.

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::collector::ProcessStats;
use crate::engine::iokit_sensors::HardwareSnapshot;

// ── Constants ───────────────────────────────────────────────────────────────

/// CO2 intensity: kg CO2 per kWh.
/// Global average is ~0.475 kg/kWh (IEA 2023). We use a moderate estimate
/// that assumes a partial renewable mix. Users in clean-grid regions (e.g.,
/// France nuclear, Norway hydro) will see lower real emissions; users on
/// coal-heavy grids will see higher. This is a rough order-of-magnitude guide.
const CO2_KG_PER_KWH: f64 = 0.390;

/// Maximum delta-time (seconds) we accept per update call. If the daemon was
/// suspended or paused longer than this, we cap `dt` to avoid a single huge
/// energy spike that would distort cumulative totals.
const MAX_DT_SECS: f64 = 30.0;

/// Minimum delta-time (seconds). Ignore updates with sub-millisecond intervals
/// to avoid amplifying floating-point noise.
const MIN_DT_SECS: f64 = 0.001;

/// Processes not seen for this many seconds are decayed from the tracker.
const DECAY_TIMEOUT_SECS: f64 = 3600.0; // 1 hour

/// Maximum number of tracked process names. Hard ceiling to prevent unbounded
/// memory growth even if decay is somehow not cleaning up (e.g., rapid process
/// churn with unique names).
const MAX_TRACKED_PROCESSES: usize = 2000;

// ── Data types ──────────────────────────────────────────────────────────────

/// Per-process energy accumulator (internal bookkeeping).
#[derive(Debug, Clone)]
struct EnergyAccumulator {
    /// Current instantaneous power draw estimate (watts).
    current_watts: f64,
    /// Cumulative energy consumed since tracking started (watt-hours).
    cumulative_wh: f64,
    /// Last time this process was observed in an update call.
    last_seen: Instant,
}

/// Per-app energy report (public, serializable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppEnergy {
    /// Normalized process name.
    pub name: String,
    /// Estimated instantaneous power draw (watts).
    pub current_watts: f64,
    /// Cumulative energy since tracking started (watt-hours).
    pub cumulative_wh: f64,
    /// This process's share of total system CPU power (0.0..100.0).
    pub percentage_of_total: f64,
}

/// Session-level energy summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergySummary {
    /// Total CPU energy consumed across all tracked processes (Wh).
    pub total_cpu_wh: f64,
    /// Total GPU energy consumed (unattributed lump sum) (Wh).
    pub total_gpu_wh: f64,
    /// Total package energy consumed (Wh).
    pub total_package_wh: f64,
    /// Estimated CO2 emissions (kg).
    pub estimated_co2_kg: f64,
    /// Estimated energy saved by Apollo's optimizations (Wh).
    pub estimated_savings_wh: f64,
    /// Number of processes being tracked.
    pub tracked_processes: usize,
    /// Session duration in seconds.
    pub session_duration_secs: f64,
}

/// Main energy tracking engine.
pub struct EnergyTracker {
    /// Per-process-name energy accumulators.
    accumulators: HashMap<String, EnergyAccumulator>,
    /// Cumulative GPU energy (unattributed) in Wh.
    gpu_cumulative_wh: f64,
    /// Cumulative package energy in Wh.
    package_cumulative_wh: f64,
    /// Cumulative CPU energy (sum of all process attributions) in Wh.
    cpu_cumulative_wh: f64,
    /// Hypothetical energy if no optimization were applied (Wh).
    /// We estimate this as total_cpu_watts (unthrottled) accumulated over time.
    hypothetical_wh: f64,
    /// When tracking started.
    session_start: Instant,
    /// Last time `update()` was called (for dt calculation if caller doesn't provide it).
    last_update: Option<Instant>,
}

impl EnergyTracker {
    /// Create a new energy tracker. Lightweight; no allocations until first `update`.
    pub fn new() -> Self {
        Self {
            accumulators: HashMap::new(),
            gpu_cumulative_wh: 0.0,
            package_cumulative_wh: 0.0,
            cpu_cumulative_wh: 0.0,
            hypothetical_wh: 0.0,
            session_start: Instant::now(),
            last_update: None,
        }
    }

    /// Update energy estimates with current process stats and hardware snapshot.
    ///
    /// # Arguments
    ///
    /// * `processes` — Slice of per-process stats from the collector. May be
    ///   empty (idle system) or contain only the top-N processes.
    /// * `hw` — Current hardware power/thermal snapshot. Power fields may be
    ///   `None` if `SmcReader` hasn't produced data yet.
    /// * `dt_secs` — Time elapsed since the last update, in seconds. If this
    ///   is unreasonably large (daemon was paused), it will be capped.
    pub fn update(&mut self, processes: &[ProcessStats], hw: &HardwareSnapshot, dt_secs: f64) {
        // Sanitize dt: clamp to [MIN_DT_SECS, MAX_DT_SECS].
        let dt = dt_secs.clamp(MIN_DT_SECS, MAX_DT_SECS);

        // If dt was provided as NaN or negative, bail out entirely.
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }

        let now = Instant::now();
        self.last_update = Some(now);

        // Extract power readings, defaulting to 0 when unavailable.
        let cpu_watts = hw.power.cpu_watts.map(|w| w as f64).unwrap_or(0.0);
        let gpu_watts = hw.power.gpu_watts.map(|w| w as f64).unwrap_or(0.0);
        let package_watts = hw.power.package_watts.map(|w| w as f64).unwrap_or(0.0);

        // Clamp negative power readings (sensor glitch) to zero.
        let cpu_watts = cpu_watts.max(0.0);
        let gpu_watts = gpu_watts.max(0.0);
        let package_watts = package_watts.max(0.0);

        let dt_hours = dt / 3600.0;

        // Accumulate GPU and package energy (lump sums, not per-process).
        self.gpu_cumulative_wh += gpu_watts * dt_hours;
        self.package_cumulative_wh += package_watts * dt_hours;

        // Accumulate hypothetical energy (what power would be without throttling).
        // Conservative estimate: use the actual package watts as a lower bound.
        // A more accurate model would track pre-throttle TDP, but that data isn't
        // available. We use package_watts * 1.0 as baseline; the savings come from
        // processes we froze/throttled that no longer contribute to cpu_watts.
        self.hypothetical_wh += package_watts * dt_hours;

        // Sum total CPU% across all processes. sysinfo reports per-core percentages,
        // so a fully loaded 10-core system would sum to ~1000%.
        let total_cpu_pct: f64 = processes
            .iter()
            .map(|p| sanitize_f64(p.cpu_usage as f64))
            .sum();

        // Reset all current_watts to 0 before this cycle's attribution.
        for acc in self.accumulators.values_mut() {
            acc.current_watts = 0.0;
        }

        // Attribute CPU power proportionally if we have meaningful data.
        if total_cpu_pct > 0.01 && cpu_watts > 0.0 {
            for proc in processes {
                let proc_cpu = sanitize_f64(proc.cpu_usage as f64);
                if proc_cpu <= 0.0 {
                    continue;
                }

                let fraction = proc_cpu / total_cpu_pct;
                // fraction is guaranteed in (0, 1] since proc_cpu > 0 and
                // total_cpu_pct >= proc_cpu.
                let proc_watts = fraction * cpu_watts;
                let proc_wh = proc_watts * dt_hours;

                let name = normalize_process_name(&proc.name);
                let acc = self
                    .accumulators
                    .entry(name)
                    .or_insert_with(|| EnergyAccumulator {
                        current_watts: 0.0,
                        cumulative_wh: 0.0,
                        last_seen: now,
                    });

                acc.current_watts += proc_watts;
                acc.cumulative_wh += proc_wh;
                acc.last_seen = now;

                // Guard against NaN propagation from floating-point edge cases.
                if !acc.cumulative_wh.is_finite() {
                    acc.cumulative_wh = 0.0;
                }
                if !acc.current_watts.is_finite() {
                    acc.current_watts = 0.0;
                }

                self.cpu_cumulative_wh += proc_wh;
            }
        } else {
            // No meaningful CPU activity or no power data — just mark seen processes.
            for proc in processes {
                let name = normalize_process_name(&proc.name);
                if let Some(acc) = self.accumulators.get_mut(&name) {
                    acc.last_seen = now;
                }
            }
        }

        // Guard cumulative totals against NaN.
        if !self.cpu_cumulative_wh.is_finite() {
            self.cpu_cumulative_wh = 0.0;
        }
        if !self.gpu_cumulative_wh.is_finite() {
            self.gpu_cumulative_wh = 0.0;
        }
        if !self.package_cumulative_wh.is_finite() {
            self.package_cumulative_wh = 0.0;
        }
        if !self.hypothetical_wh.is_finite() {
            self.hypothetical_wh = 0.0;
        }

        // Periodic decay: remove processes not seen in DECAY_TIMEOUT_SECS.
        // Only run cleanup when the map is above a threshold to avoid scanning
        // on every cycle. Also enforce the hard cap.
        if self.accumulators.len() > 50 {
            self.decay_stale(now);
        }
        if self.accumulators.len() > MAX_TRACKED_PROCESSES {
            self.evict_oldest(now);
        }
    }

    /// Return the top `n` energy consumers sorted by current watts (descending).
    pub fn top_consumers(&self, n: usize) -> Vec<AppEnergy> {
        let total_watts: f64 = self
            .accumulators
            .values()
            .map(|a| a.current_watts)
            .sum::<f64>()
            .max(0.0);

        let mut entries: Vec<AppEnergy> = self
            .accumulators
            .iter()
            .filter(|(_, acc)| acc.current_watts > 0.0 || acc.cumulative_wh > 0.0)
            .map(|(name, acc)| {
                let pct = if total_watts > 1e-9 {
                    (acc.current_watts / total_watts) * 100.0
                } else {
                    0.0
                };
                AppEnergy {
                    name: name.clone(),
                    current_watts: acc.current_watts,
                    cumulative_wh: acc.cumulative_wh,
                    percentage_of_total: pct,
                }
            })
            .collect();

        entries.sort_by(|a, b| {
            b.current_watts
                .partial_cmp(&a.current_watts)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        entries.truncate(n);
        entries
    }

    /// Generate a session-level energy summary.
    pub fn session_summary(&self) -> EnergySummary {
        let total_wh = self.package_cumulative_wh;
        let total_kwh = total_wh / 1000.0;
        let co2_kg = total_kwh * CO2_KG_PER_KWH;

        EnergySummary {
            total_cpu_wh: self.cpu_cumulative_wh,
            total_gpu_wh: self.gpu_cumulative_wh,
            total_package_wh: self.package_cumulative_wh,
            estimated_co2_kg: co2_kg,
            estimated_savings_wh: self.savings_estimate_wh(),
            tracked_processes: self.accumulators.len(),
            session_duration_secs: self.session_start.elapsed().as_secs_f64(),
        }
    }

    /// Estimate energy saved by Apollo's throttling and freezing.
    ///
    /// The savings model is conservative: we compare the cumulative CPU energy
    /// attributed to processes against the hypothetical energy if all processes
    /// had run unthrottled. Since we don't have a true "unthrottled baseline,"
    /// we use the difference between package power accumulation (which includes
    /// DRAM and other fixed costs) and the attributed CPU energy as a rough
    /// proxy. This yields a lower-bound savings estimate.
    ///
    /// For a more accurate model, the daemon would need to track which processes
    /// were frozen/throttled and estimate their counterfactual power draw.
    pub fn savings_estimate_wh(&self) -> f64 {
        // Savings = hypothetical - actual_package.
        // Since hypothetical currently equals actual (we don't have pre-throttle
        // TDP data), savings will be near zero. This is intentionally conservative;
        // inflated savings claims are worse than under-reporting.
        //
        // Future improvement: when the daemon freezes process P that was using X%
        // CPU, call `record_savings(X% * cpu_watts * dt_hours)` to accumulate
        // real savings data.
        let savings = (self.hypothetical_wh - self.package_cumulative_wh).max(0.0);
        if savings.is_finite() {
            savings
        } else {
            0.0
        }
    }

    /// Record energy saved by a specific optimization action.
    ///
    /// Call this when the daemon freezes or throttles a process. Provide the
    /// estimated watts that process was consuming before the action.
    ///
    /// # Arguments
    ///
    /// * `saved_watts` — Power the process was drawing before freeze/throttle.
    /// * `duration_secs` — How long the process has been frozen/throttled.
    pub fn record_savings(&mut self, saved_watts: f64, duration_secs: f64) {
        if saved_watts > 0.0
            && duration_secs > 0.0
            && saved_watts.is_finite()
            && duration_secs.is_finite()
        {
            let dt = duration_secs.min(MAX_DT_SECS);
            self.hypothetical_wh += saved_watts * (dt / 3600.0);
        }
    }

    /// Number of processes currently being tracked.
    pub fn tracked_count(&self) -> usize {
        self.accumulators.len()
    }

    /// Reset all tracking data. Useful when the daemon restarts or the user
    /// requests a fresh session.
    pub fn reset(&mut self) {
        self.accumulators.clear();
        self.gpu_cumulative_wh = 0.0;
        self.package_cumulative_wh = 0.0;
        self.cpu_cumulative_wh = 0.0;
        self.hypothetical_wh = 0.0;
        self.session_start = Instant::now();
        self.last_update = None;
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    /// Remove processes not seen in DECAY_TIMEOUT_SECS.
    fn decay_stale(&mut self, now: Instant) {
        self.accumulators.retain(|_, acc| {
            let age_secs = now.duration_since(acc.last_seen).as_secs_f64();
            age_secs < DECAY_TIMEOUT_SECS
        });
    }

    /// Emergency eviction when we exceed MAX_TRACKED_PROCESSES.
    /// Removes the oldest-seen entries until we're under the limit.
    fn evict_oldest(&mut self, now: Instant) {
        if self.accumulators.len() <= MAX_TRACKED_PROCESSES {
            return;
        }

        // Collect entries with their age, sort by age descending (oldest first),
        // and remove the oldest ones.
        let mut entries: Vec<(String, f64)> = self
            .accumulators
            .iter()
            .map(|(name, acc)| {
                let age = now.duration_since(acc.last_seen).as_secs_f64();
                (name.clone(), age)
            })
            .collect();

        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let to_remove = self.accumulators.len() - MAX_TRACKED_PROCESSES;
        for (name, _) in entries.into_iter().take(to_remove) {
            self.accumulators.remove(&name);
        }
    }
}

impl Default for EnergyTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Utility functions ───────────────────────────────────────────────────────

/// Normalize a process name for consistent aggregation.
/// Trims whitespace and collapses empty names to "<unknown>".
fn normalize_process_name(name: &str) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        "<unknown>".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Sanitize an f64 value: replace NaN/Inf with 0.0, clamp negatives to 0.0.
fn sanitize_f64(v: f64) -> f64 {
    if v.is_finite() && v >= 0.0 {
        v
    } else {
        0.0
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::iokit_sensors::{ClusterTemps, PowerReading, ThermalState};

    /// Build a HardwareSnapshot with specified power values.
    fn make_hw(cpu_w: Option<f32>, gpu_w: Option<f32>, pkg_w: Option<f32>) -> HardwareSnapshot {
        HardwareSnapshot {
            thermal_state: ThermalState::Normal,
            temps: ClusterTemps {
                p_cluster_celsius: Some(60.0),
                e_cluster_celsius: Some(45.0),
                gpu_celsius: None,
                nand_celsius: None,
            },
            power: PowerReading {
                package_watts: pkg_w,
                cpu_watts: cpu_w,
                gpu_watts: gpu_w,
                dram_watts: None,
                ane_watts: None,
                ane_util_pct: None,
                ane_tflops: None,
            },
            p_cluster_util: None,
            e_cluster_util: None,
            battery_percent: None,
            battery_watts: None,
        }
    }

    fn make_proc(name: &str, cpu_usage: f32) -> ProcessStats {
        ProcessStats {
            pid: 1,
            name: name.to_string(),
            cpu_usage,
            memory_usage: 0,
            cpu_wall_ratio: None,
        }
    }

    #[test]
    fn basic_energy_attribution() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), Some(2.0), Some(10.0));
        let procs = vec![
            make_proc("Chrome", 150.0), // 150% of one core
            make_proc("Finder", 50.0),  // 50% of one core
        ];

        // 1 second elapsed
        tracker.update(&procs, &hw, 1.0);

        let top = tracker.top_consumers(10);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].name, "Chrome");

        // Chrome: 150/200 * 6.0 = 4.5 W
        let chrome = &top[0];
        assert!(
            (chrome.current_watts - 4.5).abs() < 0.01,
            "Chrome watts: {}",
            chrome.current_watts
        );
        assert!((chrome.percentage_of_total - 75.0).abs() < 0.1);

        // Finder: 50/200 * 6.0 = 1.5 W
        let finder = &top[1];
        assert!(
            (finder.current_watts - 1.5).abs() < 0.01,
            "Finder watts: {}",
            finder.current_watts
        );

        // Cumulative Wh after 1 second:
        // Chrome: 4.5 * (1/3600) = 0.00125 Wh
        assert!((chrome.cumulative_wh - 0.00125).abs() < 0.0001);
    }

    #[test]
    fn zero_cpu_no_division_by_zero() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), None, Some(8.0));
        let procs = vec![make_proc("idle_proc", 0.0)];

        // Should not panic or produce NaN.
        tracker.update(&procs, &hw, 1.0);
        let top = tracker.top_consumers(10);
        // No process has meaningful watts, so top should be empty or have 0 watts.
        for entry in &top {
            assert!(entry.current_watts.is_finite());
            assert!(entry.cumulative_wh.is_finite());
        }
    }

    #[test]
    fn empty_processes_list() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), Some(2.0), Some(10.0));

        tracker.update(&[], &hw, 1.0);
        let top = tracker.top_consumers(10);
        assert!(top.is_empty());

        // Package and GPU energy should still accumulate.
        let summary = tracker.session_summary();
        assert!(summary.total_package_wh > 0.0);
        assert!(summary.total_gpu_wh > 0.0);
    }

    #[test]
    fn missing_power_data() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(None, None, None);
        let procs = vec![make_proc("Safari", 100.0)];

        // No power data — should not panic, should not accumulate energy.
        tracker.update(&procs, &hw, 1.0);
        let top = tracker.top_consumers(10);
        // Safari was seen but has 0 watts (no cpu_watts data).
        assert!(top.is_empty() || top[0].current_watts == 0.0);
    }

    #[test]
    fn dt_capping() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), None, Some(8.0));
        let procs = vec![make_proc("App", 100.0)];

        // Simulate daemon pause: dt = 600 seconds. Should be capped to MAX_DT_SECS.
        tracker.update(&procs, &hw, 600.0);

        let top = tracker.top_consumers(10);
        assert_eq!(top.len(), 1);
        // With capped dt of 30s: 6.0 * (30/3600) = 0.05 Wh
        let app = &top[0];
        let expected_wh = 6.0 * (MAX_DT_SECS / 3600.0);
        assert!(
            (app.cumulative_wh - expected_wh).abs() < 0.001,
            "Expected ~{:.4} Wh, got {:.4} Wh",
            expected_wh,
            app.cumulative_wh
        );
    }

    #[test]
    fn negative_dt_ignored() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), None, Some(8.0));
        let procs = vec![make_proc("App", 100.0)];

        // Negative dt should be clamped to MIN_DT_SECS, not cause issues.
        tracker.update(&procs, &hw, -5.0);
        // Should still produce finite values, just with minimal energy.
        let top = tracker.top_consumers(10);
        for entry in &top {
            assert!(entry.current_watts.is_finite());
            assert!(entry.cumulative_wh.is_finite());
        }
    }

    #[test]
    fn nan_dt_ignored() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), None, Some(8.0));
        let procs = vec![make_proc("App", 100.0)];

        // NaN dt: after clamping NaN to [MIN, MAX], f64::clamp returns NaN.
        // Our explicit is_finite check catches this.
        tracker.update(&procs, &hw, f64::NAN);
        let top = tracker.top_consumers(10);
        assert!(top.is_empty());
    }

    #[test]
    fn nan_cpu_usage_handled() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), None, Some(8.0));
        let procs = vec![
            make_proc("Good", 100.0),
            ProcessStats {
                pid: 2,
                name: "Bad".to_string(),
                cpu_usage: f32::NAN,
                memory_usage: 0,
                cpu_wall_ratio: None,
            },
        ];

        tracker.update(&procs, &hw, 1.0);
        let top = tracker.top_consumers(10);
        // "Bad" should be sanitized to 0 CPU and not appear with watts.
        for entry in &top {
            assert!(entry.current_watts.is_finite());
            assert!(entry.cumulative_wh.is_finite());
            assert!(!entry.current_watts.is_nan());
        }
    }

    #[test]
    fn negative_power_reading_clamped() {
        let mut tracker = EnergyTracker::new();
        // Negative watts (sensor glitch).
        let hw = make_hw(Some(-2.0), Some(-1.0), Some(-5.0));
        let procs = vec![make_proc("App", 100.0)];

        tracker.update(&procs, &hw, 1.0);
        let summary = tracker.session_summary();
        assert!(summary.total_cpu_wh >= 0.0);
        assert!(summary.total_gpu_wh >= 0.0);
        assert!(summary.total_package_wh >= 0.0);
    }

    #[test]
    fn cumulative_over_multiple_cycles() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), None, Some(8.0));
        let procs = vec![make_proc("App", 100.0)];

        // 10 cycles of 1 second each.
        for _ in 0..10 {
            tracker.update(&procs, &hw, 1.0);
        }

        let top = tracker.top_consumers(10);
        assert_eq!(top.len(), 1);
        // 6.0 W * 10 s / 3600 s = 0.01667 Wh
        let expected = 6.0 * 10.0 / 3600.0;
        assert!(
            (top[0].cumulative_wh - expected).abs() < 0.001,
            "Expected {:.5}, got {:.5}",
            expected,
            top[0].cumulative_wh
        );
    }

    #[test]
    fn process_name_normalization() {
        assert_eq!(normalize_process_name("  Chrome  "), "Chrome");
        assert_eq!(normalize_process_name(""), "<unknown>");
        assert_eq!(normalize_process_name("   "), "<unknown>");
        assert_eq!(normalize_process_name("Safari"), "Safari");
    }

    #[test]
    fn sanitize_f64_edge_cases() {
        assert_eq!(sanitize_f64(f64::NAN), 0.0);
        assert_eq!(sanitize_f64(f64::INFINITY), 0.0);
        assert_eq!(sanitize_f64(f64::NEG_INFINITY), 0.0);
        assert_eq!(sanitize_f64(-1.0), 0.0);
        assert_eq!(sanitize_f64(0.0), 0.0);
        assert_eq!(sanitize_f64(42.5), 42.5);
    }

    #[test]
    fn decay_removes_stale_entries() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), None, Some(8.0));
        let procs = vec![make_proc("OldApp", 100.0)];

        tracker.update(&procs, &hw, 1.0);
        assert_eq!(tracker.tracked_count(), 1);

        // Manually age the entry beyond DECAY_TIMEOUT_SECS.
        if let Some(acc) = tracker.accumulators.get_mut("OldApp") {
            // Set last_seen to a time far in the past by subtracting from now.
            // We can't easily fake Instant, but we can test decay_stale directly
            // with a future "now".
            acc.last_seen = Instant::now() - std::time::Duration::from_secs(7200);
        }

        // Add enough dummy entries to trigger decay (threshold is 50).
        for i in 0..55 {
            let name = format!("Proc{}", i);
            tracker.accumulators.insert(
                name,
                EnergyAccumulator {
                    current_watts: 0.0,
                    cumulative_wh: 0.0,
                    last_seen: Instant::now(),
                },
            );
        }

        // Trigger an update which will run decay.
        tracker.update(&[], &hw, 1.0);

        // OldApp should have been decayed.
        assert!(
            !tracker.accumulators.contains_key("OldApp"),
            "OldApp should have been decayed"
        );
    }

    #[test]
    fn evict_oldest_enforces_cap() {
        let mut tracker = EnergyTracker::new();

        // Insert MAX_TRACKED_PROCESSES + 100 entries.
        let now = Instant::now();
        for i in 0..(MAX_TRACKED_PROCESSES + 100) {
            let age = if i < 100 {
                // First 100 entries are old.
                std::time::Duration::from_secs(3600)
            } else {
                std::time::Duration::from_secs(0)
            };
            let last_seen = now.checked_sub(age).unwrap_or(now);
            tracker.accumulators.insert(
                format!("Proc{}", i),
                EnergyAccumulator {
                    current_watts: 0.0,
                    cumulative_wh: 0.0,
                    last_seen,
                },
            );
        }

        tracker.evict_oldest(now);
        assert!(
            tracker.accumulators.len() <= MAX_TRACKED_PROCESSES,
            "Map size {} exceeds cap {}",
            tracker.accumulators.len(),
            MAX_TRACKED_PROCESSES
        );
    }

    #[test]
    fn session_summary_co2_sanity() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), Some(2.0), Some(10.0));
        let procs = vec![make_proc("App", 100.0)];

        // Simulate 1 hour of running at 10W package power.
        // 3600 cycles of 1 second each.
        for _ in 0..3600 {
            tracker.update(&procs, &hw, 1.0);
        }

        let summary = tracker.session_summary();

        // Package: 10W * 1hr = 10 Wh = 0.01 kWh
        assert!(
            (summary.total_package_wh - 10.0).abs() < 0.1,
            "Package Wh: {}",
            summary.total_package_wh
        );

        // CO2: 0.01 kWh * 0.390 kg/kWh = 0.0039 kg
        let expected_co2 = 10.0 / 1000.0 * CO2_KG_PER_KWH;
        assert!(
            (summary.estimated_co2_kg - expected_co2).abs() < 0.001,
            "CO2: {} kg, expected: {} kg",
            summary.estimated_co2_kg,
            expected_co2
        );
    }

    #[test]
    fn record_savings_accumulates() {
        let mut tracker = EnergyTracker::new();

        // Record 5W saved for 10 seconds.
        tracker.record_savings(5.0, 10.0);
        let savings = tracker.savings_estimate_wh();
        // 5 * 10/3600 = 0.01389 Wh
        assert!(
            (savings - 5.0 * 10.0 / 3600.0).abs() < 0.001,
            "Savings: {}",
            savings
        );
    }

    #[test]
    fn record_savings_rejects_invalid() {
        let mut tracker = EnergyTracker::new();
        let before = tracker.savings_estimate_wh();

        tracker.record_savings(-1.0, 10.0);
        tracker.record_savings(5.0, -10.0);
        tracker.record_savings(f64::NAN, 10.0);
        tracker.record_savings(5.0, f64::INFINITY);

        assert_eq!(tracker.savings_estimate_wh(), before);
    }

    #[test]
    fn reset_clears_everything() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), Some(2.0), Some(10.0));
        let procs = vec![make_proc("App", 100.0)];

        tracker.update(&procs, &hw, 1.0);
        assert!(tracker.tracked_count() > 0);

        tracker.reset();
        assert_eq!(tracker.tracked_count(), 0);
        let summary = tracker.session_summary();
        assert_eq!(summary.total_cpu_wh, 0.0);
        assert_eq!(summary.total_gpu_wh, 0.0);
        assert_eq!(summary.total_package_wh, 0.0);
    }

    #[test]
    fn top_consumers_limits_output() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(10.0), None, Some(10.0));

        let mut procs = Vec::new();
        for i in 0..20 {
            procs.push(make_proc(&format!("App{}", i), 50.0));
        }

        tracker.update(&procs, &hw, 1.0);

        let top5 = tracker.top_consumers(5);
        assert_eq!(top5.len(), 5);

        let top100 = tracker.top_consumers(100);
        assert_eq!(top100.len(), 20);
    }

    #[test]
    fn multiple_pids_same_name_aggregate() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), None, Some(8.0));

        // Two processes with the same name (e.g., Chrome Helper instances).
        let procs = vec![
            ProcessStats {
                pid: 100,
                name: "Chrome Helper".to_string(),
                cpu_usage: 80.0,
                memory_usage: 0,
                cpu_wall_ratio: None,
            },
            ProcessStats {
                pid: 101,
                name: "Chrome Helper".to_string(),
                cpu_usage: 20.0,
                memory_usage: 0,
                cpu_wall_ratio: None,
            },
        ];

        tracker.update(&procs, &hw, 1.0);

        let top = tracker.top_consumers(10);
        // Both should be aggregated under one name.
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].name, "Chrome Helper");
        // Total CPU: 100%, so watts = 6.0 W (100/100 * 6.0).
        assert!((top[0].current_watts - 6.0).abs() < 0.01);
    }

    #[test]
    fn whitespace_process_name() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), None, Some(8.0));
        let procs = vec![make_proc("  Safari  ", 100.0), make_proc("Safari", 50.0)];

        tracker.update(&procs, &hw, 1.0);
        let top = tracker.top_consumers(10);
        // Both should aggregate under "Safari".
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].name, "Safari");
    }

    #[test]
    fn very_small_cpu_usage_below_threshold() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), None, Some(8.0));
        // Process with negligibly small CPU usage (below 0.01% threshold).
        let procs = vec![make_proc("Daemon", 0.001)];

        tracker.update(&procs, &hw, 1.0);
        let top = tracker.top_consumers(10);
        // Total CPU% is 0.001 which is below the 0.01 threshold, so no
        // power is attributed. This avoids amplifying measurement noise.
        assert!(top.is_empty() || top[0].current_watts == 0.0);
    }

    #[test]
    fn small_but_meaningful_cpu_usage() {
        let mut tracker = EnergyTracker::new();
        let hw = make_hw(Some(6.0), None, Some(8.0));
        // Process with small but above-threshold CPU usage.
        let procs = vec![make_proc("Daemon", 0.1)];

        tracker.update(&procs, &hw, 1.0);
        let top = tracker.top_consumers(10);
        assert_eq!(top.len(), 1);
        // Should get all of the CPU watts since it's the only process.
        assert!((top[0].current_watts - 6.0).abs() < 0.01);
        assert!(top[0].cumulative_wh.is_finite());
    }

    #[test]
    fn default_trait() {
        let tracker = EnergyTracker::default();
        assert_eq!(tracker.tracked_count(), 0);
    }
}
