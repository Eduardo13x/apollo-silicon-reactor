//! # Daemon Sensor Tick
//!
//! Per-cycle hardware-telemetry reads extracted from the daemon main loop.
//!
//! Covers the fire-and-forget sensor pass that runs *before* the effective-
//! pressure aggregation downstream:
//!
//! - IOReport delta sample (P/E cluster, GPU, ANE, AMC bandwidth, per-component mW).
//! - SMC direct snapshot (power, lid, battery, per-die temps).
//! - KPC hardware performance counters (system-wide IPC).
//! - Rosetta AOT monitor poll (oahd-helper activity).
//! - Per-process energy ranking (`ri_billed_energy`) with derived hint maps.
//! - Syscall-aware profiling (JIT-compiling PID set).
//! - IOPMrootDomain direct thermal (every 10 cycles, aligned with HwPredictor).
//! - Upstream boost factors that feed `effective_pressure::compute`:
//!   memory bandwidth saturation, SMC direct thermal, battery overheat.
//!
//! ## Design invariants
//!
//! - **No lock acquisition.** Every sensor is owned by the main-loop (no
//!   `Arc<Mutex>` touched from here). This keeps the hot path deadlock-free
//!   and independent of the `mach_qos` / `frozen_state` lock hierarchy.
//! - **Fire-and-forget.** Per-sensor timeouts and `Option` fallbacks live
//!   inside the underlying readers; this module never blocks.
//! - **No pressure aggregation.** Boost factors are *produced* here but never
//!   mixed with `snapshot.pressure`. `effective_pressure::compute` remains
//!   the authoritative aggregator [Saltzer & Schroeder 1975 — Economy of
//!   Mechanism].
//!
//! Extracted from `main.rs` during the V1.1.0 Strangler Fig pass
//! [Fowler 2004].

use std::collections::{HashMap, HashSet};

use apollo_engine::collector::SystemSnapshot;
use apollo_engine::engine::energy_pid::{EnergyPidTracker, ProcessEnergyDelta};
use apollo_engine::engine::ioreport::{IOReportReader, IOReportSnapshot};
use apollo_engine::engine::kpc_counters::{KpcReader, KpcSnapshot};
use apollo_engine::engine::rosetta_monitor::RosettaMonitor;
use apollo_engine::engine::smc_direct::{SmcDirectReader, SmcSnapshot};
use apollo_engine::engine::syscall_classifier::SyscallClassifier;
use apollo_engine::engine::thermal_iokit::IoPmSnapshot;

/// Aggregate output of the per-cycle sensor pass.
///
/// Ownership semantics: the caller owns the produced values. `last_ioreport`
/// and `last_smc` are written back into long-lived `Option`s so they remain
/// available across cycles when the underlying reader is unavailable.
pub struct SensorTickOutput {
    /// KPC system-wide performance counters for this cycle (if available).
    pub kpc_snap: Option<KpcSnapshot>,
    /// Per-process energy deltas produced by `EnergyPidTracker::sample`.
    pub energy_pid_results: Vec<ProcessEnergyDelta>,
    /// `pid → IPC` hint map (filtered to IPC > 0.0).
    pub ipc_hints: HashMap<u32, f64>,
    /// `pid → wakeups/s` for processes above the 50 w/s vampire threshold.
    pub wakeup_hints: HashMap<u32, f64>,
    /// `pid → physical footprint (MB)` for accurate freeze ranking.
    pub footprint_hints: HashMap<u32, f64>,
    /// `pid → disk MB/s` for background processes above the 5 MB/s burst
    /// threshold (used to throttle I/O competition with LLM weight loads).
    pub io_burst_hints: HashMap<u32, f64>,
    /// `pid → anomaly score` for processes exceeding the effective MAD
    /// threshold for their learned behavioral baseline.
    pub anomaly_hints: HashMap<u32, f64>,
    /// Effective MAD threshold used to build `anomaly_hints`. Exposed so
    /// downstream metrics reporting can filter `energy_pid_results` against
    /// the *same* threshold without recomputing.
    pub anomaly_thresh: f64,
    /// PIDs currently in the `JitCompiling` syscall-profile state. Merged
    /// into `behavior_interactive_pids` downstream so `decide_actions`
    /// protects them from throttling.
    pub jit_protected_pids: HashSet<u32>,
    /// IOPMrootDomain thermal snapshot (every 10 cycles, aligned with
    /// HwPredictor).
    pub iopm_snap: Option<IoPmSnapshot>,
    /// +0.10 pressure boost when AMC bandwidth > 80% (memory-bound).
    pub mem_bw_boost: f64,
    /// Thermal boost derived from SMC direct CPU temperature
    /// (moderate / severe / critical tiers).
    pub smc_thermal_boost: f64,
    /// +0.12 pressure boost when SMC reports the battery is overheating.
    pub battery_overheat_boost: f64,
}

/// Run the per-cycle sensor pass.
///
/// Writes the latest IOReport/SMC snapshots back into the caller-owned
/// `Option`s so stale telemetry survives transient reader failures. Returns
/// the derived hint maps + boost factors for downstream consumption.
///
/// **Hot path.** This function must never acquire a `SharedState` lock and
/// never block. Preserve the error-swallowing (`.ok()`, `.unwrap_or_default()`)
/// semantics of the underlying sensor readers — they are intentional.
#[allow(clippy::too_many_arguments)]
pub fn run_sensor_tick(
    snapshot: &SystemSnapshot,
    cycle_count: u64,
    cycle_dt_secs: f64,
    ioreport: &mut IOReportReader,
    last_ioreport: &mut Option<IOReportSnapshot>,
    last_ioreport_sample: &mut std::time::Instant,
    smc_direct: &SmcDirectReader,
    last_smc: &mut Option<SmcSnapshot>,
    kpc_reader: &mut KpcReader,
    rosetta_monitor: &mut RosettaMonitor,
    energy_pid_tracker: &mut EnergyPidTracker,
    syscall_classifier: &mut SyscallClassifier,
) -> SensorTickOutput {
    // ── IOReport delta sample (throttled to ≥900ms between samples) ──────
    // end_sample() + begin_sample() gives a rolling inter-cycle window.
    if ioreport.available && last_ioreport_sample.elapsed() >= std::time::Duration::from_millis(900)
    {
        #[cfg(target_os = "macos")]
        {
            *last_ioreport = ioreport.end_sample();
            ioreport.begin_sample();
        }
        *last_ioreport_sample = std::time::Instant::now();
    }

    // ── SMC Direct: power, lid, sleep/wake, battery ─────────────
    if smc_direct.available {
        *last_smc = smc_direct.read_snapshot();
    }

    // ── KPC: hardware performance counters (IPC) ────────────────
    let kpc_snap = if kpc_reader.available {
        kpc_reader.sample()
    } else {
        None
    };

    // ── Rosetta AOT: poll for oahd-helper activity ──────────────
    rosetta_monitor.poll();

    // ── Per-process energy ranking (ri_billed_energy) ────────────
    let energy_pid_results = {
        let procs: Vec<(u32, &str)> = snapshot
            .top_processes
            .iter()
            .map(|p| (p.pid, p.name.as_str()))
            .collect();
        energy_pid_tracker.sample(&procs, cycle_dt_secs)
    };

    // Build IPC hint map for decide_actions (pid → IPC from rusage).
    let ipc_hints: HashMap<u32, f64> = energy_pid_results
        .iter()
        .filter(|e| e.ipc > 0.0)
        .map(|e| (e.pid, e.ipc))
        .collect();

    // Battery vampire detection: processes with >50 wakeups/s get priority throttle.
    let wakeup_hints = apollo_engine::engine::energy_pid::EnergyPidTracker::build_wakeup_hints(
        &energy_pid_results,
        50.0,
    );
    // Physical footprint hints for accurate freeze ranking.
    let footprint_hints =
        apollo_engine::engine::energy_pid::EnergyPidTracker::build_footprint_hints(
            &energy_pid_results,
        );
    // I/O burst hints: background processes writing >5 MB/s compete for
    // disk bandwidth with LLM model weight loading — throttle during inference.
    let io_burst_hints = apollo_engine::engine::energy_pid::EnergyPidTracker::build_io_burst_hints(
        &energy_pid_results,
        5.0,
    );
    // Behavioral anomaly hints: processes deviating ≥ threshold MADs from
    // their learned {ipc, wakeup_rate, disk_mbps} baseline get priority throttle.
    // Threshold is raised during cold start (< 10 warm baselines) to suppress
    // false positives from poorly-trained detectors. [Chandola 2009 §4.1]
    let anomaly_thresh = apollo_engine::engine::process_baseline::effective_threshold(
        energy_pid_tracker.baseline.warm_count(),
    );
    let anomaly_hints: HashMap<u32, f64> = energy_pid_results
        .iter()
        .filter(|r| r.anomaly_score >= anomaly_thresh)
        .map(|r| (r.pid, r.anomaly_score))
        .collect();

    // ── Syscall-aware profiling: identify JIT-compiling processes ──
    // Sample top processes through the syscall classifier and collect
    // PIDs currently in JitCompiling state.  These are merged into
    // behavior_interactive_pids below so decide_actions protects them
    // from throttling (same path as I/O-bound interactive processes).
    // Evict stale entries every 60 cycles to keep the HashMap bounded.
    let jit_protected_pids: HashSet<u32> = {
        let pids: Vec<u32> = snapshot.top_processes.iter().map(|p| p.pid).collect();
        if cycle_count.is_multiple_of(60) {
            syscall_classifier.evict_stale(&pids);
        }
        pids.iter()
            .filter_map(|&pid| {
                syscall_classifier
                    .sample(pid)
                    .filter(|p| {
                        *p == apollo_engine::engine::syscall_classifier::SyscallProfile::JitCompiling
                    })
                    .map(|_| pid)
            })
            .collect()
    };

    // ── IOPMrootDomain direct thermal (every 10 cycles, aligned with HwPredictor) ──
    let iopm_snap = if cycle_count.is_multiple_of(10) {
        apollo_engine::engine::thermal_iokit::read_iopm_state()
    } else {
        None
    };

    // ── Memory bandwidth pressure boost ─────────────────────────
    // AMC bandwidth > 80% = memory-bound → freeze more aggressively.
    let mem_bw_boost = last_ioreport
        .as_ref()
        .filter(|ir| ir.memory_bandwidth_saturated())
        .map(|_| 0.10)
        .unwrap_or(0.0);

    // ── SMC thermal direct boost ────────────────────────────────
    // CPU temp from SMC is real-time (<100µs). Use it to augment
    // thermal_bailout when powermetrics is stale.
    let smc_thermal_boost = last_smc
        .as_ref()
        .and_then(|s| s.cpu_temp_celsius)
        .map(|t| {
            if t >= 100.0 {
                0.30
            }
            // critical
            else if t >= 90.0 {
                0.15
            }
            // severe
            else if t >= 80.0 {
                0.05
            }
            // moderate
            else {
                0.0
            }
        })
        .unwrap_or(0.0);

    // ── Battery overheat protection ─────────────────────────────
    let battery_overheat_boost = last_smc
        .as_ref()
        .filter(|s| s.battery_overheating())
        .map(|_| 0.12)
        .unwrap_or(0.0);

    SensorTickOutput {
        kpc_snap,
        energy_pid_results,
        ipc_hints,
        wakeup_hints,
        footprint_hints,
        io_burst_hints,
        anomaly_hints,
        anomaly_thresh,
        jit_protected_pids,
        iopm_snap,
        mem_bw_boost,
        smc_thermal_boost,
        battery_overheat_boost,
    }
}
