//! # Daemon Process Scan Tick
//!
//! MemoryAnalyzer profiling loop + WakeStormDetector per-cycle tick extracted
//! from main.rs (Wave 32). [Fowler 2004] Strangler Fig — pure move.
//!
//! ## Responsibilities
//! - MemoryAnalyzer: profile top-50 processes; refine WSS for top-10 via TASK_VM_INFO
//! - Register confirmed memory leaks (probability ≥ 0.75) to ProcessRecoveryManager
//! - WakeStormDetector: record elevated wakeup rates, GC stale entries
//!
//! ## Ordering invariant
//! Must run AFTER proc_snaps is populated (process_enrichment) and BEFORE
//! the freeze decision pass (which reads proc_recovery for confirmed leakers).

use std::time::Duration;

use apollo_optimizer::engine::compressor_aware::query_memory_profile;
use apollo_optimizer::engine::memory_analyzer::MemoryAnalyzer;
use apollo_optimizer::engine::process_classifier::ProcessSnapshot;
use apollo_optimizer::engine::process_recovery::ProcessRecoveryManager;
use apollo_optimizer::engine::wake_storm_detector::WakeStormDetector;

/// Profile top-50 processes for memory leaks and elevated wakeup rates.
///
/// # Parameters
/// - `proc_snaps` — enriched process snapshots for this cycle
/// - `mem_analyzer` — MemoryAnalyzer instance (stateful per-PID leak tracking)
/// - `proc_recovery` — registers confirmed leakers for freeze consideration
/// - `wake_storm` — records wakeup-rate spikes, GCs stale entries
pub fn run_proc_scan_tick(
    proc_snaps: &[ProcessSnapshot],
    mem_analyzer: &mut MemoryAnalyzer,
    proc_recovery: &mut ProcessRecoveryManager,
    wake_storm: &mut WakeStormDetector,
) {
    for (i, snap) in proc_snaps.iter().take(50).enumerate() {
        let mut profile = mem_analyzer.analyze_process(
            snap.pid,
            &snap.name,
            snap.rss_bytes,
            snap.rss_bytes,
            snap.pageins_total as u64,
        );
        if i < 10 {
            if let Some(mem_profile) = query_memory_profile(snap.pid) {
                MemoryAnalyzer::refine_wss(&mut profile, mem_profile.working_set_bytes);
            }
        }
        if profile.memory_leak_probability >= 0.75 {
            proc_recovery.register_leak(
                snap.pid,
                snap.name.clone(),
                profile.memory_leak_probability,
                snap.rss_bytes,
            );
        }
    }
    proc_recovery.cleanup_resolved();

    for snap in proc_snaps.iter().take(50) {
        if snap.wakeups_per_sec > 10.0 {
            wake_storm.record_wakeup(snap.pid, snap.name.clone());
        }
    }
    wake_storm.cleanup_stale(Duration::from_secs(300));
}
