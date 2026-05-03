//! # Daemon Process Collector
//!
//! Per-cycle process-table operations extracted from the daemon main loop:
//!
//! - `build_process_tree`       — build the parent/child tree from sysinfo.
//! - `run_pre_sleep_unfreeze`   — release all SIGSTOP'd PIDs before system sleep.
//! - `run_ghost_pid_reconciliation` — evict dead PIDs from `frozen_state` / turbo.
//!
//! All three are small, self-contained, and free of cross-cycle state.

use std::collections::HashSet;
use std::path::Path;

use apollo_optimizer::collector::SystemCollector;
use apollo_optimizer::engine::daemon_helpers::{
    unfreeze_pids_verified, write_frozen_state,
};
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::display_turbo::DisplayTurbo;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::process_tree::{ProcessEntry, ProcessTree};
use apollo_optimizer::engine::sleep_notifier::SleepNotifier;

/// Build the parent/child process tree from the latest sysinfo snapshot.
///
/// Used by foreground-family detection, enrichment, and chromium visibility
/// checks. Cost is dominated by sysinfo's internal iteration (~1–2ms for
/// ~500 processes).
pub fn build_process_tree(collector: &SystemCollector) -> ProcessTree {
    let sys = collector.system();
    let entries: Vec<ProcessEntry> = sys
        .processes()
        .iter()
        .map(|(pid, process)| ProcessEntry {
            pid: pid.as_u32(),
            ppid: process.parent().map(|p| p.as_u32()).unwrap_or(0),
            name: process.name().to_string(),
            cpu_usage: process.cpu_usage(),
            memory_bytes: process.memory(),
        })
        .collect();
    ProcessTree::build(&entries)
}

/// Pre-sleep unfreeze — release every SIGSTOP'd PID before the kernel suspends.
///
/// `kIOMessageSystemWillSleep` fires ~30s before kernel suspension. Without
/// releasing our frozen PIDs here, they remain ineligible for jetsam / compressor
/// eviction during sleep, which forces macOS to kill more interactive helpers
/// (widgets, extensions) to reclaim memory.
///
/// A-B-A defense: `unfreeze_pids_verified` re-checks (pid, start_sec, name)
/// identity before SIGCONT so PIDs recycled during the race window are skipped.
/// [Saltzer & Kaashoek 2009] §3.3 Complete Mediation.
pub fn run_pre_sleep_unfreeze(
    state: &SharedState,
    frozen_state_path: &Path,
    display_turbo: &mut DisplayTurbo,
    sleep_notifier: &SleepNotifier,
) {
    if !sleep_notifier.will_sleep_pending() {
        return;
    }
    let mut frozen_guard = state.frozen_state.lock_recover();
    // Turbo PIDs live in frozen_guard too, so this covers both regular + turbo.
    let count = unfreeze_pids_verified(&frozen_guard);
    if count > 0 {
        // Snapshot thawed PIDs before clearing for cooldown bookkeeping.
        let thawed_pids: Vec<u32> = frozen_guard.keys().copied().collect();
        tracing::info!(
            count,
            "pre-sleep: released {} frozen PID(s) — \
             handing back to macOS memory manager",
            count
        );
        frozen_guard.clear();
        write_frozen_state(frozen_state_path, &frozen_guard);
        drop(frozen_guard);
        state.metrics.lock_recover().metrics.unfreezes_applied += count;
        // Mark thawed PIDs in cooldown to prevent gate_e re-freeze oscillation.
        // [Nygard 2018] §8.5 — circuit breaker hold-down after recovery.
        {
            let mut cooldown = state.freeze_cooldown.lock_recover();
            for pid in &thawed_pids {
                cooldown.mark_thawed(*pid);
            }
        }
    }
    display_turbo.clear_frozen();
    sleep_notifier.acknowledge();
}

/// Ghost-PID reconciliation — evict frozen_state entries whose PID is dead.
///
/// A frozen process can die via manual kill, Force Quit, or jetsam while kqueue
/// `NOTE_EXIT` isn't registered (e.g., after a daemon restart). Without this,
/// `frozen_state` retains ghost entries whose RSS is counted as `frozen_ram_mb`
/// even though the OS already reclaimed that memory.
///
/// `live_pids` must come from the authoritative sysinfo snapshot used this cycle.
/// Also triggers:
/// - `display_turbo.gc_dead_pids()` (in-memory, no disk write)
/// - `mach_qos.gc_dead_pids()` every 60 cycles (~30s) — libc::kill(pid,0) is
///   cheap but the internal HashMaps can grow large under Chrome.
pub fn run_ghost_pid_reconciliation(
    state: &SharedState,
    live_pids: &HashSet<u32>,
    frozen_state_path: &Path,
    display_turbo: &mut DisplayTurbo,
    cycle_count: u64,
) {
    let mut frozen_guard = state.frozen_state.lock_recover();
    let before = frozen_guard.len();
    frozen_guard.retain(|pid, _| live_pids.contains(pid));
    let removed = before - frozen_guard.len();
    if removed > 0 {
        tracing::info!(
            removed,
            "frozen_state: evicted {} ghost PID(s) \
             (died without kqueue notification)",
            removed
        );
        write_frozen_state(frozen_state_path, &frozen_guard);
    }
    display_turbo.gc_dead_pids(live_pids);
    drop(frozen_guard);

    if cycle_count % 60 == 0 {
        state.mach_qos.lock_recover().gc_dead_pids();
    }
}
