//! # Daemon Memory Budget
//!
//! Jetsam inactive-limit enforcement from memory budget computation extracted
//! from main.rs (Wave 28). [Fowler 2004] Strangler Fig — pure move.
//!
//! ## Responsibilities
//! - When pressure ≥ 0.60: compute per-process jetsam inactive limits
//! - Use TASK_VM_INFO WSS when available, fault-rate heuristic otherwise
//! - Apply set_memlimit() to over-budget processes (active=0 = never kill)
//!
//! ## Ordering invariant
//! Must run AFTER proc_snaps is populated (process_enrichment) and BEFORE
//! the main decision pass.

use apollo_optimizer::engine::compressor_aware::query_memory_profile;
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::jetsam_control;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::memory_analyzer::MemoryAnalyzer;
use apollo_optimizer::engine::memory_budget::{self, ProcessBudgetInput};
use apollo_optimizer::engine::overflow_guard::is_build_tool_name;
use apollo_optimizer::engine::process_classifier::ProcessSnapshot;

/// Enforce jetsam inactive limits for over-budget processes under memory pressure.
///
/// # Parameters
/// - `memory_pressure` — effective memory pressure for this cycle
/// - `total_ram` — total system RAM bytes
/// - `state` — SharedState (usage_model lock for presence/interactive EMAs)
/// - `proc_snaps` — enriched process snapshots for this cycle
/// - `mem_analyzer` — for major_fault_rate() WSS fallback
pub fn run_memory_budget(
    memory_pressure: f64,
    total_ram: u64,
    state: &SharedState,
    proc_snaps: &[ProcessSnapshot],
    mem_analyzer: &MemoryAnalyzer,
) {
    if memory_pressure < 0.60 {
        return;
    }

    let usage_guard = state.usage.lock_recover();
    let budget_inputs: Vec<ProcessBudgetInput> = proc_snaps
        .iter()
        .take(30)
        .filter(|s| s.rss_bytes > 50 * 1024 * 1024)
        .map(|s| {
            let (presence, interactive) = usage_guard
                .usage_model
                .entries()
                .get(&s.name.to_ascii_lowercase())
                .map(|e| (e.presence_ema, e.interactive_ema))
                .unwrap_or((0.1, 0.0));
            // Use real WSS from TASK_VM_INFO when available,
            // fall back to fault-rate heuristic.
            let wss_bytes = query_memory_profile(s.pid)
                .map(|p| p.working_set_bytes)
                .unwrap_or_else(|| {
                    let fault_rate = mem_analyzer.major_fault_rate(s.pid);
                    if fault_rate > 50.0 {
                        (s.rss_bytes as f64 * 1.3) as u64
                    } else {
                        s.rss_bytes
                    }
                });
            ProcessBudgetInput {
                pid: s.pid,
                name: s.name.clone(),
                rss_bytes: s.rss_bytes,
                working_set_bytes: wss_bytes,
                is_foreground: s.has_gui_window && s.secs_since_foreground == 0,
                is_build_tool: is_build_tool_name(&s.name),
                presence_ema: presence,
                interactive_ema: interactive,
            }
        })
        .collect();
    drop(usage_guard);

    if budget_inputs.is_empty() {
        return;
    }

    let budgets = memory_budget::compute_budgets(total_ram, &budget_inputs);
    for budget in budgets.iter().filter(|b| b.over_budget) {
        let _ = jetsam_control::set_memlimit(
            budget.pid,
            0, // active: unlimited (don't kill foreground)
            budget.inactive_limit_mb,
        );
    }
}
