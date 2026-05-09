//! Memory Budget Allocator — per-process memory quotas for 8GB systems.
//!
//! Instead of reacting to global pressure, proactively assign each process
//! a memory budget based on its importance and behavior.  When a process
//! exceeds its budget, Apollo asks the kernel to reclaim from it first
//! (via jetsam inactive limits) rather than triggering system-wide pressure.
//!
//! Budget allocation principles:
//!   - Foreground app gets the largest share (up to 40% of available)
//!   - Build tools (rustc/cargo) get a temporary large allocation
//!   - Background apps get proportional shares based on usage_model EMAs
//!   - System reserve: 2.0 GB always untouched (kernel, WindowServer, etc.)
//!
//! Enforcement is soft: exceeding budget ≠ immediate kill.  The kernel's
//! jetsam inactive limit triggers reclamation when the process backgrounds,
//! providing a "graceful shed" rather than a hard cap.

/// System reserve that is never allocated to user processes (bytes).
/// On 8GB: kernel_task + WindowServer + coreaudiod + system daemons ≈ 2.0 GB.
const SYSTEM_RESERVE_BYTES: u64 = 2_000_000_000;

/// Minimum budget for any tracked process (bytes).  Below this, jetsam
/// limits are too tight and cause constant reclamation churn.
const MIN_BUDGET_BYTES: u64 = 64 * 1024 * 1024; // 64 MB

/// Maximum share a single foreground app can claim.
const MAX_FOREGROUND_SHARE: f64 = 0.40;

/// Maximum share a single background app can claim.
const MAX_BACKGROUND_SHARE: f64 = 0.15;

/// A computed memory budget for one process.
#[derive(Debug, Clone)]
pub struct ProcessBudget {
    pub pid: u32,
    pub name: String,
    /// Allocated budget in bytes.
    pub budget_bytes: u64,
    /// Current RSS in bytes.
    pub current_rss: u64,
    /// Working set (measured or estimated) in bytes.
    pub working_set_bytes: u64,
    /// Budget as jetsam inactive limit (MB, for set_memlimit).
    pub inactive_limit_mb: i32,
    /// True if process is currently over budget.
    pub over_budget: bool,
    /// How much over budget (bytes, 0 if under).
    pub excess_bytes: u64,
}

/// Input data for one process to the budget allocator.
#[derive(Debug, Clone)]
pub struct ProcessBudgetInput {
    pub pid: u32,
    pub name: String,
    pub rss_bytes: u64,
    pub working_set_bytes: u64,
    pub is_foreground: bool,
    pub is_build_tool: bool,
    /// Usage model presence EMA (0.0-1.0). Higher = more important.
    pub presence_ema: f64,
    /// Usage model interactive EMA (0.0-1.0). Higher = more user-facing.
    pub interactive_ema: f64,
}

/// Compute memory budgets for a set of processes.
///
/// `total_ram`: System total RAM in bytes (e.g., 8 × 1024³ for 8GB).
/// `processes`: Input data for each tracked process.
///
/// Returns budgets sorted by excess (most over-budget first).
pub fn compute_budgets(total_ram: u64, processes: &[ProcessBudgetInput]) -> Vec<ProcessBudget> {
    let allocatable = total_ram.saturating_sub(SYSTEM_RESERVE_BYTES);

    // Phase 1: Compute raw weights for each process.
    let mut weights: Vec<(usize, f64)> = Vec::with_capacity(processes.len());
    let mut total_weight = 0.0;

    for (i, p) in processes.iter().enumerate() {
        let weight = if p.is_foreground {
            // Foreground: high weight, scaled by interactive EMA.
            3.0 + p.interactive_ema * 2.0
        } else if p.is_build_tool {
            // Build tools: temporarily important.
            2.5
        } else {
            // Background: proportional to how often the user actually uses this app.
            0.5 + p.presence_ema * 2.0 + p.interactive_ema * 1.0
        };

        weights.push((i, weight));
        total_weight += weight;
    }

    if total_weight < 0.001 || processes.is_empty() {
        return vec![];
    }

    // Phase 2: Allocate proportional to weight, respecting caps.
    let mut budgets = Vec::with_capacity(processes.len());

    for &(i, weight) in &weights {
        let p = &processes[i];
        let share = weight / total_weight;

        // Cap shares.
        let capped_share = if p.is_foreground {
            share.min(MAX_FOREGROUND_SHARE)
        } else {
            share.min(MAX_BACKGROUND_SHARE)
        };

        let mut budget_bytes = (allocatable as f64 * capped_share) as u64;

        // Floor: at least the working set or MIN_BUDGET, whichever is larger.
        budget_bytes = budget_bytes.max(p.working_set_bytes).max(MIN_BUDGET_BYTES);

        // Ceiling: don't allocate more than 50% of total allocatable to one process.
        budget_bytes = budget_bytes.min(allocatable / 2);

        let over_budget = p.rss_bytes > budget_bytes;
        let excess = p.rss_bytes.saturating_sub(budget_bytes);

        // Jetsam inactive limit: budget in MB (kernel uses MB granularity).
        // Add 20% headroom to avoid aggressive reclamation of legitimate spikes.
        let inactive_limit_mb = ((budget_bytes as f64 * 1.2) / (1024.0 * 1024.0)) as i32;

        budgets.push(ProcessBudget {
            pid: p.pid,
            name: p.name.clone(),
            budget_bytes,
            current_rss: p.rss_bytes,
            working_set_bytes: p.working_set_bytes,
            inactive_limit_mb: inactive_limit_mb.max(64), // Never below 64MB
            over_budget,
            excess_bytes: excess,
        });
    }

    // Sort: most over-budget first (for priority enforcement).
    budgets.sort_by(|a, b| b.excess_bytes.cmp(&a.excess_bytes));
    budgets
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const RAM_8GB: u64 = 8 * 1024 * 1024 * 1024;

    fn make_input(
        name: &str,
        rss_mb: u64,
        fg: bool,
        build: bool,
        presence: f64,
        interactive: f64,
    ) -> ProcessBudgetInput {
        ProcessBudgetInput {
            pid: 1000,
            name: name.to_string(),
            rss_bytes: rss_mb * 1024 * 1024,
            working_set_bytes: rss_mb * 1024 * 1024, // assume WSS ≈ RSS for tests
            is_foreground: fg,
            is_build_tool: build,
            presence_ema: presence,
            interactive_ema: interactive,
        }
    }

    #[test]
    fn foreground_gets_most() {
        let inputs = vec![
            make_input("Claude", 500, true, false, 0.8, 0.9),
            make_input("Dropbox", 200, false, false, 0.3, 0.0),
        ];
        let budgets = compute_budgets(RAM_8GB, &inputs);
        assert_eq!(budgets.len(), 2);
        let claude = budgets.iter().find(|b| b.name == "Claude").unwrap();
        let dropbox = budgets.iter().find(|b| b.name == "Dropbox").unwrap();
        assert!(
            claude.budget_bytes > dropbox.budget_bytes,
            "foreground should get more: {} vs {}",
            claude.budget_bytes,
            dropbox.budget_bytes
        );
    }

    #[test]
    fn build_tools_get_large_allocation() {
        let inputs = vec![
            make_input("rustc", 1200, false, true, 0.2, 0.0),
            make_input("Dropbox", 200, false, false, 0.3, 0.0),
        ];
        let budgets = compute_budgets(RAM_8GB, &inputs);
        let rustc = budgets.iter().find(|b| b.name == "rustc").unwrap();
        let dropbox = budgets.iter().find(|b| b.name == "Dropbox").unwrap();
        assert!(
            rustc.budget_bytes > dropbox.budget_bytes,
            "build tool > daemon: {} vs {}",
            rustc.budget_bytes,
            dropbox.budget_bytes
        );
    }

    #[test]
    fn over_budget_detection() {
        // Brave at 2.5GB RSS but WSS only 800MB (lots of cached/stale tabs).
        let inputs = vec![
            ProcessBudgetInput {
                pid: 1000,
                name: "Brave".to_string(),
                rss_bytes: 2500 * 1024 * 1024,
                working_set_bytes: 800 * 1024 * 1024,
                is_foreground: false,
                is_build_tool: false,
                presence_ema: 0.5,
                interactive_ema: 0.3,
            },
            make_input("Claude", 500, true, false, 0.8, 0.9),
            make_input("Terminal", 100, false, false, 0.3, 0.2),
        ];
        let budgets = compute_budgets(RAM_8GB, &inputs);
        let brave = budgets.iter().find(|b| b.name == "Brave").unwrap();
        assert!(
            brave.over_budget,
            "2.5GB Brave (WSS=800MB) should be over budget"
        );
        assert!(brave.excess_bytes > 0);
    }

    #[test]
    fn respects_minimum_budget() {
        let inputs = vec![make_input("tiny", 10, false, false, 0.01, 0.0)];
        let budgets = compute_budgets(RAM_8GB, &inputs);
        assert!(
            budgets[0].budget_bytes >= MIN_BUDGET_BYTES,
            "budget {} should be >= min {}",
            budgets[0].budget_bytes,
            MIN_BUDGET_BYTES
        );
    }

    #[test]
    fn empty_input() {
        let budgets = compute_budgets(RAM_8GB, &[]);
        assert!(budgets.is_empty());
    }

    #[test]
    fn inactive_limit_has_headroom() {
        let inputs = vec![make_input("app", 500, true, false, 0.5, 0.5)];
        let budgets = compute_budgets(RAM_8GB, &inputs);
        let budget_mb = budgets[0].budget_bytes / (1024 * 1024);
        // Inactive limit should be ~120% of budget.
        assert!(
            budgets[0].inactive_limit_mb as u64 > budget_mb,
            "inactive_limit {}MB should be > budget {}MB",
            budgets[0].inactive_limit_mb,
            budget_mb
        );
    }
}
