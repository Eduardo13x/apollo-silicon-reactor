//! # Daemon Cluster Actions
//!
//! Coordinated multi-process freezing + Spotlight pressure gate extracted from main.rs (Wave 18).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Coordinated cluster freezing: when A is actioned AND B co-occurs with A (≥8 events),
//!   throttle B to exploit causal graph pressure-drop synergy [Pearl 2009]
//! - Spotlight pressure gate: pause/resume mdutil based on memory + swap pressure
//!   [mdutil handshake avoids SIGSTOP → no index corruption risk]
//!
//! ## Ordering invariant
//! Must run AFTER skill_tick (so actioned set reflects skill throttles) and AFTER
//! signal_digest + reclaim_forecast are computed.

use std::collections::HashSet;

use apollo_engine::collector::SystemCollector;
use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::types::RootAction;

pub struct ClusterActionsOutput {
    /// New throttle actions to append to the main actions vec.
    pub new_actions: Vec<RootAction>,
}

fn first_matching_process<'a>(
    eligible_processes: &'a [(u32, String)],
    pattern: &str,
) -> Option<(u32, &'a str)> {
    eligible_processes
        .iter()
        .find(|(_, name)| name.contains(pattern))
        .map(|(pid, name)| (*pid, name.as_str()))
}

/// Run coordinated cluster freezing for this cycle.
///
/// # Parameters
/// - `causal_pairs` — top co-occurrence pairs from outcome_tracker.top_causal_pairs()
/// - `current_actions` — actions accumulated so far (for actioned-set dedup)
/// - `collector` — SystemCollector (process iterator for partner lookup)
/// - `memory_pressure` — raw memory_pressure from snapshot
/// - `bg_pressure_threshold` — overflow_thresholds.bg_pressure (f64)
pub fn run_cluster_actions(
    causal_pairs: &[(&str, &str, u32)],
    current_actions: &[RootAction],
    collector: &SystemCollector,
    memory_pressure: f64,
    bg_pressure_threshold: f64,
) -> ClusterActionsOutput {
    let mut new_actions: Vec<RootAction> = Vec::new();

    // ── Coordinated multi-process freezing ──────────────────────────────────
    // [Pearl 2009] Causal graph clusters: if A is already actioned AND B always
    // co-occurs with A during pressure spikes (≥8 observed events), throttle B.
    // "Safari + cloudd together cause 20% drop; individually each is only 10%."
    // Gate: only triggers near the overflow threshold.
    if memory_pressure >= bg_pressure_threshold - 0.05 {
        let actioned: HashSet<String> = current_actions
            .iter()
            .filter_map(|a| match a {
                RootAction::ThrottleProcess { name, .. }
                | RootAction::FreezeProcess { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        // Preserve collector iteration order and protection semantics while
        // paying name allocation and safety classification only once per
        // cycle, rather than once per causal pair.
        let eligible_processes: Vec<(u32, String)> = collector
            .system()
            .processes()
            .iter()
            .filter_map(|(pid, proc)| {
                let name = proc.name().to_string();
                (!apollo_engine::engine::safety::is_protected_name(&name))
                    .then(|| (pid.as_u32(), name))
            })
            .collect();
        for (pa, pb, count) in causal_pairs {
            if *count < 8 {
                continue;
            }
            let a_acted = actioned.iter().any(|n| n.contains(pa));
            let b_acted = actioned.iter().any(|n| n.contains(pb));
            if a_acted == b_acted {
                continue; // both already actioned or neither
            }
            let missing = if a_acted { pb } else { pa };
            let partner = if a_acted { pa } else { pb };
            if actioned.iter().any(|n| n.contains(missing)) {
                continue;
            }
            if let Some((pid, proc_name)) = first_matching_process(&eligible_processes, missing) {
                new_actions.push(RootAction::throttle(
                    pid,
                    proc_name.to_owned(),
                    false,
                    format!(
                        "coordinated-cluster: co-occurs with {} (n={})",
                        partner, count
                    ),
                    DecisionReason::PressureContext,
                ));
            }
        }
    }

    // Spotlight pause gate removed 2026-04-30. The gate fired `mdutil -i off`
    // on transient pressure spikes (mem=1.0, swap=2.7GB), but `mdutil -i off`
    // ABORTS the in-progress indexing run rather than pausing it. When pressure
    // normalized and the gate re-enabled indexing, mds restarted from scratch.
    // Result: indexing never finished, pressure cycled forever.
    //
    // Root causes that justified the gate are now addressed elsewhere:
    //   • Podman VM right-sized (5GB → 2GB, 2026-04-30 manual)
    //   • Rust target/ excluded via `.metadata_never_index` (2026-04-30 manual)
    //   • SystemLogIngester gated on p_oom_30s > 0.50 (commit 631b1ac)
    //
    // Letting macOS manage Spotlight without interference lets indexing
    // actually complete, which is the user's stated goal. If pressure
    // genuinely spikes, jetsam handles mds_stores natively.

    ClusterActionsOutput { new_actions }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_preserves_collector_order() {
        let processes = vec![
            (7, "worker helper".to_owned()),
            (3, "worker renderer".to_owned()),
        ];
        assert_eq!(
            first_matching_process(&processes, "worker"),
            Some((7, "worker helper"))
        );
    }

    #[test]
    fn matching_returns_none_without_name_hit() {
        let processes = vec![(7, "worker helper".to_owned())];
        assert_eq!(first_matching_process(&processes, "browser"), None);
    }
}
