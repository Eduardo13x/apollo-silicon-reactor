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
            for (pid, proc) in collector.system().processes() {
                let proc_name = proc.name().to_string();
                if proc_name.contains(missing) && !actioned.iter().any(|n| n.contains(missing)) {
                    new_actions.push(RootAction::throttle(
                        pid.as_u32(),
                        proc_name,
                        false,
                        format!(
                            "coordinated-cluster: co-occurs with {} (n={})",
                            partner, count
                        ),
                        DecisionReason::PressureContext,
                    ));
                    break;
                }
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
