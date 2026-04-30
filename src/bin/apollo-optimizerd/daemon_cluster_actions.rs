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
use std::time::{Duration, Instant};

use apollo_optimizer::collector::SystemCollector;
use apollo_optimizer::engine::types::RootAction;

pub struct ClusterActionsOutput {
    /// New throttle/spotlight actions to append to the main actions vec.
    pub new_actions: Vec<RootAction>,
    /// Updated spotlight paused state.
    pub spotlight_paused: bool,
    /// Updated timestamp when spotlight was paused.
    pub spotlight_paused_at: Option<Instant>,
    /// Timestamp of last `mdutil -i off` re-assertion. Tracked separately
    /// from `spotlight_paused_at` so re-asserting doesn't reset the
    /// minimum-hold gate (300s) used by the resume path.
    pub spotlight_last_assert_at: Option<Instant>,
}

/// Run coordinated cluster freezing and spotlight pressure gate for this cycle.
///
/// # Parameters
/// - `causal_pairs` — top co-occurrence pairs from outcome_tracker.top_causal_pairs()
/// - `current_actions` — actions accumulated so far (for actioned-set dedup)
/// - `collector` — SystemCollector (process iterator for partner lookup)
/// - `memory_pressure` — raw memory_pressure from snapshot
/// - `swap_used_bytes` — raw swap_used_bytes from snapshot
/// - `bg_pressure_threshold` — overflow_thresholds.bg_pressure (f64)
/// - `spotlight_paused` — current spotlight paused state (updated in-place)
/// - `spotlight_paused_at` — timestamp when spotlight was paused (updated in-place)
#[allow(clippy::too_many_arguments)]
pub fn run_cluster_actions(
    causal_pairs: &[(&str, &str, u32)],
    current_actions: &[RootAction],
    collector: &SystemCollector,
    memory_pressure: f64,
    swap_used_bytes: u64,
    bg_pressure_threshold: f64,
    spotlight_paused: bool,
    spotlight_paused_at: Option<Instant>,
    spotlight_last_assert_at: Option<Instant>,
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
                if proc_name.contains(missing)
                    && !actioned.iter().any(|n| n.contains(missing))
                {
                    new_actions.push(RootAction::throttle(
                        pid.as_u32(),
                        proc_name,
                        false,
                        format!(
                            "coordinated-cluster: co-occurs with {} (n={})",
                            partner, count
                        ),
                    ));
                    break;
                }
            }
        }
    }

    // ── Spotlight pressure gate ──────────────────────────────────────────────
    // Pause Spotlight indexing when swap is heavy; resume when pressure normalizes.
    // Uses mdutil (clean handshake with Spotlight server) — no index corruption.
    // Gate: memory_pressure ≥ 0.75 AND swap ≥ 1.5 GB → pause.
    // Re-enable: memory_pressure < 0.35 AND swap < 1.0 GB AND paused ≥ 300s.
    // [Previously 0.55 re-enable was too aggressive: rapid on/off + mdworker storms]
    let mut new_spotlight_paused = spotlight_paused;
    let mut new_spotlight_paused_at = spotlight_paused_at;
    let mut new_spotlight_last_assert_at = spotlight_last_assert_at;
    let swap_gb = swap_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if std::path::Path::new("/usr/bin/mdutil").exists() {
        if !spotlight_paused && memory_pressure >= 0.75 && swap_gb >= 1.5 {
            new_actions.push(RootAction::ToggleSpotlight {
                enabled: false,
                reason: format!(
                    "swap-pressure: mem={:.2} swap={:.1}GB",
                    memory_pressure, swap_gb
                ),
            });
            new_spotlight_paused = true;
            let now = Instant::now();
            new_spotlight_paused_at = Some(now);
            new_spotlight_last_assert_at = Some(now);
        } else if spotlight_paused
            && memory_pressure < 0.35
            && swap_gb < 1.0
            && spotlight_paused_at
                .map(|t| t.elapsed() >= Duration::from_secs(300))
                .unwrap_or(true)
        {
            new_actions.push(RootAction::ToggleSpotlight {
                enabled: true,
                reason: "pressure-normalized: re-enabling spotlight".to_string(),
            });
            new_spotlight_paused = false;
            new_spotlight_paused_at = None;
            new_spotlight_last_assert_at = None;
        } else if spotlight_paused
            && memory_pressure >= 0.60
            && spotlight_last_assert_at
                .map(|t| t.elapsed() >= Duration::from_secs(120))
                .unwrap_or(true)
        {
            // Re-assert pause every 120s while pressure is still elevated.
            // mdutil only acts on the edge — macOS or other apps can flip
            // indexing back on (volume mounts, mdfind, system events) and
            // Apollo would silently drift out of sync with reality.
            // Observed 2026-04-30: Spotlight kept re-indexing despite
            // Apollo believing it was paused.
            new_actions.push(RootAction::ToggleSpotlight {
                enabled: false,
                reason: format!(
                    "re-assert-pause: mem={:.2} swap={:.1}GB",
                    memory_pressure, swap_gb
                ),
            });
            new_spotlight_last_assert_at = Some(Instant::now());
        }
    }

    ClusterActionsOutput {
        new_actions,
        spotlight_paused: new_spotlight_paused,
        spotlight_paused_at: new_spotlight_paused_at,
        spotlight_last_assert_at: new_spotlight_last_assert_at,
    }
}
