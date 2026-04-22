//! # Daemon Behavior PIDs
//!
//! Behavioral-interactive PID set builder extracted from main.rs (Wave 15).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Build behavior-interactive PID set from UsageModel EMA data
//! - Suppress false positives via BEHAVIOR_DENYLIST (I/O-bound OS daemons)
//! - Merge JIT-compiling PIDs so decide_actions skips throttling them
//!
//! ## Ordering invariant
//! Must run AFTER jit_protected_pids are known (from daemon_sensor_tick)
//! and AFTER snapshot.top_processes is populated for this cycle.

use std::collections::HashSet;

use apollo_optimizer::collector::SystemSnapshot;
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::lock_ext::LockRecover;

/// Daemons that are I/O-bound (low cpu_wall_ratio) but must NEVER be classified
/// as interactive. [Android LMK] Protection earned by user interaction, not I/O.
const BEHAVIOR_DENYLIST: &[&str] = &[
    "searchpartyd",         // Find My / Handoff BLE scanning
    "corespeechd",          // Siri speech (background)
    "suggestd",             // Spotlight/Siri suggestions ML
    "duetexpertd",          // Siri predictions / Proactive
    "photoanalysisd",       // Photos ML tagging
    "mediaanalysisd",       // Media content analysis
    "intelligencecontextd", // Apple Intelligence
    "mlhostd",              // Core ML inference host
    "modelmanagerd",        // On-device model cache
    "rtcreportingd",        // RealTimeComm diagnostics
    "cfprefsd",             // Preference caching — 33 false boosts observed
    "xpcproxy",             // XPC launcher — ephemeral, 40 false boosts
    "log",                  // Unified log CLI — 30 false boosts
    "apollo-optimizerd",    // Self — 45 wasted journal entries
    "apollo-optimizerctl",  // Our own CLI client
    "diagnostics_agent",    // System diagnostics — throttled 749x + boosted
    "socketfilterfw",       // Application firewall — I/O-bound, not interactive
    "stable",               // /usr/libexec/stable — 67 false boosts
];

/// Build the behavior-interactive PID set for this cycle.
///
/// Processes with sustained low cpu_wall_ratio are I/O-bound and are classified
/// as interactive so decide_actions skips throttling them. JIT-compiling PIDs
/// from syscall_classifier are merged in unconditionally.
pub fn build_behavior_interactive_pids(
    state: &SharedState,
    snapshot: &SystemSnapshot,
    jit_protected_pids: &HashSet<u32>,
) -> HashSet<u32> {
    let model = state.usage.lock_recover();
    let interactive_names: HashSet<&str> = model
        .usage_model
        .entries()
        .iter()
        .filter(|(name, entry)| {
            apollo_optimizer::engine::usage_model::is_behavior_interactive(entry)
                && !BEHAVIOR_DENYLIST.iter().any(|d| name.contains(d))
        })
        .map(|(name, _)| name.as_str())
        .collect();
    let mut pids: HashSet<u32> = snapshot
        .top_processes
        .iter()
        .filter(|p| interactive_names.contains(p.name.as_str()))
        .map(|p| p.pid)
        .collect();
    pids.extend(jit_protected_pids.iter().copied());
    pids
}
