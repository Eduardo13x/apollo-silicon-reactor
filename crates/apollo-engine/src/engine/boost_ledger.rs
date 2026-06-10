//! Boost ledger — anti-ratchet decay for BoostProcess side-effects.
//!
//! Fight-hunt finding (2026-06-10, verified live): the Boost arm writes
//! `nice = -10` + `SchedulingTier::Foreground` and NOTHING ever reverts
//! them. Observed on a 20h-uptime system: 13 processes pinned at -10,
//! including a Photos widget, background daemons, and — via fork
//! inheritance — the user's shell and every child it spawns (alacritty
//! boosted → zsh inherits -10 → ps/awk/head all run at -10). Priority
//! inflation: when everything is high-priority, nothing is, and genuinely
//! normal-priority daemons starve. This is the "gets crazy after a while"
//! failure mode.
//!
//! Design (cooperative, conservative):
//! - Producer: the execute_actions Boost arm records (pid, start_sec) here
//!   after a successful boost. Re-boosts refresh the timestamp.
//! - Consumer: a periodic main-loop sweep asks for entries older than
//!   [`BOOST_TTL`]; for each, IF the pid is not the current foreground and
//!   its identity still matches (PID-recycle guard), it reverts
//!   nice → 0 and tier → Normal, then drops the entry.
//! - The current foreground pid is never reverted — its boost is live.
//! - Reverting the parent does not un-nice already-forked children, but
//!   shell children are ephemeral and new forks inherit the restored 0.
//!
//! Global-handle pattern mirrors `effect_decay::GLOBAL_WATCHDOG` — the
//! producer call site sits deep in execute_actions where threading a new
//! parameter through the signature chain is disruptive.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Boost side-effects are reverted once the entry is older than this and
/// the process is no longer foreground. Re-boosts refresh the clock, so a
/// continuously-qualifying interactive app never expires while in use.
pub const BOOST_TTL: Duration = Duration::from_secs(600);

struct Entry {
    boosted_at: Instant,
    start_sec: u64,
}

static LEDGER: Mutex<Option<HashMap<u32, Entry>>> = Mutex::new(None);

/// Producer: record (or refresh) a successful boost. Call after the Mach
/// tier write + renice landed.
pub fn record_boost(pid: u32, start_sec: u64) {
    let mut guard = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .get_or_insert_with(HashMap::new)
        .insert(pid, Entry { boosted_at: Instant::now(), start_sec });
}

/// Consumer: drain entries older than [`BOOST_TTL`], excluding the current
/// foreground pid. Returns (pid, start_sec) pairs the caller must verify
/// (identity) and revert. Entries returned are removed — a process that
/// gets boosted again simply re-enters the ledger.
pub fn drain_expired(foreground_pid: Option<u32>) -> Vec<(u32, u64)> {
    let mut guard = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = guard.as_mut() else {
        return Vec::new();
    };
    let now = Instant::now();
    let expired: Vec<u32> = map
        .iter()
        .filter(|(pid, e)| {
            Some(**pid) != foreground_pid && now.duration_since(e.boosted_at) >= BOOST_TTL
        })
        .map(|(pid, _)| *pid)
        .collect();
    expired
        .into_iter()
        .filter_map(|pid| map.remove(&pid).map(|e| (pid, e.start_sec)))
        .collect()
}

/// Drop entries for PIDs that no longer exist (process exited — nothing to
/// revert; the kernel already reclaimed everything).
pub fn cleanup(live_pids: &[u32]) {
    let mut guard = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = guard.as_mut() {
        let live: std::collections::HashSet<u32> = live_pids.iter().copied().collect();
        map.retain(|pid, _| live.contains(pid));
    }
}

/// Current ledger size (observability).
pub fn len() -> usize {
    let guard = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().map_or(0, HashMap::len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_drain_respects_ttl_and_foreground() {
        // Fresh entry: not expired → drain returns nothing.
        record_boost(999_001, 42);
        assert!(drain_expired(None).iter().all(|(p, _)| *p != 999_001));

        // Force-expire by rewriting the timestamp.
        {
            let mut g = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(m) = g.as_mut() {
                if let Some(e) = m.get_mut(&999_001) {
                    e.boosted_at = Instant::now() - BOOST_TTL - Duration::from_secs(1);
                }
            }
        }
        // Foreground exclusion holds even when expired.
        assert!(drain_expired(Some(999_001)).iter().all(|(p, _)| *p != 999_001));
        // Non-foreground expired entry drains exactly once.
        let drained = drain_expired(None);
        assert!(drained.contains(&(999_001, 42)));
        assert!(drain_expired(None).iter().all(|(p, _)| *p != 999_001));
    }

    #[test]
    fn cleanup_drops_dead_pids() {
        record_boost(999_002, 7);
        cleanup(&[1]); // 999_002 not alive
        let mut g = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!g.get_or_insert_with(HashMap::new).contains_key(&999_002));
    }
}
