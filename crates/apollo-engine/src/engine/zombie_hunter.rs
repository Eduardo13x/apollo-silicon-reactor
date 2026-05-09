//! Zombie Hunter — detects genuinely useless processes
//!
//! Identifies five classes of "dead weight":
//!
//! 1. True zombies   — kernel `SZOMB` state (Z in `ps`)
//! 2. Orphans        — parent PID dead, process left running
//! 3. Ghost helpers  — XPC/helper for an app unused in >24 h
//! 4. Wakeup burners — high wakeup rate but zero user benefit
//! 5. Memory hoarders— large RSS, no UI, unused >1 h

use std::collections::HashMap;
use std::time::Instant;

// ── Result types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZombieClass {
    /// Process in `Z` (zombie) kernel state — parent must call wait().
    TrueZombie,
    /// Parent process is dead; child still running.
    Orphan,
    /// Helper / XPC process whose host application hasn't been used in >N hours.
    GhostHelper,
    /// Process that wakes up >N times/sec but produces no visible output.
    WakeupBurner,
    /// Process holding >N MB of RAM with no UI and no recent user interaction.
    MemoryHoarder,
}

#[derive(Debug, Clone)]
pub struct DeadWeightProcess {
    pub pid: u32,
    pub name: String,
    pub zombie_class: ZombieClass,
    pub wasted_rss_bytes: u64,
    pub wakeups_per_sec: f32,
    /// Conservative recommendation: Kill, Suspend, or NiceToMax.
    pub recommended_action: ZombieAction,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZombieAction {
    /// Immediately send SIGKILL — safe for true zombies / clear orphans.
    Kill,
    /// Send SIGSTOP — reversible, good for ghost helpers and hoarders.
    Suspend,
    /// renice to +20 — least invasive, for uncertain cases.
    NiceToMax,
}

// ── Detection snapshot ────────────────────────────────────────────────────────

/// Minimal process info needed for zombie-hunting decisions.
#[derive(Debug, Clone)]
pub struct HuntSnapshot {
    pub pid: u32,
    pub ppid: u32,
    pub name: String,
    pub is_kernel_zombie: bool, // 'Z' in ps STAT column
    pub parent_alive: bool,
    pub has_gui_window: bool,
    pub rss_bytes: u64,
    pub cpu_percent: f32,
    pub wakeups_per_sec: f32,
    pub secs_since_user_interaction: u64,
    /// PID of the "parent app" (if this is a helper process).
    pub host_app_pid: Option<u32>,
    /// Is the host application currently running?
    pub host_app_running: bool,
    /// How long has the host app been absent (0 if still running).
    pub host_app_absent_secs: u64,
}

// ── Thresholds ────────────────────────────────────────────────────────────────

/// Heuristic thresholds — all tuneable.
pub struct HunterConfig {
    /// Helpers whose host has been absent longer than this are ghost helpers.
    pub ghost_helper_threshold_secs: u64,
    /// Wakeups/sec above this with no user interaction = wakeup burner.
    pub wakeup_burner_threshold: f32,
    /// RSS above this with no UI and no recent interaction = hoarder.
    pub memory_hoarder_threshold_bytes: u64,
    /// Interaction timeout for hoarder classification.
    pub hoarder_idle_threshold_secs: u64,
}

impl Default for HunterConfig {
    fn default() -> Self {
        Self {
            ghost_helper_threshold_secs: 86_400,               // 24 h
            wakeup_burner_threshold: 20.0,                     // 20 wakeups/sec
            memory_hoarder_threshold_bytes: 256 * 1024 * 1024, // 256 MB (was 512 MB)
            hoarder_idle_threshold_secs: 1800,                 // 30 min (was 1 h)
        }
    }
}

// ── Tracker for persistent state ─────────────────────────────────────────────

/// Tracks how long a process has shown suspicious behaviour.
pub struct ZombieHunter {
    pub config: HunterConfig,
    /// pid → (first_seen_suspicious, consecutive_suspicious_cycles)
    suspicious_history: HashMap<u32, (Instant, u32)>,
    /// Cycles a process must be suspicious before we act.
    confirmation_cycles: u32,
}

impl ZombieHunter {
    pub fn new() -> Self {
        Self {
            config: HunterConfig::default(),
            suspicious_history: HashMap::new(),
            confirmation_cycles: 3, // Must be suspicious 3 consecutive cycles
        }
    }

    /// Classify a single snapshot into `Option<DeadWeightProcess>`.
    /// Returns `None` when the process appears legitimate.
    pub fn evaluate(&mut self, snap: &HuntSnapshot) -> Option<DeadWeightProcess> {
        // ── Rule 1: true zombie — immediate, no confirmation needed ────────
        if snap.is_kernel_zombie {
            return Some(DeadWeightProcess {
                pid: snap.pid,
                name: snap.name.clone(),
                zombie_class: ZombieClass::TrueZombie,
                wasted_rss_bytes: snap.rss_bytes,
                wakeups_per_sec: 0.0,
                recommended_action: ZombieAction::Kill,
                reason: "Process in kernel ZOMBIE state; parent must call wait()".into(),
            });
        }

        // ── Rule 2: orphan — immediate, no confirmation needed ─────────────
        if !snap.parent_alive && snap.ppid != 1 {
            // ppid==1 means launchd adopted it (legitimate re-parent)
            return Some(DeadWeightProcess {
                pid: snap.pid,
                name: snap.name.clone(),
                zombie_class: ZombieClass::Orphan,
                wasted_rss_bytes: snap.rss_bytes,
                wakeups_per_sec: snap.wakeups_per_sec,
                recommended_action: ZombieAction::Kill,
                reason: "Parent process is dead and process was not re-parented to launchd".into(),
            });
        }

        // ── Soft rules: build candidate if any rule fires, then confirm ────
        // We check all soft rules first, pick the best match, then confirm
        // ONCE per evaluation cycle (counter +1 per cycle, not per rule).

        let candidate = self.check_soft_rules(snap);

        if candidate.is_some() {
            // This process is suspicious — bump confirmation counter once
            let entry = self
                .suspicious_history
                .entry(snap.pid)
                .or_insert((Instant::now(), 0));
            entry.1 += 1;

            if entry.1 >= self.confirmation_cycles {
                return candidate;
            }
            // Not yet confirmed — keep counter, return nothing this cycle
        } else {
            // No rule matched — process appears legitimate, reset counter
            self.suspicious_history.remove(&snap.pid);
        }

        None
    }

    /// Evaluate all soft rules and return the first matching candidate
    /// (or None if the process looks legitimate).
    fn check_soft_rules(&self, snap: &HuntSnapshot) -> Option<DeadWeightProcess> {
        // ── Rule 3: ghost helper ───────────────────────────────────────────
        if snap.host_app_pid.is_some()
            && !snap.host_app_running
            && snap.host_app_absent_secs >= self.config.ghost_helper_threshold_secs
        {
            return Some(DeadWeightProcess {
                pid: snap.pid,
                name: snap.name.clone(),
                zombie_class: ZombieClass::GhostHelper,
                wasted_rss_bytes: snap.rss_bytes,
                wakeups_per_sec: snap.wakeups_per_sec,
                recommended_action: ZombieAction::Suspend,
                reason: format!(
                    "Helper process whose host app has been absent for {:.1} h",
                    snap.host_app_absent_secs as f32 / 3600.0
                ),
            });
        }

        // ── Rule 4: wakeup burner ──────────────────────────────────────────
        if snap.wakeups_per_sec >= self.config.wakeup_burner_threshold
            && !snap.has_gui_window
            && snap.secs_since_user_interaction > 300
        {
            return Some(DeadWeightProcess {
                pid: snap.pid,
                name: snap.name.clone(),
                zombie_class: ZombieClass::WakeupBurner,
                wasted_rss_bytes: snap.rss_bytes,
                wakeups_per_sec: snap.wakeups_per_sec,
                recommended_action: ZombieAction::NiceToMax,
                reason: format!(
                    "{:.0} wakeups/sec with no GUI and no user interaction for {}s",
                    snap.wakeups_per_sec, snap.secs_since_user_interaction
                ),
            });
        }

        // ── Rule 5: memory hoarder ─────────────────────────────────────────
        if snap.rss_bytes >= self.config.memory_hoarder_threshold_bytes
            && !snap.has_gui_window
            && snap.secs_since_user_interaction >= self.config.hoarder_idle_threshold_secs
        {
            return Some(DeadWeightProcess {
                pid: snap.pid,
                name: snap.name.clone(),
                zombie_class: ZombieClass::MemoryHoarder,
                wasted_rss_bytes: snap.rss_bytes,
                wakeups_per_sec: snap.wakeups_per_sec,
                recommended_action: ZombieAction::Suspend,
                reason: format!(
                    "Holds {} MB RSS with no UI and idle for {:.1} h",
                    snap.rss_bytes / 1024 / 1024,
                    snap.secs_since_user_interaction as f32 / 3600.0
                ),
            });
        }

        None
    }

    /// Evaluate many snapshots and return all dead-weight processes, sorted
    /// by `wasted_rss_bytes` descending (biggest memory waste first).
    pub fn evaluate_all(&mut self, snaps: &[HuntSnapshot]) -> Vec<DeadWeightProcess> {
        let mut dead_weight: Vec<DeadWeightProcess> =
            snaps.iter().filter_map(|s| self.evaluate(s)).collect();

        dead_weight.sort_by(|a, b| b.wasted_rss_bytes.cmp(&a.wasted_rss_bytes));

        dead_weight
    }

    /// Remove stale entries for PIDs that are gone.
    pub fn cleanup(&mut self, live_pids: &[u32]) {
        let live_set: std::collections::HashSet<u32> = live_pids.iter().copied().collect();
        self.suspicious_history
            .retain(|pid, _| live_set.contains(pid));
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    /// Total memory that could be reclaimed from all detected dead weight.
    pub fn total_reclaimable_bytes(dead_weight: &[DeadWeightProcess]) -> u64 {
        dead_weight.iter().map(|p| p.wasted_rss_bytes).sum()
    }
}

impl Default for ZombieHunter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_snap() -> HuntSnapshot {
        HuntSnapshot {
            pid: 1234,
            ppid: 1,
            name: "test_proc".to_string(),
            is_kernel_zombie: false,
            parent_alive: true,
            has_gui_window: false,
            rss_bytes: 10 * 1024 * 1024,
            cpu_percent: 0.5,
            wakeups_per_sec: 1.0,
            secs_since_user_interaction: 100,
            host_app_pid: None,
            host_app_running: true,
            host_app_absent_secs: 0,
        }
    }

    // ── Rule 1: True zombie ───────────────────────────────────────────────────

    #[test]
    fn true_zombie_detected_immediately() {
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            is_kernel_zombie: true,
            ..base_snap()
        };
        let result = hunter.evaluate(&snap);
        assert!(result.is_some());
        let dw = result.unwrap();
        assert_eq!(dw.zombie_class, ZombieClass::TrueZombie);
        assert_eq!(dw.recommended_action, ZombieAction::Kill);
    }

    #[test]
    fn true_zombie_fires_on_first_cycle_no_confirmation_needed() {
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            is_kernel_zombie: true,
            ..base_snap()
        };
        // Should fire immediately — no need for 3 confirmation cycles.
        assert!(hunter.evaluate(&snap).is_some());
    }

    // ── Rule 2: Orphan ────────────────────────────────────────────────────────

    #[test]
    fn orphan_with_dead_parent_non_launchd_is_killed() {
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            parent_alive: false,
            ppid: 500,
            ..base_snap()
        };
        let result = hunter.evaluate(&snap);
        assert!(result.is_some());
        let dw = result.unwrap();
        assert_eq!(dw.zombie_class, ZombieClass::Orphan);
        assert_eq!(dw.recommended_action, ZombieAction::Kill);
    }

    #[test]
    fn launchd_reparented_process_is_not_orphan() {
        // ppid == 1 means launchd adopted it — legitimate, must not be killed.
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            parent_alive: false,
            ppid: 1,
            ..base_snap()
        };
        // Even after 3 cycles, no hard rules fire.
        for _ in 0..5 {
            let result = hunter.evaluate(&snap);
            // Soft rules also don't fire (low wakeups, small RSS, not a ghost helper).
            assert!(
                result.is_none(),
                "launchd-adopted process should not be flagged"
            );
        }
    }

    // ── Rule 3: Ghost helper ──────────────────────────────────────────────────

    #[test]
    fn ghost_helper_confirmed_after_3_cycles() {
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            host_app_pid: Some(9000),
            host_app_running: false,
            host_app_absent_secs: 90_000, // > 24 h
            ..base_snap()
        };
        // Cycles 1 and 2: suspicious but not confirmed.
        assert!(hunter.evaluate(&snap).is_none());
        assert!(hunter.evaluate(&snap).is_none());
        // Cycle 3: confirmed.
        let result = hunter.evaluate(&snap);
        assert!(result.is_some());
        let dw = result.unwrap();
        assert_eq!(dw.zombie_class, ZombieClass::GhostHelper);
        assert_eq!(dw.recommended_action, ZombieAction::Suspend);
    }

    #[test]
    fn ghost_helper_below_threshold_not_flagged() {
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            host_app_pid: Some(9001),
            host_app_running: false,
            host_app_absent_secs: 3600, // only 1 h — below 24 h threshold
            ..base_snap()
        };
        for _ in 0..5 {
            assert!(hunter.evaluate(&snap).is_none());
        }
    }

    #[test]
    fn running_host_app_does_not_trigger_ghost_helper() {
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            host_app_pid: Some(9002),
            host_app_running: true,
            host_app_absent_secs: 100_000,
            ..base_snap()
        };
        for _ in 0..5 {
            assert!(hunter.evaluate(&snap).is_none());
        }
    }

    // ── Rule 4: Wakeup burner ─────────────────────────────────────────────────

    #[test]
    fn wakeup_burner_confirmed_after_3_cycles() {
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            wakeups_per_sec: 50.0, // > 20 threshold
            has_gui_window: false,
            secs_since_user_interaction: 600, // > 300
            ..base_snap()
        };
        assert!(hunter.evaluate(&snap).is_none()); // cycle 1
        assert!(hunter.evaluate(&snap).is_none()); // cycle 2
        let result = hunter.evaluate(&snap); // cycle 3
        assert!(result.is_some());
        let dw = result.unwrap();
        assert_eq!(dw.zombie_class, ZombieClass::WakeupBurner);
        assert_eq!(dw.recommended_action, ZombieAction::NiceToMax);
    }

    #[test]
    fn wakeup_burner_with_gui_window_not_flagged() {
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            wakeups_per_sec: 100.0,
            has_gui_window: true, // has a window — legitimate
            secs_since_user_interaction: 600,
            ..base_snap()
        };
        for _ in 0..5 {
            assert!(hunter.evaluate(&snap).is_none());
        }
    }

    #[test]
    fn wakeup_burner_recent_interaction_not_flagged() {
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            wakeups_per_sec: 100.0,
            has_gui_window: false,
            secs_since_user_interaction: 100, // < 300 threshold — user is active
            ..base_snap()
        };
        for _ in 0..5 {
            assert!(hunter.evaluate(&snap).is_none());
        }
    }

    // ── Rule 5: Memory hoarder ────────────────────────────────────────────────

    #[test]
    fn memory_hoarder_confirmed_after_3_cycles() {
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            rss_bytes: 512 * 1024 * 1024, // 512 MB > 256 MB threshold
            has_gui_window: false,
            secs_since_user_interaction: 3600, // > 1800 threshold
            ..base_snap()
        };
        assert!(hunter.evaluate(&snap).is_none());
        assert!(hunter.evaluate(&snap).is_none());
        let result = hunter.evaluate(&snap);
        assert!(result.is_some());
        let dw = result.unwrap();
        assert_eq!(dw.zombie_class, ZombieClass::MemoryHoarder);
        assert_eq!(dw.recommended_action, ZombieAction::Suspend);
    }

    #[test]
    fn memory_hoarder_with_gui_not_flagged() {
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            rss_bytes: 1024 * 1024 * 1024,
            has_gui_window: true,
            secs_since_user_interaction: 7200,
            ..base_snap()
        };
        for _ in 0..5 {
            assert!(hunter.evaluate(&snap).is_none());
        }
    }

    // ── Confirmation counter reset ────────────────────────────────────────────

    #[test]
    fn counter_resets_when_process_becomes_legitimate() {
        let mut hunter = ZombieHunter::new();
        let suspicious = HuntSnapshot {
            wakeups_per_sec: 50.0,
            has_gui_window: false,
            secs_since_user_interaction: 600,
            ..base_snap()
        };
        // 2 suspicious cycles.
        hunter.evaluate(&suspicious);
        hunter.evaluate(&suspicious);
        // Now process becomes legitimate.
        let legitimate = HuntSnapshot {
            wakeups_per_sec: 1.0,
            ..base_snap()
        };
        assert!(hunter.evaluate(&legitimate).is_none());
        // Counter should be reset — suspicious again needs 3 more cycles to confirm.
        assert!(hunter.evaluate(&suspicious).is_none()); // cycle 1 after reset
        assert!(hunter.evaluate(&suspicious).is_none()); // cycle 2
        let result = hunter.evaluate(&suspicious); // cycle 3 — now confirmed
        assert!(result.is_some());
    }

    // ── evaluate_all ─────────────────────────────────────────────────────────

    #[test]
    fn evaluate_all_returns_sorted_by_rss_descending() {
        let mut hunter = ZombieHunter::new();
        let snaps = vec![
            HuntSnapshot {
                pid: 1,
                is_kernel_zombie: true,
                rss_bytes: 100 * 1024 * 1024,
                ..base_snap()
            },
            HuntSnapshot {
                pid: 2,
                is_kernel_zombie: true,
                rss_bytes: 500 * 1024 * 1024,
                name: "bigger".to_string(),
                ..base_snap()
            },
        ];
        let dead_weight = hunter.evaluate_all(&snaps);
        assert_eq!(dead_weight.len(), 2);
        assert!(
            dead_weight[0].wasted_rss_bytes >= dead_weight[1].wasted_rss_bytes,
            "should be sorted by RSS descending"
        );
    }

    #[test]
    fn evaluate_all_empty_input_returns_empty() {
        let mut hunter = ZombieHunter::new();
        let result = hunter.evaluate_all(&[]);
        assert!(result.is_empty());
    }

    // ── total_reclaimable_bytes ───────────────────────────────────────────────

    #[test]
    fn total_reclaimable_bytes_sums_correctly() {
        let dead = vec![
            DeadWeightProcess {
                pid: 1,
                name: "a".into(),
                zombie_class: ZombieClass::TrueZombie,
                wasted_rss_bytes: 100 * 1024 * 1024,
                wakeups_per_sec: 0.0,
                recommended_action: ZombieAction::Kill,
                reason: "".into(),
            },
            DeadWeightProcess {
                pid: 2,
                name: "b".into(),
                zombie_class: ZombieClass::MemoryHoarder,
                wasted_rss_bytes: 200 * 1024 * 1024,
                wakeups_per_sec: 0.0,
                recommended_action: ZombieAction::Suspend,
                reason: "".into(),
            },
        ];
        let total = ZombieHunter::total_reclaimable_bytes(&dead);
        assert_eq!(total, 300 * 1024 * 1024);
    }

    #[test]
    fn total_reclaimable_bytes_empty_is_zero() {
        assert_eq!(ZombieHunter::total_reclaimable_bytes(&[]), 0);
    }

    // ── cleanup ───────────────────────────────────────────────────────────────

    #[test]
    fn cleanup_removes_stale_pids_from_history() {
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            pid: 42,
            wakeups_per_sec: 50.0,
            has_gui_window: false,
            secs_since_user_interaction: 600,
            ..base_snap()
        };
        // Build up history entry.
        hunter.evaluate(&snap);
        hunter.evaluate(&snap);
        assert!(hunter.suspicious_history.contains_key(&42));
        // Cleanup without 42 in live_pids.
        hunter.cleanup(&[1, 2, 3]);
        assert!(
            !hunter.suspicious_history.contains_key(&42),
            "stale PID should be purged"
        );
    }

    #[test]
    fn cleanup_retains_live_pids() {
        let mut hunter = ZombieHunter::new();
        let snap = HuntSnapshot {
            pid: 99,
            wakeups_per_sec: 50.0,
            has_gui_window: false,
            secs_since_user_interaction: 600,
            ..base_snap()
        };
        hunter.evaluate(&snap);
        hunter.cleanup(&[99, 100]);
        assert!(
            hunter.suspicious_history.contains_key(&99),
            "live PID should be retained"
        );
    }

    // ── Default impl ─────────────────────────────────────────────────────────

    #[test]
    fn zombie_hunter_default_matches_new() {
        let h = ZombieHunter::default();
        assert_eq!(h.config.ghost_helper_threshold_secs, 86_400);
        assert!((h.config.wakeup_burner_threshold - 20.0).abs() < 0.1);
        assert_eq!(h.config.memory_hoarder_threshold_bytes, 256 * 1024 * 1024);
    }

    // ── Micro-benchmark: evaluate latency ────────────────────────────────────

    #[test]
    fn bench_evaluate_latency() {
        let mut hunter = ZombieHunter::new();
        let snaps: Vec<HuntSnapshot> = (0..100)
            .map(|i| HuntSnapshot {
                pid: i,
                ..base_snap()
            })
            .collect();
        // Warm-up.
        for s in &snaps {
            let _ = hunter.evaluate(s);
        }
        let start = std::time::Instant::now();
        let n = 1000usize;
        for _ in 0..n {
            for s in &snaps {
                let _ = hunter.evaluate(s);
            }
        }
        let per_call_us = start.elapsed().as_secs_f64() * 1_000_000.0 / (n * snaps.len()) as f64;
        // evaluate() is pure HashMap + comparisons — should be < 10µs each.
        assert!(
            per_call_us < 10.0,
            "evaluate() too slow: {per_call_us:.2}µs/call (expected < 10µs)"
        );
    }
}
