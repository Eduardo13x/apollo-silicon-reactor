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
            ghost_helper_threshold_secs: 86_400,             // 24 h
            wakeup_burner_threshold: 20.0,                   // 20 wakeups/sec
            memory_hoarder_threshold_bytes: 256 * 1024 * 1024, // 256 MB (was 512 MB)
            hoarder_idle_threshold_secs: 1800,               // 30 min (was 1 h)
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
        let mut dead_weight: Vec<DeadWeightProcess> = snaps
            .iter()
            .filter_map(|s| self.evaluate(s))
            .collect();

        dead_weight.sort_by(|a, b| {
            b.wasted_rss_bytes
                .cmp(&a.wasted_rss_bytes)
        });

        dead_weight
    }

    /// Remove stale entries for PIDs that are gone.
    pub fn cleanup(&mut self, live_pids: &[u32]) {
        let live_set: std::collections::HashSet<u32> = live_pids.iter().copied().collect();
        self.suspicious_history.retain(|pid, _| live_set.contains(pid));
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
