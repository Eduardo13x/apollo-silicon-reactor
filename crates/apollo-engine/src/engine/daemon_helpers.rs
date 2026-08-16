//! Daemon Helpers — pure functions extracted from apollo-optimizerd.rs.
//!
//! These helpers have no dependency on SharedState and can be tested independently.
//! Includes: path resolution, persistence I/O, freeze logic, policy seeding.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Child;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TryRecvError, TrySendError};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::engine::decision_ledger::{
    ActuatorDecisionEvent, ActuatorDecisionOutcome, CycleDecisionEvents,
};
use crate::engine::policy_store::{append_jsonl, write_json, LearnedPolicy};
use crate::engine::power_management::PowerManager;
use crate::engine::process_identity::ProcessIdentity;
use crate::engine::profile_governor::{GovernorPersisted, ProfileGovernor};
use crate::engine::types::{
    FreezeSource, FrozenEntry, FrozenPidEntry, FrozenStatePersisted, HardPath, OptimizationProfile,
    ProfileTransition, RuntimeMetrics,
};

// ── Constants ───────────────────────────────────────────────────────────────

pub const FREEZE_TTL_SECS: i64 = 3 * 60;
pub const ASYNC_COMMAND_QUEUE_CAPACITY: usize = 32;
const ASYNC_COMMAND_PENDING_CAPACITY: usize = ASYNC_COMMAND_QUEUE_CAPACITY * 2 + 1;

/// Seed policy embedded at compile time — guarantees Brave, Claude, Warp, etc.
/// are always in interactive_patterns even on fresh installs or corrupt disk policy.
static SEED_POLICY: &str = include_str!("../../policy_clean.json");

// ── Path Functions ──────────────────────────────────────────────────────────
// Root paths: /var/lib/apollo/ or /var/run/
// Non-root paths: /tmp/

fn is_root() -> bool {
    let euid = unsafe { libc::geteuid() };
    euid == 0
}

pub fn socket_path() -> &'static str {
    static CACHED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CACHED.get_or_init(|| {
        std::env::var("APOLLO_SOCKET_PATH").unwrap_or_else(|_| {
            if is_root() {
                "/var/run/apollo-optimizer.sock".to_string()
            } else {
                "/tmp/apollo-optimizer.sock".to_string()
            }
        })
    })
}

pub fn kill_switch_path() -> &'static str {
    static CACHED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CACHED.get_or_init(|| {
        std::env::var("APOLLO_KILL_SWITCH_PATH").unwrap_or_else(|_| {
            if is_root() {
                "/var/run/apollo.disable".to_string()
            } else {
                "/tmp/apollo.disable".to_string()
            }
        })
    })
}

/// Explicit opt-in for the paired Microexperiment Lab. Absent by default, so
/// the lab stays in its diagnostic Shadow phase unless the operator creates the
/// file. It is the mirror image of the kill switch: presence enables, absence
/// abstains.
pub fn experiments_opt_in_path() -> &'static str {
    static CACHED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CACHED.get_or_init(|| {
        std::env::var("APOLLO_EXPERIMENTS_OPT_IN_PATH").unwrap_or_else(|_| {
            if is_root() {
                "/var/lib/apollo/experiments.enable".to_string()
            } else {
                "/tmp/apollo-experiments.enable".to_string()
            }
        })
    })
}

pub fn journal_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/journal.jsonl"
    } else {
        "/tmp/apollo-journal.jsonl"
    }
}

pub fn policy_audit_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/policy_audit.jsonl"
    } else {
        "/tmp/apollo-policy_audit.jsonl"
    }
}

pub fn audit_log_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/deep_scan_audit.jsonl"
    } else {
        "/tmp/apollo-deep_scan_audit.jsonl"
    }
}

pub fn metrics_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/runtime_metrics.json"
    } else {
        "/tmp/apollo-runtime_metrics.json"
    }
}

pub fn reflex_state_path() -> &'static str {
    static CACHED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CACHED.get_or_init(|| {
        std::env::var("APOLLO_REFLEX_STATE_PATH").unwrap_or_else(|_| {
            if is_root() {
                "/var/lib/apollo/reflex_state.json".to_string()
            } else {
                "/tmp/apollo-reflex_state.json".to_string()
            }
        })
    })
}

/// Per-cycle telemetry archive (Phase 1.5a — MLP router unblock).
/// Distinct file from `runtime_metrics.json` (which is the current snapshot).
/// Mirrors the root-vs-non-root split of every other state file.
pub fn metrics_history_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/runtime_metrics_history.jsonl"
    } else {
        "/tmp/apollo-runtime_metrics_history.jsonl"
    }
}

pub fn governor_state_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/governor_state.json"
    } else {
        "/tmp/apollo-governor_state.json"
    }
}

pub fn overflow_history_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/overflow_history.json"
    } else {
        "/tmp/apollo-overflow_history.json"
    }
}

pub fn rl_threshold_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/rl_threshold.json"
    } else {
        "/tmp/apollo-rl_threshold.json"
    }
}

pub fn predictive_agent_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/predictive_agent.json"
    } else {
        "/tmp/apollo-predictive_agent.json"
    }
}

pub fn markov_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/markov_transitions.json"
    } else {
        "/tmp/apollo-markov_transitions.json"
    }
}

pub fn signal_intelligence_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/signal_intelligence.json"
    } else {
        "/tmp/apollo-signal_intelligence.json"
    }
}

pub fn holt_winters_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/holt_winters.json"
    } else {
        "/tmp/apollo-holt_winters.json"
    }
}

pub fn timeline_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/profile_timeline.jsonl"
    } else {
        "/tmp/apollo-profile_timeline.jsonl"
    }
}

pub fn wake_state_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/wake_state.json"
    } else {
        "/tmp/apollo-wake_state.json"
    }
}

pub fn frozen_state_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/frozen_state.json"
    } else {
        "/tmp/apollo-frozen_state.json"
    }
}

pub fn hop_groups_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/hrpo_groups.json"
    } else {
        "/tmp/apollo-hrpo_groups.json"
    }
}

pub fn learned_state_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/learned_state.json"
    } else {
        "/tmp/apollo-learned_state.json"
    }
}

pub fn installation_id_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/installation_id"
    } else {
        "/tmp/apollo-installation-id"
    }
}

pub fn skills_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/optimization_skills.json"
    } else {
        "/tmp/apollo-optimization_skills.json"
    }
}

pub fn temporal_histograms_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/temporal_histograms.json"
    } else {
        "/tmp/apollo-temporal_histograms.json"
    }
}

pub fn telemetry_output_dir() -> &'static str {
    if is_root() {
        "/var/lib/apollo/telemetry"
    } else {
        "/tmp/apollo-telemetry"
    }
}

pub fn recently_applied_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/recently_applied.jsonl"
    } else {
        "/tmp/apollo-recently_applied.jsonl"
    }
}

/// Seconds since the macOS kernel booted.
///
/// Reads `kern.boottime` via `sysctlbyname` and subtracts from wall clock.
/// Used by apollo's cold-boot dampener: during the first few minutes after
/// boot, load averages and memory pressure are transiently elevated by
/// Spotlight, cfprefsd, triald, etc., and apollo's stability signals would
/// otherwise compound this noise into false instability penalties.
///
/// Returns `0` if the sysctl fails (conservative — callers then treat the
/// system as "not in cold-boot", i.e. no attenuation).
///
/// References:
/// - [Jain 1991] "The Art of Computer Systems Performance Analysis" §12.2
///   — warm-up period must be excluded from steady-state measurements.
/// - [Denning 1968] "The Working Set Model for Program Behavior" — no
///   stable working set exists during startup; the same applies to system boot.
pub fn system_uptime_secs() -> u64 {
    use std::mem;
    let mut tv: libc::timeval = unsafe { mem::zeroed() };
    let mut size = mem::size_of::<libc::timeval>();
    let name = b"kern.boottime\0";
    // SAFETY: name is NUL-terminated; tv is a valid timeval; size matches.
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const libc::c_char,
            &mut tv as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 || tv.tv_sec == 0 {
        return 0;
    }
    let boot = tv.tv_sec as u64;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    now.saturating_sub(boot)
}

/// Stable boot identity plus monotonic seconds since boot. Unlike daemon-local
/// `Instant`, this survives service restart; unlike wall time, it cannot jump
/// forward when the clock is adjusted.
#[cfg(target_os = "macos")]
pub fn system_boot_session() -> Option<(u64, u64)> {
    #[repr(C)]
    struct MachTimebaseInfo {
        numer: u32,
        denom: u32,
    }

    extern "C" {
        fn mach_continuous_time() -> u64;
        fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
    }

    let mut tv: libc::timeval = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of::<libc::timeval>();
    let boot_status = unsafe {
        libc::sysctlbyname(
            b"kern.boottime\0".as_ptr() as *const libc::c_char,
            &mut tv as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if boot_status != 0 || tv.tv_sec <= 0 {
        return None;
    }

    let mut timebase = MachTimebaseInfo { numer: 0, denom: 0 };
    if unsafe { mach_timebase_info(&mut timebase) } != 0
        || timebase.numer == 0
        || timebase.denom == 0
    {
        return None;
    }
    let ticks = unsafe { mach_continuous_time() };
    let monotonic_secs = (u128::from(ticks) * u128::from(timebase.numer)
        / u128::from(timebase.denom)
        / 1_000_000_000) as u64;
    let boot_id = (tv.tv_sec as u64).rotate_left(21) ^ tv.tv_usec as u64;
    (boot_id != 0).then_some((boot_id, monotonic_secs))
}

#[cfg(not(target_os = "macos"))]
pub fn system_boot_session() -> Option<(u64, u64)> {
    use std::sync::OnceLock;
    use std::time::Instant;

    static START: OnceLock<Instant> = OnceLock::new();
    Some((1, START.get_or_init(Instant::now).elapsed().as_secs()))
}

/// Path where novel effective process patterns are logged for scenario generation.
/// Append-only JSONL; read by autoresearch loop to generate targeted tests.
pub fn novel_patterns_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/novel_patterns.jsonl"
    } else {
        "/tmp/apollo-novel-patterns.jsonl"
    }
}

fn crash_sentinel_path() -> &'static str {
    if is_root() {
        "/var/lib/apollo/session.running"
    } else {
        "/tmp/apollo-session.running"
    }
}

/// Call at daemon startup to detect if the previous session ended abnormally.
///
/// Returns `true` only if the previous session both (a) left a sentinel
/// behind (no clean-shutdown write) AND (b) had been running long enough
/// (≥60 seconds) for the crash to plausibly reflect a real runtime issue
/// rather than a startup-time failure or operator kill.
///
/// Side effect: writes a new sentinel for the current session so the next
/// startup can detect *this* crash too.
///
/// [Gray & Reuter 1992 "Transaction Processing" §3 — crash recovery via
/// write-ahead sentinel; presence = in-progress, absence = clean.]
/// The 60-second minimum-uptime gate avoids treating crash-loops or operator
/// kill cycles as genuine instability — those should be diagnosed, not masked
/// by cautious mode.
pub fn detect_prior_crash() -> bool {
    let path = crash_sentinel_path();
    let crashed = if let Ok(prev) = fs::read_to_string(path) {
        // Parse `started` timestamp from previous sentinel and require ≥60s
        // uptime before treating absence-of-clean-shutdown as a real crash.
        let prev_started = prev
            .split("\"started\":\"")
            .nth(1)
            .and_then(|s| s.split('"').next())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok());
        match prev_started {
            Some(started) => {
                let lived =
                    chrono::Utc::now().signed_duration_since(started.with_timezone(&chrono::Utc));
                lived.num_seconds() >= 60
            }
            None => true, // unparseable old format → be conservative, treat as crash
        }
    } else {
        false
    };
    // Overwrite (or create) sentinel with current PID + timestamp.
    let _ = fs::write(
        path,
        format!(
            "{{\"pid\":{},\"started\":\"{}\"}}",
            std::process::id(),
            chrono::Utc::now().to_rfc3339()
        ),
    );
    crashed
}

/// Call at the END of a clean shutdown to remove the sentinel.
/// If the daemon is killed (SIGKILL, OOM, kernel panic) this never runs —
/// the sentinel persists, and the next `detect_prior_crash()` returns true.
pub fn remove_crash_sentinel() {
    let _ = fs::remove_file(crash_sentinel_path());
}

// ── Audit Log ───────────────────────────────────────────────────────────────

/// Append a JSON line to the audit log (best-effort, never fails the caller).
pub fn audit_log(entry: &serde_json::Value) {
    audit_log_batch(std::slice::from_ref(entry));
}

/// Append several audit records with one metadata check and one file open.
/// High-cardinality evaluators use this to avoid turning diagnostics into a
/// per-process filesystem hot path.
pub fn audit_log_batch(entries: &[serde_json::Value]) {
    if entries.is_empty() {
        return;
    }

    use std::fs::OpenOptions;
    let path = audit_log_path();
    // Rotate at 2MB (tightened 2026-05-08 from 5MB after macOS flagged the
    // daemon for excessive sustained-write rate). Caps disk usage at ~4MB
    // (live + .1) and shortens rotation cadence so old policy decisions
    // roll off SSD pages sooner.
    if let Ok(meta) = fs::metadata(path) {
        if meta.len() > 2 * 1024 * 1024 {
            let rotated = format!("{}.1", path);
            let _ = fs::remove_file(&rotated);
            let _ = fs::rename(path, &rotated);
        }
    }
    if let Ok(f) = OpenOptions::new().create(true).append(true).open(path) {
        let mut writer = std::io::BufWriter::new(f);
        for entry in entries {
            let _ = writeln!(writer, "{}", entry);
        }
    }
}

// ── Persistence Helpers ─────────────────────────────────────────────────────

pub fn write_metrics(path: &Path, metrics: &RuntimeMetrics) {
    write_json(path, metrics, Some(0o600));
}

pub fn write_governor_state(path: &Path, governor: &ProfileGovernor) {
    write_json(path, &governor.to_persisted(), Some(0o600));
}

pub fn load_governor_state(path: &Path, fallback_profile: OptimizationProfile) -> ProfileGovernor {
    if let Ok(data) = HardPath::read_to_string_limited(path, 1024 * 1024) {
        if let Ok(state) = serde_json::from_str::<GovernorPersisted>(&data) {
            return ProfileGovernor::from_persisted(state);
        }
    }
    ProfileGovernor::new(fallback_profile)
}

pub fn append_timeline(path: &Path, transition: &ProfileTransition) {
    append_jsonl(path, transition);
    rotate_timeline(path);
}

pub fn rotate_timeline(path: &Path) {
    // Keep cap small (2MB) and only 1 rotation — was 10MB × 3 = 30MB per log.
    // discrepancy.jsonl alone had 3 × 10MB = 30MB of stale ML override logs.
    const MAX_BYTES: u64 = 2 * 1024 * 1024;
    if fs::symlink_metadata(path)
        .map(|m| !m.file_type().is_symlink() && m.len() > MAX_BYTES)
        .unwrap_or(false)
    {
        let p1 = format!("{}.1", path.display());
        // Remove old rotation and replace with current file.
        let _ = fs::remove_file(&p1);
        let _ = fs::rename(path, &p1);
    }
}

// ── Wake State ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WakeStatePersisted {
    pub last_wake_at: Option<DateTime<Utc>>,
    pub post_wake_grace_until: Option<DateTime<Utc>>,
    /// 2026-05-12: opposite-of-grace window. After sleep, file-backed page
    /// cache plus background daemons hold ~1-2 GB on M1 8GB that the user
    /// already paid for hours ago. This timestamp marks the deadline after
    /// which the post-wake aggressive purge mode stops bypassing the page
    /// reclaim pressure gate. Set 30s ahead at wake-detect time; expired
    /// timestamps are cleared per cycle just like grace.
    #[serde(default)]
    pub post_wake_reclaim_until: Option<DateTime<Utc>>,
    pub post_wake_policy: String,
}

#[derive(Debug, Clone)]
pub struct WakeRuntimeState {
    pub last_cycle_wallclock: DateTime<Utc>,
    pub last_wake_at: Option<DateTime<Utc>>,
    pub post_wake_grace_until: Option<DateTime<Utc>>,
    pub post_wake_reclaim_until: Option<DateTime<Utc>>,
    pub post_wake_policy: String,
}

pub fn write_wake_state(path: &Path, state: &WakeRuntimeState) {
    let persisted = WakeStatePersisted {
        last_wake_at: state.last_wake_at,
        post_wake_grace_until: state.post_wake_grace_until,
        post_wake_reclaim_until: state.post_wake_reclaim_until,
        post_wake_policy: state.post_wake_policy.clone(),
    };
    write_json(path, &persisted, Some(0o600));
}

pub fn load_wake_state(path: &Path) -> WakeRuntimeState {
    let now = Utc::now();
    if let Ok(data) = HardPath::read_to_string_limited(path, 1024 * 1024) {
        if let Ok(state) = serde_json::from_str::<WakeStatePersisted>(&data) {
            return WakeRuntimeState {
                last_cycle_wallclock: now,
                last_wake_at: state.last_wake_at,
                post_wake_grace_until: state.post_wake_grace_until,
                post_wake_reclaim_until: state.post_wake_reclaim_until,
                post_wake_policy: state.post_wake_policy,
            };
        }
    }
    WakeRuntimeState {
        last_cycle_wallclock: now,
        last_wake_at: None,
        post_wake_grace_until: None,
        post_wake_reclaim_until: None,
        post_wake_policy: "grace-60s".to_string(),
    }
}

// ── Frozen State ────────────────────────────────────────────────────────────

/// Single background writer thread for frozen_state.json.
///
/// C3 fix (round-3): callers always invoke `write_frozen_state` while holding
/// `state.frozen_state` — the previous implementation did the disk write
/// synchronously, blocking the entire main loop on a slow disk (observed
/// 200ms+ on a loaded SSD).  The snapshot (`FrozenStatePersisted`) is built
/// cheaply under the caller's lock, then handed to a dedicated writer thread
/// via mpsc.  Lock is released immediately after `send`.
fn frozen_state_writer(
) -> &'static std::sync::mpsc::Sender<(std::path::PathBuf, FrozenStatePersisted)> {
    use std::sync::OnceLock;
    static TX: OnceLock<std::sync::mpsc::Sender<(std::path::PathBuf, FrozenStatePersisted)>> =
        OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<(std::path::PathBuf, FrozenStatePersisted)>();
        std::thread::Builder::new()
            .name("apollo-frozen-writer".to_string())
            .spawn(move || {
                while let Ok((path, state)) = rx.recv() {
                    // Coalesce: if the queue already has newer entries for the
                    // same path, drop intermediate ones to avoid amplifying disk
                    // writes during rapid bursts.
                    let mut latest = state;
                    let mut target = path;
                    while let Ok((p, s)) = rx.try_recv() {
                        if p == target {
                            latest = s;
                        } else {
                            // Different path — flush current then switch.
                            write_json(&target, &latest, Some(0o600));
                            target = p;
                            latest = s;
                        }
                    }
                    write_json(&target, &latest, Some(0o600));
                }
            })
            .expect("failed to spawn apollo-frozen-writer");
        tx
    })
}

pub fn write_frozen_state(path: &Path, frozen_state: &HashMap<u32, FrozenEntry>) {
    // Build snapshot inline (cheap: names are small Option<String>).
    let persisted = FrozenStatePersisted {
        frozen: frozen_state
            .iter()
            .map(|(pid, entry)| FrozenPidEntry {
                pid: *pid,
                since: entry.frozen_at,
                name: entry.process_name.clone(),
            })
            .collect(),
    };
    // Hand off to writer thread; caller may still hold the frozen_state lock
    // but that's fine because we don't need it after the snapshot is built.
    let _ = frozen_state_writer().send((path.to_path_buf(), persisted));
}

pub fn load_frozen_state(path: &Path) -> HashMap<u32, FrozenEntry> {
    if let Ok(data) = HardPath::read_to_string_limited(path, 10 * 1024 * 1024) {
        if let Ok(state) = serde_json::from_str::<FrozenStatePersisted>(&data) {
            return state
                .frozen
                .into_iter()
                .map(|e| {
                    (
                        e.pid,
                        FrozenEntry {
                            frozen_at: e.since,
                            source: FreezeSource::MainLoop,
                            pressure_at_freeze: 1.0,
                            process_name: e.name,
                            // Legacy persisted entries: start_sec unknown and
                            // original jetsam priority unknown. Callers fall
                            // back to name-only check when start_sec == 0.
                            start_sec: 0,
                            original_jetsam_priority: None,
                        },
                    )
                })
                .collect();
        }
    }
    HashMap::new()
}

// ── Freeze Logic ────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct UnfreezeOutcome {
    pub applied_pids: Vec<u32>,
    pub stale_pids: Vec<u32>,
    pub failed_pids: Vec<u32>,
}

impl UnfreezeOutcome {
    pub fn applied_count(&self) -> u64 {
        self.applied_pids.len() as u64
    }

    pub fn forgettable_pids(&self) -> impl Iterator<Item = u32> + '_ {
        self.applied_pids
            .iter()
            .chain(self.stale_pids.iter())
            .copied()
    }
}

pub fn unfreeze_outcome_events(
    action_key: &str,
    source: &str,
    cycle: u64,
    outcome: &UnfreezeOutcome,
) -> CycleDecisionEvents {
    let mut events = CycleDecisionEvents::default();
    for pid in &outcome.applied_pids {
        events.push(ActuatorDecisionEvent::local(
            action_key,
            format!("pid:{pid}"),
            cycle,
            ActuatorDecisionOutcome::Reverted,
            source,
            "verified SIGCONT applied",
        ));
    }
    for pid in &outcome.stale_pids {
        events.push(ActuatorDecisionEvent::local(
            action_key,
            format!("pid:{pid}"),
            cycle,
            ActuatorDecisionOutcome::Blocked,
            source,
            "stale process identity",
        ));
    }
    for pid in &outcome.failed_pids {
        events.push(ActuatorDecisionEvent::local(
            action_key,
            format!("pid:{pid}"),
            cycle,
            ActuatorDecisionOutcome::Failed,
            source,
            "SIGCONT failed",
        ));
    }
    events
}

fn send_sigcont_if(pid: u32, outcome: &mut UnfreezeOutcome, authorized: &mut impl FnMut() -> bool) {
    if !authorized() {
        return;
    }
    let rc = unsafe { libc::kill(pid as i32, libc::SIGCONT) };
    if rc == 0 {
        outcome.applied_pids.push(pid);
        return;
    }

    let errno = std::io::Error::last_os_error().raw_os_error();
    if errno == Some(libc::ESRCH) {
        outcome.stale_pids.push(pid);
    } else {
        tracing::warn!(pid, ?errno, "SIGCONT failed; retaining frozen-state entry");
        outcome.failed_pids.push(pid);
    }
}

pub fn unfreeze_pids_outcome(pids: impl Iterator<Item = u32>) -> UnfreezeOutcome {
    unfreeze_pids_outcome_if(pids, || true)
}

/// Unfreeze with a revocable authority callback checked immediately before
/// every SIGCONT. Authority denial leaves the PID unresolved for its owner.
pub fn unfreeze_pids_outcome_if(
    pids: impl Iterator<Item = u32>,
    mut authorized: impl FnMut() -> bool,
) -> UnfreezeOutcome {
    let mut outcome = UnfreezeOutcome::default();
    for pid in pids {
        // A2 fix (round-3): skip zombies — SIGCONT to a zombie is a no-op
        // that still burns a syscall. The old frozen-state entry is stale.
        if crate::engine::proc_taskinfo::is_zombie_pid(pid) {
            outcome.stale_pids.push(pid);
            continue;
        }
        send_sigcont_if(pid, &mut outcome, &mut authorized);
    }
    outcome
}

pub fn unfreeze_pids(pids: impl Iterator<Item = u32>) -> u64 {
    unfreeze_pids_outcome(pids).applied_count()
}

/// Unfreeze variant that verifies kernel start-time before signalling.
///
/// A3 fix (round-3): when a `FrozenEntry::start_sec > 0` is known, this
/// helper checks that the current process at `pid` still has the same
/// start-time.  If the PID was recycled between freeze and unfreeze, we
/// skip SIGCONT — otherwise we'd be resuming an unrelated process that
/// was never stopped by us.
///
/// Entries without `start_sec` (legacy, or capture failed) fall through
/// to the plain name-based behaviour.
pub fn unfreeze_pids_verified_outcome(entries: &HashMap<u32, FrozenEntry>) -> UnfreezeOutcome {
    unfreeze_pids_verified_outcome_if(entries, || true)
}

/// Identity-verified unfreeze with a revocable callback checked at each kernel
/// mutation boundary. This retains the legacy helper's result contract.
pub fn unfreeze_pids_verified_outcome_if(
    entries: &HashMap<u32, FrozenEntry>,
    mut authorized: impl FnMut() -> bool,
) -> UnfreezeOutcome {
    let mut outcome = UnfreezeOutcome::default();
    for (&pid, entry) in entries.iter() {
        if crate::engine::proc_taskinfo::is_zombie_pid(pid) {
            outcome.stale_pids.push(pid);
            continue;
        }

        // Legacy entries may lack start_sec, but usually retain the process
        // name. Validate whichever identity fields are available so PID reuse
        // cannot resume an unrelated process.
        if entry.start_sec > 0 || entry.process_name.is_some() {
            let Some(current) = ProcessIdentity::from_pid(pid) else {
                outcome.stale_pids.push(pid);
                continue;
            };
            if !current.matches(entry.process_name.as_deref(), entry.start_sec, 0) {
                outcome.stale_pids.push(pid);
                continue;
            }
        }

        let applied_before = outcome.applied_pids.len();
        send_sigcont_if(pid, &mut outcome, &mut authorized);
        // A5/D1 restoration: if we captured a jetsam priority at freeze
        // time, restore it verbatim instead of letting the caller clobber
        // it with a blanket FOREGROUND.
        if outcome.applied_pids.len() > applied_before {
            if let Some(prio) = entry.original_jetsam_priority {
                if authorized() {
                    let _ = crate::engine::jetsam_control::set_priority(pid, prio);
                }
            }
        }
    }
    outcome
}

pub fn unfreeze_pids_verified(entries: &HashMap<u32, FrozenEntry>) -> u64 {
    unfreeze_pids_verified_outcome(entries).applied_count()
}

/// Returns true when a frozen process should be thawed.
pub fn should_unfreeze(elapsed_secs: i64, pressure_at_freeze: f64, current_pressure: f64) -> bool {
    let ttl_expired = elapsed_secs > FREEZE_TTL_SECS;
    let pressure_recovered = elapsed_secs >= 30
        && pressure_at_freeze > 0.0
        && (current_pressure <= pressure_at_freeze * 0.6
            || (pressure_at_freeze - current_pressure) >= 0.05);
    let stale_with_improvement = elapsed_secs >= 120 && current_pressure < pressure_at_freeze;
    ttl_expired || pressure_recovered || stale_with_improvement
}

/// Rotate frozen processes when >=2 frozen and oldest >=60s.
pub fn should_rotate_oldest(elapsed_secs: i64, total_frozen: usize) -> bool {
    total_frozen >= 2 && elapsed_secs >= 60
}

// ── Misc Helpers ────────────────────────────────────────────────────────────

pub fn battery_pressure_boost(power_mgr: &PowerManager) -> f64 {
    use crate::engine::power_management::BatteryMode;
    if !power_mgr.is_on_battery() {
        return 0.0;
    }
    // 2026-05-11: halved from 0.04/0.10/0.18 → 0.02/0.05/0.10 after NotebookLM
    // peer review flagged the Critical (+0.18) boost as too volatile on M1 8GB
    // — it pushed effective_pressure ≥ 0.75 (Step 2 SIGSTOP gate) at raw
    // memory_pressure as low as 0.57, increasing the risk of Brave IPC
    // timeouts (the regression that motivated commit 712b927).
    // The reduced curve keeps Step 1 (E-core demote + PurgePurgeable) as the
    // primary mobility actuator and reserves Step 2 SIGSTOP for genuine
    // physical-memory crises rather than software-induced ones.
    // [Hellerstein 2004] control targets reflect operating regime
    // [Camacho 2007] predictive control grounded in platform physical limits.
    match power_mgr.battery_mode_current() {
        BatteryMode::Normal => 0.02,
        BatteryMode::LowPower => 0.05,
        BatteryMode::Critical => 0.10,
    }
}

/// Merge seed policy patterns into `policy` as a floor.
pub fn merge_seed_into(policy: &mut LearnedPolicy) {
    let seed: LearnedPolicy =
        serde_json::from_str(SEED_POLICY).expect("BUG: embedded policy_clean.json is invalid");
    use std::sync::Arc;
    for pat in seed.protected_patterns.iter() {
        if !policy.protected_patterns.contains(pat) {
            Arc::make_mut(&mut policy.protected_patterns).push(pat.clone());
        }
    }
    for pat in seed.interactive_patterns.iter() {
        if !policy.interactive_patterns.contains(pat) && !policy.protected_patterns.contains(pat) {
            Arc::make_mut(&mut policy.interactive_patterns).push(pat.clone());
        }
    }
    for pat in seed.noise_patterns.iter() {
        if !policy.noise_patterns.contains(pat)
            && !policy.protected_patterns.contains(pat)
            && !policy.interactive_patterns.contains(pat)
        {
            Arc::make_mut(&mut policy.noise_patterns).push(pat.clone());
        }
    }
    // Snapshot Arcs (refcount bump) so retain closures can read while make_mut
    // borrows the target Arc mutably. Arc::make_mut clones the inner Vec only
    // if shared (refcount > 1), so these snapshot Arcs trigger clone-on-write.
    let protected_snap = policy.protected_patterns.clone();
    Arc::make_mut(&mut policy.interactive_patterns).retain(|p| !protected_snap.contains(p));
    let interactive_snap = policy.interactive_patterns.clone();
    Arc::make_mut(&mut policy.noise_patterns)
        .retain(|p| !protected_snap.contains(p) && !interactive_snap.contains(p));
}

pub fn pid_start_time(pid: u32) -> (u64, u64) {
    ProcessIdentity::from_pid(pid)
        .map(|id| (id.start_sec, id.start_usec))
        .unwrap_or((0, 0))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReversibleJetsamOutcome {
    Applied,
    Refreshed,
    Unchanged,
    Stale,
}

/// Temporarily move one process to the background Jetsam band while retaining
/// the exact prior band for TTL-based restoration. The live process identity
/// is checked before the read and again by the mediator immediately before the
/// write, so a recycled PID cannot inherit Apollo's mutation or ledger lease.
pub fn apply_reversible_background_jetsam(
    pid: u32,
    expected_name: Option<&str>,
    ttl: Duration,
    owner: &'static str,
) -> Result<ReversibleJetsamOutcome, String> {
    use crate::engine::effect_ledger::{
        record_global, refresh_global_if_justification, AppliedEffect,
    };
    use crate::engine::jetsam_control::{self, priority};
    use crate::engine::mediator::{mediate, Effect, JetsamEffector, JetsamTierKind, PreCondition};

    let Some(identity) = ProcessIdentity::from_pid(pid) else {
        return Ok(ReversibleJetsamOutcome::Stale);
    };
    if !identity.matches(expected_name, identity.start_sec, identity.start_usec) {
        return Ok(ReversibleJetsamOutcome::Stale);
    }

    let prior = jetsam_control::get_priority(pid)
        .ok_or_else(|| format!("jetsam priority unreadable for pid {pid}"))?;
    if !identity.is_still_valid() {
        return Ok(ReversibleJetsamOutcome::Stale);
    }

    let effect_key = AppliedEffect::JetsamPriority { pid, prior: 0 };
    if prior == priority::BACKGROUND {
        return Ok(
            if refresh_global_if_justification(&effect_key, ttl, identity.start_sec, owner) {
                ReversibleJetsamOutcome::Refreshed
            } else {
                ReversibleJetsamOutcome::Unchanged
            },
        );
    }

    let effect = Effect::SetJetsamTier {
        pid,
        start_sec: identity.start_sec,
        tier: JetsamTierKind::Background,
    };
    let precondition = PreCondition {
        pid_identity: Some((pid, identity.start_sec)),
        ..Default::default()
    };
    match mediate(&effect, &precondition, &JetsamEffector) {
        Ok(receipt) if receipt.applied_count > 0 => {
            record_global(
                AppliedEffect::JetsamPriority { pid, prior },
                ttl,
                identity.start_sec,
                owner,
            );
            Ok(ReversibleJetsamOutcome::Applied)
        }
        Ok(_) => Ok(ReversibleJetsamOutcome::Unchanged),
        Err(error) => Err(format!("{error:?}")),
    }
}

pub fn parse_profile(input: &str) -> OptimizationProfile {
    match input {
        "aggressive-root" => OptimizationProfile::AggressiveRoot,
        "safe-root" => OptimizationProfile::SafeRoot,
        _ => OptimizationProfile::BalancedRoot,
    }
}

pub fn compute_p95(samples: &[u64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    let idx = (((sorted.len() - 1) as f64) * 0.95).round() as usize;
    *sorted.select_nth_unstable(idx).1 as f64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncCommandMetric {
    MaintenancePurge,
    SurvivalPurge,
    CliPurge,
    Spotlight,
    Test,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsyncCommandSubmission {
    pub correlation_id: u64,
    pub action_key: String,
    pub target: String,
    pub source: String,
    pub metric: AsyncCommandMetric,
}

impl AsyncCommandSubmission {
    pub fn pending_event(&self, cycle: u64) -> ActuatorDecisionEvent {
        ActuatorDecisionEvent::local(
            &self.action_key,
            &self.target,
            cycle,
            ActuatorDecisionOutcome::Pending,
            &self.source,
            "command queued; completion pending",
        )
        .with_correlation(self.correlation_id)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AsyncCommandCompletionStats {
    pub completed: u64,
    pub failed: u64,
    pub expired: u64,
    pub maintenance_purges_applied: u64,
    pub survival_purges_applied: u64,
    pub cli_purges_applied: u64,
    pub spotlight_applied: u64,
}

impl AsyncCommandCompletionStats {
    fn merge(&mut self, other: Self) {
        self.completed = self.completed.saturating_add(other.completed);
        self.failed = self.failed.saturating_add(other.failed);
        self.expired = self.expired.saturating_add(other.expired);
        self.maintenance_purges_applied = self
            .maintenance_purges_applied
            .saturating_add(other.maintenance_purges_applied);
        self.survival_purges_applied = self
            .survival_purges_applied
            .saturating_add(other.survival_purges_applied);
        self.cli_purges_applied = self
            .cli_purges_applied
            .saturating_add(other.cli_purges_applied);
        self.spotlight_applied = self
            .spotlight_applied
            .saturating_add(other.spotlight_applied);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncCommandSubmitError {
    Full,
    WorkerUnavailable,
}

impl std::fmt::Display for AsyncCommandSubmitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Full => "asynchronous command queue full",
            Self::WorkerUnavailable => "asynchronous command worker unavailable",
        })
    }
}

enum AsyncCommandKind {
    Purge,
    Spotlight(bool),
    #[cfg(test)]
    TestProgram(String),
    #[cfg(test)]
    TestDelay(Duration, bool),
}

struct AsyncCommandJob {
    kind: AsyncCommandKind,
    submission: AsyncCommandSubmission,
}

struct AsyncCommandCompletion {
    submission: AsyncCommandSubmission,
    success: bool,
    detail: String,
}

/// Loop-owned, fixed-capacity command handoff. Submitting and draining use
/// non-blocking channel operations; only the dedicated worker waits for child
/// completion. The completion channel is also bounded, so memory cannot grow
/// with slow or failed external commands.
pub struct AsyncCommandQueue {
    jobs: SyncSender<AsyncCommandJob>,
    completions: Receiver<AsyncCommandCompletion>,
    next_correlation_id: AtomicU64,
    pending: RefCell<Vec<AsyncCommandSubmission>>,
    shutdown_finalized: Cell<bool>,
}

impl Default for AsyncCommandQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncCommandQueue {
    pub fn new() -> Self {
        let (job_tx, job_rx) = mpsc::sync_channel::<AsyncCommandJob>(ASYNC_COMMAND_QUEUE_CAPACITY);
        let (completion_tx, completion_rx) =
            mpsc::sync_channel::<AsyncCommandCompletion>(ASYNC_COMMAND_QUEUE_CAPACITY);
        let _ = std::thread::Builder::new()
            .name("apollo-command-completion".to_string())
            .spawn(move || async_command_worker(job_rx, completion_tx));
        Self {
            jobs: job_tx,
            completions: completion_rx,
            next_correlation_id: AtomicU64::new(1),
            pending: RefCell::new(Vec::with_capacity(ASYNC_COMMAND_PENDING_CAPACITY)),
            shutdown_finalized: Cell::new(false),
        }
    }

    pub fn submit_purge(
        &self,
        action_key: &str,
        target: &str,
        source: &str,
        metric: AsyncCommandMetric,
    ) -> Result<AsyncCommandSubmission, AsyncCommandSubmitError> {
        self.submit(AsyncCommandKind::Purge, action_key, target, source, metric)
    }

    pub fn submit_spotlight(
        &self,
        enabled: bool,
        source: &str,
    ) -> Result<AsyncCommandSubmission, AsyncCommandSubmitError> {
        self.submit(
            AsyncCommandKind::Spotlight(enabled),
            "spotlight:indexing",
            if enabled { "enabled" } else { "disabled" },
            source,
            AsyncCommandMetric::Spotlight,
        )
    }

    #[cfg(test)]
    fn submit_test_program(
        &self,
        program: &str,
        action_key: &str,
        target: &str,
        source: &str,
    ) -> Result<AsyncCommandSubmission, AsyncCommandSubmitError> {
        self.submit(
            AsyncCommandKind::TestProgram(program.to_string()),
            action_key,
            target,
            source,
            AsyncCommandMetric::Test,
        )
    }

    #[cfg(test)]
    fn submit_test_delay(
        &self,
        delay: Duration,
        success: bool,
        action_key: &str,
        target: &str,
        source: &str,
    ) -> Result<AsyncCommandSubmission, AsyncCommandSubmitError> {
        self.submit(
            AsyncCommandKind::TestDelay(delay, success),
            action_key,
            target,
            source,
            AsyncCommandMetric::Test,
        )
    }

    fn submit(
        &self,
        kind: AsyncCommandKind,
        action_key: &str,
        target: &str,
        source: &str,
        metric: AsyncCommandMetric,
    ) -> Result<AsyncCommandSubmission, AsyncCommandSubmitError> {
        if self.shutdown_finalized.get() {
            return Err(AsyncCommandSubmitError::WorkerUnavailable);
        }
        if self.pending.borrow().len() >= ASYNC_COMMAND_PENDING_CAPACITY {
            return Err(AsyncCommandSubmitError::Full);
        }
        let correlation_id = self.next_correlation_id.fetch_add(1, Ordering::Relaxed);
        let submission = AsyncCommandSubmission {
            correlation_id: correlation_id.max(1),
            action_key: bound_async_text(action_key, 320),
            target: bound_async_text(target, 256),
            source: bound_async_text(source, 48),
            metric,
        };
        match self.jobs.try_send(AsyncCommandJob {
            kind,
            submission: submission.clone(),
        }) {
            Ok(()) => {
                self.pending.borrow_mut().push(submission.clone());
                Ok(submission)
            }
            Err(TrySendError::Full(_)) => Err(AsyncCommandSubmitError::Full),
            Err(TrySendError::Disconnected(_)) => Err(AsyncCommandSubmitError::WorkerUnavailable),
        }
    }

    pub fn drain_decision_events(
        &self,
        cycle: u64,
        events: &mut CycleDecisionEvents,
    ) -> AsyncCommandCompletionStats {
        let mut stats = AsyncCommandCompletionStats::default();
        for _ in 0..ASYNC_COMMAND_QUEUE_CAPACITY {
            let completion = match self.completions.try_recv() {
                Ok(completion) => completion,
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            };
            let Some(pending_index) =
                self.pending.borrow().iter().position(|pending| {
                    pending.correlation_id == completion.submission.correlation_id
                })
            else {
                // Shutdown already terminally expired this identity.
                continue;
            };
            self.pending.borrow_mut().swap_remove(pending_index);
            stats.completed = stats.completed.saturating_add(1);
            stats.failed = stats.failed.saturating_add(u64::from(!completion.success));
            if completion.success {
                match completion.submission.metric {
                    AsyncCommandMetric::MaintenancePurge => {
                        stats.maintenance_purges_applied =
                            stats.maintenance_purges_applied.saturating_add(1);
                    }
                    AsyncCommandMetric::SurvivalPurge => {
                        stats.survival_purges_applied =
                            stats.survival_purges_applied.saturating_add(1);
                    }
                    AsyncCommandMetric::CliPurge => {
                        stats.cli_purges_applied = stats.cli_purges_applied.saturating_add(1);
                    }
                    AsyncCommandMetric::Spotlight => {
                        stats.spotlight_applied = stats.spotlight_applied.saturating_add(1);
                    }
                    AsyncCommandMetric::Test => {}
                }
            }
            events.push(
                ActuatorDecisionEvent::local(
                    completion.submission.action_key,
                    completion.submission.target,
                    cycle,
                    if completion.success {
                        ActuatorDecisionOutcome::Applied
                    } else {
                        ActuatorDecisionOutcome::Failed
                    },
                    completion.submission.source,
                    completion.detail,
                )
                .with_correlation(completion.submission.correlation_id),
            );
        }
        stats
    }

    /// Terminally account for every admitted command without waiting for an
    /// external process. Ready completions retain their exact exit status;
    /// every other correlation expires once and any later worker result is
    /// ignored because it no longer exists in the loop-owned pending set.
    pub fn finalize_shutdown(
        &self,
        cycle: u64,
        events: &mut CycleDecisionEvents,
    ) -> AsyncCommandCompletionStats {
        self.shutdown_finalized.set(true);
        let mut stats = AsyncCommandCompletionStats::default();
        // Three non-blocking drains cover the fixed 65-command admission
        // envelope even when the worker refills the 32-slot completion queue
        // while shutdown is consuming it.
        for _ in 0..3 {
            stats.merge(self.drain_decision_events(cycle, events));
        }
        let pending = self.pending.take();
        for submission in pending {
            stats.expired = stats.expired.saturating_add(1);
            events.push(
                ActuatorDecisionEvent::local(
                    submission.action_key,
                    submission.target,
                    cycle,
                    ActuatorDecisionOutcome::Expired,
                    submission.source,
                    "command unresolved at graceful shutdown; authority cancelled",
                )
                .with_correlation(submission.correlation_id),
            );
        }
        stats
    }
}

fn async_command_worker(
    jobs: Receiver<AsyncCommandJob>,
    completions: SyncSender<AsyncCommandCompletion>,
) {
    while let Ok(job) = jobs.recv() {
        let (success, detail) = match job.kind {
            AsyncCommandKind::Purge => async_command_result(
                std::process::Command::new("purge")
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status(),
            ),
            AsyncCommandKind::Spotlight(enabled) => async_command_result(
                std::process::Command::new("/usr/bin/mdutil")
                    .args(["-a", "-i", if enabled { "on" } else { "off" }])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status(),
            ),
            #[cfg(test)]
            AsyncCommandKind::TestProgram(program) => async_command_result(
                std::process::Command::new(program)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status(),
            ),
            #[cfg(test)]
            AsyncCommandKind::TestDelay(delay, success) => {
                std::thread::sleep(delay);
                (success, format!("test_delay_success={success}"))
            }
        };
        if completions
            .send(AsyncCommandCompletion {
                submission: job.submission,
                success,
                detail,
            })
            .is_err()
        {
            break;
        }
    }
}

fn async_command_result(result: std::io::Result<std::process::ExitStatus>) -> (bool, String) {
    match result {
        Ok(status) => (
            status.success(),
            format!(
                "exit_status={:?},success={}",
                status.code(),
                status.success()
            ),
        ),
        Err(error) => (false, format!("spawn_error={error}")),
    }
}

fn bound_async_text(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

type ReapJob = (Child, &'static str);

#[cfg(test)]
static REAPED_CHILD_PIDS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<u32>>> =
    std::sync::OnceLock::new();

fn child_reaper_sender() -> std::io::Result<&'static Sender<ReapJob>> {
    static REAPER: std::sync::OnceLock<Option<Sender<ReapJob>>> = std::sync::OnceLock::new();

    REAPER
        .get_or_init(|| {
            let (tx, rx) = mpsc::channel::<ReapJob>();
            std::thread::Builder::new()
                .name("apollo-child-reaper".to_string())
                .spawn(move || {
                    while let Ok((mut child, label)) = rx.recv() {
                        let pid = child.id();
                        match child.wait() {
                            Ok(status) if !status.success() => tracing::warn!(
                                child_pid = pid,
                                command = label,
                                ?status,
                                "asynchronous child exited unsuccessfully"
                            ),
                            Err(error) => tracing::warn!(
                                child_pid = pid,
                                command = label,
                                %error,
                                "failed to reap asynchronous child"
                            ),
                            _ => {}
                        }
                        #[cfg(test)]
                        REAPED_CHILD_PIDS
                            .get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .insert(pid);
                    }
                })
                .ok()
                .map(|_| tx)
        })
        .as_ref()
        .ok_or_else(|| std::io::Error::other("Apollo child reaper thread failed to start"))
}

/// Spawn a command without blocking the daemon and transfer its `Child` to a
/// dedicated waiter. Dropping `std::process::Child` does not reap it, so every
/// fire-and-forget command must use this helper (or wait explicitly).
pub fn spawn_reaped_command(
    program: &str,
    args: &[&str],
    label: &'static str,
) -> std::io::Result<u32> {
    // Start the waiter first: if thread creation fails, no unmanaged child is
    // launched. A disconnected sender is handled below by killing and waiting.
    let sender = child_reaper_sender()?;
    let child = std::process::Command::new(program).args(args).spawn()?;
    let pid = child.id();

    if let Err(error) = sender.send((child, label)) {
        let (mut orphan, _) = error.0;
        let _ = orphan.kill();
        let _ = orphan.wait();
        return Err(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "Apollo child reaper stopped",
        ));
    }

    Ok(pid)
}

pub fn spawn_reaped_purge() -> bool {
    spawn_reaped_command("purge", &[], "purge").is_ok()
}

/// mdutil communicates with the Spotlight server via XPC (com.apple.spotlightserver).
/// There is no public or private framework function equivalent — MDSetIndexingEnabled
/// does not exist in the dyld shared cache on Apple Silicon macOS 15.
pub fn spotlight_set_indexing(enabled: bool) {
    let flag = if enabled { "on" } else { "off" };
    let _ = std::process::Command::new("/usr/bin/mdutil")
        .args(["-a", "-i", flag])
        .status();
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::decision_ledger::{ActuatorDecisionOutcome, CycleDecisionEvents};

    #[test]
    fn compute_p95_matches_full_sort_reference() {
        let samples = [91, 4, 77, 12, 55, 55, 3, 99, 42, 88, 65, 21, 73];
        for len in 1..=samples.len() {
            let mut reference = samples[..len].to_vec();
            reference.sort_unstable();
            let idx = (((len - 1) as f64) * 0.95).round() as usize;
            assert_eq!(compute_p95(&samples[..len]), reference[idx] as f64);
        }
        assert_eq!(compute_p95(&[]), 0.0);
    }

    #[test]
    fn reaped_command_does_not_leave_a_waitable_child() {
        let pid = spawn_reaped_command("/usr/bin/true", &[], "test-true")
            .expect("spawn true through child reaper");
        let deadline = std::time::Instant::now() + Duration::from_secs(2);

        loop {
            let was_reaped = REAPED_CHILD_PIDS
                .get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains(&pid);
            if was_reaped {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "child reaper did not collect /usr/bin/true within 2s"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn bounded_async_command_reports_pending_then_exact_failed_completion() {
        let commands = AsyncCommandQueue::new();
        let submission = commands
            .submit_test_program(
                "/usr/bin/false",
                "predictive_purge:test",
                "host",
                "test-async-command",
            )
            .expect("bounded queue accepts one test command");
        let pending = submission.pending_event(5);
        assert_eq!(pending.outcome, ActuatorDecisionOutcome::Pending);
        assert_eq!(pending.correlation_id, Some(submission.correlation_id));

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let completion = loop {
            let mut events = CycleDecisionEvents::default();
            commands.drain_decision_events(6, &mut events);
            if let Some(event) = events.as_slice().first() {
                break event.clone();
            }
            assert!(std::time::Instant::now() < deadline, "completion timed out");
            std::thread::sleep(Duration::from_millis(5));
        };

        assert_eq!(completion.outcome, ActuatorDecisionOutcome::Failed);
        assert_eq!(completion.correlation_id, pending.correlation_id);
        assert_eq!(completion.proposal.action_key, pending.proposal.action_key);
        assert_eq!(completion.proposal.target, pending.proposal.target);
        assert!(completion.detail.contains("exit"));
    }

    #[test]
    fn shutdown_expires_inflight_command_and_ignores_late_completion() {
        let commands = AsyncCommandQueue::new();
        let submission = commands
            .submit_test_delay(
                Duration::from_millis(75),
                true,
                "predictive_purge:test",
                "host",
                "test-async-command",
            )
            .expect("bounded queue accepts delayed test command");

        let mut shutdown_events = CycleDecisionEvents::default();
        let stats = commands.finalize_shutdown(9, &mut shutdown_events);
        assert_eq!(stats.completed, 0);
        assert_eq!(stats.expired, 1);
        assert_eq!(shutdown_events.len(), 1);
        let expiry = &shutdown_events.as_slice()[0];
        assert_eq!(expiry.outcome, ActuatorDecisionOutcome::Expired);
        assert_eq!(expiry.correlation_id, Some(submission.correlation_id));
        assert_eq!(expiry.proposal.action_key, submission.action_key);
        assert_eq!(expiry.proposal.target, submission.target);

        std::thread::sleep(Duration::from_millis(125));
        let mut late_events = CycleDecisionEvents::default();
        let late_stats = commands.drain_decision_events(10, &mut late_events);
        assert_eq!(late_stats.completed, 0);
        assert!(late_events.is_empty());
    }

    #[test]
    fn shutdown_prefers_ready_exact_completion_over_expiry() {
        let commands = AsyncCommandQueue::new();
        let submission = commands
            .submit_test_program(
                "/usr/bin/false",
                "predictive_purge:test",
                "host",
                "test-async-command",
            )
            .expect("bounded queue accepts one test command");
        std::thread::sleep(Duration::from_millis(50));

        let mut shutdown_events = CycleDecisionEvents::default();
        let stats = commands.finalize_shutdown(10, &mut shutdown_events);
        assert_eq!(stats.completed, 1);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.expired, 0);
        assert_eq!(shutdown_events.len(), 1);
        let completion = &shutdown_events.as_slice()[0];
        assert_eq!(completion.outcome, ActuatorDecisionOutcome::Failed);
        assert_eq!(completion.correlation_id, Some(submission.correlation_id));
        assert!(completion.detail.contains("exit"));
    }

    #[test]
    fn unfreeze_outcome_emits_exact_reverted_blocked_and_failed_receipts() {
        let outcome = UnfreezeOutcome {
            applied_pids: vec![11],
            stale_pids: vec![12],
            failed_pids: vec![13],
        };

        let events =
            unfreeze_outcome_events("unfreeze:pre_sleep", "pre-sleep-recovery", 8, &outcome);

        assert_eq!(events.len(), 3);
        assert_eq!(events.as_slice()[0].proposal.target, "pid:11");
        assert_eq!(
            events.as_slice()[0].outcome,
            ActuatorDecisionOutcome::Reverted
        );
        assert_eq!(events.as_slice()[1].proposal.target, "pid:12");
        assert_eq!(
            events.as_slice()[1].outcome,
            ActuatorDecisionOutcome::Blocked
        );
        assert_eq!(events.as_slice()[2].proposal.target, "pid:13");
        assert_eq!(
            events.as_slice()[2].outcome,
            ActuatorDecisionOutcome::Failed
        );
    }

    #[test]
    fn test_should_unfreeze_ttl_path() {
        assert!(should_unfreeze(FREEZE_TTL_SECS + 1, 0.80, 0.80));
        assert!(should_unfreeze(FREEZE_TTL_SECS + 1, 0.80, 0.90));
    }

    #[test]
    fn test_should_unfreeze_pressure_recovery() {
        assert!(should_unfreeze(60, 0.80, 0.45));
        assert!(should_unfreeze(60, 0.80, 0.75));
        assert!(!should_unfreeze(60, 0.80, 0.77));
    }

    #[test]
    fn test_should_unfreeze_min_30s_guard() {
        assert!(!should_unfreeze(29, 0.80, 0.10));
        assert!(should_unfreeze(30, 0.80, 0.10));
    }

    #[test]
    fn test_should_unfreeze_high_pressure_at_freeze() {
        assert!(should_unfreeze(60, 1.0, 0.10));
        // 1.0 → 0.65 = delta 0.35, exceeds 0.05 threshold → should unfreeze
        assert!(should_unfreeze(60, 1.0, 0.65));
        // 1.0 → 0.96 = delta 0.04, below 0.05 AND 0.96 > 0.6 → should NOT unfreeze
        assert!(!should_unfreeze(60, 1.0, 0.96));
        assert!(should_unfreeze(FREEZE_TTL_SECS + 1, 1.0, 0.90));
    }

    #[test]
    fn test_should_unfreeze_zero_pressure_at_freeze() {
        assert!(!should_unfreeze(60, 0.0, 0.0));
        assert!(!should_unfreeze(60, 0.0, 0.10));
    }

    #[test]
    fn test_should_unfreeze_stale_at_2min() {
        assert!(should_unfreeze(120, 0.75, 0.74));
        assert!(!should_unfreeze(119, 0.75, 0.74));
        assert!(!should_unfreeze(120, 0.75, 0.75));
    }

    #[test]
    fn test_should_rotate_oldest() {
        assert!(should_rotate_oldest(60, 2));
        assert!(should_rotate_oldest(200, 5));
        assert!(!should_rotate_oldest(60, 1));
        assert!(!should_rotate_oldest(59, 2));
    }

    /// F3 — ABA race defense: unfreeze_pids_verified must return 0 (no SIGCONT
    /// issued) for a PID that is either dead or whose kernel start_sec does
    /// not match the stored FrozenEntry. Uses a very high PID unlikely to be
    /// live + a bogus start_sec so identity check always fails.
    /// [Gray & Reuter 1992 §11] crash recovery identity invariants.
    #[test]
    fn unfreeze_pids_verified_skips_dead_or_recycled_pid() {
        use crate::engine::types::{FreezeSource, FrozenEntry};
        let mut entries = HashMap::new();
        // PID 999_999 is virtually guaranteed not to exist; start_sec is a
        // bogus sentinel that won't match any live process's pbi_start_tvsec.
        entries.insert(
            999_999_u32,
            FrozenEntry {
                frozen_at: chrono::Utc::now(),
                source: FreezeSource::MainLoop,
                pressure_at_freeze: 0.8,
                process_name: Some("ghost-process".to_string()),
                start_sec: 1_u64,
                original_jetsam_priority: None,
            },
        );
        let outcome = unfreeze_pids_verified_outcome(&entries);
        assert_eq!(
            outcome.applied_count(),
            0,
            "unfreeze_pids_verified must skip dead/recycled PIDs (no SIGCONT sent)"
        );
        assert_eq!(outcome.stale_pids, vec![999_999]);
        assert!(outcome.failed_pids.is_empty());
    }

    #[test]
    fn legacy_unfreeze_entry_rejects_recycled_pid_by_name() {
        use crate::engine::types::{FreezeSource, FrozenEntry};
        let pid = std::process::id();
        let mut entries = HashMap::new();
        entries.insert(
            pid,
            FrozenEntry {
                frozen_at: chrono::Utc::now(),
                source: FreezeSource::MainLoop,
                pressure_at_freeze: 0.8,
                process_name: Some("definitely-not-this-test-process".to_string()),
                start_sec: 0,
                original_jetsam_priority: None,
            },
        );

        let outcome = unfreeze_pids_verified_outcome(&entries);
        assert_eq!(outcome.applied_count(), 0);
        assert_eq!(outcome.stale_pids, vec![pid]);
    }

    #[test]
    fn unfreeze_outcome_counts_only_successful_signal() {
        let pid = std::process::id();
        let outcome = unfreeze_pids_outcome(std::iter::once(pid));
        assert_eq!(outcome.applied_pids, vec![pid]);
        assert!(outcome.stale_pids.is_empty());
        assert!(outcome.failed_pids.is_empty());
    }

    #[test]
    fn reversible_jetsam_rejects_wrong_process_name_before_kernel_write() {
        let outcome = apply_reversible_background_jetsam(
            std::process::id(),
            Some("definitely-not-this-test-process"),
            Duration::from_secs(1),
            "test: wrong process identity",
        )
        .expect("identity rejection is a normal stale outcome");
        assert_eq!(outcome, ReversibleJetsamOutcome::Stale);
    }

    /// Serialize sentinel tests — they share a global file path.
    fn sentinel_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn clean_shutdown_no_crash_detected() {
        let _guard = sentinel_test_lock();
        let path = crash_sentinel_path();
        let _ = fs::remove_file(path); // ensure clean state
        let crashed = detect_prior_crash();
        assert!(!crashed, "fresh start should not appear as crash");
        remove_crash_sentinel();
        assert!(
            !std::path::Path::new(path).exists(),
            "sentinel should be removed after clean shutdown"
        );
    }

    #[test]
    fn crash_leaves_sentinel_detected_on_next_start() {
        let _guard = sentinel_test_lock();
        let path = crash_sentinel_path();
        let _ = fs::remove_file(path); // clean state
                                       // Inject an aged sentinel: previous session "started" 120s ago.
        let aged = chrono::Utc::now() - chrono::Duration::seconds(120);
        let _ = fs::write(
            path,
            format!("{{\"pid\":1,\"started\":\"{}\"}}", aged.to_rfc3339()),
        );
        let crashed = detect_prior_crash(); // sees aged sentinel → real crash
        assert!(
            crashed,
            "aged sentinel (≥60s uptime) should be detected as crash"
        );
        remove_crash_sentinel();
    }

    #[test]
    fn fresh_sentinel_below_uptime_floor_not_a_crash() {
        let _guard = sentinel_test_lock();
        let path = crash_sentinel_path();
        let _ = fs::remove_file(path);
        // Inject a very fresh sentinel (just now) — uptime < 60s.
        let now = chrono::Utc::now();
        let _ = fs::write(
            path,
            format!("{{\"pid\":1,\"started\":\"{}\"}}", now.to_rfc3339()),
        );
        let crashed = detect_prior_crash();
        assert!(
            !crashed,
            "sentinel with <60s uptime should not be treated as a crash (likely startup failure or operator kill)"
        );
        remove_crash_sentinel();
    }

    #[test]
    fn remove_crash_sentinel_idempotent() {
        let _guard = sentinel_test_lock();
        remove_crash_sentinel();
        remove_crash_sentinel(); // must not panic
    }
}
