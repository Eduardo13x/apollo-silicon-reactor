use crate::engine::sysctl_direct;
use std::collections::HashSet;

use chrono::Utc;

use crate::engine::activity_sensor::pids_with_assertions;
use crate::engine::amx_detector;
use crate::engine::io_tiering::{apply_io_tier, io_tier_for_throttle};
use crate::engine::jetsam_control::{apply_apollo_policy, JetsamClass};
use crate::engine::journal::append_journal;
use crate::engine::mach_qos::{LatencyTier, MachQoSManager, ThreadTier, ThroughputTier};
use crate::engine::proc_taskinfo;
use crate::engine::process_identity::{self, ProcessIdentity};
use crate::engine::safety::{
    allowlisted_sysctls, allowlisted_sysctls_with_ranges, classify_protection,
    infrastructure_processes, protected_processes, ProtectionLevel,
};
use crate::engine::types::{CapabilityReport, JournalEntry, RootAction};

/// Set the nice value for a process via `setpriority(2)`.
/// Returns `Ok(())` on success, or an error if the call failed.
fn set_nice(pid: u32, nice: i32) -> anyhow::Result<()> {
    // errno must be cleared before setpriority — a return of -1 is ambiguous
    // because -1 is a valid priority.  We use the errno convention instead.
    unsafe {
        *libc::__error() = 0;
        let rc = libc::setpriority(libc::PRIO_PROCESS, pid, nice);
        if rc == -1 && *libc::__error() != 0 {
            anyhow::bail!(
                "setpriority({}, {}) failed: {}",
                pid,
                nice,
                std::io::Error::last_os_error()
            );
        }
    }
    Ok(())
}

/// Send a signal to all processes whose name matches `daemon` exactly.
/// Equivalent to `/usr/bin/killall <signal> <daemon>` but without fork/exec.
fn killall_by_name(daemon: &str, signal: i32) -> anyhow::Result<()> {
    let pids = proc_taskinfo::list_all_pids();
    let mut matched = 0u32;
    for pid in pids {
        if let Some(name) = process_identity::proc_name_for_pid(pid) {
            if name == daemon {
                let rc = unsafe { libc::kill(pid as i32, signal) };
                if rc == 0 {
                    matched += 1;
                }
            }
        }
    }
    if matched == 0 {
        anyhow::bail!("no process found matching '{}'", daemon);
    }
    Ok(())
}

/// Toggle Spotlight indexing via `mdutil -a -i on/off`.
///
/// mdutil communicates with the Spotlight server via XPC (com.apple.spotlightserver).
/// Fire-and-forget: `.spawn()` instead of `.status()` to avoid blocking the daemon
/// loop indefinitely if the Spotlight server is unresponsive.
fn spotlight_set_indexing(enabled: bool) {
    let flag = if enabled { "on" } else { "off" };
    let _ = std::process::Command::new("/usr/bin/mdutil")
        .args(["-a", "-i", flag])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn(); // non-blocking — child reaps automatically
}

fn run_sysctl_write(key: &str, value: &str) -> anyhow::Result<()> {
    if sysctl_write_with_timeout(key, value) {
        Ok(())
    } else {
        anyhow::bail!("sysctl write failed: {}={}", key, value)
    }
}

// ── Timeout wrappers for kernel syscalls that can block as root ──────────

/// Read a sysctl with 500ms timeout. Prevents `sysctlbyname` from blocking
/// the daemon loop indefinitely under kernel lock contention.
fn sysctl_read_with_timeout(key: &str) -> Option<String> {
    let key = key.to_string();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(sysctl_direct::read_str(&key));
    });
    rx.recv_timeout(std::time::Duration::from_millis(500))
        .ok()
        .flatten()
}

/// Write a sysctl with 500ms timeout.
fn sysctl_write_with_timeout(key: &str, value: &str) -> bool {
    let key = key.to_string();
    let value = value.to_string();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(sysctl_direct::write_str_value(&key, &value));
    });
    rx.recv_timeout(std::time::Duration::from_millis(500))
        .ok()
        .unwrap_or(false)
}

/// Write an i32 sysctl with 500ms timeout.
fn sysctl_write_i32_with_timeout(key: &str, value: i32) -> bool {
    let key = key.to_string();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(sysctl_direct::write_i32(&key, value));
    });
    rx.recv_timeout(std::time::Duration::from_millis(500))
        .ok()
        .unwrap_or(false)
}

/// Verify PID identity using kernel start-time.
///
/// If `start_sec > 0`, checks that the process still has the same start-time
/// (prevents A-B-A PID recycling). Falls back to name-only check when
/// start-time is unavailable (legacy actions with `start_sec == 0`).
fn verify_pid_identity(pid: u32, expected_name: &str, start_sec: u64, start_usec: u64) -> bool {
    match ProcessIdentity::from_pid(pid) {
        Some(current) => {
            // start_sec check: guards against PID recycling between snapshot and execution.
            if start_sec > 0 && current.start_sec != start_sec {
                return false;
            }
            // start_usec check: only when explicitly provided (non-zero).
            // decide_actions passes start_usec=0 because sysinfo doesn't expose
            // sub-second precision — treating 0 as "not provided" prevents false
            // positives where pbi_start_tvusec != 0 for every live process.
            if start_sec > 0 && start_usec > 0 && current.start_usec != start_usec {
                return false;
            }
            // Name check as defense-in-depth (handles start_sec==0 fallback too).
            let name_ok = current.name == expected_name
                || (current.name.len() >= 6 && expected_name.starts_with(&current.name))
                || (expected_name.len() >= 6 && current.name.starts_with(expected_name));
            name_ok
        }
        None => false, // Process already dead.
    }
}

/// Aggregate counters returned by execute_actions so callers do not need to
/// hold a RuntimeMetrics lock during blocking I/O.
#[derive(Debug, Default)]
pub struct ExecuteOutcomes {
    pub boosts_applied: u64,
    pub throttles_applied: u64,
    pub freezes_applied: u64,
    pub unfreezes_applied: u64,
    pub paging_hints_applied: u64,
    pub sysctl_applied: u64,
    pub failures: u64,
    pub last_error: Option<String>,
    pub critical_background_skips: u64,
    pub invalid_sysctl_denied: u64,
    pub top_skipped: Vec<String>,
    pub throttle_reverted: u64,
    pub thread_qos_applied: u64,
}

impl ExecuteOutcomes {
    fn push_skip(&mut self, what: String) {
        if self.top_skipped.len() < 12 && !self.top_skipped.contains(&what) {
            self.top_skipped.push(what);
        }
    }
}

/// Execute a list of actions. Returns an [ExecuteOutcomes] accumulator that
/// the caller can merge into RuntimeMetrics **after** releasing any locks,
/// eliminating the need to hold locks across blocking I/O.
pub fn execute_actions(
    actions: Vec<RootAction>,
    caps: &CapabilityReport,
    journal_path: &std::path::Path,
    frozen: &mut HashSet<u32>,
    learned_protected: &[String],
    learned_interactive: &[String],
    mut qos_mgr: Option<&mut MachQoSManager>,
) -> ExecuteOutcomes {
    let protected = protected_processes();
    // Only infrastructure (docker, postgres, redis, etc.) gets unconditional protection
    // at execution time. Dev runtimes (python, node, etc.) are filtered upstream by
    // behavioral_protection_score in the daemon — if they reach execute_actions,
    // they've already lost their behavioral gate.
    let critical_bg = infrastructure_processes();
    let allowlist = allowlisted_sysctls();
    // Self-protection: never freeze/throttle/kill the daemon itself.
    let my_pid = std::process::id();
    // ML/AMX workloads: final safety net — never throttle or freeze inference processes.
    let ml_pids = amx_detector::ml_protected_pids();
    // Lazy: computed only if we actually have a FreezeProcess action.
    let mut assertion_pids: Option<std::collections::HashSet<u32>> = None;

    // Unified policy list for classify_protection(): learned_protected + learned_interactive.
    //
    // At execute time there is no foreground context, so learned_interactive patterns are
    // treated as unconditional skips (same as learned_protected).  Both are passed as
    // `policy_protected` to classify_protection(), which maps them to ProtectionLevel::Unconditional.
    // This is behaviorally identical to the previous three-step explicit check.
    let policy_all: Vec<String> = learned_protected
        .iter()
        .chain(learned_interactive.iter())
        .cloned()
        .collect();
    // Empty infra set — infrastructure_processes() is handled separately below
    // in ThrottleProcess (soft throttle path) and FreezeProcess (critical-bg skip path).
    let empty_infra: std::collections::HashSet<&'static str> = std::collections::HashSet::new();

    let mut out = ExecuteOutcomes::default();

    for action in actions {
        let mut success = false;
        let mut before = None;
        let mut after = None;
        let reason = match &action {
            RootAction::BoostProcess { reason, .. }
            | RootAction::ThrottleProcess { reason, .. }
            | RootAction::FreezeProcess { reason, .. }
            | RootAction::SetSysctl { reason, .. }
            | RootAction::SetMemorystatus { reason, .. }
            | RootAction::ToggleSpotlight { reason, .. }
            | RootAction::QuarantineDaemon { reason, .. }
            | RootAction::SetThreadQoS { reason, .. } => reason.clone(),
            RootAction::UnfreezeProcess { .. } => "unfreeze".to_string(),
        };

        let result: anyhow::Result<()> = (|| {
            match &action {
                RootAction::BoostProcess { pid, name, .. } => {
                    if *pid == my_pid || protected.iter().any(|p| name.contains(p)) {
                        return Ok(());
                    }
                    // Validate PID identity (name-only for boost — no start-time available).
                    if !verify_pid_identity(*pid, name, 0, 0) {
                        return Ok(());
                    }
                    if caps.can_taskpolicy {
                        // Phase 2: direct Mach syscalls (~50µs vs ~5ms fork/exec).
                        if let Some(ref mut mgr) = qos_mgr {
                            mgr.set_tier(*pid, crate::engine::mach_qos::SchedulingTier::Foreground);
                            mgr.set_latency_and_throughput(
                                *pid,
                                LatencyTier::Interactive,
                                ThroughputTier::High,
                            );
                        }
                        // Boost I/O tier to Interactive.
                        apply_io_tier(*pid, crate::engine::io_tiering::IOTier::Interactive);
                    }
                    let _ = set_nice(*pid, -10);
                    out.boosts_applied += 1;
                }
                RootAction::ThrottleProcess {
                    pid,
                    name,
                    aggressive,
                    start_sec,
                    start_usec,
                    ..
                } => {
                    if *pid == my_pid {
                        return Ok(());
                    }
                    // Unified protection check: hard OS names + policy-learned + interactive.
                    // learned_interactive is treated as Unconditional at execute time because
                    // no foreground context is available here (see policy_all pre-computation).
                    // infra (infrastructure_processes) is intentionally excluded: critical_bg
                    // below handles infra with soft-throttle semantics, not a full skip.
                    match classify_protection(name, &protected, &empty_infra, &policy_all, false) {
                        ProtectionLevel::Unconditional => {
                            out.push_skip(format!("protected:{}", name));
                            return Ok(());
                        }
                        ProtectionLevel::ConditionalForeground | ProtectionLevel::Unprotected => {}
                    }
                    // ML/AMX protection: never throttle inference workloads.
                    if ml_pids.contains(pid) {
                        out.push_skip(format!("ml-protected:{}", name));
                        return Ok(());
                    }
                    // Validate PID identity with start-time (prevents A-B-A recycling).
                    if !verify_pid_identity(*pid, name, *start_sec, *start_usec) {
                        out.push_skip(format!("pid-recycled:{}", name));
                        return Ok(());
                    }
                    let is_critical_bg = critical_bg.iter().any(|p| name.contains(p));
                    let aggressive = if is_critical_bg { false } else { *aggressive };
                    if is_critical_bg {
                        out.critical_background_skips += 1;
                        out.push_skip(format!("critical-bg:{}", name));
                    }
                    if caps.can_taskpolicy {
                        // Phase 2: direct Mach syscalls for CPU tier routing.
                        if let Some(ref mut mgr) = qos_mgr {
                            let sched_tier = if aggressive {
                                crate::engine::mach_qos::SchedulingTier::Background
                            // E-cores only
                            } else {
                                crate::engine::mach_qos::SchedulingTier::Normal // scheduler decides, less invasive than E-cores-only
                            };
                            mgr.set_tier(*pid, sched_tier);
                            let lat = if aggressive {
                                LatencyTier::Background
                            } else {
                                LatencyTier::Default
                            };
                            let thr = if aggressive {
                                ThroughputTier::Low
                            } else {
                                ThroughputTier::Default
                            };
                            mgr.set_latency_and_throughput(*pid, lat, thr);
                        }
                        // Granular I/O tiering based on aggressiveness.
                        let io_tier = io_tier_for_throttle(aggressive);
                        apply_io_tier(*pid, io_tier);
                    }
                    let nice_val: i32 = if aggressive { 20 } else { 10 };
                    let _ = set_nice(*pid, nice_val);
                    out.throttles_applied += 1;
                }
                RootAction::FreezeProcess {
                    pid,
                    name,
                    start_sec,
                    start_usec,
                    ..
                } => {
                    if *pid == my_pid {
                        return Ok(());
                    }
                    // Unified protection check: hard OS names + infra + policy-learned + interactive.
                    // Unlike ThrottleProcess, infra (critical_bg) is included here because
                    // FreezeProcess treats infra as a full skip (not a soft-throttle path).
                    // learned_interactive is treated as Unconditional: no foreground context
                    // at execute time (see policy_all pre-computation above).
                    match classify_protection(name, &protected, &critical_bg, &policy_all, false) {
                        ProtectionLevel::Unconditional => {
                            if critical_bg.iter().any(|p| name.contains(p)) {
                                out.critical_background_skips += 1;
                            }
                            out.push_skip(format!("protected:{}", name));
                            return Ok(());
                        }
                        ProtectionLevel::ConditionalForeground | ProtectionLevel::Unprotected => {}
                    }
                    // ML/AMX protection: never freeze inference workloads.
                    if ml_pids.contains(pid) {
                        out.push_skip(format!("ml-protected:{}", name));
                        return Ok(());
                    }
                    // Validate PID identity with start-time (prevents A-B-A recycling).
                    if !verify_pid_identity(*pid, name, *start_sec, *start_usec) {
                        return Ok(());
                    }
                    // Never freeze processes with active power assertions
                    // (audio playback, active downloads, background tasks).
                    let busy = assertion_pids.get_or_insert_with(pids_with_assertions);
                    if busy.contains(pid) {
                        out.push_skip(format!("assertion-active:{}", name));
                        return Ok(());
                    }
                    // Demote disk I/O to Passive before SIGSTOP.
                    // This prevents the process from hoarding SSD bandwidth on resume.
                    if caps.can_taskpolicy {
                        apply_io_tier(*pid, crate::engine::io_tiering::IOTier::Passive);
                    }
                    // Jetsam: marcar como BACKGROUND en el kernel antes de SIGSTOP.
                    // Así si el sistema entra en OOM mientras el proceso está frozen,
                    // el kernel lo mata primero en lugar de matar procesos interactivos.
                    if caps.can_memorystatus {
                        let _ = apply_apollo_policy(*pid, JetsamClass::Noise);
                    }
                    let rc = unsafe { libc::kill(*pid as i32, libc::SIGSTOP) };
                    if rc == 0 {
                        frozen.insert(*pid);
                        out.freezes_applied += 1;
                    }
                }
                RootAction::UnfreezeProcess { pid, .. } => {
                    let alive = unsafe { libc::kill(*pid as i32, 0) } == 0;
                    if alive {
                        let rc = unsafe { libc::kill(*pid as i32, libc::SIGCONT) };
                        if rc == 0 {
                            // Restore I/O tier to Standard on unfreeze.
                            if caps.can_taskpolicy {
                                apply_io_tier(*pid, crate::engine::io_tiering::IOTier::Standard);
                            }
                            // Restaurar prioridad jetsam a FOREGROUND al descongelar.
                            if caps.can_memorystatus {
                                let _ = apply_apollo_policy(*pid, JetsamClass::Interactive);
                            }
                            frozen.remove(pid);
                            out.unfreezes_applied += 1;
                            out.throttle_reverted += 1;
                        }
                        // If SIGCONT failed (e.g. permission denied), keep in frozen set
                        // so the TTL or next cycle can retry.
                    } else {
                        // Process is dead — safe to remove from frozen set.
                        frozen.remove(pid);
                    }
                }
                RootAction::SetSysctl { key, value, .. } => {
                    if !allowlist.contains(key.as_str()) || !caps.can_sysctl {
                        return Ok(());
                    }
                    // Validate value range — prevents dangerous values like kern.maxfiles=1.
                    let ranges = allowlisted_sysctls_with_ranges();
                    if let Some(range) = ranges.iter().find(|r| r.key == key.as_str()) {
                        if let Ok(numeric_val) = value.parse::<i64>() {
                            if numeric_val < range.min || numeric_val > range.max {
                                out.invalid_sysctl_denied += 1;
                                out.push_skip(format!("sysctl-out-of-range:{}={}", key, value));
                                return Ok(());
                            }
                        }
                    }
                    // Read current value — doubles as existence check.
                    // Uses timeout wrapper: sysctlbyname can block as root.
                    let read_result = sysctl_read_with_timeout(key);
                    match read_result {
                        Some(val) => {
                            before = Some(val);
                        }
                        None => {
                            out.invalid_sysctl_denied += 1;
                            out.push_skip(format!("invalid-sysctl:{}", key));
                            return Ok(());
                        }
                    }
                    run_sysctl_write(key, value)?;
                    after = sysctl_read_with_timeout(key);
                    out.sysctl_applied += 1;
                }
                RootAction::SetMemorystatus { pid, .. } => {
                    if caps.can_memorystatus {
                        // Guard: never send memory pressure to protected/critical processes.
                        let is_protected = crate::engine::process_identity::proc_name_for_pid(*pid)
                            .map(|name| {
                                let nl = name.to_ascii_lowercase();
                                protected
                                    .iter()
                                    .any(|p| nl.contains(&p.to_ascii_lowercase()))
                                    || critical_bg
                                        .iter()
                                        .any(|c| nl.contains(&c.to_ascii_lowercase()))
                            })
                            .unwrap_or(false);
                        if is_protected {
                            // Skip — do not pressure protected processes.
                        } else {
                            let _ = sysctl_write_i32_with_timeout(
                                "kern.memorystatus_vm_pressure_send",
                                *pid as i32,
                            );
                            out.paging_hints_applied += 1;
                        }
                    }
                }
                RootAction::ToggleSpotlight { enabled, .. } => {
                    if caps.can_mdutil {
                        spotlight_set_indexing(*enabled);
                    }
                }
                RootAction::QuarantineDaemon { daemon, active, .. } => {
                    // Guard: never quarantine protected/critical daemons.
                    let dl = daemon.to_ascii_lowercase();
                    let is_protected = protected
                        .iter()
                        .any(|p| dl.contains(&p.to_ascii_lowercase()))
                        || critical_bg
                            .iter()
                            .any(|c| dl.contains(&c.to_ascii_lowercase()));
                    // Validate daemon name: only alphanumeric, dots, hyphens, underscores.
                    let name_valid = !daemon.is_empty()
                        && daemon.len() <= 128
                        && daemon
                            .chars()
                            .all(|c| c.is_alphanumeric() || c == '.' || c == '-' || c == '_');
                    if is_protected {
                        // Skip — do not quarantine protected daemons.
                    } else if !name_valid {
                        // Skip — daemon name contains invalid characters.
                    } else {
                        let signal = if *active {
                            libc::SIGSTOP
                        } else {
                            libc::SIGCONT
                        };
                        let _ = killall_by_name(daemon, signal);
                    }
                }
                RootAction::SetThreadQoS {
                    pid,
                    name,
                    thread_index,
                    tier,
                    ..
                } => {
                    if protected.iter().any(|p| name.contains(p)) {
                        return Ok(());
                    }
                    if !verify_pid_identity(*pid, name, 0, 0) {
                        return Ok(());
                    }
                    let thread_tier = match tier.as_str() {
                        "interactive" => ThreadTier::Interactive,
                        "background" => ThreadTier::Background,
                        _ => ThreadTier::Utility,
                    };
                    if let Some(ref mut mgr) = qos_mgr {
                        if mgr.set_thread_qos(*pid, *thread_index, thread_tier) {
                            out.thread_qos_applied += 1;
                        }
                    }
                }
            }
            Ok(())
        })();

        if let Err(e) = result {
            out.failures += 1;
            out.last_error = Some(e.to_string());
        } else {
            success = true;
        }

        let entry = JournalEntry {
            timestamp: Utc::now(),
            action,
            before,
            after,
            success,
            reason,
        };
        let _ = append_journal(journal_path, &entry);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn make_caps() -> CapabilityReport {
        CapabilityReport {
            can_taskpolicy: false,
            can_sysctl: false,
            can_memorystatus: false,
            can_mdutil: false,
            can_tmutil: false,
            is_root: false,
            unavailable: vec![],
        }
    }

    /// Helper: run execute_actions with a temp journal and return outcomes.
    fn run(
        actions: Vec<RootAction>,
        learned_protected: &[String],
        learned_interactive: &[String],
    ) -> ExecuteOutcomes {
        let journal = std::env::temp_dir().join("apollo-test-execute-actions.jsonl");
        let mut frozen = HashSet::new();
        execute_actions(
            actions,
            &make_caps(),
            &journal,
            &mut frozen,
            learned_protected,
            learned_interactive,
            None,
        )
    }

    /// A PID unlikely to exist so SIGSTOP/setpriority don't land on a real process.
    /// Using PID 9_999_999 (exceeds typical macOS max PID of ~99_999).
    const GHOST_PID: u32 = 9_999_999;

    // ── learned_interactive skips (BUG-07) ────────────────────────────────────

    #[test]
    fn throttle_skips_learned_interactive_process() {
        let interactive = vec!["MyInteractiveApp".to_string()];
        let outcomes = run(
            vec![RootAction::ThrottleProcess {
                pid: GHOST_PID,
                name: "MyInteractiveApp".to_string(),
                aggressive: false,
                reason: "test".to_string(),
                start_sec: 0,
                start_usec: 0,
            }],
            &[],
            &interactive,
        );
        assert_eq!(
            outcomes.throttles_applied, 0,
            "learned_interactive process must not be throttled"
        );
        assert!(
            outcomes
                .top_skipped
                .iter()
                .any(|s| s.contains("MyInteractiveApp")),
            "skip reason must mention the process name"
        );
    }

    #[test]
    fn freeze_skips_learned_interactive_process() {
        let interactive = vec!["MyInteractiveApp".to_string()];
        let outcomes = run(
            vec![RootAction::FreezeProcess {
                pid: GHOST_PID,
                name: "MyInteractiveApp".to_string(),
                reason: "test".to_string(),
                start_sec: 0,
                start_usec: 0,
            }],
            &[],
            &interactive,
        );
        assert_eq!(
            outcomes.freezes_applied, 0,
            "learned_interactive process must not be frozen"
        );
        assert!(
            outcomes
                .top_skipped
                .iter()
                .any(|s| s.contains("MyInteractiveApp")),
            "skip reason must mention the process name"
        );
    }

    #[test]
    fn throttle_skips_learned_interactive_case_insensitive() {
        // Pattern stored lowercase; process name has mixed case — must still skip.
        let interactive = vec!["myinteractiveapp".to_string()];
        let outcomes = run(
            vec![RootAction::ThrottleProcess {
                pid: GHOST_PID,
                name: "MyInteractiveApp".to_string(),
                aggressive: false,
                reason: "test".to_string(),
                start_sec: 0,
                start_usec: 0,
            }],
            &[],
            &interactive,
        );
        assert_eq!(outcomes.throttles_applied, 0);
    }

    #[test]
    fn throttle_skips_learned_protected_process() {
        let protected = vec!["MyProtectedDaemon".to_string()];
        let outcomes = run(
            vec![RootAction::ThrottleProcess {
                pid: GHOST_PID,
                name: "MyProtectedDaemon".to_string(),
                aggressive: false,
                reason: "test".to_string(),
                start_sec: 0,
                start_usec: 0,
            }],
            &protected,
            &[],
        );
        assert_eq!(outcomes.throttles_applied, 0);
    }
}
