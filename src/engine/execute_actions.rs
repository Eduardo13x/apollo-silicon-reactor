use std::collections::HashSet;
use std::process::Command;

use chrono::Utc;

use crate::engine::activity_sensor::pids_with_assertions;
use crate::engine::amx_detector;
use crate::engine::io_tiering::{apply_io_tier, io_tier_for_throttle};
use crate::engine::jetsam_control::{apply_apollo_policy, JetsamClass};
use crate::engine::journal::append_journal;
use crate::engine::mach_qos::{LatencyTier, MachQoSManager, ThreadTier, ThroughputTier};
use crate::engine::process_identity::ProcessIdentity;
use crate::engine::safety::{
    allowlisted_sysctls, allowlisted_sysctls_with_ranges, critical_background_processes,
    protected_processes,
};
use crate::engine::types::{CapabilityReport, JournalEntry, RootAction};

fn run(program: &str, args: &[&str]) -> anyhow::Result<()> {
    let out = Command::new(program).args(args).output()?;
    if out.status.success() {
        Ok(())
    } else {
        anyhow::bail!(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// Verify PID identity using kernel start-time.
///
/// If `start_sec > 0`, checks that the process still has the same start-time
/// (prevents A-B-A PID recycling). Falls back to name-only check when
/// start-time is unavailable (legacy actions with `start_sec == 0`).
fn verify_pid_identity(pid: u32, expected_name: &str, start_sec: u64, start_usec: u64) -> bool {
    match ProcessIdentity::from_pid(pid) {
        Some(current) => {
            // If we have a captured start-time, verify it matches exactly.
            if start_sec > 0 && (current.start_sec != start_sec || current.start_usec != start_usec)
            {
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
    let critical_bg = critical_background_processes();
    let allowlist = allowlisted_sysctls();
    // ML/AMX workloads: final safety net — never throttle or freeze inference processes.
    let ml_pids = amx_detector::ml_protected_pids();
    // Lazy: computed only if we actually have a FreezeProcess action.
    let mut assertion_pids: Option<std::collections::HashSet<u32>> = None;

    // Pre-lowercase learned patterns once per call (avoids ~2,400 allocations/cycle).
    let learned_protected_lc: Vec<String> = learned_protected
        .iter()
        .map(|s| s.to_ascii_lowercase())
        .collect();
    let learned_interactive_lc: Vec<String> = learned_interactive
        .iter()
        .map(|s| s.to_ascii_lowercase())
        .collect();

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
                    if protected.iter().any(|p| name.contains(p)) {
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
                        } else {
                            // Fallback to CLI if no QoS manager provided.
                            let _ =
                                run("/usr/sbin/taskpolicy", &["-l", "0", "-p", &pid.to_string()]);
                            let _ =
                                run("/usr/sbin/taskpolicy", &["-t", "0", "-p", &pid.to_string()]);
                        }
                        // Boost I/O tier to Interactive.
                        apply_io_tier(*pid, crate::engine::io_tiering::IOTier::Interactive);
                    }
                    let _ = run("/usr/bin/renice", &["-10", "-p", &pid.to_string()]);
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
                    if protected.iter().any(|p| name.contains(p)) {
                        out.push_skip(format!("protected:{}", name));
                        return Ok(());
                    }
                    // ML/AMX protection: never throttle inference workloads.
                    if ml_pids.contains(pid) {
                        out.push_skip(format!("ml-protected:{}", name));
                        return Ok(());
                    }
                    {
                        let name_lc = name.to_ascii_lowercase();
                        if learned_protected_lc
                            .iter()
                            .any(|p| name_lc.contains(p.as_str()))
                        {
                            out.push_skip(format!("learned-protected:{}", name));
                            return Ok(());
                        }
                        // Never throttle interactive apps (they deserve boosted priority).
                        if learned_interactive_lc
                            .iter()
                            .any(|p| name_lc.contains(p.as_str()))
                        {
                            out.push_skip(format!("learned-interactive:{}", name));
                            return Ok(());
                        }
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
                                crate::engine::mach_qos::SchedulingTier::Background // E-cores only
                            } else {
                                crate::engine::mach_qos::SchedulingTier::Normal     // scheduler decides, less invasive than E-cores-only
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
                        } else {
                            let tier = if aggressive { "4" } else { "2" };
                            let pid_str = pid.to_string();
                            let _ = run("/usr/sbin/taskpolicy", &["-l", tier, "-p", &pid_str]);
                        }
                        // Granular I/O tiering based on aggressiveness.
                        let io_tier = io_tier_for_throttle(aggressive);
                        apply_io_tier(*pid, io_tier);
                    }
                    let nice = if aggressive { "+20" } else { "+10" };
                    let _ = run("/usr/bin/renice", &[nice, "-p", &pid.to_string()]);
                    out.throttles_applied += 1;
                }
                RootAction::FreezeProcess {
                    pid,
                    name,
                    start_sec,
                    start_usec,
                    ..
                } => {
                    if protected.iter().any(|p| name.contains(p)) {
                        return Ok(());
                    }
                    // ML/AMX protection: never freeze inference workloads.
                    if ml_pids.contains(pid) {
                        out.push_skip(format!("ml-protected:{}", name));
                        return Ok(());
                    }
                    {
                        let name_lc = name.to_ascii_lowercase();
                        if learned_protected_lc
                            .iter()
                            .any(|p| name_lc.contains(p.as_str()))
                        {
                            out.push_skip(format!("learned-protected:{}", name));
                            return Ok(());
                        }
                        if learned_interactive_lc
                            .iter()
                            .any(|p| name_lc.contains(p.as_str()))
                        {
                            out.push_skip(format!("learned-interactive:{}", name));
                            return Ok(());
                        }
                    }
                    if critical_bg.iter().any(|p| name.contains(p)) {
                        out.critical_background_skips += 1;
                        out.push_skip(format!("critical-bg:{}", name));
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
                    let read_result = Command::new("/usr/sbin/sysctl").args(["-n", key]).output();
                    match read_result {
                        Ok(o) if o.status.success() => {
                            before = Some(String::from_utf8_lossy(&o.stdout).trim().to_string());
                        }
                        _ => {
                            // Key doesn't exist on this kernel build.
                            out.invalid_sysctl_denied += 1;
                            out.push_skip(format!("invalid-sysctl:{}", key));
                            return Ok(());
                        }
                    }
                    run("/usr/sbin/sysctl", &["-w", &format!("{}={}", key, value)])?;
                    let after_out = Command::new("/usr/sbin/sysctl")
                        .args(["-n", key])
                        .output()
                        .ok();
                    if let Some(o) = after_out {
                        after = Some(String::from_utf8_lossy(&o.stdout).trim().to_string());
                    }
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
                            let pid_str = pid.to_string();
                            let _ = run(
                                "/usr/sbin/sysctl",
                                &[
                                    "-w",
                                    &format!("kern.memorystatus_vm_pressure_send={}", pid_str),
                                ],
                            );
                            out.paging_hints_applied += 1;
                        }
                    }
                }
                RootAction::ToggleSpotlight { enabled, .. } => {
                    if caps.can_mdutil {
                        let state = if *enabled { "on" } else { "off" };
                        let _ = run("/usr/sbin/mdutil", &["-i", state, "/"]);
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
                        let signal = if *active { "-STOP" } else { "-CONT" };
                        let _ = run("/usr/bin/killall", &[signal, daemon]);
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
