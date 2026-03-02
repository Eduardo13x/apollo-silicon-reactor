use std::collections::HashSet;
use std::process::Command;

use chrono::Utc;

use crate::engine::journal::append_journal;
use crate::engine::safety::{
    allowlisted_sysctls, critical_background_processes, protected_processes,
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
) -> ExecuteOutcomes {
    let protected = protected_processes();
    let critical_bg = critical_background_processes();
    let allowlist = allowlisted_sysctls();

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
            | RootAction::QuarantineDaemon { reason, .. } => reason.clone(),
            RootAction::UnfreezeProcess { .. } => "unfreeze".to_string(),
        };

        let result: anyhow::Result<()> = (|| {
            match &action {
                RootAction::BoostProcess { pid, name, .. } => {
                    if protected.iter().any(|p| name.contains(p)) {
                        return Ok(());
                    }
                    // Validate process still exists before acting.
                    if unsafe { libc::kill(*pid as i32, 0) } != 0 {
                        return Ok(());
                    }
                    if caps.can_taskpolicy {
                        let _ = run("/usr/sbin/taskpolicy", &["-l", "0", "-p", &pid.to_string()]);
                        let _ = run("/usr/sbin/taskpolicy", &["-t", "0", "-p", &pid.to_string()]);
                    }
                    let _ = run("/usr/bin/renice", &["-10", "-p", &pid.to_string()]);
                    out.boosts_applied += 1;
                }
                RootAction::ThrottleProcess {
                    pid,
                    name,
                    aggressive,
                    ..
                } => {
                    if protected.iter().any(|p| name.contains(p)) {
                        return Ok(());
                    }
                    if unsafe { libc::kill(*pid as i32, 0) } != 0 {
                        return Ok(());
                    }
                    let is_critical_bg = critical_bg.iter().any(|p| name.contains(p));
                    let aggressive = if is_critical_bg { false } else { *aggressive };
                    if is_critical_bg {
                        out.critical_background_skips += 1;
                        out.push_skip(format!("critical-bg:{}", name));
                    }
                    if caps.can_taskpolicy {
                        let tier = if aggressive { "4" } else { "2" };
                        let _ = run(
                            "/usr/sbin/taskpolicy",
                            &["-l", tier, "-p", &pid.to_string()],
                        );
                    }
                    let nice = if aggressive { "+20" } else { "+10" };
                    let _ = run("/usr/bin/renice", &[nice, "-p", &pid.to_string()]);
                    out.throttles_applied += 1;
                }
                RootAction::FreezeProcess { pid, name, .. } => {
                    if protected.iter().any(|p| name.contains(p)) {
                        return Ok(());
                    }
                    if critical_bg.iter().any(|p| name.contains(p)) {
                        out.critical_background_skips += 1;
                        out.push_skip(format!("critical-bg:{}", name));
                        return Ok(());
                    }
                    // Validate process exists before sending SIGSTOP (BUG 4 fix).
                    if unsafe { libc::kill(*pid as i32, 0) } != 0 {
                        return Ok(());
                    }
                    unsafe {
                        libc::kill(*pid as i32, libc::SIGSTOP);
                    }
                    frozen.insert(*pid);
                    out.freezes_applied += 1;
                }
                RootAction::UnfreezeProcess { pid, .. } => {
                    // Validate PID still belongs to a live process (BUG 4 fix).
                    if unsafe { libc::kill(*pid as i32, 0) } == 0 {
                        unsafe {
                            libc::kill(*pid as i32, libc::SIGCONT);
                        }
                    }
                    frozen.remove(pid);
                    out.unfreezes_applied += 1;
                    out.throttle_reverted += 1;
                }
                RootAction::SetSysctl { key, value, .. } => {
                    if !allowlist.contains(key.as_str()) || !caps.can_sysctl {
                        return Ok(());
                    }
                    // Dynamic allowlist: ignore sysctls that don't exist on this build.
                    if Command::new("/usr/sbin/sysctl")
                        .args(["-n", key])
                        .output()
                        .map(|o| !o.status.success())
                        .unwrap_or(true)
                    {
                        out.invalid_sysctl_denied += 1;
                        out.push_skip(format!("invalid-sysctl:{}", key));
                        return Ok(());
                    }
                    let before_out = Command::new("/usr/sbin/sysctl")
                        .args(["-n", key])
                        .output()
                        .ok();
                    if let Some(o) = before_out {
                        before = Some(String::from_utf8_lossy(&o.stdout).trim().to_string());
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
                RootAction::SetMemorystatus { .. } => {
                    if caps.can_memorystatus {
                        out.paging_hints_applied += 1;
                    }
                }
                RootAction::ToggleSpotlight { enabled, .. } => {
                    if caps.can_mdutil {
                        let state = if *enabled { "on" } else { "off" };
                        let _ = run("mdutil", &["-i", state, "/"]);
                    }
                }
                RootAction::QuarantineDaemon { daemon, active, .. } => {
                    let signal = if *active { "-STOP" } else { "-CONT" };
                    let _ = run("killall", &[signal, daemon]);
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
