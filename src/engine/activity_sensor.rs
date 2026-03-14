//! Activity sensor: detects processes that are actively doing work,
//! even when they are not in the foreground.
//!
//! Two signals:
//! 1. **Power assertions** — processes that told the OS "don't interrupt me"
//!    (audio playback, downloads, background tasks, etc.).
//! 2. **Active children** — processes with children consuming significant CPU
//!    (terminals running builds, scripts, long-running commands, etc.).

use std::collections::{HashMap, HashSet};
use std::process::Command;

/// Parse `pmset -g assertions` and return the set of PIDs that hold any
/// active power assertion. These processes are actively doing something
/// the user or system considers important — freezing them would break it.
///
/// Cost: ~10-20ms (one subprocess call). Cache the result per freeze cycle.
pub fn pids_with_assertions() -> HashSet<u32> {
    let output = match Command::new("pmset").args(["-g", "assertions"]).output() {
        Ok(o) => o,
        Err(_) => return HashSet::new(),
    };

    let text = String::from_utf8_lossy(&output.stdout);
    let mut pids = HashSet::new();

    for line in text.lines() {
        let line = line.trim();
        // Lines with a PID look like: "pid 92974(Electron): [0x...] ..."
        if !line.starts_with("pid ") {
            continue;
        }
        let after_pid = &line[4..];
        let pid_end = match after_pid.find('(') {
            Some(i) => i,
            None => continue,
        };
        if let Ok(pid) = after_pid[..pid_end].trim().parse::<u32>() {
            pids.insert(pid);
        }
    }

    pids
}

/// Return the set of parent PIDs whose children are collectively consuming
/// at least `threshold` percent CPU. A terminal running a build will show
/// up here even if the terminal process itself is idle.
///
/// Uses the already-refreshed `sysinfo::System` — no extra syscalls.
pub fn pids_with_active_children(
    processes: &HashMap<sysinfo::Pid, sysinfo::Process>,
    threshold: f32,
) -> HashSet<u32> {
    let mut child_cpu: HashMap<u32, f32> = HashMap::new();

    for (pid, proc_info) in processes {
        if let Some(parent) = proc_info.parent() {
            let entry = child_cpu.entry(parent.as_u32()).or_insert(0.0);
            *entry += proc_info.cpu_usage();
            let _ = pid; // suppress unused warning
        }
    }

    child_cpu
        .into_iter()
        .filter(|(_, total_cpu)| *total_cpu >= threshold)
        .map(|(pid, _)| pid)
        .collect()
}

/// Combined: returns PIDs that should NOT be frozen because they are
/// actively doing work — either via a power assertion or via active children.
///
/// `processes` should come from a `sysinfo::System` that is already refreshed.
pub fn active_pids(processes: &HashMap<sysinfo::Pid, sysinfo::Process>) -> HashSet<u32> {
    let mut result = pids_with_assertions();
    result.extend(pids_with_active_children(processes, 10.0));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assertions_returns_hashset() {
        // Just check it doesn't panic and returns something reasonable.
        let pids = pids_with_assertions();
        // pmset should always be available on macOS; result may be empty if
        // no assertions are active, but the call must succeed.
        let _ = pids;
    }

    #[test]
    fn active_children_empty_system() {
        let processes = HashMap::new();
        let result = pids_with_active_children(&processes, 10.0);
        assert!(result.is_empty());
    }
}
