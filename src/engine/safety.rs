use std::collections::HashSet;

use crate::engine::types::{ActionBudgetState, RootAction, SafetyPolicy};

pub struct SysctlRange {
    pub key: &'static str,
    pub min: i64,
    pub max: i64,
}

pub fn protected_processes() -> HashSet<&'static str> {
    [
        "kernel_task",
        "launchd",
        "WindowServer",
        "loginwindow",
        "hidd",
        "configd",
        "opendirectoryd",
        "notifyd",
        "UserEventAgent",
        "securityd",
        "syspolicyd",
        "tccd",
        "CoreServicesUIAgent",
        // XPC infrastructure — xpcproxy is the XPC service launcher; freezing it
        // is a no-op (kernel re-spawns immediately) that spams the journal.
        "xpcproxy",
        "trustd",
        "distnoted",
        // Spotlight stack (never touch — throttling these breaks app search).
        "Spotlight",
        "mds",
        "mds_stores",
        "mdworker",
        "mdworker_shared",
        "corespotlightd",
        "spotlightknowledged",
    ]
    .into_iter()
    .collect()
}

pub fn allowlisted_sysctls_with_ranges() -> Vec<SysctlRange> {
    vec![
        SysctlRange {
            key: "debug.lowpri_throttle_enabled",
            min: 0,
            max: 1,
        },
        SysctlRange {
            key: "net.inet.tcp.sendspace",
            min: 4096,
            max: 1_048_576,
        }, // 4KB-1MB
        SysctlRange {
            key: "net.inet.tcp.recvspace",
            min: 4096,
            max: 1_048_576,
        },
        SysctlRange {
            key: "net.inet.tcp.delayed_ack",
            min: 0,
            max: 3,
        },
        SysctlRange {
            key: "net.inet.tcp.min_iaj_win",
            min: 0,
            max: 64,
        },
        SysctlRange {
            key: "net.inet.tcp.win_scale_factor",
            min: 1,
            max: 14,
        },
        SysctlRange {
            key: "net.inet.tcp.autorcvbufmax",
            min: 65_536,
            max: 16_777_216,
        }, // 64KB-16MB
        SysctlRange {
            key: "net.inet.tcp.autosndbufmax",
            min: 65_536,
            max: 16_777_216,
        },
        SysctlRange {
            key: "vm.compressor_poll_interval",
            min: 1,
            max: 1000,
        },
        SysctlRange {
            key: "vm.compressor_sample_min",
            min: 0,
            max: 1000,
        },
        SysctlRange {
            key: "kern.maxvnodes",
            min: 10_000,
            max: 1_000_000,
        },
        SysctlRange {
            key: "kern.maxfiles",
            min: 65_536, // 256 era peligroso: causaría "too many open files" en cascada
            max: 1_048_576,
        },
        SysctlRange {
            key: "kern.maxfilesperproc",
            min: 2_048, // 256 podría romper apps con muchas conexiones/ventanas
            max: 524_288,
        },
        SysctlRange {
            key: "kern.ipc.somaxconn",
            min: 128,
            max: 65_535,
        },
        SysctlRange {
            key: "kern.ipc.maxsockbuf",
            min: 65_536,
            max: 67_108_864,
        }, // 64KB-64MB
        SysctlRange {
            key: "iogpu.wired_limit_mb",
            min: 256,
            max: 65_536,
        },
        SysctlRange {
            key: "debug.iogpu.wired_limit",
            min: 256,
            max: 65_536,
        },
    ]
}

pub fn allowlisted_sysctls() -> HashSet<&'static str> {
    allowlisted_sysctls_with_ranges()
        .iter()
        .map(|r| r.key)
        .collect()
}

pub fn critical_background_processes() -> HashSet<&'static str> {
    // Dev workloads that should not be slowed down by an "interactive-first" optimizer.
    // Match by substring, same style as `protected_processes()`.
    [
        // Containers / VMs
        "podman",
        "qemu-system",
        "colima",
        "lima",
        "docker",
        "com.docker",
        "vpnkit",
        "hyperkit",
        // Databases / caches
        "postgres",
        "mysqld",
        "mariadbd",
        "redis-server",
        "mongod",
        // Queues / streaming
        "kafka",
        "zookeeper",
        "rabbitmq",
        // Common dev runtimes / servers
        "node",
        "python",
        "java",
        "go",
        "nginx",
    ]
    .into_iter()
    .collect()
}

pub fn enforce_limits(actions: Vec<RootAction>, policy: &SafetyPolicy) -> Vec<RootAction> {
    let mut out = Vec::new();
    let mut boosts = 0usize;
    let mut throttles = 0usize;
    let mut hints = 0usize;
    let mut freezes = 0usize;
    let mut sysctl_writes = 0usize;
    let mut thread_qos = 0usize;

    for action in actions {
        let allowed = match &action {
            RootAction::BoostProcess { .. } => {
                boosts += 1;
                boosts <= policy.max_boosts_per_cycle
            }
            RootAction::ThrottleProcess { .. } => {
                throttles += 1;
                throttles <= policy.max_throttles_per_cycle
            }
            RootAction::SetMemorystatus { .. } => {
                hints += 1;
                hints <= policy.max_paging_hints_per_cycle
            }
            RootAction::FreezeProcess { .. } => {
                freezes += 1;
                freezes <= policy.max_freezes_per_cycle
            }
            RootAction::SetSysctl { .. } => {
                sysctl_writes += 1;
                sysctl_writes <= policy.max_sysctl_writes_per_cycle
            }
            RootAction::SetThreadQoS { .. } => {
                thread_qos += 1;
                thread_qos <= policy.max_thread_qos_per_cycle
            }
            _ => true,
        };

        if allowed {
            out.push(action);
        }
    }

    out
}

pub fn enforce_limits_with_budget(
    actions: Vec<RootAction>,
    policy: &SafetyPolicy,
    budget: &mut ActionBudgetState,
    minute_budget_cap: usize,
) -> Vec<RootAction> {
    let mut out = Vec::new();

    for action in enforce_limits(actions, policy) {
        if budget.minute_actions >= minute_budget_cap {
            budget.boost_denied_cooldown += 1;
            break;
        }

        match &action {
            RootAction::BoostProcess { .. } => budget.cycle_boosts += 1,
            RootAction::ThrottleProcess { .. } => budget.cycle_throttles += 1,
            RootAction::SetMemorystatus { .. } => budget.cycle_hints += 1,
            RootAction::FreezeProcess { .. } => budget.cycle_freezes += 1,
            RootAction::SetSysctl { .. } => budget.cycle_sysctl_writes += 1,
            RootAction::SetThreadQoS { .. } => budget.cycle_thread_qos += 1,
            _ => {}
        }

        budget.minute_actions += 1;
        out.push(action);
    }

    out
}

/// Check if a pattern could match any protected or critical process name.
/// Uses bidirectional prefix/suffix matching to catch partial evasions.
pub fn pattern_conflicts_with_protected(pattern: &str) -> bool {
    let pat = pattern.to_lowercase();
    if pat.len() < 4 {
        return true; // Too short patterns are dangerous
    }
    // Reject non-ASCII patterns to avoid UTF-8 multibyte edge cases
    if !pat.is_ascii() {
        return true;
    }

    let all_protected: Vec<&str> = protected_processes()
        .into_iter()
        .chain(critical_background_processes())
        .collect();

    for name in &all_protected {
        let name_lower = name.to_lowercase();
        // Bidirectional: pattern matches name OR name matches pattern
        if pat.contains(&name_lower) || name_lower.contains(&pat) {
            return true;
        }
        // Prefix/suffix overlap check (catches "kernel_tas" vs "kernel_task")
        let min_len = pat.len().min(name_lower.len());
        let overlap = min_len * 3 / 4; // 75% overlap threshold
        if overlap >= 4 {
            // Safe slicing: use .get() to avoid panic on invalid byte boundaries
            let pat_prefix = pat.get(..overlap);
            let name_prefix = name_lower.get(..overlap);
            let pat_suffix = pat.get(pat.len() - overlap..);
            let name_suffix = name_lower.get(name_lower.len() - overlap..);
            if let (Some(pp), Some(np)) = (pat_prefix, name_prefix) {
                if pp == np {
                    return true;
                }
            }
            if let (Some(ps), Some(ns)) = (pat_suffix, name_suffix) {
                if ps == ns {
                    return true;
                }
            }
        }
    }
    false
}
