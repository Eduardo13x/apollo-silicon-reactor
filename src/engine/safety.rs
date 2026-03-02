use std::collections::HashSet;

use crate::engine::types::{ActionBudgetState, RootAction, SafetyPolicy};

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
        // Spotlight stack (never touch).
        "Spotlight",
        "mds",
        "mds_stores",
        "mdworker",
        "mdworker_shared",
    ]
    .into_iter()
    .collect()
}

pub fn allowlisted_sysctls() -> HashSet<&'static str> {
    [
        "debug.lowpri_throttle_enabled",
        "net.inet.tcp.sendspace",
        "net.inet.tcp.recvspace",
        "net.inet.tcp.delayed_ack",
        "net.inet.tcp.min_iaj_win",
        "net.inet.tcp.win_scale_factor",
        "net.inet.tcp.autorcvbufmax",
        "net.inet.tcp.autosndbufmax",
        "vm.compressor_poll_interval",
        "vm.compressor_sample_min",
        "kern.maxvnodes",
        "kern.maxfiles",
        "kern.maxfilesperproc",
        "kern.ipc.somaxconn",
        "kern.ipc.maxsockbuf",
        "iogpu.wired_limit_mb",
        "debug.iogpu.wired_limit",
    ]
    .into_iter()
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
            _ => {}
        }

        budget.minute_actions += 1;
        out.push(action);
    }

    out
}
