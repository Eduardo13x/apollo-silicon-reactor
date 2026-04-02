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
        // Apollo itself — self-SIGSTOP is instant deadlock.
        "apollo-optimizerd",
        "apollo-optimizerctl",
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

/// Infrastructure processes: stateful services whose freeze/kill would cascade
/// to dependent processes (data loss, broken connections, orphaned containers).
/// Unconditionally protected — same as `protected_processes()` but for dev infra.
pub fn infrastructure_processes() -> HashSet<&'static str> {
    [
        // Containers / VMs — freezing kills all containers inside
        "podman",
        "qemu-system",
        "colima",
        "lima",
        "docker",
        "com.docker",
        "vpnkit",
        "hyperkit",
        // Databases / caches — freezing causes data corruption or client timeouts
        "postgres",
        "mysqld",
        "mariadbd",
        "redis-server",
        "mongod",
        // Queues / streaming — freezing loses in-flight messages
        "kafka",
        "zookeeper",
        "rabbitmq",
    ]
    .into_iter()
    .collect()
}

/// Dev runtime patterns: processes that MAY be doing useful work (web server,
/// build tool, etc.) or MAY be abandoned zombies (forgotten script leaking 6GB).
///
/// Unlike infrastructure, these earn protection dynamically through behavioral
/// signals — not by name alone. See `behavioral_protection_score()`.
///
/// Note: short patterns like "go" would false-positive on CategoriesService,
/// mongod, etc. via substring match. We use a dedicated `matches_dev_runtime()`
/// function with word-boundary awareness instead of raw `.contains()`.
pub fn dev_runtime_patterns() -> &'static [&'static str] {
    &["node", "python", "java", "go", "nginx"]
}

/// Check if a process name matches a dev runtime pattern, with word-boundary
/// awareness for short patterns (≤3 chars). "go" must appear at a word boundary
/// (start/end of string or preceded/followed by non-alphanumeric), preventing
/// false matches like CategoriesService, mongod, Cargo, etc.
///
/// Case-insensitive: macOS shows "Python" (capital P) in process names.
pub fn matches_dev_runtime(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    for &pat in dev_runtime_patterns() {
        if pat.len() <= 3 {
            // Short pattern: require word boundaries.
            let mut start = 0;
            while let Some(pos) = lower[start..].find(pat) {
                let abs_pos = start + pos;
                let end_pos = abs_pos + pat.len();
                let left_ok = abs_pos == 0
                    || !lower.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
                let right_ok = end_pos == lower.len()
                    || !lower.as_bytes()[end_pos].is_ascii_alphanumeric();
                if left_ok && right_ok {
                    return true;
                }
                start = abs_pos + 1;
                if start >= lower.len() {
                    break;
                }
            }
        } else {
            // Long pattern (≥4 chars): substring match is safe.
            if lower.contains(pat) {
                return true;
            }
        }
    }
    false
}

/// Backward-compatible: returns infrastructure ∪ dev_runtime patterns.
/// Callers that need the behavioral gate should use the split functions instead.
pub fn critical_background_processes() -> HashSet<&'static str> {
    let mut all = infrastructure_processes();
    all.extend(dev_runtime_patterns().iter().copied());
    all
}

/// Behavioral protection score: measures a process's demonstrated resource
/// demand relative to its memory cost.
///
/// Based on three principles:
/// - **Android LMK**: protection earned by activity, not name. A dormant process
///   loses its dev-workload exemption regardless of binary name.
/// - **TMO (Facebook, ASPLOS'22)**: reclaim aggressiveness proportional to memory
///   pressure. The returned score is compared against system pressure — as pressure
///   rises, the activity bar for maintaining protection rises with it.
/// - **DAMOS (SeongJae Park, arXiv:2303.05919)**: actions decided by access pattern,
///   not identity. CPU, IPC, network, and recency are all access-pattern signals.
///
/// Returns [0.0, 1.0]. Compare against `system_pressure` to decide protection:
///   score >= pressure → protected (process justifies its memory usage)
///   score <  pressure → unprotected (memory cost exceeds demonstrated demand)
///
/// At pressure 0.3 (calm): almost everything stays protected.
/// At pressure 0.7 (stressed): only active processes survive.
pub fn behavioral_protection_score(
    cpu_pct: f32,
    wakeups_per_sec: f32,
    has_network: bool,
    has_gui_window: bool,
    secs_idle: u64,
    rss_bytes: u64,
    total_ram_bytes: u64,
) -> f64 {
    // ── Activity signals ────────────────────────────────────────────────
    // Each measures a different dimension of "this process is alive."
    // Binary max() aggregation: ANY sign of life = active (Android LMK model).
    // No weights — a web server with 0 CPU but active network is just as alive
    // as a compute job with high CPU but no network.

    // CPU: kernel-measured user+system time. Nonzero = executing code.
    let cpu_signal = if cpu_pct > 0.5 { 1.0 } else { cpu_pct as f64 / 0.5 };

    // IPC: Mach message wakeups. Nonzero = receiving requests / responding.
    let ipc_signal = if wakeups_per_sec > 1.0 {
        1.0
    } else {
        wakeups_per_sec as f64
    };

    // Network: active sockets. Weak hint — socket existence ≠ active I/O.
    // Cumulative mach_msgs inflate has_network for long-lived daemons,
    // so this is circumstantial evidence, not proof of activity.
    // Reduced from 0.30 → 0.15: always-listening background daemons (suggestd,
    // cloudd, imagent) have persistent sockets but are not meaningfully active.
    let net_signal = if has_network { 0.15 } else { 0.0 };

    // GUI: visible window. User can see it → strong protection.
    let gui_signal = if has_gui_window { 0.8 } else { 0.0 };

    // Recency: exponential decay over 10 minutes (600s).
    // At 0s → 1.0, at 300s → 0.61, at 600s → 0.37, at 1200s → 0.14.
    // e^(-t/600) is the natural decay — no threshold, just diminishing relevance.
    let recency_signal = (-1.0 * secs_idle as f64 / 600.0).exp();

    // Aggregate: max of all (any sign of life suffices).
    let activity = cpu_signal
        .max(ipc_signal)
        .max(net_signal)
        .max(gui_signal)
        .max(recency_signal);

    // ── Memory cost (Darwinian resource pressure) ──────────────────────
    // Ecological fitness: foraging_success / resource_demand.
    // Factor: 1 + k·√(rss/ram).  k=1.5 (Google Borg sub-linear cost model):
    //   100MB/8GB → cost 1.17 (barely penalized)
    //   500MB     → cost 1.38 (moderate; active servers survive)
    //   4GB       → cost 2.06 (heavy; must be very active to survive)
    // Denominator ≥ 1.0 so score is naturally bounded in [0, 1].
    let mem_fraction = rss_bytes as f64 / total_ram_bytes.max(1) as f64;
    let cost = 1.0 + 1.5 * mem_fraction.sqrt();

    // ── Fitness score ───────────────────────────────────────────────────
    // Compare against system_pressure for the protection gate:
    //   score ≥ pressure → survives (Darwinian selection).
    // No clamp needed — activity ∈ [0,1] and cost ≥ 1.0 guarantees [0,1].
    activity / cost
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

#[cfg(test)]
mod tests {
    use super::*;

    const RAM_8GB: u64 = 8 * 1024 * 1024 * 1024;

    // ── List split tests ────────────────────────────────────────────────

    #[test]
    fn infrastructure_has_stateful_services() {
        let infra = infrastructure_processes();
        for name in &["docker", "postgres", "redis-server", "kafka", "qemu-system"] {
            assert!(infra.contains(name), "'{}' must be in infrastructure", name);
        }
    }

    // ── Word-boundary matching tests ──────────────────────────────────

    #[test]
    fn go_matches_go_binary() {
        assert!(matches_dev_runtime("go"));
        assert!(matches_dev_runtime("go-build"));
        assert!(matches_dev_runtime("/usr/local/bin/go"));
    }

    #[test]
    fn go_does_not_match_substrings() {
        assert!(!matches_dev_runtime("CategoriesService"));
        assert!(!matches_dev_runtime("mongod"));
        assert!(!matches_dev_runtime("Cargo"));
        assert!(!matches_dev_runtime("google-chrome"));
        assert!(!matches_dev_runtime("Diego"));
    }

    #[test]
    fn python_matches_long_pattern() {
        assert!(matches_dev_runtime("python"));
        assert!(matches_dev_runtime("python3.13"));
        assert!(matches_dev_runtime("/opt/homebrew/bin/python3"));
        // Case insensitive: macOS shows "Python" with capital P.
        assert!(matches_dev_runtime("Python"));
    }

    #[test]
    fn node_matches_correctly() {
        assert!(matches_dev_runtime("node"));
        assert!(matches_dev_runtime("/usr/local/bin/node"));
        // "node" is 4 chars → uses substring match (≥4), so this matches:
        assert!(matches_dev_runtime("nodejs"));
    }

    #[test]
    fn dev_runtimes_not_in_infrastructure() {
        let infra = infrastructure_processes();
        for name in &["python", "node", "java", "go", "nginx"] {
            assert!(!infra.contains(name), "'{}' must NOT be in infrastructure", name);
        }
    }

    #[test]
    fn dev_runtimes_in_own_set() {
        let runtimes = dev_runtime_patterns();
        for name in &["python", "node", "java", "go", "nginx"] {
            assert!(runtimes.contains(name), "'{}' must be in dev_runtime_patterns", name);
        }
    }

    #[test]
    fn backward_compat_union_has_all() {
        let all = critical_background_processes();
        let infra = infrastructure_processes();
        let runtimes = dev_runtime_patterns();
        for p in infra.iter().chain(runtimes.iter()) {
            assert!(all.contains(p), "'{}' must be in backward-compat union", p);
        }
    }

    // ── Behavioral protection score tests ───────────────────────────────

    #[test]
    fn active_server_gets_high_score() {
        // Python web server: CPU active, wakeups, network, recent interaction
        // activity=1.0, cost=1+1.5*sqrt(200M/8G)=1.17 → score≈0.76
        let score = behavioral_protection_score(5.0, 10.0, true, false, 30, 200_000_000, RAM_8GB);
        assert!(score > 0.7, "active server should have high score: {}", score);
    }

    #[test]
    fn dormant_zombie_gets_low_score() {
        // Abandoned Python script: 0 CPU, 0 wakeups, no network, idle 4 hours, 6.8GB RSS
        // activity≈e^(-24)≈0, cost=1+1.5*sqrt(0.85)=2.38 → score≈0
        let score = behavioral_protection_score(
            0.0, 0.0, false, false, 14400, 6_800_000_000, RAM_8GB,
        );
        assert!(score < 0.01, "6.8GB dormant zombie should have near-zero score: {}", score);
    }

    #[test]
    fn small_idle_process_gets_moderate_score() {
        // Small background Python (50MB): idle 10min but cheap to protect
        // recency=e^(-1)=0.37, cost=1+1.5*sqrt(50M/8G)=1.12 → score≈0.33
        let score = behavioral_protection_score(0.0, 0.0, false, false, 600, 50_000_000, RAM_8GB);
        assert!(score > 0.2, "small idle process should have moderate score: {}", score);
    }

    #[test]
    fn pressure_gate_protects_active_drops_dormant() {
        // Active 500MB server: activity=1.0, cost=1+1.5*sqrt(0.0625)=1.375 → score≈0.73
        let active_score = behavioral_protection_score(
            3.0, 5.0, true, false, 60, 500_000_000, RAM_8GB,
        );
        // Dormant 6.8GB zombie: activity≈0, score≈0
        let dormant_score = behavioral_protection_score(
            0.0, 0.0, false, false, 7200, 6_800_000_000, RAM_8GB,
        );
        let pressure = 0.67; // typical stressed system

        assert!(
            active_score >= pressure,
            "active process should survive at pressure {}: score={}",
            pressure, active_score
        );
        assert!(
            dormant_score < pressure,
            "dormant hog should lose protection at pressure {}: score={}",
            pressure, dormant_score
        );
    }

    #[test]
    fn score_scales_with_memory_cost() {
        // Same activity, different RSS — bigger process needs more activity
        let small = behavioral_protection_score(0.0, 0.5, false, false, 300, 100_000_000, RAM_8GB);
        let big = behavioral_protection_score(0.0, 0.5, false, false, 300, 4_000_000_000, RAM_8GB);
        assert!(
            small > big,
            "smaller process should have higher score at same activity: small={} big={}",
            small, big
        );
    }

    #[test]
    fn gui_process_well_protected() {
        // Process with a visible window — user can see it → strong protection
        // gui_signal=0.8, cost=1+1.5*sqrt(1G/8G)=1.53 → score≈0.52
        let score = behavioral_protection_score(0.0, 0.0, false, true, 3600, 1_000_000_000, RAM_8GB);
        assert!(score > 0.4, "GUI process should be well-protected: {}", score);
    }

    #[test]
    fn network_only_not_enough_at_medium_pressure() {
        // Idle daemon with network sockets but no other activity — should NOT be
        // immune at medium pressure. This is the CategoriesService / idle Python fix.
        // net_signal=0.3, idle=3600, cost≈1.0 → score≈0.3
        let score = behavioral_protection_score(
            0.0, 0.0, true, false, 3600, 50_000_000, RAM_8GB,
        );
        let pressure = 0.5; // medium pressure
        assert!(
            score < pressure,
            "idle daemon with only network should lose protection at pressure {}: score={}",
            pressure, score
        );
    }

    #[test]
    fn zero_rss_does_not_inflate_score() {
        // Process with rss=0 (tiny process or measurement gap) — score should NOT
        // be inflated to 1.0. With new formula: cost=1+0=1 → score=activity.
        let score = behavioral_protection_score(0.0, 0.0, true, false, 3600, 0, RAM_8GB);
        assert!(
            score < 0.4,
            "zero-RSS process with only network should not be over-protected: {}",
            score,
        );
    }
}
