use std::collections::HashSet;
use std::sync::OnceLock;

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
        // Spotlight stack (never touch — throttling these breaks app search and
        // prevents reindexing from ever completing).
        "Spotlight",
        "mds",
        "mds_stores",
        "mdworker",
        "mdworker_shared",
        "mdbulkimport",   // bulk initial import — throttle = reindex never finishes
        "mdwrite",        // writes Spotlight DB — throttle = index never updates
        "mdutil",         // Spotlight control tool — throttle = mdutil commands stall
        "corespotlightd",
        "spotlightknowledged",
        // Network / contacts / font — throttle causes app network timeouts,
        // Contacts/Settings hangs, and slow app launches with custom fonts.
        "nsurlsessiond",  // URL session broker — throttle → network timeouts in apps
        "contactsd",      // Contacts DB daemon — throttle → Settings/Contacts hang
        "fontworker",     // font catalog builder — throttle → slow app launch
        "imagent",        // iMessage agent — throttle → message send delays
        "spindump",       // crash diagnostic — short-lived, but throttle corrupts reports
        "ReportCrash",    // crash reporter — throttle → crash reports lost/corrupt
        // Display & audio rendering pipeline — freezing any of these causes frame drops,
        // animation jank, or audio glitches visible to the user. [WWDC 2021 "Tune CPU job
        // scheduling with QoS"; Apple TN2169 SIP/process policy]
        "Dock",           // Dock, Exposé, Mission Control animations
        "coreaudiod",     // Real-time audio I/O loop — latency-critical
        "mediaserverd",   // AVFoundation / CoreMedia pipeline
        "displaypolicyd", // Display power-state transitions — freeze → flicker
        "runningboardd",  // Process lifecycle manager — freeze → broken app launches
        "SystemUIServer", // Menu bar / status bar rendering
        "ControlCenter",  // Control center overlay rendering
        "Finder",         // File browser / open+save delegate — throttle → slow dialogs, nav hangs
        // Core services — freeze → Finder hangs, app associations break, no icon updates.
        // [Apple TN3113] coreservicesd manages Launch Services database, UTI resolution,
        // and app registration. SIGSTOP leaves the system unable to resolve file types.
        "coreservicesd",
        // Audio daemon — freeze → audio routing drops, Bluetooth headphones disconnect.
        // [CoreAudio documentation] audiod manages audio device policy and routing
        // decisions above coreaudiod's real-time mixing layer.
        "audiod",
        // Power management — freeze → sleep/wake state machine stalls, fans uncontrolled.
        // [IOKit Power Management] powerd manages assertion tracking, thermal policy,
        // and display sleep. SIGSTOP causes indefinite wake-lock or uncontrolled fan spin.
        "powerd",
        // AirDrop / Handoff / AirPlay sharing daemon. Freezing breaks device-to-device
        // sharing. Found in production data: gate_c freeze loop froze sharingd 173
        // times over 19 hours (2026-04-09 journal audit). sharingd respawns and
        // immediately gets re-frozen — infinite loop, AirDrop permanently broken.
        "sharingd",
        // Watchdog-adjacent daemons — SIGSTOP any of these causes kernel panic.
        // watchdogd monitors logd + WindowServer + opendirectoryd + configd checkins
        // every 120s. A single FreezeProcess on logd is enough to trigger the
        // "userspace watchdog timeout" kernel panic (observed 2026-04-14 in prod).
        "logd",
        "watchdogd",
        "syslogd",
        "OSLogService",
        // Bluetooth stack — freeze any of these = BT keyboard/trackpad dead on M1.
        // IOUserBluetoothSerialDriver seen 297 throttles in prod; BTLEServer 114.
        // No recovery without killing the freeze manually or rebooting.
        "bluetoothd",
        "BTLEServer",
        "bluetoothuserd",
        "IOUserBluetoothSerialDriver",
        // WiFi daemon — freeze = network drops immediately.
        "airportd",
        // DNS/Bonjour — freeze = all DNS resolution stops (affects every app, every URL).
        "mDNSResponder",
        // Filesystem events daemon — freeze = Time Machine, Spotlight, all FSEvents broken.
        // Observed 4 freezes in prod (2026-04-14 journal audit).
        "fseventsd",
        // NTP time sync — freeze = system clock drifts, TLS cert validation fails.
        "timed",
        // Kernel event bridge — routes disk-full, mount, unmount events to userspace.
        // 99 throttles observed in prod. Silent throttle = user never sees disk alerts.
        "KernelEventAgent",
        // Authentication daemon — freeze = TouchID / password prompts hang forever.
        "coreauthd",
        // User shells — throttling active zsh/bash blocks interactive commands.
        // 126 zsh throttles observed in prod; no GUI window so is_user_interactive misses it.
        "zsh",
        "bash",
        "fish",
        // AppKit open/save panel service — freeze = dialog hangs indefinitely.
        // 61 freezes observed in prod (2026-04-14 journal audit). User opens save
        // dialog → Apollo freezes the XPC service → dialog never responds → stuck.
        "openAndSavePanelService",
        // Audio driver helpers — throttle = audio dropouts, buffer underruns.
        // 122+40 throttles observed in prod. Includes DriverKit audio path.
        "Core-Audio-Driver-Service",
        "audio.SandboxHelper",
        "AudioComponentRegistrar",
        // Video decode XPC — throttle = video stutter in every app (YouTube, QuickTime).
        // 79 throttles observed in prod.
        "VTDecoderXPCService",
        // Camera / CMIO video capture — throttle = camera hangs mid-session.
        // 43 throttles in prod. Affects FaceTime, Zoom, OBS.
        "cmio.videodriverkithostextension",
        // Display brightness daemon — throttle = auto-brightness stuck, manual slider laggy.
        // 63 throttles in prod.
        "corebrightnessd",
        // Display extension — 33 FREEZES in prod. Freeze = display pipeline stall.
        "DisplaysExt",
        // Display timing daemon (IOMFB) — throttle = frame pacing issues, tearing.
        "IOMFB_bics_daemon",
        // WiFi DriverKit extension — freeze/throttle = WiFi drops at driver level.
        // 9 freezes + 16 throttles in prod (complements airportd above).
        "DriverKit-AppleBCMWLAN",
        // Karabiner keyboard remapper — throttle = key remapping stops mid-session.
        // 36 throttles in prod. User loses custom shortcuts silently.
        "Karabiner-DriverKit-VirtualHIDDevice",
        // Code signing integrity daemon — throttle = app launches hang at signature check.
        // 10 throttles in prod. amfid is on the critical path for every app launch.
        "amfid",
        // Apple Neural Engine user daemon — throttle = on-device ML inference stalls.
        "aneuserd",
        // login process — throttle = session management stalls.
        "login",
        // Sandbox policy daemon — throttle = sandboxed app operations hang.
        "sandboxd",
        // Security init daemon — throttle = security subsystem slow to respond.
        "secinitd",
        // UIKit system process — throttle = UI framework operations stall.
        "UIKitSystem",
        // Apollo own binary (all variants) — prevent self-freeze of old/new binary names.
        "apollo_optimizer",
        "apollo-optimizer",
    ]
    .into_iter()
    .collect()
}

/// Fast membership test against the hard-protected set (hot-path safe).
/// Uses `OnceLock` to initialise once and share across all subsequent calls,
/// avoiding the ~300 HashSet allocations per cycle that `protected_processes()`
/// would otherwise produce when called inside process-enumeration loops.
/// [GoF 1994] Flyweight pattern — share expensive read-only objects.
fn is_hard_protected(name: &str) -> bool {
    static CACHE: OnceLock<HashSet<&'static str>> = OnceLock::new();
    CACHE.get_or_init(protected_processes).contains(name)
}

/// Returns true if a process should be treated as a user-interactive app
/// — i.e. protection is foreground-conditional rather than unconditional.
///
/// Decision is purely behavioral (no hardcoded names):
/// - Has a GUI window: user can see and interact with it
/// - NOT a known system daemon (already in protected_processes)
/// - Has recent user interaction OR significant RSS (it's actively used)
///
/// Background instances of such apps are eligible for QoS hints and
/// CPU throttle but are NEVER frozen.
pub fn is_user_interactive_app(
    has_gui_window: bool,
    secs_since_user_interaction: u64,
    rss_bytes: u64,
    name: &str,
) -> bool {
    // System daemons never qualify regardless of heuristics.
    if is_hard_protected(name) {
        return false;
    }
    // Must have a GUI window — headless daemons are not user apps.
    if !has_gui_window {
        return false;
    }
    // Recent interaction OR large resident footprint (user launched and uses it).
    secs_since_user_interaction < 300 || rss_bytes > 100 * 1024 * 1024
}

/// Classification of how a process name should be treated by the protection system.
///
/// This enum captures the three-tier model established by `heuristic_critical_pids`
/// (the most correct site): hard/unconditional protection, foreground-conditional
/// protection, and no protection.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ProtectionLevel {
    /// Never throttle or freeze — OS essentials, infra services, policy-learned daemons.
    Unconditional,
    /// Protect only when the process is the foreground app; eligible for QoS/throttle
    /// when in background. Set by behavioral signals via `is_user_interactive_app()`.
    ConditionalForeground,
    /// No name-based protection — eligible for normal optimization decisions.
    Unprotected,
}

/// Classify how a process should be protected based on its name and policy lists.
///
/// This is the unified replacement for the five divergent protection checks scattered
/// across the codebase. It is purely name-based (no I/O, no async, no system calls)
/// and must be combined with a foreground check at the call site when the result is
/// `ConditionalForeground`.
///
/// # Parameters
/// - `name`: the process name as reported by the OS (case-sensitive, e.g. `"WindowServer"`)
/// - `hard_protected`: the result of `protected_processes()` — OS/system essentials
/// - `infra_protected`: the result of `infrastructure_processes()` — stateful services
/// - `policy_protected`: learned patterns from `LearnedPolicy::protected_patterns`
/// - `is_interactive`: result of `is_user_interactive_app()` for this process —
///   caller is responsible for evaluating behavioral signals before calling here.
///   Pass `false` if behavioral data is unavailable.
///
/// # Ordering
/// 1. Hard OS/system names → `Unconditional`
/// 2. Infrastructure services → `Unconditional`
/// 3. Policy-learned patterns (substring, case-insensitive) → `Unconditional`
/// 4. Behavioral interactive heuristic → `ConditionalForeground`
/// 5. Everything else → `Unprotected`
///
/// # Substring note
/// `hard_protected` and `infra_protected` use substring matching (consistent with
/// how Sites B–E use them: `name.contains(p)`). This catches bundled executables
/// like `com.docker.backend` matching `"docker"`.
pub fn classify_protection(
    name: &str,
    hard_protected: &HashSet<&'static str>,
    infra_protected: &HashSet<&'static str>,
    policy_protected: &[String],
    is_interactive: bool,
) -> ProtectionLevel {
    // Tier 1: OS/system essentials — substring to catch variants like
    // "com.apple.WindowServer" matching "WindowServer".
    if hard_protected.iter().any(|p| name.contains(p)) {
        return ProtectionLevel::Unconditional;
    }
    // Tier 2: Stateful infrastructure (Docker, Postgres, Redis, …).
    if infra_protected.iter().any(|p| name.contains(p)) {
        return ProtectionLevel::Unconditional;
    }
    // Tier 3: Policy-learned daemons (case-insensitive substring).
    {
        let name_lc = name.to_ascii_lowercase();
        if policy_protected
            .iter()
            .any(|p| name_lc.contains(p.to_ascii_lowercase().as_str()))
        {
            return ProtectionLevel::Unconditional;
        }
    }
    // Tier 4: Behavioral interactive apps — caller evaluated is_user_interactive_app().
    if is_interactive {
        return ProtectionLevel::ConditionalForeground;
    }
    ProtectionLevel::Unprotected
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
        // LLM inference servers — freezing kills token generation mid-stream,
        // causes client timeout, and post-thaw model reload spikes RAM.
        "llama-server",
        "ollama_llama_server",
        "ollama",
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
    // Production data (2026-04-06): clippy-driver frozen 7x — compiler toolchain
    // processes are dev-runtime critical; freeze during compilation → broken build.
    // [Saltzer & Kaashoek 2009] Fail-safe defaults: protect by default, not by exception.
    &[
        "node",
        "python",
        "java",
        "go",
        "nginx",
        "rustc",
        "clippy-driver",
    ]
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
                let left_ok =
                    abs_pos == 0 || !lower.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
                let right_ok =
                    end_pos == lower.len() || !lower.as_bytes()[end_pos].is_ascii_alphanumeric();
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
    let cpu_signal = if cpu_pct > 0.5 {
        1.0
    } else {
        cpu_pct as f64 / 0.5
    };

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
            assert!(
                !infra.contains(name),
                "'{}' must NOT be in infrastructure",
                name
            );
        }
    }

    #[test]
    fn dev_runtimes_in_own_set() {
        let runtimes = dev_runtime_patterns();
        for name in &["python", "node", "java", "go", "nginx"] {
            assert!(
                runtimes.contains(name),
                "'{}' must be in dev_runtime_patterns",
                name
            );
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
        assert!(
            score > 0.7,
            "active server should have high score: {}",
            score
        );
    }

    #[test]
    fn dormant_zombie_gets_low_score() {
        // Abandoned Python script: 0 CPU, 0 wakeups, no network, idle 4 hours, 6.8GB RSS
        // activity≈e^(-24)≈0, cost=1+1.5*sqrt(0.85)=2.38 → score≈0
        let score =
            behavioral_protection_score(0.0, 0.0, false, false, 14400, 6_800_000_000, RAM_8GB);
        assert!(
            score < 0.01,
            "6.8GB dormant zombie should have near-zero score: {}",
            score
        );
    }

    #[test]
    fn small_idle_process_gets_moderate_score() {
        // Small background Python (50MB): idle 10min but cheap to protect
        // recency=e^(-1)=0.37, cost=1+1.5*sqrt(50M/8G)=1.12 → score≈0.33
        let score = behavioral_protection_score(0.0, 0.0, false, false, 600, 50_000_000, RAM_8GB);
        assert!(
            score > 0.2,
            "small idle process should have moderate score: {}",
            score
        );
    }

    #[test]
    fn pressure_gate_protects_active_drops_dormant() {
        // Active 500MB server: activity=1.0, cost=1+1.5*sqrt(0.0625)=1.375 → score≈0.73
        let active_score =
            behavioral_protection_score(3.0, 5.0, true, false, 60, 500_000_000, RAM_8GB);
        // Dormant 6.8GB zombie: activity≈0, score≈0
        let dormant_score =
            behavioral_protection_score(0.0, 0.0, false, false, 7200, 6_800_000_000, RAM_8GB);
        let pressure = 0.67; // typical stressed system

        assert!(
            active_score >= pressure,
            "active process should survive at pressure {}: score={}",
            pressure,
            active_score
        );
        assert!(
            dormant_score < pressure,
            "dormant hog should lose protection at pressure {}: score={}",
            pressure,
            dormant_score
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
            small,
            big
        );
    }

    #[test]
    fn gui_process_well_protected() {
        // Process with a visible window — user can see it → strong protection
        // gui_signal=0.8, cost=1+1.5*sqrt(1G/8G)=1.53 → score≈0.52
        let score =
            behavioral_protection_score(0.0, 0.0, false, true, 3600, 1_000_000_000, RAM_8GB);
        assert!(
            score > 0.4,
            "GUI process should be well-protected: {}",
            score
        );
    }

    #[test]
    fn network_only_not_enough_at_medium_pressure() {
        // Idle daemon with network sockets but no other activity — should NOT be
        // immune at medium pressure. This is the CategoriesService / idle Python fix.
        // net_signal=0.3, idle=3600, cost≈1.0 → score≈0.3
        let score = behavioral_protection_score(0.0, 0.0, true, false, 3600, 50_000_000, RAM_8GB);
        let pressure = 0.5; // medium pressure
        assert!(
            score < pressure,
            "idle daemon with only network should lose protection at pressure {}: score={}",
            pressure,
            score
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

    // ── is_user_interactive_app() tests ──────────────────────────────────────

    const MB: u64 = 1024 * 1024;

    #[test]
    fn interactive_app_recent_interaction() {
        // GUI + interaction 60s ago → true regardless of RSS
        assert!(is_user_interactive_app(true, 60, 10 * MB, "MyApp"));
    }

    #[test]
    fn interactive_app_large_rss_idle() {
        // GUI + RSS > 100MB + idle 1h → still true (user launched, large footprint)
        assert!(is_user_interactive_app(true, 3600, 200 * MB, "BigApp"));
    }

    #[test]
    fn interactive_app_small_rss_long_idle_is_false() {
        // GUI + RSS ≤ 100MB + idle > 300s → false (looks abandoned)
        assert!(!is_user_interactive_app(true, 400, 50 * MB, "TinyApp"));
    }

    #[test]
    fn headless_daemon_never_interactive() {
        // No GUI → always false regardless of RSS or interaction
        assert!(!is_user_interactive_app(false, 0, 500 * MB, "coreaudiod"));
    }

    #[test]
    fn protected_process_never_interactive() {
        // kernel_task is in protected_processes() → false even with GUI
        assert!(!is_user_interactive_app(true, 0, 500 * MB, "kernel_task"));
        assert!(!is_user_interactive_app(true, 0, 500 * MB, "WindowServer"));
        assert!(!is_user_interactive_app(true, 0, 500 * MB, "launchd"));
    }

    #[test]
    fn interactive_app_at_interaction_boundary() {
        // secs == 299 (< 300) → true
        assert!(is_user_interactive_app(true, 299, 10 * MB, "BrowserApp"));
        // secs == 300 (= 300, not < 300) → false if RSS also small
        assert!(!is_user_interactive_app(true, 300, 10 * MB, "BrowserApp"));
    }

    #[test]
    fn interactive_app_at_rss_boundary() {
        // RSS == 100MB + 1 byte (> 100MB) → true even if idle
        assert!(is_user_interactive_app(
            true,
            9999,
            100 * MB + 1,
            "HeavyApp"
        ));
        // RSS == exactly 100MB (not >) → false if also idle
        assert!(!is_user_interactive_app(true, 9999, 100 * MB, "HeavyApp"));
    }

    #[test]
    fn unknown_app_name_follows_behavior() {
        // Arbitrary unknown name — decision is purely behavioral
        assert!(is_user_interactive_app(
            true,
            10,
            5 * MB,
            "com.example.MyRandomApp"
        ));
        assert!(!is_user_interactive_app(
            false,
            10,
            5 * MB,
            "com.example.MyRandomApp"
        ));
    }

    // ── classify_protection() tests ────────────────────────────────────────

    fn test_policy() -> Vec<String> {
        vec!["mypostgres-wrapper".to_string(), "custom-db".to_string()]
    }

    #[test]
    fn classify_kernel_task_is_unconditional() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        assert_eq!(
            classify_protection("kernel_task", &hard, &infra, &[], false),
            ProtectionLevel::Unconditional,
            "kernel_task must be Unconditional"
        );
    }

    #[test]
    fn classify_windowserver_is_unconditional() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        assert_eq!(
            classify_protection("WindowServer", &hard, &infra, &[], false),
            ProtectionLevel::Unconditional
        );
    }

    #[test]
    fn classify_docker_infra_is_unconditional() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        assert_eq!(
            classify_protection("com.docker.backend", &hard, &infra, &[], false),
            ProtectionLevel::Unconditional,
            "com.docker.backend must be Unconditional via infra substring match"
        );
    }

    #[test]
    fn classify_postgres_is_unconditional() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        assert_eq!(
            classify_protection("postgres", &hard, &infra, &[], false),
            ProtectionLevel::Unconditional
        );
    }

    #[test]
    fn classify_policy_protected_is_unconditional() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        let policy = test_policy();
        assert_eq!(
            classify_protection("mypostgres-wrapper", &hard, &infra, &policy, false),
            ProtectionLevel::Unconditional,
            "policy-protected process must be Unconditional"
        );
    }

    #[test]
    fn classify_policy_protected_case_insensitive() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        let policy = vec!["Custom-DB".to_string()];
        // Process name in lowercase
        assert_eq!(
            classify_protection("custom-db", &hard, &infra, &policy, false),
            ProtectionLevel::Unconditional
        );
        // Process name uppercase, pattern lowercase in policy
        let policy2 = vec!["custom-db".to_string()];
        assert_eq!(
            classify_protection("Custom-DB", &hard, &infra, &policy2, false),
            ProtectionLevel::Unconditional
        );
    }

    #[test]
    fn classify_interactive_app_foreground_conditional() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        // is_interactive=true (caller evaluated is_user_interactive_app)
        assert_eq!(
            classify_protection("Brave Browser", &hard, &infra, &[], true),
            ProtectionLevel::ConditionalForeground,
            "user interactive app must be ConditionalForeground"
        );
    }

    #[test]
    fn classify_unknown_process_is_unprotected() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        assert_eq!(
            classify_protection("com.example.random-daemon", &hard, &infra, &[], false),
            ProtectionLevel::Unprotected
        );
    }

    #[test]
    fn classify_substring_false_positive_dropbox_with_box() {
        // "Dropbox" contains "box" — but "box" is NOT in any protection list, so
        // it falls through to Unprotected (or ConditionalForeground if interactive).
        // This verifies we only match against actual protected patterns, not arbitrary
        // short strings.
        let hard = protected_processes();
        let infra = infrastructure_processes();
        assert_eq!(
            classify_protection("Dropbox", &hard, &infra, &[], false),
            ProtectionLevel::Unprotected,
            "Dropbox must not be Unconditional — 'box' is not a protected pattern"
        );
        // If Dropbox has a GUI window and recent interaction, caller sets is_interactive=true
        assert_eq!(
            classify_protection("Dropbox", &hard, &infra, &[], true),
            ProtectionLevel::ConditionalForeground,
            "Dropbox with GUI should be ConditionalForeground"
        );
    }

    #[test]
    fn classify_interactive_overridden_by_hard_protection() {
        // Even if is_interactive=true, a hard-protected name wins (Tier 1 first).
        let hard = protected_processes();
        let infra = infrastructure_processes();
        assert_eq!(
            classify_protection("kernel_task", &hard, &infra, &[], true),
            ProtectionLevel::Unconditional,
            "hard protection must take precedence over is_interactive"
        );
    }

    /// Verify critical macOS system daemons added for audio/power/services safety.
    /// [Apple TN3113] coreservicesd, [CoreAudio] audiod, [IOKit PM] powerd.
    #[test]
    fn protected_includes_coreservicesd_audiod_powerd() {
        let protected = protected_processes();
        assert!(protected.contains("coreservicesd"),
            "coreservicesd must be protected — freeze breaks Finder, app associations, UTI resolution");
        assert!(protected.contains("audiod"),
            "audiod must be protected — freeze drops audio routing, disconnects Bluetooth headphones");
        assert!(
            protected.contains("powerd"),
            "powerd must be protected — freeze stalls sleep/wake state machine, uncontrolled fans"
        );
    }

    /// Protected processes must include ALL rendering pipeline components.
    /// Freezing any of these causes visible user-facing degradation.
    #[test]
    fn protected_covers_rendering_pipeline() {
        let protected = protected_processes();
        let pipeline = [
            "WindowServer",
            "Dock",
            "SystemUIServer",
            "ControlCenter",
            "coreaudiod",
            "mediaserverd",
            "displaypolicyd",
        ];
        for name in &pipeline {
            assert!(
                protected.contains(name),
                "'{}' must be in rendering pipeline protection",
                name
            );
        }
    }

    /// The hard-protected fast path (OnceLock) must agree with the full set.
    #[test]
    fn is_hard_protected_matches_full_set() {
        let full = protected_processes();
        for &name in full.iter() {
            assert!(
                is_hard_protected(name),
                "is_hard_protected('{}') must return true",
                name
            );
        }
        // Negative check.
        assert!(!is_hard_protected("random_process_xyz"));
    }
}
