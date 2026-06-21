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
        "mdbulkimport", // bulk initial import — throttle = reindex never finishes
        "mdwrite",      // writes Spotlight DB — throttle = index never updates
        "mdutil",       // Spotlight control tool — throttle = mdutil commands stall
        "corespotlightd",
        "spotlightknowledged",
        // Network / contacts / font — throttle causes app network timeouts,
        // Contacts/Settings hangs, and slow app launches with custom fonts.
        "nsurlsessiond", // URL session broker — throttle → network timeouts in apps
        "contactsd",     // Contacts DB daemon — throttle → Settings/Contacts hang
        "fontworker",    // font catalog builder — throttle → slow app launch
        "imagent",       // iMessage agent — throttle → message send delays
        "spindump",      // crash diagnostic — short-lived, but throttle corrupts reports
        "ReportCrash",   // crash reporter — throttle → crash reports lost/corrupt
        // Display & audio rendering pipeline — freezing any of these causes frame drops,
        // animation jank, or audio glitches visible to the user. [WWDC 2021 "Tune CPU job
        // scheduling with QoS"; Apple TN2169 SIP/process policy]
        "Dock",           // Dock, Exposé, Mission Control animations
        "coreaudiod",     // Real-time audio I/O loop — latency-critical
        "mediaserverd",   // AVFoundation / CoreMedia pipeline
        "mediaplaybackd", // 2026-06-21 (P3): video playback daemon — acting on it
        // during 4K = decode stall / frame drop. Was only PID-protected via the
        // apple_owned path, which the name-keyed SetMemorystatus guard never hits.
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
        // Apple GPU/Metal shader-compiler server. Apollo throttled it 69× at
        // 4.2% effectiveness (2026-06 audit `weight-futile`) — pure wasted
        // budget, and throttling the shader compiler risks graphics jank.
        // It IS Apple-owned but slipped the is_protected_pid path; pin it by
        // name. [2026-06-18 audit finding]
        "CVMServer",
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
        // `log` CLI tool — ephemeral (<3s) subprocess spawned by apollo's own
        // system_log_ingester (src/engine/system_log_ingester.rs). Night-mode
        // heuristic was targeting these; by execute_actions time the PID had
        // recycled/died, producing 4052 consecutive `skip:pid-recycled:log`
        // entries with 0% throttle success (prod observation 2026-04-16, 11h
        // window). Self-inflicted cascade [Nygard 2018 Ch.4].
        "log",
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
        // Media players — freeze = audio/video playback drops immediately.
        // VLC runs its own audio thread decoupled from the GUI window; background
        // playback (e.g. music while coding) has no foreground window so
        // is_user_interactive_app() misses it → freeze kills audio silently.
        "VLC",
        // IDE language servers — freeze stalls code completion/diagnostics and
        // can cascade into IDE process death (Antigravity, VS Code). They have
        // no GUI window of their own, so is_user_interactive_app() misses them.
        "language_server_macos_arm",
        "language_server_linux_x64",
        "rust-analyzer",
        "gopls",
        "pyright",
        "typescript-language-server",
        "tsserver",
        "clangd",
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

/// Fast membership test against the infrastructure-protected set (hot-path safe).
/// Uses `OnceLock` + Aho-Corasick automaton to do multi-pattern substring scan
/// in a single pass over `name`, instead of N iter() × name.contains() loops.
/// Substring semantics preserved — catches bundled executables like
/// `com.docker.backend`. Case-sensitive (matches prior behavior).
fn is_infra_protected(name: &str) -> bool {
    static MATCHER: OnceLock<aho_corasick::AhoCorasick> = OnceLock::new();
    MATCHER
        .get_or_init(|| {
            let patterns: Vec<&'static str> = infrastructure_processes().into_iter().collect();
            aho_corasick::AhoCorasick::new(patterns).expect("infra patterns build")
        })
        .is_match(name)
}

/// Fast substring membership test against the hard-protected set.
/// Differs from `is_hard_protected` (exact match) — this matches as substring
/// so `com.apple.WindowServer` matches pattern `WindowServer`. Used by
/// `decide_actions` and `classify_protection` Tier 1.
pub fn hard_protected_contains(name: &str) -> bool {
    static MATCHER: OnceLock<aho_corasick::AhoCorasick> = OnceLock::new();
    MATCHER
        .get_or_init(|| {
            let patterns: Vec<&'static str> = protected_processes().into_iter().collect();
            aho_corasick::AhoCorasick::new(patterns).expect("hard patterns build")
        })
        .is_match(name)
}

/// Single source of truth for the BOOST veto. Apollo must never BOOST these
/// — Chromium IPC contract + structural-low-effectiveness misclassification trap.
///
/// Returns true when `name` is either hard-protected (kernel/WindowServer/…) OR
/// a Chromium family-root (Brave / Google Chrome / Microsoft Edge / …). Production
/// matches Brave via `match_engine::is_family_root`, NOT `hard_protected_contains`,
/// so the latter alone leaves the Brave Boost guard as dead code (FIX-1).
pub fn is_boost_forbidden(name: &str) -> bool {
    hard_protected_contains(name) || crate::engine::match_engine::is_family_root(name)
}

/// Fast substring membership test against the softly-protected set (LLM
/// inference servers). Mirrors `hard_protected_contains` — single AC walk
/// instead of HashSet build + N×contains. Frozen only when the alternative
/// is OOM reboot (decide_actions checks `survival_mode` before honoring).
pub fn softly_protected_contains(name: &str) -> bool {
    static MATCHER: OnceLock<aho_corasick::AhoCorasick> = OnceLock::new();
    MATCHER
        .get_or_init(|| {
            let patterns: Vec<&'static str> = softly_protected_processes().into_iter().collect();
            aho_corasick::AhoCorasick::new(patterns).expect("softly patterns build")
        })
        .is_match(name)
}

/// Fast critical-background membership test (infra ∪ dev_runtime) — replaces
/// the slow `critical_background_processes().iter().any(|p| name.contains(p))`
/// pattern at hot decide_actions sites.
pub fn is_critical_background_name(name: &str) -> bool {
    is_infra_protected(name) || matches_dev_runtime(name)
}

// ── B.6 Chromium Non-Invasive Containment ─────────────────────────────────────

/// Process intervention class — determines which actions Apollo may take
/// against a process. Single source of truth replacing scattered `contains("Brave")`
/// and `is_family_root` checks scattered across action decision code.
///
/// # Memory Safety Notes
/// Chromium-family processes are treated as memory-toxic but alive: Apollo
/// must not interrupt their IPC contract (SIGSTOP breaks Brave WebContents
/// async communication — observed 2026-04-14 in prod). The strategy is
/// non-invasive containment: observe, attribute, demote, hint — never freeze,
/// boost, or hard-throttle.
///
/// See: B.6 Chromium Non-Invasive Containment (CLAUDE.md)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessInterventionClass {
    /// Standard process — Apollo may apply any permitted action.
    Normal,
    /// Chromium family (Brave/Chrome/Chromium/Electron/Safari/Firefox).
    /// SIGSTOP blocks the async IPC message loop. Apollo must NOT freeze,
    /// boost, or hard-throttle these — only non-invasive containment.
    ChromiumFamily,
    /// OS/system essentials — kernel, WindowServer, launchd, etc.
    /// No interventions whatsoever.
    ProtectedSystem,
    /// Media-critical: audio/video playback. Hard-throttle causes dropouts.
    MediaCritical,
    /// Build tools: rustc, cargo, node, etc. Throttle during active builds.
    BuildTool,
}

impl ProcessInterventionClass {
    /// Classify a process by name into its intervention class.
    pub fn for_name(name: &str) -> Self {
        if is_hard_protected(name) {
            return ProcessInterventionClass::ProtectedSystem;
        }
        if crate::engine::match_engine::is_family_root(name) {
            return ProcessInterventionClass::ChromiumFamily;
        }
        if matches_dev_runtime(name) {
            return ProcessInterventionClass::BuildTool;
        }
        ProcessInterventionClass::Normal
    }
}

/// What Apollo is allowed to do to a process in a given intervention class.
/// All fields are `bool` for clarity at callsites — `true` = allowed, `false` = forbidden.
///
/// # Example
/// ```
/// use apollo_engine::engine::safety::{InterventionPolicy, ProcessInterventionClass};
/// let policy = InterventionPolicy::for_class(ProcessInterventionClass::ChromiumFamily);
/// assert!(!policy.allow_freeze);
/// assert!(!policy.allow_boost);
/// assert!(policy.allow_ecore_demote);
/// assert!(policy.allow_purge_hint);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct InterventionPolicy {
    pub allow_freeze: bool,
    pub allow_boost: bool,
    pub allow_hard_throttle: bool,
    pub allow_ecore_demote: bool,
    pub allow_purge_hint: bool,
    pub allow_priority_demote: bool,
    pub allow_memory_budget_pressure: bool,
}

impl InterventionPolicy {
    /// Return the policy for a given intervention class.
    pub fn for_class(class: ProcessInterventionClass) -> Self {
        match class {
            ProcessInterventionClass::Normal => Self {
                allow_freeze: true,
                allow_boost: true,
                allow_hard_throttle: true,
                allow_ecore_demote: true,
                allow_purge_hint: true,
                allow_priority_demote: true,
                allow_memory_budget_pressure: true,
            },
            ProcessInterventionClass::ChromiumFamily => Self {
                allow_freeze: false,
                allow_boost: false,
                allow_hard_throttle: false,
                allow_ecore_demote: true,
                allow_purge_hint: true,
                allow_priority_demote: true,
                allow_memory_budget_pressure: true,
            },
            ProcessInterventionClass::ProtectedSystem => Self {
                allow_freeze: false,
                allow_boost: false,
                allow_hard_throttle: false,
                allow_ecore_demote: false,
                allow_purge_hint: false,
                allow_priority_demote: false,
                allow_memory_budget_pressure: false,
            },
            ProcessInterventionClass::MediaCritical => Self {
                allow_freeze: false,
                allow_boost: true,
                allow_hard_throttle: false,
                allow_ecore_demote: true,
                allow_purge_hint: true,
                allow_priority_demote: true,
                allow_memory_budget_pressure: true,
            },
            ProcessInterventionClass::BuildTool => Self {
                allow_freeze: true,
                allow_boost: true,
                allow_hard_throttle: true,
                allow_ecore_demote: true,
                allow_purge_hint: true,
                allow_priority_demote: true,
                allow_memory_budget_pressure: true,
            },
        }
    }

    /// Shortcut: get the policy directly from a process name.
    pub fn for_name(name: &str) -> Self {
        Self::for_class(ProcessInterventionClass::for_name(name))
    }
}

/// Returns true if the named process is a Chromium family member.
/// This is the single point of truth for B.6 Chromium non-invasive containment —
/// freeze/throttle/boost code should call this, not `is_family_root` directly.
///
/// Use `InterventionPolicy::for_name(name).allow_freeze` when you need the
/// full policy; use this only when you need a fast boolean skip.
pub fn is_chromium_family(name: &str) -> bool {
    crate::engine::match_engine::is_family_root(name)
}

/// Returns true if Apollo MAY freeze this process.
/// Shortcut for `InterventionPolicy::for_name(name).allow_freeze`.
pub fn can_freeze(name: &str) -> bool {
    InterventionPolicy::for_name(name).allow_freeze
}

/// Returns true if Apollo MAY boost this process.
/// Shortcut for `InterventionPolicy::for_name(name).allow_boost`.
pub fn can_boost(name: &str) -> bool {
    InterventionPolicy::for_name(name).allow_boost
}

/// Returns true if Apollo MAY hard-throttle this process.
/// Shortcut for `InterventionPolicy::for_name(name).allow_hard_throttle`.
pub fn can_hard_throttle(name: &str) -> bool {
    InterventionPolicy::for_name(name).allow_hard_throttle
}

/// Returns true if Apollo MAY apply E-core demotion to this process.
pub fn can_ecore_demote(name: &str) -> bool {
    InterventionPolicy::for_name(name).allow_ecore_demote
}

/// Returns true if Apollo MAY send purgeable memory hints to this process.
pub fn can_purge_hint(name: &str) -> bool {
    InterventionPolicy::for_name(name).allow_purge_hint
}

// ── macOS Cooperation Layer ────────────────────────────────────────────────────

/// How aggressively Apollo should act, based on what macOS is already doing.
///
/// Apollo is a **cooperator**, not a director. It observes what macOS kernel
/// is already handling (compressor, jetsam, purgeable memory) and supplements
/// with hints — it does not override or duplicate what the kernel is doing.
///
/// Design principle: when macOS is already acting, Apollo steps back and
/// helps via hints rather than taking direct intervention actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacOSCooperationMode {
    /// macOS memory subsystem is calm. Apollo may act normally.
    Normal,
    /// macOS compressor is actively compressing pages. Apollo should step back
    /// from freeze/throttle and focus on purgeable hints and jetsam tier hints.
    /// The compressor is already handling memory compression — Apollo would be
    /// duplicating work and causing extra latency.
    CompressorActive,
    /// macOS is actively swapping to disk. Apollo should be conservative —
    /// only send jetsam tier hints and purgeable hints. Heavy interventions
    /// will only add more I/O pressure to an already stressed system.
    SwapActive,
    /// macOS jetsam has fired (processes killed). Apollo should be in
    /// observation mode — the kernel just made hard decisions, Apollo should
    /// not override them. Just track and learn.
    JetsamFired,
}

impl MacOSCooperationMode {
    /// Determine the cooperation mode from system pressure signals.
    ///
    /// Priority: JetsamFired > CompressorActive > SwapActive > Normal
    ///
    /// When jetsam has fired, Apollo must not act — the kernel just made
    /// hard decisions. When compressor is active, Apollo should step back
    /// from freeze/throttle since macOS is already compressing.
    pub fn from_pressure_signals(
        compressor_pressure: f64,
        swap_delta_bytes_per_sec: f64,
        jetsam_kill_count: u32,
    ) -> Self {
        if jetsam_kill_count > 0 {
            return MacOSCooperationMode::JetsamFired;
        }
        if compressor_pressure > 0.50 {
            return MacOSCooperationMode::CompressorActive;
        }
        if swap_delta_bytes_per_sec > 524_288.0 {
            return MacOSCooperationMode::SwapActive;
        }
        MacOSCooperationMode::Normal
    }

    /// Returns true if Apollo should step back from direct interventions
    /// (freeze, hard-throttle) and focus only on cooperative hints.
    ///
    /// When macOS is already handling memory pressure, Apollo adding more
    /// interventions creates contention and can make things worse.
    pub fn should_step_back(self) -> bool {
        matches!(
            self,
            MacOSCooperationMode::CompressorActive
                | MacOSCooperationMode::SwapActive
                | MacOSCooperationMode::JetsamFired
        )
    }

    /// Returns true if Apollo should emit jetsam tier hints (mark process
    /// as sacrificable) rather than direct interventions.
    pub fn should_emit_jetsam_hints(self) -> bool {
        matches!(
            self,
            MacOSCooperationMode::CompressorActive
                | MacOSCooperationMode::SwapActive
                | MacOSCooperationMode::JetsamFired
        )
    }
}

/// Returns true if Apollo should step back from direct interventions
/// (freeze, hard-throttle) because macOS is already handling memory pressure.
///
/// When the compressor is active or macOS is swapping, Apollo should not
/// add more pressure — just send hints and let the kernel manage.
pub fn apollo_should_step_back(compressor_pressure: f64, swap_delta_bps: f64) -> bool {
    MacOSCooperationMode::from_pressure_signals(compressor_pressure, swap_delta_bps, 0)
        .should_step_back()
}

/// Single truth point for name-based process protection.
///
/// Returns `true` if the process should NEVER receive a Freeze, Throttle, or Kill
/// action, regardless of memory pressure or optimization decisions.
///
/// This is the [Saltzer & Kaashoek 2009] §3.3 Complete Mediation guard — every
/// code path that emits a potentially harmful action (freeze_gate, thermal_interrupt,
/// process_enrichment, decide_actions) can call this one function and be guaranteed
/// complete coverage across all protection tiers.
///
/// **Tiers checked (in order):**
/// 1. OS/system essentials via `protected_processes()` — kernel, WindowServer, Dock, etc.
///    Hard-protected: unconditional, no exceptions. Uses exact name match.
/// 2. Infrastructure services via `infrastructure_processes()` — Docker, Postgres, Redis.
///    Uses substring match to catch bundled executables (`com.docker.backend`).
/// 3. Dev runtime patterns via `matches_dev_runtime()` — rustc, clippy-driver, node, etc.
///    Word-boundary aware to avoid false positives (e.g. "go" vs "mongod").
///
/// **NOT included:** behavioral/interactive app checks (e.g. "Brave Browser" with GUI).
/// Those are context-dependent (foreground vs background) and handled by
/// `is_user_interactive_app()` + `classify_protection()` at callsites that have
/// runtime behavioral data.
///
/// # Performance
/// Hot-path safe: all three checks use `OnceLock` caches or static slices.
/// No allocations on repeated calls.
///
/// # When to use this vs `classify_protection()`
/// - Use `is_protected_name()` when you only have a name and need a fast boolean skip
///   (e.g., before building action lists, in filter loops).
/// - Use `classify_protection()` when you have full runtime context and need to
///   distinguish `Unconditional` vs `ConditionalForeground` vs `Unprotected`.
pub fn is_protected_name(name: &str) -> bool {
    // Tier 1: OS/system essentials — exact match (fast OnceLock path).
    if is_hard_protected(name) {
        return true;
    }
    // Tier 2: Infrastructure services — substring match (docker, postgres, redis…).
    if is_infra_protected(name) {
        return true;
    }
    // Tier 3: Dev runtime patterns — word-boundary aware (rustc, clippy-driver, node…).
    matches_dev_runtime(name)
}

/// MatchEngine-backed protected check returning `(matched, confidence)`.
/// Layers the 3-tier MatchEngine on top of `is_protected_name` without
/// breaking bool-only callers. Saltzer & Kaashoek 2009 §3.3 complete mediation.
///
/// Composition contract (peer-consult item #2, 2026-05-30): `confidence` is
/// epistemic identification, NEVER utility. Route through
/// `match_engine::IdentityUncertaintyFeature` for RSS composition (65f310d);
/// do NOT multiply into PolicyScorer benefit/composite.
pub fn is_protected_with_confidence(
    name: &str,
    learned_protected: &[String],
    weights: &std::collections::HashMap<String, crate::engine::outcome_tracker::PatternWeight>,
) -> (bool, f64) {
    // Hard chokepoint wins: known protected → unconditional 1.0.
    if is_protected_name(name) {
        return (true, 1.0);
    }
    let res = crate::engine::match_engine::match_name(
        name,
        &std::collections::HashSet::new(),
        learned_protected,
    );
    let conf = crate::engine::match_engine::confidence_for(res, None, weights);
    (res.matched(), conf)
}

/// Returns true if the binary path belongs to protected infrastructure territory.
///
/// Covers:
/// - `/opt/homebrew/` — Apple-Silicon Homebrew root (includes `Cellar/` keg-only paths).
/// - `/usr/local/bin/` — traditional system / Intel-Mac Homebrew binaries.
/// - `/usr/local/sbin/` — system-service binaries (haproxy, custom daemons).
/// - `/usr/local/Cellar/` — Intel-Mac Homebrew keg-only paths (nginx, postgresql).
/// - `/Library/PrivilegedHelperTools/` — Apple privileged helpers.
///
/// 2026-05-16 (adversarial review): the original list missed `sbin` and both
/// Cellar roots. Homebrew-managed nginx / postgresql are frequently invoked
/// from `*/Cellar/<pkg>/<ver>/bin/...` rather than the symlink, allowing
/// kqueue-launched processes to escape infrastructure protection.
pub fn is_infrastructure_path(path: &str) -> bool {
    // `/opt/homebrew/Cellar/...` is already covered by the `/opt/homebrew/`
    // prefix; the Cellar test below is a regression guard against future
    // narrowing of that prefix.
    path.starts_with("/opt/homebrew/")
        || path.starts_with("/usr/local/bin/")
        || path.starts_with("/usr/local/sbin/")
        || path.starts_with("/usr/local/Cellar/")
        || path.starts_with("/Library/PrivilegedHelperTools/")
}

/// Fully-mediated protection check for a PID. Checks name, path, and signing.
pub fn is_protected_pid(pid: u32) -> bool {
    if let Some(name) = crate::engine::process_identity::proc_name_for_pid(pid) {
        if is_protected_name(&name) {
            return true;
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(path) = crate::engine::apple_owned::resolve_pid_path(pid) {
            if is_infrastructure_path(&path) {
                return true;
            }
        }
        if crate::engine::apple_owned::is_apple_owned(pid) {
            return true;
        }
    }
    false
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
/// Build an Aho-Corasick matcher for the policy_protected dynamic pattern list.
/// Returns None if the list is empty. Callers that invoke `classify_protection`
/// in a loop over many names should build this once before the loop and pass
/// `Some(&ac)` to all iterations — eliminates per-call `p.to_ascii_lowercase()`
/// allocs inside Tier 3 substring scan.
///
/// Case-insensitive (mirrors prior `name_lc.contains(p.to_ascii_lowercase())`
/// semantics). Build cost is amortized over the loop iterations.
pub fn build_policy_protected_ac(policy_protected: &[String]) -> Option<aho_corasick::AhoCorasick> {
    if policy_protected.is_empty() {
        return None;
    }
    aho_corasick::AhoCorasickBuilder::new()
        .ascii_case_insensitive(true)
        .build(policy_protected)
        .ok()
}

pub fn classify_protection(
    name: &str,
    hard_protected: &HashSet<&'static str>,
    infra_protected: &HashSet<&'static str>,
    policy_protected: &[String],
    policy_protected_ac: Option<&aho_corasick::AhoCorasick>,
    is_interactive: bool,
) -> ProtectionLevel {
    // Tier 1: OS/system essentials — substring to catch variants like
    // "com.apple.WindowServer" matching "WindowServer". Fast path via
    // Aho-Corasick OnceLock. Caller-supplied HashSet args are ignored in favor
    // of the canonical static set (all production callers pass that anyway).
    let _ = hard_protected;
    let _ = infra_protected;
    if hard_protected_contains(name) {
        return ProtectionLevel::Unconditional;
    }
    // Tier 2: Stateful infrastructure (Docker, Postgres, Redis, …).
    if is_infra_protected(name) {
        return ProtectionLevel::Unconditional;
    }
    // Tier 3: Policy-learned daemons (case-insensitive substring).
    // Fast path: caller-supplied AC scans all patterns in single O(name.len) pass.
    // Slow path: per-pattern `to_ascii_lowercase()` chain — used by tests and any
    // caller that hasn't built the AC yet. Both preserve identical semantics.
    if !policy_protected.is_empty() {
        if let Some(ac) = policy_protected_ac {
            if ac.is_match(name) {
                return ProtectionLevel::Unconditional;
            }
        } else {
            let name_lc = name.to_ascii_lowercase();
            if policy_protected
                .iter()
                .any(|p| name_lc.contains(p.to_ascii_lowercase().as_str()))
            {
                return ProtectionLevel::Unconditional;
            }
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
        // macOS Virtualization.framework helpers used by modern container
        // tooling. Same blast radius as qemu-system: freezing them mid-write
        // can corrupt the guest filesystem / journal. Added 2026-05-10 after
        // a Podman VM journal-corruption incident — Apollo journal showed no
        // direct hits, but `vfkit` was eligible for SIGSTOP because the
        // pre-2024 name list ("podman", "qemu-system", "lima") missed the
        // basename returned by sysinfo. NotebookLM peer review expanded the
        // missing-helper set; all named here.
        "vfkit",           // Virtualization.framework wrapper (Podman 5.x default)
        "gvproxy",         // gvisor-tap-vsock networking helper (Podman/Lima)
        "krunkit",         // libkrun-based microVM (alt Podman backend)
        "vmapple",         // Tart VM helper (Virtualization.framework child)
        "vmnetd",          // macOS vmnet daemon (bridge networking for Lima/Podman)
        "slirp4netns",     // user-mode networking for rootless Podman
        "containerd-shim", // Docker / containerd process shim — frozen = stuck container
        "runc",            // OCI runtime — frozen mid-spawn = orphaned container
        "orbstack",        // OrbStack main process
        "orbstack-helper",
        "orbstack-node",   // OrbStack VM node
        "lima-guestagent", // Lima guest agent helper
        "dockerd",         // Docker daemon (not just "com.docker.backend")
        "containerd",      // Standalone containerd
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
        // LLM inference servers — moved to softly_protected_processes().
        // Unconditional protection removed: at survival pressure (≥0.85) these
        // become freeze candidates. See softly_protected_processes().
    ]
    .into_iter()
    .collect()
}

/// Processes protected at NORMAL pressure but expendable at survival pressure
/// (memory_pressure ≥ 0.85 AND swap_gb ≥ 2.0).
///
/// Tier philosophy [Nygard 2018 "Release It!" Ch.5 — Load Shedding]:
///   Tier 0 (always protected): OS daemons, Claude, foreground app.
///   Tier 1 (softly protected): AI model servers, background IDE helpers.
///     Frozen only when the alternative is OOM reboot.
///
/// Adding a new model server: put it here, not in infrastructure_processes().
/// It will be protected at healthy pressure but shedable in crisis.
pub fn softly_protected_processes() -> HashSet<&'static str> {
    [
        // LLM inference servers — freezing kills token generation mid-stream
        // at normal pressure. At ≥0.85 / 2GB swap the OOM risk outweighs
        // a single interrupted inference call (caller retries on ECONNRESET).
        "llama-server",
        "ollama_llama_server",
        "ollama",
        "lm-studio",
        "lmstudio",
    ]
    .into_iter()
    .collect()
}

/// Survival pressure threshold above which softly_protected processes lose
/// their protection and become freeze candidates.
pub const SOFT_PROTECTION_PRESSURE_THRESHOLD: f64 = 0.85;

/// Swap threshold (GB) that must ALSO be exceeded before soft protection drops.
/// Prevents premature shedding on transient pressure spikes with healthy swap headroom.
pub const SOFT_PROTECTION_SWAP_GB_THRESHOLD: f64 = 2.0;

/// Absolute swap exhaustion floor: trigger survival regardless of kernel pressure.
/// On M1 8GB, swap maxes at ~5GB. At 4GB the system is minutes from OOM panic.
/// kernel pressure can read LOW while swap is near-full because the compressor
/// has already absorbed the pages — the pressure signal lags the real risk
/// [Nygard 2018 §4, macOS memorystatus internals].
///
/// Used as an absolute floor; the effective threshold is `max(this, swap_total × pct)`
/// so machines with larger swap (16GB+) also trigger at a reasonable utilization.
pub const SWAP_EXHAUSTION_GB: f64 = 4.0;

/// Fraction of total swap that counts as "exhaustion" on larger machines.
/// 35% of swap_total is aggressive enough to fire before the compressor saturates
/// but high enough to avoid false positives during normal paging bursts.
/// [Denning 1968] — working-set approximation: beyond this fraction, the kernel is
/// paging hot working-set pages, not just cold tail.
pub const SWAP_EXHAUSTION_PCT: f64 = 0.35;

/// Hard survival trigger — fires when swap_used / swap_total ≥ this fraction,
/// regardless of absolute swap size. Closes a calibration trap on M1 8GB where
/// swap_total = 4 GB makes the `max(4 GB, ...)` absolute floor unreachable
/// before 100% saturation: production swap = 3.92 GB / 4 GB (91 %) never fires
/// `swap_exhaustion_threshold_bytes` (= 4 GB). Production cycles=91k,
/// freezes_applied=0, freeze_gate=thrashing — the gate fires but candidates
/// (softly_protected) are still protected because survival mode never crosses.
/// Original design intent (memory.md "Época 8"): `swap_used >= 0.80 * swap_total`.
/// Cross-hardware sanity: 16 GB swap → fires at 13.6 GB (severe but legitimate
/// crisis); 32 GB swap → 27.2 GB (effectively kernel-panic territory).
/// [Nygard 2018 §5] Load Shedding — at this saturation, shed before kernel does.
pub const SURVIVAL_SWAP_PCT_THRESHOLD: f64 = 0.85;

/// Effective swap-exhaustion byte threshold for a given swap_total.
/// Returns `max(SWAP_EXHAUSTION_GB, swap_total × SWAP_EXHAUSTION_PCT)`.
/// When `swap_total == 0` (misreported / no swap), falls back to the absolute floor.
#[inline]
pub fn swap_exhaustion_threshold_bytes(swap_total_bytes: u64) -> u64 {
    let gib: u64 = 1024 * 1024 * 1024;
    let abs_floor = (SWAP_EXHAUSTION_GB * gib as f64) as u64;
    let pct_floor = ((swap_total_bytes as f64) * SWAP_EXHAUSTION_PCT) as u64;
    abs_floor.max(pct_floor)
}

/// Returns true when conditions justify shedding softly_protected processes.
/// Two independent triggers (OR):
///   1. Both pressure AND swap above normal thresholds (sustained crisis)
///   2. Swap alone near exhaustion — scales with swap_total via
///      `swap_exhaustion_threshold_bytes()`
///
/// When `swap_total_bytes == 0` the function degrades to the absolute 4GB floor.
#[inline]
pub fn survival_mode_active_total(
    memory_pressure: f64,
    swap_used_bytes: u64,
    swap_total_bytes: u64,
) -> bool {
    let swap_gb = swap_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let exhausted = swap_used_bytes >= swap_exhaustion_threshold_bytes(swap_total_bytes);
    let pct_exhausted = swap_total_bytes > 0
        && (swap_used_bytes as f64 / swap_total_bytes as f64) >= SURVIVAL_SWAP_PCT_THRESHOLD;
    (memory_pressure >= SOFT_PROTECTION_PRESSURE_THRESHOLD
        && swap_gb >= SOFT_PROTECTION_SWAP_GB_THRESHOLD)
        || exhausted
        || pct_exhausted
}

/// Backward-compatible shim: assumes swap_total is unknown → absolute 4GB floor only.
/// Call sites that have `swap_total_bytes` should prefer `survival_mode_active_total`.
#[inline]
pub fn survival_mode_active(memory_pressure: f64, swap_used_bytes: u64) -> bool {
    survival_mode_active_total(memory_pressure, swap_used_bytes, 0)
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
    // Long patterns (≥4 chars) → single-pass Aho-Corasick (case-insensitive).
    static LONG_MATCHER: OnceLock<aho_corasick::AhoCorasick> = OnceLock::new();
    let long = LONG_MATCHER.get_or_init(|| {
        let pats: Vec<&'static str> = dev_runtime_patterns()
            .iter()
            .copied()
            .filter(|p| p.len() > 3)
            .collect();
        aho_corasick::AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(pats)
            .expect("dev_runtime long patterns build")
    });
    if long.is_match(name) {
        return true;
    }
    // Short patterns (≤3 chars) — word-boundary aware (currently just "go").
    let lower = name.to_ascii_lowercase();
    for &pat in dev_runtime_patterns() {
        if pat.len() > 3 {
            continue;
        }
        let mut start = 0;
        while let Some(pos) = lower[start..].find(pat) {
            let abs_pos = start + pos;
            let end_pos = abs_pos + pat.len();
            let left_ok = abs_pos == 0 || !lower.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
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
    let recency_signal = (-(secs_idle as f64) / 600.0).exp();

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
            RootAction::SetSysctl(_) => {
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
            RootAction::SetSysctl(_) => budget.cycle_sysctl_writes += 1,
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
    // 2026-05-31 (post-MatchEngine 29f09e0): removed the <4 early-reject.
    // The MatchEngine substring tier (confidence 0.30, below freeze floor
    // 0.35) now provides structural safety against short-pattern ambiguity
    // at runtime. socket_handler enforces MIN_PATTERN_LEN=3 at apply time
    // so 2-char patterns never reach this function — anything we see is
    // ≥3 chars. The bidirectional substring + 75% overlap checks below
    // still catch any real conflict with hardcoded protected names.
    if pat.is_empty() {
        return true;
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

    // ── is_infrastructure_path coverage tests (2026-05-16 adversarial review) ─

    /// Homebrew-managed nginx / postgres are frequently invoked from
    /// `/opt/homebrew/Cellar/<pkg>/<ver>/bin/...` instead of the
    /// `/opt/homebrew/bin/<pkg>` symlink. The pre-2026-05-16 prefix list
    /// missed the Cellar root, allowing kqueue-launched processes that
    /// bypass the symlink to escape infrastructure protection.
    #[test]
    fn infrastructure_path_covers_homebrew_cellar() {
        assert!(is_infrastructure_path(
            "/opt/homebrew/Cellar/postgresql@16/16.2/bin/postgres"
        ));
    }

    /// Intel-Mac Homebrew layout puts the Cellar under `/usr/local/Cellar/`.
    /// The pre-2026-05-16 prefix list only had `/usr/local/bin/`, so any
    /// service invoked via the keg-only path was unprotected.
    #[test]
    fn infrastructure_path_covers_usr_local_cellar() {
        assert!(is_infrastructure_path(
            "/usr/local/Cellar/nginx/1.27.4/bin/nginx"
        ));
    }

    /// System service binaries traditionally live in `/usr/local/sbin/`
    /// (haproxy, custom daemons), distinct from `/usr/local/bin/`. The
    /// pre-2026-05-16 prefix list omitted this directory.
    #[test]
    fn infrastructure_path_covers_usr_local_sbin() {
        assert!(is_infrastructure_path("/usr/local/sbin/foobar"));
    }

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
            classify_protection("kernel_task", &hard, &infra, &[], None, false),
            ProtectionLevel::Unconditional,
            "kernel_task must be Unconditional"
        );
    }

    #[test]
    fn classify_windowserver_is_unconditional() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        assert_eq!(
            classify_protection("WindowServer", &hard, &infra, &[], None, false),
            ProtectionLevel::Unconditional
        );
    }

    #[test]
    fn classify_docker_infra_is_unconditional() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        assert_eq!(
            classify_protection("com.docker.backend", &hard, &infra, &[], None, false),
            ProtectionLevel::Unconditional,
            "com.docker.backend must be Unconditional via infra substring match"
        );
    }

    #[test]
    fn classify_postgres_is_unconditional() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        assert_eq!(
            classify_protection("postgres", &hard, &infra, &[], None, false),
            ProtectionLevel::Unconditional
        );
    }

    #[test]
    fn classify_policy_protected_is_unconditional() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        let policy = test_policy();
        assert_eq!(
            classify_protection("mypostgres-wrapper", &hard, &infra, &policy, None, false),
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
            classify_protection("custom-db", &hard, &infra, &policy, None, false),
            ProtectionLevel::Unconditional
        );
        // Process name uppercase, pattern lowercase in policy
        let policy2 = vec!["custom-db".to_string()];
        assert_eq!(
            classify_protection("Custom-DB", &hard, &infra, &policy2, None, false),
            ProtectionLevel::Unconditional
        );
    }

    #[test]
    fn classify_interactive_app_foreground_conditional() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        // is_interactive=true (caller evaluated is_user_interactive_app)
        assert_eq!(
            classify_protection("Brave Browser", &hard, &infra, &[], None, true),
            ProtectionLevel::ConditionalForeground,
            "user interactive app must be ConditionalForeground"
        );
    }

    #[test]
    fn classify_unknown_process_is_unprotected() {
        let hard = protected_processes();
        let infra = infrastructure_processes();
        assert_eq!(
            classify_protection("com.example.random-daemon", &hard, &infra, &[], None, false),
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
            classify_protection("Dropbox", &hard, &infra, &[], None, false),
            ProtectionLevel::Unprotected,
            "Dropbox must not be Unconditional — 'box' is not a protected pattern"
        );
        // If Dropbox has a GUI window and recent interaction, caller sets is_interactive=true
        assert_eq!(
            classify_protection("Dropbox", &hard, &infra, &[], None, true),
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
            classify_protection("kernel_task", &hard, &infra, &[], None, true),
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

    #[test]
    fn survival_mode_tiers() {
        // Normal: llama-server softly protected
        assert!(!survival_mode_active(
            0.70,
            (1.5 * 1024.0 * 1024.0 * 1024.0) as u64
        ));
        // High pressure but low swap: still protected
        assert!(!survival_mode_active(
            0.90,
            (1.0 * 1024.0 * 1024.0 * 1024.0) as u64
        ));
        // Survival: both pressure+swap thresholds crossed
        assert!(survival_mode_active(
            0.85,
            (2.0 * 1024.0 * 1024.0 * 1024.0) as u64
        ));
        assert!(survival_mode_active(
            0.95,
            (4.0 * 1024.0 * 1024.0 * 1024.0) as u64
        ));
        // Swap exhaustion alone (≥4GB) triggers even with low pressure
        // — kernel pressure lags when compressor absorbed the crisis [Nygard 2018]
        assert!(survival_mode_active(
            0.59,
            (4.0 * 1024.0 * 1024.0 * 1024.0) as u64
        ));
        assert!(survival_mode_active(
            0.40,
            (4.5 * 1024.0 * 1024.0 * 1024.0) as u64
        ));
        // Below exhaustion floor: not triggered by swap alone
        assert!(!survival_mode_active(
            0.50,
            (3.9 * 1024.0 * 1024.0 * 1024.0) as u64
        ));
        // llama-server in soft list, not hard list
        assert!(!protected_processes().contains("llama-server"));
        assert!(softly_protected_processes().contains("llama-server"));
    }

    #[test]
    fn swap_exhaustion_threshold_scales_with_total() {
        let gib = 1024u64 * 1024 * 1024;
        // Small swap (8GB): absolute 4GB floor dominates (35% = 2.8GB < 4GB).
        assert_eq!(swap_exhaustion_threshold_bytes(8 * gib), 4 * gib);
        // Medium swap (12GB): 35% = 4.2GB > absolute floor → relative dominates.
        let t12 = swap_exhaustion_threshold_bytes(12 * gib);
        assert!(t12 > 4 * gib);
        // Large swap (16GB): 35% = 5.6GB — relative dominates.
        let t16 = swap_exhaustion_threshold_bytes(16 * gib);
        assert!(t16 >= (5.5 * gib as f64) as u64);
        assert!(t16 <= (5.7 * gib as f64) as u64);
        // Zero total (unknown): falls back to absolute floor.
        assert_eq!(swap_exhaustion_threshold_bytes(0), 4 * gib);
    }

    #[test]
    fn survival_total_variant_respects_relative_threshold() {
        let gib = 1024u64 * 1024 * 1024;
        // On 16GB swap machine, 4.5GB used should NOT yet trigger survival
        // (threshold ≈ 5.6GB at 35%).
        assert!(!survival_mode_active_total(
            0.50,
            (4.5 * gib as f64) as u64,
            16 * gib,
        ));
        // Same 4.5GB on 8GB swap machine SHOULD trigger (absolute 4GB floor).
        assert!(survival_mode_active_total(
            0.50,
            (4.5 * gib as f64) as u64,
            8 * gib,
        ));
        // 6GB on 16GB machine triggers (above relative 5.6GB).
        assert!(survival_mode_active_total(0.50, 6 * gib, 16 * gib,));
        // Back-compat shim: no total known → absolute floor.
        assert!(survival_mode_active(0.50, (4.0 * gib as f64) as u64));
    }

    /// M1 8GB trap: swap_total = 4 GB makes the absolute 4 GB floor unreachable
    /// before 100 % saturation. Production observed swap_used = 3.92 GB / 4 GB
    /// (91 %) with `freezes_applied=0` because survival never activated. The
    /// percentage trigger (0.85) must fire here.
    #[test]
    fn survival_pct_trigger_closes_m1_8gb_trap() {
        let gib = 1024u64 * 1024 * 1024;
        let four_gb = 4 * gib;
        // 91 % of 4 GB swap → percentage trigger fires even with moderate pressure.
        let used_91 = ((4.0 * gib as f64) * 0.91) as u64;
        assert!(survival_mode_active_total(0.78, used_91, four_gb));
        // 80 % of 4 GB → below 0.85 trigger AND below 4 GB absolute → must NOT fire.
        let used_80 = ((4.0 * gib as f64) * 0.80) as u64;
        assert!(!survival_mode_active_total(0.50, used_80, four_gb));
        // 86 % of 4 GB → just above the percentage trigger.
        let used_86 = ((4.0 * gib as f64) * 0.86) as u64;
        assert!(survival_mode_active_total(0.50, used_86, four_gb));
        // 16 GB swap at 14 GB used (87 %) → percentage trigger also fires.
        assert!(survival_mode_active_total(0.50, 14 * gib, 16 * gib));
        // 16 GB swap at 13 GB used (81 %) → below pct trigger AND below
        // 35 % × 16 = 5.6 GB absolute → AND-branch needs pressure ≥ 0.85.
        // 13 GB > 5.6 GB → exhausted branch fires; survival should be true.
        assert!(survival_mode_active_total(0.50, 13 * gib, 16 * gib));
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

    // ── is_protected_name() — Single Truth Point tests ──────────────────────
    //
    // These tests prove that is_protected_name() closes the three known bypass
    // classes identified in the 360° Muscle Map:
    //
    //   Bypass class 1 (sharingd loop): OS daemons not in INTERACTIVE_APPS
    //     were missed by callers that only called is_interactive_app_name().
    //     is_protected_name() covers Tier 1 (OS essentials) unconditionally.
    //
    //   Bypass class 2 (Notion/Antigravity): Callers checked critical_pids
    //     (PID-based) but not process names for ConditionalForeground helpers.
    //     is_protected_name() + classify_protection() together close this.
    //
    //   Bypass class 3 (clippy-driver/rustc): Compiler toolchain missed because
    //     the freeze guard only checked INTERACTIVE_APPS names, not dev_runtime
    //     patterns. is_protected_name() covers Tier 3 (dev_runtime) explicitly.

    /// Tier 1: OS/system essentials must all be protected.
    #[test]
    fn is_protected_name_covers_os_essentials() {
        // Representative sample — exact names from protected_processes().
        for name in &[
            "kernel_task",
            "launchd",
            "WindowServer",
            "Dock",
            "sharingd", // Bug 6: frozen 173× via gate_c (missed protected_processes)
            "logd",     // watchdog-adjacent — SIGSTOP triggers kernel panic
            "watchdogd",
            "coreaudiod",
            "mDNSResponder",
            "airportd",
            "bluetoothd",
        ] {
            assert!(
                is_protected_name(name),
                "is_protected_name('{}') must return true — OS essential",
                name
            );
        }
    }

    /// Tier 2: Infrastructure services must be protected via substring match.
    #[test]
    fn is_protected_name_covers_infrastructure() {
        // Exact names from infrastructure_processes().
        assert!(is_protected_name("postgres"), "postgres must be protected");
        assert!(
            is_protected_name("redis-server"),
            "redis-server must be protected"
        );
        assert!(is_protected_name("docker"), "docker must be protected");
        // Bundled variant — substring match catches this.
        assert!(
            is_protected_name("com.docker.backend"),
            "com.docker.backend must be protected via substring match on 'docker'"
        );
    }

    /// Tier 3: Dev runtime patterns must be protected (compiler toolchain bypass fix).
    #[test]
    fn zombie_demote_guard_contract_is_protected_name_not_hard_only() {
        // 2026-06-20: the zombie-hunter jetsam-demote guard was broadened from
        // hard_protected_contains to is_protected_name after the regression
        // probe caught `node` nominated 3x. Pin the two tiers the hard-only
        // check missed: dev-runtime (node) and infra (docker). Both MUST be
        // is_protected_name=true but are NOT in the hard-protected set.
        assert!(
            is_protected_name("node"),
            "dev-runtime node must be protected"
        );
        assert!(
            is_protected_name("com.docker.backend"),
            "infra (docker) must be protected by is_protected_name"
        );
        assert!(
            !hard_protected_contains("node"),
            "node is NOT hard-protected — proves the old guard let it through"
        );
    }

    #[test]
    fn playback_working_set_protection_contract_p1_p2_p3() {
        // 2026-06-21 (playback-easing Wave 1): the zombie-hunter jetsam-demote
        // and the SetMemorystatus execute chokepoint must never act on the 4K
        // playback working set. Two protections, two mechanisms:
        // P3 — mediaplaybackd is now hard-listed → is_protected_name covers it.
        assert!(
            is_protected_name("mediaplaybackd"),
            "P3: mediaplaybackd (video playback daemon) must be protected"
        );
        // P1/P2 — Chromium/Brave renderers are NOT in is_protected_name (verified
        // here), which is exactly why P1 (zombie guard) and P2 (execute
        // chokepoint) add an explicit is_chromium_family check. If this ever
        // becomes true, the explicit guards become redundant — but until then
        // they are load-bearing.
        assert!(
            is_chromium_family("Brave Browser Helper (GPU)"),
            "P1/P2: a Brave GPU/renderer helper must match is_chromium_family"
        );
        assert!(
            is_chromium_family("Brave Browser Helper (Renderer)"),
            "P1/P2: a Brave Renderer helper must match is_chromium_family"
        );
        assert!(
            !is_protected_name("Brave Browser Helper (GPU)"),
            "is_protected_name does NOT cover chromium — proves P1/P2's explicit \
             is_chromium_family guard is necessary, not redundant"
        );
    }

    #[test]
    fn is_protected_name_covers_dev_runtimes() {
        // Bug 3 / Bypass class 3: rustc and clippy-driver were frozen because the
        // freeze guard only checked INTERACTIVE_APPS, not dev_runtime_patterns.
        assert!(
            is_protected_name("rustc"),
            "rustc must be protected — compiler toolchain"
        );
        assert!(
            is_protected_name("clippy-driver"),
            "clippy-driver must be protected — 7 freezes during builds in prod"
        );
        assert!(is_protected_name("node"), "node must be protected");
        assert!(
            is_protected_name("python3.13"),
            "python3.13 must be protected via substring"
        );
    }

    /// Negative cases: non-protected apps must NOT be blocked.
    #[test]
    fn is_protected_name_does_not_over_protect() {
        // User apps that are NOT in the protection tiers — they're
        // ConditionalForeground (behavioral), not Unconditional.
        // is_protected_name() correctly returns false for these;
        // the caller then uses classify_protection() for full nuance.
        assert!(
            !is_protected_name("Safari"),
            "Safari is ConditionalForeground (behavioral), not hard-protected"
        );
        assert!(
            !is_protected_name("Spotify"),
            "Spotify is ConditionalForeground, not hard-protected"
        );
        assert!(
            !is_protected_name("com.example.random-daemon"),
            "Unknown process must not be over-protected"
        );
        // Word-boundary: "go" must not match "google-chrome" or "Cargo".
        // Note: "mongod" IS in infrastructure_processes() (MongoDB daemon) so it
        // correctly returns true — the word-boundary test for "go" is separate
        // (see go_does_not_match_substrings test for matches_dev_runtime).
        assert!(
            !is_protected_name("google-chrome"),
            "google-chrome must NOT match 'go'"
        );
        assert!(!is_protected_name("Cargo"), "Cargo must NOT match 'go'");
    }

    /// Self-protection: Apollo's own binaries must be protected (prevent self-freeze).
    #[test]
    fn is_protected_name_covers_apollo_self() {
        assert!(
            is_protected_name("apollo-optimizerd"),
            "apollo-optimizerd must be self-protected"
        );
        assert!(
            is_protected_name("apollo-optimizerctl"),
            "apollo-optimizerctl must be self-protected"
        );
    }
}
