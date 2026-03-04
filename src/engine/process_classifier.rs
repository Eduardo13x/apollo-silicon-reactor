//! Process Classifier — heuristic-based process categorization
//!
//! Classifies every process into a tier without any LLM.
//! Decision logic is a scored rule set:
//!   score = utility_score / (resource_cost + 1)
//! Low-score processes are throttle / kill candidates.

use std::collections::HashSet;

// ── Category ─────────────────────────────────────────────────────────────────

/// How a process relates to the user right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessTier {
    /// User is actively interacting (frontmost window, keyboard input).
    ActiveForeground,
    /// Opened by the user, not focused but recently used.
    BackgroundVisible,
    /// Helper / extension process for an app the user runs.
    AppHelper,
    /// System daemon that the user genuinely needs (launchd, coreaudio…).
    SystemEssential,
    /// Background service with no visible user benefit.
    SilentDaemon,
    /// Process that has not been interacted with in >24 h.
    Stale,
    /// Zombie (Z) or parent-dead orphan.
    ZombieOrphan,
    /// Apple / 3rd-party telemetry / analytics.
    Telemetry,
}

// ── Known-process knowledge base ─────────────────────────────────────────────

/// Processes that must never be touched.
pub fn essential_process_names() -> HashSet<&'static str> {
    [
        // macOS kernel & core
        "kernel_task",
        "launchd",
        "kextd",
        "watchdogd",
        "securityd",
        "configd",
        "notifyd",
        "opendirectoryd",
        "syslogd",
        "diskarbitrationd",
        "powerd",
        "WindowServer",
        // Audio / video pipeline
        "coreaudiod",
        "audioclocksyncd",
        "avconferenced",
        // Input
        "hidd",
        "bluetoothd",
        // Network
        "mDNSResponder",
        "networkd",
        "trustd",
        "neagent",
        // Our own daemon
        "apollo-optimizerd",
    ]
    .iter()
    .copied()
    .collect()
}

/// Processes known to be telemetry / analytics — safe to throttle hard.
pub fn telemetry_process_names() -> HashSet<&'static str> {
    [
        // Apple telemetry
        "DiagnosticReportsTrigger",
        "ReportPanic",
        "spindump",
        "SubmitDiagInfo",
        "osanalyticshelper",
        "CrashReporter",
        "analyticsd",
        "WirelessRadioManagerd",
        // Common 3rd-party
        "GoogleCrashHandler",
        "GoogleUpdate",
        "firefox-crash-reporter",
        "SentinelStatsd",
        "MacFUSEHelper",
        "adobe_crash_reporter",
        // Electron / CEF helpers that report telemetry
        "Electron Helper (GPU)",
        "Electron Helper (Renderer)",
    ]
    .iter()
    .copied()
    .collect()
}

/// Well-known "helper" suffixes / substrings — child processes of user apps.
pub fn helper_name_patterns() -> &'static [&'static str] {
    &[
        " Helper",
        " Helper (GPU)",
        " Helper (Renderer)",
        " Helper (Plugin)",
        " Agent",
        "Extension",
        "Service",
        "XPCService",
        "BrokerExtension",
    ]
}

// ── Scoring ───────────────────────────────────────────────────────────────────

/// Lightweight snapshot of a process for classification.
#[derive(Debug, Clone)]
pub struct ProcessSnapshot {
    pub pid: u32,
    pub name: String,
    pub cpu_percent: f32,
    pub rss_bytes: u64,
    pub is_zombie: bool,
    /// Seconds since the process had a foreground window.
    pub secs_since_foreground: u64,
    /// Seconds since user last interacted with any window of this app.
    pub secs_since_user_interaction: u64,
    /// True if the process has any active network connections.
    pub has_network: bool,
    /// True if there is any open GUI window.
    pub has_gui_window: bool,
    /// Number of voluntary wakeups per second (Mach port messages received).
    pub wakeups_per_sec: f32,
    /// Parent process is still alive.
    pub parent_alive: bool,
}

/// Score a process's "user utility" on a 0.0–1.0 scale.
/// Higher = more useful / should not be throttled.
pub fn score_utility(snap: &ProcessSnapshot) -> f32 {
    if snap.is_zombie || !snap.parent_alive {
        return 0.0;
    }

    let mut score: f32 = 0.0;

    // Active interaction is the strongest signal
    score += match snap.secs_since_user_interaction {
        0..=30 => 1.0,
        31..=300 => 0.7,
        301..=3600 => 0.4,
        3601..=86400 => 0.1,
        _ => 0.0,
    };

    // GUI window visible
    if snap.has_gui_window {
        score += 0.3;
    }

    // Normalise to 0..1
    (score / 1.3).min(1.0)
}

/// Score a process's resource waste on a 0.0–1.0 scale.
/// Higher = more wasteful.
pub fn score_waste(snap: &ProcessSnapshot) -> f32 {
    let cpu_waste = (snap.cpu_percent / 10.0).min(1.0); // 10 % CPU → max
    let wakeup_waste = (snap.wakeups_per_sec / 50.0).min(1.0); // 50/s → max
    ((cpu_waste + wakeup_waste) / 2.0).min(1.0)
}

// ── Classifier ────────────────────────────────────────────────────────────────

pub struct ProcessClassifier {
    essential: HashSet<&'static str>,
    telemetry: HashSet<&'static str>,
}

impl ProcessClassifier {
    pub fn new() -> Self {
        Self {
            essential: essential_process_names(),
            telemetry: telemetry_process_names(),
        }
    }

    /// Classify a single process snapshot deterministically.
    pub fn classify(&self, snap: &ProcessSnapshot) -> ProcessTier {
        // Hard rules first — order matters
        if snap.is_zombie || !snap.parent_alive {
            return ProcessTier::ZombieOrphan;
        }
        if self.essential.contains(snap.name.as_str()) {
            return ProcessTier::SystemEssential;
        }
        if self.telemetry.contains(snap.name.as_str()) {
            return ProcessTier::Telemetry;
        }
        if snap.secs_since_user_interaction == 0 {
            return ProcessTier::ActiveForeground;
        }
        if snap.secs_since_user_interaction <= 300 && snap.has_gui_window {
            return ProcessTier::BackgroundVisible;
        }

        // Heuristic rules
        let is_helper = helper_name_patterns()
            .iter()
            .any(|p| snap.name.contains(p));

        if snap.secs_since_user_interaction > 86400 {
            return ProcessTier::Stale;
        }
        if is_helper {
            return ProcessTier::AppHelper;
        }
        if !snap.has_gui_window && !snap.has_network && snap.wakeups_per_sec > 5.0 {
            return ProcessTier::SilentDaemon;
        }

        // Default
        ProcessTier::SilentDaemon
    }

    /// Classify many processes and return sorted by descending waste score.
    pub fn classify_all(&self, snaps: &[ProcessSnapshot]) -> Vec<(ProcessSnapshot, ProcessTier, f32)> {
        let mut results: Vec<(ProcessSnapshot, ProcessTier, f32)> = snaps
            .iter()
            .map(|s| {
                let tier = self.classify(s);
                let waste = score_waste(s);
                (s.clone(), tier, waste)
            })
            .collect();

        results.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Return only processes that are safe throttle/kill candidates.
    pub fn throttle_candidates<'a>(
        &self,
        snaps: &'a [ProcessSnapshot],
    ) -> Vec<&'a ProcessSnapshot> {
        snaps
            .iter()
            .filter(|s| {
                let tier = self.classify(s);
                matches!(
                    tier,
                    ProcessTier::Stale
                        | ProcessTier::SilentDaemon
                        | ProcessTier::Telemetry
                        | ProcessTier::ZombieOrphan
                )
            })
            .collect()
    }
}

impl Default for ProcessClassifier {
    fn default() -> Self {
        Self::new()
    }
}
