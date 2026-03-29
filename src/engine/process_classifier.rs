//! Process Classifier — heuristic-based process tier classification.
//!
//! Classifies every process into a tier without any LLM.
//! Decision logic uses behavioral signals: CPU, RSS, wakeups, GUI presence, etc.
//!
//!   tier = classify(name, cpu, rss, wakeups, gui, network, uptime, ...)
//!
//! score_utility() maps a snapshot to a [0,1] utility score used by AdaptiveGovernor.

use std::collections::HashSet;

// ── Tier ──────────────────────────────────────────────────────────────────────

/// Priority tier for a process. Higher = more important = never throttle/freeze.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProcessTier {
    /// System-critical (launchd, WindowServer, kernel helpers). Never touch.
    SystemEssential,
    /// The foreground app the user is actively using.
    ActiveForeground,
    /// Visible background app (e.g. music player, notification app).
    BackgroundVisible,
    /// Helper/renderer spawned by a foreground app (Chrome Helper, etc.).
    AppHelper,
    /// Background daemon with no GUI and low wakeup rate.
    SilentDaemon,
    /// Process that has been idle for a long time — freeze candidate.
    Stale,
    /// Telemetry / analytics process — always throttle.
    Telemetry,
    /// Zombie or orphan — kill candidate.
    ZombieOrphan,
}

// ── ProcessSnapshot ───────────────────────────────────────────────────────────

/// Lightweight snapshot of a process for classification.
#[derive(Debug, Clone)]
pub struct ProcessSnapshot {
    pub pid: u32,
    pub name: String,
    pub cpu_percent: f32,
    pub rss_bytes: u64,
    pub is_zombie: bool,
    pub secs_since_foreground: u64,
    pub secs_since_user_interaction: u64,
    pub has_network: bool,
    pub has_gui_window: bool,
    pub wakeups_per_sec: f32,
    pub parent_alive: bool,
    pub process_uptime_secs: u64,
    pub faults_total: u32,
    pub pageins_total: u32,
    pub is_translated: bool,
    pub mach_port_count: u32,
}

// ── Name lists ────────────────────────────────────────────────────────────────

pub fn essential_process_names() -> HashSet<&'static str> {
    [
        "launchd", "kernel_task", "WindowServer", "loginwindow", "cfprefsd",
        "UserEventAgent", "diskarbitrationd", "powerd", "configd", "notifyd",
        "syslogd", "opendirectoryd", "coreaudiod", "bluetoothd", "airportd",
        "mDNSResponder", "systemstats", "logd", "nsurlsessiond", "symptomsd",
    ]
    .iter()
    .copied()
    .collect()
}

pub fn telemetry_process_names() -> HashSet<&'static str> {
    [
        "DiagnosticReporter", "CrashReporter", "ReportCrash", "SubmitDiagInfo",
        "Siri", "SiriNCService", "parsec-fbf", "OSLogHelper",
        "com.apple.telemetry", "EscrowSecurityAlert", "analyticsd",
        "symptomsd", "rapportd",
    ]
    .iter()
    .copied()
    .collect()
}

pub fn helper_name_patterns() -> &'static [&'static str] {
    &[
        "Helper", "Renderer", "GPU Process", "Crashpad", "BraveSoftware Helper",
        "Google Chrome Helper", "Electron Helper", "plugin-container",
    ]
}

// ── ProcessClassifier ─────────────────────────────────────────────────────────

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

    /// Classify all snapshots. Returns (snap, tier, waste_score) triples.
    pub fn classify_all<'a>(
        &self,
        snaps: &'a [ProcessSnapshot],
    ) -> Vec<(&'a ProcessSnapshot, ProcessTier, f32)> {
        snaps
            .iter()
            .map(|s| {
                let tier = self.classify(s);
                let waste = waste_score(s, tier);
                (s, tier, waste)
            })
            .collect()
    }

    pub fn classify(&self, snap: &ProcessSnapshot) -> ProcessTier {
        // Zombie / orphan
        if snap.is_zombie || (!snap.parent_alive && snap.pid > 1) {
            return ProcessTier::ZombieOrphan;
        }

        // System essential
        if self.essential.contains(snap.name.as_str()) {
            return ProcessTier::SystemEssential;
        }

        // Telemetry
        if self.telemetry.contains(snap.name.as_str()) {
            return ProcessTier::Telemetry;
        }
        for pat in helper_name_patterns() {
            if snap.name.contains(pat) {
                return if snap.has_gui_window {
                    ProcessTier::AppHelper
                } else {
                    ProcessTier::SilentDaemon
                };
            }
        }

        // Foreground
        if snap.has_gui_window && snap.secs_since_user_interaction < 30 {
            return ProcessTier::ActiveForeground;
        }

        // Background visible
        if snap.has_gui_window {
            return ProcessTier::BackgroundVisible;
        }

        // Stale: no GUI, low CPU, long idle
        if snap.cpu_percent < 0.5
            && snap.wakeups_per_sec < 1.0
            && snap.secs_since_foreground > 300
        {
            return ProcessTier::Stale;
        }

        ProcessTier::SilentDaemon
    }

    /// Return processes that are candidates for throttling (non-essential, non-foreground).
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
                    ProcessTier::Telemetry
                        | ProcessTier::Stale
                        | ProcessTier::SilentDaemon
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

// ── Scoring ───────────────────────────────────────────────────────────────────

/// Waste score [0,1] for a process standalone — high CPU + wakeups = wasteful.
/// Used by external callers that don't have the tier yet.
pub fn score_waste(snap: &ProcessSnapshot) -> f32 {
    let mut w = 0.0_f32;
    // High CPU with no GUI = likely wasteful
    if snap.cpu_percent > 5.0 && !snap.has_gui_window {
        w += 0.40;
    }
    // High wakeups
    if snap.wakeups_per_sec > 20.0 {
        w += 0.25;
    }
    // Large RSS in background
    if snap.rss_bytes > 200 * 1024 * 1024 && !snap.has_gui_window {
        w += 0.15;
    }
    // Long stale
    if snap.secs_since_foreground > 3600 && snap.cpu_percent < 1.0 {
        w += 0.10;
    }
    // Penalty reduction for active GUI
    if snap.has_gui_window {
        w -= 0.30;
    }
    w.clamp(0.0, 1.0)
}

/// Utility score [0,1]: how valuable is this process to the user right now?
/// Higher = keep alive. Lower = freeze/throttle candidate.
pub fn score_utility(snap: &ProcessSnapshot) -> f32 {
    let mut score: f32 = 0.5;

    // GUI presence is the strongest signal
    if snap.has_gui_window {
        score += 0.25;
    }
    if snap.secs_since_user_interaction < 10 {
        score += 0.20;
    }

    // Active network = doing something useful
    if snap.has_network {
        score += 0.05;
    }

    // High CPU = doing real work (or being a nuisance — context-dependent)
    if snap.cpu_percent > 10.0 {
        score += 0.05;
    }

    // Penalty: high wakeups with no GUI = chatty daemon
    if snap.wakeups_per_sec > 50.0 && !snap.has_gui_window {
        score -= 0.15;
    }

    // Penalty: translated binary (Rosetta) = legacy, lower priority
    if snap.is_translated {
        score -= 0.05;
    }

    score.clamp(0.0, 1.0)
}

/// Waste score [0,1]: how wasteful is this process?
/// Higher = stronger candidate for throttle/freeze.
pub fn waste_score(snap: &ProcessSnapshot, tier: ProcessTier) -> f32 {
    match tier {
        ProcessTier::SystemEssential | ProcessTier::ActiveForeground => 0.0,
        ProcessTier::ZombieOrphan => 1.0,
        ProcessTier::Telemetry => 0.85,
        ProcessTier::Stale => 0.70,
        ProcessTier::SilentDaemon => {
            let mut w = 0.30_f32;
            if snap.wakeups_per_sec > 20.0 {
                w += 0.20;
            }
            if snap.rss_bytes > 200 * 1024 * 1024 {
                w += 0.15;
            }
            w.min(0.80)
        }
        ProcessTier::AppHelper => 0.20,
        ProcessTier::BackgroundVisible => 0.10,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(name: &str) -> ProcessSnapshot {
        ProcessSnapshot {
            pid: 100,
            name: name.to_string(),
            cpu_percent: 0.0,
            rss_bytes: 10 * 1024 * 1024,
            is_zombie: false,
            secs_since_foreground: 3600,
            secs_since_user_interaction: 3600,
            has_network: false,
            has_gui_window: false,
            wakeups_per_sec: 0.0,
            parent_alive: true,
            process_uptime_secs: 3600,
            faults_total: 0,
            pageins_total: 0,
            is_translated: false,
            mach_port_count: 0,
        }
    }

    #[test]
    fn essential_process_classified_correctly() {
        let c = ProcessClassifier::new();
        let s = snap("WindowServer");
        assert_eq!(c.classify(&s), ProcessTier::SystemEssential);
    }

    #[test]
    fn zombie_classified_correctly() {
        let c = ProcessClassifier::new();
        let mut s = snap("some_app");
        s.is_zombie = true;
        assert_eq!(c.classify(&s), ProcessTier::ZombieOrphan);
    }

    #[test]
    fn foreground_gui_recent_interaction() {
        let c = ProcessClassifier::new();
        let mut s = snap("MyApp");
        s.has_gui_window = true;
        s.secs_since_user_interaction = 5;
        assert_eq!(c.classify(&s), ProcessTier::ActiveForeground);
    }

    #[test]
    fn background_gui_no_recent_interaction() {
        let c = ProcessClassifier::new();
        let mut s = snap("MusicApp");
        s.has_gui_window = true;
        s.secs_since_user_interaction = 300;
        assert_eq!(c.classify(&s), ProcessTier::BackgroundVisible);
    }

    #[test]
    fn stale_process_no_gui_idle() {
        let c = ProcessClassifier::new();
        let mut s = snap("idled");
        s.secs_since_foreground = 600;
        s.cpu_percent = 0.1;
        s.wakeups_per_sec = 0.5;
        assert_eq!(c.classify(&s), ProcessTier::Stale);
    }

    #[test]
    fn telemetry_process_classified() {
        let c = ProcessClassifier::new();
        let s = snap("DiagnosticReporter");
        assert_eq!(c.classify(&s), ProcessTier::Telemetry);
    }

    #[test]
    fn score_utility_gui_process_high() {
        let mut s = snap("App");
        s.has_gui_window = true;
        s.secs_since_user_interaction = 5;
        assert!(score_utility(&s) > 0.7);
    }

    #[test]
    fn score_utility_chatty_daemon_penalized() {
        let mut s = snap("daemon");
        s.wakeups_per_sec = 100.0;
        assert!(score_utility(&s) < 0.5);
    }

    #[test]
    fn waste_score_zombie_is_one() {
        let s = snap("zombie");
        assert_eq!(waste_score(&s, ProcessTier::ZombieOrphan), 1.0);
    }

    #[test]
    fn waste_score_essential_is_zero() {
        let s = snap("WindowServer");
        assert_eq!(waste_score(&s, ProcessTier::SystemEssential), 0.0);
    }
}
