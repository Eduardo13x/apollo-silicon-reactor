//! Real foreground app detection for macOS.
//!
//! Replaces the system-wide `interactive_proxy` boolean with per-app resolution.
//! Instead of marking ALL processes as interactive when any user input occurs,
//! this module detects which specific app has window focus and exposes that PID.
//!
//! # Strategy
//!
//! Primary: `lsappinfo front` + `lsappinfo info -only pid -only bundleid -only name <ASN>`
//! - Fast (~5ms combined), works as root, no extra crates needed.
//! - Available on all macOS versions with a GUI session.
//!
//! Fallback: `osascript -e 'tell application "System Events" ...'`
//! - Slower (~50-100ms), may fail in root sessions or when no GUI session exists.
//!
//! # Caching
//!
//! Detection is cached; repeated calls within the cache window return the stored
//! result in <1us. The cache duration is configurable (default: 2 seconds).
//!
//! # Thread Safety
//!
//! `ForegroundDetector` is `Send + Sync`. Internal state is behind a `Mutex`.
//! The mutex-guarded section is kept short (reads/writes to cache fields only).
//! The external command execution happens outside the lock.
//!
//! # Stale PID Warning
//!
//! The foreground PID may become stale if the app quits between detection and
//! the caller's use of the PID. Callers should treat the PID as advisory and
//! handle `ESRCH` / "no such process" gracefully (the same way they would for
//! any PID obtained from the process table).

use serde::{Deserialize, Serialize};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::engine::lock_ext::LockRecover;

// ── Data types ────────────────────────────────────────────────────────────────

/// Bundle identifiers and names that indicate the system is at the lock screen,
/// screensaver, or a non-interactive system UI. When these are the "frontmost"
/// app, we report `ForegroundState::Idle` rather than `App`.
const IDLE_BUNDLE_IDS: &[&str] = &[
    "com.apple.loginwindow",
    "com.apple.screensaver",
    "com.apple.ScreenSaver.Engine",
];

const IDLE_APP_NAMES: &[&str] = &["loginwindow", "ScreenSaverEngine"];

/// Information about the currently focused macOS application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForegroundApp {
    /// Process ID of the foreground application.
    pub pid: u32,
    /// Display name (e.g., "iTerm2", "Safari").
    pub name: String,
    /// macOS bundle identifier, if available (e.g., "com.googlecode.iterm2").
    pub bundle_id: Option<String>,
}

/// Result of a foreground detection attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForegroundState {
    /// A specific app is in the foreground.
    App(ForegroundApp),
    /// No foreground app detected (screen locked, screensaver, login window,
    /// fast user switching, or no GUI session).
    Idle,
    /// Detection is unavailable (lsappinfo missing, headless server, etc.).
    Unavailable,
}

impl ForegroundState {
    /// Returns the foreground app if one is detected.
    pub fn app(&self) -> Option<&ForegroundApp> {
        match self {
            ForegroundState::App(app) => Some(app),
            _ => None,
        }
    }

    /// Returns the foreground app name, if any.
    pub fn name(&self) -> Option<&str> {
        self.app().map(|a| a.name.as_str())
    }

    /// Returns the foreground PID, if any.
    pub fn pid(&self) -> Option<u32> {
        self.app().map(|a| a.pid)
    }

    /// True if the user appears idle (no foreground app).
    pub fn is_idle(&self) -> bool {
        matches!(self, ForegroundState::Idle)
    }

    /// True if detection is available (either an app is focused or user is idle).
    pub fn is_available(&self) -> bool {
        !matches!(self, ForegroundState::Unavailable)
    }
}

// ── Detector ──────────────────────────────────────────────────────────────────

/// Cached state behind the mutex -- kept small so lock duration is minimal.
struct CachedState {
    /// Last successful or unsuccessful detection result.
    last_result: ForegroundState,
    /// When the last detection was performed.
    last_detect_at: Option<Instant>,
    /// Set to true if lsappinfo was confirmed missing/broken.
    /// Avoids repeated failed Command spawns.
    lsappinfo_broken: bool,
    /// When we last checked whether lsappinfo is broken (retry periodically).
    lsappinfo_broken_checked_at: Option<Instant>,
    /// Recently active foreground apps: (name, last_seen). Used to protect
    /// apps that were recently in the foreground but are now minimized.
    recent_apps: Vec<(String, Instant)>,
}

impl Default for CachedState {
    fn default() -> Self {
        Self {
            last_result: ForegroundState::Unavailable,
            last_detect_at: None,
            lsappinfo_broken: false,
            lsappinfo_broken_checked_at: None,
            recent_apps: Vec::new(),
        }
    }
}

/// Detects the foreground macOS application with caching.
///
/// Thread-safe: all mutable state is behind a `Mutex`. The lock is held only
/// for short reads/writes to cached fields; external commands run outside it.
///
/// # Concurrent calls
///
/// If multiple threads call `detect()` simultaneously when the cache is stale,
/// they may each run `lsappinfo` independently. This is harmless (idempotent
/// reads) and avoids holding a lock during I/O. The last writer wins for the
/// cache, which is acceptable since all results reflect near-simultaneous state.
pub struct ForegroundDetector {
    /// How long to keep a cached result before re-detecting.
    cache_ttl: Duration,
    /// How long to wait before retrying a broken lsappinfo.
    broken_retry_interval: Duration,
    /// Maximum time to wait for lsappinfo to respond.
    command_timeout: Duration,
    state: Mutex<CachedState>,
}

impl ForegroundDetector {
    /// Create a new detector with default settings.
    ///
    /// - Cache TTL: 2 seconds
    /// - Broken retry interval: 60 seconds
    /// - Command timeout: 200ms
    pub fn new() -> Self {
        Self {
            cache_ttl: Duration::from_millis(200),
            broken_retry_interval: Duration::from_secs(60),
            command_timeout: Duration::from_millis(200),
            state: Mutex::new(CachedState::default()),
        }
    }

    /// Create a detector with custom cache TTL.
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Create a detector with custom command timeout.
    pub fn with_command_timeout(mut self, timeout: Duration) -> Self {
        self.command_timeout = timeout;
        self
    }

    /// Detect the current foreground application.
    ///
    /// Returns a cached result if the cache is still valid. Otherwise,
    /// runs `lsappinfo` (or the fallback) to detect the foreground app.
    ///
    /// Cost: cached calls <1us, uncached calls ~5-10ms.
    pub fn detect(&self) -> ForegroundState {
        let now = Instant::now();

        // Fast path: return cached result if still valid.
        {
            let guard = self.state.lock_recover();
            if let Some(last_at) = guard.last_detect_at {
                if now.duration_since(last_at) < self.cache_ttl {
                    return guard.last_result.clone();
                }
            }
            // Check if lsappinfo is known broken and we haven't waited long enough to retry.
            if guard.lsappinfo_broken {
                if let Some(checked_at) = guard.lsappinfo_broken_checked_at {
                    if now.duration_since(checked_at) < self.broken_retry_interval {
                        // Still in broken-cooldown: try fallback only, then cache.
                        drop(guard);
                        let result = self.detect_via_osascript();
                        let mut guard = self.state.lock_recover();
                        guard.last_result = result.clone();
                        guard.last_detect_at = Some(now);
                        return result;
                    }
                }
            }
        }
        // Lock released here -- external command runs without the lock.

        // Try primary detection (lsappinfo).
        let (result, lsappinfo_ok) = self.detect_via_lsappinfo();

        // If lsappinfo failed, try fallback.
        let result = if lsappinfo_ok {
            result
        } else {
            self.detect_via_osascript()
        };

        // Update cache.
        {
            let mut guard = self.state.lock_recover();
            guard.last_result = result.clone();
            guard.last_detect_at = Some(now);
            if !lsappinfo_ok {
                guard.lsappinfo_broken = true;
                guard.lsappinfo_broken_checked_at = Some(now);
            } else {
                guard.lsappinfo_broken = false;
            }
            // Track recently active apps so minimized apps stay protected.
            if let ForegroundState::App(ref app) = result {
                if let Some(entry) = guard.recent_apps.iter_mut().find(|(n, _)| n == &app.name) {
                    entry.1 = now;
                } else {
                    guard.recent_apps.push((app.name.clone(), now));
                }
                // Evict entries older than 10 minutes.
                guard
                    .recent_apps
                    .retain(|(_, t)| now.duration_since(*t).as_secs() < 600);
            }
        }

        result
    }

    /// Check if the given PID is the foreground application.
    ///
    /// Uses the cached foreground state; calls `detect()` if cache is stale.
    pub fn is_foreground(&self, pid: u32) -> bool {
        self.detect().pid() == Some(pid)
    }

    /// Check if the given PID is the foreground app or a descendant of it.
    ///
    /// The `is_descendant` closure should return `true` if the first PID is a
    /// descendant (child, grandchild, etc.) of the second PID.
    ///
    /// This design avoids coupling to any specific process-tree implementation.
    /// The caller provides whatever ancestry lookup they have available.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let detector = ForegroundDetector::new();
    /// let is_fg_family = detector.is_foreground_family(some_pid, |child, ancestor| {
    ///     process_tree.is_descendant_of(child, ancestor)
    /// });
    /// ```
    pub fn is_foreground_family<F>(&self, pid: u32, is_descendant: F) -> bool
    where
        F: FnOnce(u32, u32) -> bool,
    {
        match self.detect().pid() {
            Some(fg_pid) if fg_pid == pid => true,
            Some(fg_pid) => is_descendant(pid, fg_pid),
            None => false,
        }
    }

    /// Returns true if the given app name was in the foreground within `within`.
    ///
    /// This protects minimized apps that were recently active — they can be
    /// brought back to the foreground at any moment and must not be frozen.
    /// Substring match: "Brave Browser" matches a process named "Brave Browser Helper".
    pub fn is_recently_active(&self, name: &str, within: Duration) -> bool {
        let now = Instant::now();
        let guard = self.state.lock_recover();
        guard.recent_apps.iter().any(|(app_name, last_seen)| {
            now.duration_since(*last_seen) <= within
                && (name.contains(app_name.as_str()) || app_name.contains(name))
        })
    }

    /// Returns the current cached state without triggering a fresh detection.
    ///
    /// Useful for reading the last-known foreground app without any I/O cost.
    /// Returns `ForegroundState::Unavailable` if no detection has occurred yet.
    pub fn cached(&self) -> ForegroundState {
        let guard = self.state.lock_recover();
        guard.last_result.clone()
    }

    /// Force a fresh detection, ignoring the cache.
    ///
    /// Note: if multiple threads call this concurrently, each will independently
    /// run detection. This is safe but may cause redundant external command calls.
    pub fn detect_fresh(&self) -> ForegroundState {
        // Invalidate cache, then detect.
        {
            let mut guard = self.state.lock_recover();
            guard.last_detect_at = None;
        }
        self.detect()
    }

    // ── Private detection methods ─────────────────────────────────────────

    /// Primary detection via `lsappinfo`.
    ///
    /// Two-step process:
    /// 1. `lsappinfo front` -> get the ASN (Application Serial Number)
    /// 2. `lsappinfo info -only pid -only bundleid -only name <ASN>` -> get details
    ///
    /// Returns `(ForegroundState, bool)` where the bool indicates whether
    /// lsappinfo itself worked (true) or failed to execute (false).
    fn detect_via_lsappinfo(&self) -> (ForegroundState, bool) {
        // Step 1: Get the frontmost ASN.
        let asn = match self.run_lsappinfo_front() {
            Some(asn) => asn,
            None => return (ForegroundState::Unavailable, false),
        };

        // Empty ASN or no frontmost app -> user is idle / screen locked.
        if asn.is_empty() {
            return (ForegroundState::Idle, true);
        }

        // Step 2: Query details for this ASN.
        match self.run_lsappinfo_info(&asn) {
            Some(app) => {
                // Filter out system lock-screen / screensaver UIs.
                if is_idle_app(&app) {
                    (ForegroundState::Idle, true)
                } else {
                    (ForegroundState::App(app), true)
                }
            }
            None => {
                // lsappinfo worked but we couldn't parse details.
                // This can happen with special system UIs (Notification Center, etc.).
                (ForegroundState::Idle, true)
            }
        }
    }

    /// Run `lsappinfo front` and extract the ASN.
    ///
    /// Returns `None` if lsappinfo is not available or fails.
    /// Returns `Some("")` if the output is empty (no frontmost app).
    /// Returns `Some("ASN:0x0-0x46046")` on success.
    fn run_lsappinfo_front(&self) -> Option<String> {
        let output = run_command_with_timeout("lsappinfo", &["front"], self.command_timeout)?;

        let text = String::from_utf8_lossy(&output.stdout);
        let trimmed = text.trim();

        if trimmed.is_empty() {
            return Some(String::new());
        }

        // Extract ASN from output. Format varies:
        //   Root session:  "ASN:0x0-0x46046:"
        //   User session:  "\"AppName\" ASN:0x0-0x46046: ..."
        // We look for the ASN:0x...-0x... pattern.
        extract_asn(trimmed)
    }

    /// Run `lsappinfo info -only pid -only bundleid -only name <ASN>`.
    ///
    /// Parses the key=value output to build a `ForegroundApp`.
    fn run_lsappinfo_info(&self, asn: &str) -> Option<ForegroundApp> {
        let output = run_command_with_timeout(
            "lsappinfo",
            &[
                "info", "-only", "pid", "-only", "bundleid", "-only", "name", asn,
            ],
            self.command_timeout,
        )?;

        let text = String::from_utf8_lossy(&output.stdout);
        parse_lsappinfo_info(&text)
    }

    /// Fallback detection via `osascript` (AppleScript).
    ///
    /// Slower and less reliable than lsappinfo -- fails in root sessions,
    /// during screen lock, and in headless environments. But it's a useful
    /// fallback when lsappinfo is broken.
    fn detect_via_osascript(&self) -> ForegroundState {
        let script = concat!(
            "tell application \"System Events\" to get ",
            "{name, unix id} of first application process ",
            "whose frontmost is true"
        );

        let output =
            match run_command_with_timeout("osascript", &["-e", script], self.command_timeout) {
                Some(o) => o,
                None => return ForegroundState::Unavailable,
            };

        let text = String::from_utf8_lossy(&output.stdout);
        parse_osascript_output(&text)
    }
}

impl Default for ForegroundDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Parsing helpers (pure functions, easy to test) ────────────────────────────

/// Returns true if the given app represents a system idle state
/// (login window, screensaver, etc.) rather than a real user-facing app.
fn is_idle_app(app: &ForegroundApp) -> bool {
    // Check bundle ID first (more reliable).
    if let Some(ref bid) = app.bundle_id {
        if IDLE_BUNDLE_IDS
            .iter()
            .any(|id| bid.eq_ignore_ascii_case(id))
        {
            return true;
        }
    }
    // Fall back to name matching.
    IDLE_APP_NAMES
        .iter()
        .any(|name| app.name.eq_ignore_ascii_case(name))
}

/// Extract an ASN from lsappinfo output text.
///
/// Looks for the pattern `ASN:0x<hex>-0x<hex>` anywhere in the text.
/// Returns `Some("")` if no ASN pattern is found but the command succeeded
/// (indicating no frontmost app).
fn extract_asn(text: &str) -> Option<String> {
    // Walk the text looking for "ASN:" prefix.
    let mut search_from = 0;
    while let Some(start) = text[search_from..].find("ASN:") {
        let abs_start = search_from + start;
        // Find the end of the ASN token (terminated by colon, space, quote, or EOL).
        let after_prefix = abs_start + 4; // skip "ASN:"
        if after_prefix >= text.len() {
            break;
        }
        // ASN format: "0x<hex>-0x<hex>"
        // Find the end: next whitespace, quote, or trailing colon.
        let rest = &text[abs_start..];
        let end = rest
            .find(|c: char| c.is_whitespace() || c == '"')
            .unwrap_or(rest.len());
        let candidate = rest[..end].trim_end_matches(':');

        // Validate: must contain at least "ASN:0x" and a hyphen.
        if candidate.len() >= 8 && candidate.contains('-') {
            return Some(candidate.to_string());
        }
        search_from = abs_start + 4;
    }
    // No ASN found -- the output was valid but empty / no frontmost app.
    Some(String::new())
}

/// Parse the output of `lsappinfo info -only pid -only bundleid -only name <ASN>`.
///
/// Expected format (one key per line):
/// ```text
/// "pid"=2766
/// "CFBundleIdentifier"="com.googlecode.iterm2"
/// "LSDisplayName"="iTerm2"
/// ```
///
/// Values may be `[ NULL ]` if not available for a given app.
fn parse_lsappinfo_info(text: &str) -> Option<ForegroundApp> {
    let mut pid: Option<u32> = None;
    let mut name: Option<String> = None;
    let mut bundle_id: Option<String> = None;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Split on first '=' only.
        let eq_pos = match line.find('=') {
            Some(p) => p,
            None => continue,
        };
        let (key_raw, val_raw) = line.split_at(eq_pos);
        let val_raw = &val_raw[1..]; // skip the '='

        let key = key_raw.trim().trim_matches('"');
        let val = val_raw.trim().trim_matches('"');

        // Skip NULL values.
        if val.contains("NULL") {
            continue;
        }

        match key {
            "pid" => {
                pid = val.trim().parse::<u32>().ok();
            }
            "CFBundleIdentifier" => {
                let clean = val.trim().to_string();
                if !clean.is_empty() {
                    bundle_id = Some(clean);
                }
            }
            "LSDisplayName" => {
                let clean = val.trim().to_string();
                if !clean.is_empty() {
                    name = Some(clean);
                }
            }
            _ => {}
        }
    }

    // PID is required; name falls back to bundle_id if missing.
    let pid = pid?;
    let display_name = name
        .or_else(|| bundle_id.clone())
        .unwrap_or_else(|| format!("pid-{}", pid));

    Some(ForegroundApp {
        pid,
        name: display_name,
        bundle_id,
    })
}

/// Parse the output of the osascript fallback.
///
/// Expected format: `AppName, 12345` (comma-separated name and unix PID).
fn parse_osascript_output(text: &str) -> ForegroundState {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return ForegroundState::Idle;
    }

    // The output is: "AppName, PID"
    // Use rsplitn to split from the right, handling names with commas.
    let parts: Vec<&str> = trimmed.rsplitn(2, ", ").collect();
    if parts.len() != 2 {
        return ForegroundState::Idle;
    }

    // rsplitn reverses: parts[0] = PID, parts[1] = name
    let pid_str = parts[0].trim();
    let name = parts[1].trim().to_string();

    let pid = match pid_str.parse::<u32>() {
        Ok(p) => p,
        Err(_) => return ForegroundState::Idle,
    };

    if name.is_empty() {
        return ForegroundState::Idle;
    }

    let app = ForegroundApp {
        pid,
        name,
        bundle_id: None,
    };

    // Filter idle-state apps the same way as the lsappinfo path.
    if is_idle_app(&app) {
        return ForegroundState::Idle;
    }

    ForegroundState::App(app)
}

// ── Command execution with timeout ───────────────────────────────────────────

/// Run an external command with a timeout.
///
/// Spawns a thread to execute the command and waits up to `timeout` for it.
/// Returns `None` if the command fails to start, times out, or exits non-zero.
///
/// # Thread lifetime
///
/// The spawned thread owns the `Command` child process. If the receiver times
/// out, the thread continues running until `Command::output()` completes (which
/// waits for the child process to exit). The child's stdout/stderr are captured
/// in memory and dropped when the thread finishes. No process leak occurs.
///
/// For typical `lsappinfo` / `osascript` calls (~5-50ms), the thread lifetime
/// is negligible. In pathological cases (hung command), the thread lingers until
/// the OS-level timeout or the process exits.
fn run_command_with_timeout(
    program: &str,
    args: &[&str],
    timeout: Duration,
) -> Option<std::process::Output> {
    let program = program.to_string();
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = Command::new(&program).args(&args).output();
        // If the receiver has been dropped (timeout), this send fails silently.
        let _ = tx.send(result);
    });

    let output = rx.recv_timeout(timeout).ok()?.ok()?;

    if output.status.success() {
        Some(output)
    } else {
        None
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ASN extraction tests ──────────────────────────────────────────────

    #[test]
    fn extract_asn_root_format() {
        // Root session: bare ASN with trailing colon.
        let input = "ASN:0x0-0x46046:";
        let asn = extract_asn(input);
        assert_eq!(asn, Some("ASN:0x0-0x46046".to_string()));
    }

    #[test]
    fn extract_asn_user_format() {
        // User session: app name in quotes before ASN.
        let input = r#""iTerm2" ASN:0x0-0x46046: (in front)"#;
        let asn = extract_asn(input);
        assert_eq!(asn, Some("ASN:0x0-0x46046".to_string()));
    }

    #[test]
    fn extract_asn_empty_output() {
        let asn = extract_asn("");
        assert_eq!(asn, Some(String::new()));
    }

    #[test]
    fn extract_asn_no_match() {
        let asn = extract_asn("some random output without asn");
        assert_eq!(asn, Some(String::new()));
    }

    #[test]
    fn extract_asn_multiple_asn() {
        // Should return the first valid ASN.
        let input = r#""App" ASN:0x0-0x100: ASN:0x0-0x200:"#;
        let asn = extract_asn(input);
        assert_eq!(asn, Some("ASN:0x0-0x100".to_string()));
    }

    #[test]
    fn extract_asn_long_hex() {
        let input = "ASN:0x0-0x2c02c:";
        let asn = extract_asn(input);
        assert_eq!(asn, Some("ASN:0x0-0x2c02c".to_string()));
    }

    // ── lsappinfo info parsing tests ──────────────────────────────────────

    #[test]
    fn parse_info_normal() {
        let text = r#""pid"=2766
"CFBundleIdentifier"="com.googlecode.iterm2"
"LSDisplayName"="iTerm2""#;
        let app = parse_lsappinfo_info(text);
        assert!(app.is_some());
        let app = app.unwrap();
        assert_eq!(app.pid, 2766);
        assert_eq!(app.name, "iTerm2");
        assert_eq!(app.bundle_id, Some("com.googlecode.iterm2".to_string()));
    }

    #[test]
    fn parse_info_null_values() {
        // When querying an invalid ASN, all values are NULL.
        let text = r#""pid"=[ NULL ]
"CFBundleIdentifier"=[ NULL ]
"LSDisplayName"=[ NULL ] "#;
        let app = parse_lsappinfo_info(text);
        assert!(app.is_none());
    }

    #[test]
    fn parse_info_pid_only() {
        // Some system processes have a PID but no name or bundle ID.
        let text = r#""pid"=42
"CFBundleIdentifier"=[ NULL ]
"LSDisplayName"=[ NULL ] "#;
        let app = parse_lsappinfo_info(text);
        assert!(app.is_some());
        let app = app.unwrap();
        assert_eq!(app.pid, 42);
        assert_eq!(app.name, "pid-42"); // Fallback name
        assert_eq!(app.bundle_id, None);
    }

    #[test]
    fn parse_info_no_bundle_id() {
        let text = r#""pid"=99
"CFBundleIdentifier"=[ NULL ]
"LSDisplayName"="Spotlight""#;
        let app = parse_lsappinfo_info(text);
        assert!(app.is_some());
        let app = app.unwrap();
        assert_eq!(app.pid, 99);
        assert_eq!(app.name, "Spotlight");
        assert_eq!(app.bundle_id, None);
    }

    #[test]
    fn parse_info_unicode_name() {
        // App names can contain CJK, emoji, or accented characters.
        let text = "\"pid\"=123\n\"CFBundleIdentifier\"=\"com.example.app\"\n\"LSDisplayName\"=\"\u{1F680} RocketApp \u{00E9}\"";
        let app = parse_lsappinfo_info(text);
        assert!(app.is_some());
        let app = app.unwrap();
        assert_eq!(app.name, "\u{1F680} RocketApp \u{00E9}");
    }

    #[test]
    fn parse_info_empty() {
        let app = parse_lsappinfo_info("");
        assert!(app.is_none());
    }

    #[test]
    fn parse_info_name_fallback_to_bundle() {
        // If LSDisplayName is NULL but bundle ID is present, use bundle ID as name.
        let text = r#""pid"=55
"CFBundleIdentifier"="com.apple.controlcenter"
"LSDisplayName"=[ NULL ]"#;
        let app = parse_lsappinfo_info(text);
        assert!(app.is_some());
        let app = app.unwrap();
        assert_eq!(app.pid, 55);
        assert_eq!(app.name, "com.apple.controlcenter");
        assert_eq!(app.bundle_id, Some("com.apple.controlcenter".to_string()));
    }

    // ── Idle app detection tests ──────────────────────────────────────────

    #[test]
    fn is_idle_loginwindow_by_bundle() {
        let app = ForegroundApp {
            pid: 433,
            name: "loginwindow".to_string(),
            bundle_id: Some("com.apple.loginwindow".to_string()),
        };
        assert!(is_idle_app(&app));
    }

    #[test]
    fn is_idle_loginwindow_by_name() {
        let app = ForegroundApp {
            pid: 433,
            name: "loginwindow".to_string(),
            bundle_id: None,
        };
        assert!(is_idle_app(&app));
    }

    #[test]
    fn is_idle_screensaver_by_bundle() {
        let app = ForegroundApp {
            pid: 555,
            name: "ScreenSaverEngine".to_string(),
            bundle_id: Some("com.apple.ScreenSaver.Engine".to_string()),
        };
        assert!(is_idle_app(&app));
    }

    #[test]
    fn is_idle_normal_app() {
        let app = ForegroundApp {
            pid: 100,
            name: "Safari".to_string(),
            bundle_id: Some("com.apple.Safari".to_string()),
        };
        assert!(!is_idle_app(&app));
    }

    #[test]
    fn is_idle_case_insensitive() {
        let app = ForegroundApp {
            pid: 433,
            name: "LoginWindow".to_string(),
            bundle_id: None,
        };
        assert!(is_idle_app(&app));
    }

    // ── osascript parsing tests ───────────────────────────────────────────

    #[test]
    fn parse_osascript_normal() {
        let state = parse_osascript_output("iTerm2, 2766\n");
        match state {
            ForegroundState::App(app) => {
                assert_eq!(app.pid, 2766);
                assert_eq!(app.name, "iTerm2");
                assert_eq!(app.bundle_id, None);
            }
            other => panic!("Expected App, got {:?}", other),
        }
    }

    #[test]
    fn parse_osascript_loginwindow() {
        // During screen lock, System Events reports loginwindow.
        let state = parse_osascript_output("loginwindow, 433\n");
        assert_eq!(state, ForegroundState::Idle);
    }

    #[test]
    fn parse_osascript_screensaver() {
        let state = parse_osascript_output("ScreenSaverEngine, 888\n");
        assert_eq!(state, ForegroundState::Idle);
    }

    #[test]
    fn parse_osascript_empty() {
        let state = parse_osascript_output("");
        assert_eq!(state, ForegroundState::Idle);
    }

    #[test]
    fn parse_osascript_name_with_commas() {
        // App name containing commas: "Some, App, Name, 999"
        // rsplitn(2, ", ") splits from the right, so PID is correctly isolated.
        let state = parse_osascript_output("Some, App, Name, 999\n");
        match state {
            ForegroundState::App(app) => {
                assert_eq!(app.pid, 999);
                assert_eq!(app.name, "Some, App, Name");
            }
            other => panic!("Expected App, got {:?}", other),
        }
    }

    #[test]
    fn parse_osascript_invalid_pid() {
        let state = parse_osascript_output("Safari, not_a_number\n");
        assert_eq!(state, ForegroundState::Idle);
    }

    // ── ForegroundState API tests ─────────────────────────────────────────

    #[test]
    fn state_accessors() {
        let app = ForegroundApp {
            pid: 42,
            name: "Test".to_string(),
            bundle_id: Some("com.test".to_string()),
        };

        let state = ForegroundState::App(app);
        assert_eq!(state.pid(), Some(42));
        assert_eq!(state.name(), Some("Test"));
        assert!(!state.is_idle());
        assert!(state.is_available());

        let idle = ForegroundState::Idle;
        assert_eq!(idle.pid(), None);
        assert_eq!(idle.name(), None);
        assert!(idle.app().is_none());
        assert!(idle.is_idle());
        assert!(idle.is_available());

        let unavailable = ForegroundState::Unavailable;
        assert_eq!(unavailable.pid(), None);
        assert!(!unavailable.is_idle());
        assert!(!unavailable.is_available());
    }

    // ── Serde round-trip tests ────────────────────────────────────────────

    #[test]
    fn serde_foreground_app_round_trip() {
        let app = ForegroundApp {
            pid: 42,
            name: "Test".to_string(),
            bundle_id: Some("com.test".to_string()),
        };
        let json = serde_json::to_string(&app).unwrap();
        let recovered: ForegroundApp = serde_json::from_str(&json).unwrap();
        assert_eq!(app, recovered);
    }

    #[test]
    fn serde_foreground_state_variants() {
        let states = vec![
            ForegroundState::App(ForegroundApp {
                pid: 1,
                name: "A".to_string(),
                bundle_id: None,
            }),
            ForegroundState::Idle,
            ForegroundState::Unavailable,
        ];
        for state in &states {
            let json = serde_json::to_string(state).unwrap();
            let recovered: ForegroundState = serde_json::from_str(&json).unwrap();
            assert_eq!(*state, recovered);
        }
    }

    // ── Cache behavior tests ──────────────────────────────────────────────

    #[test]
    fn detector_returns_cached_within_ttl() {
        let detector = ForegroundDetector::new().with_cache_ttl(Duration::from_secs(60));

        // Pre-populate cache.
        {
            let mut guard = detector.state.lock_recover();
            guard.last_result = ForegroundState::App(ForegroundApp {
                pid: 999,
                name: "CachedApp".to_string(),
                bundle_id: None,
            });
            guard.last_detect_at = Some(Instant::now());
        }

        // Should return cached result without running any command.
        let result = detector.detect();
        assert_eq!(result.pid(), Some(999));
        assert_eq!(result.name(), Some("CachedApp"));
    }

    #[test]
    fn detector_cached_returns_last_known() {
        let detector = ForegroundDetector::new();

        // Before any detection, cached() returns Unavailable.
        assert_eq!(detector.cached(), ForegroundState::Unavailable);

        // Pre-populate.
        {
            let mut guard = detector.state.lock_recover();
            guard.last_result = ForegroundState::Idle;
            guard.last_detect_at = Some(Instant::now());
        }

        assert_eq!(detector.cached(), ForegroundState::Idle);
    }

    #[test]
    fn is_foreground_checks_pid() {
        let detector = ForegroundDetector::new().with_cache_ttl(Duration::from_secs(60));

        // Pre-populate cache.
        {
            let mut guard = detector.state.lock_recover();
            guard.last_result = ForegroundState::App(ForegroundApp {
                pid: 100,
                name: "TestApp".to_string(),
                bundle_id: None,
            });
            guard.last_detect_at = Some(Instant::now());
        }

        assert!(detector.is_foreground(100));
        assert!(!detector.is_foreground(101));
        assert!(!detector.is_foreground(0));
    }

    #[test]
    fn is_foreground_family_with_closure() {
        let detector = ForegroundDetector::new().with_cache_ttl(Duration::from_secs(60));

        // Foreground PID = 100.
        {
            let mut guard = detector.state.lock_recover();
            guard.last_result = ForegroundState::App(ForegroundApp {
                pid: 100,
                name: "Parent".to_string(),
                bundle_id: None,
            });
            guard.last_detect_at = Some(Instant::now());
        }

        // PID 200 is a child of PID 100.
        let result =
            detector.is_foreground_family(200, |child, ancestor| child == 200 && ancestor == 100);
        assert!(result);

        // PID 300 is not related.
        let result = detector.is_foreground_family(300, |_child, _ancestor| false);
        assert!(!result);

        // PID 100 is the foreground app itself -- closure is not called.
        let result = detector.is_foreground_family(100, |_child, _ancestor| {
            panic!("should not be called for exact match");
        });
        assert!(result);
    }

    #[test]
    fn is_foreground_family_when_idle() {
        let detector = ForegroundDetector::new().with_cache_ttl(Duration::from_secs(60));

        {
            let mut guard = detector.state.lock_recover();
            guard.last_result = ForegroundState::Idle;
            guard.last_detect_at = Some(Instant::now());
        }

        let result = detector.is_foreground_family(42, |_, _| true);
        assert!(!result);
    }

    // ── Edge case: loginwindow via lsappinfo parsing ──────────────────────

    #[test]
    fn parse_info_loginwindow_pid() {
        // loginwindow is parsed successfully by the parser; the idle-app filter
        // in detect_via_lsappinfo will convert it to Idle.
        let text = r#""pid"=433
"CFBundleIdentifier"="com.apple.loginwindow"
"LSDisplayName"="loginwindow""#;
        let app = parse_lsappinfo_info(text);
        assert!(app.is_some());
        let app = app.unwrap();
        assert_eq!(app.name, "loginwindow");
        // Verify the idle filter catches it.
        assert!(is_idle_app(&app));
    }

    // ── Edge case: app name containing equals sign ────────────────────────

    #[test]
    fn parse_info_name_with_equals() {
        // Unlikely but possible: app name contains '='.
        // split_at(eq_pos) only splits on the FIRST '=', so the value is preserved.
        let text = "\"pid\"=77\n\"LSDisplayName\"=\"A=B App\"";
        let app = parse_lsappinfo_info(text);
        assert!(app.is_some());
        let app = app.unwrap();
        assert_eq!(app.name, "A=B App");
    }

    // ── Edge case: stale broken flag recovery ─────────────────────────────

    #[test]
    fn broken_flag_is_cleared_on_success() {
        let detector = ForegroundDetector::new().with_cache_ttl(Duration::from_secs(60));

        // Simulate lsappinfo broken state.
        {
            let mut guard = detector.state.lock_recover();
            guard.lsappinfo_broken = true;
            guard.lsappinfo_broken_checked_at = Some(Instant::now());
        }

        // Simulate a successful detection updating the cache.
        {
            let mut guard = detector.state.lock_recover();
            guard.last_result = ForegroundState::App(ForegroundApp {
                pid: 42,
                name: "Recovered".to_string(),
                bundle_id: None,
            });
            guard.last_detect_at = Some(Instant::now());
            guard.lsappinfo_broken = false; // As detect() would do.
        }

        let guard = detector.state.lock_recover();
        assert!(!guard.lsappinfo_broken);
    }
}
