//! System Log Ingester — reads macOS unified logs for OOM/Jetsam kills and crashes.
//!
//! Every 60 seconds, queries `log show` for Jetsam memory kills and process crashes
//! from the last 2 minutes. Parsed events feed into the hazard model (OOM events)
//! and NARS beliefs (both OOM and crash events).
//!
//! # Design
//! - Uses `std::process::Command` — zero new dependencies
//! - 2-second timeout prevents blocking the daemon
//! - Protected processes are never targeted by signals from the log
//! - Read-only: we only observe, never act on crash/OOM events directly

use std::process::Command;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// A system event parsed from macOS unified logs.
#[derive(Debug, Clone)]
pub enum SystemEvent {
    /// Jetsam/OOM kill — a process was killed by the kernel due to memory pressure.
    OomKill {
        /// Name of the killed process (may be truncated by the kernel).
        process_name: String,
        /// Reason string from the log (e.g., "per-process-limit", "vm-pageshortage").
        reason: String,
    },
    /// Process crash — EXC_CRASH, SIGKILL, SIGABRT, etc.
    Crash {
        /// Name of the crashed process.
        process_name: String,
        /// Signal or exception type.
        signal: String,
    },
}

/// Number of consecutive `log show` query failures before we classify the
/// platform's log subsystem (logd) as unhealthy.
///
/// At the default 60-s poll interval, 3 failures ≈ 3 minutes of silence —
/// long enough to rule out a single transient failure but short enough to
/// catch logd crashes within one human-noticeable timeframe.
///
/// [Nygard 2018] "Release It!" Ch.5 — adjacent subsystems going silent are
/// as informative as error responses; detect them as first-class signals.
pub const PLATFORM_UNHEALTHY_FAIL_THRESHOLD: u32 = 3;

/// Ingester that periodically queries macOS system logs.
pub struct SystemLogIngester {
    /// When we last polled the logs.
    last_poll: Instant,
    /// How often to poll (default 60s).
    poll_interval: Duration,
    /// Maximum time to wait for `log show` (default 2s).
    timeout: Duration,
    /// Total OOM events observed (lifetime counter).
    pub total_oom_events: u64,
    /// Total crash events observed (lifetime counter).
    pub total_crash_events: u64,
    /// Background thread channel receiver. When `Some`, `poll()` drains this
    /// channel non-blockingly. Each message is `Some(events)` on success or
    /// `None` when the underlying `log show` invocation failed, so the main
    /// thread can track query-failure streaks without re-running the query.
    receiver: Option<mpsc::Receiver<Option<Vec<SystemEvent>>>>,
    /// Consecutive `log show` query failures. Reset on any success.
    consecutive_query_failures: u32,
    /// Sticky "platform unhealthy" flag. Set when consecutive failures cross
    /// `PLATFORM_UNHEALTHY_FAIL_THRESHOLD`; cleared on the first success after
    /// the flag was raised (and a single-line stderr transition is logged on
    /// both edges). Exists so the daemon can subtly back off optimisation
    /// aggression when the host OS log pipeline is degraded — a common
    /// precursor to full system freezes.
    platform_unhealthy: bool,
}

impl SystemLogIngester {
    pub fn new() -> Self {
        Self {
            // Start with last_poll in the past so the first poll happens immediately.
            last_poll: Instant::now() - Duration::from_secs(120),
            poll_interval: Duration::from_secs(60),
            timeout: Duration::from_secs(2),
            total_oom_events: 0,
            total_crash_events: 0,
            receiver: None,
            consecutive_query_failures: 0,
            platform_unhealthy: false,
        }
    }

    /// Returns `true` when the platform's log subsystem has been failing long
    /// enough that we should treat it as degraded. Currently driven by
    /// consecutive failed `log show` invocations.
    pub fn is_platform_unhealthy(&self) -> bool {
        self.platform_unhealthy
    }

    /// Consecutive failed query count (for diagnostics/metrics).
    pub fn consecutive_query_failures(&self) -> u32 {
        self.consecutive_query_failures
    }

    /// Record a successful query (empty or not). Clears the failure streak
    /// and emits a "recovered" log line if we were previously unhealthy.
    fn record_query_success(&mut self) {
        self.consecutive_query_failures = 0;
        if self.platform_unhealthy {
            eprintln!("[log_ingester] platform log subsystem recovered (`log show` succeeded)");
            self.platform_unhealthy = false;
        }
    }

    /// Record a failed query. Increments the streak and raises the unhealthy
    /// flag (with a single-line edge-triggered log message) once the streak
    /// crosses `PLATFORM_UNHEALTHY_FAIL_THRESHOLD`.
    fn record_query_failure(&mut self) {
        self.consecutive_query_failures = self.consecutive_query_failures.saturating_add(1);
        if !self.platform_unhealthy
            && self.consecutive_query_failures >= PLATFORM_UNHEALTHY_FAIL_THRESHOLD
        {
            eprintln!(
                "[log_ingester] platform log subsystem degraded: {} consecutive `log show` failures (logd may be unhealthy)",
                self.consecutive_query_failures
            );
            self.platform_unhealthy = true;
        }
    }

    /// Start a background thread that polls `log show` every `poll_interval`.
    /// After calling this, `poll()` becomes non-blocking: it drains the channel
    /// without spawning a subprocess inline. Eliminates 100-300ms spikes in the
    /// daemon hot path when `log show` fires.
    ///
    /// The thread shuts down automatically when the ingester is dropped (sender
    /// disconnect causes the thread loop to exit).
    pub fn start_background(&mut self) {
        let (tx, rx) = mpsc::channel::<Option<Vec<SystemEvent>>>();
        let poll_interval = self.poll_interval;
        let timeout = self.timeout;
        std::thread::Builder::new()
            .name("apollo-log-ingester".to_string())
            .spawn(move || {
                // Wait one full interval before first poll so startup isn't
                // burdened with a potentially slow `log show` call.
                std::thread::sleep(poll_interval);
                loop {
                    let result = Self::run_query_static(timeout);
                    if tx.send(result).is_err() {
                        // Main thread dropped receiver — exit cleanly.
                        return;
                    }
                    std::thread::sleep(poll_interval);
                }
            })
            .ok(); // spawn failure is non-fatal; falls back to blocking poll
        self.receiver = Some(rx);
    }

    /// Static version of `query_logs` + `parse_log_output` for use in background thread.
    ///
    /// Returns `None` when the `log show` invocation itself failed (spawn error,
    /// non-zero exit, timeout). Returns `Some(vec)` (possibly empty) on success,
    /// so the main thread can distinguish "no events" from "query broken" —
    /// the latter is a platform-health signal.
    fn run_query_static(timeout: Duration) -> Option<Vec<SystemEvent>> {
        let result = Command::new("log")
            .args([
                "show",
                "--last",
                "2m",
                "--predicate",
                r#"(subsystem == "com.apple.kernel" AND category == "memorystatus") OR (process == "ReportCrash") OR (eventMessage CONTAINS "Jetsam") OR (eventMessage CONTAINS "EXC_CRASH") OR (eventMessage CONTAINS "SIGKILL")"#,
                "--style",
                "compact",
                "--info",
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn();
        let mut child = match result {
            Ok(c) => c,
            Err(_) => return None,
        };
        match child.wait_timeout(timeout) {
            Ok(Some(status)) if status.success() => {}
            _ => {
                let _ = child.kill();
                return None;
            }
        }
        let output = match child.stdout.take().and_then(|out| {
            use std::io::Read;
            let mut buf = String::new();
            let mut reader = out;
            reader.read_to_string(&mut buf).ok()?;
            Some(buf)
        }) {
            Some(s) => s,
            None => return None,
        };
        Some(Self::parse_log_output(&output))
    }

    /// Check if it's time to poll. If so, query the logs and return any events found.
    /// Returns an empty vec if it's not time yet or if no events were found.
    ///
    /// In background mode (after `start_background()`), this is always non-blocking:
    /// it drains whatever the background thread has collected without spawning a subprocess.
    pub fn poll(&mut self) -> Vec<SystemEvent> {
        // Background mode: drain channel non-blockingly. Each message is
        // `Some(batch)` on success or `None` on query failure; the latter
        // feeds the platform-health detector.
        if self.receiver.is_some() {
            let mut events = Vec::new();
            // Collect all messages first to avoid holding a borrow across
            // the health-tracking calls below (which take &mut self).
            let mut batches = Vec::new();
            if let Some(rx) = &self.receiver {
                while let Ok(msg) = rx.try_recv() {
                    batches.push(msg);
                }
            }
            for msg in batches {
                match msg {
                    Some(batch) => {
                        self.record_query_success();
                        for ev in batch {
                            match &ev {
                                SystemEvent::OomKill { .. } => self.total_oom_events += 1,
                                SystemEvent::Crash { .. } => self.total_crash_events += 1,
                            }
                            events.push(ev);
                        }
                    }
                    None => self.record_query_failure(),
                }
            }
            return events;
        }
        // Blocking fallback (used in tests and when start_background() was not called).
        if self.last_poll.elapsed() < self.poll_interval {
            return Vec::new();
        }
        self.last_poll = Instant::now();

        let output = match self.query_logs() {
            Some(text) => text,
            None => {
                self.record_query_failure();
                return Vec::new();
            }
        };
        self.record_query_success();

        let events = Self::parse_log_output(&output);

        for ev in &events {
            match ev {
                SystemEvent::OomKill { .. } => self.total_oom_events += 1,
                SystemEvent::Crash { .. } => self.total_crash_events += 1,
            }
        }

        events
    }

    /// Run `log show` to get Jetsam and crash events from the last 2 minutes.
    fn query_logs(&self) -> Option<String> {
        // Predicate matches:
        // 1. Jetsam kills (kernel memorystatus subsystem)
        // 2. Process crashes (ReportCrash / exc_crash)
        let result = Command::new("log")
            .args([
                "show",
                "--last",
                "2m",
                "--predicate",
                r#"(subsystem == "com.apple.kernel" AND category == "memorystatus") OR (process == "ReportCrash") OR (eventMessage CONTAINS "Jetsam") OR (eventMessage CONTAINS "EXC_CRASH") OR (eventMessage CONTAINS "SIGKILL")"#,
                "--style",
                "compact",
                "--info",
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn();

        let mut child = match result {
            Ok(c) => c,
            Err(_) => return None,
        };

        // Wait with timeout to prevent blocking the daemon.
        match child.wait_timeout(self.timeout) {
            Ok(Some(status)) if status.success() => {}
            _ => {
                let _ = child.kill();
                return None;
            }
        }

        child
            .stdout
            .take()
            .and_then(|out| {
                use std::io::Read;
                let mut buf = String::new();
                let mut reader = out;
                reader.read_to_string(&mut buf).ok()?;
                Some(buf)
            })
    }

    /// Parse raw log output into structured events.
    /// Handles both compact and ndjson styles.
    pub fn parse_log_output(output: &str) -> Vec<SystemEvent> {
        let mut events = Vec::new();

        for line in output.lines() {
            let lower = line.to_ascii_lowercase();

            // Jetsam / OOM kill detection
            if lower.contains("jetsam") || lower.contains("memorystatus") {
                if let Some(ev) = Self::parse_jetsam_line(line) {
                    events.push(ev);
                }
            }

            // Crash detection
            if lower.contains("exc_crash")
                || lower.contains("sigkill")
                || lower.contains("sigabrt")
                || lower.contains("reportcrash")
            {
                if let Some(ev) = Self::parse_crash_line(line) {
                    events.push(ev);
                }
            }
        }

        events
    }

    /// Extract process name and reason from a Jetsam log line.
    fn parse_jetsam_line(line: &str) -> Option<SystemEvent> {
        // Common patterns:
        // "memorystatus: killing process <pid> [<name>] (per-process-limit)"
        // "Jetsam: killing pid <pid> [<name>]"
        // Look for [name] pattern
        let name = Self::extract_bracketed_name(line)?;
        let reason = if line.contains("per-process-limit") {
            "per-process-limit".to_string()
        } else if line.contains("vm-pageshortage") {
            "vm-pageshortage".to_string()
        } else if line.contains("highwater") {
            "highwater".to_string()
        } else {
            "jetsam".to_string()
        };
        Some(SystemEvent::OomKill {
            process_name: name,
            reason,
        })
    }

    /// Extract process name and signal from a crash log line.
    fn parse_crash_line(line: &str) -> Option<SystemEvent> {
        let name = Self::extract_bracketed_name(line)
            .or_else(|| Self::extract_process_name_heuristic(line))?;
        let signal = if line.contains("EXC_CRASH") {
            "EXC_CRASH".to_string()
        } else if line.contains("SIGKILL") {
            "SIGKILL".to_string()
        } else if line.contains("SIGABRT") {
            "SIGABRT".to_string()
        } else {
            "crash".to_string()
        };
        Some(SystemEvent::Crash {
            process_name: name,
            signal,
        })
    }

    /// Extract a process name enclosed in square brackets: [Name]
    fn extract_bracketed_name(line: &str) -> Option<String> {
        let start = line.find('[')?;
        let end = line[start + 1..].find(']')?;
        let name = &line[start + 1..start + 1 + end];
        // Validate: not a number (pid), not empty, not too long
        if name.is_empty() || name.len() > 64 || name.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        Some(name.to_string())
    }

    /// Heuristic: extract process name from "process: <name>" or "Process <name>"
    fn extract_process_name_heuristic(line: &str) -> Option<String> {
        for prefix in &["process: ", "Process ", "killing "] {
            if let Some(idx) = line.find(prefix) {
                let rest = &line[idx + prefix.len()..];
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
                    .collect();
                if name.len() >= 2 && name.len() <= 64 {
                    return Some(name);
                }
            }
        }
        None
    }
}

/// Extension trait for `std::process::Child` to add wait-with-timeout.
trait ChildExt {
    fn wait_timeout(&mut self, timeout: Duration) -> std::io::Result<Option<std::process::ExitStatus>>;
}

impl ChildExt for std::process::Child {
    fn wait_timeout(&mut self, timeout: Duration) -> std::io::Result<Option<std::process::ExitStatus>> {
        let start = Instant::now();
        loop {
            match self.try_wait()? {
                Some(status) => return Ok(Some(status)),
                None => {
                    if start.elapsed() >= timeout {
                        return Ok(None);
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_unhealthy_flag_rises_after_threshold_failures() {
        let mut ing = SystemLogIngester::new();
        assert!(!ing.is_platform_unhealthy());
        assert_eq!(ing.consecutive_query_failures(), 0);
        for _ in 0..PLATFORM_UNHEALTHY_FAIL_THRESHOLD {
            ing.record_query_failure();
        }
        assert!(ing.is_platform_unhealthy());
        assert_eq!(
            ing.consecutive_query_failures(),
            PLATFORM_UNHEALTHY_FAIL_THRESHOLD
        );
    }

    #[test]
    fn platform_unhealthy_clears_after_success() {
        let mut ing = SystemLogIngester::new();
        for _ in 0..5 {
            ing.record_query_failure();
        }
        assert!(ing.is_platform_unhealthy());
        ing.record_query_success();
        assert!(!ing.is_platform_unhealthy());
        assert_eq!(ing.consecutive_query_failures(), 0);
    }

    #[test]
    fn isolated_failures_below_threshold_do_not_trip_flag() {
        let mut ing = SystemLogIngester::new();
        // Fail twice (below threshold=3), then succeed — never unhealthy.
        ing.record_query_failure();
        ing.record_query_failure();
        assert!(!ing.is_platform_unhealthy());
        ing.record_query_success();
        ing.record_query_failure();
        assert!(!ing.is_platform_unhealthy());
    }

    #[test]
    fn test_parse_jetsam_bracketed_name() {
        let line = "2026-04-06 memorystatus: killing process 1234 [Safari] (per-process-limit) pid: 1234";
        let events = SystemLogIngester::parse_log_output(line);
        assert_eq!(events.len(), 1);
        match &events[0] {
            SystemEvent::OomKill { process_name, reason } => {
                assert_eq!(process_name, "Safari");
                assert_eq!(reason, "per-process-limit");
            }
            _ => panic!("expected OomKill"),
        }
    }

    #[test]
    fn test_parse_jetsam_vm_pageshortage() {
        let line = "Jetsam: killing pid 5678 [com.apple.WebKit] (vm-pageshortage)";
        let events = SystemLogIngester::parse_log_output(line);
        assert_eq!(events.len(), 1);
        match &events[0] {
            SystemEvent::OomKill { process_name, reason } => {
                assert_eq!(process_name, "com.apple.WebKit");
                assert_eq!(reason, "vm-pageshortage");
            }
            _ => panic!("expected OomKill"),
        }
    }

    #[test]
    fn test_parse_crash_exc_crash() {
        let line = "ReportCrash: process [myapp] hit EXC_CRASH";
        let events = SystemLogIngester::parse_log_output(line);
        assert!(!events.is_empty());
        match &events[0] {
            SystemEvent::Crash { process_name, signal } => {
                assert_eq!(process_name, "myapp");
                assert_eq!(signal, "EXC_CRASH");
            }
            _ => panic!("expected Crash"),
        }
    }

    #[test]
    fn test_parse_sigkill() {
        let line = "process: daemon_helper received SIGKILL";
        let events = SystemLogIngester::parse_log_output(line);
        assert!(!events.is_empty());
        match &events[0] {
            SystemEvent::Crash { process_name, signal } => {
                assert_eq!(process_name, "daemon_helper");
                assert_eq!(signal, "SIGKILL");
            }
            _ => panic!("expected Crash"),
        }
    }

    #[test]
    fn test_parse_empty_output() {
        let events = SystemLogIngester::parse_log_output("");
        assert!(events.is_empty());
    }

    #[test]
    fn test_parse_no_events() {
        let line = "2026-04-06 12:00:00 normal log entry with nothing relevant";
        let events = SystemLogIngester::parse_log_output(line);
        assert!(events.is_empty());
    }

    #[test]
    fn test_bracketed_name_skips_pid_only() {
        // [1234] should not be treated as a process name
        assert!(SystemLogIngester::extract_bracketed_name("killed [1234]").is_none());
    }

    #[test]
    fn test_bracketed_name_skips_empty() {
        assert!(SystemLogIngester::extract_bracketed_name("killed []").is_none());
    }

    #[test]
    fn test_poll_respects_interval() {
        let mut ingester = SystemLogIngester::new();
        ingester.last_poll = Instant::now(); // just polled
        let events = ingester.poll();
        assert!(events.is_empty(), "should not poll again so soon");
    }

    #[test]
    fn test_mixed_log_output() {
        let output = "\
memorystatus: killing process 100 [Dropbox] (vm-pageshortage)\n\
normal log line\n\
ReportCrash: process [node] hit EXC_CRASH signal\n\
another normal line";
        let events = SystemLogIngester::parse_log_output(output);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], SystemEvent::OomKill { .. }));
        assert!(matches!(&events[1], SystemEvent::Crash { .. }));
    }

    #[test]
    fn test_lifetime_counters() {
        let ingester = SystemLogIngester::new();
        assert_eq!(ingester.total_oom_events, 0);
        assert_eq!(ingester.total_crash_events, 0);
    }
}
