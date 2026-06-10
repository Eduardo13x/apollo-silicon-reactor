//! Realtime user-activity signals beyond the CoreAudio full-duplex probe.
//!
//! B.2 (2026-06-09 prod incident follow-up): the Meet-call guard landed in
//! `257baea` keyed exclusively on `coreaudio_active::is_realtime_call_active`
//! (default-output AND default-input both running). That misses the
//! *screen-share* leg of a call: post-screen-share traffic whipsaw showed
//! "high retransmissions scaling UP" immediately followed by "low
//! retransmissions -25% scale-down". Screen capture on macOS is mediated by
//! a small set of system processes (`replayd`, `screencaptureui`,
//! `ScreenSharingAgent`) — their presence is a cheap, reliable proxy for an
//! active capture session.
//!
//! The proc-table scan costs ~2-4ms over ~600 pids, which is too expensive
//! to pay every daemon cycle inside the 100ms p95 budget. `ScreenCaptureCache`
//! amortises it behind a 6s TTL — capture sessions last minutes, so a 6s
//! staleness window is harmless while cutting the scan to ~once per 6s.

use std::time::{Duration, Instant};

use crate::engine::proc_taskinfo::{get_proc_path, list_all_pids};

/// How long a scan result stays valid before the proc table is re-scanned.
/// 6s balances detection latency (screen shares last minutes) against the
/// ~2-4ms scan cost on the daemon hot path (p95 budget 100ms).
const SCREEN_CAPTURE_TTL: Duration = Duration::from_secs(6);

/// Executable-path suffixes that indicate an active screen-capture session.
/// Matched against the *full* proc_pidpath so a user binary named e.g.
/// `myreplayd-tool` cannot spoof the gate — the path must end with the
/// exact `/name` component.
fn is_screen_capture_path(path: &str) -> bool {
    path.ends_with("/replayd")
        || path.ends_with("/screencaptureui")
        || path.ends_with("/ScreenSharingAgent")
}

/// TTL-cached detector for active screen capture (replayd et al.).
///
/// Lives as per-loop `mut` state in the daemon (same pattern as
/// `last_netstat_tick`); NOT a global — the cache is single-consumer.
#[derive(Debug)]
pub struct ScreenCaptureCache {
    last_check: Option<Instant>,
    cached: bool,
}

impl Default for ScreenCaptureCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ScreenCaptureCache {
    pub fn new() -> Self {
        Self {
            last_check: None,
            cached: false,
        }
    }

    /// Returns true when a screen-capture process is running.
    ///
    /// Re-scans the proc table at most once per `SCREEN_CAPTURE_TTL`;
    /// otherwise returns the cached verdict. The scan early-returns on the
    /// first matching pid.
    pub fn check(&mut self) -> bool {
        if let Some(at) = self.last_check {
            if at.elapsed() < SCREEN_CAPTURE_TTL {
                return self.cached;
            }
        }
        self.cached = scan_for_screen_capture();
        self.last_check = Some(Instant::now());
        self.cached
    }
}

/// One full proc-table pass: ~2-4ms over ~600 pids. Early-returns true on
/// the first screen-capture binary found. Pids that vanish mid-scan (or
/// deny proc_pidpath) are skipped — best-effort per daemon discipline.
fn scan_for_screen_capture() -> bool {
    for pid in list_all_pids() {
        if let Some(path) = get_proc_path(pid) {
            if is_screen_capture_path(&path) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screen_capture_path_matches_exact_suffixes() {
        assert!(is_screen_capture_path("/usr/libexec/replayd"));
        assert!(is_screen_capture_path(
            "/System/Library/CoreServices/screencaptureui"
        ));
        assert!(is_screen_capture_path(
            "/System/Library/CoreServices/RemoteManagement/ScreenSharingAgent"
        ));
    }

    #[test]
    fn screen_capture_path_rejects_lookalikes_and_substrings() {
        // Suffix match must require the exact `/name` component.
        assert!(!is_screen_capture_path("/usr/local/bin/myreplayd-tool"));
        assert!(!is_screen_capture_path("/usr/local/bin/notreplayd"));
        assert!(!is_screen_capture_path("/usr/libexec/replayd/helper"));
        assert!(!is_screen_capture_path(""));
        assert!(!is_screen_capture_path("/usr/sbin/cfprefsd"));
    }

    #[test]
    fn check_does_not_panic_and_returns_bool() {
        // Live proc-table scan on the test machine — outcome is environment
        // dependent (replayd may or may not be running); we only assert the
        // scan completes and the cache records the verdict.
        let mut cache = ScreenCaptureCache::new();
        let first = cache.check();
        assert!(cache.last_check.is_some());
        assert_eq!(cache.cached, first);
    }

    #[test]
    fn ttl_cache_returns_cached_value_within_window() {
        // Force a cached verdict that almost certainly disagrees with a live
        // scan in CI (cached=true) — a fresh TTL must short-circuit the scan
        // and echo the cached value back.
        let mut cache = ScreenCaptureCache {
            last_check: Some(Instant::now()),
            cached: true,
        };
        assert!(cache.check(), "fresh TTL must return cached value");

        // Same shape with cached=false to prove the short-circuit isn't a
        // constant-true accident.
        let mut cache = ScreenCaptureCache {
            last_check: Some(Instant::now()),
            cached: false,
        };
        assert!(!cache.check(), "fresh TTL must return cached value");
    }

    #[test]
    fn ttl_cache_rescans_after_window_expires() {
        let stale = Instant::now() - SCREEN_CAPTURE_TTL - Duration::from_secs(1);
        let mut cache = ScreenCaptureCache {
            last_check: Some(stale),
            cached: true,
        };
        let _ = cache.check();
        // The rescan must refresh the timestamp regardless of verdict.
        assert!(
            cache.last_check.expect("timestamp refreshed").elapsed() < Duration::from_secs(2),
            "expired TTL must trigger a rescan and refresh last_check"
        );
    }
}
