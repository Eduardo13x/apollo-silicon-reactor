//! User Context Collector — "telepathy" signals about what the user is doing.
//!
//! Collects 4 lightweight signals invisible to pure process metrics:
//!
//! - `idle_secs`: seconds since last keyboard/mouse event (IOHIDSystem HIDIdleTime)
//! - `has_sleep_assertion`: any non-Apollo sleep-prevention assertion active
//! - `call_in_progress`: video/audio call detected via pmset assertion owner
//! - `audio_active`: audio output currently playing (coreaudiod NoIdleSleepAssertion)
//!
//! **Collection cost:** one `ioreg` subprocess (~2ms) + one `pmset` subprocess (~5ms).
//! The caller should rate-limit `pmset` to every N cycles; `ioreg` is safe every cycle.
//!
//! [Riva & Mantovani 2014] "User context awareness for mobile computing" —
//! idle time + media state are the two highest-signal contextual cues.

use serde::{Deserialize, Serialize};
#[cfg(target_os = "macos")]
use std::process::Command;
use std::sync::Mutex;
use std::time::Instant;

/// App names that indicate an active video/audio call.
#[cfg(target_os = "macos")]
const CALL_APP_NAMES: &[&str] = &[
    "zoom.us", "facetime", "teams", "webex", "skype", "discord", "meet", "slack", "whereby",
    "around", "loom",
];

// ── HID event-rate tracker (Phase 5.1-D) ─────────────────────────────────────
//
// macOS exposes `HIDIdleTime` (nanoseconds since last keyboard / mouse event)
// but does NOT expose a monotonic "events since boot" counter. We approximate
// the event rate by observing **resets** of HIDIdleTime between consecutive
// samples: when the new reading is *smaller* than the previous one, at least
// one HID event fired in the interval. Each sampled cycle therefore contributes
// a binary observation `(reset_detected, wall_clock_at_sample)`. The public
// accessor [`hid_events_per_minute`] converts a rolling window of these
// observations into a normalised events-per-minute estimate.
//
// Why 30 samples: the daemon's cognitive tick samples `UserContext::collect()`
// once every ~10 main cycles (≈ 20 s on a healthy daemon). 30 real samples
// therefore spans ≈ 10 min of wall-clock, long enough to smooth bursty input
// (a user typing for 8 s in the middle of an otherwise idle minute) without
// lagging the modulator past the 2-min "is_idle_long" threshold the gate
// pipeline uses to swap policy tiers.
//
// Bounded per-cycle work: O(1) sample, O(30) for the rate read.
// Memory: 30 × `Sample` = 30 × 16 B = 480 B + Mutex overhead.

const HID_RATE_WINDOW: usize = 30;
/// Minimum observed wall-clock span required before reporting a non-zero rate.
/// Below this the divisor is too small (and noisy) to produce a stable rate;
/// returning 0.0 keeps the modulator on its idle-time fallback during the
/// daemon's first few minutes of operation.
const HID_RATE_MIN_SPAN_SECS: f64 = 5.0;

#[derive(Clone, Copy)]
struct HidSample {
    /// True if the HIDIdleTime reading was lower than the previous reading,
    /// i.e. at least one keyboard / mouse event fired between samples.
    reset: bool,
    /// Wall-clock instant at which this sample was taken — used by
    /// [`hid_events_per_minute`] to compute the actual window span.
    at: Instant,
}

struct HidEventRateTracker {
    /// Most recent HIDIdleTime reading (seconds). `None` until first sample.
    last_idle_secs: Option<f64>,
    /// Rolling window of binary reset observations (newest = back).
    samples: std::collections::VecDeque<HidSample>,
}

impl HidEventRateTracker {
    const fn new() -> Self {
        Self {
            last_idle_secs: None,
            samples: std::collections::VecDeque::new(),
        }
    }

    /// Record one HID idle sample. `O(1)`. The first sample is bootstrap-only
    /// (no previous reading to compare against) and registers `reset = false`.
    fn observe(&mut self, current_idle_secs: f64, now: Instant) {
        let reset = match self.last_idle_secs {
            // A drop of any magnitude indicates at least one HID event fired.
            Some(prev) => current_idle_secs < prev,
            None => false,
        };
        self.last_idle_secs = Some(current_idle_secs);
        if self.samples.len() == HID_RATE_WINDOW {
            self.samples.pop_front();
        }
        self.samples.push_back(HidSample { reset, at: now });
    }

    /// Convert the window into an events-per-minute estimate. `O(window)`.
    ///
    /// Returns 0.0 when the window is empty, when the observed span is too
    /// short to be meaningful (< [`HID_RATE_MIN_SPAN_SECS`]), or when the
    /// computed span would be non-positive (clock anomaly / single sample).
    fn events_per_minute(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let first = self.samples.front().expect("checked len >= 2").at;
        let last = self.samples.back().expect("checked len >= 2").at;
        // `Instant::saturating_duration_since` guarantees non-negative.
        let span_secs = last.saturating_duration_since(first).as_secs_f64();
        if !span_secs.is_finite() || span_secs < HID_RATE_MIN_SPAN_SECS {
            return 0.0;
        }
        let resets = self.samples.iter().filter(|s| s.reset).count() as f64;
        let per_minute = resets * 60.0 / span_secs;
        // Defensive: NaN/inf cannot occur given the span guard above, but be
        // explicit so the modulator never receives a poisoned input.
        if per_minute.is_finite() {
            per_minute
        } else {
            0.0
        }
    }

    #[cfg(test)]
    fn reset_for_tests(&mut self) {
        self.last_idle_secs = None;
        self.samples.clear();
    }
}

static HID_RATE_TRACKER: Mutex<HidEventRateTracker> = Mutex::new(HidEventRateTracker::new());

/// Record one HID idle observation. Called automatically by
/// [`collect_idle_secs`] on macOS; exposed at `pub(crate)` only for the
/// behavioural tests further down this file.
pub(crate) fn record_hid_idle_sample(current_idle_secs: f64, now: Instant) {
    if let Ok(mut t) = HID_RATE_TRACKER.lock() {
        t.observe(current_idle_secs, now);
    }
    // Mutex poisoning: a poisoned tracker means a panic crossed the lock.
    // Silently drop the sample — the daemon loop is best-effort and the
    // modulator will continue on its `idle_secs`-only fallback.
}

/// Public accessor used by the daemon main loop to fill
/// `PresenceInputs.hid_events_per_minute`. See module docs and
/// [`HidEventRateTracker::events_per_minute`] for the calculation.
///
/// Returns 0.0 — a neutral "fall back to idle_seconds" signal — whenever
/// the tracker is empty, poisoned, or has not yet observed enough span.
///
/// [Iqbal & Bailey 2008] "Effects of Interruptions on Task Performance":
/// interruption cost rises with active keyboard work even when the user has
/// briefly paused (idle_secs ≥ 5). A 30-sample reset-window catches those
/// bursty active periods that pure idle-time misses.
pub fn hid_events_per_minute() -> f64 {
    HID_RATE_TRACKER
        .lock()
        .map(|t| t.events_per_minute())
        .unwrap_or(0.0)
}

#[cfg(test)]
fn reset_hid_tracker_for_tests() {
    if let Ok(mut t) = HID_RATE_TRACKER.lock() {
        t.reset_for_tests();
    }
}

/// User context snapshot — what is the user actually doing right now?
///
/// All fields default to "safe/unknown" values so callers that skip collection
/// (e.g., tests, non-macOS) behave conservatively.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    /// Seconds since last keyboard or mouse event.
    /// 0.0 = recently active or unknown. Use `is_idle_long()` / `is_recently_active()`.
    pub idle_secs: f64,

    /// True if any non-Apollo process holds a sleep-prevention assertion.
    /// Indicates active media playback, presentation, or video call.
    pub has_sleep_assertion: bool,

    /// True when a video/audio call is likely in progress (assertion owner
    /// matches a known conferencing app).
    pub call_in_progress: bool,

    /// True if audio is currently being output.
    /// Derived from `coreaudiod` holding a `NoIdleSleepAssertion`.
    pub audio_active: bool,
}

impl Default for UserContext {
    fn default() -> Self {
        Self {
            idle_secs: 0.0,
            has_sleep_assertion: false,
            call_in_progress: false,
            audio_active: false,
        }
    }
}

impl UserContext {
    /// Collect all user context signals.
    ///
    /// `collect_assertions`: pass `true` every N cycles to amortise the `pmset`
    /// cost. When `false`, assertion fields are left at their previous values —
    /// the caller merges with the cached context.
    #[cfg(target_os = "macos")]
    pub fn collect(collect_assertions: bool) -> Self {
        let idle_secs = collect_idle_secs();
        let (has_sleep_assertion, call_in_progress, audio_active) = if collect_assertions {
            collect_pmset_assertions()
        } else {
            (false, false, false)
        };
        Self {
            idle_secs,
            has_sleep_assertion,
            call_in_progress,
            audio_active,
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn collect(_collect_assertions: bool) -> Self {
        Self::default()
    }

    /// User has been away long enough that aggressive optimization is safe.
    /// Threshold: 2 minutes (120s).
    #[inline]
    pub fn is_idle_long(&self) -> bool {
        self.idle_secs >= 120.0
    }

    /// User was very recently active — protect fluidity, avoid jank.
    /// Threshold: 15 seconds.
    #[inline]
    pub fn is_recently_active(&self) -> bool {
        // idle_secs == 0.0 is the "unknown" sentinel — treat as active.
        self.idle_secs < 15.0
    }

    /// Any signal that means "don't freeze interactive processes".
    ///
    /// `call_in_progress` is unconditional — interrupting a video call is never OK.
    /// `has_sleep_assertion` is bypassed when the system is in genuine crisis.
    ///
    /// Crisis signals (bypass sleep-assertion if ANY fires):
    ///   • `memory_pressure >= 0.70` — kernel-reported RAM level critical
    ///   • `thrashing_score >= 10_000` — Gate C flow crisis: compressor churning
    ///   • `p_oom_30s >= 0.40` — hazard-model predicts ≥40% OOM probability in 30s
    ///
    /// Why `p_oom_30s` replaces the old swap-bytes bypass:
    /// The old check (`swap_used >= 4 GB`) was hardware-hardcoded for 16+ GB Macs
    /// and never fired on M1 8GB before OOM. macOS dynamic-swap also makes
    /// absolute-bytes fragile: `swap_used ≈ swap_total` whenever swap is in use.
    /// `p_oom_30s` is the learned aggregate of pressure + swap-velocity +
    /// compressor state, calibrated against actual OOM events by OutcomeTracker.
    /// It incorporates swap implicitly and scales with hardware automatically.
    ///
    /// [Denning 1968] fault rate > residency defines working-set quality;
    /// [Nygard 2018] load shedding must override politeness under overload;
    /// [Camacho 2007] predictive control > reactive snapshots under lag.
    #[inline]
    pub fn freeze_protected(
        &self,
        memory_pressure: f64,
        thrashing_score: f64,
        p_oom_30s: f64,
    ) -> bool {
        if self.call_in_progress {
            return true;
        }
        if memory_pressure >= 0.70 || thrashing_score >= 10_000.0 || p_oom_30s >= 0.40 {
            return false;
        }
        self.has_sleep_assertion
    }

    /// Pressure threshold delta in percentage-points based on idle state.
    ///
    /// Returns a signed offset to add to the effective bg_pressure gate:
    /// - Idle long → `-0.10` (lower gate → allow earlier, more aggressive optimization)
    /// - Recently active → `+0.05` (raise gate → be gentle)
    /// - Otherwise → `0.0`
    #[inline]
    pub fn pressure_gate_offset(&self) -> f64 {
        if self.is_idle_long() {
            -0.10
        } else if self.is_recently_active() {
            0.05
        } else {
            0.0
        }
    }
}

// ── Internal collection helpers ───────────────────────────────────────────────

/// Read HIDIdleTime from IOHIDSystem via `ioreg`.
///
/// The value is reported in nanoseconds. Returns 30.0 on any error — neutral
/// zone (15 ≤ idle < 120), so a collection failure doesn't falsely trigger
/// "recently active" conservatism (which would raise freeze gates).
/// [Gray & Reuter 1993] "Transaction Processing: Concepts and Techniques" —
/// safe-default under partial failure: use neutral, not worst-case assumption.
#[cfg(target_os = "macos")]
fn collect_idle_secs() -> f64 {
    let idle = collect_idle_secs_inner().unwrap_or(30.0);
    // Side effect: feed the HID-event-rate tracker. Recording on the
    // fallback path (30.0) is intentional — a stuck reading produces no
    // resets and therefore no spurious "events" in the window.
    record_hid_idle_sample(idle, Instant::now());
    idle
}

#[cfg(target_os = "macos")]
fn collect_idle_secs_inner() -> Option<f64> {
    // 2026-05-12: direct IOKit FFI replaces `ioreg -c IOHIDSystem` subprocess.
    // Subprocess fork+exec+parse cost ~25ms — the single biggest p95 outlier
    // contributor in stage_reason_usercontext_max. IOKit registry query is
    // ~50µs (500× faster) and matches what `ioreg` itself does internally.
    //
    // Path: IOServiceGetMatchingService(kIOMasterPortDefault,
    //       IOServiceMatching("IOHIDSystem")) →
    //       IORegistryEntryCreateCFProperty(service, CFSTR("HIDIdleTime"),
    //       kCFAllocatorDefault, 0) → CFNumberGetValue(num, kCFNumberSInt64Type)
    //
    // Returns idle time in nanoseconds.
    use std::ffi::{c_void, CString};

    type MachPortT = u32;
    type CFAllocatorRef = *const c_void;
    type CFDictionaryRef = *const c_void;
    type CFTypeRef = *const c_void;
    type CFStringRef = *const c_void;
    type IoServiceT = u32;
    type KernReturnT = i32;
    type CFNumberType = i64;

    const K_CF_NUMBER_S_INT64: CFNumberType = 4;
    const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

    #[link(name = "CoreFoundation", kind = "framework")]
    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        static kIOMasterPortDefault: MachPortT;
        fn IOServiceMatching(name: *const i8) -> *mut c_void;
        fn IOServiceGetMatchingService(
            master_port: MachPortT,
            matching: CFDictionaryRef,
        ) -> IoServiceT;
        fn IORegistryEntryCreateCFProperty(
            entry: IoServiceT,
            key: CFStringRef,
            allocator: CFAllocatorRef,
            options: u32,
        ) -> CFTypeRef;
        fn IOObjectRelease(entry: IoServiceT) -> KernReturnT;
        fn CFStringCreateWithCString(
            alloc: CFAllocatorRef,
            cstr: *const i8,
            encoding: u32,
        ) -> CFStringRef;
        fn CFNumberGetValue(num: CFTypeRef, ty: CFNumberType, value_ptr: *mut c_void) -> bool;
        fn CFRelease(cf: CFTypeRef);
    }

    unsafe {
        let service_name = CString::new("IOHIDSystem").ok()?;
        let matching = IOServiceMatching(service_name.as_ptr());
        if matching.is_null() {
            return None;
        }
        // IOServiceGetMatchingService consumes the matching dict — no release.
        let service = IOServiceGetMatchingService(kIOMasterPortDefault, matching);
        if service == 0 {
            return None;
        }
        let key_cstr = CString::new("HIDIdleTime").ok()?;
        let key = CFStringCreateWithCString(
            std::ptr::null(),
            key_cstr.as_ptr(),
            K_CF_STRING_ENCODING_UTF8,
        );
        if key.is_null() {
            IOObjectRelease(service);
            return None;
        }
        let prop = IORegistryEntryCreateCFProperty(service, key, std::ptr::null(), 0);
        CFRelease(key);
        IOObjectRelease(service);
        if prop.is_null() {
            return None;
        }
        let mut ns: i64 = 0;
        let ok = CFNumberGetValue(prop, K_CF_NUMBER_S_INT64, &mut ns as *mut _ as *mut c_void);
        CFRelease(prop);
        if !ok || ns < 0 {
            return None;
        }
        Some(ns as f64 / 1_000_000_000.0)
    }
}

/// Parse `pmset -g assertions` for sleep-prevention, call, and audio signals.
///
/// Returns `(has_sleep_assertion, call_in_progress, audio_active)`.
/// All false on error.
///
/// [Apple TN3115 / IOKit Power Assertions] pmset assertions:
///   `PreventUserIdleSleep` — active user task (CPU+display)
///   `PreventUserIdleSystemSleep` — active background task (CPU only)
///   `PreventUserIdleDisplaySleep` — display kept awake (streaming video,
///                                   slide-deck presenter mode, PDF/dashboard
///                                   viewers that mute audio but keep the
///                                   screen lit). 2026-05-16 reviewer-found
///                                   gap: prior code only matched the two
///                                   above, letting passive-content viewers
///                                   fall through into aggressive throttling.
///   `NoIdleSleepAssertion` from coreaudiod → audio output active
///
/// `audio_active` is OR'd with a CoreAudio direct query
/// (`kAudioDevicePropertyDeviceIsRunningSomewhere` on the default output
/// device). Modern macOS no longer reliably emits the coreaudiod assertion
/// for browser-sourced audio (HTML5 audio, podcasts, YouTube background
/// playback), so the pmset path was missing those cases — letting the
/// maintenance purge gate fire during media playback and cause stutter.
#[cfg(target_os = "macos")]
fn collect_pmset_assertions() -> (bool, bool, bool) {
    let (sleep, call, audio_pmset) = collect_pmset_inner().unwrap_or((false, false, false));
    let audio = audio_pmset || super::coreaudio_active::is_audio_running_somewhere();
    (sleep, call, audio)
}

#[cfg(target_os = "macos")]
fn collect_pmset_inner() -> Option<(bool, bool, bool)> {
    let output = Command::new("pmset")
        .args(["-g", "assertions"])
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(parse_pmset_assertions(&stdout))
}

/// Pure parser for `pmset -g assertions` stdout.
///
/// Factored out of [`collect_pmset_inner`] so the assertion-matching logic is
/// testable without shelling out. Returns the same
/// `(has_sleep_assertion, call_in_progress, audio_active)` tuple.
///
/// Recognised assertion kinds (case-sensitive match on the line):
///   * `PreventUserIdleSleep`
///   * `PreventUserIdleSystemSleep`
///   * `PreventUserIdleDisplaySleep` (added 2026-05-16 — IOKit Power
///     Assertions documentation lists this as the canonical kind for apps
///     keeping the display awake without audio output).
///
/// Apollo's own assertions are skipped to avoid self-recursion (Apollo
/// publishes `PreventUserIdleSystemSleep` during the maintenance purge gate;
/// counting that would silence the very heuristic that gates it).
fn parse_pmset_assertions(stdout: &str) -> (bool, bool, bool) {
    let mut has_sleep_assertion = false;
    let mut call_in_progress = false;
    let mut audio_active = false;

    // Only parse the "Listed by owning process:" section — the summary
    // section at the top has aggregate counts that could be misleading.
    let mut in_process_section = false;

    for line in stdout.lines() {
        if line.starts_with("Listed by owning process:") {
            in_process_section = true;
            continue;
        }
        if !in_process_section || line.trim().is_empty() {
            continue;
        }

        let line_lc = line.to_ascii_lowercase();

        // Skip Apollo's own assertions.
        if line_lc.contains("apollo-optimizer") || line_lc.contains("apollo-optimizerd") {
            continue;
        }

        // Sleep-prevention assertions indicate active user task.
        // Three kinds are treated as equivalent for the user-presence signal:
        //   - PreventUserIdleSleep         (CPU + display)
        //   - PreventUserIdleSystemSleep   (CPU only — long background task)
        //   - PreventUserIdleDisplaySleep  (display only — passive viewing)
        // The third kind closes the 2026-05-16 reviewer-found gap (streaming,
        // PDF/dashboard, presenter mode mute the audio but keep the screen lit).
        if line.contains("PreventUserIdleSleep")
            || line.contains("PreventUserIdleSystemSleep")
            || line.contains("PreventUserIdleDisplaySleep")
        {
            has_sleep_assertion = true;
            // If a conferencing app owns the assertion → call in progress.
            if CALL_APP_NAMES.iter().any(|n| line_lc.contains(n)) {
                call_in_progress = true;
            }
        }

        // coreaudiod holding NoIdleSleepAssertion → audio output is active.
        if line_lc.contains("coreaudiod") && line.contains("NoIdleSleepAssertion") {
            audio_active = true;
        }
    }

    (has_sleep_assertion, call_in_progress, audio_active)
}

// Cross-platform alias so the parser test compiles on non-macOS too.
#[cfg(not(target_os = "macos"))]
#[cfg(test)]
fn parse_pmset_assertions(stdout: &str) -> (bool, bool, bool) {
    // Same body as the macOS variant — kept inline because the function above
    // is `#[cfg(target_os = "macos")]`-gated and unavailable for tests on
    // other hosts. The duplication is trivial; the alternative (moving the
    // function out from under the cfg) would expose macOS-only call sites.
    let mut has_sleep_assertion = false;
    let mut call_in_progress = false;
    let mut audio_active = false;
    let mut in_process_section = false;
    for line in stdout.lines() {
        if line.starts_with("Listed by owning process:") {
            in_process_section = true;
            continue;
        }
        if !in_process_section || line.trim().is_empty() {
            continue;
        }
        let line_lc = line.to_ascii_lowercase();
        if line_lc.contains("apollo-optimizer") || line_lc.contains("apollo-optimizerd") {
            continue;
        }
        if line.contains("PreventUserIdleSleep")
            || line.contains("PreventUserIdleSystemSleep")
            || line.contains("PreventUserIdleDisplaySleep")
        {
            has_sleep_assertion = true;
            if ["zoom.us", "facetime", "teams"]
                .iter()
                .any(|n| line_lc.contains(n))
            {
                call_in_progress = true;
            }
        }
        if line_lc.contains("coreaudiod") && line.contains("NoIdleSleepAssertion") {
            audio_active = true;
        }
    }
    (has_sleep_assertion, call_in_progress, audio_active)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_safe() {
        let ctx = UserContext::default();
        assert_eq!(ctx.idle_secs, 0.0);
        assert!(!ctx.has_sleep_assertion);
        assert!(!ctx.call_in_progress);
        assert!(!ctx.audio_active);
    }

    #[test]
    fn is_recently_active_when_idle_zero() {
        let ctx = UserContext {
            idle_secs: 0.0,
            ..Default::default()
        };
        assert!(ctx.is_recently_active());
        assert!(!ctx.is_idle_long());
    }

    #[test]
    fn is_idle_long_at_threshold() {
        let ctx = UserContext {
            idle_secs: 120.0,
            ..Default::default()
        };
        assert!(ctx.is_idle_long());
        assert!(!ctx.is_recently_active());
    }

    #[test]
    fn recently_active_boundary() {
        let active = UserContext {
            idle_secs: 14.9,
            ..Default::default()
        };
        let not_active = UserContext {
            idle_secs: 15.0,
            ..Default::default()
        };
        assert!(active.is_recently_active());
        assert!(!not_active.is_recently_active());
    }

    #[test]
    fn freeze_protected_from_call_or_assertion() {
        let call = UserContext {
            call_in_progress: true,
            ..Default::default()
        };
        let assertion = UserContext {
            has_sleep_assertion: true,
            ..Default::default()
        };
        let normal = UserContext::default();
        let no_thrash = 0.0_f64;
        let thrashing = 15_000.0_f64; // above 10k bypass
        let low_p_oom = 0.05_f64;
        let high_p_oom = 0.55_f64; // above 0.40 predictive bypass

        // Low pressure + no thrashing + low p_oom: assertion blocks freeze, normal does not.
        assert!(call.freeze_protected(0.30, no_thrash, low_p_oom));
        assert!(assertion.freeze_protected(0.30, no_thrash, low_p_oom));
        assert!(!normal.freeze_protected(0.30, no_thrash, low_p_oom));
        // High pressure: call still blocks; assertion no longer.
        assert!(call.freeze_protected(0.85, no_thrash, low_p_oom));
        assert!(!assertion.freeze_protected(0.85, no_thrash, low_p_oom));
        assert!(!normal.freeze_protected(0.85, no_thrash, low_p_oom));
        // Low pressure BUT thrashing ≥ 10k: assertion no longer blocks.
        assert!(call.freeze_protected(0.59, thrashing, low_p_oom));
        assert!(!assertion.freeze_protected(0.59, thrashing, low_p_oom));
        assert!(!normal.freeze_protected(0.59, thrashing, low_p_oom));
        // Low pressure + low thrashing BUT p_oom_30s ≥ 0.40: assertion no longer blocks.
        // This is the root-cause fix — predictive signal catches crises the old
        // 4GB-swap-bytes bypass missed on M1 8GB (prod hit 1.5GB swap + 0.55 p_oom
        // with thrashing only 2549 — all old bypasses silent, freeze blocked for hours).
        assert!(call.freeze_protected(0.59, no_thrash, high_p_oom));
        assert!(!assertion.freeze_protected(0.59, no_thrash, high_p_oom));
        assert!(!normal.freeze_protected(0.59, no_thrash, high_p_oom));
    }

    #[test]
    fn pressure_gate_offset_idle() {
        let idle = UserContext {
            idle_secs: 300.0,
            ..Default::default()
        };
        assert!((idle.pressure_gate_offset() - (-0.10)).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_gate_offset_active() {
        let active = UserContext {
            idle_secs: 5.0,
            ..Default::default()
        };
        assert!((active.pressure_gate_offset() - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_gate_offset_neutral() {
        let neutral = UserContext {
            idle_secs: 60.0,
            ..Default::default()
        };
        assert!((neutral.pressure_gate_offset() - 0.0).abs() < f64::EPSILON);
    }

    // ── pmset parser tests ────────────────────────────────────────────────────

    /// Step 1 — Gap A regression test (reviewer 2026-05-16).
    ///
    /// `PreventUserIdleDisplaySleep` is the canonical IOKit Power Assertion
    /// kind for apps that keep the display awake without audio output
    /// (streaming video on external monitor, PDF/dashboard viewers,
    /// presenter mode). Prior parser only matched the `CPU`/`SystemSleep`
    /// variants and silently dropped these — a presence gap.
    #[test]
    fn pmset_detects_prevent_display_sleep_as_active() {
        let stdout = "\
Assertion status system-wide:
   PreventUserIdleSystemSleep   0
   PreventUserIdleDisplaySleep  1
Listed by owning process:
pid 4321(QuickTime Player): [0x0000000100000abc] 00:30:11 PreventUserIdleDisplaySleep named: \"com.apple.QuickTimePlayerX playback\"
";
        let (sleep, call, audio) = parse_pmset_assertions(stdout);
        assert!(
            sleep,
            "PreventUserIdleDisplaySleep must register as a sleep assertion"
        );
        // No conferencing app name in the line → call_in_progress stays false.
        assert!(!call, "no conferencing-app name → call_in_progress=false");
        // No coreaudiod NoIdleSleepAssertion line → audio_active stays false.
        assert!(!audio, "no coreaudiod line → audio_active=false");
    }

    /// Parser still recognises the two pre-existing kinds.
    #[test]
    fn pmset_detects_existing_sleep_kinds() {
        let stdout = "\
Listed by owning process:
pid 100(otherApp): [0x0000000100000001] 00:01:00 PreventUserIdleSleep named: \"x\"
pid 101(daemonish): [0x0000000100000002] 00:01:00 PreventUserIdleSystemSleep named: \"y\"
";
        let (sleep, _, _) = parse_pmset_assertions(stdout);
        assert!(sleep);
    }

    /// Apollo's own assertion lines must be skipped — otherwise the parser
    /// would self-trigger every cycle the maintenance purge gate is active.
    #[test]
    fn pmset_skips_apollo_own_assertions() {
        let stdout = "\
Listed by owning process:
pid 1234(apollo-optimizerd): [0x0000000100000abc] 00:30:11 PreventUserIdleDisplaySleep named: \"maintenance-purge-gate\"
";
        let (sleep, _, _) = parse_pmset_assertions(stdout);
        assert!(
            !sleep,
            "Apollo's own DisplaySleep assertion must NOT register"
        );
    }

    /// Conferencing-app heuristic still fires for DisplaySleep — a Zoom call
    /// that mutes audio but keeps the camera preview on shows up as
    /// DisplaySleep + zoom.us owner.
    #[test]
    fn pmset_call_in_progress_fires_for_display_sleep_with_zoom_owner() {
        let stdout = "\
Listed by owning process:
pid 9999(zoom.us): [0x000000010000beef] 00:05:00 PreventUserIdleDisplaySleep named: \"Zoom preview\"
";
        let (sleep, call, _) = parse_pmset_assertions(stdout);
        assert!(sleep);
        assert!(call, "zoom.us owning DisplaySleep → call_in_progress=true");
    }

    // ── HID event-rate tracker tests (Phase 5.1-D) ────────────────────────────
    //
    // These tests share a process-global static mutex (`HID_RATE_TRACKER`).
    // They acquire a dedicated test-mutex first so they cannot interleave
    // and corrupt each other's window state, then call
    // `reset_hid_tracker_for_tests` to start from a known-empty window.

    use std::sync::Mutex as StdMutex;
    use std::time::Duration;
    static HID_TEST_SERIAL: StdMutex<()> = StdMutex::new(());

    /// No HID resets observed → events_per_minute reports 0.0.
    ///
    /// Models a fully-idle laptop: HIDIdleTime monotonically grows, the
    /// tracker observes no resets, and the modulator stays on its
    /// `idle_seconds`-only fallback.
    #[test]
    fn hid_rate_zero_when_idle_constant() {
        let _guard = HID_TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset_hid_tracker_for_tests();
        let start = Instant::now();
        // 10 samples, idle strictly increasing — zero resets.
        for i in 0..10 {
            let idle = 5.0 + i as f64; // 5.0, 6.0, … 14.0
            let at = start + Duration::from_secs(i as u64 * 2); // 2-s cadence
            record_hid_idle_sample(idle, at);
        }
        let rate = hid_events_per_minute();
        assert!(
            rate.abs() < f64::EPSILON,
            "expected 0.0 events/min with no resets, got {rate}"
        );
    }

    /// An idle-time reset (current < previous) → positive events_per_minute.
    ///
    /// Two samples with `idle_secs` dropping from 5.0 to 0.1 over a 2-s
    /// interval mean at least one keyboard / mouse event fired between
    /// them. Spread the same pattern over a 10-s window so the min-span
    /// guard (5 s) passes, and assert the rate is strictly positive.
    #[test]
    fn hid_rate_positive_on_idle_reset() {
        let _guard = HID_TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset_hid_tracker_for_tests();
        let start = Instant::now();
        // Three samples spread over 10 s: idle 5.0 → 0.1 → 1.0.
        // The 5.0 → 0.1 transition counts as one reset.
        record_hid_idle_sample(5.0, start);
        record_hid_idle_sample(0.1, start + Duration::from_secs(5));
        record_hid_idle_sample(1.0, start + Duration::from_secs(10));
        let rate = hid_events_per_minute();
        assert!(
            rate > 0.0,
            "expected positive events/min after idle reset, got {rate}"
        );
        // Sanity: 1 reset over 10 s → 6 events/min.
        let expected = 1.0 * 60.0 / 10.0;
        assert!(
            (rate - expected).abs() < 1e-6,
            "expected {expected} events/min, got {rate}"
        );
    }

    /// Window caps at 30 samples and the rate divisor is the rolling
    /// wall-clock span across the retained samples (NOT the lifetime of
    /// the process). Pushing more than 30 samples must evict the oldest
    /// and the reported rate must reflect only what is still in the window.
    #[test]
    fn hid_rate_window_averages_over_30_cycles() {
        let _guard = HID_TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset_hid_tracker_for_tests();
        let start = Instant::now();
        let step = Duration::from_secs(2); // 2-s spacing per sample

        // (1) Push 40 samples — 10 will be evicted, 30 retained.
        //     Idle alternates 1.0 → 0.5 → 1.0 → 0.5 …
        //     Every transition into 0.5 is a reset (1.0 > 0.5).
        //     Over 40 samples that is 20 resets total, but only 30 samples
        //     remain in the window (samples 10..40) which contain 15 resets.
        for i in 0..40 {
            let idle = if i % 2 == 0 { 1.0 } else { 0.5 };
            record_hid_idle_sample(idle, start + step * (i as u32));
        }
        let rate = hid_events_per_minute();
        // Window now spans samples 10..40 → 30 samples → span = 29 * 2 s = 58 s.
        // 15 resets / 58 s * 60 s/min ≈ 15.517 events/min.
        let expected = 15.0 * 60.0 / 58.0;
        assert!(
            (rate - expected).abs() < 1e-6,
            "rolling 30-sample window: expected {expected:.3} events/min, got {rate}"
        );

        // (2) After pushing 40 samples the queue length is capped at 30:
        //     read the tracker directly to confirm the bound is enforced.
        let len = HID_RATE_TRACKER
            .lock()
            .map(|t| t.samples.len())
            .unwrap_or(0);
        assert_eq!(len, HID_RATE_WINDOW, "window must cap at 30 samples");
    }
}
