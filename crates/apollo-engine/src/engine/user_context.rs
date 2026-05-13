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

/// App names that indicate an active video/audio call.
#[cfg(target_os = "macos")]
const CALL_APP_NAMES: &[&str] = &[
    "zoom.us", "facetime", "teams", "webex", "skype", "discord", "meet", "slack", "whereby",
    "around", "loom",
];

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
    collect_idle_secs_inner().unwrap_or(30.0)
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
        fn IOServiceMatching(name: *const i8) -> CFDictionaryRef;
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
        fn CFNumberGetValue(
            num: CFTypeRef,
            ty: CFNumberType,
            value_ptr: *mut c_void,
        ) -> bool;
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
/// [Apple TN3115] pmset assertions:
///   `PreventUserIdleSleep` / `PreventUserIdleSystemSleep` → active media/call
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
        if line.contains("PreventUserIdleSleep") || line.contains("PreventUserIdleSystemSleep") {
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

    Some((has_sleep_assertion, call_in_progress, audio_active))
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
}
