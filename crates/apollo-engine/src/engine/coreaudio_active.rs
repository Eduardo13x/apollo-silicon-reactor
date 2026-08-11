//! CoreAudio direct query for "is audio actively playing right now".
//!
//! Modern macOS no longer reliably emits `coreaudiod NoIdleSleepAssertion` for
//! browser-sourced audio (Brave/Chrome YouTube, podcasts in HTML5 audio).
//! Pmset-only detection misses these → maintenance purge fires during media
//! playback → page-cache invalidation → audio glitches.
//!
//! Fix: query CoreAudio's `kAudioDevicePropertyDeviceIsRunningSomewhere` on
//! the default output device. This is the canonical macOS API for "is anyone
//! using this output". True iff at least one IOProc on the device is active.
//!
//! Direct HAL queries are cached process-wide. Successful probes have a short
//! TTL; unavailable devices use exponential backoff. Root LaunchDaemons skip
//! the direct probe entirely because their system bootstrap session has no
//! per-user default device. The daemon still keeps the independent pmset,
//! process, and screen-capture signals in that environment.

#[cfg(target_os = "macos")]
use std::mem;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const SUCCESS_CACHE_TTL: Duration = Duration::from_secs(1);
const FAILURE_BACKOFF_INITIAL: Duration = Duration::from_secs(15);
const FAILURE_BACKOFF_MAX: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AudioActivitySnapshot {
    pub output_active: bool,
    pub input_active: bool,
    pub output_probe_available: bool,
    pub input_probe_available: bool,
    pub session_supported: bool,
    pub direct_samples: u64,
    pub cache_hits: u64,
    pub failures: u64,
}

impl AudioActivitySnapshot {
    #[inline]
    pub fn realtime_call_active(self) -> bool {
        self.output_active && self.input_active
    }

    pub fn probe_state(self) -> &'static str {
        if !self.session_supported {
            "session-fallback"
        } else if self.output_probe_available && self.input_probe_available {
            "direct"
        } else if self.output_probe_available || self.input_probe_available {
            "degraded"
        } else {
            "backoff"
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct DeviceReading {
    active: bool,
    available: bool,
}

#[derive(Debug, Default)]
struct DeviceProbeCache {
    reading: DeviceReading,
    next_probe_at: Option<Instant>,
    failure_backoff: Option<Duration>,
}

impl DeviceProbeCache {
    fn read_with(
        &mut self,
        now: Instant,
        probe: impl FnOnce() -> DeviceReading,
    ) -> (DeviceReading, bool, bool) {
        if self.next_probe_at.is_some_and(|next| now < next) {
            return (self.reading, true, false);
        }

        let reading = probe();
        self.reading = reading;
        if reading.available {
            self.failure_backoff = None;
            self.next_probe_at = Some(now + SUCCESS_CACHE_TTL);
        } else {
            let backoff = self
                .failure_backoff
                .unwrap_or(FAILURE_BACKOFF_INITIAL)
                .min(FAILURE_BACKOFF_MAX);
            self.next_probe_at = Some(now + backoff);
            self.failure_backoff = Some((backoff * 2).min(FAILURE_BACKOFF_MAX));
        }
        (reading, false, !reading.available)
    }
}

#[derive(Debug, Default)]
struct AudioProbeCache {
    output: DeviceProbeCache,
    input: DeviceProbeCache,
    direct_samples: u64,
    cache_hits: u64,
    failures: u64,
}

impl AudioProbeCache {
    fn sample_with(
        &mut self,
        now: Instant,
        session_supported: bool,
        output_probe: impl FnOnce() -> DeviceReading,
        input_probe: impl FnOnce() -> DeviceReading,
    ) -> AudioActivitySnapshot {
        if !session_supported {
            return AudioActivitySnapshot {
                session_supported: false,
                direct_samples: self.direct_samples,
                cache_hits: self.cache_hits,
                failures: self.failures,
                ..AudioActivitySnapshot::default()
            };
        }

        let (output, output_cached, output_failed) = self.output.read_with(now, output_probe);
        let (input, input_cached, input_failed) = self.input.read_with(now, input_probe);
        self.direct_samples = self
            .direct_samples
            .saturating_add(u64::from(!output_cached) + u64::from(!input_cached));
        self.cache_hits = self
            .cache_hits
            .saturating_add(u64::from(output_cached) + u64::from(input_cached));
        self.failures = self
            .failures
            .saturating_add(u64::from(output_failed) + u64::from(input_failed));

        AudioActivitySnapshot {
            output_active: output.active,
            input_active: input.active,
            output_probe_available: output.available,
            input_probe_available: input.available,
            session_supported: true,
            direct_samples: self.direct_samples,
            cache_hits: self.cache_hits,
            failures: self.failures,
        }
    }
}

fn audio_probe_cache() -> &'static Mutex<AudioProbeCache> {
    static CACHE: OnceLock<Mutex<AudioProbeCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(AudioProbeCache::default()))
}

#[cfg(target_os = "macos")]
type AudioObjectID = u32;
#[cfg(target_os = "macos")]
type OSStatus = i32;

#[cfg(target_os = "macos")]
const K_AUDIO_OBJECT_SYSTEM_OBJECT: AudioObjectID = 1;
// Four-char codes (big-endian when typed as u32):
// 'dOut' = default output device selector
#[cfg(target_os = "macos")]
const K_AUDIO_HARDWARE_PROPERTY_DEFAULT_OUTPUT_DEVICE: u32 = 0x644F_7574;
// 'dIn ' = default input device selector (note trailing space, big-endian: 'd','I','n',' ')
#[cfg(target_os = "macos")]
const K_AUDIO_HARDWARE_PROPERTY_DEFAULT_INPUT_DEVICE: u32 = 0x6449_6E20;
// 'gone' = device-is-running-somewhere selector
#[cfg(target_os = "macos")]
const K_AUDIO_DEVICE_PROPERTY_DEVICE_IS_RUNNING_SOMEWHERE: u32 = 0x676F_6E65;
// 'glob' = global scope
#[cfg(target_os = "macos")]
const K_AUDIO_OBJECT_PROPERTY_SCOPE_GLOBAL: u32 = 0x676C_6F62;
#[cfg(target_os = "macos")]
const K_AUDIO_OBJECT_PROPERTY_ELEMENT_MAIN: u32 = 0;

#[cfg(target_os = "macos")]
#[repr(C)]
struct AudioObjectPropertyAddress {
    selector: u32,
    scope: u32,
    element: u32,
}

#[cfg(target_os = "macos")]
#[link(name = "CoreAudio", kind = "framework")]
extern "C" {
    fn AudioObjectGetPropertyData(
        in_object_id: AudioObjectID,
        in_address: *const AudioObjectPropertyAddress,
        in_qualifier_data_size: u32,
        in_qualifier_data: *const std::ffi::c_void,
        io_data_size: *mut u32,
        out_data: *mut std::ffi::c_void,
    ) -> OSStatus;
}

#[cfg(target_os = "macos")]
fn probe_output_device() -> DeviceReading {
    unsafe {
        let default_out_addr = AudioObjectPropertyAddress {
            selector: K_AUDIO_HARDWARE_PROPERTY_DEFAULT_OUTPUT_DEVICE,
            scope: K_AUDIO_OBJECT_PROPERTY_SCOPE_GLOBAL,
            element: K_AUDIO_OBJECT_PROPERTY_ELEMENT_MAIN,
        };
        let mut device_id: AudioObjectID = 0;
        let mut size: u32 = mem::size_of::<AudioObjectID>() as u32;
        let status = AudioObjectGetPropertyData(
            K_AUDIO_OBJECT_SYSTEM_OBJECT,
            &default_out_addr,
            0,
            std::ptr::null(),
            &mut size,
            &mut device_id as *mut _ as *mut std::ffi::c_void,
        );
        if status != 0 || device_id == 0 {
            return DeviceReading::default();
        }

        let running_addr = AudioObjectPropertyAddress {
            selector: K_AUDIO_DEVICE_PROPERTY_DEVICE_IS_RUNNING_SOMEWHERE,
            scope: K_AUDIO_OBJECT_PROPERTY_SCOPE_GLOBAL,
            element: K_AUDIO_OBJECT_PROPERTY_ELEMENT_MAIN,
        };
        let mut running: u32 = 0;
        let mut size2: u32 = mem::size_of::<u32>() as u32;
        let status2 = AudioObjectGetPropertyData(
            device_id,
            &running_addr,
            0,
            std::ptr::null(),
            &mut size2,
            &mut running as *mut _ as *mut std::ffi::c_void,
        );
        if status2 != 0 {
            return DeviceReading::default();
        }
        DeviceReading {
            active: running != 0,
            available: true,
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn probe_output_device() -> DeviceReading {
    DeviceReading::default()
}

/// True when the default INPUT device (microphone) is actively capturing.
///
/// Same FFI shape as `is_audio_running_somewhere` but selecting the default
/// input device. Mirrors the output-side detection so callers can build a
/// realtime-call gate (`output_active AND input_active = full-duplex call`).
///
/// Returns `false` on any error path. The signal is composed with other media
/// indicators — a missed detection only weakens the inhibit, never fires it
/// spuriously.
///
/// WebRTC ROOT-CAUSE (2026-06-09 prod incident): Apollo's sysctl_governor
/// scaled down TCP send/recv buffers by 25% mid-Meet (sysctl_governor.rs:641
/// path: "low retransmissions + low throughput") and set `delayed_ack=3` on
/// battery (sysctl_governor.rs:669), which dropped audio frames and froze
/// video on the user's call. `is_realtime_call_active()` gates both branches
/// from re-firing during a live full-duplex call.
#[cfg(target_os = "macos")]
fn probe_input_device() -> DeviceReading {
    unsafe {
        let default_in_addr = AudioObjectPropertyAddress {
            selector: K_AUDIO_HARDWARE_PROPERTY_DEFAULT_INPUT_DEVICE,
            scope: K_AUDIO_OBJECT_PROPERTY_SCOPE_GLOBAL,
            element: K_AUDIO_OBJECT_PROPERTY_ELEMENT_MAIN,
        };
        let mut device_id: AudioObjectID = 0;
        let mut size: u32 = mem::size_of::<AudioObjectID>() as u32;
        let status = AudioObjectGetPropertyData(
            K_AUDIO_OBJECT_SYSTEM_OBJECT,
            &default_in_addr,
            0,
            std::ptr::null(),
            &mut size,
            &mut device_id as *mut _ as *mut std::ffi::c_void,
        );
        if status != 0 || device_id == 0 {
            return DeviceReading::default();
        }

        let running_addr = AudioObjectPropertyAddress {
            selector: K_AUDIO_DEVICE_PROPERTY_DEVICE_IS_RUNNING_SOMEWHERE,
            scope: K_AUDIO_OBJECT_PROPERTY_SCOPE_GLOBAL,
            element: K_AUDIO_OBJECT_PROPERTY_ELEMENT_MAIN,
        };
        let mut running: u32 = 0;
        let mut size2: u32 = mem::size_of::<u32>() as u32;
        let status2 = AudioObjectGetPropertyData(
            device_id,
            &running_addr,
            0,
            std::ptr::null(),
            &mut size2,
            &mut running as *mut _ as *mut std::ffi::c_void,
        );
        if status2 != 0 {
            return DeviceReading::default();
        }
        DeviceReading {
            active: running != 0,
            available: true,
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn probe_input_device() -> DeviceReading {
    DeviceReading::default()
}

#[inline]
fn direct_session_supported() -> bool {
    #[cfg(target_os = "macos")]
    {
        // A root LaunchDaemon lives in the system bootstrap namespace. HAL's
        // default-device selectors are per-login-session and emit an error on
        // every query from that namespace. Apollo's pmset/process/capture
        // fallbacks remain active, so skipping a known-invalid direct source
        // is both cheaper and semantically honest.
        unsafe { libc::geteuid() != 0 }
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Cached direct CoreAudio state and probe diagnostics.
///
/// All callers share this snapshot, so maintenance, Chromium, sysctl, policy,
/// and user-context paths cannot independently hammer HAL in the same cycle.
pub fn audio_activity_snapshot() -> AudioActivitySnapshot {
    if !direct_session_supported() {
        return AudioActivitySnapshot {
            session_supported: false,
            ..AudioActivitySnapshot::default()
        };
    }
    let mut cache = audio_probe_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.sample_with(
        Instant::now(),
        true,
        probe_output_device,
        probe_input_device,
    )
}

/// True when audio is actively flowing through the default output device.
///
/// Returns `false` when the direct source is unavailable. Callers combine it
/// with pmset, process, screen-capture, and workload signals as appropriate.
#[inline]
pub fn is_audio_running_somewhere() -> bool {
    audio_activity_snapshot().output_active
}

#[inline]
pub fn is_audio_input_active() -> bool {
    audio_activity_snapshot().input_active
}

/// True when BOTH default output AND default input devices are running.
///
/// Full-duplex audio = realtime call (Google Meet / Zoom / FaceTime / Discord /
/// Teams). Apollo MUST NOT mutate network sysctls or apply Battery network
/// profile during this state — buffer reductions and ACK coalescing degrade
/// WebRTC quality (jitter, audio cutouts, video freezes).
///
/// Cost: ~100µs (two CoreAudio round-trips, one per device). Caller is expected
/// to cache at the same cadence as other media probes (≥3 cycles).
///
/// Composed signal (not "and"-of-noisy): both APIs must positively report
/// running — eliminates false positives from output-only playback (YouTube)
/// or input-only background ASR.
#[inline]
pub fn is_realtime_call_active() -> bool {
    audio_activity_snapshot().realtime_call_active()
}

/// Provisional fault-in storm threshold (pages/sec). Phase 0 baseline on M1
/// 8GB measured typical ~4-6k pages/s under load and a peak of ~150k
/// (≈2.46 GB/s). 30k (~0.5 GB/s) sits well above typical and below the storm
/// peak — a conservative "genuine storm in progress" line. Tunable as more
/// baseline accumulates. [Phase 1]
pub const STORM_REFAULT_PAGES_PER_SEC: f64 = 30_000.0;

/// Physical-pressure floor above which memory RELIEF always wins over
/// anti-stutter suppression. Survival beats UX politeness — Apollo must never
/// strangle its own relief (purge/freeze/demote) while memory drowns.
///
/// REGRESSION SCAR (2026-06-15): the first cut of this gate OR'd in plain
/// `is_audio_running_somewhere()`, so with background music it was permanently
/// true and suppressed the maintenance purge 127,959× (vs 147 fired) → no
/// cache flush → thrashing 69k, refault peaks of 22 GB/s, system "horrible"
/// until the user killed the daemon. Two fixes: (1) drop plain audio so the
/// signal is TRANSIENT (storm/call only); (2) this survival escape, mirroring
/// `user_presence::CRITICAL_PRESSURE_BYPASS`.
pub const SURVIVAL_PRESSURE_FLOOR: f64 = 0.70;

/// True when Apollo should hold off its own memory churn (purge, stale-freeze,
/// jetsam-demote) because a TRANSIENT high-volume workload is in progress and
/// memory is NOT in danger. Suppressing churn here avoids adding faults to the
/// app the user is switching to (the microstutter).
///
/// Two guards make this safe:
/// - **Survival escape**: if `physical_pressure >= SURVIVAL_PRESSURE_FLOOR`,
///   returns `false` — relief wins, no matter the workload. Never strangle.
/// - **Transient only**: a realtime call (output AND input) OR a fault-in
///   storm above [`STORM_REFAULT_PAGES_PER_SEC`]. Plain background audio is
///   deliberately EXCLUDED — including it made the gate permanent (the scar).
///
/// Pass the PHYSICAL pressure (`memory_pressure_raw`, falling back to
/// `memory_pressure`) — purge cannot fix thermal/battery boost.
/// [Hellerstein 2004 §9 disturbance rejection; project survival doctrine]
#[inline]
pub fn is_high_bw_workload_active(refault_pages_per_sec: f64, physical_pressure: f64) -> bool {
    if physical_pressure >= SURVIVAL_PRESSURE_FLOOR {
        return false; // drowning — relief wins, never suppress.
    }
    if refault_pages_per_sec > STORM_REFAULT_PAGES_PER_SEC {
        return true;
    }
    audio_activity_snapshot().realtime_call_active()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn available(active: bool) -> DeviceReading {
        DeviceReading {
            active,
            available: true,
        }
    }

    #[test]
    fn shared_cache_samples_each_device_once_inside_success_ttl() {
        let mut cache = AudioProbeCache::default();
        let now = Instant::now();
        let output_calls = Cell::new(0_u32);
        let input_calls = Cell::new(0_u32);
        let first = cache.sample_with(
            now,
            true,
            || {
                output_calls.set(output_calls.get() + 1);
                available(true)
            },
            || {
                input_calls.set(input_calls.get() + 1);
                available(false)
            },
        );
        let second = cache.sample_with(
            now + Duration::from_millis(500),
            true,
            || {
                output_calls.set(output_calls.get() + 1);
                available(false)
            },
            || {
                input_calls.set(input_calls.get() + 1);
                available(true)
            },
        );

        assert_eq!(output_calls.get(), 1);
        assert_eq!(input_calls.get(), 1);
        assert!(first.output_active);
        assert_eq!(first.output_active, second.output_active);
        assert_eq!(first.input_active, second.input_active);
        assert_eq!(second.direct_samples, 2);
        assert_eq!(second.cache_hits, 2);
    }

    #[test]
    fn failed_device_uses_backoff_without_hiding_healthy_device() {
        let mut cache = AudioProbeCache::default();
        let now = Instant::now();
        let output_calls = Cell::new(0_u32);
        let input_calls = Cell::new(0_u32);
        let _ = cache.sample_with(
            now,
            true,
            || {
                output_calls.set(output_calls.get() + 1);
                available(true)
            },
            || {
                input_calls.set(input_calls.get() + 1);
                DeviceReading::default()
            },
        );
        let next = cache.sample_with(
            now + Duration::from_secs(2),
            true,
            || {
                output_calls.set(output_calls.get() + 1);
                available(true)
            },
            || {
                input_calls.set(input_calls.get() + 1);
                available(true)
            },
        );

        assert_eq!(output_calls.get(), 2, "healthy output keeps its short TTL");
        assert_eq!(input_calls.get(), 1, "failed input stays in backoff");
        assert!(next.output_probe_available);
        assert!(!next.input_probe_available);
        assert_eq!(next.failures, 1);
        assert_eq!(next.probe_state(), "degraded");
    }

    #[test]
    fn unsupported_session_never_calls_hal() {
        let mut cache = AudioProbeCache::default();
        let called = Cell::new(false);
        let snapshot = cache.sample_with(
            Instant::now(),
            false,
            || {
                called.set(true);
                available(true)
            },
            || {
                called.set(true);
                available(true)
            },
        );

        assert!(!called.get());
        assert!(!snapshot.session_supported);
        assert_eq!(snapshot.probe_state(), "session-fallback");
        assert_eq!(snapshot.direct_samples, 0);
    }

    #[test]
    fn query_does_not_panic() {
        // On macOS: returns true or false depending on current playback state.
        // On other OSes: always false. Either way, must not panic.
        let _ = is_audio_running_somewhere();
    }

    #[test]
    fn input_query_does_not_panic() {
        let _ = is_audio_input_active();
    }

    #[test]
    fn realtime_call_does_not_panic() {
        let _ = is_realtime_call_active();
    }

    #[test]
    fn high_bw_workload_fires_on_storm_when_memory_safe() {
        // A storm above threshold, with memory safe (low pressure), suppresses.
        assert!(
            is_high_bw_workload_active(STORM_REFAULT_PAGES_PER_SEC + 1.0, 0.40),
            "storm + safe memory → suppress churn"
        );
        // Quiet rate, low pressure → no storm, no call: do not suppress.
        let quiet = is_high_bw_workload_active(0.0, 0.40);
        assert_eq!(
            quiet,
            is_realtime_call_active(),
            "quiet → only a call counts"
        );
    }

    #[test]
    fn survival_escape_beats_any_workload() {
        // THE regression guard (2026-06-15): even a massive storm must NOT
        // suppress relief once physical pressure reaches the survival floor.
        // Suppressing purge while memory drowns is the bug that strangled
        // Apollo (127,959 skipped purges, thrashing 69k).
        assert!(
            !is_high_bw_workload_active(
                STORM_REFAULT_PAGES_PER_SEC * 100.0,
                SURVIVAL_PRESSURE_FLOOR
            ),
            "at the survival floor, relief wins regardless of the storm"
        );
        assert!(
            !is_high_bw_workload_active(1_000_000.0, 0.95),
            "drowning → never suppress"
        );
    }

    #[test]
    fn storm_threshold_is_strict_greater_than() {
        // Pin the `>` semantics at safe pressure: exactly AT the threshold the
        // storm branch must NOT fire (only a call would).
        let at = is_high_bw_workload_active(STORM_REFAULT_PAGES_PER_SEC, 0.40);
        assert_eq!(at, is_realtime_call_active(), "at-threshold → storm off");
        assert!(is_high_bw_workload_active(
            STORM_REFAULT_PAGES_PER_SEC * 2.0,
            0.40
        ));
    }

    #[test]
    fn realtime_call_implies_both_branches() {
        // Logical invariant — if realtime fires, both individual probes must agree.
        // Cannot fail spuriously: when both probes are false, composite is false.
        let composite = is_realtime_call_active();
        if composite {
            assert!(is_audio_running_somewhere(), "realtime requires output");
            assert!(is_audio_input_active(), "realtime requires input");
        }
    }
}
