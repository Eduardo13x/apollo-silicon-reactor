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
//! Cost: ~50µs per call (two `AudioObjectGetPropertyData` round-trips).
//! Cached at the same 3-cycle cadence as the existing pmset poll, so net
//! cost in the daemon hot path is negligible.

#[cfg(target_os = "macos")]
use std::mem;

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

/// True when audio is actively flowing through the default output device.
///
/// Returns `false` on any error (no default output, query failure, non-macOS).
/// Errors are silent because this signal is OR'd with other media indicators
/// — a missed detection only weakens the gate; never falsely fires it.
#[cfg(target_os = "macos")]
pub fn is_audio_running_somewhere() -> bool {
    unsafe {
        // Step 1: resolve default output device id.
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
            return false;
        }

        // Step 2: query is-running-somewhere on that device.
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
            return false;
        }
        running != 0
    }
}

#[cfg(not(target_os = "macos"))]
pub fn is_audio_running_somewhere() -> bool {
    false
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
pub fn is_audio_input_active() -> bool {
    unsafe {
        // Step 1: resolve default input device id.
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
            return false;
        }

        // Step 2: query is-running-somewhere on that input device.
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
            return false;
        }
        running != 0
    }
}

#[cfg(not(target_os = "macos"))]
pub fn is_audio_input_active() -> bool {
    false
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
    is_audio_running_somewhere() && is_audio_input_active()
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
    is_realtime_call_active() || refault_pages_per_sec > STORM_REFAULT_PAGES_PER_SEC
}

#[cfg(test)]
mod tests {
    use super::*;

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
