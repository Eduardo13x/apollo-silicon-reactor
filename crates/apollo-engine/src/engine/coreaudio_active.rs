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
