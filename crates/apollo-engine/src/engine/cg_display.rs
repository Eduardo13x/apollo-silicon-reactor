//! CoreGraphics display enumeration — count active displays + flag external 4K.
//!
//! Apollo needs to know when a 4K external monitor is attached because the
//! WindowServer compositor cost roughly doubles per cycle (extra render
//! pipeline + larger frame buffer) and background browser renderers
//! compete with the compositor for unified-memory bandwidth on M1 8GB.
//!
//! When `external_4k_attached == true` the chromium Step 2 gate tightens
//! to (0.65, 6_000) — same regime as build, between media and default.
//! Sample at ~5 s cadence; display topology rarely changes faster.
//!
//! Reference: Apple CGDirectDisplay.h
//!   CGGetActiveDisplayList(maxDisplays, activeDisplays, displayCount)
//!   CGDisplayPixelsWide(display) / CGDisplayPixelsHigh(display)
//!   CGMainDisplayID() returns the built-in / "main" display ID.

#[cfg(target_os = "macos")]
type CGDirectDisplayID = u32;
#[cfg(target_os = "macos")]
type CGError = i32;

/// Snapshot of the current display topology. Cheap value type; copy freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct DisplayState {
    /// Total number of active displays (built-in + external).
    pub display_count: u8,
    /// True when at least one display has a pixel area ≥ 4K (3840×2160).
    /// Includes the built-in display only on the rare 16-inch MBP M3 Max,
    /// so on the typical M1 8GB Air this flag is true iff an external 4K
    /// monitor is attached.
    pub external_4k_attached: bool,
}

/// 4K threshold in pixels. Slightly relaxed to catch "Ultra-wide 4K" and
/// "Studio Display" variants. The lower bound 3440x1440 ≈ 4.96 M pixels
/// is treated as "near-4K" because the compositor cost is dominated by
/// the pixel count regardless of aspect ratio.
#[cfg(target_os = "macos")]
const FOUR_K_PIXEL_AREA: u64 = 3_440 * 1_440; // 4.96 M pixels — generous

#[cfg(target_os = "macos")]
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGGetActiveDisplayList(
        max_displays: u32,
        active_displays: *mut CGDirectDisplayID,
        display_count: *mut u32,
    ) -> CGError;
    fn CGDisplayPixelsWide(display: CGDirectDisplayID) -> usize;
    fn CGDisplayPixelsHigh(display: CGDirectDisplayID) -> usize;
}

/// Snapshot the current display topology.
///
/// Cost: ~50µs per call. Caller is expected to cache for ~5 s.
///
/// Returns `DisplayState::default()` (count=0, no 4k) on any failure —
/// matches the cg_window.rs convention of "unknown ≡ skip gating".
#[cfg(target_os = "macos")]
pub fn snapshot() -> DisplayState {
    const MAX_DISPLAYS: usize = 8;
    let mut displays: [CGDirectDisplayID; MAX_DISPLAYS] = [0; MAX_DISPLAYS];
    let mut count: u32 = 0;
    let err = unsafe {
        CGGetActiveDisplayList(
            MAX_DISPLAYS as u32,
            displays.as_mut_ptr(),
            &mut count as *mut u32,
        )
    };
    if err != 0 || count == 0 {
        return DisplayState::default();
    }
    let count = count.min(MAX_DISPLAYS as u32);
    let mut any_4k = false;
    for &id in displays.iter().take(count as usize) {
        let w = unsafe { CGDisplayPixelsWide(id) } as u64;
        let h = unsafe { CGDisplayPixelsHigh(id) } as u64;
        if w.saturating_mul(h) >= FOUR_K_PIXEL_AREA {
            any_4k = true;
            break;
        }
    }
    DisplayState {
        display_count: count.min(255) as u8,
        external_4k_attached: any_4k,
    }
}

#[cfg(not(target_os = "macos"))]
pub fn snapshot() -> DisplayState {
    DisplayState::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_does_not_panic() {
        // On macOS: returns whatever CoreGraphics reports.
        // On other OSes: returns default (count=0).
        // Either way, must not panic.
        let st = snapshot();
        // count is bounded by MAX_DISPLAYS (8). cast to u32 to avoid lint on
        // `u8 < u8` always-true comparisons when MAX_DISPLAYS fits in u8.
        assert!((st.display_count as u32) < 32);
    }
}
