//! Sysctl value-range clamp helper.
//!
//! Single-source-of-truth for clamping a proposed sysctl integer value to the
//! allowlisted range from `safety::allowlisted_sysctls_with_ranges()`.
//!
//! Sprint 4 Phase 4 (2026-05-07) — extracted from `sysctl_governor` to remove
//! the cyclic implication that only the governor needs to clamp. Bug 6 (the
//! `network-optimizer` site at main.rs:3577 emitting raw 4 MB buffers) was
//! caused by exactly that assumption: a non-governor emitter constructed
//! `RootAction::SetSysctl { value: "4194304", .. }` via struct literal,
//! bypassing the four governor-internal clamp sites.
//!
//! With Sprint 4 Phase 4 the only construction path is
//! `SetSysctlAction::new_clamped(...)`, which routes every `value` through
//! `clamp_to_allowed_range` automatically.
//!
//! [Anti-Corruption Layer Pattern — 1001 patterns slide 48]

/// Clamp a proposed sysctl value to the allowed range from
/// `safety::allowlisted_sysctls_with_ranges()`.
///
/// Returns the clamped value (always within allowed range), or the original
/// if no range exists for this key (execute_actions rejects non-allowlist
/// keys with `BlockReason::InvalidSysctl` — defense in depth).
pub fn clamp_to_allowed_range(key: &str, proposed: i64) -> i64 {
    let ranges = crate::engine::safety::allowlisted_sysctls_with_ranges();
    if let Some(r) = ranges.iter().find(|r| r.key == key) {
        proposed.clamp(r.min, r.max)
    } else {
        // Key not in allowlist — execute_actions will reject with
        // InvalidSysctl. Pass through unchanged so the failure surfaces.
        proposed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_value_within_range_passes_through() {
        // First, find an actual range from safety to use realistic values.
        let ranges = crate::engine::safety::allowlisted_sysctls_with_ranges();
        if let Some(r) = ranges.first() {
            // Pick a value in the middle of the range
            let mid = (r.min + r.max) / 2;
            let v = clamp_to_allowed_range(&r.key, mid);
            assert_eq!(v, mid, "value within range must pass through unchanged");
        }
    }

    #[test]
    fn clamp_value_above_max_clamps_to_max() {
        let ranges = crate::engine::safety::allowlisted_sysctls_with_ranges();
        if let Some(r) = ranges.first() {
            let v = clamp_to_allowed_range(&r.key, r.max + 999_999);
            assert_eq!(v, r.max, "out-of-range high must clamp to max");
        }
    }

    #[test]
    fn clamp_value_below_min_clamps_to_min() {
        let ranges = crate::engine::safety::allowlisted_sysctls_with_ranges();
        if let Some(r) = ranges.first() {
            let v = clamp_to_allowed_range(&r.key, r.min - 1);
            assert_eq!(v, r.min, "out-of-range low must clamp to min");
        }
    }

    #[test]
    fn clamp_unknown_key_passes_through() {
        // Unknown keys: pass through unchanged (execute_actions catches via
        // InvalidSysctl block reason)
        let v = clamp_to_allowed_range("not.in.allowlist", 12345);
        assert_eq!(v, 12345);
    }
}
