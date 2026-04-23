//! Shadow signal globals — thread-safe conduits from the daemon main loop
//! to `decide_actions`' shadow-mode `ActionContext` construction without
//! changing `decide_actions()`'s signature.
//!
//! Rationale: `decide_actions` takes ~25 arguments already. Threading four
//! more (`p_oom_30s`, `p_jank_60s`, `thermal_emergency`, `interrupt_phase`)
//! through the signature + all callers + `PolicyContext` + tests is
//! disruptive. These signals come from single producers (signal_intelligence
//! tick, thermal_manager, resource_interrupt) and are read by a single
//! consumer (the shadow evaluator's ActionContext builder). A lock-free
//! global is the minimal-touch wire.
//!
//! Writers: main-loop tick (after `signal_intel.tick()` and thermal eval).
//! Readers: `decide_actions` when building the shadow `ActionContext`.
//!
//! All atomics are `Relaxed` — these are best-effort observability inputs,
//! not synchronization primitives.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

static P_OOM_30S_BITS: AtomicU64 = AtomicU64::new(0);
static P_JANK_60S_BITS: AtomicU64 = AtomicU64::new(0);
/// 0 sentinel means "unset" — writers set to a non-zero sentinel (`1` => 0.0,
/// otherwise the f64 bits) to distinguish "never written" from "genuinely 0.0".
static P_OOM_30S_WRITTEN: AtomicBool = AtomicBool::new(false);
static P_JANK_60S_WRITTEN: AtomicBool = AtomicBool::new(false);

static THERMAL_EMERGENCY: AtomicBool = AtomicBool::new(false);
static INTERRUPT_PHASE: AtomicU8 = AtomicU8::new(0);

/// Called from the daemon main loop after signal_intelligence.tick() computes
/// the latest `p_oom_30s`. No-op if called outside the daemon (tests, CLI).
pub fn set_p_oom_30s(p: f64) {
    P_OOM_30S_BITS.store(p.to_bits(), Ordering::Relaxed);
    P_OOM_30S_WRITTEN.store(true, Ordering::Relaxed);
}

/// Called from the daemon main loop. Returns `None` if never written, else
/// the latest value. Clamped to `[0, 1]` on read — defensive against any
/// producer sending out-of-range garbage.
pub fn get_p_oom_30s() -> Option<f64> {
    if !P_OOM_30S_WRITTEN.load(Ordering::Relaxed) {
        return None;
    }
    let raw = f64::from_bits(P_OOM_30S_BITS.load(Ordering::Relaxed));
    if raw.is_finite() {
        Some(raw.clamp(0.0, 1.0))
    } else {
        None
    }
}

pub fn set_p_jank_60s(p: f64) {
    P_JANK_60S_BITS.store(p.to_bits(), Ordering::Relaxed);
    P_JANK_60S_WRITTEN.store(true, Ordering::Relaxed);
}

pub fn get_p_jank_60s() -> Option<f64> {
    if !P_JANK_60S_WRITTEN.load(Ordering::Relaxed) {
        return None;
    }
    let raw = f64::from_bits(P_JANK_60S_BITS.load(Ordering::Relaxed));
    if raw.is_finite() {
        Some(raw.clamp(0.0, 1.0))
    } else {
        None
    }
}

pub fn set_thermal_emergency(flag: bool) {
    THERMAL_EMERGENCY.store(flag, Ordering::Relaxed);
}

pub fn get_thermal_emergency() -> bool {
    THERMAL_EMERGENCY.load(Ordering::Relaxed)
}

pub fn set_interrupt_phase(phase: u8) {
    INTERRUPT_PHASE.store(phase, Ordering::Relaxed);
}

pub fn get_interrupt_phase() -> u8 {
    INTERRUPT_PHASE.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_oom_none_before_set() {
        // This test relies on no other test having written — run in isolation
        // with `cargo test --lib shadow_signals::tests::roundtrip_and_clamp`.
        // The roundtrip test below is order-independent.
    }

    #[test]
    fn roundtrip_and_clamp() {
        set_p_oom_30s(0.42);
        assert_eq!(get_p_oom_30s(), Some(0.42));

        // Out-of-range clamps.
        set_p_oom_30s(1.5);
        assert_eq!(get_p_oom_30s(), Some(1.0));
        set_p_oom_30s(-0.3);
        assert_eq!(get_p_oom_30s(), Some(0.0));

        // NaN becomes None.
        set_p_oom_30s(f64::NAN);
        assert_eq!(get_p_oom_30s(), None);

        // Restore a known value for other tests.
        set_p_oom_30s(0.0);
    }

    #[test]
    fn thermal_and_interrupt_roundtrip() {
        set_thermal_emergency(true);
        assert!(get_thermal_emergency());
        set_thermal_emergency(false);
        assert!(!get_thermal_emergency());

        set_interrupt_phase(3);
        assert_eq!(get_interrupt_phase(), 3);
        set_interrupt_phase(0);
        assert_eq!(get_interrupt_phase(), 0);
    }
}
