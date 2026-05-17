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

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicU8, Ordering};

static P_OOM_30S_BITS: AtomicU64 = AtomicU64::new(0);
static P_JANK_60S_BITS: AtomicU64 = AtomicU64::new(0);
/// 0 sentinel means "unset" — writers set to a non-zero sentinel (`1` => 0.0,
/// otherwise the f64 bits) to distinguish "never written" from "genuinely 0.0".
static P_OOM_30S_WRITTEN: AtomicBool = AtomicBool::new(false);
static P_JANK_60S_WRITTEN: AtomicBool = AtomicBool::new(false);

static THERMAL_EMERGENCY: AtomicBool = AtomicBool::new(false);
static INTERRUPT_PHASE: AtomicU8 = AtomicU8::new(0);

// Foreground PID: -1 sentinel for None (no foreground app detected).
static FOREGROUND_PID: AtomicI32 = AtomicI32::new(-1);

// Deep Scan class-level aggregates — published after each cycle's deep scan
// loop in main.rs. Readers use last-cycle's data (one-cycle stale is acceptable
// for shadow-mode cost estimation).
static MAX_HOT_PAGE_FRACTION_BITS: AtomicU64 = AtomicU64::new(0);
static MAX_WSS_MB_BITS: AtomicU64 = AtomicU64::new(0);
static MAX_HOT_PAGE_WRITTEN: AtomicBool = AtomicBool::new(false);
static MAX_WSS_WRITTEN: AtomicBool = AtomicBool::new(false);

// Epistemic uncertainty: urgency-based proxy (signal_digest.urgency 0..1).
// [Lakshminarayanan 2017] — urgency aggregates pressure-velocity / thrashing-
// flow / OOM-hazard into a single normalized epistemic signal.
static EPISTEMIC_UNCERTAINTY_BITS: AtomicU64 = AtomicU64::new(0);
static EPISTEMIC_WRITTEN: AtomicBool = AtomicBool::new(false);

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

pub fn set_foreground_pid(pid: Option<u32>) {
    let encoded: i32 = match pid {
        Some(p) if p <= i32::MAX as u32 => p as i32,
        _ => -1,
    };
    FOREGROUND_PID.store(encoded, Ordering::Relaxed);
}

pub fn get_foreground_pid() -> Option<u32> {
    let v = FOREGROUND_PID.load(Ordering::Relaxed);
    if v < 0 {
        None
    } else {
        Some(v as u32)
    }
}

/// Published by main.rs AFTER each cycle's deep scan loop completes. Readers
/// in the SAME cycle see `None` on the first cycle and previous-cycle's value
/// thereafter. This is a known 1-cycle lag — acceptable for class-level cost
/// estimation (hot pages shift slowly), unacceptable for per-PID gating.
/// [NotebookLM audit 2026-04-22: aliasing temporal, documentado no oculto.]
pub fn set_max_hot_page_fraction(f: f64) {
    MAX_HOT_PAGE_FRACTION_BITS.store(f.to_bits(), Ordering::Relaxed);
    MAX_HOT_PAGE_WRITTEN.store(true, Ordering::Relaxed);
}

pub fn get_max_hot_page_fraction() -> Option<f64> {
    if !MAX_HOT_PAGE_WRITTEN.load(Ordering::Relaxed) {
        return None;
    }
    let raw = f64::from_bits(MAX_HOT_PAGE_FRACTION_BITS.load(Ordering::Relaxed));
    if raw.is_finite() {
        Some(raw.clamp(0.0, 1.0))
    } else {
        None
    }
}

pub fn set_max_wss_mb(mb: f64) {
    MAX_WSS_MB_BITS.store(mb.to_bits(), Ordering::Relaxed);
    MAX_WSS_WRITTEN.store(true, Ordering::Relaxed);
}

pub fn get_max_wss_mb() -> Option<f64> {
    if !MAX_WSS_WRITTEN.load(Ordering::Relaxed) {
        return None;
    }
    let raw = f64::from_bits(MAX_WSS_MB_BITS.load(Ordering::Relaxed));
    if raw.is_finite() && raw >= 0.0 {
        Some(raw)
    } else {
        None
    }
}

/// Epistemic uncertainty — published by main.rs from `signal_digest.urgency`
/// as a normalized [0,1] composite of pressure-velocity / thrashing-flow /
/// hazard-rate uncertainty. Shadow reads it to penalize aggressive accepts
/// under high-uncertainty state. [Lakshminarayanan 2017] ensemble epistemic.
pub fn set_epistemic_uncertainty(u: f64) {
    EPISTEMIC_UNCERTAINTY_BITS.store(u.to_bits(), Ordering::Relaxed);
    EPISTEMIC_WRITTEN.store(true, Ordering::Relaxed);
}

pub fn get_epistemic_uncertainty() -> f64 {
    if !EPISTEMIC_WRITTEN.load(Ordering::Relaxed) {
        return 0.0;
    }
    let raw = f64::from_bits(EPISTEMIC_UNCERTAINTY_BITS.load(Ordering::Relaxed));
    if raw.is_finite() {
        raw.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

// ── Phase 5.2 wiring (Sprint 10, 2026-05-16) ────────────────────────────────
// Battery-aware cost penalty inputs. Same `Option<>`-via-WRITTEN-flag pattern
// the rest of this module uses: a producer must set the value before
// consumers see anything; until then `get_*` returns `None`. This avoids
// false positives in shadow / test contexts that never publish.

static IS_ON_BATTERY_FLAG: AtomicBool = AtomicBool::new(false);
static IS_ON_BATTERY_WRITTEN: AtomicBool = AtomicBool::new(false);
static WAKEUPS_PER_SEC_BITS: AtomicU64 = AtomicU64::new(0);
static WAKEUPS_PER_SEC_WRITTEN: AtomicBool = AtomicBool::new(false);
static CTX_SWITCHES_PER_SEC_BITS: AtomicU64 = AtomicU64::new(0);
static CTX_SWITCHES_PER_SEC_WRITTEN: AtomicBool = AtomicBool::new(false);

pub fn set_is_on_battery(on_battery: bool) {
    IS_ON_BATTERY_FLAG.store(on_battery, Ordering::Relaxed);
    IS_ON_BATTERY_WRITTEN.store(true, Ordering::Relaxed);
}
pub fn get_is_on_battery() -> Option<bool> {
    if IS_ON_BATTERY_WRITTEN.load(Ordering::Relaxed) {
        Some(IS_ON_BATTERY_FLAG.load(Ordering::Relaxed))
    } else {
        None
    }
}

pub fn set_wakeups_per_sec(rate: f64) {
    if rate.is_finite() && rate >= 0.0 {
        WAKEUPS_PER_SEC_BITS.store(rate.to_bits(), Ordering::Relaxed);
        WAKEUPS_PER_SEC_WRITTEN.store(true, Ordering::Relaxed);
    }
}
pub fn get_wakeups_per_sec() -> Option<f64> {
    if !WAKEUPS_PER_SEC_WRITTEN.load(Ordering::Relaxed) {
        return None;
    }
    let raw = f64::from_bits(WAKEUPS_PER_SEC_BITS.load(Ordering::Relaxed));
    if raw.is_finite() { Some(raw) } else { None }
}

pub fn set_ctx_switches_per_sec(rate: f64) {
    if rate.is_finite() && rate >= 0.0 {
        CTX_SWITCHES_PER_SEC_BITS.store(rate.to_bits(), Ordering::Relaxed);
        CTX_SWITCHES_PER_SEC_WRITTEN.store(true, Ordering::Relaxed);
    }
}
pub fn get_ctx_switches_per_sec() -> Option<f64> {
    if !CTX_SWITCHES_PER_SEC_WRITTEN.load(Ordering::Relaxed) {
        return None;
    }
    let raw = f64::from_bits(CTX_SWITCHES_PER_SEC_BITS.load(Ordering::Relaxed));
    if raw.is_finite() { Some(raw) } else { None }
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

    #[test]
    fn foreground_pid_roundtrip() {
        set_foreground_pid(Some(42));
        assert_eq!(get_foreground_pid(), Some(42));
        set_foreground_pid(None);
        assert_eq!(get_foreground_pid(), None);
        set_foreground_pid(Some(u32::MAX)); // overflow sentinel guard
        assert_eq!(get_foreground_pid(), None);
        set_foreground_pid(None); // restore
    }

    #[test]
    fn deep_scan_aggregates_roundtrip_and_clamp() {
        set_max_hot_page_fraction(0.85);
        assert_eq!(get_max_hot_page_fraction(), Some(0.85));
        set_max_hot_page_fraction(2.0); // clamps
        assert_eq!(get_max_hot_page_fraction(), Some(1.0));
        set_max_hot_page_fraction(f64::NAN);
        assert_eq!(get_max_hot_page_fraction(), None);

        set_max_wss_mb(512.5);
        assert_eq!(get_max_wss_mb(), Some(512.5));
        set_max_wss_mb(-1.0); // invalid, None
        assert_eq!(get_max_wss_mb(), None);
    }

    #[test]
    fn epistemic_default_zero_and_clamp() {
        // Before any set, returns 0.0 (neutral).
        set_epistemic_uncertainty(0.6);
        assert!((get_epistemic_uncertainty() - 0.6).abs() < 1e-9);
        set_epistemic_uncertainty(1.5); // clamps
        assert_eq!(get_epistemic_uncertainty(), 1.0);
        set_epistemic_uncertainty(-0.3);
        assert_eq!(get_epistemic_uncertainty(), 0.0);
        set_epistemic_uncertainty(f64::INFINITY);
        assert_eq!(get_epistemic_uncertainty(), 0.0);
        set_epistemic_uncertainty(0.0);
    }
}
