//! User Presence Modulator — HID-event-aware action-aggressiveness suppression.
//!
//! Phase 5.1 (Sprint 8, 2026-05-16). Differentiates "active work" from
//! "background idle" using HID idle time + HID event rate, and emits a
//! multiplier ∈ [0.5, 1.0] applied to action aggressiveness when the user is
//! visibly working at the machine.
//!
//! Why a sibling pure function instead of extending `ArousalState`:
//! Phase 3.2 (`arousal_modulated_decay_factor`) already depends on the current
//! `ArousalState::level` interface. Touching that struct would risk a cascade
//! through Phase 3.2's invariant tests. The cheap fix: a stateless adapter
//! that takes the raw signals as arguments and returns a multiplier.
//!
//! Caller contract — the modulator is a pure suppression knob, not a
//! replacement for arousal:
//!   * input  `idle_seconds`        — seconds since last HID event (IOHIDSystem)
//!   * input  `hid_events_per_minute` — recent keyboard+mouse event rate
//!   * input  `current_arousal`     — `ArousalState::level` ∈ [0,1]
//!   * output multiplier ∈ [0.5, 1.0] to scale action-aggressiveness
//!
//! [Iqbal & Bailey 2008] "Effects of Interruptions on Task Performance" —
//! background activity during periods of high user engagement carries
//! disproportionate UX cost; the rational policy is to defer non-survival
//! optimization while the user is actively typing/clicking.
//!
//! NOTE: this module is **not** yet wired to any caller. The intended
//! injection site is in `decide_actions.rs` cost composition (multiply the
//! computed action cost by `1.0 / modulator` so non-survival actions become
//! "more expensive" relative to the gate threshold during active work), or
//! equivalently inside `daemon_cognitive_tick.rs::apply_specialist_voting`
//! where specialist confidence is already being modulated by the Phase 3.1
//! skill-aware factor. See `OPENS: 1` on the introducing commit.

use crate::engine::lse_counters::LSE_COUNTERS;

/// Arousal threshold above which the survival policy overrides UX-politeness.
///
/// Matches `ArousalState::zone()` "Crisis" boundary (0.80) — see
/// `nars_belief.rs`. Above this, the user-presence modulator returns 1.0
/// regardless of HID activity: if the system is in crisis (high p_oom,
/// thrashing) the cost of *not* acting outweighs the cost of a UX hiccup.
///
/// [Nygard 2018] "Release It!" — load shedding must override politeness
/// under overload.
pub const CRISIS_AROUSAL_THRESHOLD: f64 = 0.80;

/// Multiplier returned when the user is actively working at the machine.
/// 0.5 = "halve the aggressiveness of non-survival actions".
pub const ACTIVE_MULTIPLIER: f64 = 0.5;

/// Multiplier returned when the user is semi-active (typing intermittently
/// or recently idled). 0.75 = "trim aggressiveness by a quarter".
pub const SEMI_ACTIVE_MULTIPLIER: f64 = 0.75;

/// Multiplier returned when the user is truly idle (no recent HID activity).
/// 1.0 = "no suppression — Apollo can optimize freely".
pub const IDLE_MULTIPLIER: f64 = 1.0;

/// Idle threshold for the "active" tier (seconds).
/// Below this, the user is almost certainly at the keyboard.
const ACTIVE_IDLE_SECONDS: f64 = 5.0;

/// Idle threshold for the "semi-active" tier (seconds).
/// Below this, recent activity is still likely to be interrupted.
const SEMI_ACTIVE_IDLE_SECONDS: f64 = 30.0;

/// HID event-rate threshold for "active" tier (events/min).
/// Above this, sustained typing/clicking is in progress.
const ACTIVE_HID_EVENTS_PER_MIN: f64 = 30.0;

/// HID event-rate threshold for "semi-active" tier (events/min).
/// Above this, sporadic but real interaction is happening.
const SEMI_ACTIVE_HID_EVENTS_PER_MIN: f64 = 5.0;

/// Compute the user-presence suppression multiplier.
///
/// Returns a multiplier in `[ACTIVE_MULTIPLIER, IDLE_MULTIPLIER]` to apply to
/// the aggressiveness of non-survival actions. **Pure function** — no I/O,
/// no global mutation. The caller is responsible for collecting the inputs
/// (idle time from `user_context::collect_idle_secs`, event rate from the
/// activity sensor) and applying the multiplier at the cost-composition
/// site.
///
/// Side effect: increments `user_presence_suppressions_total` when the
/// returned multiplier is strictly less than 1.0, so dashboards can verify
/// the feature is actually firing in prod (mirrors the Phase 3.1 design
/// against the "scaffolding-without-wiring" anti-pattern flagged by
/// NotebookLM 2026-04-22).
///
/// Tier rules (first match wins):
///   1. `current_arousal >= CRISIS_AROUSAL_THRESHOLD` → 1.0 (survival overrides UX)
///   2. `idle_seconds < 5.0` OR `hid_events_per_minute > 30.0` → 0.5 (active)
///   3. `idle_seconds < 30.0` OR `hid_events_per_minute > 5.0` → 0.75 (semi-active)
///   4. else → 1.0 (idle)
pub fn user_presence_modulator(
    idle_seconds: f64,
    hid_events_per_minute: f64,
    current_arousal: f64,
) -> f64 {
    // (1) Crisis override: survival always wins over UX politeness.
    if current_arousal >= CRISIS_AROUSAL_THRESHOLD {
        return IDLE_MULTIPLIER;
    }

    // (2) Actively at the keyboard — strongest suppression.
    if idle_seconds < ACTIVE_IDLE_SECONDS || hid_events_per_minute > ACTIVE_HID_EVENTS_PER_MIN {
        LSE_COUNTERS.add_user_presence_suppressions(1);
        return ACTIVE_MULTIPLIER;
    }

    // (3) Semi-active — partial suppression.
    if idle_seconds < SEMI_ACTIVE_IDLE_SECONDS
        || hid_events_per_minute > SEMI_ACTIVE_HID_EVENTS_PER_MIN
    {
        LSE_COUNTERS.add_user_presence_suppressions(1);
        return SEMI_ACTIVE_MULTIPLIER;
    }

    // (4) Idle — no suppression, full aggressiveness.
    IDLE_MULTIPLIER
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tier 4 — long idle, no recent HID events, low arousal: no suppression.
    #[test]
    fn presence_idle_user_no_suppression() {
        let m = user_presence_modulator(120.0, 0.0, 0.2);
        assert!(
            (m - IDLE_MULTIPLIER).abs() < f64::EPSILON,
            "expected {} (idle), got {}",
            IDLE_MULTIPLIER,
            m
        );
    }

    /// Tier 2 — actively typing (idle < 5s): suppress to 0.5.
    #[test]
    fn presence_active_user_suppresses_to_half() {
        let m = user_presence_modulator(1.5, 60.0, 0.3);
        assert!(
            (m - ACTIVE_MULTIPLIER).abs() < f64::EPSILON,
            "expected {} (active), got {}",
            ACTIVE_MULTIPLIER,
            m
        );
    }

    /// Tier 3 — semi-active (idle in [5, 30)): suppress to 0.75.
    #[test]
    fn presence_semi_active_user_three_quarters() {
        // 20s since last HID event, low event rate, low arousal → tier 3.
        let m = user_presence_modulator(20.0, 2.0, 0.3);
        assert!(
            (m - SEMI_ACTIVE_MULTIPLIER).abs() < f64::EPSILON,
            "expected {} (semi-active), got {}",
            SEMI_ACTIVE_MULTIPLIER,
            m
        );
    }

    /// Tier 1 — crisis arousal overrides all suppression: return 1.0 even if
    /// the user is actively typing. Survival > UX.
    #[test]
    fn presence_crisis_arousal_overrides_suppression() {
        // User is actively typing (idle=1s, 80 ev/min) AND arousal is in
        // Crisis (>= 0.80). The crisis override must win.
        let m = user_presence_modulator(1.0, 80.0, 0.85);
        assert!(
            (m - IDLE_MULTIPLIER).abs() < f64::EPSILON,
            "crisis must override suppression: expected {}, got {}",
            IDLE_MULTIPLIER,
            m
        );
        // Boundary check: exactly at the Crisis threshold also overrides.
        let m_boundary = user_presence_modulator(1.0, 80.0, CRISIS_AROUSAL_THRESHOLD);
        assert!(
            (m_boundary - IDLE_MULTIPLIER).abs() < f64::EPSILON,
            "boundary arousal must override: got {}",
            m_boundary
        );
    }

    /// Tier 2 — high HID event rate triggers active tier even when the
    /// last HID event happened > 5s ago. The OR semantics catch bursty
    /// input (think a user dragging the mouse: many events per minute, but
    /// the "last event" timestamp can be stale by the time we sample).
    #[test]
    fn presence_high_hid_events_suppresses_even_when_idle() {
        // idle_seconds is "idle" but event rate is far above the active
        // threshold. The disjunction `idle < 5 OR events > 30` must still
        // trigger the active tier.
        let m = user_presence_modulator(45.0, 50.0, 0.3);
        assert!(
            (m - ACTIVE_MULTIPLIER).abs() < f64::EPSILON,
            "high HID rate must trigger active suppression even with stale idle: got {}",
            m
        );
    }

    // ── Boundary-condition tests for the tier thresholds ──────────────────

    #[test]
    fn presence_exact_active_idle_boundary_falls_through() {
        // idle_seconds == ACTIVE_IDLE_SECONDS is NOT < threshold → semi-active.
        let m = user_presence_modulator(5.0, 0.0, 0.3);
        assert!((m - SEMI_ACTIVE_MULTIPLIER).abs() < f64::EPSILON);
    }

    #[test]
    fn presence_exact_semi_active_idle_boundary_falls_through() {
        // idle_seconds == SEMI_ACTIVE_IDLE_SECONDS is NOT < threshold → idle.
        let m = user_presence_modulator(30.0, 0.0, 0.3);
        assert!((m - IDLE_MULTIPLIER).abs() < f64::EPSILON);
    }

    #[test]
    fn presence_multiplier_always_in_bounds() {
        // Property: regardless of inputs, the multiplier stays in [0.5, 1.0].
        for &idle in &[0.0_f64, 1.0, 5.0, 15.0, 29.9, 30.0, 120.0, 3600.0] {
            for &events in &[0.0_f64, 1.0, 5.0, 5.1, 30.0, 30.1, 100.0, 1000.0] {
                for &arousal in &[0.0_f64, 0.3, 0.5, 0.79, 0.80, 0.9, 1.0] {
                    let m = user_presence_modulator(idle, events, arousal);
                    assert!(
                        (ACTIVE_MULTIPLIER..=IDLE_MULTIPLIER).contains(&m),
                        "multiplier out of bounds for idle={}, events={}, arousal={}: {}",
                        idle,
                        events,
                        arousal,
                        m
                    );
                }
            }
        }
    }
}
