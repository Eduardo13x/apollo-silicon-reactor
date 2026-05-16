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

/// Narrowed "active" tier multiplier — Phase 5.1 wiring (2026-05-16).
///
/// NotebookLM peer-review post-Step-2 recommended narrowing the active-tier
/// floor from 0.5 → 0.7 before wiring into `apply_specialist_voting`. The
/// 0.5 floor multiplied with `SpecialistAccuracyTracker::weight()` (≈0.6),
/// `skill_aware_factor` (0.85 at min damp) and a typical specialist confidence
/// of 0.7 yields `0.6 × 0.85 × 0.7 × 0.5 = 0.179`, below the disagreement-
/// safety floor (0.4). Result: every active-user cycle would collapse to
/// Observe regardless of physical pressure ("Regression Paralysis"). With
/// 0.7 the cascade min lifts to `0.6 × 0.85 × 0.7 × 0.7 = 0.250` — still
/// roughly half typical confidence, but above the floor.
///
/// Used by [`user_presence_modulator_narrowed`].
pub const ACTIVE_MULTIPLIER_NARROWED: f64 = 0.7;

/// Narrowed "semi-active" tier multiplier — Phase 5.1 wiring.
///
/// 0.85 mirrors the same logic as [`ACTIVE_MULTIPLIER_NARROWED`]: keep the
/// suppressive bite informative (15% damp) without dragging the cascade
/// under the disagreement-safety floor. Centred symmetrically around the
/// active-tier narrowing (0.7 → 0.85 → 1.0 = 15-point increments).
pub const SEMI_ACTIVE_MULTIPLIER_NARROWED: f64 = 0.85;

/// Phase 5.1 wiring — narrowed-band variant of
/// [`user_presence_modulator_with_passive`].
///
/// Identical decision tree to `_with_passive` except the band is
/// `[0.7, 1.0]` instead of `[0.5, 1.0]`. Preferred entry-point for the
/// `apply_specialist_voting` wiring site. The wider-band variants remain
/// exported for callers that want the original [Iqbal & Bailey 2008]
/// damping profile (e.g., future `decide_actions.rs` cost composition).
///
/// Tier rules (first match wins):
///   1. `current_arousal >= CRISIS_AROUSAL_THRESHOLD` → 1.0 (survival > UX)
///   2. `audio_active || has_sleep_assertion` → 1.0 (passive content)
///   3. active (idle < 5s OR hid_rate > 30) → 0.7
///   4. semi-active (idle < 30s OR hid_rate > 5) → 0.85
///   5. else → 1.0
///
/// Side effects: increments `user_presence_suppressions_total` ONCE per call
/// that returns `< 1.0` (mirroring the base function's contract). The caller
/// in `apply_specialist_voting` does NOT bump the counter again — that would
/// double-count. The "fire per modulated vote" semantics requested by
/// NotebookLM are implemented at the call site by issuing the
/// `add_user_presence_suppressions(modulated)` only when this function
/// returns `< 1.0`, with `modulated` equal to the number of non-Observe
/// votes scaled in that cycle. To prevent the double-count we suppress the
/// in-function increment via [`user_presence_modulator_narrowed_no_counter`],
/// the variant invoked by the wiring site.
pub fn user_presence_modulator_narrowed(
    idle_seconds: f64,
    hid_events_per_minute: f64,
    current_arousal: f64,
    audio_active: bool,
    has_sleep_assertion: bool,
) -> f64 {
    if current_arousal >= CRISIS_AROUSAL_THRESHOLD {
        return IDLE_MULTIPLIER;
    }
    if audio_active || has_sleep_assertion {
        return IDLE_MULTIPLIER;
    }
    if idle_seconds < ACTIVE_IDLE_SECONDS || hid_events_per_minute > ACTIVE_HID_EVENTS_PER_MIN {
        LSE_COUNTERS.add_user_presence_suppressions(1);
        return ACTIVE_MULTIPLIER_NARROWED;
    }
    if idle_seconds < SEMI_ACTIVE_IDLE_SECONDS
        || hid_events_per_minute > SEMI_ACTIVE_HID_EVENTS_PER_MIN
    {
        LSE_COUNTERS.add_user_presence_suppressions(1);
        return SEMI_ACTIVE_MULTIPLIER_NARROWED;
    }
    IDLE_MULTIPLIER
}

/// Counter-free twin of [`user_presence_modulator_narrowed`] for the
/// `apply_specialist_voting` wiring site, which increments the counter once
/// per modulated vote (NotebookLM 2026-05-16 recommendation) rather than
/// once per call.
///
/// Returns the same multiplier set `{0.7, 0.85, 1.0}` with identical
/// decision logic; only the counter side-effect is omitted. The wiring
/// site is the SOLE caller — do not export under a more general name.
pub fn user_presence_modulator_narrowed_no_counter(
    idle_seconds: f64,
    hid_events_per_minute: f64,
    current_arousal: f64,
    audio_active: bool,
    has_sleep_assertion: bool,
) -> f64 {
    if current_arousal >= CRISIS_AROUSAL_THRESHOLD {
        return IDLE_MULTIPLIER;
    }
    if audio_active || has_sleep_assertion {
        return IDLE_MULTIPLIER;
    }
    if idle_seconds < ACTIVE_IDLE_SECONDS || hid_events_per_minute > ACTIVE_HID_EVENTS_PER_MIN {
        return ACTIVE_MULTIPLIER_NARROWED;
    }
    if idle_seconds < SEMI_ACTIVE_IDLE_SECONDS
        || hid_events_per_minute > SEMI_ACTIVE_HID_EVENTS_PER_MIN
    {
        return SEMI_ACTIVE_MULTIPLIER_NARROWED;
    }
    IDLE_MULTIPLIER
}

/// Phase 5.1 (Gap B, 2026-05-16) — passive-content-aware variant.
///
/// Extends [`user_presence_modulator`] with two binary signals from
/// [`UserContext`](super::user_context::UserContext):
///
///   * `audio_active`        — coreaudiod is producing output OR CoreAudio
///                             reports the default output device running
///   * `has_sleep_assertion` — some non-Apollo process holds an IOKit
///                             `PreventUserIdle*Sleep` assertion (after Step 1,
///                             that includes `PreventUserIdleDisplaySleep`)
///
/// Tier rules (first match wins):
///   1. `current_arousal >= CRISIS_AROUSAL_THRESHOLD` → 1.0 (survival > UX,
///      identical to the base modulator)
///   2. `audio_active || has_sleep_assertion` → 1.0
///      (passive-content override: the user is consuming media or an app
///      has explicitly asked the kernel to preserve wakefulness — throttling
///      now is not a UX win, it's a stutter. Note: this returns 1.0, **not**
///      a suppression multiplier — Apollo can still act on non-Observe
///      votes, just without the extra penalty for "user is at keyboard".)
///   3. else → delegate to [`user_presence_modulator`] (base band logic)
///
/// Why "passive content" wins over "active user":
///   The base modulator treats `idle < 5s` + high HID rate as "active typing"
///   and damps to 0.5. But a user watching a movie on the external display
///   is `idle = 0s` (cursor still warm), `hid_rate = low`, and `audio_active
///   = true`. Damping to 0.5 there penalises the very workload that needs
///   smooth playback. The passive override returns 1.0 — no suppression of
///   the throttle-class actions that keep memory healthy underneath the
///   media playback.
///
/// Why crisis still beats passive:
///   In a true OOM-imminent state (arousal ≥ 0.80), continued audio playback
///   is going to stutter anyway because the kernel is about to start jetsam
///   killing. Acting now (1.0 multiplier = full aggressiveness on votes,
///   same as the survival policy) is strictly better than waiting for the
///   audio to die naturally.
///
/// [Iqbal & Bailey 2008] "Effects of Interruptions on Task Performance" —
/// passive content consumption is a different cost profile from active
/// keyboard work: the relevant cost is glitch-free output, not
/// interruption-free attention. A multiplier of 0.5 (the "active" tier)
/// over-corrects for the passive case.
///
/// Side effects: increments `user_presence_suppressions_total` ONLY when the
/// returned multiplier is `< 1.0`; the passive/crisis pass-through paths
/// (which return 1.0) deliberately do not bump the counter — they are
/// non-events for the dashboard purpose of "verify the feature is firing".
pub fn user_presence_modulator_with_passive(
    idle_seconds: f64,
    hid_events_per_minute: f64,
    current_arousal: f64,
    audio_active: bool,
    has_sleep_assertion: bool,
) -> f64 {
    // (1) Crisis override: survival always wins over UX politeness.
    //     Identical to the base modulator's first check.
    if current_arousal >= CRISIS_AROUSAL_THRESHOLD {
        return IDLE_MULTIPLIER;
    }

    // (2) Passive-content override: media playback / display-keep-awake
    //     assertion. Apollo can still optimize, but the "active user"
    //     penalty is wrong here — return 1.0 (full aggressiveness on votes).
    if audio_active || has_sleep_assertion {
        return IDLE_MULTIPLIER;
    }

    // (3) Delegate to the base band logic for the active/semi-active/idle
    //     tiers. The base function owns the counter increment for the
    //     suppressing branches — we do not double-count here.
    user_presence_modulator(idle_seconds, hid_events_per_minute, current_arousal)
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

    // ── Phase 5.1 Gap B — passive-content override ────────────────────────

    /// `audio_active=true` overrides the "active user" tier — the user is
    /// listening to music while typing; the typing should not penalise the
    /// throttle-class actions that keep memory healthy.
    #[test]
    fn presence_passive_audio_overrides_active_user_no_suppression() {
        // Inputs that would normally yield ACTIVE_MULTIPLIER (0.5):
        //   idle < 5s, high HID rate, sub-crisis arousal.
        let m = user_presence_modulator_with_passive(
            1.5,  // active idle
            60.0, // active HID rate
            0.3,  // sub-crisis
            true, // audio_active
            false,
        );
        assert!(
            (m - IDLE_MULTIPLIER).abs() < f64::EPSILON,
            "audio_active must override active-user suppression: got {m}"
        );
    }

    /// `has_sleep_assertion=true` overrides the "active user" tier — an app
    /// is keeping the display awake (presentation, streaming on external
    /// monitor) and we should not throttle.
    #[test]
    fn presence_passive_sleep_assertion_overrides_no_suppression() {
        let m = user_presence_modulator_with_passive(
            1.5,
            60.0,
            0.3,
            false,
            true, // has_sleep_assertion
        );
        assert!(
            (m - IDLE_MULTIPLIER).abs() < f64::EPSILON,
            "has_sleep_assertion must override active-user suppression: got {m}"
        );
    }

    /// Crisis arousal still wins over the passive-content override — if the
    /// system is on the verge of OOM, audio is going to stutter regardless;
    /// act now (returns 1.0 = full aggressiveness, same shape as both other
    /// overrides but with strictly higher precedence).
    ///
    /// This test exercises the precedence: a single function with both
    /// crisis AND passive flags set must still return 1.0 (which both
    /// branches happen to do), and the early-return on crisis means the
    /// passive branch is never evaluated.
    #[test]
    fn presence_crisis_still_overrides_passive_content() {
        let m = user_presence_modulator_with_passive(
            1.5,
            60.0,
            0.90, // crisis
            true, // also passive
            true,
        );
        assert!(
            (m - IDLE_MULTIPLIER).abs() < f64::EPSILON,
            "crisis must override even when passive flags are set: got {m}"
        );
    }

    /// Passive flag set, low arousal, otherwise-idle inputs: still 1.0.
    /// This is the no-op case — the user is consuming media and Apollo is
    /// fully free to act (no suppression in the base path either).
    #[test]
    fn presence_passive_without_arousal_returns_one() {
        let m = user_presence_modulator_with_passive(
            120.0, // long idle
            0.0,   // no HID events
            0.2,   // far from crisis
            true,  // audio playing in the background
            false,
        );
        assert!(
            (m - IDLE_MULTIPLIER).abs() < f64::EPSILON,
            "passive + idle + low arousal → 1.0: got {m}"
        );
    }

    /// When NEITHER passive flag is set, the function delegates to the base
    /// band. Property: for any (idle, hid, arousal) tuple, the no-passive
    /// path of `_with_passive` returns the same value as the base
    /// `user_presence_modulator`.
    #[test]
    fn presence_non_passive_delegates_to_base_band() {
        // Sample the active, semi-active, and idle tiers and the crisis
        // override path of the base modulator and check identity.
        let cases: &[(f64, f64, f64)] = &[
            (1.5, 60.0, 0.3),   // tier 2 (active)
            (20.0, 2.0, 0.3),   // tier 3 (semi-active)
            (120.0, 0.0, 0.3),  // tier 4 (idle)
            (1.0, 80.0, 0.85),  // tier 1 (crisis)
        ];
        for &(idle, hid, arousal) in cases {
            let base = user_presence_modulator(idle, hid, arousal);
            let extended =
                user_presence_modulator_with_passive(idle, hid, arousal, false, false);
            assert!(
                (extended - base).abs() < f64::EPSILON,
                "no-passive delegation must match base: idle={idle}, hid={hid}, \
                 arousal={arousal} → base={base}, extended={extended}"
            );
        }
    }

    // ── Phase 5.1 wiring — narrowed band [0.7, 1.0] ───────────────────────

    /// Active tier (idle < 5s) under the narrowed band → 0.7.
    #[test]
    fn presence_narrowed_active_returns_07() {
        let m = user_presence_modulator_narrowed(1.5, 60.0, 0.3, false, false);
        assert!(
            (m - ACTIVE_MULTIPLIER_NARROWED).abs() < f64::EPSILON,
            "narrowed active tier must be 0.7, got {m}"
        );
    }

    /// Semi-active tier (idle in [5, 30)) under the narrowed band → 0.85.
    #[test]
    fn presence_narrowed_semi_active_returns_085() {
        let m = user_presence_modulator_narrowed(20.0, 2.0, 0.3, false, false);
        assert!(
            (m - SEMI_ACTIVE_MULTIPLIER_NARROWED).abs() < f64::EPSILON,
            "narrowed semi-active tier must be 0.85, got {m}"
        );
    }

    /// Crisis + passive overrides still return 1.0 under the narrowed band.
    /// The no-counter twin returns the same multipliers as the counter
    /// variant, by construction. Guard against future drift.
    #[test]
    fn presence_narrowed_no_counter_matches_counter_variant() {
        let cases: &[(f64, f64, f64, bool, bool)] = &[
            (1.5, 60.0, 0.3, false, false), // active
            (20.0, 2.0, 0.3, false, false), // semi-active
            (120.0, 0.0, 0.2, false, false), // idle
            (1.0, 80.0, 0.85, false, false), // crisis
            (1.0, 80.0, 0.3, true, false),   // passive audio
            (1.0, 80.0, 0.3, false, true),   // passive sleep assertion
        ];
        for &(idle, hid, arousal, audio, assertion) in cases {
            let with_counter = user_presence_modulator_narrowed(
                idle, hid, arousal, audio, assertion,
            );
            let no_counter = user_presence_modulator_narrowed_no_counter(
                idle, hid, arousal, audio, assertion,
            );
            assert!(
                (with_counter - no_counter).abs() < f64::EPSILON,
                "narrowed twins diverged at idle={idle}, hid={hid}, arousal={arousal}, \
                 audio={audio}, assertion={assertion}: with={with_counter}, no={no_counter}"
            );
        }
    }

    #[test]
    fn presence_narrowed_overrides_preserved() {
        // Crisis
        let crisis = user_presence_modulator_narrowed(1.0, 80.0, 0.85, false, false);
        assert!((crisis - IDLE_MULTIPLIER).abs() < f64::EPSILON);
        // Passive (audio)
        let audio = user_presence_modulator_narrowed(1.0, 80.0, 0.3, true, false);
        assert!((audio - IDLE_MULTIPLIER).abs() < f64::EPSILON);
        // Passive (sleep assertion)
        let assertion = user_presence_modulator_narrowed(1.0, 80.0, 0.3, false, true);
        assert!((assertion - IDLE_MULTIPLIER).abs() < f64::EPSILON);
        // Idle (long, no HID, low arousal)
        let idle = user_presence_modulator_narrowed(120.0, 0.0, 0.2, false, false);
        assert!((idle - IDLE_MULTIPLIER).abs() < f64::EPSILON);
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
