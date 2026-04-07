//! ══════════════════════════════════════════════════════════════════════════════
//! Apollo AutoResearch — Latency Monitor Benchmark
//! ══════════════════════════════════════════════════════════════════════════════
//!
//! THIS FILE IS READ-ONLY. The agent must NEVER modify it.
//!
//! Tests whether latency_monitor correctly classifies UI responsiveness
//! across realistic scenarios. The goal is to ensure:
//!   - True responsive states aren't flagged as needing boost (false positives)
//!   - True sluggish/broken states are always caught (false negatives)
//!   - Boost threshold triggers at the right severity level
//!   - Signal weights produce correct rankings across edge cases
//!
//! Target file: src/engine/latency_monitor.rs

#[cfg(test)]
mod scenarios {
    use apollo_optimizer::engine::latency_monitor::{
        compute_latency, LatencyCategory, LatencySignals,
    };

    fn nominal() -> LatencySignals {
        LatencySignals {
            jitter_us: 30.0,
            windowserver_cpu: 10.0,
            foreground_cpu: 15.0,
            foreground_csw_per_sec: 200.0,
            has_foreground: true,
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 1: FALSE POSITIVE PREVENTION
    // Scenarios where the system is fine — boost must NOT trigger.
    // ══════════════════════════════════════════════════════════════════════════

    /// L01: Perfectly nominal system — score must be well under 0.3.
    #[test]
    fn l01_nominal_is_responsive() {
        let s = compute_latency(&nominal());
        assert_eq!(s.category, LatencyCategory::Responsive);
        assert!(!s.needs_boost, "nominal system should not need boost");
        assert!(
            s.score < 0.15,
            "nominal score should be near zero, got {}",
            s.score
        );
    }

    /// L02: Mild jitter (300µs) alone — common during context switches.
    /// Should NOT trigger boost. This is normal M1 behavior under moderate load.
    #[test]
    fn l02_mild_jitter_no_boost() {
        let mut sig = nominal();
        sig.jitter_us = 300.0;
        let s = compute_latency(&sig);
        assert!(
            !s.needs_boost,
            "300µs jitter is normal M1 variance — no boost. Score: {}",
            s.score
        );
    }

    /// L03: WindowServer at 25% is normal during window resize/animation.
    /// Should NOT trigger boost (it's just compositing, not a backlog).
    #[test]
    fn l03_moderate_ws_no_boost() {
        let mut sig = nominal();
        sig.windowserver_cpu = 25.0;
        let s = compute_latency(&sig);
        assert!(
            !s.needs_boost,
            "WS at 25% is normal compositing — no boost. Score: {}",
            s.score
        );
    }

    /// L04: High context switches (3000/s) alone with everything else nominal.
    /// Should be Noticeable at most, but NOT trigger boost — CSW alone
    /// doesn't mean the foreground is suffering.
    #[test]
    fn l04_high_csw_alone_no_boost() {
        let mut sig = nominal();
        sig.foreground_csw_per_sec = 3000.0;
        let s = compute_latency(&sig);
        assert!(
            !s.needs_boost,
            "High CSW alone shouldn't trigger boost — foreground may be fine. Score: {}",
            s.score
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 2: TRUE POSITIVE — MUST DETECT
    // Scenarios where the UI is genuinely degraded — boost MUST trigger.
    // ══════════════════════════════════════════════════════════════════════════

    /// L05: Jitter at 1500µs + WS at 40% — compositor falling behind.
    /// This is a real "jank" scenario: user sees stuttered animations.
    #[test]
    fn l05_jitter_plus_ws_triggers_boost() {
        let mut sig = nominal();
        sig.jitter_us = 1500.0;
        sig.windowserver_cpu = 40.0;
        let s = compute_latency(&sig);
        assert!(
            s.needs_boost,
            "Jitter 1500µs + WS 40% = real jank — must boost. Score: {}",
            s.score
        );
    }

    /// L06: Foreground completely starved (0% CPU) — app frozen to user.
    /// Even with everything else nominal, this is broken.
    #[test]
    fn l06_fg_starved_triggers_boost() {
        let mut sig = nominal();
        sig.foreground_cpu = 0.0;
        sig.jitter_us = 400.0; // slight jitter from whatever stole the CPU
        let s = compute_latency(&sig);
        assert!(
            s.needs_boost,
            "FG at 0% CPU = completely starved — must boost. Score: {}",
            s.score
        );
    }

    /// L07: Full system stress — everything bad. Must be Sluggish or Broken.
    #[test]
    fn l07_full_stress_is_broken() {
        let sig = LatencySignals {
            jitter_us: 5000.0,
            windowserver_cpu: 70.0,
            foreground_cpu: 0.5,
            foreground_csw_per_sec: 8000.0,
            has_foreground: true,
        };
        let s = compute_latency(&sig);
        assert!(
            s.category == LatencyCategory::Sluggish || s.category == LatencyCategory::Broken,
            "Full stress must be Sluggish or Broken, got {:?} ({})",
            s.category,
            s.score
        );
        assert!(s.needs_boost);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 3: EDGE CASES & SENSITIVITY
    // Tricky scenarios that test the weight balance.
    // ══════════════════════════════════════════════════════════════════════════

    /// L08: Two moderate signals together should outweigh one bad signal.
    /// Jitter 800µs + WS 35% should score HIGHER than jitter 2000µs alone,
    /// because two concurrent degradations compound the user impact.
    #[test]
    fn l08_two_moderate_beats_one_bad() {
        let mut two_mod = nominal();
        two_mod.jitter_us = 800.0;
        two_mod.windowserver_cpu = 35.0;

        let mut one_bad = nominal();
        one_bad.jitter_us = 2000.0;

        let s_two = compute_latency(&two_mod);
        let s_one = compute_latency(&one_bad);
        assert!(
            s_two.score >= s_one.score,
            "Two moderate degradations ({:.3}) should score >= one bad ({:.3})",
            s_two.score,
            s_one.score
        );
    }

    /// L09: No foreground app → always responsive, regardless of other signals.
    #[test]
    fn l09_no_foreground_always_responsive() {
        let sig = LatencySignals {
            jitter_us: 5000.0,
            windowserver_cpu: 80.0,
            foreground_cpu: 0.0,
            foreground_csw_per_sec: 10000.0,
            has_foreground: false,
        };
        let s = compute_latency(&sig);
        assert_eq!(s.category, LatencyCategory::Responsive);
        assert!(!s.needs_boost);
    }

    /// L10: Jitter at exactly the "bad" threshold (2000µs) with nominal rest.
    /// Should be Noticeable (borderline) but the single signal shouldn't push
    /// past the boost threshold — jitter alone at 35% weight maxes at 0.35.
    #[test]
    fn l10_jitter_at_bad_threshold_is_noticeable() {
        let mut sig = nominal();
        sig.jitter_us = 2000.0;
        let s = compute_latency(&sig);
        assert!(
            s.category == LatencyCategory::Noticeable || s.category == LatencyCategory::Responsive,
            "Single bad signal should be at most Noticeable, got {:?} ({})",
            s.category,
            s.score
        );
    }
}
