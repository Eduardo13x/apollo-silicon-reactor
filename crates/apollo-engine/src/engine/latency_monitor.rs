//! Perceptual Latency Monitor — composite score for UI responsiveness.
//!
//! Instead of direct HID event tracking (requires IOHIDManager + CFRunLoop,
//! fragile in root daemon context), this module combines existing signals
//! into a perceptual latency score:
//!
//!   1. Schedule jitter (from hw_predictor): high jitter = CPU contention
//!   2. WindowServer CPU%: high = compositing backlog, frame drops
//!   3. Foreground app CPU usage: very low = starved, very high = overloaded
//!   4. Context switch rate: high = scheduler thrashing
//!
//! Score interpretation (Card, Moran & Newell 1983):
//!   0.0–0.3: responsive (<100ms perceived latency)
//!   0.3–0.6: noticeable (100–300ms)
//!   0.6–0.8: sluggish (300ms–1s)
//!   0.8–1.0: broken (>1s)
//!
//! When score > 0.5, Apollo should immediately boost the foreground app
//! and throttle background noise more aggressively.

/// Per-process CPU and scheduler thresholds. These inputs use macOS-normalized
/// units and are independent of the number of physical cores.
const JITTER_NOMINAL_US: f64 = 50.0;
const JITTER_BAD_US: f64 = 2000.0;
const WS_CPU_NOMINAL: f64 = 15.0;
const WS_CPU_BAD: f64 = 60.0;
const CSW_NOMINAL: f64 = 500.0; // context switches/s
const CSW_BAD: f64 = 5000.0;

/// Input signals for latency estimation.
#[derive(Debug, Clone)]
pub struct LatencySignals {
    /// Schedule jitter in microseconds (from hw_predictor).
    pub jitter_us: f64,
    /// WindowServer CPU usage (0–100).
    pub windowserver_cpu: f64,
    /// Foreground app CPU usage (0–100). 0 if no foreground.
    pub foreground_cpu: f64,
    /// Foreground app context switches per second.
    pub foreground_csw_per_sec: f64,
    /// True if there is an active foreground app.
    pub has_foreground: bool,
}

/// Result of latency analysis.
#[derive(Debug, Clone)]
pub struct LatencyScore {
    /// Composite latency score [0.0, 1.0]. Higher = worse.
    pub score: f64,
    /// True if immediate foreground boost is recommended.
    pub needs_boost: bool,
    /// Human-readable category.
    pub category: LatencyCategory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyCategory {
    /// <100ms perceived: no action needed.
    Responsive,
    /// 100–300ms: user notices slight delay.
    Noticeable,
    /// 300ms–1s: user perceives sluggishness.
    Sluggish,
    /// >1s: UI feels broken.
    Broken,
}

impl LatencyCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Responsive => "responsive",
            Self::Noticeable => "noticeable",
            Self::Sluggish => "sluggish",
            Self::Broken => "broken",
        }
    }
}

/// Compute the perceptual latency score from available signals.
pub fn compute_latency(signals: &LatencySignals) -> LatencyScore {
    if !signals.has_foreground {
        return LatencyScore {
            score: 0.0,
            needs_boost: false,
            category: LatencyCategory::Responsive,
        };
    }

    // Normalize each signal to [0, 1].
    let jitter_norm = ((signals.jitter_us - JITTER_NOMINAL_US)
        / (JITTER_BAD_US - JITTER_NOMINAL_US))
        .clamp(0.0, 1.0);

    let ws_norm = ((signals.windowserver_cpu - WS_CPU_NOMINAL) / (WS_CPU_BAD - WS_CPU_NOMINAL))
        .clamp(0.0, 1.0);

    let csw_norm =
        ((signals.foreground_csw_per_sec - CSW_NOMINAL) / (CSW_BAD - CSW_NOMINAL)).clamp(0.0, 1.0);

    // Low foreground CPU is ambiguous: most GUI applications legitimately
    // sleep while waiting for input. Treat it as starvation only when the
    // scheduler or the process itself also shows contention. This prevents a
    // quiet editor/browser from creating a self-reinforcing boost loop while
    // retaining the true-positive path for a runnable app that is not getting
    // scheduled.
    let runnable_pressure = jitter_norm > 0.10 || csw_norm > 0.10;
    let fg_starved = if signals.foreground_cpu < 5.0 && runnable_pressure {
        (5.0 - signals.foreground_cpu) / 5.0
    } else {
        0.0
    };

    // Weighted composite:
    //   jitter: 32% — most direct signal of scheduler contention
    //   foreground starvation: 30% — most user-visible (app frozen/unresponsive)
    //   WindowServer: 25% — frame compositing backlog
    //   context switches: 13% — scheduler thrashing (least directly visible)
    let linear = jitter_norm * 0.32 + ws_norm * 0.25 + fg_starved * 0.30 + csw_norm * 0.13;

    // Starvation severity: when fg_starved > 0.8 (cpu < 1%), the app is
    // effectively frozen — this alone should trigger boost regardless of
    // other signals. Add an extra 0.25 for severe starvation.
    let starvation_severity = if fg_starved > 0.8 { 0.25 } else { 0.0 };

    // Synergy bonus: multiple elevated signals compound perceptual impact.
    // Two moderate problems feel worse than one bad problem (Card et al. 1983).
    // Count signals above 0.3 and add a bonus that scales with count.
    let elevated_count = [jitter_norm, ws_norm, fg_starved, csw_norm]
        .iter()
        .filter(|&&v| v > 0.3)
        .count();
    let synergy = if elevated_count >= 2 {
        0.12 * (elevated_count - 1) as f64
    } else {
        0.0
    };

    let score = (linear + starvation_severity + synergy).clamp(0.0, 1.0);

    let category = if score < 0.3 {
        LatencyCategory::Responsive
    } else if score < 0.6 {
        LatencyCategory::Noticeable
    } else if score < 0.8 {
        LatencyCategory::Sluggish
    } else {
        LatencyCategory::Broken
    };

    LatencyScore {
        score,
        // Boost threshold at 0.45: for 90-120fps consistency, we need to act
        // earlier than the Noticeable/Sluggish boundary. At 0.45 we catch
        // compound degradations (jitter + WS backlog) before they become visible.
        needs_boost: score > 0.45,
        category,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn nominal_signals() -> LatencySignals {
        LatencySignals {
            jitter_us: 30.0,
            windowserver_cpu: 10.0,
            foreground_cpu: 15.0,
            foreground_csw_per_sec: 200.0,
            has_foreground: true,
        }
    }

    #[test]
    fn nominal_is_responsive() {
        let score = compute_latency(&nominal_signals());
        assert_eq!(score.category, LatencyCategory::Responsive);
        assert!(!score.needs_boost);
        assert!(score.score < 0.3, "score {} should be < 0.3", score.score);
    }

    #[test]
    fn high_jitter_raises_score() {
        let mut signals = nominal_signals();
        signals.jitter_us = 3000.0; // Way above threshold
        let score = compute_latency(&signals);
        assert!(
            score.score > 0.3,
            "high jitter should raise score above 0.3: {}",
            score.score
        );
    }

    #[test]
    fn contended_starved_foreground_raises_score() {
        let mut signals = nominal_signals();
        signals.foreground_cpu = 0.5; // Nearly starved
        signals.jitter_us = 400.0; // Runnable work is waiting on the scheduler
        let score = compute_latency(&signals);
        assert!(
            score.score > 0.1,
            "starved foreground should raise score: {}",
            score.score
        );
    }

    #[test]
    fn combined_stress_is_sluggish_or_broken() {
        let signals = LatencySignals {
            jitter_us: 5000.0,
            windowserver_cpu: 70.0,
            foreground_cpu: 0.5,
            foreground_csw_per_sec: 8000.0,
            has_foreground: true,
        };
        let score = compute_latency(&signals);
        assert!(
            score.category == LatencyCategory::Sluggish
                || score.category == LatencyCategory::Broken,
            "combined stress should be sluggish or broken: {:?} ({})",
            score.category,
            score.score,
        );
        assert!(score.needs_boost);
    }

    #[test]
    fn no_foreground_is_responsive() {
        let mut signals = nominal_signals();
        signals.has_foreground = false;
        let score = compute_latency(&signals);
        assert_eq!(score.category, LatencyCategory::Responsive);
        assert!(!score.needs_boost);
    }

    #[test]
    fn quiet_foreground_is_not_mistaken_for_cpu_starvation() {
        let signals = LatencySignals {
            jitter_us: 0.0,
            windowserver_cpu: 44.0,
            foreground_cpu: 0.7,
            foreground_csw_per_sec: 120.0,
            has_foreground: true,
        };

        let score = compute_latency(&signals);

        assert!(!score.needs_boost);
        assert_ne!(score.category, LatencyCategory::Sluggish);
        assert_ne!(score.category, LatencyCategory::Broken);
    }
}
