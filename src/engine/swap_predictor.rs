//! Swap Predictor — proactive swap trend forecasting.
//!
//! Tracks swap usage over time and predicts when swap will become critical.
//! Feeds SwapTrend into PredictiveAgent and SysctlGovernor for proactive throttling.

use std::collections::VecDeque;

// ── SwapTrend ─────────────────────────────────────────────────────────────────

/// Direction and urgency of swap usage change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwapTrend {
    /// Swap usage is decreasing — memory pressure easing.
    Decreasing,
    /// Swap usage is stable — no urgent action needed.
    Stable,
    /// Swap usage is growing — consider throttling.
    Increasing,
    /// Swap usage growing rapidly — near-critical, freeze candidates.
    Critical,
}

// ── SwapForecast ──────────────────────────────────────────────────────────────

/// Output of SwapPredictor::update().
#[derive(Debug, Clone)]
pub struct SwapForecast {
    /// Current trend classification.
    pub swap_trend: SwapTrend,
    /// Seconds until swap is predicted to reach critical threshold.
    /// -1 if not trending toward critical.
    pub time_to_swap_critical: i32,
    /// Current swap utilization ratio [0,1].
    pub swap_ratio: f64,
    /// Predicted swap usage bytes (extrapolated forward).
    pub swap_predicted_bytes: u64,
    /// Human-readable action recommendations based on trend.
    pub recommended_actions: Vec<String>,
}

// ── SwapPredictor ─────────────────────────────────────────────────────────────

/// Rolling window swap predictor using linear regression on recent samples.
pub struct SwapPredictor {
    /// Ring buffer of (swap_used_bytes, swap_total_bytes) samples.
    samples: VecDeque<(u64, u64)>,
    /// Maximum samples to retain (~5 min at 5s cycle = 60 samples).
    max_samples: usize,
}

impl SwapPredictor {
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(60),
            max_samples: 60,
        }
    }

    /// Update with current swap stats, return forecast.
    pub fn update(&mut self, swap_used_bytes: u64, swap_total_bytes: u64) -> SwapForecast {
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back((swap_used_bytes, swap_total_bytes));

        let total = if swap_total_bytes > 0 {
            swap_total_bytes
        } else {
            // Default to 2GB if unknown
            2 * 1024 * 1024 * 1024
        };

        let ratio = swap_used_bytes as f64 / total as f64;

        let trend = self.compute_trend(total);
        let tte = self.time_to_critical(swap_used_bytes, total, &trend);
        let predicted = self.predict_bytes(swap_used_bytes);
        let actions = Self::recommend(&trend, ratio);

        SwapForecast {
            swap_trend: trend,
            time_to_swap_critical: tte,
            swap_ratio: ratio,
            swap_predicted_bytes: predicted,
            recommended_actions: actions,
        }
    }

    fn compute_trend(&self, total: u64) -> SwapTrend {
        let n = self.samples.len();
        if n < 3 {
            return SwapTrend::Stable;
        }

        // Compare recent half vs older half
        let mid = n / 2;
        let older_avg = self
            .samples
            .iter()
            .take(mid)
            .map(|(u, _)| *u as f64)
            .sum::<f64>()
            / mid as f64;
        let newer_avg = self
            .samples
            .iter()
            .skip(mid)
            .map(|(u, _)| *u as f64)
            .sum::<f64>()
            / (n - mid) as f64;

        let delta_ratio = (newer_avg - older_avg) / total as f64;

        // Critical: growing >5% of total swap in window
        if delta_ratio > 0.05 {
            return SwapTrend::Critical;
        }
        // Increasing: growing >1%
        if delta_ratio > 0.01 {
            return SwapTrend::Increasing;
        }
        // Decreasing: shrinking >1%
        if delta_ratio < -0.01 {
            return SwapTrend::Decreasing;
        }
        SwapTrend::Stable
    }

    fn predict_bytes(&self, current: u64) -> u64 {
        let n = self.samples.len();
        if n < 2 {
            return current;
        }
        let first = self
            .samples
            .front()
            .map(|(u, _)| *u as f64)
            .unwrap_or(current as f64);
        let rate = (current as f64 - first) / n as f64;
        // Predict 6 samples ahead (~30s)
        let predicted = current as f64 + rate * 6.0;
        predicted.max(0.0) as u64
    }

    fn recommend(trend: &SwapTrend, ratio: f64) -> Vec<String> {
        let mut actions = Vec::new();
        match trend {
            SwapTrend::Critical => {
                actions.push("CRITICAL: Swap growing rapidly — freeze background processes".into());
                if ratio > 0.80 {
                    actions.push("CRITICAL: Swap near capacity — consider emergency purge".into());
                }
            }
            SwapTrend::Increasing => {
                actions.push("Swap increasing — throttle heavy background processes".into());
            }
            SwapTrend::Stable | SwapTrend::Decreasing => {}
        }
        actions
    }

    fn time_to_critical(&self, current: u64, total: u64, trend: &SwapTrend) -> i32 {
        match trend {
            SwapTrend::Stable | SwapTrend::Decreasing => -1,
            SwapTrend::Increasing | SwapTrend::Critical => {
                let n = self.samples.len();
                if n < 2 {
                    return -1;
                }
                // Rate of change: bytes per sample (5s cycle)
                let first = self.samples.front().map(|(u, _)| *u).unwrap_or(current);
                let rate_per_cycle = (current as f64 - first as f64) / n as f64;
                if rate_per_cycle <= 0.0 {
                    return -1;
                }
                let critical_threshold = total as f64 * 0.85;
                let bytes_remaining = critical_threshold - current as f64;
                if bytes_remaining <= 0.0 {
                    return 0;
                }
                let cycles_remaining = bytes_remaining / rate_per_cycle;
                (cycles_remaining * 5.0).round() as i32 // 5s per cycle
            }
        }
    }
}

impl Default for SwapPredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn new_predictor_returns_stable() {
        let mut p = SwapPredictor::new();
        let f = p.update(0, 2 * GB);
        assert_eq!(f.swap_trend, SwapTrend::Stable);
        assert_eq!(f.time_to_swap_critical, -1);
    }

    #[test]
    fn stable_swap_detected() {
        let mut p = SwapPredictor::new();
        for _ in 0..10 {
            p.update(500 * 1024 * 1024, 2 * GB);
        }
        let f = p.update(500 * 1024 * 1024, 2 * GB);
        assert_eq!(f.swap_trend, SwapTrend::Stable);
    }

    #[test]
    fn rapidly_growing_swap_is_critical() {
        let mut p = SwapPredictor::new();
        let total = 2 * GB;
        // Simulate fast growth: 100MB per sample
        let step = 100 * 1024 * 1024_u64;
        for i in 0..10_u64 {
            p.update(i * step, total);
        }
        let f = p.update(10 * step, total);
        assert!(
            matches!(f.swap_trend, SwapTrend::Critical | SwapTrend::Increasing),
            "Expected Increasing or Critical, got {:?}",
            f.swap_trend
        );
    }

    #[test]
    fn decreasing_swap_detected() {
        let mut p = SwapPredictor::new();
        let total = 2 * GB;
        let start = 800 * 1024 * 1024_u64;
        let step = 20 * 1024 * 1024_u64;
        for i in 0..10_u64 {
            p.update(start - i * step, total);
        }
        let f = p.update(start - 10 * step, total);
        assert_eq!(f.swap_trend, SwapTrend::Decreasing);
    }

    #[test]
    fn swap_ratio_computed_correctly() {
        let mut p = SwapPredictor::new();
        let f = p.update(GB, 2 * GB);
        assert!((f.swap_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn time_to_critical_negative_when_stable() {
        let mut p = SwapPredictor::new();
        for _ in 0..10 {
            p.update(GB, 4 * GB);
        }
        let f = p.update(GB, 4 * GB);
        assert_eq!(f.time_to_swap_critical, -1);
    }
}
