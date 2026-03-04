//! Swap prediction and proactive memory management
//!
//! Predicts when swap will be needed and takes proactive actions.

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct SwapForecast {
    pub swap_used_bytes: u64,
    pub swap_predicted_bytes: u64,
    pub time_to_swap_critical: i32, // seconds, -1 if none
    pub swap_trend: SwapTrend,
    pub recommended_actions: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapTrend {
    Decreasing,  // Swap usage going down
    Stable,      // No significant change
    Increasing,  // Gradual increase
    Critical,    // Rapid increase
}

pub struct SwapPredictor {
    history: VecDeque<SwapSnapshot>,
    max_history: usize,
    swap_critical_threshold: u64, // When to alert (1GB)
    swap_max_safe: u64,           // Max safe swap (2GB)
}

#[derive(Debug, Clone)]
struct SwapSnapshot {
    timestamp: std::time::Instant,
    swap_used: u64,
}

impl SwapPredictor {
    pub fn new() -> Self {
        Self {
            history: VecDeque::new(),
            max_history: 120, // 2 minutes at 1Hz
            swap_critical_threshold: 1024 * 1024 * 1024, // 1GB
            swap_max_safe: 2 * 1024 * 1024 * 1024,       // 2GB
        }
    }

    /// Update swap metrics and generate forecast.
    /// `swap_total` is accepted for future use (e.g. percentage-based thresholds).
    pub fn update(&mut self, swap_used: u64, swap_total: u64) -> SwapForecast {
        let _ = swap_total;
        let snapshot = SwapSnapshot {
            timestamp: std::time::Instant::now(),
            swap_used,
        };

        self.history.push_back(snapshot.clone());
        if self.history.len() > self.max_history {
            let _ = self.history.pop_front();
        }

        let trend = self.calculate_trend();
        let predicted = self.predict_swap_usage();
        let time_to_critical = self.time_to_critical(swap_used);
        let recommendations = self.get_recommendations(&trend, swap_used);

        SwapForecast {
            swap_used_bytes: swap_used,
            swap_predicted_bytes: predicted,
            time_to_swap_critical: time_to_critical,
            swap_trend: trend,
            recommended_actions: recommendations,
        }
    }

    fn calculate_trend(&self) -> SwapTrend {
        if self.history.len() < 3 {
            return SwapTrend::Stable;
        }

        let recent: Vec<_> = self.history.iter().rev().take(10).collect();
        let mut increases = 0;

        for i in 1..recent.len() {
            if recent[i].swap_used < recent[i - 1].swap_used {
                increases += 1;
            }
        }

        let increase_ratio = increases as f32 / (recent.len() - 1).max(1) as f32;

        if recent[0].swap_used > self.swap_critical_threshold {
            SwapTrend::Critical
        } else if increase_ratio > 0.7 {
            SwapTrend::Increasing
        } else if increase_ratio > 0.3 {
            SwapTrend::Stable
        } else {
            SwapTrend::Decreasing
        }
    }

    fn predict_swap_usage(&self) -> u64 {
        if self.history.len() < 2 {
            return self.history.back().map(|s| s.swap_used).unwrap_or(0);
        }

        let recent: Vec<_> = self.history.iter().rev().take(20).collect();
        let first = recent[recent.len() - 1];
        let last = recent[0];

        let time_delta = last
            .timestamp
            .duration_since(first.timestamp)
            .as_secs_f64()
            .max(1.0);
        let swap_delta = last.swap_used as i64 - first.swap_used as i64;

        let rate = swap_delta as f64 / time_delta; // bytes/sec

        if rate > 0.0 {
            // Extrapolate 30 seconds into future
            let predicted = (last.swap_used as f64 + (rate * 30.0)).max(0.0) as u64;
            predicted.min(self.swap_max_safe * 2)
        } else {
            last.swap_used
        }
    }

    fn time_to_critical(&self, current_swap: u64) -> i32 {
        if current_swap >= self.swap_critical_threshold {
            return 0;
        }

        if self.history.len() < 2 {
            return -1;
        }

        let recent: Vec<_> = self.history.iter().rev().take(10).collect();
        let first = recent[recent.len() - 1];
        let last = recent[0];

        let time_delta = last
            .timestamp
            .duration_since(first.timestamp)
            .as_secs_f64()
            .max(1.0);
        let swap_delta = last.swap_used as f64 - first.swap_used as f64;

        if swap_delta <= 0.0 {
            return -1; // Decreasing or stable
        }

        let rate = swap_delta / time_delta;
        let threshold_delta = self.swap_critical_threshold as f64 - current_swap as f64;
        let seconds = (threshold_delta / rate) as i32;

        seconds.max(0)
    }

    fn get_recommendations(&self, trend: &SwapTrend, swap_used: u64) -> Vec<String> {
        let mut recommendations = Vec::new();

        match trend {
            SwapTrend::Critical => {
                recommendations.push("🔴 CRITICAL: Enable aggressive memory compression".to_string());
                recommendations.push("🔴 Kill non-essential processes immediately".to_string());
            }
            SwapTrend::Increasing => {
                if swap_used > self.swap_critical_threshold / 2 {
                    recommendations.push("🟡 Swap usage increasing: Consider reducing background load".to_string());
                }
            }
            _ => {}
        }

        recommendations
    }
}

impl Default for SwapPredictor {
    fn default() -> Self {
        Self::new()
    }
}
