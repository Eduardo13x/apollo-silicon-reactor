//! Swap Predictor — proactive swap trend forecasting.
//!
//! Tracks swap usage over time and predicts when swap will become critical.
//! Feeds SwapTrend into PredictiveAgent and SysctlGovernor for proactive throttling.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

const HISTORY_CAPACITY: usize = 120;
const HISTORY_WINDOW: Duration = Duration::from_secs(5 * 60);
const MAX_CONTIGUOUS_GAP: Duration = Duration::from_secs(10);
const REGRESSION_REBASE_INTERVAL: Duration = Duration::from_secs(60 * 60);
const FORECAST_HORIZON_SECONDS: f64 = 30.0;
const CRITICAL_ENVELOPE_FRACTION_OF_RAM: f64 = 0.50;
// Match or exceed the legacy M1 cadence thresholds while scaling with RAM.
// This prevents the normalized predictor from increasing actuator frequency.
const TREND_FRACTION_OF_RAM_PER_MINUTE: f64 = 0.002;
const CRITICAL_FRACTION_OF_RAM_PER_MINUTE: f64 = 0.010;

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
#[derive(Debug, Clone, PartialEq)]
pub struct SwapForecast {
    /// Current trend classification.
    pub swap_trend: SwapTrend,
    /// Seconds until swap is predicted to reach critical threshold.
    /// None = not trending toward critical (stable or decreasing).
    pub time_to_swap_critical: Option<i32>,
    /// Current swap utilization ratio [0,1].
    pub swap_ratio: f64,
    /// Predicted swap usage bytes (extrapolated forward).
    pub swap_predicted_bytes: u64,
    /// Human-readable action recommendations based on trend.
    pub recommended_actions: Vec<String>,
}

// ── SwapPredictor ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct TimedSwapSample {
    generation: u64,
    observed_at: Instant,
    elapsed_seconds: f64,
    used_fraction_of_ram: f64,
}

/// Rolling, wall-clock swap predictor. It observes only fresh collector
/// generations and keeps bounded regression sums, so duplicate daemon cycles
/// are an O(1) cached return.
pub struct SwapPredictor {
    samples: VecDeque<TimedSwapSample>,
    origin: Instant,
    sum_t: f64,
    sum_y: f64,
    sum_tt: f64,
    sum_ty: f64,
    last_forecast: SwapForecast,
    compatibility_generation: u64,
    compatibility_elapsed: Duration,
}

impl SwapPredictor {
    pub fn new() -> Self {
        let origin = Instant::now();
        Self {
            samples: VecDeque::with_capacity(HISTORY_CAPACITY),
            origin,
            sum_t: 0.0,
            sum_y: 0.0,
            sum_tt: 0.0,
            sum_ty: 0.0,
            last_forecast: stable_forecast(0, 0),
            compatibility_generation: 0,
            compatibility_elapsed: Duration::ZERO,
        }
    }

    /// Backward-compatible deterministic adapter. Production uses
    /// [`Self::update_at`] with the collector's real generation and timestamp.
    pub fn update(&mut self, swap_used_bytes: u64, swap_total_bytes: u64) -> SwapForecast {
        self.compatibility_generation = self.compatibility_generation.saturating_add(1);
        self.compatibility_elapsed += Duration::from_secs(5);
        let physical_memory_bytes = swap_total_bytes.max(2 * 1024 * 1024 * 1024);
        self.update_at(
            self.compatibility_generation,
            self.origin + self.compatibility_elapsed,
            swap_used_bytes,
            swap_total_bytes,
            physical_memory_bytes,
        )
    }

    pub fn update_at(
        &mut self,
        generation: u64,
        observed_at: Instant,
        swap_used_bytes: u64,
        swap_total_bytes: u64,
        physical_memory_bytes: u64,
    ) -> SwapForecast {
        if self
            .samples
            .back()
            .is_some_and(|sample| sample.generation == generation)
        {
            return self.last_forecast.clone();
        }
        if physical_memory_bytes == 0 {
            self.clear_samples();
            self.last_forecast = stable_forecast(swap_used_bytes, swap_total_bytes);
            return self.last_forecast.clone();
        }

        if self.samples.is_empty() {
            self.origin = observed_at;
        }

        if self.samples.back().is_some_and(|sample| {
            observed_at.saturating_duration_since(sample.observed_at) > MAX_CONTIGUOUS_GAP
        }) || observed_at.saturating_duration_since(self.origin) > REGRESSION_REBASE_INTERVAL
        {
            self.clear_samples();
            self.origin = observed_at;
        }

        while self.samples.len() >= HISTORY_CAPACITY
            || self.samples.front().is_some_and(|front| {
                observed_at.saturating_duration_since(front.observed_at) > HISTORY_WINDOW
            })
        {
            self.pop_front();
        }

        let sample = TimedSwapSample {
            generation,
            observed_at,
            elapsed_seconds: observed_at
                .saturating_duration_since(self.origin)
                .as_secs_f64(),
            used_fraction_of_ram: swap_used_bytes as f64 / physical_memory_bytes as f64,
        };
        self.push_sample(sample);

        let slope_fraction_per_second = self.regression_slope().unwrap_or(0.0);
        let recent_slope = self.recent_slope().unwrap_or(0.0);
        let trend_floor = TREND_FRACTION_OF_RAM_PER_MINUTE / 60.0;
        let critical_floor = CRITICAL_FRACTION_OF_RAM_PER_MINUTE / 60.0;
        let span = self
            .samples
            .front()
            .map(|front| observed_at.saturating_duration_since(front.observed_at))
            .unwrap_or_default();
        let trend = if self.samples.len() < 3 || span < Duration::from_secs(2) {
            SwapTrend::Stable
        } else if recent_slope >= critical_floor && slope_fraction_per_second >= critical_floor {
            SwapTrend::Critical
        } else if recent_slope >= trend_floor && slope_fraction_per_second >= trend_floor {
            SwapTrend::Increasing
        } else if recent_slope <= -trend_floor && slope_fraction_per_second <= -trend_floor {
            SwapTrend::Decreasing
        } else {
            SwapTrend::Stable
        };

        let predicted_fraction = (sample.used_fraction_of_ram
            + slope_fraction_per_second * FORECAST_HORIZON_SECONDS)
            .max(0.0);
        let predicted =
            (predicted_fraction * physical_memory_bytes as f64).clamp(0.0, u64::MAX as f64) as u64;
        let critical_fraction = CRITICAL_ENVELOPE_FRACTION_OF_RAM;
        let time_to_swap_critical = if matches!(trend, SwapTrend::Increasing | SwapTrend::Critical)
            && slope_fraction_per_second > 0.0
        {
            let remaining = critical_fraction - sample.used_fraction_of_ram;
            if remaining <= 0.0 {
                Some(0)
            } else {
                Some(
                    (remaining / slope_fraction_per_second)
                        .round()
                        .clamp(0.0, 604_800.0) as i32,
                )
            }
        } else {
            None
        };
        let ratio = if swap_total_bytes > 0 {
            (swap_used_bytes as f64 / swap_total_bytes as f64).max(0.0)
        } else {
            0.0
        };
        self.last_forecast = SwapForecast {
            swap_trend: trend,
            time_to_swap_critical,
            swap_ratio: ratio,
            swap_predicted_bytes: predicted,
            recommended_actions: Self::recommend(&trend, ratio),
        };
        self.last_forecast.clone()
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

    fn push_sample(&mut self, sample: TimedSwapSample) {
        self.sum_t += sample.elapsed_seconds;
        self.sum_y += sample.used_fraction_of_ram;
        self.sum_tt += sample.elapsed_seconds * sample.elapsed_seconds;
        self.sum_ty += sample.elapsed_seconds * sample.used_fraction_of_ram;
        self.samples.push_back(sample);
    }

    fn pop_front(&mut self) {
        if let Some(sample) = self.samples.pop_front() {
            self.sum_t -= sample.elapsed_seconds;
            self.sum_y -= sample.used_fraction_of_ram;
            self.sum_tt -= sample.elapsed_seconds * sample.elapsed_seconds;
            self.sum_ty -= sample.elapsed_seconds * sample.used_fraction_of_ram;
        }
    }

    fn clear_samples(&mut self) {
        self.samples.clear();
        self.sum_t = 0.0;
        self.sum_y = 0.0;
        self.sum_tt = 0.0;
        self.sum_ty = 0.0;
    }

    pub fn invalidate(&mut self, swap_used_bytes: u64, swap_total_bytes: u64) -> SwapForecast {
        self.clear_samples();
        self.last_forecast = stable_forecast(swap_used_bytes, swap_total_bytes);
        self.last_forecast.clone()
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    fn regression_slope(&self) -> Option<f64> {
        let n = self.samples.len() as f64;
        if n < 2.0 {
            return None;
        }
        let denominator = n * self.sum_tt - self.sum_t * self.sum_t;
        (denominator.abs() > f64::EPSILON)
            .then(|| (n * self.sum_ty - self.sum_t * self.sum_y) / denominator)
    }

    fn recent_slope(&self) -> Option<f64> {
        let mut samples = self.samples.iter().rev();
        let current = samples.next()?;
        let previous = samples.next()?;
        let dt = current
            .observed_at
            .saturating_duration_since(previous.observed_at)
            .as_secs_f64();
        (dt > 0.0).then(|| (current.used_fraction_of_ram - previous.used_fraction_of_ram) / dt)
    }
}

fn stable_forecast(swap_used_bytes: u64, swap_total_bytes: u64) -> SwapForecast {
    SwapForecast {
        swap_trend: SwapTrend::Stable,
        time_to_swap_critical: None,
        swap_ratio: if swap_total_bytes > 0 {
            swap_used_bytes as f64 / swap_total_bytes as f64
        } else {
            0.0
        },
        swap_predicted_bytes: swap_used_bytes,
        recommended_actions: Vec::new(),
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
        assert_eq!(f.time_to_swap_critical, None);
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
        assert_eq!(f.time_to_swap_critical, None);
    }

    #[test]
    fn real_cadence_produces_equivalent_normalized_trend() {
        let start = Instant::now();
        let mut fast = SwapPredictor::new();
        let mut slow = SwapPredictor::new();
        let ram = 8 * GB;
        let total = 2 * GB;
        let rate = 16 * 1024 * 1024_u64;

        let mut fast_forecast = stable_forecast(0, total);
        for index in 0..=8_u64 {
            let elapsed_ms = index * 500;
            fast_forecast = fast.update_at(
                index + 1,
                start + Duration::from_millis(elapsed_ms),
                rate * elapsed_ms / 1_000,
                total,
                ram,
            );
        }
        let mut slow_forecast = stable_forecast(0, total);
        for index in 0..=2_u64 {
            let elapsed_secs = index * 2;
            slow_forecast = slow.update_at(
                index + 1,
                start + Duration::from_secs(elapsed_secs),
                rate * elapsed_secs,
                total,
                ram,
            );
        }

        assert_eq!(fast_forecast.swap_trend, slow_forecast.swap_trend);
        assert_eq!(fast_forecast.swap_trend, SwapTrend::Critical);
        assert!(
            fast_forecast
                .swap_predicted_bytes
                .abs_diff(slow_forecast.swap_predicted_bytes)
                < 1024
        );
    }

    #[test]
    fn duplicate_generation_is_a_cached_noop() {
        let start = Instant::now();
        let mut predictor = SwapPredictor::new();
        let first = predictor.update_at(1, start, GB, 2 * GB, 8 * GB);
        let duplicate =
            predictor.update_at(1, start + Duration::from_secs(1), 2 * GB, 4 * GB, 8 * GB);

        assert_eq!(first, duplicate);
        assert_eq!(predictor.samples.len(), 1);
    }

    #[test]
    fn burst_then_flat_does_not_remain_critical() {
        let start = Instant::now();
        let mut predictor = SwapPredictor::new();
        let ram = 8 * GB;
        predictor.update_at(1, start, GB, 2 * GB, ram);
        predictor.update_at(2, start + Duration::from_secs(1), 2 * GB, 3 * GB, ram);
        let forecast = predictor.update_at(3, start + Duration::from_secs(2), 2 * GB, 4 * GB, ram);

        assert_eq!(forecast.swap_trend, SwapTrend::Stable);
    }

    #[test]
    fn dynamic_swap_total_does_not_change_physical_prediction() {
        let start = Instant::now();
        let ram = 16 * GB;
        let mut a = SwapPredictor::new();
        let mut b = SwapPredictor::new();
        let mut fa = stable_forecast(0, 0);
        let mut fb = stable_forecast(0, 0);
        for index in 0..4_u64 {
            let at = start + Duration::from_secs(index);
            let used = GB + index * 128 * 1024 * 1024;
            fa = a.update_at(index + 1, at, used, 2 * GB, ram);
            fb = b.update_at(index + 1, at, used, 2 * GB + index * GB, ram);
        }
        assert_eq!(fa.swap_trend, fb.swap_trend);
        assert_eq!(fa.swap_predicted_bytes, fb.swap_predicted_bytes);
    }

    #[test]
    fn sleep_sized_gap_resets_the_slope() {
        let start = Instant::now();
        let mut predictor = SwapPredictor::new();
        predictor.update_at(1, start, 0, 2 * GB, 8 * GB);
        predictor.update_at(2, start + Duration::from_secs(1), GB, 2 * GB, 8 * GB);
        let forecast =
            predictor.update_at(3, start + Duration::from_secs(60), 2 * GB, 3 * GB, 8 * GB);

        assert_eq!(forecast.swap_trend, SwapTrend::Stable);
        assert_eq!(predictor.samples.len(), 1);
    }

    #[test]
    fn invalidation_discards_cached_critical_forecast() {
        let start = Instant::now();
        let mut predictor = SwapPredictor::new();
        let ram = 8 * GB;
        predictor.update_at(1, start, 0, 2 * GB, ram);
        predictor.update_at(2, start + Duration::from_secs(1), GB, 2 * GB, ram);
        let critical = predictor.update_at(3, start + Duration::from_secs(2), 2 * GB, 2 * GB, ram);
        assert_ne!(critical.swap_trend, SwapTrend::Stable);

        let passive = predictor.invalidate(2 * GB, 2 * GB);

        assert_eq!(passive.swap_trend, SwapTrend::Stable);
        assert_eq!(passive.time_to_swap_critical, None);
    }
}
